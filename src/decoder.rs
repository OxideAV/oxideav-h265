//! HEVC decoder glue.
//!
//! Ties the parameter-set and slice-header parsers to the CTU pipeline and
//! exposes an `oxideav_core::Decoder` implementation. Scope:
//!
//! * **I, P, and B slices, 8-bit 4:2:0** — full pixel decode. Reconstructed
//!   luma and chroma are emitted as a `VideoFrame` (pixel format
//!   `Yuv420P`). Inter slices pull references from a small DPB keyed by
//!   POC. B slices expose both L0 and L1 lists; the CTU walker combines
//!   two MC predictors for bi-prediction.
//! * **Everything else** — returns a specific `Error::Unsupported` message
//!   surfacing the feature that is not yet implemented.

use std::collections::HashMap;
use std::collections::VecDeque;

use oxideav_core::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, TimeBase, VideoFrame,
    VideoPlane,
};

use crate::cabac::InitType;
use crate::ctu::{decode_slice_ctus, CtuContext, Picture};
use crate::deblock::deblock_picture;
use crate::hvcc::parse_hvcc;
use crate::inter::{Dpb, RefPicture};
use crate::nal::{extract_rbsp, iter_annex_b, iter_length_prefixed, NalRef, NalUnitType};
use crate::pps::{parse_pps, PicParameterSet};
use crate::sao::apply_sao;
use crate::slice::{parse_slice_segment_header, SliceSegmentHeader, SliceType};
use crate::sps::{parse_sps, SeqParameterSet};
use crate::vps::{parse_vps, VideoParameterSet};

/// Build a decoder for the registry.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let mut dec = HevcDecoder::new(params.codec_id.clone());
    if !params.extradata.is_empty() {
        dec.consume_extradata(&params.extradata)?;
    }
    Ok(Box::new(dec))
}

pub struct HevcDecoder {
    codec_id: CodecId,
    length_size: Option<u8>,
    pub vps: HashMap<u8, VideoParameterSet>,
    pub sps: HashMap<u32, SeqParameterSet>,
    pub pps: HashMap<u32, PicParameterSet>,
    pub last_slice: Option<SliceSegmentHeader>,
    pending: VecDeque<VideoFrame>,
    eof: bool,
    last_pts: Option<i64>,
    last_time_base: TimeBase,
    /// Decoded picture buffer holding previously decoded reference pictures.
    dpb: Dpb,
    /// Running POC of the previous picture (§8.3.1).
    prev_poc_msb: i32,
    prev_poc_lsb: i32,
}

impl HevcDecoder {
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            length_size: None,
            vps: HashMap::new(),
            sps: HashMap::new(),
            pps: HashMap::new(),
            last_slice: None,
            pending: VecDeque::new(),
            eof: false,
            last_pts: None,
            last_time_base: TimeBase::new(1, 1),
            dpb: Dpb::new(8),
            prev_poc_msb: 0,
            prev_poc_lsb: 0,
        }
    }

    pub fn consume_extradata(&mut self, extradata: &[u8]) -> Result<()> {
        let cfg = parse_hvcc(extradata)?;
        self.length_size = Some(cfg.length_size_minus_one + 1);
        for arr in &cfg.nal_arrays {
            for nal in &arr.nals {
                self.process_nal_bytes(nal)?;
            }
        }
        Ok(())
    }

    pub fn process_nal_bytes(&mut self, raw: &[u8]) -> Result<()> {
        if raw.len() < 3 {
            return Err(Error::invalid("h265: NAL too short"));
        }
        let nals: Vec<NalRef<'_>> = vec![NalRef {
            header: crate::nal::NalHeader::parse(raw)?,
            raw,
        }];
        self.process_nals(&nals)
    }

    fn process_nals(&mut self, nals: &[NalRef<'_>]) -> Result<()> {
        for nal in nals {
            let rbsp = extract_rbsp(nal.payload());
            match nal.header.nal_unit_type {
                NalUnitType::Vps => {
                    let vps = parse_vps(&rbsp)?;
                    self.vps.insert(vps.vps_video_parameter_set_id, vps);
                }
                NalUnitType::Sps => {
                    let sps = parse_sps(&rbsp)?;
                    self.sps.insert(sps.sps_seq_parameter_set_id, sps);
                }
                NalUnitType::Pps => {
                    let pps = parse_pps(&rbsp)?;
                    self.pps.insert(pps.pps_pic_parameter_set_id, pps);
                }
                t if t.is_vcl() => {
                    if let Some((sps, pps)) = self.active_sps_pps_for(&nal.header, &rbsp) {
                        let hdr = parse_slice_segment_header(&rbsp, &nal.header, &sps, &pps)?;
                        let is_idr = matches!(
                            nal.header.nal_unit_type,
                            NalUnitType::IdrWRadl | NalUnitType::IdrNLp
                        );
                        self.last_slice = Some(hdr.clone());
                        if hdr.is_full_i_slice && hdr.slice_type == SliceType::I {
                            match self.decode_intra_slice(&rbsp, &hdr, &sps, &pps, is_idr) {
                                Ok(frame) => self.pending.push_back(frame),
                                Err(Error::Unsupported(msg)) => {
                                    return Err(Error::unsupported(msg));
                                }
                                Err(e) => return Err(e),
                            }
                        } else if (hdr.is_full_p_slice && hdr.slice_type == SliceType::P)
                            || (hdr.is_full_b_slice && hdr.slice_type == SliceType::B)
                        {
                            match self.decode_inter_slice(&rbsp, &hdr, &sps, &pps) {
                                Ok(frame) => self.pending.push_back(frame),
                                Err(Error::Unsupported(msg)) => {
                                    return Err(Error::unsupported(msg));
                                }
                                Err(e) => return Err(e),
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn decode_intra_slice(
        &mut self,
        rbsp: &[u8],
        hdr: &SliceSegmentHeader,
        sps: &SeqParameterSet,
        pps: &PicParameterSet,
        is_idr: bool,
    ) -> Result<VideoFrame> {
        let width = sps.pic_width_in_luma_samples;
        let height = sps.pic_height_in_luma_samples;
        let mut pic = Picture::new(width, height);
        let empty: Vec<RefPicture> = Vec::new();
        let cctx = CtuContext {
            sps,
            pps,
            slice: hdr,
            init_type: InitType::I,
            ref_list_l0: &empty,
            ref_list_l1: &empty,
            collocated_ref: None,
            weighted_pred: None,
            cur_poc: 0,
        };
        let byte_off = (hdr.slice_data_bit_offset / 8) as usize;
        decode_slice_ctus(rbsp, byte_off, &cctx, &mut pic)?;

        // Apply in-loop deblocking filter (§8.7.2) to reconstructed samples
        // before the picture is stored for reference and output.
        deblock_picture(&mut pic, sps, pps, hdr);
        // SAO (§8.7.3) runs after deblocking and before the DPB write.
        apply_sao(&mut pic, sps, hdr);

        let poc = self.derive_poc(sps, hdr, is_idr);
        self.store_ref_pic(&pic, poc);
        Ok(self.emit_frame(&pic, sps))
    }

    fn decode_inter_slice(
        &mut self,
        rbsp: &[u8],
        hdr: &SliceSegmentHeader,
        sps: &SeqParameterSet,
        pps: &PicParameterSet,
    ) -> Result<VideoFrame> {
        let width = sps.pic_width_in_luma_samples;
        let height = sps.pic_height_in_luma_samples;
        let current_poc = self.derive_poc(sps, hdr, false);
        let (rpl0, rpl1) = self.build_ref_pic_lists(sps, hdr, current_poc);
        if rpl0.is_empty() {
            return Err(Error::unsupported(
                "h265 inter slice: no reference pictures",
            ));
        }
        if hdr.slice_type == SliceType::B && rpl1.is_empty() {
            return Err(Error::unsupported("h265 B slice: no L1 reference pictures"));
        }

        // Select the collocated reference picture for TMVP, if enabled.
        let collocated_ref = if hdr.slice_temporal_mvp_enabled_flag {
            let list = if hdr.collocated_from_l0_flag || hdr.slice_type == SliceType::P {
                &rpl0
            } else {
                &rpl1
            };
            list.get(hdr.collocated_ref_idx as usize)
        } else {
            None
        };

        let mut pic = Picture::new(width, height);
        let is_b = hdr.slice_type == SliceType::B;
        let init_type = crate::cabac::InitType::for_slice(false, is_b, hdr.cabac_init_flag);
        let cctx = CtuContext {
            sps,
            pps,
            slice: hdr,
            init_type,
            ref_list_l0: &rpl0,
            ref_list_l1: &rpl1,
            collocated_ref,
            weighted_pred: hdr.weighted_pred.as_ref(),
            cur_poc: current_poc,
        };
        let byte_off = (hdr.slice_data_bit_offset / 8) as usize;
        decode_slice_ctus(rbsp, byte_off, &cctx, &mut pic)?;

        deblock_picture(&mut pic, sps, pps, hdr);
        apply_sao(&mut pic, sps, hdr);

        self.store_ref_pic(&pic, current_poc);
        Ok(self.emit_frame(&pic, sps))
    }

    fn build_ref_pic_lists(
        &mut self,
        sps: &SeqParameterSet,
        hdr: &SliceSegmentHeader,
        current_poc: i32,
    ) -> (Vec<RefPicture>, Vec<RefPicture>) {
        let Some(rps) = hdr.current_rps.as_ref() else {
            return (Vec::new(), Vec::new());
        };
        // §8.3.2: RefPicSetStCurrBefore (negative deltas) and
        // RefPicSetStCurrAfter (positive deltas), used to build RPL0 and
        // RPL1 (default initialisation §8.3.4).
        let mut st_curr_before: Vec<RefPicture> = Vec::new();
        for (i, delta) in rps.delta_poc_s0.iter().enumerate() {
            if !rps.used_by_curr_pic_s0.get(i).copied().unwrap_or(false) {
                continue;
            }
            let target = current_poc + delta;
            if let Some(p) = self.dpb.get_by_poc(target) {
                st_curr_before.push(p.clone());
            }
        }
        let mut st_curr_after: Vec<RefPicture> = Vec::new();
        for (i, delta) in rps.delta_poc_s1.iter().enumerate() {
            if !rps.used_by_curr_pic_s1.get(i).copied().unwrap_or(false) {
                continue;
            }
            let target = current_poc + delta;
            if let Some(p) = self.dpb.get_by_poc(target) {
                st_curr_after.push(p.clone());
            }
        }

        // §8.3.2 long-term reference resolution. For each slice-level LT
        // entry with `used_by_curr_pic_lt_flag == 1`, derive the full POC
        // from `poc_lsb_lt` (+ optional `delta_poc_msb_cycle_lt`) and look
        // the picture up in the DPB. Matches are marked LT on the DPB so
        // they survive later st_curr_* rotations.
        let max_poc_lsb = 1i32 << (sps.log2_max_pic_order_cnt_lsb_minus4 + 4);
        let cur_lsb = current_poc & (max_poc_lsb - 1);
        let mut lt_curr: Vec<RefPicture> = Vec::new();
        let mut lts_to_mark: Vec<i32> = Vec::new();
        for lt in &hdr.long_term_refs {
            if !lt.used_by_curr_pic_lt_flag {
                continue;
            }
            let target_poc: Option<i32> = if let Some(cycle) = lt.delta_poc_msb_cycle_lt {
                // eq. 8-5: PocLt = current_poc - cycle * MaxPocLsb -
                //                  (cur_lsb - poc_lsb_lt).
                Some(current_poc - (cycle as i32) * max_poc_lsb - (cur_lsb - lt.poc_lsb_lt as i32))
            } else {
                self.dpb
                    .find_by_poc_lsb(lt.poc_lsb_lt, max_poc_lsb as u32)
                    .map(|p| p.poc)
            };
            if let Some(poc) = target_poc {
                if let Some(p) = self.dpb.get_by_poc(poc) {
                    let mut clone = p.clone();
                    clone.is_long_term = true;
                    lt_curr.push(clone);
                    lts_to_mark.push(poc);
                }
            }
        }
        for poc in lts_to_mark {
            self.dpb.mark_long_term_by_poc(poc);
        }

        // §8.3.4: build RefPicListTempX from the three curr-sets, then
        // apply `ref_pic_list_modification()` if signalled. Absent the
        // modification, the slice uses the first NumRefIdxActive entries
        // of RefPicListTempX directly.
        let mut temp0: Vec<RefPicture> = Vec::new();
        temp0.extend(st_curr_before.iter().cloned());
        temp0.extend(st_curr_after.iter().cloned());
        temp0.extend(lt_curr.iter().cloned());
        let active0 = (hdr.num_ref_idx_l0_active_minus1 + 1) as usize;
        let rpl0 = apply_list_modification(
            &temp0,
            hdr.ref_pic_list_modification_flag_l0,
            &hdr.list_entry_l0,
            active0,
        );

        // §8.3.4 RefPicListTemp1: after-first, before, lt.
        let mut rpl1: Vec<RefPicture> = Vec::new();
        if hdr.slice_type == SliceType::B {
            let mut temp1: Vec<RefPicture> = Vec::new();
            temp1.extend(st_curr_after.iter().cloned());
            temp1.extend(st_curr_before.iter().cloned());
            temp1.extend(lt_curr.iter().cloned());
            let active1 = (hdr.num_ref_idx_l1_active_minus1 + 1) as usize;
            rpl1 = apply_list_modification(
                &temp1,
                hdr.ref_pic_list_modification_flag_l1,
                &hdr.list_entry_l1,
                active1,
            );
        }
        (rpl0, rpl1)
    }

    fn derive_poc(&mut self, sps: &SeqParameterSet, hdr: &SliceSegmentHeader, is_idr: bool) -> i32 {
        // §8.3.1. For IDR, POC = 0 and MSB/LSB reset.
        if is_idr {
            self.prev_poc_msb = 0;
            self.prev_poc_lsb = 0;
            return 0;
        }
        let max_poc_lsb = 1i32 << (sps.log2_max_pic_order_cnt_lsb_minus4 + 4);
        let poc_lsb = hdr.slice_pic_order_cnt_lsb as i32;
        let poc_msb =
            if poc_lsb < self.prev_poc_lsb && self.prev_poc_lsb - poc_lsb >= max_poc_lsb / 2 {
                self.prev_poc_msb + max_poc_lsb
            } else if poc_lsb > self.prev_poc_lsb && poc_lsb - self.prev_poc_lsb > max_poc_lsb / 2 {
                self.prev_poc_msb - max_poc_lsb
            } else {
                self.prev_poc_msb
            };
        let poc = poc_msb + poc_lsb;
        self.prev_poc_msb = poc_msb;
        self.prev_poc_lsb = poc_lsb;
        poc
    }

    fn store_ref_pic(&mut self, pic: &Picture, poc: i32) {
        self.dpb.push(RefPicture {
            poc,
            width: pic.width,
            height: pic.height,
            luma: pic.luma.clone(),
            cb: pic.cb.clone(),
            cr: pic.cr.clone(),
            luma_stride: pic.luma_stride,
            chroma_stride: pic.chroma_stride,
            inter: pic.inter.clone(),
            // New pictures enter as short-term refs by default; the slice
            // header that consumes them may mark older DPB entries as LT
            // via `mark_long_term_refs`.
            is_long_term: false,
        });
    }

    fn emit_frame(&self, pic: &Picture, sps: &SeqParameterSet) -> VideoFrame {
        let cropped_w = sps.cropped_width();
        let cropped_h = sps.cropped_height();
        let (cw, ch) = (cropped_w as usize, cropped_h as usize);
        let (cwc, chc) = (cw / 2, ch / 2);
        let bit_depth_y = sps.bit_depth_y();
        let bit_depth_c = sps.bit_depth_c();
        // Main 10 (bit_depth_y == 10) uses `Yuv420P10Le`: two bytes per
        // sample, little-endian, low 10 bits valid. Main 8 narrows u16→u8.
        // Higher bit depths fall through to the 10-bit surface for now —
        // only Main 10 is exercised in tests.
        if bit_depth_y > 8 {
            let bps_y = 2usize;
            let bps_c = 2usize;
            let y_stride = cw * bps_y;
            let c_stride = cwc * bps_c;
            let max_y = (1u16 << bit_depth_y) - 1;
            let max_c = (1u16 << bit_depth_c) - 1;
            let mut y_plane = vec![0u8; y_stride * ch];
            let mut cb_plane = vec![0u8; c_stride * chc];
            let mut cr_plane = vec![0u8; c_stride * chc];
            for y in 0..ch {
                let src = &pic.luma[y * pic.luma_stride..y * pic.luma_stride + cw];
                let dst = &mut y_plane[y * y_stride..y * y_stride + y_stride];
                for (i, &s) in src.iter().enumerate() {
                    let v = s.min(max_y);
                    dst[i * 2] = (v & 0xFF) as u8;
                    dst[i * 2 + 1] = (v >> 8) as u8;
                }
            }
            for y in 0..chc {
                let src_cb = &pic.cb[y * pic.chroma_stride..y * pic.chroma_stride + cwc];
                let src_cr = &pic.cr[y * pic.chroma_stride..y * pic.chroma_stride + cwc];
                let dst_cb = &mut cb_plane[y * c_stride..y * c_stride + c_stride];
                let dst_cr = &mut cr_plane[y * c_stride..y * c_stride + c_stride];
                for i in 0..cwc {
                    let vcb = src_cb[i].min(max_c);
                    let vcr = src_cr[i].min(max_c);
                    dst_cb[i * 2] = (vcb & 0xFF) as u8;
                    dst_cb[i * 2 + 1] = (vcb >> 8) as u8;
                    dst_cr[i * 2] = (vcr & 0xFF) as u8;
                    dst_cr[i * 2 + 1] = (vcr >> 8) as u8;
                }
            }
            return VideoFrame {
                format: PixelFormat::Yuv420P10Le,
                width: cw as u32,
                height: ch as u32,
                pts: self.last_pts,
                time_base: self.last_time_base,
                planes: vec![
                    VideoPlane {
                        stride: y_stride,
                        data: y_plane,
                    },
                    VideoPlane {
                        stride: c_stride,
                        data: cb_plane,
                    },
                    VideoPlane {
                        stride: c_stride,
                        data: cr_plane,
                    },
                ],
            };
        }
        let mut y_plane = vec![0u8; cw * ch];
        let mut cb_plane = vec![0u8; cwc * chc];
        let mut cr_plane = vec![0u8; cwc * chc];
        for y in 0..ch {
            let src = &pic.luma[y * pic.luma_stride..y * pic.luma_stride + cw];
            let dst = &mut y_plane[y * cw..(y + 1) * cw];
            for (i, &s) in src.iter().enumerate() {
                dst[i] = s.min(255) as u8;
            }
        }
        for y in 0..chc {
            let src_cb = &pic.cb[y * pic.chroma_stride..y * pic.chroma_stride + cwc];
            let src_cr = &pic.cr[y * pic.chroma_stride..y * pic.chroma_stride + cwc];
            let dst_cb = &mut cb_plane[y * cwc..(y + 1) * cwc];
            let dst_cr = &mut cr_plane[y * cwc..(y + 1) * cwc];
            for i in 0..cwc {
                dst_cb[i] = src_cb[i].min(255) as u8;
                dst_cr[i] = src_cr[i].min(255) as u8;
            }
        }
        let _ = bit_depth_c;
        VideoFrame {
            format: PixelFormat::Yuv420P,
            width: cw as u32,
            height: ch as u32,
            pts: self.last_pts,
            time_base: self.last_time_base,
            planes: vec![
                VideoPlane {
                    stride: cw,
                    data: y_plane,
                },
                VideoPlane {
                    stride: cwc,
                    data: cb_plane,
                },
                VideoPlane {
                    stride: cwc,
                    data: cr_plane,
                },
            ],
        }
    }

    fn active_sps_pps_for(
        &self,
        nal: &crate::nal::NalHeader,
        rbsp: &[u8],
    ) -> Option<(SeqParameterSet, PicParameterSet)> {
        let pps_id = peek_slice_pps_id(nal, rbsp)?;
        let pps = self.pps.get(&pps_id).cloned()?;
        let sps = self.sps.get(&pps.pps_seq_parameter_set_id).cloned()?;
        Some((sps, pps))
    }
}

fn peek_slice_pps_id(nal: &crate::nal::NalHeader, rbsp: &[u8]) -> Option<u32> {
    use crate::bitreader::BitReader;
    let mut br = BitReader::new(rbsp);
    let _first_slice_segment_in_pic_flag = br.u1().ok()?;
    if nal.nal_unit_type.is_irap() {
        let _no_output_of_prior_pics_flag = br.u1().ok()?;
    }
    br.ue().ok()
}

impl Decoder for HevcDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.last_pts = packet.pts;
        self.last_time_base = packet.time_base;
        let nals: Vec<NalRef<'_>> = match self.length_size {
            Some(n) => iter_length_prefixed(&packet.data, n)?,
            None => iter_annex_b(&packet.data).collect(),
        };
        self.process_nals(&nals)?;
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(vf) = self.pending.pop_front() {
            return Ok(Frame::Video(vf));
        }
        if self.eof {
            return Err(Error::Eof);
        }
        if let Some(s) = &self.last_slice {
            if !s.is_full_i_slice && !s.is_full_p_slice && !s.is_full_b_slice {
                return Err(Error::unsupported(
                    "h265 slice shape not yet supported (tiles/wavefront/extension)",
                ));
            }
        }
        Err(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.last_slice = None;
        self.pending.clear();
        self.eof = false;
        self.dpb = Dpb::new(8);
        self.prev_poc_msb = 0;
        self.prev_poc_lsb = 0;
        Ok(())
    }
}

/// Apply §8.3.4 ref-picture-list reordering. When `mod_flag == 0` the
/// default RefPicListTemp (the first `active` entries) is returned
/// unchanged. When `mod_flag == 1`, each output slot `i` is filled with
/// `temp[list_entry[i]]`. Out-of-range indices are clamped so the decoder
/// is robust against malformed streams — the spec forbids them, but
/// silently clamping beats an erratic panic.
fn apply_list_modification(
    temp: &[RefPicture],
    mod_flag: bool,
    list_entry: &[u32],
    active: usize,
) -> Vec<RefPicture> {
    if !mod_flag {
        let mut out = temp.to_vec();
        if out.len() > active {
            out.truncate(active);
        }
        return out;
    }
    if temp.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<RefPicture> = Vec::with_capacity(active);
    for i in 0..active {
        let idx = list_entry.get(i).copied().unwrap_or(0) as usize;
        let idx = idx.min(temp.len() - 1);
        out.push(temp[idx].clone());
    }
    out
}
