//! HEVC decoder glue.
//!
//! Ties the parameter-set and slice-header parsers to the CTU pipeline and
//! exposes an `oxideav_codec::Decoder` implementation. Scope:
//!
//! * **I slice / P slice, 8-bit 4:2:0** — full pixel decode. Reconstructed
//!   luma and chroma are emitted as a `VideoFrame` (pixel format
//!   `Yuv420P`). P slices pull references from a small DPB keyed by POC.
//! * **Everything else** — returns `Error::Unsupported("h265 B-slice
//!   decode pending")` (for B slices) or a specific unsupported message
//!   surfacing the feature that is not yet implemented.

use std::collections::HashMap;
use std::collections::VecDeque;

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, TimeBase, VideoFrame,
    VideoPlane,
};

use crate::cabac::InitType;
use crate::ctu::{decode_slice_ctus, CtuContext, Picture};
use crate::hvcc::parse_hvcc;
use crate::inter::{Dpb, RefPicture};
use crate::nal::{extract_rbsp, iter_annex_b, iter_length_prefixed, NalRef, NalUnitType};
use crate::pps::{parse_pps, PicParameterSet};
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
                        } else if hdr.is_full_p_slice && hdr.slice_type == SliceType::P {
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
        };
        let byte_off = (hdr.slice_data_bit_offset / 8) as usize;
        decode_slice_ctus(rbsp, byte_off, &cctx, &mut pic)?;

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
        // Build RPL0 from the active RPS.
        let current_poc = self.derive_poc(sps, hdr, false);
        let rpl0 = self.build_rpl0(hdr, current_poc);
        if rpl0.is_empty() {
            return Err(Error::unsupported(
                "h265 inter slice: no reference pictures",
            ));
        }

        let mut pic = Picture::new(width, height);
        let init_type = crate::cabac::InitType::for_slice(false, false, hdr.cabac_init_flag);
        let cctx = CtuContext {
            sps,
            pps,
            slice: hdr,
            init_type,
            ref_list_l0: &rpl0,
        };
        let byte_off = (hdr.slice_data_bit_offset / 8) as usize;
        decode_slice_ctus(rbsp, byte_off, &cctx, &mut pic)?;

        self.store_ref_pic(&pic, current_poc);
        Ok(self.emit_frame(&pic, sps))
    }

    fn build_rpl0(&self, hdr: &SliceSegmentHeader, current_poc: i32) -> Vec<RefPicture> {
        let Some(rps) = hdr.current_rps.as_ref() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        // Negative-POC entries first (closer in display order), then positive.
        for (i, delta) in rps.delta_poc_s0.iter().enumerate() {
            if !rps.used_by_curr_pic_s0.get(i).copied().unwrap_or(false) {
                continue;
            }
            let target = current_poc + delta;
            if let Some(p) = self.dpb.get_by_poc(target) {
                out.push(p.clone());
            }
        }
        for (i, delta) in rps.delta_poc_s1.iter().enumerate() {
            if !rps.used_by_curr_pic_s1.get(i).copied().unwrap_or(false) {
                continue;
            }
            let target = current_poc + delta;
            if let Some(p) = self.dpb.get_by_poc(target) {
                out.push(p.clone());
            }
        }
        // If the slice requests a larger num_ref_idx_l0_active than we
        // resolved, leave as-is — out-of-range ref_idx will trip a clean
        // error at CTU walk time.
        let active = (hdr.num_ref_idx_l0_active_minus1 + 1) as usize;
        if out.len() > active {
            out.truncate(active);
        }
        out
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
        });
    }

    fn emit_frame(&self, pic: &Picture, sps: &SeqParameterSet) -> VideoFrame {
        let cropped_w = sps.cropped_width();
        let cropped_h = sps.cropped_height();
        let (cw, ch) = (cropped_w as usize, cropped_h as usize);
        let (cwc, chc) = (cw / 2, ch / 2);
        let mut y_plane = vec![0u8; cw * ch];
        let mut cb_plane = vec![0u8; cwc * chc];
        let mut cr_plane = vec![0u8; cwc * chc];
        for y in 0..ch {
            y_plane[y * cw..(y + 1) * cw]
                .copy_from_slice(&pic.luma[y * pic.luma_stride..y * pic.luma_stride + cw]);
        }
        for y in 0..chc {
            cb_plane[y * cwc..(y + 1) * cwc]
                .copy_from_slice(&pic.cb[y * pic.chroma_stride..y * pic.chroma_stride + cwc]);
            cr_plane[y * cwc..(y + 1) * cwc]
                .copy_from_slice(&pic.cr[y * pic.chroma_stride..y * pic.chroma_stride + cwc]);
        }
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
        // Distinguish between "slice already parsed but decode not attempted"
        // (e.g. B slice) versus "no slice yet".
        if let Some(s) = &self.last_slice {
            if s.slice_type == SliceType::B {
                return Err(Error::unsupported("h265 B-slice decode pending"));
            }
            if !s.is_full_i_slice && !s.is_full_p_slice {
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
