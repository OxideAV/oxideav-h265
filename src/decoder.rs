//! HEVC decoder glue.
//!
//! Ties the parameter-set and slice-header parsers to the CTU pipeline and
//! exposes an `oxideav_codec::Decoder` implementation. Scope:
//!
//! * **I slice, 8-bit 4:2:0** — full pixel decode. Reconstructed luma and
//!   chroma are emitted as a `VideoFrame` (pixel format `Yuv420P`).
//! * **Everything else** — returns `Error::Unsupported("h265 inter slice
//!   pending")` (for P/B slices) or a specific unsupported message
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
                        // If this is an IDR I-slice and we have a full header,
                        // attempt the full CTU decode. Otherwise surface
                        // appropriate `Unsupported` on `receive_frame`.
                        let can_decode = hdr.is_full_i_slice && hdr.slice_type == SliceType::I;
                        self.last_slice = Some(hdr.clone());
                        if can_decode {
                            match self.decode_i_slice(&rbsp, &hdr, &sps, &pps) {
                                Ok(frame) => self.pending.push_back(frame),
                                Err(Error::Unsupported(msg)) => {
                                    // Feature gap: bubble it up later.
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

    fn decode_i_slice(
        &self,
        rbsp: &[u8],
        hdr: &SliceSegmentHeader,
        sps: &SeqParameterSet,
        pps: &PicParameterSet,
    ) -> Result<VideoFrame> {
        let width = sps.pic_width_in_luma_samples;
        let height = sps.pic_height_in_luma_samples;
        let cropped_w = sps.cropped_width();
        let cropped_h = sps.cropped_height();
        let mut pic = Picture::new(width, height);
        let cctx = CtuContext {
            sps,
            pps,
            slice: hdr,
            init_type: InitType::I,
        };
        let byte_off = (hdr.slice_data_bit_offset / 8) as usize;
        decode_slice_ctus(rbsp, byte_off, &cctx, &mut pic)?;
        // Build a VideoFrame from pic. Apply cropping by narrowing the
        // emitted plane views to the conformance-window extent.
        let (cw, ch) = (cropped_w as usize, cropped_h as usize);
        let (w, h) = (width as usize, height as usize);
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
        let _ = (w, h);
        Ok(VideoFrame {
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
        })
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
        // (e.g. P/B slice) versus "no slice yet".
        if let Some(s) = &self.last_slice {
            if s.slice_type != SliceType::I {
                return Err(Error::unsupported("h265 inter slice pending"));
            }
            if !s.is_full_i_slice {
                return Err(Error::unsupported(
                    "h265 I-slice shape not yet supported (tiles/wavefront/extension)",
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
        Ok(())
    }
}
