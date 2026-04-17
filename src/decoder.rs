//! HEVC decoder scaffold.
//!
//! Parses VPS / SPS / PPS / slice headers from the input stream and the
//! `extradata` (HEVCDecoderConfigurationRecord) but **does not** decode the
//! coded picture itself. The actual CTU pipeline — CABAC entropy decode,
//! intra / inter prediction, transforms, loop filtering — is not yet
//! implemented; `receive_frame` returns
//! `Error::Unsupported("hevc CTU decode not yet implemented")`.
//!
//! What *is* exercised end-to-end for an I-slice packet:
//! 1. Annex B / length-prefixed NAL framing.
//! 2. VPS / SPS / PPS capture.
//! 3. Slice segment header parse against the active PPS/SPS pair.
//! 4. Derivation of the byte-aligned `slice_data()` offset and the initial
//!    `SliceQpY` (§8.6.1) — surfaced on the last-slice state so callers
//!    and tests can verify the decoder reached the CTU-decode boundary.
//! 5. `receive_frame` then returns the `Unsupported` error above.

use std::collections::HashMap;
use std::collections::VecDeque;

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, Result};

use crate::hvcc::parse_hvcc;
use crate::nal::{extract_rbsp, iter_annex_b, iter_length_prefixed, NalRef, NalUnitType};
use crate::pps::{parse_pps, PicParameterSet};
use crate::slice::{parse_slice_segment_header, SliceSegmentHeader};
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
    /// Length-prefix size from the configuration record, when present. If
    /// `None` the input is treated as Annex B byte-stream.
    length_size: Option<u8>,
    pub vps: HashMap<u8, VideoParameterSet>,
    pub sps: HashMap<u32, SeqParameterSet>,
    pub pps: HashMap<u32, PicParameterSet>,
    pub last_slice: Option<SliceSegmentHeader>,
    pending: VecDeque<()>,
    eof: bool,
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
        }
    }

    /// Consume an HEVCDecoderConfigurationRecord (`hvcC` body) — populates
    /// VPS/SPS/PPS tables and selects the length-prefix size.
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

    /// Parse one NAL given its raw bytes (NAL header + payload, still
    /// emulation-prevented). Updates parameter-set tables and the last
    /// slice header.
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
                    // Slice NAL — parse the header so we surface acceptably
                    // useful state to callers, but stop short of any actual
                    // pixel decode.
                    if let Some((sps, pps)) = self.active_sps_pps_for(&nal.header, &rbsp) {
                        let hdr = parse_slice_segment_header(&rbsp, &nal.header, &sps, &pps)?;
                        self.last_slice = Some(hdr);
                    }
                }
                _ => {
                    // AUD, SEI, EOS, EOB, filler — ignored in v1.
                }
            }
        }
        Ok(())
    }

    /// Resolve the SPS/PPS pair active for a slice whose RBSP starts with
    /// `first_slice_segment_in_pic_flag` followed by an optional
    /// `no_output_of_prior_pics_flag` bit (for IRAP NALs) and then
    /// `slice_pic_parameter_set_id` as ue(v). Needs the NAL header to know
    /// whether the extra IRAP bit is present.
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

/// Peek the PPS id from the start of a slice segment header RBSP. Reads
/// exactly the syntax elements §7.3.6.1 specifies up to
/// `slice_pic_parameter_set_id`. Returns `None` if the bitstream is too
/// short or malformed.
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
        // Decide framing: if extradata gave us a length size, treat samples
        // as length-prefixed; otherwise scan for Annex B start codes.
        let nals: Vec<NalRef<'_>> = match self.length_size {
            Some(n) => iter_length_prefixed(&packet.data, n)?,
            None => iter_annex_b(&packet.data).collect(),
        };
        self.process_nals(&nals)?;
        // We never emit a frame.
        let _ = self.pending.len();
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if self.eof {
            return Err(Error::Eof);
        }
        Err(Error::unsupported("hevc CTU decode not yet implemented"))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        // Scaffold decoder — pixel reconstruction is unsupported, so the
        // only state that could stale is the last parsed slice header and
        // any transient pending markers. VPS/SPS/PPS tables + the
        // `length_size` from hvcC are stream-level and must survive the
        // reset or the decoder would reject subsequent slices.
        self.last_slice = None;
        self.pending.clear();
        self.eof = false;
        Ok(())
    }
}
