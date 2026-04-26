//! `HevcEncoder` — wires the VPS / SPS / PPS / slice emitters into the
//! `oxideav_core::Encoder` trait.
//!
//! Scope:
//!
//! * First frame of every GOP is an IDR I-slice with the access unit
//!   consisting of (VPS, SPS, PPS, IDR_slice) Annex B NAL units. Subsequent
//!   frames in the same GOP are TrailR P-slices that reference the
//!   previously-reconstructed frame (L0 only). The GOP length defaults to
//!   64 to match a typical keyint.
//! * Picture dimensions must be a multiple of the 16-pixel CTU size.
//! * Only 8-bit 4:2:0 (`PixelFormat::Yuv420P`).

use std::collections::VecDeque;

use oxideav_core::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Rational, Result,
    TimeBase,
};

use crate::encoder::p_slice_writer::{build_p_slice_with_reconstruction, ReferenceFrame};
use crate::encoder::params::{build_pps_nal, build_sps_nal, build_vps_nal, EncoderConfig};
use crate::encoder::slice_writer::build_idr_slice_nal_with_reconstruction;

/// Default GOP size (distance between IDR keyframes).
const GOP_SIZE: u32 = 64;

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    Ok(Box::new(HevcEncoder::from_params(params)?))
}

pub struct HevcEncoder {
    output_params: CodecParameters,
    cfg: EncoderConfig,
    pending: VecDeque<Packet>,
    eof: bool,
    time_base: TimeBase,
    /// Cache of the once-per-stream VPS/SPS/PPS Annex B bytes.
    vps_sps_pps: Vec<u8>,
    /// Frame counter — used to pick I vs P and to derive POC LSB.
    frame_count: u32,
    /// Last reconstruction retained as the L0 reference for the next P frame.
    last_recon: Option<ReferenceFrame>,
}

impl HevcEncoder {
    pub fn from_params(params: &CodecParameters) -> Result<Self> {
        let width = params
            .width
            .ok_or_else(|| Error::invalid("h265 encoder: missing width"))?;
        let height = params
            .height
            .ok_or_else(|| Error::invalid("h265 encoder: missing height"))?;
        if width % 16 != 0 || height % 16 != 0 {
            return Err(Error::unsupported(format!(
                "h265 encoder: width and height must be multiples of 16 (got {width}x{height})"
            )));
        }
        let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
        if pix != PixelFormat::Yuv420P {
            return Err(Error::unsupported(format!(
                "h265 encoder: only Yuv420P is supported (got {pix:?})"
            )));
        }
        let frame_rate = params.frame_rate.unwrap_or(Rational::new(30, 1));

        let mut output_params = params.clone();
        output_params.media_type = MediaType::Video;
        output_params.codec_id = CodecId::new(super::super::CODEC_ID_STR);
        output_params.width = Some(width);
        output_params.height = Some(height);
        output_params.pixel_format = Some(PixelFormat::Yuv420P);
        output_params.frame_rate = Some(frame_rate);

        let time_base = TimeBase::new(frame_rate.den.max(1), frame_rate.num.max(1));
        let cfg = EncoderConfig::new(width, height);

        let mut vps_sps_pps = Vec::new();
        vps_sps_pps.extend_from_slice(&build_vps_nal());
        vps_sps_pps.extend_from_slice(&build_sps_nal(&cfg));
        vps_sps_pps.extend_from_slice(&build_pps_nal());

        Ok(Self {
            output_params,
            cfg,
            pending: VecDeque::new(),
            eof: false,
            time_base,
            vps_sps_pps,
            frame_count: 0,
            last_recon: None,
        })
    }
}

impl Encoder for HevcEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let vf = match frame {
            Frame::Video(v) => v,
            _ => return Err(Error::invalid("h265 encoder: video frames only")),
        };
        // Pixel format / dimensions live on the stream's CodecParameters
        // and on `self.cfg`; the slim VideoFrame no longer carries them so
        // we trust the caller to honour the encoder's `output_params()`.
        if vf.planes.len() != 3 {
            return Err(Error::invalid("h265 encoder: expected 3 planes"));
        }

        let is_keyframe = self.frame_count % GOP_SIZE == 0 || self.last_recon.is_none();
        let frame_idx_in_gop = self.frame_count % GOP_SIZE;
        let mut data: Vec<u8> = Vec::with_capacity(
            self.vps_sps_pps.len() + 64 + (self.cfg.width * self.cfg.height) as usize * 3 / 2 + 32,
        );

        if is_keyframe {
            data.extend_from_slice(&self.vps_sps_pps);
            let (nal, recon) = build_idr_slice_nal_with_reconstruction(&self.cfg, vf);
            data.extend_from_slice(&nal);
            self.last_recon = Some(recon);
        } else {
            let ref_frame = self
                .last_recon
                .as_ref()
                .expect("P-slice without L0 reference");
            let (nal, recon) =
                build_p_slice_with_reconstruction(&self.cfg, vf, frame_idx_in_gop, ref_frame);
            data.extend_from_slice(&nal);
            self.last_recon = Some(recon);
        }

        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = vf.pts;
        pkt.dts = vf.pts;
        pkt.flags.keyframe = is_keyframe;
        self.pending.push_back(pkt);
        self.frame_count += 1;
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(p) = self.pending.pop_front() {
            return Ok(p);
        }
        if self.eof {
            Err(Error::Eof)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}
