//! `HevcEncoder` — wires the VPS / SPS / PPS / slice emitters into the
//! `oxideav_core::Encoder` trait.
//!
//! Scope:
//!
//! * First frame of every GOP is an IDR I-slice with the access unit
//!   consisting of (VPS, SPS, PPS, IDR_slice) Annex B NAL units.
//! * Subsequent frames in the same GOP are TrailR slices. By default
//!   every non-key frame is a P-slice referencing the previously
//!   reconstructed frame (L0 only). When the encoder is constructed
//!   with `with_mini_gop_size(2)` (round 22), the GOP follows the
//!   I-P-B-P-B decode-order pattern: every other display position is a
//!   B-slice with one past + one future reference. The caller still
//!   delivers frames in display order; the encoder reorders them
//!   internally.
//! * Picture dimensions must be a multiple of the 16-pixel CTU size.
//! * Only 8-bit 4:2:0 (`PixelFormat::Yuv420P`).

use std::collections::VecDeque;

use oxideav_core::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Rational, Result,
    TimeBase, VideoFrame,
};

use crate::encoder::b_slice_writer::build_b_slice_with_reconstruction;
use crate::encoder::p_slice_writer::{
    build_p_slice_with_reconstruction, build_p_slice_with_reconstruction_delta, ReferenceFrame,
};
use crate::encoder::params::{
    build_pps_nal, build_sps_nal, build_vps_nal_with_bit_depth, build_vps_nal_with_profile,
    EncoderConfig,
};
use crate::encoder::slice_writer::build_idr_slice_nal_with_reconstruction;
use crate::encoder::slice_writer_main10::build_idr_slice_nal_main10;
use crate::encoder::slice_writer_main12::build_idr_slice_nal_main12;
use crate::encoder::slice_writer_main444::build_idr_slice_nal_main444_8;
use crate::encoder::slice_writer_main444_10::build_idr_slice_nal_main444_10;

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
    /// Frame counter — number of source frames the caller has sent so far.
    /// In display order (one increment per `send_frame`).
    frame_count: u32,
    /// Last reconstruction retained as the L0 reference for the next P frame.
    last_recon: Option<ReferenceFrame>,
    /// POC of the picture in `last_recon` (display-order POC).
    last_recon_poc: u32,
    /// Mini-GOP size (1 = P-only, 2 = I-P-B-P-B with one B between each
    /// pair of P/I anchor frames).
    mini_gop_size: u32,
    /// Frame held back in display order while we wait for the next anchor
    /// (P or I) so the held frame can be emitted as a B-slice. Only used
    /// when `mini_gop_size > 1`.
    pending_b: Option<(VideoFrame, u32 /* display-order POC */)>,
}

impl HevcEncoder {
    pub fn from_params(params: &CodecParameters) -> Result<Self> {
        Self::from_params_with_mini_gop(params, 1)
    }

    /// Construct a B-slice-enabled encoder with the given mini-GOP size.
    /// `mini_gop = 1` is P-only (legacy). `mini_gop = 2` interleaves one
    /// B-slice between each pair of anchor frames.
    pub fn from_params_with_mini_gop(params: &CodecParameters, mini_gop: u32) -> Result<Self> {
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
        // Round 25 / 26 / 30 / 31: accept Yuv420P (Main, 8-bit),
        // Yuv420P10Le (Main 10), Yuv420P12Le (Main 12 / RExt),
        // Yuv444P (Main 4:4:4 8-bit), and Yuv444P10Le (Main 4:4:4 10).
        // The high-bit-depth + 4:4:4 paths currently only cover the IDR
        // I-slice path — B / P slices at non-Main configs will fail on
        // the next `send_frame` with a clear error (the mini_gop check
        // below also rejects mini_gop > 1 outside the 8-bit 4:2:0 path).
        let (bit_depth, chroma_format_idc) = match pix {
            PixelFormat::Yuv420P => (8u32, 1u32),
            PixelFormat::Yuv420P10Le => (10u32, 1u32),
            PixelFormat::Yuv420P12Le => (12u32, 1u32),
            PixelFormat::Yuv444P => (8u32, 3u32),
            PixelFormat::Yuv444P10Le => (10u32, 3u32),
            _ => {
                return Err(Error::unsupported(format!(
                    "h265 encoder: only Yuv420P / Yuv420P10Le / Yuv420P12Le / \
                     Yuv444P / Yuv444P10Le are supported (got {pix:?})"
                )));
            }
        };
        if mini_gop == 0 || mini_gop > 2 {
            return Err(Error::unsupported(format!(
                "h265 encoder: mini_gop_size must be 1 or 2 (got {mini_gop})"
            )));
        }
        if bit_depth == 10 && mini_gop > 1 {
            return Err(Error::unsupported(
                "h265 encoder: Main 10 (Yuv420P10Le) only supports mini_gop_size = 1 \
                 (and currently emits keyframe-only — P/B at 10 bit is a follow-up)",
            ));
        }
        if bit_depth == 12 && mini_gop > 1 {
            return Err(Error::unsupported(
                "h265 encoder: Main 12 (Yuv420P12Le) only supports mini_gop_size = 1 \
                 (and currently emits keyframe-only — P/B at 12 bit is a follow-up)",
            ));
        }
        if chroma_format_idc == 3 && mini_gop > 1 {
            return Err(Error::unsupported(
                "h265 encoder: Main 4:4:4 (Yuv444P) only supports mini_gop_size = 1 \
                 (currently keyframe-only — P/B at 4:4:4 is a follow-up)",
            ));
        }
        let frame_rate = params.frame_rate.unwrap_or(Rational::new(30, 1));

        let mut output_params = params.clone();
        output_params.media_type = MediaType::Video;
        output_params.codec_id = CodecId::new(super::super::CODEC_ID_STR);
        output_params.width = Some(width);
        output_params.height = Some(height);
        output_params.pixel_format = Some(pix);
        output_params.frame_rate = Some(frame_rate);

        let time_base = TimeBase::new(frame_rate.den.max(1), frame_rate.num.max(1));
        let cfg = match (bit_depth, chroma_format_idc) {
            (8, 3) => EncoderConfig::new_main444_8(width, height),
            (10, 3) => EncoderConfig::new_main444_10(width, height),
            (12, _) => EncoderConfig::new_main12(width, height),
            (10, _) => EncoderConfig::new_main10(width, height),
            _ => EncoderConfig::new(width, height),
        };

        let mut vps_sps_pps = Vec::new();
        // The VPS PTL must declare the same profile as the SPS; ffmpeg's
        // HEVC parser cross-checks the two and refuses streams where they
        // differ. The 4:2:0 helper preserves the round-1..29 VPS bytes
        // byte-for-byte; the new `with_profile` overload threads through
        // chroma_format_idc to engage Main 4:4:4 (round 30).
        if chroma_format_idc == 3 {
            vps_sps_pps.extend_from_slice(&build_vps_nal_with_profile(bit_depth, 3));
        } else {
            vps_sps_pps.extend_from_slice(&build_vps_nal_with_bit_depth(bit_depth));
        }
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
            last_recon_poc: 0,
            mini_gop_size: mini_gop,
            pending_b: None,
        })
    }

    /// Emit an I-slice access unit (VPS+SPS+PPS+IDR) for `frame` at POC 0.
    fn emit_idr(&mut self, frame: &VideoFrame) -> Packet {
        let mut data: Vec<u8> = Vec::with_capacity(
            self.vps_sps_pps.len() + 64 + (self.cfg.width * self.cfg.height) as usize * 3 / 2 + 32,
        );
        data.extend_from_slice(&self.vps_sps_pps);
        match (self.cfg.bit_depth, self.cfg.chroma_format_idc) {
            (8, 3) => {
                // Main 4:4:4 IDR — see `slice_writer_main444`. Keyframe
                // only at 4:4:4; we do not cache a reconstruction since
                // P/B at 4:4:4 is not yet implemented.
                let nal = build_idr_slice_nal_main444_8(&self.cfg, frame);
                data.extend_from_slice(&nal);
                self.last_recon = None;
                self.last_recon_poc = 0;
            }
            (10, 3) => {
                // Main 4:4:4 10 IDR — see `slice_writer_main444_10`.
                // Combines the 4:4:4 chroma topology of the round-30
                // 8-bit 4:4:4 path with the 10-bit precision of the
                // round-25 Main 10 writer. Keyframe only — no
                // reconstruction is cached since P/B at 4:4:4 is a
                // follow-up.
                let nal = build_idr_slice_nal_main444_10(&self.cfg, frame);
                data.extend_from_slice(&nal);
                self.last_recon = None;
                self.last_recon_poc = 0;
            }
            (12, _) => {
                // Main 12 IDR — see `slice_writer_main12`. We do NOT cache
                // a P-slice reference since 12-bit P/B is not yet
                // implemented; every frame this encoder emits at 12-bit
                // is currently a keyframe.
                let nal = build_idr_slice_nal_main12(&self.cfg, frame);
                data.extend_from_slice(&nal);
                self.last_recon = None;
                self.last_recon_poc = 0;
            }
            (10, _) => {
                // Main 10 IDR — see `slice_writer_main10`. We do NOT cache a
                // P-slice reference since 10-bit P/B is not yet implemented;
                // every frame this encoder emits at 10-bit is currently a
                // keyframe (the GOP gate forces an IDR every `GOP_SIZE` frames
                // in 8-bit mode; in 10-bit mode the next non-keyframe will
                // surface `Error::Unsupported` in `send_frame`).
                let nal = build_idr_slice_nal_main10(&self.cfg, frame);
                data.extend_from_slice(&nal);
                self.last_recon = None;
                self.last_recon_poc = 0;
            }
            _ => {
                let (nal, recon) = build_idr_slice_nal_with_reconstruction(&self.cfg, frame);
                data.extend_from_slice(&nal);
                self.last_recon = Some(recon);
                self.last_recon_poc = 0;
            }
        }
        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = frame.pts;
        pkt.dts = frame.pts;
        pkt.flags.keyframe = true;
        pkt
    }

    /// Emit a P-slice referencing `self.last_recon`. `display_poc` is the
    /// frame's display-order POC; the frame index inside the GOP is the
    /// same value because the GOP starts at display POC 0.
    fn emit_p(&mut self, frame: &VideoFrame, display_poc: u32) -> Packet {
        let ref_frame = self.last_recon.clone().expect("P-slice without ref");
        let delta_l0 = self.last_recon_poc as i32 - display_poc as i32;
        let (nal, recon) = if delta_l0 == -1 {
            // Default delta = -1 path keeps the original P-only emission
            // byte-for-byte identical (regression guard for the existing
            // P-slice / cross-decode tests).
            build_p_slice_with_reconstruction(&self.cfg, frame, display_poc, &ref_frame)
        } else {
            build_p_slice_with_reconstruction_delta(
                &self.cfg,
                frame,
                display_poc,
                delta_l0,
                &ref_frame,
            )
        };
        let mut data: Vec<u8> = Vec::with_capacity(nal.len() + 16);
        data.extend_from_slice(&nal);
        self.last_recon = Some(recon);
        self.last_recon_poc = display_poc;
        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = frame.pts;
        pkt.dts = frame.pts;
        pkt.flags.keyframe = false;
        pkt
    }

    /// Emit a B-slice. `b_frame` is the held-back source; `b_poc` is its
    /// display-order POC (between the L0 ref and the just-emitted P/I
    /// anchor's POC). `l0` and `l1` are the reconstructions used for
    /// motion compensation.
    fn emit_b(
        &mut self,
        b_frame: &VideoFrame,
        b_poc: u32,
        l0_poc: u32,
        l1_poc: u32,
        l0: &ReferenceFrame,
        l1: &ReferenceFrame,
    ) -> Packet {
        let delta_l0 = l0_poc as i32 - b_poc as i32; // negative
        let delta_l1 = l1_poc as i32 - b_poc as i32; // positive
        let (nal, _recon) = build_b_slice_with_reconstruction(
            &self.cfg, b_frame, b_poc, delta_l0, delta_l1, l0, l1,
        );
        // The B-frame is non-reference (TrailR is fine here per spec; we
        // don't store its reconstruction back into `last_recon` because
        // the next anchor P-slice already has its own L0 = the previous
        // anchor's reconstruction).
        let mut data: Vec<u8> = Vec::with_capacity(nal.len() + 16);
        data.extend_from_slice(&nal);
        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = b_frame.pts;
        pkt.dts = b_frame.pts;
        pkt.flags.keyframe = false;
        pkt
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
        if vf.planes.len() != 3 {
            return Err(Error::invalid("h265 encoder: expected 3 planes"));
        }

        let display_poc = self.frame_count;
        let is_keyframe = display_poc % GOP_SIZE == 0 || self.last_recon.is_none();

        if is_keyframe {
            // IDR resets the GOP. Any pending B (shouldn't happen if the
            // caller respects mini-GOP boundaries, but be defensive) is
            // dropped — its references would be invalidated by the IDR.
            self.pending_b = None;
            let pkt = self.emit_idr(vf);
            self.pending.push_back(pkt);
        } else if self.mini_gop_size == 1 {
            // P-only path — encode immediately referencing last_recon.
            let pkt = self.emit_p(vf, display_poc);
            self.pending.push_back(pkt);
        } else {
            // mini_gop == 2: anchor-pair pattern. Display-order positions:
            //   0 (anchor / I), 1 (B), 2 (anchor / P), 3 (B), 4 (anchor / P), ...
            // Decode-order: I, P_at_poc2, B_at_poc1, P_at_poc4, B_at_poc3, ...
            // i.e. odd display POCs are B, even ones are anchors.
            let is_anchor = display_poc % 2 == 0;
            if is_anchor {
                // This is the next P-anchor. Emit it FIRST so the
                // intervening B-frame can reference it as L1.
                let p_poc = display_poc;
                // Stash the B-frame (if any) before we mutate last_recon.
                let pending_b = self.pending_b.take();
                let l0_poc = self.last_recon_poc;
                let l0_for_b = self.last_recon.clone();
                let p_pkt = self.emit_p(vf, p_poc);
                self.pending.push_back(p_pkt);
                // Now emit the held B-frame referencing the previous
                // anchor (L0) and the just-emitted P (L1).
                if let Some((b_frame, b_poc)) = pending_b {
                    let l0 = l0_for_b.expect("B-slice without L0 ref");
                    let l1 = self
                        .last_recon
                        .as_ref()
                        .expect("B-slice without L1 ref")
                        .clone();
                    let b_pkt = self.emit_b(&b_frame, b_poc, l0_poc, p_poc, &l0, &l1);
                    self.pending.push_back(b_pkt);
                }
            } else {
                // Hold back as the pending B; it will be emitted right
                // after the next anchor P.
                self.pending_b = Some((vf.clone(), display_poc));
            }
        }
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
        // If a B-frame is still buffered at flush, we don't have a future
        // anchor to bipredict from; degrade it to a P-slice referencing
        // the previous anchor (L0 only). This keeps the stream
        // self-consistent without losing the source frame.
        if let Some((b_frame, b_poc)) = self.pending_b.take() {
            let pkt = self.emit_p(&b_frame, b_poc);
            self.pending.push_back(pkt);
        }
        self.eof = true;
        Ok(())
    }
}
