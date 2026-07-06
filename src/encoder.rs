//! H.265 / HEVC encoder — write-side building blocks.
//!
//! The encode stack mirrors the decode stack bottom-up:
//!
//! * [`bitwriter::BitWriter`] — the MSB-first RBSP bit sink (`u(n)`,
//!   `ue(v)` / `se(v)`, `rbsp_trailing_bits()`).
//! * [`nal`] — §7.4.1.1 emulation-prevention escaping, the §7.3.1.2
//!   NAL header, and Annex B framing.
//! * [`cabac::CabacEncoder`] — the §9.3.5 arithmetic encoding engine
//!   (the spec's informative encoder that "matches the arithmetic
//!   decoding engine described in clause 9.3.4.3").
//!
//! Every layer is pinned against the crate's own decode side: bit
//! fields re-read through [`crate::bitreader::BitReader`], NAL units
//! re-walked through [`crate::nal`], and CABAC bin streams re-decoded
//! through [`crate::cabac::CabacEngine`].

pub mod bitwriter;
pub mod cabac;
pub mod nal;
pub mod pcm;
pub mod residual;

use std::collections::VecDeque;

use oxideav_core::{
    CodecId, CodecParameters, Encoder, Error, Frame, Packet, PixelFormat, Result, TimeBase,
};

/// H.265 registry encoder — the PCM-only IDR bootstrap behind the
/// [`oxideav_core::Encoder`] contract. Every input frame becomes one
/// self-contained IDR access unit (`VPS + SPS + PPS + IDR_N_LP`, Annex
/// B form) whose coding units are all §7.3.8.7 PCM blocks: the encode
/// is bit-exact lossless and every packet is an independent random
/// access point. 4:2:0 8-bit, dimensions multiples of 16.
pub struct H265PcmEncoder {
    codec_id: CodecId,
    output_params: CodecParameters,
    width: usize,
    height: usize,
    ready: VecDeque<Packet>,
    frame_index: i64,
}

impl std::fmt::Debug for H265PcmEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("H265PcmEncoder")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("ready", &self.ready.len())
            .finish()
    }
}

/// Direct factory endpoint: construct the software H.265 encoder
/// (PCM-only IDR bootstrap).
///
/// # Errors
/// [`Error::InvalidData`] when width / height are missing or not
/// nonzero multiples of 16, or the pixel format is declared and is not
/// 4:2:0 8-bit planar.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::InvalidData("h265 encode: width is required".into()))?
        as usize;
    let height = params
        .height
        .ok_or_else(|| Error::InvalidData("h265 encode: height is required".into()))?
        as usize;
    if width == 0 || height == 0 || width % 16 != 0 || height % 16 != 0 {
        return Err(Error::InvalidData(format!(
            "h265 encode: dimensions must be nonzero multiples of 16, got {width}x{height}"
        )));
    }
    if let Some(pf) = params.pixel_format {
        if pf != PixelFormat::Yuv420P {
            return Err(Error::InvalidData(format!(
                "h265 encode: only yuv420p input is supported, got {pf:?}"
            )));
        }
    }
    let mut output_params = params.clone();
    output_params.media_type = oxideav_core::MediaType::Video;
    output_params.pixel_format = Some(PixelFormat::Yuv420P);
    // Parameter sets ride in band in every access unit; no extradata.
    output_params.extradata.clear();
    Ok(Box::new(H265PcmEncoder {
        codec_id: params.codec_id.clone(),
        output_params,
        width,
        height,
        ready: VecDeque::new(),
        frame_index: 0,
    }))
}

impl Encoder for H265PcmEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let v = match frame {
            Frame::Video(v) => v,
            _ => return Err(Error::InvalidData("h265 encode: video frames only".into())),
        };
        if v.planes.len() != 3 {
            return Err(Error::InvalidData(format!(
                "h265 encode: expected 3 planes (yuv420p), got {}",
                v.planes.len()
            )));
        }
        // Repack each plane row-by-row (strides may exceed the width).
        let pack = |idx: usize, w: usize, h: usize| -> Result<Vec<u8>> {
            let plane = &v.planes[idx];
            if plane.stride < w || plane.data.len() < plane.stride * h {
                return Err(Error::InvalidData(format!(
                    "h265 encode: plane {idx} too small (stride {}, len {})",
                    plane.stride,
                    plane.data.len()
                )));
            }
            let mut out = Vec::with_capacity(w * h);
            for row in 0..h {
                out.extend_from_slice(&plane.data[row * plane.stride..row * plane.stride + w]);
            }
            Ok(out)
        };
        let y = pack(0, self.width, self.height)?;
        let cb = pack(1, self.width / 2, self.height / 2)?;
        let cr = pack(2, self.width / 2, self.height / 2)?;

        let au = pcm::encode_idr_pcm_au(&y, &cb, &cr, self.width, self.height)
            .map_err(|e| Error::InvalidData(format!("h265 encode: {e}")))?;
        let mut pkt = Packet::new(0, TimeBase::new(1, 25), au);
        pkt.pts = v.pts.or(Some(self.frame_index));
        pkt.dts = pkt.pts;
        pkt.flags.keyframe = true;
        self.frame_index += 1;
        self.ready.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.ready.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        // Every frame is emitted eagerly; nothing is buffered.
        Ok(())
    }
}
