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

// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod bitwriter;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod cabac;
pub mod inter;
pub mod intra;
// internal — exposed for tests/fuzz; not part of the stable API
#[cfg(test)]
mod ccp_streams;
#[doc(hidden)]
pub mod nal;
#[cfg(test)]
mod palette_streams;
pub mod pcm;
#[cfg(test)]
mod rdpcm_streams;
#[cfg(test)]
mod scc_streams;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod residual;

/// Registry-encoder coding mode, selected by the `mode` codec option.
#[derive(Debug)]
enum EncodeMode {
    /// PCM-only IDR bootstrap: bit-exact lossless, every CU a
    /// §7.3.8.7 PCM block. The default (`mode` absent or `"pcm"`).
    Pcm,
    /// Real CABAC intra coding ([`intra::encode_idr_intra_au`]) at
    /// the carried `SliceQpY` (`mode = "intra"`, `qp` option, default
    /// 26).
    Intra(i32),
    /// Low-delay inter coding (`mode = "inter"`): `IDR, P, P, …`
    /// GOPs through [`inter::LowDelayPEncoder`] (`qp` option, default
    /// 26; `gop` option = GOP length in frames, default 0 = a single
    /// leading IDR).
    Inter(inter::LowDelayPEncoder),
}

use std::collections::VecDeque;

use oxideav_core::{
    CodecId, CodecParameters, Encoder, Error, Frame, Packet, PixelFormat, Result, TimeBase,
};

/// H.265 registry encoder behind the [`oxideav_core::Encoder`]
/// contract. Every input frame becomes one Annex B access unit — a
/// self-contained IDR (`VPS + SPS + PPS + IDR_N_LP`) in the two intra
/// modes and at inter-mode GOP starts, or a `TRAIL_R` P slice inside
/// an inter-mode GOP. Three coding modes, selected by the `mode`
/// codec option:
///
/// * `"pcm"` (default) — the PCM-only bootstrap: every coding unit a
///   §7.3.8.7 PCM block, bit-exact lossless.
/// * `"intra"` — real CABAC intra coding (§8.4 prediction + forward
///   transform/quant + §7.3.8 syntax) at the `qp` option's
///   `SliceQpY` (0..=51, default 26).
/// * `"inter"` — low-delay `IDR, P, P, …` coding (§8.5 inter
///   prediction with per-CTU skip / merge / AMVP / intra decisions)
///   at the `qp` option's `SliceQpY`; the `gop` option sets the IDR
///   period in frames (default 0 = a single leading IDR); the
///   `bslices` option (`"1"` / `"true"`) codes the non-IDR frames as
///   low-delay B slices.
///
/// 4:2:0 8-bit, dimensions multiples of 16.
pub struct H265Encoder {
    codec_id: CodecId,
    output_params: CodecParameters,
    width: usize,
    height: usize,
    mode: EncodeMode,
    ready: VecDeque<Packet>,
    frame_index: i64,
}

/// Former name of [`H265Encoder`] (from when the registry encoder was
/// PCM-only).
pub type H265PcmEncoder = H265Encoder;

impl std::fmt::Debug for H265Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("H265Encoder")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("mode", &self.mode)
            .field("ready", &self.ready.len())
            .finish()
    }
}

/// Direct factory endpoint: construct the software H.265 encoder.
/// Codec options: `mode` (`"pcm"` lossless bootstrap, the default,
/// `"intra"` real CABAC intra coding, or `"inter"` low-delay P GOPs),
/// `qp` (`SliceQpY` 0..=51 for the intra / inter modes, default 26)
/// and `gop` (inter mode IDR period, default 0 = single leading
/// IDR).
///
/// # Errors
/// [`Error::InvalidData`] when width / height are missing or not
/// nonzero multiples of 16, the pixel format is declared and is not
/// 4:2:0 8-bit planar, or a codec option is malformed.
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
    let parse_qp = |params: &CodecParameters| -> Result<i32> {
        match params.options.get("qp") {
            None => Ok(26),
            Some(v) => v
                .parse::<i32>()
                .ok()
                .filter(|q| (0..=51).contains(q))
                .ok_or_else(|| {
                    Error::InvalidData(format!("h265 encode: qp must be 0..=51, got {v:?}"))
                }),
        }
    };
    let mode = match params.options.get("mode") {
        None | Some("pcm") => EncodeMode::Pcm,
        Some("intra") => EncodeMode::Intra(parse_qp(params)?),
        Some("inter") => {
            let qp = parse_qp(params)?;
            let gop = match params.options.get("gop") {
                None => 0usize,
                Some(v) => v.parse::<usize>().map_err(|_| {
                    Error::InvalidData(format!(
                        "h265 encode: gop must be a non-negative integer, got {v:?}"
                    ))
                })?,
            };
            let b_slices = match params.options.get("bslices") {
                None | Some("0") | Some("false") => false,
                Some("1") | Some("true") => true,
                Some(v) => {
                    return Err(Error::InvalidData(format!(
                        "h265 encode: bslices must be 0/1/true/false, got {v:?}"
                    )))
                }
            };
            let enc = inter::LowDelayPEncoder::new(width, height, qp, gop)
                .map_err(|e| Error::InvalidData(format!("h265 encode: {e}")))?
                .with_b_slices(b_slices);
            EncodeMode::Inter(enc)
        }
        Some(other) => {
            return Err(Error::InvalidData(format!(
                "h265 encode: unknown mode {other:?} (expected \"pcm\", \"intra\" or \"inter\")"
            )))
        }
    };
    let mut output_params = params.clone();
    output_params.media_type = oxideav_core::MediaType::Video;
    output_params.pixel_format = Some(PixelFormat::Yuv420P);
    // Parameter sets ride in band in every access unit; no extradata.
    output_params.extradata.clear();
    Ok(Box::new(H265Encoder {
        codec_id: params.codec_id.clone(),
        output_params,
        width,
        height,
        mode,
        ready: VecDeque::new(),
        frame_index: 0,
    }))
}

impl Encoder for H265Encoder {
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

        let (au, keyframe) = match &mut self.mode {
            EncodeMode::Pcm => (
                pcm::encode_idr_pcm_au(&y, &cb, &cr, self.width, self.height)
                    .map_err(|e| Error::InvalidData(format!("h265 encode: {e}")))?,
                true,
            ),
            EncodeMode::Intra(qp) => (
                intra::encode_idr_intra_au(&y, &cb, &cr, self.width, self.height, *qp)
                    .map_err(|e| Error::InvalidData(format!("h265 encode: {e}")))?
                    .au,
                true,
            ),
            EncodeMode::Inter(enc) => {
                let f = enc
                    .encode_frame(&inter::YuvFrame {
                        y: &y,
                        cb: &cb,
                        cr: &cr,
                    })
                    .map_err(|e| Error::InvalidData(format!("h265 encode: {e}")))?;
                (f.au, f.keyframe)
            }
        };
        let mut pkt = Packet::new(0, TimeBase::new(1, 25), au);
        pkt.pts = v.pts.or(Some(self.frame_index));
        pkt.dts = pkt.pts;
        pkt.flags.keyframe = keyframe;
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
