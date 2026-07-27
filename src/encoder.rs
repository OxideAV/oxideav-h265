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
pub mod loopfilter;
pub mod pyramid;
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
    /// 26) with the carried §8.7 in-loop filters (`deblock` / `sao`
    /// options).
    Intra(i32, loopfilter::LoopFilterCfg),
    /// Low-delay inter coding (`mode = "inter"`): `IDR, P, P, …`
    /// GOPs through [`inter::LowDelayPEncoder`] (`qp` option, default
    /// 26; `gop` option = GOP length in frames, default 0 = a single
    /// leading IDR).
    Inter(inter::LowDelayPEncoder),
    /// Hierarchical-B coding (`mode = "inter"` + `pyramid = <gop>`):
    /// dyadic B pyramids through [`pyramid::PyramidEncoder`] —
    /// out-of-order coding, so packets arrive in bursts per mini-GOP
    /// and [`Encoder::flush`] emits the low-delay tail.
    Pyramid {
        /// The buffering pyramid encoder.
        enc: pyramid::PyramidEncoder,
        /// `log2(gop)` — the dts-behind-pts reorder delay in frames.
        delay: i64,
        /// The caller-supplied pts of each pushed frame, by display
        /// index (`None` ⇒ the display index itself).
        pts_by_display: Vec<Option<i64>>,
        /// Decode-order packet counter (drives dts).
        decode_count: i64,
    },
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
///   low-delay B slices; the `amp` option switches to the AMP stream
///   configuration (`MinCbSizeY == 8`, `amp_enabled_flag == 1`, the
///   four asymmetric PART_2NxnU/2NxnD/nLx2N/nRx2N shapes competing in
///   the inter CU ladder); the `pyramid` option (a power of two in
///   2..=16) switches to hierarchical-B coding — dyadic mini-GOPs
///   coded out of display order with per-layer QP offsets; packets
///   then arrive in bursts per mini-GOP (dts trails pts by
///   `log2(pyramid)` frames) and `flush()` emits the tail. `pyramid`
///   excludes the `gop` / `bslices` options.
///
/// The `"intra"` and `"inter"` modes accept the §8.7 in-loop filter
/// options `deblock` and `sao` (`"1"` / `"true"` to enable): the
/// encoder then signals the filters, elects them per slice / CTB
/// against distortion, and reconstructs through them (the reference
/// path a conforming decoder holds).
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
/// `qp` (`SliceQpY` 0..=51 for the intra / inter modes, default 26),
/// `gop` (inter mode IDR period, default 0 = single leading IDR),
/// `amp` (inter mode: asymmetric motion partitions), and the in-loop
/// filter switches `deblock` / `sao` (intra / inter modes; default
/// off).
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
    let parse_flag = |params: &CodecParameters, key: &str| -> Result<bool> {
        match params.options.get(key) {
            None | Some("0") | Some("false") => Ok(false),
            Some("1") | Some("true") => Ok(true),
            Some(v) => Err(Error::InvalidData(format!(
                "h265 encode: {key} must be 0/1/true/false, got {v:?}"
            ))),
        }
    };
    let parse_lf = |params: &CodecParameters| -> Result<loopfilter::LoopFilterCfg> {
        let sao = parse_flag(params, "sao")?;
        Ok(loopfilter::LoopFilterCfg {
            deblocking: parse_flag(params, "deblock")?,
            sao_luma: sao,
            sao_chroma: sao,
        })
    };
    let mode = match params.options.get("mode") {
        None | Some("pcm") => {
            if parse_lf(params)?.any() {
                return Err(Error::InvalidData(
                    "h265 encode: deblock/sao options require mode \"intra\" or \"inter\"".into(),
                ));
            }
            if parse_flag(params, "amp")? {
                return Err(Error::InvalidData(
                    "h265 encode: the amp option requires mode \"inter\"".into(),
                ));
            }
            EncodeMode::Pcm
        }
        Some("intra") => {
            if parse_flag(params, "amp")? {
                return Err(Error::InvalidData(
                    "h265 encode: the amp option requires mode \"inter\"".into(),
                ));
            }
            EncodeMode::Intra(parse_qp(params)?, parse_lf(params)?)
        }
        Some("inter") => {
            let qp = parse_qp(params)?;
            if let Some(v) = params.options.get("pyramid") {
                if params.options.get("gop").is_some() || params.options.get("bslices").is_some() {
                    return Err(Error::InvalidData(
                        "h265 encode: pyramid excludes the gop / bslices options".into(),
                    ));
                }
                let g = v
                    .parse::<usize>()
                    .ok()
                    .filter(|g| g.is_power_of_two() && (2..=16).contains(g))
                    .ok_or_else(|| {
                        Error::InvalidData(format!(
                            "h265 encode: pyramid must be a power of two in 2..=16, got {v:?}"
                        ))
                    })?;
                let enc = pyramid::PyramidEncoder::new(width, height, qp, g)
                    .map_err(|e| Error::InvalidData(format!("h265 encode: {e}")))?
                    .with_amp(parse_flag(params, "amp")?)
                    .with_loop_filters(parse_lf(params)?);
                EncodeMode::Pyramid {
                    enc,
                    delay: i64::from(g.trailing_zeros()),
                    pts_by_display: Vec::new(),
                    decode_count: 0,
                }
            } else {
                let gop = match params.options.get("gop") {
                    None => 0usize,
                    Some(v) => v.parse::<usize>().map_err(|_| {
                        Error::InvalidData(format!(
                            "h265 encode: gop must be a non-negative integer, got {v:?}"
                        ))
                    })?,
                };
                let b_slices = parse_flag(params, "bslices")?;
                let enc = inter::LowDelayPEncoder::new(width, height, qp, gop)
                    .map_err(|e| Error::InvalidData(format!("h265 encode: {e}")))?
                    .with_b_slices(b_slices)
                    .with_amp(parse_flag(params, "amp")?)
                    .with_loop_filters(parse_lf(params)?);
                EncodeMode::Inter(enc)
            }
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
            EncodeMode::Intra(qp, lf) => (
                intra::encode_idr_intra_au_lf(&y, &cb, &cr, self.width, self.height, *qp, lf)
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
            EncodeMode::Pyramid {
                enc,
                delay,
                pts_by_display,
                decode_count,
            } => {
                pts_by_display.push(v.pts);
                let aus = enc
                    .encode_frame(&inter::YuvFrame {
                        y: &y,
                        cb: &cb,
                        cr: &cr,
                    })
                    .map_err(|e| Error::InvalidData(format!("h265 encode: {e}")))?;
                self.frame_index += 1;
                let (delay, decode_count) = (*delay, decode_count);
                let mut packets = Self::pyramid_packets(aus, pts_by_display, delay, decode_count);
                for pkt in packets.drain(..) {
                    self.ready.push_back(pkt);
                }
                return Ok(());
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
        // Only the pyramid mode buffers: emit its low-delay tail.
        if let EncodeMode::Pyramid {
            enc,
            delay,
            pts_by_display,
            decode_count,
        } = &mut self.mode
        {
            let aus = enc.flush();
            let (delay, decode_count) = (*delay, decode_count);
            let mut packets = Self::pyramid_packets(aus, pts_by_display, delay, decode_count);
            for pkt in packets.drain(..) {
                self.ready.push_back(pkt);
            }
        }
        Ok(())
    }
}

impl H265Encoder {
    /// Turn a burst of decode-ordered pyramid access units into
    /// packets: pts = the pushed frame's pts (default: its display
    /// index), dts = the decode counter minus the reorder delay (a
    /// dts never after its pts on the dyadic schedule).
    fn pyramid_packets(
        aus: Vec<pyramid::PyramidAu>,
        pts_by_display: &[Option<i64>],
        delay: i64,
        decode_count: &mut i64,
    ) -> Vec<Packet> {
        aus.into_iter()
            .map(|au| {
                let mut pkt = Packet::new(0, TimeBase::new(1, 25), au.au);
                pkt.pts = Some(
                    pts_by_display
                        .get(au.display_order)
                        .copied()
                        .flatten()
                        .unwrap_or(au.display_order as i64),
                );
                pkt.dts = Some(*decode_count - delay);
                *decode_count += 1;
                pkt.flags.keyframe = au.keyframe;
                pkt
            })
            .collect()
    }
}
