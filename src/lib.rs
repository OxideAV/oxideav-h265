//! Pure-Rust HEVC / H.265 (ITU-T H.265 | ISO/IEC 23008-2) decoder.
//!
//! ## Scope
//!
//! * **NAL unit framing** — Annex B byte-stream and length-prefixed (HVCC)
//!   modes; emulation-prevention byte removal.
//! * **VPS / SPS / PPS parsers** — every field needed to validate stream
//!   shape (resolution, chroma format, bit depth, CTU size, tiles /
//!   wavefront flags, …) is decoded.
//! * **Slice segment header parser** — reaches `byte_alignment()` for IDR
//!   I-slice packets under the current decode shape.
//! * **HEVCDecoderConfigurationRecord (`hvcC`) parser** — used by the MP4
//!   demuxer to populate `extradata`.
//! * **CABAC** — full arithmetic engine and I-slice context tables.
//! * **Coding-tree walker** — CTU → CU → PU → TU with intra-mode decode,
//!   residual coding (sig_coeff / greater1 / greater2 / signs / rice tail),
//!   inverse transform, flat dequantisation, and reconstruction.
//! * **Intra prediction** — all 35 modes (planar, DC, 33 angular), MDIS
//!   filter, strong-intra-smoothing at 32×32.
//!
//! ## Restricted to
//!
//! * 8-bit depth, 4:2:0 chroma subsampling, no `separate_colour_plane`.
//! * Single-tile, wavefront-off, no transform-skip, no scaling lists,
//!   no PCM CUs.
//! * **I-slices only** — P / B inter prediction returns
//!   `Error::Unsupported("h265 inter slice pending")`.
//!
//! ## Out of scope
//!
//! * **Deblocking filter** (§8.7.2) — reconstructed frames carry visible
//!   block-edge artefacts.
//! * **SAO** (§8.7.3) — parsed but not applied.
//! * **Scalable / multiview / 3D extensions** (SHVC, MV-HEVC, 3D-HEVC).
//! * **Encoder** — write side is not in scope.
//!
//! ## Crate layout
//!
//! * [`bitreader`] — MSB-first reader, ue(v)/se(v) Exp-Golomb helpers.
//! * [`nal`] — start-code scanner, length-prefix iterator, NAL header
//!   parsing, emulation-prevention removal.
//! * [`ptl`] — `profile_tier_level()` (§7.3.3) shared by VPS/SPS.
//! * [`vps`], [`sps`], [`pps`] — parameter-set parsers.
//! * [`slice`] — slice segment header and `SliceQpY` derivation.
//! * [`cabac`] — arithmetic engine and I-slice context-init tables.
//! * [`intra_pred`] — 35-mode intra prediction + MDIS reference filter.
//! * [`transform`] — DST-VII 4×4 and DCT-II 4 / 8 / 16 / 32, plus flat
//!   dequantisation.
//! * [`scan`] — 4×4 sub-block scan tables (diagonal / horizontal /
//!   vertical) and the intra-mode → scan_idx rule.
//! * [`ctu`] — coding-tree walker and residual-coding pipeline.
//! * [`hvcc`] — HEVCDecoderConfigurationRecord (ISO/IEC 14496-15 §8.3.3).
//! * [`decoder`] — registry factory and `HevcDecoder` wiring.

#![allow(clippy::too_many_arguments)]

pub mod bitreader;
pub mod cabac;
pub mod ctu;
pub mod decoder;
pub mod hvcc;
pub mod intra_pred;
pub mod nal;
pub mod pps;
pub mod ptl;
pub mod scan;
pub mod slice;
pub mod sps;
pub mod transform;
pub mod vps;

use oxideav_codec::{CodecInfo, CodecRegistry};
use oxideav_core::{CodecCapabilities, CodecId, CodecTag};

/// Canonical oxideav codec id for H.265 / HEVC.
pub const CODEC_ID_STR: &str = "h265";

/// Register the HEVC implementation with a codec registry.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("h265_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(8192, 8192);
    // AVI FourCC claims — HEVC, H265, HVC1, HEV1, X265 (libx265), DXHE
    // (DivX HEVC). Unambiguous.
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(decoder::make_decoder)
            .tags([
                CodecTag::fourcc(b"HEVC"),
                CodecTag::fourcc(b"H265"),
                CodecTag::fourcc(b"HVC1"),
                CodecTag::fourcc(b"HEV1"),
                CodecTag::fourcc(b"X265"),
                CodecTag::fourcc(b"DXHE"),
            ]),
    );
}
