//! Pure-Rust HEVC / H.265 (ITU-T H.265 | ISO/IEC 23008-2) decoder.
//!
//! ## Scope
//!
//! * **NAL unit framing** — Annex B byte-stream and length-prefixed (HVCC)
//!   modes; emulation-prevention byte removal.
//! * **VPS / SPS / PPS parsers** — every field needed to validate stream
//!   shape (resolution, chroma format, bit depth, CTU size, tiles /
//!   wavefront flags, …) is decoded.
//! * **Slice segment header parser** — full I-slice and P-slice extension
//!   parses (inline RPS, num_ref_idx, max_num_merge_cand, cabac_init_flag,
//!   slice_qp_delta, SAO flags, …).
//! * **HEVCDecoderConfigurationRecord (`hvcC`) parser** — used by the MP4
//!   demuxer to populate `extradata`.
//! * **CABAC** — full arithmetic engine, I-slice and inter context tables.
//! * **Coding-tree walker** — CTU → CU → PU → TU with intra-mode decode,
//!   inter prediction (merge / AMVP / MVD / ref_idx), residual coding
//!   (sig_coeff / greater1 / greater2 / signs / rice tail), inverse
//!   transform, flat dequantisation, and reconstruction.
//! * **Intra prediction** — all 35 modes (planar, DC, 33 angular), MDIS
//!   filter, strong-intra-smoothing at 32×32.
//! * **Inter prediction** — 8-tap luma / 4-tap chroma sub-pel MC,
//!   spatial merge + AMVP + TMVP, bi-pred averaging and weighted
//!   bi-prediction, DPB keyed by POC.
//!
//! ## Restricted to
//!
//! * 8-bit depth, 4:2:0 chroma subsampling, no `separate_colour_plane`.
//! * Single-tile, wavefront-off.
//! * **I, P, and B slices.** Inter PB shapes are limited to
//!   2Nx2N / 2NxN / Nx2N / NxN (no AMP).
//!
//! ## Out of scope
//!
//! * **Deblocking filter** (§8.7.2) — applied post-reconstruction with
//!   the spec's β/tC tables; boundary-strength derivation is an
//!   approximation over the crate's per-4×4 block state (intra flag,
//!   motion, cqt_depth) since per-TU CBFs are not yet tracked.
//! * **SAO** (§8.7.3) — edge-offset + band-offset applied after
//!   deblocking, before the DPB write. Luma and chroma both handled;
//!   SAO-merge (left/up) inheritance follows §7.4.9.3.
//! * **Long-term reference pictures** (§8.3.2) — SPS-level candidate
//!   list (`lt_ref_pic_poc_lsb_sps[]`, `used_by_curr_pic_lt_sps_flag[]`)
//!   and slice-level LT RPS entries (including the optional MSB-cycle
//!   delta) are parsed. LT refs are resolved against the DPB, marked so
//!   they survive short-term rotations, and appended to RPL0/RPL1 after
//!   the st_curr_* sets per §8.3.4.
//! * **Scalable / multiview / 3D extensions** (SHVC, MV-HEVC, 3D-HEVC).
//!
//! ## Encoder (MVP)
//!
//! * **[`encoder`]** — writes a single-IDR Annex B stream (VPS + SPS + PPS
//!   + IDR slice) using PCM CUs for lossless round-trip; 8-bit 4:2:0 Main
//!   profile only; width/height must be multiples of 64. Not a production
//!   encoder — no transform / quantisation / rate control — but the
//!   bit-writer, NAL framing, parameter-set emitters, CABAC writer, and
//!   slice header layer are all spec-accurate, and the output decodes
//!   byte-for-byte through both our decoder and ffmpeg.
//!
//! ## Crate layout
//!
//! * [`bitreader`] — MSB-first reader, ue(v)/se(v) Exp-Golomb helpers.
//! * [`nal`] — start-code scanner, length-prefix iterator, NAL header
//!   parsing, emulation-prevention removal.
//! * [`ptl`] — `profile_tier_level()` (§7.3.3) shared by VPS/SPS.
//! * [`vps`], [`sps`], [`pps`] — parameter-set parsers.
//! * [`slice`] — slice segment header (I + P extension) and `SliceQpY`
//!   derivation.
//! * [`cabac`] — arithmetic engine, I-slice and inter context-init tables.
//! * [`intra_pred`] — 35-mode intra prediction + MDIS reference filter.
//! * [`inter`] — DPB, merge list, AMVP, 8-tap luma / 4-tap chroma MC.
//! * [`transform`] — DST-VII 4×4 and DCT-II 4 / 8 / 16 / 32, plus flat
//!   dequantisation.
//! * [`scan`] — 4×4 sub-block scan tables (diagonal / horizontal /
//!   vertical) and the intra-mode → scan_idx rule.
//! * [`ctu`] — coding-tree walker and residual-coding pipeline.
//! * [`sao`] — Sample Adaptive Offset filter (§8.7.3): per-CTB grid of
//!   edge-offset + band-offset parameters, applied post-deblock.
//! * [`scaling_list`] — `scaling_list_data()` parser (§7.3.4), default
//!   tables (§7.4.5 Tables 7-5 / 7-6), and ScalingFactor expansion for
//!   dequantisation (§8.6.3 eq. 8-309 with `m[x][y] ≠ 16`).
//! * [`hvcc`] — HEVCDecoderConfigurationRecord (ISO/IEC 14496-15 §8.3.3).
//! * [`decoder`] — registry factory and `HevcDecoder` wiring.

#![allow(clippy::too_many_arguments)]

pub mod bitreader;
pub mod cabac;
pub mod ctu;
pub mod deblock;
pub mod decoder;
pub mod encoder;
pub mod hvcc;
pub mod inter;
pub mod intra_pred;
pub mod nal;
pub mod pps;
pub mod ptl;
pub mod sao;
pub mod scaling_list;
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
            .encoder(encoder::make_encoder)
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
