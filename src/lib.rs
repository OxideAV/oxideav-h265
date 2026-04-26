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
//! * Bit depth 8 (Main), 10 (Main 10), and 12 (Main 12). Chroma and luma
//!   depth must match. All bit depths support full I/P/B slice decode.
//! * 4:2:0 (`chroma_format_idc == 1`) and 4:2:2 (`chroma_format_idc == 2`)
//!   — full I/P/B slice decode at 8/10/12 bits. 4:2:2 P/B uses the
//!   §8.5.3.2.10 chroma MV derivation (`mvCLX[1] = mvLX[1] * 2 /
//!   SubHeightC`). 4:4:4 / 4:0:0 / `separate_colour_plane` all rejected
//!   with a clean `Error::Unsupported`.
//! * Single-tile, wavefront-off.
//! * **I, P, and B slices.** Inter PB shapes: 2Nx2N / 2NxN / Nx2N / NxN
//!   (the four symmetric Table 7-10 partitions) plus the four AMP shapes
//!   (`Mode2NxnU` / `Mode2NxnD` / `ModenLx2N` / `ModenRx2N`) when the SPS
//!   sets `amp_enabled_flag`. AMP partition geometry, neighbour-context
//!   suppression for partIdx 1, and the merge / AMVP candidate
//!   derivation share the symmetric-rect code path. The §7.4.9.10
//!   `interSplitFlag` interaction with non-2Nx2N inter CUs at
//!   `max_transform_hierarchy_depth_inter == 0` follows libx265's
//!   empirical bin-emission (round-7/8 investigation) — see the
//!   `transform_tree_inter_inner` comment in `ctu.rs` for the full
//!   trade-off; some content-dependent P-slice drift remains on
//!   non-trivial fixtures (the existing tests use loose
//!   "frame-changes-after-P" assertions on this path).
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
//! ## Encoder
//!
//! * **[`encoder`]** — writes a single-IDR Annex B stream (VPS + SPS + PPS
//!   + IDR slice) using a real intra pipeline: per-CU mode decision from a
//!     small cardinal/diagonal subset, forward DCT-II / DST-VII + flat-list
//!     quantisation at QP 26, and CABAC residual coding
//!     (`sig_coeff_flag` / `greater1/2` / `coeff_abs_level_remaining` /
//!     sign). 8-bit 4:2:0 Main profile, width/height multiples of 16
//!     (fixed 16×16 CTU). Self-roundtrip through `HevcDecoder` produces
//!     ~45 dB PSNR on synthetic fixtures at roughly 50× compression vs
//!     raw-sample budget. ffmpeg accepts the stream without decode errors;
//!     exact-match between ffmpeg and the encoder's locally-reconstructed
//!     picture is a work-in-progress (see `tests/ffmpeg_accepts.rs`).
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

#[cfg(feature = "heif")]
pub mod heif;

use oxideav_core::{CodecCapabilities, CodecId, CodecTag};
use oxideav_core::{CodecInfo, CodecRegistry};

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
