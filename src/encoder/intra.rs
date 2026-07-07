//! Real CABAC intra encoder — I-slice IDR access units with §8.4
//! intra prediction, forward transform + quantization, and full
//! §7.3.8 CABAC syntax emission (no PCM).
//!
//! Geometry (this bootstrap's fixed shape, mirroring the PCM
//! encoder): `CtbSizeY == MinCbSizeY == 16`, so every CTB is one
//! unsplit intra CU. Two partition modes compete per CU
//! (rate-distortion heuristic):
//!
//! * `PART_2Nx2N` — one 16x16 luma PB/TB + two 8x8 chroma TBs;
//! * `PART_NxN` — four 8x8 luma PBs, each its own mode; the §7.4.9.8
//!   `IntraSplitFlag` forces the transform tree to depth 1 (four
//!   8x8 luma TBs + four 4x4 chroma TBs per plane), the 8x8 luma and
//!   4x4 chroma TBs picking their §7.4.9.11 mode-dependent scans.
//!
//! Per CTU the encoder:
//!
//! 1. gathers the §8.4.4.2.1 reference samples from its own
//!    reconstruction buffer, marking availability per the §6.4.1
//!    z-scan decode order (CTB raster + TU z-order within the CTB),
//!    and runs the decode-side [`crate::intra_pred`] pipeline
//!    (§8.4.4.2.2 substitution + §8.4.4.2.3 filtering + planar / DC /
//!    angular prediction) for every candidate mode, picking the
//!    SAD-best per PB;
//! 2. forward-transforms the prediction residual (the transpose of
//!    the §8.6.4.2 DCT-II basis) and quantizes against the §8.6.3
//!    `levelScale`-derived reciprocal at the slice QP (chroma via the
//!    Table 8-10 QP mapping);
//! 3. reconstructs through the crate's own DECODE-side §8.6.2
//!    scaling/transform ([`crate::transform::residual_block`]) so the
//!    encoder's reference buffer is bit-identical to what a
//!    conforming decoder reconstructs, and picks the partition with
//!    the smaller SSD + partition-cost heuristic;
//! 4. emits the §7.3.8.5 coding-unit syntax (`part_mode`, the
//!    §7.3.8.5 two-loop `prev_intra_luma_pred_flag[]` then
//!    `mpm_idx` / `rem_intra_luma_pred_mode` group against the
//!    §8.4.2 candidate lists, `intra_chroma_pred_mode` =
//!    derived-from-luma), the §7.3.8.8 transform tree with its cbf
//!    inheritance, and the §7.3.8.11 residual blocks through
//!    [`crate::encoder::residual::encode_residual_coding`].
//!
//! In-loop filters are off (SAO off in the SPS, deblocking disabled
//! in the PPS), so a conforming decoder's output equals the encoder's
//! reconstruction exactly — pinned by the roundtrip tests.

use crate::binarization::intra_luma_cand_mode_list;
use crate::binarization::{cbf_cb_ctx_inc, cbf_cr_ctx_inc, cbf_luma_ctx_inc};
use crate::cabac::init_type;
use crate::ctx_init::SliceContexts;
use crate::encoder::bitwriter::BitWriter;
use crate::encoder::cabac::CabacEncoder;
use crate::encoder::nal::{annexb, nal_unit};
use crate::encoder::pcm::{level_idc_for, write_pps, write_ptl, write_vps};
use crate::encoder::residual::encode_residual_coding;
use crate::intra_mode_field::{IntraModeField, Neighbour};
use crate::intra_pred::{
    intra_predict_with_substitution, Component as PredComponent, IntraPredParams,
    MarkedReferenceSamples,
};
use crate::residual::{residual_coding_scan_idx, ResidualCodingParams};
use crate::transform::{forward_dct_1d, residual_block, BlockParams, Component, PredMode};

/// The fixed CTB / coding-block log2 size (16x16).
const CTB_LOG2: u32 = 4;
/// The fixed CTB size.
const CTB: usize = 1 << CTB_LOG2;
/// Fixed 8-bit depth.
const BIT_DEPTH: u32 = 8;
/// The z-order offsets of the four NxN prediction blocks / depth-1
/// transform units within a CTB (§6.5.2 z-scan of the four halves).
const Z_OFFSETS: [(usize, usize); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];

/// Errors from the intra encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraEncodeError {
    /// Width or height is zero or not a multiple of the 16-sample CTB.
    BadDimensions {
        /// Requested luma width.
        width: usize,
        /// Requested luma height.
        height: usize,
    },
    /// A supplied plane's length does not match the 4:2:0 geometry.
    PlaneSize {
        /// Which plane (`"y"`, `"cb"`, `"cr"`).
        plane: &'static str,
        /// Required sample count.
        expected: usize,
        /// Supplied sample count.
        got: usize,
    },
    /// `SliceQpY` outside 0..=51.
    BadQp(i32),
}

impl core::fmt::Display for IntraEncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadDimensions { width, height } => write!(
                f,
                "intra encoder requires nonzero dimensions that are multiples of 16, got {width}x{height}"
            ),
            Self::PlaneSize {
                plane,
                expected,
                got,
            } => write!(f, "{plane} plane has {got} samples, expected {expected}"),
            Self::BadQp(qp) => write!(f, "slice QP {qp} outside 0..=51"),
        }
    }
}

impl std::error::Error for IntraEncodeError {}

/// The encoded access unit plus the encoder's own reconstruction
/// (what a conforming decoder outputs — in-loop filters are off).
#[derive(Debug, Clone)]
pub struct IntraEncodedAu {
    /// The Annex B access unit (`VPS + SPS + PPS + IDR_N_LP`).
    pub au: Vec<u8>,
    /// Reconstructed luma plane (`width * height`).
    pub recon_y: Vec<u8>,
    /// Reconstructed Cb plane (`width/2 * height/2`).
    pub recon_cb: Vec<u8>,
    /// Reconstructed Cr plane.
    pub recon_cr: Vec<u8>,
}

/// §7.3.2.2 — the fixed-geometry SPS (4:2:0, 8-bit, CTB 16, PCM off,
/// SAO off). Shared by the intra and low-delay-P encoders (nothing in
/// it is slice-type specific: the P slices code their §7.4.8
/// short-term RPS inline, `sps_temporal_mvp_enabled_flag` is 0, and
/// `sps_max_dec_pic_buffering_minus1[0] == 1` covers the one-reference
/// low-delay GOP).
pub(crate) fn write_sps(width: usize, height: usize, level_idc: u8) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(0, 4); // sps_video_parameter_set_id
    w.put_bits(0, 3); // sps_max_sub_layers_minus1
    w.put_bit(1); // sps_temporal_id_nesting_flag
    write_ptl(&mut w, level_idc);
    w.ue(0); // sps_seq_parameter_set_id
    w.ue(1); // chroma_format_idc = 4:2:0
    w.ue(width as u32); // pic_width_in_luma_samples
    w.ue(height as u32); // pic_height_in_luma_samples
    w.put_bit(0); // conformance_window_flag
    w.ue(0); // bit_depth_luma_minus8
    w.ue(0); // bit_depth_chroma_minus8
    w.ue(4); // log2_max_pic_order_cnt_lsb_minus4
    w.put_bit(1); // sps_sub_layer_ordering_info_present_flag
    w.ue(1); // sps_max_dec_pic_buffering_minus1[0]
    w.ue(0); // sps_max_num_reorder_pics[0]
    w.ue(0); // sps_max_latency_increase_plus1[0]
    w.ue(CTB_LOG2 - 3); // log2_min_luma_coding_block_size_minus3 (16)
    w.ue(0); // log2_diff_max_min_luma_coding_block_size (CTB 16)
    w.ue(0); // log2_min_luma_transform_block_size_minus2 (4)
    w.ue(2); // log2_diff_max_min_luma_transform_block_size (16)
    w.ue(0); // max_transform_hierarchy_depth_inter
    w.ue(0); // max_transform_hierarchy_depth_intra
    w.put_bit(0); // scaling_list_enabled_flag
    w.put_bit(0); // amp_enabled_flag
    w.put_bit(0); // sample_adaptive_offset_enabled_flag
    w.put_bit(0); // pcm_enabled_flag
    w.ue(0); // num_short_term_ref_pic_sets
    w.put_bit(0); // long_term_ref_pics_present_flag
    w.put_bit(0); // sps_temporal_mvp_enabled_flag
    w.put_bit(0); // strong_intra_smoothing_enabled_flag
    w.put_bit(0); // vui_parameters_present_flag
    w.put_bit(0); // sps_extension_present_flag
    w.rbsp_trailing_bits();
    w.finish()
}

/// Table 8-10 — the `ChromaArrayType == 1` chroma QP mapping
/// `qPC = f(qPi)` (§8.6.1; `QpBdOffsetC == 0` at 8-bit).
pub(crate) fn chroma_qp_420(qp_y: i32) -> u32 {
    let qpi = qp_y.clamp(0, 57);
    (match qpi {
        x if x < 30 => x,
        30..=33 => qpi - 1,             // 30..=33 -> 29, 30, 31, 32
        34..=43 => 33 + (qpi - 34) / 2, // 34..=43 -> 33, 33, 34, 34 .. 37, 37
        x => x - 6,
    }) as u32
}

/// §8.6.3-derived reciprocal quantizer scale: `levelScale[qP % 6]` is
/// `{40, 45, 51, 57, 64, 72}`; the forward reciprocal is
/// `round(2^20 / levelScale)` so `quant ∘ dequant` has unity gain.
fn quant_scale(qp_rem: u32) -> i64 {
    let ls = i64::from(crate::transform::LEVEL_SCALE[qp_rem as usize]);
    ((1i64 << 20) + ls / 2) / ls
}

/// Forward 2-D DCT-II (the transpose of the §8.6.4 inverse): stage 1
/// over rows with `shift1 = log2TbS + BitDepth − 9`, stage 2 over
/// columns with `shift2 = log2TbS + 6` — the normalization that makes
/// the §8.6.3 dequant + §8.6.4 inverse reproduce the residual.
fn forward_transform(res: &[i32], n: usize) -> Vec<i32> {
    let log2 = n.trailing_zeros();
    let shift1 = log2 + BIT_DEPTH - 9;
    let shift2 = log2 + 6;
    let r1 = 1i64 << (shift1 - 1);
    let r2 = 1i64 << (shift2 - 1);
    // Stage 1: horizontal analysis per row y -> a[y][u].
    let mut a = vec![0i64; n * n];
    for y in 0..n {
        let row: Vec<i64> = (0..n).map(|x| i64::from(res[y * n + x])).collect();
        let t = forward_dct_1d(&row, n);
        for (u, &v) in t.iter().enumerate() {
            a[y * n + u] = (v + r1) >> shift1;
        }
    }
    // Stage 2: vertical analysis per column u -> coef[v][u].
    let mut coef = vec![0i32; n * n];
    for u in 0..n {
        let col: Vec<i64> = (0..n).map(|y| a[y * n + u]).collect();
        let t = forward_dct_1d(&col, n);
        for (v, &val) in t.iter().enumerate() {
            coef[v * n + u] = ((val + r2) >> shift2) as i32;
        }
    }
    coef
}

/// Scalar quantization to `TransCoeffLevel`: `level = sign ·
/// (|coef| · quantScale + offset) >> qBits` with `qBits = 14 + qP/6 +
/// (15 − BitDepth − log2TbS)` (the inverse of the §8.6.3 eq. 8-309
/// scaling chain) and a one-third rounding offset; clamped to the
/// §7.4.9.11 CoeffMax.
fn quantize(coef: &[i32], n: usize, qp: u32) -> Vec<i32> {
    let log2 = n.trailing_zeros();
    let qbits = 14 + qp / 6 + (15 - BIT_DEPTH - log2);
    let scale = quant_scale(qp % 6);
    let offset = (1i64 << qbits) / 3;
    coef.iter()
        .map(|&c| {
            let level = ((i64::from(c.unsigned_abs()) * scale + offset) >> qbits).min(0x7FFF);
            (level as i32) * c.signum()
        })
        .collect()
}

/// §6.4.1-shaped z-scan availability of the sample at `(nx, ny)`
/// relative to the block being decoded: available iff inside the
/// plane AND its covering coding block precedes in decode order —
/// an earlier CTB (raster), or the same CTB with a smaller depth-1
/// z-order quadrant index (`cur_z`; pass 0 when the current TB is the
/// whole CTB, making every same-CTB neighbour unavailable).
#[allow(clippy::too_many_arguments)]
pub(crate) fn zscan_avail(
    nx: i64,
    ny: i64,
    plane_w: usize,
    plane_h: usize,
    blk: usize,
    ctbs_x: usize,
    cur_ctb: usize,
    cur_z: u32,
) -> bool {
    if nx < 0 || ny < 0 || nx >= plane_w as i64 || ny >= plane_h as i64 {
        return false;
    }
    let (nx, ny) = (nx as usize, ny as usize);
    let nctb = (ny / blk) * ctbs_x + nx / blk;
    match nctb.cmp(&cur_ctb) {
        core::cmp::Ordering::Less => true,
        core::cmp::Ordering::Greater => false,
        core::cmp::Ordering::Equal => {
            let half = blk / 2;
            let z = ((ny % blk) / half) * 2 + ((nx % blk) / half);
            (z as u32) < cur_z
        }
    }
}

/// Gather the §8.4.4.2.1 marked reference array for an `n`-sample TB
/// at `(x0, y0)`: values through `read`, availability through `avail`.
pub(crate) fn gather_refs(
    read: &dyn Fn(usize, usize) -> i32,
    avail: &dyn Fn(i64, i64) -> bool,
    x0: usize,
    y0: usize,
    n: usize,
) -> MarkedReferenceSamples {
    let get = |x: i64, y: i64| -> (i32, bool) {
        if avail(x, y) {
            (read(x as usize, y as usize), true)
        } else {
            (0, false)
        }
    };
    let corner = get(x0 as i64 - 1, y0 as i64 - 1);
    let left: Vec<(i32, bool)> = (0..2 * n)
        .map(|k| get(x0 as i64 - 1, (y0 + k) as i64))
        .collect();
    let top: Vec<(i32, bool)> = (0..2 * n)
        .map(|k| get((x0 + k) as i64, y0 as i64 - 1))
        .collect();
    MarkedReferenceSamples::new(n, corner, left, top).expect("legal TB geometry")
}

pub(crate) fn pred_params(mode: u8, cidx: PredComponent) -> IntraPredParams {
    IntraPredParams {
        pred_mode_intra: mode,
        cidx,
        bit_depth: BIT_DEPTH as u8,
        bit_depth_luma: BIT_DEPTH as u8,
        intra_smoothing_disabled: false,
        strong_intra_smoothing_enabled: false,
        chroma_array_type_3: false,
        disable_boundary_filter: false,
    }
}

/// SAD-search all 35 §8.4.2 modes; returns `(mode, prediction)`.
pub(crate) fn search_best_mode(marked: &MarkedReferenceSamples, src: &[i32]) -> (u8, Vec<i32>) {
    let mut best = (0u8, Vec::new());
    let mut best_cost = u64::MAX;
    for mode in 0..=34u8 {
        let pred = intra_predict_with_substitution(marked, &pred_params(mode, PredComponent::Luma))
            .expect("legal prediction params");
        let cost: u64 = src
            .iter()
            .zip(pred.iter())
            .map(|(&s, &p)| u64::from(s.abs_diff(p)))
            .sum();
        if cost < best_cost {
            best_cost = cost;
            best = (mode, pred);
        }
    }
    best
}

/// Transform + quantize one component TB and reconstruct it through
/// the DECODE-side §8.6.2 path. Returns `(levels, recon_samples)`;
/// `levels` all-zero ⇔ cbf 0 (recon = clipped prediction).
///
/// `pred_mode` selects the §8.6.4 transform family exactly as the
/// decoder does (the intra-luma 4x4 DST case; every TB the intra and
/// low-delay-P encoders emit at other geometries is DCT either way).
pub(crate) fn code_tb(
    src: &[i32],
    pred: &[i32],
    n: usize,
    qp: u32,
    component: Component,
    pred_mode: PredMode,
) -> (Vec<i32>, Vec<u8>) {
    let res: Vec<i32> = src.iter().zip(pred.iter()).map(|(&s, &p)| s - p).collect();
    let coef = forward_transform(&res, n);
    let levels = quantize(&coef, n, qp);
    let recon: Vec<u8> = if levels.iter().all(|&v| v == 0) {
        pred.iter().map(|&p| p.clamp(0, 255) as u8).collect()
    } else {
        let r = residual_block(
            &levels,
            None,
            BlockParams {
                n_tbs: n,
                q_p: qp,
                component,
                pred_mode,
                bit_depth: BIT_DEPTH as u8,
                extended_precision: false,
                transquant_bypass: false,
                transform_skip: false,
                transform_skip_rotation_enabled: false,
            },
        )
        .expect("legal block params");
        pred.iter()
            .zip(r.iter())
            .map(|(&p, &d)| (p + d).clamp(0, 255) as u8)
            .collect()
    };
    (levels, recon)
}

pub(crate) fn ssd(a: &[u8], b: &[i32]) -> u64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = i64::from(x) - i64::from(y);
            (d * d) as u64
        })
        .sum()
}

/// Crude bit-cost proxy for one TB's quantized levels: each nonzero
/// coefficient costs roughly its magnitude's bit length (sig flag +
/// sign + level bins); a coded-but-empty TB is nearly free, a coded
/// TB pays a small last-sig overhead.
fn rate_proxy(levels: &[i32]) -> u64 {
    let bits: u64 = levels
        .iter()
        .filter(|&&l| l != 0)
        .map(|&l| 3 + 2 * u64::from(32 - l.unsigned_abs().leading_zeros()))
        .sum();
    if bits == 0 {
        1
    } else {
        bits + 8
    }
}

/// One coded luma partition candidate for a CTB.
struct LumaPlan {
    /// `PART_NxN`?
    nxn: bool,
    /// PB modes (1 used for 2Nx2N, 4 for NxN, z-order).
    modes: [u8; 4],
    /// TB level arrays (1 x 16x16 or 4 x 8x8, z-order).
    levels: Vec<Vec<i32>>,
    /// The CTB's 16x16 luma reconstruction, row-major.
    recon: Vec<u8>,
}

/// Encode one 4:2:0 8-bit frame as a self-contained intra IDR access
/// unit at `SliceQpY == qp` and return it with the reconstruction a
/// conforming decoder produces.
///
/// # Errors
/// [`IntraEncodeError`] on bad dimensions / plane sizes / QP.
#[allow(clippy::too_many_lines)]
pub fn encode_idr_intra_au(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    qp: i32,
) -> Result<IntraEncodedAu, IntraEncodeError> {
    if width == 0 || height == 0 || width % CTB != 0 || height % CTB != 0 {
        return Err(IntraEncodeError::BadDimensions { width, height });
    }
    if !(0..=51).contains(&qp) {
        return Err(IntraEncodeError::BadQp(qp));
    }
    let check = |plane: &'static str, buf: &[u8], expected: usize| {
        if buf.len() != expected {
            Err(IntraEncodeError::PlaneSize {
                plane,
                expected,
                got: buf.len(),
            })
        } else {
            Ok(())
        }
    };
    check("y", y, width * height)?;
    check("cb", cb, width * height / 4)?;
    check("cr", cr, width * height / 4)?;

    let (cw, ch) = (width / 2, height / 2);
    let ctbs_x = width / CTB;
    let ctbs_y = height / CTB;
    let qp_y = qp as u32;
    let qp_c = chroma_qp_420(qp);
    // Rate-distortion tradeoff for the partition decision: SSD per
    // estimated bit, doubling every 3 QP (integer, deterministic).
    let lambda: u64 = 1u64 << (qp.unsigned_abs().saturating_sub(9) / 3);

    let mut recon_y = vec![0u8; width * height];
    let mut recon_cb = vec![0u8; cw * ch];
    let mut recon_cr = vec![0u8; cw * ch];
    let mut modes = IntraModeField::new(width, height, CTB_LOG2);

    // ---- slice_segment_header( ) ----
    let mut w = BitWriter::new();
    w.put_bit(1); // first_slice_segment_in_pic_flag
    w.put_bit(0); // no_output_of_prior_pics_flag (IRAP NAL)
    w.ue(0); // slice_pic_parameter_set_id
    w.ue(2); // slice_type = I
    w.se(qp - 26); // slice_qp_delta (init_qp_minus26 == 0)
                   // SAO off + deblocking disabled: no more slice-level fields.
    w.rbsp_trailing_bits(); // byte_alignment()

    // ---- slice_segment_data( ) ----
    let mut cabac = CabacEncoder::new();
    // Table 9-4: I slice => initType 0 (raw slice_type 2).
    let mut ctxs = SliceContexts::init(init_type(2, false), qp);

    let extract = |plane: &[u8], pw: usize, x0: usize, y0: usize, n: usize| -> Vec<i32> {
        let mut out = Vec::with_capacity(n * n);
        for j in 0..n {
            for i in 0..n {
                out.push(i32::from(plane[(y0 + j) * pw + x0 + i]));
            }
        }
        out
    };
    let store = |plane: &mut [u8], pw: usize, x0: usize, y0: usize, n: usize, s: &[u8]| {
        for j in 0..n {
            plane[(y0 + j) * pw + x0..(y0 + j) * pw + x0 + n]
                .copy_from_slice(&s[j * n..(j + 1) * n]);
        }
    };

    for ctb in 0..ctbs_x * ctbs_y {
        let x0 = (ctb % ctbs_x) * CTB;
        let y0 = (ctb / ctbs_x) * CTB;
        let src16 = extract(y, width, x0, y0, CTB);

        // ---- candidate PART_2Nx2N: one 16x16 PB/TB ----
        let plan_2n = {
            let read = |x: usize, yy: usize| i32::from(recon_y[yy * width + x]);
            let avail = |nx: i64, ny: i64| zscan_avail(nx, ny, width, height, CTB, ctbs_x, ctb, 0);
            let marked = gather_refs(&read, &avail, x0, y0, CTB);
            let (mode, pred) = search_best_mode(&marked, &src16);
            let (levels, recon) =
                code_tb(&src16, &pred, CTB, qp_y, Component::Luma, PredMode::Intra);
            LumaPlan {
                nxn: false,
                modes: [mode; 4],
                levels: vec![levels],
                recon,
            }
        };

        // ---- candidate PART_NxN: four 8x8 PBs/TBs, z-order ----
        let plan_nxn = {
            let mut scratch = vec![0u8; CTB * CTB]; // in-progress CTB recon
            let mut pb_modes = [0u8; 4];
            let mut pb_levels: Vec<Vec<i32>> = Vec::with_capacity(4);
            for (k, &(zx, zy)) in Z_OFFSETS.iter().enumerate() {
                let (px, py) = (x0 + zx * 8, y0 + zy * 8);
                let read = |x: usize, yy: usize| -> i32 {
                    if (x0..x0 + CTB).contains(&x) && (y0..y0 + CTB).contains(&yy) {
                        i32::from(scratch[(yy - y0) * CTB + (x - x0)])
                    } else {
                        i32::from(recon_y[yy * width + x])
                    }
                };
                let avail = |nx: i64, ny: i64| {
                    zscan_avail(nx, ny, width, height, CTB, ctbs_x, ctb, k as u32)
                };
                let marked = gather_refs(&read, &avail, px, py, 8);
                let src8 = extract(y, width, px, py, 8);
                let (mode, pred) = search_best_mode(&marked, &src8);
                let (levels, recon8) =
                    code_tb(&src8, &pred, 8, qp_y, Component::Luma, PredMode::Intra);
                for j in 0..8 {
                    scratch[(zy * 8 + j) * CTB + zx * 8..(zy * 8 + j) * CTB + zx * 8 + 8]
                        .copy_from_slice(&recon8[j * 8..(j + 1) * 8]);
                }
                pb_modes[k] = mode;
                pb_levels.push(levels);
            }
            LumaPlan {
                nxn: true,
                modes: pb_modes,
                levels: pb_levels,
                recon: scratch,
            }
        };

        // Luma-only rate-distortion comparison: SSD of the coded
        // reconstruction + lambda times a bit proxy (residual levels +
        // mode signalling: ~6 bits per PB).
        let cost = |plan: &LumaPlan| -> u64 {
            let rate: u64 = plan.levels.iter().map(|lv| rate_proxy(lv)).sum::<u64>()
                + plan.levels.len() as u64 * 6;
            ssd(&plan.recon, &src16) + lambda * rate
        };
        let plan = if cost(&plan_nxn) < cost(&plan_2n) {
            plan_nxn
        } else {
            plan_2n
        };
        store(&mut recon_y, width, x0, y0, CTB, &plan.recon);
        // §8.4.3: IntraPredModeC derives from the CU's first PB.
        let mode_c = plan.modes[0];

        // ---- chroma: 8x8 TBs (2Nx2N) or four 4x4 TBs (NxN) ----
        let (cx0, cy0) = (x0 / 2, y0 / 2);
        let code_chroma = |plane: &[u8],
                           recon: &mut Vec<u8>,
                           comp: Component,
                           pc: PredComponent|
         -> Vec<Vec<i32>> {
            if !plan.nxn {
                let read = |x: usize, yy: usize| i32::from(recon[yy * cw + x]);
                let avail = |nx: i64, ny: i64| zscan_avail(nx, ny, cw, ch, CTB / 2, ctbs_x, ctb, 0);
                let marked = gather_refs(&read, &avail, cx0, cy0, 8);
                let pred = intra_predict_with_substitution(&marked, &pred_params(mode_c, pc))
                    .expect("legal prediction params");
                let src = extract(plane, cw, cx0, cy0, 8);
                let (levels, rec) = code_tb(&src, &pred, 8, qp_c, comp, PredMode::Intra);
                store(recon, cw, cx0, cy0, 8, &rec);
                vec![levels]
            } else {
                let mut out = Vec::with_capacity(4);
                let mut scratch = vec![0u8; 64]; // 8x8 chroma CTB recon
                for (k, &(zx, zy)) in Z_OFFSETS.iter().enumerate() {
                    let (px, py) = (cx0 + zx * 4, cy0 + zy * 4);
                    let read = |x: usize, yy: usize| -> i32 {
                        if (cx0..cx0 + 8).contains(&x) && (cy0..cy0 + 8).contains(&yy) {
                            i32::from(scratch[(yy - cy0) * 8 + (x - cx0)])
                        } else {
                            i32::from(recon[yy * cw + x])
                        }
                    };
                    let avail = |nx: i64, ny: i64| {
                        zscan_avail(nx, ny, cw, ch, CTB / 2, ctbs_x, ctb, k as u32)
                    };
                    let marked = gather_refs(&read, &avail, px, py, 4);
                    let pred = intra_predict_with_substitution(&marked, &pred_params(mode_c, pc))
                        .expect("legal prediction params");
                    let src = extract(plane, cw, px, py, 4);
                    let (levels, rec) = code_tb(&src, &pred, 4, qp_c, comp, PredMode::Intra);
                    for j in 0..4 {
                        scratch[(zy * 4 + j) * 8 + zx * 4..(zy * 4 + j) * 8 + zx * 4 + 4]
                            .copy_from_slice(&rec[j * 4..(j + 1) * 4]);
                    }
                    out.push(levels);
                }
                store(recon, cw, cx0, cy0, 8, &scratch);
                out
            }
        };
        let cb_levels = code_chroma(cb, &mut recon_cb, Component::Cb, PredComponent::Cb);
        let cr_levels = code_chroma(cr, &mut recon_cr, Component::Cr, PredComponent::Cr);

        // ---- §7.3.8.5 coding_unit( ) syntax ----
        // part_mode: §9.3.3.7 intra at MinCb — "1" = PART_2Nx2N,
        // "0" = PART_NxN.
        cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], u8::from(!plan.nxn));
        // §7.3.8.5 two-loop luma mode group: all
        // prev_intra_luma_pred_flag bins first, then the mpm_idx /
        // rem_intra_luma_pred_mode group. The §8.4.2 candidate list of
        // PB k sees the recorded modes of PBs < k, so record as we go
        // through the SECOND loop (derivation order).
        let n_pb = if plan.nxn { 4 } else { 1 };
        let pb_size = if plan.nxn { 8 } else { CTB };
        let pb_pos = |k: usize| (x0 + Z_OFFSETS[k].0 * 8, y0 + Z_OFFSETS[k].1 * 8);
        let mut selections: Vec<Option<usize>> = Vec::with_capacity(n_pb);
        {
            // The candidate list depends only on ALREADY-decoded PBs,
            // which are identical in both loop passes; precompute the
            // per-PB MPM position by simulating the record order.
            for k in 0..n_pb {
                let (px, py) = pb_pos(k);
                let avail_l = zscan_avail(
                    px as i64 - 1,
                    py as i64,
                    width,
                    height,
                    CTB,
                    ctbs_x,
                    ctb,
                    k as u32,
                );
                let avail_a = zscan_avail(
                    px as i64,
                    py as i64 - 1,
                    width,
                    height,
                    CTB,
                    ctbs_x,
                    ctb,
                    k as u32,
                );
                let cand_a = modes.cand_intra_pred_mode(px, py, Neighbour::Left, avail_l);
                let cand_b = modes.cand_intra_pred_mode(px, py, Neighbour::Above, avail_a);
                let list = intra_luma_cand_mode_list(cand_a, cand_b);
                selections.push(list.iter().position(|&m| m == plan.modes[k]));
                // Loop 1: prev_intra_luma_pred_flag[k].
                cabac.encode_decision(
                    &mut w,
                    &mut ctxs.prev_intra_luma_pred_flag[0],
                    u8::from(selections[k].is_some()),
                );
                // Record now: the next PB's candidates must see this
                // one (§8.4.2 derivation order).
                modes.record_intra_pb(px, py, pb_size, plan.modes[k], false);
            }
            // Loop 2: mpm_idx / rem_intra_luma_pred_mode.
            for (k, sel) in selections.iter().enumerate() {
                match *sel {
                    Some(0) => cabac.encode_bypass(&mut w, 0),
                    Some(1) => {
                        cabac.encode_bypass(&mut w, 1);
                        cabac.encode_bypass(&mut w, 0);
                    }
                    Some(_) => {
                        cabac.encode_bypass(&mut w, 1);
                        cabac.encode_bypass(&mut w, 1);
                    }
                    None => {
                        // §8.4.2: rem = mode with each smaller
                        // candidate removed. Recompute the list the
                        // same way the decoder will (earlier PBs
                        // recorded).
                        let (px, py) = pb_pos(k);
                        let avail_l = zscan_avail(
                            px as i64 - 1,
                            py as i64,
                            width,
                            height,
                            CTB,
                            ctbs_x,
                            ctb,
                            k as u32,
                        );
                        let avail_a = zscan_avail(
                            px as i64,
                            py as i64 - 1,
                            width,
                            height,
                            CTB,
                            ctbs_x,
                            ctb,
                            k as u32,
                        );
                        let cand_a = modes.cand_intra_pred_mode(px, py, Neighbour::Left, avail_l);
                        let cand_b = modes.cand_intra_pred_mode(px, py, Neighbour::Above, avail_a);
                        let list = intra_luma_cand_mode_list(cand_a, cand_b);
                        let mut rem = u32::from(plan.modes[k]);
                        for &c in &list {
                            if u32::from(plan.modes[k]) > u32::from(c) {
                                rem -= 1;
                            }
                        }
                        cabac.encode_bypass_bits(&mut w, rem, 5); // FL cMax 31
                    }
                }
            }
        }
        // intra_chroma_pred_mode = 4 (derived from luma): bin "0".
        cabac.encode_decision(&mut w, &mut ctxs.intra_chroma_pred_mode[0], 0);

        // ---- §7.3.8.8 transform_tree + §7.3.8.10 transform_unit ----
        let rc_params = |log2: u32, is_chroma: bool, mode: u8| ResidualCodingParams {
            log2_trafo_size: log2,
            is_chroma,
            // §7.4.9.11 mode-dependent scan (only 4x4 / 8x8-luma TBs
            // are eligible; larger TBs come back Diagonal).
            scan_idx: residual_coding_scan_idx(true, log2, u8::from(is_chroma), 1, u32::from(mode)),
            sign_data_hiding_enabled_flag: false,
            sign_hidden_suppressed: false,
            transform_skip_sig_ctx: false,
        };
        if !plan.nxn {
            // Single 16x16 TU at depth 0.
            let cbf_cb = cb_levels[0].iter().any(|&v| v != 0);
            let cbf_cr = cr_levels[0].iter().any(|&v| v != 0);
            let cbf_luma = plan.levels[0].iter().any(|&v| v != 0);
            cabac.encode_decision(
                &mut w,
                &mut ctxs.cbf_chroma[cbf_cb_ctx_inc(0) as usize],
                u8::from(cbf_cb),
            );
            cabac.encode_decision(
                &mut w,
                &mut ctxs.cbf_chroma[cbf_cr_ctx_inc(0) as usize],
                u8::from(cbf_cr),
            );
            cabac.encode_decision(
                &mut w,
                &mut ctxs.cbf_luma[cbf_luma_ctx_inc(0) as usize],
                u8::from(cbf_luma),
            );
            if cbf_luma {
                encode_residual_coding(
                    &mut w,
                    &mut cabac,
                    &mut ctxs.residual,
                    &rc_params(4, false, plan.modes[0]),
                    &plan.levels[0],
                )
                .expect("validated luma levels");
            }
            if cbf_cb {
                encode_residual_coding(
                    &mut w,
                    &mut cabac,
                    &mut ctxs.residual,
                    &rc_params(3, true, mode_c),
                    &cb_levels[0],
                )
                .expect("validated cb levels");
            }
            if cbf_cr {
                encode_residual_coding(
                    &mut w,
                    &mut cabac,
                    &mut ctxs.residual,
                    &rc_params(3, true, mode_c),
                    &cr_levels[0],
                )
                .expect("validated cr levels");
            }
        } else {
            // IntraSplitFlag == 1: split_transform_flag inferred 1 at
            // depth 0 (§7.4.9.8); four 8x8 leaves at depth 1. Root
            // cbf_cb / cbf_cr gate the per-leaf chroma flags
            // (§7.3.8.8 inheritance).
            let leaf_cbf = |lv: &Vec<i32>| lv.iter().any(|&v| v != 0);
            let root_cb = cb_levels.iter().any(&leaf_cbf);
            let root_cr = cr_levels.iter().any(&leaf_cbf);
            cabac.encode_decision(
                &mut w,
                &mut ctxs.cbf_chroma[cbf_cb_ctx_inc(0) as usize],
                u8::from(root_cb),
            );
            cabac.encode_decision(
                &mut w,
                &mut ctxs.cbf_chroma[cbf_cr_ctx_inc(0) as usize],
                u8::from(root_cr),
            );
            for k in 0..4 {
                let cbf_cb_k = leaf_cbf(&cb_levels[k]);
                let cbf_cr_k = leaf_cbf(&cr_levels[k]);
                let cbf_luma_k = leaf_cbf(&plan.levels[k]);
                if root_cb {
                    cabac.encode_decision(
                        &mut w,
                        &mut ctxs.cbf_chroma[cbf_cb_ctx_inc(1) as usize],
                        u8::from(cbf_cb_k),
                    );
                }
                if root_cr {
                    cabac.encode_decision(
                        &mut w,
                        &mut ctxs.cbf_chroma[cbf_cr_ctx_inc(1) as usize],
                        u8::from(cbf_cr_k),
                    );
                }
                cabac.encode_decision(
                    &mut w,
                    &mut ctxs.cbf_luma[cbf_luma_ctx_inc(1) as usize],
                    u8::from(cbf_luma_k),
                );
                if cbf_luma_k {
                    encode_residual_coding(
                        &mut w,
                        &mut cabac,
                        &mut ctxs.residual,
                        &rc_params(3, false, plan.modes[k]),
                        &plan.levels[k],
                    )
                    .expect("validated luma levels");
                }
                if root_cb && cbf_cb_k {
                    encode_residual_coding(
                        &mut w,
                        &mut cabac,
                        &mut ctxs.residual,
                        &rc_params(2, true, mode_c),
                        &cb_levels[k],
                    )
                    .expect("validated cb levels");
                }
                if root_cr && cbf_cr_k {
                    encode_residual_coding(
                        &mut w,
                        &mut cabac,
                        &mut ctxs.residual,
                        &rc_params(2, true, mode_c),
                        &cr_levels[k],
                    )
                    .expect("validated cr levels");
                }
            }
        }

        // end_of_slice_segment_flag.
        cabac.encode_terminate(&mut w, u8::from(ctb == ctbs_x * ctbs_y - 1));
    }
    // The final terminate-1 flush wrote the rbsp_stop_one_bit;
    // rbsp_slice_segment_trailing_bits() is alignment zeros from here.
    w.align_zero();
    let slice_rbsp = w.finish();

    let level_idc = level_idc_for(width * height);
    let units = vec![
        nal_unit(32, 0, 0, &write_vps(level_idc)), // VPS_NUT
        nal_unit(33, 0, 0, &write_sps(width, height, level_idc)), // SPS_NUT
        nal_unit(34, 0, 0, &write_pps(false, false, None)), // PPS_NUT
        nal_unit(20, 0, 0, &slice_rbsp),           // IDR_N_LP
    ];
    Ok(IntraEncodedAu {
        au: annexb(&units),
        recon_y,
        recon_cb,
        recon_cr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binarization::PartMode;
    use crate::sequence::{decode_annexb_sequence, decode_annexb_sequence_debug};
    use crate::slice_data::CodingQuadtree;

    /// Deterministic test content: smooth gradients + a diagonal
    /// texture component so directional modes and residuals both work.
    fn planes(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let y: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, yy) = (i % w, i / w);
                ((x * 3 + yy * 2 + (x * yy / 7) % 31) % 256) as u8
            })
            .collect();
        let cb: Vec<u8> = (0..w * h / 4)
            .map(|i| ((i % (w / 2)) * 4 % 200 + 20) as u8)
            .collect();
        let cr: Vec<u8> = (0..w * h / 4)
            .map(|i| (240 - (i / (w / 2)) * 3 % 200) as u8)
            .collect();
        (y, cb, cr)
    }

    /// Content with per-8x8 alternating strong directions: drives the
    /// partition decision toward PART_NxN.
    fn blocky_planes(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let y: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, yy) = (i % w, i / w);
                let (bx, by) = (x / 8, yy / 8);
                match (bx + by) % 3 {
                    0 => ((x % 8) * 30) as u8,
                    1 => ((yy % 8) * 30) as u8,
                    _ => (((x + yy) % 16) * 15) as u8,
                }
            })
            .collect();
        let cb = vec![100u8; w * h / 4];
        let cr = vec![160u8; w * h / 4];
        (y, cb, cr)
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let mse: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = f64::from(x) - f64::from(y);
                d * d
            })
            .sum::<f64>()
            / a.len() as f64;
        if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0f64 * 255.0 / mse).log10()
        }
    }

    fn assert_roundtrip_exact(y: &[u8], cb: &[u8], cr: &[u8], w: usize, h: usize, qp: i32) {
        let enc = encode_idr_intra_au(y, cb, cr, w, h, qp).expect("encode");
        let frames = decode_annexb_sequence(&enc.au).expect("decode");
        assert_eq!(frames.len(), 1, "{w}x{h} qp{qp}");
        let mut recon = Vec::new();
        recon.extend_from_slice(&enc.recon_y);
        recon.extend_from_slice(&enc.recon_cb);
        recon.extend_from_slice(&enc.recon_cr);
        assert_eq!(
            frames[0].picture.to_planar_u8().expect("8-bit"),
            recon,
            "{w}x{h} qp{qp}: decoder output == encoder reconstruction"
        );
    }

    /// The core contract: the crate's own decoder reproduces the
    /// encoder's reconstruction EXACTLY (dual-decoder bit-exactness),
    /// on both smooth and NxN-inducing content.
    #[test]
    fn intra_au_decodes_to_encoder_recon_exactly() {
        for (w, h) in [(16usize, 16usize), (64, 48), (48, 80)] {
            let (y, cb, cr) = planes(w, h);
            for qp in [4i32, 22, 32, 45] {
                assert_roundtrip_exact(&y, &cb, &cr, w, h, qp);
            }
            let (y, cb, cr) = blocky_planes(w, h);
            for qp in [10i32, 27, 38] {
                assert_roundtrip_exact(&y, &cb, &cr, w, h, qp);
            }
        }
    }

    /// The partition decision really selects PART_NxN on content with
    /// per-8x8 directional structure (and the stream decodes exactly).
    #[test]
    fn nxn_partition_is_selected_and_decodes() {
        let (w, h) = (64usize, 64usize);
        let (y, cb, cr) = blocky_planes(w, h);
        let enc = encode_idr_intra_au(&y, &cb, &cr, w, h, 27).expect("encode");
        let ctus = decode_annexb_sequence_debug(&enc.au).expect("walk");
        let mut nxn = 0usize;
        let mut two_n = 0usize;
        for (_, _, ctu) in &ctus {
            if let CodingQuadtree::Leaf(cu) = &ctu.quadtree {
                match cu.part_mode {
                    PartMode::PartNxN => nxn += 1,
                    PartMode::Part2Nx2N => two_n += 1,
                    other => panic!("unexpected part mode {other:?}"),
                }
            }
        }
        assert!(nxn > 0, "no PART_NxN CU selected ({two_n} 2Nx2N)");
    }

    /// Rate/distortion sanity: low QP is near-transparent, and
    /// quality degrades monotonically-ish while staying decodable.
    #[test]
    fn intra_quality_tracks_qp() {
        let (w, h) = (64usize, 64usize);
        let (y, cb, cr) = planes(w, h);
        let at = |qp: i32| {
            let enc = encode_idr_intra_au(&y, &cb, &cr, w, h, qp).expect("encode");
            (psnr(&enc.recon_y, &y), enc.au.len())
        };
        let (p4, s4) = at(4);
        let (p22, s22) = at(22);
        let (p40, s40) = at(40);
        assert!(p4 > 45.0, "qp4 luma PSNR {p4:.1} dB");
        assert!(p22 > 33.0, "qp22 luma PSNR {p22:.1} dB");
        assert!(p40 > 22.0, "qp40 luma PSNR {p40:.1} dB");
        assert!(p4 > p22 && p22 > p40, "PSNR decreases with QP");
        assert!(
            s4 > s22 && s22 > s40,
            "bytes decrease with QP ({s4} > {s22} > {s40})"
        );
    }

    /// QP 22 on this content should be visually transparent while far
    /// smaller than the PCM (raw) coding — i.e. the transform path
    /// actually compresses.
    #[test]
    fn intra_beats_pcm_size_at_high_quality() {
        let (w, h) = (64usize, 64usize);
        let (y, cb, cr) = planes(w, h);
        let enc = encode_idr_intra_au(&y, &cb, &cr, w, h, 22).expect("encode");
        let raw = w * h * 3 / 2;
        assert!(
            enc.au.len() < raw / 2,
            "compressed {} bytes vs raw {raw}",
            enc.au.len()
        );
    }

    #[test]
    fn rejects_bad_inputs() {
        let (y, cb, cr) = planes(16, 16);
        assert!(matches!(
            encode_idr_intra_au(&y, &cb, &cr, 20, 16, 26),
            Err(IntraEncodeError::BadDimensions { .. })
        ));
        assert!(matches!(
            encode_idr_intra_au(&y, &cb, &cr, 16, 16, 52),
            Err(IntraEncodeError::BadQp(52))
        ));
        assert!(matches!(
            encode_idr_intra_au(&y, &cb, &cr, 32, 16, 26),
            Err(IntraEncodeError::PlaneSize { .. })
        ));
    }

    /// Table 8-10 spot pins.
    #[test]
    fn chroma_qp_mapping_matches_table_8_10() {
        assert_eq!(chroma_qp_420(0), 0);
        assert_eq!(chroma_qp_420(29), 29);
        assert_eq!(chroma_qp_420(30), 29);
        assert_eq!(chroma_qp_420(33), 32);
        assert_eq!(chroma_qp_420(34), 33);
        assert_eq!(chroma_qp_420(35), 33);
        assert_eq!(chroma_qp_420(36), 34);
        assert_eq!(chroma_qp_420(37), 34);
        assert_eq!(chroma_qp_420(38), 35);
        assert_eq!(chroma_qp_420(43), 37);
        assert_eq!(chroma_qp_420(44), 38);
        assert_eq!(chroma_qp_420(51), 45);
    }
}
