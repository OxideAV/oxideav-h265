//! Quantization tools of the quadtree coder: rate-distortion
//! optimised quantization (RDOQ) over the exact §7.3.8.11
//! `residual_coding( )` bin structure, and the §7.3.8.11 sign-data-
//! hiding parity adjustment.
//!
//! **Bin costs.** Every context-coded bin is priced from the CABAC
//! probability state of its context model: the state's LPS
//! probability is read off Table 9-52 (`rangeTabLps[ pStateIdx ][ qRangeIdx ]`
//! over the four `ivlCurrRange` quarters, averaged), and the cost is
//! `−log2( p )` in 1/256-bit units — computed with integer arithmetic
//! only ([`fixed_log2_ratio`]) so the decisions are bit-identical on
//! every platform. Bypass bins cost one bit.
//!
//! **RDOQ** ([`quantize_tb`] with a [`RdoqModel`]) walks the block in
//! the decoder's reverse §6.5 scan and, per coefficient, picks the
//! level among `{ round-nearest, round-nearest − 1, 0 }` minimising
//! `D + λ·R` where `D` is the coefficient-domain squared error mapped
//! to the sample domain through the transform gain and `R` the exact
//! bins the level costs at that position (`sig_coeff_flag` with the
//! §9.3.4.2.5 ctxInc, `coeff_abs_level_greater1/2_flag` through the
//! §9.3.4.2.6/.7 context sets, the bypass sign, and the §9.3.3.11
//! Rice-adapted `coeff_abs_level_remaining` bins). It then elects the
//! last significant position (each candidate priced with its
//! `last_sig_coeff_{x,y}_prefix/suffix` bins against the distortion of
//! zeroing everything after it) and zeroes whole coded sub-blocks
//! whose `coded_sub_block_flag == 0` alternative is cheaper.
//!
//! **Sign data hiding** ([`apply_sign_hiding`]): for every 4x4
//! sub-block where `signHidden` holds (`lastSigScanPos −
//! firstSigScanPos > 3`), the decoder infers the sign of the
//! first-in-scan-order coefficient from the parity of `sumAbsLevel`;
//! when the quantized parity disagrees, the cheapest ±1 level
//! adjustment (quantization residue distortion + a rate estimate,
//! sub-block consistency re-checked after the change) is applied.

use std::sync::OnceLock;

use crate::binarization::{
    coded_sub_block_flag_ctx_inc_with_edge, coeff_abs_level_greater2_flag_ctx_inc,
    coeff_abs_level_remaining_c_max_eq_9_26, coeff_abs_level_remaining_c_rice_param_eq_9_24,
    last_sig_coeff_position, last_sig_coeff_prefix_cmax, last_sig_coeff_prefix_ctx_inc,
    last_sig_coeff_prefix_ctx_offset_shift, last_sig_coeff_suffix_n_bits,
    sig_coeff_flag_ctx_inc_from_sig_ctx, sig_coeff_flag_sig_ctx_dc, sig_coeff_flag_sig_ctx_general,
    sig_coeff_flag_sig_ctx_log2_2, Greater1State,
};
use crate::cabac::{ContextModel, RANGE_TAB_LPS};
use crate::encoder::bitwriter::BitWriter;
use crate::residual::ResidualContexts;
use crate::scaling_list::{ScalingFactorMatrix, ScalingListData, NUM_MATRIX_IDS, NUM_SIZE_IDS};
use crate::scan::{scan_order, ScanIdx};
use crate::transform::LEVEL_SCALE;

/// Fixed 8-bit depth.
const BIT_DEPTH: u32 = 8;
/// Cost of one bypass bin (1/256-bit units).
const BYPASS_COST: u64 = 256;
/// §7.4.9.11 CoeffMax for the non-extended-precision profiles.
const COEFF_MAX: i64 = 0x7FFF;

/// `round( 256 · log2( den / num ) )` for `0 < num <= den`, in
/// integer arithmetic: the integer part by halving, eight fractional
/// bits by squaring a Q32 mantissa.
#[must_use]
pub fn fixed_log2_ratio(num: u64, den: u64) -> u32 {
    debug_assert!(num > 0 && num <= den);
    let mut int_part = 0u32;
    let mut n = num;
    while n * 2 <= den {
        n *= 2;
        int_part += 1;
    }
    // mantissa = den / n in [1, 2), Q32.
    let mut m: u64 = ((u128::from(den) << 32) / u128::from(n)) as u64;
    let mut frac = 0u32;
    for _ in 0..9 {
        m = ((u128::from(m) * u128::from(m)) >> 32) as u64;
        frac <<= 1;
        if m >= 2u64 << 32 {
            frac |= 1;
            m >>= 1;
        }
    }
    // Nine fractional bits computed; round to eight.
    ((int_part << 9) + frac).div_ceil(2)
}

/// Per-probability-state bin costs: `[pStateIdx][bin_is_lps]` in
/// 1/256-bit units (Table 9-52-derived, see the module docs).
fn bin_cost_table() -> &'static [[u32; 2]; 64] {
    static TABLE: OnceLock<[[u32; 2]; 64]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [[0u32; 2]; 64];
        for (s, row) in RANGE_TAB_LPS.iter().enumerate() {
            // Average the LPS fraction over the four range quarters
            // (midpoints 288, 352, 416, 480), scaled to a common
            // denominator so the mean is exact.
            let quarters = [288u64, 352, 416, 480];
            let mut num = 0u64;
            let den = 4 * 288 * 352 * 416 * 480;
            for (q, &lps) in row.iter().enumerate() {
                let others: u64 = quarters
                    .iter()
                    .enumerate()
                    .filter(|&(k, _)| k != q)
                    .map(|(_, &v)| v)
                    .product();
                num += u64::from(lps) * others;
            }
            // p_lps = num / den; cost_lps = -log2(p_lps),
            // cost_mps = -log2(1 - p_lps).
            t[s][1] = fixed_log2_ratio(num, den);
            t[s][0] = fixed_log2_ratio(den - num, den);
        }
        t
    })
}

/// Cost (1/256 bit) of coding `bin` with context `ctx`.
#[inline]
fn ctx_cost(ctx: &ContextModel, bin: u8) -> u64 {
    u64::from(bin_cost_table()[ctx.p_state_idx as usize][usize::from(bin != ctx.val_mps)])
}

/// The RDOQ rate model: a snapshot of the slice's residual-coding
/// context models (the shadow emission's state at the CTB start).
#[derive(Debug, Clone)]
pub struct RdoqModel {
    /// The §7.3.8.11 context banks the bins are priced against.
    pub contexts: ResidualContexts,
}

impl RdoqModel {
    /// A model over the slice-initial context states (§9.3.2.2
    /// `initType` / `SliceQpY`).
    #[must_use]
    pub fn slice_initial(init_type: u8, slice_qp_y: i32) -> Self {
        Self {
            contexts: ResidualContexts::init(init_type, slice_qp_y),
        }
    }
}

/// The scaling lists a stream configuration selects
/// ([`crate::encoder::ctu::TreeCfg::scaling_lists`]): `0` none
/// (`scaling_list_enabled_flag == 0`), `1` the §7.4.5 Table 7-5 /
/// 7-6 defaults (enabled, no `scaling_list_data( )` transmitted), `2`
/// a flattened custom family (each default factor's deviation from
/// 16 halved) and `3` a steepened one (deviation doubled), the custom
/// families transmitted explicitly in the SPS. Returns `(data,
/// transmitted)`; `None` for `0`.
#[must_use]
pub fn scaling_lists_for(mode: u8) -> Option<(ScalingListData, bool)> {
    match mode {
        0 => None,
        1 => Some((ScalingListData::all_default(), false)),
        m => {
            let mut data = ScalingListData::all_default();
            for size_lists in data.lists.iter_mut() {
                for list in size_lists.iter_mut() {
                    for c in list.coef.iter_mut() {
                        let dev = i32::from(*c) - 16;
                        let dev = if m == 2 { dev / 2 } else { dev * 2 };
                        *c = (16 + dev).clamp(1, 255) as u16;
                    }
                    // Custom DC factors stay at the default 16.
                    list.dc_coef = 16;
                }
            }
            Some((data, true))
        }
    }
}

/// §7.3.4 `scaling_list_data( )` — every slot coded explicitly
/// (`scaling_list_pred_mode_flag == 1`): the `sizeId > 1` DC
/// coefficient, then the DPCM `scaling_list_delta_coef` run over the
/// up-right-diagonal-ordered list (each delta the −128..=127
/// representative of `coef − nextCoef` mod 256).
pub fn write_scaling_list_data(w: &mut BitWriter, data: &ScalingListData) {
    for size_id in 0..NUM_SIZE_IDS {
        let step = if size_id == 3 { 3 } else { 1 };
        let mut matrix_id = 0usize;
        while matrix_id < NUM_MATRIX_IDS {
            let list = &data.lists[size_id][matrix_id];
            w.put_bit(1); // scaling_list_pred_mode_flag
            let coef_num = 64.min(1usize << (4 + (size_id << 1)));
            let mut next_coef: i32 = 8;
            if size_id > 1 {
                w.se(i32::from(list.dc_coef) - 8); // scaling_list_dc_coef_minus8
                next_coef = i32::from(list.dc_coef);
            }
            for &c in list.coef.iter().take(coef_num) {
                let delta = (i32::from(c) - next_coef).rem_euclid(256);
                let delta = if delta > 127 { delta - 256 } else { delta };
                w.se(delta); // scaling_list_delta_coef
                next_coef = i32::from(c);
            }
            matrix_id += step;
        }
    }
}

/// One transform block's quantization request.
#[derive(Debug, Clone, Copy)]
pub struct TbQuant<'a> {
    /// `log2TrafoSize` (2..=5).
    pub log2: u32,
    /// The block's `qP` (luma `QpY` or the Table 8-10-mapped chroma QP).
    pub qp: u32,
    /// `cIdx > 0`.
    pub is_chroma: bool,
    /// The §7.4.9.11 scan.
    pub scan: ScanIdx,
    /// Sample-domain Lagrangian (SSD per bin).
    pub lambda: u64,
    /// RDOQ model; `None` selects the plain deadzone quantizer.
    pub model: Option<&'a RdoqModel>,
    /// Apply the sign-data-hiding parity adjustment.
    pub sign_hiding: bool,
    /// The §7.4.5 `ScalingFactor[ sizeId ][ matrixId ]` matrix of the
    /// block (`scaling_list_enabled_flag == 1`); `None` is the flat
    /// `m[ x ][ y ] == 16` of §8.6.3.
    pub scaling: Option<&'a ScalingFactorMatrix>,
}

/// §8.6.3-derived reciprocal quantizer scale (`round( 2^20 /
/// levelScale[ qP % 6 ] )`) — the flat `m == 16` case.
fn quant_scale(qp_rem: u32) -> u64 {
    let ls = u64::from(LEVEL_SCALE[qp_rem as usize] as u32);
    ((1u64 << 20) + ls / 2) / ls
}

/// The per-position reciprocal scale under a scaling list:
/// `round( 2^20 · 16 / ( levelScale[ qP % 6 ] · m[ x ][ y ] ) )`,
/// the inverse of the eq. 8-309 `m · levelScale` product.
fn quant_scale_m(qp_rem: u32, m: u16) -> u64 {
    let d = u64::from(LEVEL_SCALE[qp_rem as usize] as u32) * u64::from(m.max(1));
    ((1u64 << 24) + d / 2) / d
}

/// One reciprocal scale per coefficient (row-major).
fn scales_of(req: &TbQuant<'_>, count: usize) -> Vec<u64> {
    match req.scaling {
        None => vec![quant_scale(req.qp % 6); count],
        Some(sf) => sf
            .coef
            .iter()
            .map(|&m| quant_scale_m(req.qp % 6, m))
            .collect(),
    }
}

/// `qBits = 14 + qP / 6 + ( 15 − BitDepth − log2TbS )`.
fn q_bits(log2: u32, qp: u32) -> u32 {
    14 + qp / 6 + (15 - BIT_DEPTH - log2)
}

/// Coefficient-domain squared error of coding the scaled magnitude
/// `q` (`|coef| · scale`) as `level`.
fn coef_sq_err(q: u64, level: u64, qbits: u32, scale: u64) -> u64 {
    let e = q.abs_diff(level << qbits);
    ((u128::from(e) * u128::from(e)) / (u128::from(scale) * u128::from(scale))) as u64
}

/// Number of bypass bins `coeff_abs_level_remaining == value` takes
/// with Rice parameter `k` (§9.3.3.11).
fn remaining_bins(value: u32, k: u32) -> u64 {
    let c_max = coeff_abs_level_remaining_c_max_eq_9_26(k);
    if value < c_max {
        u64::from((value >> k) + 1 + k)
    } else {
        let kk = k + 1;
        let v = u64::from(value - c_max);
        let mut prefix_ones = 0u64;
        while (((1u64 << (prefix_ones + 1)) - 1) << kk) <= v {
            prefix_ones += 1;
        }
        4 + prefix_ones + 1 + prefix_ones + u64::from(kk)
    }
}

/// Plain deadzone quantizer (one-third rounding offset) on one
/// coefficient magnitude.
fn deadzone_level(q: u64, qbits: u32) -> u64 {
    ((q + (1u64 << qbits) / 3) >> qbits).min(COEFF_MAX as u64)
}

/// Quantize one TB's forward-transform coefficients (row-major) to
/// `TransCoeffLevel` under `req`: the deadzone quantizer, or RDOQ
/// when a model is supplied, then the sign-hiding parity adjustment
/// when requested.
#[must_use]
pub fn quantize_tb(coef: &[i32], req: &TbQuant<'_>) -> Vec<i32> {
    let qbits = q_bits(req.log2, req.qp);
    let scales = scales_of(req, coef.len());
    let q: Vec<u64> = coef
        .iter()
        .zip(&scales)
        .map(|(&c, &scale)| u64::from(c.unsigned_abs()) * scale)
        .collect();
    let mut levels: Vec<i32> = match req.model {
        Some(model) => rdoq_levels(&q, coef, req, model, qbits, &scales),
        None => q
            .iter()
            .zip(coef)
            .map(|(&qv, &c)| deadzone_level(qv, qbits) as i32 * c.signum())
            .collect(),
    };
    if req.sign_hiding && levels.iter().any(|&l| l != 0) {
        apply_sign_hiding(&mut levels, &q, coef, req, qbits, &scales);
    }
    levels
}

/// Sample-domain λ lifted to the coefficient-squared-error domain
/// (`coef = 2^(7 − log2) · orthonormal`), in 1/256-bit-rate units.
fn lambda_coef(lambda: u64, log2: u32) -> u64 {
    lambda << (2 * (7 - log2))
}

/// One coefficient's RDOQ bookkeeping.
#[derive(Clone, Copy, Default)]
struct CoefDecision {
    /// Row-major index.
    idx: usize,
    /// Chosen |level|.
    level: u64,
    /// `256 · D` of the chosen level.
    dist: u64,
    /// `256 · D` of level 0.
    dist0: u64,
    /// λ·R of the chosen level, sig flag included.
    rate: u64,
    /// λ·R of the `sig_coeff_flag == 1` bin alone (0 when not coded).
    rate_sig1: u64,
    /// Whether the position sits in a coded sub-block.
    in_coded_sb: bool,
}

/// The RDOQ level decisions (see the module docs).
#[allow(clippy::too_many_lines)]
fn rdoq_levels(
    q: &[u64],
    coef: &[i32],
    req: &TbQuant<'_>,
    model: &RdoqModel,
    qbits: u32,
    scales: &[u64],
) -> Vec<i32> {
    let log2 = req.log2;
    let size = 1usize << log2;
    let is_chroma = req.is_chroma;
    let scan_idx_num = u32::from(req.scan.index());
    let lam = lambda_coef(req.lambda, log2);
    let ctxs = &model.contexts;
    let pos_scan = scan_order(2, req.scan).expect("4x4 scan");
    let sub_scan = scan_order((log2 - 2) as u8, req.scan).expect("sub-block scan");
    let num_sb_1d = 1usize << (log2 - 2);
    let n_sb = num_sb_1d * num_sb_1d;
    let half = 1u64 << (qbits - 1);

    let index_of = |sb_i: usize, n: usize| -> (usize, u32, u32, u32, u32) {
        let sb = sub_scan[sb_i];
        let (xs, ys) = (u32::from(sb.x), u32::from(sb.y));
        let xc = (xs << 2) + u32::from(pos_scan[n].x);
        let yc = (ys << 2) + u32::from(pos_scan[n].y);
        (yc as usize * size + xc as usize, xc, yc, xs, ys)
    };

    // Naive last position (round-nearest levels).
    let mut naive_last: Option<(usize, usize)> = None;
    for sb_i in 0..n_sb {
        for n in 0..16 {
            let (idx, ..) = index_of(sb_i, n);
            if (q[idx] + half) >> qbits > 0 {
                naive_last = Some((sb_i, n));
            }
        }
    }
    let Some((last_sb, last_n)) = naive_last else {
        return vec![0; size * size];
    };

    // Sequential decisions in reverse scan order.
    let mut decisions: Vec<CoefDecision> = Vec::with_capacity(size * size);
    let mut csbf = vec![0u8; n_sb];
    let csbf_at = |grid: &[u8], xs: usize, ys: usize| -> u8 {
        if xs < num_sb_1d && ys < num_sb_1d {
            grid[ys * num_sb_1d + xs]
        } else {
            0
        }
    };
    // Per sub-block accounting for the zero-out election:
    // (first decision index, csbf-1 rate, csbf-0 rate).
    let mut sb_spans: Vec<(usize, usize, u64, u64)> = Vec::with_capacity(n_sb);
    let mut g1_state = Greater1State::new();
    let mut last_g1_bin: u8 = 0;

    for sb_i in (0..=last_sb).rev() {
        let start_n = if sb_i == last_sb { last_n } else { 15 };
        let sb = sub_scan[sb_i];
        let (xs, ys) = (u32::from(sb.x), u32::from(sb.y));
        let right = csbf_at(&csbf, xs as usize + 1, ys as usize);
        let below = csbf_at(&csbf, xs as usize, ys as usize + 1);
        // coded_sub_block_flag rates (coded for 0 < i < lastSubBlock).
        let (csbf_r1, csbf_r0) = if sb_i < last_sb && sb_i > 0 {
            let inc = coded_sub_block_flag_ctx_inc_with_edge(is_chroma, xs, ys, log2, right, below);
            let ctx = &ctxs.coded_sub_block_flag[inc as usize];
            (lam * ctx_cost(ctx, 1), lam * ctx_cost(ctx, 0))
        } else {
            (0, 0)
        };
        let span_start = decisions.len();

        let mut num_sig: u32 = 0;
        let mut first_g1_done = false;
        let mut entered = false;
        let mut c_last_abs_level: u32 = 0;
        let mut c_last_rice_param: u32 = 0;
        let mut any_nonzero = false;
        for n in (0..=start_n).rev() {
            let (idx, xc, yc, _, _) = index_of(sb_i, n);
            let qv = q[idx];
            let scale = scales[idx];
            let dist0 = coef_sq_err(qv, 0, qbits, scale) * 256;
            // sig_coeff_flag context (always priced; the last-position
            // election subtracts it for the elected last).
            let sig_ctx = if log2 == 2 {
                sig_coeff_flag_sig_ctx_log2_2(xc & 3, yc & 3)
            } else if xc + yc == 0 {
                sig_coeff_flag_sig_ctx_dc(is_chroma, log2, scan_idx_num)
            } else {
                sig_coeff_flag_sig_ctx_general(
                    is_chroma,
                    log2,
                    xc,
                    yc,
                    xs,
                    ys,
                    right,
                    below,
                    scan_idx_num,
                )
            };
            let sig_inc = sig_coeff_flag_ctx_inc_from_sig_ctx(sig_ctx, is_chroma) as usize;
            let sig_model = &ctxs.sig_coeff_flag[sig_inc];
            let rate_sig1 = lam * ctx_cost(sig_model, 1);
            let rate_sig0 = lam * ctx_cost(sig_model, 0);

            let l_max = ((qv + half) >> qbits).min(COEFF_MAX as u64);
            let mut best = CoefDecision {
                idx,
                level: 0,
                dist: dist0,
                dist0,
                rate: rate_sig0,
                rate_sig1,
                in_coded_sb: true,
            };
            if l_max > 0 {
                // Price the greater1 / greater2 / remaining bins of
                // each nonzero candidate at this position's state.
                let g1_coded = num_sig < 8;
                let g1_inc = if g1_coded {
                    if !entered {
                        g1_state.on_subblock_entry(sb_i as u32, is_chroma, last_g1_bin);
                    }
                    Some(g1_state.current_ctx_inc(is_chroma) as usize)
                } else {
                    None
                };
                let lo = if l_max > 1 { l_max - 1 } else { 1 };
                for level in (lo..=l_max).rev() {
                    let dist = coef_sq_err(qv, level, qbits, scale) * 256;
                    let mut bits = rate_sig1 + lam * BYPASS_COST; // sig + sign
                    let mut threshold = 1u32;
                    if let Some(inc) = g1_inc {
                        let g1 = u8::from(level > 1);
                        bits += lam * ctx_cost(&ctxs.coeff_abs_level_greater1_flag[inc], g1);
                        threshold = 2;
                        if g1 == 1 && !first_g1_done {
                            let g2_inc = coeff_abs_level_greater2_flag_ctx_inc(
                                g1_state.ctx_set(),
                                is_chroma,
                            ) as usize;
                            bits += lam
                                * ctx_cost(
                                    &ctxs.coeff_abs_level_greater2_flag[g2_inc],
                                    u8::from(level > 2),
                                );
                            threshold = 3;
                        }
                    }
                    if level >= u64::from(threshold) {
                        let rice = coeff_abs_level_remaining_c_rice_param_eq_9_24(
                            c_last_abs_level,
                            c_last_rice_param,
                        );
                        bits += lam
                            * BYPASS_COST
                            * remaining_bins((level - u64::from(threshold)) as u32, rice);
                    }
                    if dist + bits < best.dist + best.rate {
                        best.level = level;
                        best.dist = dist;
                        best.rate = bits;
                    }
                }
                // Commit the state transitions of the chosen level.
                if best.level > 0 {
                    any_nonzero = true;
                    if let Some(_inc) = g1_inc {
                        entered = true;
                        let g1 = u8::from(best.level > 1);
                        g1_state.on_coeff_abs_level_greater1_flag(g1);
                        last_g1_bin = g1;
                        let threshold = if g1 == 1 && !first_g1_done {
                            first_g1_done = true;
                            3
                        } else {
                            2
                        };
                        if best.level >= threshold {
                            let rice = coeff_abs_level_remaining_c_rice_param_eq_9_24(
                                c_last_abs_level,
                                c_last_rice_param,
                            );
                            c_last_abs_level = best.level as u32;
                            c_last_rice_param = rice;
                        }
                    } else {
                        let rice = coeff_abs_level_remaining_c_rice_param_eq_9_24(
                            c_last_abs_level,
                            c_last_rice_param,
                        );
                        c_last_abs_level = best.level as u32;
                        c_last_rice_param = rice;
                    }
                    num_sig += 1;
                }
            }
            decisions.push(best);
        }
        csbf[ys as usize * num_sb_1d + xs as usize] = u8::from(any_nonzero);
        sb_spans.push((sb_i, span_start, csbf_r1, csbf_r0));
    }

    // Last-position election: candidate = every nonzero decision;
    // cost(p) = Σ_{before p in decode order} dist0 + last-pos bins +
    // (chosen(p) − sig1 bin) + Σ_{after p} chosen.
    let (ctx_off, ctx_shift) = last_sig_coeff_prefix_ctx_offset_shift(log2, is_chroma);
    let c_max = last_sig_coeff_prefix_cmax(log2);
    let last_pos_rate = |xc: u32, yc: u32| -> u64 {
        let (wx, wy) = if req.scan == ScanIdx::Vertical {
            (yc, xc)
        } else {
            (xc, yc)
        };
        let mut bits = 0u64;
        for (v, bank) in [
            (wx, &ctxs.last_sig_coeff_x_prefix),
            (wy, &ctxs.last_sig_coeff_y_prefix),
        ] {
            let (prefix, suffix_bits) = split_last_position(v, c_max);
            for bin_idx in 0..prefix {
                let inc = last_sig_coeff_prefix_ctx_inc(bin_idx, ctx_off, ctx_shift) as usize;
                bits += ctx_cost(&bank[inc], 1);
            }
            if prefix < c_max {
                let inc = last_sig_coeff_prefix_ctx_inc(prefix, ctx_off, ctx_shift) as usize;
                bits += ctx_cost(&bank[inc], 0);
            }
            bits += BYPASS_COST * u64::from(suffix_bits);
        }
        lam * bits
    };
    let total_chosen: u64 = decisions.iter().map(|d| d.dist + d.rate).sum();
    let mut best_last: Option<usize> = None;
    let mut best_cost = u64::MAX;
    let mut prefix_zero = 0u64; // Σ dist0 of decisions before k
    let mut prefix_chosen = 0u64; // Σ (dist + rate) of decisions before k
    for (k, d) in decisions.iter().enumerate() {
        if d.level > 0 {
            let xc = (d.idx % size) as u32;
            let yc = (d.idx / size) as u32;
            let tail = total_chosen - prefix_chosen - (d.dist + d.rate);
            let cost = prefix_zero + last_pos_rate(xc, yc) + d.dist + d.rate - d.rate_sig1 + tail;
            if cost < best_cost {
                best_cost = cost;
                best_last = Some(k);
            }
        }
        prefix_zero += d.dist0;
        prefix_chosen += d.dist + d.rate;
    }
    let Some(last_k) = best_last else {
        return vec![0; size * size];
    };
    for d in &mut decisions[..last_k] {
        d.level = 0;
    }

    // Sub-block zero-out: sub-blocks strictly inside (0, lastSubBlock)
    // whose csbf == 0 alternative is cheaper.
    let n_dec = decisions.len();
    for (s, &(sb_i, start, r1, r0)) in sb_spans.iter().enumerate() {
        let end = sb_spans.get(s + 1).map_or(n_dec, |&(_, st, _, _)| st);
        let span = &mut decisions[start..end];
        let last_sb_now = start <= last_k && last_k < end;
        if last_sb_now || sb_i == 0 {
            continue;
        }
        let coded: u64 = span.iter().map(|d| d.dist + d.rate).sum::<u64>() + r1;
        let zeroed: u64 = span.iter().map(|d| d.dist0).sum::<u64>() + r0;
        if zeroed <= coded {
            for d in span.iter_mut() {
                d.level = 0;
                d.in_coded_sb = false;
            }
        }
    }

    let mut levels = vec![0i32; size * size];
    for d in &decisions {
        if d.level > 0 {
            levels[d.idx] = d.level as i32 * coef[d.idx].signum();
        }
    }
    levels
}

/// Split a `LastSignificantCoeff*` position into `(prefix,
/// suffix_bit_count)` (the inverse of §7.4.9.11 eqs. 7-74..7-77).
fn split_last_position(v: u32, c_max: u32) -> (u32, u32) {
    for prefix in 0..=c_max {
        let n_bits = last_sig_coeff_suffix_n_bits(prefix);
        if n_bits == 0 {
            if last_sig_coeff_position(prefix, None) == v {
                return (prefix, 0);
            }
        } else {
            let base = last_sig_coeff_position(prefix, Some(0));
            if v >= base && v < base + (1u32 << n_bits) {
                return (prefix, n_bits);
            }
        }
    }
    unreachable!("last-sig position exceeds the TB")
}

/// Rough bit estimate of one |level| (sig + sign + greater flags +
/// Rice bins) for the sign-hiding rate term.
fn level_bits_estimate(level: u64) -> u64 {
    if level == 0 {
        0
    } else {
        3 + 2 * u64::from(64 - level.leading_zeros())
    }
}

/// The §7.3.8.11 sign-hiding parity adjustment (see the module docs).
/// `levels` is row-major and is modified in place.
fn apply_sign_hiding(
    levels: &mut [i32],
    q: &[u64],
    coef: &[i32],
    req: &TbQuant<'_>,
    qbits: u32,
    scales: &[u64],
) {
    let log2 = req.log2;
    let size = 1usize << log2;
    let lam = lambda_coef(req.lambda, log2);
    let pos_scan = scan_order(2, req.scan).expect("4x4 scan");
    let sub_scan = scan_order((log2 - 2) as u8, req.scan).expect("sub-block scan");
    let num_sb_1d = 1usize << (log2 - 2);
    let n_sb = num_sb_1d * num_sb_1d;

    // Sub-block descriptor: (firstSigScanPos, lastSigScanPos, sumAbs,
    // first-coefficient index) in the §7.3.8.11 sense.
    let describe = |levels: &[i32], sb_i: usize| -> Option<(usize, usize, u64, usize)> {
        let sb = sub_scan[sb_i];
        let mut first = 16usize;
        let mut last: Option<usize> = None;
        let mut sum = 0u64;
        let mut first_idx = 0usize;
        for n in (0..16).rev() {
            let xc = ((sb.x as usize) << 2) + pos_scan[n].x as usize;
            let yc = ((sb.y as usize) << 2) + pos_scan[n].y as usize;
            let idx = yc * size + xc;
            let l = levels[idx];
            if l != 0 {
                if last.is_none() {
                    last = Some(n);
                }
                first = n;
                first_idx = idx;
                sum += u64::from(l.unsigned_abs());
            }
        }
        last.map(|l| (first, l, sum, first_idx))
    };
    // Whether the sub-block is consistent: not hidden, or parity
    // matches the first coefficient's sign.
    let consistent = |levels: &[i32], sb_i: usize| -> bool {
        match describe(levels, sb_i) {
            None => true,
            Some((first, last, sum, first_idx)) => {
                if last - first <= 3 {
                    true
                } else {
                    let negative = levels[first_idx] < 0;
                    (sum % 2 == 1) == negative
                }
            }
        }
    };

    for (sb_i, sb) in sub_scan.iter().enumerate().take(n_sb) {
        if consistent(levels, sb_i) {
            continue;
        }
        let mut best: Option<(i128, usize, i32)> = None;
        for pos in pos_scan.iter().take(16) {
            let xc = ((sb.x as usize) << 2) + pos.x as usize;
            let yc = ((sb.y as usize) << 2) + pos.y as usize;
            let idx = yc * size + xc;
            let cur = i64::from(levels[idx]);
            let cur_abs = cur.unsigned_abs();
            let sign: i64 = if cur != 0 {
                cur.signum()
            } else if coef[idx] < 0 {
                -1
            } else {
                1
            };
            let rate_of = |l: u64| -> u64 { lam * 256 * level_bits_estimate(l) };
            let scale = scales[idx];
            let d_cur = coef_sq_err(q[idx], cur_abs, qbits, scale) * 256 + rate_of(cur_abs);
            for delta in [1i64, -1] {
                let new_abs = cur_abs as i64 + delta;
                if !(0..=COEFF_MAX).contains(&new_abs) {
                    continue;
                }
                let new_abs = new_abs as u64;
                let d_new = coef_sq_err(q[idx], new_abs, qbits, scale) * 256 + rate_of(new_abs);
                let cost = d_new as i128 - d_cur as i128;
                let candidate = (sign * new_abs as i64) as i32;
                let old = levels[idx];
                levels[idx] = candidate;
                let ok = consistent(levels, sb_i);
                levels[idx] = old;
                if ok && best.map_or(true, |(c, _, _)| cost < c) {
                    best = Some((cost, idx, candidate));
                }
            }
        }
        if let Some((_, idx, candidate)) = best {
            levels[idx] = candidate;
        }
        debug_assert!(consistent(levels, sb_i), "sign-hiding parity unresolved");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;
    use crate::cabac::CabacEngine;
    use crate::encoder::bitwriter::BitWriter;
    use crate::encoder::cabac::CabacEncoder;
    use crate::encoder::residual::encode_residual_coding;
    use crate::residual::{decode_residual_coding, ResidualCodingParams};

    #[test]
    fn fixed_log2_matches_known_points() {
        assert_eq!(fixed_log2_ratio(1, 1), 0);
        assert_eq!(fixed_log2_ratio(1, 2), 256);
        assert_eq!(fixed_log2_ratio(1, 4), 512);
        // log2(4/3) = 0.41504 -> 106.25 -> 106
        assert_eq!(fixed_log2_ratio(3, 4), 106);
        // log2(2) exactly 256; log2(1.5) = 0.58496 -> 149.75 -> 150
        assert_eq!(fixed_log2_ratio(2, 3), 150);
    }

    #[test]
    fn bin_costs_are_monotone_in_state() {
        let t = bin_cost_table();
        // State 0 is p = 0.5-ish: both bins near one bit.
        assert!(t[0][0] > 200 && t[0][0] < 300);
        for s in 1..64 {
            assert!(t[s][1] >= t[s - 1][1], "lps cost grows with state");
            assert!(t[s][0] <= t[s - 1][0], "mps cost shrinks with state");
        }
        // rangeTabLps[ 63 ] = 2 over ~384: about 7.6 bits.
        assert!(t[63][1] > 7 * 256, "state 63 LPS costs > 7 bits");
    }

    fn params(log2: u32, scan: ScanIdx, sdh: bool) -> ResidualCodingParams {
        ResidualCodingParams {
            log2_trafo_size: log2,
            is_chroma: false,
            scan_idx: scan,
            sign_data_hiding_enabled_flag: sdh,
            sign_hidden_suppressed: false,
            transform_skip_sig_ctx: false,
            persistent_rice_adaptation_enabled_flag: false,
            cabac_bypass_alignment_enabled_flag: false,
            extended_precision_processing_flag: false,
            bit_depth: 8,
            rice_stat_transform_skip: false,
        }
    }

    fn roundtrip(p: &ResidualCodingParams, levels: &[i32]) {
        let mut w = BitWriter::new();
        let mut cabac = CabacEncoder::new();
        let mut enc_ctx = ResidualContexts::init(0, 26);
        encode_residual_coding(&mut w, &mut cabac, &mut enc_ctx, p, levels).expect("encode");
        cabac.encode_terminate(&mut w, 1);
        w.align_zero();
        let bytes = w.finish();
        let mut engine = CabacEngine::new(BitReader::new(&bytes)).expect("init");
        let mut dec_ctx = ResidualContexts::init(0, 26);
        let block = decode_residual_coding(&mut engine, &mut dec_ctx, p).expect("decode");
        assert_eq!(block.levels, levels);
    }

    fn synthetic_coefs(log2: u32, seed: u32, amp: i32) -> Vec<i32> {
        let size = 1usize << log2;
        let mut x = seed;
        (0..size * size)
            .map(|i| {
                x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let (u, v) = (i % size, i / size);
                let decay = (u + v + 1) as i32;
                let r = ((x >> 16) % (2 * amp as u32 + 1)) as i32 - amp;
                r * 8 / decay
            })
            .collect()
    }

    #[test]
    fn rdoq_levels_roundtrip_and_never_exceed_round_nearest() {
        let model = RdoqModel::slice_initial(0, 30);
        for log2 in 2..=5u32 {
            for (seed, amp) in [(1u32, 400i32), (7, 4000), (9, 60)] {
                let coef = synthetic_coefs(log2, seed, amp);
                for qp in [22u32, 30, 38] {
                    let req = TbQuant {
                        log2,
                        qp,
                        is_chroma: false,
                        scan: ScanIdx::Diagonal,
                        lambda: 1u64 << ((qp - 9) / 3),
                        model: Some(&model),
                        sign_hiding: false,
                        scaling: None,
                    };
                    let levels = quantize_tb(&coef, &req);
                    let plain = quantize_tb(&coef, &TbQuant { model: None, ..req });
                    let qbits = q_bits(log2, qp);
                    let scale = quant_scale(qp % 6);
                    for (i, (&l, &c)) in levels.iter().zip(&coef).enumerate() {
                        let nearest =
                            (u64::from(c.unsigned_abs()) * scale + (1 << (qbits - 1))) >> qbits;
                        assert!(u64::from(l.unsigned_abs()) <= nearest, "coef {i}");
                        assert!(l == 0 || l.signum() == c.signum());
                    }
                    let nz: usize = levels.iter().filter(|&&l| l != 0).count();
                    let nz_plain: usize = plain.iter().filter(|&&l| l != 0).count();
                    assert!(nz <= nz_plain + 2, "rdoq should not inflate the block");
                    if levels.iter().any(|&l| l != 0) {
                        roundtrip(&params(log2, ScanIdx::Diagonal, false), &levels);
                    }
                }
            }
        }
    }

    #[test]
    fn sign_hiding_parity_holds_and_roundtrips() {
        let model = RdoqModel::slice_initial(0, 27);
        for log2 in 2..=5u32 {
            for scan in [ScanIdx::Diagonal, ScanIdx::Horizontal, ScanIdx::Vertical] {
                if log2 > 3 && scan != ScanIdx::Diagonal {
                    continue;
                }
                for seed in 1..12u32 {
                    let coef = synthetic_coefs(log2, seed, 900);
                    for use_model in [false, true] {
                        let req = TbQuant {
                            log2,
                            qp: 27,
                            is_chroma: false,
                            scan,
                            lambda: 64,
                            model: use_model.then_some(&model),
                            sign_hiding: true,
                            scaling: None,
                        };
                        let levels = quantize_tb(&coef, &req);
                        if levels.iter().any(|&l| l != 0) {
                            roundtrip(&params(log2, scan, true), &levels);
                        }
                    }
                }
            }
        }
    }

    /// Under the default §7.4.5 lists a flat-16 4x4 matrix reproduces
    /// the unscaled levels, and the 8x8 intra list (larger factors
    /// at high frequencies) quantizes those positions coarser; the
    /// levels stay encodable with and without RDOQ / sign hiding.
    #[test]
    fn scaling_lists_scale_per_position_and_roundtrip() {
        use crate::scaling_list::ScalingListData;
        let sf = ScalingListData::all_default().scaling_factors(1);
        let model = RdoqModel::slice_initial(0, 30);
        for log2 in 2..=5u32 {
            let coef = synthetic_coefs(log2, 5, 900);
            let matrix = &sf.factors[log2 as usize - 2][0];
            for (use_model, sdh) in [(false, false), (true, false), (true, true)] {
                let base = TbQuant {
                    log2,
                    qp: 30,
                    is_chroma: false,
                    scan: ScanIdx::Diagonal,
                    lambda: 64,
                    model: use_model.then_some(&model),
                    sign_hiding: sdh,
                    scaling: None,
                };
                let flat = quantize_tb(&coef, &base);
                let scaled = quantize_tb(
                    &coef,
                    &TbQuant {
                        scaling: Some(matrix),
                        ..base
                    },
                );
                if log2 == 2 {
                    assert_eq!(flat, scaled, "flat 4x4 list is a no-op");
                } else if !use_model {
                    // Every factor >= 16: never a larger magnitude.
                    for (f, s) in flat.iter().zip(&scaled) {
                        assert!(s.unsigned_abs() <= f.unsigned_abs());
                    }
                }
                if scaled.iter().any(|&l| l != 0) {
                    roundtrip(&params(log2, ScanIdx::Diagonal, sdh), &scaled);
                }
            }
        }
    }

    /// The §7.3.4 writer round-trips through the crate's parser for
    /// every family, and the custom families differ from the default.
    #[test]
    fn scaling_list_writer_roundtrips_through_the_parser() {
        use crate::bitreader::BitReader;
        for mode in 1..=3u8 {
            let (data, transmitted) = scaling_lists_for(mode).expect("lists");
            assert_eq!(transmitted, mode >= 2);
            let mut w = BitWriter::new();
            write_scaling_list_data(&mut w, &data);
            w.rbsp_trailing_bits();
            let bytes = w.finish();
            let parsed = ScalingListData::parse(&mut BitReader::new(&bytes)).expect("parse");
            // sizeId 3 only transmits matrixId 0 / 3; the other slots
            // keep their defaults in both.
            for size_id in 0..NUM_SIZE_IDS {
                for matrix_id in 0..NUM_MATRIX_IDS {
                    if size_id == 3 && matrix_id % 3 != 0 {
                        continue;
                    }
                    let (a, b) = (
                        &parsed.lists[size_id][matrix_id],
                        &data.lists[size_id][matrix_id],
                    );
                    assert_eq!(
                        a.coef, b.coef,
                        "size {size_id} matrix {matrix_id} mode {mode}"
                    );
                    // The DC factor only exists for sizeId > 1 (the
                    // parser leaves the unused slots at 8).
                    if size_id > 1 {
                        assert_eq!(a.dc_coef, b.dc_coef, "dc size {size_id} matrix {matrix_id}");
                    }
                }
            }
            if mode >= 2 {
                assert_ne!(
                    data.lists[1][0].coef,
                    ScalingListData::all_default().lists[1][0].coef
                );
            }
        }
    }

    #[test]
    fn deadzone_matches_legacy_quantizer() {
        for log2 in 2..=5u32 {
            let coef = synthetic_coefs(log2, 3, 700);
            for qp in [4u32, 22, 37, 51] {
                let req = TbQuant {
                    log2,
                    qp,
                    is_chroma: false,
                    scan: ScanIdx::Diagonal,
                    lambda: 1,
                    model: None,
                    sign_hiding: false,
                    scaling: None,
                };
                assert_eq!(
                    quantize_tb(&coef, &req),
                    crate::encoder::intra::quantize(&coef, 1 << log2, qp)
                );
            }
        }
    }
}
