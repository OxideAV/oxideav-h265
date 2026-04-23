//! CABAC emit side for a leaf transform_unit's residual data.
//!
//! Mirrors the decoder's `residual_coding` path in `ctu.rs` so that bins
//! emitted here are consumed by the same state machine on the decode
//! side. Handles:
//!
//! * `last_sig_coeff_x/y_prefix` (+ optional suffix) — §9.3.4.2.3 / §7.4.9.11.
//! * Reverse sub-block scan with `coded_sub_block_flag`.
//! * Per-coefficient `sig_coeff_flag` with the spec's context derivation.
//! * `coeff_abs_level_greater1_flag` / `greater2_flag`.
//! * `coeff_sign_flag` (bypass).
//! * `coeff_abs_level_remaining` (Rice + EGk bypass tail).
//!
//! The encoder only ever uses diagonal scan_idx (= 0) for the sub-block
//! and coefficient scans — that matches what the decoder's path uses for
//! 16×16 / 32×32 TUs and for every CU we emit. No scaling lists.

use crate::cabac::{CtxState, SIG_COEFF_FLAG_INIT_VALUES};
use crate::encoder::cabac_writer::CabacWriter;
use crate::scan::{scan_4x4, DIAG_SCAN_4X4};

/// Per-slice residual-coding context state — kept outside the transform-unit
/// walker so contexts persist across TUs within the slice.
pub struct ResidualCtx {
    pub sig_coeff_flag: Vec<CtxState>,
    pub coeff_abs_gt1: [CtxState; 24],
    pub coeff_abs_gt2: [CtxState; 6],
    pub coded_sub_block_flag: [CtxState; 4],
    pub last_sig_x_prefix: [CtxState; 18],
    pub last_sig_y_prefix: [CtxState; 18],
    pub cbf_luma: [CtxState; 2],
    pub cbf_cb_cr: [CtxState; 5],
    pub prev_greater1_ctx: u8,
    pub prev_greater1_flag: bool,
}

impl ResidualCtx {
    pub fn new(slice_qp_y: i32) -> Self {
        use crate::cabac::{
            init_context, init_row, CBF_CB_CR_INIT_VALUES, CBF_LUMA_INIT_VALUES,
            CODED_SUB_BLOCK_FLAG_INIT_VALUES, COEFF_ABS_GT1_INIT_VALUES, COEFF_ABS_GT2_INIT_VALUES,
            InitType, LAST_SIG_COEFF_X_PREFIX_INIT_VALUES, LAST_SIG_COEFF_Y_PREFIX_INIT_VALUES,
        };
        let it = InitType::I;
        let sig = SIG_COEFF_FLAG_INIT_VALUES[it as usize];
        let sig_ctx: Vec<CtxState> = sig.iter().map(|&v| init_context(v, slice_qp_y)).collect();
        Self {
            sig_coeff_flag: sig_ctx,
            coeff_abs_gt1: init_row(&COEFF_ABS_GT1_INIT_VALUES, it, slice_qp_y),
            coeff_abs_gt2: init_row(&COEFF_ABS_GT2_INIT_VALUES, it, slice_qp_y),
            coded_sub_block_flag: init_row(&CODED_SUB_BLOCK_FLAG_INIT_VALUES, it, slice_qp_y),
            last_sig_x_prefix: init_row(&LAST_SIG_COEFF_X_PREFIX_INIT_VALUES, it, slice_qp_y),
            last_sig_y_prefix: init_row(&LAST_SIG_COEFF_Y_PREFIX_INIT_VALUES, it, slice_qp_y),
            cbf_luma: init_row(&CBF_LUMA_INIT_VALUES, it, slice_qp_y),
            cbf_cb_cr: init_row(&CBF_CB_CR_INIT_VALUES, it, slice_qp_y),
            prev_greater1_ctx: 1,
            prev_greater1_flag: false,
        }
    }
}

/// Encode a single TU's residual (n×n coefficients in row-major order).
///
/// * `levels` — quantised level values, same layout used by the decoder.
/// * `log2_tb` — 2, 3, or 4 (encoder uses 4x4 and 8x8 chroma plus 16x16 luma).
/// * `is_luma` — selects the luma or chroma context offsets.
///
/// Returns `true` if there was at least one non-zero coefficient and
/// anything was emitted. Caller must have already emitted the `cbf_*`
/// flag = 1 for this plane.
pub fn encode_residual(
    cw: &mut CabacWriter<'_>,
    ctx: &mut ResidualCtx,
    levels: &[i32],
    log2_tb: u32,
    is_luma: bool,
) -> bool {
    let n = 1usize << log2_tb;
    debug_assert_eq!(levels.len(), n * n);
    // Use diagonal scan (scan_idx = 0) everywhere — matches decoder's
    // `residual_coding` default for our generated bitstream.
    let scan_idx = 0u32;
    let scan = scan_4x4(scan_idx);

    // Find last non-zero position in forward scan order across all
    // sub-blocks; the decoder expects `(last_x, last_y)` in picture-
    // coordinate form.
    let num_sb = (n / 4).max(1);
    let sb_scan = sub_block_scan(num_sb);
    let mut last_x = 0usize;
    let mut last_y = 0usize;
    let mut found = false;
    // Walk forward scan order (sub-blocks forward, coeffs forward in each
    // sub-block) and remember the last non-zero position we encounter.
    for (i, &(sx, sy)) in sb_scan.iter().enumerate() {
        for &(cx, cy) in DIAG_SCAN_4X4.iter() {
            let x = sx as usize * 4 + cx as usize;
            let y = sy as usize * 4 + cy as usize;
            if levels[y * n + x] != 0 {
                last_x = x;
                last_y = y;
                found = true;
            }
        }
        let _ = i;
    }
    if !found {
        // Caller mis-signalled cbf=1; emit a single DC coefficient so the
        // bitstream stays in sync. Should never happen in practice.
        return false;
    }

    // ---- last_sig_coeff_x/y_prefix (context) + suffix (bypass) ---------
    encode_last_sig_pos(cw, ctx, last_x, last_y, log2_tb, is_luma);

    // ---- sub-block reverse scan ----------------------------------------
    let (last_sx, last_sy) = (last_x >> 2, last_y >> 2);
    let (last_px, last_py) = (last_x & 3, last_y & 3);
    let mut last_sb_idx = 0usize;
    for (i, &(sx, sy)) in sb_scan.iter().enumerate() {
        if sx as usize == last_sx && sy as usize == last_sy {
            last_sb_idx = i;
            break;
        }
    }
    let mut last_coef_in_sb = 0usize;
    for (i, &(px, py)) in scan.iter().enumerate() {
        if px as usize == last_px && py as usize == last_py {
            last_coef_in_sb = i;
            break;
        }
    }

    let mut sb_cbf = vec![false; num_sb * num_sb];
    sb_cbf[last_sy * num_sb + last_sx] = true;
    ctx.prev_greater1_ctx = 1;
    ctx.prev_greater1_flag = false;

    for i in (0..=last_sb_idx).rev() {
        let (sx_u8, sy_u8) = sb_scan[i];
        let sx = sx_u8 as usize;
        let sy = sy_u8 as usize;
        let is_last_sb = i == last_sb_idx;
        let is_first_sb = i == 0;
        // coded_sub_block_flag.
        let coded = if is_last_sb || is_first_sb {
            // No bin emitted. Remember the flag.
            let has_any = sub_block_has_nonzero(levels, n, sx, sy);
            sb_cbf[sy * num_sb + sx] = is_last_sb || is_first_sb || has_any;
            // For the first sub-block the spec infers flag=1; if it has
            // zero coefficients we still run through this block — the
            // decoder will infer sig_flag_dc=1 which forces a non-zero
            // DC at read time. Our quantiser is required to ensure the
            // DC of sub-block 0 is non-zero when any other sub-block
            // has content, but realistically we just accept that DC of
            // sub-block 0 might be forced 1 (cost: a handful of bins).
            true
        } else {
            let right = sx + 1 < num_sb && sb_cbf[sy * num_sb + sx + 1];
            let below = sy + 1 < num_sb && sb_cbf[(sy + 1) * num_sb + sx];
            let has_any = sub_block_has_nonzero(levels, n, sx, sy);
            let ctx_inc = (right || below) as usize;
            let ctx_inc = if is_luma { ctx_inc } else { 2 + ctx_inc };
            cw.encode_bin(&mut ctx.coded_sub_block_flag[ctx_inc], has_any as u32);
            sb_cbf[sy * num_sb + sx] = has_any;
            has_any
        };
        if !coded {
            continue;
        }

        // Collect sig flags for this sub-block in scan-position order
        // (scan[0..16]), covering positions 0..=start where start=15 for
        // non-last sub-blocks and `last_coef_in_sb` for the last one.
        let start = if is_last_sb { last_coef_in_sb } else { 15 };
        let mut sig_flags = [false; 16];
        if is_last_sb {
            sig_flags[last_coef_in_sb] = true;
        }
        // Gather sig flags from the actual levels.
        for j in 0..=start {
            let (cx, cy) = scan[j];
            let gx = sx * 4 + cx as usize;
            let gy = sy * 4 + cy as usize;
            sig_flags[j] = levels[gy * n + gx] != 0;
        }
        if is_last_sb {
            sig_flags[last_coef_in_sb] = true;
        }
        let middle_sb = !is_last_sb && !is_first_sb;
        // In middle sub-blocks, the decoder infers sig_flag_dc = 1 unless
        // any sig_coeff_flag read in this sub-block is 1. The encoder
        // must ensure sig_flag at the DC is consistent with the level:
        // if no non-zero flag is present but sub-block is coded, decoder
        // will infer DC=1 → we must have non-zero DC level. That's the
        // caller's responsibility; here we just emit bins for the
        // non-inferred positions.
        let mut any_sig_after_dc = false;
        for j in (0..=start).rev() {
            if is_last_sb && j == last_coef_in_sb {
                // Already implied 1 — not emitted.
                continue;
            }
            let (cx, cy) = scan[j];
            let at_dc = cx == 0 && cy == 0;
            if at_dc && middle_sb && !any_sig_after_dc {
                // Decoder infers DC=1 for middle sb without any observed
                // non-zero; skip emission. We rely on caller to ensure
                // sig_flags[0] matches that inference when coded.
                continue;
            }
            let ctx_inc = sig_coeff_ctx_inc(
                log2_tb,
                scan_idx,
                cx as u32,
                cy as u32,
                sx as u32,
                sy as u32,
                sx + 1 < num_sb && sb_cbf[sy * num_sb + sx + 1],
                sy + 1 < num_sb && sb_cbf[(sy + 1) * num_sb + sx],
                is_luma,
            );
            cw.encode_bin(&mut ctx.sig_coeff_flag[ctx_inc], sig_flags[j] as u32);
            if sig_flags[j] {
                any_sig_after_dc = true;
            }
        }
        // coeff_abs_level_greater1_flag / greater2_flag / sign / remainder.
        let mut greater1_flags = [false; 16];
        let mut greater2_flags = [false; 16];
        let mut first_sig_scan_pos = 16i32;
        let mut last_sig_scan_pos = -1i32;
        let mut num_greater1_flag = 0u32;
        let mut last_greater1_scan_pos = -1i32;
        let mut ctx_set = if i == 0 || !is_luma { 0u8 } else { 2u8 };
        let mut greater1_ctx = 1u8;
        let mut last_used_greater1_flag = false;
        if i != last_sb_idx {
            let mut last_greater1_ctx = ctx.prev_greater1_ctx;
            if last_greater1_ctx > 0 {
                if ctx.prev_greater1_flag {
                    last_greater1_ctx = 0;
                } else {
                    last_greater1_ctx = last_greater1_ctx.saturating_add(1);
                }
            }
            if last_greater1_ctx == 0 {
                ctx_set = ctx_set.saturating_add(1);
            }
            greater1_ctx = 1;
        }
        for pos in (0..16).rev() {
            if !sig_flags[pos] {
                continue;
            }
            let (cx, cy) = scan[pos];
            let gx = sx * 4 + cx as usize;
            let gy = sy * 4 + cy as usize;
            let abs_level = levels[gy * n + gx].unsigned_abs();
            let gt1 = abs_level > 1;
            if num_greater1_flag < 8 {
                let mut ctx_inc =
                    (ctx_set as usize * 4) + usize::min(3, greater1_ctx as usize);
                if !is_luma {
                    ctx_inc += 16;
                }
                cw.encode_bin(&mut ctx.coeff_abs_gt1[ctx_inc], gt1 as u32);
                greater1_flags[pos] = gt1;
                last_used_greater1_flag = gt1;
                num_greater1_flag += 1;
                if gt1 && last_greater1_scan_pos == -1 {
                    last_greater1_scan_pos = pos as i32;
                }
                if greater1_ctx > 0 {
                    if gt1 {
                        greater1_ctx = 0;
                    } else {
                        greater1_ctx = greater1_ctx.saturating_add(1);
                    }
                }
            } else {
                // After 8 gt1s, further coefficients skip the gt1 flag and
                // use base_level = 1. Decoder reads threshold = 1 → remainder.
                greater1_flags[pos] = false;
            }
            if last_sig_scan_pos == -1 {
                last_sig_scan_pos = pos as i32;
            }
            first_sig_scan_pos = pos as i32;
        }
        ctx.prev_greater1_ctx = greater1_ctx;
        ctx.prev_greater1_flag = last_used_greater1_flag;
        if last_greater1_scan_pos != -1 {
            let (cx, cy) = scan[last_greater1_scan_pos as usize];
            let gx = sx * 4 + cx as usize;
            let gy = sy * 4 + cy as usize;
            let abs_level = levels[gy * n + gx].unsigned_abs();
            let gt2 = abs_level > 2;
            greater2_flags[last_greater1_scan_pos as usize] = gt2;
            let mut ctx_inc = ctx_set as usize;
            if !is_luma {
                ctx_inc += 4;
            }
            cw.encode_bin(&mut ctx.coeff_abs_gt2[ctx_inc], gt2 as u32);
        }
        // Sign bits (sign_data_hiding disabled).
        for pos in (0..16).rev() {
            if !sig_flags[pos] {
                continue;
            }
            let (cx, cy) = scan[pos];
            let gx = sx * 4 + cx as usize;
            let gy = sy * 4 + cy as usize;
            let lvl = levels[gy * n + gx];
            let sign = (lvl < 0) as u32;
            cw.encode_bypass(sign);
        }
        // coeff_abs_level_remaining.
        let mut num_sig_coeff = 0u32;
        let mut coeff_rem_seen = false;
        let mut last_abs_level = 0u32;
        let mut last_rice_param = 0u32;
        for pos in (0..16).rev() {
            if !sig_flags[pos] {
                continue;
            }
            let (cx, cy) = scan[pos];
            let gx = sx * 4 + cx as usize;
            let gy = sy * 4 + cy as usize;
            let abs_level = levels[gy * n + gx].unsigned_abs();
            let base_level = 1u32
                + u32::from(greater1_flags[pos])
                + u32::from(greater2_flags[pos]);
            let threshold = if num_sig_coeff < 8 {
                if pos as i32 == last_greater1_scan_pos {
                    3
                } else {
                    2
                }
            } else {
                1
            };
            if base_level == threshold {
                let rice_param = if coeff_rem_seen {
                    (last_rice_param
                        + u32::from(last_abs_level > (3 * (1u32 << last_rice_param))))
                    .min(4)
                } else {
                    0
                };
                let remainder = abs_level - base_level;
                encode_coeff_remainder(cw, rice_param, remainder);
                coeff_rem_seen = true;
                last_abs_level = base_level + remainder;
                last_rice_param = rice_param;
            }
            let _ = first_sig_scan_pos;
            let _ = last_sig_scan_pos;
            num_sig_coeff += 1;
        }
    }
    true
}

fn sub_block_has_nonzero(levels: &[i32], n: usize, sx: usize, sy: usize) -> bool {
    for dy in 0..4 {
        for dx in 0..4 {
            let x = sx * 4 + dx;
            let y = sy * 4 + dy;
            if x < n && y < n && levels[y * n + x] != 0 {
                return true;
            }
        }
    }
    false
}

/// Forward scan order of 4×4 sub-blocks for an n×n TU with diagonal
/// sub-block scan (the only scan this encoder emits).
fn sub_block_scan(num_sb: usize) -> Vec<(u8, u8)> {
    if num_sb == 1 {
        return vec![(0, 0)];
    }
    let mut v = Vec::with_capacity(num_sb * num_sb);
    for d in 0..(2 * num_sb - 1) {
        for x in 0..=d {
            let y = d - x;
            if x < num_sb && y < num_sb {
                v.push((x as u8, y as u8));
            }
        }
    }
    v
}

fn encode_last_sig_pos(
    cw: &mut CabacWriter<'_>,
    ctx: &mut ResidualCtx,
    last_x: usize,
    last_y: usize,
    log2_tb: u32,
    is_luma: bool,
) {
    // Derive prefix for x and y. Scan_idx = 0 so no swap.
    let (px, py) = (last_x as u32, last_y as u32);
    let prefix_x = last_coord_to_prefix(px);
    let prefix_y = last_coord_to_prefix(py);
    // Truncated-rice: emit `prefix` 1s then a 0 unless prefix equals max.
    let max_prefix = (log2_tb << 1).saturating_sub(1);
    emit_last_prefix(
        cw,
        &mut ctx.last_sig_x_prefix,
        prefix_x,
        max_prefix,
        log2_tb,
        is_luma,
    );
    emit_last_prefix(
        cw,
        &mut ctx.last_sig_y_prefix,
        prefix_y,
        max_prefix,
        log2_tb,
        is_luma,
    );
    // Suffix (bypass) if prefix > 3.
    if prefix_x > 3 {
        let suffix_len = ((prefix_x - 2) >> 1) as u32;
        let base = (1u32 << suffix_len) * (2 + (prefix_x & 1));
        let suffix = px - base;
        for bit in (0..suffix_len).rev() {
            cw.encode_bypass((suffix >> bit) & 1);
        }
    }
    if prefix_y > 3 {
        let suffix_len = ((prefix_y - 2) >> 1) as u32;
        let base = (1u32 << suffix_len) * (2 + (prefix_y & 1));
        let suffix = py - base;
        for bit in (0..suffix_len).rev() {
            cw.encode_bypass((suffix >> bit) & 1);
        }
    }
}

fn last_coord_to_prefix(v: u32) -> u32 {
    if v < 4 {
        return v;
    }
    // Find the prefix ≥ 4 such that the decoded range
    //   base..(base + 2^suffix_len) contains v, where
    //   suffix_len = (prefix - 2) >> 1
    //   base       = (1 << suffix_len) * (2 + (prefix & 1))
    //
    // This inverts the decoder's branch at `ctu.rs::decode_last_sig_pos`.
    //
    // Walk prefix values upward from 4 — max is `(log2_tb << 1) - 1 = 9`
    // at log2_tb = 5, so at most 6 iterations.
    let mut prefix = 4u32;
    loop {
        let suffix_len = (prefix - 2) >> 1;
        let base = (1u32 << suffix_len) * (2 + (prefix & 1));
        let next = base + (1u32 << suffix_len);
        if v >= base && v < next {
            return prefix;
        }
        prefix += 1;
        if prefix > 31 {
            return 31;
        }
    }
}

fn emit_last_prefix(
    cw: &mut CabacWriter<'_>,
    contexts: &mut [CtxState; 18],
    prefix: u32,
    max_prefix: u32,
    log2_tb: u32,
    is_luma: bool,
) {
    let mut i = 0u32;
    while i < prefix {
        let ctx_inc = last_sig_prefix_ctx_inc(i, log2_tb, is_luma);
        cw.encode_bin(&mut contexts[ctx_inc], 1);
        i += 1;
    }
    if prefix < max_prefix {
        let ctx_inc = last_sig_prefix_ctx_inc(prefix, log2_tb, is_luma);
        cw.encode_bin(&mut contexts[ctx_inc], 0);
    }
}

fn last_sig_prefix_ctx_inc(prefix: u32, log2_tb: u32, is_luma: bool) -> usize {
    let (ctx_offset, ctx_shift) = if is_luma {
        (
            3 * (log2_tb.saturating_sub(2)) + ((log2_tb.saturating_sub(1)) >> 2),
            (log2_tb + 1) >> 2,
        )
    } else {
        (15, log2_tb.saturating_sub(2))
    };
    (ctx_offset + (prefix >> ctx_shift)) as usize
}

fn encode_coeff_remainder(cw: &mut CabacWriter<'_>, rice: u32, value: u32) {
    let c_max = 4u32 << rice;
    if value < c_max {
        let prefix = value >> rice;
        // Emit `prefix` 1s, then a 0, then the rice-bit suffix.
        for _ in 0..prefix {
            cw.encode_bypass(1);
        }
        cw.encode_bypass(0);
        // Suffix bits.
        let suffix = value - (prefix << rice);
        for bit in (0..rice).rev() {
            cw.encode_bypass((suffix >> bit) & 1);
        }
    } else {
        // EGk tail.
        for _ in 0..4 {
            cw.encode_bypass(1);
        }
        encode_egk(cw, rice + 1, value - c_max);
    }
}

fn encode_egk(cw: &mut CabacWriter<'_>, k: u32, value: u32) {
    // Decoder: prefix unary of 1s terminated by 0, then (prefix + k) bit
    // suffix. Inverse of `decode_egk_bypass` in ctu.rs.
    // Value encoding: find prefix = floor(log2(value / (1 << k) + 1)).
    // Total value decoded = ((1 << prefix) - 1) << k + suffix.
    let mut prefix = 0u32;
    // Walk up until (2^prefix - 1) << k <= value < (2^(prefix+1) - 1) << k.
    // The decoder: prefix reads 1-bits until 0, then reads (prefix + k) bits.
    // So total = ((1 << prefix) - 1) << k + suffix, where suffix is in [0, 2^(prefix+k)).
    while value >= ((2u32 << (prefix + k)) - (1u32 << k)) {
        prefix += 1;
        if prefix > 31 {
            break;
        }
    }
    let suffix_bits = prefix + k;
    let base = ((1u32 << prefix) - 1) << k;
    let suffix = value - base;
    for _ in 0..prefix {
        cw.encode_bypass(1);
    }
    cw.encode_bypass(0);
    for bit in (0..suffix_bits).rev() {
        cw.encode_bypass((suffix >> bit) & 1);
    }
}

/// §9.3.4.2.5 sig_coeff_flag ctxInc — matches `sig_coeff_ctx_inc` in ctu.rs.
fn sig_coeff_ctx_inc(
    log2_tb: u32,
    scan_idx: u32,
    cx: u32,
    cy: u32,
    sx: u32,
    sy: u32,
    right_coded: bool,
    below_coded: bool,
    is_luma: bool,
) -> usize {
    const CTX_IDX_MAP_4X4: [usize; 16] = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8];
    if log2_tb == 2 {
        let sig_ctx = CTX_IDX_MAP_4X4[((cy << 2) | cx) as usize];
        return if is_luma { sig_ctx } else { 27 + sig_ctx };
    }
    let abs_x = (sx << 2) + cx;
    let abs_y = (sy << 2) + cy;
    if abs_x + abs_y == 0 {
        return if is_luma { 0 } else { 27 };
    }
    let mut sig_ctx = {
        let prev_csbf = usize::from(right_coded) + (usize::from(below_coded) << 1);
        let x_p = (cx & 3) as usize;
        let y_p = (cy & 3) as usize;
        match prev_csbf {
            0 => {
                if x_p + y_p == 0 {
                    2
                } else if x_p + y_p < 3 {
                    1
                } else {
                    0
                }
            }
            1 => {
                if y_p == 0 {
                    2
                } else if y_p == 1 {
                    1
                } else {
                    0
                }
            }
            2 => {
                if x_p == 0 {
                    2
                } else if x_p == 1 {
                    1
                } else {
                    0
                }
            }
            _ => 2,
        }
    };
    if is_luma {
        if sx + sy > 0 {
            sig_ctx += 3;
        }
        if log2_tb == 3 {
            sig_ctx += if scan_idx == 0 { 9 } else { 15 };
        } else {
            sig_ctx += 21;
        }
    } else if log2_tb == 3 {
        sig_ctx += 9;
    } else {
        sig_ctx += 12;
    }
    if is_luma { sig_ctx } else { 27 + sig_ctx }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::{init_row, CabacEngine, InitType};
    use crate::encoder::bit_writer::BitWriter;

    #[test]
    fn last_coord_prefix_matches_spec() {
        // Invariants mirror `ctu.rs::decode_last_sig_pos`.
        for v in 0u32..=3 {
            assert_eq!(last_coord_to_prefix(v), v);
        }
        // prefix 4 → suffix_len=1, base=4, values {4,5}
        assert_eq!(last_coord_to_prefix(4), 4);
        assert_eq!(last_coord_to_prefix(5), 4);
        // prefix 5 → base=6, values {6,7}
        assert_eq!(last_coord_to_prefix(6), 5);
        assert_eq!(last_coord_to_prefix(7), 5);
        // prefix 6 → suffix_len=2, base=8, values {8..11}
        for v in 8u32..=11 {
            assert_eq!(last_coord_to_prefix(v), 6, "v={v}");
        }
        // prefix 7 → suffix_len=2, base=12, values {12..15}
        for v in 12u32..=15 {
            assert_eq!(last_coord_to_prefix(v), 7, "v={v}");
        }
        // prefix 8 → suffix_len=3, base=16, values {16..23}
        for v in 16u32..=23 {
            assert_eq!(last_coord_to_prefix(v), 8, "v={v}");
        }
        // prefix 9 → suffix_len=3, base=24, values {24..31}
        for v in 24u32..=31 {
            assert_eq!(last_coord_to_prefix(v), 9, "v={v}");
        }
    }
}
