//! HEVC intra prediction (§8.4.4.2).
//!
//! Covers reference sample derivation and substitution (§8.4.4.2.2 /
//! §8.4.4.2.3), the optional MDIS low-pass reference filter
//! (§8.4.4.2.3), and the 35 prediction modes:
//!
//! * mode 0 — **PLANAR** (§8.4.4.2.5)
//! * mode 1 — **DC** (§8.4.4.2.6)
//! * modes 2..=34 — **33 angular** predictions (§8.4.4.2.6)
//!
//! Operates on a square luma or chroma block of size `n×n` where
//! `n ∈ {4, 8, 16, 32}` (luma) or `{4, 8, 16, 16}` (chroma in 4:2:0 —
//! chroma TB caps at 16). Samples are `u16` so the same code path
//! covers 8-bit (Main) and 10-bit (Main 10) content. The `bit_depth`
//! parameter supplies the output sample range `[0, (1<<bit_depth) - 1]`
//! for the Clip1 operator (§A.3 / eq. 8-242).

/// Clip to the sample range for `bit_depth`.
#[inline]
fn clip_bd(v: i32, bit_depth: u32) -> u16 {
    let max = ((1i32 << bit_depth) - 1).max(0);
    v.clamp(0, max) as u16
}

/// §8.4.4.2.6 Table 8-4: mapping from intra prediction mode (2..=34) to
/// `intraPredModeC` compatible angle parameters.
///
/// For angular mode `m` (2..=34), `intra_pred_angle[m]` gives the
/// displacement (in 1/32 sample units) per reference-sample row / column.
#[rustfmt::skip]
pub const INTRA_PRED_ANGLE: [i32; 35] = [
    0, 0,
    32, 26, 21, 17, 13,  9,  5,  2,
     0, -2, -5, -9, -13, -17, -21, -26,
    -32, -26, -21, -17, -13, -9, -5, -2,
     0,  2,  5,  9, 13, 17, 21, 26,
    32,
];

/// `invAngle` for modes 11..=25 — used to project the left-side reference
/// samples into the top reference row (§8.4.4.2.6).
#[rustfmt::skip]
pub const INV_ANGLE: [i32; 35] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -4096, -1638, -910, -630, -482, -390, -315,
    -256,
    -315, -390, -482, -630, -910, -1638, -4096,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Assemble the reference sample vector used by intra prediction.
///
/// Layout (§8.4.4.2.2): returns a 1-D vector of length `4*n + 1` whose
/// indices correspond to:
///
/// ```text
///   idx 0            -> top-left corner sample
///   idx 1..=2n       -> top row, left to right: p[0,-1] .. p[2n-1,-1]
///   idx 2n+1..=4n    -> left column, top to bottom: p[-1,0] .. p[-1,2n-1]
/// ```
///
/// `available[i]` tells whether reference sample `i` is available (i.e.,
/// inside the already-decoded region of the picture). If none are
/// available, the entire vector is filled with the neutral value
/// `1 << (bit_depth - 1)`.
///
/// When at least one sample is available the spec's substitution rule runs:
/// scan from `idx 4n` downward and upward, replacing unavailable samples
/// with the closest available neighbour.
pub fn build_ref_samples(
    samples: &[u16],
    available: &[bool],
    n: usize,
    bit_depth: u32,
) -> Vec<u16> {
    let len = 4 * n + 1;
    debug_assert_eq!(samples.len(), len);
    debug_assert_eq!(available.len(), len);

    // Storage order of `samples` / `available` is:
    //   [0]         top-left corner p[-1, -1]
    //   [1..=2n]    top row p[0..2n-1, -1]
    //   [2n+1..=4n] left column p[-1, 0..2n-1]
    //
    // HEVC §8.4.4.2.2 substitutes in a different order:
    //   1. left column BOTTOM-UP (p[-1, 2n-1] → p[-1, 0])
    //   2. top-left corner (p[-1, -1])
    //   3. top row LEFT-TO-RIGHT (p[0, -1] → p[2n-1, -1])
    // Samples before the first available slot get the first available value;
    // samples at or after that slot carry forward the most recent available
    // value. The translation to our storage order is non-trivial because
    // the left column is walked in reverse.

    let any = available.iter().any(|&a| a);
    let neutral = (1u16 << (bit_depth - 1)) as u16;
    let mut out = vec![neutral; len];
    if !any {
        return out;
    }

    // Helper: linear index for a given "spec position" (0..=4n).
    //   spec 0..=2n-1   → left col indices 4n, 4n-1, ..., 2n+1.
    //   spec 2n         → top-left = index 0.
    //   spec 2n+1..=4n  → top row indices 1, 2, ..., 2n.
    let spec_to_linear = |s: usize| -> usize {
        if s < 2 * n {
            // Left col bottom-up: spec s=0 → left[2n-1] → linear 4n.
            4 * n - s
        } else if s == 2 * n {
            0
        } else {
            // Top row L-R: spec s=2n+1 → top[0] → linear 1.
            s - 2 * n
        }
    };

    // Find first available in spec order.
    let mut first_avail_spec = 0;
    while first_avail_spec <= 4 * n && !available[spec_to_linear(first_avail_spec)] {
        first_avail_spec += 1;
    }
    let seed = samples[spec_to_linear(first_avail_spec)];

    let mut last = seed;
    for s in 0..=(4 * n) {
        let lin = spec_to_linear(s);
        if s < first_avail_spec {
            out[lin] = seed;
        } else if available[lin] {
            out[lin] = samples[lin];
            last = out[lin];
        } else {
            out[lin] = last;
        }
    }
    out
}

/// Apply the reference-sample smoothing filter of §8.4.4.2.3.
/// `filter_mode` selects between the 3-tap `[1 2 1]/4` filter (0) and the
/// strong-intra-smoothing bilinear path (1).
pub fn filter_ref_samples(refs: &mut [u16], n: usize, strong: bool, bit_depth: u32) {
    let len = 4 * n + 1;
    debug_assert_eq!(refs.len(), len);

    if strong {
        // Bilinear between the three anchors: p[-1,-1], p[2n-1,-1], p[-1,2n-1].
        let tl = refs[0] as i32;
        let top_right = refs[2 * n] as i32;
        let bot_left = refs[4 * n] as i32;
        let shift = n.ilog2() + 1;
        // Top row: i in 1..2n interpolates tl→top_right.
        let tn = n as i32;
        for (i, slot) in refs.iter_mut().enumerate().take(2 * n).skip(1) {
            let w = i as i32;
            let val = ((2 * tn - w) * tl + w * top_right + tn) >> shift;
            *slot = clip_bd(val, bit_depth);
        }
        refs[2 * n] = top_right as u16;
        // Left column: indices 2n+1 .. 4n.
        for i in 1..(2 * n) {
            let w = i as i32;
            let val = ((2 * tn - w) * tl + w * bot_left + tn) >> shift;
            refs[2 * n + i] = clip_bd(val, bit_depth);
        }
        refs[4 * n] = bot_left as u16;
    } else {
        // `[1 2 1]/4` 3-tap filter per §8.4.4.2.3. The top-row and
        // left-column segments of `refs` are geometrically disjoint, so
        // we apply the filter separately: the far endpoints of each
        // segment stay as-is, and the corner `refs[0]` is the shared
        // top-left that filters from the first top-row and first
        // left-column sample.
        //
        // refs layout: [0] top-left, [1..=2n] top row (left to right),
        // [2n+1..=4n] left column (top to bottom).
        let mut tmp = refs.to_vec();
        // Top-left corner (shared between top[-1] and left[-1]):
        // (left[0] + 2*refs[0] + top[0] + 2) >> 2.
        tmp[0] = ((refs[2 * n + 1] as u32 + 2 * refs[0] as u32 + refs[1] as u32 + 2) >> 2) as u16;
        // Top row: indices 1..=2n. Interior filter uses neighbour in-row;
        // left neighbour of refs[1] is refs[0] (corner). Endpoint refs[2n]
        // preserved (do not filter the 2n-1 sample, matching ffmpeg).
        for i in 1..(2 * n) {
            tmp[i] =
                ((refs[i - 1] as u32 + 2 * refs[i] as u32 + refs[i + 1] as u32 + 2) >> 2) as u16;
        }
        // tmp[2n] stays = refs[2n] (the farthest top sample).
        // Left column: indices 2n+1..=4n. Neighbour of refs[2n+1] above is
        // refs[0] (corner). Endpoint refs[4n] preserved.
        tmp[2 * n + 1] =
            ((refs[0] as u32 + 2 * refs[2 * n + 1] as u32 + refs[2 * n + 2] as u32 + 2) >> 2)
                as u16;
        for i in (2 * n + 2)..(4 * n) {
            tmp[i] =
                ((refs[i - 1] as u32 + 2 * refs[i] as u32 + refs[i + 1] as u32 + 2) >> 2) as u16;
        }
        // tmp[4n] stays = refs[4n].
        refs.copy_from_slice(&tmp);
    }
}

/// Decide whether the `[1 2 1]/4` filter should be applied to the
/// reference samples of an intra-luma block (§8.4.4.2.3 Table 8-3).
/// Returns `(apply_filter, use_strong)`.
pub fn filter_decision(
    log2_tb: u32,
    intra_pred_mode: u32,
    strong_intra_smoothing_enabled: bool,
    refs: &[u16],
    n: usize,
    bit_depth: u32,
) -> (bool, bool) {
    // §8.4.4.2.3: for sizes <= 4 (log2_tb == 2), no filtering.
    if log2_tb <= 2 {
        return (false, false);
    }
    // Determine `filterFlag` per Table 8-3.
    // Angular modes with minDistVerHor >= threshold get filtered;
    // mode 1 (DC) and mode 0 (planar), log2_tb > 2 — planar always filters
    // for sizes >= 8; DC never filters.
    //
    // The threshold is 7 for 8x8, 1 for 16x16, 0 for 32x32.
    if intra_pred_mode == 1 {
        return (false, false); // DC never
    }
    let threshold: i32 = match log2_tb {
        3 => 7,
        4 => 1,
        5 => 0,
        _ => 0,
    };
    let m = intra_pred_mode as i32;
    let dist = (m - 26).abs().min((m - 10).abs());
    let apply = if intra_pred_mode == 0 {
        // Planar filtering enabled iff log2_tb > 2 (i.e. size >= 8).
        true
    } else {
        // Angular: filter when minDistVerHor > threshold.
        dist > threshold
    };
    if !apply {
        return (false, false);
    }
    // Strong filter applies only at 32x32 (log2_tb == 5) with strong-intra-
    // smoothing enabled and when the corner deviations are small enough.
    // §8.4.4.2.3: threshold scales with bit depth (`1 << (BitDepthY - 5)`).
    let mut strong = false;
    if strong_intra_smoothing_enabled && log2_tb == 5 {
        let th = 1i32 << (bit_depth as i32 - 5);
        let tl = refs[0] as i32;
        let top_right = refs[2 * n] as i32;
        let bot_left = refs[4 * n] as i32;
        let bot_mid = refs[3 * n] as i32;
        let top_mid = refs[n] as i32;
        let top_ok = (tl + top_right - 2 * top_mid).abs() < th;
        let left_ok = (tl + bot_left - 2 * bot_mid).abs() < th;
        if top_ok && left_ok {
            strong = true;
        }
    }
    (true, strong)
}

/// Apply PLANAR prediction (mode 0) to fill an `n×n` block.
///
/// `refs` uses the layout documented on [`build_ref_samples`]. The output
/// buffer is written row-major; `dst_stride` is the number of samples per
/// row in `dst` (can be larger than `n` if writing into a bigger picture
/// plane).
pub fn predict_planar(refs: &[u16], n: usize, dst: &mut [u16], dst_stride: usize) {
    debug_assert_eq!(refs.len(), 4 * n + 1);
    let log2_n = n.ilog2() as i32;
    for y in 0..n {
        for x in 0..n {
            let top = refs[1 + x] as i32; // p[x, -1]
            let left = refs[2 * n + 1 + y] as i32; // p[-1, y]
                                                   // Spec §8.4.4.2.5:
                                                   //   predSamples[x][y] = ((nT - 1 - x) * p[-1, y] + (x+1)*p[nT, -1]
                                                   //                       + (nT - 1 - y) * p[x, -1] + (y+1)*p[-1, nT]
                                                   //                       + nT) >> (log2(nT)+1)
                                                   // p[nT,-1] is refs[1 + n] (top row slot at x=n), p[-1,nT]=refs[2n+1+n]
            let p_top_r = refs[1 + n] as i32; // p[nT, -1]
            let p_bot_l = refs[2 * n + 1 + n] as i32; // p[-1, nT]
            let v = ((n as i32 - 1 - x as i32) * left
                + (x as i32 + 1) * p_top_r
                + (n as i32 - 1 - y as i32) * top
                + (y as i32 + 1) * p_bot_l
                + n as i32)
                >> (log2_n + 1);
            // Planar output is already in range when refs are — no clip needed
            // strictly, but store as u16 for type compat.
            dst[y * dst_stride + x] = v as u16;
        }
    }
}

/// Apply DC prediction (mode 1) to fill an `n×n` block.
pub fn predict_dc(refs: &[u16], n: usize, dst: &mut [u16], dst_stride: usize, luma: bool) {
    let mut sum: u32 = 0;
    for x in 0..n {
        sum += refs[1 + x] as u32;
    }
    for y in 0..n {
        sum += refs[2 * n + 1 + y] as u32;
    }
    let dc = ((sum + n as u32) / (2 * n as u32)) as i32;
    // Fill interior.
    for y in 0..n {
        for x in 0..n {
            dst[y * dst_stride + x] = dc as u16;
        }
    }
    // §8.4.4.2.6 DC edge filtering — only for luma AND n < 32.
    if luma && n < 32 {
        let top0 = refs[1] as i32;
        let left0 = refs[2 * n + 1] as i32;
        dst[0] = ((top0 + left0 + 2 * dc + 2) >> 2) as u16;
        for x in 1..n {
            let top = refs[1 + x] as i32;
            dst[x] = ((top + 3 * dc + 2) >> 2) as u16;
        }
        for y in 1..n {
            let left = refs[2 * n + 1 + y] as i32;
            dst[y * dst_stride] = ((left + 3 * dc + 2) >> 2) as u16;
        }
    }
}

/// Apply an angular prediction (mode ∈ 2..=34) to fill an `n×n` block.
pub fn predict_angular(
    refs: &[u16],
    n: usize,
    dst: &mut [u16],
    dst_stride: usize,
    mode: u32,
    luma: bool,
    bit_depth: u32,
) {
    debug_assert!((2..=34).contains(&mode));
    let mode_is_vertical = mode >= 18;
    let intra_pred_angle = INTRA_PRED_ANGLE[mode as usize];
    let inv_angle = INV_ANGLE[mode as usize];

    // Build the 1-D reference array `ref1[]` per §8.4.4.2.6 eq 8-42..8-46.
    // `ref1` indices run from -n..=2n (inclusive), so length 3n+1 if
    // intra_pred_angle >= 0; otherwise negative indices up to -n are
    // populated by projecting left-side refs via inv_angle.
    //
    // Allocate `ref_extended[idx + n]` so that logical index -n maps to 0
    // and 2n maps to 3n.
    let mut ref_ext = vec![0u16; 3 * n + 1];

    if mode_is_vertical {
        // ref1[x] for x in 0..=2n comes from top-row refs: refs[1..=2n] and
        // ref_ext layout: negative-side indices 0..n (only populated for
        // angles < 0 via the inv_angle projection below), then
        // ref_ext[n..=n+2n] holds ref1[0..=2n] from the top row of refs.
        ref_ext[n..n + 2 * n + 1].copy_from_slice(&refs[..2 * n + 1]);
        if intra_pred_angle < 0 {
            // Extend negative side by projecting left column samples.
            let inv_angle_sum_step = inv_angle;
            let mut inv_angle_sum: i32 = 128;
            for i in 1..=n {
                inv_angle_sum += inv_angle_sum_step;
                let src_idx = (inv_angle_sum >> 8).clamp(-(n as i32), n as i32);
                let li = (-src_idx - 1).clamp(0, 4 * n as i32) as usize;
                let src = refs[2 * n + 1 + li];
                ref_ext[n - i] = src;
            }
        }
    } else {
        // Horizontal mode: ref1[0] = top-left = refs[0],
        // ref1[x] for x >= 1 = refs[2n + x] (left column), at x=1 we want
        // refs[2n+1] == p[-1, 0].
        ref_ext[n] = refs[0];
        for i in 1..=(2 * n) {
            ref_ext[n + i] = refs[2 * n + i];
        }
        if intra_pred_angle < 0 {
            let inv_angle_sum_step = inv_angle;
            let mut inv_angle_sum: i32 = 128;
            for i in 1..=n {
                inv_angle_sum += inv_angle_sum_step;
                let src_idx = (inv_angle_sum >> 8).clamp(-(n as i32), n as i32);
                let ti = (-src_idx - 1).clamp(0, 2 * n as i32) as usize;
                let src = refs[1 + ti];
                ref_ext[n - i] = src;
            }
        }
    }

    // Now predict.
    //
    // For the vertical direction (mode >= 18), we scan (x,y) normally and
    // look up samples in `ref_ext` (which holds the top-row refs extended
    // from the left column for negative angles). For the horizontal
    // direction (mode < 18), the coordinate roles swap: the spec
    // essentially mirrors the block around the diagonal and uses the left-
    // column refs instead.
    for y in 0..n {
        for x in 0..n {
            // Per §8.4.4.2.6: for vertical modes predSamples[x][y] references
            //   ref1[x + ((y+1)*intra_pred_angle >> 5)]; for horizontal modes
            //   it references ref1[y + ((x+1)*intra_pred_angle >> 5)].
            let (ax, ay) = if mode_is_vertical { (x, y) } else { (y, x) };
            let idx_i = ((ay as i32 + 1) * intra_pred_angle) >> 5;
            let frac = ((ay as i32 + 1) * intra_pred_angle) & 31;
            let r_idx = (ax as i32 + idx_i + 1 + n as i32) as usize;
            let r_idx = r_idx.min(3 * n);
            let r_a = ref_ext[r_idx] as i32;
            let r_b = ref_ext[r_idx.saturating_add(1).min(3 * n)] as i32;
            let v = ((32 - frac) * r_a + frac * r_b + 16) >> 5;
            let mut pred = clip_bd(v, bit_depth);
            // Post-prediction edge filter for horizontal (mode 10) /
            // vertical (mode 26) — §8.4.4.2.6.
            if luma && n <= 16 && (mode == 10 || mode == 26) {
                if mode == 26 && x == 0 {
                    let top = refs[1] as i32;
                    let tl = refs[0] as i32;
                    let left = refs[2 * n + 1 + y] as i32;
                    let v2 = top + ((left - tl) >> 1);
                    pred = clip_bd(v2, bit_depth);
                }
                if mode == 10 && y == 0 {
                    let left = refs[2 * n + 1] as i32;
                    let tl = refs[0] as i32;
                    let top = refs[1 + x] as i32;
                    let v2 = left + ((top - tl) >> 1);
                    pred = clip_bd(v2, bit_depth);
                }
            }
            dst[y * dst_stride + x] = pred;
        }
    }
}

/// Top-level entry: dispatch on `intra_pred_mode` and fill an `n×n` block.
pub fn predict(
    refs: &[u16],
    n: usize,
    dst: &mut [u16],
    dst_stride: usize,
    intra_pred_mode: u32,
    luma: bool,
    bit_depth: u32,
) {
    match intra_pred_mode {
        0 => predict_planar(refs, n, dst, dst_stride),
        1 => predict_dc(refs, n, dst, dst_stride, luma),
        _ => predict_angular(refs, n, dst, dst_stride, intra_pred_mode, luma, bit_depth),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planar_flat() {
        // Uniform refs → uniform prediction.
        let n = 4;
        let refs = vec![128u16; 4 * n + 1];
        let mut dst = vec![0u16; n * n];
        predict_planar(&refs, n, &mut dst, n);
        assert!(dst.iter().all(|&v| v == 128));
    }

    #[test]
    fn dc_mean_flat_interior() {
        let n = 8;
        let refs = vec![100u16; 4 * n + 1];
        let mut dst = vec![0u16; n * n];
        predict_dc(&refs, n, &mut dst, n, false);
        assert!(dst.iter().all(|&v| v == 100));
    }

    #[test]
    fn angular_horizontal_replicates() {
        // Mode 10 (pure horizontal) copies the left column across all rows.
        let n = 4;
        let mut refs = vec![0u16; 4 * n + 1];
        for y in 0..n {
            refs[2 * n + 1 + y] = (10 + y as u16) * 10;
        }
        let mut dst = vec![0u16; n * n];
        predict_angular(&refs, n, &mut dst, n, 10, false, 8);
        for y in 0..n {
            let expected = (10 + y as u16) * 10;
            for x in 0..n {
                assert_eq!(dst[y * n + x], expected, "row {y} col {x}");
            }
        }
    }

    #[test]
    fn angular_vertical_replicates() {
        // Mode 26 (pure vertical) copies the top row down all columns.
        let n = 4;
        let mut refs = vec![0u16; 4 * n + 1];
        for x in 0..n {
            refs[1 + x] = (10 + x as u16) * 10;
        }
        let mut dst = vec![0u16; n * n];
        predict_angular(&refs, n, &mut dst, n, 26, false, 8);
        for y in 0..n {
            for x in 0..n {
                assert_eq!(dst[y * n + x], (10 + x as u16) * 10, "row {y} col {x}");
            }
        }
    }

    #[test]
    fn build_ref_samples_fills_uniform_when_unavailable() {
        let n = 4;
        let samples = vec![0u16; 4 * n + 1];
        let avail = vec![false; 4 * n + 1];
        let out = build_ref_samples(&samples, &avail, n, 8);
        assert!(out.iter().all(|&v| v == 128));
    }

    #[test]
    fn build_ref_samples_fills_uniform_10bit() {
        let n = 4;
        let samples = vec![0u16; 4 * n + 1];
        let avail = vec![false; 4 * n + 1];
        let out = build_ref_samples(&samples, &avail, n, 10);
        assert!(out.iter().all(|&v| v == 512));
    }
}
