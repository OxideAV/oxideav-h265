//! HEVC inverse transforms and dequantisation (§8.6).
//!
//! Implements:
//!
//! * 4×4 **DST-VII** (Annex §8.6.4.2 eq 8-284..8-287), used for 4×4 intra
//!   luma residual blocks.
//! * 4/8/16/32 **DCT-II** (§8.6.4.2 eq 8-279..8-283) — a single integer
//!   butterfly shared across all four sizes.
//! * Inverse quantisation for 8-bit luma / chroma without scaling lists
//!   (§8.6.3). Flat scaling is implied per §7.4.4 when
//!   `scaling_list_enabled_flag == 0`.
//!
//! All integer maths matches the reference implementation bit-for-bit; the
//! module has no `unwrap` or `unsafe`. Inputs are `i32` coefficients, output
//! is written into a 2-D residual buffer of `i32`.

/// `levelScale[]` from §8.6.3 (rem6 lookup).
pub const LEVEL_SCALE: [i32; 6] = [40, 45, 51, 57, 64, 72];

/// 8-point DCT-II basis rows (§8.6.4.2 Table 8-7) for the 32-point
/// transform. Only the rows used for sizes 4/8/16/32 are included.
#[rustfmt::skip]
pub const TRANSFORM_MATRIX: [[i16; 32]; 32] = [
    [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    [90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13, 4, -4, -13, -22, -31, -38, -46, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90],
    [90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90, -90, -87, -80, -70, -57, -43, -25, -9, 9, 25, 43, 57, 70, 80, 87, 90],
    [90, 82, 67, 46, 22, -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13, 13, 38, 61, 78, 88, 90, 85, 73, 54, 31, 4, -22, -46, -67, -82, -90],
    [89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89, 89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89],
    [88, 67, 31, -13, -54, -82, -90, -78, -46, -4, 38, 73, 90, 85, 61, 22, -22, -61, -85, -90, -73, -38, 4, 46, 78, 90, 82, 54, 13, -31, -67, -88],
    [87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87, -87, -57, -9, 43, 80, 90, 70, 25, -25, -70, -90, -80, -43, 9, 57, 87],
    [85, 46, -13, -67, -90, -73, -22, 38, 82, 88, 54, -4, -61, -90, -78, -31, 31, 78, 90, 61, 4, -54, -88, -82, -38, 22, 73, 90, 67, 13, -46, -85],
    [83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83],
    [82, 22, -54, -90, -61, 13, 78, 85, 31, -46, -90, -67, 4, 73, 88, 38, -38, -88, -73, -4, 67, 90, 46, -31, -85, -78, -13, 61, 90, 54, -22, -82],
    [80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80, -80, -9, 70, 87, 25, -57, -90, -43, 43, 90, 57, -25, -87, -70, 9, 80],
    [78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 31, 90, 54, -38, -90, -46, 46, 90, 38, -54, -90, -31, 61, 88, 22, -67, -85, -13, 73, 82, 4, -78],
    [75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75, 75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75],
    [73, -31, -90, -22, 78, 67, -38, -90, -13, 82, 61, -46, -88, -4, 85, 54, -54, -85, 4, 88, 46, -61, -82, 13, 90, 38, -67, -78, 22, 90, 31, -73],
    [70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70, -70, 43, 87, -9, -90, -25, 80, 57, -57, -80, 25, 90, 9, -87, -43, 70],
    [67, -54, -78, 38, 85, -22, -90, 4, 90, 13, -88, -31, 82, 46, -73, -61, 61, 73, -46, -82, 31, 88, -13, -90, -4, 90, 22, -85, -38, 78, 54, -67],
    [64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64],
    [61, -73, -46, 82, 31, -88, -13, 90, -4, -90, 22, 85, -38, -78, 54, 67, -67, -54, 78, 38, -85, -22, 90, 4, -90, 13, 88, -31, -82, 46, 73, -61],
    [57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57, -57, 80, 25, -90, 9, 87, -43, -70, 70, 43, -87, -9, 90, -25, -80, 57],
    [54, -85, -4, 88, -46, -61, 82, 13, -90, 38, 67, -78, -22, 90, -31, -73, 73, 31, -90, 22, 78, -67, -38, 90, -13, -82, 61, 46, -88, 4, 85, -54],
    [50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50, 50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50],
    [46, -90, 38, 54, -90, 31, 61, -88, 22, 67, -85, 13, 73, -82, 4, 78, -78, -4, 82, -73, -13, 85, -67, -22, 88, -61, -31, 90, -54, -38, 90, -46],
    [43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43, -43, 90, -57, -25, 87, -70, -9, 80, -80, 9, 70, -87, 25, 57, -90, 43],
    [38, -88, 73, -4, -67, 90, -46, -31, 85, -78, 13, 61, -90, 54, 22, -82, 82, -22, -54, 90, -61, -13, 78, -85, 31, 46, -90, 67, 4, -73, 88, -38],
    [36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36],
    [31, -78, 90, -61, 4, 54, -88, 82, -38, -22, 73, -90, 67, -13, -46, 85, -85, 46, 13, -67, 90, -73, 22, 38, -82, 88, -54, -4, 61, -90, 78, -31],
    [25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25, -25, 70, -90, 80, -43, -9, 57, -87, 87, -57, 9, 43, -80, 90, -70, 25],
    [22, -61, 85, -90, 73, -38, -4, 46, -78, 90, -82, 54, -13, -31, 67, -88, 88, -67, 31, 13, -54, 82, -90, 78, -46, 4, 38, -73, 90, -85, 61, -22],
    [18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18, 18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18],
    [13, -38, 61, -78, 88, -90, 85, -73, 54, -31, 4, 22, -46, 67, -82, 90, -90, 82, -67, 46, -22, -4, 31, -54, 73, -85, 90, -88, 78, -61, 38, -13],
    [9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9, -9, 25, -43, 57, -70, 80, -87, 90, -90, 87, -80, 70, -57, 43, -25, 9],
    [4, -13, 22, -31, 38, -46, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90, 90, -90, 88, -85, 82, -78, 73, -67, 61, -54, 46, -38, 31, -22, 13, -4],
];

/// 4×4 DST-VII basis (§8.6.4.2 Table 8-8).
#[rustfmt::skip]
pub const DST_MATRIX: [[i16; 4]; 4] = [
    [29, 55, 74, 84],
    [74, 74,  0, -74],
    [84, -29, -74, 55],
    [55, -84, 74, -29],
];

/// Apply an `N×N` inverse integer transform to a row/column.
///
/// `src[0..n]` → `dst[0..n]`. `mat_stride` is 32 (the matrix is stored
/// as one 32-column row per basis vector).
fn apply_transform_1d(src: &[i32], dst: &mut [i32], n: usize, shift: i32, is_dst: bool) {
    let round = 1i32 << (shift - 1);
    // HEVC stores the 32×32 DCT matrix row-by-row as basis × sample
    // (row = basis index, column = sample position). For an N-point
    // inverse transform, the k-th basis function is row k·(32/N) and we
    // read its value at sample position i directly from column i — no
    // stride on the sample side — matching ffmpeg's TR_N partial butterfly.
    let basis_step = 1usize << (5 - n.ilog2());
    for i in 0..n {
        let mut sum: i32 = 0;
        for j in 0..n {
            let coef = if is_dst {
                DST_MATRIX[j][i] as i32
            } else {
                TRANSFORM_MATRIX[j * basis_step][i] as i32
            };
            sum += coef * src[j];
        }
        dst[i] = ((sum + round) >> shift).clamp(-32768, 32767);
    }
}

/// Run the 2-D HEVC inverse transform over an `n×n` coefficient block.
///
/// * `coeffs` is the `n*n` dequantised coefficient array in row-major order.
/// * `out` receives the reconstructed residual samples.
/// * `log2_tb` = log2(n), ∈ {2, 3, 4, 5}.
/// * `is_dst`: true for 4×4 intra luma (DST-VII), false for DCT-II.
pub fn inverse_transform_2d(
    coeffs: &[i32],
    out: &mut [i32],
    log2_tb: u32,
    is_dst: bool,
    bit_depth: u32,
) {
    let n = 1usize << log2_tb;
    debug_assert_eq!(coeffs.len(), n * n);
    debug_assert_eq!(out.len(), n * n);

    // Shift1: 7 (applied after the first 1-D pass).
    // Shift2: 20 - bit_depth (second pass).
    let shift1: i32 = 7;
    let shift2: i32 = 20 - bit_depth as i32;

    // First pass: transform each column (treating input as n columns, the
    // DCT inverse per spec acts first on columns, then rows).
    let mut tmp = vec![0i32; n * n];
    let mut col = vec![0i32; n];
    let mut col_out = vec![0i32; n];
    for x in 0..n {
        for y in 0..n {
            col[y] = coeffs[y * n + x];
        }
        apply_transform_1d(&col, &mut col_out, n, shift1, is_dst);
        for y in 0..n {
            tmp[y * n + x] = col_out[y];
        }
    }
    // Second pass: transform each row.
    let mut row = vec![0i32; n];
    let mut row_out = vec![0i32; n];
    for y in 0..n {
        for x in 0..n {
            row[x] = tmp[y * n + x];
        }
        apply_transform_1d(&row, &mut row_out, n, shift2, is_dst);
        for x in 0..n {
            out[y * n + x] = row_out[x];
        }
    }
}

/// Dequantise an n×n coefficient block without scaling lists.
///
/// * `coeffs_in` — signed level values from the bitstream.
/// * `coeffs_out` — dequantised `i32` coefficients (caller can feed these
///   straight into [`inverse_transform_2d`]).
/// * `qp` — the per-TB QP value (luma or chroma-derived).
/// * `log2_tb` — block size log2.
/// * `bit_depth` — component bit depth (8 for 8-bit).
pub fn dequantize_flat(
    coeffs_in: &[i32],
    coeffs_out: &mut [i32],
    qp: i32,
    log2_tb: u32,
    bit_depth: u32,
) {
    let n = 1usize << log2_tb;
    debug_assert_eq!(coeffs_in.len(), n * n);
    debug_assert_eq!(coeffs_out.len(), n * n);

    // qp' = qp
    let q_div = qp / 6;
    let q_rem = qp.rem_euclid(6) as usize;
    let scale = LEVEL_SCALE[q_rem];

    // §8.6.3 eq. (8-309): with scaling lists disabled, m[x][y] = 16 and
    // bdShift = bitDepth + log2(nTbS) + 10 - log2TransformRange.
    let bd_shift = bit_depth as i32 + log2_tb as i32 - 5;

    for i in 0..n * n {
        let lvl = coeffs_in[i];
        let mut v = lvl * 16 * scale;
        if q_div > 0 {
            v <<= q_div;
        }
        let round = 1i32 << (bd_shift - 1);
        v = (v + round) >> bd_shift;
        coeffs_out[i] = v.clamp(-32768, 32767);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_zero_transforms_to_zero() {
        let n = 4;
        let coeffs = vec![0i32; n * n];
        let mut out = vec![99i32; n * n];
        inverse_transform_2d(&coeffs, &mut out, 2, false, 8);
        assert!(out.iter().all(|&v| v == 0));
        inverse_transform_2d(&coeffs, &mut out, 2, true, 8);
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn dequant_scale_positive() {
        let coeffs = vec![1i32; 16];
        let mut out = vec![0i32; 16];
        dequantize_flat(&coeffs, &mut out, 22, 2, 8);
        // Each value should be > 0 (positive level, positive scale).
        assert!(out.iter().all(|&v| v != 0));
    }
}
