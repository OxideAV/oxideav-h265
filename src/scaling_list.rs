//! HEVC scaling list data (§7.3.4, §7.4.5) and derived scaling factors
//! (§7.4.5 eq. 7-44 .. 7-51).
//!
//! Provides:
//!
//! * [`ScalingListData`] — parsed `scaling_list_data()` payload, holding
//!   the four-way `ScalingList[sizeId][matrixId][i]` (sizeId 0..3,
//!   matrixId 0..5 for sizeId<3 and 0/3 for sizeId==3) plus the 16×16 /
//!   32×32 DC coefficients (`scaling_list_dc_coef_minus8`).
//! * [`ScalingListData::default_flat`] — ScalingList filled with 16s.
//!   Used when the SPS/PPS does not signal explicit lists but the spec
//!   still needs a non-null matrix for default initialisation.
//! * [`ScalingListData::spec_defaults`] — loads Table 7-5 / 7-6 default
//!   values. Used when `scaling_list_pred_matrix_id_delta == 0`.
//! * [`ScalingListData::scaling_factor`] — returns `m[x][y]` for a given
//!   TU by expanding the 8×8 coefficients of sizeId 2/3 per eq. 7-46 /
//!   7-48 and overriding the DC per eq. 7-47 / 7-49.
//!
//! The parser matches `scaling_list_data()` byte-for-byte per §7.3.4.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::scan::DIAG_SCAN_4X4;

/// Number of matrices per size: {6, 6, 6, 6} (sizeId 0..3). For sizeId==3
/// only matrixId 0 and 3 are active but we allocate the full 6 slots for
/// uniformity and copy from 0/3 for chroma to keep indexing simple.
const NUM_MATRICES: [usize; 4] = [6, 6, 6, 6];

/// `coefNum = Min(64, 1 << (4 + (sizeId << 1)))` — 16, 64, 64, 64.
const COEF_NUM: [usize; 4] = [16, 64, 64, 64];

/// Table 7-6 default values for sizeId >= 1 intra (matrixId 0..2). 64
/// entries in up-right diagonal scan order.
pub const DEFAULT_8X8_INTRA: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 16, 17, 16, 17, 18, 17, 18, 18, 17, 18, 21, 19, 20,
    21, 20, 19, 21, 24, 22, 22, 24, 24, 22, 22, 24, 25, 25, 27, 30, 27, 25, 25, 29, 31, 35, 35, 31,
    29, 36, 41, 44, 41, 36, 47, 54, 54, 47, 65, 70, 65, 88, 88, 115,
];

/// Table 7-6 default values for sizeId >= 1 inter (matrixId 3..5). 64
/// entries in up-right diagonal scan order.
pub const DEFAULT_8X8_INTER: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 20, 20, 20,
    20, 20, 20, 20, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 28, 28, 28, 28, 28,
    28, 33, 33, 33, 33, 33, 41, 41, 41, 41, 54, 54, 54, 71, 71, 91,
];

/// All four ScalingList arrays plus the 16×16 and 32×32 DC coefficients.
#[derive(Clone, Debug)]
pub struct ScalingListData {
    /// `ScalingList[sizeId][matrixId][i]` — flattened to
    /// `lists[sizeId][matrixId]` with `coefNum(sizeId)` entries.
    pub lists: [[Vec<u8>; 6]; 4],
    /// `scaling_list_dc_coef_minus8[0][matrixId] + 8` for sizeId == 2.
    pub dc_16x16: [u8; 6],
    /// `scaling_list_dc_coef_minus8[1][matrixId] + 8` for sizeId == 3.
    pub dc_32x32: [u8; 6],
}

impl Default for ScalingListData {
    fn default() -> Self {
        Self::default_flat()
    }
}

impl ScalingListData {
    /// Every entry 16 (equivalent to "scaling lists disabled" — matches the
    /// §8.6.3 branch that sets `m[x][y] = 16`).
    pub fn default_flat() -> Self {
        let mk_list = |n: usize| vec![16u8; n];
        let lists_one_size = |coef_num: usize| -> [Vec<u8>; 6] {
            [
                mk_list(coef_num),
                mk_list(coef_num),
                mk_list(coef_num),
                mk_list(coef_num),
                mk_list(coef_num),
                mk_list(coef_num),
            ]
        };
        Self {
            lists: [
                lists_one_size(COEF_NUM[0]),
                lists_one_size(COEF_NUM[1]),
                lists_one_size(COEF_NUM[2]),
                lists_one_size(COEF_NUM[3]),
            ],
            dc_16x16: [16; 6],
            dc_32x32: [16; 6],
        }
    }

    /// Populate from the spec's default ScalingList tables (Table 7-5 /
    /// 7-6). Used when the bitstream signals
    /// `scaling_list_pred_mode_flag == 0` with
    /// `scaling_list_pred_matrix_id_delta == 0`.
    pub fn spec_defaults() -> Self {
        let mut d = Self::default_flat();
        // sizeId 0: all 16 (Table 7-5) — already set.
        // sizeId 1..3: Table 7-6 split by matrixId.
        for size_id in 1..4 {
            for matrix_id in 0..6 {
                let src = if matrix_id < 3 {
                    &DEFAULT_8X8_INTRA
                } else {
                    &DEFAULT_8X8_INTER
                };
                d.lists[size_id][matrix_id] = src.to_vec();
            }
        }
        // DC defaults (when derived from defaults) = 16 per §7.4.5
        // "inferred to be equal to 8" for `scaling_list_dc_coef_minus8`,
        // which means 8+8 = 16.
        d
    }

    /// Expand a ScalingFactor matrix into a row-major `n×n` buffer, ready
    /// for [`crate::transform::dequantize_with_matrix`]. `n = 1 << (2 +
    /// size_id)`. Matches eq. 7-44 .. 7-51 with the DC override at (0, 0)
    /// for sizeId ∈ {2, 3}.
    pub fn expand_matrix(&self, size_id: usize, matrix_id: usize) -> Vec<u8> {
        let n = 1usize << (2 + size_id);
        let mut out = vec![16u8; n * n];
        for y in 0..n {
            for x in 0..n {
                out[y * n + x] = self.scaling_factor(size_id, matrix_id, x, y);
            }
        }
        out
    }

    /// Return the scaling factor `m[x][y]` for a TU (§7.4.5 eq. 7-44 ..
    /// 7-51). Indexes:
    ///
    /// * `size_id` — 0 (4×4), 1 (8×8), 2 (16×16), 3 (32×32).
    /// * `matrix_id` — 0..5 (sizeId<3) or 0/3 (sizeId=3).
    /// * `x, y` — coordinate within the TU.
    pub fn scaling_factor(&self, size_id: usize, matrix_id: usize, x: usize, y: usize) -> u8 {
        debug_assert!(size_id < 4);
        // matrixId for sizeId=3 collapses {1,2,4,5} to 0/3 for the base
        // 32×32 luma/chroma lists. The spec (eq. 7-50) derives chroma
        // 32×32 from the 16×16 list when ChromaArrayType == 3, but for
        // 4:2:0 we only ever hit size_id == 3 with matrix_id == 0 or 3.
        let matrix_id = matrix_id.min(5);
        if size_id == 0 {
            // 4×4: direct lookup in up-right diagonal scan order (eq. 7-44).
            let (sx, sy) = find_scan_pos_4x4(x, y);
            let i = sy * 4 + sx; // Not actually used: we look up by (i).
            let _ = i;
            // Actually eq. 7-44 says: x = ScanOrder[...][i][0], y = [1].
            // So for a given (x, y), we need the inverse scan: find i such
            // that ScanOrder[i] == (x, y). Precomputed below.
            let i = inverse_diag_scan_4x4(x as u8, y as u8);
            return self.lists[0][matrix_id][i as usize];
        }
        if size_id == 1 {
            // 8×8 eq. 7-45.
            let i = inverse_diag_scan_8x8(x as u8, y as u8);
            return self.lists[1][matrix_id][i as usize];
        }
        // sizeId 2 / 3: x = 2*xScan + k (or 4*xScan + k), y similarly.
        let div = 1usize << (size_id - 1); // 2 for sizeId=2, 4 for sizeId=3.
        let xs = x / div;
        let ys = y / div;
        let i = inverse_diag_scan_8x8(xs as u8, ys as u8) as usize;
        // Override DC at (0, 0) per eq. 7-47 / 7-49.
        if x == 0 && y == 0 {
            return match size_id {
                2 => self.dc_16x16[matrix_id],
                3 => self.dc_32x32[matrix_id],
                _ => 16,
            };
        }
        self.lists[size_id][matrix_id][i]
    }
}

/// Find the scan-order index `i` for position (x, y) in the 4×4 up-right
/// diagonal scan. Returns 0..15.
fn inverse_diag_scan_4x4(x: u8, y: u8) -> u8 {
    for (i, &(px, py)) in DIAG_SCAN_4X4.iter().enumerate() {
        if px == x && py == y {
            return i as u8;
        }
    }
    0
}

/// Find the scan-order index `i` for (x, y) in the 8×8 up-right diagonal
/// scan. The 8×8 diagonal scan can be described as: for each anti-diagonal
/// `d = 0..14`, enumerate (x, y) with `x + y = d`, `0 <= x, y < 8`, in
/// ascending `y`. Returns 0..63.
fn inverse_diag_scan_8x8(x: u8, y: u8) -> u8 {
    let mut idx = 0u8;
    for d in 0..=14u8 {
        let y_min = d.saturating_sub(7);
        let y_max = d.min(7);
        for yy in y_min..=y_max {
            let xx = d - yy;
            if xx == x && yy == y {
                return idx;
            }
            idx += 1;
        }
    }
    0
}

/// Kept as an alias for documentation; `inverse_diag_scan_4x4` is the real
/// callable. Returns the (scan_x, scan_y) at position `(x, y)` — in
/// practice this is the same pair, but the spec phrases it via a scan-index
/// variable.
fn find_scan_pos_4x4(x: usize, y: usize) -> (usize, usize) {
    (x, y)
}

/// Parse `scaling_list_data()` (§7.3.4).
///
/// Returns a [`ScalingListData`] with ScalingList arrays populated
/// per-spec. Default tables and reference-matrix copies are handled
/// per §7.4.5 (eq. 7-42, 7-43, inferred DC values).
pub fn parse_scaling_list_data(br: &mut BitReader<'_>) -> Result<ScalingListData> {
    let mut out = ScalingListData::default_flat();
    for size_id in 0..4usize {
        let step = if size_id == 3 { 3 } else { 1 };
        let mut matrix_id = 0usize;
        while matrix_id < NUM_MATRICES[size_id] {
            let scaling_list_pred_mode_flag = br.u1()? == 1;
            if !scaling_list_pred_mode_flag {
                // Copy from either defaults or a lower matrixId.
                let delta = br.ue()? as usize;
                if delta == 0 {
                    // §7.4.5: default scaling list.
                    let src = default_list(size_id, matrix_id);
                    out.lists[size_id][matrix_id] = src;
                    if size_id >= 2 {
                        // Inferred DC when pred_matrix_id_delta == 0: 8+8=16.
                        match size_id {
                            2 => out.dc_16x16[matrix_id] = 16,
                            3 => out.dc_32x32[matrix_id] = 16,
                            _ => {}
                        }
                    }
                } else {
                    // §7.4.5 eq. 7-42: refMatrixId = matrixId - delta * step.
                    let ref_matrix_id = (matrix_id as i32) - (delta as i32) * (step as i32);
                    if ref_matrix_id < 0 {
                        return Err(Error::invalid(
                            "h265 scaling_list: pred_matrix_id_delta out of range",
                        ));
                    }
                    let ref_matrix_id = ref_matrix_id as usize;
                    out.lists[size_id][matrix_id] = out.lists[size_id][ref_matrix_id].clone();
                    if size_id == 2 {
                        out.dc_16x16[matrix_id] = out.dc_16x16[ref_matrix_id];
                    } else if size_id == 3 {
                        out.dc_32x32[matrix_id] = out.dc_32x32[ref_matrix_id];
                    }
                }
            } else {
                let coef_num = COEF_NUM[size_id];
                let mut next_coef: i32 = 8;
                if size_id > 1 {
                    let dc = br.se()?; // range [-7, 247]
                    next_coef = dc + 8;
                    let dc_val = (dc + 8).clamp(1, 255) as u8;
                    match size_id {
                        2 => out.dc_16x16[matrix_id] = dc_val,
                        3 => out.dc_32x32[matrix_id] = dc_val,
                        _ => {}
                    }
                }
                let mut list = vec![0u8; coef_num];
                for coef in list.iter_mut().take(coef_num) {
                    let delta = br.se()?;
                    next_coef = (next_coef + delta + 256).rem_euclid(256);
                    // Spec requires >0, but guard against malformed streams.
                    *coef = next_coef.clamp(1, 255) as u8;
                }
                out.lists[size_id][matrix_id] = list;
            }
            matrix_id += step;
        }
    }
    // For sizeId==3 matrixId 1,2,4,5 are not present; copy from 0/3 so
    // `scaling_factor` can still return a value (per §8.6.3 m[x][y] ignored
    // for those combinations in 4:2:0 but we keep a valid default).
    for mid in [1usize, 2, 4, 5] {
        let src = if mid < 3 { 0 } else { 3 };
        out.lists[3][mid] = out.lists[3][src].clone();
    }
    Ok(out)
}

/// §7.4.5 default scaling list: flat 16s for size_id==0, Table 7-6 split
/// (intra matrixId 0..2, inter 3..5) for size_id>=1.
fn default_list(size_id: usize, matrix_id: usize) -> Vec<u8> {
    if size_id == 0 {
        return vec![16u8; 16];
    }
    if matrix_id < 3 {
        DEFAULT_8X8_INTRA.to_vec()
    } else {
        DEFAULT_8X8_INTER.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_flat_is_all_16() {
        let d = ScalingListData::default_flat();
        for size_id in 0..4 {
            for matrix_id in 0..6 {
                for &v in &d.lists[size_id][matrix_id] {
                    assert_eq!(v, 16);
                }
            }
        }
        // Scaling factor at any position is 16.
        for size_id in 0..4 {
            let n = 1usize << (2 + size_id);
            for y in 0..n.min(4) {
                for x in 0..n.min(4) {
                    assert_eq!(d.scaling_factor(size_id, 0, x, y), 16);
                }
            }
        }
    }

    #[test]
    fn inverse_diag_4x4_matches_forward_scan() {
        for (i, &(x, y)) in DIAG_SCAN_4X4.iter().enumerate() {
            assert_eq!(inverse_diag_scan_4x4(x, y) as usize, i);
        }
    }

    #[test]
    fn inverse_diag_8x8_covers_all_positions() {
        let mut seen = [false; 64];
        for y in 0..8 {
            for x in 0..8 {
                let i = inverse_diag_scan_8x8(x, y);
                assert!(i < 64, "oob index {i} at ({x},{y})");
                assert!(!seen[i as usize], "duplicate index {i}");
                seen[i as usize] = true;
            }
        }
    }

    #[test]
    fn spec_defaults_match_tables() {
        let d = ScalingListData::spec_defaults();
        // sizeId 0 all 16.
        for v in &d.lists[0][0] {
            assert_eq!(*v, 16);
        }
        // sizeId 1 intra (matrixId 0) matches DEFAULT_8X8_INTRA.
        assert_eq!(d.lists[1][0], DEFAULT_8X8_INTRA.to_vec());
        // sizeId 1 inter (matrixId 3) matches DEFAULT_8X8_INTER.
        assert_eq!(d.lists[1][3], DEFAULT_8X8_INTER.to_vec());
    }

    #[test]
    fn parse_all_defaults_yields_flat_plus_tables() {
        // Synthesise a `scaling_list_data()` where every (sizeId, matrixId)
        // uses pred_mode_flag=0 and pred_matrix_id_delta=0. That's one
        // `u(1)=0` bit followed by one `ue(v)=0` bit per matrix.
        //
        // Matrix count: sizeId 0..2 have 6 matrices each (18 total), sizeId 3
        // has 2 (0, 3). Total = 20. Each entry = 2 bits (0b0 + 0b1 for ue(0)
        // = '1' = value 0). Wait: ue(0) is '1' (one bit). So 0 then 1 = 2
        // bits per matrix × 20 = 40 bits.
        let mut bits: Vec<u8> = Vec::new();
        let mut cur: u8 = 0;
        let mut n: u8 = 0;
        let push_bit = |b: u8, cur: &mut u8, n: &mut u8, bits: &mut Vec<u8>| {
            *cur = (*cur << 1) | (b & 1);
            *n += 1;
            if *n == 8 {
                bits.push(*cur);
                *cur = 0;
                *n = 0;
            }
        };
        for size_id in 0..4 {
            let step = if size_id == 3 { 3 } else { 1 };
            let mut mid = 0;
            while mid < 6 {
                push_bit(0, &mut cur, &mut n, &mut bits); // pred_mode_flag
                push_bit(1, &mut cur, &mut n, &mut bits); // ue(0) = '1'
                mid += step;
            }
        }
        if n > 0 {
            cur <<= 8 - n;
            bits.push(cur);
        }
        let mut br = BitReader::new(&bits);
        let d = parse_scaling_list_data(&mut br).expect("parse");
        assert_eq!(d.lists[0][0], vec![16u8; 16]);
        assert_eq!(d.lists[1][0], DEFAULT_8X8_INTRA.to_vec());
        assert_eq!(d.lists[2][5], DEFAULT_8X8_INTER.to_vec());
    }
}
