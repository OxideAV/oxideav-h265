//! §7.3.4 `scaling_list_data()` parse + §7.4.5 `ScalingList` derivation.
//!
//! H.265 carries an optional set of quantization scaling lists in the
//! SPS (`sps_scaling_list_data_present_flag == 1`) and PPS
//! (`pps_scaling_list_data_present_flag == 1`). The
//! `scaling_list_data()` syntax structure (ITU-T Rec. H.265 §7.3.4)
//! signals, for each of the 24 (sizeId, matrixId) slots, either:
//!
//! * a **prediction** from an earlier list of the same `sizeId`
//!   (`scaling_list_pred_mode_flag == 0`), where
//!   `scaling_list_pred_matrix_id_delta` selects the reference list
//!   (delta 0 means "use the default list", per §7.4.5); or
//! * an **explicit** delta-coded list
//!   (`scaling_list_pred_mode_flag == 1`), built from a running
//!   `nextCoef` accumulator modulo 256, optionally preceded by a DC
//!   coefficient (`scaling_list_dc_coef_minus8`) for sizeId > 1.
//!
//! This module parses that structure and derives the flat
//! `ScalingList[sizeId][matrixId][i]` coefficient arrays plus the
//! per-slot DC coefficients, applying the §7.4.5 default-list and
//! prediction-inference rules (equations 7-42 and 7-43). The
//! up-right-diagonal scan into the two-dimensional `ScalingFactor`
//! array (equations 7-44..7-51, requiring §6.5.3 `ScanOrder`) is left
//! to a follow-up; this module stops at the flat per-`i` lists, which
//! is the form the SPS/PPS parse needs to no longer reject scaling
//! lists.

use crate::bitreader::{BitReader, BitReaderError};

/// Number of `sizeId` values (4x4, 8x8, 16x16, 32x32) — §7.4.5
/// Table 7-3.
pub const NUM_SIZE_IDS: usize = 4;

/// Number of `matrixId` values per `sizeId` — §7.4.5 Table 7-4
/// (intra Y/Cb/Cr + inter Y/Cb/Cr).
pub const NUM_MATRIX_IDS: usize = 6;

/// Maximum number of coefficients in a flat scaling list. `sizeId` 0
/// carries 16; `sizeId` 1..3 carry 64
/// (`Min(64, 1 << (4 + (sizeId << 1)))`).
pub const MAX_COEF_NUM: usize = 64;

/// Number of explicit coefficients for a given `sizeId`:
/// `Min(64, 1 << (4 + (sizeId << 1)))` (§7.3.4) — 16 for `sizeId` 0,
/// 64 otherwise.
fn coef_num(size_id: usize) -> usize {
    (1usize << (4 + (size_id << 1))).min(MAX_COEF_NUM)
}

/// Errors that can arise while parsing `scaling_list_data()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingListError {
    /// The RBSP ran out of bits before the structure was fully parsed.
    Truncated,
    /// `scaling_list_pred_matrix_id_delta` exceeded the §7.4.5 bound
    /// (`matrixId` for `sizeId <= 2`, `matrixId / 3` for `sizeId == 3`),
    /// which would index a non-existent or future reference list.
    PredMatrixIdDeltaOutOfRange {
        /// The `sizeId` of the offending slot.
        size_id: u8,
        /// The `matrixId` of the offending slot.
        matrix_id: u8,
        /// The (illegal) `scaling_list_pred_matrix_id_delta` value.
        got: u32,
    },
    /// `scaling_list_dc_coef_minus8` was outside the §7.4.5 range of
    /// −7 to 247, inclusive.
    DcCoefOutOfRange {
        /// The `sizeId` of the offending slot (always > 1).
        size_id: u8,
        /// The `matrixId` of the offending slot.
        matrix_id: u8,
        /// The (illegal) `scaling_list_dc_coef_minus8` value.
        got: i32,
    },
    /// A derived coefficient `ScalingList[sizeId][matrixId][i]` was not
    /// greater than 0, violating the §7.4.5 conformance requirement.
    NonPositiveCoef {
        /// The `sizeId` of the offending slot.
        size_id: u8,
        /// The `matrixId` of the offending slot.
        matrix_id: u8,
        /// The coefficient index `i`.
        index: u8,
    },
    /// An unexpected bitstream-level error surfaced from the reader.
    Bitstream(BitReaderError),
}

impl core::fmt::Display for ScalingListError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => f.write_str("scaling_list_data() truncated"),
            Self::PredMatrixIdDeltaOutOfRange {
                size_id,
                matrix_id,
                got,
            } => write!(
                f,
                "scaling_list_pred_matrix_id_delta[{size_id}][{matrix_id}] out of range: {got}"
            ),
            Self::DcCoefOutOfRange {
                size_id,
                matrix_id,
                got,
            } => write!(
                f,
                "scaling_list_dc_coef_minus8[{}][{matrix_id}] out of range: {got}",
                size_id - 2
            ),
            Self::NonPositiveCoef {
                size_id,
                matrix_id,
                index,
            } => write!(
                f,
                "ScalingList[{size_id}][{matrix_id}][{index}] is not greater than 0"
            ),
            Self::Bitstream(e) => write!(f, "bitstream error during scaling_list_data(): {e}"),
        }
    }
}

impl std::error::Error for ScalingListError {}

impl From<BitReaderError> for ScalingListError {
    fn from(e: BitReaderError) -> Self {
        match e {
            BitReaderError::EndOfBuffer => Self::Truncated,
            other => Self::Bitstream(other),
        }
    }
}

/// Default 4x4 list — §7.4.5 Table 7-5: every coefficient is 16 for
/// all six `matrixId`s.
const DEFAULT_4X4: [u16; 16] = [16; 16];

/// Default 8x8 intra list (`matrixId` 0..2) — §7.4.5 Table 7-6, rows
/// for `ScalingList[1..3][0..2]`. Also serves as the default for the
/// 16x16 and 32x32 intra lists (sizeId 1, 2, 3).
const DEFAULT_8X8_INTRA: [u16; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 16, 17, 16, 17, 18, // i = 0..15
    17, 18, 18, 17, 18, 21, 19, 20, 21, 20, 19, 21, 24, 22, 22, 24, // i = 16..31
    24, 22, 22, 24, 25, 25, 27, 30, 27, 25, 25, 29, 31, 35, 35, 31, // i = 32..47
    29, 36, 41, 44, 41, 36, 47, 54, 54, 47, 65, 70, 65, 88, 88, 115, // i = 48..63
];

/// Default 8x8 inter list (`matrixId` 3..5) — §7.4.5 Table 7-6, rows
/// for `ScalingList[1..3][3..5]`. Also serves as the default for the
/// 16x16 and 32x32 inter lists.
const DEFAULT_8X8_INTER: [u16; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, // i = 0..15
    18, 18, 18, 18, 18, 20, 20, 20, 20, 20, 20, 20, 24, 24, 24, 24, // i = 16..31
    24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 28, 28, 28, 28, 28, // i = 32..47
    28, 33, 33, 33, 33, 33, 41, 41, 41, 41, 54, 54, 54, 71, 71, 91, // i = 48..63
];

/// The §7.4.5 default `ScalingList[sizeId][matrixId][i]` values, for
/// `i = 0..coefNum(sizeId) - 1`. `sizeId == 0` is the flat 4x4 list;
/// every other `sizeId` reuses the 8x8 intra/inter defaults
/// (`matrixId < 3` → intra, `matrixId >= 3` → inter).
pub fn default_scaling_list(size_id: usize, matrix_id: usize) -> &'static [u16] {
    if size_id == 0 {
        &DEFAULT_4X4
    } else if matrix_id < 3 {
        &DEFAULT_8X8_INTRA
    } else {
        &DEFAULT_8X8_INTER
    }
}

/// One parsed-and-derived scaling list: the flat coefficient array
/// `ScalingList[sizeId][matrixId][i]` plus its DC coefficient (the
/// §7.4.5 `scaling_list_dc_coef_minus8 + 8`, only meaningful for
/// `sizeId > 1`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalingListMatrix {
    /// The flat coefficients, length `coefNum(sizeId)` (16 for
    /// `sizeId` 0, else 64).
    pub coef: Vec<u16>,
    /// `scaling_list_dc_coef_minus8 + 8` — the DC coefficient for
    /// `sizeId > 1`, supplying `ScalingFactor[sizeId][matrixId][0][0]`.
    /// Inferred to 8 (and unused) for `sizeId <= 1`.
    pub dc_coef: u16,
}

/// The full parsed-and-derived `scaling_list_data()` structure: all 24
/// `(sizeId, matrixId)` slots. Slots not signalled for `sizeId == 3`
/// (matrixId 1, 2, 4, 5 are skipped by the `matrixId += 3` step)
/// retain their default value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalingListData {
    /// `[sizeId][matrixId]` — the derived per-slot scaling lists.
    pub lists: [[ScalingListMatrix; NUM_MATRIX_IDS]; NUM_SIZE_IDS],
}

impl ScalingListData {
    /// Parse `scaling_list_data()` (§7.3.4) and derive the per-slot
    /// `ScalingList` coefficient arrays (§7.4.5), applying the
    /// default-list and prediction-inference rules.
    pub fn parse(br: &mut BitReader<'_>) -> Result<Self, ScalingListError> {
        // Seed every slot with its default; signalled slots overwrite.
        // sizeId == 3 only visits matrixId 0 and 3 (the `+= 3` step),
        // so the other four slots stay at their (unused) default.
        let mut lists: [[ScalingListMatrix; NUM_MATRIX_IDS]; NUM_SIZE_IDS] =
            core::array::from_fn(|size_id| {
                core::array::from_fn(|matrix_id| ScalingListMatrix {
                    coef: default_scaling_list(size_id, matrix_id).to_vec(),
                    dc_coef: 8,
                })
            });

        for (size_id, size_lists) in lists.iter_mut().enumerate() {
            // matrixId += (sizeId == 3) ? 3 : 1  (§7.3.4)
            let step = if size_id == 3 { 3 } else { 1 };
            let mut matrix_id = 0usize;
            while matrix_id < NUM_MATRIX_IDS {
                let pred_mode_flag = br.u1()? != 0;
                if !pred_mode_flag {
                    // Prediction from a reference list of the same sizeId.
                    let delta = br.ue()?;
                    let max_delta = if size_id <= 2 {
                        matrix_id as u32
                    } else {
                        matrix_id as u32 / 3
                    };
                    if delta > max_delta {
                        return Err(ScalingListError::PredMatrixIdDeltaOutOfRange {
                            size_id: size_id as u8,
                            matrix_id: matrix_id as u8,
                            got: delta,
                        });
                    }
                    if delta == 0 {
                        // §7.4.5: inferred from the default scaling list.
                        let def = default_scaling_list(size_id, matrix_id);
                        size_lists[matrix_id] = ScalingListMatrix {
                            coef: def.to_vec(),
                            // DC default for sizeId > 1 is also 8.
                            dc_coef: 8,
                        };
                    } else {
                        // refMatrixId = matrixId − delta * step  (7-42),
                        // and copy the reference list (7-43) including
                        // its DC coefficient (§7.4.5 last paragraph).
                        let ref_matrix_id = matrix_id - (delta as usize) * step;
                        size_lists[matrix_id] = size_lists[ref_matrix_id].clone();
                    }
                } else {
                    // Explicitly signalled list.
                    let n = coef_num(size_id);
                    let mut next_coef: i32 = 8;
                    let mut dc_coef: u16 = 8;
                    if size_id > 1 {
                        let dc = br.se()?;
                        if !(-7..=247).contains(&dc) {
                            return Err(ScalingListError::DcCoefOutOfRange {
                                size_id: size_id as u8,
                                matrix_id: matrix_id as u8,
                                got: dc,
                            });
                        }
                        next_coef = dc + 8;
                        dc_coef = (dc + 8) as u16;
                    }
                    let mut coef = Vec::with_capacity(n);
                    for i in 0..n {
                        let delta_coef = br.se()?;
                        // nextCoef = (nextCoef + delta_coef + 256) % 256
                        next_coef = (next_coef + delta_coef + 256).rem_euclid(256);
                        if next_coef <= 0 {
                            return Err(ScalingListError::NonPositiveCoef {
                                size_id: size_id as u8,
                                matrix_id: matrix_id as u8,
                                index: i as u8,
                            });
                        }
                        coef.push(next_coef as u16);
                    }
                    size_lists[matrix_id] = ScalingListMatrix { coef, dc_coef };
                }
                matrix_id += step;
            }
        }

        Ok(Self { lists })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack a sequence of `(value, bit_width)` MSB-first into bytes.
    fn pack(fields: &[(u64, u32)]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut cur = 0u8;
        let mut nbits = 0u32;
        for &(val, w) in fields {
            for i in (0..w).rev() {
                let bit = ((val >> i) & 1) as u8;
                cur = (cur << 1) | bit;
                nbits += 1;
                if nbits == 8 {
                    out.push(cur);
                    cur = 0;
                    nbits = 0;
                }
            }
        }
        if nbits > 0 {
            cur <<= 8 - nbits;
            out.push(cur);
        }
        out
    }

    /// ue(v) codeword for a small codeNum, as `(value, width)` fields.
    /// codeNum 0 → '1'; codeNum N → prefix of `m` zeros, a 1, then
    /// `m`-bit suffix where `m = floor(log2(N+1))`.
    fn ue(code_num: u32) -> Vec<(u64, u32)> {
        let x = code_num + 1;
        let m = 31 - x.leading_zeros(); // floor(log2(x))
                                        // m leading zeros + the implicit leading 1 of `x` (so `x` in
                                        // (m+1) bits) gives the standard Exp-Golomb codeword.
        vec![(0, m), (x as u64, m + 1)]
    }

    /// se(v) codeword: map value → codeNum per Table 9-3, then ue().
    fn se(value: i32) -> Vec<(u64, u32)> {
        let code_num = if value > 0 {
            (2 * value - 1) as u32
        } else {
            (-2 * value) as u32
        };
        ue(code_num)
    }

    /// A `scaling_list_data()` where every one of the 24 slots is
    /// pred_mode_flag=0, delta=0 (i.e. "use the default list").
    /// 24 slots visited: sizeId 0/1/2 each 6, sizeId 3 only 2.
    fn all_default_bits() -> Vec<(u64, u32)> {
        let mut f = Vec::new();
        for size_id in 0..4 {
            let step = if size_id == 3 { 3 } else { 1 };
            let mut m = 0;
            while m < 6 {
                f.push((0, 1)); // scaling_list_pred_mode_flag = 0
                f.extend(ue(0)); // scaling_list_pred_matrix_id_delta = 0
                m += step;
            }
        }
        f
    }

    #[test]
    fn all_default_lists_match_tables() {
        let bits = all_default_bits();
        let buf = pack(&bits);
        let mut br = BitReader::new(&buf);
        let data = ScalingListData::parse(&mut br).unwrap();

        // 4x4 — Table 7-5: all 16.
        for m in 0..6 {
            assert_eq!(data.lists[0][m].coef, vec![16u16; 16]);
        }
        // 8x8 intra (matrixId 0..2) — Table 7-6.
        for m in 0..3 {
            assert_eq!(data.lists[1][m].coef, DEFAULT_8X8_INTRA.to_vec());
        }
        // 8x8 inter (matrixId 3..5).
        for m in 3..6 {
            assert_eq!(data.lists[1][m].coef, DEFAULT_8X8_INTER.to_vec());
        }
        // 16x16 reuses the 8x8 defaults; DC default 8.
        assert_eq!(data.lists[2][0].coef, DEFAULT_8X8_INTRA.to_vec());
        assert_eq!(data.lists[2][0].dc_coef, 8);
        assert_eq!(data.lists[2][3].coef, DEFAULT_8X8_INTER.to_vec());
        // 32x32 visits only matrixId 0 (intra) and 3 (inter).
        assert_eq!(data.lists[3][0].coef, DEFAULT_8X8_INTRA.to_vec());
        assert_eq!(data.lists[3][3].coef, DEFAULT_8X8_INTER.to_vec());
        assert_eq!(data.lists[3][0].dc_coef, 8);
    }

    /// Coefficient-numbering: sizeId 0 carries 16, sizeId 1..3 carry 64.
    #[test]
    fn coef_num_per_size_id() {
        assert_eq!(coef_num(0), 16);
        assert_eq!(coef_num(1), 64);
        assert_eq!(coef_num(2), 64);
        assert_eq!(coef_num(3), 64);
    }

    /// An explicitly-signalled 4x4 list (sizeId 0, matrixId 0): all 16
    /// slots carry delta 0, so every coefficient equals the
    /// `nextCoef` seed (8). Verifies the running modulo accumulator and
    /// that no DC coefficient is read for sizeId <= 1.
    #[test]
    fn explicit_flat_4x4_list() {
        let mut f = Vec::new();
        // sizeId 0, matrixId 0: pred_mode = 1, 16 delta_coef = 0.
        f.push((1, 1));
        for _ in 0..16 {
            f.extend(se(0));
        }
        // Remaining 23 slots: default (pred_mode = 0, delta = 0).
        for size_id in 0..4 {
            let step = if size_id == 3 { 3 } else { 1 };
            let mut m = 0;
            while m < 6 {
                if !(size_id == 0 && m == 0) {
                    f.push((0, 1));
                    f.extend(ue(0));
                }
                m += step;
            }
        }
        let buf = pack(&f);
        let mut br = BitReader::new(&buf);
        let data = ScalingListData::parse(&mut br).unwrap();
        // nextCoef seeds at 8; every delta is 0 ⇒ all coefs = 8.
        assert_eq!(data.lists[0][0].coef, vec![8u16; 16]);
    }

    /// Explicit 16x16 list (sizeId 2) reads a DC coefficient first.
    /// dc = +4 ⇒ scaling_list_dc_coef_minus8 = 4 ⇒ dc_coef = 12,
    /// nextCoef seeds at 12; deltas all 0 ⇒ every coef = 12.
    #[test]
    fn explicit_16x16_reads_dc_coef() {
        let mut f = Vec::new();
        // Default sizeId 0 and 1 (12 slots).
        for _ in 0..2 {
            for _ in 0..6 {
                f.push((0, 1));
                f.extend(ue(0));
            }
        }
        // sizeId 2, matrixId 0: explicit with DC = +4.
        f.push((1, 1));
        f.extend(se(4)); // scaling_list_dc_coef_minus8 = 4
        for _ in 0..64 {
            f.extend(se(0));
        }
        // sizeId 2, matrixId 1..5: default.
        for _ in 1..6 {
            f.push((0, 1));
            f.extend(ue(0));
        }
        // sizeId 3, matrixId 0 and 3: default.
        for _ in 0..2 {
            f.push((0, 1));
            f.extend(ue(0));
        }
        let buf = pack(&f);
        let mut br = BitReader::new(&buf);
        let data = ScalingListData::parse(&mut br).unwrap();
        assert_eq!(data.lists[2][0].dc_coef, 12);
        assert_eq!(data.lists[2][0].coef, vec![12u16; 64]);
    }

    /// Prediction with delta != 0 copies the referenced list (eq 7-43),
    /// including its DC coefficient. Set sizeId 2, matrixId 0 explicit
    /// with DC=+4, then matrixId 1 predicts from it (delta = 1 ⇒
    /// refMatrixId = 0).
    #[test]
    fn pred_matrix_id_delta_copies_reference() {
        let mut f = Vec::new();
        for _ in 0..2 {
            for _ in 0..6 {
                f.push((0, 1));
                f.extend(ue(0));
            }
        }
        // sizeId 2, matrixId 0: explicit, DC = +4, all deltas 0.
        f.push((1, 1));
        f.extend(se(4));
        for _ in 0..64 {
            f.extend(se(0));
        }
        // sizeId 2, matrixId 1: predicts from matrixId 0 (delta = 1).
        f.push((0, 1));
        f.extend(ue(1));
        // sizeId 2, matrixId 2..5: default.
        for _ in 2..6 {
            f.push((0, 1));
            f.extend(ue(0));
        }
        // sizeId 3, matrixId 0 and 3: default.
        for _ in 0..2 {
            f.push((0, 1));
            f.extend(ue(0));
        }
        let buf = pack(&f);
        let mut br = BitReader::new(&buf);
        let data = ScalingListData::parse(&mut br).unwrap();
        // matrixId 1 == copy of matrixId 0 (coefs and DC).
        assert_eq!(data.lists[2][1].coef, data.lists[2][0].coef);
        assert_eq!(data.lists[2][1].dc_coef, 12);
    }

    /// `scaling_list_pred_matrix_id_delta` out of range is rejected:
    /// for sizeId 0, matrixId 0 the max allowed delta is 0, so delta=1
    /// must error.
    #[test]
    fn rejects_pred_matrix_id_delta_out_of_range() {
        let mut f = Vec::new();
        f.push((0, 1)); // pred_mode = 0
        f.extend(ue(1)); // delta = 1 > matrixId(0)
        let buf = pack(&f);
        let mut br = BitReader::new(&buf);
        let err = ScalingListData::parse(&mut br).unwrap_err();
        assert_eq!(
            err,
            ScalingListError::PredMatrixIdDeltaOutOfRange {
                size_id: 0,
                matrix_id: 0,
                got: 1,
            }
        );
    }

    /// A delta-coef sequence that drives `nextCoef` to 0 (modulo 256)
    /// must be rejected (§7.4.5: ScalingList coefficient > 0).
    #[test]
    fn rejects_non_positive_coef() {
        let mut f = Vec::new();
        // sizeId 0, matrixId 0: explicit. nextCoef seeds at 8; a
        // delta_coef of −8 wraps it to 0.
        f.push((1, 1));
        f.extend(se(-8));
        let buf = pack(&f);
        let mut br = BitReader::new(&buf);
        let err = ScalingListData::parse(&mut br).unwrap_err();
        assert_eq!(
            err,
            ScalingListError::NonPositiveCoef {
                size_id: 0,
                matrix_id: 0,
                index: 0,
            }
        );
    }

    /// `scaling_list_dc_coef_minus8` outside −7..=247 is rejected.
    /// se(v) for −8 (codeNum 16) gives dc = −8 < −7.
    #[test]
    fn rejects_dc_coef_out_of_range() {
        let mut f = Vec::new();
        for _ in 0..2 {
            for _ in 0..6 {
                f.push((0, 1));
                f.extend(ue(0));
            }
        }
        // sizeId 2, matrixId 0: explicit with DC = −8 (out of range).
        f.push((1, 1));
        f.extend(se(-8));
        let buf = pack(&f);
        let mut br = BitReader::new(&buf);
        let err = ScalingListData::parse(&mut br).unwrap_err();
        assert_eq!(
            err,
            ScalingListError::DcCoefOutOfRange {
                size_id: 2,
                matrix_id: 0,
                got: -8,
            }
        );
    }

    /// Truncation midway through the structure surfaces `Truncated`.
    #[test]
    fn truncation_is_reported() {
        // Just one bit — not enough for the first slot.
        let buf = [0u8; 0];
        let mut br = BitReader::new(&buf);
        assert_eq!(
            ScalingListData::parse(&mut br).unwrap_err(),
            ScalingListError::Truncated
        );
    }
}
