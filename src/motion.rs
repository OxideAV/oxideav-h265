//! §8.5.3.2 — motion-vector reconstruction, chroma MV derivation, the
//! zero-MV merge fallback, and the per-block motion field.
//!
//! This module turns the §7.3.8.6 `prediction_unit()` syntax (the
//! signalled `mvd`, `mvp_lX_flag`, `ref_idx`, `inter_pred_idc` /
//! `merge_idx`) into the resolved per-PU motion data the §8.5.3.3.1
//! block-walk driver ([`crate::inter_pred::predict_inter_pu`]) consumes,
//! and stores it in a per-4×4-block [`MotionField`] that the §8.7.2.4
//! boundary-filtering-strength derivation reads.
//!
//! Three numeric §8.5.3.2 sub-processes are self-contained and live here:
//!
//! * §8.5.3.2.1 step 4 / 5 — [`reconstruct_mv`]: the
//!   `uLX = (mvpLX + mvdLX + 2^16) % 2^16` wrap of equations 8-94..8-101,
//!   for both the fractional-MV path (eqs 8-94..8-97) and the
//!   integer-MV path (eqs 8-98..8-101, used when the reference picture is
//!   the current picture or `use_integer_mv_flag == 1`).
//! * §8.5.3.2.10 — [`derive_chroma_mv`]: `mvCLX = mvLX * 2 / SubWidthC`
//!   (equations 8-210 / 8-211).
//! * §8.5.3.2.5 — [`append_zero_merge_candidates`]: the zero-MV merge
//!   fallback (equations 8-152..8-169) that pads the merge candidate
//!   list out to `MaxNumMergeCand` when the spatial / temporal / combined
//!   candidates do not fill it.
//!
//! The §8.5.3.2.2 merge-mode driver, the §8.5.3.2.3 spatial / §8.5.3.2.8
//! temporal candidate derivation, and the §8.5.3.2.6 / §8.5.3.2.7 MVP
//! candidate construction depend on the neighbour reconstruction state
//! (the spatial neighbour PUs and the collocated picture) and are the
//! picture-driver's follow-up; this module provides the arithmetic the
//! drivers compose.

/// A `[mv[0], mv[1]]` motion vector. Luma MVs are in quarter-luma-sample
/// units; chroma MVs are in eighth-chroma-sample units.
pub type Mv = [i32; 2];

/// `2^16` — the §8.5.3.2.1 motion-vector wrap modulus (eqs 8-94..8-101).
const MV_WRAP: i32 = 1 << 16;
/// `2^15` — the §8.5.3.2.1 wrap threshold; values `>= 2^15` fold to the
/// negative half so the result lands in `[−2^15, 2^15 − 1]`.
const MV_HALF: i32 = 1 << 15;

/// §8.5.3.2.1 step 4 / 5 — reconstruct one luma motion-vector component
/// from its predictor and delta with the `2^16` wrap.
///
/// `integer_mv` selects between the fractional path (eqs 8-94..8-97,
/// `integer_mv == false`) and the integer-MV path (eqs 8-98..8-101,
/// `integer_mv == true`), the latter applying when the reference picture
/// is the current picture or `use_integer_mv_flag == 1`.
#[inline]
#[must_use]
fn reconstruct_component(mvp: i32, mvd: i32, integer_mv: bool) -> i32 {
    // eq 8-94 / 8-98: uLX = (mvpLX + mvdLX + 2^16) % 2^16, with the
    // integer path first quantizing the predictor to full-sample units
    // ( (mvp >> 2) + mvd ) << 2.
    let sum = if integer_mv {
        (((mvp >> 2) + mvd) << 2) + MV_WRAP
    } else {
        mvp + mvd + MV_WRAP
    };
    let u = sum.rem_euclid(MV_WRAP);
    // eq 8-95 / 8-99: fold the upper half to the negative range.
    if u >= MV_HALF {
        u - MV_WRAP
    } else {
        u
    }
}

/// §8.5.3.2.1 step 4 / 5 — reconstruct a luma motion vector `mvLX` from
/// the motion-vector predictor `mvpLX` and the decoded delta `mvdLX`
/// (equations 8-94..8-101), with the `2^16` wrap on each component.
///
/// `integer_mv` is `true` for the §8.5.3.2.1 step-5 integer path
/// (reference picture is the current picture, or `use_integer_mv_flag`),
/// `false` for the step-4 fractional path. The result is guaranteed to
/// lie in `[−2^15, 2^15 − 1]` per spec NOTE 1 / NOTE 2.
#[must_use]
pub fn reconstruct_mv(mvp: Mv, mvd: Mv, integer_mv: bool) -> Mv {
    [
        reconstruct_component(mvp[0], mvd[0], integer_mv),
        reconstruct_component(mvp[1], mvd[1], integer_mv),
    ]
}

/// §8.5.3.2.10 — derive the chroma motion vector `mvCLX` from the luma
/// motion vector `mvLX` (equations 8-210 / 8-211):
/// `mvCLX[0] = mvLX[0] * 2 / SubWidthC`, `mvCLX[1] = mvLX[1] * 2 / SubHeightC`.
///
/// `(sub_w, sub_h)` are `(SubWidthC, SubHeightC)` from Table 6-1. The
/// spec division is signed integer division (truncation toward zero);
/// `i32::wrapping_div` is exact here because the operands never overflow.
#[must_use]
pub fn derive_chroma_mv(mv_l: Mv, sub_w: i32, sub_h: i32) -> Mv {
    [mv_l[0] * 2 / sub_w, mv_l[1] * 2 / sub_h]
}

/// One merge candidate's motion data — the per-list reference index,
/// utilization flag and motion vector (§8.5.3.2.x candidate lists).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MergeCandidate {
    /// `refIdxL0N` (−1 when L0 is unused).
    pub ref_idx_l0: i32,
    /// `refIdxL1N` (−1 when L1 is unused).
    pub ref_idx_l1: i32,
    /// `predFlagL0N`.
    pub pred_flag_l0: bool,
    /// `predFlagL1N`.
    pub pred_flag_l1: bool,
    /// `mvL0N`.
    pub mv_l0: Mv,
    /// `mvL1N`.
    pub mv_l1: Mv,
}

/// §8.5.3.2.5 — append zero-motion-vector merge candidates to a partial
/// merge candidate list until it holds `max_num_merge_cand` entries
/// (equations 8-152..8-169).
///
/// `cand_list` is the list after the spatial / temporal / combined
/// candidates (`numCurrMergeCand` entries); `slice_is_b` selects the B
/// (bi-predictive zero candidate, eqs 8-161..8-168) versus P
/// (uni-L0 zero candidate, eqs 8-152..8-159) form. `num_ref_idx` is the
/// §8.5.3.2.5 `numRefIdx` (P: `num_ref_idx_l0_active`; B:
/// `Min(l0_active, l1_active)`). The list is grown in place.
pub fn append_zero_merge_candidates(
    cand_list: &mut Vec<MergeCandidate>,
    slice_is_b: bool,
    num_ref_idx: i32,
    max_num_merge_cand: usize,
) {
    let mut zero_idx = 0i32;
    while cand_list.len() < max_num_merge_cand {
        // eq 8-152 / 8-161: refIdxL0 = (zeroIdx < numRefIdx) ? zeroIdx : 0.
        let ref_idx = if zero_idx < num_ref_idx { zero_idx } else { 0 };
        let cand = if slice_is_b {
            MergeCandidate {
                ref_idx_l0: ref_idx,
                ref_idx_l1: ref_idx,
                pred_flag_l0: true,
                pred_flag_l1: true,
                mv_l0: [0, 0],
                mv_l1: [0, 0],
            }
        } else {
            MergeCandidate {
                ref_idx_l0: ref_idx,
                ref_idx_l1: -1,
                pred_flag_l0: true,
                pred_flag_l1: false,
                mv_l0: [0, 0],
                mv_l1: [0, 0],
            }
        };
        cand_list.push(cand);
        zero_idx += 1;
    }
}

/// Per-4×4-block motion / mode information for one decoded picture, the
/// store the §8.7.2.4 boundary-strength derivation and the inter
/// reconstruction driver read out of.
///
/// The grid is indexed in 4×4-luma-sample units (`PuMvField` granularity
/// of §8.5.3): block `(bx, by)` covers luma samples
/// `[bx*4, bx*4+3] × [by*4, by*4+3]`. Intra blocks carry `predMode ==
/// intra` and undefined MVs; inter blocks carry the §8.5.3.2-resolved
/// `predFlagLX` / `refIdxLX` (as an opaque reference-picture identity)
/// / `mvLX`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MotionField {
    width_4: usize,
    height_4: usize,
    cells: Vec<MotionCell>,
}

/// One 4×4 block's motion / mode record in a [`MotionField`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MotionCell {
    /// `true` when the covering coding unit is intra (`MODE_INTRA`).
    pub is_intra: bool,
    /// `true` when the covering luma transform block holds one or more
    /// non-zero transform-coefficient levels (the §8.7.2.4 `cbf` test).
    pub has_nonzero_coeff: bool,
    /// `PredFlagL0[x][y]`.
    pub pred_flag_l0: bool,
    /// `PredFlagL1[x][y]`.
    pub pred_flag_l1: bool,
    /// A reference-picture identity for list 0 (e.g. the POC) — the
    /// §8.7.2.4 "different reference pictures" test compares these, not
    /// the `RefIdxLX` list positions (NOTE 1). `i32::MIN` when unused.
    pub ref_poc_l0: i32,
    /// A reference-picture identity for list 1; `i32::MIN` when unused.
    pub ref_poc_l1: i32,
    /// `MvL0[x][y]` in quarter-luma-sample units.
    pub mv_l0: Mv,
    /// `MvL1[x][y]`.
    pub mv_l1: Mv,
}

impl MotionField {
    /// Allocate a motion field covering a `width_luma × height_luma`
    /// picture, rounded up to whole 4×4 blocks. Every cell starts as an
    /// intra cell with no motion.
    #[must_use]
    pub fn new(width_luma: usize, height_luma: usize) -> Self {
        let width_4 = width_luma.div_ceil(4);
        let height_4 = height_luma.div_ceil(4);
        // The unwritten background is an intra cell with no reference
        // pictures (the §8.7.2.4 intra test reads `is_intra`).
        let background = MotionCell {
            is_intra: true,
            ref_poc_l0: i32::MIN,
            ref_poc_l1: i32::MIN,
            ..MotionCell::default()
        };
        Self {
            width_4,
            height_4,
            cells: vec![background; width_4 * height_4],
        }
    }

    /// Width of the field in 4×4 blocks.
    #[inline]
    #[must_use]
    pub fn width_4(&self) -> usize {
        self.width_4
    }

    /// Height of the field in 4×4 blocks.
    #[inline]
    #[must_use]
    pub fn height_4(&self) -> usize {
        self.height_4
    }

    /// The motion cell covering luma sample `(x, y)`.
    ///
    /// # Panics
    /// Panics if `(x, y)` lies outside the field.
    #[must_use]
    pub fn cell_at(&self, x: usize, y: usize) -> MotionCell {
        let bx = x / 4;
        let by = y / 4;
        self.cells[by * self.width_4 + bx]
    }

    /// Set every 4×4 cell covering the luma rectangle `[(x0, y0),
    /// (x0 + w, y0 + h))` to `cell`. Coordinates and dimensions are in
    /// luma samples; the rectangle is clipped to the field.
    pub fn fill_rect(&mut self, x0: usize, y0: usize, w: usize, h: usize, cell: MotionCell) {
        let bx0 = x0 / 4;
        let by0 = y0 / 4;
        let bx1 = ((x0 + w).min(self.width_4 * 4)).div_ceil(4);
        let by1 = ((y0 + h).min(self.height_4 * 4)).div_ceil(4);
        for by in by0..by1 {
            for bx in bx0..bx1 {
                self.cells[by * self.width_4 + bx] = cell;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mv_reconstruct_fractional_no_wrap() {
        // mvp + mvd, no wrap needed.
        assert_eq!(reconstruct_mv([10, -4], [3, 2], false), [13, -2]);
    }

    #[test]
    fn mv_reconstruct_wraps_to_negative_half() {
        // A component that exceeds 2^15 − 1 folds into the negative half.
        // mvp = 2^15 − 1, mvd = 1 ⇒ u = 2^15 ⇒ result = 2^15 − 2^16 = −2^15.
        let r = reconstruct_mv([MV_HALF - 1, 0], [1, 0], false);
        assert_eq!(r[0], -MV_HALF);
    }

    #[test]
    fn mv_reconstruct_wraps_negative_underflow() {
        // mvp = −2^15, mvd = −1 ⇒ sum = −2^15 − 1 + 2^16 = 2^15 − 1 ⇒
        // stays positive (the wrap keeps the value in range).
        let r = reconstruct_mv([-MV_HALF, 0], [-1, 0], false);
        assert_eq!(r[0], MV_HALF - 1);
    }

    #[test]
    fn mv_reconstruct_integer_path_quantizes_predictor() {
        // integer path: ((mvp >> 2) + mvd) << 2. mvp = 13 (>>2 = 3),
        // mvd = 1 ⇒ (3 + 1) << 2 = 16.
        let r = reconstruct_mv([13, 0], [1, 0], true);
        assert_eq!(r[0], 16);
    }

    #[test]
    fn chroma_mv_420() {
        // 4:2:0: SubWidthC = SubHeightC = 2 ⇒ mvC = mvL * 2 / 2 = mvL.
        assert_eq!(derive_chroma_mv([7, -9], 2, 2), [7, -9]);
    }

    #[test]
    fn chroma_mv_422() {
        // 4:2:2: SubWidthC = 2, SubHeightC = 1 ⇒ mvCx = mvLx, mvCy = mvLy*2.
        assert_eq!(derive_chroma_mv([7, -9], 2, 1), [7, -18]);
    }

    #[test]
    fn chroma_mv_444() {
        // 4:4:4: SubWidthC = SubHeightC = 1 ⇒ mvC = mvL * 2.
        assert_eq!(derive_chroma_mv([7, -9], 1, 1), [14, -18]);
    }

    #[test]
    fn zero_merge_p_slice_pads_uni_l0() {
        let mut list = vec![MergeCandidate {
            ref_idx_l0: 0,
            ref_idx_l1: -1,
            pred_flag_l0: true,
            pred_flag_l1: false,
            mv_l0: [4, 4],
            mv_l1: [0, 0],
        }];
        append_zero_merge_candidates(&mut list, false, 2, 5);
        assert_eq!(list.len(), 5);
        // The four added candidates are uni-L0 zero MVs; refIdx ramps
        // 0,1 then clamps to 0 (zeroIdx >= numRefIdx).
        assert_eq!(list[1].ref_idx_l0, 0);
        assert_eq!(list[2].ref_idx_l0, 1);
        assert_eq!(list[3].ref_idx_l0, 0);
        assert_eq!(list[4].ref_idx_l0, 0);
        for c in &list[1..] {
            assert_eq!(c.mv_l0, [0, 0]);
            assert!(c.pred_flag_l0 && !c.pred_flag_l1);
        }
    }

    #[test]
    fn zero_merge_b_slice_pads_bi() {
        let mut list = Vec::new();
        append_zero_merge_candidates(&mut list, true, 1, 3);
        assert_eq!(list.len(), 3);
        for c in &list {
            assert!(c.pred_flag_l0 && c.pred_flag_l1, "B zero cand is bi");
            assert_eq!(c.mv_l0, [0, 0]);
            assert_eq!(c.mv_l1, [0, 0]);
        }
        // numRefIdx = 1 ⇒ only zeroIdx 0 maps to refIdx 0; rest clamp.
        assert_eq!(list[0].ref_idx_l0, 0);
        assert_eq!(list[1].ref_idx_l0, 0);
    }

    #[test]
    fn motion_field_fill_and_query() {
        let mut mf = MotionField::new(64, 64);
        assert_eq!(mf.width_4(), 16);
        assert_eq!(mf.height_4(), 16);
        let inter = MotionCell {
            is_intra: false,
            has_nonzero_coeff: true,
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_poc_l0: 0,
            ref_poc_l1: i32::MIN,
            mv_l0: [8, -4],
            mv_l1: [0, 0],
        };
        mf.fill_rect(16, 16, 32, 32, inter);
        // Inside the rect.
        assert_eq!(mf.cell_at(20, 20), inter);
        assert_eq!(mf.cell_at(47, 47), inter);
        // Outside stays intra default.
        assert!(mf.cell_at(0, 0).is_intra);
        assert!(mf.cell_at(48, 48).is_intra);
    }
}
