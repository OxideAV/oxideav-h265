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

/// A spatial neighbour prediction unit's motion data, as read out of the
/// per-block motion field at a neighbour sample location for the
/// §8.5.3.2.3 spatial-merge / §8.5.3.2.7 MVP-candidate derivations.
///
/// This is the `(MvLX, RefIdxLX, PredFlagLX)` tuple the spec reads at
/// `[ xNbN ][ yNbN ]`. Two merge candidates "have the same motion vectors
/// and the same reference indices" (the §8.5.3.2.3 pruning test) iff their
/// `pred_flag`/`ref_idx`/`mv` for both lists are equal — i.e. iff their
/// [`NeighbourPu`] values compare equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NeighbourPu {
    /// `PredFlagL0[ xNb ][ yNb ]`.
    pub pred_flag_l0: bool,
    /// `PredFlagL1[ xNb ][ yNb ]`.
    pub pred_flag_l1: bool,
    /// `RefIdxL0[ xNb ][ yNb ]` (−1 when L0 unused).
    pub ref_idx_l0: i32,
    /// `RefIdxL1[ xNb ][ yNb ]` (−1 when L1 unused).
    pub ref_idx_l1: i32,
    /// `MvL0[ xNb ][ yNb ]`.
    pub mv_l0: Mv,
    /// `MvL1[ xNb ][ yNb ]`.
    pub mv_l1: Mv,
}

impl NeighbourPu {
    /// The §8.5.3.2.3 candidate built from this neighbour PU's motion
    /// (eqs 8-128..8-130 and the B0/A0/B1/B2 analogues): the neighbour's
    /// per-list MV / refIdx / predFlag, copied verbatim into a
    /// [`MergeCandidate`].
    #[inline]
    #[must_use]
    fn to_candidate(self) -> MergeCandidate {
        MergeCandidate {
            ref_idx_l0: self.ref_idx_l0,
            ref_idx_l1: self.ref_idx_l1,
            pred_flag_l0: self.pred_flag_l0,
            pred_flag_l1: self.pred_flag_l1,
            mv_l0: self.mv_l0,
            mv_l1: self.mv_l1,
        }
    }
}

/// The five §8.5.3.2.3 spatial neighbour positions, in the order the
/// availability tests reference them. Each carries the neighbour sample
/// location `(xNbN, yNbN)` relative to the picture and the neighbour's
/// motion data when the §6.4.2 prediction-block availability test passed
/// (`None` when the neighbour is unavailable).
///
/// Positions (eqs in §8.5.3.2.3), with `(xPb, yPb)` the PB top-left and
/// `nPbW`/`nPbH` its width/height:
/// * `A1` = `(xPb − 1,        yPb + nPbH − 1)`
/// * `B1` = `(xPb + nPbW − 1, yPb − 1)`
/// * `B0` = `(xPb + nPbW,     yPb − 1)`
/// * `A0` = `(xPb − 1,        yPb + nPbH)`
/// * `B2` = `(xPb − 1,        yPb − 1)`
#[derive(Debug, Clone, Copy, Default)]
pub struct SpatialMergeNeighbours {
    /// Neighbour `A1` motion (`None` when §6.4.2-unavailable).
    pub a1: Option<NeighbourPu>,
    /// Neighbour `B1` motion.
    pub b1: Option<NeighbourPu>,
    /// Neighbour `B0` motion.
    pub b0: Option<NeighbourPu>,
    /// Neighbour `A0` motion.
    pub a0: Option<NeighbourPu>,
    /// Neighbour `B2` motion.
    pub b2: Option<NeighbourPu>,
}

/// §8.5.3.2.3 PartMode-dependent partition exclusion for the current PU.
///
/// The spatial-merge derivation forces `availableA1` / `availableB1` to
/// FALSE for the second partition of certain `PartMode`s so the two PUs of
/// a split CU don't merge into one another (which would defeat the split).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PartitionContext {
    /// `partIdx` of the current PU within its CU (0 or 1 for two-PU modes).
    pub part_idx: u32,
    /// `true` when `PartMode ∈ {PART_Nx2N, PART_nLx2N, PART_nRx2N}` — the
    /// vertical-split modes that exclude `A1` at `partIdx == 1`.
    pub part_mode_vertical_split: bool,
    /// `true` when `PartMode ∈ {PART_2NxN, PART_2NxnU, PART_2NxnD}` — the
    /// horizontal-split modes that exclude `B1` at `partIdx == 1`.
    pub part_mode_horizontal_split: bool,
}

/// §8.5.3.2.3 — derive the up-to-five spatial merging candidates and their
/// availability flags from the neighbouring prediction units.
///
/// `neigh` carries each neighbour position's §6.4.2 availability (as
/// `Some`/`None`) plus its motion data; `part` carries the current PU's
/// `partIdx` + `PartMode` split class for the A1/B1 partition-exclusion
/// rules; `same_par_mrg` is `true` when the neighbour falls in the *same*
/// `Log2ParMrgLevel` parallel-merge region as the current PB (the
/// `xPb >> L == xNb >> L && yPb >> L == yNb >> L` test that forces a
/// neighbour unavailable). The five returned `Option`s are the available
/// candidates A1/B1/B0/A0/B2 (eqs 8-128..8-142), already pruned against the
/// earlier-derived neighbours exactly as the spec's redundancy checks
/// require.
///
/// `same_par_mrg` is a 5-tuple `(a1, b1, b0, a0, b2)` of the per-neighbour
/// parallel-merge-region tests.
#[must_use]
pub fn derive_spatial_merge_candidates(
    neigh: &SpatialMergeNeighbours,
    part: PartitionContext,
    same_par_mrg: (bool, bool, bool, bool, bool),
) -> SpatialMergeCandidates {
    let (smrg_a1, smrg_b1, smrg_b0, smrg_a0, smrg_b2) = same_par_mrg;

    // --- A1 (eqs 8-128..8-130) ---
    // availableA1 starts from §6.4.2, then is forced FALSE when the
    // neighbour is in the same parallel-merge region, or for the second
    // partition of a vertical-split PartMode.
    let a1_excluded = smrg_a1 || (part.part_mode_vertical_split && part.part_idx == 1);
    let cand_a1 = if a1_excluded { None } else { neigh.a1 };

    // --- B1 (eqs 8-131..8-133) ---
    // Forced FALSE in the same-region case, the second partition of a
    // horizontal-split PartMode, or when A1 is available with identical
    // motion (the redundancy check).
    let b1_excluded = smrg_b1
        || (part.part_mode_horizontal_split && part.part_idx == 1)
        || same_motion(cand_a1, neigh.b1);
    let cand_b1 = if b1_excluded { None } else { neigh.b1 };

    // --- B0 (eqs 8-134..8-136) ---
    // Forced FALSE in the same-region case or when B1 is available with
    // identical motion.
    let b0_excluded = smrg_b0 || same_motion(cand_b1, neigh.b0);
    let cand_b0 = if b0_excluded { None } else { neigh.b0 };

    // --- A0 (eqs 8-137..8-139) ---
    // Forced FALSE in the same-region case or when A1 is available with
    // identical motion.
    let a0_excluded = smrg_a0 || same_motion(cand_a1, neigh.a0);
    let cand_a0 = if a0_excluded { None } else { neigh.a0 };

    // --- B2 (eqs 8-140..8-142) ---
    // Forced FALSE in the same-region case, when A1 or B1 is available
    // with identical motion, or when all four of A0/A1/B0/B1 are already
    // available (the list is full before B2).
    let four_available =
        cand_a0.is_some() && cand_a1.is_some() && cand_b0.is_some() && cand_b1.is_some();
    let b2_excluded = smrg_b2
        || four_available
        || same_motion(cand_a1, neigh.b2)
        || same_motion(cand_b1, neigh.b2);
    let cand_b2 = if b2_excluded { None } else { neigh.b2 };

    SpatialMergeCandidates {
        a1: cand_a1.map(NeighbourPu::to_candidate),
        b1: cand_b1.map(NeighbourPu::to_candidate),
        b0: cand_b0.map(NeighbourPu::to_candidate),
        a0: cand_a0.map(NeighbourPu::to_candidate),
        b2: cand_b2.map(NeighbourPu::to_candidate),
    }
}

/// The §8.5.3.2.3 "have the same motion vectors and the same reference
/// indices" redundancy test: `true` iff `earlier` is available and equal
/// (per-list predFlag / refIdx / MV) to `candidate`.
#[inline]
#[must_use]
fn same_motion(earlier: Option<NeighbourPu>, candidate: Option<NeighbourPu>) -> bool {
    match (earlier, candidate) {
        (Some(e), Some(c)) => e == c,
        _ => false,
    }
}

/// The five §8.5.3.2.3 spatial merging candidates, in their named slots.
/// A slot is `Some` iff its availability flag is 1 after the redundancy /
/// partition-exclusion checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SpatialMergeCandidates {
    /// Candidate A1 (`availableFlagA1`).
    pub a1: Option<MergeCandidate>,
    /// Candidate B1 (`availableFlagB1`).
    pub b1: Option<MergeCandidate>,
    /// Candidate B0 (`availableFlagB0`).
    pub b0: Option<MergeCandidate>,
    /// Candidate A0 (`availableFlagA0`).
    pub a0: Option<MergeCandidate>,
    /// Candidate B2 (`availableFlagB2`).
    pub b2: Option<MergeCandidate>,
}

impl SpatialMergeCandidates {
    /// §8.5.3.2.2 step 5 — append the available spatial candidates into a
    /// `mergeCandList` in the spec order A1, B1, B0, A0, B2 (eq 8-119,
    /// minus the temporal `Col` candidate, which the driver inserts after
    /// these). The list is grown in place.
    pub fn append_to(&self, list: &mut Vec<MergeCandidate>) {
        for c in [self.a1, self.b1, self.b0, self.a0, self.b2]
            .into_iter()
            .flatten()
        {
            list.push(c);
        }
    }
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

/// Table 8-7 — `(l0CandIdx, l1CandIdx)` for `combIdx = 0..11`, the source
/// candidate positions the §8.5.3.2.4 combined bi-predictive candidate
/// pairs draw its L0 motion and L1 motion from.
const COMB_CAND_IDX: [(usize, usize); 12] = [
    (0, 1),
    (1, 0),
    (0, 2),
    (2, 0),
    (1, 2),
    (2, 1),
    (0, 3),
    (3, 0),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
];

/// §8.5.3.2.4 — append combined bi-predictive merging candidates to the
/// `mergeCandList`, drawing each new candidate's L0 motion from one
/// existing candidate and its L1 motion from another per Table 8-7.
///
/// `list` is the merge list after the spatial + temporal candidates
/// (`numOrigMergeCand` entries on entry); it is grown in place up to
/// `max_num_merge_cand`. `ref_poc` resolves `(list_x, ref_idx)` to the
/// picture order count of `RefPicListX[ ref_idx ]` so the eq 8-143
/// `DiffPicOrderCnt(...) != 0 || mvL0 != mvL1` distinctness test can run
/// (the candidate is only added when its L0 and L1 motion are not the
/// identical-picture/identical-MV degenerate case).
///
/// Per the spec this process only runs for B slices, and only when
/// `numOrigMergeCand` is in `2..MaxNumMergeCand`; the driver gates on
/// those before calling. The loop stops at
/// `combIdx == numOrig * (numOrig − 1)` or when the list is full.
pub fn append_combined_bi_candidates<F>(
    list: &mut Vec<MergeCandidate>,
    num_orig_merge_cand: usize,
    max_num_merge_cand: usize,
    mut ref_poc: F,
) where
    F: FnMut(usize, i32) -> i32,
{
    if !(num_orig_merge_cand > 1 && num_orig_merge_cand < max_num_merge_cand) {
        return;
    }
    let comb_limit = num_orig_merge_cand * (num_orig_merge_cand - 1);
    let mut comb_idx = 0usize;
    while comb_idx < comb_limit && list.len() < max_num_merge_cand {
        let (l0_idx, l1_idx) = COMB_CAND_IDX[comb_idx];
        comb_idx += 1;
        // The source candidates are positions in the *original* part of
        // the list (both < numOrigMergeCand by Table 8-7's range when
        // numOrig <= 5, which it always is: 5 spatial + 1 temporal).
        let l0_cand = list[l0_idx];
        let l1_cand = list[l1_idx];
        // eq 8-143 conditions: L0 of l0Cand and L1 of l1Cand both used,
        // and not the degenerate same-picture+same-MV pair.
        if !(l0_cand.pred_flag_l0 && l1_cand.pred_flag_l1) {
            continue;
        }
        let poc_l0 = ref_poc(0, l0_cand.ref_idx_l0);
        let poc_l1 = ref_poc(1, l1_cand.ref_idx_l1);
        let distinct = poc_l0 != poc_l1 || l0_cand.mv_l0 != l1_cand.mv_l1;
        if !distinct {
            continue;
        }
        list.push(MergeCandidate {
            ref_idx_l0: l0_cand.ref_idx_l0,
            ref_idx_l1: l1_cand.ref_idx_l1,
            pred_flag_l0: true,
            pred_flag_l1: true,
            mv_l0: l0_cand.mv_l0,
            mv_l1: l1_cand.mv_l1,
        });
    }
}

/// Inputs that vary per slice / SPS for the §8.5.3.2.2 merge-list build.
#[derive(Debug, Clone, Copy)]
pub struct MergeListParams {
    /// `true` for a B slice (enables the §8.5.3.2.4 combined step and the
    /// bi-predictive §8.5.3.2.5 zero candidates).
    pub slice_is_b: bool,
    /// `MaxNumMergeCand` (§7.4.7.1; `5 − five_minus_max_num_merge_cand`).
    pub max_num_merge_cand: usize,
    /// `numRefIdx` for §8.5.3.2.5 — P: `num_ref_idx_l0_active`;
    /// B: `Min(l0_active, l1_active)`.
    pub zero_num_ref_idx: i32,
}

/// §8.5.3.2.2 steps 5–10 — assemble the full `mergeCandList` and select the
/// candidate at `merge_idx`.
///
/// `spatial` is the §8.5.3.2.3 output; `col` is the §8.5.3.2.8 temporal
/// candidate (`None` when `availableFlagCol == 0` — e.g.
/// `slice_temporal_mvp_enabled_flag == 0`, which is the present decoder
/// state until the collocated-picture path lands). `ref_poc` resolves
/// `(list_x, ref_idx)` to a POC for the §8.5.3.2.4 distinctness test.
///
/// The returned [`MergeCandidate`] is `mergeCandList[ merge_idx ]` after
/// step 9, with the step-10 `(nOrigPbW + nOrigPbH == 12)` bi→uni-L0
/// reduction applied when `pb_w_plus_h == 12`.
///
/// `merge_idx` is clamped into the assembled list; a conformant bitstream
/// always indexes a valid entry because the zero-candidate padding fills
/// the list to `MaxNumMergeCand`.
#[must_use]
pub fn build_merge_candidate<F>(
    spatial: &SpatialMergeCandidates,
    col: Option<MergeCandidate>,
    params: MergeListParams,
    merge_idx: usize,
    pb_w_plus_h: u32,
    ref_poc: F,
) -> MergeCandidate
where
    F: FnMut(usize, i32) -> i32,
{
    // Step 5: spatial candidates in A1,B1,B0,A0,B2 order, then Col.
    let mut list: Vec<MergeCandidate> = Vec::with_capacity(params.max_num_merge_cand);
    spatial.append_to(&mut list);
    if let Some(c) = col {
        list.push(c);
    }
    // Step 6: numOrigMergeCand snapshot (after spatial + temporal).
    let num_orig = list.len();
    // Step 7: combined bi-predictive candidates (B slices only).
    if params.slice_is_b {
        append_combined_bi_candidates(&mut list, num_orig, params.max_num_merge_cand, ref_poc);
    }
    // Step 8: zero-MV padding to MaxNumMergeCand.
    append_zero_merge_candidates(
        &mut list,
        params.slice_is_b,
        params.zero_num_ref_idx,
        params.max_num_merge_cand,
    );
    // Step 9: select N = mergeCandList[ merge_idx ].
    let idx = merge_idx.min(list.len().saturating_sub(1));
    let mut chosen = list[idx];
    // Step 10: (nOrigPbW + nOrigPbH == 12) ⇒ drop L1, keep uni-L0.
    if chosen.pred_flag_l0 && chosen.pred_flag_l1 && pb_w_plus_h == 12 {
        chosen.pred_flag_l1 = false;
        chosen.ref_idx_l1 = -1;
    }
    chosen
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

    /// A uni-L0 neighbour PU with the given refIdx + MV.
    fn uni_l0(ref_idx: i32, mv: Mv) -> NeighbourPu {
        NeighbourPu {
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_idx_l0: ref_idx,
            ref_idx_l1: -1,
            mv_l0: mv,
            mv_l1: [0, 0],
        }
    }

    fn no_split() -> PartitionContext {
        PartitionContext::default()
    }

    fn no_par_mrg() -> (bool, bool, bool, bool, bool) {
        (false, false, false, false, false)
    }

    #[test]
    fn spatial_merge_all_unavailable_yields_empty() {
        let neigh = SpatialMergeNeighbours::default();
        let c = derive_spatial_merge_candidates(&neigh, no_split(), no_par_mrg());
        let mut list = Vec::new();
        c.append_to(&mut list);
        assert!(list.is_empty());
    }

    #[test]
    fn spatial_merge_b1_pruned_against_identical_a1() {
        // A1 and B1 carry identical motion ⇒ B1 is dropped (redundancy).
        let pu = uni_l0(0, [4, 8]);
        let neigh = SpatialMergeNeighbours {
            a1: Some(pu),
            b1: Some(pu),
            ..Default::default()
        };
        let c = derive_spatial_merge_candidates(&neigh, no_split(), no_par_mrg());
        assert!(c.a1.is_some());
        assert!(c.b1.is_none(), "B1 == A1 ⇒ pruned");
    }

    #[test]
    fn spatial_merge_b1_kept_when_motion_differs() {
        let neigh = SpatialMergeNeighbours {
            a1: Some(uni_l0(0, [4, 8])),
            b1: Some(uni_l0(0, [4, 9])),
            ..Default::default()
        };
        let c = derive_spatial_merge_candidates(&neigh, no_split(), no_par_mrg());
        assert!(c.a1.is_some() && c.b1.is_some());
        // Order into the list is A1 then B1 (eq 8-119).
        let mut list = Vec::new();
        c.append_to(&mut list);
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].mv_l0, [4, 8]);
        assert_eq!(list[1].mv_l0, [4, 9]);
    }

    #[test]
    fn spatial_merge_b2_dropped_when_four_available() {
        // A0,A1,B0,B1 all available with distinct motion ⇒ list full ⇒ B2
        // excluded even though it is itself available + distinct.
        let neigh = SpatialMergeNeighbours {
            a1: Some(uni_l0(0, [1, 0])),
            b1: Some(uni_l0(0, [2, 0])),
            b0: Some(uni_l0(0, [3, 0])),
            a0: Some(uni_l0(0, [4, 0])),
            b2: Some(uni_l0(0, [5, 0])),
        };
        let c = derive_spatial_merge_candidates(&neigh, no_split(), no_par_mrg());
        assert!(c.a1.is_some() && c.b1.is_some() && c.b0.is_some() && c.a0.is_some());
        assert!(c.b2.is_none(), "four already available ⇒ B2 excluded");
    }

    #[test]
    fn spatial_merge_a1_excluded_for_vertical_split_part1() {
        let neigh = SpatialMergeNeighbours {
            a1: Some(uni_l0(0, [7, 7])),
            ..Default::default()
        };
        let part = PartitionContext {
            part_idx: 1,
            part_mode_vertical_split: true,
            part_mode_horizontal_split: false,
        };
        let c = derive_spatial_merge_candidates(&neigh, part, no_par_mrg());
        assert!(c.a1.is_none(), "PART_Nx2N partIdx 1 ⇒ A1 excluded");
    }

    #[test]
    fn spatial_merge_same_par_mrg_forces_unavailable() {
        let neigh = SpatialMergeNeighbours {
            a1: Some(uni_l0(0, [1, 1])),
            b1: Some(uni_l0(0, [2, 2])),
            ..Default::default()
        };
        // A1's neighbour is in the same parallel-merge region.
        let c =
            derive_spatial_merge_candidates(&neigh, no_split(), (true, false, false, false, false));
        assert!(c.a1.is_none());
        assert!(c.b1.is_some());
    }

    #[test]
    fn combined_bi_pairs_l0_and_l1_per_table_8_7() {
        // Two B candidates: cand0 uni-L0, cand1 uni-L1. combIdx 0 pairs
        // l0CandIdx=0 (cand0's L0) with l1CandIdx=1 (cand1's L1).
        let cand0 = MergeCandidate {
            ref_idx_l0: 0,
            ref_idx_l1: -1,
            pred_flag_l0: true,
            pred_flag_l1: false,
            mv_l0: [4, 0],
            mv_l1: [0, 0],
        };
        let cand1 = MergeCandidate {
            ref_idx_l0: -1,
            ref_idx_l1: 0,
            pred_flag_l0: false,
            pred_flag_l1: true,
            mv_l0: [0, 0],
            mv_l1: [-4, 0],
        };
        let mut list = vec![cand0, cand1];
        // POCs distinct so the eq 8-143 distinctness test passes.
        append_combined_bi_candidates(&mut list, 2, 5, |list_x, _| if list_x == 0 { 0 } else { 8 });
        assert!(list.len() > 2, "at least one combined candidate added");
        let comb = list[2];
        assert!(comb.pred_flag_l0 && comb.pred_flag_l1, "combined is bi");
        assert_eq!(comb.mv_l0, [4, 0], "L0 from cand0");
        assert_eq!(comb.mv_l1, [-4, 0], "L1 from cand1");
    }

    #[test]
    fn combined_bi_skips_degenerate_same_pic_same_mv() {
        // Both candidates point at the same POC with the same MV ⇒ every
        // pair is degenerate ⇒ no combined candidate is produced.
        let bi = MergeCandidate {
            ref_idx_l0: 0,
            ref_idx_l1: 0,
            pred_flag_l0: true,
            pred_flag_l1: true,
            mv_l0: [2, 2],
            mv_l1: [2, 2],
        };
        let mut list = vec![bi, bi];
        append_combined_bi_candidates(&mut list, 2, 5, |_, _| 0);
        assert_eq!(list.len(), 2, "all pairs degenerate ⇒ nothing added");
    }

    #[test]
    fn combined_bi_noop_below_two_orig() {
        let mut list = vec![MergeCandidate::default()];
        append_combined_bi_candidates(&mut list, 1, 5, |_, _| 0);
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn build_merge_selects_index_and_pads_with_zero() {
        // One spatial candidate; merge_idx 2 lands in the zero padding.
        let spatial = SpatialMergeCandidates {
            a1: Some(uni_l0(0, [4, 4]).to_candidate()),
            ..Default::default()
        };
        let params = MergeListParams {
            slice_is_b: false,
            max_num_merge_cand: 5,
            zero_num_ref_idx: 1,
        };
        let chosen = build_merge_candidate(&spatial, None, params, 2, 16, |_, _| 0);
        // Index 2 is a zero-MV uni-L0 candidate.
        assert_eq!(chosen.mv_l0, [0, 0]);
        assert!(chosen.pred_flag_l0 && !chosen.pred_flag_l1);
    }

    #[test]
    fn build_merge_step10_drops_l1_for_8x4_or_4x8() {
        // A bi candidate selected at a PU with nOrigPbW + nOrigPbH == 12
        // (8x4 / 4x8) ⇒ L1 is dropped (step 10).
        let bi = MergeCandidate {
            ref_idx_l0: 0,
            ref_idx_l1: 0,
            pred_flag_l0: true,
            pred_flag_l1: true,
            mv_l0: [1, 1],
            mv_l1: [2, 2],
        };
        let spatial = SpatialMergeCandidates {
            a1: Some(bi),
            ..Default::default()
        };
        let params = MergeListParams {
            slice_is_b: true,
            max_num_merge_cand: 5,
            zero_num_ref_idx: 1,
        };
        let chosen = build_merge_candidate(&spatial, None, params, 0, 12, |_, _| 0);
        assert!(chosen.pred_flag_l0);
        assert!(!chosen.pred_flag_l1, "8x4/4x8 bi ⇒ uni-L0");
        assert_eq!(chosen.ref_idx_l1, -1);
    }

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
