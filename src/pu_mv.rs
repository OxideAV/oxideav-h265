//! §8.5.3.1 / §8.5.3.2.1 — the per-PU motion-vector / reference-index
//! resolution driver.
//!
//! This module is the seam between the §7.3.8.6 `prediction_unit()`
//! syntax (parsed by [`crate::slice_data`] into a
//! [`crate::slice_data::PredictionUnit`]) and the resolved per-PU motion
//! data the §8.5.3.3 inter-sample-prediction path
//! ([`crate::recon::reconstruct_inter_pu`]) consumes. It runs the
//! §8.5.3.2.1 "Derivation process for motion vector components and
//! reference indices": it gathers the §8.5.3.2.3 spatial neighbours out
//! of the per-block [`MotionField`], dispatches the merge-mode
//! (§8.5.3.2.2) versus MVP (§8.5.3.2.6) candidate derivation that already
//! lives in [`crate::motion`], reconstructs each list's
//! `mvLX = mvpLX + mvdLX` (eqs 8-94..8-101), applies the §8.5.3.2.1
//! `nPbSw == 8 && nPbSh == 8` bi→uni reduction (eqs 8-102/8-103), and
//! returns the fully-resolved [`PuMotion`] — which the caller then writes
//! back into the motion field (eqs 8-80..8-85) before reconstruction.
//!
//! The arithmetic sub-processes (spatial / temporal candidate lists, MVP
//! candidate, the MV wrap) stay in [`crate::motion`]; this module owns the
//! neighbour-location plumbing and the §8.5.3.2.1 ordered-step control
//! flow that ties them to the parsed PU syntax.

use crate::binarization::InterPredIdc;
use crate::motion::{
    build_merge_candidate, derive_mvp_candidate, derive_spatial_merge_candidates,
    derive_temporal_mv, reconstruct_mv, MergeCandidate, MergeListParams, MotionCell, MotionField,
    Mv, MvpContext, NeighbourPu, PartitionContext, RefPicId, SpatialMergeNeighbours,
    TemporalMvContext,
};
use crate::slice_data::PredictionUnit;

/// The §8.5.3.2.1 `PartMode` split class the `nPbSw` / `nPbSh` (eqs
/// 8-86/8-87) and the §8.5.3.2.3 partition-exclusion rules read.
///
/// The values mirror the §7.4.9.5 `PartMode` enumeration but carry only
/// the geometry the MV-resolution driver needs (the §7.3.8.5 part-mode
/// binarization itself lives in [`crate::binarization`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartMode {
    /// `PART_2Nx2N` — one PU covering the whole CU.
    Part2Nx2N,
    /// `PART_2NxN` — top / bottom horizontal split.
    Part2NxN,
    /// `PART_Nx2N` — left / right vertical split.
    PartNx2N,
    /// `PART_NxN` — four quadrants.
    PartNxN,
    /// `PART_2NxnU` — asymmetric top (¼) / bottom (¾) horizontal split.
    Part2NxnU,
    /// `PART_2NxnD` — asymmetric top (¾) / bottom (¼) horizontal split.
    Part2NxnD,
    /// `PART_nLx2N` — asymmetric left (¼) / right (¾) vertical split.
    PartNLx2N,
    /// `PART_nRx2N` — asymmetric left (¾) / right (¼) vertical split.
    PartNRx2N,
}

impl PartMode {
    /// `true` for the vertical-split modes (`PART_Nx2N`, `PART_nLx2N`,
    /// `PART_nRx2N`) — the §8.5.3.2.3 modes that exclude `A1` for the
    /// second partition.
    #[must_use]
    fn is_vertical_split(self) -> bool {
        matches!(self, Self::PartNx2N | Self::PartNLx2N | Self::PartNRx2N)
    }

    /// `true` for the horizontal-split modes (`PART_2NxN`, `PART_2NxnU`,
    /// `PART_2NxnD`) — the §8.5.3.2.3 modes that exclude `B1` for the
    /// second partition.
    #[must_use]
    fn is_horizontal_split(self) -> bool {
        matches!(self, Self::Part2NxN | Self::Part2NxnU | Self::Part2NxnD)
    }
}

/// The fully-resolved §8.5.3.2.1 output for one prediction unit: the
/// per-list utilization flag, reference index and luma motion vector.
///
/// This is the `(predFlagLX, refIdxLX, mvLX)` triple that drives both the
/// §8.5.3.3 inter sample prediction and the eqs 8-80..8-85 motion-field
/// store. A reference index of `−1` (with `pred_flag == false`) marks an
/// unused list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PuMotion {
    /// `predFlagL0`.
    pub pred_flag_l0: bool,
    /// `predFlagL1`.
    pub pred_flag_l1: bool,
    /// `refIdxL0` (`−1` when L0 unused).
    pub ref_idx_l0: i32,
    /// `refIdxL1` (`−1` when L1 unused).
    pub ref_idx_l1: i32,
    /// `mvL0` in quarter-luma-sample units.
    pub mv_l0: Mv,
    /// `mvL1` in quarter-luma-sample units.
    pub mv_l1: Mv,
}

impl PuMotion {
    /// Convert the resolved motion into the §8.5.3 [`MotionCell`] the
    /// per-block [`MotionField`] stores. `ref_poc_l0` / `ref_poc_l1`
    /// resolve each used list's reference index to a reference-picture
    /// POC (the §8.7.2.4 / §8.5.3.2.9 identity, not the list position).
    #[must_use]
    pub fn to_cell(self, ref_poc_l0: i32, ref_poc_l1: i32) -> MotionCell {
        MotionCell {
            is_intra: false,
            has_nonzero_coeff: false,
            pred_flag_l0: self.pred_flag_l0,
            pred_flag_l1: self.pred_flag_l1,
            ref_poc_l0: if self.pred_flag_l0 {
                ref_poc_l0
            } else {
                i32::MIN
            },
            ref_poc_l1: if self.pred_flag_l1 {
                ref_poc_l1
            } else {
                i32::MIN
            },
            mv_l0: if self.pred_flag_l0 {
                self.mv_l0
            } else {
                [0, 0]
            },
            mv_l1: if self.pred_flag_l1 {
                self.mv_l1
            } else {
                [0, 0]
            },
        }
    }
}

/// The reference-picture resolvers the §8.5.3.2 derivations need, plus the
/// current-picture / slice context that does not vary per PU.
///
/// All POC / long-term / short-term lookups go through closures so this
/// driver stays independent of the [`crate::dpb`] reference-list layout:
/// the picture-level driver binds them to `RefPicListX[ refIdx ]`.
pub struct PuMvContext<'a> {
    /// `PicOrderCntVal` of the current picture.
    pub curr_poc: i32,
    /// `true` for a B slice (enables L1 + the §8.5.3.2.4 combined step).
    pub slice_is_b: bool,
    /// `CtbLog2SizeY`.
    pub ctb_log2_size_y: u32,
    /// `pic_width_in_luma_samples` (the §8.5.3.2.8 bottom-right bound).
    pub pic_width_luma: u32,
    /// `pic_height_in_luma_samples`.
    pub pic_height_luma: u32,
    /// `MaxNumMergeCand` (§7.4.7.1).
    pub max_num_merge_cand: usize,
    /// `num_ref_idx_l0_active` (the active L0 size, not the minus1 form).
    pub num_ref_idx_l0_active: i32,
    /// `num_ref_idx_l1_active`.
    pub num_ref_idx_l1_active: i32,
    /// `Log2ParMrgLevel` (§7.4.3.3.1) — the parallel-merge-region size.
    pub log2_par_mrg_level: u32,
    /// `slice_temporal_mvp_enabled_flag` (the §8.5.3.2.8 gate).
    pub temporal_mvp_enabled: bool,
    /// `collocated_from_l0_flag` (the §8.5.3.2.9 collocated list selector).
    pub collocated_from_l0_flag: bool,
    /// `PicOrderCnt( ColPic )` — the collocated picture's POC (§8.5.3.2.9).
    pub col_poc: i32,
    /// `NoBackwardPredFlag` (§8.3.5) — the §8.5.3.2.9 list-selection gate.
    pub no_backward_pred: bool,
    /// Resolve `(list, ref_idx)` of the current PU's reference list to a
    /// reference-picture POC.
    pub ref_poc: &'a dyn Fn(usize, i32) -> i32,
    /// Resolve `(list, ref_idx)` to whether the picture is long-term.
    pub ref_long_term: &'a dyn Fn(usize, i32) -> bool,
    /// Resolve `(list, ref_idx)` to whether the picture is short-term.
    pub ref_short_term: &'a dyn Fn(usize, i32) -> bool,
    /// The collocated picture's per-block motion field (§8.5.3.2.9), when
    /// `slice_temporal_mvp_enabled_flag` and a `ColPic` was selected.
    pub col_field: Option<&'a MotionField>,
    /// Resolve the collocated picture's stored `refIdxCol` reference POC to
    /// whether that reference is long-term (the §8.5.3.2.9 scaling gate).
    pub col_ref_long_term: &'a dyn Fn(i32) -> bool,
}

impl std::fmt::Debug for PuMvContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PuMvContext")
            .field("curr_poc", &self.curr_poc)
            .field("slice_is_b", &self.slice_is_b)
            .field("ctb_log2_size_y", &self.ctb_log2_size_y)
            .field("max_num_merge_cand", &self.max_num_merge_cand)
            .field("log2_par_mrg_level", &self.log2_par_mrg_level)
            .field("temporal_mvp_enabled", &self.temporal_mvp_enabled)
            .field("col_poc", &self.col_poc)
            .field("no_backward_pred", &self.no_backward_pred)
            .finish_non_exhaustive()
    }
}

/// The per-PU geometry the §8.5.3.2.1 driver needs: the CU + PU locations
/// and sizes plus the `PartMode` / `partIdx` split context.
#[derive(Debug, Clone, Copy)]
pub struct PuGeometry {
    /// `(xCb, yCb)` — luma top-left of the coding block.
    pub x_cb: usize,
    /// `yCb`.
    pub y_cb: usize,
    /// `nCbS` — luma coding-block size.
    pub n_cb_s: usize,
    /// `(xPb, yPb)` — luma top-left of the prediction block.
    pub x_pb: usize,
    /// `yPb`.
    pub y_pb: usize,
    /// `nPbW` — prediction-block width.
    pub n_pb_w: usize,
    /// `nPbH` — prediction-block height.
    pub n_pb_h: usize,
    /// `PartMode` of the coding unit.
    pub part_mode: PartMode,
    /// `partIdx` of this PU within the CU.
    pub part_idx: u32,
}

/// A spatial neighbour location and whether the §6.4.2 prediction-block
/// availability test passed there. The driver reads the [`MotionField`]
/// cell at `(x, y)` when `available`.
#[derive(Debug, Clone, Copy)]
struct NeighbourLoc {
    x: i32,
    y: i32,
    available: bool,
}

/// Read the [`NeighbourPu`] motion for a spatial neighbour, mapping the
/// motion field's stored reference POCs back to the current PU's reference
/// indices (the §8.5.3.2.3 candidate copies the neighbour's `refIdxLX`,
/// which the field does not store directly).
///
/// Returns `None` when the neighbour is §6.4.2-unavailable, off-picture, or
/// covered by an intra block (intra neighbours contribute no candidate).
fn neighbour_pu(field: &MotionField, loc: NeighbourLoc, ctx: &PuMvContext) -> Option<NeighbourPu> {
    if !loc.available || loc.x < 0 || loc.y < 0 {
        return None;
    }
    let (x, y) = (loc.x as usize, loc.y as usize);
    if x >= field.width_4() * 4 || y >= field.height_4() * 4 {
        return None;
    }
    let cell = field.cell_at(x, y);
    if cell.is_intra {
        return None;
    }
    // Map the stored reference POC back to a refIdx in the current PU's
    // reference list, scanning the active entries (the merge candidate
    // carries the neighbour's refIdx, which must index *this* slice's
    // RefPicListX since both slices share the same DPB).
    let ref_idx_l0 = if cell.pred_flag_l0 {
        poc_to_ref_idx(0, cell.ref_poc_l0, ctx.num_ref_idx_l0_active, ctx.ref_poc)
    } else {
        -1
    };
    let ref_idx_l1 = if cell.pred_flag_l1 {
        poc_to_ref_idx(1, cell.ref_poc_l1, ctx.num_ref_idx_l1_active, ctx.ref_poc)
    } else {
        -1
    };
    Some(NeighbourPu {
        pred_flag_l0: cell.pred_flag_l0 && ref_idx_l0 >= 0,
        pred_flag_l1: cell.pred_flag_l1 && ref_idx_l1 >= 0,
        ref_idx_l0,
        ref_idx_l1,
        mv_l0: cell.mv_l0,
        mv_l1: cell.mv_l1,
    })
}

/// Resolve a reference POC to a reference index in list `x` by scanning
/// the active entries; `−1` when no active entry matches.
fn poc_to_ref_idx(x: usize, poc: i32, active: i32, ref_poc: &dyn Fn(usize, i32) -> i32) -> i32 {
    for r in 0..active {
        if ref_poc(x, r) == poc {
            return r;
        }
    }
    -1
}

/// The five §8.5.3.2.3 neighbour locations (A1, B1, B0, A0, B2) for a PU
/// at `(xPb, yPb)` of size `(nPbW, nPbH)`, with the §6.4.2 availability
/// closure applied.
fn spatial_neighbours(
    field: &MotionField,
    geom: &PuGeometry,
    ctx: &PuMvContext,
    available: &dyn Fn(i32, i32) -> bool,
) -> SpatialMergeNeighbours {
    let (x_pb, y_pb) = (geom.x_pb as i32, geom.y_pb as i32);
    let (w, h) = (geom.n_pb_w as i32, geom.n_pb_h as i32);
    let locs = [
        // A1 = (xPb − 1, yPb + nPbH − 1)
        NeighbourLoc {
            x: x_pb - 1,
            y: y_pb + h - 1,
            available: available(x_pb - 1, y_pb + h - 1),
        },
        // B1 = (xPb + nPbW − 1, yPb − 1)
        NeighbourLoc {
            x: x_pb + w - 1,
            y: y_pb - 1,
            available: available(x_pb + w - 1, y_pb - 1),
        },
        // B0 = (xPb + nPbW, yPb − 1)
        NeighbourLoc {
            x: x_pb + w,
            y: y_pb - 1,
            available: available(x_pb + w, y_pb - 1),
        },
        // A0 = (xPb − 1, yPb + nPbH)
        NeighbourLoc {
            x: x_pb - 1,
            y: y_pb + h,
            available: available(x_pb - 1, y_pb + h),
        },
        // B2 = (xPb − 1, yPb − 1)
        NeighbourLoc {
            x: x_pb - 1,
            y: y_pb - 1,
            available: available(x_pb - 1, y_pb - 1),
        },
    ];
    SpatialMergeNeighbours {
        a1: neighbour_pu(field, locs[0], ctx),
        b1: neighbour_pu(field, locs[1], ctx),
        b0: neighbour_pu(field, locs[2], ctx),
        a0: neighbour_pu(field, locs[3], ctx),
        b2: neighbour_pu(field, locs[4], ctx),
    }
}

/// The §8.5.3.2.3 parallel-merge-region (`Log2ParMrgLevel`) test for each
/// of the five neighbour positions — `true` when the neighbour shares the
/// current PB's parallel-merge region (`xPb >> L == xNb >> L && yPb >> L ==
/// yNb >> L`), which forces it unavailable.
fn par_mrg_tests(geom: &PuGeometry, log2_par: u32) -> (bool, bool, bool, bool, bool) {
    let (x_pb, y_pb) = (geom.x_pb as i32, geom.y_pb as i32);
    let (w, h) = (geom.n_pb_w as i32, geom.n_pb_h as i32);
    let same = |nx: i32, ny: i32| -> bool {
        (x_pb >> log2_par) == (nx >> log2_par) && (y_pb >> log2_par) == (ny >> log2_par)
    };
    (
        same(x_pb - 1, y_pb + h - 1), // A1
        same(x_pb + w - 1, y_pb - 1), // B1
        same(x_pb + w, y_pb - 1),     // B0
        same(x_pb - 1, y_pb + h),     // A0
        same(x_pb - 1, y_pb - 1),     // B2
    )
}

/// §8.5.3.2.8 — derive the temporal merge candidate `Col` for one list as a
/// [`MergeCandidate`], or `None` when `availableFlagLXCol == 0`.
fn temporal_merge_candidate(
    geom: &PuGeometry,
    ctx: &PuMvContext,
    is_b: bool,
) -> Option<MergeCandidate> {
    let col_field = ctx.col_field?;
    if !ctx.temporal_mvp_enabled {
        return None;
    }
    // refIdxLXCol = 0 (§8.5.3.2.2 step 2); the merge candidate's reference
    // pictures are RefPicListX[ 0 ].
    let l0_col = temporal_mv_for_list(col_field, geom, ctx, 0);
    let l1_col = if is_b {
        temporal_mv_for_list(col_field, geom, ctx, 1)
    } else {
        None
    };
    if l0_col.is_none() && l1_col.is_none() {
        return None;
    }
    Some(MergeCandidate {
        ref_idx_l0: if l0_col.is_some() { 0 } else { -1 },
        ref_idx_l1: if l1_col.is_some() { 0 } else { -1 },
        pred_flag_l0: l0_col.is_some(),
        pred_flag_l1: l1_col.is_some(),
        mv_l0: l0_col.unwrap_or([0, 0]),
        mv_l1: l1_col.unwrap_or([0, 0]),
    })
}

/// §8.5.3.2.8 — the temporal MV `mvLXCol` for one list with `refIdxCol == 0`.
fn temporal_mv_for_list(
    col_field: &MotionField,
    geom: &PuGeometry,
    ctx: &PuMvContext,
    x: usize,
) -> Option<Mv> {
    let curr_ref_poc = (ctx.ref_poc)(x, 0);
    let curr_ref_short = (ctx.ref_short_term)(x, 0);
    let tctx = TemporalMvContext {
        ctb_log2_size_y: ctx.ctb_log2_size_y,
        pic_width_luma: ctx.pic_width_luma,
        pic_height_luma: ctx.pic_height_luma,
        curr_poc: ctx.curr_poc,
        col_poc: ctx.col_poc,
        curr_ref_poc,
        curr_ref_long_term: !curr_ref_short,
        no_backward_pred: ctx.no_backward_pred,
        collocated_from_l0_flag: ctx.collocated_from_l0_flag,
        col_ref_long_term: ctx.col_ref_long_term,
    };
    derive_temporal_mv(
        col_field,
        geom.x_pb as u32,
        geom.y_pb as u32,
        geom.n_pb_w as u32,
        geom.n_pb_h as u32,
        &tctx,
    )
}

/// §8.5.3.2.1 — resolve one prediction unit's motion vectors, reference
/// indices and prediction-list utilization flags.
///
/// `field` is the *current* picture's motion field as built so far (the
/// spatial neighbours are read out of it); `geom` is the PU geometry;
/// `pu` is the parsed §7.3.8.6 syntax; `ctx` carries the reference-picture
/// resolvers and slice context; `available` is the §6.4.2 prediction-block
/// availability test (`(xNb, yNb) -> bool`).
///
/// Returns the fully-resolved [`PuMotion`]; the caller writes it back into
/// the motion field (eqs 8-80..8-85) with [`PuMotion::to_cell`].
#[must_use]
pub fn resolve_pu_motion(
    field: &MotionField,
    geom: &PuGeometry,
    pu: &PredictionUnit,
    ctx: &PuMvContext,
    available: &dyn Fn(i32, i32) -> bool,
) -> PuMotion {
    // §8.5.3.2.1: merge_flag selects the §8.5.3.2.2 merge branch. A skip CU
    // (merge_flag inferred 1) carries only merge_idx and no AMVP motion;
    // an explicit AMVP PU always signals at least one mvp_lX_flag.
    if pu.merge_flag || !has_explicit_mv(pu) {
        resolve_merge(field, geom, pu, ctx, available)
    } else {
        resolve_amvp(field, geom, pu, ctx, available)
    }
}

/// `true` when the PU carries explicit AMVP motion (an `mvp_lX_flag`),
/// i.e. it is not a merge / skip PU.
fn has_explicit_mv(pu: &PredictionUnit) -> bool {
    pu.mvp_l0_flag.is_some() || pu.mvp_l1_flag.is_some()
}

/// §8.5.3.2.2 — the merge-mode branch of §8.5.3.2.1.
fn resolve_merge(
    field: &MotionField,
    geom: &PuGeometry,
    pu: &PredictionUnit,
    ctx: &PuMvContext,
    available: &dyn Fn(i32, i32) -> bool,
) -> PuMotion {
    // §8.5.3.2.2 eqs 8-110..8-113: when Log2ParMrgLevel > 2 and nCbS == 8,
    // all PUs of the CU share the 2Nx2N merge list (PU expands to the CU).
    let merge_geom = if ctx.log2_par_mrg_level > 2 && geom.n_cb_s == 8 {
        PuGeometry {
            x_pb: geom.x_cb,
            y_pb: geom.y_cb,
            n_pb_w: geom.n_cb_s,
            n_pb_h: geom.n_cb_s,
            part_idx: 0,
            part_mode: PartMode::Part2Nx2N,
            ..*geom
        }
    } else {
        *geom
    };

    let part = PartitionContext {
        part_idx: merge_geom.part_idx,
        part_mode_vertical_split: merge_geom.part_mode.is_vertical_split(),
        part_mode_horizontal_split: merge_geom.part_mode.is_horizontal_split(),
    };
    let neigh = spatial_neighbours(field, &merge_geom, ctx, available);
    let par = par_mrg_tests(&merge_geom, ctx.log2_par_mrg_level);
    let spatial = derive_spatial_merge_candidates(&neigh, part, par);
    let col = temporal_merge_candidate(&merge_geom, ctx, ctx.slice_is_b);

    let zero_num_ref_idx = if ctx.slice_is_b {
        ctx.num_ref_idx_l0_active.min(ctx.num_ref_idx_l1_active)
    } else {
        ctx.num_ref_idx_l0_active
    };
    let params = MergeListParams {
        slice_is_b: ctx.slice_is_b,
        max_num_merge_cand: ctx.max_num_merge_cand,
        zero_num_ref_idx,
    };
    let merge_idx = pu.merge_idx.unwrap_or(0) as usize;
    // The §8.5.3.2.4 combined-candidate distinctness test (eq 8-143) reads
    // RefPicListX[ refIdx ] POC; the original PB size drives the eq 8-122
    // step-10 reduction.
    let pb_w_plus_h = (geom.n_pb_w + geom.n_pb_h) as u32;
    let ref_poc = ctx.ref_poc;
    let chosen = build_merge_candidate(&spatial, col, params, merge_idx, pb_w_plus_h, ref_poc);

    PuMotion {
        pred_flag_l0: chosen.pred_flag_l0,
        pred_flag_l1: chosen.pred_flag_l1,
        ref_idx_l0: chosen.ref_idx_l0,
        ref_idx_l1: chosen.ref_idx_l1,
        mv_l0: chosen.mv_l0,
        mv_l1: chosen.mv_l1,
    }
}

/// §8.5.3.2.1 steps 1–5 — the AMVP (non-merge) branch.
fn resolve_amvp(
    field: &MotionField,
    geom: &PuGeometry,
    pu: &PredictionUnit,
    ctx: &PuMvContext,
    available: &dyn Fn(i32, i32) -> bool,
) -> PuMotion {
    let idc = pu.inter_pred_idc.unwrap_or(InterPredIdc::PredL0);
    let uses_l0 = matches!(idc, InterPredIdc::PredL0 | InterPredIdc::PredBi);
    let uses_l1 = matches!(idc, InterPredIdc::PredL1 | InterPredIdc::PredBi);

    // The §8.5.3.2.7 A/B MVP derivation reads the raw §6.4.2-gated
    // neighbours; the §8.5.3.2.3 merge redundancy / partition-exclusion
    // and the parallel-merge-region test apply only to merge mode.
    let neigh = spatial_neighbours(field, geom, ctx, available);

    let mut out = PuMotion {
        ref_idx_l0: -1,
        ref_idx_l1: -1,
        ..PuMotion::default()
    };

    for (x, uses, ref_idx_field, mvd_field, mvp_flag_field) in [
        (0usize, uses_l0, pu.ref_idx_l0, pu.mvd_l0, pu.mvp_l0_flag),
        (1usize, uses_l1, pu.ref_idx_l1, pu.mvd_l1, pu.mvp_l1_flag),
    ] {
        if !uses {
            continue;
        }
        let ref_idx = ref_idx_field.unwrap_or(0) as i32;
        let cur_ref_poc = (ctx.ref_poc)(x, ref_idx);
        let cur_ref_long = (ctx.ref_long_term)(x, ref_idx);
        let mvp_ctx = MvpContext {
            x,
            curr_poc: ctx.curr_poc,
            cur_ref: RefPicId {
                poc: cur_ref_poc,
                long_term: cur_ref_long,
            },
            neigh_ref_poc: ctx.ref_poc,
            neigh_ref_long_term: ctx.ref_long_term,
            neigh_ref_short_term: ctx.ref_short_term,
        };
        let col = temporal_mvp(geom, ctx, x, ref_idx);
        let mvp_flag = mvp_flag_field.unwrap_or(0) != 0;
        let mvp = derive_mvp_candidate(&neigh, &mvp_ctx, col, mvp_flag);
        let mvd = mvd_field
            .map(|c| [c[0].value, c[1].value])
            .unwrap_or([0, 0]);
        // The integer-MV path (eqs 8-98..8-101) applies for SCC current-
        // picture-referencing or use_integer_mv_flag; the present
        // decoder state uses the fractional path (integer_mv == false).
        let mv = reconstruct_mv(mvp, mvd, false);
        match x {
            0 => {
                out.pred_flag_l0 = true;
                out.ref_idx_l0 = ref_idx;
                out.mv_l0 = mv;
            }
            _ => {
                out.pred_flag_l1 = true;
                out.ref_idx_l1 = ref_idx;
                out.mv_l1 = mv;
            }
        }
    }

    apply_bi_to_uni_reduction(&mut out, geom, ctx);
    out
}

/// §8.5.3.2.8 — the temporal MVP `mvLXCol` for the AMVP branch (one list,
/// the PU's signalled `refIdxLX`).
fn temporal_mvp(geom: &PuGeometry, ctx: &PuMvContext, x: usize, ref_idx: i32) -> Option<Mv> {
    if !ctx.temporal_mvp_enabled {
        return None;
    }
    let col_field = ctx.col_field?;
    let curr_ref_poc = (ctx.ref_poc)(x, ref_idx);
    let curr_ref_short = (ctx.ref_short_term)(x, ref_idx);
    let tctx = TemporalMvContext {
        ctb_log2_size_y: ctx.ctb_log2_size_y,
        pic_width_luma: ctx.pic_width_luma,
        pic_height_luma: ctx.pic_height_luma,
        curr_poc: ctx.curr_poc,
        col_poc: ctx.col_poc,
        curr_ref_poc,
        curr_ref_long_term: !curr_ref_short,
        no_backward_pred: ctx.no_backward_pred,
        collocated_from_l0_flag: ctx.collocated_from_l0_flag,
        col_ref_long_term: ctx.col_ref_long_term,
    };
    derive_temporal_mv(
        col_field,
        geom.x_pb as u32,
        geom.y_pb as u32,
        geom.n_pb_w as u32,
        geom.n_pb_h as u32,
        &tctx,
    )
}

/// §8.5.3.2.1 eqs 8-102/8-103 — the `nPbSw == 8 && nPbSh == 8` bi→uni-L0
/// reduction (the SCC `TwoVersionsOfCurrDecPicFlag` clause). For the
/// non-SCC main path this clause never fires (`TwoVersionsOfCurrDecPicFlag
/// == 0`), so the reduction is a no-op; the function is structured so the
/// SCC path can be enabled when current-picture referencing lands.
fn apply_bi_to_uni_reduction(out: &mut PuMotion, geom: &PuGeometry, _ctx: &PuMvContext) {
    // nPbSw / nPbSh (eqs 8-86/8-87).
    let n_pb_sw = geom.n_cb_s
        / if matches!(geom.part_mode, PartMode::Part2Nx2N | PartMode::Part2NxN) {
            1
        } else {
            2
        };
    let n_pb_sh = geom.n_cb_s
        / if matches!(geom.part_mode, PartMode::Part2Nx2N | PartMode::PartNx2N) {
            1
        } else {
            2
        };
    // The full eq 8-102/8-103 clause additionally requires
    // TwoVersionsOfCurrDecPicFlag == 1 (SCC). The non-SCC main path keeps
    // bi-prediction intact, so we gate the reduction off for now.
    let two_versions_of_curr_dec_pic = false;
    if out.pred_flag_l0
        && out.pred_flag_l1
        && n_pb_sw == 8
        && n_pb_sh == 8
        && two_versions_of_curr_dec_pic
    {
        out.pred_flag_l1 = false;
        out.ref_idx_l1 = -1;
        out.mv_l1 = [0, 0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binarization::MvdComponent;

    fn intra_field(w: usize, h: usize) -> MotionField {
        MotionField::new(w, h)
    }

    fn inter_cell_l0(ref_poc: i32, mv: Mv) -> MotionCell {
        MotionCell {
            is_intra: false,
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_poc_l0: ref_poc,
            ref_poc_l1: i32::MIN,
            mv_l0: mv,
            ..MotionCell::default()
        }
    }

    fn base_ctx<'a>(
        ref_poc: &'a dyn Fn(usize, i32) -> i32,
        long: &'a dyn Fn(usize, i32) -> bool,
        short: &'a dyn Fn(usize, i32) -> bool,
        col_long: &'a dyn Fn(i32) -> bool,
    ) -> PuMvContext<'a> {
        PuMvContext {
            curr_poc: 4,
            slice_is_b: false,
            ctb_log2_size_y: 6,
            pic_width_luma: 64,
            pic_height_luma: 64,
            max_num_merge_cand: 5,
            num_ref_idx_l0_active: 1,
            num_ref_idx_l1_active: 1,
            log2_par_mrg_level: 2,
            temporal_mvp_enabled: false,
            collocated_from_l0_flag: true,
            col_poc: 0,
            no_backward_pred: true,
            ref_poc,
            ref_long_term: long,
            ref_short_term: short,
            col_field: None,
            col_ref_long_term: col_long,
        }
    }

    /// A merge / skip PU with no explicit AMVP motion.
    fn merge_pu(merge_idx: u8) -> PredictionUnit {
        PredictionUnit {
            merge_flag: true,
            merge_idx: Some(merge_idx),
            inter_pred_idc: None,
            ref_idx_l0: None,
            mvd_l0: None,
            mvp_l0_flag: None,
            ref_idx_l1: None,
            mvd_l1: None,
            mvp_l1_flag: None,
        }
    }

    fn geom_2nx2n(x: usize, y: usize, n: usize) -> PuGeometry {
        PuGeometry {
            x_cb: x,
            y_cb: y,
            n_cb_s: n,
            x_pb: x,
            y_pb: y,
            n_pb_w: n,
            n_pb_h: n,
            part_mode: PartMode::Part2Nx2N,
            part_idx: 0,
        }
    }

    #[test]
    fn merge_picks_left_neighbour_motion() {
        // Left neighbour A1 carries an L0 MV referencing POC 0; the merge
        // candidate at merge_idx 0 copies it verbatim.
        let mut field = intra_field(32, 32);
        field.fill_rect(0, 0, 16, 32, inter_cell_l0(0, [12, -4]));
        let ref_poc = |_l: usize, r: i32| if r == 0 { 0 } else { -999 };
        let long = |_l: usize, _r: i32| false;
        let short = |_l: usize, _r: i32| true;
        let col_long = |_p: i32| false;
        let ctx = base_ctx(&ref_poc, &long, &short, &col_long);
        let geom = geom_2nx2n(16, 0, 16);
        let pu = merge_pu(0);
        // A1 at (15, 15) is available; everything else unavailable.
        let avail = |x: i32, y: i32| x < 16 && y < 32 && x >= 0 && y >= 0;
        let out = resolve_pu_motion(&field, &geom, &pu, &ctx, &avail);
        assert!(out.pred_flag_l0);
        assert!(!out.pred_flag_l1);
        assert_eq!(out.ref_idx_l0, 0);
        assert_eq!(out.mv_l0, [12, -4]);
    }

    #[test]
    fn merge_all_intra_neighbours_pads_zero() {
        // No inter neighbours ⇒ the list is filled with zero candidates;
        // merge_idx 0 selects a zero-MV uni-L0 candidate (P slice).
        let field = intra_field(32, 32);
        let ref_poc = |_l: usize, r: i32| if r == 0 { 0 } else { -999 };
        let long = |_l: usize, _r: i32| false;
        let short = |_l: usize, _r: i32| true;
        let col_long = |_p: i32| false;
        let ctx = base_ctx(&ref_poc, &long, &short, &col_long);
        let geom = geom_2nx2n(0, 0, 16);
        let pu = merge_pu(0);
        let avail = |_x: i32, _y: i32| false;
        let out = resolve_pu_motion(&field, &geom, &pu, &ctx, &avail);
        assert!(out.pred_flag_l0);
        assert!(!out.pred_flag_l1);
        assert_eq!(out.mv_l0, [0, 0]);
        assert_eq!(out.ref_idx_l0, 0);
    }

    #[test]
    fn amvp_adds_mvd_to_predictor() {
        // AMVP L0: a left neighbour A1 provides the MVP; the signalled mvd
        // is added (eqs 8-94..8-97). With one neighbour and no scaling the
        // predictor is the neighbour MV.
        let mut field = intra_field(32, 32);
        field.fill_rect(0, 0, 16, 32, inter_cell_l0(0, [8, 0]));
        let ref_poc = |_l: usize, r: i32| if r == 0 { 0 } else { -999 };
        let long = |_l: usize, _r: i32| false;
        let short = |_l: usize, _r: i32| true;
        let col_long = |_p: i32| false;
        let ctx = base_ctx(&ref_poc, &long, &short, &col_long);
        let geom = geom_2nx2n(16, 0, 16);
        let mvd = |v: i32| MvdComponent {
            greater0_flag: u8::from(v != 0),
            greater1_flag: None,
            minus2: None,
            sign_flag: None,
            value: v,
        };
        let pu = PredictionUnit {
            merge_flag: false,
            merge_idx: None,
            inter_pred_idc: Some(InterPredIdc::PredL0),
            ref_idx_l0: Some(0),
            mvd_l0: Some([mvd(3), mvd(2)]),
            mvp_l0_flag: Some(0),
            ref_idx_l1: None,
            mvd_l1: None,
            mvp_l1_flag: None,
        };
        let avail = |x: i32, y: i32| x < 16 && y < 32 && x >= 0 && y >= 0;
        let out = resolve_pu_motion(&field, &geom, &pu, &ctx, &avail);
        assert!(out.pred_flag_l0);
        // predictor [8,0] + mvd [3,2] = [11, 2].
        assert_eq!(out.mv_l0, [11, 2]);
        assert_eq!(out.ref_idx_l0, 0);
    }

    #[test]
    fn to_cell_round_trips_used_lists() {
        let m = PuMotion {
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_idx_l0: 0,
            ref_idx_l1: -1,
            mv_l0: [4, -8],
            mv_l1: [0, 0],
        };
        let cell = m.to_cell(2, 0);
        assert!(!cell.is_intra);
        assert!(cell.pred_flag_l0);
        assert!(!cell.pred_flag_l1);
        assert_eq!(cell.ref_poc_l0, 2);
        assert_eq!(cell.ref_poc_l1, i32::MIN);
        assert_eq!(cell.mv_l0, [4, -8]);
    }
}
