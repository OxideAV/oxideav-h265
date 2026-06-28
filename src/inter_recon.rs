//! §8.5 picture-level inter reconstruction driver.
//!
//! This module is the rung between the §7.3.8 slice-data parse tree (the
//! decoded [`crate::slice_data::CodingTreeUnit`] structures, with their
//! inter coding units carrying §7.3.8.6 prediction units) and the per-PU
//! §8.5.3.3 motion-compensated prediction + §8.6.2 residual reconstruction
//! already implemented in [`crate::inter_pred`] and [`crate::recon`]. It
//! walks a picture's decoded CTUs in tile-scan (decode) order and, for each
//! inter coding unit:
//!
//! 1. §8.5.3.2.1 — resolve every prediction unit's motion vectors,
//!    reference indices and `predFlagLX` from the parsed syntax, gathering
//!    the spatial merge / MVP neighbours from the *current* picture's
//!    motion field (built up by the earlier CUs) and the temporal `Col`
//!    candidate from the collocated picture's motion field
//!    ([`crate::pu_mv::resolve_cu_motion`]).
//! 2. §8.5.3.3 — for each PU, build the §8.5.3.3.2 reference planes from the
//!    resolved `RefPicListX[ refIdxLX ]` pictures, interpolate + combine,
//!    add the §8.6.2 residual sliced from the CU residual planes
//!    ([`crate::recon::extract_cu_residual`]) and clip into the target
//!    picture ([`crate::recon::reconstruct_inter_pu`]).
//!
//! Intra coding units inside a P / B slice are reconstructed by the §8.4
//! intra path; the motion field records them as intra so a later inter CU's
//! §6.4.2 prediction-block availability denies them as motion neighbours.

use crate::dpb::{DpbEntry, RefPicLists};
use crate::motion::{derive_chroma_mv, MotionField};
use crate::picture::{sub_wh_c, Picture};
use crate::pu_mv::{resolve_cu_motion, InterCuDesc, PuMotion, PuMvContext, PuRect};
use crate::recon::{
    extract_cu_residual, reconstruct_inter_pu, CuResidual, ReconError, ReconParams, ResolvedList,
};
use crate::slice_data::{CodingUnit, PredictionUnit};

/// The resolved reference-picture access an inter CU reconstruction needs:
/// `RefPicListX[ refIdx ]` → a borrowed reference [`Picture`] and its POC.
///
/// The picture-level driver binds this to the §8.3.4 [`RefPicLists`] + the
/// [`crate::dpb::Dpb`] entries; the per-CU reconstruction reads it through
/// the [`PuMvContext`] resolvers (for the candidate derivation) and to
/// fetch each used list's reference planes (for the interpolation).
#[derive(Debug)]
pub struct RefListAccess<'a> {
    /// `RefPicList0` / `RefPicList1` as DPB entry indices.
    pub lists: &'a RefPicLists,
    /// The DPB entries the indices point into.
    pub entries: &'a [DpbEntry],
}

impl<'a> RefListAccess<'a> {
    /// Borrow `RefPicListX[ ref_idx ]`'s reconstructed picture, or `None`
    /// when the list slot is "no reference picture" / out of range.
    #[must_use]
    pub fn ref_pic(&self, list: usize, ref_idx: i32) -> Option<&'a Picture> {
        self.entry(list, ref_idx).map(|e| &e.picture)
    }

    /// `PicOrderCnt( RefPicListX[ ref_idx ] )`, or `i32::MIN` when absent.
    #[must_use]
    pub fn ref_poc(&self, list: usize, ref_idx: i32) -> i32 {
        self.entry(list, ref_idx).map_or(i32::MIN, |e| e.poc)
    }

    fn entry(&self, list: usize, ref_idx: i32) -> Option<&'a DpbEntry> {
        if ref_idx < 0 {
            return None;
        }
        let slot = if list == 0 {
            self.lists.list0.get(ref_idx as usize)
        } else {
            self.lists.list1.as_ref()?.get(ref_idx as usize)
        };
        let idx = (*slot?)?;
        self.entries.get(idx)
    }
}

/// §8.5 — reconstruct one inter coding unit into `pic`.
///
/// `motions` are the §8.5.3.2.1-resolved per-PU motions (same order as the
/// CU's §7.3.8.6 prediction units), already written into the picture's
/// motion field by [`resolve_cu_motion`]. `residual` is the CU's
/// per-component residual planes ([`extract_cu_residual`]). `refs` resolves
/// each list's reference picture. Each PU's covering residual is sliced from
/// the CU residual planes and added onto the motion-compensated prediction.
///
/// # Errors
/// Propagates [`ReconError`] from the §8.5.3.3 interpolation / combine.
pub fn reconstruct_inter_cu(
    pic: &mut Picture,
    params: &ReconParams,
    cu: &CodingUnit,
    rects: &[PuRect],
    motions: &[PuMotion],
    residual: &CuResidual,
    refs: &RefListAccess,
) -> Result<(), ReconError> {
    let cat = params.chroma_array_type;
    let (sub_w, sub_h) = if cat != 0 { sub_wh_c(cat) } else { (1, 1) };

    for (rect, motion) in rects.iter().zip(motions.iter()) {
        let l0 = resolve_list(
            params,
            refs,
            0,
            motion.pred_flag_l0,
            motion.ref_idx_l0,
            motion.mv_l0,
        )?;
        let l1 = resolve_list(
            params,
            refs,
            1,
            motion.pred_flag_l1,
            motion.ref_idx_l1,
            motion.mv_l1,
        )?;

        // Slice the PU's covering residual out of the CU residual planes.
        let res_luma = residual
            .luma
            .slice_region(rect.x_pb, rect.y_pb, rect.n_pb_w, rect.n_pb_h);
        let (res_cb, res_cr) = if cat != 0 {
            let (cx, cy) = (rect.x_pb / sub_w, rect.y_pb / sub_h);
            let (cw, ch) = (rect.n_pb_w / sub_w, rect.n_pb_h / sub_h);
            (
                residual.cb.as_ref().map(|p| p.slice_region(cx, cy, cw, ch)),
                residual.cr.as_ref().map(|p| p.slice_region(cx, cy, cw, ch)),
            )
        } else {
            (None, None)
        };

        reconstruct_inter_pu(
            pic,
            params,
            rect.x_pb,
            rect.y_pb,
            rect.n_pb_w,
            rect.n_pb_h,
            l0,
            l1,
            Some(res_luma.as_slice()),
            res_cb.as_deref(),
            res_cr.as_deref(),
        )?;
    }
    let _ = cu;
    Ok(())
}

/// Build a [`ResolvedList`] for one reference list, deriving the §8.5.3.2.10
/// chroma MV and fetching `RefPicListX[ refIdx ]`. An unused list (or a
/// list whose reference resolves to "no reference picture") becomes a
/// `pred_flag == false` entry pointing at a fallback picture (never read).
fn resolve_list<'a>(
    params: &ReconParams,
    refs: &RefListAccess<'a>,
    list: usize,
    pred_flag: bool,
    ref_idx: i32,
    mv_l: [i32; 2],
) -> Result<ResolvedList<'a>, ReconError> {
    if pred_flag {
        if let Some(ref_pic) = refs.ref_pic(list, ref_idx) {
            let (sw, sh) = if params.chroma_array_type != 0 {
                sub_wh_c(params.chroma_array_type)
            } else {
                (1, 1)
            };
            let mv_c = derive_chroma_mv(mv_l, sw as i32, sh as i32);
            return Ok(ResolvedList {
                pred_flag: true,
                mv_l,
                mv_c,
                ref_pic,
            });
        }
        // A used list with an unresolvable reference is a malformed stream;
        // surface it rather than silently dropping the prediction.
        return Err(ReconError::InterNotSupported);
    }
    // Unused list: point at any available picture (the prediction skips it).
    let fallback = refs
        .ref_pic(0, 0)
        .or_else(|| refs.ref_pic(1, 0))
        .ok_or(ReconError::InterNotSupported)?;
    Ok(ResolvedList {
        pred_flag: false,
        mv_l: [0, 0],
        mv_c: [0, 0],
        ref_pic: fallback,
    })
}

/// §8.5.3.2.1 — resolve one inter CU's per-PU motion and write it into
/// `field`, then reconstruct its samples into `pic`.
///
/// This composes [`resolve_cu_motion`] (the candidate derivation reading the
/// in-progress `field`) with [`extract_cu_residual`] + [`reconstruct_inter_cu`].
/// `pus` are the parsed §7.3.8.6 prediction units; `ctx` carries the
/// reference-picture resolvers + slice context; `available` is the §6.4.2
/// prediction-block availability test.
///
/// # Errors
/// Propagates [`ReconError`] from the residual extraction / reconstruction.
#[allow(clippy::too_many_arguments)]
pub fn resolve_and_reconstruct_inter_cu(
    pic: &mut Picture,
    field: &mut MotionField,
    params: &ReconParams,
    cu: &CodingUnit,
    pus: &[PredictionUnit],
    ctx: &PuMvContext,
    available: &dyn Fn(i32, i32) -> bool,
    refs: &RefListAccess,
) -> Result<(), ReconError> {
    let n_cb_s = 1usize << cu.log2_cb_size;
    let desc = InterCuDesc {
        x0: cu.x0 as usize,
        y0: cu.y0 as usize,
        n_cb_s,
        part_mode: cu.part_mode.into(),
    };
    let motions = resolve_cu_motion(field, desc, pus, ctx, available);
    let rects =
        crate::pu_mv::pu_partitions(cu.x0 as usize, cu.y0 as usize, n_cb_s, cu.part_mode.into());

    // §7.4.9.14 CuQpDeltaVal carried by the CU (0 for the single-QG case).
    let cu_qp_delta_val = cu
        .transform_tree
        .as_ref()
        .and_then(first_leaf_cu_qp_delta)
        .unwrap_or(0);
    let residual = extract_cu_residual(
        params,
        cu.transform_tree.as_ref(),
        cu.x0 as usize,
        cu.y0 as usize,
        n_cb_s,
        cu_qp_delta_val,
        cu.cu_transquant_bypass_flag,
    )?;

    reconstruct_inter_cu(pic, params, cu, &rects, &motions, &residual, refs)
}

/// Find the first transform-unit `cu_qp_delta` in a transform tree (the
/// single-quantization-group case threads exactly one delta per CU).
fn first_leaf_cu_qp_delta(tree: &crate::transform_tree::TransformTree) -> Option<i32> {
    match tree {
        crate::transform_tree::TransformTree::Leaf { unit, .. } => {
            unit.cu_qp_delta.as_ref().map(|d| d.value)
        }
        crate::transform_tree::TransformTree::Split { children, .. } => {
            children.iter().find_map(first_leaf_cu_qp_delta)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binarization::{CuPredMode, PartMode};
    use crate::dpb::{Marking, RefPicLists};
    use crate::motion::MotionField;
    use crate::picture::{Picture, Plane};
    use crate::pu_mv::PuMvContext;
    use crate::residual::ResidualBlock;
    use crate::slice_data::{CodingUnit, PredictionUnit};
    use crate::transform_tree::TransformTree;
    use crate::transform_unit::TransformUnit;

    fn p_params() -> ReconParams {
        ReconParams {
            chroma_array_type: 1,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            intra_smoothing_disabled: false,
            strong_intra_smoothing_enabled: true,
            slice_qp_y: 25,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            transform_skip_rotation_enabled: false,
            extended_precision: false,
        }
    }

    /// A 32×32 reference picture with a flat luma + chroma value.
    fn flat_ref(luma: i32, chroma: i32) -> Picture {
        let mut p = Picture::new(32, 32, 1, 8, 8);
        for y in 0..32 {
            for x in 0..32 {
                p.set_sample(Plane::Luma, x, y, luma);
            }
        }
        for y in 0..16 {
            for x in 0..16 {
                p.set_sample(Plane::Cb, x, y, chroma);
                p.set_sample(Plane::Cr, x, y, chroma);
            }
        }
        p
    }

    fn dpb_entry(poc: i32, pic: Picture) -> DpbEntry {
        DpbEntry {
            poc,
            layer_id: 0,
            marking: Marking::ShortTerm,
            picture: pic,
            motion: MotionField::new(32, 32),
        }
    }

    /// A merge-mode P prediction unit selecting merge candidate `idx`.
    fn merge_pu(idx: u8) -> PredictionUnit {
        PredictionUnit {
            merge_flag: true,
            merge_idx: Some(idx),
            inter_pred_idc: None,
            ref_idx_l0: None,
            mvd_l0: None,
            mvp_l0_flag: None,
            ref_idx_l1: None,
            mvd_l1: None,
            mvp_l1_flag: None,
        }
    }

    /// A 16×16 inter CU at (0,0), PART_2Nx2N, with one prediction unit and
    /// an optional flat luma DC residual.
    fn inter_cu_16(pu: PredictionUnit, luma_dc: Option<i32>) -> CodingUnit {
        let tree = luma_dc.map(|dc| {
            let mut levels = vec![0i32; 16 * 16];
            levels[0] = dc;
            TransformTree::Leaf {
                cbf_luma: true,
                unit: TransformUnit {
                    residual_luma: Some(ResidualBlock {
                        log2_trafo_size: 4,
                        last_sig_coeff_x: 0,
                        last_sig_coeff_y: 0,
                        levels,
                    }),
                    ..Default::default()
                },
            }
        });
        CodingUnit {
            x0: 0,
            y0: 0,
            log2_cb_size: 4,
            cu_pred_mode: CuPredMode::Inter,
            cu_transquant_bypass_flag: false,
            part_mode: PartMode::Part2Nx2N,
            pcm_flag: false,
            prediction_units: vec![pu],
            intra_luma: vec![],
            intra_chroma_pred_mode: vec![],
            rqt_root_cbf: luma_dc.is_some(),
            transform_tree: tree,
        }
    }

    /// Build a single-reference P-slice context: one short-term reference at
    /// POC 0, current picture at POC 4, temporal MVP disabled.
    fn p_ctx<'a>(
        ref_poc: &'a dyn Fn(usize, i32) -> i32,
        long: &'a dyn Fn(usize, i32) -> bool,
        short: &'a dyn Fn(usize, i32) -> bool,
        col_long: &'a dyn Fn(i32) -> bool,
    ) -> PuMvContext<'a> {
        PuMvContext {
            curr_poc: 4,
            slice_is_b: false,
            ctb_log2_size_y: 4,
            pic_width_luma: 32,
            pic_height_luma: 32,
            max_num_merge_cand: 5,
            num_ref_idx_l0_active: 1,
            num_ref_idx_l1_active: 0,
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

    /// A merge PU with no spatial / temporal neighbours falls through to the
    /// §8.5.3.2.5 zero-MV candidate (mvL0 = 0, refIdxL0 = 0), so the PU
    /// reconstructs to the reference picture's co-located samples.
    #[test]
    fn merge_zero_mv_p_cu_reconstructs_reference() {
        let params = p_params();
        let refpic = flat_ref(100, 120);
        let entries = vec![dpb_entry(0, refpic)];
        let lists = RefPicLists {
            list0: vec![Some(0)],
            list1: None,
        };
        let refs = RefListAccess {
            lists: &lists,
            entries: &entries,
        };

        let ref_poc = |_l: usize, _r: i32| 0i32;
        let long = |_l: usize, _r: i32| false;
        let short = |_l: usize, _r: i32| true;
        let col_long = |_p: i32| false;
        let ctx = p_ctx(&ref_poc, &long, &short, &col_long);

        let cu = inter_cu_16(merge_pu(0), None);
        let mut field = MotionField::new(32, 32);
        let mut pic = Picture::new(32, 32, 1, 8, 8);
        // No neighbours available (single isolated CU at picture origin).
        let available = |_x: i32, _y: i32| false;
        resolve_and_reconstruct_inter_cu(
            &mut pic,
            &mut field,
            &params,
            &cu,
            &cu.prediction_units,
            &ctx,
            &available,
            &refs,
        )
        .unwrap();

        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(pic.sample(Plane::Luma, x, y), 100, "luma ({x},{y})");
            }
        }
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(pic.sample(Plane::Cb, x, y), 120, "cb ({x},{y})");
                assert_eq!(pic.sample(Plane::Cr, x, y), 120, "cr ({x},{y})");
            }
        }
        // eqs 8-80..8-85: the CU's motion is written into the field.
        let cell = field.cell_at(0, 0);
        assert!(!cell.is_intra);
        assert!(cell.pred_flag_l0 && !cell.pred_flag_l1);
        assert_eq!(cell.mv_l0, [0, 0]);
    }

    /// The CU residual is dequantized + inverse-transformed and added onto
    /// the motion-compensated prediction.
    #[test]
    fn merge_zero_mv_p_cu_adds_residual() {
        let params = p_params();
        let refpic = flat_ref(100, 120);
        let entries = vec![dpb_entry(0, refpic)];
        let lists = RefPicLists {
            list0: vec![Some(0)],
            list1: None,
        };
        let refs = RefListAccess {
            lists: &lists,
            entries: &entries,
        };
        let ref_poc = |_l: usize, _r: i32| 0i32;
        let long = |_l: usize, _r: i32| false;
        let short = |_l: usize, _r: i32| true;
        let col_long = |_p: i32| false;
        let ctx = p_ctx(&ref_poc, &long, &short, &col_long);

        // A DC luma residual produces a uniform offset over the 16×16 block.
        let cu = inter_cu_16(merge_pu(0), Some(40));
        let mut field = MotionField::new(32, 32);
        let mut pic = Picture::new(32, 32, 1, 8, 8);
        let available = |_x: i32, _y: i32| false;
        resolve_and_reconstruct_inter_cu(
            &mut pic,
            &mut field,
            &params,
            &cu,
            &cu.prediction_units,
            &ctx,
            &available,
            &refs,
        )
        .unwrap();

        // Uniform DC residual ⇒ every luma sample is prediction(100) + r for
        // the same r; assert the block is uniform and offset from 100.
        let v = pic.sample(Plane::Luma, 0, 0);
        assert_ne!(v, 100, "residual shifts the prediction");
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(pic.sample(Plane::Luma, x, y), v, "uniform at ({x},{y})");
            }
        }
    }
}
