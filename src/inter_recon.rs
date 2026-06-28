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

    /// Borrow `RefPicListX[ ref_idx ]`'s DPB entry, or `None` when the
    /// list slot is "no reference picture" / out of range.
    #[must_use]
    pub fn entry(&self, list: usize, ref_idx: i32) -> Option<&'a DpbEntry> {
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

    // §8.7.2.4 — mark the per-4×4 cells covered by a transform block with a
    // coded luma coefficient so the deblocking boundary-strength `cbf` test
    // (bS = 1 at a transform-block edge with a non-zero coefficient) reads
    // them.
    if let Some(tree) = cu.transform_tree.as_ref() {
        mark_nonzero_luma(field, tree, cu.x0 as usize, cu.y0 as usize, cu.log2_cb_size);
    }

    reconstruct_inter_cu(pic, params, cu, &rects, &motions, &residual, refs)
}

/// Walk an inter CU's transform tree and mark each leaf transform block that
/// carries a non-zero luma coefficient into the motion field
/// ([`MotionField::mark_nonzero_coeff`]).
fn mark_nonzero_luma(
    field: &mut MotionField,
    tree: &crate::transform_tree::TransformTree,
    x0: usize,
    y0: usize,
    log2_trafo_size: u32,
) {
    use crate::transform_tree::TransformTree;
    let n = 1usize << log2_trafo_size;
    match tree {
        TransformTree::Leaf { cbf_luma, unit } => {
            let has_coeff = *cbf_luma
                && unit
                    .residual_luma
                    .as_ref()
                    .is_some_and(|rb| rb.levels.iter().any(|&l| l != 0));
            if has_coeff {
                field.mark_nonzero_coeff(x0, y0, n, n);
            }
        }
        TransformTree::Split { children, .. } => {
            let half = n / 2;
            let offsets = [(0, 0), (half, 0), (0, half), (half, half)];
            for (child, (dx, dy)) in children.iter().zip(offsets) {
                mark_nonzero_luma(field, child, x0 + dx, y0 + dy, log2_trafo_size - 1);
            }
        }
    }
}

/// Slice-level inputs constant across one P / B slice's inter
/// reconstruction — every field a §8.5.3.2 / §8.3 input that does not vary
/// per coding unit. The picture-level driver
/// ([`reconstruct_inter_picture`]) binds the [`PuMvContext`] reference
/// resolvers to the [`RefListAccess`] + collocated field internally.
#[derive(Debug, Clone, Copy)]
pub struct InterSliceContext {
    /// `PicOrderCntVal` of the current picture.
    pub curr_poc: i32,
    /// `true` for a B slice (enables L1 + the §8.5.3.2.4 combined step).
    pub slice_is_b: bool,
    /// `CtbLog2SizeY`.
    pub ctb_log2_size_y: u32,
    /// `pic_width_in_luma_samples`.
    pub pic_width_luma: u32,
    /// `pic_height_in_luma_samples`.
    pub pic_height_luma: u32,
    /// `MaxNumMergeCand` (§7.4.7.1).
    pub max_num_merge_cand: usize,
    /// `num_ref_idx_l0_active`.
    pub num_ref_idx_l0_active: i32,
    /// `num_ref_idx_l1_active`.
    pub num_ref_idx_l1_active: i32,
    /// `Log2ParMrgLevel` (§7.4.3.3.1).
    pub log2_par_mrg_level: u32,
    /// `slice_temporal_mvp_enabled_flag`.
    pub temporal_mvp_enabled: bool,
    /// `collocated_from_l0_flag`.
    pub collocated_from_l0_flag: bool,
    /// `PicOrderCnt( ColPic )` (§8.5.3.2.9).
    pub col_poc: i32,
    /// `NoBackwardPredFlag` (§8.3.5).
    pub no_backward_pred: bool,
    /// `MinTbLog2SizeY` (the transform-block grid base).
    pub min_tb_log2_size_y: u32,
    /// `slice_deblocking_filter_disabled_flag == 0` — run the §8.7.2
    /// in-loop deblocking pass after reconstruction.
    pub deblock_enabled: bool,
    /// `slice_beta_offset_div2` (§8.7.2.5.3).
    pub beta_offset_div2: i32,
    /// `slice_tc_offset_div2` (§8.7.2.5.3).
    pub tc_offset_div2: i32,
    /// `SliceQpY` — the per-CU QP for the single-quantization-group case
    /// (the deblocking β/tC derivation reads it).
    pub slice_qp_y: i32,
    /// `pps_cb_qp_offset + slice_cb_qp_offset`.
    pub cb_qp_offset: i32,
    /// `pps_cr_qp_offset + slice_cr_qp_offset`.
    pub cr_qp_offset: i32,
    /// `slice_sao_luma_flag` (§8.7.3.1 luma gate).
    pub slice_sao_luma_flag: bool,
    /// `slice_sao_chroma_flag` (§8.7.3.1 chroma gate).
    pub slice_sao_chroma_flag: bool,
    /// `log2_sao_offset_scale_luma` (§7.4.3.3.2; 0 for 8-bit Main).
    pub log2_sao_offset_scale_luma: u8,
    /// `log2_sao_offset_scale_chroma`.
    pub log2_sao_offset_scale_chroma: u8,
}

/// One placed coding tree unit for the §8.5 picture-level inter driver.
#[derive(Debug)]
pub struct PlacedInterCtu<'a> {
    /// CTB luma top-left x.
    pub x_ctb: u32,
    /// CTB luma top-left y.
    pub y_ctb: u32,
    /// `SliceAddrRs` of the independent slice segment owning this CTB.
    pub slice_addr_rs: u32,
    /// The decoded coding tree unit.
    pub ctu: &'a crate::slice_data::CodingTreeUnit,
}

/// §8.5 — reconstruct a full P / B picture from its decoded CTUs.
///
/// Walks the placed CTUs in decode order, dispatching each leaf coding unit:
/// an intra CU goes through the §8.4 intra path
/// ([`crate::recon::reconstruct_intra_cu_ctx`]), an inter CU through
/// [`resolve_and_reconstruct_inter_cu`] (§8.5.3.2 candidate derivation from
/// the in-progress motion field + the collocated `col_field`, then §8.5.3.3
/// motion-compensated reconstruction). The §6.4.2 prediction-block
/// availability the candidate derivation needs is evaluated against the
/// shared [`crate::recon::ReconCtx`] tiling + the per-cell intra / inter
/// flag of the motion field built up so far.
///
/// Returns the reconstructed picture and its per-PU motion field (the
/// §8.5.3.2.9 collocated arrays a later picture's temporal MVP reads). The
/// returned picture is the full in-loop-filtered output: when
/// `slice.deblock_enabled` the §8.7.2 deblocking pass runs first, then the
/// §8.7.3 SAO pass (a no-op when both slice SAO flags are clear).
///
/// # Errors
/// Propagates [`ReconError`] from the per-CU reconstruction.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_inter_picture(
    pic_width_luma: usize,
    pic_height_luma: usize,
    params: &ReconParams,
    slice: &InterSliceContext,
    tiles: &crate::availability::TilingParams,
    ctus: &[PlacedInterCtu<'_>],
    refs: &RefListAccess,
    col_field: Option<&MotionField>,
) -> Result<(Picture, MotionField), ReconError> {
    let mut pic = Picture::new(
        pic_width_luma,
        pic_height_luma,
        params.chroma_array_type,
        params.bit_depth_luma,
        params.bit_depth_chroma,
    );
    let mut ctx = crate::recon::ReconCtx::new(
        pic_width_luma,
        pic_height_luma,
        slice.ctb_log2_size_y,
        slice.min_tb_log2_size_y,
        tiles,
    )?;
    let mut field = MotionField::new(pic_width_luma, pic_height_luma);

    let ctb_size = 1usize << slice.ctb_log2_size_y;
    let pic_w_ctbs = pic_width_luma.div_ceil(ctb_size);
    let pic_h_ctbs = pic_height_luma.div_ceil(ctb_size);
    let mut slice_addr_map = vec![0u32; pic_w_ctbs * pic_h_ctbs];
    for placed in ctus {
        let rx = (placed.x_ctb as usize) >> slice.ctb_log2_size_y;
        let ry = (placed.y_ctb as usize) >> slice.ctb_log2_size_y;
        slice_addr_map[ry * pic_w_ctbs + rx] = placed.slice_addr_rs;
    }
    ctx.set_slice_addr_rs(slice_addr_map.clone());

    // §8.5.3.2 reference-picture resolvers, bound to the §8.3.4 ref lists.
    let ref_poc = |list: usize, ref_idx: i32| refs.ref_poc(list, ref_idx);
    let ref_long_term = |list: usize, ref_idx: i32| {
        refs.entry(list, ref_idx)
            .is_some_and(|e| e.marking == crate::dpb::Marking::LongTerm)
    };
    let ref_short_term = |list: usize, ref_idx: i32| {
        refs.entry(list, ref_idx)
            .is_some_and(|e| e.marking == crate::dpb::Marking::ShortTerm)
    };
    let col_ref_long_term = |_poc: i32| false;

    let mv_ctx = PuMvContext {
        curr_poc: slice.curr_poc,
        slice_is_b: slice.slice_is_b,
        ctb_log2_size_y: slice.ctb_log2_size_y,
        pic_width_luma: slice.pic_width_luma,
        pic_height_luma: slice.pic_height_luma,
        max_num_merge_cand: slice.max_num_merge_cand,
        num_ref_idx_l0_active: slice.num_ref_idx_l0_active,
        num_ref_idx_l1_active: slice.num_ref_idx_l1_active,
        log2_par_mrg_level: slice.log2_par_mrg_level,
        temporal_mvp_enabled: slice.temporal_mvp_enabled,
        collocated_from_l0_flag: slice.collocated_from_l0_flag,
        col_poc: slice.col_poc,
        no_backward_pred: slice.no_backward_pred,
        ref_poc: &ref_poc,
        ref_long_term: &ref_long_term,
        ref_short_term: &ref_short_term,
        col_field,
        col_ref_long_term: &col_ref_long_term,
    };

    let mut deblock_cus: Vec<crate::deblock::DeblockCuDesc> = Vec::new();
    for placed in ctus {
        reconstruct_inter_quadtree(
            &mut pic,
            &mut ctx,
            &mut field,
            params,
            &mv_ctx,
            refs,
            slice,
            &mut deblock_cus,
            &placed.ctu.quadtree,
        )?;
    }

    // §8.7.2 — in-loop deblocking (all vertical edges, then horizontal),
    // ahead of the §8.7.3 SAO pass.
    if slice.deblock_enabled {
        crate::deblock::deblock_picture(&mut pic, &field, &deblock_cus);
    }

    // §8.7.3 — sample-adaptive offset (on the deblocked samples). Resolve
    // each CTB's §7.4.9.3 SAO parameters with left / above merge (denied
    // across slice boundaries), then run the picture-level filter.
    let mut sao_grid = vec![crate::sao::ResolvedSao::off(); pic_w_ctbs * pic_h_ctbs];
    for placed in ctus {
        let rx = (placed.x_ctb as usize) >> slice.ctb_log2_size_y;
        let ry = (placed.y_ctb as usize) >> slice.ctb_log2_size_y;
        let here = slice_addr_map[ry * pic_w_ctbs + rx];
        if let Some(sao_params) = &placed.ctu.sao {
            let left = (rx > 0 && slice_addr_map[ry * pic_w_ctbs + (rx - 1)] == here)
                .then(|| sao_grid[ry * pic_w_ctbs + (rx - 1)]);
            let above = (ry > 0 && slice_addr_map[(ry - 1) * pic_w_ctbs + rx] == here)
                .then(|| sao_grid[(ry - 1) * pic_w_ctbs + rx]);
            sao_grid[ry * pic_w_ctbs + rx] = crate::sao::ResolvedSao::resolve(
                sao_params,
                left.as_ref(),
                above.as_ref(),
                slice.log2_sao_offset_scale_luma,
                slice.log2_sao_offset_scale_chroma,
            );
        }
    }
    let filtered = crate::sao::apply_sao_picture(
        &pic,
        &sao_grid,
        slice.ctb_log2_size_y,
        params.chroma_array_type,
        slice.slice_sao_luma_flag,
        slice.slice_sao_chroma_flag,
    );

    Ok((filtered, field))
}

/// Walk one §7.3.8.4 coding quadtree, dispatching each leaf coding unit to
/// the intra or inter reconstruction path.
#[allow(clippy::too_many_arguments)]
fn reconstruct_inter_quadtree(
    pic: &mut Picture,
    ctx: &mut crate::recon::ReconCtx,
    field: &mut MotionField,
    params: &ReconParams,
    mv_ctx: &PuMvContext,
    refs: &RefListAccess,
    slice: &InterSliceContext,
    deblock_cus: &mut Vec<crate::deblock::DeblockCuDesc>,
    qt: &crate::slice_data::CodingQuadtree,
) -> Result<(), ReconError> {
    use crate::slice_data::CodingQuadtree;
    match qt {
        CodingQuadtree::Split(children) => {
            for child in children {
                reconstruct_inter_quadtree(
                    pic,
                    ctx,
                    field,
                    params,
                    mv_ctx,
                    refs,
                    slice,
                    deblock_cus,
                    child,
                )?;
            }
            Ok(())
        }
        CodingQuadtree::Leaf(cu) => reconstruct_inter_leaf_cu(
            pic,
            ctx,
            field,
            params,
            mv_ctx,
            refs,
            slice,
            deblock_cus,
            cu,
        ),
    }
}

/// Build the §8.7.2 [`crate::deblock::DeblockCuDesc`] for one coding unit
/// (its geometry, transform-split topology, partition mode, QP context, and
/// the CB-boundary edge-flag gates) and append it to `deblock_cus`.
fn collect_deblock_cu(
    cu: &CodingUnit,
    slice: &InterSliceContext,
    chroma_array_type: u8,
    bit_depth_luma: u8,
    bit_depth_chroma: u8,
    deblock_cus: &mut Vec<crate::deblock::DeblockCuDesc>,
) {
    let cu_params = crate::deblock::DeblockCuParams {
        qp_y: slice.slice_qp_y,
        beta_offset_div2: slice.beta_offset_div2,
        tc_offset_div2: slice.tc_offset_div2,
        cb_qp_offset: slice.cb_qp_offset,
        cr_qp_offset: slice.cr_qp_offset,
        bit_depth_luma,
        bit_depth_chroma,
        chroma_array_type,
    };
    deblock_cus.push(crate::deblock::DeblockCuDesc {
        cu: crate::deblock::DeblockCu {
            x_cb: cu.x0 as usize,
            y_cb: cu.y0 as usize,
            log2_cb_size: cu.log2_cb_size,
            params: cu_params,
            qp_y_p: slice.slice_qp_y,
        },
        transform_split: crate::deblock::TransformSplit::from_tree(cu.transform_tree.as_ref()),
        part_mode: cu.part_mode,
        // §8.7.2.1 — the CB-boundary edges are filtered except at the
        // picture's left / top border (the single-slice / single-tile case;
        // a slice / tile boundary with loop-filter-across disabled would
        // additionally clear these, threaded by the caller).
        filter_left: cu.x0 != 0,
        filter_top: cu.y0 != 0,
    });
}

/// Reconstruct one leaf coding unit (intra → §8.4 path + intra-stamp the
/// motion field; inter → §8.5 path), and collect its deblocking descriptor.
#[allow(clippy::too_many_arguments)]
fn reconstruct_inter_leaf_cu(
    pic: &mut Picture,
    ctx: &mut crate::recon::ReconCtx,
    field: &mut MotionField,
    params: &ReconParams,
    mv_ctx: &PuMvContext,
    refs: &RefListAccess,
    slice: &InterSliceContext,
    deblock_cus: &mut Vec<crate::deblock::DeblockCuDesc>,
    cu: &CodingUnit,
) -> Result<(), ReconError> {
    use crate::binarization::CuPredMode;
    let n_cb_s = 1usize << cu.log2_cb_size;
    if slice.deblock_enabled {
        collect_deblock_cu(
            cu,
            slice,
            params.chroma_array_type,
            params.bit_depth_luma,
            params.bit_depth_chroma,
            deblock_cus,
        );
    }
    if matches!(cu.cu_pred_mode, CuPredMode::Intra) {
        // §8.4 intra reconstruction; stamp the motion field intra so a
        // later inter CU's §6.4.2 availability denies it as a candidate.
        crate::recon::reconstruct_intra_cu_ctx(pic, params, ctx, cu)?;
        field.fill_rect(
            cu.x0 as usize,
            cu.y0 as usize,
            n_cb_s,
            n_cb_s,
            crate::motion::MotionCell {
                is_intra: true,
                ..crate::motion::MotionCell::default()
            },
        );
        // §8.7.2.4 — mark the intra CU's coded transform blocks (intra
        // neighbours give bS = 2 regardless, but the cbf flag is read for
        // an inter-side q neighbour at the shared edge).
        if let Some(tree) = cu.transform_tree.as_ref() {
            mark_nonzero_luma(field, tree, cu.x0 as usize, cu.y0 as usize, cu.log2_cb_size);
        }
        return Ok(());
    }

    // §6.4.2 prediction-block availability against the shared tiling + the
    // per-cell intra / inter flag of the motion field. The candidate
    // derivation also *mutates* `field` (writing each PU's resolved motion),
    // so the closure cannot borrow `field` directly; snapshot the per-4×4
    // intra-flag grid up front. A neighbour outside the current CU was
    // decoded before it, so its intra flag is already final; positions
    // inside the current CU are excluded by the §6.4.2 z-scan test.
    let x_cb = cu.x0;
    let y_cb = cu.y0;
    let w4 = field.width_4();
    let h4 = field.height_4();
    let mut intra_grid = vec![false; w4 * h4];
    for (gy, row) in intra_grid.chunks_mut(w4).enumerate() {
        for (gx, cell) in row.iter_mut().enumerate() {
            *cell = field.cell_at(gx * 4, gy * 4).is_intra;
        }
    }
    let tiling = ctx.tiling();
    let available = |x_nb: i32, y_nb: i32| -> bool {
        let cu_pred_mode = |x: u32, y: u32| -> u8 {
            let (gx, gy) = ((x as usize) / 4, (y as usize) / 4);
            if gx < w4 && gy < h4 && intra_grid[gy * w4 + gx] {
                crate::availability::MODE_INTRA
            } else {
                0
            }
        };
        tiling.prediction_block_availability(
            x_cb,
            y_cb,
            n_cb_s as u32,
            x_cb,
            y_cb,
            n_cb_s as u32,
            n_cb_s as u32,
            0,
            x_nb,
            y_nb,
            |ctb_rs| ctx.slice_addr_rs_of(ctb_rs),
            cu_pred_mode,
        )
    };

    resolve_and_reconstruct_inter_cu(
        pic,
        field,
        params,
        cu,
        &cu.prediction_units,
        mv_ctx,
        &available,
        refs,
    )
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

/// §8.3 + §8.5 — decode one inter (P / B) picture end to end against the
/// decoded-picture buffer, completing the per-picture reference cycle.
///
/// Ties the §8.3.1 → §8.3.2 → §8.3.4 → §8.3.5 reference derivation
/// ([`crate::decode::PictureSequenceState::begin_picture`]) to the
/// picture-level inter reconstruction: it resolves `RefPicList0` /
/// `RefPicList1` + `ColPic` into the [`RefListAccess`] + collocated motion
/// field the inter driver reads, runs [`reconstruct_inter_picture`] (recon →
/// deblock → SAO), then inserts the reconstructed picture + its motion field
/// into the DPB ([`crate::decode::PictureSequenceState::store_picture`]) as a
/// short-term reference for the next picture.
///
/// `header` / `slice_ref` carry the §8.3 inputs; `slice` the §8.5.3.2 /
/// §8.7 slice-constant inputs; `ctus` the decoded CTUs in decode order.
/// Returns the (output) reconstructed picture.
///
/// # Errors
/// [`ReconError::InterNotSupported`] when the picture is an I picture (no
/// reference lists were built — use the intra driver instead) or a used
/// reference resolves to "no reference picture"; otherwise the per-CU
/// reconstruction errors.
#[allow(clippy::too_many_arguments)]
pub fn decode_inter_picture(
    seq: &mut crate::decode::PictureSequenceState,
    header: &crate::decode::PictureHeaderInfo,
    slice_ref: &crate::decode::SliceRefParams,
    pic_width_luma: usize,
    pic_height_luma: usize,
    params: &ReconParams,
    slice: &InterSliceContext,
    tiles: &crate::availability::TilingParams,
    ctus: &[PlacedInterCtu<'_>],
) -> Result<Picture, ReconError> {
    // §8.3.1 → §8.3.5 — POC, RPS marking, reference lists, ColPic.
    let ref_state = seq.begin_picture(header, slice_ref);
    let lists = ref_state
        .ref_pic_lists
        .clone()
        .ok_or(ReconError::InterNotSupported)?;
    let layer_id = header.layer_id;
    let poc = ref_state.poc;

    // The collocated picture's motion field for the §8.5.3.2.9 temporal MVP.
    let col_field = ref_state
        .col_pic
        .map(|idx| &seq.dpb().entries()[idx].motion);

    let refs = RefListAccess {
        lists: &lists,
        entries: seq.dpb().entries(),
    };

    let (picture, motion) = reconstruct_inter_picture(
        pic_width_luma,
        pic_height_luma,
        params,
        slice,
        tiles,
        ctus,
        &refs,
        col_field,
    )?;

    // §8.3.2 — insert as a short-term reference for the following pictures.
    // (The output picture is returned to the caller before the move.)
    let output = picture.clone();
    seq.store_picture(poc, layer_id, picture, motion);
    Ok(output)
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

    fn p_slice_ctx() -> InterSliceContext {
        InterSliceContext {
            curr_poc: 4,
            slice_is_b: false,
            ctb_log2_size_y: 5,
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
            min_tb_log2_size_y: 2,
            deblock_enabled: false,
            beta_offset_div2: 0,
            tc_offset_div2: 0,
            slice_qp_y: 25,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            slice_sao_luma_flag: false,
            slice_sao_chroma_flag: false,
            log2_sao_offset_scale_luma: 0,
            log2_sao_offset_scale_chroma: 0,
        }
    }

    /// The §8.5 picture-level driver reconstructs a single-CTU P picture
    /// whose one 32×32 inter merge CU (zero-MV fallback) copies the flat
    /// reference samples, and records the CU's motion into the returned
    /// field.
    #[test]
    fn picture_driver_single_inter_ctu_copies_reference() {
        let params = p_params();
        let refpic = flat_ref(77, 99);
        let entries = vec![dpb_entry(0, refpic)];
        let lists = RefPicLists {
            list0: vec![Some(0)],
            list1: None,
        };
        let refs = RefListAccess {
            lists: &lists,
            entries: &entries,
        };

        // One 32×32 inter merge CU at (0,0) — a single coding tree unit
        // covering the whole picture.
        let mut cu = inter_cu_16(merge_pu(0), None);
        cu.log2_cb_size = 5;
        let ctu = crate::slice_data::CodingTreeUnit {
            sao: None,
            quadtree: crate::slice_data::CodingQuadtree::Leaf(Box::new(cu)),
        };
        let placed = vec![PlacedInterCtu {
            x_ctb: 0,
            y_ctb: 0,
            slice_addr_rs: 0,
            ctu: &ctu,
        }];

        let slice = p_slice_ctx();
        let tiles = crate::availability::TilingParams::single_tile();
        let (pic, field) =
            reconstruct_inter_picture(32, 32, &params, &slice, &tiles, &placed, &refs, None)
                .unwrap();

        for y in 0..32 {
            for x in 0..32 {
                assert_eq!(pic.sample(Plane::Luma, x, y), 77, "luma ({x},{y})");
            }
        }
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(pic.sample(Plane::Cb, x, y), 99, "cb ({x},{y})");
            }
        }
        let cell = field.cell_at(16, 16);
        assert!(!cell.is_intra && cell.pred_flag_l0);
        assert_eq!(cell.mv_l0, [0, 0]);
    }

    /// A mixed P picture: an intra CU and an inter CU side by side both
    /// reconstruct, and the intra CU is stamped intra in the motion field.
    #[test]
    fn picture_driver_mixed_intra_inter() {
        let params = p_params();
        let refpic = flat_ref(60, 90);
        let entries = vec![dpb_entry(0, refpic)];
        let lists = RefPicLists {
            list0: vec![Some(0)],
            list1: None,
        };
        let refs = RefListAccess {
            lists: &lists,
            entries: &entries,
        };

        // CTU split into four 16×16 quadrants: top-left intra (DC), the
        // other three inter merge.
        let intra_cu = CodingUnit {
            x0: 0,
            y0: 0,
            log2_cb_size: 4,
            cu_pred_mode: CuPredMode::Intra,
            cu_transquant_bypass_flag: false,
            part_mode: PartMode::Part2Nx2N,
            pcm_flag: false,
            prediction_units: vec![],
            intra_luma: vec![crate::slice_data::IntraLumaMode {
                prev_intra_luma_pred_flag: true,
                mpm_idx: Some(0),
                rem_intra_luma_pred_mode: None,
            }],
            intra_chroma_pred_mode: vec![4],
            rqt_root_cbf: true,
            transform_tree: Some(TransformTree::Leaf {
                cbf_luma: false,
                unit: TransformUnit::default(),
            }),
        };
        let mut inter_tr = inter_cu_16(merge_pu(0), None);
        inter_tr.x0 = 16;
        let mut inter_bl = inter_cu_16(merge_pu(0), None);
        inter_bl.y0 = 16;
        let mut inter_br = inter_cu_16(merge_pu(0), None);
        inter_br.x0 = 16;
        inter_br.y0 = 16;

        let ctu = crate::slice_data::CodingTreeUnit {
            sao: None,
            quadtree: crate::slice_data::CodingQuadtree::Split(vec![
                crate::slice_data::CodingQuadtree::Leaf(Box::new(intra_cu)),
                crate::slice_data::CodingQuadtree::Leaf(Box::new(inter_tr)),
                crate::slice_data::CodingQuadtree::Leaf(Box::new(inter_bl)),
                crate::slice_data::CodingQuadtree::Leaf(Box::new(inter_br)),
            ]),
        };
        let placed = vec![PlacedInterCtu {
            x_ctb: 0,
            y_ctb: 0,
            slice_addr_rs: 0,
            ctu: &ctu,
        }];

        let slice = p_slice_ctx();
        let tiles = crate::availability::TilingParams::single_tile();
        let (pic, field) =
            reconstruct_inter_picture(32, 32, &params, &slice, &tiles, &placed, &refs, None)
                .unwrap();

        // The three inter quadrants copy the reference value 60.
        assert_eq!(pic.sample(Plane::Luma, 24, 8), 60, "top-right inter");
        assert_eq!(pic.sample(Plane::Luma, 8, 24), 60, "bottom-left inter");
        assert_eq!(pic.sample(Plane::Luma, 24, 24), 60, "bottom-right inter");
        // The intra quadrant is stamped intra; the inter quadrants are not.
        assert!(field.cell_at(0, 0).is_intra, "TL intra-stamped");
        assert!(!field.cell_at(16, 0).is_intra, "TR inter");
        assert!(!field.cell_at(16, 16).is_intra, "BR inter");
    }

    /// With deblocking enabled, the §8.7.2 in-loop pass runs as part of the
    /// picture driver: a strong luma step at the intra/inter CU boundary
    /// (bS = 2) is smoothed, so the boundary samples differ from the
    /// undeblocked reconstruction.
    #[test]
    fn picture_driver_deblock_smooths_cu_boundary() {
        let params = p_params();
        // Reference for the inter CUs: flat 140 — a modest step from the
        // intra DC (128) so the §8.7.2.5.3 dE decision classifies the
        // boundary as a blocking artifact (small gradient) rather than a
        // true edge, and the weak filter engages.
        let refpic = flat_ref(140, 128);
        let entries = vec![dpb_entry(0, refpic)];
        let lists = RefPicLists {
            list0: vec![Some(0)],
            list1: None,
        };
        let refs = RefListAccess {
            lists: &lists,
            entries: &entries,
        };

        // CTU split into four 16×16 quadrants: the two left quadrants intra
        // (DC ⇒ mid-grey 128, no neighbours), the two right inter (copy 200).
        let intra = |x0: u32, y0: u32| CodingUnit {
            x0,
            y0,
            log2_cb_size: 4,
            cu_pred_mode: CuPredMode::Intra,
            cu_transquant_bypass_flag: false,
            part_mode: PartMode::Part2Nx2N,
            pcm_flag: false,
            prediction_units: vec![],
            intra_luma: vec![crate::slice_data::IntraLumaMode {
                prev_intra_luma_pred_flag: false,
                mpm_idx: None,
                rem_intra_luma_pred_mode: Some(1),
            }],
            intra_chroma_pred_mode: vec![4],
            rqt_root_cbf: false,
            transform_tree: Some(TransformTree::Leaf {
                cbf_luma: false,
                unit: TransformUnit::default(),
            }),
        };
        let inter = |x0: u32, y0: u32| {
            let mut c = inter_cu_16(merge_pu(0), None);
            c.x0 = x0;
            c.y0 = y0;
            c
        };
        let ctu = crate::slice_data::CodingTreeUnit {
            sao: None,
            quadtree: crate::slice_data::CodingQuadtree::Split(vec![
                crate::slice_data::CodingQuadtree::Leaf(Box::new(intra(0, 0))),
                crate::slice_data::CodingQuadtree::Leaf(Box::new(inter(16, 0))),
                crate::slice_data::CodingQuadtree::Leaf(Box::new(intra(0, 16))),
                crate::slice_data::CodingQuadtree::Leaf(Box::new(inter(16, 16))),
            ]),
        };
        let placed = vec![PlacedInterCtu {
            x_ctb: 0,
            y_ctb: 0,
            slice_addr_rs: 0,
            ctu: &ctu,
        }];

        let tiles = crate::availability::TilingParams::single_tile();
        // Undeblocked baseline.
        let mut undeb = p_slice_ctx();
        undeb.deblock_enabled = false;
        let (plain, _) =
            reconstruct_inter_picture(32, 32, &params, &undeb, &tiles, &placed, &refs, None)
                .unwrap();
        // Deblocked.
        let mut deb = p_slice_ctx();
        deb.deblock_enabled = true;
        let (filtered, _) =
            reconstruct_inter_picture(32, 32, &params, &deb, &tiles, &placed, &refs, None).unwrap();

        // The vertical boundary at x == 16 separates intra (128) from inter
        // (200); the deblock pass adjusts samples on at least one side, so
        // the filtered picture differs from the plain one near the edge.
        let mut changed = false;
        for y in 0..32 {
            for x in 13..19 {
                if filtered.sample(Plane::Luma, x, y) != plain.sample(Plane::Luma, x, y) {
                    changed = true;
                }
            }
        }
        assert!(changed, "deblocking modifies samples at the CU boundary");
        // Far-from-edge interior samples are untouched.
        assert_eq!(filtered.sample(Plane::Luma, 28, 8), 140, "inter interior");
    }

    /// With `slice_sao_luma_flag` set and a band-offset SAO on the CTB, the
    /// §8.7.3 SAO pass runs as part of the picture driver: the inter CU's
    /// flat luma is shifted by the band offset covering its value.
    #[test]
    fn picture_driver_applies_sao() {
        let params = p_params();
        // Inter reference flat 100. Luma 100 is in band 100 >> 3 == 12.
        let refpic = flat_ref(100, 128);
        let entries = vec![dpb_entry(0, refpic)];
        let lists = RefPicLists {
            list0: vec![Some(0)],
            list1: None,
        };
        let refs = RefListAccess {
            lists: &lists,
            entries: &entries,
        };

        let mut cu = inter_cu_16(merge_pu(0), None);
        cu.log2_cb_size = 5;
        // Band-offset SAO on luma: band_position 12 (covers value 100),
        // first offset +5 ⇒ luma 100 → 105.
        let sao = crate::slice_data::SaoCtbParams {
            merge_left: false,
            merge_up: false,
            components: [
                crate::slice_data::SaoComponent {
                    sao_type_idx: 1,
                    offset_abs: [5, 0, 0, 0],
                    offset_sign: [0, 0, 0, 0],
                    band_position: 12,
                    eo_class: 0,
                },
                crate::slice_data::SaoComponent::default(),
                crate::slice_data::SaoComponent::default(),
            ],
        };
        let ctu = crate::slice_data::CodingTreeUnit {
            sao: Some(sao),
            quadtree: crate::slice_data::CodingQuadtree::Leaf(Box::new(cu)),
        };
        let placed = vec![PlacedInterCtu {
            x_ctb: 0,
            y_ctb: 0,
            slice_addr_rs: 0,
            ctu: &ctu,
        }];

        let mut slice = p_slice_ctx();
        slice.slice_sao_luma_flag = true;
        let tiles = crate::availability::TilingParams::single_tile();
        let (pic, _) =
            reconstruct_inter_picture(32, 32, &params, &slice, &tiles, &placed, &refs, None)
                .unwrap();
        // The reconstructed inter samples (100) fall in SAO band 12 and get
        // the +5 offset.
        assert_eq!(
            pic.sample(Plane::Luma, 8, 8),
            105,
            "SAO band offset applied"
        );
    }

    /// §8.3 + §8.5 end-to-end inter picture cycle: an IDR reference picture
    /// is stored in the DPB, then a P picture (one 32×32 inter merge CU)
    /// decodes against it via `decode_inter_picture` — resolving
    /// RefPicList0[0] → the IDR, copying its samples, and landing in the DPB
    /// as a short-term reference.
    #[test]
    fn decode_inter_picture_full_cycle() {
        use crate::poc::NalKind;
        use crate::sps::MaterializedShortTermRefPicSet;

        let params = p_params();
        let mut seq = crate::decode::PictureSequenceState::new();

        // IDR (POC 0): a flat-110 reference picture stored directly.
        let idr_header = crate::decode::PictureHeaderInfo {
            nal_kind: NalKind::new(NalKind::IDR_N_LP),
            temporal_id: 0,
            layer_id: 0,
            no_rasl_output: true,
            poc_lsb: 0,
            max_poc_lsb: 256,
            short_term_rps: MaterializedShortTermRefPicSet {
                delta_poc_s0: vec![],
                used_by_curr_pic_s0: vec![],
                delta_poc_s1: vec![],
                used_by_curr_pic_s1: vec![],
            },
            long_term: vec![],
        };
        let i_slice = crate::decode::SliceRefParams {
            is_inter: false,
            is_b: false,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            num_pic_total_curr: 0,
            temporal_mvp_enabled: false,
            collocated_from_l0_flag: true,
            collocated_ref_idx: 0,
        };
        let idr = seq.begin_picture(&idr_header, &i_slice);
        seq.store_picture(idr.poc, 0, flat_ref(110, 128), MotionField::new(32, 32));

        // P picture (POC 1): one short-term-before reference at POC 0.
        let p_header = crate::decode::PictureHeaderInfo {
            nal_kind: NalKind::new(NalKind::TRAIL_R),
            no_rasl_output: false,
            poc_lsb: 1,
            short_term_rps: MaterializedShortTermRefPicSet {
                delta_poc_s0: vec![-1],
                used_by_curr_pic_s0: vec![true],
                delta_poc_s1: vec![],
                used_by_curr_pic_s1: vec![],
            },
            ..idr_header.clone()
        };
        let p_slice_ref = crate::decode::SliceRefParams {
            is_inter: true,
            num_pic_total_curr: 1,
            ..i_slice
        };

        let mut cu = inter_cu_16(merge_pu(0), None);
        cu.log2_cb_size = 5;
        let ctu = crate::slice_data::CodingTreeUnit {
            sao: None,
            quadtree: crate::slice_data::CodingQuadtree::Leaf(Box::new(cu)),
        };
        let placed = vec![PlacedInterCtu {
            x_ctb: 0,
            y_ctb: 0,
            slice_addr_rs: 0,
            ctu: &ctu,
        }];
        let slice = p_slice_ctx();
        let tiles = crate::availability::TilingParams::single_tile();

        let out = decode_inter_picture(
            &mut seq,
            &p_header,
            &p_slice_ref,
            32,
            32,
            &params,
            &slice,
            &tiles,
            &placed,
        )
        .unwrap();

        // The P picture copies the IDR reference's flat 110.
        for y in 0..32 {
            for x in 0..32 {
                assert_eq!(out.sample(Plane::Luma, x, y), 110, "P luma ({x},{y})");
            }
        }
        // The DPB now holds two pictures (IDR POC 0 + P POC 1), both
        // short-term references.
        assert_eq!(seq.dpb().entries().len(), 2);
        assert_eq!(seq.dpb().entries()[1].poc, 1);
        assert_eq!(
            seq.dpb().entries()[1].marking,
            crate::dpb::Marking::ShortTerm
        );
        // The P picture's motion field records the inter CU (for a future
        // picture's temporal MVP).
        assert!(!seq.dpb().entries()[1].motion.cell_at(16, 16).is_intra);
    }
}
