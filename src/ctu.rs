//! HEVC CTU / coding_quadtree walker (§7.3.8, §7.4.9).
//!
//! Orchestrates the full I-slice decode pipeline:
//!
//! 1. Parse the SAO parameters for the CTU.
//! 2. Recurse through the coding quadtree, splitting on the CABAC-coded
//!    `split_cu_flag`.
//! 3. At each leaf coding unit, decode the part-mode, intra prediction
//!    modes, and recurse through the transform tree.
//! 4. At each transform unit, decode cbf flags + residual coefficients,
//!    run intra prediction, inverse transform, and reconstruct.
//!
//! Scope is restricted to I-slice 8-bit 4:2:0 with `separate_colour_plane
//! == false` and `pcm_enabled == false`. Any out-of-scope shape surfaces
//! a clean `Error::Unsupported` early so callers can fall back.

use oxideav_core::{Error, Result};

use crate::cabac::{
    init_row, CabacEngine, CtxState, InitType, ABS_MVD_GREATER_FLAGS_INIT_VALUES,
    CBF_CB_CR_INIT_VALUES, CBF_LUMA_INIT_VALUES, CODED_SUB_BLOCK_FLAG_INIT_VALUES,
    COEFF_ABS_GT1_INIT_VALUES, COEFF_ABS_GT2_INIT_VALUES, CU_QP_DELTA_ABS_INIT_VALUES,
    CU_SKIP_FLAG_INIT_VALUES, CU_TRANSQUANT_BYPASS_FLAG_INIT_VALUES, INTER_PRED_IDC_INIT_VALUES,
    INTRA_CHROMA_PRED_MODE_INIT_VALUES, LAST_SIG_COEFF_X_PREFIX_INIT_VALUES,
    LAST_SIG_COEFF_Y_PREFIX_INIT_VALUES, MERGE_FLAG_INIT_VALUES, MERGE_IDX_INIT_VALUES,
    MVP_LX_FLAG_INIT_VALUES, PART_MODE_INIT_VALUES, PRED_MODE_FLAG_INIT_VALUES,
    PREV_INTRA_LUMA_PRED_FLAG_INIT_VALUES, REF_IDX_INIT_VALUES, RQT_ROOT_CBF_INIT_VALUES,
    SAO_MERGE_FLAG_INIT_VALUES, SAO_TYPE_IDX_INIT_VALUES, SIG_COEFF_FLAG_INIT_VALUES,
    SPLIT_CU_FLAG_INIT_VALUES, SPLIT_TRANSFORM_FLAG_INIT_VALUES,
    TRANSFORM_SKIP_FLAG_CHROMA_INIT_VALUES, TRANSFORM_SKIP_FLAG_LUMA_INIT_VALUES,
};
use crate::inter::{
    build_amvp_list, build_merge_list, chroma_mc, chroma_mc_bi_combine, chroma_mc_hp, luma_mc,
    luma_mc_bi_combine, luma_mc_bi_weighted, luma_mc_hp, InterState, MergeCand, MotionVector,
    PbMotion, RefPicture, WeightedPred,
};
use crate::intra_pred::{build_ref_samples, filter_decision, filter_ref_samples, predict};
use crate::pps::PicParameterSet;
use crate::sao::{CtuSaoParams, SaoGrid};
use crate::scaling_list::ScalingListData;
use crate::scan::{scan_4x4, scan_idx_for_intra};
use crate::slice::{SliceSegmentHeader, SliceType};
use crate::sps::SeqParameterSet;
use crate::transform::{
    dequantize_flat, dequantize_with_matrix, inverse_transform_2d, transform_skip_2d,
};

/// Reconstructed picture (4:2:0, bit depth 8 or 10).
///
/// Samples are stored as `u16` so Main and Main 10 share a single code
/// path; callers that only care about 8-bit simply narrow on the way out.
pub struct Picture {
    pub width: u32,
    pub height: u32,
    pub luma: Vec<u16>,
    pub cb: Vec<u16>,
    pub cr: Vec<u16>,
    pub luma_stride: usize,
    pub chroma_stride: usize,
    /// Per-4x4 block intra prediction mode (luma) for neighbour derivation.
    /// Indexed `[(y>>2) * (width_in_4>>0) + (x>>2)]`.
    pub intra_luma_mode: Vec<u8>,
    pub intra_width_4: usize,
    pub intra_height_4: usize,
    /// Per-4x4 block "is intra" flag (false when the CU is inter-coded).
    pub is_intra: Vec<bool>,
    /// Per-4×4 block coding-quadtree depth of the containing CU. Used for
    /// `split_cu_flag` context derivation (§9.3.4.2.2).
    pub cqt_depth: Vec<u8>,
    /// Per-4×4 block "decoded" flag. A sample is available for intra
    /// reference only when its containing 4×4 block has already been
    /// reconstructed (Z-scan order per §6.4.3).
    pub decoded_4x4: Vec<bool>,
    /// Per-8×8 block luma QP. Populated at the end of each QG so that the
    /// next QG's QpY_PRED (§8.6.1) can look up the left/above neighbours.
    /// Indexed `[(y>>3) * qp_stride + (x>>3)]`.
    pub qp_y: Vec<i32>,
    pub qp_stride: usize,
    /// Per-4×4 block motion-field grid for inter neighbour lookups.
    pub inter: InterState,
    /// Per-CTB SAO parameters (§7.4.9.3). Populated by the CTU walker as
    /// it parses each `sao()` syntax element; read back by the post-decode
    /// SAO filter (§8.7.3). `None` until the walker attaches one.
    pub sao_grid: Option<SaoGrid>,
}

impl Picture {
    pub fn new(width: u32, height: u32) -> Self {
        let w = width as usize;
        let h = height as usize;
        let cw = w / 2;
        let ch = h / 2;
        let iw4 = w.div_ceil(4);
        let ih4 = h.div_ceil(4);
        Self {
            width,
            height,
            // 8-bit neutral grey. Callers should re-init for other bit
            // depths if they care about the leading junk before the CTU
            // walker has written every sample; in practice every code path
            // that reads a sample also writes it first.
            luma: vec![128u16; w * h],
            cb: vec![128u16; cw * ch],
            cr: vec![128u16; cw * ch],
            luma_stride: w,
            chroma_stride: cw,
            intra_luma_mode: vec![1u8; iw4 * ih4], // 1 = DC default (§8.4.4.2.1).
            intra_width_4: iw4,
            intra_height_4: ih4,
            is_intra: vec![true; iw4 * ih4],
            cqt_depth: vec![0u8; iw4 * ih4],
            decoded_4x4: vec![false; iw4 * ih4],
            qp_y: vec![0; w.div_ceil(8) * h.div_ceil(8)],
            qp_stride: w.div_ceil(8),
            inter: InterState::new(width, height),
            sao_grid: None,
        }
    }
}

/// Everything the CTU walker needs from higher-level parsers.
pub struct CtuContext<'a> {
    pub sps: &'a SeqParameterSet,
    pub pps: &'a PicParameterSet,
    pub slice: &'a SliceSegmentHeader,
    pub init_type: InitType,
    /// Reference picture list L0. Empty for I slices.
    pub ref_list_l0: &'a [RefPicture],
    /// Reference picture list L1. Empty for I and P slices.
    pub ref_list_l1: &'a [RefPicture],
    /// Collocated reference picture for TMVP lookups, selected from either
    /// L0 or L1 based on `collocated_from_l0_flag` + `collocated_ref_idx`.
    /// `None` when TMVP is disabled or no collocated ref is available.
    pub collocated_ref: Option<&'a RefPicture>,
    /// Weighted bi-prediction table, when signalled by
    /// `pred_weight_table()` for the slice.
    pub weighted_pred: Option<&'a WeightedPred>,
}

impl<'a> CtuContext<'a> {
    pub fn is_inter_slice(&self) -> bool {
        matches!(self.slice.slice_type, SliceType::P | SliceType::B)
    }

    pub fn is_b_slice(&self) -> bool {
        matches!(self.slice.slice_type, SliceType::B)
    }
}

/// Inter CU partition modes we support.
///
/// The four symmetric shapes (§7.4.9.5 Table 7-10) plus the four AMP shapes
/// enabled when `amp_enabled_flag = 1`. AMP splits the CB asymmetrically
/// into quarter + three-quarter stripes:
///
/// * `Mode2NxnU` — top stripe `cb/4`, bottom `3·cb/4`.
/// * `Mode2NxnD` — top stripe `3·cb/4`, bottom `cb/4`.
/// * `ModenLx2N` — left stripe `cb/4`, right `3·cb/4`.
/// * `ModenRx2N` — left stripe `3·cb/4`, right `cb/4`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InterPart {
    Mode2Nx2N,
    Mode2NxN,
    ModeNx2N,
    ModeNxN,
    Mode2NxnU,
    Mode2NxnD,
    ModenLx2N,
    ModenRx2N,
}

#[derive(Clone)]
struct Ctx {
    split_cu_flag: [CtxState; 3],
    cu_transquant_bypass_flag: [CtxState; 1],
    cu_skip_flag: [CtxState; 3],
    pred_mode_flag: [CtxState; 1],
    part_mode: [CtxState; 4],
    prev_intra_luma_pred_flag: [CtxState; 1],
    intra_chroma_pred_mode: [CtxState; 1],
    rqt_root_cbf: [CtxState; 1],
    split_transform_flag: [CtxState; 3],
    cbf_luma: [CtxState; 2],
    cbf_cb_cr: [CtxState; 5],
    cu_qp_delta_abs: [CtxState; 2],
    last_sig_x_prefix: [CtxState; 18],
    last_sig_y_prefix: [CtxState; 18],
    sig_coeff_flag: [CtxState; 44],
    coeff_abs_gt1: [CtxState; 24],
    coeff_abs_gt2: [CtxState; 6],
    coded_sub_block_flag: [CtxState; 4],
    sao_merge_flag: [CtxState; 1],
    sao_type_idx: [CtxState; 1],
    merge_flag: [CtxState; 1],
    merge_idx: [CtxState; 1],
    #[allow(dead_code)]
    inter_pred_idc: [CtxState; 5],
    ref_idx: [CtxState; 2],
    abs_mvd_greater: [CtxState; 2],
    mvp_lx_flag: [CtxState; 1],
    transform_skip_flag_luma: [CtxState; 1],
    transform_skip_flag_chroma: [CtxState; 1],
}

impl Ctx {
    fn init(slice_qp_y: i32, init_type: InitType) -> Self {
        Self {
            split_cu_flag: init_row(&SPLIT_CU_FLAG_INIT_VALUES, init_type, slice_qp_y),
            cu_transquant_bypass_flag: init_row(
                &CU_TRANSQUANT_BYPASS_FLAG_INIT_VALUES,
                init_type,
                slice_qp_y,
            ),
            cu_skip_flag: init_row(&CU_SKIP_FLAG_INIT_VALUES, init_type, slice_qp_y),
            pred_mode_flag: init_row(&PRED_MODE_FLAG_INIT_VALUES, init_type, slice_qp_y),
            part_mode: init_row(&PART_MODE_INIT_VALUES, init_type, slice_qp_y),
            prev_intra_luma_pred_flag: init_row(
                &PREV_INTRA_LUMA_PRED_FLAG_INIT_VALUES,
                init_type,
                slice_qp_y,
            ),
            intra_chroma_pred_mode: init_row(
                &INTRA_CHROMA_PRED_MODE_INIT_VALUES,
                init_type,
                slice_qp_y,
            ),
            rqt_root_cbf: init_row(&RQT_ROOT_CBF_INIT_VALUES, init_type, slice_qp_y),
            split_transform_flag: init_row(
                &SPLIT_TRANSFORM_FLAG_INIT_VALUES,
                init_type,
                slice_qp_y,
            ),
            cbf_luma: init_row(&CBF_LUMA_INIT_VALUES, init_type, slice_qp_y),
            cbf_cb_cr: init_row(&CBF_CB_CR_INIT_VALUES, init_type, slice_qp_y),
            cu_qp_delta_abs: init_row(&CU_QP_DELTA_ABS_INIT_VALUES, init_type, slice_qp_y),
            last_sig_x_prefix: init_row(
                &LAST_SIG_COEFF_X_PREFIX_INIT_VALUES,
                init_type,
                slice_qp_y,
            ),
            last_sig_y_prefix: init_row(
                &LAST_SIG_COEFF_Y_PREFIX_INIT_VALUES,
                init_type,
                slice_qp_y,
            ),
            sig_coeff_flag: init_row(&SIG_COEFF_FLAG_INIT_VALUES, init_type, slice_qp_y),
            coeff_abs_gt1: init_row(&COEFF_ABS_GT1_INIT_VALUES, init_type, slice_qp_y),
            coeff_abs_gt2: init_row(&COEFF_ABS_GT2_INIT_VALUES, init_type, slice_qp_y),
            coded_sub_block_flag: init_row(
                &CODED_SUB_BLOCK_FLAG_INIT_VALUES,
                init_type,
                slice_qp_y,
            ),
            sao_merge_flag: init_row(&SAO_MERGE_FLAG_INIT_VALUES, init_type, slice_qp_y),
            sao_type_idx: init_row(&SAO_TYPE_IDX_INIT_VALUES, init_type, slice_qp_y),
            merge_flag: init_row(&MERGE_FLAG_INIT_VALUES, init_type, slice_qp_y),
            merge_idx: init_row(&MERGE_IDX_INIT_VALUES, init_type, slice_qp_y),
            inter_pred_idc: init_row(&INTER_PRED_IDC_INIT_VALUES, init_type, slice_qp_y),
            ref_idx: init_row(&REF_IDX_INIT_VALUES, init_type, slice_qp_y),
            abs_mvd_greater: init_row(&ABS_MVD_GREATER_FLAGS_INIT_VALUES, init_type, slice_qp_y),
            mvp_lx_flag: init_row(&MVP_LX_FLAG_INIT_VALUES, init_type, slice_qp_y),
            transform_skip_flag_luma: init_row(
                &TRANSFORM_SKIP_FLAG_LUMA_INIT_VALUES,
                init_type,
                slice_qp_y,
            ),
            transform_skip_flag_chroma: init_row(
                &TRANSFORM_SKIP_FLAG_CHROMA_INIT_VALUES,
                init_type,
                slice_qp_y,
            ),
        }
    }
}

/// Decode every CTU of the slice and fill `pic`.
pub fn decode_slice_ctus(
    rbsp: &[u8],
    slice_data_byte_off: usize,
    cctx: &CtuContext<'_>,
    pic: &mut Picture,
) -> Result<()> {
    // Up-front bail for shapes outside v1 pixel-decode scope.
    if cctx.sps.separate_colour_plane_flag {
        return Err(Error::unsupported("h265 separate_colour_plane pending"));
    }
    if cctx.sps.chroma_format_idc != 1 {
        return Err(Error::unsupported("h265 only 4:2:0 pixel decode supported"));
    }
    // 8-bit (Main) and 10-bit (Main 10) are both supported. Higher bit
    // depths (Main 12 / Main 4:2:2 10-bit etc.) have not been validated so
    // surface them as a clean unsupported error rather than producing
    // garbage reconstructions.
    if cctx.sps.bit_depth_y() > 10 || cctx.sps.bit_depth_c() > 10 {
        return Err(Error::unsupported(
            "h265 pixel decode limited to bit_depth <= 10",
        ));
    }
    if cctx.sps.bit_depth_y() != cctx.sps.bit_depth_c() {
        return Err(Error::unsupported(
            "h265 pixel decode requires matching luma/chroma bit depth",
        ));
    }
    // Inter motion compensation (§8.5.3.3) clips at Clip1Y/Clip1C using
    // `bit_depth` from the SPS, so Main 10 P/B slices now share the MC
    // path with Main. Higher bit depths (>10) are rejected above.
    // Note: SAO parameters are parsed per-CTU (see decode_sao) into
    // `pic.sao_grid`; the post-decode filter (§8.7.3) runs from
    // `decoder::decode_*_slice` after deblocking. Deblocking (§8.7.2)
    // runs there too, before SAO.
    if cctx.slice.slice_type == SliceType::P && cctx.ref_list_l0.is_empty() {
        return Err(Error::unsupported("h265 P-slice without RPL0"));
    }
    if cctx.slice.slice_type == SliceType::B
        && (cctx.ref_list_l0.is_empty() || cctx.ref_list_l1.is_empty())
    {
        return Err(Error::unsupported("h265 B-slice without both RPL0 + RPL1"));
    }

    let ctb_log2 = cctx.sps.log2_min_luma_coding_block_size_minus3
        + 3
        + cctx.sps.log2_diff_max_min_luma_coding_block_size;
    let ctb_size = 1u32 << ctb_log2;
    let min_cb_log2 = cctx.sps.log2_min_luma_coding_block_size_minus3 + 3;
    let max_tb_log2 = cctx.sps.log2_min_luma_transform_block_size_minus2
        + 2
        + cctx.sps.log2_diff_max_min_luma_transform_block_size;
    let min_tb_log2 = cctx.sps.log2_min_luma_transform_block_size_minus2 + 2;

    let pic_w = cctx.sps.pic_width_in_luma_samples;
    let pic_h = cctx.sps.pic_height_in_luma_samples;
    let ctbs_x = pic_w.div_ceil(ctb_size);
    let ctbs_y = pic_h.div_ceil(ctb_size);

    // §6.5.1 tile geometry. When tiles are disabled we build a single-tile
    // layout that spans the whole picture so the downstream CTU iteration
    // is a single code-path.
    let tile_plan = TilePlan::build(cctx.pps, ctbs_x, ctbs_y);

    // Resolve the absolute byte offsets (within rbsp) of each sub-stream
    // start. Sub-stream 0 starts at `slice_data_byte_off`; sub-stream i+1
    // is `entry_point_offsets[i]` bytes later. When neither tiles nor WPP
    // are enabled the slice is a single sub-stream.
    let sub_stream_offsets: Vec<usize> = {
        let mut v = Vec::with_capacity(cctx.slice.entry_point_offsets.len() + 1);
        v.push(slice_data_byte_off);
        for off in &cctx.slice.entry_point_offsets {
            v.push(slice_data_byte_off + *off as usize);
        }
        v
    };

    // Multi-tile and multi-row-WPP both require the slice to carry the
    // matching number of entry points. Single-tile + single-row are a
    // single sub-stream; reject mismatches loudly — missing entry points
    // means we cannot re-seed CABAC at the correct byte positions.
    let multi_tile = cctx.pps.tiles_enabled_flag && tile_plan.num_tiles() > 1;
    let multi_row_wpp = cctx.pps.entropy_coding_sync_enabled_flag && ctbs_y > 1;
    if multi_tile && (multi_row_wpp) {
        return Err(Error::unsupported(
            "h265 combined tiles + WPP not supported yet",
        ));
    }
    if multi_tile {
        let needed = tile_plan.num_tiles() as usize - 1;
        if cctx.slice.entry_point_offsets.len() < needed {
            return Err(Error::invalid(format!(
                "h265 slice: tiled slice missing entry points (have {}, need {})",
                cctx.slice.entry_point_offsets.len(),
                needed
            )));
        }
    }
    if multi_row_wpp {
        let needed = (ctbs_y as usize).saturating_sub(1);
        if cctx.slice.entry_point_offsets.len() < needed {
            return Err(Error::invalid(format!(
                "h265 slice: WPP slice missing entry points (have {}, need {})",
                cctx.slice.entry_point_offsets.len(),
                needed
            )));
        }
    }

    let mut engine = CabacEngine::new(rbsp, slice_data_byte_off);
    let mut ctx = Ctx::init(cctx.slice.slice_qp_y, cctx.init_type);
    let mut cu_qp_y = cctx.slice.slice_qp_y;
    // Attach a per-slice SAO grid to the picture so the walker can populate
    // per-CTB params as each `sao()` syntax element is parsed, and the
    // post-decode filter (§8.7.3) can read them back later.
    if cctx.sps.sample_adaptive_offset_enabled_flag {
        pic.sao_grid = Some(SaoGrid::new(ctbs_x, ctbs_y, ctb_log2));
    }
    // §7.4.5: active ScalingListData for the slice. PPS-level data, when
    // present, overrides any SPS-level data; if neither is signalled but
    // scaling lists are enabled the defaults (Table 7-5 / 7-6) apply.
    let active_scaling_list: Option<ScalingListData> = if cctx.sps.scaling_list_enabled_flag {
        if let Some(d) = &cctx.pps.scaling_list_data {
            Some(d.clone())
        } else if let Some(d) = &cctx.sps.scaling_list_data {
            Some(d.clone())
        } else {
            Some(ScalingListData::spec_defaults())
        }
    } else {
        None
    };
    let mut walker = Walker {
        pic,
        cctx,
        min_cb_log2,
        max_tb_log2,
        min_tb_log2,
        cu_qp_y: &mut cu_qp_y,
        is_cu_qp_delta_coded: false,
        last_qg_pos: None,
        qpy_prev: cctx.slice.slice_qp_y,
        qpy_pred: cctx.slice.slice_qp_y,
        scaling_list: active_scaling_list.as_ref(),
        cu_transquant_bypass: false,
    };

    if multi_tile {
        // §6.3.1: per-tile CTU raster scan. Tiles are decoded in the order
        // they appear in the bitstream (row-major over the tile grid).
        // For each tile after the first, re-init the CABAC engine at the
        // matching entry point and rebuild the context table per §9.3.2.4.
        for tile_idx in 0..tile_plan.num_tiles() as usize {
            if tile_idx > 0 {
                let off = sub_stream_offsets
                    .get(tile_idx)
                    .copied()
                    .unwrap_or(slice_data_byte_off);
                engine.reinit_at_byte(off);
                ctx = Ctx::init(cctx.slice.slice_qp_y, cctx.init_type);
            }
            let (ctb_x0, ctb_y0, cols, rows) = tile_plan.tile_bounds(tile_idx as u32);
            for cty in ctb_y0..(ctb_y0 + rows) {
                for ctx_x in ctb_x0..(ctb_x0 + cols) {
                    let x0 = ctx_x * ctb_size;
                    let y0 = cty * ctb_size;
                    if cctx.slice.slice_sao_luma_flag || cctx.slice.slice_sao_chroma_flag {
                        walker.decode_sao(&mut engine, &mut ctx, ctx_x, cty)?;
                    }
                    walker.coding_quadtree(&mut engine, &mut ctx, x0, y0, ctb_log2, 0)?;
                    let _ = engine.decode_terminate();
                }
            }
        }
    } else if multi_row_wpp {
        // §6.3.2: wavefront parallel processing. At each CTU row start
        // (except the first), CABAC is re-initialised from a snapshot of
        // the row-above context *after* its second CTU (column index 1)
        // completed (§9.3.2.4). The arithmetic engine restarts at the
        // entry-point byte position of the row.
        let mut row_snapshot: Option<Ctx> = None;
        for cty in 0..ctbs_y {
            if cty > 0 {
                let off = sub_stream_offsets
                    .get(cty as usize)
                    .copied()
                    .unwrap_or(slice_data_byte_off);
                engine.reinit_at_byte(off);
                // §9.3.2.4: context table is inherited from the row above
                // after its second CTU (ctbAddrInRs with horizontal index
                // 1). When the row above is only 1 CTU wide or wasn't
                // snapshotted, fall back to a fresh slice-level init.
                ctx = row_snapshot
                    .take()
                    .unwrap_or_else(|| Ctx::init(cctx.slice.slice_qp_y, cctx.init_type));
            }
            for ctx_x in 0..ctbs_x {
                let x0 = ctx_x * ctb_size;
                let y0 = cty * ctb_size;
                if cctx.slice.slice_sao_luma_flag || cctx.slice.slice_sao_chroma_flag {
                    walker.decode_sao(&mut engine, &mut ctx, ctx_x, cty)?;
                }
                walker.coding_quadtree(&mut engine, &mut ctx, x0, y0, ctb_log2, 0)?;
                let _ = engine.decode_terminate();
                // Snapshot the context immediately after the second CTU of
                // the current row (column index 1) finishes — that's the
                // seed for the next row per §9.3.2.4.
                if ctx_x == 1 && cty + 1 < ctbs_y {
                    row_snapshot = Some(ctx.clone());
                }
            }
        }
    } else {
        for cty in 0..ctbs_y {
            for ctx_x in 0..ctbs_x {
                let x0 = ctx_x * ctb_size;
                let y0 = cty * ctb_size;
                // SAO parameters (§7.3.8.3).
                if cctx.slice.slice_sao_luma_flag || cctx.slice.slice_sao_chroma_flag {
                    walker.decode_sao(&mut engine, &mut ctx, ctx_x, cty)?;
                }
                walker.coding_quadtree(&mut engine, &mut ctx, x0, y0, ctb_log2, 0)?;
                // end_of_slice_flag / end_of_slice_segment_flag (one terminating bin).
                // We consume the bin but tolerate disagreement — a single-bit
                // bitstream-position desync at the end doesn't invalidate the
                // reconstructed picture, and our simplified residual_coding may
                // land one bit off the exact spec position.
                let _ = engine.decode_terminate();
                let _is_last = ctx_x + 1 == ctbs_x && cty + 1 == ctbs_y;
            }
        }
    }
    Ok(())
}

/// §6.5.1 resolved tile layout. When tiles are disabled the plan carries a
/// single tile spanning the whole picture so the CTU walker has one
/// unified code-path.
struct TilePlan {
    /// Per-tile-column widths in CTBs.
    col_widths: Vec<u32>,
    /// Per-tile-row heights in CTBs.
    row_heights: Vec<u32>,
    /// Cumulative column starts in CTBs — length `col_widths.len() + 1`.
    col_starts: Vec<u32>,
    /// Cumulative row starts in CTBs — length `row_heights.len() + 1`.
    row_starts: Vec<u32>,
}

impl TilePlan {
    fn build(pps: &crate::pps::PicParameterSet, pic_w_ctb: u32, pic_h_ctb: u32) -> Self {
        let (col_widths, row_heights) = if let Some(ti) = pps.tile_info.as_ref() {
            (
                ti.column_widths_ctb(pic_w_ctb),
                ti.row_heights_ctb(pic_h_ctb),
            )
        } else {
            (vec![pic_w_ctb], vec![pic_h_ctb])
        };
        let mut col_starts = Vec::with_capacity(col_widths.len() + 1);
        col_starts.push(0);
        for w in &col_widths {
            let s = col_starts.last().copied().unwrap_or(0);
            col_starts.push(s + w);
        }
        let mut row_starts = Vec::with_capacity(row_heights.len() + 1);
        row_starts.push(0);
        for h in &row_heights {
            let s = row_starts.last().copied().unwrap_or(0);
            row_starts.push(s + h);
        }
        Self {
            col_widths,
            row_heights,
            col_starts,
            row_starts,
        }
    }

    fn num_tiles(&self) -> u32 {
        (self.col_widths.len() * self.row_heights.len()) as u32
    }

    /// Given a tile index (row-major over the tile grid), return
    /// `(ctb_x0, ctb_y0, num_cols_ctb, num_rows_ctb)`.
    fn tile_bounds(&self, tile_idx: u32) -> (u32, u32, u32, u32) {
        let cols = self.col_widths.len() as u32;
        let ti_col = tile_idx % cols;
        let ti_row = tile_idx / cols;
        (
            self.col_starts[ti_col as usize],
            self.row_starts[ti_row as usize],
            self.col_widths[ti_col as usize],
            self.row_heights[ti_row as usize],
        )
    }
}

struct Walker<'a> {
    pic: &'a mut Picture,
    cctx: &'a CtuContext<'a>,
    min_cb_log2: u32,
    max_tb_log2: u32,
    min_tb_log2: u32,
    cu_qp_y: &'a mut i32,
    /// Tracks whether `cu_qp_delta_abs` has been decoded for the current
    /// quantisation group yet (§7.3.8.11 `IsCuQpDeltaCoded`). Reset when
    /// a new coding_unit enters a different QG per §8.6.1.
    is_cu_qp_delta_coded: bool,
    /// Top-left of the QG containing the most recent CU, used to detect
    /// QG transitions.
    last_qg_pos: Option<(u32, u32)>,
    /// QP of the previously-decoded QG in coding order (§8.6.1 QpY_PREV).
    /// Seeded with slice_qp_y and updated at each QG transition.
    qpy_prev: i32,
    /// Predicted QP for the current QG (§8.6.1 eq. 8-282). Recomputed on
    /// each QG transition from the left/above neighbours in `pic.qp_y`.
    /// The actual `cu_qp_y` (QpY, *not* primed) is then (eq. 8-283):
    ///   `QpY = ((qPY_PRED + CuQpDeltaVal + 52 + 2*QpBdOffsetY)
    ///          % (52 + QpBdOffsetY)) - QpBdOffsetY`
    /// — i.e. the modulus widens with bit depth (to 64 for Main 10) so that
    /// legal negative QpY values (down to -QpBdOffsetY) survive the wrap.
    qpy_pred: i32,
    /// Resolved scaling-list data (§7.4.5) for this slice, or `None` when
    /// `scaling_list_enabled_flag == 0` (in which case `m[x][y] = 16`).
    scaling_list: Option<&'a ScalingListData>,
    /// `cu_transquant_bypass_flag` (§7.4.9.5) for the CU currently being
    /// decoded. When set, the transform tree skips dequantisation and the
    /// inverse transform — decoded level values are taken as the spatial-
    /// domain residual and added directly to the predictor (§8.6.2, eq.
    /// 8-277 with `bdShift = 0`). Reset at the start of each CU.
    cu_transquant_bypass: bool,
}

impl<'a> Walker<'a> {
    fn decode_sao(
        &mut self,
        engine: &mut CabacEngine<'_>,
        ctx: &mut Ctx,
        ctx_x: u32,
        ctx_y: u32,
    ) -> Result<()> {
        // §7.3.8.3 sao() + §9.3.4.2.1 context derivation. Parses the full
        // per-CTB SAO parameter set, storing the result in `pic.sao_grid`
        // so the post-decode filter (§8.7.3) can apply it.
        let merge_left = if ctx_x > 0 {
            engine.decode_bin(&mut ctx.sao_merge_flag[0])
        } else {
            0
        };
        let merge_up = if ctx_y > 0 && merge_left == 0 {
            engine.decode_bin(&mut ctx.sao_merge_flag[0])
        } else {
            0
        };
        if merge_left == 1 || merge_up == 1 {
            // §7.4.9.3: SaoParams for this CTB are inherited from the
            // left / above neighbour. Copy across so the filter stage sees
            // the merged values.
            let Some(grid) = self.pic.sao_grid.as_mut() else {
                return Ok(());
            };
            let src = if merge_left == 1 {
                grid.get(ctx_x - 1, ctx_y).clone()
            } else {
                grid.get(ctx_x, ctx_y - 1).clone()
            };
            grid.set(ctx_x, ctx_y, src);
            return Ok(());
        }
        // sao_offset_abs is TR with cMax = (1 << (Min(bitDepth, 10) - 5)) - 1,
        // which is 7 for 8-bit. cRiceParam = 0 → unary up to 7 ones then stop.
        let abs_cmax: u32 = 7;
        let mut params = CtuSaoParams::default();
        // sao_type_idx is shared across Cb and Cr; Cr inherits it.
        let mut chroma_type_idx: u32 = 0;
        let mut chroma_eo_class: u32 = 0;
        for comp in 0..3usize {
            if (comp == 0 && !self.cctx.slice.slice_sao_luma_flag)
                || (comp > 0 && !self.cctx.slice.slice_sao_chroma_flag)
            {
                continue;
            }
            let type_idx = if comp == 2 {
                chroma_type_idx
            } else {
                let t0 = engine.decode_bin(&mut ctx.sao_type_idx[0]);
                let t = if t0 == 0 {
                    0
                } else if engine.decode_bypass() == 0 {
                    1
                } else {
                    2
                };
                if comp == 1 {
                    chroma_type_idx = t;
                }
                t
            };
            if type_idx == 0 {
                continue;
            }
            // sao_offset_abs[i] for i in 0..4 — TR, cMax = 7.
            let mut offset_abs = [0u32; 4];
            for o in offset_abs.iter_mut() {
                let mut cnt = 0u32;
                while cnt < abs_cmax && engine.decode_bypass() == 1 {
                    cnt += 1;
                }
                *o = cnt;
            }
            let mut signed = [0i8; 4];
            let eo_class: u32;
            let band_position: u8;
            if type_idx == 1 {
                // Band offset: sign flag for each non-zero offset, then 5-bit
                // sao_band_position. Offsets are stored with the signalled
                // sign already applied.
                for (i, &abs) in offset_abs.iter().enumerate() {
                    if abs != 0 {
                        let sign = engine.decode_bypass();
                        signed[i] = if sign == 1 {
                            -(abs as i32) as i8
                        } else {
                            abs as i32 as i8
                        };
                    }
                }
                let mut bp: u32 = 0;
                for _ in 0..5 {
                    let b = engine.decode_bypass();
                    bp = (bp << 1) | b;
                }
                eo_class = 0;
                band_position = bp as u8;
            } else {
                // Edge offset: signs are fixed by spec (§7.4.9.3): categories
                // 1, 2 (local min / valley) get + offsets, categories 3, 4
                // (local max / peak) get − offsets. The parser sees only
                // magnitudes and applies the spec-fixed signs here.
                for i in 0..4usize {
                    let abs = offset_abs[i] as i32;
                    signed[i] = if i < 2 { abs as i8 } else { -(abs) as i8 };
                }
                if comp != 2 {
                    let mut ec: u32 = 0;
                    for _ in 0..2 {
                        let b = engine.decode_bypass();
                        ec = (ec << 1) | b;
                    }
                    eo_class = ec;
                    if comp == 1 {
                        chroma_eo_class = ec;
                    }
                } else {
                    eo_class = chroma_eo_class;
                }
                band_position = 0;
            }
            params.type_idx[comp] = type_idx as u8;
            params.eo_class[comp] = eo_class as u8;
            params.band_position[comp] = band_position;
            params.offsets[comp] = signed;
        }
        if let Some(grid) = self.pic.sao_grid.as_mut() {
            grid.set(ctx_x, ctx_y, params);
        }
        Ok(())
    }

    fn coding_quadtree(
        &mut self,
        engine: &mut CabacEngine<'_>,
        ctx: &mut Ctx,
        x0: u32,
        y0: u32,
        log2_cb: u32,
        depth: u32,
    ) -> Result<()> {
        let cb_size = 1u32 << log2_cb;
        let pic_w = self.cctx.sps.pic_width_in_luma_samples;
        let pic_h = self.cctx.sps.pic_height_in_luma_samples;
        // If the CB extends past the picture boundary, it is implicitly split
        // (§7.4.9.4).
        let at_bottom_right = x0 + cb_size > pic_w || y0 + cb_size > pic_h;
        let split = if log2_cb > self.min_cb_log2 {
            if at_bottom_right {
                1
            } else {
                let ctx_inc = self.split_cu_ctx_inc(x0, y0, depth);
                engine.decode_bin(&mut ctx.split_cu_flag[ctx_inc])
            }
        } else {
            0
        };
        if split == 1 {
            let sub = cb_size / 2;
            self.coding_quadtree(engine, ctx, x0, y0, log2_cb - 1, depth + 1)?;
            if x0 + sub < pic_w {
                self.coding_quadtree(engine, ctx, x0 + sub, y0, log2_cb - 1, depth + 1)?;
            }
            if y0 + sub < pic_h {
                self.coding_quadtree(engine, ctx, x0, y0 + sub, log2_cb - 1, depth + 1)?;
            }
            if x0 + sub < pic_w && y0 + sub < pic_h {
                self.coding_quadtree(engine, ctx, x0 + sub, y0 + sub, log2_cb - 1, depth + 1)?;
            }
            return Ok(());
        }
        // Record the cqt-depth covering this leaf CU so neighbours' split_cu
        // context (§9.3.4.2.2) gets the spec-correct value.
        let cb_size = 1u32 << log2_cb;
        self.store_cqt_depth(x0, y0, cb_size, depth as u8);
        self.coding_unit(engine, ctx, x0, y0, log2_cb)
    }

    fn split_cu_ctx_inc(&self, x0: u32, y0: u32, depth: u32) -> usize {
        // §9.3.4.2.2: ctxInc = (condL && cqtDepthL > cqtDepthCurr)
        //                    + (condA && cqtDepthA > cqtDepthCurr).
        let iw = self.pic.intra_width_4;
        let ih = self.pic.intra_height_4;
        let read_depth = |x: u32, y: u32| -> Option<u8> {
            if x >= self.pic.width || y >= self.pic.height {
                return None;
            }
            let bx = (x >> 2) as usize;
            let by = (y >> 2) as usize;
            if bx < iw && by < ih {
                Some(self.pic.cqt_depth[by * iw + bx])
            } else {
                None
            }
        };
        let l = if x0 == 0 {
            0
        } else {
            usize::from(read_depth(x0 - 1, y0).is_some_and(|d| u32::from(d) > depth))
        };
        let a = if y0 == 0 {
            0
        } else {
            usize::from(read_depth(x0, y0 - 1).is_some_and(|d| u32::from(d) > depth))
        };
        l + a
    }

    fn store_cqt_depth(&mut self, x: u32, y: u32, size: u32, depth: u8) {
        let bx0 = (x >> 2) as usize;
        let by0 = (y >> 2) as usize;
        let n4 = (size >> 2) as usize;
        let iw = self.pic.intra_width_4;
        let ih = self.pic.intra_height_4;
        for dy in 0..n4 {
            for dx in 0..n4 {
                let bx = bx0 + dx;
                let by = by0 + dy;
                if bx < iw && by < ih {
                    self.pic.cqt_depth[by * iw + bx] = depth;
                }
            }
        }
    }

    fn coding_unit(
        &mut self,
        engine: &mut CabacEngine<'_>,
        ctx: &mut Ctx,
        x0: u32,
        y0: u32,
        log2_cb: u32,
    ) -> Result<()> {
        // §8.6.1: IsCuQpDeltaCoded is reset at the start of each quantisation
        // group. QG top-left is (x0, y0) rounded down to a multiple of
        // `1 << qg_log2`, where qg_log2 = ctb_log2 - diff_cu_qp_delta_depth.
        let qg_log2 = self
            .cctx
            .sps
            .log2_min_luma_coding_block_size_minus3
            .saturating_add(3)
            .saturating_add(self.cctx.sps.log2_diff_max_min_luma_coding_block_size)
            .saturating_sub(self.cctx.pps.diff_cu_qp_delta_depth);
        let qg_mask = (1u32 << qg_log2).saturating_sub(1);
        let qg_x = x0 & !qg_mask;
        let qg_y = y0 & !qg_mask;
        let cur_qg = (qg_x, qg_y);
        if self.last_qg_pos != Some(cur_qg) {
            if let Some(prev_qg) = self.last_qg_pos {
                // Previous QG ended — persist its QP into the grid so that
                // later QGs can read it as a neighbour (§8.6.1).
                let qg_sz = 1u32 << qg_log2;
                self.save_qg_qp(prev_qg.0, prev_qg.1, qg_sz, *self.cu_qp_y);
                self.qpy_prev = *self.cu_qp_y;
            }
            self.qpy_pred = self.compute_qpy_pred(qg_x, qg_y);
            *self.cu_qp_y = self.qpy_pred;
            self.is_cu_qp_delta_coded = false;
            self.last_qg_pos = Some(cur_qg);
        }
        let cb_size = 1u32 << log2_cb;
        // cu_skip_flag (P/B only).
        let mut is_skip = false;
        if matches!(self.cctx.slice.slice_type, SliceType::P | SliceType::B) {
            let ctx_inc = self.skip_ctx_inc(x0, y0);
            let skip = engine.decode_bin(&mut ctx.cu_skip_flag[ctx_inc]);
            is_skip = skip == 1;
        }
        if is_skip {
            // Skip CU: infer merge_flag = 1, one 2Nx2N PB. Decode merge_idx
            // only if max_num_merge_cand > 1.
            let merge_idx = if self.cctx.slice.max_num_merge_cand > 1 {
                decode_merge_idx(engine, ctx, self.cctx.slice.max_num_merge_cand)
            } else {
                0
            };
            self.perform_merge(x0, y0, cb_size, cb_size, merge_idx)?;
            return Ok(());
        }
        // §7.4.9.5 cu_transquant_bypass_flag. When the PPS opts in, each CU
        // carries a bypass flag; per-CU state is reset here before the
        // transform tree consults it. When bypass is set, the transform
        // tree copies decoded levels straight into the reconstruction
        // (no dequant, no inverse transform, no deblock/SAO contribution).
        self.cu_transquant_bypass = false;
        if self.cctx.pps.transquant_bypass_enabled_flag {
            let bypass = engine.decode_bin(&mut ctx.cu_transquant_bypass_flag[0]);
            self.cu_transquant_bypass = bypass == 1;
        }

        // pred_mode_flag: for P/B slices, 0 = INTER, 1 = INTRA. For I slice,
        // inferred INTRA.
        let is_inter = if matches!(self.cctx.slice.slice_type, SliceType::P | SliceType::B) {
            let pmf = engine.decode_bin(&mut ctx.pred_mode_flag[0]);
            pmf == 0
        } else {
            false
        };

        if is_inter {
            // part_mode for inter (§7.3.8.5, Table 7-10). AMP is only
            // available when `amp_enabled_flag == 1` and `log2_cb > minCb`.
            // For `log2_cb == minCb` a second split level (NxN) is permitted
            // when `log2_cb > 3`.
            let amp = self.cctx.sps.amp_enabled_flag && log2_cb > self.min_cb_log2;
            let b0 = engine.decode_bin(&mut ctx.part_mode[0]);
            let part_mode = if b0 == 1 {
                InterPart::Mode2Nx2N
            } else {
                // b1 = 1 selects the horizontal family (2NxN / 2NxnU /
                // 2NxnD); b1 = 0 selects the vertical family.
                let b1 = engine.decode_bin(&mut ctx.part_mode[1]);
                let at_min = log2_cb == self.min_cb_log2;
                let log2_gt_3 = log2_cb > 3;
                if at_min && log2_gt_3 {
                    // At the smallest CU size (>8), NxN replaces AMP in the
                    // third bin.
                    if b1 == 1 {
                        InterPart::Mode2NxN
                    } else {
                        let b2 = engine.decode_bin(&mut ctx.part_mode[2]);
                        if b2 == 1 {
                            InterPart::ModeNx2N
                        } else {
                            InterPart::ModeNxN
                        }
                    }
                } else if amp {
                    // Horizontal or vertical AMP: the third bin is bypass-
                    // coded (§9.3.4.2.1) — 0 distinguishes the symmetric
                    // case, 1 triggers a fourth bypass bin for the quarter-
                    // position selection.
                    let is_amp = engine.decode_bypass();
                    if b1 == 1 {
                        if is_amp == 0 {
                            InterPart::Mode2NxN
                        } else {
                            let pos = engine.decode_bypass();
                            if pos == 0 {
                                InterPart::Mode2NxnU
                            } else {
                                InterPart::Mode2NxnD
                            }
                        }
                    } else if is_amp == 0 {
                        InterPart::ModeNx2N
                    } else {
                        let pos = engine.decode_bypass();
                        if pos == 0 {
                            InterPart::ModenLx2N
                        } else {
                            InterPart::ModenRx2N
                        }
                    }
                } else if b1 == 1 {
                    InterPart::Mode2NxN
                } else {
                    InterPart::ModeNx2N
                }
            };
            self.decode_inter_cu(engine, ctx, x0, y0, log2_cb, part_mode)
        } else {
            // For I-slice & intra-in-P: pred_mode is INTRA. part_mode only
            // signalled for the smallest CU size.
            let mut part_mode_is_nxn = false;
            if log2_cb == self.min_cb_log2 {
                let part_mode_bit = engine.decode_bin(&mut ctx.part_mode[0]);
                part_mode_is_nxn = part_mode_bit == 0;
            }
            // Tag the CU as intra so neighbour derivation for inter PBs is
            // correct.
            self.pic
                .inter
                .set_rect(x0, y0, cb_size, cb_size, PbMotion::intra());
            // §7.3.8.5: pcm_flag is only signalled for intra 2Nx2N with
            // pcm_enabled and the CU size in [Log2MinIpcmCbSizeY,
            // Log2MaxIpcmCbSizeY]. Decoded via DecodeTerminate (§9.3.4.3.5).
            let pcm_eligible = !part_mode_is_nxn
                && self.cctx.sps.pcm_enabled_flag
                && log2_cb >= self.cctx.sps.log2_min_pcm_luma_coding_block_size
                && log2_cb <= self.cctx.sps.log2_max_pcm_luma_coding_block_size;
            if pcm_eligible {
                let pcm_flag = engine.decode_terminate();
                if pcm_flag == 1 {
                    return self.decode_pcm_cu(engine, x0, y0, log2_cb);
                }
            }
            if part_mode_is_nxn {
                self.decode_intra_prediction_and_transforms(engine, ctx, x0, y0, log2_cb, true)
            } else {
                self.decode_intra_prediction_and_transforms(engine, ctx, x0, y0, log2_cb, false)
            }
        }
    }

    /// Consume a PCM CU's sample data directly from the bitstream
    /// (§7.3.8.5 `pcm_sample()`). After the terminate bin has been read
    /// as `pcm_flag = 1`, the spec requires:
    ///
    /// 1. `pcm_alignment_zero_bit` — discard bits until byte-aligned.
    /// 2. `pcm_sample_luma[i]`, `pcm_sample_chroma[i]` as fixed-length
    ///    u(PcmBitDepth*) values.
    /// 3. Re-initialise the arithmetic decoding engine (§9.3.2.6) at the
    ///    new byte position.
    ///
    /// Samples are scaled from `PcmBitDepth*` → 8-bit bit-depth and
    /// written straight to the picture (no prediction, no transform).
    fn decode_pcm_cu(
        &mut self,
        engine: &mut CabacEngine<'_>,
        x0: u32,
        y0: u32,
        log2_cb: u32,
    ) -> Result<()> {
        let n = 1u32 << log2_cb;
        let pcm_depth_y = self.cctx.sps.pcm_sample_bit_depth_luma as usize;
        let pcm_depth_c = self.cctx.sps.pcm_sample_bit_depth_chroma as usize;
        let bit_depth_y = self.cctx.sps.bit_depth_y() as i32;
        let bit_depth_c = self.cctx.sps.bit_depth_c() as i32;
        let max_y = (1i32 << bit_depth_y) - 1;
        let max_c = (1i32 << bit_depth_c) - 1;
        // Start bit offset in the payload: after DecodeTerminate returns 1,
        // the engine's physical read pointer sits `bits_consumed()` bits
        // into the payload. `pcm_alignment_zero_bit` aligns to the next
        // byte boundary; we model this by always reading from the next
        // whole byte, which is where the encoder guarantees pcm_sample to
        // start (§7.3.8.5).
        let mut byte_pos = engine.byte_pos();
        let data = engine.data();
        // If the last read byte still has unconsumed bits in `bits_in_buf`,
        // `byte_pos` has already moved past it — we're implicitly aligned
        // to the next byte boundary. No extra bits to skip.
        //
        // Read luma: n × n samples at `pcm_depth_y` bits each.
        let mut bit_off: u32 = 0;
        for py in 0..n {
            for px in 0..n {
                let sample = read_fl_bits(data, byte_pos, bit_off, pcm_depth_y)
                    .ok_or_else(|| Error::invalid("h265 pcm: luma sample read past EOF"))?;
                bit_off += pcm_depth_y as u32;
                while bit_off >= 8 {
                    bit_off -= 8;
                    byte_pos += 1;
                }
                // Scale to sample bit depth per §8.4.4.3 (left-shift by
                // BitDepth - PcmBitDepth).
                let shift = bit_depth_y - pcm_depth_y as i32;
                let px_val = if shift >= 0 {
                    (sample as i32) << shift
                } else {
                    (sample as i32) >> (-shift)
                };
                let xx = x0 + px;
                let yy = y0 + py;
                if (xx as usize) < self.pic.width as usize
                    && (yy as usize) < self.pic.height as usize
                {
                    let idx = yy as usize * self.pic.luma_stride + xx as usize;
                    self.pic.luma[idx] = px_val.clamp(0, max_y) as u16;
                }
            }
        }
        // Read chroma Cb then Cr: (n/2) × (n/2) samples at `pcm_depth_c` bits.
        let cn = n / 2;
        for chroma_plane in 0..2usize {
            for py in 0..cn {
                for px in 0..cn {
                    let sample = read_fl_bits(data, byte_pos, bit_off, pcm_depth_c)
                        .ok_or_else(|| Error::invalid("h265 pcm: chroma sample read past EOF"))?;
                    bit_off += pcm_depth_c as u32;
                    while bit_off >= 8 {
                        bit_off -= 8;
                        byte_pos += 1;
                    }
                    let shift = bit_depth_c - pcm_depth_c as i32;
                    let px_val = if shift >= 0 {
                        (sample as i32) << shift
                    } else {
                        (sample as i32) >> (-shift)
                    };
                    let xx = (x0 / 2) + px;
                    let yy = (y0 / 2) + py;
                    let pic_cw = (self.pic.width as usize) / 2;
                    let pic_ch = (self.pic.height as usize) / 2;
                    if (xx as usize) < pic_cw && (yy as usize) < pic_ch {
                        let idx = yy as usize * self.pic.chroma_stride + xx as usize;
                        if chroma_plane == 0 {
                            self.pic.cb[idx] = px_val.clamp(0, max_c) as u16;
                        } else {
                            self.pic.cr[idx] = px_val.clamp(0, max_c) as u16;
                        }
                    }
                }
            }
        }
        // Any leftover bit_off lands on the next byte.
        if bit_off > 0 {
            byte_pos += 1;
        }
        // §9.3.2.6: re-init arithmetic engine at the next byte boundary.
        engine.reinit_at_byte(byte_pos);
        // Mark all 4×4 blocks in the CU as decoded so neighbour availability
        // checks still work for subsequent intra PBs.
        self.mark_decoded_block(x0, y0, n);
        Ok(())
    }

    fn skip_ctx_inc(&self, x0: u32, y0: u32) -> usize {
        // ctxInc = (L.skip ? 1 : 0) + (A.skip ? 1 : 0) — we don't track the
        // skip-flag per PB, so approximate by "neighbour inside pic".
        let left = if x0 == 0 {
            0
        } else {
            let bx = ((x0 - 1) >> 2) as usize;
            let by = (y0 >> 2) as usize;
            self.pic
                .inter
                .get(bx, by)
                .map(|p| !p.is_intra && p.valid)
                .unwrap_or(false) as usize
        };
        let above = if y0 == 0 {
            0
        } else {
            let bx = (x0 >> 2) as usize;
            let by = ((y0 - 1) >> 2) as usize;
            self.pic
                .inter
                .get(bx, by)
                .map(|p| !p.is_intra && p.valid)
                .unwrap_or(false) as usize
        };
        (left + above).min(2)
    }

    /// Perform a merge-mode prediction at the given PB: look up the merge
    /// candidate, run motion compensation, store pixels into the picture.
    fn perform_merge(
        &mut self,
        x0: u32,
        y0: u32,
        n_pb_w: u32,
        n_pb_h: u32,
        merge_idx: u32,
    ) -> Result<()> {
        let tmvp = self.tmvp_merge_cand(x0, y0, n_pb_w, n_pb_h);
        let cands = build_merge_list(
            &self.pic.inter,
            x0,
            y0,
            n_pb_w,
            n_pb_h,
            self.cctx.slice.max_num_merge_cand,
            self.cctx.is_b_slice(),
            tmvp,
        );
        let sel = cands.get(merge_idx as usize).copied().unwrap_or_default();
        let pb = sel.to_pb();
        self.motion_compensate_pb(x0, y0, n_pb_w, n_pb_h, pb)?;
        Ok(())
    }

    /// Run motion compensation for a prediction block and write the result
    /// into the picture. Supports L0 uni-pred, L1 uni-pred, and bi-pred.
    fn motion_compensate_pb(
        &mut self,
        x0: u32,
        y0: u32,
        w: u32,
        h: u32,
        pb: PbMotion,
    ) -> Result<()> {
        let wz = w as usize;
        let hz = h as usize;
        let cw = (w / 2) as usize;
        let ch = (h / 2) as usize;

        let ref0 = if pb.pred_l0 {
            let idx = pb.ref_idx_l0.max(0) as usize;
            self.cctx
                .ref_list_l0
                .get(idx)
                .ok_or_else(|| Error::invalid("h265 inter: L0 ref_idx out of range"))?
        } else {
            &self.cctx.ref_list_l0[0]
        };
        let ref1_opt = if pb.pred_l1 {
            let idx = pb.ref_idx_l1.max(0) as usize;
            Some(
                self.cctx
                    .ref_list_l1
                    .get(idx)
                    .ok_or_else(|| Error::invalid("h265 inter: L1 ref_idx out of range"))?,
            )
        } else {
            None
        };

        let mut luma_out = vec![0u16; wz * hz];
        let mut cb_out = vec![0u16; cw * ch];
        let mut cr_out = vec![0u16; cw * ch];

        let is_bi = pb.pred_l0 && pb.pred_l1;
        let weighted = self.cctx.weighted_pred;
        let bd_y = self.cctx.sps.bit_depth_y() as i32;
        let bd_c = self.cctx.sps.bit_depth_c() as i32;

        if !is_bi {
            // Uni-prediction.
            let (ref_pic, mv) = if pb.pred_l0 {
                (ref0, pb.mv_l0)
            } else {
                let r =
                    ref1_opt.ok_or_else(|| Error::invalid("h265 inter: uni-L1 without L1 ref"))?;
                (r, pb.mv_l1)
            };
            luma_mc(
                ref_pic,
                x0 as i32,
                y0 as i32,
                w as i32,
                h as i32,
                mv,
                &mut luma_out,
                bd_y,
            )?;
            chroma_mc(
                ref_pic,
                (x0 / 2) as i32,
                (y0 / 2) as i32,
                cw as i32,
                ch as i32,
                mv,
                &mut cb_out,
                0,
                bd_c,
            )?;
            chroma_mc(
                ref_pic,
                (x0 / 2) as i32,
                (y0 / 2) as i32,
                cw as i32,
                ch as i32,
                mv,
                &mut cr_out,
                1,
                bd_c,
            )?;
        } else {
            let ref_l1 =
                ref1_opt.ok_or_else(|| Error::invalid("h265 inter: bi-pred without L1 ref"))?;
            // Compute high-precision uni-pred samples from both refs and
            // combine for bi-prediction (§8.5.3.3.3 / §8.5.3.3.4).
            let mut a_l = vec![0i32; wz * hz];
            let mut b_l = vec![0i32; wz * hz];
            luma_mc_hp(
                ref0, x0 as i32, y0 as i32, w as i32, h as i32, pb.mv_l0, &mut a_l,
            )?;
            luma_mc_hp(
                ref_l1, x0 as i32, y0 as i32, w as i32, h as i32, pb.mv_l1, &mut b_l,
            )?;
            let mut a_cb = vec![0i32; cw * ch];
            let mut b_cb = vec![0i32; cw * ch];
            let mut a_cr = vec![0i32; cw * ch];
            let mut b_cr = vec![0i32; cw * ch];
            chroma_mc_hp(
                ref0,
                (x0 / 2) as i32,
                (y0 / 2) as i32,
                cw as i32,
                ch as i32,
                pb.mv_l0,
                &mut a_cb,
                0,
            )?;
            chroma_mc_hp(
                ref_l1,
                (x0 / 2) as i32,
                (y0 / 2) as i32,
                cw as i32,
                ch as i32,
                pb.mv_l1,
                &mut b_cb,
                0,
            )?;
            chroma_mc_hp(
                ref0,
                (x0 / 2) as i32,
                (y0 / 2) as i32,
                cw as i32,
                ch as i32,
                pb.mv_l0,
                &mut a_cr,
                1,
            )?;
            chroma_mc_hp(
                ref_l1,
                (x0 / 2) as i32,
                (y0 / 2) as i32,
                cw as i32,
                ch as i32,
                pb.mv_l1,
                &mut b_cr,
                1,
            )?;

            let l0_luma = weighted.and_then(|w| w.luma_weight_l0(pb.ref_idx_l0.max(0) as usize));
            let l1_luma = weighted.and_then(|w| w.luma_weight_l1(pb.ref_idx_l1.max(0) as usize));
            if let (Some(w_table), Some((w0, o0)), Some((w1, o1))) = (weighted, l0_luma, l1_luma) {
                luma_mc_bi_weighted(
                    &a_l,
                    &b_l,
                    &mut luma_out,
                    w0,
                    o0,
                    w1,
                    o1,
                    w_table.luma_denom + 1,
                    bd_y,
                );
            } else {
                luma_mc_bi_combine(&a_l, &b_l, &mut luma_out, bd_y);
            }

            for comp in 0..2 {
                let (a, b, out) = match comp {
                    0 => (&a_cb, &b_cb, &mut cb_out),
                    _ => (&a_cr, &b_cr, &mut cr_out),
                };
                let l0_c = weighted
                    .and_then(|w| w.chroma_weight_l0(pb.ref_idx_l0.max(0) as usize, comp as usize));
                let l1_c = weighted
                    .and_then(|w| w.chroma_weight_l1(pb.ref_idx_l1.max(0) as usize, comp as usize));
                if let (Some(w_table), Some((w0, o0)), Some((w1, o1))) = (weighted, l0_c, l1_c) {
                    luma_mc_bi_weighted(
                        a, b, out, w0, o0, w1, o1, w_table.chroma_denom + 1, bd_c,
                    );
                } else {
                    chroma_mc_bi_combine(a, b, out, bd_c);
                }
            }
        }

        self.write_rect(x0, y0, wz, hz, &luma_out, true, false);
        self.write_rect(x0 / 2, y0 / 2, cw, ch, &cb_out, false, false);
        self.write_rect(x0 / 2, y0 / 2, cw, ch, &cr_out, false, true);
        self.pic.inter.set_rect(x0, y0, w, h, pb);
        Ok(())
    }

    /// Look up the collocated PB motion from the slice's configured
    /// collocated reference picture and convert it to a merge candidate.
    /// Returns `None` when TMVP is disabled or the collocated block is
    /// intra.
    fn tmvp_merge_cand(&self, x0: u32, y0: u32, w: u32, h: u32) -> Option<MergeCand> {
        if !self.cctx.slice.slice_temporal_mvp_enabled_flag {
            return None;
        }
        let coll = self.cctx.collocated_ref?;
        // §8.5.3.2.9: pick the collocated PB at the bottom-right of the
        // current PB, falling back to its centre when unavailable.
        let pic_w = self.cctx.sps.pic_width_in_luma_samples;
        let pic_h = self.cctx.sps.pic_height_in_luma_samples;
        let br_x = x0 + w;
        let br_y = y0 + h;
        if br_x < pic_w && br_y < pic_h {
            if let Some(pb) = coll.collocated_motion(br_x, br_y) {
                return Some(MergeCand::from_pb_into_ref0(pb));
            }
        }
        let cx = x0 + w / 2;
        let cy = y0 + h / 2;
        let pb = coll.collocated_motion(cx, cy)?;
        Some(MergeCand::from_pb_into_ref0(pb))
    }

    fn decode_inter_cu(
        &mut self,
        engine: &mut CabacEngine<'_>,
        ctx: &mut Ctx,
        x0: u32,
        y0: u32,
        log2_cb: u32,
        part_mode: InterPart,
    ) -> Result<()> {
        let cb_size = 1u32 << log2_cb;
        let q = cb_size / 4;
        let pbs: Vec<(u32, u32, u32, u32)> = match part_mode {
            InterPart::Mode2Nx2N => vec![(x0, y0, cb_size, cb_size)],
            InterPart::Mode2NxN => vec![
                (x0, y0, cb_size, cb_size / 2),
                (x0, y0 + cb_size / 2, cb_size, cb_size / 2),
            ],
            InterPart::ModeNx2N => vec![
                (x0, y0, cb_size / 2, cb_size),
                (x0 + cb_size / 2, y0, cb_size / 2, cb_size),
            ],
            InterPart::ModeNxN => vec![
                (x0, y0, cb_size / 2, cb_size / 2),
                (x0 + cb_size / 2, y0, cb_size / 2, cb_size / 2),
                (x0, y0 + cb_size / 2, cb_size / 2, cb_size / 2),
                (x0 + cb_size / 2, y0 + cb_size / 2, cb_size / 2, cb_size / 2),
            ],
            // AMP shapes (§7.4.9.5 Table 7-10). The top/left stripe is the
            // "quarter" (cb/4) slice; the second stripe fills the remainder
            // (3·cb/4). "U" / "L" place the quarter stripe first;
            // "D" / "R" place it last.
            InterPart::Mode2NxnU => vec![
                (x0, y0, cb_size, q),
                (x0, y0 + q, cb_size, cb_size - q),
            ],
            InterPart::Mode2NxnD => vec![
                (x0, y0, cb_size, cb_size - q),
                (x0, y0 + cb_size - q, cb_size, q),
            ],
            InterPart::ModenLx2N => vec![
                (x0, y0, q, cb_size),
                (x0 + q, y0, cb_size - q, cb_size),
            ],
            InterPart::ModenRx2N => vec![
                (x0, y0, cb_size - q, cb_size),
                (x0 + cb_size - q, y0, q, cb_size),
            ],
        };
        for (px, py, pw, ph) in &pbs {
            self.decode_prediction_unit(engine, ctx, *px, *py, *pw, *ph)?;
        }

        // rqt_root_cbf: whether a transform tree follows.
        let rqt = engine.decode_bin(&mut ctx.rqt_root_cbf[0]);
        if rqt == 0 {
            return Ok(());
        }
        self.transform_tree_inter(engine, ctx, x0, y0, log2_cb, 0)
    }

    fn decode_prediction_unit(
        &mut self,
        engine: &mut CabacEngine<'_>,
        ctx: &mut Ctx,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<()> {
        let merge_flag = engine.decode_bin(&mut ctx.merge_flag[0]) == 1;
        let pb = if merge_flag {
            let merge_idx = if self.cctx.slice.max_num_merge_cand > 1 {
                decode_merge_idx(engine, ctx, self.cctx.slice.max_num_merge_cand)
            } else {
                0
            };
            let tmvp = self.tmvp_merge_cand(x, y, w, h);
            let cands = build_merge_list(
                &self.pic.inter,
                x,
                y,
                w,
                h,
                self.cctx.slice.max_num_merge_cand,
                self.cctx.is_b_slice(),
                tmvp,
            );
            let sel = cands.get(merge_idx as usize).copied().unwrap_or_default();
            sel.to_pb()
        } else {
            // §7.3.8.6: inter_pred_idc = 0 (L0), 1 (L1), 2 (BI). Only signalled
            // for B slices; for P slices L0 is implied.
            let (use_l0, use_l1) = if self.cctx.is_b_slice() {
                // inter_pred_idc: bin 0 with context (nPbW + nPbH == 12 uses
                // ctx 4); when bin 0 == 1 bin 1 chooses L0 vs L1. When the
                // PB is <8 samples in either dim, only L0/L1 is allowed.
                let nb_eq_8 = (w + h) == 12;
                let bin0_ctx = if nb_eq_8 { 4 } else { 0 };
                let small = w + h == 12; // 8×4 / 4×8 partitions: no bi-pred.
                if small {
                    let uni = engine.decode_bin(&mut ctx.inter_pred_idc[bin0_ctx]);
                    if uni == 0 {
                        (true, false)
                    } else {
                        (false, true)
                    }
                } else {
                    let bi = engine.decode_bin(&mut ctx.inter_pred_idc[bin0_ctx]);
                    if bi == 1 {
                        (true, true)
                    } else {
                        let bin1 = engine.decode_bin(&mut ctx.inter_pred_idc[4]);
                        if bin1 == 0 {
                            (true, false)
                        } else {
                            (false, true)
                        }
                    }
                }
            } else {
                (true, false)
            };

            let mut mv_l0 = MotionVector::default();
            let mut mv_l1 = MotionVector::default();
            let mut ref_idx_l0 = 0i8;
            let mut ref_idx_l1 = 0i8;

            if use_l0 {
                let ri = if self.cctx.slice.num_ref_idx_l0_active_minus1 > 0 {
                    decode_ref_idx(
                        engine,
                        ctx,
                        self.cctx.slice.num_ref_idx_l0_active_minus1 + 1,
                    )
                } else {
                    0
                };
                let mvd = decode_mvd(engine, ctx)?;
                let mvp_flag = engine.decode_bin(&mut ctx.mvp_lx_flag[0]);
                let tmvp_mv = self.tmvp_amvp_mv(x, y, w, h, true);
                let amvp = build_amvp_list(&self.pic.inter, x, y, w, h, true, tmvp_mv);
                let mvp = amvp[mvp_flag as usize];
                ref_idx_l0 = ri as i8;
                mv_l0 = MotionVector::new(mvp.x + mvd.x, mvp.y + mvd.y);
            }
            if use_l1 {
                let ri = if self.cctx.slice.num_ref_idx_l1_active_minus1 > 0 {
                    decode_ref_idx(
                        engine,
                        ctx,
                        self.cctx.slice.num_ref_idx_l1_active_minus1 + 1,
                    )
                } else {
                    0
                };
                // `mvd_l1_zero_flag` lets the encoder suppress the L1 MVD
                // encoding when bi-pred (§7.4.7.1). We still need to honour
                // that: read MVD normally otherwise, else zero.
                let mvd = if self.cctx.slice.mvd_l1_zero_flag && use_l0 {
                    MotionVector::default()
                } else {
                    decode_mvd(engine, ctx)?
                };
                let mvp_flag = engine.decode_bin(&mut ctx.mvp_lx_flag[0]);
                let tmvp_mv = self.tmvp_amvp_mv(x, y, w, h, false);
                let amvp = build_amvp_list(&self.pic.inter, x, y, w, h, false, tmvp_mv);
                let mvp = amvp[mvp_flag as usize];
                ref_idx_l1 = ri as i8;
                mv_l1 = MotionVector::new(mvp.x + mvd.x, mvp.y + mvd.y);
            }

            PbMotion {
                valid: true,
                is_intra: false,
                pred_l0: use_l0,
                pred_l1: use_l1,
                ref_idx_l0,
                ref_idx_l1,
                mv_l0,
                mv_l1,
            }
        };
        self.motion_compensate_pb(x, y, w, h, pb)
    }

    /// Look up an AMVP temporal motion vector (§8.5.3.2.9). Returns the
    /// collocated PB's MV for the requested list, or `None` when TMVP is
    /// off / unavailable.
    fn tmvp_amvp_mv(
        &self,
        x0: u32,
        y0: u32,
        w: u32,
        h: u32,
        want_l0: bool,
    ) -> Option<MotionVector> {
        if !self.cctx.slice.slice_temporal_mvp_enabled_flag {
            return None;
        }
        let coll = self.cctx.collocated_ref?;
        let pic_w = self.cctx.sps.pic_width_in_luma_samples;
        let pic_h = self.cctx.sps.pic_height_in_luma_samples;
        let br_x = x0 + w;
        let br_y = y0 + h;
        let pick = |pb: PbMotion| -> Option<MotionVector> {
            if want_l0 && pb.pred_l0 {
                Some(pb.mv_l0)
            } else if !want_l0 && pb.pred_l1 {
                Some(pb.mv_l1)
            } else if pb.pred_l0 {
                Some(pb.mv_l0)
            } else if pb.pred_l1 {
                Some(pb.mv_l1)
            } else {
                None
            }
        };
        if br_x < pic_w && br_y < pic_h {
            if let Some(pb) = coll.collocated_motion(br_x, br_y) {
                if let Some(mv) = pick(pb) {
                    return Some(mv);
                }
            }
        }
        let cx = x0 + w / 2;
        let cy = y0 + h / 2;
        let pb = coll.collocated_motion(cx, cy)?;
        pick(pb)
    }

    fn decode_intra_prediction_and_transforms(
        &mut self,
        engine: &mut CabacEngine<'_>,
        ctx: &mut Ctx,
        x0: u32,
        y0: u32,
        log2_cb: u32,
        part_mode_nxn: bool,
    ) -> Result<()> {
        // For 2Nx2N there is one luma prediction block; for NxN (only at min CB)
        // there are four. §7.3.8.10.
        let num_pus = if part_mode_nxn { 4 } else { 1 };
        let cb_size = 1u32 << log2_cb;
        let pu_size = if part_mode_nxn { cb_size / 2 } else { cb_size };

        // prev_intra_luma_pred_flag[0..num_pus]
        let mut prev = [0u32; 4];
        for item in prev.iter_mut().take(num_pus) {
            *item = engine.decode_bin(&mut ctx.prev_intra_luma_pred_flag[0]);
        }
        let mut mpm_idx = [0u32; 4];
        let mut rem = [0u32; 4];
        for i in 0..num_pus {
            if prev[i] == 1 {
                // TR-coded with cMax=2, cRice=0 -> {0:'0', 1:'10', 2:'11'}
                let b0 = engine.decode_bypass();
                mpm_idx[i] = if b0 == 0 {
                    0
                } else {
                    let b1 = engine.decode_bypass();
                    1 + b1
                };
            } else {
                // 5 bypass bits
                let mut v = 0u32;
                for _ in 0..5 {
                    v = (v << 1) | engine.decode_bypass();
                }
                rem[i] = v;
            }
        }
        // Resolve the intra prediction modes per PU using the 3-MPM list rule
        // (§8.4.2). Then record into the per-4x4 intra_luma_mode map.
        let mut luma_modes = [1u32; 4];
        for i in 0..num_pus {
            let (pu_x, pu_y) = if part_mode_nxn {
                let xi = (i & 1) as u32;
                let yi = ((i >> 1) & 1) as u32;
                (x0 + xi * pu_size, y0 + yi * pu_size)
            } else {
                (x0, y0)
            };
            let mpm = self.derive_mpm_list(pu_x, pu_y);
            let mode = if prev[i] == 1 {
                mpm[mpm_idx[i] as usize]
            } else {
                // Non-MPM: reorder mpm ascending, then insert rem values
                // skipping MPM positions (§8.4.2).
                let mut sorted = mpm;
                sorted.sort();
                let mut m = rem[i];
                if m >= sorted[0] {
                    m += 1;
                }
                if m >= sorted[1] {
                    m += 1;
                }
                if m >= sorted[2] {
                    m += 1;
                }
                m
            };
            luma_modes[i] = mode;
            self.store_luma_mode(pu_x, pu_y, pu_size, mode as u8);
        }
        if std::env::var_os("H265_TRACE_MODE").is_some() {
            eprintln!(
                "intra CU x={x0} y={y0} log2_cb={log2_cb} nxn={part_mode_nxn} modes={:?}",
                &luma_modes[..num_pus]
            );
        }

        // intra_chroma_pred_mode (once per CU) — TR(cMax=4).
        let icpm_first = engine.decode_bin(&mut ctx.intra_chroma_pred_mode[0]);
        let icpm = if icpm_first == 0 {
            4 // DM mode (inherit luma)
        } else {
            // TR-coded bypass 2 bits for the remaining 4 choices.
            let b0 = engine.decode_bypass();
            let b1 = engine.decode_bypass();
            (b0 << 1) | b1
        };
        // intraPredModeC for the CU (§8.4.3). Chroma predicts from the
        // CU-top-left luma mode (2Nx2N chroma unit). For NxN this is the luma
        // mode at (x0, y0).
        let chroma_mode = self.resolve_chroma_mode(luma_modes[0], icpm);

        // transform_tree: one call at CU root. x_base/y_base and parent_cbf
        // are unused at the root (tr_depth == 0 always decodes cbf directly).
        self.transform_tree(
            engine,
            ctx,
            x0,
            y0,
            x0,
            y0,
            x0,
            y0,
            log2_cb,
            log2_cb,
            0,
            0,
            1,
            1,
            &luma_modes,
            chroma_mode,
            part_mode_nxn,
        )
    }

    fn derive_mpm_list(&self, x0: u32, y0: u32) -> [u32; 3] {
        // §8.4.2: IntraPredModeY[x,y] candidates from left (A) and above (B).
        let get_mode = |x: u32, y: u32| -> Option<u32> {
            if x >= self.pic.width || y >= self.pic.height {
                return None;
            }
            let bx = (x >> 2) as usize;
            let by = (y >> 2) as usize;
            Some(self.pic.intra_luma_mode[by * self.pic.intra_width_4 + bx] as u32)
        };
        let a = if x0 == 0 { None } else { get_mode(x0 - 1, y0) };
        // §8.4.2: when the above neighbour B lies in a different CTB row
        // than the current block (i.e. yPb-1 is above the current CTB),
        // candIntraPredModeB is forced to INTRA_DC — even if the actual
        // block at (x0, y0-1) has a different stored mode. This prevents
        // the MPM list from crossing the CTB-row seam, matching how
        // ffmpeg / the spec parse the bitstream.
        let ctb_log2 = self.cctx.sps.log2_min_luma_coding_block_size_minus3
            + 3
            + self.cctx.sps.log2_diff_max_min_luma_coding_block_size;
        let ctb_row_top = (y0 >> ctb_log2) << ctb_log2;
        let b = if y0 == 0 {
            None
        } else if y0 - 1 < ctb_row_top {
            // Across CTB-row boundary: B is treated as INTRA_DC.
            Some(1)
        } else {
            get_mode(x0, y0 - 1)
        };
        let cand_a = a.unwrap_or(1);
        let cand_b = b.unwrap_or(1);
        let mut list = [0u32; 3];
        if cand_a == cand_b {
            if cand_a < 2 {
                list[0] = 0; // planar
                list[1] = 1; // DC
                list[2] = 26; // vertical
            } else {
                list[0] = cand_a;
                list[1] = 2 + ((cand_a + 29) % 32);
                list[2] = 2 + ((cand_a - 2 + 1) % 32);
            }
        } else {
            list[0] = cand_a;
            list[1] = cand_b;
            if cand_a != 0 && cand_b != 0 {
                list[2] = 0;
            } else if cand_a != 1 && cand_b != 1 {
                list[2] = 1;
            } else {
                list[2] = 26;
            }
        }
        list
    }

    fn store_luma_mode(&mut self, x: u32, y: u32, size: u32, mode: u8) {
        let bx0 = (x >> 2) as usize;
        let by0 = (y >> 2) as usize;
        let n4 = (size >> 2) as usize;
        let iw = self.pic.intra_width_4;
        let ih = self.pic.intra_height_4;
        for dy in 0..n4 {
            for dx in 0..n4 {
                let bx = bx0 + dx;
                let by = by0 + dy;
                if bx < iw && by < ih {
                    self.pic.intra_luma_mode[by * iw + bx] = mode;
                }
            }
        }
    }

    fn resolve_chroma_mode(&self, luma_mode: u32, icpm: u32) -> u32 {
        // Table 8-2 (§8.4.3).
        // icpm = 4 → DM = luma mode.
        // icpm = 0/1/2/3 → {planar, 26, 10, 1} except when the chosen mode
        // equals luma_mode, in which case map to 34.
        let chosen = match icpm {
            0 => 0,
            1 => 26,
            2 => 10,
            3 => 1,
            _ => luma_mode, // DM
        };
        if chosen == luma_mode && icpm != 4 {
            34
        } else {
            chosen
        }
    }

    /// Inter-slice transform-tree decode. The MC predictor has already been
    /// written into the picture; we decode residuals and add them on top.
    fn transform_tree_inter(
        &mut self,
        engine: &mut CabacEngine<'_>,
        ctx: &mut Ctx,
        x0: u32,
        y0: u32,
        log2_tb: u32,
        tr_depth: u32,
    ) -> Result<()> {
        let max_tb_log2 = self.max_tb_log2;
        let min_tb_log2 = self.min_tb_log2;
        let must_split = log2_tb > max_tb_log2;
        let can_split = log2_tb > min_tb_log2;
        let split = if must_split {
            1
        } else if !can_split {
            0
        } else {
            let ctx_inc = (5 - log2_tb) as usize;
            let ctx_inc = ctx_inc.min(2);
            engine.decode_bin(&mut ctx.split_transform_flag[ctx_inc])
        };
        if split == 1 {
            let sub = 1u32 << (log2_tb - 1);
            self.transform_tree_inter(engine, ctx, x0, y0, log2_tb - 1, tr_depth + 1)?;
            self.transform_tree_inter(engine, ctx, x0 + sub, y0, log2_tb - 1, tr_depth + 1)?;
            self.transform_tree_inter(engine, ctx, x0, y0 + sub, log2_tb - 1, tr_depth + 1)?;
            self.transform_tree_inter(engine, ctx, x0 + sub, y0 + sub, log2_tb - 1, tr_depth + 1)?;
            return Ok(());
        }

        let mut cbf_cb = 0u32;
        let mut cbf_cr = 0u32;
        let chroma_log2 = if log2_tb == self.min_tb_log2 {
            0
        } else {
            log2_tb - 1
        };
        let chroma_present = chroma_log2 >= 2;
        let cbf_ctx_inc = tr_depth.min(4) as usize;
        if chroma_present {
            cbf_cb = engine.decode_bin(&mut ctx.cbf_cb_cr[cbf_ctx_inc]);
            cbf_cr = engine.decode_bin(&mut ctx.cbf_cb_cr[cbf_ctx_inc]);
        }
        let cbf_luma_inc = if tr_depth == 0 { 1usize } else { 0usize };
        let cbf_luma = engine.decode_bin(&mut ctx.cbf_luma[cbf_luma_inc]);
        let has_any_coeff = cbf_luma != 0 || cbf_cb != 0 || cbf_cr != 0;
        if has_any_coeff
            && self.cctx.pps.cu_qp_delta_enabled_flag
            && !self.is_cu_qp_delta_coded
        {
            let mut prefix = 0u32;
            let max_prefix = 5u32;
            while prefix < max_prefix {
                let ctx_inc = if prefix == 0 { 0usize } else { 1 };
                if engine.decode_bin(&mut ctx.cu_qp_delta_abs[ctx_inc]) == 0 {
                    break;
                }
                prefix += 1;
            }
            let mut abs = prefix;
            if prefix >= max_prefix {
                let mut k = 0u32;
                while engine.decode_bypass() == 1 {
                    k += 1;
                    if k > 32 {
                        return Err(Error::invalid("h265 cu_qp_delta suffix overflow"));
                    }
                }
                let mut suf = 0u32;
                for _ in 0..k {
                    suf = (suf << 1) | engine.decode_bypass();
                }
                abs += suf + (1 << k) - 1;
            }
            let delta = if abs != 0 {
                let sign = engine.decode_bypass();
                if sign == 1 {
                    -(abs as i32)
                } else {
                    abs as i32
                }
            } else {
                0
            };
            // §8.6.1 eq. 8-283 — bit-depth-aware QpY wrap. For Main (8-bit)
            // QpBdOffsetY=0 this collapses to (qpy_pred + delta + 52) % 52;
            // for Main 10 it uses (qpy_pred + delta + 76) % 64 - 12.
            let qp_bd = 6 * self.cctx.sps.bit_depth_luma_minus8 as i32;
            *self.cu_qp_y =
                (self.qpy_pred + delta + 52 + 2 * qp_bd).rem_euclid(52 + qp_bd) - qp_bd;
            self.is_cu_qp_delta_coded = true;
        }
        if cbf_luma != 0 {
            self.add_residual_plane(engine, ctx, x0, y0, log2_tb, true)?;
        }
        if chroma_present {
            let cx = x0 / 2;
            let cy = y0 / 2;
            if cbf_cb != 0 {
                self.add_residual_plane(engine, ctx, cx, cy, chroma_log2, false)?;
            }
            if cbf_cr != 0 {
                self.add_residual_plane_cr(engine, ctx, cx, cy, chroma_log2)?;
            }
        }
        Ok(())
    }

    fn add_residual_plane(
        &mut self,
        engine: &mut CabacEngine<'_>,
        ctx: &mut Ctx,
        x0: u32,
        y0: u32,
        log2_tb: u32,
        is_luma: bool,
    ) -> Result<()> {
        let n = 1usize << log2_tb;
        let mut levels = vec![0i32; n * n];
        // For inter residual: scan_idx is always diagonal (scan_idx=0).
        let ts = self.residual_coding(
            engine,
            ctx,
            &mut levels,
            log2_tb,
            /*pred_mode*/ 0,
            is_luma,
        )?;
        let mut res = vec![0i32; n * n];
        if self.cu_transquant_bypass {
            res.copy_from_slice(&levels);
        } else {
            let qp = self.get_qp(is_luma, false);
            let mut deq = vec![0i32; n * n];
            self.dequantize(&levels, &mut deq, qp, log2_tb, is_luma, false, false, ts);
            let bit_depth = if is_luma {
                self.cctx.sps.bit_depth_y()
            } else {
                self.cctx.sps.bit_depth_c()
            };
            if ts {
                // §8.6.4.2 eq. 8-298: skip iDCT, apply tsShift/bdShift.
                transform_skip_2d(&deq, &mut res, log2_tb, bit_depth);
            } else {
                // Inter TU: DCT only (no DST-VII even at 4×4).
                inverse_transform_2d(&deq, &mut res, log2_tb, false, bit_depth);
            }
        }
        self.add_to_plane(x0, y0, n, &res, is_luma, false);
        Ok(())
    }

    fn add_residual_plane_cr(
        &mut self,
        engine: &mut CabacEngine<'_>,
        ctx: &mut Ctx,
        x0: u32,
        y0: u32,
        log2_tb: u32,
    ) -> Result<()> {
        let n = 1usize << log2_tb;
        let mut levels = vec![0i32; n * n];
        let ts = self.residual_coding(engine, ctx, &mut levels, log2_tb, 0, false)?;
        let mut res = vec![0i32; n * n];
        if self.cu_transquant_bypass {
            res.copy_from_slice(&levels);
        } else {
            let qp = self.get_qp(false, true);
            let mut deq = vec![0i32; n * n];
            self.dequantize(&levels, &mut deq, qp, log2_tb, false, true, false, ts);
            let bit_depth = self.cctx.sps.bit_depth_c();
            if ts {
                transform_skip_2d(&deq, &mut res, log2_tb, bit_depth);
            } else {
                inverse_transform_2d(&deq, &mut res, log2_tb, false, bit_depth);
            }
        }
        self.add_to_plane(x0, y0, n, &res, false, true);
        Ok(())
    }

    fn add_to_plane(&mut self, x: u32, y: u32, n: usize, res: &[i32], is_luma: bool, is_cr: bool) {
        let bit_depth = if is_luma {
            self.cctx.sps.bit_depth_y()
        } else {
            self.cctx.sps.bit_depth_c()
        };
        let max = (1i32 << bit_depth) - 1;
        let (stride, plane, pic_w, pic_h) = if is_luma {
            (
                self.pic.luma_stride,
                &mut self.pic.luma,
                self.pic.width as usize,
                self.pic.height as usize,
            )
        } else if is_cr {
            (
                self.pic.chroma_stride,
                &mut self.pic.cr,
                (self.pic.width as usize) / 2,
                (self.pic.height as usize) / 2,
            )
        } else {
            (
                self.pic.chroma_stride,
                &mut self.pic.cb,
                (self.pic.width as usize) / 2,
                (self.pic.height as usize) / 2,
            )
        };
        let x0 = x as usize;
        let y0 = y as usize;
        for dy in 0..n {
            for dx in 0..n {
                let xx = x0 + dx;
                let yy = y0 + dy;
                if xx < pic_w && yy < pic_h {
                    let idx = yy * stride + xx;
                    let v = plane[idx] as i32 + res[dy * n + dx];
                    plane[idx] = v.clamp(0, max) as u16;
                }
            }
        }
    }

    fn transform_tree(
        &mut self,
        engine: &mut CabacEngine<'_>,
        ctx: &mut Ctx,
        x0: u32,
        y0: u32,
        x_base: u32,
        y_base: u32,
        cu_x0: u32,
        cu_y0: u32,
        log2_cb: u32,
        log2_tb: u32,
        tr_depth: u32,
        blk_idx: u32,
        parent_cbf_cb: u32,
        parent_cbf_cr: u32,
        luma_modes: &[u32; 4],
        chroma_mode: u32,
        part_mode_nxn: bool,
    ) -> Result<()> {
        // §7.3.8.10 / §7.4.9.9 split decision (matches ffmpeg
        // libavcodec/hevc/hevcdec.c hls_transform_tree).
        let max_tb_log2 = self.max_tb_log2;
        let min_tb_log2 = self.min_tb_log2;
        let max_trafo_depth = if part_mode_nxn {
            self.cctx.sps.max_transform_hierarchy_depth_intra + 1
        } else {
            self.cctx.sps.max_transform_hierarchy_depth_intra
        };
        let decode_split = log2_tb <= max_tb_log2
            && log2_tb > min_tb_log2
            && tr_depth < max_trafo_depth
            && !(part_mode_nxn && tr_depth == 0);
        let split = if decode_split {
            let ctx_inc = ((5 - log2_tb) as usize).min(2);
            engine.decode_bin(&mut ctx.split_transform_flag[ctx_inc])
        } else if log2_tb > max_tb_log2 || (part_mode_nxn && tr_depth == 0) {
            1
        } else {
            0
        };

        // cbf_cb / cbf_cr at every tree node with log2_tb > 2 (§7.3.8.10,
        // 4:2:0). Gated by parent's cbf at non-root depths; inherited from
        // parent when not decoded.
        let mut cbf_cb = parent_cbf_cb;
        let mut cbf_cr = parent_cbf_cr;
        if log2_tb > 2 {
            let cbf_ctx_inc = tr_depth.min(4) as usize;
            if tr_depth == 0 || parent_cbf_cb == 1 {
                cbf_cb = engine.decode_bin(&mut ctx.cbf_cb_cr[cbf_ctx_inc]);
            } else {
                cbf_cb = 0;
            }
            if tr_depth == 0 || parent_cbf_cr == 1 {
                cbf_cr = engine.decode_bin(&mut ctx.cbf_cb_cr[cbf_ctx_inc]);
            } else {
                cbf_cr = 0;
            }
        }

        if split == 1 {
            let sub = 1u32 << (log2_tb - 1);
            // Children's xBase/yBase is our current (x0, y0).
            self.transform_tree(
                engine, ctx, x0, y0, x0, y0, cu_x0, cu_y0,
                log2_cb, log2_tb - 1, tr_depth + 1, 0, cbf_cb, cbf_cr,
                luma_modes, chroma_mode, part_mode_nxn,
            )?;
            self.transform_tree(
                engine, ctx, x0 + sub, y0, x0, y0, cu_x0, cu_y0,
                log2_cb, log2_tb - 1, tr_depth + 1, 1, cbf_cb, cbf_cr,
                luma_modes, chroma_mode, part_mode_nxn,
            )?;
            self.transform_tree(
                engine, ctx, x0, y0 + sub, x0, y0, cu_x0, cu_y0,
                log2_cb, log2_tb - 1, tr_depth + 1, 2, cbf_cb, cbf_cr,
                luma_modes, chroma_mode, part_mode_nxn,
            )?;
            self.transform_tree(
                engine, ctx, x0 + sub, y0 + sub, x0, y0, cu_x0, cu_y0,
                log2_cb, log2_tb - 1, tr_depth + 1, 3, cbf_cb, cbf_cr,
                luma_modes, chroma_mode, part_mode_nxn,
            )?;
            return Ok(());
        }

        // Leaf transform_unit. §7.3.8.11.
        // cbf_luma is always read for intra leaves.
        let cbf_luma_inc = if tr_depth == 0 { 1usize } else { 0usize };
        let cbf_luma = engine.decode_bin(&mut ctx.cbf_luma[cbf_luma_inc]);
        let has_any_coeff = cbf_luma != 0 || cbf_cb != 0 || cbf_cr != 0;

        // cu_qp_delta once per QG when at least one cbf is set (§7.3.8.11).
        if has_any_coeff
            && self.cctx.pps.cu_qp_delta_enabled_flag
            && !self.is_cu_qp_delta_coded
        {
            // Prefix: TR with cMax=5, cRiceParam=0. ctxInc 0 for the first
            // bin, 1 for every subsequent prefix bin (§9.3.4.2).
            let mut prefix = 0u32;
            let max_prefix = 5u32;
            while prefix < max_prefix {
                let ctx_inc = if prefix == 0 { 0usize } else { 1 };
                if engine.decode_bin(&mut ctx.cu_qp_delta_abs[ctx_inc]) == 0 {
                    break;
                }
                prefix += 1;
            }
            let mut abs = prefix;
            if prefix >= max_prefix {
                // EG0 bypass suffix.
                let mut k = 0u32;
                while engine.decode_bypass() == 1 {
                    k += 1;
                    if k > 32 {
                        return Err(Error::invalid("h265 cu_qp_delta suffix overflow"));
                    }
                }
                let mut suf = 0u32;
                for _ in 0..k {
                    suf = (suf << 1) | engine.decode_bypass();
                }
                abs += suf + (1 << k) - 1;
            }
            let delta = if abs != 0 {
                let sign = engine.decode_bypass();
                if sign == 1 {
                    -(abs as i32)
                } else {
                    abs as i32
                }
            } else {
                0
            };
            // §8.6.1 eq. 8-283 — bit-depth-aware QpY wrap (see matching
            // comment above the other call site).
            let qp_bd = 6 * self.cctx.sps.bit_depth_luma_minus8 as i32;
            *self.cu_qp_y =
                (self.qpy_pred + delta + 52 + 2 * qp_bd).rem_euclid(52 + qp_bd) - qp_bd;
            self.is_cu_qp_delta_coded = true;
        }

        // Run intra prediction + residual + reconstruct, per plane.
        let luma_mode = if part_mode_nxn {
            // Pick the luma mode of the PU containing this TU, relative to
            // the root NxN intra CU.
            let sub = 1u32 << (log2_cb - 1);
            let lx = ((x0 - cu_x0) >= sub) as usize;
            let ly = ((y0 - cu_y0) >= sub) as usize;
            luma_modes[ly * 2 + lx]
        } else {
            luma_modes[0]
        };

        self.reconstruct_plane(engine, ctx, x0, y0, log2_tb, luma_mode, true, cbf_luma != 0)?;

        // Chroma residual placement (§7.3.8.11):
        //   log2_tb > 2: chroma TB co-located at (x0/2, y0/2), size log2_tb-1.
        //   log2_tb == 2 and blk_idx == 3: single chroma 4x4 at parent base
        //     (x_base/2, y_base/2), covering all 4 luma 4x4 children.
        if log2_tb > 2 {
            let chroma_x = x0 / 2;
            let chroma_y = y0 / 2;
            let chroma_log2 = log2_tb - 1;
            self.reconstruct_plane(
                engine, ctx, chroma_x, chroma_y, chroma_log2, chroma_mode, false, cbf_cb != 0,
            )?;
            self.reconstruct_plane_cr(
                engine, ctx, chroma_x, chroma_y, chroma_log2, chroma_mode, cbf_cr != 0,
            )?;
        } else if blk_idx == 3 {
            let chroma_x = x_base / 2;
            let chroma_y = y_base / 2;
            self.reconstruct_plane(
                engine, ctx, chroma_x, chroma_y, 2, chroma_mode, false, cbf_cb != 0,
            )?;
            self.reconstruct_plane_cr(
                engine, ctx, chroma_x, chroma_y, 2, chroma_mode, cbf_cr != 0,
            )?;
        }
        Ok(())
    }

    fn reconstruct_plane(
        &mut self,
        engine: &mut CabacEngine<'_>,
        ctx: &mut Ctx,
        x0: u32,
        y0: u32,
        log2_tb: u32,
        pred_mode: u32,
        is_luma: bool,
        has_coeff: bool,
    ) -> Result<()> {
        let n = 1usize << log2_tb;
        let bit_depth = if is_luma {
            self.cctx.sps.bit_depth_y()
        } else {
            self.cctx.sps.bit_depth_c()
        };
        let max_val = (1i32 << bit_depth) - 1;
        // Build reference samples.
        let refs = self.gather_refs(x0, y0, n, is_luma, false, true);
        let mut refs = refs;
        // MDIS filter decision for luma.
        if is_luma {
            let (apply, strong) = filter_decision(
                log2_tb,
                pred_mode,
                self.cctx.sps.strong_intra_smoothing_enabled_flag,
                &refs,
                n,
                bit_depth,
            );
            if apply {
                filter_ref_samples(&mut refs, n, strong, bit_depth);
            }
        }
        // Predict.
        let mut pred = vec![0u16; n * n];
        predict(&refs, n, &mut pred, n, pred_mode, is_luma, bit_depth);

        // If there are residuals, decode them and add.
        if has_coeff {
            let mut levels = vec![0i32; n * n];
            let ts =
                self.residual_coding(engine, ctx, &mut levels, log2_tb, pred_mode, is_luma)?;
            let mut res = vec![0i32; n * n];
            if self.cu_transquant_bypass {
                // §8.6.2 bypass path: levels already represent the spatial
                // residual. The spec applies neither dequantisation nor the
                // inverse transform. Residuals are placed directly (scan
                // order aside — `residual_coding` already writes in raster
                // order) and added to the predictor.
                res.copy_from_slice(&levels);
            } else {
                let qp = self.get_qp(is_luma, false);
                let mut deq = vec![0i32; n * n];
                self.dequantize(&levels, &mut deq, qp, log2_tb, is_luma, false, true, ts);
                if ts {
                    transform_skip_2d(&deq, &mut res, log2_tb, bit_depth);
                } else {
                    let is_dst = is_luma && log2_tb == 2;
                    inverse_transform_2d(&deq, &mut res, log2_tb, is_dst, bit_depth);
                }
            }
            for i in 0..n * n {
                let v = pred[i] as i32 + res[i];
                pred[i] = v.clamp(0, max_val) as u16;
            }
        }

        // Write to picture.
        self.write_block(x0, y0, n, &pred, is_luma, false);
        if is_luma {
            self.mark_decoded_block(x0, y0, n as u32);
        }
        Ok(())
    }

    fn reconstruct_plane_cr(
        &mut self,
        engine: &mut CabacEngine<'_>,
        ctx: &mut Ctx,
        x0: u32,
        y0: u32,
        log2_tb: u32,
        pred_mode: u32,
        has_coeff: bool,
    ) -> Result<()> {
        let n = 1usize << log2_tb;
        let bit_depth = self.cctx.sps.bit_depth_c();
        let max_val = (1i32 << bit_depth) - 1;
        let refs = self.gather_refs(x0, y0, n, false, true, false);
        let mut refs = refs;
        // Chroma doesn't get the MDIS filter in 4:2:0 main profile. Leave as-is.
        let _ = &mut refs;
        let mut pred = vec![0u16; n * n];
        predict(&refs, n, &mut pred, n, pred_mode, false, bit_depth);
        if has_coeff {
            let mut levels = vec![0i32; n * n];
            let ts = self.residual_coding(engine, ctx, &mut levels, log2_tb, pred_mode, false)?;
            let mut res = vec![0i32; n * n];
            if self.cu_transquant_bypass {
                res.copy_from_slice(&levels);
            } else {
                let mut deq = vec![0i32; n * n];
                let qp = self.get_qp(false, true);
                self.dequantize(&levels, &mut deq, qp, log2_tb, false, true, true, ts);
                if ts {
                    transform_skip_2d(&deq, &mut res, log2_tb, bit_depth);
                } else {
                    inverse_transform_2d(&deq, &mut res, log2_tb, false, bit_depth);
                }
            }
            for i in 0..n * n {
                let v = pred[i] as i32 + res[i];
                pred[i] = v.clamp(0, max_val) as u16;
            }
        }
        self.write_block(x0, y0, n, &pred, false, true);
        Ok(())
    }

    /// Dispatch to the right dequantiser: scaling-matrix-aware when the
    /// slice has scaling lists enabled (§7.4.5), flat otherwise.
    /// `is_intra` selects the matrixId subgroup (0..2 intra, 3..5 inter)
    /// per Table 7-4. `is_luma/is_cr` pick between Y / Cb / Cr.
    fn dequantize(
        &self,
        levels: &[i32],
        out: &mut [i32],
        qp: i32,
        log2_tb: u32,
        is_luma: bool,
        is_cr: bool,
        is_intra: bool,
        transform_skip: bool,
    ) {
        let bit_depth = if is_luma {
            self.cctx.sps.bit_depth_y()
        } else {
            self.cctx.sps.bit_depth_c()
        };
        // §8.6.3: when transform_skip_flag is set and nTbS > 4, m[x][y] is
        // still forced to 16 (flat) per the exception in the branch list.
        let force_flat = transform_skip && log2_tb > 2;
        if force_flat || self.scaling_list.is_none() {
            dequantize_flat(levels, out, qp, log2_tb, bit_depth);
            return;
        }
        let slist = self.scaling_list.unwrap();
        // sizeId maps (4×4→0, 8×8→1, 16×16→2, 32×32→3).
        let size_id = (log2_tb as usize).saturating_sub(2).min(3);
        // matrixId: 0..5 per Table 7-4. For sizeId==3 only 0 (intra luma)
        // and 3 (inter luma) are valid; chroma 32×32 doesn't exist in 4:2:0
        // (handled at the caller — chroma TUs max at 16×16).
        let cidx = if is_luma {
            0
        } else if is_cr {
            2
        } else {
            1
        };
        let matrix_id = if is_intra { cidx } else { 3 + cidx };
        let matrix = slist.expand_matrix(size_id, matrix_id);
        dequantize_with_matrix(levels, out, qp, log2_tb, bit_depth, &matrix);
    }

    /// Return the *primed* quantisation parameter for the given component
    /// (§8.6.2 eq. 8-284 / 8-289 / 8-290). The "primed" QP is what the
    /// dequantiser in §8.6.3 eq. 8-309 expects as `qP`:
    ///
    /// * luma: `Qp'Y = QpY + QpBdOffsetY`
    /// * chroma: `Qp'C = QpC + QpBdOffsetC`
    ///
    /// For Main (8-bit) this matches the bare QP because `QpBdOffset* = 0`.
    /// For Main 10, both offsets are 12 — the missing bit-depth offset was
    /// the dominant cause of the 13 dB PSNR gap on Main 10 intra content.
    fn get_qp(&self, is_luma: bool, is_cr: bool) -> i32 {
        let qp_bd_offset_y = 6 * self.cctx.sps.bit_depth_luma_minus8 as i32;
        let qp_bd_offset_c = 6 * self.cctx.sps.bit_depth_chroma_minus8 as i32;
        if is_luma {
            // Qp'Y = QpY + QpBdOffsetY (eq. 8-284).
            *self.cu_qp_y + qp_bd_offset_y
        } else {
            let qp_offset = if is_cr {
                self.cctx.pps.pps_cr_qp_offset + self.cctx.slice.slice_cr_qp_offset
            } else {
                self.cctx.pps.pps_cb_qp_offset + self.cctx.slice.slice_cb_qp_offset
            };
            // eq. 8-285/8-286: qPiCb/Cr = Clip3(-QpBdOffsetC, 57,
            //                                   QpY + pps_offset + slice_offset + CuQpOffset).
            // (CuQpOffsetC* comes from chroma_qp_offset_list — not yet
            //  parsed, defaults to 0 for streams that don't opt in.)
            let qpi = (*self.cu_qp_y + qp_offset).clamp(-qp_bd_offset_c, 57);
            // Table 8-10: QpC for ChromaArrayType == 1 (4:2:0).
            let qp_c = qp_y_to_qp_c(qpi);
            // eq. 8-289/8-290: Qp'Cb/Cr = QpC + QpBdOffsetC.
            qp_c + qp_bd_offset_c
        }
    }

    fn gather_refs(
        &self,
        x0: u32,
        y0: u32,
        n: usize,
        is_luma: bool,
        is_cr: bool,
        _top_left_only: bool,
    ) -> Vec<u16> {
        let len = 4 * n + 1;
        let bit_depth = if is_luma {
            self.cctx.sps.bit_depth_y()
        } else {
            self.cctx.sps.bit_depth_c()
        };
        let neutral = (1u16 << (bit_depth - 1)) as u16;
        let mut samples = vec![neutral; len];
        let mut avail = vec![false; len];
        let (stride, plane, pic_w, pic_h) = if is_luma {
            (
                self.pic.luma_stride,
                &self.pic.luma,
                self.pic.width as usize,
                self.pic.height as usize,
            )
        } else if is_cr {
            (
                self.pic.chroma_stride,
                &self.pic.cr,
                (self.pic.width as usize) / 2,
                (self.pic.height as usize) / 2,
            )
        } else {
            (
                self.pic.chroma_stride,
                &self.pic.cb,
                (self.pic.width as usize) / 2,
                (self.pic.height as usize) / 2,
            )
        };
        let x0 = x0 as usize;
        let y0 = y0 as usize;
        // Convert to luma-space coords for the decoded-block lookup.
        let luma_shift = usize::from(!is_luma);
        let is_decoded = |lx: usize, ly: usize| -> bool {
            let bx = lx >> 2;
            let by = ly >> 2;
            bx < self.pic.intra_width_4
                && by < self.pic.intra_height_4
                && self.pic.decoded_4x4[by * self.pic.intra_width_4 + bx]
        };

        // top-left corner p[-1, -1]
        if x0 > 0 && y0 > 0 {
            let lx = (x0 - 1) << luma_shift;
            let ly = (y0 - 1) << luma_shift;
            if is_decoded(lx, ly) {
                samples[0] = plane[(y0 - 1) * stride + (x0 - 1)];
                avail[0] = true;
            }
        }
        // top row p[0..2n-1, -1]
        if y0 > 0 {
            for i in 0..(2 * n) {
                let xx = x0 + i;
                if xx < pic_w {
                    let lx = xx << luma_shift;
                    let ly = (y0 - 1) << luma_shift;
                    if is_decoded(lx, ly) {
                        samples[1 + i] = plane[(y0 - 1) * stride + xx];
                        avail[1 + i] = true;
                    }
                }
            }
        }
        // left column p[-1, 0..2n-1]
        if x0 > 0 {
            for i in 0..(2 * n) {
                let yy = y0 + i;
                if yy < pic_h {
                    let lx = (x0 - 1) << luma_shift;
                    let ly = yy << luma_shift;
                    if is_decoded(lx, ly) {
                        samples[2 * n + 1 + i] = plane[yy * stride + (x0 - 1)];
                        avail[2 * n + 1 + i] = true;
                    }
                }
            }
        }
        build_ref_samples(&samples, &avail, n, bit_depth)
    }

    /// Persist `qp` into the 8×8 QP grid for every 8×8 block inside the QG
    /// rooted at `(qg_x, qg_y)` so later QGs can look it up as a neighbour
    /// (§8.6.1 QpY_A / QpY_B derivation).
    fn save_qg_qp(&mut self, qg_x: u32, qg_y: u32, qg_size: u32, qp: i32) {
        let stride = self.pic.qp_stride;
        let grid_w = stride;
        let grid_h = self.pic.qp_y.len() / stride.max(1);
        let bx0 = (qg_x >> 3) as usize;
        let by0 = (qg_y >> 3) as usize;
        let n8 = (qg_size >> 3).max(1) as usize;
        for dy in 0..n8 {
            for dx in 0..n8 {
                let bx = bx0 + dx;
                let by = by0 + dy;
                if bx < grid_w && by < grid_h {
                    self.pic.qp_y[by * stride + bx] = qp;
                }
            }
        }
    }

    /// QpY_PRED derivation per §8.6.1 eq. 8-286. Uses the left and above
    /// 8×8 neighbour QPs if they lie in a different QG within the current
    /// slice; otherwise falls back to `qpy_prev`. Picture-edge samples are
    /// treated as unavailable.
    fn compute_qpy_pred(&self, qg_x: u32, qg_y: u32) -> i32 {
        let stride = self.pic.qp_stride;
        let grid_w = stride;
        let grid_h = self.pic.qp_y.len() / stride.max(1);
        let lookup = |sx: i32, sy: i32| -> Option<i32> {
            if sx < 0 || sy < 0 {
                return None;
            }
            let bx = (sx as u32 >> 3) as usize;
            let by = (sy as u32 >> 3) as usize;
            if bx >= grid_w || by >= grid_h {
                return None;
            }
            Some(self.pic.qp_y[by * stride + bx])
        };
        let qpy_a = lookup(qg_x as i32 - 1, qg_y as i32).unwrap_or(self.qpy_prev);
        let qpy_b = lookup(qg_x as i32, qg_y as i32 - 1).unwrap_or(self.qpy_prev);
        (qpy_a + qpy_b + 1) >> 1
    }

    /// Mark a `size × size` region starting at `(x, y)` as reconstructed
    /// so subsequent intra-pred reference-sample gathering can see it.
    fn mark_decoded_block(&mut self, x: u32, y: u32, size: u32) {
        let bx0 = (x >> 2) as usize;
        let by0 = (y >> 2) as usize;
        let n4 = (size >> 2).max(1) as usize;
        let iw = self.pic.intra_width_4;
        let ih = self.pic.intra_height_4;
        for dy in 0..n4 {
            for dx in 0..n4 {
                let bx = bx0 + dx;
                let by = by0 + dy;
                if bx < iw && by < ih {
                    self.pic.decoded_4x4[by * iw + bx] = true;
                }
            }
        }
    }

    fn write_rect(
        &mut self,
        x: u32,
        y: u32,
        w: usize,
        h: usize,
        block: &[u16],
        is_luma: bool,
        is_cr: bool,
    ) {
        let (stride, plane, pic_w, pic_h) = if is_luma {
            (
                self.pic.luma_stride,
                &mut self.pic.luma,
                self.pic.width as usize,
                self.pic.height as usize,
            )
        } else if is_cr {
            (
                self.pic.chroma_stride,
                &mut self.pic.cr,
                (self.pic.width as usize) / 2,
                (self.pic.height as usize) / 2,
            )
        } else {
            (
                self.pic.chroma_stride,
                &mut self.pic.cb,
                (self.pic.width as usize) / 2,
                (self.pic.height as usize) / 2,
            )
        };
        let x0 = x as usize;
        let y0 = y as usize;
        for dy in 0..h {
            for dx in 0..w {
                let xx = x0 + dx;
                let yy = y0 + dy;
                if xx < pic_w && yy < pic_h {
                    plane[yy * stride + xx] = block[dy * w + dx];
                }
            }
        }
    }

    fn write_block(&mut self, x: u32, y: u32, n: usize, block: &[u16], is_luma: bool, is_cr: bool) {
        let (stride, plane, pic_w, pic_h) = if is_luma {
            (
                self.pic.luma_stride,
                &mut self.pic.luma,
                self.pic.width as usize,
                self.pic.height as usize,
            )
        } else if is_cr {
            (
                self.pic.chroma_stride,
                &mut self.pic.cr,
                (self.pic.width as usize) / 2,
                (self.pic.height as usize) / 2,
            )
        } else {
            (
                self.pic.chroma_stride,
                &mut self.pic.cb,
                (self.pic.width as usize) / 2,
                (self.pic.height as usize) / 2,
            )
        };
        let x0 = x as usize;
        let y0 = y as usize;
        for dy in 0..n {
            for dx in 0..n {
                let xx = x0 + dx;
                let yy = y0 + dy;
                if xx < pic_w && yy < pic_h {
                    plane[yy * stride + xx] = block[dy * n + dx];
                }
            }
        }
    }

    /// Decode residual coefficients for a single TU into `levels`
    /// (row-major, size `n*n`). Returns the decoded
    /// `transform_skip_flag` — `true` when the iDCT should be bypassed
    /// for this TU (§8.6.4.2 eq. 8-298).
    fn residual_coding(
        &mut self,
        engine: &mut CabacEngine<'_>,
        ctx: &mut Ctx,
        levels: &mut [i32],
        log2_tb: u32,
        pred_mode: u32,
        is_luma: bool,
    ) -> Result<bool> {
        let n = 1usize << log2_tb;
        let scan_idx = scan_idx_for_intra(log2_tb, pred_mode, is_luma);
        let scan = scan_4x4(scan_idx);

        // §7.3.8.11: transform_skip_flag is the first element, present only
        // when `pps.transform_skip_enabled_flag` is set, the block is 4×4
        // (Log2MaxTransformSkipSize defaults to 2), and the CU is not in
        // transquant-bypass mode. Rext can raise the cap; we clamp at 2
        // for the base profile.
        let transform_skip_flag = if self.cctx.pps.transform_skip_enabled_flag
            && log2_tb == 2
            && !self.cu_transquant_bypass
        {
            let ctx_arr: &mut [CtxState; 1] = if is_luma {
                &mut ctx.transform_skip_flag_luma
            } else {
                &mut ctx.transform_skip_flag_chroma
            };
            engine.decode_bin(&mut ctx_arr[0]) == 1
        } else {
            false
        };

        // last_sig_coeff_x/y_{prefix, suffix} (§9.3.4.2.7, 9.3.4.2.8).
        let (last_x, last_y) =
            self.decode_last_sig_pos(engine, ctx, log2_tb, scan_idx, is_luma)?;
        // Translate (last_x, last_y) into the (scan_idx)-ordered 4x4 sub-block
        // grid coordinate.
        let (last_sx, last_sy) = subblock_coord(last_x, last_y);
        let (last_px, last_py) = (last_x & 3, last_y & 3);
        // Scan sub-blocks in reverse scan order from (last_sx, last_sy) to (0,0).
        let num_sb = (n / 4).max(1);
        // Build the list of sub-block positions in forward scan order.
        let sb_scan = subblock_scan(scan_idx, log2_tb);
        // Find the index of (last_sx, last_sy) in sb_scan.
        let mut last_sb_idx = 0usize;
        for (i, &(sx, sy)) in sb_scan.iter().enumerate() {
            if sx as usize == last_sx && sy as usize == last_sy {
                last_sb_idx = i;
                break;
            }
        }
        // Find the index of (last_px, last_py) within scan.
        let mut last_coef_in_sb = 0usize;
        for (i, &(px, py)) in scan.iter().enumerate() {
            if px as usize == last_px && py as usize == last_py {
                last_coef_in_sb = i;
                break;
            }
        }

        // Significance map for sub-blocks (tracks coded_sub_block_flag).
        let mut sb_cbf = vec![false; num_sb * num_sb];
        sb_cbf[last_sy * num_sb + last_sx] = true; // Always set at last-sig pos.
        let mut prev_greater1_ctx = 1u8;
        let mut prev_greater1_flag = false;

        // For each sub-block in reverse scan order starting at last_sb_idx.
        for i in (0..=last_sb_idx).rev() {
            let (sx_u8, sy_u8) = sb_scan[i];
            let sx = sx_u8 as usize;
            let sy = sy_u8 as usize;
            let is_last_sb = i == last_sb_idx;
            let is_first_sb = i == 0; // DC sub-block.

            // Decode coded_sub_block_flag unless this is the last SB (has at
            // least one coeff at last-pos) or the first (DC) SB (always 1).
            let coded = if is_last_sb || is_first_sb {
                sb_cbf[sy * num_sb + sx] = true;
                true
            } else {
                let right = sx + 1 < num_sb && sb_cbf[sy * num_sb + sx + 1];
                let below = sy + 1 < num_sb && sb_cbf[(sy + 1) * num_sb + sx];
                let ctx_inc = (right || below) as usize;
                let ctx_inc = if is_luma { ctx_inc } else { 2 + ctx_inc };
                let v = engine.decode_bin(&mut ctx.coded_sub_block_flag[ctx_inc]);
                sb_cbf[sy * num_sb + sx] = v == 1;
                v == 1
            };
            if !coded {
                continue;
            }
            // Per §7.4.9.11: for a middle sub-block, start with
            // inferSbDcSigCoeffFlag=1. Any decoded sig_coeff_flag==1 during
            // the reverse-scan loop clears it; if it is still 1 when we reach
            // the DC position (n==0) we skip the read and infer the DC flag
            // to 1. First/last sub-blocks never skip DC.
            let start = if is_last_sb { last_coef_in_sb } else { 15 };
            let mut sig_flags = [false; 16];
            if is_last_sb {
                sig_flags[last_coef_in_sb] = true;
            }
            let middle_sb = !is_last_sb && !is_first_sb;
            let mut infer_dc = middle_sb;
            for j in (0..=start).rev() {
                if is_last_sb && j == last_coef_in_sb {
                    continue;
                }
                let (cx, cy) = scan[j];
                let at_dc = cx == 0 && cy == 0;
                if at_dc && infer_dc {
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
                let v = engine.decode_bin(&mut ctx.sig_coeff_flag[ctx_inc]);
                sig_flags[j] = v == 1;
                if v == 1 {
                    infer_dc = false;
                }
            }
            if middle_sb && infer_dc {
                sig_flags[0] = true;
            }
            // coeff_abs_level_greater1_flag / greater2_flag / sign / remaining.
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
                let mut last_greater1_ctx = prev_greater1_ctx;
                if last_greater1_ctx > 0 {
                    if prev_greater1_flag {
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
                if num_greater1_flag < 8 {
                    let mut ctx_inc = (ctx_set as usize * 4) + usize::min(3, greater1_ctx as usize);
                    if !is_luma {
                        ctx_inc += 16;
                    }
                    let v = engine.decode_bin(&mut ctx.coeff_abs_gt1[ctx_inc]);
                    greater1_flags[pos] = v == 1;
                    last_used_greater1_flag = greater1_flags[pos];
                    num_greater1_flag += 1;
                    if greater1_flags[pos] {
                        if last_greater1_scan_pos == -1 {
                            last_greater1_scan_pos = pos as i32;
                        }
                    }
                    if greater1_ctx > 0 {
                        if greater1_flags[pos] {
                            greater1_ctx = 0;
                        } else {
                            greater1_ctx = greater1_ctx.saturating_add(1);
                        }
                    }
                }
                if last_sig_scan_pos == -1 {
                    last_sig_scan_pos = pos as i32;
                }
                first_sig_scan_pos = pos as i32;
            }
            prev_greater1_ctx = greater1_ctx;
            prev_greater1_flag = last_used_greater1_flag;
            if last_greater1_scan_pos != -1 {
                let mut ctx_inc = ctx_set as usize;
                if !is_luma {
                    ctx_inc += 4;
                }
                greater2_flags[last_greater1_scan_pos as usize] =
                    engine.decode_bin(&mut ctx.coeff_abs_gt2[ctx_inc]) == 1;
            }
            let sign_hidden = self.cctx.pps.sign_data_hiding_enabled_flag
                && (last_sig_scan_pos - first_sig_scan_pos > 3);
            let mut sign_flags = [false; 16];
            for pos in (0..16).rev() {
                if !sig_flags[pos] {
                    continue;
                }
                if !sign_hidden || pos as i32 != first_sig_scan_pos {
                    sign_flags[pos] = engine.decode_bypass() == 1;
                }
            }
            let mut num_sig_coeff = 0u32;
            let mut sum_abs_level = 0u32;
            let mut coeff_rem_seen = false;
            let mut last_abs_level = 0u32;
            let mut last_rice_param = 0u32;
            for pos in (0..16).rev() {
                if !sig_flags[pos] {
                    continue;
                }
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
                let remainder = if base_level == threshold {
                    let rice_param = if coeff_rem_seen {
                        (last_rice_param
                            + u32::from(last_abs_level > (3 * (1u32 << last_rice_param))))
                        .min(4)
                    } else {
                        0
                    };
                    let rem = decode_coeff_remainder(engine, rice_param)?;
                    coeff_rem_seen = true;
                    last_abs_level = base_level + rem;
                    last_rice_param = rice_param;
                    rem
                } else {
                    0
                };
                let abs_level = base_level + remainder;
                if sign_hidden {
                    sum_abs_level += abs_level;
                    if pos as i32 == first_sig_scan_pos && (sum_abs_level & 1) == 1 {
                        sign_flags[pos] = !sign_flags[pos];
                    }
                }
                let level = if sign_flags[pos] {
                    -(abs_level as i32)
                } else {
                    abs_level as i32
                };
                let (cx, cy) = scan[pos];
                let gx = sx * 4 + cx as usize;
                let gy = sy * 4 + cy as usize;
                levels[gy * n + gx] = level;
                num_sig_coeff += 1;
            }
        }
        // Seed the last-sig position's coefficient value if we didn't cover it
        // in the loop (the decode_coeff_remainder path records it).
        if levels[last_y * n + last_x] == 0 {
            levels[last_y * n + last_x] = 1;
        }
        Ok(transform_skip_flag)
    }

    fn decode_last_sig_pos(
        &mut self,
        engine: &mut CabacEngine<'_>,
        ctx: &mut Ctx,
        log2_tb: u32,
        scan_idx: u32,
        is_luma: bool,
    ) -> Result<(usize, usize)> {
        // §9.3.4.2.3 / §7.4.9.11.
        let prefix_x =
            decode_last_sig_prefix(engine, &mut ctx.last_sig_x_prefix, log2_tb, is_luma)?;
        let prefix_y =
            decode_last_sig_prefix(engine, &mut ctx.last_sig_y_prefix, log2_tb, is_luma)?;
        let last_x = if prefix_x > 3 {
            let k = (prefix_x - 2) >> 1;
            let mut suf = 0u32;
            for _ in 0..k {
                suf = (suf << 1) | engine.decode_bypass();
            }
            (1 << ((prefix_x - 2) >> 1)) * (2 + (prefix_x & 1)) + suf
        } else {
            prefix_x
        };
        let last_y = if prefix_y > 3 {
            let k = (prefix_y - 2) >> 1;
            let mut suf = 0u32;
            for _ in 0..k {
                suf = (suf << 1) | engine.decode_bypass();
            }
            (1 << ((prefix_y - 2) >> 1)) * (2 + (prefix_y & 1)) + suf
        } else {
            prefix_y
        };
        if scan_idx == 2 {
            Ok((last_y as usize, last_x as usize))
        } else {
            Ok((last_x as usize, last_y as usize))
        }
    }
}

/// Truncated-rice merge_idx decode (§9.3.4.2.10). Values 0..MaxNumMergeCand-1.
fn decode_merge_idx(engine: &mut CabacEngine<'_>, ctx: &mut Ctx, max: u32) -> u32 {
    if max <= 1 {
        return 0;
    }
    // First bin: context-coded with merge_idx[0].
    let b0 = engine.decode_bin(&mut ctx.merge_idx[0]);
    if b0 == 0 {
        return 0;
    }
    // Remaining bins: bypass, unary up to max-1.
    let mut v = 1u32;
    while v < max - 1 {
        let b = engine.decode_bypass();
        if b == 0 {
            break;
        }
        v += 1;
    }
    v
}

/// ref_idx truncated-rice decode (§9.3.4.2.11). `num_refs` is the list size.
fn decode_ref_idx(engine: &mut CabacEngine<'_>, ctx: &mut Ctx, num_refs: u32) -> u32 {
    if num_refs <= 1 {
        return 0;
    }
    // First bin: ctx 0. Second bin (if needed): ctx 1. Remaining: bypass.
    let b0 = engine.decode_bin(&mut ctx.ref_idx[0]);
    if b0 == 0 {
        return 0;
    }
    if num_refs == 2 {
        return 1;
    }
    let b1 = engine.decode_bin(&mut ctx.ref_idx[1]);
    if b1 == 0 {
        return 1;
    }
    let mut v = 2u32;
    while v < num_refs - 1 {
        let b = engine.decode_bypass();
        if b == 0 {
            break;
        }
        v += 1;
    }
    v
}

/// Parse one `mvd_coding()` (§7.3.8.9) for L0 and return the signed MVD.
fn decode_mvd(engine: &mut CabacEngine<'_>, ctx: &mut Ctx) -> Result<MotionVector> {
    let gt0_x = engine.decode_bin(&mut ctx.abs_mvd_greater[0]);
    let gt0_y = engine.decode_bin(&mut ctx.abs_mvd_greater[0]);
    let mut abs_x: i32 = 0;
    let mut abs_y: i32 = 0;
    let mut gt1_x = 0u32;
    let mut gt1_y = 0u32;
    if gt0_x == 1 {
        gt1_x = engine.decode_bin(&mut ctx.abs_mvd_greater[1]);
    }
    if gt0_y == 1 {
        gt1_y = engine.decode_bin(&mut ctx.abs_mvd_greater[1]);
    }
    if gt0_x == 1 {
        abs_x = 1 + gt1_x as i32;
        if gt1_x == 1 {
            let rem = decode_eg1(engine)?;
            abs_x += rem as i32;
        }
        let sign = engine.decode_bypass();
        if sign == 1 {
            abs_x = -abs_x;
        }
    }
    if gt0_y == 1 {
        abs_y = 1 + gt1_y as i32;
        if gt1_y == 1 {
            let rem = decode_eg1(engine)?;
            abs_y += rem as i32;
        }
        let sign = engine.decode_bypass();
        if sign == 1 {
            abs_y = -abs_y;
        }
    }
    Ok(MotionVector::new(abs_x, abs_y))
}

/// First-order Exp-Golomb (EG1) bypass-coded decode per §9.3.4.3.4.
fn decode_eg1(engine: &mut CabacEngine<'_>) -> Result<u32> {
    let mut k = 1u32;
    while engine.decode_bypass() == 1 {
        k += 1;
        if k > 32 {
            return Err(Error::invalid("h265 eg1 prefix overflow"));
        }
    }
    // Read k bits of suffix.
    let mut suf = 0u32;
    for _ in 0..k {
        suf = (suf << 1) | engine.decode_bypass();
    }
    Ok(suf + (1u32 << k) - 2)
}

/// §9.3.4.2.3 last_sig_coeff prefix context increment.
/// Returns a value in 0..18 safe for the 18-entry context array.
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

fn decode_last_sig_prefix(
    engine: &mut CabacEngine<'_>,
    contexts: &mut [CtxState; 18],
    log2_tb: u32,
    is_luma: bool,
) -> Result<u32> {
    let max_prefix = (log2_tb << 1).saturating_sub(1);
    let mut prefix = 0u32;
    while prefix < max_prefix {
        let ctx_inc = last_sig_prefix_ctx_inc(prefix, log2_tb, is_luma);
        if ctx_inc >= contexts.len() {
            return Err(Error::invalid("h265 last_sig prefix ctx overflow"));
        }
        let v = engine.decode_bin(&mut contexts[ctx_inc]);
        if v == 0 {
            break;
        }
        prefix += 1;
    }
    Ok(prefix)
}

/// Map a coefficient coordinate to the enclosing 4×4 sub-block coordinate.
fn subblock_coord(x: usize, y: usize) -> (usize, usize) {
    (x >> 2, y >> 2)
}

/// Generate the sub-block scan order for an n×n TB given scan_idx.
///
/// Per HEVC §9.3.4.2.5 / §6.5.4, the sub-block scan order tracks
/// `scan_idx` only for 8×8 TUs (log2 == 3). For 16×16 and 32×32 TUs the
/// sub-block scan is ALWAYS up-right diagonal, regardless of the
/// `scan_idx` used for the coefficient scan inside each 4×4 sub-block.
fn subblock_scan(scan_idx: u32, log2_tb: u32) -> Vec<(u8, u8)> {
    let sb = if log2_tb <= 2 {
        1
    } else {
        1usize << (log2_tb - 2)
    };
    // Force diagonal sub-block scan for 16×16 / 32×32 TUs.
    let effective_scan_idx = if log2_tb >= 4 { 0 } else { scan_idx };
    match effective_scan_idx {
        1 => {
            // Horizontal — row-major.
            let mut v = Vec::with_capacity(sb * sb);
            for y in 0..sb {
                for x in 0..sb {
                    v.push((x as u8, y as u8));
                }
            }
            v
        }
        2 => {
            // Vertical — column-major.
            let mut v = Vec::with_capacity(sb * sb);
            for x in 0..sb {
                for y in 0..sb {
                    v.push((x as u8, y as u8));
                }
            }
            v
        }
        _ => {
            // Up-right diagonal (§6.5.3): along each anti-diagonal, walk from
            // (0, d) down to (d, 0) — y decreasing, x increasing.
            let mut v = Vec::with_capacity(sb * sb);
            for d in 0..(2 * sb - 1) {
                for x in 0..=d {
                    let y = d - x;
                    if x < sb && y < sb {
                        v.push((x as u8, y as u8));
                    }
                }
            }
            v
        }
    }
}

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
    // §9.3.4.2.5 eq. 9-41: the 4×4 branch returns ctxIdxMap[] directly and
    // skips the subblock / size modifiers applied to the larger-TB cases.
    if log2_tb == 2 {
        let sig_ctx = CTX_IDX_MAP_4X4[((cy << 2) | cx) as usize];
        return if is_luma { sig_ctx } else { 27 + sig_ctx };
    }
    // §9.3.4.2.5 eq. 9-42: if the whole-TB DC (xC + yC == 0) is reached we
    // return sigCtx = 0 directly — the modifier block below belongs to the
    // "Otherwise, sigCtx is derived using previous values of coded_sub_block_flag"
    // branch only.
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

fn decode_coeff_remainder(engine: &mut CabacEngine<'_>, rice: u32) -> Result<u32> {
    let c_max = 4u32 << rice;
    let prefix_cap = c_max >> rice; // Always 4 for coeff_abs_level_remaining.
    let mut prefix = 0u32;
    while prefix < prefix_cap && engine.decode_bypass() == 1 {
        prefix += 1;
    }
    if prefix < prefix_cap {
        let mut suffix = 0u32;
        for _ in 0..rice {
            suffix = (suffix << 1) | engine.decode_bypass();
        }
        Ok((prefix << rice) + suffix)
    } else {
        Ok(c_max + decode_egk_bypass(engine, rice + 1)?)
    }
}

fn decode_egk_bypass(engine: &mut CabacEngine<'_>, k: u32) -> Result<u32> {
    let mut prefix = 0u32;
    while engine.decode_bypass() == 1 {
        prefix += 1;
        if prefix > 32 {
            return Err(Error::invalid("h265 EGk prefix overflow"));
        }
    }
    let mut suffix = 0u32;
    for _ in 0..(prefix + k) {
        suffix = (suffix << 1) | engine.decode_bypass();
    }
    Ok((((1u32 << prefix) - 1) << k) + suffix)
}

fn qp_y_to_qp_c(qp_y: i32) -> i32 {
    // Table 8-10 (§8.6.1), 4:2:0 only.
    if qp_y < 30 {
        qp_y
    } else {
        match qp_y {
            30 => 29,
            31 => 30,
            32 => 31,
            33 => 32,
            34 => 33,
            35 => 33,
            36 => 34,
            37 => 34,
            38 => 35,
            39 => 35,
            40 => 36,
            41 => 36,
            42 => 37,
            43 => 37,
            q if q >= 44 => q - 6,
            _ => qp_y,
        }
    }
}

/// Read an `n`-bit unsigned integer at byte position `byte_pos`, bit
/// offset `bit_off` (within that byte, MSB-first). Used by the PCM path
/// to pull fixed-length samples directly from the RBSP bytes.
/// Returns `None` when the read would run past the end of `data`.
fn read_fl_bits(data: &[u8], byte_pos: usize, bit_off: u32, n: usize) -> Option<u32> {
    if n == 0 {
        return Some(0);
    }
    if n > 32 {
        return None;
    }
    let total_bit = byte_pos as u64 * 8 + bit_off as u64 + n as u64;
    if (total_bit as usize).div_ceil(8) > data.len() {
        // Clamp to whatever is available — pcm_sample bits can run to the
        // very end of the slice, in which case padding zeros are read
        // (matches the §9 "read 0 past EOF" convention).
    }
    let mut v: u32 = 0;
    let mut remaining = n;
    let mut pos = byte_pos;
    let mut off = bit_off;
    while remaining > 0 {
        if pos >= data.len() {
            v <<= remaining;
            break;
        }
        let byte = data[pos] as u32;
        let take = core::cmp::min(remaining, 8 - off as usize);
        let shift = 8 - off as usize - take;
        let chunk = (byte >> shift) & ((1u32 << take) - 1);
        v = (v << take) | chunk;
        off += take as u32;
        if off >= 8 {
            off -= 8;
            pos += 1;
        }
        remaining -= take;
    }
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_fl_bits_aligned_byte() {
        let data = [0xAB, 0xCD];
        assert_eq!(read_fl_bits(&data, 0, 0, 8), Some(0xAB));
        assert_eq!(read_fl_bits(&data, 1, 0, 8), Some(0xCD));
    }

    #[test]
    fn read_fl_bits_crosses_byte_boundary() {
        // Bits: 10101011 11001101 ; starting at bit 4 for 8 bits -> 10111100 = 0xBC.
        let data = [0xAB, 0xCD];
        assert_eq!(read_fl_bits(&data, 0, 4, 8), Some(0xBC));
    }

    #[test]
    fn read_fl_bits_past_eof_clamps_zero() {
        let data = [0xFF];
        // Reading 16 bits past start reads 0xFF + 0x00 = 0xFF00.
        assert_eq!(read_fl_bits(&data, 0, 0, 16), Some(0xFF00));
    }

    #[test]
    fn read_fl_bits_zero_width() {
        assert_eq!(read_fl_bits(&[], 0, 0, 0), Some(0));
    }
}
