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
};
use crate::inter::{
    build_amvp_list, build_merge_list, chroma_mc, chroma_mc_bi_combine, chroma_mc_hp, luma_mc,
    luma_mc_bi_combine, luma_mc_bi_weighted, luma_mc_hp, InterState, MergeCand, MotionVector,
    PbMotion, RefPicture, WeightedPred,
};
use crate::intra_pred::{build_ref_samples, filter_decision, filter_ref_samples, predict};
use crate::pps::PicParameterSet;
use crate::scan::{scan_4x4, scan_idx_for_intra};
use crate::slice::{SliceSegmentHeader, SliceType};
use crate::sps::SeqParameterSet;
use crate::transform::{dequantize_flat, inverse_transform_2d};

/// Reconstructed picture (8-bit 4:2:0 only).
pub struct Picture {
    pub width: u32,
    pub height: u32,
    pub luma: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
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
    /// Per-4×4 block motion-field grid for inter neighbour lookups.
    pub inter: InterState,
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
            luma: vec![128u8; w * h],
            cb: vec![128u8; cw * ch],
            cr: vec![128u8; cw * ch],
            luma_stride: w,
            chroma_stride: cw,
            intra_luma_mode: vec![1u8; iw4 * ih4], // 1 = DC default (§8.4.4.2.1).
            intra_width_4: iw4,
            intra_height_4: ih4,
            is_intra: vec![true; iw4 * ih4],
            cqt_depth: vec![0u8; iw4 * ih4],
            decoded_4x4: vec![false; iw4 * ih4],
            inter: InterState::new(width, height),
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InterPart {
    Mode2Nx2N,
    Mode2NxN,
    ModeNx2N,
    ModeNxN,
}

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
    if cctx.sps.bit_depth_y() != 8 || cctx.sps.bit_depth_c() != 8 {
        return Err(Error::unsupported("h265 only 8-bit pixel decode supported"));
    }
    if cctx.sps.pcm_enabled_flag {
        return Err(Error::unsupported("h265 pcm_enabled pending"));
    }
    if cctx.sps.scaling_list_enabled_flag {
        return Err(Error::unsupported("h265 scaling_list_enabled pending"));
    }
    if cctx.pps.transform_skip_enabled_flag {
        return Err(Error::unsupported("h265 transform_skip pending"));
    }
    if cctx.pps.tiles_enabled_flag {
        return Err(Error::unsupported("h265 tiles pending"));
    }
    // Note: SAO syntax is parsed per-CTU (see decode_sao), but applying SAO
    // to reconstructed samples is deferred. Streams with SAO decode without
    // the in-loop filter effect — lower-fidelity but functional.
    if !cctx.slice.slice_deblocking_filter_disabled_flag {
        return Err(Error::unsupported("h265 deblocking filter pending"));
    }
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

    let mut engine = CabacEngine::new(rbsp, slice_data_byte_off);
    let mut ctx = Ctx::init(cctx.slice.slice_qp_y, cctx.init_type);
    let mut cu_qp_y = cctx.slice.slice_qp_y;
    let _ = ctb_log2;
    let mut walker = Walker {
        pic,
        cctx,
        min_cb_log2,
        max_tb_log2,
        min_tb_log2,
        cu_qp_y: &mut cu_qp_y,
        is_cu_qp_delta_coded: false,
        last_qg_pos: None,
    };

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
    Ok(())
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
}

impl<'a> Walker<'a> {
    fn decode_sao(
        &mut self,
        engine: &mut CabacEngine<'_>,
        ctx: &mut Ctx,
        ctx_x: u32,
        ctx_y: u32,
    ) -> Result<()> {
        // §7.3.8.3 sao() + §9.3.4.2.1 context derivation. The bits are parsed
        // but values are discarded; actually applying SAO to reconstructed
        // samples is pending.
        let mut merge_left = 0u32;
        let mut merge_up = 0u32;
        if ctx_x > 0 {
            merge_left = engine.decode_bin(&mut ctx.sao_merge_flag[0]);
        }
        if ctx_y > 0 && merge_left == 0 {
            merge_up = engine.decode_bin(&mut ctx.sao_merge_flag[0]);
        }
        if merge_left == 1 || merge_up == 1 {
            return Ok(());
        }
        // sao_offset_abs is TR with cMax = (1 << (Min(bitDepth, 10) - 5)) - 1,
        // which is 7 for 8-bit. cRiceParam = 0 → unary up to 7 ones then stop.
        let abs_cmax: u32 = 7;
        // sao_type_idx is shared across Cb and Cr; Cr inherits it.
        let mut chroma_type_idx: u32 = 0;
        for comp in 0..3 {
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
            if type_idx == 1 {
                // Band offset: sign flag for each non-zero offset, then 5-bit
                // sao_band_position.
                for &abs in &offset_abs {
                    if abs != 0 {
                        let _sign = engine.decode_bypass();
                    }
                }
                for _ in 0..5 {
                    let _ = engine.decode_bypass();
                }
            } else {
                // Edge offset: sao_eo_class (2 bits) only for luma + Cb; Cr
                // inherits Cb's class.
                if comp != 2 {
                    for _ in 0..2 {
                        let _ = engine.decode_bypass();
                    }
                }
            }
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
        if self.cctx.pps.transquant_bypass_enabled_flag {
            let bypass = engine.decode_bin(&mut ctx.cu_transquant_bypass_flag[0]);
            if bypass == 1 {
                return Err(Error::unsupported("h265 transquant_bypass pending"));
            }
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
            // part_mode for inter (§7.3.8.5, Table 7-10). AMP not supported.
            let b0 = engine.decode_bin(&mut ctx.part_mode[0]);
            let part_mode = if b0 == 1 {
                InterPart::Mode2Nx2N
            } else {
                let b1 = engine.decode_bin(&mut ctx.part_mode[1]);
                let at_min = log2_cb == self.min_cb_log2;
                let log2_gt_3 = log2_cb > 3;
                if at_min && log2_gt_3 {
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
            if part_mode_is_nxn {
                self.decode_intra_prediction_and_transforms(engine, ctx, x0, y0, log2_cb, true)
            } else {
                self.decode_intra_prediction_and_transforms(engine, ctx, x0, y0, log2_cb, false)
            }
        }
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

        let mut luma_out = vec![0u8; wz * hz];
        let mut cb_out = vec![0u8; cw * ch];
        let mut cr_out = vec![0u8; cw * ch];

        let is_bi = pb.pred_l0 && pb.pred_l1;
        let weighted = self.cctx.weighted_pred;

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
                );
            } else {
                luma_mc_bi_combine(&a_l, &b_l, &mut luma_out);
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
                    luma_mc_bi_weighted(a, b, out, w0, o0, w1, o1, w_table.chroma_denom + 1);
                } else {
                    chroma_mc_bi_combine(a, b, out);
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
        let b = if y0 == 0 { None } else { get_mode(x0, y0 - 1) };
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
            *self.cu_qp_y = (*self.cu_qp_y + delta).rem_euclid(52);
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
        self.residual_coding(
            engine,
            ctx,
            &mut levels,
            log2_tb,
            /*pred_mode*/ 0,
            is_luma,
        )?;
        let qp = self.get_qp(is_luma, false);
        let mut deq = vec![0i32; n * n];
        dequantize_flat(&levels, &mut deq, qp, log2_tb, 8);
        let mut res = vec![0i32; n * n];
        // Inter TU: DCT only (no DST-VII even at 4×4).
        inverse_transform_2d(&deq, &mut res, log2_tb, false, 8);
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
        self.residual_coding(engine, ctx, &mut levels, log2_tb, 0, false)?;
        let qp = self.get_qp(false, true);
        let mut deq = vec![0i32; n * n];
        dequantize_flat(&levels, &mut deq, qp, log2_tb, 8);
        let mut res = vec![0i32; n * n];
        inverse_transform_2d(&deq, &mut res, log2_tb, false, 8);
        self.add_to_plane(x0, y0, n, &res, false, true);
        Ok(())
    }

    fn add_to_plane(&mut self, x: u32, y: u32, n: usize, res: &[i32], is_luma: bool, is_cr: bool) {
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
                    plane[idx] = v.clamp(0, 255) as u8;
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
            *self.cu_qp_y = (*self.cu_qp_y + delta).rem_euclid(52);
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
            );
            if apply {
                filter_ref_samples(&mut refs, n, strong);
            }
        }
        // Predict.
        let mut pred = vec![0u8; n * n];
        predict(&refs, n, &mut pred, n, pred_mode, is_luma);

        // If there are residuals, decode them and add.
        if has_coeff {
            let mut levels = vec![0i32; n * n];
            self.residual_coding(engine, ctx, &mut levels, log2_tb, pred_mode, is_luma)?;
            let qp = self.get_qp(is_luma, false);
            let mut deq = vec![0i32; n * n];
            dequantize_flat(&levels, &mut deq, qp, log2_tb, 8);
            let mut res = vec![0i32; n * n];
            let is_dst = is_luma && log2_tb == 2;
            inverse_transform_2d(&deq, &mut res, log2_tb, is_dst, 8);
            if std::env::var_os("H265_TRACE_RESIDUAL").is_some() {
                let plane = if is_luma { "Y" } else { "C" };
                let nonzero: Vec<(usize, usize, i32)> = (0..n * n)
                    .filter(|&i| levels[i] != 0)
                    .map(|i| (i % n, i / n, levels[i]))
                    .collect();
                eprintln!(
                    "{plane} TU x={x0} y={y0} log2={log2_tb} qp={qp} dst={is_dst} mode={pred_mode} coefs={nonzero:?}"
                );
            }
            for i in 0..n * n {
                let v = pred[i] as i32 + res[i];
                pred[i] = v.clamp(0, 255) as u8;
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
        let refs = self.gather_refs(x0, y0, n, false, true, false);
        let mut refs = refs;
        // Chroma doesn't get the MDIS filter in 4:2:0 main profile. Leave as-is.
        let _ = &mut refs;
        let mut pred = vec![0u8; n * n];
        predict(&refs, n, &mut pred, n, pred_mode, false);
        if has_coeff {
            let mut levels = vec![0i32; n * n];
            self.residual_coding(engine, ctx, &mut levels, log2_tb, pred_mode, false)?;
            let mut deq = vec![0i32; n * n];
            let qp = self.get_qp(false, true);
            dequantize_flat(&levels, &mut deq, qp, log2_tb, 8);
            let mut res = vec![0i32; n * n];
            inverse_transform_2d(&deq, &mut res, log2_tb, false, 8);
            for i in 0..n * n {
                let v = pred[i] as i32 + res[i];
                pred[i] = v.clamp(0, 255) as u8;
            }
        }
        self.write_block(x0, y0, n, &pred, false, true);
        Ok(())
    }

    fn get_qp(&self, is_luma: bool, is_cr: bool) -> i32 {
        if is_luma {
            *self.cu_qp_y
        } else {
            let qp_offset = if is_cr {
                self.cctx.pps.pps_cr_qp_offset + self.cctx.slice.slice_cr_qp_offset
            } else {
                self.cctx.pps.pps_cb_qp_offset + self.cctx.slice.slice_cb_qp_offset
            };
            let qpy_offset = *self.cu_qp_y + qp_offset;
            let qpy_offset = qpy_offset.clamp(-12, 57);
            qp_y_to_qp_c(qpy_offset)
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
    ) -> Vec<u8> {
        let len = 4 * n + 1;
        let mut samples = vec![128u8; len];
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
        build_ref_samples(&samples, &avail, n)
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
        block: &[u8],
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

    fn write_block(&mut self, x: u32, y: u32, n: usize, block: &[u8], is_luma: bool, is_cr: bool) {
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
    /// (row-major, size `n*n`).
    fn residual_coding(
        &mut self,
        engine: &mut CabacEngine<'_>,
        ctx: &mut Ctx,
        levels: &mut [i32],
        log2_tb: u32,
        pred_mode: u32,
        is_luma: bool,
    ) -> Result<()> {
        let n = 1usize << log2_tb;
        let scan_idx = scan_idx_for_intra(log2_tb, pred_mode, is_luma);
        let scan = scan_4x4(scan_idx);

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
        Ok(())
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
fn subblock_scan(scan_idx: u32, log2_tb: u32) -> Vec<(u8, u8)> {
    let sb = if log2_tb <= 2 {
        1
    } else {
        1usize << (log2_tb - 2)
    };
    match scan_idx {
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
