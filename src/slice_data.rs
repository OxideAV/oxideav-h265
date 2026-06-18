//! §7.3.8.1 .. §7.3.8.6 slice-data CABAC syntax-element walk.
//!
//! This module is the upper rung of the §7.3.8 slice-data parse loop:
//! it drives the CABAC engine through the per-CTU syntax structures —
//! §7.3.8.3 `sao( )`, §7.3.8.4 `coding_tree_unit( )` /
//! `coding_quadtree( )`, §7.3.8.5 `coding_unit( )`, and §7.3.8.6
//! `prediction_unit( )` — composing the §7.3.8.3 / §7.3.8.5 / §7.3.8.6
//! leaf decode primitives ([`crate::binarization`]) with the already
//! implemented §7.3.8.8 `transform_tree( )` recursion
//! ([`crate::transform_tree`]) and its §7.3.8.10 `transform_unit( )`
//! leaf.
//!
//! The driver produces a structured parse tree (`CodingTreeUnit` →
//! `CodingQuadtree` → `CodingUnit`) rather than reconstructed samples:
//! it decodes the complete CABAC syntax-element stream of a CTU, which
//! is the prerequisite the §8.4 / §8.5 picture-reconstruction passes
//! consume. Picture-level neighbour availability (§6.4.1) and the
//! `CtDepth` / `cu_skip_flag` neighbour grids feeding the §9.3.4.2.2
//! `split_cu_flag` / `cu_skip_flag` ctxInc derivations are threaded
//! through a small per-CTU [`CtuGrid`] keyed at minimum-coding-block
//! (`MinCbSizeY`) granularity.
//!
//! The §6.5.1 quantization-group reset (`IsCuQpDeltaCoded`,
//! `CuQpDeltaVal`, `IsCuChromaQpOffsetCoded`) is performed by the
//! §7.3.8.4 `coding_quadtree( )` walk at each node whose `log2CbSize`
//! meets the `Log2MinCuQpDeltaSize` / `Log2MinCuChromaQpOffsetSize`
//! threshold, mirroring the syntax table; the resulting
//! [`crate::transform_unit::QuantGroupState`] is threaded into the
//! transform tree.

use crate::binarization::{
    cu_pred_mode_from_flag, cu_pred_mode_from_skip, cu_skip_flag_ctx_inc, decode_cu_skip_flag,
    decode_cu_transquant_bypass_flag, decode_end_of_slice_segment_flag, decode_inter_pred_idc,
    decode_intra_chroma_pred_mode, decode_merge_flag, decode_merge_idx, decode_mpm_idx,
    decode_mvd_component, decode_mvp_flag, decode_part_mode, decode_pcm_flag,
    decode_pred_mode_flag, decode_prev_intra_luma_pred_flag, decode_ref_idx,
    decode_rem_intra_luma_pred_mode, decode_rqt_root_cbf, decode_sao_band_position,
    decode_sao_eo_class, decode_sao_merge_flag, decode_sao_offset_abs, decode_sao_offset_sign,
    decode_sao_type_idx, decode_split_cu_flag, split_cu_flag_ctx_inc, CuPredMode, InterPredIdc,
    MvdComponent, PartMode, PartModeResult,
};
use crate::cabac::CabacEngine;
use crate::ctx_init::SliceContexts;
use crate::residual::ResidualCodingError;
use crate::transform_tree::{decode_transform_tree, TransformTree, TransformTreeParams};
use crate::transform_unit::{CuPredMode as TuCuPredMode, QuantGroupState, TransformUnitParams};

/// Map the §7.3.8.5 [`binarization::CuPredMode`](CuPredMode) (which
/// carries the `MODE_SKIP` not-present variant) to the two-state
/// [`crate::transform_unit::CuPredMode`] the transform tree / unit
/// consume. A skip CU never enters the transform tree (it has no
/// residual), so `Skip` collapses to `Inter` defensively.
fn to_tu_pred_mode(m: CuPredMode) -> TuCuPredMode {
    match m {
        CuPredMode::Intra => TuCuPredMode::Intra,
        CuPredMode::Inter | CuPredMode::Skip => TuCuPredMode::Inter,
    }
}

/// Per-CTU sequence / picture / slice constants the §7.3.8 walk reads.
/// These derive from the active SPS / PPS / slice header (§7.4.3) and
/// are constant for one slice segment's worth of CTUs.
#[derive(Debug, Clone, Copy)]
pub struct SliceDataParams {
    /// `CtbLog2SizeY` (§7.4.3.2) — the luma coding-tree-block log2 size.
    pub ctb_log2_size_y: u32,
    /// `MinCbLog2SizeY` (§7.4.3.2) — the minimum luma coding-block log2
    /// size; bounds the §7.3.8.4 `split_cu_flag` presence gate.
    pub min_cb_log2_size_y: u32,
    /// `MaxTbLog2SizeY` (§7.4.3.2).
    pub max_tb_log2_size_y: u32,
    /// `MinTbLog2SizeY` (§7.4.3.2).
    pub min_tb_log2_size_y: u32,
    /// `pic_width_in_luma_samples` (§7.4.3.2.1).
    pub pic_width_in_luma_samples: u32,
    /// `pic_height_in_luma_samples` (§7.4.3.2.1).
    pub pic_height_in_luma_samples: u32,
    /// `ChromaArrayType` (0 = monochrome, 1 = 4:2:0, 2 = 4:2:2,
    /// 3 = 4:4:4).
    pub chroma_array_type: u8,
    /// `BitDepthY` (§7.4.3.2.1) — used by §7.3.8.3 `sao_offset_abs`.
    pub bit_depth_luma: u32,
    /// `BitDepthC` (§7.4.3.2.1) — used by §7.3.8.3 `sao_offset_abs`.
    pub bit_depth_chroma: u32,
    /// `slice_type == I` (§7.4.7.1).
    pub slice_type_is_i: bool,
    /// `slice_type == B` (§7.4.7.1).
    pub slice_type_is_b: bool,
    /// `slice_sao_luma_flag` (§7.4.7.1).
    pub slice_sao_luma_flag: bool,
    /// `slice_sao_chroma_flag` (§7.4.7.1).
    pub slice_sao_chroma_flag: bool,
    /// `transquant_bypass_enabled_flag` (§7.4.3.3.1).
    pub transquant_bypass_enabled_flag: bool,
    /// `cu_qp_delta_enabled_flag` (§7.4.3.3.1).
    pub cu_qp_delta_enabled_flag: bool,
    /// `Log2MinCuQpDeltaSize` (§7.4.3.3.1) = `CtbLog2SizeY −
    /// diff_cu_qp_delta_depth`.
    pub log2_min_cu_qp_delta_size: u32,
    /// per-slice `cu_chroma_qp_offset_enabled_flag` (§7.4.9.10).
    pub cu_chroma_qp_offset_enabled_flag: bool,
    /// `Log2MinCuChromaQpOffsetSize` (§7.4.3.3.1).
    pub log2_min_cu_chroma_qp_offset_size: u32,
    /// `chroma_qp_offset_list_len_minus1` (§7.4.3.3.1).
    pub chroma_qp_offset_list_len_minus1: u32,
    /// `amp_enabled_flag` (§7.4.3.2.1).
    pub amp_enabled_flag: bool,
    /// PCM block: `pcm_enabled_flag` (§7.4.3.2.1).
    pub pcm_enabled_flag: bool,
    /// `Log2MinIpcmCbSizeY` (§7.4.3.2.1).
    pub log2_min_ipcm_cb_size_y: u32,
    /// `Log2MaxIpcmCbSizeY` (§7.4.3.2.1).
    pub log2_max_ipcm_cb_size_y: u32,
    /// `max_transform_hierarchy_depth_intra` (§7.4.3.2.1).
    pub max_transform_hierarchy_depth_intra: u32,
    /// `max_transform_hierarchy_depth_inter` (§7.4.3.2.1).
    pub max_transform_hierarchy_depth_inter: u32,
    /// `MaxNumMergeCand` (§7.4.7.1) — `5 −
    /// five_minus_max_num_merge_cand`.
    pub max_num_merge_cand: u32,
    /// `num_ref_idx_l0_active_minus1` (§7.4.7.1).
    pub num_ref_idx_l0_active_minus1: u32,
    /// `num_ref_idx_l1_active_minus1` (§7.4.7.1).
    pub num_ref_idx_l1_active_minus1: u32,
    /// `mvd_l1_zero_flag` (§7.4.7.1).
    pub mvd_l1_zero_flag: bool,
    /// PPS `sign_data_hiding_enabled_flag` (§7.4.3.3.1).
    pub sign_data_hiding_enabled_flag: bool,
    /// PPS `cross_component_prediction_enabled_flag` (§7.4.3.3.1).
    pub cross_component_prediction_enabled_flag: bool,
    /// SCC `residual_adaptive_colour_transform_enabled_flag`
    /// (§7.4.3.3.1).
    pub residual_adaptive_colour_transform_enabled_flag: bool,
}

/// §7.4.9.3 decoded SAO parameters for one colour component of one CTB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SaoComponent {
    /// `SaoTypeIdx[cIdx][rx][ry]` — 0 (not applied), 1 (band offset),
    /// 2 (edge offset).
    pub sao_type_idx: u8,
    /// `sao_offset_abs[cIdx][rx][ry][0..4]` magnitudes.
    pub offset_abs: [u32; 4],
    /// `sao_offset_sign[cIdx][rx][ry][0..4]` (band offset only).
    pub offset_sign: [u8; 4],
    /// `sao_band_position[cIdx][rx][ry]` (band offset only).
    pub band_position: u8,
    /// `SaoEoClass[cIdx][rx][ry]` (edge offset only).
    pub eo_class: u8,
}

/// §7.3.8.3 decoded SAO parameters for one CTB (all three components).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SaoCtbParams {
    /// `sao_merge_left_flag` — this CTB copies the left CTB's params.
    pub merge_left: bool,
    /// `sao_merge_up_flag` — this CTB copies the above CTB's params.
    pub merge_up: bool,
    /// Per-component parameters: `[Y, Cb, Cr]`. Only populated when
    /// neither merge flag is set (otherwise the caller resolves the
    /// merged source).
    pub components: [SaoComponent; 3],
}

/// One decoded §7.3.8.5 `coding_unit( )`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodingUnit {
    /// Luma top-left position `(x0, y0)`.
    pub x0: u32,
    /// Luma top-left position `(x0, y0)`.
    pub y0: u32,
    /// `log2CbSize`.
    pub log2_cb_size: u32,
    /// `CuPredMode[x0][y0]`.
    pub cu_pred_mode: CuPredMode,
    /// `cu_transquant_bypass_flag`.
    pub cu_transquant_bypass_flag: bool,
    /// `PartMode` (§7.4.9.5).
    pub part_mode: PartMode,
    /// `pcm_flag[x0][y0]`.
    pub pcm_flag: bool,
    /// Decoded prediction units (intra: empty — the luma/chroma intra
    /// modes carry the prediction; inter: 1..=4 entries).
    pub prediction_units: Vec<PredictionUnit>,
    /// `prev_intra_luma_pred_flag` / `mpm_idx` /
    /// `rem_intra_luma_pred_mode` per luma prediction block (intra
    /// only), in §7.3.8.5 PB-loop order.
    pub intra_luma: Vec<IntraLumaMode>,
    /// `intra_chroma_pred_mode` values (intra only).
    pub intra_chroma_pred_mode: Vec<u8>,
    /// `rqt_root_cbf` (inter only; intra CUs always enter the tree).
    pub rqt_root_cbf: bool,
    /// The decoded §7.3.8.8 transform tree, present when the CU codes
    /// residual (`!pcm_flag && (intra || rqt_root_cbf)`).
    pub transform_tree: Option<TransformTree>,
}

/// §7.3.8.5 per-luma-prediction-block intra-mode signalling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntraLumaMode {
    /// `prev_intra_luma_pred_flag[xPb][yPb]`.
    pub prev_intra_luma_pred_flag: bool,
    /// `mpm_idx[xPb][yPb]` (present when `prev_intra_luma_pred_flag`).
    pub mpm_idx: Option<u8>,
    /// `rem_intra_luma_pred_mode[xPb][yPb]` (present otherwise).
    pub rem_intra_luma_pred_mode: Option<u8>,
}

/// §7.3.8.6 one decoded `prediction_unit( )`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PredictionUnit {
    /// `merge_flag[x0][y0]` (`false` for a skip CU, which carries only
    /// `merge_idx`).
    pub merge_flag: bool,
    /// `merge_idx[x0][y0]` (present for skip or merge).
    pub merge_idx: Option<u8>,
    /// `inter_pred_idc[x0][y0]` (present for non-merge B PUs; a P-slice
    /// non-merge PU is `PRED_L0`).
    pub inter_pred_idc: Option<InterPredIdc>,
    /// `ref_idx_l0[x0][y0]` (present for L0/BI non-merge PUs).
    pub ref_idx_l0: Option<u8>,
    /// `mvd_coding(…, 0)` (present for L0/BI non-merge PUs).
    pub mvd_l0: Option<[MvdComponent; 2]>,
    /// `mvp_l0_flag[x0][y0]`.
    pub mvp_l0_flag: Option<u8>,
    /// `ref_idx_l1[x0][y0]` (present for L1/BI non-merge PUs).
    pub ref_idx_l1: Option<u8>,
    /// `mvd_coding(…, 1)` (present for L1/BI non-merge PUs unless the
    /// `mvd_l1_zero_flag && PRED_BI` zero-inference path applies).
    pub mvd_l1: Option<[MvdComponent; 2]>,
    /// `mvp_l1_flag[x0][y0]`.
    pub mvp_l1_flag: Option<u8>,
}

/// One node of a decoded §7.3.8.4 `coding_quadtree( )`: a split node
/// with up to four in-picture children, or a leaf coding unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodingQuadtree {
    /// `split_cu_flag == 1`: the present (in-picture) children, raster
    /// order; off-picture children are absent (the §7.3.8.4 boundary
    /// `if( x1 < … )` guards).
    Split(Vec<CodingQuadtree>),
    /// `split_cu_flag == 0`: a leaf coding unit.
    Leaf(Box<CodingUnit>),
}

/// One decoded §7.3.8.2 `coding_tree_unit( )`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodingTreeUnit {
    /// The §7.3.8.3 SAO parameters, when `slice_sao_luma_flag ||
    /// slice_sao_chroma_flag`.
    pub sao: Option<SaoCtbParams>,
    /// The §7.3.8.4 coding-quadtree root.
    pub quadtree: CodingQuadtree,
}

/// Per-CTU neighbour-state grid at minimum-coding-block granularity.
/// Records `CtDepth[x][y]` and `cu_skip_flag[x][y]` for the
/// §9.3.4.2.2 `split_cu_flag` / `cu_skip_flag` left/above ctxInc
/// derivations.
///
/// The grid spans one CTB (`(1 << CtbLog2SizeY)` luma samples square),
/// addressed in `MinCb` units. Neighbour lookups that fall outside the
/// CTB to the left or above are treated as unavailable for ctxInc
/// purposes (a conservative within-CTU model: full cross-CTU
/// availability is the picture-level §6.4.1 reconstruction pass's
/// responsibility).
#[derive(Debug, Clone)]
pub struct CtuGrid {
    min_cb_log2: u32,
    ctb_log2: u32,
    dim: u32,
    ct_depth: Vec<u8>,
    cu_skip: Vec<u8>,
    x_ctb: u32,
    y_ctb: u32,
}

impl CtuGrid {
    /// Construct a fresh grid for the CTB whose top-left luma position
    /// is `(x_ctb, y_ctb)`.
    #[must_use]
    pub fn new(params: &SliceDataParams, x_ctb: u32, y_ctb: u32) -> Self {
        let dim = 1u32 << (params.ctb_log2_size_y - params.min_cb_log2_size_y);
        let n = (dim * dim) as usize;
        Self {
            min_cb_log2: params.min_cb_log2_size_y,
            ctb_log2: params.ctb_log2_size_y,
            dim,
            ct_depth: vec![0u8; n],
            cu_skip: vec![0u8; n],
            x_ctb,
            y_ctb,
        }
    }

    /// Convert an absolute luma position to a grid index, or `None`
    /// when it lies outside this CTB.
    fn idx(&self, x: u32, y: u32) -> Option<usize> {
        if x < self.x_ctb || y < self.y_ctb {
            return None;
        }
        let lx = (x - self.x_ctb) >> self.min_cb_log2;
        let ly = (y - self.y_ctb) >> self.min_cb_log2;
        if lx >= self.dim || ly >= self.dim {
            return None;
        }
        Some((ly * self.dim + lx) as usize)
    }

    /// Mark a coding block covering `[x0, x0 + size) × [y0, y0 + size)`
    /// with the given `CtDepth` and `cu_skip_flag`.
    fn mark(&mut self, x0: u32, y0: u32, log2_cb_size: u32, ct_depth: u8, cu_skip: u8) {
        let size = 1u32 << log2_cb_size;
        let step = 1u32 << self.min_cb_log2;
        let mut y = y0;
        while y < y0 + size {
            let mut x = x0;
            while x < x0 + size {
                if let Some(i) = self.idx(x, y) {
                    self.ct_depth[i] = ct_depth;
                    self.cu_skip[i] = cu_skip;
                }
                x += step;
            }
            y += step;
        }
        let _ = self.ctb_log2;
    }

    /// `(CtDepth, available)` for the block to the left of `(x0, y0)`.
    fn left_ct_depth(&self, x0: u32, y0: u32) -> (u32, bool) {
        if x0 == 0 {
            return (0, false);
        }
        match self.idx(x0 - 1, y0) {
            Some(i) => (self.ct_depth[i] as u32, true),
            None => (0, false),
        }
    }

    /// `(CtDepth, available)` for the block above `(x0, y0)`.
    fn above_ct_depth(&self, x0: u32, y0: u32) -> (u32, bool) {
        if y0 == 0 {
            return (0, false);
        }
        match self.idx(x0, y0 - 1) {
            Some(i) => (self.ct_depth[i] as u32, true),
            None => (0, false),
        }
    }

    /// `(cu_skip_flag, available)` for the block to the left.
    fn left_cu_skip(&self, x0: u32, y0: u32) -> (u8, bool) {
        if x0 == 0 {
            return (0, false);
        }
        match self.idx(x0 - 1, y0) {
            Some(i) => (self.cu_skip[i], true),
            None => (0, false),
        }
    }

    /// `(cu_skip_flag, available)` for the block above.
    fn above_cu_skip(&self, x0: u32, y0: u32) -> (u8, bool) {
        if y0 == 0 {
            return (0, false);
        }
        match self.idx(x0, y0 - 1) {
            Some(i) => (self.cu_skip[i], true),
            None => (0, false),
        }
    }
}

/// Decode one §7.3.8.3 `sao( rx, ry )` syntax structure.
///
/// `merge_left_allowed` / `merge_up_allowed` are the §7.3.8.3 presence
/// conditions for the two merge flags (`rx > 0 && leftCtbInSliceSeg &&
/// leftCtbInTile` and the symmetric up condition), already evaluated by
/// the caller against the §6.5 tile / slice geometry. When the merge
/// path is taken the per-component fields are left at their defaults.
pub fn decode_sao(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &SliceDataParams,
    merge_left_allowed: bool,
    merge_up_allowed: bool,
) -> Result<SaoCtbParams, ResidualCodingError> {
    let mut out = SaoCtbParams::default();

    if merge_left_allowed {
        out.merge_left = decode_sao_merge_flag(engine, &mut ctx.sao_merge_flag[0])? != 0;
    }
    if merge_up_allowed && !out.merge_left {
        out.merge_up = decode_sao_merge_flag(engine, &mut ctx.sao_merge_flag[0])? != 0;
    }
    if out.merge_left || out.merge_up {
        return Ok(out);
    }

    let num_comp = if params.chroma_array_type != 0 { 3 } else { 1 };
    for c_idx in 0..num_comp {
        let read = (params.slice_sao_luma_flag && c_idx == 0)
            || (params.slice_sao_chroma_flag && c_idx > 0);
        if !read {
            continue;
        }
        // sao_type_idx_luma (cIdx 0) / sao_type_idx_chroma (cIdx 1) are
        // both read from the wire (sharing the Table 9-5 bank). For
        // cIdx == 2 the §7.3.8.3 syntax has no read — SaoTypeIdx[2] is
        // inferred equal to SaoTypeIdx[1] (§7.4.9.3).
        let type_idx = if c_idx < 2 {
            decode_sao_type_idx(engine, &mut ctx.sao_type_idx[0])?
        } else {
            out.components[1].sao_type_idx
        };
        out.components[c_idx as usize].sao_type_idx = type_idx;

        if type_idx != 0 {
            let bit_depth = if c_idx == 0 {
                params.bit_depth_luma
            } else {
                params.bit_depth_chroma
            };
            for i in 0..4 {
                out.components[c_idx as usize].offset_abs[i] =
                    decode_sao_offset_abs(engine, bit_depth)?;
            }
            if type_idx == 1 {
                // Band offset: per-i sign + band position.
                for i in 0..4 {
                    if out.components[c_idx as usize].offset_abs[i] != 0 {
                        out.components[c_idx as usize].offset_sign[i] =
                            decode_sao_offset_sign(engine)?;
                    }
                }
                out.components[c_idx as usize].band_position = decode_sao_band_position(engine)?;
            } else {
                // Edge offset: eo_class for cIdx 0 and 1 (cIdx 2 shares
                // cIdx 1's eo_class per §7.4.9.3).
                if c_idx == 0 || c_idx == 1 {
                    out.components[c_idx as usize].eo_class = decode_sao_eo_class(engine)?;
                } else {
                    out.components[2].eo_class = out.components[1].eo_class;
                }
            }
        }
    }
    // §7.4.9.3: SaoTypeIdx[2] / eo_class[2] inherit cIdx 1.
    if num_comp == 3 {
        out.components[2].sao_type_idx = out.components[1].sao_type_idx;
        if out.components[2].sao_type_idx == 2 {
            out.components[2].eo_class = out.components[1].eo_class;
        }
    }
    Ok(out)
}

/// Build the constant (non-geometry) part of a §7.3.8.10
/// `TransformUnitParams` template from the slice-data params + CU
/// context. The §7.3.8.8 transform-tree walk overwrites the per-node
/// geometry / cbf fields before each leaf.
fn tu_template(
    params: &SliceDataParams,
    cu_pred_mode: CuPredMode,
    cu_transquant_bypass_flag: bool,
    part_mode_2nx2n: bool,
) -> TransformUnitParams {
    TransformUnitParams {
        log2_trafo_size: 0,
        trafo_depth: 0,
        blk_idx: 0,
        cu_pred_mode: to_tu_pred_mode(cu_pred_mode),
        chroma_array_type: params.chroma_array_type,
        cbf_luma: false,
        cbf_cb: false,
        cbf_cb_lower: false,
        cbf_cr: false,
        cbf_cr_lower: false,
        intra_pred_mode_y: 0,
        intra_pred_mode_c: 0,
        intra_chroma_pred_mode: 0,
        cu_qp_delta_enabled_flag: params.cu_qp_delta_enabled_flag,
        cu_chroma_qp_offset_enabled_flag: params.cu_chroma_qp_offset_enabled_flag,
        chroma_qp_offset_list_len_minus1: params.chroma_qp_offset_list_len_minus1,
        cu_transquant_bypass_flag,
        sign_data_hiding_enabled_flag: params.sign_data_hiding_enabled_flag,
        cross_component_prediction_enabled_flag: params.cross_component_prediction_enabled_flag,
        residual_adaptive_colour_transform_enabled_flag: params
            .residual_adaptive_colour_transform_enabled_flag,
        part_mode_2nx2n,
        intra_chroma_pred_mode_corners: [0; 4],
    }
}

/// Decode one §7.3.8.6 `prediction_unit( x0, y0, nPbW, nPbH )`.
fn decode_prediction_unit(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &SliceDataParams,
    cu_skip_flag: bool,
    ct_depth: u32,
    n_pb_w: u32,
    n_pb_h: u32,
) -> Result<PredictionUnit, ResidualCodingError> {
    let mut pu = PredictionUnit {
        merge_flag: false,
        merge_idx: None,
        inter_pred_idc: None,
        ref_idx_l0: None,
        mvd_l0: None,
        mvp_l0_flag: None,
        ref_idx_l1: None,
        mvd_l1: None,
        mvp_l1_flag: None,
    };

    if cu_skip_flag {
        if params.max_num_merge_cand > 1 {
            pu.merge_idx = Some(decode_merge_idx(
                engine,
                &mut ctx.merge_idx[0],
                params.max_num_merge_cand,
            )?);
        } else {
            pu.merge_idx = Some(0);
        }
        return Ok(pu);
    }

    // MODE_INTER.
    pu.merge_flag = decode_merge_flag(engine, &mut ctx.merge_flag[0])? != 0;
    if pu.merge_flag {
        if params.max_num_merge_cand > 1 {
            pu.merge_idx = Some(decode_merge_idx(
                engine,
                &mut ctx.merge_idx[0],
                params.max_num_merge_cand,
            )?);
        } else {
            pu.merge_idx = Some(0);
        }
        return Ok(pu);
    }

    // Non-merge inter.
    let pred_idc = if params.slice_type_is_b {
        // inter_pred_idc bin 0 ctxInc = (nPbW+nPbH != 12) ? CtDepth : 4;
        // bin 1 ctxInc = 4. Bank slot layout: indices 0..=3 are the
        // CtDepth-keyed slots, index 4 is the shared bin-1 slot.
        let b0_slot = if n_pb_w + n_pb_h != 12 {
            ct_depth.min(3) as usize
        } else {
            4
        };
        // Borrow two distinct slots from the inter_pred_idc bank.
        let idc = decode_inter_pred_idc_banked(engine, ctx, b0_slot, n_pb_w, n_pb_h)?;
        pu.inter_pred_idc = Some(idc);
        idc
    } else {
        // P slice: a non-merge inter PB is PRED_L0 (§7.4.9.6) — no
        // inter_pred_idc on the wire.
        InterPredIdc::PredL0
    };

    if pred_idc != InterPredIdc::PredL1 {
        // L0 path.
        if params.num_ref_idx_l0_active_minus1 > 0 {
            pu.ref_idx_l0 = Some(decode_ref_idx_l0(engine, ctx, params)?);
        } else {
            pu.ref_idx_l0 = Some(0);
        }
        let c0 = decode_mvd_l(engine, ctx)?;
        let c1 = decode_mvd_l(engine, ctx)?;
        pu.mvd_l0 = Some([c0, c1]);
        pu.mvp_l0_flag = Some(decode_mvp_flag(engine, &mut ctx.mvp_flag[0])?);
    }
    if pred_idc != InterPredIdc::PredL0 {
        // L1 path.
        if params.num_ref_idx_l1_active_minus1 > 0 {
            pu.ref_idx_l1 = Some(decode_ref_idx_l1(engine, ctx, params)?);
        } else {
            pu.ref_idx_l1 = Some(0);
        }
        if params.mvd_l1_zero_flag && pred_idc == InterPredIdc::PredBi {
            // MvdL1 inferred zero; mvd_coding not read.
            pu.mvd_l1 = None;
        } else {
            let c0 = decode_mvd_l(engine, ctx)?;
            let c1 = decode_mvd_l(engine, ctx)?;
            pu.mvd_l1 = Some([c0, c1]);
        }
        pu.mvp_l1_flag = Some(decode_mvp_flag(engine, &mut ctx.mvp_flag[0])?);
    }

    Ok(pu)
}

/// Helper: decode `inter_pred_idc` borrowing two distinct slots from
/// the `inter_pred_idc[5]` bank (bin-0 slot `b0_slot`, bin-1 slot 4).
fn decode_inter_pred_idc_banked(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    b0_slot: usize,
    n_pb_w: u32,
    n_pb_h: u32,
) -> Result<InterPredIdc, ResidualCodingError> {
    // The bank is `[ContextModel; 5]`; bin-0 slot is in 0..=4 and bin-1
    // slot is fixed at 4. When they collide (b0_slot == 4, the
    // nPbW+nPbH==12 single-bin case) only bin 0 is read, so bin 1's
    // context is never dereferenced — a throwaway copy is harmless.
    if b0_slot == 4 {
        let mut dummy = ctx.inter_pred_idc[4];
        let r = decode_inter_pred_idc(
            engine,
            &mut ctx.inter_pred_idc[4],
            &mut dummy,
            n_pb_w,
            n_pb_h,
        )?;
        return Ok(r);
    }
    let (head, tail) = ctx.inter_pred_idc.split_at_mut(4);
    let b0 = &mut head[b0_slot];
    let b1 = &mut tail[0];
    Ok(decode_inter_pred_idc(engine, b0, b1, n_pb_w, n_pb_h)?)
}

/// Decode one `mvd_coding( )` component, borrowing the two
/// `abs_mvd_greater0_flag` / `abs_mvd_greater1_flag` contexts.
fn decode_mvd_l(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
) -> Result<MvdComponent, ResidualCodingError> {
    Ok(decode_mvd_component(
        engine,
        &mut ctx.abs_mvd_greater0_flag[0],
        &mut ctx.abs_mvd_greater1_flag[0],
    )?)
}

fn decode_ref_idx_l0(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &SliceDataParams,
) -> Result<u8, ResidualCodingError> {
    let (c0, c1) = ctx.ref_idx.split_at_mut(1);
    Ok(decode_ref_idx(
        engine,
        &mut c0[0],
        &mut c1[0],
        params.num_ref_idx_l0_active_minus1,
    )?)
}

fn decode_ref_idx_l1(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &SliceDataParams,
) -> Result<u8, ResidualCodingError> {
    let (c0, c1) = ctx.ref_idx.split_at_mut(1);
    Ok(decode_ref_idx(
        engine,
        &mut c0[0],
        &mut c1[0],
        params.num_ref_idx_l1_active_minus1,
    )?)
}

/// Decode one §7.3.8.5 `coding_unit( x0, y0, log2CbSize )`.
#[allow(clippy::too_many_arguments)]
fn decode_coding_unit(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &SliceDataParams,
    grid: &mut CtuGrid,
    qg: &mut QuantGroupState,
    x0: u32,
    y0: u32,
    log2_cb_size: u32,
    ct_depth: u32,
) -> Result<CodingUnit, ResidualCodingError> {
    let n_cb_s = 1u32 << log2_cb_size;

    let cu_transquant_bypass_flag = if params.transquant_bypass_enabled_flag {
        decode_cu_transquant_bypass_flag(engine, &mut ctx.cu_transquant_bypass_flag[0])? != 0
    } else {
        false
    };

    // cu_skip_flag (P/B only).
    let cu_skip_flag = if !params.slice_type_is_i {
        let (l_skip, l_avail) = grid.left_cu_skip(x0, y0);
        let (a_skip, a_avail) = grid.above_cu_skip(x0, y0);
        let inc = cu_skip_flag_ctx_inc(l_skip, l_avail, a_skip, a_avail) as usize;
        decode_cu_skip_flag(engine, &mut ctx.cu_skip_flag[inc])? != 0
    } else {
        false
    };
    grid.mark(x0, y0, log2_cb_size, ct_depth as u8, cu_skip_flag as u8);

    let mut cu = CodingUnit {
        x0,
        y0,
        log2_cb_size,
        cu_pred_mode: CuPredMode::Intra,
        cu_transquant_bypass_flag,
        part_mode: PartMode::Part2Nx2N,
        pcm_flag: false,
        prediction_units: Vec::new(),
        intra_luma: Vec::new(),
        intra_chroma_pred_mode: Vec::new(),
        rqt_root_cbf: false,
        transform_tree: None,
    };

    if cu_skip_flag {
        // §7.4.9.5: MODE_SKIP. One prediction_unit covering the CU.
        cu.cu_pred_mode = cu_pred_mode_from_skip(params.slice_type_is_i, 1).unwrap();
        let pu = decode_prediction_unit(engine, ctx, params, true, ct_depth, n_cb_s, n_cb_s)?;
        cu.prediction_units.push(pu);
        return Ok(cu);
    }

    // pred_mode_flag (P/B), else inferred MODE_INTRA on I slices.
    let cu_pred_mode = if params.slice_type_is_i {
        CuPredMode::Intra
    } else {
        let flag = decode_pred_mode_flag(engine, &mut ctx.pred_mode_flag[0])?;
        cu_pred_mode_from_flag(flag)
    };
    cu.cu_pred_mode = cu_pred_mode;

    // part_mode: present when MODE_INTER or log2CbSize == MinCbLog2SizeY.
    let part_present =
        cu_pred_mode != CuPredMode::Intra || log2_cb_size == params.min_cb_log2_size_y;
    let part_result: PartModeResult = if part_present {
        decode_part_mode_banked(engine, ctx, cu_pred_mode, log2_cb_size, params)?
    } else {
        crate::binarization::part_mode_inferred()
    };
    cu.part_mode = part_result.part_mode;

    if cu_pred_mode == CuPredMode::Intra {
        decode_intra_cu(
            engine,
            ctx,
            params,
            &mut cu,
            x0,
            y0,
            log2_cb_size,
            part_result,
            qg,
        )?;
    } else {
        decode_inter_cu(
            engine,
            ctx,
            params,
            &mut cu,
            x0,
            y0,
            log2_cb_size,
            ct_depth,
            part_result,
            qg,
        )?;
    }

    Ok(cu)
}

/// Decode `part_mode` borrowing three distinct slots from the
/// `part_mode[4]` bank. Bin 0 → slot 0, bin 1 → slot 1, bin 2 → slot 2
/// (`log2CbSize == MinCbLog2SizeY`) or slot 3 (`>`).
fn decode_part_mode_banked(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    cu_pred_mode: CuPredMode,
    log2_cb_size: u32,
    params: &SliceDataParams,
) -> Result<PartModeResult, ResidualCodingError> {
    let bin2_slot = if log2_cb_size == params.min_cb_log2_size_y {
        2usize
    } else {
        3usize
    };
    // Borrow slot 0, 1 and bin2_slot (2 or 3) — all distinct.
    let (lo, hi) = ctx.part_mode.split_at_mut(2);
    let (c0, c1) = lo.split_at_mut(1);
    let bin2 = &mut hi[bin2_slot - 2];
    Ok(decode_part_mode(
        engine,
        &mut c0[0],
        &mut c1[0],
        bin2,
        cu_pred_mode,
        log2_cb_size,
        params.min_cb_log2_size_y,
        params.amp_enabled_flag,
    )?)
}

#[allow(clippy::too_many_arguments)]
fn decode_intra_cu(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &SliceDataParams,
    cu: &mut CodingUnit,
    x0: u32,
    y0: u32,
    log2_cb_size: u32,
    part_result: PartModeResult,
    qg: &mut QuantGroupState,
) -> Result<(), ResidualCodingError> {
    let n_cb_s = 1u32 << log2_cb_size;

    // PCM gate.
    let pcm_present = part_result.part_mode == PartMode::Part2Nx2N
        && params.pcm_enabled_flag
        && log2_cb_size >= params.log2_min_ipcm_cb_size_y
        && log2_cb_size <= params.log2_max_ipcm_cb_size_y;
    if pcm_present {
        cu.pcm_flag = decode_pcm_flag(engine)? != 0;
    }
    if cu.pcm_flag {
        // PCM samples are byte-aligned raw u(v) reads — out of scope for
        // the CABAC syntax walk (handled by the reconstruction pass that
        // re-aligns the bit reader). Mark and return.
        return Ok(());
    }

    // Luma intra mode signalling group.
    let pb_offset = if part_result.part_mode == PartMode::PartNxN {
        n_cb_s / 2
    } else {
        n_cb_s
    };
    let n_pb = (n_cb_s / pb_offset) as usize; // 1 or 2 per axis
    let count = n_pb * n_pb;

    let mut prev_flags = Vec::with_capacity(count);
    for _ in 0..count {
        let f =
            decode_prev_intra_luma_pred_flag(engine, &mut ctx.prev_intra_luma_pred_flag[0])? != 0;
        prev_flags.push(f);
    }
    for &f in &prev_flags {
        let mut entry = IntraLumaMode {
            prev_intra_luma_pred_flag: f,
            mpm_idx: None,
            rem_intra_luma_pred_mode: None,
        };
        if f {
            entry.mpm_idx = Some(decode_mpm_idx(engine)?);
        } else {
            entry.rem_intra_luma_pred_mode = Some(decode_rem_intra_luma_pred_mode(engine)?);
        }
        cu.intra_luma.push(entry);
    }

    // Chroma intra mode.
    if params.chroma_array_type == 3 {
        for _ in 0..count {
            cu.intra_chroma_pred_mode
                .push(decode_intra_chroma_pred_mode(
                    engine,
                    &mut ctx.intra_chroma_pred_mode[0],
                )?);
        }
    } else if params.chroma_array_type != 0 {
        cu.intra_chroma_pred_mode
            .push(decode_intra_chroma_pred_mode(
                engine,
                &mut ctx.intra_chroma_pred_mode[0],
            )?);
    }

    // Intra CUs always enter the transform tree (rqt_root_cbf is not
    // coded; cbf_luma presence at the root is unconditional).
    let max_trafo_depth =
        params.max_transform_hierarchy_depth_intra + part_result.intra_split_flag as u32;
    let tt_params = TransformTreeParams {
        max_tb_log2_size_y: params.max_tb_log2_size_y,
        min_tb_log2_size_y: params.min_tb_log2_size_y,
        max_trafo_depth,
        intra_split_flag: part_result.intra_split_flag,
        inter_split_flag: false,
        cu_pred_mode: TuCuPredMode::Intra,
        chroma_array_type: params.chroma_array_type,
        tu_template: tu_template(
            params,
            CuPredMode::Intra,
            cu.cu_transquant_bypass_flag,
            part_result.part_mode == PartMode::Part2Nx2N,
        ),
    };
    let tree = decode_transform_tree(
        engine,
        ctx,
        &tt_params,
        qg,
        x0,
        y0,
        x0,
        y0,
        log2_cb_size,
        0,
        0,
        false,
        false,
    )?;
    cu.transform_tree = Some(tree);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_inter_cu(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &SliceDataParams,
    cu: &mut CodingUnit,
    x0: u32,
    y0: u32,
    log2_cb_size: u32,
    ct_depth: u32,
    part_result: PartModeResult,
    qg: &mut QuantGroupState,
) -> Result<(), ResidualCodingError> {
    let n = 1u32 << log2_cb_size;
    let pm = part_result.part_mode;

    // §7.3.8.5: emit the prediction_unit calls per PartMode.
    let pu_rects: Vec<(u32, u32, u32, u32)> = match pm {
        PartMode::Part2Nx2N => vec![(x0, y0, n, n)],
        PartMode::Part2NxN => vec![(x0, y0, n, n / 2), (x0, y0 + n / 2, n, n / 2)],
        PartMode::PartNx2N => vec![(x0, y0, n / 2, n), (x0 + n / 2, y0, n / 2, n)],
        PartMode::Part2NxnU => vec![(x0, y0, n, n / 4), (x0, y0 + n / 4, n, n * 3 / 4)],
        PartMode::Part2NxnD => vec![(x0, y0, n, n * 3 / 4), (x0, y0 + n * 3 / 4, n, n / 4)],
        PartMode::PartNLx2N => vec![(x0, y0, n / 4, n), (x0 + n / 4, y0, n * 3 / 4, n)],
        PartMode::PartNRx2N => vec![(x0, y0, n * 3 / 4, n), (x0 + n * 3 / 4, y0, n / 4, n)],
        PartMode::PartNxN => vec![
            (x0, y0, n / 2, n / 2),
            (x0 + n / 2, y0, n / 2, n / 2),
            (x0, y0 + n / 2, n / 2, n / 2),
            (x0 + n / 2, y0 + n / 2, n / 2, n / 2),
        ],
    };
    for (px, py, pw, ph) in pu_rects {
        let _ = (px, py);
        let pu = decode_prediction_unit(engine, ctx, params, false, ct_depth, pw, ph)?;
        cu.prediction_units.push(pu);
    }

    // rqt_root_cbf: present unless PART_2Nx2N + merge.
    let single_merge = pm == PartMode::Part2Nx2N
        && cu
            .prediction_units
            .first()
            .map(|p| p.merge_flag)
            .unwrap_or(false);
    let rqt_root_cbf = if !single_merge {
        decode_rqt_root_cbf(engine, &mut ctx.rqt_root_cbf[0])? != 0
    } else {
        // §7.4.9.5: not present ⇒ inferred 1.
        true
    };
    cu.rqt_root_cbf = rqt_root_cbf;

    if rqt_root_cbf {
        let inter_split =
            params.max_transform_hierarchy_depth_inter == 0 && pm != PartMode::Part2Nx2N;
        let max_trafo_depth = params.max_transform_hierarchy_depth_inter;
        let tt_params = TransformTreeParams {
            max_tb_log2_size_y: params.max_tb_log2_size_y,
            min_tb_log2_size_y: params.min_tb_log2_size_y,
            max_trafo_depth,
            intra_split_flag: false,
            inter_split_flag: inter_split,
            cu_pred_mode: TuCuPredMode::Inter,
            chroma_array_type: params.chroma_array_type,
            tu_template: tu_template(
                params,
                CuPredMode::Inter,
                cu.cu_transquant_bypass_flag,
                pm == PartMode::Part2Nx2N,
            ),
        };
        let tree = decode_transform_tree(
            engine,
            ctx,
            &tt_params,
            qg,
            x0,
            y0,
            x0,
            y0,
            log2_cb_size,
            0,
            0,
            false,
            false,
        )?;
        cu.transform_tree = Some(tree);
    }
    Ok(())
}

/// Decode one §7.3.8.4 `coding_quadtree( x0, y0, log2CbSize, cqtDepth )`.
#[allow(clippy::too_many_arguments)]
pub fn decode_coding_quadtree(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &SliceDataParams,
    grid: &mut CtuGrid,
    qg: &mut QuantGroupState,
    x0: u32,
    y0: u32,
    log2_cb_size: u32,
    cqt_depth: u32,
) -> Result<CodingQuadtree, ResidualCodingError> {
    let size = 1u32 << log2_cb_size;
    let fits_w = x0 + size <= params.pic_width_in_luma_samples;
    let fits_h = y0 + size <= params.pic_height_in_luma_samples;

    // split_cu_flag presence gate (§7.3.8.4).
    let split_present = fits_w && fits_h && log2_cb_size > params.min_cb_log2_size_y;
    let split = if split_present {
        let (l_depth, l_avail) = grid.left_ct_depth(x0, y0);
        let (a_depth, a_avail) = grid.above_ct_depth(x0, y0);
        let inc = split_cu_flag_ctx_inc(l_depth, l_avail, a_depth, a_avail, cqt_depth) as usize;
        decode_split_cu_flag(engine, &mut ctx.split_cu_flag[inc])? != 0
    } else {
        // §7.4.9.4 inference: 1 when the block extends past the picture
        // boundary OR log2CbSize > MinCbLog2SizeY, else 0.
        (!fits_w || !fits_h) || log2_cb_size > params.min_cb_log2_size_y
    };

    // §6.5.1 quantization-group resets at the QG threshold.
    if params.cu_qp_delta_enabled_flag && log2_cb_size >= params.log2_min_cu_qp_delta_size {
        qg.is_cu_qp_delta_coded = false;
        qg.cu_qp_delta_val = 0;
    }
    if params.cu_chroma_qp_offset_enabled_flag
        && log2_cb_size >= params.log2_min_cu_chroma_qp_offset_size
    {
        qg.is_cu_chroma_qp_offset_coded = false;
    }

    if split {
        let half = 1u32 << (log2_cb_size - 1);
        let x1 = x0 + half;
        let y1 = y0 + half;
        let child_log2 = log2_cb_size - 1;
        let child_depth = cqt_depth + 1;
        let mut children = Vec::with_capacity(4);
        // First child always present.
        children.push(decode_coding_quadtree(
            engine,
            ctx,
            params,
            grid,
            qg,
            x0,
            y0,
            child_log2,
            child_depth,
        )?);
        if x1 < params.pic_width_in_luma_samples {
            children.push(decode_coding_quadtree(
                engine,
                ctx,
                params,
                grid,
                qg,
                x1,
                y0,
                child_log2,
                child_depth,
            )?);
        }
        if y1 < params.pic_height_in_luma_samples {
            children.push(decode_coding_quadtree(
                engine,
                ctx,
                params,
                grid,
                qg,
                x0,
                y1,
                child_log2,
                child_depth,
            )?);
        }
        if x1 < params.pic_width_in_luma_samples && y1 < params.pic_height_in_luma_samples {
            children.push(decode_coding_quadtree(
                engine,
                ctx,
                params,
                grid,
                qg,
                x1,
                y1,
                child_log2,
                child_depth,
            )?);
        }
        Ok(CodingQuadtree::Split(children))
    } else {
        let cu = decode_coding_unit(
            engine,
            ctx,
            params,
            grid,
            qg,
            x0,
            y0,
            log2_cb_size,
            cqt_depth,
        )?;
        Ok(CodingQuadtree::Leaf(Box::new(cu)))
    }
}

/// Decode one §7.3.8.2 `coding_tree_unit( )` rooted at CTB top-left
/// `(x_ctb, y_ctb)`.
///
/// `sao_merge_left_allowed` / `sao_merge_up_allowed` are the §7.3.8.3
/// merge-flag presence conditions (the slice-segment / tile boundary
/// tests), evaluated by the caller.
#[allow(clippy::too_many_arguments)]
pub fn decode_coding_tree_unit(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &SliceDataParams,
    x_ctb: u32,
    y_ctb: u32,
    sao_merge_left_allowed: bool,
    sao_merge_up_allowed: bool,
) -> Result<CodingTreeUnit, ResidualCodingError> {
    let mut grid = CtuGrid::new(params, x_ctb, y_ctb);
    let mut qg = QuantGroupState::default();

    let sao = if params.slice_sao_luma_flag || params.slice_sao_chroma_flag {
        Some(decode_sao(
            engine,
            ctx,
            params,
            sao_merge_left_allowed,
            sao_merge_up_allowed,
        )?)
    } else {
        None
    };

    let quadtree = decode_coding_quadtree(
        engine,
        ctx,
        params,
        &mut grid,
        &mut qg,
        x_ctb,
        y_ctb,
        params.ctb_log2_size_y,
        0,
    )?;

    Ok(CodingTreeUnit { sao, quadtree })
}

/// Decode the §7.3.8.1 `end_of_slice_segment_flag` that follows each
/// CTU. Re-exported here for the slice-data loop convenience; it is the
/// §9.3.4.3.5 terminate path.
pub fn end_of_slice_segment_flag(
    engine: &mut CabacEngine<'_>,
) -> Result<bool, ResidualCodingError> {
    Ok(decode_end_of_slice_segment_flag(engine)? != 0)
}

#[cfg(test)]
mod tests;
