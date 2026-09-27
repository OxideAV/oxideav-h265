//! General coding-tree encoder — recursive §7.3.8.4 coding quadtrees
//! at CTB 16 / 32 / 64 with rate-distortion-elected `split_cu_flag`,
//! recursive §7.3.8.8 residual quadtrees
//! (`max_transform_hierarchy_depth_*` 0..=3, RD-elected at every
//! node), §8.6.4 DST-VII 4x4 intra luma TUs, and `MinCbSizeY == 8`
//! coding units (intra `PART_NxN` with four 4x4 luma PBs, and the
//! 8x4 / 4x8 `PART_2NxN` / `PART_Nx2N` inter PUs — uni-predicted per
//! §8.5.3.2.2 step 10 / Table 9-46).
//!
//! This module is the quadtree twin of the fixed-geometry
//! [`crate::encoder::intra`] / [`crate::encoder::inter`] bootstrap
//! coders (which keep the historical `CtbSizeY == 16`, one-CU-per-CTB
//! streams byte-stable): a [`TreeCfg`] on the stream configuration
//! routes I / P / B slices here instead. Every decision is validated
//! through the crate's own DECODE-side machinery:
//!
//! * §6.4.1 z-scan availability through
//!   [`crate::availability::PictureTiling::z_scan_availability`] (and
//!   §6.4.2 for prediction blocks);
//! * intra prediction through [`crate::intra_pred`], inter prediction
//!   through [`crate::inter_pred`], motion resolution through
//!   [`crate::pu_mv::resolve_pu_motion`];
//! * reconstruction through the decode-side §8.6.2 scaling /
//!   transform ([`crate::transform::residual_block`] — the 4x4 intra
//!   luma TBs taking the eq. 8-316 DST-VII path, mirrored by the
//!   encoder's forward DST);
//! * the §8.7 loop filters through the decode-side apply, with
//!   per-CU [`DeblockCuDesc`] lists and the per-4x4 §8.6.1 `QpY` map.
//!
//! Pass 1 walks each CTB's quadtree bottom-up-comparably: at every
//! node the best unsplit CU (the full skip / merge / AMVP / two-PU /
//! intra ladder on P / B slices; `PART_2Nx2N` with an RD-elected RQT,
//! plus `PART_NxN` at `MinCbSizeY`, on intra) competes against the
//! four coded children under the same SSD + λ·bins cost, with the
//! encoder state (reconstruction, motion field, mode field, `CtDepth`
//! / skip cells) snapshotted and rolled back around each trial. Pass
//! 2 emits the §7.3.8 syntax; the emission mirrors the decoder's
//! parse tree rule for rule (`split_transform_flag` presence /
//! inference, the §7.3.8.8 chroma-cbf inheritance, the `blkIdx == 3`
//! deferred-chroma 4x4 leaves, §7.3.8.14 `delta_qp( )` once per
//! quantization group).

use crate::availability::{PictureTiling, TilingParams, MODE_INTRA};
use crate::binarization::{
    cbf_cb_ctx_inc, cbf_cr_ctx_inc, cbf_luma_ctx_inc, cu_skip_flag_ctx_inc,
    intra_luma_cand_mode_list, split_cu_flag_ctx_inc, split_transform_flag_ctx_inc, CuPredMode,
};
use crate::cabac::init_type;
use crate::ctx_init::SliceContexts;
use crate::deblock::{DeblockCu, DeblockCuDesc, DeblockCuParams, TransformSplit};
use crate::encoder::bitwriter::BitWriter;
use crate::encoder::cabac::CabacEncoder;
use crate::encoder::inter::{
    amvp_search, bin_part_mode, blit, choose_pu, encode_merge_idx, encode_pu_syntax_at,
    entry_point_offsets, extract, merge_pu, part_is_amp, part_is_horizontal, predict_block_wp,
    store, sub_block, write_entry_points, FrameRecon, FrameStats, PuChooseCtx, PuSyntax, RefPlanes,
    SliceLfSignalling, SliceSpec, YuvFrame,
};
use crate::encoder::intra::{
    encode_cu_qp_delta, forward_transform_bd, rate_proxy, IntraEncodeError, IntraEncodedAu,
    IntraEncodedAuWide, SpsCfg,
};
use crate::encoder::loopfilter::{
    encode_sao_ctb_fmt, filter_frame, FilterInput, LoopFilterCfg, TreeLayout,
};
use crate::encoder::pcm::TileGrid;
use crate::encoder::quant::{quantize_tb, scaling_lists_for, RdoqModel, TbQuant};
use crate::encoder::residual::encode_residual_coding;
use crate::encoder::sample::{Sample, SampleFmt};
use crate::inter_recon::SliceWpTables;
use crate::intra_mode_field::{IntraModeField, Neighbour};
use crate::intra_pred::{
    intra_predict_with_substitution, Component as PredComponent, IntraPredParams,
    MarkedReferenceSamples,
};
use crate::motion::{MotionCell, MotionField};
use crate::pu_mv::{pu_partitions, resolve_pu_motion, PartMode, PuGeometry, PuMotion, PuMvContext};
use crate::residual::{residual_coding_scan_idx, ResidualCodingParams};
use crate::scaling_list::{ScalingFactorMatrix, ScalingFactors};
use crate::scan::ScanIdx;
use crate::slice_data::SaoCtbParams;
use crate::transform::{forward_dst4_1d, residual_block, BlockParams, Component, PredMode};

/// `MaxNumMergeCand` (§7.4.7.1) — `five_minus_max_num_merge_cand = 0`.
const MAX_MERGE: usize = 5;
/// The z-order offsets of the four quadrants of a split node.
const Z_OFFSETS: [(usize, usize); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];

/// The quadtree coder's stream geometry: `CtbLog2SizeY`, the residual
/// quadtree depths, always `MinCbLog2SizeY == 3` and
/// `MaxTbLog2SizeY == min( CtbLog2SizeY, 5 )`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct TreeCfg {
    /// `CtbLog2SizeY` (4..=6).
    pub ctb_log2: u32,
    /// `max_transform_hierarchy_depth_intra`.
    pub th_depth_intra: u32,
    /// `max_transform_hierarchy_depth_inter`.
    pub th_depth_inter: u32,
    /// Rate-distortion optimised quantization
    /// ([`crate::encoder::quant`]): every TB's levels are elected
    /// under `D + λ·R` over the exact §7.3.8.11 bin costs at the
    /// running CABAC context states.
    pub rdoq: bool,
    /// PPS `sign_data_hiding_enabled_flag == 1`: the §7.3.8.11
    /// `signHidden` sub-blocks omit their first sign, the levels
    /// parity-adjusted by the cheapest ±1 move.
    pub sign_hiding: bool,
    /// Scaling lists ([`crate::encoder::quant::scaling_lists_for`]):
    /// 0 off, 1 the §7.4.5 defaults, 2 / 3 the flattened / steepened
    /// custom families (transmitted in the SPS).
    pub scaling_lists: u8,
    /// PPS `weighted_pred_flag` / `weighted_bipred_flag == 1`: every
    /// P / B slice carries a §7.3.6.3 `pred_weight_table( )` estimated
    /// by fade detection ([`crate::encoder::wp`]).
    pub weighted_pred: bool,
    /// PPS `entropy_coding_sync_enabled_flag == 1` (wavefront
    /// parallel processing): one substream per CTB row of a tile with
    /// the §9.3.2.2 context storage after the row's second CTB and
    /// synchronization at the next row start, `end_of_subset_one_bit`
    /// + byte alignment between rows and §7.3.6.1 entry points.
    pub wpp: bool,
    /// The tile grid (`tiles_enabled_flag == 1` when more than one
    /// tile): CTBs coded in the §6.5.1 tile scan, availability cut at
    /// tile boundaries, fresh contexts + engine per tile, one subset
    /// per tile with entry points; the in-loop filters stay
    /// picture-wide (`loop_filter_across_tiles_enabled_flag == 1`).
    pub tiles: TileLayout,
    /// Intra mode-decision effort: 0 = the historical SAD search with
    /// derived (DM) chroma; 1 = SATD + λ·signalling-bins rough
    /// decision over all 35 modes with a rough chroma-mode election
    /// (`intra_chroma_pred_mode` 0..=4), one full coding pass; 2 = the
    /// same rough decision keeping a short list (3 modes at 4x4 / 8x8,
    /// 2 above, plus the most-probable mode) each coded for real
    /// through the RD-elected RQT, the cheapest kept.
    pub intra_rd: u8,
}

/// A tile grid for [`TreeCfg`]: uniform spacing, or explicit
/// `column_width_minus1[ ]` / `row_height_minus1[ ]` spans (up to 8
/// columns / rows).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileLayout {
    /// Tile columns (1..=8).
    pub cols: u8,
    /// Tile rows (1..=8).
    pub rows: u8,
    /// `uniform_spacing_flag`.
    pub uniform: bool,
    /// `column_width_minus1[ i ]` for `i < cols − 1` (explicit only).
    pub col_w_minus1: [u8; 7],
    /// `row_height_minus1[ j ]` for `j < rows − 1` (explicit only).
    pub row_h_minus1: [u8; 7],
}

impl TileLayout {
    /// The single-tile layout (`tiles_enabled_flag == 0`).
    #[must_use]
    pub const fn single() -> Self {
        Self {
            cols: 1,
            rows: 1,
            uniform: true,
            col_w_minus1: [0; 7],
            row_h_minus1: [0; 7],
        }
    }

    /// A `uniform_spacing_flag == 1` grid (each dimension clamped to
    /// 1..=8).
    #[must_use]
    pub fn uniform(cols: u8, rows: u8) -> Self {
        Self {
            cols: cols.clamp(1, 8),
            rows: rows.clamp(1, 8),
            ..Self::single()
        }
    }

    /// An explicit grid from the column widths / row heights in CTBs
    /// of every column / row but the last (`uniform_spacing_flag ==
    /// 0`); at most 7 entries each are kept.
    #[must_use]
    pub fn explicit(col_widths: &[u8], row_heights: &[u8]) -> Self {
        let mut l = Self::single();
        l.uniform = false;
        l.cols = (col_widths.len().min(7) + 1) as u8;
        l.rows = (row_heights.len().min(7) + 1) as u8;
        for (i, &w) in col_widths.iter().take(7).enumerate() {
            l.col_w_minus1[i] = w.max(1) - 1;
        }
        for (j, &h) in row_heights.iter().take(7).enumerate() {
            l.row_h_minus1[j] = h.max(1) - 1;
        }
        l
    }

    /// More than one tile?
    #[must_use]
    pub fn on(&self) -> bool {
        self.cols > 1 || self.rows > 1
    }

    /// The PPS grid (`None` for a single tile).
    pub(crate) fn grid(&self) -> Option<TileGrid> {
        self.on().then(|| TileGrid {
            cols: u32::from(self.cols),
            rows: u32::from(self.rows),
            uniform: self.uniform,
            column_width_minus1: self.col_w_minus1[..usize::from(self.cols) - 1]
                .iter()
                .map(|&v| u32::from(v))
                .collect(),
            row_height_minus1: self.row_h_minus1[..usize::from(self.rows) - 1]
                .iter()
                .map(|&v| u32::from(v))
                .collect(),
        })
    }

    /// The §6.5.1 tiling parameters.
    pub(crate) fn params(&self) -> TilingParams {
        self.grid()
            .map_or_else(TilingParams::single_tile, |g| g.tiling_params())
    }
}

impl TreeCfg {
    /// A quadtree configuration for CTB size 16 / 32 / 64 with one
    /// level of residual-quadtree freedom on both prediction types.
    #[must_use]
    pub fn new(ctb: usize) -> Option<Self> {
        let ctb_log2 = match ctb {
            16 => 4,
            32 => 5,
            64 => 6,
            _ => return None,
        };
        Some(Self {
            ctb_log2,
            th_depth_intra: 1,
            th_depth_inter: 1,
            rdoq: false,
            sign_hiding: false,
            scaling_lists: 0,
            weighted_pred: false,
            wpp: false,
            tiles: TileLayout::single(),
            intra_rd: 0,
        })
    }

    /// Switch wavefront parallel processing signalling on / off.
    #[must_use]
    pub fn with_wpp(mut self, on: bool) -> Self {
        self.wpp = on;
        self
    }

    /// Intra mode-decision effort (see [`Self::intra_rd`]; values
    /// above 2 clamp to 2).
    #[must_use]
    pub fn with_intra_rd(mut self, level: u8) -> Self {
        self.intra_rd = level.min(2);
        self
    }

    /// Select the tile grid.
    #[must_use]
    pub fn with_tiles(mut self, tiles: TileLayout) -> Self {
        self.tiles = tiles;
        self
    }

    /// Switch explicit weighted prediction (fade estimation) on / off.
    #[must_use]
    pub fn with_weighted_pred(mut self, on: bool) -> Self {
        self.weighted_pred = on;
        self
    }

    /// Select the scaling lists (0 off, 1 default, 2 flattened, 3
    /// steepened).
    #[must_use]
    pub fn with_scaling_lists(mut self, mode: u8) -> Self {
        self.scaling_lists = mode.min(3);
        self
    }

    /// Set `max_transform_hierarchy_depth_intra` / `_inter` (0..=3;
    /// the residual quadtrees may then split that many levels below
    /// the coding block, down to 4x4 luma TBs).
    #[must_use]
    pub fn with_tu_depth(mut self, intra: u32, inter: u32) -> Self {
        self.th_depth_intra = intra.min(3);
        self.th_depth_inter = inter.min(3);
        self
    }

    /// Switch rate-distortion optimised quantization on / off.
    #[must_use]
    pub fn with_rdoq(mut self, on: bool) -> Self {
        self.rdoq = on;
        self
    }

    /// Switch sign data hiding on / off.
    #[must_use]
    pub fn with_sign_hiding(mut self, on: bool) -> Self {
        self.sign_hiding = on;
        self
    }

    /// `MinCbLog2SizeY` (always 3: 8x8 minimum coding blocks).
    #[must_use]
    pub fn min_cb_log2(&self) -> u32 {
        3
    }

    /// `MaxTbLog2SizeY` (32x32 transform ceiling, CTB-clamped).
    #[must_use]
    pub fn max_tb_log2(&self) -> u32 {
        self.ctb_log2.min(5)
    }
}

// ---------------------------------------------------------------------
// Coded-tree data model
// ---------------------------------------------------------------------

/// One residual-quadtree node's coded levels. Leaves whose chroma is
/// coded in place (§7.3.8.10 `log2TrafoSize > 2 || ChromaArrayType ==
/// 3`) carry their chroma blocks; 4x4 luma leaves of a 4:2:0 / 4:2:2
/// tree defer chroma to their parent split node (`blkIdx == 3`),
/// which carries the CU-quadrant 4x4 chroma blocks itself. A chroma
/// level vector holds the node's `ChromaArrayType == 2 ? 2 : 1`
/// square blocks back to back (upper half first), each
/// `(1 << log2TrafoSizeC)²` long; monochrome trees keep them empty.
enum TuNode {
    /// `split_transform_flag == 0` leaf: the luma levels, plus the
    /// chroma levels when coded in place.
    Leaf {
        y: Vec<i32>,
        cb: Vec<i32>,
        cr: Vec<i32>,
    },
    /// `split_transform_flag == 1` node: four z-order children, plus
    /// the deferred 4x4 chroma blocks when the children are 4x4 luma
    /// leaves (`log2TrafoSize == 3` here, 4:2:0 / 4:2:2 only).
    Split {
        children: Box<[TuNode; 4]>,
        cb: Vec<i32>,
        cr: Vec<i32>,
    },
}

impl TuNode {
    fn any_nonzero(v: &[i32]) -> bool {
        v.iter().any(|&x| x != 0)
    }

    /// The per-block cbf flags of a chroma level vector holding
    /// `blocks` stacked square blocks (`[upper, lower]`; the second
    /// entry is `false` unless `ChromaArrayType == 2`).
    fn cbf_halves(v: &[i32], blocks: usize) -> [bool; 2] {
        if blocks == 2 && !v.is_empty() {
            let half = v.len() / 2;
            [Self::any_nonzero(&v[..half]), Self::any_nonzero(&v[half..])]
        } else {
            [Self::any_nonzero(v), false]
        }
    }

    /// This node's own chroma level vectors `(cb, cr)` (empty at a
    /// split node above the deferred-chroma level).
    fn own_chroma(&self) -> (&[i32], &[i32]) {
        match self {
            TuNode::Leaf { cb, cr, .. } | TuNode::Split { cb, cr, .. } => (cb, cr),
        }
    }

    /// Whether any luma level in the subtree is nonzero.
    fn cbf_luma_any(&self) -> bool {
        match self {
            TuNode::Leaf { y, .. } => Self::any_nonzero(y),
            TuNode::Split { children, .. } => children.iter().any(TuNode::cbf_luma_any),
        }
    }

    /// `cbf_cb` of this node (OR over the subtree — §7.3.8.8
    /// inheritance means a node's flag covers its descendants).
    fn cbf_cb(&self) -> bool {
        match self {
            TuNode::Leaf { cb, .. } => Self::any_nonzero(cb),
            TuNode::Split { children, cb, .. } => {
                Self::any_nonzero(cb) || children.iter().any(TuNode::cbf_cb)
            }
        }
    }

    fn cbf_cr(&self) -> bool {
        match self {
            TuNode::Leaf { cr, .. } => Self::any_nonzero(cr),
            TuNode::Split { children, cr, .. } => {
                Self::any_nonzero(cr) || children.iter().any(TuNode::cbf_cr)
            }
        }
    }

    fn any_cbf(&self) -> bool {
        self.cbf_luma_any() || self.cbf_cb() || self.cbf_cr()
    }

    /// The deblocking [`TransformSplit`] twin of this node.
    fn to_transform_split(&self) -> TransformSplit {
        match self {
            TuNode::Leaf { .. } => TransformSplit::Leaf,
            TuNode::Split { children, .. } => TransformSplit::Split(Box::new([
                children[0].to_transform_split(),
                children[1].to_transform_split(),
                children[2].to_transform_split(),
                children[3].to_transform_split(),
            ])),
        }
    }
}

/// How a coded CU is signalled.
enum TreeCuKind {
    /// `cu_skip_flag == 1`.
    Skip { merge_idx: usize },
    /// `merge_flag == 1` 2Nx2N with residual (`rqt_root_cbf`
    /// inferred 1).
    Merge { merge_idx: usize },
    /// 2Nx2N AMVP (`rqt_root_cbf` signalled).
    Amvp { pu: PuSyntax },
    /// Two-PU inter partition.
    TwoPu { part: PartMode, pus: [PuSyntax; 2] },
    /// Intra: `PART_2Nx2N` (one PB) or, at `MinCbSizeY`, `PART_NxN`
    /// (four PBs). `modes[0]` is replicated for 2Nx2N; `chroma` holds
    /// the coded `intra_chroma_pred_mode` per PB (4 = derived from
    /// luma) — one value replicated unless `ChromaArrayType == 3`
    /// `PART_NxN`, where §7.3.8.5 signals four.
    Intra {
        modes: [u8; 4],
        nxn: bool,
        chroma: [u8; 4],
    },
}

/// One coded coding unit (a quadtree leaf).
struct CuCoded {
    x0: usize,
    y0: usize,
    log2: u32,
    kind: TreeCuKind,
    /// Resolved per-PU motion in §7.3.8.6 order (empty for intra).
    motions: Vec<PuMotion>,
    /// The coded transform tree (`None` ⇔ skip or `rqt_root_cbf == 0`).
    tree: Option<TuNode>,
    /// SSD + λ·bins cost of the CU (its own syntax included).
    cost: u64,
}

/// One coding-quadtree node of a coded CTB.
enum CuNode {
    Leaf(Box<CuCoded>),
    /// Children in z-order; children outside the picture are `None`
    /// (the decoder never visits them).
    Split(Box<[Option<CuNode>; 4]>),
}

impl CuNode {
    fn for_each_cu<'a>(&'a self, f: &mut dyn FnMut(&'a CuCoded)) {
        match self {
            CuNode::Leaf(cu) => f(cu),
            CuNode::Split(ch) => {
                for c in ch.iter().flatten() {
                    c.for_each_cu(f);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Shared per-slice context + mutable encoder state
// ---------------------------------------------------------------------

/// Everything a slice's CU decisions read (immutable).
struct SliceCtx<'a> {
    cfg: TreeCfg,
    /// The picture's chroma format and bit depths (the sample path is
    /// `u16` at every depth; 8-bit pictures are widened on entry).
    fmt: SampleFmt,
    amp: bool,
    width: usize,
    height: usize,
    /// Source planes `[Y, Cb, Cr]` (the chroma planes empty for
    /// monochrome).
    src: [&'a [u16]; 3],
    qp: i32,
    aq_deltas: &'a [i32],
    /// CTU-level rate feedback: the frame's bit budget. When set, each
    /// CTB's QP moves off `qp + aq` by up to ±3 against the running
    /// coded size (a shadow CABAC emission of every coded CTB) versus
    /// the pro-rata budget.
    ctu_rc: Option<u64>,
    b_slice: bool,
    intra_slice: bool,
    refs_l0: &'a [RefPlanes],
    refs_l1: &'a [RefPlanes],
    mv_ctx: Option<&'a PuMvContext<'a>>,
    two_sided: bool,
    tiling: &'a PictureTiling,
    /// The §7.4.5 `ScalingFactor` matrices when the stream enables
    /// scaling lists.
    scaling: Option<&'a ScalingFactors>,
    /// The slice's explicit weighted-prediction tables.
    wp: Option<&'a SliceWpTables>,
    /// Motion-search reference planes (luma-weighted copies under
    /// weighted prediction; else the reference planes themselves).
    me_refs_l0: &'a [RefPlanes],
    me_refs_l1: &'a [RefPlanes],
    /// Pass-1 worker budget (tiles decided in parallel; 1 = serial).
    threads: usize,
    /// `pps_cb_qp_offset == pps_cr_qp_offset`.
    chroma_qp_offset: i32,
}

impl SliceCtx<'_> {
    /// The mode-decision λ (SSD per bin): the power-of-two ladder
    /// `2^((QP−9)/3)` of the historical coders, halved under the
    /// `intra_rd >= 1` decision — a {25, 35, 50, 70, 85, 100, 125} %
    /// sweep on two photographs put the optimum at one half (−1.5 % /
    /// −1.0 % BD-rate on the 1024x768 / 4032x3024 stills; the SATD
    /// rough decision and the exact-bin RDOQ leave less for the proxy
    /// bins to guard against).
    ///
    /// `q` is the `QpY`; the ladder runs on `Qp′Y = QpY + QpBdOffsetY`
    /// so that at a higher bit depth — where the same `QpY` quantizes
    /// `2^(BitDepth − 8)` times coarser in sample units and the SSD
    /// scales by its square — λ scales by exactly the same
    /// `2^(QpBdOffsetY / 3) = 4^(BitDepth − 8)` (identity at 8 bits).
    fn lambda_of(&self, q: i32) -> u64 {
        let q = q + self.fmt.qp_bd_offset_y();
        let base = 1u64 << (q.unsigned_abs().saturating_sub(9) / 3);
        if self.cfg.intra_rd == 0 {
            base
        } else {
            base.div_ceil(2)
        }
    }

    /// Chroma plane width (0 for monochrome).
    fn cw(&self) -> usize {
        self.fmt.chroma_dims(self.width, self.height).0
    }

    /// Chroma plane height (0 for monochrome).
    fn ch(&self) -> usize {
        self.fmt.chroma_dims(self.width, self.height).1
    }

    /// The luma `qP` (`Qp′Y`) of the §8.6.2 scaling process at `QpY`.
    fn qp_y_prime(&self, qp_y: i32) -> u32 {
        self.fmt.luma_qp_prime(qp_y)
    }

    /// The chroma `qP` (`Qp′Cb == Qp′Cr`: one PPS offset for both) at
    /// `QpY`.
    fn qp_c_prime(&self, qp_y: i32) -> u32 {
        self.fmt.chroma_qp_prime(qp_y, self.chroma_qp_offset)
    }

    fn ctbs_x(&self) -> usize {
        self.width.div_ceil(1 << self.cfg.ctb_log2)
    }

    fn ctbs_y(&self) -> usize {
        self.height.div_ceil(1 << self.cfg.ctb_log2)
    }

    /// Raster address of the CTB at tile-scan index `ts`.
    fn ctb_rs(&self, ts: usize) -> usize {
        self.tiling.ctb_addr_ts_to_rs(ts as u32) as usize
    }

    /// The CTB at tile-scan index `ts` starts a tile (fresh contexts,
    /// engine and `qPY_PREV`).
    fn tile_start(&self, ts: usize) -> bool {
        ts == 0 || self.tiling.tile_id(ts as u32) != self.tiling.tile_id(ts as u32 - 1)
    }

    /// The CTB at tile-scan index `ts` starts a CTB row of its tile
    /// (§9.3.2.2 WPP synchronization point, §8.6.1 `qPY_PREV` reset)
    /// — only meaningful when `cfg.wpp`.
    fn row_start_in_tile(&self, ts: usize) -> bool {
        let rs = self.ctb_rs(ts) as u32;
        let w = self.ctbs_x() as u32;
        rs % w == 0
            || self.tiling.tile_id(self.tiling.ctb_addr_rs_to_ts(rs - 1))
                != self.tiling.tile_id(ts as u32)
    }

    /// The CTB at tile-scan index `ts` is the one after which the
    /// §9.3.2.2 WPP storage fires (the second CTB of a row of a tile).
    fn wpp_store_after(&self, ts: usize) -> bool {
        let rs = self.ctb_rs(ts) as u32;
        let w = self.ctbs_x() as u32;
        rs % w == 1
            || (rs > 1
                && self.tiling.tile_id(ts as u32)
                    != self.tiling.tile_id(self.tiling.ctb_addr_rs_to_ts(rs - 2)))
    }

    /// Whether the §8.6.1 `qPY_PREV` resets to `SliceQpY` at the CTB
    /// at tile-scan index `ts` (first QG in the slice / tile / WPP row).
    fn qp_prev_resets(&self, ts: usize) -> bool {
        self.tile_start(ts) || (self.cfg.wpp && self.row_start_in_tile(ts))
    }

    /// A subset boundary follows the CTB at tile-scan index `ts`
    /// (`end_of_subset_one_bit` + byte alignment): the next CTB starts
    /// a tile, or (WPP) a CTB row of a tile.
    fn subset_ends_after(&self, ts: usize) -> bool {
        let next = ts + 1;
        if next >= self.ctbs_x() * self.ctbs_y() {
            return false;
        }
        (self.cfg.tiles.on() && self.tile_start(next))
            || (self.cfg.wpp && self.row_start_in_tile(next))
    }

    /// §9.3.2.2 availability of the spatial neighbour T (eq. 9-3) of
    /// the CTB at tile-scan index `ts`.
    fn wpp_sync_available(&self, ts: usize) -> bool {
        let rs = self.ctb_rs(ts);
        let ctb = 1usize << self.cfg.ctb_log2;
        let (x0, y0) = ((rs % self.ctbs_x()) * ctb, (rs / self.ctbs_x()) * ctb);
        let (xt, yt) = ((x0 + ctb) as i64, y0 as i64 - ctb as i64);
        self.z_avail(x0, y0, xt, yt)
    }

    /// §6.4.1 z-scan availability of luma location `(nx, ny)` for the
    /// block whose top-left is `(x_cur, y_cur)` (single slice + tile).
    fn z_avail(&self, x_cur: usize, y_cur: usize, nx: i64, ny: i64) -> bool {
        if nx < 0 || ny < 0 || nx >= self.width as i64 || ny >= self.height as i64 {
            return false;
        }
        self.tiling
            .z_scan_availability(x_cur as u32, y_cur as u32, nx as i32, ny as i32, |_| 0)
    }
}

/// The reconstruction planes the quadtree coder writes: `u16` at any
/// bit depth (an 8-bit picture narrows back on output).
#[derive(Clone)]
pub(crate) struct ReconPlanes {
    pub y: Vec<u16>,
    /// Empty for monochrome.
    pub cb: Vec<u16>,
    pub cr: Vec<u16>,
}

impl ReconPlanes {
    fn new(fmt: &SampleFmt, width: usize, height: usize) -> Self {
        let (cw, ch) = fmt.chroma_dims(width, height);
        Self {
            y: vec![0; width * height],
            cb: vec![0; cw * ch],
            cr: vec![0; cw * ch],
        }
    }

    /// Narrow to the 8-bit [`FrameRecon`] of the inter coders (every
    /// value is already `<= 255` on an 8-bit picture).
    fn into_frame_recon(self) -> FrameRecon {
        let narrow = |v: Vec<u16>| -> Vec<u8> { v.into_iter().map(|s| s as u8).collect() };
        FrameRecon {
            y: narrow(self.y),
            cb: narrow(self.cb),
            cr: narrow(self.cr),
            motion_field: None,
        }
    }
}

/// Widen an 8-bit plane to the coder's `u16` sample path.
pub(crate) fn widen(plane: &[u8]) -> Vec<u16> {
    plane.iter().map(|&s| u16::from(s)).collect()
}

/// The picture state the decisions mutate (and snapshots roll back).
struct EncState {
    recon: ReconPlanes,
    field: MotionField,
    modes: IntraModeField,
    /// Per-4x4-cell `CtDepth` (−1 until coded).
    ct_depth: Vec<i8>,
    /// Per-4x4-cell `cu_skip_flag`.
    skip: Vec<u8>,
    w_cells: usize,
    h_cells: usize,
    /// The RDOQ rate model for the CTB being decided (the shadow
    /// emission's residual context states at the CTB start); `None`
    /// when RDOQ is off.
    rdoq_model: Option<RdoqModel>,
}

impl EncState {
    fn new(fmt: &SampleFmt, width: usize, height: usize, ctb_log2: u32) -> Self {
        let w_cells = width.div_ceil(4);
        let h_cells = height.div_ceil(4);
        Self {
            recon: ReconPlanes::new(fmt, width, height),
            field: MotionField::new(width, height),
            modes: IntraModeField::new(width, height, ctb_log2),
            ct_depth: vec![-1; w_cells * h_cells],
            skip: vec![0; w_cells * h_cells],
            w_cells,
            h_cells,
            rdoq_model: None,
        }
    }

    fn cell(&self, x: usize, y: usize) -> usize {
        (y / 4) * self.w_cells + x / 4
    }

    /// `(value, available)` cell reads for the §9.3.4.2.2 ctxIncs.
    fn nb_ct_depth(&self, x0: usize, y0: usize, nb: Neighbour) -> (u32, bool) {
        let (x, y) = match nb {
            Neighbour::Left => (x0 as i64 - 1, y0 as i64),
            Neighbour::Above => (x0 as i64, y0 as i64 - 1),
        };
        if x < 0 || y < 0 || x >= (self.w_cells * 4) as i64 || y >= (self.h_cells * 4) as i64 {
            return (0, false);
        }
        let d = self.ct_depth[self.cell(x as usize, y as usize)];
        if d < 0 {
            (0, false)
        } else {
            (d as u32, true)
        }
    }

    fn nb_skip(&self, x0: usize, y0: usize, nb: Neighbour) -> (u8, bool) {
        let (x, y) = match nb {
            Neighbour::Left => (x0 as i64 - 1, y0 as i64),
            Neighbour::Above => (x0 as i64, y0 as i64 - 1),
        };
        if x < 0 || y < 0 || x >= (self.w_cells * 4) as i64 || y >= (self.h_cells * 4) as i64 {
            return (0, false);
        }
        let c = self.cell(x as usize, y as usize);
        if self.ct_depth[c] < 0 {
            (0, false)
        } else {
            (self.skip[c], true)
        }
    }

    fn fill_cells(&mut self, x0: usize, y0: usize, n: usize, depth: i8, skip: u8) {
        let bx1 = ((x0 + n).min(self.w_cells * 4)).div_ceil(4);
        let by1 = ((y0 + n).min(self.h_cells * 4)).div_ceil(4);
        for by in y0 / 4..by1 {
            for bx in x0 / 4..bx1 {
                self.ct_depth[by * self.w_cells + bx] = depth;
                self.skip[by * self.w_cells + bx] = skip;
            }
        }
    }
}

/// A rectangular rollback snapshot of the encoder state.
struct Snap {
    x0: usize,
    y0: usize,
    n: usize,
    y: Vec<u16>,
    cb: Vec<u16>,
    cr: Vec<u16>,
    field: Vec<MotionCell>,
    modes: Vec<u8>,
    depth: Vec<i8>,
    skip: Vec<u8>,
}

fn rect_copy<T: Copy>(plane: &[T], pw: usize, x0: usize, y0: usize, w: usize, h: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(w * h);
    for j in 0..h {
        out.extend_from_slice(&plane[(y0 + j) * pw + x0..(y0 + j) * pw + x0 + w]);
    }
    out
}

fn rect_paste<T: Copy>(
    plane: &mut [T],
    pw: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    s: &[T],
) {
    for j in 0..h {
        plane[(y0 + j) * pw + x0..(y0 + j) * pw + x0 + w].copy_from_slice(&s[j * w..(j + 1) * w]);
    }
}

impl EncState {
    fn snapshot(&self, ctx: &SliceCtx<'_>, x0: usize, y0: usize, n: usize) -> Snap {
        let w = (ctx.width - x0).min(n);
        let h = (ctx.height - y0).min(n);
        let (sw, sh) = ctx.fmt.sub_wh();
        let (cw, cx0, cy0) = (ctx.cw(), x0 / sw, y0 / sh);
        let (wc, hc) = if ctx.fmt.has_chroma() {
            (w / sw, h / sh)
        } else {
            (0, 0)
        };
        let bx0 = x0 / 4;
        let by0 = y0 / 4;
        let bx1 = (x0 + w).div_ceil(4).min(self.w_cells);
        let by1 = (y0 + h).div_ceil(4).min(self.h_cells);
        let mut depth = Vec::with_capacity((bx1 - bx0) * (by1 - by0));
        let mut skip = Vec::with_capacity((bx1 - bx0) * (by1 - by0));
        for by in by0..by1 {
            depth.extend_from_slice(
                &self.ct_depth[by * self.w_cells + bx0..by * self.w_cells + bx1],
            );
            skip.extend_from_slice(&self.skip[by * self.w_cells + bx0..by * self.w_cells + bx1]);
        }
        Snap {
            x0,
            y0,
            n,
            y: rect_copy(&self.recon.y, ctx.width, x0, y0, w, h),
            cb: rect_copy(&self.recon.cb, cw, cx0, cy0, wc, hc),
            cr: rect_copy(&self.recon.cr, cw, cx0, cy0, wc, hc),
            field: self.field.snapshot_rect(x0, y0, n, n),
            modes: self.modes.snapshot_rect(x0, y0, n, n),
            depth,
            skip,
        }
    }

    fn restore(&mut self, ctx: &SliceCtx<'_>, snap: &Snap) {
        let (x0, y0, n) = (snap.x0, snap.y0, snap.n);
        let w = (ctx.width - x0).min(n);
        let h = (ctx.height - y0).min(n);
        let (sw, sh) = ctx.fmt.sub_wh();
        let (cw, cx0, cy0) = (ctx.cw(), x0 / sw, y0 / sh);
        let (wc, hc) = if ctx.fmt.has_chroma() {
            (w / sw, h / sh)
        } else {
            (0, 0)
        };
        rect_paste(&mut self.recon.y, ctx.width, x0, y0, w, h, &snap.y);
        rect_paste(&mut self.recon.cb, cw, cx0, cy0, wc, hc, &snap.cb);
        rect_paste(&mut self.recon.cr, cw, cx0, cy0, wc, hc, &snap.cr);
        self.field.restore_rect(x0, y0, n, n, &snap.field);
        self.modes.restore_rect(x0, y0, n, n, &snap.modes);
        let bx0 = x0 / 4;
        let by0 = y0 / 4;
        let bx1 = (x0 + w).div_ceil(4).min(self.w_cells);
        let by1 = (y0 + h).div_ceil(4).min(self.h_cells);
        let row = bx1 - bx0;
        for (i, by) in (by0..by1).enumerate() {
            self.ct_depth[by * self.w_cells + bx0..by * self.w_cells + bx1]
                .copy_from_slice(&snap.depth[i * row..(i + 1) * row]);
            self.skip[by * self.w_cells + bx0..by * self.w_cells + bx1]
                .copy_from_slice(&snap.skip[i * row..(i + 1) * row]);
        }
    }
}

// ---------------------------------------------------------------------
// Transform-block coding (forward transform with the DST-VII case)
// ---------------------------------------------------------------------

/// Forward DST-VII for the intra-luma 4x4 case (the transpose of the
/// eq. 8-316 synthesis, at the encoder's DCT normalization shifts for
/// `bit_depth`).
fn forward_transform_dst4(res: &[i32], bit_depth: u8) -> Vec<i32> {
    let shift1 = 2 + u32::from(bit_depth) - 9; // log2TbS + BitDepth − 9
    let shift2 = 2 + 6;
    let r1 = 1i64 << (shift1 - 1);
    let r2 = 1i64 << (shift2 - 1);
    let mut a = [0i64; 16];
    for y in 0..4 {
        let row: Vec<i64> = (0..4).map(|x| i64::from(res[y * 4 + x])).collect();
        let t = forward_dst4_1d(&row);
        for (u, &v) in t.iter().enumerate() {
            a[y * 4 + u] = (v + r1) >> shift1;
        }
    }
    let mut coef = vec![0i32; 16];
    for u in 0..4 {
        let col: Vec<i64> = (0..4).map(|y| a[y * 4 + u]).collect();
        let t = forward_dst4_1d(&col);
        for (v, &val) in t.iter().enumerate() {
            coef[v * 4 + u] = ((val + r2) >> shift2) as i32;
        }
    }
    coef
}

/// The quantization tools one TB is coded with.
#[derive(Clone, Copy)]
struct TbTools<'a> {
    /// The §7.4.9.11 scan of the block (mode-dependent on intra).
    scan: ScanIdx,
    /// The component's bit depth (transform normalization, the
    /// quantizer's `qBits`, the reconstruction clip).
    bit_depth: u8,
    /// Sample-domain λ.
    lambda: u64,
    /// RDOQ model (`None` = deadzone quantizer).
    model: Option<&'a RdoqModel>,
    /// Sign-data-hiding parity adjustment.
    sign_hiding: bool,
    /// The block's `ScalingFactor[ sizeId ][ matrixId ]` (Table 7-3 /
    /// 7-4) when scaling lists are on.
    scaling: Option<&'a ScalingFactorMatrix>,
}

impl<'a> TbTools<'a> {
    /// The tools for a TB of the CU under decision: the scan from
    /// the prediction type / intra mode, the model from the state.
    fn new(
        ctx: &SliceCtx<'a>,
        model: Option<&'a RdoqModel>,
        lambda: u64,
        cu_is_intra: bool,
        log2: u32,
        c_idx: u8,
        mode: u8,
    ) -> Self {
        // Table 7-3 sizeId = log2 − 2; Table 7-4 matrixId = 3·inter +
        // cIdx.
        let scaling = ctx.scaling.map(|sf| {
            &sf.factors[log2 as usize - 2][3 * usize::from(!cu_is_intra) + usize::from(c_idx)]
        });
        // The mode-decision λ prices PROXY bins (`rate_proxy`
        // overstates residual bits); the quantizer prices exact bins,
        // and λ/2 is the measured optimum (a {1/8 .. 3/4}·λ sweep on
        // the rd_measure corpus: −9.4 % BD-rate on the pyramid path).
        Self {
            scan: residual_coding_scan_idx(
                cu_is_intra,
                log2,
                c_idx,
                ctx.fmt.chroma_format_idc,
                u32::from(mode),
            ),
            bit_depth: ctx.fmt.bit_depth(c_idx),
            lambda: lambda / 2,
            model: if ctx.cfg.rdoq { model } else { None },
            sign_hiding: ctx.cfg.sign_hiding,
            scaling,
        }
    }
}

/// Transform + quantize one TB and reconstruct through the decode-side
/// §8.6.2 path (the intra-luma 4x4 case taking the DST-VII pair).
/// Returns `(levels, recon_samples)`.
fn code_tb(
    src: &[i32],
    pred: &[i32],
    n: usize,
    qp: u32,
    component: Component,
    pred_mode: PredMode,
    tools: TbTools<'_>,
) -> (Vec<i32>, Vec<u16>) {
    let res: Vec<i32> = src.iter().zip(pred.iter()).map(|(&s, &p)| s - p).collect();
    let coef = if pred_mode == PredMode::Intra && component == Component::Luma && n == 4 {
        forward_transform_dst4(&res, tools.bit_depth)
    } else {
        forward_transform_bd(&res, n, tools.bit_depth)
    };
    let levels = quantize_tb(
        &coef,
        &TbQuant {
            log2: n.trailing_zeros(),
            qp,
            is_chroma: component != Component::Luma,
            bit_depth: tools.bit_depth,
            scan: tools.scan,
            lambda: tools.lambda,
            model: tools.model,
            sign_hiding: tools.sign_hiding,
            scaling: tools.scaling,
        },
    );
    let max = (1i32 << tools.bit_depth) - 1;
    let recon: Vec<u16> = if levels.iter().all(|&v| v == 0) {
        pred.iter().map(|&p| u16::clipped(p, max)).collect()
    } else {
        let r = residual_block(
            &levels,
            tools.scaling,
            BlockParams {
                n_tbs: n,
                q_p: qp,
                component,
                pred_mode,
                bit_depth: tools.bit_depth,
                extended_precision: false,
                transquant_bypass: false,
                transform_skip: false,
                transform_skip_rotation_enabled: false,
            },
        )
        .expect("legal block params");
        pred.iter()
            .zip(r.iter())
            .map(|(&p, &d)| u16::clipped(p + d, max))
            .collect()
    };
    (levels, recon)
}

/// Sum of squared differences of a stored block against an `i32` one.
fn ssd<S: Sample>(a: &[S], b: &[i32]) -> u64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = i64::from(x.to_i32()) - i64::from(y);
            (d * d) as u64
        })
        .sum()
}

/// Clip a prediction into stored samples at the component ceiling.
fn clip_to_samples(v: &[i32], max: i32) -> Vec<u16> {
    v.iter().map(|&p| u16::clipped(p, max)).collect()
}

// ---------------------------------------------------------------------
// Intra CU coding
// ---------------------------------------------------------------------

/// The §8.4.4.2 prediction parameters of one component at the
/// picture's format (`strong_intra_smoothing_enabled_flag == 0`, the
/// §8.4.4.2.3 filtering gate open for chroma only at 4:4:4).
fn pred_params(fmt: &SampleFmt, mode: u8, cidx: PredComponent) -> IntraPredParams {
    IntraPredParams {
        pred_mode_intra: mode,
        cidx,
        bit_depth: if cidx == PredComponent::Luma {
            fmt.bit_depth_luma
        } else {
            fmt.bit_depth_chroma
        },
        bit_depth_luma: fmt.bit_depth_luma,
        intra_smoothing_disabled: false,
        strong_intra_smoothing_enabled: false,
        chroma_array_type_3: fmt.chroma_format_idc == 3,
        disable_boundary_filter: false,
    }
}

/// Gather the §8.4.4.2.1 marked luma reference array for an `n`-TB at
/// `(x0, y0)` from the frame reconstruction, availability per §6.4.1.
fn gather_luma_refs(
    ctx: &SliceCtx<'_>,
    recon_y: &[u16],
    x0: usize,
    y0: usize,
    n: usize,
) -> MarkedReferenceSamples {
    let get = |x: i64, y: i64| -> (i32, bool) {
        if ctx.z_avail(x0, y0, x, y) {
            (
                i32::from(recon_y[y as usize * ctx.width + x as usize]),
                true,
            )
        } else {
            (0, false)
        }
    };
    let corner = get(x0 as i64 - 1, y0 as i64 - 1);
    let left: Vec<(i32, bool)> = (0..2 * n)
        .map(|k| get(x0 as i64 - 1, (y0 + k) as i64))
        .collect();
    let top: Vec<(i32, bool)> = (0..2 * n)
        .map(|k| get((x0 + k) as i64, y0 as i64 - 1))
        .collect();
    MarkedReferenceSamples::new(n, corner, left, top).expect("legal TB geometry")
}

/// A source plane plus the chroma rectangle `(x, y, w, h)` whose
/// reference reads it substitutes (see [`gather_chroma_refs`]).
type SrcOverride<'a> = (&'a [u16], (usize, usize, usize, usize));

/// Chroma twin of [`gather_luma_refs`] (`n` chroma samples at chroma
/// `(cx0, cy0)`; availability tested at the co-located luma — the
/// §8.4.4.2.2 `( xTbCmp, yTbCmp )` scaled by `SubWidthC` /
/// `SubHeightC`, so a 4:2:2 lower block is judged from its own
/// position). `override` substitutes reads inside a chroma rectangle
/// (`x, y, w, h`) with the SOURCE plane — the rough mode elections'
/// stand-in for not-yet-reconstructed samples of the current CU.
fn gather_chroma_refs(
    ctx: &SliceCtx<'_>,
    plane: &[u16],
    cx0: usize,
    cy0: usize,
    n: usize,
    override_src: Option<SrcOverride<'_>>,
) -> MarkedReferenceSamples {
    let cw = ctx.cw();
    let ch = ctx.ch();
    let (sw, sh) = ctx.fmt.sub_wh();
    let get = |x: i64, y: i64| -> (i32, bool) {
        if x < 0 || y < 0 || x >= cw as i64 || y >= ch as i64 {
            return (0, false);
        }
        let (xu, yu) = (x as usize, y as usize);
        if let Some((src, (rx, ry, rw, rh))) = override_src {
            if (rx..rx + rw).contains(&xu) && (ry..ry + rh).contains(&yu) {
                return (i32::from(src[yu * cw + xu]), true);
            }
        }
        if ctx.z_avail(cx0 * sw, cy0 * sh, x * sw as i64, y * sh as i64) {
            (i32::from(plane[yu * cw + xu]), true)
        } else {
            (0, false)
        }
    };
    let corner = get(cx0 as i64 - 1, cy0 as i64 - 1);
    let left: Vec<(i32, bool)> = (0..2 * n)
        .map(|k| get(cx0 as i64 - 1, (cy0 + k) as i64))
        .collect();
    let top: Vec<(i32, bool)> = (0..2 * n)
        .map(|k| get((cx0 + k) as i64, cy0 as i64 - 1))
        .collect();
    MarkedReferenceSamples::new(n, corner, left, top).expect("legal TB geometry")
}

/// SAD-search all 35 §8.4.2 modes for a luma TB read from the frame
/// reconstruction; `override_read` (inside-CU source samples for the
/// 64x64 multi-TU search) substitutes reference reads when set.
fn search_best_mode(
    fmt: &SampleFmt,
    marked: &MarkedReferenceSamples,
    src: &[i32],
) -> (u8, Vec<i32>) {
    let mut best = (0u8, Vec::new());
    let mut best_cost = u64::MAX;
    for mode in 0..=34u8 {
        let pred =
            intra_predict_with_substitution(marked, &pred_params(fmt, mode, PredComponent::Luma))
                .expect("legal prediction params");
        let cost: u64 = src
            .iter()
            .zip(pred.iter())
            .map(|(&s, &p)| u64::from(s.abs_diff(p)))
            .sum();
        if cost < best_cost {
            best_cost = cost;
            best = (mode, pred);
        }
    }
    best
}

/// In-place 1-D Walsh–Hadamard butterfly over `v` (a power-of-two
/// length): the sequency-agnostic sum-of-absolute-transformed-
/// differences kernel.
fn hadamard_1d(v: &mut [i64]) {
    let mut h = 1;
    while h < v.len() {
        let mut i = 0;
        while i < v.len() {
            for j in i..i + h {
                let (a, b) = (v[j], v[j + h]);
                v[j] = a + b;
                v[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
}

/// SATD of one `m x m` block (`m` 4 or 8) of `src − pred` read at
/// `(bx, by)` inside `n x n` buffers: the 2-D Hadamard of the
/// difference, summed in magnitude.
fn satd_block(src: &[i32], pred: &[i32], n: usize, bx: usize, by: usize, m: usize) -> u64 {
    let mut d = [0i64; 64];
    for y in 0..m {
        for x in 0..m {
            d[y * m + x] = i64::from(src[(by + y) * n + bx + x] - pred[(by + y) * n + bx + x]);
        }
    }
    for y in 0..m {
        hadamard_1d(&mut d[y * m..(y + 1) * m]);
    }
    let mut col = [0i64; 8];
    let mut sum = 0u64;
    for x in 0..m {
        for y in 0..m {
            col[y] = d[y * m + x];
        }
        hadamard_1d(&mut col[..m]);
        sum += col[..m].iter().map(|c| c.unsigned_abs()).sum::<u64>();
    }
    sum
}

/// Sum of absolute Hadamard-transformed differences of an `n x n`
/// block (`n` >= 4): 4x4 kernels for a 4x4 block, 8x8 kernels tiled
/// over larger ones, normalized to the SAD scale (the same λ prices
/// both).
fn satd(src: &[i32], pred: &[i32], n: usize) -> u64 {
    if n == 4 {
        return (satd_block(src, pred, 4, 0, 0, 4) + 1) >> 1;
    }
    let mut sum = 0u64;
    let mut by = 0;
    while by < n {
        let mut bx = 0;
        while bx < n {
            sum += satd_block(src, pred, n, bx, by, 8);
            bx += 8;
        }
        by += 8;
    }
    (sum + 2) >> 2
}

/// §7.3.8.5 luma mode signalling cost in bins: `prev_intra_luma_pred_
/// flag` + `mpm_idx` (1 / 2 bypass bins) for a most-probable mode,
/// else the flag + the 5-bit `rem_intra_luma_pred_mode`.
fn luma_mode_bins(mode: u8, mpm: &[u8; 3]) -> u64 {
    match mpm.iter().position(|&m| m == mode) {
        Some(0) => 2,
        Some(_) => 3,
        None => 6,
    }
}

/// The §8.4.2 `candModeList` of the PB at `(x, y)` from the recorded
/// mode field (left / above availability per §6.4.1).
fn mpm_list(ctx: &SliceCtx<'_>, st: &EncState, x: usize, y: usize) -> [u8; 3] {
    let avail_l = ctx.z_avail(x, y, x as i64 - 1, y as i64);
    let avail_a = ctx.z_avail(x, y, x as i64, y as i64 - 1);
    let a = st
        .modes
        .cand_intra_pred_mode(x, y, Neighbour::Left, avail_l);
    let b = st
        .modes
        .cand_intra_pred_mode(x, y, Neighbour::Above, avail_a);
    intra_luma_cand_mode_list(a, b)
}

/// Rough mode decision: every §8.4.2 mode scored by SATD of its
/// prediction against the source plus `λ_me` times its signalling
/// bins; the best `keep` modes in ascending cost (the most-probable
/// mode appended when it did not make the cut — it is the cheapest
/// to signal and a frequent RD winner).
fn rough_intra_modes(
    fmt: &SampleFmt,
    marked: &MarkedReferenceSamples,
    src: &[i32],
    n: usize,
    mpm: &[u8; 3],
    lambda_me: u64,
    keep: usize,
) -> Vec<u8> {
    let mut scored: Vec<(u64, u8)> = (0..=34u8)
        .map(|mode| {
            let pred = intra_predict_with_substitution(
                marked,
                &pred_params(fmt, mode, PredComponent::Luma),
            )
            .expect("legal prediction params");
            (
                satd(src, &pred, n) + lambda_me * luma_mode_bins(mode, mpm),
                mode,
            )
        })
        .collect();
    scored.sort_by_key(|&(c, m)| (c, m));
    let mut out: Vec<u8> = scored.iter().take(keep).map(|&(_, m)| m).collect();
    if !out.contains(&mpm[0]) {
        out.push(mpm[0]);
    }
    out
}

/// Rough chroma mode election for the intra prediction block at luma
/// `(x0, y0)` size `n` whose luma mode is `luma_mode`: the five
/// `intra_chroma_pred_mode` values (Table 8-2: planar / 26 / 10 / 1
/// with the mode-34 substitution, or 4 = the luma mode; the Table 8-3
/// 4:2:2 remap applied) scored by the SATD of the Cb + Cr predictions
/// from the current reconstruction plus `λ_me` times the §9.3.3.8
/// bins (1 for 4, 3 otherwise). A 4:2:2 block's lower half is
/// predicted with the upper half's SOURCE samples standing in for its
/// not-yet-coded reconstruction. Returns `(intra_chroma_pred_mode,
/// IntraPredModeC)`; `(4, luma_mode)` for a monochrome picture.
fn elect_chroma_mode(
    ctx: &SliceCtx<'_>,
    st: &EncState,
    x0: usize,
    y0: usize,
    n: usize,
    luma_mode: u8,
    lambda_me: u64,
) -> (u8, u8) {
    let fmt = &ctx.fmt;
    if !fmt.has_chroma() {
        return (4, luma_mode);
    }
    let (sw, sh) = fmt.sub_wh();
    let (cx0, cy0) = (x0 / sw, y0 / sh);
    // The chroma block(s) of an n x n luma block: side n / SubWidthC,
    // stacked `chroma_blocks()` times — each scored as square blocks
    // of at most the 32x32 maximum TB (a 4:4:4 64x64 CU is predicted
    // as four 32x32 chroma TBs).
    let side = n / sw;
    let nc = side.min(32);
    let per_row = side / nc;
    let blocks = fmt.chroma_blocks();
    let cw = ctx.cw();
    let rect = (cx0, cy0, side, side * blocks);
    let mut best = (4u8, luma_mode, u64::MAX);
    let mut cache: Vec<(u8, u64)> = Vec::with_capacity(5);
    for idx in [4u8, 0, 1, 2, 3] {
        let mode_c = crate::binarization::derive_intra_pred_mode_c(
            idx,
            luma_mode,
            fmt.chroma_format_idc == 2,
        );
        // Distinct indices can map to one mode (idx 4 vs an explicit
        // one): the SATD is a function of the mode alone.
        let satd_c = match cache.iter().find(|(m, _)| *m == mode_c) {
            Some(&(_, d)) => d,
            None => {
                let mut d = 0u64;
                for (plane_idx, pc) in [(1usize, PredComponent::Cb), (2, PredComponent::Cr)] {
                    let recon_plane = if plane_idx == 1 {
                        &st.recon.cb
                    } else {
                        &st.recon.cr
                    };
                    for v in 0..blocks * per_row {
                        for u in 0..per_row {
                            let (cx, cy) = (cx0 + u * nc, cy0 + v * nc);
                            let marked = gather_chroma_refs(
                                ctx,
                                recon_plane,
                                cx,
                                cy,
                                nc,
                                Some((ctx.src[plane_idx], rect)),
                            );
                            let pred = intra_predict_with_substitution(
                                &marked,
                                &pred_params(fmt, mode_c, pc),
                            )
                            .expect("legal prediction params");
                            let src = extract(ctx.src[plane_idx], cw, cx, cy, nc);
                            d += satd(&src, &pred, nc);
                        }
                    }
                }
                cache.push((mode_c, d));
                d
            }
        };
        let cost = satd_c + lambda_me * if idx == 4 { 1 } else { 3 };
        if cost < best.2 {
            best = (idx, mode_c, cost);
        }
    }
    (best.0, best.1)
}

/// Pick the 64x64 intra CU's single PB mode: the CU is forced to four
/// 32x32 TUs, so each candidate mode is scored by SAD over the four
/// TU predictions with in-CU (not-yet-reconstructed) reference reads
/// substituted from the SOURCE picture — a search heuristic only; the
/// actual coding pass predicts from the real reconstruction. With
/// `rough` set (`intra_rd >= 1`) the score is SATD + `λ_me` times the
/// mode's signalling bins against `mpm`, as for the smaller CUs.
fn search_mode_64(
    ctx: &SliceCtx<'_>,
    st: &EncState,
    x0: usize,
    y0: usize,
    rough: Option<(&[u8; 3], u64)>,
) -> u8 {
    let mut best = (0u8, u64::MAX);
    let read = |x: i64, y: i64| -> i32 {
        let (xu, yu) = (x as usize, y as usize);
        if xu >= x0 && xu < x0 + 64 && yu >= y0 && yu < y0 + 64 {
            i32::from(ctx.src[0][yu * ctx.width + xu])
        } else {
            i32::from(st.recon.y[yu * ctx.width + xu])
        }
    };
    for mode in 0..=34u8 {
        let mut cost = 0u64;
        for &(zx, zy) in &Z_OFFSETS {
            let (tx, ty) = (x0 + zx * 32, y0 + zy * 32);
            let get = |x: i64, y: i64| -> (i32, bool) {
                if ctx.z_avail(x0, y0, x, y)
                    || (x >= x0 as i64 && x < (x0 + 64) as i64 && y >= y0 as i64 && y < ty as i64)
                    || (y >= ty as i64 && y < (ty + 32) as i64 && x >= x0 as i64 && x < tx as i64)
                {
                    (read(x, y), true)
                } else {
                    (0, false)
                }
            };
            let corner = get(tx as i64 - 1, ty as i64 - 1);
            let left: Vec<(i32, bool)> = (0..64)
                .map(|k| get(tx as i64 - 1, (ty + k) as i64))
                .collect();
            let top: Vec<(i32, bool)> = (0..64)
                .map(|k| get((tx + k) as i64, ty as i64 - 1))
                .collect();
            let marked =
                MarkedReferenceSamples::new(32, corner, left, top).expect("legal TB geometry");
            let pred = intra_predict_with_substitution(
                &marked,
                &pred_params(&ctx.fmt, mode, PredComponent::Luma),
            )
            .expect("legal prediction params");
            let src = extract(ctx.src[0], ctx.width, tx, ty, 32);
            cost += match rough {
                Some(_) => satd(&src, &pred, 32),
                None => src
                    .iter()
                    .zip(pred.iter())
                    .map(|(&s, &p)| u64::from(s.abs_diff(p)))
                    .sum::<u64>(),
            };
        }
        if let Some((mpm, lambda_me)) = rough {
            cost += lambda_me * luma_mode_bins(mode, mpm);
        }
        if cost < best.1 {
            best = (mode, cost);
        }
    }
    best.0
}

/// Code one intra luma TB at `(x, y)` (predict from the frame recon,
/// transform, reconstruct into the frame recon). Returns
/// `(levels, dist, mode_pred_sad_unused)`.
#[allow(clippy::too_many_arguments)]
fn code_intra_luma_tb(
    ctx: &SliceCtx<'_>,
    st: &mut EncState,
    x: usize,
    y: usize,
    n: usize,
    mode: u8,
    qp_y: u32,
    lambda: u64,
) -> (Vec<i32>, u64) {
    let marked = gather_luma_refs(ctx, &st.recon.y, x, y, n);
    let pred =
        intra_predict_with_substitution(&marked, &pred_params(&ctx.fmt, mode, PredComponent::Luma))
            .expect("legal prediction params");
    let src = extract(ctx.src[0], ctx.width, x, y, n);
    let tools = TbTools::new(
        ctx,
        st.rdoq_model.as_ref(),
        lambda,
        true,
        n.trailing_zeros(),
        0,
        mode,
    );
    let (levels, recon) = code_tb(
        &src,
        &pred,
        n,
        qp_y,
        Component::Luma,
        PredMode::Intra,
        tools,
    );
    let dist = ssd(&recon, &src);
    store(&mut st.recon.y, ctx.width, x, y, n, &recon);
    (levels, dist)
}

/// Code the intra chroma blocks of the transform-tree node at luma
/// `(x, y)` of luma size `log2` (`log2_c` per §7.3.8.10; the deferred
/// 4x4 chroma of a `log2TrafoSize == 3` split passes `log2 == 3`):
/// the `ChromaArrayType == 2 ? 2 : 1` stacked square blocks per
/// component, each predicted from the reconstruction after the
/// previous one (§8.4.4.1: the lower 4:2:2 block reads the upper's
/// reconstructed samples), transformed, reconstructed INTO the frame.
/// Returns `(cb_levels, cr_levels, dist)` with the blocks' levels
/// back to back; empty for a monochrome picture.
#[allow(clippy::too_many_arguments)]
fn code_intra_chroma_tbs(
    ctx: &SliceCtx<'_>,
    st: &mut EncState,
    x: usize,
    y: usize,
    log2: u32,
    mode_c: u8,
    qp_c: u32,
    lambda: u64,
) -> (Vec<i32>, Vec<i32>, u64) {
    let fmt = ctx.fmt;
    if !fmt.has_chroma() {
        return (Vec::new(), Vec::new(), 0);
    }
    let (sw, sh) = fmt.sub_wh();
    let (cx, cy) = (x / sw, y / sh);
    let log2_c = fmt.log2_chroma_tb(log2);
    let n = 1usize << log2_c;
    let blocks = fmt.chroma_blocks();
    let cw = ctx.cw();
    let model = st.rdoq_model.clone();
    let mut do_plane = |plane_idx: usize, comp: Component, pc: PredComponent| -> (Vec<i32>, u64) {
        let mut all_levels = Vec::with_capacity(n * n * blocks);
        let mut dist = 0u64;
        for v in 0..blocks {
            let cy_v = cy + v * n;
            let recon_plane = match plane_idx {
                1 => &st.recon.cb,
                _ => &st.recon.cr,
            };
            let marked = gather_chroma_refs(ctx, recon_plane, cx, cy_v, n, None);
            let pred = intra_predict_with_substitution(&marked, &pred_params(&fmt, mode_c, pc))
                .expect("legal prediction params");
            let src = extract(ctx.src[plane_idx], cw, cx, cy_v, n);
            let tools = TbTools::new(
                ctx,
                model.as_ref(),
                lambda,
                true,
                log2_c,
                plane_idx as u8,
                mode_c,
            );
            let (levels, recon) = code_tb(&src, &pred, n, qp_c, comp, PredMode::Intra, tools);
            dist += ssd(&recon, &src);
            let recon_plane = match plane_idx {
                1 => &mut st.recon.cb,
                _ => &mut st.recon.cr,
            };
            store(recon_plane, cw, cx, cy_v, n, &recon);
            all_levels.extend(levels);
        }
        (all_levels, dist)
    };
    let (cb, d1) = do_plane(1, Component::Cb, PredComponent::Cb);
    let (cr, d2) = do_plane(2, Component::Cr, PredComponent::Cr);
    (cb, cr, d1 + d2)
}

/// The recursive intra residual quadtree for a `PART_2Nx2N` CU with
/// PB mode `mode` / chroma mode `mode_c`: at each node the unsplit TU
/// competes against the four-child split under SSD + λ·bins (splits
/// that the §7.3.8.8 gate cannot signal are never elected; forced
/// splits are always taken). Codes INTO the frame reconstruction.
/// Returns `(node, dist, rate_bins)`.
#[allow(clippy::too_many_arguments)]
fn intra_rqt(
    ctx: &SliceCtx<'_>,
    st: &mut EncState,
    x: usize,
    y: usize,
    log2: u32,
    depth: u32,
    max_depth: u32,
    mode: u8,
    mode_c: u8,
    qp_y: u32,
    qp_c: u32,
    lambda: u64,
) -> (TuNode, u64, u64) {
    let max_tb = ctx.cfg.max_tb_log2();
    let split_forced = log2 > max_tb;
    let split_allowed = log2 <= max_tb && log2 > 2 && depth < max_depth;
    // Whether split_transform_flag is CODED here (vs inferred): the
    // §7.3.8.8 presence gate. IntraSplitFlag CUs never reach this
    // function at depth 0 (the NxN path codes its forced tree itself).
    let flag_coded = split_allowed;

    // The chroma cbf bins a node codes: one per component per block
    // (two blocks at 4:2:2).
    let chroma_cbf_bins = 2 * ctx.fmt.chroma_blocks() as u64;
    let leaf_eval = |st: &mut EncState| -> (TuNode, u64, u64) {
        let n = 1usize << log2;
        let (y_lv, d_y) = code_intra_luma_tb(ctx, st, x, y, n, mode, qp_y, lambda);
        let (cb, cr, d_c) = if ctx.fmt.chroma_in_place(log2) {
            code_intra_chroma_tbs(ctx, st, x, y, log2, mode_c, qp_c, lambda)
        } else {
            (Vec::new(), Vec::new(), 0)
        };
        // cbf bins: luma always signalled on intra; the chroma flags
        // when chroma is coded in place at this node.
        let rate = rate_proxy(&y_lv)
            + if ctx.fmt.chroma_in_place(log2) {
                rate_proxy(&cb) + rate_proxy(&cr) + chroma_cbf_bins
            } else {
                0
            }
            + 1;
        let dist = d_y + d_c;
        (TuNode::Leaf { y: y_lv, cb, cr }, dist, rate)
    };

    let split_eval = |ctx: &SliceCtx<'_>, st: &mut EncState| -> (TuNode, u64, u64) {
        let half = 1usize << (log2 - 1);
        let mut children: Vec<TuNode> = Vec::with_capacity(4);
        let mut dist = 0u64;
        let mut rate = 0u64;
        for &(zx, zy) in &Z_OFFSETS {
            let (node, d, r) = intra_rqt(
                ctx,
                st,
                x + zx * half,
                y + zy * half,
                log2 - 1,
                depth + 1,
                max_depth,
                mode,
                mode_c,
                qp_y,
                qp_c,
                lambda,
            );
            children.push(node);
            dist += d;
            rate += r;
        }
        // Deferred 4x4 chroma at the log2 == 3 split parent (4:2:0 /
        // 4:2:2: the children are 4x4 luma leaves without chroma).
        let (cb, cr) = if log2 == 3 && ctx.fmt.has_chroma() && ctx.fmt.chroma_format_idc != 3 {
            let (cb, cr, d_c) = code_intra_chroma_tbs(ctx, st, x, y, 3, mode_c, qp_c, lambda);
            dist += d_c;
            rate += rate_proxy(&cb) + rate_proxy(&cr) + chroma_cbf_bins;
            (cb, cr)
        } else {
            if ctx.fmt.has_chroma() {
                rate += 2; // this node's cbf_cb / cbf_cr pair
            }
            (Vec::new(), Vec::new())
        };
        let children: Box<[TuNode; 4]> = match children.try_into() {
            Ok(c) => Box::new(c),
            Err(_) => unreachable!("four children pushed"),
        };
        (TuNode::Split { children, cb, cr }, dist, rate)
    };

    if split_forced {
        let (node, dist, rate) = split_eval(ctx, st);
        return (node, dist, rate);
    }
    if !split_allowed {
        return leaf_eval(st);
    }
    // Both are possible: measure the leaf, roll back, measure the
    // split, keep the cheaper (restoring the loser's state).
    let n = 1usize << log2;
    let before = st.snapshot(ctx, x, y, n);
    let (leaf_node, leaf_dist, leaf_rate) = leaf_eval(st);
    let leaf_cost = leaf_dist + lambda * (leaf_rate + u64::from(flag_coded));
    let after_leaf = st.snapshot(ctx, x, y, n);
    st.restore(ctx, &before);
    let (split_node, split_dist, split_rate) = split_eval(ctx, st);
    let split_cost = split_dist + lambda * (split_rate + u64::from(flag_coded));
    if leaf_cost <= split_cost {
        st.restore(ctx, &after_leaf);
        (leaf_node, leaf_dist, leaf_rate + u64::from(flag_coded))
    } else {
        (split_node, split_dist, split_rate + u64::from(flag_coded))
    }
}

/// Code the best intra CU at `(x0, y0)` size `1 << log2` INTO the
/// state, dispatching on the configured mode-decision effort
/// ([`TreeCfg::intra_rd`]).
#[allow(clippy::too_many_arguments)]
fn code_intra_cu(
    ctx: &SliceCtx<'_>,
    st: &mut EncState,
    x0: usize,
    y0: usize,
    log2: u32,
    depth: u32,
    ctb_qp: i32,
) -> CuCoded {
    if ctx.cfg.intra_rd == 0 {
        code_intra_cu_legacy(ctx, st, x0, y0, log2, depth, ctb_qp)
    } else {
        code_intra_cu_rd(ctx, st, x0, y0, log2, depth, ctb_qp)
    }
}

/// The `PART_NxN` tail shared by both intra CU coders: the chroma of
/// a four-4x4-PB CU. At 4:2:0 / 4:2:2 the CU has ONE chroma PB whose
/// mode derives from PB 0 and whose 4x4 (pair of 4x4) blocks are
/// deferred to the split node; at 4:4:4 every 4x4 luma leaf carries
/// its own 4x4 chroma blocks with its OWN `intra_chroma_pred_mode`
/// (§7.3.8.5 signals four). `elect` picks each chroma PB's index (the
/// legacy coder passes the derived mode 4). Returns the coded tree,
/// the per-PB chroma indices and the added `(dist, rate_bins,
/// chroma_mode_bins)`.
#[allow(clippy::too_many_arguments)]
fn code_nxn_chroma(
    ctx: &SliceCtx<'_>,
    st: &mut EncState,
    x0: usize,
    y0: usize,
    pb_modes: &[u8; 4],
    luma_lv: Vec<Vec<i32>>,
    qp_c: u32,
    lambda: u64,
    lambda_me: u64,
    elect: bool,
) -> (TuNode, [u8; 4], u64, u64, u64) {
    let fmt = ctx.fmt;
    let chroma_bins = |idx: u8| if idx == 4 { 1u64 } else { 3 };
    let mut dist = 0u64;
    let mut rate = 0u64;
    let mut mode_bins = 0u64;
    let mut chroma_idx = [4u8; 4];
    if fmt.chroma_format_idc == 3 {
        // Per-PB chroma, coded in place at each 4x4 leaf.
        let mut leaves: Vec<TuNode> = Vec::with_capacity(4);
        for (k, (y, &(zx, zy))) in luma_lv.into_iter().zip(Z_OFFSETS.iter()).enumerate() {
            let (px, py) = (x0 + zx * 4, y0 + zy * 4);
            let (idx, mode_c) = if elect {
                elect_chroma_mode(ctx, st, px, py, 4, pb_modes[k], lambda_me)
            } else {
                (4, pb_modes[k])
            };
            let (cb, cr, d_c) = code_intra_chroma_tbs(ctx, st, px, py, 2, mode_c, qp_c, lambda);
            dist += d_c;
            rate += rate_proxy(&cb) + rate_proxy(&cr) + 2;
            mode_bins += chroma_bins(idx);
            chroma_idx[k] = idx;
            leaves.push(TuNode::Leaf { y, cb, cr });
        }
        let children: Box<[TuNode; 4]> =
            Box::new(leaves.try_into().map_err(|_| ()).expect("four leaves"));
        return (
            TuNode::Split {
                children,
                cb: Vec::new(),
                cr: Vec::new(),
            },
            chroma_idx,
            dist,
            rate,
            mode_bins,
        );
    }
    let children: Box<[TuNode; 4]> = Box::new(
        luma_lv
            .into_iter()
            .map(|y| TuNode::Leaf {
                y,
                cb: Vec::new(),
                cr: Vec::new(),
            })
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| ())
            .expect("four leaves"),
    );
    let (cb, cr) = if fmt.has_chroma() {
        let (idx, mode_c) = if elect {
            elect_chroma_mode(ctx, st, x0, y0, 8, pb_modes[0], lambda_me)
        } else {
            (4, pb_modes[0])
        };
        let (cb, cr, d_c) = code_intra_chroma_tbs(ctx, st, x0, y0, 3, mode_c, qp_c, lambda);
        dist += d_c;
        rate += rate_proxy(&cb) + rate_proxy(&cr) + 2 * fmt.chroma_blocks() as u64;
        mode_bins += chroma_bins(idx);
        chroma_idx = [idx; 4];
        (cb, cr)
    } else {
        (Vec::new(), Vec::new())
    };
    (
        TuNode::Split { children, cb, cr },
        chroma_idx,
        dist,
        rate,
        mode_bins,
    )
}

/// The `intra_rd == 0` intra CU coder (byte-stable with the historical
/// streams): SAD mode search, derived chroma mode. Codes the best
/// intra CU at `(x0, y0)` size `1 << log2` INTO the
/// state (reconstruction + mode field + cells): `PART_2Nx2N` with the
/// RD-elected RQT, and additionally `PART_NxN` (four 4x4 PBs, DST
/// TUs) at `MinCbSizeY`.
#[allow(clippy::too_many_arguments)]
fn code_intra_cu_legacy(
    ctx: &SliceCtx<'_>,
    st: &mut EncState,
    x0: usize,
    y0: usize,
    log2: u32,
    depth: u32,
    ctb_qp: i32,
) -> CuCoded {
    let n = 1usize << log2;
    let qp_y = ctx.qp_y_prime(ctb_qp);
    let qp_c = ctx.qp_c_prime(ctb_qp);
    let lambda = ctx.lambda_of(ctb_qp);
    // Per-CU syntax overhead proxy: pred_mode (P/B) + part_mode (at
    // MinCb) + luma mode ~6/PB + chroma mode 1.
    let base_bins = u64::from(!ctx.intra_slice) + u64::from(log2 == ctx.cfg.min_cb_log2());
    let before = st.snapshot(ctx, x0, y0, n);

    // ---- PART_2Nx2N ----
    let mode = if log2 == 6 {
        search_mode_64(ctx, st, x0, y0, None)
    } else {
        let marked = gather_luma_refs(ctx, &st.recon.y, x0, y0, n);
        let src = extract(ctx.src[0], ctx.width, x0, y0, n);
        search_best_mode(&ctx.fmt, &marked, &src).0
    };
    // The derived chroma mode (intra_chroma_pred_mode 4) — Table 8-3
    // remapped at 4:2:2.
    let mode_c =
        crate::binarization::derive_intra_pred_mode_c(4, mode, ctx.fmt.chroma_format_idc == 2);
    // §8.4.2 derivation order: the PB's own recorded mode must be in
    // place before its TUs' neighbours inside the CU are derived? No —
    // the mode field is only consulted by LATER PBs; record after.
    let max_depth_2n = ctx.cfg.th_depth_intra; // IntraSplitFlag == 0
    let (tree_2n, dist_2n, rate_2n) = intra_rqt(
        ctx,
        st,
        x0,
        y0,
        log2,
        0,
        max_depth_2n,
        mode,
        mode_c,
        qp_y,
        qp_c,
        lambda,
    );
    let cost_2n = dist_2n + lambda * (rate_2n + base_bins + 7);
    let cu_2n = CuCoded {
        x0,
        y0,
        log2,
        kind: TreeCuKind::Intra {
            modes: [mode; 4],
            nxn: false,
            chroma: [4; 4],
        },
        motions: Vec::new(),
        tree: Some(tree_2n),
        cost: cost_2n,
    };
    st.modes.record_intra_pb(x0, y0, n, mode, false);

    // ---- PART_NxN at MinCbSizeY (four 4x4 PBs, forced depth-1) ----
    let cu = if log2 == ctx.cfg.min_cb_log2() && log2 == 3 {
        let after_2n = st.snapshot(ctx, x0, y0, n);
        st.restore(ctx, &before);
        let mut pb_modes = [0u8; 4];
        let mut luma_lv: Vec<Vec<i32>> = Vec::with_capacity(4);
        let mut dist = 0u64;
        let mut rate = 0u64;
        for (k, &(zx, zy)) in Z_OFFSETS.iter().enumerate() {
            let (px, py) = (x0 + zx * 4, y0 + zy * 4);
            let marked = gather_luma_refs(ctx, &st.recon.y, px, py, 4);
            let src = extract(ctx.src[0], ctx.width, px, py, 4);
            let (m, _) = search_best_mode(&ctx.fmt, &marked, &src);
            let (lv, d) = code_intra_luma_tb(ctx, st, px, py, 4, m, qp_y, lambda);
            // §8.4.2: later PBs' candidate lists see this PB's mode.
            st.modes.record_intra_pb(px, py, 4, m, false);
            pb_modes[k] = m;
            rate += rate_proxy(&lv) + 1;
            dist += d;
            luma_lv.push(lv);
        }
        let (tree, chroma_idx, d_c, r_c, _) =
            code_nxn_chroma(ctx, st, x0, y0, &pb_modes, luma_lv, qp_c, lambda, 0, false);
        dist += d_c;
        rate += r_c;
        let cost_nxn = dist + lambda * (rate + base_bins + 4 * 7);
        if cost_nxn < cost_2n {
            CuCoded {
                x0,
                y0,
                log2,
                kind: TreeCuKind::Intra {
                    modes: pb_modes,
                    nxn: true,
                    chroma: chroma_idx,
                },
                motions: Vec::new(),
                tree: Some(tree),
                cost: cost_nxn,
            }
        } else {
            st.restore(ctx, &after_2n);
            cu_2n
        }
    } else {
        cu_2n
    };

    // Commit the non-recon state.
    st.field.fill_rect(
        x0,
        y0,
        n,
        n,
        MotionCell {
            is_intra: true,
            ref_poc_l0: i32::MIN,
            ref_poc_l1: i32::MIN,
            ..MotionCell::default()
        },
    );
    st.fill_cells(x0, y0, n, depth as i8, 0);
    cu
}

/// The `intra_rd >= 1` intra CU coder: SATD + λ·bins rough decision,
/// chroma-mode election, full RD over the short list at level 2.
/// Codes the best intra CU at `(x0, y0)` size `1 << log2` INTO the
/// state (reconstruction + mode field + cells): `PART_2Nx2N` with the
/// RD-elected RQT, and additionally `PART_NxN` (four 4x4 PBs, DST
/// TUs) at `MinCbSizeY`.
#[allow(clippy::too_many_arguments)]
fn code_intra_cu_rd(
    ctx: &SliceCtx<'_>,
    st: &mut EncState,
    x0: usize,
    y0: usize,
    log2: u32,
    depth: u32,
    ctb_qp: i32,
) -> CuCoded {
    let n = 1usize << log2;
    let qp_y = ctx.qp_y_prime(ctb_qp);
    let qp_c = ctx.qp_c_prime(ctb_qp);
    let lambda = ctx.lambda_of(ctb_qp);
    let lambda_me = crate::encoder::rate::motion_lambda(lambda);
    // Per-CU syntax overhead proxy: pred_mode (P/B) + part_mode (at
    // MinCb); the luma / chroma mode bins are priced per candidate.
    let base_bins = u64::from(!ctx.intra_slice) + u64::from(log2 == ctx.cfg.min_cb_log2());
    let chroma_bins = |idx: u8| if idx == 4 { 1 } else { 3 };

    let before = st.snapshot(ctx, x0, y0, n);
    let mpm = mpm_list(ctx, st, x0, y0);

    // ---- PART_2Nx2N ----
    // Rough decision (SATD + λ_me·bins over all 35 modes) keeps a
    // short list; each survivor is then coded for real through the
    // RD-elected RQT (with its own chroma mode election) and the
    // cheapest SSD + λ·bins candidate is kept.
    let cands: Vec<u8> = if log2 == 6 {
        vec![search_mode_64(ctx, st, x0, y0, Some((&mpm, lambda_me)))]
    } else {
        let marked = gather_luma_refs(ctx, &st.recon.y, x0, y0, n);
        let src = extract(ctx.src[0], ctx.width, x0, y0, n);
        let keep = match (ctx.cfg.intra_rd, log2) {
            (1, _) => 1,
            (_, 3) => 3,
            _ => 2,
        };
        let mut c = rough_intra_modes(&ctx.fmt, &marked, &src, n, &mpm, lambda_me, keep);
        if ctx.cfg.intra_rd == 1 {
            c.truncate(1);
        }
        c
    };
    let max_depth_2n = ctx.cfg.th_depth_intra; // IntraSplitFlag == 0
    let mut best_2n: Option<(u8, u8, TuNode, u64, Snap)> = None;
    for (k, &mode) in cands.iter().enumerate() {
        if k > 0 {
            st.restore(ctx, &before);
        }
        let (chroma_idx, mode_c) = elect_chroma_mode(ctx, st, x0, y0, n, mode, lambda_me);
        let (tree, dist, rate) = intra_rqt(
            ctx,
            st,
            x0,
            y0,
            log2,
            0,
            max_depth_2n,
            mode,
            mode_c,
            qp_y,
            qp_c,
            lambda,
        );
        // A monochrome CU signals no chroma mode.
        let c_bins = if ctx.fmt.has_chroma() {
            chroma_bins(chroma_idx)
        } else {
            0
        };
        let cost = dist + lambda * (rate + base_bins + luma_mode_bins(mode, &mpm) + c_bins);
        if best_2n.as_ref().map_or(true, |b| cost < b.3) {
            best_2n = Some((mode, chroma_idx, tree, cost, st.snapshot(ctx, x0, y0, n)));
        }
    }
    let (mode, chroma_2n, tree_2n, cost_2n, after_2n_coded) = best_2n.expect("a candidate");
    if cands.len() > 1 {
        st.restore(ctx, &after_2n_coded);
    }
    let cu_2n = CuCoded {
        x0,
        y0,
        log2,
        kind: TreeCuKind::Intra {
            modes: [mode; 4],
            nxn: false,
            chroma: [chroma_2n; 4],
        },
        motions: Vec::new(),
        tree: Some(tree_2n),
        cost: cost_2n,
    };
    // The mode field is only consulted by LATER PBs; record after.
    st.modes.record_intra_pb(x0, y0, n, mode, false);

    // ---- PART_NxN at MinCbSizeY (four 4x4 PBs, forced depth-1) ----
    let cu = if log2 == ctx.cfg.min_cb_log2() && log2 == 3 {
        let after_2n = st.snapshot(ctx, x0, y0, n);
        st.restore(ctx, &before);
        let mut pb_modes = [0u8; 4];
        let mut luma_lv: Vec<Vec<i32>> = Vec::with_capacity(4);
        let mut dist = 0u64;
        let mut rate = 0u64;
        let mut mode_bins = 0u64;
        for (k, &(zx, zy)) in Z_OFFSETS.iter().enumerate() {
            let (px, py) = (x0 + zx * 4, y0 + zy * 4);
            let marked = gather_luma_refs(ctx, &st.recon.y, px, py, 4);
            let src = extract(ctx.src[0], ctx.width, px, py, 4);
            let pb_mpm = mpm_list(ctx, st, px, py);
            let mut pb_cands = rough_intra_modes(&ctx.fmt, &marked, &src, 4, &pb_mpm, lambda_me, 3);
            if ctx.cfg.intra_rd == 1 {
                pb_cands.truncate(1);
            }
            // Full RD over the survivors on this 4x4 TB.
            let pb_before = st.snapshot(ctx, px, py, 4);
            let mut best: Option<(u8, Vec<i32>, u64, u64, Snap)> = None;
            for (i, &m) in pb_cands.iter().enumerate() {
                if i > 0 {
                    st.restore(ctx, &pb_before);
                }
                let (lv, d) = code_intra_luma_tb(ctx, st, px, py, 4, m, qp_y, lambda);
                let bins = rate_proxy(&lv) + 1 + luma_mode_bins(m, &pb_mpm);
                let cost = d + lambda * bins;
                if best.as_ref().map_or(true, |b| cost < b.3) {
                    best = Some((m, lv, d, cost, st.snapshot(ctx, px, py, 4)));
                }
            }
            let (m, lv, d, _, after) = best.expect("a candidate");
            if pb_cands.len() > 1 {
                st.restore(ctx, &after);
            }
            // §8.4.2: later PBs' candidate lists see this PB's mode.
            st.modes.record_intra_pb(px, py, 4, m, false);
            pb_modes[k] = m;
            mode_bins += luma_mode_bins(m, &pb_mpm);
            rate += rate_proxy(&lv) + 1;
            dist += d;
            luma_lv.push(lv);
        }
        let (tree, chroma_nxn, d_c, r_c, c_bins) = code_nxn_chroma(
            ctx, st, x0, y0, &pb_modes, luma_lv, qp_c, lambda, lambda_me, true,
        );
        dist += d_c;
        rate += r_c;
        let cost_nxn = dist + lambda * (rate + base_bins + mode_bins + c_bins);
        if cost_nxn < cost_2n {
            CuCoded {
                x0,
                y0,
                log2,
                kind: TreeCuKind::Intra {
                    modes: pb_modes,
                    nxn: true,
                    chroma: chroma_nxn,
                },
                motions: Vec::new(),
                tree: Some(tree),
                cost: cost_nxn,
            }
        } else {
            st.restore(ctx, &after_2n);
            cu_2n
        }
    } else {
        cu_2n
    };

    // Commit the non-recon state.
    st.field.fill_rect(
        x0,
        y0,
        n,
        n,
        MotionCell {
            is_intra: true,
            ref_poc_l0: i32::MIN,
            ref_poc_l1: i32::MIN,
            ..MotionCell::default()
        },
    );
    st.fill_cells(x0, y0, n, depth as i8, 0);
    cu
}

// ---------------------------------------------------------------------
// Inter CU coding
// ---------------------------------------------------------------------

/// The recursive inter residual quadtree over a fixed CU-wide
/// prediction. Pure: codes nothing into the state. `x`/`y` are
/// CU-relative; `pred`/`src` are CU-local buffers (luma `n_cb`x`n_cb`,
/// chroma half). Returns `(node, recon_local, dist, rate_bins)` where
/// `recon_local` holds the node's luma + chroma reconstructions.
#[allow(clippy::too_many_arguments)]
fn inter_rqt(
    ctx: &SliceCtx<'_>,
    bufs: &InterCuBufs<'_>,
    x: usize,
    y: usize,
    log2: u32,
    depth: u32,
    max_depth: u32,
    inter_split: bool,
    qp_y: u32,
    qp_c: u32,
    lambda: u64,
) -> (TuNode, LocalRecon, u64, u64) {
    let max_tb = ctx.cfg.max_tb_log2();
    let split_forced = log2 > max_tb || (inter_split && depth == 0);
    let split_allowed = log2 <= max_tb && log2 > 2 && depth < max_depth && !split_forced;

    let leaf_eval = || -> (TuNode, LocalRecon, u64, u64) {
        let n = 1usize << log2;
        let n_cb = bufs.n_cb;
        let src_y = sub_block(bufs.src_y, n_cb, x, y, n);
        let pred_y = sub_block(bufs.pred_y, n_cb, x, y, n);
        let tools = |c_idx: u8| TbTools::new(ctx, bufs.model, lambda, false, log2, c_idx, 0);
        let (y_lv, y_rc) = code_tb(
            &src_y,
            &pred_y,
            n,
            qp_y,
            Component::Luma,
            PredMode::Inter,
            tools(0),
        );
        let mut dist = ssd(&y_rc, &src_y);
        let mut rate = rate_proxy(&y_lv) + 1;
        let mut local = LocalRecon {
            y: y_rc,
            cb: Vec::new(),
            cr: Vec::new(),
        };
        let (cb_lv, cr_lv) = if log2 >= 3 {
            let hc = n / 2;
            let src_cb = sub_block(bufs.src_cb, n_cb / 2, x / 2, y / 2, hc);
            let pred_cb = sub_block(bufs.pred_cb, n_cb / 2, x / 2, y / 2, hc);
            let chroma_tools =
                |c_idx: u8| TbTools::new(ctx, bufs.model, lambda, false, log2 - 1, c_idx, 0);
            let (cb_lv, cb_rc) = code_tb(
                &src_cb,
                &pred_cb,
                hc,
                qp_c,
                Component::Cb,
                PredMode::Inter,
                chroma_tools(1),
            );
            let src_cr = sub_block(bufs.src_cr, n_cb / 2, x / 2, y / 2, hc);
            let pred_cr = sub_block(bufs.pred_cr, n_cb / 2, x / 2, y / 2, hc);
            let (cr_lv, cr_rc) = code_tb(
                &src_cr,
                &pred_cr,
                hc,
                qp_c,
                Component::Cr,
                PredMode::Inter,
                chroma_tools(2),
            );
            dist += ssd(&cb_rc, &src_cb) + ssd(&cr_rc, &src_cr);
            rate += rate_proxy(&cb_lv) + rate_proxy(&cr_lv) + 2;
            local.cb = cb_rc;
            local.cr = cr_rc;
            (cb_lv, cr_lv)
        } else {
            (Vec::new(), Vec::new())
        };
        (
            TuNode::Leaf {
                y: y_lv,
                cb: cb_lv,
                cr: cr_lv,
            },
            local,
            dist,
            rate,
        )
    };

    let split_eval = || -> (TuNode, LocalRecon, u64, u64) {
        let half = 1usize << (log2 - 1);
        let n = 1usize << log2;
        let mut children: Vec<TuNode> = Vec::with_capacity(4);
        let mut local = LocalRecon {
            y: vec![0u16; n * n],
            cb: vec![0u16; (n / 2) * (n / 2)],
            cr: vec![0u16; (n / 2) * (n / 2)],
        };
        let mut dist = 0u64;
        let mut rate = 0u64;
        for &(zx, zy) in &Z_OFFSETS {
            let (node, child, d, r) = inter_rqt(
                ctx,
                bufs,
                x + zx * half,
                y + zy * half,
                log2 - 1,
                depth + 1,
                max_depth,
                inter_split,
                qp_y,
                qp_c,
                lambda,
            );
            // Paste the child's luma into the node-local recon.
            rect_paste(&mut local.y, n, zx * half, zy * half, half, half, &child.y);
            if log2 > 3 {
                rect_paste(
                    &mut local.cb,
                    n / 2,
                    zx * half / 2,
                    zy * half / 2,
                    half / 2,
                    half / 2,
                    &child.cb,
                );
                rect_paste(
                    &mut local.cr,
                    n / 2,
                    zx * half / 2,
                    zy * half / 2,
                    half / 2,
                    half / 2,
                    &child.cr,
                );
            }
            children.push(node);
            dist += d;
            rate += r;
        }
        // Deferred 4x4 chroma at a log2 == 3 split parent.
        let (cb_lv, cr_lv) = if log2 == 3 {
            let n_cb = bufs.n_cb;
            let src_cb = sub_block(bufs.src_cb, n_cb / 2, x / 2, y / 2, 4);
            let pred_cb = sub_block(bufs.pred_cb, n_cb / 2, x / 2, y / 2, 4);
            let chroma_tools =
                |c_idx: u8| TbTools::new(ctx, bufs.model, lambda, false, 2, c_idx, 0);
            let (cb_lv, cb_rc) = code_tb(
                &src_cb,
                &pred_cb,
                4,
                qp_c,
                Component::Cb,
                PredMode::Inter,
                chroma_tools(1),
            );
            let src_cr = sub_block(bufs.src_cr, n_cb / 2, x / 2, y / 2, 4);
            let pred_cr = sub_block(bufs.pred_cr, n_cb / 2, x / 2, y / 2, 4);
            let (cr_lv, cr_rc) = code_tb(
                &src_cr,
                &pred_cr,
                4,
                qp_c,
                Component::Cr,
                PredMode::Inter,
                chroma_tools(2),
            );
            dist += ssd(&cb_rc, &src_cb) + ssd(&cr_rc, &src_cr);
            rate += rate_proxy(&cb_lv) + rate_proxy(&cr_lv) + 2;
            local.cb = cb_rc;
            local.cr = cr_rc;
            (cb_lv, cr_lv)
        } else {
            rate += 2;
            (Vec::new(), Vec::new())
        };
        let children: Box<[TuNode; 4]> = match children.try_into() {
            Ok(c) => Box::new(c),
            Err(_) => unreachable!("four children pushed"),
        };
        (
            TuNode::Split {
                children,
                cb: cb_lv,
                cr: cr_lv,
            },
            local,
            dist,
            rate,
        )
    };

    if split_forced {
        return split_eval();
    }
    if !split_allowed {
        return leaf_eval();
    }
    let (leaf_node, leaf_local, leaf_dist, leaf_rate) = leaf_eval();
    let (split_node, split_local, split_dist, split_rate) = split_eval();
    let leaf_cost = leaf_dist + lambda * (leaf_rate + 1);
    let split_cost = split_dist + lambda * (split_rate + 1);
    if leaf_cost <= split_cost {
        (leaf_node, leaf_local, leaf_dist, leaf_rate + 1)
    } else {
        (split_node, split_local, split_dist, split_rate + 1)
    }
}

/// One CU node's local reconstruction (luma + chroma half).
struct LocalRecon {
    y: Vec<u16>,
    cb: Vec<u16>,
    cr: Vec<u16>,
}

/// The CU-local buffers an inter residual quadtree reads.
struct InterCuBufs<'a> {
    n_cb: usize,
    src_y: &'a [i32],
    src_cb: &'a [i32],
    src_cr: &'a [i32],
    pred_y: &'a [i32],
    pred_cb: &'a [i32],
    pred_cr: &'a [i32],
    /// The CTB's RDOQ model.
    model: Option<&'a RdoqModel>,
}

/// One fully-evaluated inter candidate (pre-commit).
struct InterCand {
    kind: TreeCuKind,
    motions: Vec<PuMotion>,
    tree: Option<TuNode>,
    recon: LocalRecon,
    cost: u64,
}

/// Code the best CU at `(x0, y0)` size `1 << log2` on a P / B slice
/// INTO the state: the skip / merge / AMVP / two-PU / intra ladder.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn code_inter_cu(
    ctx: &SliceCtx<'_>,
    st: &mut EncState,
    x0: usize,
    y0: usize,
    log2: u32,
    depth: u32,
    ctb_qp: i32,
) -> CuCoded {
    debug_assert!(
        ctx.fmt.is_yuv420_8(),
        "the inter coder is 4:2:0 8-bit (the intra quadtree coder is format-general)"
    );
    let n = 1usize << log2;
    let qp_y = ctx.qp_y_prime(ctb_qp);
    let qp_c = ctx.qp_c_prime(ctb_qp);
    let lambda = ctx.lambda_of(ctb_qp);
    let lambda_me = crate::encoder::rate::motion_lambda(lambda);
    let mv_ctx = ctx.mv_ctx.expect("inter slice has motion context");
    let (cw, cx0, cy0) = (ctx.cw(), x0 / 2, y0 / 2);
    let (max_y, max_c) = (ctx.fmt.max_luma(), ctx.fmt.max_chroma());
    let src = [
        extract(ctx.src[0], ctx.width, x0, y0, n),
        extract(ctx.src[1], cw, cx0, cy0, n / 2),
        extract(ctx.src[2], cw, cx0, cy0, n / 2),
    ];

    let geom = PuGeometry {
        x_cb: x0,
        y_cb: y0,
        n_cb_s: n,
        x_pb: x0,
        y_pb: y0,
        n_pb_w: n,
        n_pb_h: n,
        part_mode: PartMode::Part2Nx2N,
        part_idx: 0,
    };
    let choose_ctx = PuChooseCtx {
        refs_l0: ctx.refs_l0,
        refs_l1: ctx.refs_l1,
        me_refs_l0: ctx.me_refs_l0,
        me_refs_l1: ctx.me_refs_l1,
        wp: ctx.wp,
        mv_ctx,
        lambda_me,
        b_slice: ctx.b_slice,
        two_sided: ctx.two_sided,
    };
    let max_depth = ctx.cfg.th_depth_inter;

    // Residual coder over a CU-wide prediction.
    let code_residual = |pred: &crate::inter_pred::InterPrediction,
                         part_2nx2n: bool|
     -> (Option<TuNode>, LocalRecon, u64, u64) {
        let bufs = InterCuBufs {
            n_cb: n,
            src_y: &src[0],
            src_cb: &src[1],
            src_cr: &src[2],
            pred_y: &pred.luma,
            pred_cb: &pred.cb,
            pred_cr: &pred.cr,
            model: st.rdoq_model.as_ref(),
        };
        let inter_split = max_depth == 0 && !part_2nx2n;
        let (node, local, dist, rate) = inter_rqt(
            ctx,
            &bufs,
            0,
            0,
            log2,
            0,
            max_depth,
            inter_split,
            qp_y,
            qp_c,
            lambda,
        );
        if node.any_cbf() {
            (Some(node), local, dist, rate)
        } else {
            // rqt_root_cbf == 0: prediction-only reconstruction.
            let local = LocalRecon {
                y: clip_to_samples(&pred.luma, max_y),
                cb: clip_to_samples(&pred.cb, max_c),
                cr: clip_to_samples(&pred.cr, max_c),
            };
            let dist = ssd(&local.y, &src[0]) + ssd(&local.cb, &src[1]) + ssd(&local.cr, &src[2]);
            (None, local, dist, 0)
        }
    };

    let mut cands: Vec<InterCand> = Vec::with_capacity(8);
    {
        let available = |x_nb: i32, y_nb: i32| -> bool {
            ctx.tiling.prediction_block_availability(
                x0 as u32,
                y0 as u32,
                n as u32,
                x0 as u32,
                y0 as u32,
                n as u32,
                n as u32,
                0,
                x_nb,
                y_nb,
                |_ctb_rs| 0,
                |x, y| {
                    if st.field.cell_at(x as usize, y as usize).is_intra {
                        MODE_INTRA
                    } else {
                        0
                    }
                },
            )
        };

        // ---- merge / skip candidates ----
        let mut merge_cands: Vec<(usize, PuMotion)> = Vec::with_capacity(MAX_MERGE);
        for idx in 0..MAX_MERGE {
            let m = resolve_pu_motion(&st.field, &geom, &merge_pu(idx), mv_ctx, &available);
            if !merge_cands.iter().any(|(_, prev)| *prev == m) {
                merge_cands.push((idx, m));
            }
        }
        let (best_merge_idx, best_merge_motion) = merge_cands
            .iter()
            .map(|&(idx, m)| {
                let pred =
                    predict_block_wp(ctx.refs_l0, ctx.refs_l1, x0, y0, n, n, &m, false, ctx.wp);
                (
                    crate::encoder::inter::sad(&pred.luma, &src[0]) + lambda_me * (idx as u64 + 1),
                    idx,
                    m,
                )
            })
            .min_by_key(|&(cost, idx, _)| (cost, idx))
            .map(|(_, idx, m)| (idx, m))
            .expect("merge list is never empty");

        // Skip.
        let merge_pred = predict_block_wp(
            ctx.refs_l0,
            ctx.refs_l1,
            x0,
            y0,
            n,
            n,
            &best_merge_motion,
            true,
            ctx.wp,
        );
        let skip_recon = LocalRecon {
            y: clip_to_samples(&merge_pred.luma, max_y),
            cb: clip_to_samples(&merge_pred.cb, max_c),
            cr: clip_to_samples(&merge_pred.cr, max_c),
        };
        let skip_dist = ssd(&skip_recon.y, &src[0])
            + ssd(&skip_recon.cb, &src[1])
            + ssd(&skip_recon.cr, &src[2]);
        cands.push(InterCand {
            kind: TreeCuKind::Skip {
                merge_idx: best_merge_idx,
            },
            motions: vec![best_merge_motion],
            tree: None,
            recon: skip_recon,
            cost: skip_dist + lambda * (best_merge_idx as u64 + 2),
        });

        // Merge + residual (legal only with some coded level).
        let (m_tree, m_recon, m_dist, m_rate) = code_residual(&merge_pred, true);
        if m_tree.is_some() {
            cands.push(InterCand {
                kind: TreeCuKind::Merge {
                    merge_idx: best_merge_idx,
                },
                motions: vec![best_merge_motion],
                tree: m_tree,
                recon: m_recon,
                cost: m_dist + lambda * (m_rate + best_merge_idx as u64 + 3),
            });
        }

        // AMVP.
        let (amvp_syntax, amvp_motion, amvp_rate, _) = amvp_search(
            &st.field,
            &geom,
            &available,
            &src[0],
            &choose_ctx,
            &merge_cands,
        );
        let amvp_pred = predict_block_wp(
            ctx.refs_l0,
            ctx.refs_l1,
            x0,
            y0,
            n,
            n,
            &amvp_motion,
            true,
            ctx.wp,
        );
        let (a_tree, a_recon, a_dist, a_rate) = code_residual(&amvp_pred, true);
        cands.push(InterCand {
            kind: TreeCuKind::Amvp { pu: amvp_syntax },
            motions: vec![amvp_motion],
            tree: a_tree,
            recon: a_recon,
            cost: a_dist + lambda * (a_rate + amvp_rate + 2),
        });
    }

    // ---- two-PU partitions (8x4 / 4x8 PUs at log2 == 3, uni-pred
    // only per §8.5.3.2.2 step 10 / Table 9-46; AMP shapes only above
    // MinCbSizeY per Table 9-45) ----
    {
        let mut parts = vec![PartMode::Part2NxN, PartMode::PartNx2N];
        if ctx.amp && log2 > ctx.cfg.min_cb_log2() {
            parts.extend([
                PartMode::Part2NxnU,
                PartMode::Part2NxnD,
                PartMode::PartNLx2N,
                PartMode::PartNRx2N,
            ]);
        }
        for part in parts {
            let rects = pu_partitions(x0, y0, n, part);
            let field_snap = st.field.snapshot_rect(x0, y0, n, n);
            let mut pus = [PuSyntax::Merge { merge_idx: 0 }; 2];
            let mut motions_r: Vec<PuMotion> = Vec::with_capacity(2);
            let mut pred_y = vec![0i32; n * n];
            let mut pred_cb = vec![0i32; (n / 2) * (n / 2)];
            let mut pred_cr = vec![0i32; (n / 2) * (n / 2)];
            let mut motion_rate = if part_is_amp(part) { 5u64 } else { 3u64 };
            for (k, r) in rects.iter().enumerate() {
                let g = PuGeometry {
                    x_cb: x0,
                    y_cb: y0,
                    n_cb_s: n,
                    x_pb: r.x_pb,
                    y_pb: r.y_pb,
                    n_pb_w: r.n_pb_w,
                    n_pb_h: r.n_pb_h,
                    part_mode: part,
                    part_idx: k as u32,
                };
                let avail_k = |x_nb: i32, y_nb: i32| -> bool {
                    ctx.tiling.prediction_block_availability(
                        x0 as u32,
                        y0 as u32,
                        n as u32,
                        x0 as u32,
                        y0 as u32,
                        n as u32,
                        n as u32,
                        0,
                        x_nb,
                        y_nb,
                        |_ctb_rs| 0,
                        |x, y| {
                            let (xu, yu) = (x as usize, y as usize);
                            let inside_cu =
                                (x0..x0 + n).contains(&xu) && (y0..y0 + n).contains(&yu);
                            if !inside_cu && st.field.cell_at(xu, yu).is_intra {
                                MODE_INTRA
                            } else {
                                0
                            }
                        },
                    )
                };
                let mut src_pu = Vec::with_capacity(r.n_pb_w * r.n_pb_h);
                for j in 0..r.n_pb_h {
                    for i in 0..r.n_pb_w {
                        src_pu.push(i32::from(ctx.src[0][(r.y_pb + j) * ctx.width + r.x_pb + i]));
                    }
                }
                let (pu_syntax, motion, rate_k) =
                    choose_pu(&st.field, &g, &avail_k, &src_pu, &choose_ctx);
                // eqs 8-80..8-85: PU1's derivation sees PU0's motion.
                let (p0, p1) = cell_pocs(ctx, &motion);
                st.field
                    .fill_rect(r.x_pb, r.y_pb, r.n_pb_w, r.n_pb_h, motion.to_cell(p0, p1));
                let p = predict_block_wp(
                    ctx.refs_l0,
                    ctx.refs_l1,
                    r.x_pb,
                    r.y_pb,
                    r.n_pb_w,
                    r.n_pb_h,
                    &motion,
                    true,
                    ctx.wp,
                );
                blit(
                    &mut pred_y,
                    n,
                    r.x_pb - x0,
                    r.y_pb - y0,
                    &p.luma,
                    r.n_pb_w,
                    r.n_pb_h,
                );
                blit(
                    &mut pred_cb,
                    n / 2,
                    (r.x_pb - x0) / 2,
                    (r.y_pb - y0) / 2,
                    &p.cb,
                    r.n_pb_w / 2,
                    r.n_pb_h / 2,
                );
                blit(
                    &mut pred_cr,
                    n / 2,
                    (r.x_pb - x0) / 2,
                    (r.y_pb - y0) / 2,
                    &p.cr,
                    r.n_pb_w / 2,
                    r.n_pb_h / 2,
                );
                pus[k] = pu_syntax;
                motions_r.push(motion);
                motion_rate += rate_k;
            }
            st.field.restore_rect(x0, y0, n, n, &field_snap);
            let pred = crate::inter_pred::InterPrediction {
                luma: pred_y,
                cb: pred_cb,
                cr: pred_cr,
            };
            let (tree, recon, dist, rate) = code_residual(&pred, false);
            cands.push(InterCand {
                kind: TreeCuKind::TwoPu { part, pus },
                motions: motions_r,
                tree,
                recon,
                cost: dist + lambda * (rate + motion_rate + 1),
            });
        }
    }

    // ---- intra fallback (2Nx2N, leaf TU shape kept simple) ----
    {
        let mode = if log2 == 6 {
            search_mode_64(ctx, st, x0, y0, None)
        } else {
            let marked = gather_luma_refs(ctx, &st.recon.y, x0, y0, n);
            if ctx.cfg.intra_rd == 0 {
                search_best_mode(&ctx.fmt, &marked, &src[0]).0
            } else {
                let mpm = mpm_list(ctx, st, x0, y0);
                rough_intra_modes(&ctx.fmt, &marked, &src[0], n, &mpm, lambda_me, 1)[0]
            }
        };
        // Predict + code the whole CU as its (possibly forced-split)
        // intra transform tree, on a scratch copy of the recon rect.
        let before = st.snapshot(ctx, x0, y0, n);
        let (tree, dist, rate) = intra_rqt(
            ctx,
            st,
            x0,
            y0,
            log2,
            0,
            ctx.cfg.th_depth_intra,
            mode,
            mode,
            qp_y,
            qp_c,
            lambda,
        );
        let recon = LocalRecon {
            y: rect_copy(&st.recon.y, ctx.width, x0, y0, n, n),
            cb: rect_copy(&st.recon.cb, cw, cx0, cy0, n / 2, n / 2),
            cr: rect_copy(&st.recon.cr, cw, cx0, cy0, n / 2, n / 2),
        };
        st.restore(ctx, &before);
        cands.push(InterCand {
            kind: TreeCuKind::Intra {
                modes: [mode; 4],
                nxn: false,
                chroma: [4; 4],
            },
            motions: Vec::new(),
            tree: Some(tree),
            recon,
            cost: dist + lambda * (rate + 9),
        });
    }

    let chosen = cands
        .into_iter()
        .min_by_key(|c| c.cost)
        .expect("at least the skip candidate");

    // ---- commit ----
    store(&mut st.recon.y, ctx.width, x0, y0, n, &chosen.recon.y);
    store(&mut st.recon.cb, cw, cx0, cy0, n / 2, &chosen.recon.cb);
    store(&mut st.recon.cr, cw, cx0, cy0, n / 2, &chosen.recon.cr);
    let is_skip = matches!(chosen.kind, TreeCuKind::Skip { .. });
    match &chosen.kind {
        TreeCuKind::Intra { modes, nxn, .. } => {
            st.field.fill_rect(
                x0,
                y0,
                n,
                n,
                MotionCell {
                    is_intra: true,
                    ref_poc_l0: i32::MIN,
                    ref_poc_l1: i32::MIN,
                    ..MotionCell::default()
                },
            );
            if *nxn {
                for (k, &(zx, zy)) in Z_OFFSETS.iter().enumerate() {
                    st.modes.record_intra_pb(
                        x0 + zx * n / 2,
                        y0 + zy * n / 2,
                        n / 2,
                        modes[k],
                        false,
                    );
                }
            } else {
                st.modes.record_intra_pb(x0, y0, n, modes[0], false);
            }
        }
        kind => {
            let part = match kind {
                TreeCuKind::TwoPu { part, .. } => *part,
                _ => PartMode::Part2Nx2N,
            };
            let rects = pu_partitions(x0, y0, n, part);
            for (r, m) in rects.iter().zip(chosen.motions.iter()) {
                let (p0, p1) = cell_pocs(ctx, m);
                st.field
                    .fill_rect(r.x_pb, r.y_pb, r.n_pb_w, r.n_pb_h, m.to_cell(p0, p1));
            }
            let cu_mode = if is_skip {
                CuPredMode::Skip
            } else {
                CuPredMode::Inter
            };
            st.modes.record_non_intra_cu(x0, y0, n, cu_mode);
            // §8.7.2.4: mark each luma TB leaf carrying a coefficient.
            if let Some(tree) = &chosen.tree {
                mark_nonzero(&mut st.field, tree, x0, y0, log2);
            }
        }
    }
    st.fill_cells(x0, y0, n, depth as i8, u8::from(is_skip));

    CuCoded {
        x0,
        y0,
        log2,
        kind: chosen.kind,
        motions: chosen.motions,
        tree: chosen.tree,
        cost: chosen.cost,
    }
}

/// The referenced-picture POC pair a motion cell stores (the
/// decoder's cells key the §8.7.2.4 comparisons on the referenced
/// picture's POC).
fn cell_pocs(ctx: &SliceCtx<'_>, m: &PuMotion) -> (i32, i32) {
    let mv_ctx = ctx.mv_ctx.expect("inter slice");
    (
        (mv_ctx.ref_poc)(0, m.ref_idx_l0),
        (mv_ctx.ref_poc)(1, m.ref_idx_l1),
    )
}

/// Stamp `has_nonzero_coeff` per luma TB leaf of a coded tree.
fn mark_nonzero(field: &mut MotionField, node: &TuNode, x: usize, y: usize, log2: u32) {
    match node {
        TuNode::Leaf { y: lv, .. } => {
            if lv.iter().any(|&v| v != 0) {
                field.mark_nonzero_coeff(x, y, 1 << log2, 1 << log2);
            }
        }
        TuNode::Split { children, .. } => {
            let half = 1usize << (log2 - 1);
            for (k, &(zx, zy)) in Z_OFFSETS.iter().enumerate() {
                mark_nonzero(field, &children[k], x + zx * half, y + zy * half, log2 - 1);
            }
        }
    }
}

// ---------------------------------------------------------------------
// The coding quadtree (pass 1)
// ---------------------------------------------------------------------

/// Code the quadtree node at `(x0, y0)` size `1 << log2`, committing
/// the winning decisions into the state. Returns the node + its cost
/// (including the node's own `split_cu_flag` when coded).
fn code_quadtree(
    ctx: &SliceCtx<'_>,
    st: &mut EncState,
    x0: usize,
    y0: usize,
    log2: u32,
    depth: u32,
    ctb_qp: i32,
) -> (CuNode, u64) {
    let n = 1usize << log2;
    let fits = x0 + n <= ctx.width && y0 + n <= ctx.height;
    let min_cb = ctx.cfg.min_cb_log2();

    let code_leaf = |ctx: &SliceCtx<'_>, st: &mut EncState| -> (CuNode, u64) {
        let cu = if ctx.intra_slice {
            code_intra_cu(ctx, st, x0, y0, log2, depth, ctb_qp)
        } else {
            code_inter_cu(ctx, st, x0, y0, log2, depth, ctb_qp)
        };
        let cost = cu.cost;
        (CuNode::Leaf(Box::new(cu)), cost)
    };

    let code_split = |ctx: &SliceCtx<'_>, st: &mut EncState| -> (CuNode, u64) {
        let half = n / 2;
        let mut children: [Option<CuNode>; 4] = [None, None, None, None];
        let mut cost = 0u64;
        for (k, &(zx, zy)) in Z_OFFSETS.iter().enumerate() {
            let (cx, cy) = (x0 + zx * half, y0 + zy * half);
            if cx < ctx.width && cy < ctx.height {
                let (node, c) = code_quadtree(ctx, st, cx, cy, log2 - 1, depth + 1, ctb_qp);
                children[k] = Some(node);
                cost += c;
            }
        }
        (CuNode::Split(Box::new(children)), cost)
    };

    if !fits {
        // §7.4.9.4: split inferred 1 (no flag).
        return code_split(ctx, st);
    }
    if log2 == min_cb {
        // Split inferred 0.
        return code_leaf(ctx, st);
    }
    // The flag is coded: RD-compare the unsplit CU vs the four
    // children (one flag bin either way).
    let lambda = ctx.lambda_of(ctb_qp);
    let before = st.snapshot(ctx, x0, y0, n);
    let (leaf_node, leaf_cost) = code_leaf(ctx, st);
    let leaf_cost = leaf_cost + lambda;
    // Skip-CU shortcut: a skip whose prediction is already tight will
    // not be beaten by four coded children (deterministic bound).
    if let CuNode::Leaf(cu) = &leaf_node {
        if matches!(cu.kind, TreeCuKind::Skip { .. }) && cu.cost <= (n * n) as u64 * 2 {
            return (leaf_node, leaf_cost);
        }
    }
    let after_leaf = st.snapshot(ctx, x0, y0, n);
    st.restore(ctx, &before);
    let (split_node, split_cost) = code_split(ctx, st);
    let split_cost = split_cost + lambda;
    if leaf_cost <= split_cost {
        st.restore(ctx, &after_leaf);
        (leaf_node, leaf_cost)
    } else {
        (split_node, split_cost)
    }
}

// ---------------------------------------------------------------------
// Pass 1.5 — per-CU QP thread + deblocking descriptors
// ---------------------------------------------------------------------

/// Walk the coded picture in coding order and mirror the decoder's
/// §8.6.1 QP derivation (one quantization group per CTB): each CU's
/// `QpY`, the per-4x4 QP map, and the coding-order deblocking
/// descriptors.
struct QpWalk {
    /// Per-4x4 `QpY` cells.
    cells: Vec<i8>,
    w_cells: usize,
    /// Per-CU descriptors in coding order.
    descs: Vec<DeblockCuDesc>,
}

fn qp_walk(ctx: &SliceCtx<'_>, plans: &[CuNode], ctb_qps: &[i32], aq_on: bool) -> QpWalk {
    let w_cells = ctx.width.div_ceil(4);
    let h_cells = ctx.height.div_ceil(4);
    let mut walk = QpWalk {
        cells: vec![0; w_cells * h_cells],
        w_cells,
        descs: Vec::new(),
    };
    let mut qp_prev = ctx.qp; // qPY_PREV (SliceQpY at the slice start)
    for (ctb_idx, plan) in plans.iter().enumerate() {
        let ctb_qp = ctb_qps[ctb_idx];
        // §8.6.1: the first QG of a tile / WPP row restarts at SliceQpY.
        if ctx.qp_prev_resets(ctb_idx) {
            qp_prev = ctx.qp;
        }
        // §7.3.8.14: the delta is transmitted in the first TU of the
        // CTB (== quantization group) with any cbf.
        let mut delta_coded = false;
        let mut last_qp = qp_prev;
        plan.for_each_cu(&mut |cu| {
            let has_cbf = cu.tree.as_ref().is_some_and(TuNode::any_cbf);
            if aq_on && has_cbf {
                delta_coded = true;
            }
            let qp_y = if aq_on {
                if delta_coded {
                    ctb_qp
                } else {
                    qp_prev
                }
            } else {
                ctx.qp
            };
            last_qp = qp_y;
            let n = 1usize << cu.log2;
            let bx1 = ((cu.x0 + n).min(w_cells * 4)).div_ceil(4);
            let by1 = ((cu.y0 + n).min(h_cells * 4)).div_ceil(4);
            for by in cu.y0 / 4..by1 {
                for bx in cu.x0 / 4..bx1 {
                    walk.cells[by * w_cells + bx] = qp_y as i8;
                }
            }
        });
        qp_prev = if aq_on { last_qp } else { ctx.qp };
        // Second sweep for the descriptors (the p-side scalars read the
        // now-final cells of earlier CUs).
        plan.for_each_cu(&mut |cu| {
            let qp_y = i32::from(walk.cells[(cu.y0 / 4) * w_cells + cu.x0 / 4]);
            let params = DeblockCuParams {
                qp_y,
                beta_offset_div2: 0,
                tc_offset_div2: 0,
                cb_qp_offset: ctx.chroma_qp_offset,
                cr_qp_offset: ctx.chroma_qp_offset,
                bit_depth_luma: ctx.fmt.bit_depth_luma,
                bit_depth_chroma: ctx.fmt.bit_depth_chroma,
                chroma_array_type: ctx.fmt.chroma_format_idc,
            };
            let qp_at = |x: i64, y: i64| -> i32 {
                if x < 0 || y < 0 {
                    qp_y
                } else {
                    i32::from(walk.cells[(y as usize / 4) * w_cells + x as usize / 4])
                }
            };
            walk.descs.push(DeblockCuDesc {
                cu: DeblockCu {
                    x_cb: cu.x0,
                    y_cb: cu.y0,
                    log2_cb_size: cu.log2,
                    params,
                    qp_y_p_left: qp_at(cu.x0 as i64 - 1, cu.y0 as i64),
                    qp_y_p_top: qp_at(cu.x0 as i64, cu.y0 as i64 - 1),
                },
                transform_split: cu
                    .tree
                    .as_ref()
                    .map_or(TransformSplit::Leaf, TuNode::to_transform_split),
                part_mode: match &cu.kind {
                    TreeCuKind::TwoPu { part, .. } => bin_part_mode(*part),
                    TreeCuKind::Intra { nxn: true, .. } => crate::binarization::PartMode::PartNxN,
                    _ => crate::binarization::PartMode::Part2Nx2N,
                },
                filter_left: cu.x0 > 0,
                filter_top: cu.y0 > 0,
            });
        });
        let _ = ctb_idx;
    }
    walk
}

// ---------------------------------------------------------------------
// Pass 2 — syntax emission
// ---------------------------------------------------------------------

/// Per-CTB quantization-group emission state.
struct QgState {
    /// `IsCuQpDeltaCoded` for the current CTB.
    coded: bool,
    /// The running `qPY_PREV` (last CU's `QpY`).
    qp_prev: i32,
    /// This CTB's target QP (slice QP + AQ delta).
    ctb_qp: i32,
    /// PPS `cu_qp_delta_enabled_flag`.
    enabled: bool,
}

struct Emitter<'a, 'b> {
    ctx: &'a SliceCtx<'a>,
    st: &'a EncState,
    w: &'b mut BitWriter,
    cabac: &'b mut CabacEncoder,
    ctxs: &'b mut SliceContexts,
}

impl Emitter<'_, '_> {
    fn emit_quadtree(
        &mut self,
        node: &CuNode,
        x0: usize,
        y0: usize,
        log2: u32,
        depth: u32,
        qg: &mut QgState,
    ) {
        let n = 1usize << log2;
        let fits = x0 + n <= self.ctx.width && y0 + n <= self.ctx.height;
        let split = matches!(node, CuNode::Split(_));
        if fits && log2 > self.ctx.cfg.min_cb_log2() {
            // §9.3.4.2.2 availability is the §6.4.1 one (slice / tile
            // boundaries cut it), not merely "already coded".
            let (l_depth, l_avail) = self.st.nb_ct_depth(x0, y0, Neighbour::Left);
            let (a_depth, a_avail) = self.st.nb_ct_depth(x0, y0, Neighbour::Above);
            let l_avail = l_avail && self.ctx.z_avail(x0, y0, x0 as i64 - 1, y0 as i64);
            let a_avail = a_avail && self.ctx.z_avail(x0, y0, x0 as i64, y0 as i64 - 1);
            let inc = split_cu_flag_ctx_inc(l_depth, l_avail, a_depth, a_avail, depth) as usize;
            self.cabac
                .encode_decision(self.w, &mut self.ctxs.split_cu_flag[inc], u8::from(split));
        }
        match node {
            CuNode::Split(children) => {
                let half = n / 2;
                for (k, &(zx, zy)) in Z_OFFSETS.iter().enumerate() {
                    if let Some(child) = &children[k] {
                        self.emit_quadtree(
                            child,
                            x0 + zx * half,
                            y0 + zy * half,
                            log2 - 1,
                            depth + 1,
                            qg,
                        );
                    }
                }
            }
            CuNode::Leaf(cu) => self.emit_cu(cu, depth, qg),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn emit_cu(&mut self, cu: &CuCoded, depth: u32, qg: &mut QgState) {
        let (x0, y0, n) = (cu.x0, cu.y0, 1usize << cu.log2);
        let min_cb = self.ctx.cfg.min_cb_log2();
        let is_skip = matches!(cu.kind, TreeCuKind::Skip { .. });
        if !self.ctx.intra_slice {
            let (l_skip, l_avail) = self.st.nb_skip(x0, y0, Neighbour::Left);
            let (a_skip, a_avail) = self.st.nb_skip(x0, y0, Neighbour::Above);
            let l_avail = l_avail && self.ctx.z_avail(x0, y0, x0 as i64 - 1, y0 as i64);
            let a_avail = a_avail && self.ctx.z_avail(x0, y0, x0 as i64, y0 as i64 - 1);
            let inc = cu_skip_flag_ctx_inc(l_skip, l_avail, a_skip, a_avail) as usize;
            self.cabac
                .encode_decision(self.w, &mut self.ctxs.cu_skip_flag[inc], u8::from(is_skip));
        }
        let n_l0 = self.ctx.refs_l0.len();
        let n_l1 = self.ctx.refs_l1.len();
        match &cu.kind {
            TreeCuKind::Skip { merge_idx } => {
                encode_merge_idx(self.w, self.cabac, self.ctxs, *merge_idx);
                // No transform tree; the CU's QpY threads through
                // unchanged (no delta transmitted here).
                return;
            }
            TreeCuKind::Merge { merge_idx } => {
                self.cabac
                    .encode_decision(self.w, &mut self.ctxs.pred_mode_flag[0], 0);
                self.emit_part_mode(PartMode::Part2Nx2N, cu.log2);
                encode_pu_syntax_at(
                    self.w,
                    self.cabac,
                    self.ctxs,
                    &PuSyntax::Merge {
                        merge_idx: *merge_idx,
                    },
                    self.ctx.b_slice,
                    n_l0,
                    n_l1,
                    depth,
                    (n, n),
                );
                // rqt_root_cbf inferred 1.
            }
            TreeCuKind::Amvp { pu } => {
                self.cabac
                    .encode_decision(self.w, &mut self.ctxs.pred_mode_flag[0], 0);
                self.emit_part_mode(PartMode::Part2Nx2N, cu.log2);
                encode_pu_syntax_at(
                    self.w,
                    self.cabac,
                    self.ctxs,
                    pu,
                    self.ctx.b_slice,
                    n_l0,
                    n_l1,
                    depth,
                    (n, n),
                );
                self.cabac.encode_decision(
                    self.w,
                    &mut self.ctxs.rqt_root_cbf[0],
                    u8::from(cu.tree.is_some()),
                );
            }
            TreeCuKind::TwoPu { part, pus } => {
                self.cabac
                    .encode_decision(self.w, &mut self.ctxs.pred_mode_flag[0], 0);
                self.emit_part_mode(*part, cu.log2);
                let rects = pu_partitions(x0, y0, n, *part);
                for (pu, r) in pus.iter().zip(rects.iter()) {
                    encode_pu_syntax_at(
                        self.w,
                        self.cabac,
                        self.ctxs,
                        pu,
                        self.ctx.b_slice,
                        n_l0,
                        n_l1,
                        depth,
                        (r.n_pb_w, r.n_pb_h),
                    );
                }
                self.cabac.encode_decision(
                    self.w,
                    &mut self.ctxs.rqt_root_cbf[0],
                    u8::from(cu.tree.is_some()),
                );
            }
            TreeCuKind::Intra { modes, nxn, chroma } => {
                if !self.ctx.intra_slice {
                    self.cabac
                        .encode_decision(self.w, &mut self.ctxs.pred_mode_flag[0], 1);
                }
                if cu.log2 == min_cb {
                    // Intra part_mode at MinCb: "1" 2Nx2N, "0" NxN.
                    self.cabac.encode_decision(
                        self.w,
                        &mut self.ctxs.part_mode[0],
                        u8::from(!*nxn),
                    );
                }
                // §7.3.8.5 two-loop luma mode group.
                let n_pb = if *nxn { 4 } else { 1 };
                let pb_size = if *nxn { n / 2 } else { n };
                let pb_pos =
                    |k: usize| (x0 + Z_OFFSETS[k].0 * pb_size, y0 + Z_OFFSETS[k].1 * pb_size);
                let mut selections: Vec<Option<usize>> = Vec::with_capacity(n_pb);
                for k in 0..n_pb {
                    let (px, py) = pb_pos(k);
                    let avail_l = self.ctx.z_avail(px, py, px as i64 - 1, py as i64);
                    let avail_a = self.ctx.z_avail(px, py, px as i64, py as i64 - 1);
                    let cand_a =
                        self.st
                            .modes
                            .cand_intra_pred_mode(px, py, Neighbour::Left, avail_l);
                    let cand_b =
                        self.st
                            .modes
                            .cand_intra_pred_mode(px, py, Neighbour::Above, avail_a);
                    let list = intra_luma_cand_mode_list(cand_a, cand_b);
                    selections.push(list.iter().position(|&m| m == modes[k]));
                    self.cabac.encode_decision(
                        self.w,
                        &mut self.ctxs.prev_intra_luma_pred_flag[0],
                        u8::from(selections[k].is_some()),
                    );
                }
                for (k, sel) in selections.iter().enumerate() {
                    match *sel {
                        Some(0) => self.cabac.encode_bypass(self.w, 0),
                        Some(1) => {
                            self.cabac.encode_bypass(self.w, 1);
                            self.cabac.encode_bypass(self.w, 0);
                        }
                        Some(_) => {
                            self.cabac.encode_bypass(self.w, 1);
                            self.cabac.encode_bypass(self.w, 1);
                        }
                        None => {
                            let (px, py) = pb_pos(k);
                            let avail_l = self.ctx.z_avail(px, py, px as i64 - 1, py as i64);
                            let avail_a = self.ctx.z_avail(px, py, px as i64, py as i64 - 1);
                            let cand_a = self.st.modes.cand_intra_pred_mode(
                                px,
                                py,
                                Neighbour::Left,
                                avail_l,
                            );
                            let cand_b = self.st.modes.cand_intra_pred_mode(
                                px,
                                py,
                                Neighbour::Above,
                                avail_a,
                            );
                            let list = intra_luma_cand_mode_list(cand_a, cand_b);
                            let mut rem = u32::from(modes[k]);
                            for &c in &list {
                                if u32::from(modes[k]) > u32::from(c) {
                                    rem -= 1;
                                }
                            }
                            self.cabac.encode_bypass_bits(self.w, rem, 5);
                        }
                    }
                }
                // intra_chroma_pred_mode (Table 9-46): "0" for 4
                // (derived from luma), else "1" + two bypass bins —
                // §7.3.8.5: one per PB at 4:4:4, one per CU otherwise,
                // none for monochrome.
                let n_chroma = match self.ctx.fmt.chroma_format_idc {
                    0 => 0,
                    3 => n_pb,
                    _ => 1,
                };
                for &c in chroma.iter().take(n_chroma) {
                    if c == 4 {
                        self.cabac.encode_decision(
                            self.w,
                            &mut self.ctxs.intra_chroma_pred_mode[0],
                            0,
                        );
                    } else {
                        self.cabac.encode_decision(
                            self.w,
                            &mut self.ctxs.intra_chroma_pred_mode[0],
                            1,
                        );
                        self.cabac.encode_bypass_bits(self.w, u32::from(c), 2);
                    }
                }
            }
        }
        // ---- transform tree ----
        let cu_is_intra = matches!(cu.kind, TreeCuKind::Intra { .. });
        let (intra_split, modes4, modes_c) = match &cu.kind {
            TreeCuKind::Intra { modes, nxn, chroma } => {
                // §8.4.3 IntraPredModeC per PB from the coded chroma
                // index and the PB's luma mode (the chroma TBs' scan;
                // the 4:2:2 Table 8-3 remap applied) — one chroma PB
                // derived from PB 0 unless 4:4:4 PART_NxN.
                let per_pb = self.ctx.fmt.chroma_format_idc == 3 && *nxn;
                let mut mc = [0u8; 4];
                for (k, m) in mc.iter_mut().enumerate() {
                    let (idx, luma) = if per_pb {
                        (chroma[k], modes[k])
                    } else {
                        (chroma[0], modes[0])
                    };
                    *m = crate::binarization::derive_intra_pred_mode_c(
                        idx,
                        luma,
                        self.ctx.fmt.chroma_format_idc == 2,
                    );
                }
                (*nxn, *modes, mc)
            }
            _ => (false, [0u8; 4], [0u8; 4]),
        };
        if let Some(tree) = &cu.tree {
            let max_depth = if cu_is_intra {
                self.ctx.cfg.th_depth_intra + u32::from(intra_split)
            } else {
                self.ctx.cfg.th_depth_inter
            };
            let inter_split = !cu_is_intra
                && self.ctx.cfg.th_depth_inter == 0
                && !matches!(
                    &cu.kind,
                    TreeCuKind::Amvp { .. } | TreeCuKind::Merge { .. } | TreeCuKind::Skip { .. }
                );
            let tt = TtCtx {
                cu,
                cu_is_intra,
                intra_split,
                modes4,
                modes_c,
                max_depth,
                inter_split,
            };
            self.emit_transform_tree(&tt, tree, x0, y0, cu.log2, 0, true, true, [false; 2], qg);
        }
    }

    /// Write-side §9.3.3.7 `part_mode` (inter forms; the intra MinCb
    /// bin is written inline by the caller).
    fn emit_part_mode(&mut self, part: PartMode, log2: u32) {
        let min_cb = self.ctx.cfg.min_cb_log2();
        let two_nx2n = part == PartMode::Part2Nx2N;
        self.cabac
            .encode_decision(self.w, &mut self.ctxs.part_mode[0], u8::from(two_nx2n));
        if two_nx2n {
            return;
        }
        let horizontal = part_is_horizontal(part);
        self.cabac
            .encode_decision(self.w, &mut self.ctxs.part_mode[1], u8::from(horizontal));
        if log2 > min_cb {
            if self.ctx.amp {
                let amp_shape = part_is_amp(part);
                self.cabac.encode_decision(
                    self.w,
                    &mut self.ctxs.part_mode[3],
                    u8::from(!amp_shape),
                );
                if amp_shape {
                    let second = matches!(part, PartMode::Part2NxnD | PartMode::PartNRx2N);
                    self.cabac.encode_bypass(self.w, u8::from(second));
                }
            }
            // !amp: two bins total ("01" 2NxN / "00" Nx2N).
        } else {
            // log2 == MinCb == 3: two bins total (no NxN inter CUs).
            debug_assert_eq!(min_cb, 3, "quadtree streams keep MinCbLog2SizeY == 3");
        }
    }

    /// Emit one §7.3.8.8 transform-tree node. `parent_cbf_cb` /
    /// `parent_cbf_cr` are the parent node's flags (`true` at the
    /// root per the depth-0 read rule); `parent_lower` the parent's
    /// `ChromaArrayType == 2` lower-block companions (a 4x4 leaf reads
    /// them for its deferred chroma).
    #[allow(clippy::too_many_arguments)]
    fn emit_transform_tree(
        &mut self,
        tt: &TtCtx<'_>,
        node: &TuNode,
        x: usize,
        y: usize,
        log2: u32,
        depth: u32,
        parent_cbf_cb: bool,
        parent_cbf_cr: bool,
        parent_lower: [bool; 2],
        qg: &mut QgState,
    ) {
        let fmt = self.ctx.fmt;
        let max_tb = self.ctx.cfg.max_tb_log2();
        let split = matches!(node, TuNode::Split { .. });
        // §7.3.8.8 split_transform_flag presence gate.
        if log2 <= max_tb && log2 > 2 && depth < tt.max_depth && !(tt.intra_split && depth == 0) {
            let inc = split_transform_flag_ctx_inc(log2) as usize;
            self.cabac.encode_decision(
                self.w,
                &mut self.ctxs.split_transform_flag[inc],
                u8::from(split),
            );
        } else {
            // Inferred: must match.
            let inferred =
                log2 > max_tb || (tt.intra_split && depth == 0) || (tt.inter_split && depth == 0);
            debug_assert_eq!(split, inferred, "unsignallable transform tree");
        }
        // Chroma cbf block, present per `( log2 > 2 && ChromaArrayType
        // != 0 ) || ChromaArrayType == 3`, each component gated on the
        // parent. The node-level flag is the OR over its subtree; at
        // 4:2:2 a leaf (or the log2 == 3 split parent of deferred
        // chroma) codes the two stacked blocks' flags instead.
        let cbf_cb = node.cbf_cb();
        let cbf_cr = node.cbf_cr();
        let blocks = fmt.chroma_blocks();
        let (own_cb, own_cr) = node.own_chroma();
        let halves_cb = TuNode::cbf_halves(own_cb, blocks);
        let halves_cr = TuNode::cbf_halves(own_cr, blocks);
        let two_flags = fmt.chroma_format_idc == 2 && (!split || log2 == 3);
        let mut lower = [false; 2];
        if fmt.chroma_cbf_present(log2) {
            if depth == 0 || parent_cbf_cb {
                let first = if two_flags { halves_cb[0] } else { cbf_cb };
                self.cabac.encode_decision(
                    self.w,
                    &mut self.ctxs.cbf_chroma[cbf_cb_ctx_inc(depth) as usize],
                    u8::from(first),
                );
                if two_flags {
                    self.cabac.encode_decision(
                        self.w,
                        &mut self.ctxs.cbf_chroma[cbf_cb_ctx_inc(depth) as usize],
                        u8::from(halves_cb[1]),
                    );
                    lower[0] = halves_cb[1];
                }
            }
            if depth == 0 || parent_cbf_cr {
                let first = if two_flags { halves_cr[0] } else { cbf_cr };
                self.cabac.encode_decision(
                    self.w,
                    &mut self.ctxs.cbf_chroma[cbf_cr_ctx_inc(depth) as usize],
                    u8::from(first),
                );
                if two_flags {
                    self.cabac.encode_decision(
                        self.w,
                        &mut self.ctxs.cbf_chroma[cbf_cr_ctx_inc(depth) as usize],
                        u8::from(halves_cr[1]),
                    );
                    lower[1] = halves_cr[1];
                }
            }
        }
        match node {
            TuNode::Split { children, cb, cr } => {
                let half = 1usize << (log2 - 1);
                // The gate the children read is the node-level flag
                // `cbf_cb[ xBase ][ yBase ]` — at a log2 == 3 split that
                // is the upper deferred block's flag (its lower
                // companion rides along for the blkIdx == 3 leaf).
                let (gate_cb, gate_cr) = if two_flags {
                    (halves_cb[0], halves_cr[0])
                } else {
                    (cbf_cb, cbf_cr)
                };
                for (k, &(zx, zy)) in Z_OFFSETS.iter().enumerate() {
                    self.emit_transform_tree(
                        tt,
                        &children[k],
                        x + zx * half,
                        y + zy * half,
                        log2 - 1,
                        depth + 1,
                        gate_cb,
                        gate_cr,
                        lower,
                        qg,
                    );
                    // The blkIdx == 3 deferred chroma rides inside the
                    // last child's transform_unit — emitted right after
                    // its luma residual, below.
                    if log2 == 3 && k == 3 && fmt.has_chroma() && fmt.chroma_format_idc != 3 {
                        self.emit_deferred_chroma(tt, cb, cr, halves_cb, halves_cr, log2);
                    }
                }
            }
            TuNode::Leaf { y: y_lv, cb, cr } => {
                let cbf_luma = TuNode::any_nonzero(y_lv);
                // Chroma coded in place at this node, else (a 4x4 luma
                // leaf of a 4:2:0 / 4:2:2 tree) the parent's flags.
                let in_place = fmt.chroma_in_place(log2);
                let any_chroma = if in_place {
                    cbf_cb || cbf_cr
                } else {
                    parent_cbf_cb || parent_cbf_cr || parent_lower[0] || parent_lower[1]
                };
                let cbf_luma_present = tt.cu_is_intra || depth != 0 || any_chroma;
                if cbf_luma_present {
                    self.cabac.encode_decision(
                        self.w,
                        &mut self.ctxs.cbf_luma[cbf_luma_ctx_inc(depth) as usize],
                        u8::from(cbf_luma),
                    );
                } else {
                    debug_assert!(cbf_luma, "an all-zero root inter TU must not be coded");
                }
                // ---- transform_unit ----
                if cbf_luma || any_chroma {
                    self.emit_delta_qp(qg);
                    if cbf_luma {
                        let pb_idx = tt.pb_idx(x, y);
                        let mode = tt.modes4[pb_idx];
                        self.emit_residual(y_lv, log2, 0, tt.cu_is_intra, mode);
                    }
                    if in_place {
                        let mode_c = tt.modes_c[tt.pb_idx(x, y)];
                        let log2_c = fmt.log2_chroma_tb(log2);
                        self.emit_chroma_blocks(cb, halves_cb, log2_c, 1, tt.cu_is_intra, mode_c);
                        self.emit_chroma_blocks(cr, halves_cr, log2_c, 2, tt.cu_is_intra, mode_c);
                    }
                    // Otherwise chroma is deferred to blkIdx 3, handled
                    // by the parent (emit_deferred_chroma).
                }
            }
        }
    }

    /// The stacked chroma blocks of one component at a transform unit
    /// (`residual_coding( )` per coded block, upper first).
    fn emit_chroma_blocks(
        &mut self,
        levels: &[i32],
        cbf: [bool; 2],
        log2_c: u32,
        c_idx: u8,
        cu_is_intra: bool,
        mode_c: u8,
    ) {
        let blocks = self.ctx.fmt.chroma_blocks();
        let len = 1usize << (2 * log2_c);
        for v in 0..blocks {
            if cbf[v] {
                self.emit_residual(
                    &levels[v * len..(v + 1) * len],
                    log2_c,
                    c_idx,
                    cu_is_intra,
                    mode_c,
                );
            }
        }
    }

    /// The §7.3.8.10 `blkIdx == 3` deferred-chroma tail (invoked right
    /// after the fourth 4x4 luma child of a `log2TrafoSize == 3`
    /// split node): the parent's 4x4 chroma blocks, gated per block.
    fn emit_deferred_chroma(
        &mut self,
        tt: &TtCtx<'_>,
        cb: &[i32],
        cr: &[i32],
        cbf_cb: [bool; 2],
        cbf_cr: [bool; 2],
        parent_log2: u32,
    ) {
        // The last luma leaf's transform_unit fired (and consumed
        // delta_qp) iff its own cbf_luma or the parent chroma was set;
        // the deferred chroma is coded in that same transform_unit.
        let mode_c = tt.modes_c[0];
        let log2_c = self.ctx.fmt.log2_chroma_tb(parent_log2);
        self.emit_chroma_blocks(cb, cbf_cb, log2_c, 1, tt.cu_is_intra, mode_c);
        self.emit_chroma_blocks(cr, cbf_cr, log2_c, 2, tt.cu_is_intra, mode_c);
    }

    /// §7.3.8.14 `delta_qp( )`, once per quantization group.
    fn emit_delta_qp(&mut self, qg: &mut QgState) {
        if qg.enabled && !qg.coded {
            qg.coded = true;
            let delta = qg.ctb_qp - qg.qp_prev;
            encode_cu_qp_delta(self.w, self.cabac, self.ctxs, delta);
        }
    }

    fn emit_residual(&mut self, levels: &[i32], log2: u32, c_idx: u8, cu_is_intra: bool, mode: u8) {
        let params = ResidualCodingParams {
            log2_trafo_size: log2,
            is_chroma: c_idx != 0,
            scan_idx: residual_coding_scan_idx(
                cu_is_intra,
                log2,
                c_idx,
                self.ctx.fmt.chroma_format_idc,
                u32::from(mode),
            ),
            sign_data_hiding_enabled_flag: self.ctx.cfg.sign_hiding,
            sign_hidden_suppressed: false,
            transform_skip_sig_ctx: false,
            persistent_rice_adaptation_enabled_flag: false,
            cabac_bypass_alignment_enabled_flag: false,
            extended_precision_processing_flag: false,
            bit_depth: self.ctx.fmt.bit_depth(c_idx),
            rice_stat_transform_skip: false,
        };
        encode_residual_coding(self.w, self.cabac, &mut self.ctxs.residual, &params, levels)
            .expect("validated levels");
    }
}

/// Per-CU transform-tree emission context.
struct TtCtx<'a> {
    cu: &'a CuCoded,
    cu_is_intra: bool,
    intra_split: bool,
    modes4: [u8; 4],
    /// `IntraPredModeC` per PB (intra CUs): the chroma TBs' scan
    /// selector (all equal outside 4:4:4 `PART_NxN`).
    modes_c: [u8; 4],
    max_depth: u32,
    inter_split: bool,
}

impl TtCtx<'_> {
    /// The prediction block a transform block at luma `(x, y)` lies in
    /// (`PART_NxN`: the quadrant; else 0).
    fn pb_idx(&self, x: usize, y: usize) -> usize {
        if self.intra_split {
            let half = 1usize << (self.cu.log2 - 1);
            (usize::from(y.wrapping_sub(self.cu.y0) >= half) << 1)
                | usize::from(x.wrapping_sub(self.cu.x0) >= half)
        } else {
            0
        }
    }
}

// ---------------------------------------------------------------------
// Substream entropy coder (tiles / WPP subsets)
// ---------------------------------------------------------------------

/// The slice's arithmetic coder over its §7.3.8.1 subsets: one
/// [`CabacEncoder`] + [`SliceContexts`] pair re-armed at every tile
/// start (§9.3.2.2 initialization) and, under WPP, synchronized from
/// the stored second-CTB state at every CTB row start of a tile
/// (§9.3.2.5), the bytes of each subset kept apart for the §7.3.6.1
/// entry points. Shared by the shadow emission (rate feedback / RDOQ
/// context states) and the final slice-data emission so both see the
/// identical context evolution.
struct EntropyCoder {
    init_type: u8,
    slice_qp: i32,
    cabac: CabacEncoder,
    ctxs: SliceContexts,
    w: BitWriter,
    /// `TableStateIdxWpp` / `TableMpsValWpp` storage.
    wpp_store: Option<SliceContexts>,
    subsets: Vec<Vec<u8>>,
    /// Bits of the finished subsets.
    bits_done: u64,
}

impl EntropyCoder {
    fn new(init_type: u8, slice_qp: i32) -> Self {
        Self {
            init_type,
            slice_qp,
            cabac: CabacEncoder::new(),
            ctxs: SliceContexts::init(init_type, slice_qp),
            w: BitWriter::new(),
            wpp_store: None,
            subsets: Vec::new(),
            bits_done: 0,
        }
    }

    /// §9.3.2.1 at the CTB at tile-scan index `ts`: re-initialize at a
    /// tile start, synchronize (or re-initialize) at a WPP row start.
    fn begin_ctb(&mut self, ctx: &SliceCtx<'_>, ts: usize) {
        if ts == 0 {
            return;
        }
        if ctx.tile_start(ts) {
            self.ctxs = SliceContexts::init(self.init_type, self.slice_qp);
            self.wpp_store = None;
        } else if ctx.cfg.wpp && ctx.row_start_in_tile(ts) {
            self.ctxs = match (&self.wpp_store, ctx.wpp_sync_available(ts)) {
                (Some(stored), true) => stored.clone(),
                _ => SliceContexts::init(self.init_type, self.slice_qp),
            };
        }
    }

    /// After the CTU syntax of the CTB at tile-scan index `ts`: the
    /// WPP storage, `end_of_slice_segment_flag`, and the subset end
    /// (`end_of_subset_one_bit` + byte alignment, fresh engine).
    fn end_ctb(&mut self, ctx: &SliceCtx<'_>, ts: usize, last: bool) {
        if ctx.cfg.wpp && ctx.wpp_store_after(ts) {
            self.wpp_store = Some(self.ctxs.clone());
        }
        self.cabac.encode_terminate(&mut self.w, u8::from(last));
        if last {
            self.w.align_zero();
            self.close_subset();
        } else if ctx.subset_ends_after(ts) {
            self.cabac.encode_terminate(&mut self.w, 1); // end_of_subset_one_bit
            self.w.align_zero(); // byte_alignment( )
            self.close_subset();
        }
    }

    fn close_subset(&mut self) {
        let bytes = std::mem::take(&mut self.w).finish();
        self.bits_done += 8 * bytes.len() as u64;
        self.subsets.push(bytes);
        self.cabac = CabacEncoder::new();
    }

    /// Bits emitted so far (finished subsets + the open one).
    fn bit_len(&self) -> u64 {
        self.bits_done + self.w.bit_len() as u64
    }
}

// ---------------------------------------------------------------------
// Whole-picture drivers
// ---------------------------------------------------------------------

/// The coded picture a slice driver wraps: the slice-data RBSP tail
/// (everything after the slice header), the filtered reconstruction,
/// the loop-filter elections, and the per-CU stats.
struct CodedPicture {
    /// Per CTB in CODING (tile-scan) order.
    plans: Vec<CuNode>,
    /// The QP each CTB was coded at (slice QP + AQ + rate feedback),
    /// coding order.
    ctb_qps: Vec<i32>,
    /// The pass-1 cell / field state (pass 2 reads the ctxInc cells).
    st: EncState,
    recon: ReconPlanes,
    stats: FrameStats,
    deblock_on: bool,
    beta_offset_div2: i32,
    tc_offset_div2: i32,
    sao_luma: bool,
    sao_chroma: bool,
    sao_ctbs: Vec<SaoCtbParams>,
}

/// One tile's pass-1 job: its tile-scan range and luma rectangle.
struct TileJob {
    ts_start: usize,
    ts_end: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
}

/// The §6.5.1 tile grid as pass-1 jobs (tile-scan order).
fn tile_jobs(ctx: &SliceCtx<'_>) -> Vec<TileJob> {
    let ctb = 1usize << ctx.cfg.ctb_log2;
    let ctbs_x = ctx.ctbs_x();
    let col_bd = ctx.tiling.col_bd();
    let row_bd = ctx.tiling.row_bd();
    let mut jobs = Vec::new();
    for r in 0..row_bd.len() - 1 {
        for c in 0..col_bd.len() - 1 {
            let (cx0, cx1) = (col_bd[c] as usize, col_bd[c + 1] as usize);
            let (cy0, cy1) = (row_bd[r] as usize, row_bd[r + 1] as usize);
            let rs = cy0 * ctbs_x + cx0;
            let ts_start = ctx.tiling.ctb_addr_rs_to_ts(rs as u32) as usize;
            let x0 = cx0 * ctb;
            let y0 = cy0 * ctb;
            jobs.push(TileJob {
                ts_start,
                ts_end: ts_start + (cx1 - cx0) * (cy1 - cy0),
                x0,
                y0,
                w: (cx1 * ctb).min(ctx.width) - x0,
                h: (cy1 * ctb).min(ctx.height) - y0,
            });
        }
    }
    jobs
}

/// One tile's pass-1 output: the state it decided into (only its
/// rectangle is meaningful) and its CTB plans in tile-scan order.
struct TileOut {
    st: EncState,
    plans: Vec<CuNode>,
    ctb_qps: Vec<i32>,
}

/// Pass 1 over one tile: the CTU decisions in tile-scan order with the
/// tile's own shadow entropy coder (§9.3.2.1 re-initializes at the
/// tile start, so the shadow's context states — and with them RDOQ —
/// never depend on other tiles) and, under CTU-level rate feedback,
/// the tile's pro-rata share of the frame budget.
fn code_tile(ctx: &SliceCtx<'_>, job: &TileJob, slice_type_raw: u8, cu_qp_delta: bool) -> TileOut {
    let mut st = EncState::new(&ctx.fmt, ctx.width, ctx.height, ctx.cfg.ctb_log2);
    let ctbs_x = ctx.ctbs_x();
    let n_ctbs = ctbs_x * ctx.ctbs_y();
    let n_tile = job.ts_end - job.ts_start;
    let ctb = 1usize << ctx.cfg.ctb_log2;
    let mut plans: Vec<CuNode> = Vec::with_capacity(n_tile);
    let mut ctb_qps: Vec<i32> = Vec::with_capacity(n_tile);
    // CTU-level rate feedback: a shadow CABAC emission of every coded
    // CTB (no SAO syntax — that is elected after the filters) tracks
    // the tile's running coded size against its pro-rata budget.
    // RDOQ prices its bins at the same shadow's residual context
    // states (refreshed at every CTB start). The shadow walks the
    // same WPP subset structure as the final emission.
    let tile_budget = ctx
        .ctu_rc
        .map(|b| b.saturating_mul(n_tile as u64) / n_ctbs as u64);
    let mut shadow: Option<(EntropyCoder, i32)> =
        (ctx.ctu_rc.is_some() || ctx.cfg.rdoq).then(|| {
            (
                EntropyCoder::new(init_type(slice_type_raw, false), ctx.qp),
                ctx.qp,
            )
        });
    for ts in job.ts_start..job.ts_end {
        let local = ts - job.ts_start;
        let rs = ctx.ctb_rs(ts);
        let x0 = (rs % ctbs_x) * ctb;
        let y0 = (rs / ctbs_x) * ctb;
        if let Some((coder, qp_prev)) = shadow.as_mut() {
            coder.begin_ctb(ctx, ts);
            if ctx.qp_prev_resets(ts) {
                *qp_prev = ctx.qp;
            }
        }
        let rc_adj = match (tile_budget, &shadow) {
            (Some(budget), Some((coder, _))) if local > 0 => {
                let so_far = coder.bit_len();
                let expected = budget.saturating_mul(local as u64) / n_tile as u64;
                // Ratio in Q8: over-spending raises the QP, under-
                // spending lowers it, in bounded steps (±3).
                match so_far.saturating_mul(256).checked_div(expected) {
                    None => 0,
                    Some(r) if r > 512 => 3,
                    Some(r) if r > 384 => 2,
                    Some(r) if r > 320 => 1,
                    Some(r) if r < 128 => -3,
                    Some(r) if r < 192 => -2,
                    Some(r) if r < 224 => -1,
                    Some(_) => 0,
                }
            }
            _ => 0,
        };
        let ctb_qp = (ctx.qp + ctx.aq_deltas[rs] + rc_adj).clamp(-ctx.fmt.qp_bd_offset_y(), 51);
        if ctx.cfg.rdoq {
            st.rdoq_model = shadow.as_ref().map(|(coder, _)| RdoqModel {
                contexts: coder.ctxs.residual.clone(),
            });
        }
        let (node, _cost) = code_quadtree(ctx, &mut st, x0, y0, ctx.cfg.ctb_log2, 0, ctb_qp);
        if let Some((coder, qp_prev)) = shadow.as_mut() {
            let mut qg = QgState {
                coded: false,
                qp_prev: *qp_prev,
                ctb_qp,
                enabled: cu_qp_delta,
            };
            let mut em = Emitter {
                ctx,
                st: &st,
                w: &mut coder.w,
                cabac: &mut coder.cabac,
                ctxs: &mut coder.ctxs,
            };
            em.emit_quadtree(&node, x0, y0, ctx.cfg.ctb_log2, 0, &mut qg);
            if qg.coded {
                *qp_prev = ctb_qp;
            }
            coder.end_ctb(ctx, ts, ts + 1 == job.ts_end);
        }
        plans.push(node);
        ctb_qps.push(ctb_qp);
    }
    TileOut { st, plans, ctb_qps }
}

/// WPP-parallel pass 1 over one tile: its CTB rows are decided by up
/// to `workers` threads in a wavefront — the CTB at column `c` of a
/// row starts once the row above has finished column `c + 1` (every
/// decision reads at most the left, above-left, above and above-right
/// CTBs: intra reference samples reach one 32-sample TB past the
/// current CTB, merge candidates one PU), each worker deciding into
/// its own state after pulling that above-row neighbourhood from the
/// shared picture state and publishing each finished CTB back. The
/// RDOQ shadow coder of a row starts from the §9.3.2.2 storage the
/// row above made after its second CTB — exactly the serial
/// wavefront's context sequence, so the decisions (and bytes) equal
/// the serial pass for any worker count.
/// One wavefront row's decided plans and CTB QPs.
type RowOut = (Vec<CuNode>, Vec<i32>);

fn code_tile_wpp(
    ctx: &SliceCtx<'_>,
    job: &TileJob,
    slice_type_raw: u8,
    cu_qp_delta: bool,
    workers: usize,
) -> TileOut {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Condvar, Mutex};
    let ctb = 1usize << ctx.cfg.ctb_log2;
    let ctbs_x = ctx.ctbs_x();
    let (col0, cols) = (job.x0 / ctb, job.w.div_ceil(ctb));
    let (row0, rows) = (job.y0 / ctb, job.h.div_ceil(ctb));
    let master = Mutex::new(EncState::new(
        &ctx.fmt,
        ctx.width,
        ctx.height,
        ctx.cfg.ctb_log2,
    ));
    let progress: Vec<AtomicUsize> = (0..rows).map(|_| AtomicUsize::new(0)).collect();
    let stored: Vec<Mutex<Option<SliceContexts>>> = (0..rows).map(|_| Mutex::new(None)).collect();
    let results: Vec<Mutex<Option<RowOut>>> = (0..rows).map(|_| Mutex::new(None)).collect();
    let wake = (Mutex::new(()), Condvar::new());
    let next_row = AtomicUsize::new(0);
    // The luma rectangle of the CTB at (row, col) of the tile.
    let rect = |r: usize, c: usize| -> (usize, usize, usize, usize) {
        let (x0, y0) = ((col0 + c) * ctb, (row0 + r) * ctb);
        (
            x0,
            y0,
            (ctx.width - x0).min(ctb),
            (ctx.height - y0).min(ctb),
        )
    };
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                let mut st = EncState::new(&ctx.fmt, ctx.width, ctx.height, ctx.cfg.ctb_log2);
                loop {
                    let r = next_row.fetch_add(1, Ordering::SeqCst);
                    if r >= rows {
                        break;
                    }
                    let mut plans = Vec::with_capacity(cols);
                    let mut ctb_qps = Vec::with_capacity(cols);
                    let mut shadow: Option<(EntropyCoder, i32)> = None;
                    for c in 0..cols {
                        if r > 0 {
                            let need = (c + 2).min(cols);
                            let (lock, cvar) = &wake;
                            let mut guard = lock.lock().expect("wavefront lock");
                            while progress[r - 1].load(Ordering::SeqCst) < need {
                                guard = cvar.wait(guard).expect("wavefront wait");
                            }
                            drop(guard);
                            // Pull the above-row neighbourhood (columns
                            // c − 1 ..= c + 1) from the picture state.
                            let (xa, ya, _, ha) = rect(r - 1, c.saturating_sub(1));
                            let x_end = ((col0 + need) * ctb).min(ctx.width);
                            let m = master.lock().expect("picture state");
                            merge_rect(&mut st, &m, ctx, (xa, ya, x_end - xa, ha));
                        }
                        let (x0, y0, _, _) = rect(r, c);
                        let rs = (row0 + r) * ctbs_x + col0 + c;
                        let ts = ctx.tiling.ctb_addr_rs_to_ts(rs as u32) as usize;
                        if ctx.cfg.rdoq && c == 0 {
                            // §9.3.2.2 at the row start: the stored
                            // contexts of the row above when its
                            // spatial neighbour T is available.
                            let mut coder =
                                EntropyCoder::new(init_type(slice_type_raw, false), ctx.qp);
                            if r > 0 && ctx.wpp_sync_available(ts) {
                                if let Some(ctxs) =
                                    stored[r - 1].lock().expect("stored contexts").as_ref()
                                {
                                    coder.ctxs = ctxs.clone();
                                }
                            }
                            shadow = Some((coder, ctx.qp));
                        }
                        let ctb_qp =
                            (ctx.qp + ctx.aq_deltas[rs]).clamp(-ctx.fmt.qp_bd_offset_y(), 51);
                        st.rdoq_model = shadow.as_ref().map(|(coder, _)| RdoqModel {
                            contexts: coder.ctxs.residual.clone(),
                        });
                        let (node, _cost) =
                            code_quadtree(ctx, &mut st, x0, y0, ctx.cfg.ctb_log2, 0, ctb_qp);
                        if let Some((coder, qp_prev)) = shadow.as_mut() {
                            let mut qg = QgState {
                                coded: false,
                                qp_prev: *qp_prev,
                                ctb_qp,
                                enabled: cu_qp_delta,
                            };
                            let mut em = Emitter {
                                ctx,
                                st: &st,
                                w: &mut coder.w,
                                cabac: &mut coder.cabac,
                                ctxs: &mut coder.ctxs,
                            };
                            em.emit_quadtree(&node, x0, y0, ctx.cfg.ctb_log2, 0, &mut qg);
                            if qg.coded {
                                *qp_prev = ctb_qp;
                            }
                            if ctx.wpp_store_after(ts) {
                                *stored[r].lock().expect("stored contexts") =
                                    Some(coder.ctxs.clone());
                            }
                        }
                        {
                            let mut m = master.lock().expect("picture state");
                            merge_rect(&mut m, &st, ctx, rect(r, c));
                        }
                        {
                            let (lock, cvar) = &wake;
                            let _guard = lock.lock().expect("wavefront lock");
                            progress[r].store(c + 1, Ordering::SeqCst);
                            cvar.notify_all();
                        }
                        plans.push(node);
                        ctb_qps.push(ctb_qp);
                    }
                    *results[r].lock().expect("row result") = Some((plans, ctb_qps));
                }
            });
        }
    });
    let mut plans = Vec::with_capacity(rows * cols);
    let mut ctb_qps = Vec::with_capacity(rows * cols);
    for slot in results {
        let (p, q) = slot
            .into_inner()
            .expect("row result")
            .expect("every row decided");
        plans.extend(p);
        ctb_qps.extend(q);
    }
    TileOut {
        st: master.into_inner().expect("picture state"),
        plans,
        ctb_qps,
    }
}

/// Copy a tile's rectangle of decided state into the picture state.
fn merge_tile(master: &mut EncState, tile: &EncState, ctx: &SliceCtx<'_>, job: &TileJob) {
    merge_rect(master, tile, ctx, (job.x0, job.y0, job.w, job.h));
}

/// Copy the luma rectangle `(x0, y0, w, h)` of decided state
/// (reconstruction, motion / mode fields, `CtDepth` / skip cells) from
/// `from` into `to`; `x0` / `y0` / `w` / `h` are multiples of 8
/// (CTB-aligned, clipped by the picture edge).
fn merge_rect(
    to: &mut EncState,
    from: &EncState,
    ctx: &SliceCtx<'_>,
    (x0, y0, w, h): (usize, usize, usize, usize),
) {
    let (master, tile) = (to, from);
    let (sw, sh) = ctx.fmt.sub_wh();
    let (cw, cx0, cy0) = (ctx.cw(), x0 / sw, y0 / sh);
    let y = rect_copy(&tile.recon.y, ctx.width, x0, y0, w, h);
    rect_paste(&mut master.recon.y, ctx.width, x0, y0, w, h, &y);
    if ctx.fmt.has_chroma() {
        let cb = rect_copy(&tile.recon.cb, cw, cx0, cy0, w / sw, h / sh);
        rect_paste(&mut master.recon.cb, cw, cx0, cy0, w / sw, h / sh, &cb);
        let cr = rect_copy(&tile.recon.cr, cw, cx0, cy0, w / sw, h / sh);
        rect_paste(&mut master.recon.cr, cw, cx0, cy0, w / sw, h / sh, &cr);
    }
    let field = tile.field.snapshot_rect(x0, y0, w, h);
    master.field.restore_rect(x0, y0, w, h, &field);
    let modes = tile.modes.snapshot_rect(x0, y0, w, h);
    master.modes.restore_rect(x0, y0, w, h, &modes);
    let bx1 = (x0 + w).div_ceil(4).min(master.w_cells);
    let by1 = (y0 + h).div_ceil(4).min(master.h_cells);
    for by in y0 / 4..by1 {
        for bx in x0 / 4..bx1 {
            let c = by * master.w_cells + bx;
            master.ct_depth[c] = tile.ct_depth[c];
            master.skip[c] = tile.skip[c];
        }
    }
}

/// Pass 1 + filters for one picture (I, P or B). The tiles are decided
/// independently (each in its own state — §6.4.1 availability never
/// crosses a tile boundary) on up to `ctx.threads` workers, then
/// merged in tile-scan order; the result is identical for any budget.
#[allow(clippy::too_many_lines)]
fn code_picture(
    ctx: &SliceCtx<'_>,
    lf: &LoopFilterCfg,
    cu_qp_delta: bool,
    slice_type_raw: u8,
) -> CodedPicture {
    let ctbs_x = ctx.ctbs_x();
    let ctbs_y = ctx.ctbs_y();
    let n_ctbs = ctbs_x * ctbs_y;
    let ctb = 1usize << ctx.cfg.ctb_log2;
    let jobs = tile_jobs(ctx);
    let workers = ctx.threads.min(jobs.len()).max(1);
    // A single-tile WPP picture fans its CTB rows out instead (the
    // CTU-level rate feedback reads the running size of everything
    // coded before, so it stays serial).
    let wpp_parallel =
        ctx.threads > 1 && ctx.cfg.wpp && jobs.len() == 1 && ctx.ctu_rc.is_none() && ctbs_y > 1;
    let outs: Vec<TileOut> = if wpp_parallel {
        vec![code_tile_wpp(
            ctx,
            &jobs[0],
            slice_type_raw,
            cu_qp_delta,
            ctx.threads.min(ctbs_y),
        )]
    } else if workers <= 1 {
        jobs.iter()
            .map(|job| code_tile(ctx, job, slice_type_raw, cu_qp_delta))
            .collect()
    } else {
        let next = std::sync::atomic::AtomicUsize::new(0);
        let mut done: Vec<Option<TileOut>> = (0..jobs.len()).map(|_| None).collect();
        let handles: Vec<Vec<(usize, TileOut)>> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..workers)
                .map(|_| {
                    scope.spawn(|| {
                        let mut mine = Vec::new();
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if i >= jobs.len() {
                                break;
                            }
                            mine.push((i, code_tile(ctx, &jobs[i], slice_type_raw, cu_qp_delta)));
                        }
                        mine
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|h| h.join().expect("tile worker"))
                .collect()
        });
        for (i, out) in handles.into_iter().flatten() {
            done[i] = Some(out);
        }
        done.into_iter()
            .map(|o| o.expect("every tile decided"))
            .collect()
    };
    let (mut st, plans, ctb_qps) = if outs.len() == 1 {
        let TileOut { st, plans, ctb_qps } = outs.into_iter().next().expect("one tile");
        (st, plans, ctb_qps)
    } else {
        let mut st = EncState::new(&ctx.fmt, ctx.width, ctx.height, ctx.cfg.ctb_log2);
        let mut plans = Vec::with_capacity(n_ctbs);
        let mut ctb_qps = Vec::with_capacity(n_ctbs);
        for (job, out) in jobs.iter().zip(outs) {
            merge_tile(&mut st, &out.st, ctx, job);
            plans.extend(out.plans);
            ctb_qps.extend(out.ctb_qps);
        }
        (st, plans, ctb_qps)
    };
    st.rdoq_model = None;

    // Stats.
    let mut stats = FrameStats::default();
    for plan in &plans {
        plan.for_each_cu(&mut |cu| match &cu.kind {
            TreeCuKind::Skip { .. } => stats.skip += 1,
            TreeCuKind::Merge { .. } => stats.merge += 1,
            TreeCuKind::Amvp { .. } => stats.amvp += 1,
            TreeCuKind::Intra { .. } => stats.intra += 1,
            TreeCuKind::TwoPu { part, pus } => {
                if pus.iter().any(|p| matches!(p, PuSyntax::Amvp { .. })) {
                    stats.amvp += 1;
                } else {
                    stats.merge += 1;
                }
                if part_is_amp(*part) {
                    stats.amp += 1;
                } else {
                    stats.rect += 1;
                }
            }
        });
        plan.for_each_cu(&mut |cu| {
            if cu.motions.iter().any(|m| m.pred_flag_l0 && m.pred_flag_l1) {
                stats.bi += 1;
            }
            if cu.motions.iter().any(|m| {
                (m.pred_flag_l0 && m.ref_idx_l0 > 0) || (m.pred_flag_l1 && m.ref_idx_l1 > 0)
            }) {
                stats.ref1 += 1;
            }
        });
    }

    // ---- §8.7 loop filters ----
    let walk = qp_walk(ctx, &plans, &ctb_qps, cu_qp_delta);
    let recon = st.recon.clone();
    let mut out = CodedPicture {
        plans,
        ctb_qps,
        st,
        recon,
        stats,
        deblock_on: false,
        beta_offset_div2: 0,
        tc_offset_div2: 0,
        sao_luma: false,
        sao_chroma: false,
        sao_ctbs: Vec::new(),
    };
    if lf.any() {
        let tile_ids: Option<Vec<u32>> = ctx.cfg.tiles.on().then(|| {
            (0..n_ctbs as u32)
                .map(|rs| ctx.tiling.tile_id(ctx.tiling.ctb_addr_rs_to_ts(rs)))
                .collect()
        });
        let ctb_qps: Vec<i32> = (0..ctbs_x * ctbs_y)
            .map(|i| {
                let x0 = (i % ctbs_x) * ctb;
                let y0 = (i / ctbs_x) * ctb;
                i32::from(walk.cells[(y0 / 4) * walk.w_cells + x0 / 4])
            })
            .collect();
        let filtered = filter_frame(
            &FilterInput {
                width: ctx.width,
                height: ctx.height,
                fmt: ctx.fmt,
                ctb_qps: &ctb_qps,
                lambda: ctx.lambda_of(ctx.qp),
                recon: [&out.recon.y, &out.recon.cb, &out.recon.cr],
                src: [ctx.src[0], ctx.src[1], ctx.src[2]],
                field: &out.st.field,
                shapes: &[],
                ctb_log2: ctx.cfg.ctb_log2,
                tree: Some(TreeLayout {
                    descs: &walk.descs,
                    qp_cells: &walk.cells,
                    w_cells: walk.w_cells,
                }),
                tile_ids: tile_ids.as_deref(),
            },
            lf,
        );
        out.deblock_on = filtered.deblock_on;
        out.beta_offset_div2 = filtered.beta_offset_div2;
        out.tc_offset_div2 = filtered.tc_offset_div2;
        out.sao_luma = filtered.slice_sao_luma;
        out.sao_chroma = filtered.slice_sao_chroma;
        out.sao_ctbs = filtered.sao_ctbs;
        out.recon.y = filtered.y;
        out.recon.cb = filtered.cb;
        out.recon.cr = filtered.cr;
    }
    out
}

/// Pass 2: emit the slice data (CTU loop, §6.5.1 tile scan) for a
/// coded picture as its §7.3.8.1 subsets (one per tile / WPP row;
/// a single subset otherwise). `slice_type_raw` per §7.4.7.1.
fn emit_slice_data(
    ctx: &SliceCtx<'_>,
    coded: &CodedPicture,
    slice_type_raw: u8,
    cu_qp_delta: bool,
) -> Vec<Vec<u8>> {
    let st = &coded.st;
    let mut coder = EntropyCoder::new(init_type(slice_type_raw, false), ctx.qp);
    let ctbs_x = ctx.ctbs_x();
    let ctbs_y = ctx.ctbs_y();
    let n_ctbs = ctbs_x * ctbs_y;
    let ctb = 1usize << ctx.cfg.ctb_log2;
    let mut qp_prev = ctx.qp;
    for (ctb_idx, plan) in coded.plans.iter().enumerate() {
        let rs = ctx.ctb_rs(ctb_idx);
        let x0 = (rs % ctbs_x) * ctb;
        let y0 = (rs / ctbs_x) * ctb;
        coder.begin_ctb(ctx, ctb_idx);
        if ctx.qp_prev_resets(ctb_idx) {
            qp_prev = ctx.qp;
        }
        if coded.sao_luma || coded.sao_chroma {
            let same_tile = |other: usize| {
                ctx.tiling
                    .tile_id(ctx.tiling.ctb_addr_rs_to_ts(other as u32))
                    == ctx.tiling.tile_id(ctb_idx as u32)
            };
            let can_left = rs % ctbs_x > 0 && same_tile(rs - 1);
            let can_up = rs / ctbs_x > 0 && same_tile(rs - ctbs_x);
            encode_sao_ctb_fmt(
                &mut coder.w,
                &mut coder.cabac,
                &mut coder.ctxs,
                &coded.sao_ctbs[rs],
                can_left,
                can_up,
                coded.sao_luma,
                coded.sao_chroma,
                &ctx.fmt,
            );
        }
        let ctb_qp = coded.ctb_qps[ctb_idx];
        let mut qg = QgState {
            coded: false,
            qp_prev,
            ctb_qp,
            enabled: cu_qp_delta,
        };
        let mut em = Emitter {
            ctx,
            st,
            w: &mut coder.w,
            cabac: &mut coder.cabac,
            ctxs: &mut coder.ctxs,
        };
        em.emit_quadtree(plan, x0, y0, ctx.cfg.ctb_log2, 0, &mut qg);
        if qg.coded {
            qp_prev = ctb_qp;
        }
        coder.end_ctb(ctx, ctb_idx, ctb_idx + 1 == n_ctbs);
    }
    coder.subsets
}

/// Append the slice-data subsets after a byte-aligned header.
fn append_subsets(w: &mut BitWriter, subsets: &[Vec<u8>]) {
    for s in subsets {
        for &b in s {
            w.put_bits(u32::from(b), 8);
        }
    }
}

/// The §7.3.6.1 entry-point block is present iff the PPS enables
/// tiles or entropy coding sync.
fn entry_points_present(cfg: &TreeCfg) -> bool {
    cfg.tiles.on() || cfg.wpp
}

/// Encode one 4:2:0 8-bit frame as a quadtree intra IDR slice RBSP +
/// reconstruction. The caller wraps VPS/SPS/PPS/NAL around it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_intra_picture_tree(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    qp: i32,
    cfg: &SpsCfg,
    lf: &LoopFilterCfg,
    aq: u8,
    ctu_rc: Option<u64>,
) -> Result<IntraEncodedAu, IntraEncodeError> {
    debug_assert!(
        cfg.fmt.is_yuv420_8(),
        "the u8 entry codes 4:2:0 8-bit pictures"
    );
    let (wy, wcb, wcr) = (widen(y), widen(cb), widen(cr));
    let wide =
        encode_intra_picture_tree_wide([&wy, &wcb, &wcr], width, height, qp, cfg, lf, aq, ctu_rc)?;
    let narrow = |v: Vec<u16>| -> Vec<u8> { v.into_iter().map(|s| s as u8).collect() };
    Ok(IntraEncodedAu {
        au: wide.au,
        recon_y: narrow(wide.recon_y),
        recon_cb: narrow(wide.recon_cb),
        recon_cr: narrow(wide.recon_cr),
    })
}

/// Encode one frame of any chroma format / bit depth (`cfg.fmt`) as a
/// quadtree intra IDR access unit + reconstruction: the planes are
/// `[Y, Cb, Cr]` at `width x height` luma (chroma per Table 6-1;
/// empty for monochrome), `qp` the `SliceQpY` in `−QpBdOffsetY ..=
/// 51`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_intra_picture_tree_wide(
    planes: [&[u16]; 3],
    width: usize,
    height: usize,
    qp: i32,
    cfg: &SpsCfg,
    lf: &LoopFilterCfg,
    aq: u8,
    ctu_rc: Option<u64>,
) -> Result<IntraEncodedAuWide, IntraEncodeError> {
    let tree = cfg.tree.expect("tree config present");
    let fmt = cfg.fmt;
    if width == 0 || height == 0 || width % 16 != 0 || height % 16 != 0 {
        return Err(IntraEncodeError::BadDimensions { width, height });
    }
    if !fmt.qp_range().contains(&qp) {
        return Err(IntraEncodeError::BadQp(qp));
    }
    if aq > 3 {
        return Err(IntraEncodeError::BadAq(aq));
    }
    // Sample VALUES are not range-checked here: an out-of-range source
    // sample only enters the residual (the reconstruction is clipped),
    // so the caller owns that validation (the registry encoder does).
    let check = |plane: &'static str, buf: &[u16], expected: usize| {
        if buf.len() != expected {
            Err(IntraEncodeError::PlaneSize {
                plane,
                expected,
                got: buf.len(),
            })
        } else {
            Ok(())
        }
    };
    let (cw, ch) = fmt.chroma_dims(width, height);
    check("y", planes[0], width * height)?;
    check("cb", planes[1], cw * ch)?;
    check("cr", planes[2], cw * ch)?;

    let ctb = 1usize << tree.ctb_log2;
    let aq_deltas = crate::encoder::aq::ctb_aq_deltas_wide(
        planes[0],
        width,
        height,
        aq,
        ctb,
        fmt.bit_depth_luma,
    );
    let tiling = make_tiling(width, height, &tree);
    let scaling = scaling_lists_for(tree.scaling_lists)
        .map(|(d, _)| d.scaling_factors(fmt.chroma_format_idc));
    let ctx = SliceCtx {
        cfg: tree,
        fmt,
        amp: cfg.amp,
        width,
        height,
        src: planes,
        qp,
        aq_deltas: &aq_deltas,
        ctu_rc,
        b_slice: false,
        intra_slice: true,
        refs_l0: &[],
        refs_l1: &[],
        mv_ctx: None,
        two_sided: false,
        tiling: &tiling,
        scaling: scaling.as_ref(),
        wp: None,
        me_refs_l0: &[],
        me_refs_l1: &[],
        threads: cfg.threads,
        chroma_qp_offset: cfg.chroma_qp_offset,
    };
    let cu_qp_delta = aq > 0 || ctu_rc.is_some();
    debug_assert!(
        cfg.cu_qp_delta || !cu_qp_delta,
        "cu_qp_delta signalling needs the PPS flag"
    );
    let coded = code_picture(&ctx, lf, cu_qp_delta, 2);
    let subsets = emit_slice_data(&ctx, &coded, 2, cu_qp_delta);

    // ---- slice_segment_header( ) — I slice, IDR ----
    let mut w = BitWriter::new();
    w.put_bit(1); // first_slice_segment_in_pic_flag
    w.put_bit(0); // no_output_of_prior_pics_flag
    w.ue(u32::from(cfg.ids.pps)); // slice_pic_parameter_set_id
    w.ue(2); // slice_type = I
    if lf.sao() {
        w.put_bit(u8::from(coded.sao_luma));
        if fmt.has_chroma() {
            w.put_bit(u8::from(coded.sao_chroma));
        }
    }
    w.se(qp - 26); // slice_qp_delta
    if lf.deblocking {
        w.put_bit(u8::from(coded.deblock_on)); // deblocking_filter_override_flag
        if coded.deblock_on {
            w.put_bit(0); // slice_deblocking_filter_disabled_flag
            w.se(coded.beta_offset_div2);
            w.se(coded.tc_offset_div2);
        }
    }
    if coded.sao_luma || coded.sao_chroma || coded.deblock_on {
        w.put_bit(1); // slice_loop_filter_across_slices_enabled_flag
    }
    if entry_points_present(&tree) {
        write_entry_points(&mut w, &entry_point_offsets(&subsets));
    }
    w.rbsp_trailing_bits();
    append_subsets(&mut w, &subsets);
    let slice_rbsp = w.finish();
    let au = crate::encoder::intra::assemble_idr_au(width, height, cfg, lf, &slice_rbsp);
    Ok(IntraEncodedAuWide {
        au,
        recon_y: coded.recon.y,
        recon_cb: coded.recon.cb,
        recon_cr: coded.recon.cr,
    })
}

fn make_tiling(width: usize, height: usize, cfg: &TreeCfg) -> PictureTiling {
    let ctb = 1usize << cfg.ctb_log2;
    let ctbs_x = width.div_ceil(ctb) as u32;
    let ctbs_y = height.div_ceil(ctb) as u32;
    // A grid the picture cannot hold (more columns / rows than CTBs,
    // or explicit spans past the edge) falls back to a single tile.
    let params = cfg.tiles.params();
    let fits = u32::from(cfg.tiles.cols) <= ctbs_x
        && u32::from(cfg.tiles.rows) <= ctbs_y
        && (cfg.tiles.uniform
            || (params
                .column_width_minus1
                .iter()
                .map(|w| w + 1)
                .sum::<u32>()
                < ctbs_x
                && params.row_height_minus1.iter().map(|h| h + 1).sum::<u32>() < ctbs_y));
    let params = if fits {
        params
    } else {
        TilingParams::single_tile()
    };
    PictureTiling::new(
        ctbs_x,
        ctbs_y,
        width as u32,
        height as u32,
        cfg.ctb_log2,
        2,
        &params,
    )
    .expect("legal tile geometry")
}

/// Encode one P / B frame as a quadtree TRAIL_R slice (the tree twin
/// of [`crate::encoder::inter::encode_inter_slice`]).
pub(crate) fn encode_inter_slice_tree(
    frame: &YuvFrame<'_>,
    spec: &SliceSpec<'_>,
    width: usize,
    height: usize,
) -> (Vec<u8>, FrameRecon, FrameStats) {
    let tree = spec.tree.expect("tree config present");
    let ctb = 1usize << tree.ctb_log2;
    let aq_deltas = crate::encoder::aq::ctb_aq_deltas(frame.y, width, height, spec.aq, ctb);
    let tiling = make_tiling(width, height, &tree);
    let scaling = scaling_lists_for(tree.scaling_lists).map(|(d, _)| d.scaling_factors(1));

    let to_i32 = |p: &[u8]| -> Vec<i32> { p.iter().map(|&v| i32::from(v)).collect() };
    let to_planes = |list: &[(i32, &FrameRecon)]| -> Vec<RefPlanes> {
        list.iter()
            .map(|&(_, rec)| RefPlanes {
                y: to_i32(&rec.y),
                cb: to_i32(&rec.cb),
                cr: to_i32(&rec.cr),
                width,
                height,
            })
            .collect()
    };
    let refs_l0 = to_planes(&spec.l0);
    let refs_l1 = to_planes(&spec.l1);
    // Explicit weighted prediction: fade-fitted tables per reference,
    // luma-weighted reference copies for the motion search.
    let wp_tables = tree.weighted_pred.then(|| {
        let l0: Vec<&FrameRecon> = spec.l0.iter().map(|&(_, r)| r).collect();
        let l1: Vec<&FrameRecon> = spec.l1.iter().map(|&(_, r)| r).collect();
        crate::encoder::wp::estimate(frame, &l0, &l1, width, height)
    });
    let me_planes = wp_tables.as_ref().map(|t| {
        (
            crate::encoder::wp::weighted_ref_planes(&refs_l0, &t.l0),
            crate::encoder::wp::weighted_ref_planes(&refs_l1, &t.l1),
        )
    });
    let (me_refs_l0, me_refs_l1): (&[RefPlanes], &[RefPlanes]) = match &me_planes {
        Some((a, b)) => (a, b),
        None => (&refs_l0, &refs_l1),
    };
    let l0_pocs: Vec<i32> = spec.l0.iter().map(|&(p, _)| p).collect();
    let l1_pocs: Vec<i32> = spec.l1.iter().map(|&(p, _)| p).collect();
    let n_l0 = l0_pocs.len() as i32;
    let n_l1 = l1_pocs.len() as i32;
    let two_sided = spec.b_slice && l0_pocs != l1_pocs;
    let list_pocs = |list: usize| -> &Vec<i32> {
        if list == 0 {
            &l0_pocs
        } else {
            &l1_pocs
        }
    };
    let ref_poc = |list: usize, ref_idx: i32| -> i32 {
        usize::try_from(ref_idx)
            .ok()
            .and_then(|i| list_pocs(list).get(i).copied())
            .unwrap_or(i32::MIN)
    };
    let ref_long_term = |_list: usize, _ref_idx: i32| false;
    let ref_short_term = |list: usize, ref_idx: i32| {
        usize::try_from(ref_idx).is_ok_and(|i| i < list_pocs(list).len())
    };
    let col_ref_long_term = |_poc: i32| false;
    let no_backward_pred = !l0_pocs.iter().chain(l1_pocs.iter()).any(|&p| p > spec.poc);
    let (col_poc, col_field) = crate::encoder::inter::collocated_picture(spec);
    let mv_ctx = PuMvContext {
        curr_poc: spec.poc,
        slice_is_b: spec.b_slice,
        ctb_log2_size_y: tree.ctb_log2,
        pic_width_luma: width as u32,
        pic_height_luma: height as u32,
        max_num_merge_cand: MAX_MERGE,
        num_ref_idx_l0_active: n_l0,
        num_ref_idx_l1_active: if spec.b_slice { n_l1 } else { 0 },
        log2_par_mrg_level: 2,
        temporal_mvp_enabled: spec.tmvp.slice_enabled,
        collocated_from_l0_flag: !spec.b_slice || spec.tmvp.collocated_from_l0,
        col_poc,
        no_backward_pred,
        ref_poc: &ref_poc,
        ref_long_term: &ref_long_term,
        ref_short_term: &ref_short_term,
        col_field,
        col_ref_long_term: &col_ref_long_term,
        use_integer_mv: false,
        two_versions_curr_pic: false,
        is_curr_pic: &|_, _| false,
    };
    let (wy, wcb, wcr) = (widen(frame.y), widen(frame.cb), widen(frame.cr));
    let ctx = SliceCtx {
        cfg: tree,
        fmt: SampleFmt::YUV420_8,
        amp: spec.big_cu,
        width,
        height,
        src: [&wy, &wcb, &wcr],
        qp: spec.qp,
        aq_deltas: &aq_deltas,
        ctu_rc: spec.ctu_rc,
        b_slice: spec.b_slice,
        intra_slice: false,
        refs_l0: &refs_l0,
        refs_l1: &refs_l1,
        mv_ctx: Some(&mv_ctx),
        two_sided,
        tiling: &tiling,
        scaling: scaling.as_ref(),
        wp: wp_tables.as_ref().map(|t| &t.resolved),
        me_refs_l0,
        me_refs_l1,
        threads: spec.threads,
        chroma_qp_offset: 0,
    };
    let cu_qp_delta = spec.aq > 0 || spec.ctu_rc.is_some();
    let raw_slice_type: u8 = if spec.b_slice { 0 } else { 1 };
    let coded = code_picture(&ctx, spec.lf, cu_qp_delta, raw_slice_type);

    let lf_sig = SliceLfSignalling {
        cfg: spec.lf,
        deblock_on: coded.deblock_on,
        beta_offset_div2: coded.beta_offset_div2,
        tc_offset_div2: coded.tc_offset_div2,
        sao_luma: coded.sao_luma,
        sao_chroma: coded.sao_chroma,
    };
    let subsets = emit_slice_data(&ctx, &coded, raw_slice_type, cu_qp_delta);
    let offsets = entry_points_present(&tree).then(|| entry_point_offsets(&subsets));
    let mut w = BitWriter::new();
    crate::encoder::inter::write_inter_slice_header(
        &mut w,
        spec,
        &lf_sig,
        wp_tables.as_ref(),
        offsets.as_deref(),
    );
    append_subsets(&mut w, &subsets);
    let CodedPicture {
        recon, st, stats, ..
    } = coded;
    let mut recon = recon.into_frame_recon();
    recon.motion_field = Some(st.field);
    (w.finish(), recon, stats)
}

#[cfg(test)]
mod tests {
    /// SATD: zero difference is zero; a constant difference of `d`
    /// over an `n x n` block is `n * n * d` before the SAD-scale
    /// normalization (only the DC Hadamard coefficient is nonzero);
    /// a single-sample difference spreads to every coefficient.
    #[test]
    fn satd_kernels_match_hadamard_identities() {
        for n in [4usize, 8, 16, 32] {
            let src: Vec<i32> = (0..n * n).map(|i| (i % 17) as i32 + 40).collect();
            assert_eq!(satd(&src, &src, n), 0, "n={n}");
            let pred: Vec<i32> = src.iter().map(|&v| v - 3).collect();
            // Every 8x8 (or the 4x4) kernel sees DC = m*m*3.
            let expected = if n == 4 {
                (4 * 4 * 3 + 1) >> 1
            } else {
                ((n / 8) * (n / 8) * 8 * 8 * 3 + 2) >> 2
            };
            assert_eq!(satd(&src, &pred, n), expected as u64, "n={n}");
        }
        let mut pred = vec![0i32; 16];
        let src = vec![0i32; 16];
        pred[5] = 8;
        // One impulse of 8: all 16 4x4 Hadamard coefficients are ±8.
        assert_eq!(satd(&src, &pred, 4), (16 * 8 + 1) >> 1);
    }

    /// §7.3.8.5 luma mode bins: 2 / 3 / 3 for the three most-probable
    /// modes, 6 (flag + 5-bit remainder) otherwise.
    #[test]
    fn luma_mode_bins_follow_the_mpm_binarization() {
        let mpm = [26u8, 10, 0];
        assert_eq!(luma_mode_bins(26, &mpm), 2);
        assert_eq!(luma_mode_bins(10, &mpm), 3);
        assert_eq!(luma_mode_bins(0, &mpm), 3);
        assert_eq!(luma_mode_bins(1, &mpm), 6);
        assert_eq!(luma_mode_bins(34, &mpm), 6);
    }

    use super::*;
    use crate::encoder::inter::YuvFrame;
    use crate::sequence::decode_annexb_sequence;

    fn planes(w: usize, h: usize, seed: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut y = vec![0u8; w * h];
        for j in 0..h {
            for i in 0..w {
                let v = (i * 3 + j * 5 + usize::from(seed) * 7) % 256;
                let block = if (i / 20 + j / 14) % 3 == 0 { 40 } else { 0 };
                y[j * w + i] = ((v / 2) + block) as u8;
            }
        }
        let cb: Vec<u8> = (0..w * h / 4).map(|k| (k % 32 + 100) as u8).collect();
        let cr: Vec<u8> = (0..w * h / 4).map(|k| (k % 24 + 90) as u8).collect();
        (y, cb, cr)
    }

    fn tree_cfg(ctb: usize) -> SpsCfg {
        SpsCfg {
            min_cb_log2: 3,
            tree: TreeCfg::new(ctb),
            ..SpsCfg::legacy(1)
        }
    }

    fn assert_intra_roundtrip(w: usize, h: usize, qp: i32, ctb: usize) {
        let (y, cb, cr) = planes(w, h, 3);
        let au = crate::encoder::intra::encode_idr_intra_au_full(
            &y,
            &cb,
            &cr,
            w,
            h,
            qp,
            &tree_cfg(ctb),
            &LoopFilterCfg::off(),
            0,
            None,
        )
        .expect("encode");
        let frames = decode_annexb_sequence(&au.au).expect("decode");
        assert_eq!(frames.len(), 1);
        let f = &frames[0];
        assert_eq!(f.picture.to_planar_u8().unwrap()[..w * h], au.recon_y[..]);
        let planar = f.picture.to_planar_u8().unwrap();
        assert_eq!(planar[w * h..w * h + w * h / 4], au.recon_cb[..]);
        assert_eq!(planar[w * h + w * h / 4..], au.recon_cr[..]);
    }

    /// Deterministic textured planes of a sample format (u16 samples
    /// at the format's depths; chroma empty for monochrome).
    fn planes_fmt(fmt: &SampleFmt, w: usize, h: usize) -> [Vec<u16>; 3] {
        let tex = |x: usize, y: usize, seed: usize, bd: u8| -> u16 {
            let v = (x * 3 + y * 5 + seed * 7) % 256;
            let block = if (x / 20 + y / 14) % 3 == 0 { 40 } else { 0 };
            let ripple = ((x * y + seed) % 9) as u32;
            let s8 = (v / 2 + block) as u32;
            // Scale to the depth and add sub-8-bit detail.
            ((s8 << (bd - 8)) + ripple * ((1u32 << (bd - 8)) - 1) / 8) as u16
        };
        let y: Vec<u16> = (0..w * h)
            .map(|i| tex(i % w, i / w, 3, fmt.bit_depth_luma))
            .collect();
        let (cw, ch) = fmt.chroma_dims(w, h);
        let cb: Vec<u16> = (0..cw * ch)
            .map(|i| tex(i % cw.max(1), i / cw.max(1), 11, fmt.bit_depth_chroma))
            .collect();
        let cr: Vec<u16> = (0..cw * ch)
            .map(|i| tex(i / ch.max(1), i % ch.max(1), 17, fmt.bit_depth_chroma))
            .collect();
        [y, cb, cr]
    }

    /// Encode a picture of any format through the quadtree intra coder
    /// and check the crate's decoder reconstructs the encoder's own
    /// reconstruction exactly, plane by plane; returns the AU size.
    fn assert_intra_roundtrip_fmt(
        fmt: SampleFmt,
        (w, h): (usize, usize),
        qp: i32,
        tree: TreeCfg,
        lf: LoopFilterCfg,
    ) -> usize {
        let planes = planes_fmt(&fmt, w, h);
        let cfg = SpsCfg {
            min_cb_log2: 3,
            tree: Some(tree),
            fmt,
            ..SpsCfg::legacy(1)
        };
        let au = crate::encoder::intra::encode_idr_intra_au_wide(
            [&planes[0], &planes[1], &planes[2]],
            w,
            h,
            qp,
            &cfg,
            &lf,
            0,
            None,
        )
        .expect("encode");
        let frames = decode_annexb_sequence(&au.au).expect("decode");
        assert_eq!(frames.len(), 1);
        let pic = &frames[0].picture;
        assert_eq!(pic.chroma_array_type(), fmt.chroma_format_idc);
        let as_u16 = |p: crate::picture::Plane| -> Vec<u16> {
            pic.plane(p).iter().map(|&v| v as u16).collect()
        };
        let what = format!("{fmt:?} {w}x{h} qp {qp}");
        assert_eq!(
            as_u16(crate::picture::Plane::Luma),
            au.recon_y,
            "{what}: luma"
        );
        if fmt.has_chroma() {
            assert_eq!(as_u16(crate::picture::Plane::Cb), au.recon_cb, "{what}: cb");
            assert_eq!(as_u16(crate::picture::Plane::Cr), au.recon_cr, "{what}: cr");
        } else {
            assert!(au.recon_cb.is_empty() && au.recon_cr.is_empty());
        }
        // Lossy but faithful: the luma PSNR against the source stays
        // well above a broken reconstruction's.
        let peak = f64::from(fmt.max_luma());
        let mse = planes[0]
            .iter()
            .zip(&au.recon_y)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum::<f64>()
            / (w * h) as f64;
        let psnr = 10.0 * (peak * peak / mse.max(1e-9)).log10();
        assert!(psnr > 30.0, "{what}: luma PSNR {psnr:.2}");
        au.au.len()
    }

    #[test]
    fn deep_layout_intra_roundtrips_every_format() {
        // 4:2:0 10 / 12-bit, 4:2:2, 4:4:4 and monochrome at 8 / 10 /
        // 12 bits: CTB 32 so both a forced-split 32x32 and the 8x8
        // NxN / 4x4 leaves (4:4:4 in-place chroma, 4:2:2 stacked
        // deferred blocks) are exercised on a small picture.
        for (cfi, bd) in [
            (1u8, 10u8),
            (1, 12),
            (2, 8),
            (2, 10),
            (2, 12),
            (3, 8),
            (3, 10),
            (3, 12),
            (0, 8),
            (0, 10),
            (0, 12),
        ] {
            let fmt = SampleFmt::new(cfi, bd).expect("format");
            let tree = TreeCfg::new(32).expect("ctb").with_intra_rd(2);
            assert_intra_roundtrip_fmt(fmt, (48, 32), 24, tree, LoopFilterCfg::off());
        }
    }

    #[test]
    fn deep_layout_intra_roundtrips_with_filters_and_quant_tools() {
        // Deblocking + SAO (the 10-bit offset range, band shift, the
        // 4:2:2 / 4:4:4 chroma CTB geometry, no chroma for monochrome),
        // RDOQ + sign hiding + scaling lists (the 4:4:4 32x32 chroma
        // matrices), a CTB-64 picture whose 64x64 CUs split to 32x32
        // TBs (4:4:4 chroma TBs at 32x32).
        for (cfi, bd) in [(1u8, 10u8), (2, 10), (3, 10), (0, 12), (3, 12)] {
            let fmt = SampleFmt::new(cfi, bd).expect("format");
            let tree = TreeCfg::new(64)
                .expect("ctb")
                .with_rdoq(true)
                .with_sign_hiding(true)
                .with_scaling_lists(1)
                .with_tu_depth(2, 2)
                .with_intra_rd(1);
            assert_intra_roundtrip_fmt(fmt, (64, 48), 20, tree, LoopFilterCfg::all());
        }
    }

    #[test]
    fn deep_layout_intra_accepts_the_extended_qp_range() {
        // SliceQpY −QpBdOffsetY (−12 at 10 bits, −24 at 12) up to 51.
        let fmt10 = SampleFmt::new(1, 10).expect("format");
        let tree = TreeCfg::new(16).expect("ctb");
        let fine = assert_intra_roundtrip_fmt(fmt10, (32, 32), -12, tree, LoopFilterCfg::off());
        let coarse = assert_intra_roundtrip_fmt(fmt10, (32, 32), 20, tree, LoopFilterCfg::off());
        assert!(fine > coarse, "QP −12 spends more bits than QP 20");
        let fmt12 = SampleFmt::new(3, 12).expect("format");
        assert_intra_roundtrip_fmt(fmt12, (32, 32), -24, tree, LoopFilterCfg::off());
        let planes = planes_fmt(&fmt12, 32, 32);
        let cfg = SpsCfg {
            min_cb_log2: 3,
            tree: Some(tree),
            fmt: fmt12,
            ..SpsCfg::legacy(1)
        };
        let bad = crate::encoder::intra::encode_idr_intra_au_wide(
            [&planes[0], &planes[1], &planes[2]],
            32,
            32,
            -25,
            &cfg,
            &LoopFilterCfg::off(),
            0,
            None,
        );
        assert!(matches!(bad, Err(IntraEncodeError::BadQp(-25))));
    }

    #[test]
    fn tree_intra_roundtrips_ctb16() {
        assert_intra_roundtrip(64, 32, 30, 16);
    }

    #[test]
    fn tree_intra_roundtrips_ctb32() {
        assert_intra_roundtrip(96, 64, 27, 32);
        assert_intra_roundtrip(48, 48, 40, 32);
    }

    #[test]
    fn tree_intra_roundtrips_ctb64() {
        assert_intra_roundtrip(96, 80, 32, 64);
    }

    #[test]
    fn tree_intra_elects_splits_and_dst() {
        // A busy picture at moderate QP must produce at least one
        // split (8x8 or NxN) somewhere and still round-trip.
        assert_intra_roundtrip(80, 64, 22, 64);
    }

    fn scene(w: usize, h: usize, n_frames: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        (0..n_frames)
            .map(|f| {
                let mut y = vec![0u8; w * h];
                for j in 0..h {
                    for i in 0..w {
                        // A moving edge + static texture.
                        let sq =
                            usize::from(i >= 8 + f * 2 && i < 24 + f * 2 && (8..24).contains(&j));
                        y[j * w + i] = ((i * 5 + j * 3) % 128 + sq * 90) as u8;
                    }
                }
                let cb = vec![110u8; w * h / 4];
                let cr = vec![120u8; w * h / 4];
                (y, cb, cr)
            })
            .collect()
    }

    fn assert_gop_tree_roundtrip(ctb: usize, b_slices: bool, lf: LoopFilterCfg, aq: u8) {
        assert_gop_tree_roundtrip_cfg(TreeCfg::new(ctb).expect("legal ctb"), b_slices, lf, aq);
    }

    fn assert_gop_tree_roundtrip_cfg(cfg: TreeCfg, b_slices: bool, lf: LoopFilterCfg, aq: u8) {
        use crate::encoder::inter::LowDelayPEncoder;
        let (w, h) = (96, 64);
        let frames = scene(w, h, 4);
        let mut enc = LowDelayPEncoder::new(w, h, 30, 0)
            .expect("encoder")
            .with_tree(cfg)
            .with_b_slices(b_slices)
            .with_loop_filters(lf)
            .with_aq(aq);
        let mut stream = Vec::new();
        let mut recons = Vec::new();
        for (y, cb, cr) in &frames {
            let f = enc.encode_frame(&YuvFrame { y, cb, cr }).expect("frame");
            stream.extend_from_slice(&f.au);
            recons.push(f.recon);
        }
        let decoded = decode_annexb_sequence(&stream).expect("decode");
        assert_eq!(decoded.len(), recons.len());
        for (f, rec) in decoded.iter().zip(recons.iter()) {
            let planar = f.picture.to_planar_u8().unwrap();
            assert_eq!(planar[..w * h], rec.y[..], "luma mismatch");
            assert_eq!(planar[w * h..w * h + w * h / 4], rec.cb[..]);
            assert_eq!(planar[w * h + w * h / 4..], rec.cr[..]);
        }
    }

    /// Depth-2 / depth-3 residual quadtrees with 8x4 / 4x8 PUs and
    /// the quantization tools on, P and B slices: every stream must
    /// reconstruct sample-exact through the decoder.
    #[test]
    fn tree_deep_rqt_and_small_pus_roundtrip() {
        for (depth, b) in [(2u32, false), (3, true), (2, true)] {
            let cfg = TreeCfg::new(32)
                .expect("ctb 32")
                .with_tu_depth(depth, depth)
                .with_rdoq(true)
                .with_sign_hiding(true);
            assert_gop_tree_roundtrip_cfg(cfg, b, LoopFilterCfg::all(), 0);
        }
    }

    /// The deeper ladder actually elects 8x4 / 4x8 PUs and depth-2
    /// transform splits on a busy P frame.
    /// Default and custom scaling lists on P / B GOPs (with RDOQ +
    /// sign hiding): the SPS carries scaling_list_enabled_flag (and
    /// the §7.3.4 body for the custom families) and every stream
    /// reconstructs sample-exact through the decoder.
    /// Weighted prediction on a fading clip, P and B (pyramid) paths:
    /// the slices carry pred_weight_table( ) entries and the streams
    /// reconstruct sample-exact.
    #[test]
    fn tree_weighted_pred_roundtrips_on_a_fade() {
        use crate::encoder::inter::LowDelayPEncoder;
        use crate::encoder::pyramid::PyramidEncoder;
        let (w, h) = (96, 64);
        let base = scene(w, h, 1).remove(0);
        // Brightness ramps 100 % -> 55 % over six frames.
        let frames: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..6u32)
            .map(|f| {
                let g = 100 - f * 9;
                let fade = |p: &[u8]| -> Vec<u8> {
                    p.iter().map(|&v| (u32::from(v) * g / 100) as u8).collect()
                };
                (fade(&base.0), fade(&base.1), fade(&base.2))
            })
            .collect();
        let cfg = TreeCfg::new(32).expect("ctb 32").with_weighted_pred(true);
        let mut enc = LowDelayPEncoder::new(w, h, 28, 0)
            .expect("encoder")
            .with_tree(cfg)
            .with_b_slices(true)
            .with_loop_filters(LoopFilterCfg::all());
        let mut stream = Vec::new();
        let mut recons = Vec::new();
        for (y, cb, cr) in &frames {
            let f = enc.encode_frame(&YuvFrame { y, cb, cr }).expect("frame");
            stream.extend_from_slice(&f.au);
            recons.push(f.recon);
        }
        let decoded = decode_annexb_sequence(&stream).expect("decode");
        assert_eq!(decoded.len(), recons.len());
        for (f, rec) in decoded.iter().zip(recons.iter()) {
            let planar = f.picture.to_planar_u8().unwrap();
            assert_eq!(planar[..w * h], rec.y[..], "luma mismatch");
            assert_eq!(planar[w * h..w * h + w * h / 4], rec.cb[..]);
            assert_eq!(planar[w * h + w * h / 4..], rec.cr[..]);
        }
        // Pyramid (two-sided B, bi-pred weights).
        let mut enc = PyramidEncoder::new(w, h, 28, 4)
            .expect("encoder")
            .with_tree(cfg)
            .with_loop_filters(LoopFilterCfg::all());
        let mut aus = Vec::new();
        for (y, cb, cr) in &frames {
            aus.extend(enc.encode_frame(&YuvFrame { y, cb, cr }).expect("frame"));
        }
        aus.extend(enc.flush());
        let mut stream = Vec::new();
        let mut recons: Vec<Option<FrameRecon>> = (0..frames.len()).map(|_| None).collect();
        for au in aus {
            stream.extend_from_slice(&au.au);
            recons[au.display_order] = Some(au.recon);
        }
        let decoded = decode_annexb_sequence(&stream).expect("decode");
        assert_eq!(decoded.len(), recons.len());
        for (f, rec) in decoded.iter().zip(recons.iter()) {
            let rec = rec.as_ref().unwrap();
            let planar = f.picture.to_planar_u8().unwrap();
            assert_eq!(planar[..w * h], rec.y[..], "pyramid luma mismatch");
            assert_eq!(planar[w * h..w * h + w * h / 4], rec.cb[..]);
            assert_eq!(planar[w * h + w * h / 4..], rec.cr[..]);
        }
    }

    /// WPP, uniform / explicit tiles and both together — with RDOQ
    /// (the shadow coder's subset structure), AQ (per-tile / per-row
    /// qPY_PREV resets) and the in-loop filters (tile-gated SAO
    /// merges) — on I / P / B slices: every stream reconstructs
    /// sample-exact through the decoder.
    #[test]
    fn tree_wpp_and_tiles_roundtrip() {
        let cases: Vec<(TreeCfg, bool, u8)> = vec![
            (
                TreeCfg::new(16)
                    .expect("ctb")
                    .with_wpp(true)
                    .with_rdoq(true),
                false,
                1,
            ),
            (
                TreeCfg::new(16)
                    .expect("ctb")
                    .with_tiles(TileLayout::uniform(2, 2))
                    .with_rdoq(true),
                true,
                2,
            ),
            (
                TreeCfg::new(16)
                    .expect("ctb")
                    .with_tiles(TileLayout::explicit(&[1, 3], &[2]))
                    .with_wpp(true)
                    .with_sign_hiding(true),
                false,
                0,
            ),
            (
                TreeCfg::new(32)
                    .expect("ctb")
                    .with_tiles(TileLayout::uniform(2, 1))
                    .with_wpp(true),
                true,
                1,
            ),
        ];
        for (cfg, b, aq) in cases {
            assert_gop_tree_roundtrip_cfg(cfg, b, LoopFilterCfg::all(), aq);
        }
    }

    /// The tiled / WPP streams really carry the subset structure: the
    /// slice header declares the entry points and the PPS the flags.
    #[test]
    fn tree_wpp_tiles_signal_entry_points() {
        use crate::encoder::inter::LowDelayPEncoder;
        let (w, h) = (96, 64);
        let frames = scene(w, h, 2);
        let cfg = TreeCfg::new(16)
            .expect("ctb")
            .with_tiles(TileLayout::uniform(2, 2))
            .with_wpp(true);
        let mut enc = LowDelayPEncoder::new(w, h, 30, 0)
            .expect("encoder")
            .with_tree(cfg);
        let mut aus = Vec::new();
        for (y, cb, cr) in &frames {
            aus.push(enc.encode_frame(&YuvFrame { y, cb, cr }).expect("frame").au);
        }
        let units = crate::nal::collect_nal_units(&aus[0]).expect("walk");
        let sps = crate::sps::SeqParameterSet::parse(&units[1].rbsp).expect("sps");
        let pps = crate::pps::PicParameterSet::parse(&units[2].rbsp).expect("pps");
        assert!(pps.tiles_enabled_flag && pps.entropy_coding_sync_enabled_flag);
        assert!(pps.loop_filter_across_tiles_enabled_flag);
        assert_eq!(pps.tiles.num_tile_columns_minus1, 1);
        assert_eq!(pps.tiles.num_tile_rows_minus1, 1);
        // 6 x 4 CTBs, 2x2 tiles of 3x2 CTBs, WPP: 2 rows per tile => 8
        // subsets => 7 entry points on the I and the P slice.
        let p_units = crate::nal::collect_nal_units(&aus[1]).expect("walk");
        for u in [&units[3], &p_units[0]] {
            let header = crate::slice::SliceSegmentHeader::parse(
                &u.rbsp,
                u.header.nal_unit_type,
                &sps,
                &pps,
            )
            .expect("slice header");
            let eps = header.entry_point_offsets.expect("entry points present");
            assert_eq!(eps.num_entry_point_offsets, 7);
        }
        let stream: Vec<u8> = aus.concat();
        assert_eq!(decode_annexb_sequence(&stream).expect("decode").len(), 2);
    }

    /// The tile-parallel pass 1 is bit-identical to the serial one
    /// (2x2 tiles, RDOQ + WPP + AQ + filters, P and B slices).
    #[test]
    fn tree_parallel_wavefront_matches_serial() {
        // A single-tile WPP picture decides its CTB rows in a wavefront
        // on several workers: the bytes equal the serial pass — P / B
        // GOP (merge candidates reach the above-right CTB) and a deep
        // 4:4:4 intra still with RDOQ (the per-row shadow contexts
        // start from the row above's §9.3.2.2 storage).
        use crate::encoder::inter::LowDelayPEncoder;
        let (w, h) = (112, 80);
        let frames = scene(w, h, 3);
        let cfg = TreeCfg::new(16)
            .expect("ctb")
            .with_wpp(true)
            .with_rdoq(true)
            .with_sign_hiding(true)
            .with_intra_rd(2);
        let encode = |threads: usize| -> Vec<u8> {
            let mut enc = LowDelayPEncoder::new(w, h, 30, 0)
                .expect("encoder")
                .with_tree(cfg)
                .with_b_slices(true)
                .with_aq(1)
                .with_loop_filters(LoopFilterCfg::all())
                .with_threads(threads);
            let mut stream = Vec::new();
            for (y, cb, cr) in &frames {
                stream.extend_from_slice(
                    &enc.encode_frame(&YuvFrame { y, cb, cr }).expect("frame").au,
                );
            }
            stream
        };
        let serial = encode(1);
        assert_eq!(encode(3), serial, "3 workers");
        assert_eq!(encode(8), serial, "8 workers");
        assert_eq!(decode_annexb_sequence(&serial).expect("decode").len(), 3);

        let fmt = SampleFmt::new(3, 10).expect("format");
        let planes = planes_fmt(&fmt, 96, 80);
        let intra = |threads: usize| -> Vec<u8> {
            let sps = SpsCfg {
                min_cb_log2: 3,
                tree: Some(
                    TreeCfg::new(32)
                        .expect("ctb")
                        .with_wpp(true)
                        .with_rdoq(true),
                ),
                fmt,
                threads,
                cu_qp_delta: true,
                ..SpsCfg::legacy(1)
            };
            crate::encoder::intra::encode_idr_intra_au_wide(
                [&planes[0], &planes[1], &planes[2]],
                96,
                80,
                22,
                &sps,
                &LoopFilterCfg::all(),
                2,
                None,
            )
            .expect("encode")
            .au
        };
        let serial = intra(1);
        assert_eq!(intra(2), serial, "intra, 2 workers");
        assert_eq!(intra(4), serial, "intra, 4 workers");
    }

    #[test]
    fn tree_parallel_tiles_match_serial() {
        use crate::encoder::inter::LowDelayPEncoder;
        let (w, h) = (96, 64);
        let frames = scene(w, h, 3);
        let cfg = TreeCfg::new(16)
            .expect("ctb")
            .with_tiles(TileLayout::uniform(2, 2))
            .with_wpp(true)
            .with_rdoq(true);
        let encode = |threads: usize| -> Vec<u8> {
            let mut enc = LowDelayPEncoder::new(w, h, 30, 0)
                .expect("encoder")
                .with_tree(cfg)
                .with_b_slices(true)
                .with_aq(1)
                .with_loop_filters(LoopFilterCfg::all())
                .with_threads(threads);
            let mut stream = Vec::new();
            for (y, cb, cr) in &frames {
                stream.extend_from_slice(
                    &enc.encode_frame(&YuvFrame { y, cb, cr }).expect("frame").au,
                );
            }
            stream
        };
        let serial = encode(1);
        assert_eq!(encode(4), serial, "4 workers");
        assert_eq!(encode(2), serial, "2 workers");
        assert_eq!(decode_annexb_sequence(&serial).expect("decode").len(), 3);
    }

    #[test]
    fn tree_scaling_lists_roundtrip() {
        for (mode, b) in [(1u8, false), (2, true), (3, false)] {
            let cfg = TreeCfg::new(32)
                .expect("ctb 32")
                .with_scaling_lists(mode)
                .with_rdoq(true)
                .with_sign_hiding(mode == 3);
            assert_gop_tree_roundtrip_cfg(cfg, b, LoopFilterCfg::all(), 0);
        }
    }

    #[test]
    fn tree_deep_ladder_elects_small_pus_and_deep_splits() {
        use crate::encoder::inter::LowDelayPEncoder;
        let (w, h) = (96, 64);
        let frames = scene(w, h, 3);
        let cfg = TreeCfg::new(32).expect("ctb 32").with_tu_depth(2, 2);
        let mut enc = LowDelayPEncoder::new(w, h, 24, 0)
            .expect("encoder")
            .with_tree(cfg);
        let mut rect = 0usize;
        for (y, cb, cr) in &frames {
            let f = enc.encode_frame(&YuvFrame { y, cb, cr }).expect("frame");
            rect += f.stats.rect;
        }
        assert!(rect > 0, "no rectangular PU elected at all");
    }

    #[test]
    fn tree_p_gop_roundtrips_ctb32() {
        assert_gop_tree_roundtrip(32, false, LoopFilterCfg::off(), 0);
    }

    #[test]
    fn tree_b_gop_roundtrips_ctb64() {
        assert_gop_tree_roundtrip(64, true, LoopFilterCfg::off(), 0);
    }

    #[test]
    fn tree_gop_with_filters_and_aq_roundtrips() {
        assert_gop_tree_roundtrip(32, false, LoopFilterCfg::all(), 2);
    }

    #[test]
    fn tree_pyramid_roundtrips_ctb64() {
        use crate::encoder::pyramid::PyramidEncoder;
        let (w, h) = (96, 64);
        let frames = scene(w, h, 5);
        let mut enc = PyramidEncoder::new(w, h, 30, 4)
            .expect("encoder")
            .with_tree(TreeCfg::new(64).expect("legal ctb"));
        let mut stream = Vec::new();
        let mut recons_by_display: Vec<Option<FrameRecon>> = vec![None; frames.len()];
        let push = |aus: Vec<crate::encoder::pyramid::PyramidAu>,
                    stream: &mut Vec<u8>,
                    recons: &mut Vec<Option<FrameRecon>>| {
            for au in aus {
                stream.extend_from_slice(&au.au);
                recons[au.display_order] = Some(au.recon);
            }
        };
        for (y, cb, cr) in &frames {
            let aus = enc.encode_frame(&YuvFrame { y, cb, cr }).expect("frame");
            push(aus, &mut stream, &mut recons_by_display);
        }
        push(enc.flush(), &mut stream, &mut recons_by_display);
        let decoded = decode_annexb_sequence(&stream).expect("decode");
        assert_eq!(decoded.len(), frames.len());
        for (i, f) in decoded.iter().enumerate() {
            let rec = recons_by_display[i].as_ref().expect("coded");
            let planar = f.picture.to_planar_u8().unwrap();
            assert_eq!(planar[..w * h], rec.y[..], "frame {i} luma");
            assert_eq!(planar[w * h..w * h + w * h / 4], rec.cb[..]);
            assert_eq!(planar[w * h + w * h / 4..], rec.cr[..]);
        }
    }

    #[test]
    fn tree_gop_with_amp_composes() {
        use crate::encoder::inter::LowDelayPEncoder;
        let (w, h) = (96, 64);
        let frames = scene(w, h, 3);
        let mut enc = LowDelayPEncoder::new(w, h, 28, 0)
            .expect("encoder")
            .with_tree(TreeCfg::new(32).expect("legal ctb"))
            .with_amp(true);
        let mut stream = Vec::new();
        let mut recons = Vec::new();
        for (y, cb, cr) in &frames {
            let f = enc.encode_frame(&YuvFrame { y, cb, cr }).expect("frame");
            stream.extend_from_slice(&f.au);
            recons.push(f.recon);
        }
        let decoded = decode_annexb_sequence(&stream).expect("decode");
        for (f, rec) in decoded.iter().zip(recons.iter()) {
            let planar = f.picture.to_planar_u8().unwrap();
            assert_eq!(planar[..w * h], rec.y[..]);
        }
    }
}
