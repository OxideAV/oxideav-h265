//! General coding-tree encoder — recursive §7.3.8.4 coding quadtrees
//! at CTB 16 / 32 / 64 with rate-distortion-elected `split_cu_flag`,
//! recursive §7.3.8.8 residual quadtrees
//! (`max_transform_hierarchy_depth_*` > 0), §8.6.4 DST-VII 4x4 intra
//! luma TUs, and `MinCbSizeY == 8` coding units (intra `PART_NxN`
//! with four 4x4 luma PBs included).
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
    amvp_search, bin_part_mode, blit, choose_pu, encode_merge_idx, encode_pu_syntax_at, extract,
    merge_pu, part_is_amp, part_is_horizontal, predict_block, store, sub_block, FrameRecon,
    FrameStats, PuChooseCtx, PuSyntax, RefPlanes, SliceLfSignalling, SliceSpec, YuvFrame,
};
use crate::encoder::intra::{
    chroma_qp_420, encode_cu_qp_delta, forward_transform, quantize, rate_proxy, IntraEncodeError,
    IntraEncodedAu, SpsCfg,
};
use crate::encoder::loopfilter::{
    encode_sao_ctb, filter_frame, FilterInput, LoopFilterCfg, TreeLayout,
};
use crate::encoder::residual::encode_residual_coding;
use crate::intra_mode_field::{IntraModeField, Neighbour};
use crate::intra_pred::{
    intra_predict_with_substitution, Component as PredComponent, IntraPredParams,
    MarkedReferenceSamples,
};
use crate::motion::{MotionCell, MotionField};
use crate::pu_mv::{pu_partitions, resolve_pu_motion, PartMode, PuGeometry, PuMotion, PuMvContext};
use crate::residual::{residual_coding_scan_idx, ResidualCodingParams};
use crate::slice_data::SaoCtbParams;
use crate::transform::{forward_dst4_1d, residual_block, BlockParams, Component, PredMode};

/// Fixed 8-bit depth.
const BIT_DEPTH: u32 = 8;
/// `MaxNumMergeCand` (§7.4.7.1) — `five_minus_max_num_merge_cand = 0`.
const MAX_MERGE: usize = 5;
/// The z-order offsets of the four quadrants of a split node.
const Z_OFFSETS: [(usize, usize); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];

/// The quadtree coder's stream geometry: `CtbLog2SizeY`, the residual
/// quadtree depths, always `MinCbLog2SizeY == 3` and
/// `MaxTbLog2SizeY == min( CtbLog2SizeY, 5 )`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeCfg {
    /// `CtbLog2SizeY` (4..=6).
    pub ctb_log2: u32,
    /// `max_transform_hierarchy_depth_intra`.
    pub th_depth_intra: u32,
    /// `max_transform_hierarchy_depth_inter`.
    pub th_depth_inter: u32,
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
        })
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

/// One residual-quadtree node's coded levels. Leaves at
/// `log2TrafoSize >= 3` carry their own half-size chroma blocks;
/// 4x4 luma leaves defer chroma to their parent split node
/// (§7.3.8.10 `blkIdx == 3`), which carries the CU-quadrant 4x4
/// chroma blocks itself.
enum TuNode {
    /// `split_transform_flag == 0` leaf: the luma levels, plus the
    /// chroma levels when `log2TrafoSize > 2`.
    Leaf {
        y: Vec<i32>,
        cb: Vec<i32>,
        cr: Vec<i32>,
    },
    /// `split_transform_flag == 1` node: four z-order children, plus
    /// the deferred 4x4 chroma blocks when the children are 4x4 luma
    /// leaves (`log2TrafoSize == 3` here).
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
    /// (four PBs). `modes[0]` is replicated for 2Nx2N.
    Intra { modes: [u8; 4], nxn: bool },
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
    amp: bool,
    width: usize,
    height: usize,
    src: [&'a [u8]; 3],
    qp: i32,
    aq_deltas: &'a [i32],
    b_slice: bool,
    intra_slice: bool,
    refs_l0: &'a [RefPlanes],
    refs_l1: &'a [RefPlanes],
    mv_ctx: Option<&'a PuMvContext<'a>>,
    two_sided: bool,
    tiling: &'a PictureTiling,
}

impl SliceCtx<'_> {
    fn lambda_of(&self, q: i32) -> u64 {
        1u64 << (q.unsigned_abs().saturating_sub(9) / 3)
    }

    fn ctbs_x(&self) -> usize {
        self.width.div_ceil(1 << self.cfg.ctb_log2)
    }

    fn ctbs_y(&self) -> usize {
        self.height.div_ceil(1 << self.cfg.ctb_log2)
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

/// The picture state the decisions mutate (and snapshots roll back).
struct EncState {
    recon: FrameRecon,
    field: MotionField,
    modes: IntraModeField,
    /// Per-4x4-cell `CtDepth` (−1 until coded).
    ct_depth: Vec<i8>,
    /// Per-4x4-cell `cu_skip_flag`.
    skip: Vec<u8>,
    w_cells: usize,
    h_cells: usize,
}

impl EncState {
    fn new(width: usize, height: usize, ctb_log2: u32) -> Self {
        let (cw, ch) = (width / 2, height / 2);
        let w_cells = width.div_ceil(4);
        let h_cells = height.div_ceil(4);
        Self {
            recon: FrameRecon {
                y: vec![0u8; width * height],
                cb: vec![0u8; cw * ch],
                cr: vec![0u8; cw * ch],
                motion_field: None,
            },
            field: MotionField::new(width, height),
            modes: IntraModeField::new(width, height, ctb_log2),
            ct_depth: vec![-1; w_cells * h_cells],
            skip: vec![0; w_cells * h_cells],
            w_cells,
            h_cells,
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
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
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
        let (cw, cx0, cy0) = (ctx.width / 2, x0 / 2, y0 / 2);
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
            cb: rect_copy(&self.recon.cb, cw, cx0, cy0, w / 2, h / 2),
            cr: rect_copy(&self.recon.cr, cw, cx0, cy0, w / 2, h / 2),
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
        let (cw, cx0, cy0) = (ctx.width / 2, x0 / 2, y0 / 2);
        rect_paste(&mut self.recon.y, ctx.width, x0, y0, w, h, &snap.y);
        rect_paste(&mut self.recon.cb, cw, cx0, cy0, w / 2, h / 2, &snap.cb);
        rect_paste(&mut self.recon.cr, cw, cx0, cy0, w / 2, h / 2, &snap.cr);
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
/// eq. 8-316 synthesis, at the encoder's DCT normalization shifts).
fn forward_transform_dst4(res: &[i32]) -> Vec<i32> {
    let shift1 = 2 + BIT_DEPTH - 9; // log2TbS + BitDepth − 9
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
) -> (Vec<i32>, Vec<u8>) {
    let res: Vec<i32> = src.iter().zip(pred.iter()).map(|(&s, &p)| s - p).collect();
    let coef = if pred_mode == PredMode::Intra && component == Component::Luma && n == 4 {
        forward_transform_dst4(&res)
    } else {
        forward_transform(&res, n)
    };
    let levels = quantize(&coef, n, qp);
    let recon: Vec<u8> = if levels.iter().all(|&v| v == 0) {
        pred.iter().map(|&p| p.clamp(0, 255) as u8).collect()
    } else {
        let r = residual_block(
            &levels,
            None,
            BlockParams {
                n_tbs: n,
                q_p: qp,
                component,
                pred_mode,
                bit_depth: BIT_DEPTH as u8,
                extended_precision: false,
                transquant_bypass: false,
                transform_skip: false,
                transform_skip_rotation_enabled: false,
            },
        )
        .expect("legal block params");
        pred.iter()
            .zip(r.iter())
            .map(|(&p, &d)| (p + d).clamp(0, 255) as u8)
            .collect()
    };
    (levels, recon)
}

fn ssd_u8(a: &[u8], b: &[i32]) -> u64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = i64::from(x) - i64::from(y);
            (d * d) as u64
        })
        .sum()
}

// ---------------------------------------------------------------------
// Intra CU coding
// ---------------------------------------------------------------------

fn pred_params(mode: u8, cidx: PredComponent) -> IntraPredParams {
    IntraPredParams {
        pred_mode_intra: mode,
        cidx,
        bit_depth: BIT_DEPTH as u8,
        bit_depth_luma: BIT_DEPTH as u8,
        intra_smoothing_disabled: false,
        strong_intra_smoothing_enabled: false,
        chroma_array_type_3: false,
        disable_boundary_filter: false,
    }
}

/// Gather the §8.4.4.2.1 marked luma reference array for an `n`-TB at
/// `(x0, y0)` from the frame reconstruction, availability per §6.4.1.
fn gather_luma_refs(
    ctx: &SliceCtx<'_>,
    recon_y: &[u8],
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

/// Chroma twin of [`gather_luma_refs`] (`n` chroma samples at chroma
/// `(cx0, cy0)`; availability tested at the co-located luma).
fn gather_chroma_refs(
    ctx: &SliceCtx<'_>,
    plane: &[u8],
    cx0: usize,
    cy0: usize,
    n: usize,
) -> MarkedReferenceSamples {
    let cw = ctx.width / 2;
    let ch = ctx.height / 2;
    let get = |x: i64, y: i64| -> (i32, bool) {
        if x < 0 || y < 0 || x >= cw as i64 || y >= ch as i64 {
            return (0, false);
        }
        if ctx.z_avail(cx0 * 2, cy0 * 2, x * 2, y * 2) {
            (i32::from(plane[y as usize * cw + x as usize]), true)
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
fn search_best_mode(marked: &MarkedReferenceSamples, src: &[i32]) -> (u8, Vec<i32>) {
    let mut best = (0u8, Vec::new());
    let mut best_cost = u64::MAX;
    for mode in 0..=34u8 {
        let pred = intra_predict_with_substitution(marked, &pred_params(mode, PredComponent::Luma))
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

/// Pick the 64x64 intra CU's single PB mode: the CU is forced to four
/// 32x32 TUs, so each candidate mode is scored by SAD over the four
/// TU predictions with in-CU (not-yet-reconstructed) reference reads
/// substituted from the SOURCE picture — a search heuristic only; the
/// actual coding pass predicts from the real reconstruction.
fn search_mode_64(ctx: &SliceCtx<'_>, st: &EncState, x0: usize, y0: usize) -> u8 {
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
            let pred =
                intra_predict_with_substitution(&marked, &pred_params(mode, PredComponent::Luma))
                    .expect("legal prediction params");
            let src = extract(ctx.src[0], ctx.width, tx, ty, 32);
            cost += src
                .iter()
                .zip(pred.iter())
                .map(|(&s, &p)| u64::from(s.abs_diff(p)))
                .sum::<u64>();
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
fn code_intra_luma_tb(
    ctx: &SliceCtx<'_>,
    st: &mut EncState,
    x: usize,
    y: usize,
    n: usize,
    mode: u8,
    qp_y: u32,
) -> (Vec<i32>, u64) {
    let marked = gather_luma_refs(ctx, &st.recon.y, x, y, n);
    let pred = intra_predict_with_substitution(&marked, &pred_params(mode, PredComponent::Luma))
        .expect("legal prediction params");
    let src = extract(ctx.src[0], ctx.width, x, y, n);
    let (levels, recon) = code_tb(&src, &pred, n, qp_y, Component::Luma, PredMode::Intra);
    let dist = ssd_u8(&recon, &src);
    store(&mut st.recon.y, ctx.width, x, y, n, &recon);
    (levels, dist)
}

/// Code one intra chroma TB pair at chroma `(cx, cy)` size `n`.
fn code_intra_chroma_tbs(
    ctx: &SliceCtx<'_>,
    st: &mut EncState,
    cx: usize,
    cy: usize,
    n: usize,
    mode_c: u8,
    qp_c: u32,
) -> (Vec<i32>, Vec<i32>, u64) {
    let cw = ctx.width / 2;
    let mut do_plane = |plane_idx: usize, comp: Component, pc: PredComponent| -> (Vec<i32>, u64) {
        let recon_plane = match plane_idx {
            1 => &st.recon.cb,
            _ => &st.recon.cr,
        };
        let marked = gather_chroma_refs(ctx, recon_plane, cx, cy, n);
        let pred = intra_predict_with_substitution(&marked, &pred_params(mode_c, pc))
            .expect("legal prediction params");
        let src = extract(ctx.src[plane_idx], cw, cx, cy, n);
        let (levels, recon) = code_tb(&src, &pred, n, qp_c, comp, PredMode::Intra);
        let dist = ssd_u8(&recon, &src);
        let recon_plane = match plane_idx {
            1 => &mut st.recon.cb,
            _ => &mut st.recon.cr,
        };
        store(recon_plane, cw, cx, cy, n, &recon);
        (levels, dist)
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

    let leaf_eval = |st: &mut EncState| -> (TuNode, u64, u64) {
        let n = 1usize << log2;
        let (y_lv, d_y) = code_intra_luma_tb(ctx, st, x, y, n, mode, qp_y);
        let (cb, cr, d_c) = if log2 >= 3 {
            code_intra_chroma_tbs(ctx, st, x / 2, y / 2, n / 2, mode_c, qp_c)
        } else {
            (Vec::new(), Vec::new(), 0)
        };
        // cbf bins: luma always signalled on intra; chroma pair when
        // log2 > 2 at this node.
        let rate = rate_proxy(&y_lv)
            + if log2 >= 3 {
                rate_proxy(&cb) + rate_proxy(&cr) + 2
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
        // Deferred 4x4 chroma at the log2 == 3 split parent.
        let (cb, cr) = if log2 == 3 {
            let (cb, cr, d_c) = code_intra_chroma_tbs(ctx, st, x / 2, y / 2, 4, mode_c, qp_c);
            dist += d_c;
            rate += rate_proxy(&cb) + rate_proxy(&cr) + 2;
            (cb, cr)
        } else {
            rate += 2; // this node's cbf_cb / cbf_cr pair
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
/// state (reconstruction + mode field + cells): `PART_2Nx2N` with the
/// RD-elected RQT, and additionally `PART_NxN` (four 4x4 PBs, DST
/// TUs) at `MinCbSizeY`.
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
    let n = 1usize << log2;
    let qp_y = ctb_qp as u32;
    let qp_c = chroma_qp_420(ctb_qp);
    let lambda = ctx.lambda_of(ctb_qp);
    // Per-CU syntax overhead proxy: pred_mode (P/B) + part_mode (at
    // MinCb) + luma mode ~6/PB + chroma mode 1.
    let base_bins = u64::from(!ctx.intra_slice) + u64::from(log2 == ctx.cfg.min_cb_log2());

    let before = st.snapshot(ctx, x0, y0, n);

    // ---- PART_2Nx2N ----
    let mode = if log2 == 6 {
        search_mode_64(ctx, st, x0, y0)
    } else {
        let marked = gather_luma_refs(ctx, &st.recon.y, x0, y0, n);
        let src = extract(ctx.src[0], ctx.width, x0, y0, n);
        search_best_mode(&marked, &src).0
    };
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
        mode,
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
            let (m, _) = search_best_mode(&marked, &src);
            let (lv, d) = code_intra_luma_tb(ctx, st, px, py, 4, m, qp_y);
            // §8.4.2: later PBs' candidate lists see this PB's mode.
            st.modes.record_intra_pb(px, py, 4, m, false);
            pb_modes[k] = m;
            rate += rate_proxy(&lv) + 1;
            dist += d;
            luma_lv.push(lv);
        }
        let (cb, cr, d_c) = code_intra_chroma_tbs(ctx, st, x0 / 2, y0 / 2, 4, pb_modes[0], qp_c);
        dist += d_c;
        rate += rate_proxy(&cb) + rate_proxy(&cr) + 2;
        let cost_nxn = dist + lambda * (rate + base_bins + 4 * 7);
        if cost_nxn < cost_2n {
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
            CuCoded {
                x0,
                y0,
                log2,
                kind: TreeCuKind::Intra {
                    modes: pb_modes,
                    nxn: true,
                },
                motions: Vec::new(),
                tree: Some(TuNode::Split { children, cb, cr }),
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
        let (y_lv, y_rc) = code_tb(&src_y, &pred_y, n, qp_y, Component::Luma, PredMode::Inter);
        let mut dist = ssd_u8(&y_rc, &src_y);
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
            let (cb_lv, cb_rc) =
                code_tb(&src_cb, &pred_cb, hc, qp_c, Component::Cb, PredMode::Inter);
            let src_cr = sub_block(bufs.src_cr, n_cb / 2, x / 2, y / 2, hc);
            let pred_cr = sub_block(bufs.pred_cr, n_cb / 2, x / 2, y / 2, hc);
            let (cr_lv, cr_rc) =
                code_tb(&src_cr, &pred_cr, hc, qp_c, Component::Cr, PredMode::Inter);
            dist += ssd_u8(&cb_rc, &src_cb) + ssd_u8(&cr_rc, &src_cr);
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
            y: vec![0u8; n * n],
            cb: vec![0u8; (n / 2) * (n / 2)],
            cr: vec![0u8; (n / 2) * (n / 2)],
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
            let (cb_lv, cb_rc) =
                code_tb(&src_cb, &pred_cb, 4, qp_c, Component::Cb, PredMode::Inter);
            let src_cr = sub_block(bufs.src_cr, n_cb / 2, x / 2, y / 2, 4);
            let pred_cr = sub_block(bufs.pred_cr, n_cb / 2, x / 2, y / 2, 4);
            let (cr_lv, cr_rc) =
                code_tb(&src_cr, &pred_cr, 4, qp_c, Component::Cr, PredMode::Inter);
            dist += ssd_u8(&cb_rc, &src_cb) + ssd_u8(&cr_rc, &src_cr);
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
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
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
    let n = 1usize << log2;
    let qp_y = ctb_qp as u32;
    let qp_c = chroma_qp_420(ctb_qp);
    let lambda = ctx.lambda_of(ctb_qp);
    let lambda_me = crate::encoder::rate::isqrt_u64(lambda);
    let mv_ctx = ctx.mv_ctx.expect("inter slice has motion context");
    let (cw, cx0, cy0) = (ctx.width / 2, x0 / 2, y0 / 2);
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
            let clip =
                |v: &[i32]| -> Vec<u8> { v.iter().map(|&p| p.clamp(0, 255) as u8).collect() };
            let local = LocalRecon {
                y: clip(&pred.luma),
                cb: clip(&pred.cb),
                cr: clip(&pred.cr),
            };
            let dist =
                ssd_u8(&local.y, &src[0]) + ssd_u8(&local.cb, &src[1]) + ssd_u8(&local.cr, &src[2]);
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
                let pred = predict_block(ctx.refs_l0, ctx.refs_l1, x0, y0, n, n, &m, false);
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
        let merge_pred = predict_block(
            ctx.refs_l0,
            ctx.refs_l1,
            x0,
            y0,
            n,
            n,
            &best_merge_motion,
            true,
        );
        let clip = |v: &[i32]| -> Vec<u8> { v.iter().map(|&p| p.clamp(0, 255) as u8).collect() };
        let skip_recon = LocalRecon {
            y: clip(&merge_pred.luma),
            cb: clip(&merge_pred.cb),
            cr: clip(&merge_pred.cr),
        };
        let skip_dist = ssd_u8(&skip_recon.y, &src[0])
            + ssd_u8(&skip_recon.cb, &src[1])
            + ssd_u8(&skip_recon.cr, &src[2]);
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
        let amvp_pred = predict_block(ctx.refs_l0, ctx.refs_l1, x0, y0, n, n, &amvp_motion, true);
        let (a_tree, a_recon, a_dist, a_rate) = code_residual(&amvp_pred, true);
        cands.push(InterCand {
            kind: TreeCuKind::Amvp { pu: amvp_syntax },
            motions: vec![amvp_motion],
            tree: a_tree,
            recon: a_recon,
            cost: a_dist + lambda * (a_rate + amvp_rate + 2),
        });
    }

    // ---- two-PU partitions (log2 > 3: 8x4 / 4x8 PUs stay out) ----
    if log2 > 3 {
        let mut parts = vec![PartMode::Part2NxN, PartMode::PartNx2N];
        if ctx.amp {
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
                let p = predict_block(
                    ctx.refs_l0,
                    ctx.refs_l1,
                    r.x_pb,
                    r.y_pb,
                    r.n_pb_w,
                    r.n_pb_h,
                    &motion,
                    true,
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
            search_mode_64(ctx, st, x0, y0)
        } else {
            let marked = gather_luma_refs(ctx, &st.recon.y, x0, y0, n);
            search_best_mode(&marked, &src[0]).0
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
        TreeCuKind::Intra { modes, nxn } => {
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
) -> (CuNode, u64) {
    let n = 1usize << log2;
    let fits = x0 + n <= ctx.width && y0 + n <= ctx.height;
    let min_cb = ctx.cfg.min_cb_log2();

    let code_leaf = |ctx: &SliceCtx<'_>, st: &mut EncState| -> (CuNode, u64) {
        let ctb_idx = (y0 >> ctx.cfg.ctb_log2) * ctx.ctbs_x() + (x0 >> ctx.cfg.ctb_log2);
        let ctb_qp = (ctx.qp + ctx.aq_deltas[ctb_idx]).clamp(0, 51);
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
                let (node, c) = code_quadtree(ctx, st, cx, cy, log2 - 1, depth + 1);
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
    let ctb_idx = (y0 >> ctx.cfg.ctb_log2) * ctx.ctbs_x() + (x0 >> ctx.cfg.ctb_log2);
    let ctb_qp = (ctx.qp + ctx.aq_deltas[ctb_idx]).clamp(0, 51);
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

fn qp_walk(ctx: &SliceCtx<'_>, plans: &[CuNode], aq_on: bool) -> QpWalk {
    let w_cells = ctx.width.div_ceil(4);
    let h_cells = ctx.height.div_ceil(4);
    let mut walk = QpWalk {
        cells: vec![0; w_cells * h_cells],
        w_cells,
        descs: Vec::new(),
    };
    let mut qp_prev = ctx.qp; // qPY_PREV (SliceQpY at the slice start)
    for (ctb_idx, plan) in plans.iter().enumerate() {
        let ctb_qp = (ctx.qp + ctx.aq_deltas[ctb_idx]).clamp(0, 51);
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
                cb_qp_offset: 0,
                cr_qp_offset: 0,
                bit_depth_luma: 8,
                bit_depth_chroma: 8,
                chroma_array_type: 1,
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
            let (l_depth, l_avail) = self.st.nb_ct_depth(x0, y0, Neighbour::Left);
            let (a_depth, a_avail) = self.st.nb_ct_depth(x0, y0, Neighbour::Above);
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
            TreeCuKind::Intra { modes, nxn } => {
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
                // intra_chroma_pred_mode = 4 (derived): bin "0".
                self.cabac
                    .encode_decision(self.w, &mut self.ctxs.intra_chroma_pred_mode[0], 0);
            }
        }
        // ---- transform tree ----
        let cu_is_intra = matches!(cu.kind, TreeCuKind::Intra { .. });
        let (intra_split, modes4) = match &cu.kind {
            TreeCuKind::Intra { modes, nxn } => (*nxn, *modes),
            _ => (false, [0u8; 4]),
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
                max_depth,
                inter_split,
            };
            self.emit_transform_tree(&tt, tree, x0, y0, cu.log2, 0, true, true, qg);
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
    /// root per the depth-0 read rule).
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
        qg: &mut QgState,
    ) {
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
        // Chroma cbf pair, present when log2 > 2, gated on the parent.
        let cbf_cb = node.cbf_cb();
        let cbf_cr = node.cbf_cr();
        if log2 > 2 {
            if depth == 0 || parent_cbf_cb {
                self.cabac.encode_decision(
                    self.w,
                    &mut self.ctxs.cbf_chroma[cbf_cb_ctx_inc(depth) as usize],
                    u8::from(cbf_cb),
                );
            }
            if depth == 0 || parent_cbf_cr {
                self.cabac.encode_decision(
                    self.w,
                    &mut self.ctxs.cbf_chroma[cbf_cr_ctx_inc(depth) as usize],
                    u8::from(cbf_cr),
                );
            }
        }
        match node {
            TuNode::Split { children, cb, cr } => {
                let half = 1usize << (log2 - 1);
                for (k, &(zx, zy)) in Z_OFFSETS.iter().enumerate() {
                    self.emit_transform_tree(
                        tt,
                        &children[k],
                        x + zx * half,
                        y + zy * half,
                        log2 - 1,
                        depth + 1,
                        cbf_cb,
                        cbf_cr,
                        qg,
                    );
                    // The blkIdx == 3 deferred chroma rides inside the
                    // last child's transform_unit — emitted right after
                    // its luma residual, below.
                    if log2 == 3 && k == 3 {
                        self.emit_deferred_chroma(tt, cb, cr, cbf_cb, cbf_cr, log2, qg);
                    }
                }
            }
            TuNode::Leaf { y: y_lv, cb, cr } => {
                let cbf_luma = TuNode::any_nonzero(y_lv);
                // For log2 == 2 leaves the chroma state is the parent's.
                let (eff_cb, eff_cr) = if log2 > 2 {
                    (cbf_cb, cbf_cr)
                } else {
                    (parent_cbf_cb, parent_cbf_cr)
                };
                let cbf_luma_present = tt.cu_is_intra || depth != 0 || eff_cb || eff_cr;
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
                let cbf_chroma = eff_cb || eff_cr;
                if cbf_luma || cbf_chroma {
                    self.emit_delta_qp(qg);
                    if cbf_luma {
                        let pb_idx = if tt.intra_split {
                            let half = 1usize << (tt.cu.log2 - 1);
                            (usize::from(y.wrapping_sub(tt.cu.y0) >= half) << 1)
                                | usize::from(x.wrapping_sub(tt.cu.x0) >= half)
                        } else {
                            0
                        };
                        let mode = tt.modes4[pb_idx];
                        self.emit_residual(y_lv, log2, 0, tt.cu_is_intra, mode);
                    }
                    if log2 > 2 {
                        let mode_c = tt.modes4[0];
                        if TuNode::any_nonzero(cb) {
                            self.emit_residual(cb, log2 - 1, 1, tt.cu_is_intra, mode_c);
                        }
                        if TuNode::any_nonzero(cr) {
                            self.emit_residual(cr, log2 - 1, 2, tt.cu_is_intra, mode_c);
                        }
                    }
                    // log2 == 2: chroma deferred to blkIdx 3, handled
                    // by the parent (emit_deferred_chroma).
                }
            }
        }
    }

    /// The §7.3.8.10 `blkIdx == 3` deferred-chroma tail (invoked right
    /// after the fourth 4x4 luma child of a `log2TrafoSize == 3`
    /// split node). Also owns the delta_qp when the four luma leaves
    /// were all uncoded but the parent chroma is not.
    #[allow(clippy::too_many_arguments)]
    fn emit_deferred_chroma(
        &mut self,
        tt: &TtCtx<'_>,
        cb: &[i32],
        cr: &[i32],
        cbf_cb: bool,
        cbf_cr: bool,
        parent_log2: u32,
        qg: &mut QgState,
    ) {
        // The last luma leaf's transform_unit fired (and consumed
        // delta_qp) iff its own cbf_luma or the parent chroma was set;
        // the deferred chroma is coded in that same transform_unit.
        let _ = qg;
        let mode_c = tt.modes4[0];
        if cbf_cb {
            self.emit_residual(cb, parent_log2 - 1, 1, tt.cu_is_intra, mode_c);
        }
        if cbf_cr {
            self.emit_residual(cr, parent_log2 - 1, 2, tt.cu_is_intra, mode_c);
        }
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
            scan_idx: residual_coding_scan_idx(cu_is_intra, log2, c_idx, 1, u32::from(mode)),
            sign_data_hiding_enabled_flag: false,
            sign_hidden_suppressed: false,
            transform_skip_sig_ctx: false,
            persistent_rice_adaptation_enabled_flag: false,
            cabac_bypass_alignment_enabled_flag: false,
            extended_precision_processing_flag: false,
            bit_depth: 8,
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
    max_depth: u32,
    inter_split: bool,
}

// ---------------------------------------------------------------------
// Whole-picture drivers
// ---------------------------------------------------------------------

/// The coded picture a slice driver wraps: the slice-data RBSP tail
/// (everything after the slice header), the filtered reconstruction,
/// the loop-filter elections, and the per-CU stats.
struct CodedPicture {
    plans: Vec<CuNode>,
    /// The pass-1 cell / field state (pass 2 reads the ctxInc cells).
    st: EncState,
    recon: FrameRecon,
    stats: FrameStats,
    deblock_on: bool,
    beta_offset_div2: i32,
    tc_offset_div2: i32,
    sao_luma: bool,
    sao_chroma: bool,
    sao_ctbs: Vec<SaoCtbParams>,
}

/// Pass 1 + filters for one picture (I, P or B).
#[allow(clippy::too_many_lines)]
fn code_picture(ctx: &SliceCtx<'_>, lf: &LoopFilterCfg, aq_on: bool) -> CodedPicture {
    let mut st = EncState::new(ctx.width, ctx.height, ctx.cfg.ctb_log2);
    let ctbs_x = ctx.ctbs_x();
    let ctbs_y = ctx.ctbs_y();
    let ctb = 1usize << ctx.cfg.ctb_log2;
    let mut plans: Vec<CuNode> = Vec::with_capacity(ctbs_x * ctbs_y);
    for ctb_idx in 0..ctbs_x * ctbs_y {
        let x0 = (ctb_idx % ctbs_x) * ctb;
        let y0 = (ctb_idx / ctbs_x) * ctb;
        let (node, _cost) = code_quadtree(ctx, &mut st, x0, y0, ctx.cfg.ctb_log2, 0);
        plans.push(node);
    }

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
    let walk = qp_walk(ctx, &plans, aq_on);
    let recon = st.recon.clone();
    let mut out = CodedPicture {
        plans,
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

/// Pass 2: emit the slice data (CTU loop) for a coded picture into `w`
/// (already holding the slice header). `slice_type_raw` per §7.4.7.1.
fn emit_slice_data(
    ctx: &SliceCtx<'_>,
    coded: &CodedPicture,
    w: &mut BitWriter,
    slice_type_raw: u8,
    aq_on: bool,
) {
    let st = &coded.st;
    let mut cabac = CabacEncoder::new();
    let mut ctxs = SliceContexts::init(init_type(slice_type_raw, false), ctx.qp);
    let ctbs_x = ctx.ctbs_x();
    let ctbs_y = ctx.ctbs_y();
    let ctb = 1usize << ctx.cfg.ctb_log2;
    let mut qp_prev = ctx.qp;
    for (ctb_idx, plan) in coded.plans.iter().enumerate() {
        let x0 = (ctb_idx % ctbs_x) * ctb;
        let y0 = (ctb_idx / ctbs_x) * ctb;
        if coded.sao_luma || coded.sao_chroma {
            encode_sao_ctb(
                w,
                &mut cabac,
                &mut ctxs,
                &coded.sao_ctbs[ctb_idx],
                ctb_idx % ctbs_x,
                ctb_idx / ctbs_x,
                coded.sao_luma,
                coded.sao_chroma,
            );
        }
        let ctb_qp = (ctx.qp + ctx.aq_deltas[ctb_idx]).clamp(0, 51);
        let mut qg = QgState {
            coded: false,
            qp_prev,
            ctb_qp,
            enabled: aq_on,
        };
        let mut em = Emitter {
            ctx,
            st,
            w,
            cabac: &mut cabac,
            ctxs: &mut ctxs,
        };
        em.emit_quadtree(plan, x0, y0, ctx.cfg.ctb_log2, 0, &mut qg);
        if aq_on && qg.coded {
            qp_prev = ctb_qp;
        }
        cabac.encode_terminate(w, u8::from(ctb_idx == ctbs_x * ctbs_y - 1));
    }
    w.align_zero();
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
) -> Result<IntraEncodedAu, IntraEncodeError> {
    let tree = cfg.tree.expect("tree config present");
    if width == 0 || height == 0 || width % 16 != 0 || height % 16 != 0 {
        return Err(IntraEncodeError::BadDimensions { width, height });
    }
    if !(0..=51).contains(&qp) {
        return Err(IntraEncodeError::BadQp(qp));
    }
    if aq > 3 {
        return Err(IntraEncodeError::BadAq(aq));
    }
    let check = |plane: &'static str, buf: &[u8], expected: usize| {
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
    check("y", y, width * height)?;
    check("cb", cb, width * height / 4)?;
    check("cr", cr, width * height / 4)?;

    let ctb = 1usize << tree.ctb_log2;
    let aq_deltas = crate::encoder::aq::ctb_aq_deltas(y, width, height, aq, ctb);
    let tiling = make_tiling(width, height, tree.ctb_log2);
    let ctx = SliceCtx {
        cfg: tree,
        amp: cfg.amp,
        width,
        height,
        src: [y, cb, cr],
        qp,
        aq_deltas: &aq_deltas,
        b_slice: false,
        intra_slice: true,
        refs_l0: &[],
        refs_l1: &[],
        mv_ctx: None,
        two_sided: false,
        tiling: &tiling,
    };
    let coded = code_picture(&ctx, lf, aq > 0);

    // ---- slice_segment_header( ) — I slice, IDR ----
    let mut w = BitWriter::new();
    w.put_bit(1); // first_slice_segment_in_pic_flag
    w.put_bit(0); // no_output_of_prior_pics_flag
    w.ue(0); // slice_pic_parameter_set_id
    w.ue(2); // slice_type = I
    if lf.sao() {
        w.put_bit(u8::from(coded.sao_luma));
        w.put_bit(u8::from(coded.sao_chroma));
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
    w.rbsp_trailing_bits();

    emit_slice_data(&ctx, &coded, &mut w, 2, aq > 0);
    let slice_rbsp = w.finish();
    let au = crate::encoder::intra::assemble_idr_au(width, height, cfg, lf, &slice_rbsp);
    Ok(IntraEncodedAu {
        au,
        recon_y: coded.recon.y,
        recon_cb: coded.recon.cb,
        recon_cr: coded.recon.cr,
    })
}

fn make_tiling(width: usize, height: usize, ctb_log2: u32) -> PictureTiling {
    let ctb = 1usize << ctb_log2;
    PictureTiling::new(
        width.div_ceil(ctb) as u32,
        height.div_ceil(ctb) as u32,
        width as u32,
        height as u32,
        ctb_log2,
        2,
        &TilingParams::single_tile(),
    )
    .expect("legal single-tile geometry")
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
    let tiling = make_tiling(width, height, tree.ctb_log2);

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
    let ctx = SliceCtx {
        cfg: tree,
        amp: spec.big_cu,
        width,
        height,
        src: [frame.y, frame.cb, frame.cr],
        qp: spec.qp,
        aq_deltas: &aq_deltas,
        b_slice: spec.b_slice,
        intra_slice: false,
        refs_l0: &refs_l0,
        refs_l1: &refs_l1,
        mv_ctx: Some(&mv_ctx),
        two_sided,
        tiling: &tiling,
    };
    let coded = code_picture(&ctx, spec.lf, spec.aq > 0);

    let lf_sig = SliceLfSignalling {
        cfg: spec.lf,
        deblock_on: coded.deblock_on,
        beta_offset_div2: coded.beta_offset_div2,
        tc_offset_div2: coded.tc_offset_div2,
        sao_luma: coded.sao_luma,
        sao_chroma: coded.sao_chroma,
    };
    let mut w = BitWriter::new();
    crate::encoder::inter::write_inter_slice_header(&mut w, spec, &lf_sig);
    let raw_slice_type: u8 = if spec.b_slice { 0 } else { 1 };
    emit_slice_data(&ctx, &coded, &mut w, raw_slice_type, spec.aq > 0);
    let CodedPicture {
        mut recon,
        st,
        stats,
        ..
    } = coded;
    recon.motion_field = Some(st.field);
    (w.finish(), recon, stats)
}

#[cfg(test)]
mod tests {
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
        use crate::encoder::inter::LowDelayPEncoder;
        let (w, h) = (96, 64);
        let frames = scene(w, h, 4);
        let mut enc = LowDelayPEncoder::new(w, h, 30, 0)
            .expect("encoder")
            .with_tree(TreeCfg::new(ctb).expect("legal ctb"))
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
