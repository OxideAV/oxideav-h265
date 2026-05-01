//! HEVC inter prediction support (§8.5).
//!
//! This module supplies the pieces needed for P- and B-slice inter decode:
//!
//! * [`MotionVector`] + [`PbMotion`] record the per-prediction-block motion
//!   state (reference index, sub-pel MV, availability, bi-prediction flags)
//!   that the CTU walker saves into [`InterState`] and consults for spatial
//!   neighbour derivation (and for collocated TMVP lookups on later
//!   pictures).
//! * [`Dpb`] — decoded picture buffer keyed by picture order count. Each
//!   [`RefPicture`] carries its own `InterState` snapshot so a later slice
//!   can do temporal-MV lookups.
//! * [`RefPicList`] — per-slice reference list resolved from the active
//!   RPS + DPB contents (§8.3.2). B slices expose both L0 and L1.
//! * Merge-candidate construction (§8.5.3.1.2) — spatial candidates A1,
//!   B1, B0, A0, B2 with HEVC-style pruning plus, for B slices, the
//!   temporal candidate (§8.5.3.1.2.3) and combined bi-predictive fillers
//!   (§8.5.3.1.2.4). Zero-MV filler for any remaining slots.
//! * AMVP (§8.5.3.1.6) — spatial candidates + temporal candidate (TMVP)
//!   when the slice-header flag is set; zero-MV padding.
//! * 8-tap luma and 4-tap chroma sub-pel interpolation (§8.5.3.2.2 /
//!   §8.5.3.2.3) with the 16-phase filter tables. Includes bi-prediction
//!   averaging (§8.5.3.3.3) — a uni-pred variant returns the 16-bit
//!   pre-shift samples so a second reference prediction can be combined.
//! * Weighted bi-prediction (§8.5.3.3.4) — optional per-reference
//!   luma/chroma weight + offset passed through from the slice header.
//!
//! Scope limits that remain: deblocking, SAO filtering, tiles, and
//! wavefront parallel processing are all out of scope for the B-slice
//! landing.

#![allow(clippy::needless_range_loop)]

use oxideav_core::Error;
use oxideav_core::Result;

/// 1/4-pel precision motion vector (§8.5.3.2.1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MotionVector {
    pub x: i32,
    pub y: i32,
}

impl MotionVector {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// Per-prediction-block motion state. Indexed per 4×4 luma block in
/// [`InterState`], with one entry per PB grid cell for fast neighbour
/// lookups during merge/AMVP derivation.
#[derive(Clone, Copy, Debug, Default)]
pub struct PbMotion {
    /// Whether the PB has been decoded and has valid motion fields.
    pub valid: bool,
    /// Whether the PB is intra-coded (skips motion).
    pub is_intra: bool,
    /// `cu_skip_flag` value at the time this PB was decoded (§9.3.4.2.2).
    /// Used by the `cu_skip_flag` neighbour-context derivation in the next
    /// CU's CABAC parse: `condTermFlag{L,A} = cu_skip_flag` of the
    /// neighbouring PB. Without this we previously approximated with
    /// `is_inter` which is too lax (every inter PB looked like a skip PB
    /// to the context derivation), shifting CABAC contexts by one slot
    /// for some neighbour configurations.
    pub is_skip: bool,
    /// `pred_flag_l0`: this PB uses a reference from L0.
    pub pred_l0: bool,
    /// `pred_flag_l1`: this PB uses a reference from L1.
    pub pred_l1: bool,
    /// Reference index into L0 (valid iff `pred_l0`).
    pub ref_idx_l0: i8,
    /// Reference index into L1 (valid iff `pred_l1`).
    pub ref_idx_l1: i8,
    pub mv_l0: MotionVector,
    pub mv_l1: MotionVector,
    /// POC of `RefPicListX[ref_idx_l0]` at the time this PB was decoded.
    /// Resolved once and stashed so a later slice's TMVP lookup (§8.5.3.2.9)
    /// can compute `DiffPicOrderCnt(ColPic, refPicListCol[refIdxCol])`
    /// without reconstructing the collocated slice's ref list. Valid only
    /// when `pred_l0`.
    pub ref_poc_l0: i32,
    /// POC of `RefPicListX[ref_idx_l1]`. Valid only when `pred_l1`.
    pub ref_poc_l1: i32,
    /// `true` when `RefPicListX[ref_idx_l0]` was a long-term reference at
    /// the time this PB was decoded. Used by TMVP to enforce the LT-flag
    /// match gate (§8.5.3.2.9 eq. pre-8-204).
    pub ref_lt_l0: bool,
    pub ref_lt_l1: bool,
}

impl PbMotion {
    pub fn intra() -> Self {
        Self {
            valid: true,
            is_intra: true,
            ..Default::default()
        }
    }

    /// Uni-prediction from L0 (P-slice fast path).
    pub fn inter(ref_idx_l0: i8, mv: MotionVector) -> Self {
        Self {
            valid: true,
            is_intra: false,
            pred_l0: true,
            pred_l1: false,
            ref_idx_l0,
            mv_l0: mv,
            ..Default::default()
        }
    }

    /// Uni-prediction from L1 (B-slice).
    pub fn inter_l1(ref_idx_l1: i8, mv: MotionVector) -> Self {
        Self {
            valid: true,
            is_intra: false,
            pred_l0: false,
            pred_l1: true,
            ref_idx_l1,
            mv_l1: mv,
            ..Default::default()
        }
    }

    /// Bi-prediction from both L0 and L1.
    pub fn bi(ref_idx_l0: i8, mv_l0: MotionVector, ref_idx_l1: i8, mv_l1: MotionVector) -> Self {
        Self {
            valid: true,
            is_intra: false,
            is_skip: false,
            pred_l0: true,
            pred_l1: true,
            ref_idx_l0,
            ref_idx_l1,
            mv_l0,
            mv_l1,
            ref_poc_l0: 0,
            ref_poc_l1: 0,
            ref_lt_l0: false,
            ref_lt_l1: false,
        }
    }
}

/// Per-picture motion-field grid on a 4×4 luma block resolution.
#[derive(Clone, Debug)]
pub struct InterState {
    pub grid_w: usize,
    pub grid_h: usize,
    pub pb: Vec<PbMotion>,
}

impl InterState {
    pub fn new(pic_w: u32, pic_h: u32) -> Self {
        let gw = (pic_w as usize).div_ceil(4);
        let gh = (pic_h as usize).div_ceil(4);
        Self {
            grid_w: gw,
            grid_h: gh,
            pb: vec![PbMotion::default(); gw * gh],
        }
    }

    pub fn get(&self, bx: usize, by: usize) -> Option<&PbMotion> {
        if bx < self.grid_w && by < self.grid_h {
            Some(&self.pb[by * self.grid_w + bx])
        } else {
            None
        }
    }

    pub fn set_region(&mut self, x: u32, y: u32, w: u32, h: u32, pb: PbMotion) {
        let bx0 = (x >> 2) as usize;
        let by0 = (y >> 2) as usize;
        let bx1 = ((x + w).div_ceil(4)) as usize;
        let by1 = ((y + h).div_ceil(4)) as usize;
        for by in by0..bx1.max(by1) {
            if by >= self.grid_h {
                break;
            }
            for bx in bx0..bx1 {
                if bx >= self.grid_w {
                    break;
                }
                self.pb[by * self.grid_w + bx] = pb;
            }
        }
    }

    /// Set motion for a rectangular region in luma coordinates. Rounds up
    /// to the enclosing 4×4 grid cells.
    pub fn set_rect(&mut self, x: u32, y: u32, w: u32, h: u32, pb: PbMotion) {
        let bx0 = (x >> 2) as usize;
        let by0 = (y >> 2) as usize;
        let bx1 = ((x + w + 3) >> 2) as usize;
        let by1 = ((y + h + 3) >> 2) as usize;
        for by in by0..by1.min(self.grid_h) {
            for bx in bx0..bx1.min(self.grid_w) {
                self.pb[by * self.grid_w + bx] = pb;
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  Decoded picture buffer
// ---------------------------------------------------------------------------

/// A decoded reference picture entry in the DPB.
///
/// Carries both the reconstructed sample planes and a snapshot of the
/// picture's motion field so later slices can perform temporal motion
/// vector lookups (TMVP, §8.5.3.2.9).
///
/// Chroma plane dimensions are derived from `(SubWidthC, SubHeightC)`
/// per §6.2 Table 6-1: 4:2:0 → `(2, 2)`, 4:2:2 → `(2, 1)`. The
/// `sample_cb`/`sample_cr` accessors clamp on the chroma-plane geometry
/// `(width / SubWidthC, height / SubHeightC)`.
#[derive(Clone, Debug)]
pub struct RefPicture {
    pub poc: i32,
    pub width: u32,
    pub height: u32,
    /// `SubWidthC` per §6.2 Table 6-1 (1 or 2). Only 4:2:0 / 4:2:2 are
    /// in scope today, so this is always 2; tracked for future formats.
    pub sub_x: u32,
    /// `SubHeightC` per §6.2 Table 6-1 (1 or 2). 4:2:0 → 2, 4:2:2 → 1.
    pub sub_y: u32,
    pub luma: Vec<u16>,
    pub cb: Vec<u16>,
    pub cr: Vec<u16>,
    pub luma_stride: usize,
    pub chroma_stride: usize,
    /// Motion-field grid for collocated-MV (TMVP) lookups. Empty when the
    /// picture was intra-only.
    pub inter: InterState,
    /// `true` when this picture is marked as "used for long-term reference"
    /// per §8.3.2. Long-term refs are kept in the DPB until a later slice
    /// evicts them; short-term refs can be rotated out by the
    /// `StRefPicSet` management each picture.
    pub is_long_term: bool,
}

impl RefPicture {
    pub fn sample_luma(&self, x: i32, y: i32) -> u16 {
        let w = self.width as i32;
        let h = self.height as i32;
        let xc = x.clamp(0, w - 1) as usize;
        let yc = y.clamp(0, h - 1) as usize;
        self.luma[yc * self.luma_stride + xc]
    }

    pub fn sample_cb(&self, x: i32, y: i32) -> u16 {
        let w = (self.width / self.sub_x) as i32;
        let h = (self.height / self.sub_y) as i32;
        let xc = x.clamp(0, w - 1) as usize;
        let yc = y.clamp(0, h - 1) as usize;
        self.cb[yc * self.chroma_stride + xc]
    }

    pub fn sample_cr(&self, x: i32, y: i32) -> u16 {
        let w = (self.width / self.sub_x) as i32;
        let h = (self.height / self.sub_y) as i32;
        let xc = x.clamp(0, w - 1) as usize;
        let yc = y.clamp(0, h - 1) as usize;
        self.cr[yc * self.chroma_stride + xc]
    }

    /// Look up the collocated PB motion at luma position `(x, y)` in this
    /// reference picture. Returns `None` if the position is intra or
    /// outside the grid.
    pub fn collocated_motion(&self, x: u32, y: u32) -> Option<PbMotion> {
        let bx = (x >> 2) as usize;
        let by = (y >> 2) as usize;
        let pb = self.inter.get(bx, by)?;
        if !pb.valid || pb.is_intra {
            return None;
        }
        Some(*pb)
    }
}

/// Decoded picture buffer. Very small and naive — a `VecDeque` with POC
/// lookup is enough for the short reference windows used by a simple
/// I/P/B GOP.
#[derive(Clone, Debug, Default)]
pub struct Dpb {
    pub pics: Vec<RefPicture>,
    /// Capacity bound (drops least-recently-inserted when exceeded).
    pub capacity: usize,
}

impl Dpb {
    pub fn new(capacity: usize) -> Self {
        Self {
            pics: Vec::new(),
            capacity: capacity.max(1),
        }
    }

    pub fn push(&mut self, pic: RefPicture) {
        if self.pics.len() >= self.capacity {
            // Drop oldest short-term ref first; LT refs persist until a
            // later slice explicitly demotes them (§8.3.2).
            if let Some(pos) = self.pics.iter().position(|p| !p.is_long_term) {
                self.pics.remove(pos);
            } else {
                self.pics.remove(0);
            }
        }
        self.pics.push(pic);
    }

    pub fn get_by_poc(&self, poc: i32) -> Option<&RefPicture> {
        self.pics.iter().find(|p| p.poc == poc)
    }

    /// Find a DPB entry whose POC matches `poc_lsb` modulo `max_poc_lsb`
    /// — used for long-term refs signalled without the MSB delta
    /// (§8.3.2). Returns the first match; if `expect_long_term` is true,
    /// the returned entry is also marked as LT.
    pub fn find_by_poc_lsb(&self, poc_lsb: u32, max_poc_lsb: u32) -> Option<&RefPicture> {
        let mask = (max_poc_lsb - 1) as i32;
        self.pics.iter().find(|p| (p.poc & mask) as u32 == poc_lsb)
    }

    /// Mark a DPB entry as a long-term reference by POC (§8.3.2). Used when
    /// a slice's LT RPS references a picture still in the DPB.
    pub fn mark_long_term_by_poc(&mut self, poc: i32) {
        if let Some(p) = self.pics.iter_mut().find(|p| p.poc == poc) {
            p.is_long_term = true;
        }
    }
}

/// Reference picture list L0 / L1, resolved against the current slice's
/// RPS and the DPB. Holds owned POC references; the MC path then looks
/// each POC up in the DPB.
#[derive(Clone, Debug, Default)]
pub struct RefPicList {
    pub pocs: Vec<i32>,
}

impl RefPicList {
    pub fn len(&self) -> usize {
        self.pocs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pocs.is_empty()
    }
}

// ---------------------------------------------------------------------------
//  Weighted bi-prediction table
// ---------------------------------------------------------------------------

/// Per-reference weighted-prediction table (§7.4.7.3 / §8.5.3.3.4).
/// Zero denominators and unit weights disable the weighting and fall
/// back to the standard (1,1)/2 bi-pred averaging.
#[derive(Clone, Debug, Default)]
pub struct WeightedPred {
    /// `luma_log2_weight_denom` (0..=7).
    pub luma_denom: u32,
    /// `chroma_log2_weight_denom` (0..=7).
    pub chroma_denom: u32,
    /// Per-L0 entry: `(flag, luma_weight, luma_offset)`.
    pub l0_luma: Vec<(bool, i32, i32)>,
    /// Per-L0 entry: `[(flag, weight, offset)]` for Cb (0) and Cr (1).
    pub l0_chroma: Vec<[(bool, i32, i32); 2]>,
    /// Per-L1 entry.
    pub l1_luma: Vec<(bool, i32, i32)>,
    pub l1_chroma: Vec<[(bool, i32, i32); 2]>,
}

impl WeightedPred {
    pub fn luma_weight_l0(&self, idx: usize) -> Option<(i32, i32)> {
        self.l0_luma
            .get(idx)
            .and_then(|(f, w, o)| f.then_some((*w, *o)))
    }

    pub fn luma_weight_l1(&self, idx: usize) -> Option<(i32, i32)> {
        self.l1_luma
            .get(idx)
            .and_then(|(f, w, o)| f.then_some((*w, *o)))
    }

    pub fn chroma_weight_l0(&self, idx: usize, comp: usize) -> Option<(i32, i32)> {
        self.l0_chroma
            .get(idx)
            .and_then(|c| c.get(comp))
            .and_then(|(f, w, o)| f.then_some((*w, *o)))
    }

    pub fn chroma_weight_l1(&self, idx: usize, comp: usize) -> Option<(i32, i32)> {
        self.l1_chroma
            .get(idx)
            .and_then(|c| c.get(comp))
            .and_then(|(f, w, o)| f.then_some((*w, *o)))
    }
}

// ---------------------------------------------------------------------------
//  Merge candidate list
// ---------------------------------------------------------------------------

/// One merge candidate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MergeCand {
    pub pred_l0: bool,
    pub pred_l1: bool,
    pub ref_idx_l0: i8,
    pub ref_idx_l1: i8,
    pub mv_l0: MotionVector,
    pub mv_l1: MotionVector,
    pub ref_poc_l0: i32,
    pub ref_poc_l1: i32,
    pub ref_lt_l0: bool,
    pub ref_lt_l1: bool,
}

impl MergeCand {
    fn from_pb(pb: PbMotion) -> Self {
        Self {
            pred_l0: pb.pred_l0,
            pred_l1: pb.pred_l1,
            ref_idx_l0: pb.ref_idx_l0,
            ref_idx_l1: pb.ref_idx_l1,
            mv_l0: pb.mv_l0,
            mv_l1: pb.mv_l1,
            ref_poc_l0: pb.ref_poc_l0,
            ref_poc_l1: pb.ref_poc_l1,
            ref_lt_l0: pb.ref_lt_l0,
            ref_lt_l1: pb.ref_lt_l1,
        }
    }

    /// Re-map a collocated-picture PB into a temporal merge candidate that
    /// refers to the current slice's ref-idx 0 in whichever list the
    /// collocated PB used. §8.5.3.1.2.3 / §8.5.3.2.9 requires POC-distance
    /// scaling (eqs. 8-202..8-209) when `currPocDiff != colPocDiff` and
    /// both refs are short-term. The caller is expected to supply the
    /// already-scaled MV via [`from_pb_into_ref0_scaled`]; this plain
    /// helper is kept for the short-GOP case where the scaling factor
    /// collapses to 1.
    pub fn from_pb_into_ref0(pb: PbMotion) -> Self {
        // Force the candidate to reference the first entry of whichever
        // list matches the collocated PB's direction.
        Self {
            pred_l0: pb.pred_l0,
            pred_l1: pb.pred_l1,
            ref_idx_l0: 0,
            ref_idx_l1: 0,
            mv_l0: pb.mv_l0,
            mv_l1: pb.mv_l1,
            ref_poc_l0: pb.ref_poc_l0,
            ref_poc_l1: pb.ref_poc_l1,
            ref_lt_l0: pb.ref_lt_l0,
            ref_lt_l1: pb.ref_lt_l1,
        }
    }

    pub fn to_pb(self) -> PbMotion {
        PbMotion {
            valid: true,
            is_intra: false,
            // Skip status is set by the caller — `to_pb` produces a
            // non-skip merge PB by default; `perform_merge` flips
            // `is_skip = true` for skip CUs.
            is_skip: false,
            pred_l0: self.pred_l0,
            pred_l1: self.pred_l1,
            ref_idx_l0: self.ref_idx_l0,
            ref_idx_l1: self.ref_idx_l1,
            mv_l0: self.mv_l0,
            mv_l1: self.mv_l1,
            ref_poc_l0: self.ref_poc_l0,
            ref_poc_l1: self.ref_poc_l1,
            ref_lt_l0: self.ref_lt_l0,
            ref_lt_l1: self.ref_lt_l1,
        }
    }
}

/// Per-PU spatial-neighbour availability overrides implementing
/// §8.5.3.2.3 partIdx rules and the §6.4.2 PB availability exception.
///
/// The grid (`InterState`) is written at PU granularity as each PU is
/// decoded, so the naive "read the cell at (xNb, yNb)" approach from
/// the partIdx==1 PU would pick up the just-decoded partIdx==0 motion.
/// The spec marks those neighbour slots UNAVAILABLE so this struct
/// lets the caller tell `build_merge_list`/`build_amvp_list` which to
/// suppress.
#[derive(Clone, Copy, Debug, Default)]
pub struct NeighbourContext {
    /// `availableA1 = FALSE` — set for PART_Nx2N / PART_nLx2N /
    /// PART_nRx2N at partIdx == 1.
    pub suppress_a1: bool,
    /// `availableB1 = FALSE` — set for PART_2NxN / PART_2NxnU /
    /// PART_2NxnD at partIdx == 1.
    pub suppress_b1: bool,
    /// `availableA0 = FALSE` — §6.4.2 NxN / partIdx == 1 exception.
    pub suppress_a0: bool,
}

/// Per-slice configuration for the zero-MV padding step of the merge
/// list builder (§8.5.3.2.5). `num_ref_idx` is the spec's `numRefIdx`
/// — `num_ref_idx_l0_active_minus1 + 1` for P slices, or the minimum
/// of `(L0+1, L1+1)` for B slices. The optional `rpl0` / `rpl1` slices
/// supply the per-list reference-picture POCs and long-term flags so
/// the padded candidates carry spec-correct `ref_poc_*` / `ref_lt_*`
/// metadata for downstream TMVP / collocated lookups (§8.5.3.2.9).
/// When the slices are empty the legacy zero-POC behaviour applies
/// (callers patch the metadata via `refresh_pb_ref_poc` instead).
#[derive(Clone, Copy, Debug, Default)]
pub struct MergeZeroPad<'a> {
    pub num_ref_idx: u32,
    pub rpl0: &'a [RefPocEntry],
    pub rpl1: &'a [RefPocEntry],
}

/// Per-slice POC context used by the §8.5.3.2.4 combined-bi-pred check.
/// Carries the (rpl0, rpl1) POC + long-term flags so the spec's
///   `DiffPicOrderCnt(RefPicList0[refIdxL0l0Cand], RefPicList1[refIdxL1l1Cand]) != 0
///        || mvL0l0Cand != mvL1l1Cand`
/// gate can be evaluated. When `rpl0` / `rpl1` are empty the gate
/// degrades to "any (mvL0, mvL1) pair is allowed" — sufficient for the
/// unit tests that don't simulate a full RPL.
#[derive(Clone, Copy, Debug, Default)]
pub struct MergeCombinedCfg<'a> {
    pub rpl0: &'a [RefPocEntry],
    pub rpl1: &'a [RefPocEntry],
}

/// Compare two merge candidates under the spec's "same motion vectors
/// and same reference indices" rule (§8.5.3.2.3). The `ref_poc_*` /
/// `ref_lt_*` shadow fields are *not* part of the equality test —
/// they're derived metadata and may differ for two PBs that the spec
/// considers equivalent (e.g. a TMVP-projected candidate vs a spatial
/// neighbour pointing at the same RPL slot).
fn merge_same_motion(a: MergeCand, b: MergeCand) -> bool {
    a.pred_l0 == b.pred_l0
        && a.pred_l1 == b.pred_l1
        && a.ref_idx_l0 == b.ref_idx_l0
        && a.ref_idx_l1 == b.ref_idx_l1
        && a.mv_l0 == b.mv_l0
        && a.mv_l1 == b.mv_l1
}

fn opt_same_motion(a: Option<MergeCand>, b: Option<MergeCand>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => merge_same_motion(x, y),
        _ => false,
    }
}

/// A temporal merge candidate looked up from a collocated reference
/// picture, with optional pre-scaled MV. If `None` is returned from
/// [`temporal_merge_cand`] there is no temporal candidate.
///
/// Build the merge candidate list for a PB at (xPb, yPb) of size
/// nPbW × nPbH. Follows §8.5.3.1.2. Returns `max_num_merge_cand`
/// candidates (padded with zero-MV if fewer qualify). Set
/// `is_b_slice` to enable combined bi-pred candidates.
///
/// Round 21 (§8.5.3.2.4 / §8.5.3.2.5 audit): the combined-bi-pred and
/// zero-MV stages now follow the spec exactly. See [`build_merge_list_full`]
/// for the form callers should use when they have the slice's RPLs in
/// scope; this thin wrapper preserves the legacy unit-test contract.
pub fn build_merge_list(
    inter: &InterState,
    x_pb: u32,
    y_pb: u32,
    n_pb_w: u32,
    n_pb_h: u32,
    max_num_merge_cand: u32,
    is_b_slice: bool,
    tmvp: Option<MergeCand>,
    nb: NeighbourContext,
) -> Vec<MergeCand> {
    build_merge_list_full(
        inter,
        x_pb,
        y_pb,
        n_pb_w,
        n_pb_h,
        max_num_merge_cand,
        is_b_slice,
        tmvp,
        nb,
        MergeCombinedCfg::default(),
        MergeZeroPad::default(),
    )
}

/// Spec-faithful form of [`build_merge_list`] that accepts the slice's
/// reference-picture POC tables so the §8.5.3.2.4 combined-bi-pred
/// check and §8.5.3.2.5 zero-MV ramp can use the spec's exact rules.
#[allow(clippy::too_many_arguments)]
pub fn build_merge_list_full(
    inter: &InterState,
    x_pb: u32,
    y_pb: u32,
    n_pb_w: u32,
    n_pb_h: u32,
    max_num_merge_cand: u32,
    is_b_slice: bool,
    tmvp: Option<MergeCand>,
    nb: NeighbourContext,
    comb: MergeCombinedCfg<'_>,
    zero: MergeZeroPad<'_>,
) -> Vec<MergeCand> {
    let mut cands: Vec<MergeCand> = Vec::with_capacity(max_num_merge_cand as usize);

    // §8.5.3.2.3 spatial availability: suppress partIdx-disallowed
    // neighbours so we don't pick up the just-decoded sibling PU's
    // motion that the grid now contains.
    let a1 = if nb.suppress_a1 {
        None
    } else {
        fetch_spatial_neighbour(inter, x_pb as i32 - 1, y_pb as i32 + n_pb_h as i32 - 1)
    };
    let b1 = if nb.suppress_b1 {
        None
    } else {
        fetch_spatial_neighbour(inter, x_pb as i32 + n_pb_w as i32 - 1, y_pb as i32 - 1)
    };
    let b0 = fetch_spatial_neighbour(inter, x_pb as i32 + n_pb_w as i32, y_pb as i32 - 1);
    let a0 = if nb.suppress_a0 {
        None
    } else {
        fetch_spatial_neighbour(inter, x_pb as i32 - 1, y_pb as i32 + n_pb_h as i32)
    };
    let b2 = fetch_spatial_neighbour(inter, x_pb as i32 - 1, y_pb as i32 - 1);

    // §8.5.3.2.3 redundancy rules — each candidate is pruned only
    // against a specific set of earlier slots (not "any existing"):
    //   A1: always inserted when available
    //   B1: dropped if equal to A1
    //   B0: dropped if equal to B1
    //   A0: dropped if equal to A1
    //   B2: dropped if equal to A1 or B1; skipped entirely when
    //       availableFlagA0 + A1 + B0 + B1 == 4.
    //
    // Round 21: the equality predicate is the spec's "same motion
    // vectors and same reference indices" — we *don't* compare the
    // shadow `ref_poc_*` / `ref_lt_*` fields, which are derived
    // metadata that can legitimately differ between two PBs the spec
    // considers equivalent for redundancy purposes.
    if let Some(c) = a1 {
        cands.push(c);
    }
    if cands.len() < max_num_merge_cand as usize {
        if let Some(c) = b1 {
            if !opt_same_motion(Some(c), a1) {
                cands.push(c);
            }
        }
    }
    if cands.len() < max_num_merge_cand as usize {
        if let Some(c) = b0 {
            if !opt_same_motion(Some(c), b1) {
                cands.push(c);
            }
        }
    }
    if cands.len() < max_num_merge_cand as usize {
        if let Some(c) = a0 {
            if !opt_same_motion(Some(c), a1) {
                cands.push(c);
            }
        }
    }
    if cands.len() < max_num_merge_cand as usize {
        let full4 = a0.is_some() && a1.is_some() && b0.is_some() && b1.is_some();
        if !full4 {
            if let Some(c) = b2 {
                if !opt_same_motion(Some(c), a1) && !opt_same_motion(Some(c), b1) {
                    cands.push(c);
                }
            }
        }
    }

    // Temporal candidate (§8.5.3.1.2.3).
    if cands.len() < max_num_merge_cand as usize {
        if let Some(t) = tmvp {
            cands.push(t);
        }
    }

    let num_orig = cands.len();

    // Combined bi-pred candidates for B slices (§8.5.3.2.4).
    //
    // Round 21: walk the spec's combIdx ordering (Table 8-7) instead
    // of the generic O(N²) double loop, and use the spec gate
    //   predFlagL0[l0Cand] && predFlagL1[l1Cand] &&
    //   ( DiffPicOrderCnt(RefPicList0[refIdxL0l0Cand],
    //                     RefPicList1[refIdxL1l1Cand]) != 0
    //     || mvL0l0Cand != mvL1l1Cand )
    // for "add this candidate". The earlier code's "skip if equal to
    // any existing entry" check was non-spec — combined-bi-pred entries
    // are added in the spec without any global dedup.
    if is_b_slice && cands.len() < max_num_merge_cand as usize && num_orig > 1 {
        // Table 8-7 — fixed (l0CandIdx, l1CandIdx) ordering for combIdx
        // 0..11. The spec's loop terminates at combIdx ==
        // numOrigMergeCand * (numOrigMergeCand - 1) (i.e. all ordered
        // pairs of distinct indices in the original list, up to 4
        // entries → 12 pairs).
        const COMB_TABLE: [(usize, usize); 12] = [
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
        let max_comb_idx = (num_orig * (num_orig - 1)).min(COMB_TABLE.len());
        for &(l0_idx, l1_idx) in COMB_TABLE.iter().take(max_comb_idx) {
            if cands.len() >= max_num_merge_cand as usize {
                break;
            }
            if l0_idx >= num_orig || l1_idx >= num_orig {
                continue;
            }
            let l0_cand = cands[l0_idx];
            let l1_cand = cands[l1_idx];
            if !l0_cand.pred_l0 || !l1_cand.pred_l1 {
                continue;
            }
            // Spec gate: skip if the L0/L1 motion is bit-identical
            // (same MV + same POC). Otherwise the candidate is added.
            let l0_poc = comb
                .rpl0
                .get(l0_cand.ref_idx_l0.max(0) as usize)
                .map(|e| e.poc)
                .unwrap_or(l0_cand.ref_poc_l0);
            let l1_poc = comb
                .rpl1
                .get(l1_cand.ref_idx_l1.max(0) as usize)
                .map(|e| e.poc)
                .unwrap_or(l1_cand.ref_poc_l1);
            let poc_eq = l0_poc == l1_poc;
            let mv_eq = l0_cand.mv_l0 == l1_cand.mv_l1;
            if poc_eq && mv_eq {
                continue;
            }
            let l0_lt = comb
                .rpl0
                .get(l0_cand.ref_idx_l0.max(0) as usize)
                .map(|e| e.is_long_term)
                .unwrap_or(l0_cand.ref_lt_l0);
            let l1_lt = comb
                .rpl1
                .get(l1_cand.ref_idx_l1.max(0) as usize)
                .map(|e| e.is_long_term)
                .unwrap_or(l1_cand.ref_lt_l1);
            cands.push(MergeCand {
                pred_l0: true,
                pred_l1: true,
                ref_idx_l0: l0_cand.ref_idx_l0,
                ref_idx_l1: l1_cand.ref_idx_l1,
                mv_l0: l0_cand.mv_l0,
                mv_l1: l1_cand.mv_l1,
                ref_poc_l0: l0_poc,
                ref_poc_l1: l1_poc,
                ref_lt_l0: l0_lt,
                ref_lt_l1: l1_lt,
            });
        }
    }

    // Zero-MV padding (§8.5.3.2.5).
    //
    // Round 21: each padded candidate uses
    //   refIdxLX = (zeroIdx < numRefIdx) ? zeroIdx : 0
    // with `zeroIdx` incremented per pad iteration. Pre-r21 we
    // hard-coded `refIdxLX = 0` for every pad slot, which silently
    // collapses N pads onto a single ref entry — wrong when the
    // selected merge_idx lands in the pad region.
    let num_ref_idx = if zero.num_ref_idx == 0 {
        // Legacy / unit-test path: behave like the pre-r21 builder.
        1
    } else {
        zero.num_ref_idx
    };
    let mut zero_idx: u32 = 0;
    while (cands.len() as u32) < max_num_merge_cand {
        let ri = if zero_idx < num_ref_idx {
            zero_idx as i8
        } else {
            0
        };
        let (l0_poc, l0_lt) = zero
            .rpl0
            .get(ri.max(0) as usize)
            .map(|e| (e.poc, e.is_long_term))
            .unwrap_or((0, false));
        let (l1_poc, l1_lt) = if is_b_slice {
            zero.rpl1
                .get(ri.max(0) as usize)
                .map(|e| (e.poc, e.is_long_term))
                .unwrap_or((0, false))
        } else {
            (0, false)
        };
        cands.push(MergeCand {
            pred_l0: true,
            pred_l1: is_b_slice,
            ref_idx_l0: ri,
            ref_idx_l1: if is_b_slice { ri } else { 0 },
            mv_l0: MotionVector::default(),
            mv_l1: MotionVector::default(),
            ref_poc_l0: l0_poc,
            ref_poc_l1: l1_poc,
            ref_lt_l0: l0_lt,
            ref_lt_l1: l1_lt,
        });
        zero_idx += 1;
    }
    cands.truncate(max_num_merge_cand as usize);
    cands
}

fn fetch_spatial_neighbour(inter: &InterState, x: i32, y: i32) -> Option<MergeCand> {
    if x < 0 || y < 0 {
        return None;
    }
    let bx = (x >> 2) as usize;
    let by = (y >> 2) as usize;
    let pb = inter.get(bx, by)?;
    if !pb.valid || pb.is_intra {
        return None;
    }
    Some(MergeCand::from_pb(*pb))
}

// ---------------------------------------------------------------------------
//  AMVP candidate list
// ---------------------------------------------------------------------------

/// Summary of a reference picture for AMVP POC-scaling (§8.5.3.2.7):
/// POC value and long-term-ref flag. One entry per L0 / L1 slot so a
/// neighbour's `(pred_flag, ref_idx)` can be resolved into the referenced
/// picture's POC without pulling the full [`RefPicture`] tree.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefPocEntry {
    pub poc: i32,
    pub is_long_term: bool,
}

/// Per-slice POC context used by AMVP MV-predictor scaling. Built once
/// per inter slice from the resolved `rpl0` / `rpl1` + current POC.
#[derive(Clone, Debug, Default)]
pub struct AmvpPocInfo<'a> {
    pub cur_poc: i32,
    pub rpl0: &'a [RefPocEntry],
    pub rpl1: &'a [RefPocEntry],
}

impl<'a> AmvpPocInfo<'a> {
    fn ref_entry(&self, want_l0: bool, ref_idx: i8) -> Option<RefPocEntry> {
        let list = if want_l0 { self.rpl0 } else { self.rpl1 };
        if ref_idx < 0 {
            return None;
        }
        list.get(ref_idx as usize).copied()
    }
}

/// Apply the §8.5.3.2.7 distance-scaling transform (eqs. 8-179..8-181)
/// to a motion vector component pair, scaling from the neighbour's
/// reference POC (`tx_src`) to the current PU's target reference POC
/// (`tx_dst`). Both are expressed as signed POC deltas from `cur_poc`.
///
/// The two POC-difference inputs follow eqs. 8-182 / 8-183:
///
/// ```text
/// td = Clip3(-128, 127, DiffPicOrderCnt(curr, neighbour_ref))   // 8-182
/// tb = Clip3(-128, 127, DiffPicOrderCnt(curr, target_ref))      // 8-183
/// tx = (16384 + (|td| >> 1)) / td                               // 8-179
/// dsf = Clip3(-4096, 4095, (tb * tx + 32) >> 6)                 // 8-180
/// mv' = Clip3(-32768, 32767, sign(dsf*mv) * ((|dsf*mv| + 127) >> 8))
/// ```
fn scale_mv_by_poc(mv: MotionVector, td: i32, tb: i32) -> MotionVector {
    let td = td.clamp(-128, 127);
    let tb = tb.clamp(-128, 127);
    if td == 0 {
        return mv;
    }
    // Integer division toward zero — matches spec semantics when td != 0.
    let tx = (16384 + (td.abs() >> 1)) / td;
    let dsf = ((tb * tx + 32) >> 6).clamp(-4096, 4095);
    let scale = |v: i32| -> i32 {
        let prod = dsf * v;
        let sign = if prod < 0 { -1 } else { 1 };
        let abs = prod.unsigned_abs() as i64;
        let rounded = ((abs + 127) >> 8) as i32;
        (sign * rounded).clamp(-32768, 32767)
    };
    MotionVector::new(scale(mv.x), scale(mv.y))
}

/// Probe the two possible neighbour MVs (L_X first, then L_Y) for an
/// exact POC match against `target_poc`. Returns the first match's MV.
fn probe_neighbour_exact(
    pb: &PbMotion,
    want_l0: bool,
    poc_info: &AmvpPocInfo<'_>,
    target_poc: i32,
) -> Option<MotionVector> {
    if !pb.valid || pb.is_intra {
        return None;
    }
    // §8.5.3.2.7 eqs. 8-171/8-172: check LX first, LY second.
    for &list_l0 in &[want_l0, !want_l0] {
        let (has, ridx, mv) = if list_l0 {
            (pb.pred_l0, pb.ref_idx_l0, pb.mv_l0)
        } else {
            (pb.pred_l1, pb.ref_idx_l1, pb.mv_l1)
        };
        if !has {
            continue;
        }
        let Some(entry) = poc_info.ref_entry(list_l0, ridx) else {
            continue;
        };
        if entry.poc == target_poc {
            return Some(mv);
        }
    }
    None
}

/// Probe the two possible neighbour MVs (L_X first, then L_Y) for a
/// long-term-flag match with `target_is_lt`. Returns the first match's
/// `(mv, ref_poc)`; caller applies POC-distance scaling when both refs
/// are short-term.
fn probe_neighbour_lt_match(
    pb: &PbMotion,
    want_l0: bool,
    poc_info: &AmvpPocInfo<'_>,
    target_is_lt: bool,
) -> Option<(MotionVector, i32, bool)> {
    if !pb.valid || pb.is_intra {
        return None;
    }
    // §8.5.3.2.7 eqs. 8-173/8-176: check LX first with LT-flag match,
    // then LY. Unlike pass 1, the gate is on LongTermRefPic parity
    // rather than exact POC.
    for &list_l0 in &[want_l0, !want_l0] {
        let (has, ridx, mv) = if list_l0 {
            (pb.pred_l0, pb.ref_idx_l0, pb.mv_l0)
        } else {
            (pb.pred_l1, pb.ref_idx_l1, pb.mv_l1)
        };
        if !has {
            continue;
        }
        let Some(entry) = poc_info.ref_entry(list_l0, ridx) else {
            continue;
        };
        if entry.is_long_term == target_is_lt {
            return Some((mv, entry.poc, entry.is_long_term));
        }
    }
    None
}

fn fetch_pb(inter: &InterState, x: i32, y: i32) -> Option<PbMotion> {
    if x < 0 || y < 0 {
        return None;
    }
    let bx = (x >> 2) as usize;
    let by = (y >> 2) as usize;
    inter.get(bx, by).copied()
}

/// Run the §8.5.3.2.7 pass-1 (exact POC match, no scaling) search across
/// the ordered `slots` for either the A-group or the B-group. Also
/// returns `any_available` — `true` when at least one slot had an
/// available (non-intra) neighbour, even if its MV did not match the
/// target POC. For the A-group this sets `isScaledFlagLX = 1`.
fn amvp_pass1_exact_match(
    inter: &InterState,
    slots: &[(i32, i32, bool)],
    want_l0: bool,
    target_poc: i32,
    poc_info: &AmvpPocInfo<'_>,
) -> (Option<MotionVector>, bool) {
    let mut any_available = false;
    for &(nx, ny, suppressed) in slots {
        if suppressed {
            continue;
        }
        let Some(pb) = fetch_pb(inter, nx, ny) else {
            continue;
        };
        if !pb.valid || pb.is_intra {
            continue;
        }
        any_available = true;
        if let Some(mv) = probe_neighbour_exact(&pb, want_l0, poc_info, target_poc) {
            return (Some(mv), any_available);
        }
    }
    (None, any_available)
}

/// Run the §8.5.3.2.7 pass-2 (LT-match + optional POC scaling) search
/// across the ordered `slots`. Returns the first usable MV (scaled when
/// both the target and the neighbour's ref are short-term).
fn amvp_pass2_scaled(
    inter: &InterState,
    slots: &[(i32, i32, bool)],
    want_l0: bool,
    target_poc: i32,
    target_is_lt: bool,
    poc_info: &AmvpPocInfo<'_>,
) -> Option<MotionVector> {
    for &(nx, ny, suppressed) in slots {
        if suppressed {
            continue;
        }
        let Some(pb) = fetch_pb(inter, nx, ny) else {
            continue;
        };
        let Some((mv, ref_poc, ref_is_lt)) =
            probe_neighbour_lt_match(&pb, want_l0, poc_info, target_is_lt)
        else {
            continue;
        };
        // When either is long-term, skip distance scaling —
        // eq. 8-181 only applies when both are short-term.
        if target_is_lt || ref_is_lt {
            return Some(mv);
        }
        let td = target_poc_to_clipped(poc_info.cur_poc, ref_poc);
        let tb = target_poc_to_clipped(poc_info.cur_poc, target_poc);
        return Some(scale_mv_by_poc(mv, td, tb));
    }
    None
}

fn target_poc_to_clipped(cur_poc: i32, ref_poc: i32) -> i32 {
    (cur_poc - ref_poc).clamp(-128, 127)
}

/// Build the two-entry AMVP MV-predictor list (§8.5.3.1.6 / §8.5.3.2.7)
/// for a given reference list.
///
/// `want_l0` selects which list's MV the current PU is asking to
/// predict (L0 for P slices / L0 side of B, L1 for L1 side of B).
/// `target_ref_idx` is the already-decoded refIdxLX the current PU uses.
/// `poc_info` carries the current slice POC + the referenced-picture POC
/// lists so neighbours whose refs differ from the target POC can be
/// distance-scaled per eqs. 8-179..8-183.
///
/// Missing slots are filled with TMVP (when available) then zero MV.
#[allow(clippy::too_many_arguments)]
pub fn build_amvp_list(
    inter: &InterState,
    x_pb: u32,
    y_pb: u32,
    n_pb_w: u32,
    n_pb_h: u32,
    want_l0: bool,
    target_ref_idx: i8,
    tmvp: Option<MotionVector>,
    nb: NeighbourContext,
    poc_info: &AmvpPocInfo<'_>,
) -> [MotionVector; 2] {
    // Resolve target reference's POC + LT flag. If unavailable (shouldn't
    // happen on a valid bitstream), fall back to current POC which makes
    // the scaling a no-op.
    let target = poc_info
        .ref_entry(want_l0, target_ref_idx)
        .unwrap_or(RefPocEntry {
            poc: poc_info.cur_poc,
            is_long_term: false,
        });

    // A-group: A0 then A1, with the §6.4.2 / §8.5.3.2.3 suppression
    // matching the existing merge-list behaviour so partIdx 1 doesn't
    // see partIdx 0's just-written motion.
    let a_slots = [
        (x_pb as i32 - 1, y_pb as i32 + n_pb_h as i32, nb.suppress_a0),
        (
            x_pb as i32 - 1,
            y_pb as i32 + n_pb_h as i32 - 1,
            nb.suppress_a1,
        ),
    ];
    // Pass 1 (exact POC) on A.
    let (mv_a1, a_any) = amvp_pass1_exact_match(inter, &a_slots, want_l0, target.poc, poc_info);
    // Pass 2 (LT-match + POC scaling) on A when pass 1 missed.
    let mv_a = mv_a1.or_else(|| {
        amvp_pass2_scaled(
            inter,
            &a_slots,
            want_l0,
            target.poc,
            target.is_long_term,
            poc_info,
        )
    });
    // `isScaledFlagLX` from §8.5.3.2.7 step 5 of A-derivation:
    // `isScaledFlagLX = 1 iff availableA0 OR availableA1`. Notably this
    // is independent of whether the A-group actually picked a valid MV.
    let is_scaled = a_any;

    // B-group: B0, B1, B2 in scan order. B1 is suppression-gated.
    let b_slots = [
        (x_pb as i32 + n_pb_w as i32, y_pb as i32 - 1, false),
        (
            x_pb as i32 + n_pb_w as i32 - 1,
            y_pb as i32 - 1,
            nb.suppress_b1,
        ),
        (x_pb as i32 - 1, y_pb as i32 - 1, false),
    ];
    // Pass 1 on B (exact POC, no scaling).
    let mv_b_pass1 = amvp_pass1_exact_match(inter, &b_slots, want_l0, target.poc, poc_info).0;

    // §8.5.3.2.7 steps 4–5 of the B-derivation:
    //   * When `isScaledFlagLX == 0` (no A-group availability at all) and
    //     `availableFlagLXB == 1`, set `mvLXA = mvLXB` (step 4, eq. 8-186)
    //     AND re-derive `mvLXB` with scaling (step 5).
    //   * Otherwise (A had availability), keep mvLXB at pass-1 value only.
    let (opt_a, opt_b) = if is_scaled {
        (mv_a, mv_b_pass1)
    } else {
        // A-group had no availability. Clone pass-1 B into A (eq. 8-186)
        // and re-derive B with scaling (step 5 / eqs. 8-193..8-197).
        let cloned_a = mv_b_pass1;
        let scaled_b = amvp_pass2_scaled(
            inter,
            &b_slots,
            want_l0,
            target.poc,
            target.is_long_term,
            poc_info,
        );
        (cloned_a, scaled_b)
    };

    // §8.5.3.1.6 list assembly: append each available candidate, drop the
    // second if equal to the first, append TMVP when `listSize < 2`, pad
    // to 2 with zero MV.
    let mut out: Vec<MotionVector> = Vec::with_capacity(3);
    if let Some(a) = opt_a {
        out.push(a);
    }
    if let Some(b) = opt_b {
        if out.is_empty() || out[0] != b {
            out.push(b);
        }
    }
    if out.len() < 2 {
        if let Some(t) = tmvp {
            if out.is_empty() || out[0] != t {
                out.push(t);
            }
        }
    }
    while out.len() < 2 {
        out.push(MotionVector::default());
    }
    [out[0], out[1]]
}

// ---------------------------------------------------------------------------
//  Sub-pel interpolation (§8.5.3.2.2.1 / §8.5.3.2.2.2 / §8.5.3.2.3)
// ---------------------------------------------------------------------------

/// 8-tap luma interpolation filters (Table 8-4).
/// Index 0 is the integer (unused in filter path — caller short-circuits),
/// 1..=3 are the 1/4, 2/4, 3/4 positions.
pub const LUMA_FILTER: [[i32; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
];

/// 4-tap chroma interpolation filters (Table 8-5).
pub const CHROMA_FILTER: [[i32; 4]; 8] = [
    [0, 64, 0, 0],
    [-2, 58, 10, -2],
    [-4, 54, 16, -2],
    [-6, 46, 28, -4],
    [-4, 36, 36, -4],
    [-4, 28, 46, -6],
    [-2, 16, 54, -4],
    [-2, 10, 58, -2],
];

/// Apply a luma 8-tap filter horizontally at a single (xi, yi) integer grid
/// position with fractional `fx` (0..=3). Returns the unclipped 16-bit
/// filtered value before rounding / shift.
fn luma_h_filter_int(p: impl Fn(i32, i32) -> i32, x: i32, y: i32, fx: usize) -> i32 {
    let f = &LUMA_FILTER[fx];
    let mut s = 0i32;
    for i in 0..8 {
        s += f[i] * p(x + i as i32 - 3, y);
    }
    s
}

/// Perform full 2-D luma sub-pel interpolation into `out` (row-major
/// `blk_w * blk_h`) with 1/4-pel MV `mv` applied to the reference at the
/// block origin (x0, y0). Output is clipped to `[0, (1 << bit_depth) - 1]`
/// per Clip1Y (§8.5.3.2.2). Supports 8- and 10-bit luma.
///
/// The integer pipeline follows the spec exactly:
///   * §8.5.3.3.3.2 derives `predSamplesLX` using `shift1 = Min(4, BitDepth-8)`
///     (per-row truncation, no rounding), `shift2 = 6` (vertical pass,
///     no rounding) and `shift3 = Max(2, 14-BitDepth)` (full-pel pre-scale).
///   * §8.5.3.3.4.2 eq. 8-262 converts `predSamplesLX` → `pbSamples` with
///     `shift = Max(2, 14-BitDepth)` and `offset = 1 << (shift-1)`, then
///     clips to `[0, (1<<bitDepth)-1]`.
///
/// At 8-bit this collapses to `(sum + 32) >> 6` (the legacy formulation);
/// at 10-bit the per-row truncation of the sub-pel filter output is
/// significant and must be preserved for bit-exact match with ffmpeg.
pub fn luma_mc(
    ref_pic: &RefPicture,
    x0: i32,
    y0: i32,
    blk_w: i32,
    blk_h: i32,
    mv: MotionVector,
    out: &mut [u16],
    bit_depth: i32,
) -> Result<()> {
    if out.len() < (blk_w * blk_h) as usize {
        return Err(Error::invalid("h265 luma_mc: output buffer too small"));
    }
    let fx = (mv.x & 3) as usize;
    let fy = (mv.y & 3) as usize;
    let ix = mv.x >> 2;
    let iy = mv.y >> 2;

    let px = |x: i32, y: i32| -> i32 { ref_pic.sample_luma(x0 + ix + x, y0 + iy + y) as i32 };
    let max_val = (1i32 << bit_depth) - 1;
    // §8.5.3.3.3.2 MC shifts.
    let shift1_mc: i32 = core::cmp::min(4, bit_depth - 8);
    let shift2_mc: i32 = 6;
    let shift3_mc: i32 = core::cmp::max(2, 14 - bit_depth);
    // §8.5.3.3.4.2 weighted-default shift back to sample range.
    let shift_w: i32 = core::cmp::max(2, 14 - bit_depth);
    let offset_w: i32 = 1 << (shift_w - 1);

    for j in 0..blk_h {
        for i in 0..blk_w {
            // First compute predSamplesLX per §8.5.3.3.3.2.
            let pred = if fx == 0 && fy == 0 {
                // Full-pel: A << shift3 (Table 8-8 row 0).
                px(i, j) << shift3_mc
            } else if fy == 0 {
                // Horizontal filter only (position a/b/c): >> shift1.
                luma_h_filter_int(px, i, j, fx) >> shift1_mc
            } else if fx == 0 {
                // Vertical filter only (position d/h/n) on full-pel
                // columns: >> shift1.
                let mut s = 0i32;
                let f = &LUMA_FILTER[fy];
                for k in 0..8 {
                    s += f[k] * px(i, j + k as i32 - 3);
                }
                s >> shift1_mc
            } else {
                // Both fractions: per spec, first horizontal row filter
                // shifted by shift1, then vertical filter on those rows
                // shifted by shift2=6. Preserving the intermediate
                // truncation is load-bearing at 10-bit.
                let mut h = [0i32; 8];
                for k in 0..8 {
                    let yk = j + k as i32 - 3;
                    h[k] = luma_h_filter_int(px, i, yk, fx) >> shift1_mc;
                }
                let mut s = 0i32;
                let f = &LUMA_FILTER[fy];
                for k in 0..8 {
                    s += f[k] * h[k];
                }
                s >> shift2_mc
            };
            // §8.5.3.3.4.2 eq. 8-262 — uni-pred weighted-default: add
            // rounding offset, right-shift, clip to sample range.
            let v = ((pred + offset_w) >> shift_w).clamp(0, max_val);
            out[(j * blk_w + i) as usize] = v as u16;
        }
    }
    Ok(())
}

/// Uni-prediction luma MC that returns the spec-scale `predSamplesLX`
/// values per §8.5.3.3.3.2 (pre-weighted). The output sits at
/// `shift3 = Max(2, 14 - bitDepth)` precision: a final `(a + b + offset)
/// >> shift` (§8.5.3.3.4.2 eq. 8-264) combines two uni-pred arrays.
///
/// The internal shifts (`shift1 = Min(4, bitDepth-8)`, `shift2 = 6`) must
/// be applied exactly as specified — the per-row truncation at `shift1`
/// is observable at 10-bit and is load-bearing for bit-exact decode.
pub fn luma_mc_hp(
    ref_pic: &RefPicture,
    x0: i32,
    y0: i32,
    blk_w: i32,
    blk_h: i32,
    mv: MotionVector,
    out: &mut [i32],
    bit_depth: i32,
) -> Result<()> {
    if out.len() < (blk_w * blk_h) as usize {
        return Err(Error::invalid("h265 luma_mc_hp: output buffer too small"));
    }
    let fx = (mv.x & 3) as usize;
    let fy = (mv.y & 3) as usize;
    let ix = mv.x >> 2;
    let iy = mv.y >> 2;

    let px = |x: i32, y: i32| -> i32 { ref_pic.sample_luma(x0 + ix + x, y0 + iy + y) as i32 };
    let shift1: i32 = core::cmp::min(4, bit_depth - 8);
    let shift2: i32 = 6;
    let shift3: i32 = core::cmp::max(2, 14 - bit_depth);

    for j in 0..blk_h {
        for i in 0..blk_w {
            let v = if fx == 0 && fy == 0 {
                // Full-pel: A << shift3 (Table 8-8 row 0).
                px(i, j) << shift3
            } else if fy == 0 {
                // Position a/b/c — horizontal filter >> shift1.
                luma_h_filter_int(px, i, j, fx) >> shift1
            } else if fx == 0 {
                // Position d/h/n — vertical filter on integer columns,
                // >> shift1.
                let mut s = 0i32;
                let f = &LUMA_FILTER[fy];
                for k in 0..8 {
                    s += f[k] * px(i, j + k as i32 - 3);
                }
                s >> shift1
            } else {
                // Two-stage sub-pel: per-row horizontal filter >> shift1,
                // then vertical filter >> shift2 = 6.
                let mut h = [0i32; 8];
                for k in 0..8 {
                    let yk = j + k as i32 - 3;
                    h[k] = luma_h_filter_int(px, i, yk, fx) >> shift1;
                }
                let mut s = 0i32;
                let f = &LUMA_FILTER[fy];
                for k in 0..8 {
                    s += f[k] * h[k];
                }
                s >> shift2
            };
            out[(j * blk_w + i) as usize] = v;
        }
    }
    Ok(())
}

/// Combine two uni-pred luma arrays produced by [`luma_mc_hp`] into a
/// bi-predicted block (§8.5.3.3.4.2 eq. 8-264). Final sample is clipped
/// by Clip1Y.
///
/// Spec `shift2 = Max(3, 15 - BitDepth)`, `offset2 = 1 << (shift2 - 1)`:
///   * 8-bit → shift=7, offset=64 (the only case covered pre-Main 10)
///   * 10-bit → shift=5, offset=16
pub fn luma_mc_bi_combine(a: &[i32], b: &[i32], out: &mut [u16], bit_depth: i32) {
    let shift = core::cmp::max(3, 15 - bit_depth);
    let offset = 1i32 << (shift - 1);
    let max_val = (1i32 << bit_depth) - 1;
    for i in 0..out.len() {
        let v = (a[i] + b[i] + offset) >> shift;
        out[i] = v.clamp(0, max_val) as u16;
    }
}

/// Weighted-bi luma combine (§8.5.3.3.4) with unit-offset rounding.
/// `(w0, o0)` and `(w1, o1)` are pulled from `pred_weight_table()`;
/// `log2_wd` is `luma_log2_weight_denom + 1`. Final sample is clipped by
/// Clip1Y to the `bit_depth` sample range.
pub fn luma_mc_bi_weighted(
    a: &[i32],
    b: &[i32],
    out: &mut [u16],
    w0: i32,
    o0: i32,
    w1: i32,
    o1: i32,
    log2_wd: u32,
    bit_depth: i32,
) {
    let shift = log2_wd;
    let round = 1i32 << shift;
    let max_val = (1i32 << bit_depth) - 1;
    for i in 0..out.len() {
        let v = (a[i] * w0 + b[i] * w1 + ((o0 + o1 + 1) << (shift - 1)) + round) >> (shift + 1);
        // Note: `(shift - 1)` is safe since HEVC requires `log2_wd >= 1`.
        out[i] = v.clamp(0, max_val) as u16;
    }
}

/// Perform 2-D chroma sub-pel interpolation. `mv` is the **luma** motion
/// vector in 1/4-pel units; the spec's §8.5.3.2.10 derivation for the
/// chroma MV is applied here: `mvCLX[0] = mvLX[0] * 2 / SubWidthC`,
/// `mvCLX[1] = mvLX[1] * 2 / SubHeightC`. The result is in 1/8-pel chroma
/// sample units, so the integer / fractional split is `>> 3` and `& 7`.
///
/// `(x0, y0)` is the chroma block origin (already divided by SubWidthC /
/// SubHeightC by the caller). `out` is `blk_w * blk_h` and `comp` selects
/// cb (0) or cr (1). Output is clipped to `[0, (1 << bit_depth) - 1]`
/// per Clip1C (§8.5.3.2.2.2). Matches §8.5.3.3.3.3 + §8.5.3.3.4.2 eq.
/// 8-262 exactly, including the `shift1 = Min(4, bitDepth-8)` per-row
/// truncation at 10-bit.
///
/// For 4:2:0 (`(sub_x, sub_y) = (2, 2)`) the chroma MV equals the luma
/// MV numerically, recovering the legacy 1/8-pel interpretation. For
/// 4:2:2 (`(2, 1)`) the vertical component doubles so each whole-luma
/// step still maps onto a whole chroma row, with `yFracC` always taking
/// only even values (0, 2, 4, 6).
#[allow(clippy::too_many_arguments)]
pub fn chroma_mc(
    ref_pic: &RefPicture,
    x0: i32,
    y0: i32,
    blk_w: i32,
    blk_h: i32,
    mv: MotionVector,
    out: &mut [u16],
    comp: u8,
    bit_depth: i32,
    sub_x: u32,
    sub_y: u32,
) -> Result<()> {
    if out.len() < (blk_w * blk_h) as usize {
        return Err(Error::invalid("h265 chroma_mc: output buffer too small"));
    }
    // §8.5.3.2.10 eq. 8-210 / 8-211 — derive chroma MV from luma MV.
    let mvc_x = mv.x * 2 / sub_x as i32;
    let mvc_y = mv.y * 2 / sub_y as i32;
    let fx = (mvc_x & 7) as usize;
    let fy = (mvc_y & 7) as usize;
    let ix = mvc_x >> 3;
    let iy = mvc_y >> 3;

    let px = |x: i32, y: i32| -> i32 {
        let sx = x0 + ix + x;
        let sy = y0 + iy + y;
        match comp {
            0 => ref_pic.sample_cb(sx, sy) as i32,
            _ => ref_pic.sample_cr(sx, sy) as i32,
        }
    };
    let max_val = (1i32 << bit_depth) - 1;
    let shift1: i32 = core::cmp::min(4, bit_depth - 8);
    let shift2: i32 = 6;
    let shift3: i32 = core::cmp::max(2, 14 - bit_depth);
    let shift_w: i32 = core::cmp::max(2, 14 - bit_depth);
    let offset_w: i32 = 1 << (shift_w - 1);

    for j in 0..blk_h {
        for i in 0..blk_w {
            let pred = if fx == 0 && fy == 0 {
                px(i, j) << shift3
            } else if fy == 0 {
                let f = &CHROMA_FILTER[fx];
                let mut s = 0i32;
                for k in 0..4 {
                    s += f[k] * px(i + k as i32 - 1, j);
                }
                s >> shift1
            } else if fx == 0 {
                let f = &CHROMA_FILTER[fy];
                let mut s = 0i32;
                for k in 0..4 {
                    s += f[k] * px(i, j + k as i32 - 1);
                }
                s >> shift1
            } else {
                let mut h = [0i32; 4];
                let fh = &CHROMA_FILTER[fx];
                for k in 0..4 {
                    let yk = j + k as i32 - 1;
                    let mut s = 0i32;
                    for m in 0..4 {
                        s += fh[m] * px(i + m as i32 - 1, yk);
                    }
                    h[k] = s >> shift1;
                }
                let fv = &CHROMA_FILTER[fy];
                let mut s = 0i32;
                for k in 0..4 {
                    s += fv[k] * h[k];
                }
                s >> shift2
            };
            let v = ((pred + offset_w) >> shift_w).clamp(0, max_val);
            out[(j * blk_w + i) as usize] = v as u16;
        }
    }
    Ok(())
}

/// Uni-prediction chroma MC returning `predSamplesLX` (§8.5.3.3.3.3) at
/// `shift3 = Max(2, 14 - bitDepth)` precision for bi-pred combine.
/// `mv` is the **luma** MV; chroma MV derivation follows §8.5.3.2.10
/// using `(sub_x, sub_y)`.
#[allow(clippy::too_many_arguments)]
pub fn chroma_mc_hp(
    ref_pic: &RefPicture,
    x0: i32,
    y0: i32,
    blk_w: i32,
    blk_h: i32,
    mv: MotionVector,
    out: &mut [i32],
    comp: u8,
    bit_depth: i32,
    sub_x: u32,
    sub_y: u32,
) -> Result<()> {
    if out.len() < (blk_w * blk_h) as usize {
        return Err(Error::invalid("h265 chroma_mc_hp: output buffer too small"));
    }
    // §8.5.3.2.10 eq. 8-210 / 8-211 — derive chroma MV from luma MV.
    let mvc_x = mv.x * 2 / sub_x as i32;
    let mvc_y = mv.y * 2 / sub_y as i32;
    let fx = (mvc_x & 7) as usize;
    let fy = (mvc_y & 7) as usize;
    let ix = mvc_x >> 3;
    let iy = mvc_y >> 3;

    let px = |x: i32, y: i32| -> i32 {
        let sx = x0 + ix + x;
        let sy = y0 + iy + y;
        match comp {
            0 => ref_pic.sample_cb(sx, sy) as i32,
            _ => ref_pic.sample_cr(sx, sy) as i32,
        }
    };
    let shift1: i32 = core::cmp::min(4, bit_depth - 8);
    let shift2: i32 = 6;
    let shift3: i32 = core::cmp::max(2, 14 - bit_depth);

    for j in 0..blk_h {
        for i in 0..blk_w {
            let v = if fx == 0 && fy == 0 {
                px(i, j) << shift3
            } else if fy == 0 {
                let f = &CHROMA_FILTER[fx];
                let mut s = 0i32;
                for k in 0..4 {
                    s += f[k] * px(i + k as i32 - 1, j);
                }
                s >> shift1
            } else if fx == 0 {
                let f = &CHROMA_FILTER[fy];
                let mut s = 0i32;
                for k in 0..4 {
                    s += f[k] * px(i, j + k as i32 - 1);
                }
                s >> shift1
            } else {
                let mut h = [0i32; 4];
                let fh = &CHROMA_FILTER[fx];
                for k in 0..4 {
                    let yk = j + k as i32 - 1;
                    let mut s = 0i32;
                    for m in 0..4 {
                        s += fh[m] * px(i + m as i32 - 1, yk);
                    }
                    h[k] = s >> shift1;
                }
                let fv = &CHROMA_FILTER[fy];
                let mut s = 0i32;
                for k in 0..4 {
                    s += fv[k] * h[k];
                }
                s >> shift2
            };
            out[(j * blk_w + i) as usize] = v;
        }
    }
    Ok(())
}

/// Combine two uni-pred chroma arrays into a bi-predicted block. Final
/// sample is clipped by Clip1C (same `(1 << bit_depth) - 1` envelope as
/// luma when `BitDepthY == BitDepthC`).
pub fn chroma_mc_bi_combine(a: &[i32], b: &[i32], out: &mut [u16], bit_depth: i32) {
    luma_mc_bi_combine(a, b, out, bit_depth);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_mv_luma_mc_matches_source() {
        let pic = RefPicture {
            poc: 0,
            width: 16,
            height: 16,
            sub_x: 2,
            sub_y: 2,
            luma: (0..(16 * 16)).map(|i| (i & 0xFF) as u16).collect(),
            cb: vec![128; 8 * 8],
            cr: vec![128; 8 * 8],
            luma_stride: 16,
            chroma_stride: 8,
            inter: InterState::new(16, 16),
            is_long_term: false,
        };
        let mut out = vec![0u16; 8 * 8];
        luma_mc(&pic, 4, 4, 8, 8, MotionVector::new(0, 0), &mut out, 8).expect("luma_mc");
        for j in 0..8 {
            for i in 0..8 {
                assert_eq!(out[j * 8 + i], pic.luma[(4 + j) * 16 + (4 + i)]);
            }
        }
    }

    #[test]
    fn merge_list_pads_with_zero_mv() {
        let inter = InterState::new(64, 64);
        let list = build_merge_list(
            &inter,
            16,
            16,
            8,
            8,
            5,
            false,
            None,
            NeighbourContext::default(),
        );
        assert_eq!(list.len(), 5);
        for c in &list {
            assert_eq!(c.mv_l0, MotionVector::default());
            assert_eq!(c.ref_idx_l0, 0);
            assert!(c.pred_l0);
            assert!(!c.pred_l1);
            // r19: merge zero-pad still sets ref_poc=0 (the merge-list
            // builder doesn't have RPL POCs in scope). The CALLER is
            // expected to refresh ref_poc against the current slice's
            // RPL[ref_idx] before stashing the PB into the inter grid.
            assert_eq!(c.ref_poc_l0, 0);
        }
    }

    #[test]
    fn pb_motion_default_is_not_skip() {
        // r19 is_skip default propagation: zero-init / `to_pb` paths
        // produce non-skip PBs. Skip CUs explicitly flip the bit at
        // the perform_merge call site (see ctu.rs `perform_merge`).
        assert!(!PbMotion::default().is_skip);
        let merge = MergeCand {
            pred_l0: true,
            pred_l1: false,
            ref_idx_l0: 0,
            ref_idx_l1: 0,
            mv_l0: MotionVector::new(1, -1),
            mv_l1: MotionVector::default(),
            ref_poc_l0: 0,
            ref_poc_l1: 0,
            ref_lt_l0: false,
            ref_lt_l1: false,
        };
        assert!(!merge.to_pb().is_skip);
    }

    #[test]
    fn merge_picks_left_neighbour() {
        let mut inter = InterState::new(64, 64);
        let pb = PbMotion::inter(0, MotionVector::new(4, -4));
        inter.set_rect(0, 16, 4, 4, pb);
        inter.set_rect(0, 20, 4, 4, pb);
        let list = build_merge_list(
            &inter,
            4,
            16,
            8,
            8,
            5,
            false,
            None,
            NeighbourContext::default(),
        );
        assert_eq!(list[0].mv_l0, MotionVector::new(4, -4));
    }

    #[test]
    fn bi_merge_zero_is_bipred() {
        let inter = InterState::new(64, 64);
        let list = build_merge_list(
            &inter,
            16,
            16,
            8,
            8,
            5,
            true,
            None,
            NeighbourContext::default(),
        );
        for c in &list {
            assert!(c.pred_l0 && c.pred_l1);
        }
    }

    #[test]
    fn combined_bi_pred_candidate_built() {
        let mut inter = InterState::new(64, 64);
        // Fill A1 with a uni-L0 candidate.
        let a1 = PbMotion::inter(0, MotionVector::new(2, 0));
        inter.set_rect(0, 16, 4, 8, a1);
        // Fill B1 with a uni-L1 candidate.
        let b1 = PbMotion::inter_l1(1, MotionVector::new(-2, 0));
        inter.set_rect(4, 12, 8, 4, b1);
        let list = build_merge_list(
            &inter,
            4,
            16,
            8,
            8,
            5,
            true,
            None,
            NeighbourContext::default(),
        );
        // There must be at least one bi-pred candidate somewhere.
        assert!(list.iter().any(|c| c.pred_l0 && c.pred_l1));
    }

    #[test]
    fn bi_combine_is_averaging() {
        let a = vec![10i32 << 6; 16]; // shift-6 encoding of value 10
        let b = vec![20i32 << 6; 16];
        let mut out = vec![0u16; 16];
        luma_mc_bi_combine(&a, &b, &mut out, 8);
        for v in &out {
            assert_eq!(*v, 15);
        }
    }

    #[test]
    fn bi_combine_10bit_scales_correctly() {
        // §8.5.3.3.4.2 eq. 8-264 — for 10-bit, shift2 = Max(3, 15-10) = 5
        // and offset2 = 1 << 4 = 16. The high-precision uni-pred inputs
        // carry an extra `<< shift3` factor relative to 8-bit (shift3 =
        // Max(2, 14-10) = 4), so the same "average value 15" encoding
        // lives at `15 << 4` per sample (combined scale = shift2).
        let a = vec![10i32 << 4; 16];
        let b = vec![20i32 << 4; 16];
        let mut out = vec![0u16; 16];
        luma_mc_bi_combine(&a, &b, &mut out, 10);
        for v in &out {
            assert_eq!(*v, 15);
        }
    }

    fn stub_pic(poc: i32, long_term: bool) -> RefPicture {
        RefPicture {
            poc,
            width: 4,
            height: 4,
            sub_x: 2,
            sub_y: 2,
            luma: vec![0; 16],
            cb: vec![128; 4],
            cr: vec![128; 4],
            luma_stride: 4,
            chroma_stride: 2,
            inter: InterState::new(4, 4),
            is_long_term: long_term,
        }
    }

    #[test]
    fn dpb_push_evicts_short_term_before_long_term() {
        let mut dpb = Dpb::new(2);
        dpb.push(stub_pic(0, true)); // LT
        dpb.push(stub_pic(1, false)); // ST
                                      // Over capacity — should drop POC 1 (short-term) not POC 0 (LT).
        dpb.push(stub_pic(2, false));
        assert!(dpb.get_by_poc(0).is_some(), "LT ref survived eviction");
        assert!(dpb.get_by_poc(1).is_none(), "ST ref was evicted");
        assert!(dpb.get_by_poc(2).is_some(), "new ref entered DPB");
    }

    #[test]
    fn dpb_find_by_poc_lsb_matches_modulo() {
        let mut dpb = Dpb::new(4);
        // POC 17, max_poc_lsb = 16 → lsb = 1.
        dpb.push(stub_pic(17, false));
        let m = dpb.find_by_poc_lsb(1, 16);
        assert!(m.is_some());
        assert_eq!(m.unwrap().poc, 17);
    }

    #[test]
    fn dpb_mark_long_term_flips_flag() {
        let mut dpb = Dpb::new(2);
        dpb.push(stub_pic(5, false));
        assert!(!dpb.get_by_poc(5).unwrap().is_long_term);
        dpb.mark_long_term_by_poc(5);
        assert!(dpb.get_by_poc(5).unwrap().is_long_term);
    }
}
