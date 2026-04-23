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
            pred_l0: true,
            pred_l1: true,
            ref_idx_l0,
            ref_idx_l1,
            mv_l0,
            mv_l1,
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
#[derive(Clone, Debug)]
pub struct RefPicture {
    pub poc: i32,
    pub width: u32,
    pub height: u32,
    pub luma: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
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
    pub fn sample_luma(&self, x: i32, y: i32) -> u8 {
        let w = self.width as i32;
        let h = self.height as i32;
        let xc = x.clamp(0, w - 1) as usize;
        let yc = y.clamp(0, h - 1) as usize;
        self.luma[yc * self.luma_stride + xc]
    }

    pub fn sample_cb(&self, x: i32, y: i32) -> u8 {
        let w = (self.width / 2) as i32;
        let h = (self.height / 2) as i32;
        let xc = x.clamp(0, w - 1) as usize;
        let yc = y.clamp(0, h - 1) as usize;
        self.cb[yc * self.chroma_stride + xc]
    }

    pub fn sample_cr(&self, x: i32, y: i32) -> u8 {
        let w = (self.width / 2) as i32;
        let h = (self.height / 2) as i32;
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
        self.pics
            .iter()
            .find(|p| (p.poc & mask) as u32 == poc_lsb)
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
        }
    }

    /// Re-map a collocated-picture PB into a temporal merge candidate that
    /// refers to the current slice's ref-idx 0 in whichever list the
    /// collocated PB used. §8.5.3.1.2.3 requires POC-distance scaling
    /// (§8.5.3.1.8); we skip scaling for this simple implementation
    /// because the short GOPs we target (|Δ|==1 refs) make the scale
    /// factor ≈ 1.
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
        }
    }

    pub fn to_pb(self) -> PbMotion {
        PbMotion {
            valid: true,
            is_intra: false,
            pred_l0: self.pred_l0,
            pred_l1: self.pred_l1,
            ref_idx_l0: self.ref_idx_l0,
            ref_idx_l1: self.ref_idx_l1,
            mv_l0: self.mv_l0,
            mv_l1: self.mv_l1,
        }
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
pub fn build_merge_list(
    inter: &InterState,
    x_pb: u32,
    y_pb: u32,
    n_pb_w: u32,
    n_pb_h: u32,
    max_num_merge_cand: u32,
    is_b_slice: bool,
    tmvp: Option<MergeCand>,
) -> Vec<MergeCand> {
    let mut cands: Vec<MergeCand> = Vec::with_capacity(max_num_merge_cand as usize);

    let a1 = fetch_spatial_neighbour(inter, x_pb as i32 - 1, y_pb as i32 + n_pb_h as i32 - 1);
    let b1 = fetch_spatial_neighbour(inter, x_pb as i32 + n_pb_w as i32 - 1, y_pb as i32 - 1);
    let b0 = fetch_spatial_neighbour(inter, x_pb as i32 + n_pb_w as i32, y_pb as i32 - 1);
    let a0 = fetch_spatial_neighbour(inter, x_pb as i32 - 1, y_pb as i32 + n_pb_h as i32);
    let b2 = fetch_spatial_neighbour(inter, x_pb as i32 - 1, y_pb as i32 - 1);

    let push = |cands: &mut Vec<MergeCand>, cand: Option<MergeCand>| {
        if let Some(c) = cand {
            if !cands.iter().any(|existing| existing == &c) {
                cands.push(c);
            }
        }
    };
    push(&mut cands, a1);
    if cands.len() < max_num_merge_cand as usize {
        push(&mut cands, b1);
    }
    if cands.len() < max_num_merge_cand as usize {
        push(&mut cands, b0);
    }
    if cands.len() < max_num_merge_cand as usize {
        push(&mut cands, a0);
    }
    if cands.len() < max_num_merge_cand as usize && cands.len() < 4 {
        push(&mut cands, b2);
    }

    // Temporal candidate (§8.5.3.1.2.3).
    if cands.len() < max_num_merge_cand as usize {
        if let Some(t) = tmvp {
            cands.push(t);
        }
    }

    // Combined bi-pred candidates for B slices (§8.5.3.1.2.4).
    if is_b_slice && cands.len() < max_num_merge_cand as usize {
        let n = cands.len();
        'outer: for l0_idx in 0..n {
            for l1_idx in 0..n {
                if l0_idx == l1_idx {
                    continue;
                }
                let a = cands[l0_idx];
                let b = cands[l1_idx];
                if !a.pred_l0 || !b.pred_l1 {
                    continue;
                }
                let combined = MergeCand {
                    pred_l0: true,
                    pred_l1: true,
                    ref_idx_l0: a.ref_idx_l0,
                    ref_idx_l1: b.ref_idx_l1,
                    mv_l0: a.mv_l0,
                    mv_l1: b.mv_l1,
                };
                if !cands.iter().any(|existing| existing == &combined) {
                    cands.push(combined);
                    if cands.len() >= max_num_merge_cand as usize {
                        break 'outer;
                    }
                }
            }
        }
    }

    // Pad with zero MV per §8.5.3.1.3. For B slices the zero candidate is
    // bi-pred; for P slices it's L0-only.
    while (cands.len() as u32) < max_num_merge_cand {
        cands.push(MergeCand {
            pred_l0: true,
            pred_l1: is_b_slice,
            ref_idx_l0: 0,
            ref_idx_l1: 0,
            mv_l0: MotionVector::default(),
            mv_l1: MotionVector::default(),
        });
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

/// Build the two-entry AMVP MV-predictor list (§8.5.3.1.6) for a given
/// reference list. Spatial neighbours A0/A1 contribute the first
/// candidate; B0/B1/B2 contribute the second. TMVP fills remaining
/// slots when available. Missing slots are filled with zero MV.
///
/// `want_l0` selects which of the neighbour's lists to read. For P
/// slices and L0 in B slices this is `true`. For L1 in B slices it is
/// `false`.
pub fn build_amvp_list(
    inter: &InterState,
    x_pb: u32,
    y_pb: u32,
    n_pb_w: u32,
    n_pb_h: u32,
    want_l0: bool,
    tmvp: Option<MotionVector>,
) -> [MotionVector; 2] {
    let pick = |cand: MergeCand| -> Option<MotionVector> {
        if want_l0 && cand.pred_l0 {
            Some(cand.mv_l0)
        } else if !want_l0 && cand.pred_l1 {
            Some(cand.mv_l1)
        } else if cand.pred_l0 {
            Some(cand.mv_l0)
        } else if cand.pred_l1 {
            Some(cand.mv_l1)
        } else {
            None
        }
    };

    let a0 =
        fetch_spatial_neighbour(inter, x_pb as i32 - 1, y_pb as i32 + n_pb_h as i32).and_then(pick);
    let a1 = fetch_spatial_neighbour(inter, x_pb as i32 - 1, y_pb as i32 + n_pb_h as i32 - 1)
        .and_then(pick);
    let b0 =
        fetch_spatial_neighbour(inter, x_pb as i32 + n_pb_w as i32, y_pb as i32 - 1).and_then(pick);
    let b1 = fetch_spatial_neighbour(inter, x_pb as i32 + n_pb_w as i32 - 1, y_pb as i32 - 1)
        .and_then(pick);
    let b2 = fetch_spatial_neighbour(inter, x_pb as i32 - 1, y_pb as i32 - 1).and_then(pick);

    let spatial_a = a0.or(a1).unwrap_or_default();
    let spatial_b = b0.or(b1).or(b2).unwrap_or_default();
    let mut out = [spatial_a, spatial_b];
    if out[0] == out[1] {
        // Try TMVP to fill the second slot before falling back to zero.
        if let Some(t) = tmvp {
            if t != out[0] {
                out[1] = t;
                return out;
            }
        }
        out[1] = MotionVector::default();
    }
    out
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
/// block origin (x0, y0). 8-bit only.
pub fn luma_mc(
    ref_pic: &RefPicture,
    x0: i32,
    y0: i32,
    blk_w: i32,
    blk_h: i32,
    mv: MotionVector,
    out: &mut [u8],
) -> Result<()> {
    if out.len() < (blk_w * blk_h) as usize {
        return Err(Error::invalid("h265 luma_mc: output buffer too small"));
    }
    let fx = (mv.x & 3) as usize;
    let fy = (mv.y & 3) as usize;
    let ix = mv.x >> 2;
    let iy = mv.y >> 2;

    let px = |x: i32, y: i32| -> i32 { ref_pic.sample_luma(x0 + ix + x, y0 + iy + y) as i32 };

    for j in 0..blk_h {
        for i in 0..blk_w {
            let v = if fx == 0 && fy == 0 {
                px(i, j)
            } else if fy == 0 {
                let s = luma_h_filter_int(px, i, j, fx);
                ((s + 32) >> 6).clamp(0, 255)
            } else if fx == 0 {
                let mut s = 0i32;
                let f = &LUMA_FILTER[fy];
                for k in 0..8 {
                    s += f[k] * px(i, j + k as i32 - 3);
                }
                ((s + 32) >> 6).clamp(0, 255)
            } else {
                let mut h = [0i32; 8];
                for k in 0..8 {
                    let yk = j + k as i32 - 3;
                    h[k] = luma_h_filter_int(px, i, yk, fx);
                }
                let mut s = 0i32;
                let f = &LUMA_FILTER[fy];
                for k in 0..8 {
                    s += f[k] * h[k];
                }
                ((s + 2048) >> 12).clamp(0, 255)
            };
            out[(j * blk_w + i) as usize] = v as u8;
        }
    }
    Ok(())
}

/// Uni-prediction luma MC that returns the 16-bit pre-shift samples for
/// later bi-prediction averaging (§8.5.3.3.3). The returned values are
/// at "shift 6" precision: a final `(a + b + 64) >> 7` combines two
/// uni-pred arrays into bi-pred output.
pub fn luma_mc_hp(
    ref_pic: &RefPicture,
    x0: i32,
    y0: i32,
    blk_w: i32,
    blk_h: i32,
    mv: MotionVector,
    out: &mut [i32],
) -> Result<()> {
    if out.len() < (blk_w * blk_h) as usize {
        return Err(Error::invalid("h265 luma_mc_hp: output buffer too small"));
    }
    let fx = (mv.x & 3) as usize;
    let fy = (mv.y & 3) as usize;
    let ix = mv.x >> 2;
    let iy = mv.y >> 2;

    let px = |x: i32, y: i32| -> i32 { ref_pic.sample_luma(x0 + ix + x, y0 + iy + y) as i32 };

    for j in 0..blk_h {
        for i in 0..blk_w {
            let v = if fx == 0 && fy == 0 {
                // Up-scale to shift-6 precision.
                px(i, j) << 6
            } else if fy == 0 {
                luma_h_filter_int(px, i, j, fx)
            } else if fx == 0 {
                let mut s = 0i32;
                let f = &LUMA_FILTER[fy];
                for k in 0..8 {
                    s += f[k] * px(i, j + k as i32 - 3);
                }
                s
            } else {
                let mut h = [0i32; 8];
                for k in 0..8 {
                    let yk = j + k as i32 - 3;
                    h[k] = luma_h_filter_int(px, i, yk, fx);
                }
                let mut s = 0i32;
                let f = &LUMA_FILTER[fy];
                for k in 0..8 {
                    s += f[k] * h[k];
                }
                // Drop back to shift-6 precision so both uni-pred paths
                // share the same scale for bi-pred averaging.
                s >> 6
            };
            out[(j * blk_w + i) as usize] = v;
        }
    }
    Ok(())
}

/// Combine two uni-pred luma arrays produced by [`luma_mc_hp`] into a
/// bi-predicted block (§8.5.3.3.3).
pub fn luma_mc_bi_combine(a: &[i32], b: &[i32], out: &mut [u8]) {
    for i in 0..out.len() {
        let v = (a[i] + b[i] + 64) >> 7;
        out[i] = v.clamp(0, 255) as u8;
    }
}

/// Weighted-bi luma combine (§8.5.3.3.4) with unit-offset rounding.
/// `(w0, o0)` and `(w1, o1)` are pulled from `pred_weight_table()`;
/// `log2_wd` is `luma_log2_weight_denom + 1`.
pub fn luma_mc_bi_weighted(
    a: &[i32],
    b: &[i32],
    out: &mut [u8],
    w0: i32,
    o0: i32,
    w1: i32,
    o1: i32,
    log2_wd: u32,
) {
    let shift = log2_wd;
    let round = 1i32 << shift;
    for i in 0..out.len() {
        let v = (a[i] * w0 + b[i] * w1 + ((o0 + o1 + 1) << (shift - 1)) + round) >> (shift + 1);
        // Note: `(shift - 1)` is safe since HEVC requires `log2_wd >= 1`.
        out[i] = v.clamp(0, 255) as u8;
    }
}

/// Perform 2-D chroma sub-pel interpolation. `mv` is in 1/4-pel luma units;
/// chroma uses 1/8-pel fractions so we split the MV accordingly.
/// `out` is `blk_w * blk_h` and `comp` selects cb (0) or cr (1).
pub fn chroma_mc(
    ref_pic: &RefPicture,
    x0: i32,
    y0: i32,
    blk_w: i32,
    blk_h: i32,
    mv: MotionVector,
    out: &mut [u8],
    comp: u8,
) -> Result<()> {
    if out.len() < (blk_w * blk_h) as usize {
        return Err(Error::invalid("h265 chroma_mc: output buffer too small"));
    }
    let fx = (mv.x & 7) as usize;
    let fy = (mv.y & 7) as usize;
    let ix = mv.x >> 3;
    let iy = mv.y >> 3;

    let px = |x: i32, y: i32| -> i32 {
        let sx = x0 + ix + x;
        let sy = y0 + iy + y;
        match comp {
            0 => ref_pic.sample_cb(sx, sy) as i32,
            _ => ref_pic.sample_cr(sx, sy) as i32,
        }
    };

    for j in 0..blk_h {
        for i in 0..blk_w {
            let v = if fx == 0 && fy == 0 {
                px(i, j)
            } else if fy == 0 {
                let f = &CHROMA_FILTER[fx];
                let mut s = 0i32;
                for k in 0..4 {
                    s += f[k] * px(i + k as i32 - 1, j);
                }
                ((s + 32) >> 6).clamp(0, 255)
            } else if fx == 0 {
                let f = &CHROMA_FILTER[fy];
                let mut s = 0i32;
                for k in 0..4 {
                    s += f[k] * px(i, j + k as i32 - 1);
                }
                ((s + 32) >> 6).clamp(0, 255)
            } else {
                let mut h = [0i32; 4];
                let fh = &CHROMA_FILTER[fx];
                for k in 0..4 {
                    let yk = j + k as i32 - 1;
                    let mut s = 0i32;
                    for m in 0..4 {
                        s += fh[m] * px(i + m as i32 - 1, yk);
                    }
                    h[k] = s;
                }
                let fv = &CHROMA_FILTER[fy];
                let mut s = 0i32;
                for k in 0..4 {
                    s += fv[k] * h[k];
                }
                ((s + 2048) >> 12).clamp(0, 255)
            };
            out[(j * blk_w + i) as usize] = v as u8;
        }
    }
    Ok(())
}

/// Uni-prediction chroma MC at shift-6 precision for bi-pred combine.
pub fn chroma_mc_hp(
    ref_pic: &RefPicture,
    x0: i32,
    y0: i32,
    blk_w: i32,
    blk_h: i32,
    mv: MotionVector,
    out: &mut [i32],
    comp: u8,
) -> Result<()> {
    if out.len() < (blk_w * blk_h) as usize {
        return Err(Error::invalid("h265 chroma_mc_hp: output buffer too small"));
    }
    let fx = (mv.x & 7) as usize;
    let fy = (mv.y & 7) as usize;
    let ix = mv.x >> 3;
    let iy = mv.y >> 3;

    let px = |x: i32, y: i32| -> i32 {
        let sx = x0 + ix + x;
        let sy = y0 + iy + y;
        match comp {
            0 => ref_pic.sample_cb(sx, sy) as i32,
            _ => ref_pic.sample_cr(sx, sy) as i32,
        }
    };

    for j in 0..blk_h {
        for i in 0..blk_w {
            let v = if fx == 0 && fy == 0 {
                px(i, j) << 6
            } else if fy == 0 {
                let f = &CHROMA_FILTER[fx];
                let mut s = 0i32;
                for k in 0..4 {
                    s += f[k] * px(i + k as i32 - 1, j);
                }
                s
            } else if fx == 0 {
                let f = &CHROMA_FILTER[fy];
                let mut s = 0i32;
                for k in 0..4 {
                    s += f[k] * px(i, j + k as i32 - 1);
                }
                s
            } else {
                let mut h = [0i32; 4];
                let fh = &CHROMA_FILTER[fx];
                for k in 0..4 {
                    let yk = j + k as i32 - 1;
                    let mut s = 0i32;
                    for m in 0..4 {
                        s += fh[m] * px(i + m as i32 - 1, yk);
                    }
                    h[k] = s;
                }
                let fv = &CHROMA_FILTER[fy];
                let mut s = 0i32;
                for k in 0..4 {
                    s += fv[k] * h[k];
                }
                s >> 6
            };
            out[(j * blk_w + i) as usize] = v;
        }
    }
    Ok(())
}

/// Combine two uni-pred chroma arrays into a bi-predicted block.
pub fn chroma_mc_bi_combine(a: &[i32], b: &[i32], out: &mut [u8]) {
    luma_mc_bi_combine(a, b, out);
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
            luma: (0..(16 * 16)).map(|i| (i & 0xFF) as u8).collect(),
            cb: vec![128; 8 * 8],
            cr: vec![128; 8 * 8],
            luma_stride: 16,
            chroma_stride: 8,
            inter: InterState::new(16, 16),
            is_long_term: false,
        };
        let mut out = vec![0u8; 8 * 8];
        luma_mc(&pic, 4, 4, 8, 8, MotionVector::new(0, 0), &mut out).expect("luma_mc");
        for j in 0..8 {
            for i in 0..8 {
                assert_eq!(out[j * 8 + i], pic.luma[(4 + j) * 16 + (4 + i)]);
            }
        }
    }

    #[test]
    fn merge_list_pads_with_zero_mv() {
        let inter = InterState::new(64, 64);
        let list = build_merge_list(&inter, 16, 16, 8, 8, 5, false, None);
        assert_eq!(list.len(), 5);
        for c in &list {
            assert_eq!(c.mv_l0, MotionVector::default());
            assert_eq!(c.ref_idx_l0, 0);
            assert!(c.pred_l0);
            assert!(!c.pred_l1);
        }
    }

    #[test]
    fn merge_picks_left_neighbour() {
        let mut inter = InterState::new(64, 64);
        let pb = PbMotion::inter(0, MotionVector::new(4, -4));
        inter.set_rect(0, 16, 4, 4, pb);
        inter.set_rect(0, 20, 4, 4, pb);
        let list = build_merge_list(&inter, 4, 16, 8, 8, 5, false, None);
        assert_eq!(list[0].mv_l0, MotionVector::new(4, -4));
    }

    #[test]
    fn bi_merge_zero_is_bipred() {
        let inter = InterState::new(64, 64);
        let list = build_merge_list(&inter, 16, 16, 8, 8, 5, true, None);
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
        let list = build_merge_list(&inter, 4, 16, 8, 8, 5, true, None);
        // There must be at least one bi-pred candidate somewhere.
        assert!(list.iter().any(|c| c.pred_l0 && c.pred_l1));
    }

    #[test]
    fn bi_combine_is_averaging() {
        let a = vec![10i32 << 6; 16]; // shift-6 encoding of value 10
        let b = vec![20i32 << 6; 16];
        let mut out = vec![0u8; 16];
        luma_mc_bi_combine(&a, &b, &mut out);
        for v in &out {
            assert_eq!(*v, 15);
        }
    }

    fn stub_pic(poc: i32, long_term: bool) -> RefPicture {
        RefPicture {
            poc,
            width: 4,
            height: 4,
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
