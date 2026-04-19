//! HEVC inter prediction support (§8.5).
//!
//! This module supplies the pieces needed for P-slice inter decode:
//!
//! * [`MotionVector`] + [`PbMotion`] record the per-prediction-block motion
//!   state (reference index, sub-pel MV, availability) that the CTU walker
//!   saves into [`InterState`] and consults for spatial neighbour derivation.
//! * [`Dpb`] — decoded picture buffer keyed by picture order count.
//! * [`RefPicList`] — per-slice reference list L0, resolved from the active
//!   RPS + DPB contents (§8.3.2).
//! * Merge-candidate construction (§8.5.3.1.2 / 8.5.3.1.3) — spatial
//!   candidates A1, B1, B0, A0, B2 with HEVC-style pruning; zero MV filler.
//! * AMVP (§8.5.3.1.6) — spatial and zero candidates; temporal MV lookups
//!   are not currently supported (TMVP is a no-op).
//! * 8-tap luma and 4-tap chroma sub-pel interpolation (§8.5.3.2.2 /
//!   8.5.3.2.3) with the 16-phase filter tables.
//!
//! Bi-prediction, B-slice specific merge/AMVP paths, temporal merge, and
//! full weighted-pred are out of scope for the P-slice landing — callers
//! should continue to surface `Unsupported` for those features.

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
    /// Reference index into L0 (P-slice only — L1 not tracked yet).
    pub ref_idx_l0: i8,
    pub mv_l0: MotionVector,
}

impl PbMotion {
    pub fn intra() -> Self {
        Self {
            valid: true,
            is_intra: true,
            ..Default::default()
        }
    }

    pub fn inter(ref_idx_l0: i8, mv: MotionVector) -> Self {
        Self {
            valid: true,
            is_intra: false,
            ref_idx_l0,
            mv_l0: mv,
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
}

/// Decoded picture buffer. Very small and naive — a `VecDeque` with POC
/// lookup is enough for single-ref P-slice decode.
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
            // Drop oldest.
            self.pics.remove(0);
        }
        self.pics.push(pic);
    }

    pub fn get_by_poc(&self, poc: i32) -> Option<&RefPicture> {
        self.pics.iter().find(|p| p.poc == poc)
    }
}

/// Reference picture list L0, resolved against the current slice's RPS and
/// the DPB. Holds owned POC references; the MC path then looks each POC up
/// in the DPB.
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
//  Merge candidate list
// ---------------------------------------------------------------------------

/// One merge candidate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MergeCand {
    pub ref_idx_l0: i8,
    pub mv_l0: MotionVector,
}

/// Build the merge candidate list for a P-slice PB at (xPb, yPb) of size
/// nPbW × nPbH. Follows §8.5.3.1.2 / §8.5.3.1.3. Returns `max_num_merge_cand`
/// candidates (padded with zero-MV if fewer spatial neighbours qualify).
///
/// Temporal merge is not currently supported; the candidate is absent.
pub fn build_merge_list(
    inter: &InterState,
    x_pb: u32,
    y_pb: u32,
    n_pb_w: u32,
    n_pb_h: u32,
    max_num_merge_cand: u32,
) -> Vec<MergeCand> {
    let mut cands: Vec<MergeCand> = Vec::with_capacity(max_num_merge_cand as usize);

    let a1 = fetch_neighbour(inter, x_pb as i32 - 1, y_pb as i32 + n_pb_h as i32 - 1);
    let b1 = fetch_neighbour(inter, x_pb as i32 + n_pb_w as i32 - 1, y_pb as i32 - 1);
    let b0 = fetch_neighbour(inter, x_pb as i32 + n_pb_w as i32, y_pb as i32 - 1);
    let a0 = fetch_neighbour(inter, x_pb as i32 - 1, y_pb as i32 + n_pb_h as i32);
    let b2 = fetch_neighbour(inter, x_pb as i32 - 1, y_pb as i32 - 1);

    let push = |cands: &mut Vec<MergeCand>, cand: Option<MergeCand>| -> bool {
        if let Some(c) = cand {
            // Prune duplicates.
            if !cands.iter().any(|existing| existing == &c) {
                cands.push(c);
                return true;
            }
        }
        false
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

    // Pad with zero MV + ref_idx=0 per §8.5.3.1.3.
    while (cands.len() as u32) < max_num_merge_cand {
        cands.push(MergeCand {
            ref_idx_l0: 0,
            mv_l0: MotionVector::default(),
        });
    }
    cands.truncate(max_num_merge_cand as usize);
    cands
}

fn fetch_neighbour(inter: &InterState, x: i32, y: i32) -> Option<MergeCand> {
    if x < 0 || y < 0 {
        return None;
    }
    let bx = (x >> 2) as usize;
    let by = (y >> 2) as usize;
    let pb = inter.get(bx, by)?;
    if !pb.valid || pb.is_intra {
        return None;
    }
    Some(MergeCand {
        ref_idx_l0: pb.ref_idx_l0,
        mv_l0: pb.mv_l0,
    })
}

// ---------------------------------------------------------------------------
//  AMVP candidate list
// ---------------------------------------------------------------------------

/// Build the two-entry AMVP MV-predictor list (§8.5.3.1.6) for L0. The
/// spatial neighbours A0/A1 contribute the first candidate; B0/B1/B2
/// contribute the second. Missing slots are filled with zero MV.
pub fn build_amvp_list(
    inter: &InterState,
    x_pb: u32,
    y_pb: u32,
    n_pb_w: u32,
    n_pb_h: u32,
) -> [MotionVector; 2] {
    let a0 = fetch_neighbour(inter, x_pb as i32 - 1, y_pb as i32 + n_pb_h as i32);
    let a1 = fetch_neighbour(inter, x_pb as i32 - 1, y_pb as i32 + n_pb_h as i32 - 1);
    let b0 = fetch_neighbour(inter, x_pb as i32 + n_pb_w as i32, y_pb as i32 - 1);
    let b1 = fetch_neighbour(inter, x_pb as i32 + n_pb_w as i32 - 1, y_pb as i32 - 1);
    let b2 = fetch_neighbour(inter, x_pb as i32 - 1, y_pb as i32 - 1);
    let spatial_a = a0.or(a1).map(|c| c.mv_l0).unwrap_or_default();
    let spatial_b = b0.or(b1).or(b2).map(|c| c.mv_l0).unwrap_or_default();
    let mut out = [spatial_a, spatial_b];
    if out[0] == out[1] {
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

/// Apply the luma 8-tap vertical filter on an 8-sample column centred at
/// (x, y) with fractional `fy`.
fn luma_v_filter_int(p: impl Fn(i32, i32) -> i32, x: i32, y: i32, fy: usize) -> i32 {
    let f = &LUMA_FILTER[fy];
    let mut s = 0i32;
    for j in 0..8 {
        s += f[j] * p(x, y + j as i32 - 3);
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
                // Horizontal only.
                let s = luma_h_filter_int(px, i, j, fx);
                ((s + 32) >> 6).clamp(0, 255)
            } else if fx == 0 {
                let s = luma_v_filter_int(px, i, j, fy);
                ((s + 32) >> 6).clamp(0, 255)
            } else {
                // Two-stage: first the 7 horizontal samples around (i, j), then
                // vertical filter those. 8-bit: intermediate shift = 0, final
                // shift = 12.
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
    // Chroma 1/8 pel fractions; MV is in luma 1/4 pel → convert: fx = mv & 7
    // for 4:2:0 after dividing luma MV by 2 via (mv.x) >> 0 with mask 7.
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
        };
        let mut out = vec![0u8; 8 * 8];
        luma_mc(&pic, 4, 4, 8, 8, MotionVector::new(0, 0), &mut out).unwrap();
        for j in 0..8 {
            for i in 0..8 {
                assert_eq!(out[j * 8 + i], pic.luma[(4 + j) * 16 + (4 + i)]);
            }
        }
    }

    #[test]
    fn merge_list_pads_with_zero_mv() {
        let inter = InterState::new(64, 64);
        let list = build_merge_list(&inter, 16, 16, 8, 8, 5);
        assert_eq!(list.len(), 5);
        for c in &list {
            assert_eq!(c.mv_l0, MotionVector::default());
            assert_eq!(c.ref_idx_l0, 0);
        }
    }

    #[test]
    fn merge_picks_left_neighbour() {
        let mut inter = InterState::new(64, 64);
        // Populate the 4×4 block at (0, 16) with a non-zero MV.
        let pb = PbMotion::inter(0, MotionVector::new(4, -4));
        inter.set_rect(0, 16, 4, 4, pb);
        // Look up merge candidates for PB at (4, 16) of size 8×8. A1 is at
        // (x-1, y+h-1) = (3, 23) which lies in the (0, 20) 4×4 block and
        // (0, 16) if y+h-1=23 rounds down to block row 5? block row = 23>>2 = 5
        // → (0, 20). Set that block too.
        inter.set_rect(0, 20, 4, 4, pb);
        let list = build_merge_list(&inter, 4, 16, 8, 8, 5);
        // First candidate must be the A1 neighbour.
        assert_eq!(list[0].mv_l0, MotionVector::new(4, -4));
    }
}
