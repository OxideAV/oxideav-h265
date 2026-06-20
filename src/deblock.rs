//! §8.7.2.4 — derivation of the deblocking boundary filtering strength.
//!
//! The deblocking filter (§8.7.2) operates on an 8×8 luma grid: vertical
//! edges first, then horizontal edges. For each candidate edge segment
//! the boundary filtering strength `bS ∈ {0, 1, 2}` is derived per
//! §8.7.2.4 from the intra/inter mode, the transform coded-block flags,
//! and the motion vectors / reference pictures of the two prediction
//! blocks straddling the edge. This module implements that derivation —
//! [`derive_boundary_strength`] — reading the per-4×4-block motion / mode
//! state from a [`crate::motion::MotionField`].
//!
//! The §8.7.2.2 / §8.7.2.3 edge-flag derivation (which marks the
//! transform-block / prediction-block boundaries an edge falls on, from
//! the coding / transform tree geometry) is the input `edgeFlags` array;
//! producing it from the decoded partition tree is the picture driver's
//! follow-up. The §8.7.2.5 edge filtering process (the actual sample
//! modification using `bS`, `β` and `tC`) is implemented by the
//! per-sample primitives [`luma_beta_tc`] (§8.7.2.5.3 β/tC, Table 8-12
//! via [`beta_prime`] / [`tc_prime`]), [`luma_sample_decision`]
//! (§8.7.2.5.6) and [`filter_luma_sample`] (§8.7.2.5.7 strong/weak).

use crate::motion::MotionField;

/// §8.7.2.1 — the edge orientation being filtered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    /// `EDGE_VER` — a vertical edge (filtered left-to-right).
    Vertical,
    /// `EDGE_HOR` — a horizontal edge (filtered top-to-bottom).
    Horizontal,
}

/// The §8.7.2.4 boundary filtering strength for one CU's edges, a
/// `(nCbS)×(nCbS)` grid indexed in luma samples (only the §8.7.2.4
/// sampled positions `xDi`/`yDj` carry meaningful values; the rest are 0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryStrength {
    n_cbs: usize,
    /// Row-major `n_cbs * n_cbs`, `bs[y * n_cbs + x]`.
    bs: Vec<u8>,
}

impl BoundaryStrength {
    /// `bS[x][y]` at luma offset `(x, y)` within the coding block.
    ///
    /// # Panics
    /// Panics if `(x, y)` lies outside the `(nCbS)×(nCbS)` block.
    #[must_use]
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.bs[y * self.n_cbs + x]
    }

    /// The coding-block side `nCbS` in luma samples.
    #[inline]
    #[must_use]
    pub fn n_cbs(&self) -> usize {
        self.n_cbs
    }
}

/// Whether the §8.7.2.4 "different reference pictures or different number
/// of motion vectors" predicate holds between the p- and q-side blocks.
///
/// The reference-picture comparison uses the picture identities
/// (`ref_poc_lX`), not the `RefIdxLX` list positions, and is order- and
/// list-independent (spec NOTE 1): block p uses reference-picture set
/// `{ p.l0?, p.l1? }`, block q uses `{ q.l0?, q.l1? }`.
fn different_refs_or_count(p: &crate::motion::MotionCell, q: &crate::motion::MotionCell) -> bool {
    let p_count = u8::from(p.pred_flag_l0) + u8::from(p.pred_flag_l1);
    let q_count = u8::from(q.pred_flag_l0) + u8::from(q.pred_flag_l1);
    if p_count != q_count {
        return true;
    }
    // Compare the *sets* of reference pictures (POCs), list-independent.
    let mut p_refs = Vec::new();
    if p.pred_flag_l0 {
        p_refs.push(p.ref_poc_l0);
    }
    if p.pred_flag_l1 {
        p_refs.push(p.ref_poc_l1);
    }
    let mut q_refs = Vec::new();
    if q.pred_flag_l0 {
        q_refs.push(q.ref_poc_l0);
    }
    if q.pred_flag_l1 {
        q_refs.push(q.ref_poc_l1);
    }
    p_refs.sort_unstable();
    q_refs.sort_unstable();
    p_refs != q_refs
}

/// `|a − b| >= 4` on either component (the §8.7.2.4 quarter-luma-sample
/// MV-difference test).
#[inline]
fn mv_diff_ge4(a: [i32; 2], b: [i32; 2]) -> bool {
    (a[0] - b[0]).abs() >= 4 || (a[1] - b[1]).abs() >= 4
}

/// §8.7.2.4 motion-only bS test for two inter prediction blocks `p` / `q`
/// (the third bullet onward: same/different references, MV differences).
/// Returns `true` when bS should be 1 on motion grounds.
fn motion_bs_is_one(p: &crate::motion::MotionCell, q: &crate::motion::MotionCell) -> bool {
    // Bullet 1: different reference pictures or different number of MVs.
    if different_refs_or_count(p, q) {
        return true;
    }

    let p_count = u8::from(p.pred_flag_l0) + u8::from(p.pred_flag_l1);

    // Bullet 2: one MV each, |Δmv| >= 4.
    if p_count == 1 {
        let pmv = if p.pred_flag_l0 { p.mv_l0 } else { p.mv_l1 };
        let qmv = if q.pred_flag_l0 { q.mv_l0 } else { q.mv_l1 };
        return mv_diff_ge4(pmv, qmv);
    }

    // Two MVs each (p_count == 2). The reference-picture sets are equal
    // (different_refs_or_count returned false).
    if p.ref_poc_l0 != p.ref_poc_l1 {
        // Bullet 3: two different reference pictures. Match each list to
        // the q-side list referencing the same picture.
        // p.l0 ↔ q.(list with same poc as p.l0), p.l1 ↔ the other.
        let (q0, q1) = if q.ref_poc_l0 == p.ref_poc_l0 {
            (q.mv_l0, q.mv_l1)
        } else {
            (q.mv_l1, q.mv_l0)
        };
        mv_diff_ge4(p.mv_l0, q0) || mv_diff_ge4(p.mv_l1, q1)
    } else {
        // Bullet 4: both MVs use the same reference picture. bS = 1 iff
        //   (|Δl0| or |Δl1| >= 4)  AND  (|p.l0−q.l1| or |p.l1−q.l0| >= 4).
        let straight = mv_diff_ge4(p.mv_l0, q.mv_l0) || mv_diff_ge4(p.mv_l1, q.mv_l1);
        let crossed = mv_diff_ge4(p.mv_l0, q.mv_l1) || mv_diff_ge4(p.mv_l1, q.mv_l0);
        straight && crossed
    }
}

/// §8.7.2.4 — derive the `(nCbS)×(nCbS)` boundary filtering strength grid
/// for one luma coding block.
///
/// Inputs:
/// * `field` — the picture's per-4×4-block motion / mode store
///   ([`MotionField`]); the p / q sample look-ups read from it.
/// * `(x_cb, y_cb)` — the coding block's luma top-left position.
/// * `log2_cb_size` — the coding-block side `log2CbSize`.
/// * `edge_type` — `EDGE_VER` or `EDGE_HOR`.
/// * `edge_flags` — the §8.7.2.2/§8.7.2.3 `(nCbS)×(nCbS)` edge-flag grid
///   (row-major, `edge_flags[y * nCbS + x]`), `true` where a transform /
///   prediction block boundary falls.
/// * `tb_edge` — `(nCbS)×(nCbS)` grid, `true` where the edge at that
///   position is *also* a transform-block edge (the §8.7.2.4 second
///   bullet `cbf` test only fires on transform-block edges).
///
/// The bS at every sampled `(xDi, yDj)` is 2 when either neighbouring
/// sample is intra; 1 when the edge is a transform-block edge and either
/// transform block holds a non-zero coefficient; 1 on the §8.7.2.4
/// motion criteria; otherwise 0. Positions not sampled by §8.7.2.4 are 0.
///
/// # Panics
/// Panics if `edge_flags` / `tb_edge` are not `(1 << log2_cb_size)`²
/// long, or if a sampled p / q location falls outside `field`.
pub fn derive_boundary_strength(
    field: &MotionField,
    x_cb: usize,
    y_cb: usize,
    log2_cb_size: u32,
    edge_type: EdgeType,
    edge_flags: &[bool],
    tb_edge: &[bool],
) -> BoundaryStrength {
    let n_cbs = 1usize << log2_cb_size;
    assert_eq!(edge_flags.len(), n_cbs * n_cbs, "edge_flags is nCbS^2");
    assert_eq!(tb_edge.len(), n_cbs * n_cbs, "tb_edge is nCbS^2");
    let mut bs = vec![0u8; n_cbs * n_cbs];

    // §8.7.2.4 sampling grid: EDGE_VER strides 8 in x, 4 in y; EDGE_HOR
    // strides 4 in x, 8 in y.
    let (step_x, step_y) = match edge_type {
        EdgeType::Vertical => (8usize, 4usize),
        EdgeType::Horizontal => (4usize, 8usize),
    };

    let mut y_dj = 0;
    while y_dj < n_cbs {
        let mut x_di = 0;
        while x_di < n_cbs {
            let idx = y_dj * n_cbs + x_di;
            if edge_flags[idx] {
                // p0 / q0 sample positions (§8.7.2.4).
                let (px, py, qx, qy) = match edge_type {
                    EdgeType::Vertical => (x_cb + x_di - 1, y_cb + y_dj, x_cb + x_di, y_cb + y_dj),
                    EdgeType::Horizontal => {
                        (x_cb + x_di, y_cb + y_dj - 1, x_cb + x_di, y_cb + y_dj)
                    }
                };
                let p = field.cell_at(px, py);
                let q = field.cell_at(qx, qy);

                // §8.7.2.4 cascade: intra ⇒ 2; else a coded-coeff
                // transform-block edge OR the motion criteria ⇒ 1; else 0.
                let coeff_edge = tb_edge[idx] && (p.has_nonzero_coeff || q.has_nonzero_coeff);
                let val = if p.is_intra || q.is_intra {
                    2
                } else if coeff_edge || motion_bs_is_one(&p, &q) {
                    1
                } else {
                    0
                };
                bs[idx] = val;
            }
            x_di += step_x;
        }
        y_dj += step_y;
    }

    BoundaryStrength { n_cbs, bs }
}

/// Table 8-12 — `β′` as a function of the input variable `Q` (0..=51).
///
/// The deblocking threshold `β′` is 0 for `Q < 16`, then rises in the
/// pattern documented in Table 8-12. `Q` outside `0..=51` returns the
/// boundary value (the callers clip `Q` to `0..=51` for `β′`, eq. 8-348).
#[must_use]
pub fn beta_prime(q: i32) -> i32 {
    const BETA: [i32; 52] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // Q 0..15
        6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, // Q 16..31
        26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, // Q 32..47
        58, 60, 62, 64, // Q 48..51
    ];
    BETA[q.clamp(0, 51) as usize]
}

/// Table 8-12 — `tC′` as a function of the input variable `Q` (0..=53).
///
/// The clipping threshold `tC′` is 0 for `Q <= 17`, `1` at `Q = 18`,
/// then rises per Table 8-12 up to `24` at `Q = 53`. `Q` outside
/// `0..=53` returns the boundary value (callers clip `Q` to `0..=53`,
/// eqs. 8-350 / 8-383).
#[must_use]
pub fn tc_prime(q: i32) -> i32 {
    const TC: [i32; 54] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, // Q 0..18
        1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, // Q 19..37
        5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24, // Q 38..53
    ];
    TC[q.clamp(0, 53) as usize]
}

/// §8.7.2.5.3 — derive `β` and `tC` for a luma block edge.
///
/// `qp_q` / `qp_p` are the `QpY` values of the coding units containing
/// the `q0,0` and `p0,0` samples; `bs` is the §8.7.2.4 boundary strength
/// (must be 1 or 2 for a filtered edge); `beta_offset_div2` /
/// `tc_offset_div2` are the per-slice `slice_beta_offset_div2` /
/// `slice_tc_offset_div2` syntax-element values; `bit_depth` is
/// `BitDepthY`.
///
/// Returns `(β, tC)` per eqs. 8-347..8-351.
#[must_use]
pub fn luma_beta_tc(
    qp_q: i32,
    qp_p: i32,
    bs: u8,
    beta_offset_div2: i32,
    tc_offset_div2: i32,
    bit_depth: u8,
) -> (i32, i32) {
    let qp_l = (qp_q + qp_p + 1) >> 1; // eq. 8-347
    let q_beta = (qp_l + (beta_offset_div2 << 1)).clamp(0, 51); // eq. 8-348
    let beta = beta_prime(q_beta) * (1 << (i32::from(bit_depth) - 8)); // eq. 8-349
    let q_tc = (qp_l + 2 * (i32::from(bs) - 1) + (tc_offset_div2 << 1)).clamp(0, 53); // eq. 8-350
    let tc = tc_prime(q_tc) * (1 << (i32::from(bit_depth) - 8)); // eq. 8-351
    (beta, tc)
}

/// §8.7.2.5.6 — decision process for a luma sample.
///
/// Returns `dSam`: `true` (1) when the long (strong) filter conditions
/// across `p0`/`p3`/`q0`/`q3` hold, given `dpq`, `β` and `tC`.
#[inline]
#[must_use]
pub fn luma_sample_decision(
    p0: i32,
    p3: i32,
    q0: i32,
    q3: i32,
    dpq: i32,
    beta: i32,
    tc: i32,
) -> bool {
    dpq < (beta >> 2)
        && (p3 - p0).abs() + (q0 - q3).abs() < (beta >> 3)
        && (p0 - q0).abs() < (5 * tc + 1) >> 1
}

/// Decisions emitted by the §8.7.2.5.3 luma edge decision process for a
/// 4-row edge segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LumaEdgeDecision {
    /// `dE`: 0 = no filtering, 1 = weak (normal), 2 = strong (long).
    pub de: u8,
    /// `dEp`: when 1, the `p1` sample is also filtered (weak filter).
    pub dep: u8,
    /// `dEq`: when 1, the `q1` sample is also filtered (weak filter).
    pub deq: u8,
    /// `β` (eq. 8-349).
    pub beta: i32,
    /// `tC` (eq. 8-351).
    pub tc: i32,
}

/// The result of §8.7.2.5.7 luma-sample filtering for one boundary row.
///
/// `p[i]` / `q[i]` are the filtered sample values (`pi'` / `qj'`); only
/// the first `ndp` `p[]` and `ndq` `q[]` entries are valid replacements
/// for the input samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LumaFilterOut {
    /// `nDp` — number of filtered samples on the p side (0..=3).
    pub ndp: usize,
    /// `nDq` — number of filtered samples on the q side (0..=3).
    pub ndq: usize,
    /// `p0'`, `p1'`, `p2'` (only `0..ndp` valid).
    pub p: [i32; 3],
    /// `q0'`, `q1'`, `q2'` (only `0..ndq` valid).
    pub q: [i32; 3],
}

/// §8.7.2.5.7 — filtering process for a luma sample row.
///
/// `p` / `q` are the four samples each side of the boundary (`p[0]` is
/// `p0`, closest to the edge); `dec` carries `dE` / `dEp` / `dEq` and
/// `tC`; `bit_depth` is `BitDepthY` for the final `Clip1Y`.
///
/// Strong filtering (`dE == 2`) writes `p0'..p2'` / `q0'..q2'`; weak
/// filtering (`dE == 1`) writes `p0'` (+`p1'` when `dEp`) and `q0'`
/// (+`q1'` when `dEq`). The PCM / transquant-bypass / palette
/// suppressions are applied by the caller (they need per-CU state).
#[must_use]
pub fn filter_luma_sample(
    p: [i32; 4],
    q: [i32; 4],
    de: u8,
    dep: u8,
    deq: u8,
    tc: i32,
    bit_depth: u8,
) -> LumaFilterOut {
    let [p0, p1, p2, p3] = p;
    let [q0, q1, q2, q3] = q;
    let clip = |v: i32| crate::picture::clip1(v, bit_depth);
    if de == 2 {
        // Strong filter (eqs. 8-389..8-394).
        let p0p =
            (p0 - 2 * tc).max((p0 + 2 * tc).min((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3));
        let p1p = (p1 - 2 * tc).max((p1 + 2 * tc).min((p2 + p1 + p0 + q0 + 2) >> 2));
        let p2p = (p2 - 2 * tc).max((p2 + 2 * tc).min((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3));
        let q0p =
            (q0 - 2 * tc).max((q0 + 2 * tc).min((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3));
        let q1p = (q1 - 2 * tc).max((q1 + 2 * tc).min((p0 + q0 + q1 + q2 + 2) >> 2));
        let q2p = (q2 - 2 * tc).max((q2 + 2 * tc).min((p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3));
        return LumaFilterOut {
            ndp: 3,
            ndq: 3,
            p: [p0p, p1p, p2p],
            q: [q0p, q1p, q2p],
        };
    }
    // Weak filter (eqs. 8-395..8-402).
    let mut out = LumaFilterOut {
        ndp: 0,
        ndq: 0,
        p: [p0, p1, p2],
        q: [q0, q1, q2],
    };
    let delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4; // eq. 8-395
    if delta.abs() < tc * 10 {
        let delta = delta.clamp(-tc, tc); // eq. 8-396
        out.p[0] = clip(p0 + delta); // eq. 8-397
        out.q[0] = clip(q0 - delta); // eq. 8-398
        if dep == 1 {
            let dp = ((((p2 + p0 + 1) >> 1) - p1 + delta) >> 1).clamp(-(tc >> 1), tc >> 1); // eq. 8-399
            out.p[1] = clip(p1 + dp); // eq. 8-400
        }
        if deq == 1 {
            let dq = ((((q2 + q0 + 1) >> 1) - q1 - delta) >> 1).clamp(-(tc >> 1), tc >> 1); // eq. 8-401
            out.q[1] = clip(q1 + dq); // eq. 8-402
        }
        out.ndp = (dep + 1) as usize; // eq. 8-403 (nDp = dEp + 1)
        out.ndq = (deq + 1) as usize; // (nDq = dEq + 1)
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::MotionCell;

    fn inter_cell(mv0: [i32; 2], poc0: i32) -> MotionCell {
        MotionCell {
            is_intra: false,
            has_nonzero_coeff: false,
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_poc_l0: poc0,
            ref_poc_l1: i32::MIN,
            mv_l0: mv0,
            mv_l1: [0, 0],
        }
    }

    /// A 16×16 CU's internal vertical edge at xDi=8, with the left
    /// (p-side) half intra ⇒ bS == 2 at the sampled positions.
    ///
    /// The §8.7.2.4 vertical-edge x stride is 8, so a 16-wide CU samples
    /// xDi ∈ {0, 8}; the internal edge is at xDi=8.
    #[test]
    fn intra_neighbour_gives_bs2() {
        let mut field = MotionField::new(32, 32);
        // CU at (0,0), 16×16. q-side (right half) inter; p-side (left)
        // stays intra (background).
        field.fill_rect(8, 0, 8, 16, inter_cell([0, 0], 0));
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let tb_edge = vec![false; n * n];
        // Internal vertical edge at xDi=8, sampled yDj 0,4,8,12.
        for yj in (0..16).step_by(4) {
            edge_flags[yj * n + 8] = true;
        }
        let bs =
            derive_boundary_strength(&field, 0, 0, 4, EdgeType::Vertical, &edge_flags, &tb_edge);
        // p0 = col 7 (intra), q0 = col 8 (inter) ⇒ bS == 2.
        assert_eq!(bs.at(8, 0), 2);
        assert_eq!(bs.at(8, 12), 2);
        // xDi=4 not sampled for EDGE_VER (8-stride).
        assert_eq!(bs.at(4, 0), 0);
    }

    /// Transform-block edge with a non-zero coefficient on an inter block
    /// ⇒ bS == 1.
    #[test]
    fn tb_edge_with_coeff_gives_bs1() {
        let mut field = MotionField::new(32, 32);
        let mut p = inter_cell([0, 0], 0);
        p.has_nonzero_coeff = true;
        let q = inter_cell([0, 0], 0);
        field.fill_rect(0, 0, 16, 16, q);
        field.fill_rect(0, 0, 8, 16, p); // left half carries the coeff
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let mut tb_edge = vec![false; n * n];
        // Vertical edge at xDi=8 (a transform-block edge).
        for yj in (0..16).step_by(4) {
            edge_flags[yj * n + 8] = true;
            tb_edge[yj * n + 8] = true;
        }
        let bs =
            derive_boundary_strength(&field, 0, 0, 4, EdgeType::Vertical, &edge_flags, &tb_edge);
        // p0 = col 7 (coeff block), q0 = col 8 ⇒ bS == 1.
        assert_eq!(bs.at(8, 0), 1);
        assert_eq!(bs.at(8, 4), 1);
    }

    /// Two inter blocks, same reference, MV difference >= 4 ⇒ bS == 1.
    #[test]
    fn mv_diff_gives_bs1() {
        let mut field = MotionField::new(32, 32);
        field.fill_rect(0, 0, 8, 16, inter_cell([0, 0], 0)); // p side
        field.fill_rect(8, 0, 8, 16, inter_cell([4, 0], 0)); // q side, Δ=4
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let tb_edge = vec![false; n * n];
        for yj in (0..16).step_by(4) {
            edge_flags[yj * n + 8] = true;
        }
        let bs =
            derive_boundary_strength(&field, 0, 0, 4, EdgeType::Vertical, &edge_flags, &tb_edge);
        assert_eq!(bs.at(8, 0), 1);
        assert_eq!(bs.at(8, 12), 1);
    }

    /// Two inter blocks, same reference + same MV ⇒ bS == 0.
    #[test]
    fn same_motion_gives_bs0() {
        let mut field = MotionField::new(32, 32);
        field.fill_rect(0, 0, 16, 16, inter_cell([3, 1], 0)); // identical motion
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let tb_edge = vec![false; n * n];
        for yj in (0..16).step_by(4) {
            edge_flags[yj * n + 8] = true;
        }
        let bs =
            derive_boundary_strength(&field, 0, 0, 4, EdgeType::Vertical, &edge_flags, &tb_edge);
        assert_eq!(bs.at(8, 0), 0);
    }

    /// Different reference pictures ⇒ bS == 1 even with identical MVs.
    #[test]
    fn different_ref_gives_bs1() {
        let mut field = MotionField::new(32, 32);
        field.fill_rect(0, 0, 8, 16, inter_cell([2, 2], 0)); // poc 0
        field.fill_rect(8, 0, 8, 16, inter_cell([2, 2], 8)); // poc 8
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let tb_edge = vec![false; n * n];
        for yj in (0..16).step_by(4) {
            edge_flags[yj * n + 8] = true;
        }
        let bs =
            derive_boundary_strength(&field, 0, 0, 4, EdgeType::Vertical, &edge_flags, &tb_edge);
        assert_eq!(bs.at(8, 0), 1);
    }

    /// Horizontal edges sample 4-stride x, 8-stride y.
    #[test]
    fn horizontal_edge_sampling() {
        let mut field = MotionField::new(32, 32);
        // top half intra (background), bottom half inter; internal
        // horizontal edge at yDj=8.
        field.fill_rect(0, 8, 16, 8, inter_cell([0, 0], 0));
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let tb_edge = vec![false; n * n];
        for xi in (0..16).step_by(4) {
            edge_flags[8 * n + xi] = true;
        }
        let bs =
            derive_boundary_strength(&field, 0, 0, 4, EdgeType::Horizontal, &edge_flags, &tb_edge);
        // yDj=8: p0 = row 7 (intra), q0 = row 8 (inter) ⇒ bS == 2.
        assert_eq!(bs.at(0, 8), 2);
        assert_eq!(bs.at(12, 8), 2);
        // yDj=4 not sampled for EDGE_HOR.
        assert_eq!(bs.at(0, 4), 0);
    }

    /// edgeFlags == 0 ⇒ bS stays 0 regardless of mode.
    #[test]
    fn no_edge_flag_gives_bs0() {
        let field = MotionField::new(16, 16); // intra background
        let n = 8;
        let edge_flags = vec![false; n * n];
        let tb_edge = vec![false; n * n];
        let bs =
            derive_boundary_strength(&field, 0, 0, 3, EdgeType::Vertical, &edge_flags, &tb_edge);
        assert_eq!(bs.at(0, 0), 0);
    }

    /// Bullet-4 path: two MVs each, same single reference picture on both
    /// sides; bS == 1 only when both the straight and crossed MV-pair
    /// differences reach 4.
    #[test]
    fn bipred_same_ref_bullet4() {
        let mut field = MotionField::new(32, 32);
        let bi = |mv0: [i32; 2], mv1: [i32; 2]| MotionCell {
            is_intra: false,
            has_nonzero_coeff: false,
            pred_flag_l0: true,
            pred_flag_l1: true,
            ref_poc_l0: 0,
            ref_poc_l1: 0, // same reference picture for both lists
            mv_l0: mv0,
            mv_l1: mv1,
        };
        // p: l0=[0,0] l1=[0,0]; q: l0=[4,0] l1=[4,0].
        // straight: |Δl0|>=4 ⇒ true. crossed: |p.l0−q.l1|=4 ⇒ true ⇒ bS 1.
        field.fill_rect(0, 0, 8, 16, bi([0, 0], [0, 0]));
        field.fill_rect(8, 0, 8, 16, bi([4, 0], [4, 0]));
        let n = 16;
        let mut edge_flags = vec![false; n * n];
        let tb_edge = vec![false; n * n];
        for yj in (0..16).step_by(4) {
            edge_flags[yj * n + 8] = true;
        }
        let bs =
            derive_boundary_strength(&field, 0, 0, 4, EdgeType::Vertical, &edge_flags, &tb_edge);
        assert_eq!(bs.at(8, 0), 1);
    }

    // ----- §8.7.2.5.3 / .6 / .7 luma filtering primitives -----

    /// Table 8-12 spot checks: β′ is 0 below Q=16 and tracks the listed
    /// breakpoints; tC′ is 0 up to Q=17 and 1 at Q=18.
    #[test]
    fn table_8_12_breakpoints() {
        assert_eq!(beta_prime(15), 0);
        assert_eq!(beta_prime(16), 6);
        assert_eq!(beta_prime(29), 20);
        assert_eq!(beta_prime(51), 64);
        assert_eq!(tc_prime(17), 0);
        assert_eq!(tc_prime(18), 1);
        assert_eq!(tc_prime(53), 24);
        // Clip behaviour at the table edges.
        assert_eq!(beta_prime(-3), 0);
        assert_eq!(tc_prime(99), 24);
    }

    /// §8.7.2.5.3 β/tC at 8-bit: qPL = (QpQ+QpP+1)>>1, no offsets, bS=2.
    /// QpQ=QpP=37 ⇒ qPL=37 ⇒ Q_beta=37 ⇒ β′=36, β=36; Q_tc=37+2=39 ⇒
    /// tC′=5, tC=5.
    #[test]
    fn luma_beta_tc_no_offset_8bit() {
        let (beta, tc) = luma_beta_tc(37, 37, 2, 0, 0, 8);
        assert_eq!(beta, 36);
        assert_eq!(tc, 5);
        // bS=1 lowers the tC index by 2 (Q_tc=37): tC′=4.
        let (_, tc1) = luma_beta_tc(37, 37, 1, 0, 0, 8);
        assert_eq!(tc1, 4);
    }

    /// 10-bit scaling: β and tC are multiplied by 1<<(BitDepthY−8)=4.
    #[test]
    fn luma_beta_tc_10bit_scaled() {
        let (beta, tc) = luma_beta_tc(37, 37, 2, 0, 0, 10);
        assert_eq!(beta, 36 * 4);
        assert_eq!(tc, 5 * 4);
    }

    /// §8.7.2.5.6: a flat region (all-equal samples) passes the decision.
    #[test]
    fn luma_sample_decision_flat_passes() {
        // dpq=0 < β>>2; |p3−p0|+|q0−q3|=0 < β>>3; |p0−q0|=0 < (5tC+1)>>1.
        assert!(luma_sample_decision(128, 128, 128, 128, 0, 32, 4));
        // A large step at the boundary fails the |p0−q0| test.
        assert!(!luma_sample_decision(64, 64, 200, 200, 0, 32, 4));
    }

    /// §8.7.2.5.7 strong filter: a flat block stays flat (idempotent).
    #[test]
    fn strong_filter_flat_is_idempotent() {
        let s = [100; 4];
        let out = filter_luma_sample(s, s, 2, 0, 0, 6, 8);
        assert_eq!(out.ndp, 3);
        assert_eq!(out.ndq, 3);
        assert_eq!(out.p, [100, 100, 100]);
        assert_eq!(out.q, [100, 100, 100]);
    }

    /// §8.7.2.5.7 strong filter clips each output to ±2·tC of the input.
    #[test]
    fn strong_filter_clips_to_two_tc() {
        // A hard step: p side 0, q side 255. tC=2 ⇒ p0' clipped to
        // p0+2·tC = 0+4 = 4; q0' clipped to q0−2·tC = 255−4 = 251.
        let out = filter_luma_sample([0, 0, 0, 0], [255, 255, 255, 255], 2, 0, 0, 2, 8);
        assert_eq!(out.p[0], 4);
        assert_eq!(out.q[0], 251);
    }

    /// §8.7.2.5.7 weak filter: small ramp, dEp=dEq=0 filters only p0/q0.
    /// p0=98,q0=102,p1=q1=100 ⇒ δ=(9·4−3·0+8)>>4=2; |δ|<10·tC; clamp to
    /// [−tC,tC]=2 ⇒ p0'=100, q0'=100; nDp=nDq=1.
    #[test]
    fn weak_filter_centres_small_step() {
        let out = filter_luma_sample([98, 100, 100, 100], [102, 100, 100, 100], 1, 0, 0, 5, 8);
        assert_eq!(out.ndp, 1);
        assert_eq!(out.ndq, 1);
        assert_eq!(out.p[0], 100);
        assert_eq!(out.q[0], 100);
    }

    /// §8.7.2.5.7 weak filter no-op when |δ| ≥ 10·tC.
    #[test]
    fn weak_filter_skips_large_delta() {
        // Big step with tiny tC ⇒ |δ| ≥ 10·tC ⇒ nDp=nDq=0, no change.
        let out = filter_luma_sample([0, 0, 0, 0], [255, 255, 255, 255], 1, 1, 1, 1, 8);
        assert_eq!(out.ndp, 0);
        assert_eq!(out.ndq, 0);
        assert_eq!(out.p[0], 0);
        assert_eq!(out.q[0], 255);
    }
}
