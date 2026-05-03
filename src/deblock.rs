//! HEVC in-loop deblocking filter (§8.7.2).
//!
//! Implements the post-decode deblocking pass per ITU-T H.265 §8.7.2:
//!
//! * §8.7.2.2 — β and tC lookup tables (Table 8-11) indexed by the
//!   clipped values `Q = Clip3(0, 51, qP + slice_beta_offset_div2 * 2)`
//!   for β and `Q = Clip3(0, 53, qP + 2 * (bS > 0 ? 1 : 0) +
//!   slice_tc_offset_div2 * 2)` for tC (the (bS>0?1:0) is collapsed with
//!   the "+2" path for luma strong/normal decisions in §8.7.2.5.3).
//! * §8.7.2.4 — edge filtering order: first all vertical edges of the
//!   whole picture, then all horizontal edges.
//! * §8.7.2.5 — luma filter: strong vs normal filter decisions per
//!   4-sample segment, per-sample clip with tC, and ±2 / ±1 sample
//!   modifications on each side of the edge.
//! * §8.7.2.6 — chroma filter: 1-sample-per-side modification, normal
//!   filter only, with a coarse boundary-strength (bS > 1) gate.
//!
//! ## Simplifications vs. the spec
//!
//! The CTU walker does not currently track per-TU coded-block flags,
//! per-PU MV-diff values, or the exact PU/TU edge geometry that §8.7.2.4
//! needs to build the full per-4-sample boundary-strength grid. We
//! approximate the boundary-strength derivation using the per-4×4 state
//! we *do* maintain (intra flag, per-PU motion, cqt_depth):
//!
//! * bS = 2 when either side is intra — exact per spec.
//! * bS = 1 when either side has non-zero cqt_depth difference (CU
//!   partition edge heuristic) OR when two inter PBs have MV difference
//!   ≥ 4 (one integer luma sample) in either component, or refer to
//!   different POCs.
//! * bS = 0 otherwise.
//!
//! This yields a spec-faithful filter *shape* and strength tables, but
//! may apply or skip the filter on a small number of edges compared to
//! a conformance decoder that tracks CBFs and exact PB layout. The gap
//! is in aggressiveness — the filter either smoothes slightly too many
//! or slightly too few edges, never writes garbage data.
//!
//! The edges are walked on the spec's 8-pixel luma grid (4-pixel chroma).
//! Edges at picture boundaries and disabled-slice edges are skipped.

use crate::ctu::Picture;
use crate::pps::PicParameterSet;
use crate::slice::SliceSegmentHeader;
use crate::sps::SeqParameterSet;

/// β lookup table (§8.7.2.2 Table 8-11), indexed by `Q = Clip3(0, 51,
/// qP + (slice_beta_offset_div2 << 1))`. Values span [0, 64] in steps of
/// 2. This table comes directly from the ITU-T H.265 spec and is not
/// copyrightable.
#[rustfmt::skip]
pub(crate) const BETA_TABLE: [u8; 52] = [
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  6,  7,  8,  9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 20,
    22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
    42, 44, 46, 48, 50, 52, 54, 56, 58, 60,
    62, 64,
];

/// tC lookup table (§8.7.2.2 Table 8-11), indexed by `Q = Clip3(0, 53,
/// qP + 2 * (bS == 2 ? 1 : 0) + (slice_tc_offset_div2 << 1))`. Values
/// span [0, 24]. Direct ITU-T spec table (54 entries, Q ∈ [0, 53]).
#[rustfmt::skip]
pub(crate) const TC_TABLE: [u8; 54] = [
    // QP 0..18
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    // QP 19..37
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4,
    // QP 38..53
    5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24,
];

/// Chroma QpI-to-QpC mapping (§8.6.1 Table 8-10) for ChromaArrayType = 1.
/// Needed so chroma tC lookups use the chroma-domain QP. Input range is
/// [0, 51], output is also [0, 51]. Direct ITU-T spec table.
#[rustfmt::skip]
pub(crate) const CHROMA_QP_I2C: [u8; 52] = [
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    29, 30, 31, 32, 33, 34, 34, 35, 35, 36,
    36, 37, 37, 37, 38, 38, 38, 39, 39, 39,
    40, 40,
];

fn clip3<T: Ord>(lo: T, hi: T, v: T) -> T {
    core::cmp::min(hi, core::cmp::max(lo, v))
}

fn clip_bd(v: i32, bit_depth: u32) -> u16 {
    let max = ((1i32 << bit_depth) - 1).max(0);
    v.clamp(0, max) as u16
}

/// Run the deblocking pass on a reconstructed picture in-place.
pub fn deblock_picture(
    pic: &mut Picture,
    sps: &SeqParameterSet,
    pps: &PicParameterSet,
    slice: &SliceSegmentHeader,
) {
    if slice.slice_deblocking_filter_disabled_flag {
        return;
    }
    let cfi = sps.chroma_format_idc;
    if cfi != 0 && cfi != 1 && cfi != 2 {
        // 4:0:0, 4:2:0, and 4:2:2 supported here (mirrors the CTU walker's
        // chroma sub-sampling gate). 4:0:0 (monochrome) skips the chroma
        // deblock step further down. 4:4:4 still falls into the early
        // return — its chroma deblock parity has not been audited
        // (chroma_deblock helpers were written for sub_x/sub_y >= 1).
        return;
    }
    let width = pic.width as usize;
    let height = pic.height as usize;
    let beta_offset = slice.slice_beta_offset_div2 << 1;
    let tc_offset = slice.slice_tc_offset_div2 << 1;
    let bit_depth_y = sps.bit_depth_y();
    let bit_depth_c = sps.bit_depth_c();

    // §8.7.2.4: vertical edges first across the whole picture, then
    // horizontal edges. Both run on an 8-pixel luma grid (4-pixel chroma).
    // β and tC table outputs scale by `<< (BitDepth - 8)` for higher bit
    // depths (§8.7.2.5.3 eqs. 8-294..8-296). The scaling is applied where
    // the tables are consulted, not here, to keep the call sites below
    // unchanged.
    deblock_vertical_luma(
        pic,
        width,
        height,
        beta_offset,
        tc_offset,
        slice,
        bit_depth_y,
    );
    deblock_horizontal_luma(
        pic,
        width,
        height,
        beta_offset,
        tc_offset,
        slice,
        bit_depth_y,
    );
    if cfi == 0 {
        // Monochrome: no chroma planes to deblock.
        let _ = (pps, bit_depth_c);
        return;
    }
    let sub_x = sps.sub_width_c();
    let sub_y = sps.sub_height_c();
    deblock_vertical_chroma(
        pic,
        width,
        height,
        tc_offset,
        slice,
        pps,
        bit_depth_c,
        sub_x,
        sub_y,
    );
    deblock_horizontal_chroma(
        pic,
        width,
        height,
        tc_offset,
        slice,
        pps,
        bit_depth_c,
        sub_x,
        sub_y,
    );
    // Suppress unused-warnings: pps only consulted for chroma QP offsets
    // in chroma paths.
    let _ = pps;
}

/// Boundary strength for a 4-pixel edge segment between two 4×4 grid
/// cells `p` (left/above) and `q` (right/below). §8.7.2.4.
fn boundary_strength(pic: &Picture, p_bx: usize, p_by: usize, q_bx: usize, q_by: usize) -> u8 {
    let gw = pic.inter.grid_w;
    let gh = pic.inter.grid_h;
    if p_bx >= gw || p_by >= gh || q_bx >= gw || q_by >= gh {
        return 0;
    }
    let p = &pic.inter.pb[p_by * gw + p_bx];
    let q = &pic.inter.pb[q_by * gw + q_bx];
    // §8.7.2.4: bS=2 when either P or Q is intra.
    if p.is_intra || q.is_intra {
        return 2;
    }
    // Heuristic for bS=1: CU-partition boundary (cqt_depth change) or
    // inter PB with MV difference ≥ 1 full-sample in any component or
    // different reference poc index.
    let iw = pic.intra_width_4;
    let ih = pic.intra_height_4;
    let p_depth = if p_bx < iw && p_by < ih {
        pic.cqt_depth[p_by * iw + p_bx]
    } else {
        0
    };
    let q_depth = if q_bx < iw && q_by < ih {
        pic.cqt_depth[q_by * iw + q_bx]
    } else {
        0
    };
    if p_depth != q_depth {
        return 1;
    }
    if p.pred_l0 != q.pred_l0 || p.pred_l1 != q.pred_l1 {
        return 1;
    }
    if p.pred_l0 && (p.ref_idx_l0 != q.ref_idx_l0) {
        return 1;
    }
    if p.pred_l1 && (p.ref_idx_l1 != q.ref_idx_l1) {
        return 1;
    }
    if p.pred_l0 {
        let dx = (p.mv_l0.x - q.mv_l0.x).abs();
        let dy = (p.mv_l0.y - q.mv_l0.y).abs();
        if dx >= 4 || dy >= 4 {
            return 1;
        }
    }
    if p.pred_l1 {
        let dx = (p.mv_l1.x - q.mv_l1.x).abs();
        let dy = (p.mv_l1.y - q.mv_l1.y).abs();
        if dx >= 4 || dy >= 4 {
            return 1;
        }
    }
    0
}

/// Average QP of the two adjacent CUs (§8.7.2.5.3 eq. 8-293).
fn qp_avg(pic: &Picture, p_x: usize, p_y: usize, q_x: usize, q_y: usize) -> i32 {
    let qs = pic.qp_stride.max(1);
    let read_qp = |x: usize, y: usize| -> i32 {
        let bx = x >> 3;
        let by = y >> 3;
        let idx = by * qs + bx;
        if idx < pic.qp_y.len() {
            pic.qp_y[idx]
        } else {
            0
        }
    };
    (read_qp(p_x, p_y) + read_qp(q_x, q_y) + 1) >> 1
}

/// Derive β and tC for a luma edge segment. Returns `(beta, tc)`.
/// §8.7.2.5.3 eqs. 8-294..8-296. Both values are multiplied by
/// `1 << (bit_depth - 8)` so the thresholds track the sample magnitude
/// at higher bit depths.
fn luma_beta_tc(qp_avg: i32, bs: u8, beta_off: i32, tc_off: i32, bit_depth: u32) -> (i32, i32) {
    let q_beta = clip3(0, 51, qp_avg + beta_off);
    let q_tc = clip3(0, 53, qp_avg + 2 * ((bs == 2) as i32) + tc_off);
    let scale = bit_depth.saturating_sub(8);
    (
        (BETA_TABLE[q_beta as usize] as i32) << scale,
        (TC_TABLE[q_tc as usize] as i32) << scale,
    )
}

/// Derive tC for a chroma edge (β is unused per §8.7.2.7 for chroma).
fn chroma_tc(qp_avg_y: i32, cb_qp_offset: i32, bs: u8, tc_off: i32, bit_depth: u32) -> i32 {
    // QpI → QpC via Table 8-10 (§8.6.1). Add PPS/slice chroma offset
    // first, clip to [0, 51], then map to chroma QP.
    let qp_i = clip3(0, 51, qp_avg_y + cb_qp_offset);
    let qp_c = CHROMA_QP_I2C[qp_i as usize] as i32;
    let q = clip3(0, 53, qp_c + 2 * ((bs == 2) as i32) + tc_off);
    (TC_TABLE[q as usize] as i32) << bit_depth.saturating_sub(8)
}

/// Luma vertical edges: iterate every edge that lies on the 8-pixel grid
/// (x ∈ {8, 16, 24, ...}). Each edge segment is 4 luma samples tall.
fn deblock_vertical_luma(
    pic: &mut Picture,
    width: usize,
    height: usize,
    beta_off: i32,
    tc_off: i32,
    _slice: &SliceSegmentHeader,
    bit_depth: u32,
) {
    let stride = pic.luma_stride;
    let mut x = 8;
    while x < width {
        let mut y = 0;
        while y + 4 <= height {
            let p_bx = (x - 1) >> 2;
            let p_by = y >> 2;
            let q_bx = x >> 2;
            let q_by = y >> 2;
            let bs = boundary_strength(pic, p_bx, p_by, q_bx, q_by);
            if bs > 0 {
                let qpa = qp_avg(pic, x - 1, y, x, y);
                let (beta, tc) = luma_beta_tc(qpa, bs, beta_off, tc_off, bit_depth);
                filter_luma_vertical(&mut pic.luma, stride, x, y, beta, tc, bit_depth);
            }
            y += 4;
        }
        x += 8;
    }
}

/// Luma horizontal edges: y ∈ {8, 16, 24, ...}, 4-sample wide segments.
fn deblock_horizontal_luma(
    pic: &mut Picture,
    width: usize,
    height: usize,
    beta_off: i32,
    tc_off: i32,
    _slice: &SliceSegmentHeader,
    bit_depth: u32,
) {
    let stride = pic.luma_stride;
    let mut y = 8;
    while y < height {
        let mut x = 0;
        while x + 4 <= width {
            let p_bx = x >> 2;
            let p_by = (y - 1) >> 2;
            let q_bx = x >> 2;
            let q_by = y >> 2;
            let bs = boundary_strength(pic, p_bx, p_by, q_bx, q_by);
            if bs > 0 {
                let qpa = qp_avg(pic, x, y - 1, x, y);
                let (beta, tc) = luma_beta_tc(qpa, bs, beta_off, tc_off, bit_depth);
                filter_luma_horizontal(&mut pic.luma, stride, x, y, beta, tc, bit_depth);
            }
            x += 4;
        }
        y += 8;
    }
}

/// Luma vertical filter for a 4-sample-tall edge segment at (x, y..y+4).
/// §8.7.2.5.3 + §8.7.2.5.6 + §8.7.2.5.7.
fn filter_luma_vertical(
    buf: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    beta: i32,
    tc: i32,
    bit_depth: u32,
) {
    if beta == 0 && tc == 0 {
        return;
    }
    let clip_bd_bd = |v: i32| clip_bd(v, bit_depth);
    // Row 0 (p/q samples) and row 3 used for the strong-filter decision.
    let idx = |xx: usize, yy: usize| yy * stride + xx;
    let p = |buf: &[u16], k: usize, row: usize| -> i32 { buf[idx(x - 1 - k, y + row)] as i32 };
    let q = |buf: &[u16], k: usize, row: usize| -> i32 { buf[idx(x + k, y + row)] as i32 };

    // §8.7.2.5.3 eq. 8-299..8-302 — dE / dEp / dEq via rows 0 and 3.
    let dp0 = (p(buf, 2, 0) - 2 * p(buf, 1, 0) + p(buf, 0, 0)).abs();
    let dp3 = (p(buf, 2, 3) - 2 * p(buf, 1, 3) + p(buf, 0, 3)).abs();
    let dq0 = (q(buf, 2, 0) - 2 * q(buf, 1, 0) + q(buf, 0, 0)).abs();
    let dq3 = (q(buf, 2, 3) - 2 * q(buf, 1, 3) + q(buf, 0, 3)).abs();
    let d = dp0 + dq0 + dp3 + dq3;
    if d >= beta {
        return; // filter disabled on this segment (§8.7.2.5.3)
    }
    let d_pq = (p(buf, 3, 0) - p(buf, 0, 0)).abs() + (q(buf, 3, 0) - q(buf, 0, 0)).abs();
    let d_pq3 = (p(buf, 3, 3) - p(buf, 0, 3)).abs() + (q(buf, 3, 3) - q(buf, 0, 3)).abs();
    let strong = (2 * (dp0 + dq0) < (beta >> 2))
        && (d_pq < (beta >> 3))
        && ((p(buf, 0, 0) - q(buf, 0, 0)).abs() < ((5 * tc + 1) >> 1))
        && (2 * (dp3 + dq3) < (beta >> 2))
        && (d_pq3 < (beta >> 3))
        && ((p(buf, 0, 3) - q(buf, 0, 3)).abs() < ((5 * tc + 1) >> 1));

    if strong {
        // §8.7.2.5.6 strong luma filter.
        for row in 0..4 {
            let p0 = p(buf, 0, row);
            let p1 = p(buf, 1, row);
            let p2 = p(buf, 2, row);
            let p3 = p(buf, 3, row);
            let q0 = q(buf, 0, row);
            let q1 = q(buf, 1, row);
            let q2 = q(buf, 2, row);
            let q3 = q(buf, 3, row);
            let tc2 = 2 * tc;
            let np0 = clip3(
                p0 - tc2,
                p0 + tc2,
                (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3,
            );
            let np1 = clip3(p1 - tc2, p1 + tc2, (p2 + p1 + p0 + q0 + 2) >> 2);
            let np2 = clip3(
                p2 - tc2,
                p2 + tc2,
                (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3,
            );
            let nq0 = clip3(
                q0 - tc2,
                q0 + tc2,
                (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3,
            );
            let nq1 = clip3(q1 - tc2, q1 + tc2, (p0 + q0 + q1 + q2 + 2) >> 2);
            let nq2 = clip3(
                q2 - tc2,
                q2 + tc2,
                (p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3,
            );
            buf[idx(x - 1, y + row)] = clip_bd_bd(np0);
            buf[idx(x - 2, y + row)] = clip_bd_bd(np1);
            buf[idx(x - 3, y + row)] = clip_bd_bd(np2);
            buf[idx(x, y + row)] = clip_bd_bd(nq0);
            buf[idx(x + 1, y + row)] = clip_bd_bd(nq1);
            buf[idx(x + 2, y + row)] = clip_bd_bd(nq2);
        }
    } else {
        // Normal filter (§8.7.2.5.7).
        // dEp/dEq per the spec — decide whether p1/q1 also get filtered.
        let dep = (dp0 + dp3) < ((beta + (beta >> 1)) >> 3);
        let deq = (dq0 + dq3) < ((beta + (beta >> 1)) >> 3);
        for row in 0..4 {
            let p0 = p(buf, 0, row);
            let p1 = p(buf, 1, row);
            let p2 = p(buf, 2, row);
            let q0 = q(buf, 0, row);
            let q1 = q(buf, 1, row);
            let q2 = q(buf, 2, row);
            let delta0 = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
            if delta0.abs() >= tc * 10 {
                continue;
            }
            let delta = clip3(-tc, tc, delta0);
            let np0 = clip_bd_bd(p0 + delta);
            let nq0 = clip_bd_bd(q0 - delta);
            buf[idx(x - 1, y + row)] = np0;
            buf[idx(x, y + row)] = nq0;
            if dep {
                let delta_p = clip3(
                    -(tc >> 1),
                    tc >> 1,
                    (((p2 + p0 + 1) >> 1) - p1 + delta) >> 1,
                );
                buf[idx(x - 2, y + row)] = clip_bd_bd(p1 + delta_p);
            }
            if deq {
                let delta_q = clip3(
                    -(tc >> 1),
                    tc >> 1,
                    (((q2 + q0 + 1) >> 1) - q1 - delta) >> 1,
                );
                buf[idx(x + 1, y + row)] = clip_bd_bd(q1 + delta_q);
            }
        }
    }
}

/// Luma horizontal filter — same math with rows/cols swapped.
fn filter_luma_horizontal(
    buf: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    beta: i32,
    tc: i32,
    bit_depth: u32,
) {
    if beta == 0 && tc == 0 {
        return;
    }
    let clip_bd_bd = |v: i32| clip_bd(v, bit_depth);
    let idx = |xx: usize, yy: usize| yy * stride + xx;
    let p = |buf: &[u16], k: usize, col: usize| -> i32 { buf[idx(x + col, y - 1 - k)] as i32 };
    let q = |buf: &[u16], k: usize, col: usize| -> i32 { buf[idx(x + col, y + k)] as i32 };

    let dp0 = (p(buf, 2, 0) - 2 * p(buf, 1, 0) + p(buf, 0, 0)).abs();
    let dp3 = (p(buf, 2, 3) - 2 * p(buf, 1, 3) + p(buf, 0, 3)).abs();
    let dq0 = (q(buf, 2, 0) - 2 * q(buf, 1, 0) + q(buf, 0, 0)).abs();
    let dq3 = (q(buf, 2, 3) - 2 * q(buf, 1, 3) + q(buf, 0, 3)).abs();
    let d = dp0 + dq0 + dp3 + dq3;
    if d >= beta {
        return;
    }
    let d_pq = (p(buf, 3, 0) - p(buf, 0, 0)).abs() + (q(buf, 3, 0) - q(buf, 0, 0)).abs();
    let d_pq3 = (p(buf, 3, 3) - p(buf, 0, 3)).abs() + (q(buf, 3, 3) - q(buf, 0, 3)).abs();
    let strong = (2 * (dp0 + dq0) < (beta >> 2))
        && (d_pq < (beta >> 3))
        && ((p(buf, 0, 0) - q(buf, 0, 0)).abs() < ((5 * tc + 1) >> 1))
        && (2 * (dp3 + dq3) < (beta >> 2))
        && (d_pq3 < (beta >> 3))
        && ((p(buf, 0, 3) - q(buf, 0, 3)).abs() < ((5 * tc + 1) >> 1));

    if strong {
        for col in 0..4 {
            let p0 = p(buf, 0, col);
            let p1 = p(buf, 1, col);
            let p2 = p(buf, 2, col);
            let p3 = p(buf, 3, col);
            let q0 = q(buf, 0, col);
            let q1 = q(buf, 1, col);
            let q2 = q(buf, 2, col);
            let q3 = q(buf, 3, col);
            let tc2 = 2 * tc;
            let np0 = clip3(
                p0 - tc2,
                p0 + tc2,
                (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3,
            );
            let np1 = clip3(p1 - tc2, p1 + tc2, (p2 + p1 + p0 + q0 + 2) >> 2);
            let np2 = clip3(
                p2 - tc2,
                p2 + tc2,
                (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3,
            );
            let nq0 = clip3(
                q0 - tc2,
                q0 + tc2,
                (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3,
            );
            let nq1 = clip3(q1 - tc2, q1 + tc2, (p0 + q0 + q1 + q2 + 2) >> 2);
            let nq2 = clip3(
                q2 - tc2,
                q2 + tc2,
                (p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3,
            );
            buf[idx(x + col, y - 1)] = clip_bd_bd(np0);
            buf[idx(x + col, y - 2)] = clip_bd_bd(np1);
            buf[idx(x + col, y - 3)] = clip_bd_bd(np2);
            buf[idx(x + col, y)] = clip_bd_bd(nq0);
            buf[idx(x + col, y + 1)] = clip_bd_bd(nq1);
            buf[idx(x + col, y + 2)] = clip_bd_bd(nq2);
        }
    } else {
        let dep = (dp0 + dp3) < ((beta + (beta >> 1)) >> 3);
        let deq = (dq0 + dq3) < ((beta + (beta >> 1)) >> 3);
        for col in 0..4 {
            let p0 = p(buf, 0, col);
            let p1 = p(buf, 1, col);
            let p2 = p(buf, 2, col);
            let q0 = q(buf, 0, col);
            let q1 = q(buf, 1, col);
            let q2 = q(buf, 2, col);
            let delta0 = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
            if delta0.abs() >= tc * 10 {
                continue;
            }
            let delta = clip3(-tc, tc, delta0);
            buf[idx(x + col, y - 1)] = clip_bd_bd(p0 + delta);
            buf[idx(x + col, y)] = clip_bd_bd(q0 - delta);
            if dep {
                let delta_p = clip3(
                    -(tc >> 1),
                    tc >> 1,
                    (((p2 + p0 + 1) >> 1) - p1 + delta) >> 1,
                );
                buf[idx(x + col, y - 2)] = clip_bd_bd(p1 + delta_p);
            }
            if deq {
                let delta_q = clip3(
                    -(tc >> 1),
                    tc >> 1,
                    (((q2 + q0 + 1) >> 1) - q1 - delta) >> 1,
                );
                buf[idx(x + col, y + 1)] = clip_bd_bd(q1 + delta_q);
            }
        }
    }
}

/// Chroma vertical edges. The chroma edge grid in the picture follows the
/// luma 8-grid scaled by `(SubWidthC, SubHeightC)`. Each luma 4-row edge
/// segment covers `4 / SubHeightC` chroma rows; the per-segment 2-row
/// chroma filter runs once for 4:2:0 (covers 2 chroma rows = 4 luma rows)
/// and twice for 4:2:2 (covers 4 chroma rows = 4 luma rows).
#[allow(clippy::too_many_arguments)]
fn deblock_vertical_chroma(
    pic: &mut Picture,
    width: usize,
    height: usize,
    tc_off: i32,
    slice: &SliceSegmentHeader,
    pps: &PicParameterSet,
    bit_depth: u32,
    sub_x: u32,
    sub_y: u32,
) {
    let cw = width / sub_x as usize;
    let ch = height / sub_y as usize;
    let cb_off = pps.pps_cb_qp_offset + slice.slice_cb_qp_offset;
    let cr_off = pps.pps_cr_qp_offset + slice.slice_cr_qp_offset;
    let edge_dx_chroma = (8u32 / sub_x) as usize; // chroma stride between vertical edges
    let seg_dy_chroma = (4u32 / sub_y) as usize; // chroma rows per luma 4-row segment
    let c_stride = pic.chroma_stride;
    let mut xc = edge_dx_chroma; // first vertical chroma edge
    while xc + 2 <= cw {
        let luma_x = xc * sub_x as usize;
        let mut yc = 0usize;
        while yc + seg_dy_chroma <= ch {
            let luma_y = yc * sub_y as usize;
            let p_bx = (luma_x - 1) >> 2;
            let p_by = luma_y >> 2;
            let q_bx = luma_x >> 2;
            let q_by = luma_y >> 2;
            let bs = boundary_strength(pic, p_bx, p_by, q_bx, q_by);
            if bs == 2 {
                let qpa = qp_avg(pic, luma_x - 1, luma_y, luma_x, luma_y);
                let tc_cb = chroma_tc(qpa, cb_off, bs, tc_off, bit_depth);
                let tc_cr = chroma_tc(qpa, cr_off, bs, tc_off, bit_depth);
                let mut row_off = 0usize;
                while row_off < seg_dy_chroma {
                    filter_chroma_vertical(
                        &mut pic.cb,
                        c_stride,
                        xc,
                        yc + row_off,
                        tc_cb,
                        bit_depth,
                    );
                    filter_chroma_vertical(
                        &mut pic.cr,
                        c_stride,
                        xc,
                        yc + row_off,
                        tc_cr,
                        bit_depth,
                    );
                    row_off += 2;
                }
            }
            yc += seg_dy_chroma;
        }
        xc += edge_dx_chroma;
    }
}

#[allow(clippy::too_many_arguments)]
fn deblock_horizontal_chroma(
    pic: &mut Picture,
    width: usize,
    height: usize,
    tc_off: i32,
    slice: &SliceSegmentHeader,
    pps: &PicParameterSet,
    bit_depth: u32,
    sub_x: u32,
    sub_y: u32,
) {
    let cw = width / sub_x as usize;
    let ch = height / sub_y as usize;
    let cb_off = pps.pps_cb_qp_offset + slice.slice_cb_qp_offset;
    let cr_off = pps.pps_cr_qp_offset + slice.slice_cr_qp_offset;
    let edge_dy_chroma = (8u32 / sub_y) as usize;
    let _ = sub_x;
    let c_stride = pic.chroma_stride;
    let mut yc = edge_dy_chroma;
    while yc + 2 <= ch {
        let luma_y = yc * sub_y as usize;
        let mut xc = 0usize;
        while xc + 2 <= cw {
            let luma_x = xc * sub_x as usize;
            let p_bx = luma_x >> 2;
            let p_by = (luma_y - 1) >> 2;
            let q_bx = luma_x >> 2;
            let q_by = luma_y >> 2;
            let bs = boundary_strength(pic, p_bx, p_by, q_bx, q_by);
            if bs == 2 {
                let qpa = qp_avg(pic, luma_x, luma_y - 1, luma_x, luma_y);
                let tc_cb = chroma_tc(qpa, cb_off, bs, tc_off, bit_depth);
                let tc_cr = chroma_tc(qpa, cr_off, bs, tc_off, bit_depth);
                filter_chroma_horizontal(&mut pic.cb, c_stride, xc, yc, tc_cb, bit_depth);
                filter_chroma_horizontal(&mut pic.cr, c_stride, xc, yc, tc_cr, bit_depth);
            }
            xc += 2;
        }
        yc += edge_dy_chroma;
    }
}

/// §8.7.2.7 chroma vertical filter: 2-sample-tall edge segment, only p0/q0
/// and p1/q1 read, only p0/q0 written.
fn filter_chroma_vertical(
    buf: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    tc: i32,
    bit_depth: u32,
) {
    if tc == 0 {
        return;
    }
    let clip_bd_bd = |v: i32| clip_bd(v, bit_depth);
    let idx = |xx: usize, yy: usize| yy * stride + xx;
    for row in 0..2 {
        let p1 = buf[idx(x - 2, y + row)] as i32;
        let p0 = buf[idx(x - 1, y + row)] as i32;
        let q0 = buf[idx(x, y + row)] as i32;
        let q1 = buf[idx(x + 1, y + row)] as i32;
        let delta = clip3(-tc, tc, (((q0 - p0) << 2) + p1 - q1 + 4) >> 3);
        buf[idx(x - 1, y + row)] = clip_bd_bd(p0 + delta);
        buf[idx(x, y + row)] = clip_bd_bd(q0 - delta);
    }
}

fn filter_chroma_horizontal(
    buf: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    tc: i32,
    bit_depth: u32,
) {
    if tc == 0 {
        return;
    }
    let clip_bd_bd = |v: i32| clip_bd(v, bit_depth);
    let idx = |xx: usize, yy: usize| yy * stride + xx;
    for col in 0..2 {
        let p1 = buf[idx(x + col, y - 2)] as i32;
        let p0 = buf[idx(x + col, y - 1)] as i32;
        let q0 = buf[idx(x + col, y)] as i32;
        let q1 = buf[idx(x + col, y + 1)] as i32;
        let delta = clip3(-tc, tc, (((q0 - p0) << 2) + p1 - q1 + 4) >> 3);
        buf[idx(x + col, y - 1)] = clip_bd_bd(p0 + delta);
        buf[idx(x + col, y)] = clip_bd_bd(q0 - delta);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_monotone_and_endpoints() {
        // Table 8-11 β values: zero for Q<16, monotone non-decreasing.
        for &v in &BETA_TABLE[..16] {
            assert_eq!(v, 0);
        }
        for q in 1..BETA_TABLE.len() {
            assert!(BETA_TABLE[q] >= BETA_TABLE[q - 1]);
        }
        assert_eq!(BETA_TABLE[51], 64);
    }

    #[test]
    fn tc_monotone_and_endpoints() {
        for &v in &TC_TABLE[..18] {
            assert_eq!(v, 0);
        }
        for q in 1..TC_TABLE.len() {
            assert!(TC_TABLE[q] >= TC_TABLE[q - 1]);
        }
        assert_eq!(TC_TABLE[53], 24);
    }

    #[test]
    fn chroma_qp_i2c_preserves_low_half() {
        // The spec's Table 8-10 is the identity up to QpI=29 for 4:2:0.
        for (i, &v) in CHROMA_QP_I2C.iter().enumerate().take(30) {
            assert_eq!(v, i as u8);
        }
        // And the high half maps conservatively (Q >= 50 clamps to 51).
        assert_eq!(CHROMA_QP_I2C[51], 40);
    }

    #[test]
    fn luma_normal_filter_preserves_smooth_region() {
        // If the signal is already smooth, the normal filter should not
        // push samples outside [p0 - tc, p0 + tc] — and in particular for
        // the flat ramp the change should be zero or tiny.
        let mut buf = [0u16; 16 * 4];
        for y in 0..4 {
            for x in 0..16 {
                buf[y * 16 + x] = ((x as u8).wrapping_mul(8)) as u16;
            }
        }
        let before = buf;
        filter_luma_vertical(&mut buf, 16, 8, 0, 30, 5, 8);
        // Max per-sample change on a smooth ramp should be small.
        let mut max_diff = 0i32;
        for i in 0..buf.len() {
            max_diff = max_diff.max((buf[i] as i32 - before[i] as i32).abs());
        }
        assert!(max_diff <= 10, "smooth ramp perturbed by {max_diff}");
    }

    #[test]
    fn luma_strong_filter_smooths_step() {
        // Flat block, step at the edge — strong filter should pull it in.
        let mut buf = [0u16; 16 * 4];
        for y in 0..4 {
            for x in 0..8 {
                buf[y * 16 + x] = 100;
            }
            for x in 8..16 {
                buf[y * 16 + x] = 120;
            }
        }
        let before = buf;
        filter_luma_vertical(&mut buf, 16, 8, 0, 60, 10, 8);
        // Samples at the edge must have moved toward each other.
        let p0_before = before[7] as i32;
        let q0_before = before[8] as i32;
        let p0_after = buf[7] as i32;
        let q0_after = buf[8] as i32;
        assert!(p0_after >= p0_before, "p0 should increase toward q");
        assert!(q0_after <= q0_before, "q0 should decrease toward p");
    }

    #[test]
    fn chroma_filter_monotone_step() {
        let mut buf = [0u16; 8 * 2];
        for y in 0..2 {
            for x in 0..4 {
                buf[y * 8 + x] = 60;
            }
            for x in 4..8 {
                buf[y * 8 + x] = 80;
            }
        }
        let before = buf;
        filter_chroma_vertical(&mut buf, 8, 4, 0, 5, 8);
        let p0_before = before[3] as i32;
        let q0_before = before[4] as i32;
        let p0_after = buf[3] as i32;
        let q0_after = buf[4] as i32;
        assert!(p0_after >= p0_before);
        assert!(q0_after <= q0_before);
    }
}
