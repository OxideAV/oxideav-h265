//! Speed-first pass-1 decisions ([`TreeCfg::speed`] ≥ 1): the same
//! coded-tree data model, emission, loop filters and reconstruction
//! contract as the rate-distortion search of the parent module, with
//! the per-CU candidate ladder decided by SATD + λ·bins and only the
//! winning candidate of each coding-unit size transform-coded:
//!
//! * coding units of 32 / 16 / 8 luma samples (a 64 CTB always
//!   splits), one transform unit per CU (`max_transform_hierarchy_
//!   depth_* == 0`), `PART_2Nx2N` only;
//! * intra: planar, DC and a coarse-to-fine angular search (step 4,
//!   then ±2, ±1, plus the most-probable modes) on the CU's real
//!   reference samples, chroma derived from luma;
//! * inter: the merge list priced by SATD, AMVP motion search per
//!   list and reference (predictor / merge / zero seeds, a shrinking
//!   square pattern at full-sample precision, then half- and
//!   quarter-sample refinement through the exact §8.5.3.3.3
//!   interpolation), the bi-predictive combination of the two list
//!   winners on two-sided B slices, and an intra candidate;
//! * the unsplit CU and its four children compete under SSE +
//!   λ·bins once both are coded; a skip whose prediction already
//!   meets the λ-scaled distortion bound stops the descent.
//!
//! The kernels ([`super::fast_dsp`]) reproduce the decode path's
//! prediction and reconstruction exactly, so the committed state is a
//! conforming decoder's.

use super::fast_dsp::{self as dsp, HP_PAD, HalfPel, MAXB, Refs};
use super::*;
use crate::binarization::derive_intra_pred_mode_c;
use crate::encoder::inter::{AmvpGroup, amvp_pu, amvp_pu_list, mvd_bits};
use crate::motion::Mv;

/// The coarse angular step of the intra search.
const ANGULAR_COARSE: [u8; 9] = [2, 6, 10, 14, 18, 22, 26, 30, 34];

/// Integer-sample motion search: the square pattern's step ladder.
const ME_STEPS: [i32; 3] = [4, 2, 1];

/// Iteration cap per step of the square pattern.
const ME_ITERS: usize = 8;

/// The λ pair of one CU decision: SSE-domain (mode / split decisions
/// on coded candidates) and its square root (SAD / SATD-domain
/// candidate pruning and motion search).
#[derive(Clone, Copy)]
struct Lambda {
    sse: f64,
    sad: f64,
}

impl Lambda {
    /// `0.57 · 2^((Qp′ − 12) / 3)` on the bit-depth-offset QP, so the
    /// SSE λ follows the squared sample scale at any depth.
    fn new(ctx: &SliceCtx<'_>, qp: i32) -> Self {
        let q = f64::from(qp + ctx.fmt.qp_bd_offset_y());
        let sse = 0.57 * (2f64).powf((q - 12.0) / 3.0);
        Self {
            sse,
            sad: sse.sqrt(),
        }
    }
}

/// One coded leaf candidate: the CU's syntax, its reconstruction
/// (luma `n`x`n`, chroma `n/2`x`n/2`, `i32` samples) and its cost.
struct Leaf {
    kind: TreeCuKind,
    motions: Vec<PuMotion>,
    tree: Option<TuNode>,
    y: Box<[i32; MAXB * MAXB]>,
    cb: Box<[i32; MAXB * MAXB / 4]>,
    cr: Box<[i32; MAXB * MAXB / 4]>,
    cost: f64,
}

/// §6.4.1 z-scan availability of luma `(xn, yn)` for the block at
/// `(xc, yc)` in a single-slice, single-tile picture: inside the
/// picture and earlier in coding order (a previous CTB row, a
/// previous CTB of the row, or an earlier minimum block in z-order of
/// the same CTB).
fn z_avail(ctx: &SliceCtx<'_>, xc: usize, yc: usize, xn: i64, yn: i64) -> bool {
    if xn < 0 || yn < 0 || xn >= ctx.width as i64 || yn >= ctx.height as i64 {
        return false;
    }
    let (xn, yn) = (xn as usize, yn as usize);
    let l = ctx.cfg.ctb_log2;
    let (cxc, cyc, cxn, cyn) = (xc >> l, yc >> l, xn >> l, yn >> l);
    if cyn != cyc {
        return cyn < cyc;
    }
    if cxn != cxc {
        return cxn < cxc;
    }
    let m = (1usize << l) - 1;
    morton((xn & m) >> 2, (yn & m) >> 2) < morton((xc & m) >> 2, (yc & m) >> 2)
}

/// Interleave the bits of `(x, y)` (x in the even positions).
fn morton(x: usize, y: usize) -> usize {
    let mut z = 0;
    for b in 0..5 {
        z |= ((x >> b) & 1) << (2 * b);
        z |= ((y >> b) & 1) << (2 * b + 1);
    }
    z
}

/// The §8.4.4.2.1 reference samples of an `n` luma TB at `(x, y)`
/// (substituted).
fn luma_refs(ctx: &SliceCtx<'_>, st: &EncState, x: usize, y: usize, n: usize) -> Refs {
    let w = ctx.width;
    let rec = &st.recon;
    let get = |xn: i64, yn: i64| -> (i32, bool) {
        if z_avail(ctx, x, y, xn, yn) {
            (
                i32::from(rec.y[rec.ly(yn as usize) * w + xn as usize]),
                true,
            )
        } else {
            (0, false)
        }
    };
    let mut left = [(0, false); 2 * MAXB + 1];
    let mut top = [(0, false); 2 * MAXB + 1];
    left[0] = get(x as i64 - 1, y as i64 - 1);
    top[0] = left[0];
    for k in 0..2 * n {
        left[1 + k] = get(x as i64 - 1, (y + k) as i64);
        top[1 + k] = get((x + k) as i64, y as i64 - 1);
    }
    Refs::substituted(n, &left, &top, ctx.fmt.bit_depth_luma)
}

/// The 4:2:0 chroma twin of [`luma_refs`] for an `n` chroma TB at
/// chroma `(cx, cy)` of plane `plane` (1 = Cb, 2 = Cr).
fn chroma_refs(
    ctx: &SliceCtx<'_>,
    st: &EncState,
    plane: usize,
    cx: usize,
    cy: usize,
    n: usize,
) -> Refs {
    let cw = ctx.cw();
    let rec = &st.recon;
    let p: &[u16] = if plane == 1 { &rec.cb } else { &rec.cr };
    let get = |xn: i64, yn: i64| -> (i32, bool) {
        if z_avail(ctx, cx * 2, cy * 2, xn * 2, yn * 2) {
            (i32::from(p[rec.cy(yn as usize) * cw + xn as usize]), true)
        } else {
            (0, false)
        }
    };
    let mut left = [(0, false); 2 * MAXB + 1];
    let mut top = [(0, false); 2 * MAXB + 1];
    left[0] = get(cx as i64 - 1, cy as i64 - 1);
    top[0] = left[0];
    for k in 0..2 * n {
        left[1 + k] = get(cx as i64 - 1, (cy + k) as i64);
        top[1 + k] = get((cx + k) as i64, cy as i64 - 1);
    }
    Refs::substituted(n, &left, &top, ctx.fmt.bit_depth_chroma)
}

/// The intra luma mode signalling bins (MPM index or the 5-bit
/// remainder).
fn mode_bins(mode: u8, mpm: &[u8; 3]) -> u32 {
    match mpm.iter().position(|&m| m == mode) {
        Some(0) => 2,
        Some(_) => 3,
        None => 6,
    }
}

/// The intra luma mode decision of an `n` CU at `(x0, y0)` by SATD +
/// λ·mode bins. Returns `(mode, cost)`.
fn intra_mode_search(
    ctx: &SliceCtx<'_>,
    st: &EncState,
    x0: usize,
    y0: usize,
    n: usize,
    lam: Lambda,
) -> (u8, f64) {
    let refs = luma_refs(ctx, st, x0, y0, n);
    let filt = refs.filtered();
    let mpm = mpm_list(ctx, st, x0, y0);
    let src = &ctx.src[0][y0 * ctx.width + x0..];
    let bd = ctx.fmt.bit_depth_luma;
    let mut pred = [0i32; MAXB * MAXB];
    let mut tried = [false; 35];
    let mut eval = |mode: u8, best: &mut (u8, f64)| {
        if tried[mode as usize] {
            return;
        }
        tried[mode as usize] = true;
        let r = if dsp::filter_flag(mode, n) {
            &filt
        } else {
            &refs
        };
        dsp::predict_intra(r, mode, true, bd, &mut pred);
        let cost = f64::from(dsp::satd(src, ctx.width, &pred, n, n))
            + lam.sad * f64::from(mode_bins(mode, &mpm));
        if cost < best.1 {
            *best = (mode, cost);
        }
    };
    let mut best = (0u8, f64::MAX);
    eval(0, &mut best);
    eval(1, &mut best);
    let mut best_ang = (2u8, f64::MAX);
    for &m in &ANGULAR_COARSE {
        eval(m, &mut best_ang);
    }
    for step in [2i32, 1] {
        let c = i32::from(best_ang.0);
        for m in [c - step, c + step] {
            if (2..=34).contains(&m) {
                eval(m as u8, &mut best_ang);
            }
        }
    }
    if best_ang.1 < best.1 {
        best = best_ang;
    }
    for &m in &mpm {
        eval(m, &mut best);
    }
    best
}

/// Transform-code one `n`x`n` block against `pred` (`w`-stride `n`),
/// writing the clipped reconstruction into `rec`; `dst` selects the
/// 4x4 intra-luma DST-VII. Returns the levels (empty when all are
/// zero) and their bin estimate.
#[allow(clippy::too_many_arguments)]
fn code_block(
    src: &[u16],
    ss: usize,
    pred: &[i32],
    n: usize,
    qp: u32,
    bit_depth: u8,
    intra: bool,
    dst: bool,
    rec: &mut [i32],
) -> (Vec<i32>, u32) {
    let mut res = [0i32; MAXB * MAXB];
    for j in 0..n {
        for i in 0..n {
            res[j * n + i] = i32::from(src[j * ss + i]) - pred[j * n + i];
        }
    }
    let mut coef = [0i32; MAXB * MAXB];
    dsp::forward(&res, n, bit_depth, dst, &mut coef);
    let mut levels = vec![0i32; n * n];
    let nz = dsp::quantize(&coef, n, qp, bit_depth, intra, &mut levels);
    let max = (1i32 << bit_depth) - 1;
    if nz == 0 {
        rec[..n * n].copy_from_slice(&pred[..n * n]);
        return (Vec::new(), 1);
    }
    let bits = dsp::levels_bits(&levels, n);
    dsp::dequant_inverse(&levels, n, qp, bit_depth, dst, &mut res);
    for k in 0..n * n {
        rec[k] = (pred[k] + res[k]).clamp(0, max);
    }
    (levels, bits)
}

/// A level vector as the tree stores it (all-zero blocks keep their
/// full-size zero vector so the emitter sees the right block size).
fn levels_or_zero(levels: Vec<i32>, n: usize) -> Vec<i32> {
    if levels.is_empty() {
        vec![0; n * n]
    } else {
        levels
    }
}

/// Code an intra 2Nx2N CU with luma mode `mode` (TU = CU, chroma
/// derived), reading reference samples from the committed state.
#[allow(clippy::too_many_arguments)]
fn code_intra_leaf(
    ctx: &SliceCtx<'_>,
    st: &EncState,
    x0: usize,
    y0: usize,
    n: usize,
    mode: u8,
    qp: i32,
    lam: Lambda,
) -> Leaf {
    let (qp_y, qp_c) = (ctx.qp_y_prime(qp), ctx.qp_c_prime(qp));
    let mut pred = [0i32; MAXB * MAXB];
    let refs = luma_refs(ctx, st, x0, y0, n);
    let r = if dsp::filter_flag(mode, n) {
        refs.filtered()
    } else {
        refs
    };
    dsp::predict_intra(&r, mode, true, ctx.fmt.bit_depth_luma, &mut pred);
    let mut y = Box::new([0i32; MAXB * MAXB]);
    let src = &ctx.src[0][y0 * ctx.width + x0..];
    let (ly, by) = code_block(
        src,
        ctx.width,
        &pred,
        n,
        qp_y,
        ctx.fmt.bit_depth_luma,
        true,
        n == 4,
        &mut y[..],
    );
    let mut dist = dsp::ssd_block(src, ctx.width, &y[..], n, n);
    let mode_c = derive_intra_pred_mode_c(4, mode, false);
    let nc = n / 2;
    let (cx, cy) = (x0 / 2, y0 / 2);
    let cw = ctx.cw();
    let mut cb = Box::new([0i32; MAXB * MAXB / 4]);
    let mut cr = Box::new([0i32; MAXB * MAXB / 4]);
    let mut bits = by;
    let mut chroma = |plane: usize, out: &mut [i32]| -> Vec<i32> {
        let r = chroma_refs(ctx, st, plane, cx, cy, nc);
        dsp::predict_intra(&r, mode_c, false, ctx.fmt.bit_depth_chroma, &mut pred);
        let src = &ctx.src[plane][cy * cw + cx..];
        let (lv, b) = code_block(
            src,
            cw,
            &pred,
            nc,
            qp_c,
            ctx.fmt.bit_depth_chroma,
            true,
            false,
            out,
        );
        dist += dsp::ssd_block(src, cw, out, nc, nc);
        bits += b;
        levels_or_zero(lv, nc)
    };
    let lcb = chroma(1, &mut cb[..]);
    let lcr = chroma(2, &mut cr[..]);
    let mpm = mpm_list(ctx, st, x0, y0);
    bits += mode_bins(mode, &mpm) + 4;
    Leaf {
        kind: TreeCuKind::Intra {
            modes: [mode; 4],
            nxn: false,
            chroma: [4; 4],
        },
        motions: Vec::new(),
        tree: Some(TuNode::Leaf {
            y: levels_or_zero(ly, n),
            cb: lcb,
            cr: lcr,
        }),
        y,
        cb,
        cr,
        cost: dist as f64 + lam.sse * f64::from(bits),
    }
}

/// Per-CU inter search state: the source, the references and the
/// scratch the candidate predictions are built in.
struct InterSearch<'c, 'a> {
    ctx: &'c SliceCtx<'a>,
    x0: usize,
    y0: usize,
    n: usize,
    lam: Lambda,
}

impl InterSearch<'_, '_> {
    /// The 14-bit luma intermediate of one list at quarter-sample
    /// `mv` from `refp`.
    fn mc(&self, refp: &RefPlanes<'_>, mv: Mv, out: &mut [i16]) {
        dsp::mc_luma(
            refp.y,
            refp.width,
            refp.height,
            self.x0,
            self.y0,
            mv,
            self.n,
            self.n,
            self.ctx.fmt.bit_depth_luma,
            out,
        );
    }

    /// The final luma prediction of resolved `motion` into `pred`.
    fn predict_luma(&self, motion: &PuMotion, pred: &mut [i32]) {
        let ctx = self.ctx;
        let len = self.n * self.n;
        let bd = ctx.fmt.bit_depth_luma;
        let mut a = [0i16; MAXB * MAXB];
        if motion.pred_flag_l0 && motion.pred_flag_l1 {
            let mut b = [0i16; MAXB * MAXB];
            self.mc(
                &ctx.refs_l0[motion.ref_idx_l0 as usize],
                motion.mv_l0,
                &mut a,
            );
            self.mc(
                &ctx.refs_l1[motion.ref_idx_l1 as usize],
                motion.mv_l1,
                &mut b,
            );
            dsp::weight_bi(&a, &b, len, bd, pred);
        } else if motion.pred_flag_l0 {
            self.mc(
                &ctx.refs_l0[motion.ref_idx_l0 as usize],
                motion.mv_l0,
                &mut a,
            );
            dsp::weight_uni(&a, len, bd, pred);
        } else {
            self.mc(
                &ctx.refs_l1[motion.ref_idx_l1 as usize],
                motion.mv_l1,
                &mut a,
            );
            dsp::weight_uni(&a, len, bd, pred);
        }
    }

    /// The final chroma predictions of resolved `motion`.
    fn predict_chroma(&self, motion: &PuMotion, pcb: &mut [i32], pcr: &mut [i32]) {
        let ctx = self.ctx;
        let nc = self.n / 2;
        let len = nc * nc;
        let bd = ctx.fmt.bit_depth_chroma;
        let (cw, ch) = (ctx.cw(), ctx.ch());
        let (cx, cy) = (self.x0 / 2, self.y0 / 2);
        for (plane, out) in [(1usize, pcb), (2usize, pcr)] {
            let mut a = [0i16; MAXB * MAXB / 4];
            let mut b = [0i16; MAXB * MAXB / 4];
            let run = |r: &RefPlanes<'_>, mv: Mv, buf: &mut [i16]| {
                let p = if plane == 1 { r.cb } else { r.cr };
                dsp::mc_chroma(p, cw, ch, cx, cy, mv, nc, nc, bd, buf);
            };
            if motion.pred_flag_l0 && motion.pred_flag_l1 {
                run(
                    &ctx.refs_l0[motion.ref_idx_l0 as usize],
                    motion.mv_l0,
                    &mut a,
                );
                run(
                    &ctx.refs_l1[motion.ref_idx_l1 as usize],
                    motion.mv_l1,
                    &mut b,
                );
                dsp::weight_bi(&a, &b, len, bd, out);
            } else if motion.pred_flag_l0 {
                run(
                    &ctx.refs_l0[motion.ref_idx_l0 as usize],
                    motion.mv_l0,
                    &mut a,
                );
                dsp::weight_uni(&a, len, bd, out);
            } else {
                run(
                    &ctx.refs_l1[motion.ref_idx_l1 as usize],
                    motion.mv_l1,
                    &mut a,
                );
                dsp::weight_uni(&a, len, bd, out);
            }
        }
    }

    /// SATD of the source block against a final prediction.
    fn satd(&self, pred: &[i32]) -> u32 {
        let ctx = self.ctx;
        dsp::satd(
            &ctx.src[0][self.y0 * ctx.width + self.x0..],
            ctx.width,
            pred,
            self.n,
            self.n,
        )
    }

    /// The reference's half-sample planes (built on first use).
    fn halfpel<'r>(&self, refp: &RefPlanes<'r>) -> &'r HalfPel {
        let cell = refp.halfpel.expect("a reconstruction's reference planes");
        cell.get_or_init(|| {
            std::sync::Arc::new(HalfPel::build(
                refp.y,
                refp.width,
                refp.height,
                self.ctx.fmt.bit_depth_luma,
                self.ctx.threads,
            ))
        })
    }

    /// Whether a block displaced by quarter-sample `mv` stays inside
    /// the half-sample planes' padding.
    fn in_pad(&self, mv: Mv) -> bool {
        let m = (HP_PAD - 8) as i32;
        let n = self.n as i32;
        let (x, y) = (self.x0 as i32 + (mv[0] >> 2), self.y0 as i32 + (mv[1] >> 2));
        x >= -m
            && y >= -m
            && x + n < self.ctx.width as i32 + m
            && y + n < self.ctx.height as i32 + m
    }

    /// The search-grade luma prediction of resolved `motion` (the
    /// exact one when a vector leaves the padded planes).
    fn approx_luma(&self, motion: &PuMotion, pred: &mut [i32]) {
        let ctx = self.ctx;
        let used = [
            (
                motion.pred_flag_l0,
                motion.mv_l0,
                ctx.refs_l0,
                motion.ref_idx_l0,
            ),
            (
                motion.pred_flag_l1,
                motion.mv_l1,
                ctx.refs_l1,
                motion.ref_idx_l1,
            ),
        ];
        if used.iter().any(|&(f, mv, _, _)| f && !self.in_pad(mv)) {
            self.predict_luma(motion, pred);
            return;
        }
        let len = self.n * self.n;
        let mut first = true;
        let mut tmp = [0i32; MAXB * MAXB];
        for (f, mv, refs, idx) in used {
            if !f {
                continue;
            }
            let hp = self.halfpel(&refs[idx as usize]);
            if first {
                dsp::approx_pred(hp, self.x0, self.y0, mv, self.n, self.n, pred);
                first = false;
            } else {
                dsp::approx_pred(hp, self.x0, self.y0, mv, self.n, self.n, &mut tmp);
                for (p, &t) in pred[..len].iter_mut().zip(&tmp[..len]) {
                    *p = (*p + t + 1) >> 1;
                }
            }
        }
    }

    /// Motion search on one reference: seeds, the integer square
    /// pattern on SAD, then half- and quarter-sample refinement on
    /// SATD of the search-grade prediction. Returns the quarter-sample
    /// MV and its SATD + λ·bins cost.
    fn search(&self, refp: &RefPlanes<'_>, mvp: &[Mv; 2], seeds: &[Mv]) -> (Mv, f64) {
        let (x0, y0, n) = (self.x0, self.y0, self.n);
        let lam = self.lam;
        let hp = self.halfpel(refp);
        let src = &self.ctx.src[0][y0 * self.ctx.width + x0..];
        let bits = |mv: Mv| -> u64 {
            mvd_bits([mv[0] - mvp[0][0], mv[1] - mvp[0][1]])
                .min(mvd_bits([mv[0] - mvp[1][0], mv[1] - mvp[1][1]]))
        };
        let int_cost = |m: [i32; 2]| -> f64 {
            if !self.in_pad([m[0] * 4, m[1] * 4]) {
                return f64::MAX;
            }
            let (plane, i) = hp.at(0, 0, x0 as i32 + m[0], y0 as i32 + m[1]);
            f64::from(dsp::sad_plane(
                src,
                self.ctx.width,
                &plane[i..],
                hp.stride(),
                n,
                n,
            )) + lam.sad * bits([m[0] * 4, m[1] * 4]) as f64
        };
        let mut best = [0i32, 0];
        let mut best_cost = f64::MAX;
        for s in seeds.iter().chain(mvp.iter()) {
            let m = [(s[0] + 2) >> 2, (s[1] + 2) >> 2];
            let c = int_cost(m);
            if c < best_cost {
                best_cost = c;
                best = m;
            }
        }
        for step in ME_STEPS {
            for _ in 0..ME_ITERS {
                let centre = best;
                for (dx, dy) in [
                    (step, 0),
                    (-step, 0),
                    (0, step),
                    (0, -step),
                    (step, step),
                    (step, -step),
                    (-step, step),
                    (-step, -step),
                ] {
                    let m = [centre[0] + dx, centre[1] + dy];
                    let c = int_cost(m);
                    if c < best_cost {
                        best_cost = c;
                        best = m;
                    }
                }
                if best == centre {
                    break;
                }
            }
        }
        let mut pred = [0i32; MAXB * MAXB];
        let mut mv = [best[0] * 4, best[1] * 4];
        let mut cost = f64::MAX;
        let mut eval = |m: Mv, cost: &mut f64, mv: &mut Mv| {
            if !self.in_pad(m) {
                return;
            }
            dsp::approx_pred(hp, x0, y0, m, n, n, &mut pred);
            let c = f64::from(self.satd(&pred)) + lam.sad * bits(m) as f64;
            if c < *cost {
                *cost = c;
                *mv = m;
            }
        };
        eval(mv, &mut cost, &mut mv);
        for step in [2i32, 1] {
            let centre = mv;
            for (dx, dy) in [
                (step, 0),
                (-step, 0),
                (0, step),
                (0, -step),
                (step, step),
                (step, -step),
                (-step, step),
                (-step, -step),
            ] {
                eval([centre[0] + dx, centre[1] + dy], &mut cost, &mut mv);
            }
        }
        (mv, cost)
    }
}

/// The best inter candidate of a CU before transform coding.
struct InterChoice {
    kind: TreeCuKind,
    motion: PuMotion,
    satd_cost: f64,
}

/// Decide and code the best leaf of an `n` CU at `(x0, y0)` on a P /
/// B slice.
#[allow(clippy::too_many_lines)]
fn inter_leaf(
    ctx: &SliceCtx<'_>,
    st: &EncState,
    x0: usize,
    y0: usize,
    n: usize,
    qp: i32,
    lam: Lambda,
) -> Leaf {
    let mv_ctx = ctx.mv_ctx.expect("inter slice has motion context");
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
    let search = InterSearch {
        ctx,
        x0,
        y0,
        n,
        lam,
    };
    let mut pred = [0i32; MAXB * MAXB];

    // ---- merge candidates ----
    let mut merges: Vec<(usize, PuMotion)> = Vec::with_capacity(MAX_MERGE);
    for idx in 0..MAX_MERGE {
        let m = resolve_pu_motion(&st.field, &geom, &merge_pu(idx), mv_ctx, &available);
        if !merges.iter().any(|(_, p)| *p == m) {
            merges.push((idx, m));
        }
    }
    let mut best = InterChoice {
        kind: TreeCuKind::Skip { merge_idx: 0 },
        motion: merges[0].1,
        satd_cost: f64::MAX,
    };
    for &(idx, m) in &merges {
        search.approx_luma(&m, &mut pred);
        let c = f64::from(search.satd(&pred)) + lam.sad * (idx as f64 + 1.0);
        if c < best.satd_cost {
            best = InterChoice {
                kind: TreeCuKind::Merge { merge_idx: idx },
                motion: m,
                satd_cost: c,
            };
        }
    }

    // ---- AMVP: per list and reference ----
    let lists: &[usize] = if ctx.two_sided { &[0, 1] } else { &[0] };
    let mut list_best: [Option<(AmvpGroup, PuMotion, f64, u64)>; 2] = [None, None];
    for &list in lists {
        let refs = if list == 0 { ctx.refs_l0 } else { ctx.refs_l1 };
        let mv_of = |m: &PuMotion| if list == 0 { m.mv_l0 } else { m.mv_l1 };
        let used = |m: &PuMotion| {
            if list == 0 {
                m.pred_flag_l0
            } else {
                m.pred_flag_l1
            }
        };
        for (r, refp) in refs.iter().enumerate() {
            let probe = |flag: u8| {
                mv_of(&resolve_pu_motion(
                    &st.field,
                    &geom,
                    &amvp_pu_list(list, r as u8, [0, 0], flag),
                    mv_ctx,
                    &available,
                ))
            };
            let mvp = [probe(0), probe(1)];
            let seeds: Vec<Mv> = merges
                .iter()
                .filter(|(_, m)| {
                    used(m)
                        && (if list == 0 {
                            m.ref_idx_l0
                        } else {
                            m.ref_idx_l1
                        }) == r as i32
                })
                .map(|(_, m)| mv_of(m))
                .chain(std::iter::once([0, 0]))
                .collect();
            let (mv, cost) = search.search(refp, &mvp, &seeds);
            let (mvp_flag, mvd) = (0u8..2)
                .map(|f| (f, [mv[0] - mvp[f as usize][0], mv[1] - mvp[f as usize][1]]))
                .min_by_key(|&(_, d)| mvd_bits(d))
                .expect("two predictors");
            let group = AmvpGroup {
                ref_idx: r as u8,
                mvd,
                mvp_flag,
            };
            let motion = resolve_pu_motion(
                &st.field,
                &geom,
                &amvp_pu_list(list, r as u8, mvd, mvp_flag),
                mv_ctx,
                &available,
            );
            let rate = 2 + mvd_bits(mvd) + u64::from(refs.len() > 1) + u64::from(ctx.b_slice) * 2;
            let total = cost + lam.sad * 2.0;
            if list_best[list].as_ref().is_none_or(|b| total < b.2) {
                list_best[list] = Some((group, motion, total, rate));
            }
        }
    }
    for (list, entry) in list_best.iter().enumerate() {
        if let Some((g, motion, cost, _)) = entry
            && *cost < best.satd_cost
        {
            best = InterChoice {
                kind: TreeCuKind::Amvp {
                    pu: if list == 0 {
                        PuSyntax::Amvp {
                            l0: Some(*g),
                            l1: None,
                        }
                    } else {
                        PuSyntax::Amvp {
                            l0: None,
                            l1: Some(*g),
                        }
                    },
                },
                motion: *motion,
                satd_cost: *cost,
            };
        }
    }
    if let (Some(b0), Some(b1)) = (&list_best[0], &list_best[1]) {
        let motion = resolve_pu_motion(
            &st.field,
            &geom,
            &amvp_pu(Some(b0.0), Some(b1.0)),
            mv_ctx,
            &available,
        );
        search.approx_luma(&motion, &mut pred);
        let rate = (b0.3 + b1.3) as f64 - 1.0;
        let c = f64::from(search.satd(&pred)) + lam.sad * rate;
        if c < best.satd_cost {
            best = InterChoice {
                kind: TreeCuKind::Amvp {
                    pu: PuSyntax::Amvp {
                        l0: Some(b0.0),
                        l1: Some(b1.0),
                    },
                },
                motion,
                satd_cost: c,
            };
        }
    }

    // ---- intra competitor ----
    let (imode, icost) = intra_mode_search(ctx, st, x0, y0, n, lam);
    if icost + lam.sad * 4.0 < best.satd_cost {
        return code_intra_leaf(ctx, st, x0, y0, n, imode, qp, lam);
    }

    // ---- code the inter winner ----
    let (qp_y, qp_c) = (ctx.qp_y_prime(qp), ctx.qp_c_prime(qp));
    let motion = best.motion;
    search.predict_luma(&motion, &mut pred);
    let nc = n / 2;
    let mut pcb = [0i32; MAXB * MAXB / 4];
    let mut pcr = [0i32; MAXB * MAXB / 4];
    search.predict_chroma(&motion, &mut pcb, &mut pcr);
    let cw = ctx.cw();
    let (cx, cy) = (x0 / 2, y0 / 2);
    let src_y = &ctx.src[0][y0 * ctx.width + x0..];
    let src_cb = &ctx.src[1][cy * cw + cx..];
    let src_cr = &ctx.src[2][cy * cw + cx..];
    let mut y = Box::new([0i32; MAXB * MAXB]);
    let mut cb = Box::new([0i32; MAXB * MAXB / 4]);
    let mut cr = Box::new([0i32; MAXB * MAXB / 4]);
    let (ly, by) = code_block(
        src_y,
        ctx.width,
        &pred,
        n,
        qp_y,
        ctx.fmt.bit_depth_luma,
        false,
        false,
        &mut y[..],
    );
    let (lcb, bcb) = code_block(
        src_cb,
        cw,
        &pcb,
        nc,
        qp_c,
        ctx.fmt.bit_depth_chroma,
        false,
        false,
        &mut cb[..],
    );
    let (lcr, bcr) = code_block(
        src_cr,
        cw,
        &pcr,
        nc,
        qp_c,
        ctx.fmt.bit_depth_chroma,
        false,
        false,
        &mut cr[..],
    );
    let coded = !(ly.is_empty() && lcb.is_empty() && lcr.is_empty());
    let dist = dsp::ssd_block(src_y, ctx.width, &y[..], n, n)
        + dsp::ssd_block(src_cb, cw, &cb[..], nc, nc)
        + dsp::ssd_block(src_cr, cw, &cr[..], nc, nc);
    let res_bits = f64::from(by + bcb + bcr);
    let tree = coded.then(|| TuNode::Leaf {
        y: levels_or_zero(ly, n),
        cb: levels_or_zero(lcb, nc),
        cr: levels_or_zero(lcr, nc),
    });
    let pred_dist = || {
        dsp::ssd_block(src_y, ctx.width, &pred, n, n)
            + dsp::ssd_block(src_cb, cw, &pcb, nc, nc)
            + dsp::ssd_block(src_cr, cw, &pcr, nc, nc)
    };
    let as_leaf = |kind: TreeCuKind,
                   tree: Option<TuNode>,
                   y: Box<[i32; MAXB * MAXB]>,
                   cb: Box<[i32; MAXB * MAXB / 4]>,
                   cr: Box<[i32; MAXB * MAXB / 4]>,
                   cost: f64| Leaf {
        kind,
        motions: vec![motion],
        tree,
        y,
        cb,
        cr,
        cost,
    };
    match best.kind {
        TreeCuKind::Merge { merge_idx } | TreeCuKind::Skip { merge_idx } => {
            let skip_cost = pred_dist() as f64 + lam.sse * (merge_idx as f64 + 1.0);
            let merge_cost = dist as f64 + lam.sse * (res_bits + merge_idx as f64 + 3.0);
            if !coded || skip_cost <= merge_cost {
                let mut py = Box::new([0i32; MAXB * MAXB]);
                py[..n * n].copy_from_slice(&pred[..n * n]);
                let mut pb = Box::new([0i32; MAXB * MAXB / 4]);
                pb[..nc * nc].copy_from_slice(&pcb[..nc * nc]);
                let mut pr = Box::new([0i32; MAXB * MAXB / 4]);
                pr[..nc * nc].copy_from_slice(&pcr[..nc * nc]);
                as_leaf(TreeCuKind::Skip { merge_idx }, None, py, pb, pr, skip_cost)
            } else {
                as_leaf(TreeCuKind::Merge { merge_idx }, tree, y, cb, cr, merge_cost)
            }
        }
        TreeCuKind::Amvp { pu } => {
            let mrate = match pu {
                PuSyntax::Amvp { l0, l1 } => {
                    l0.map_or(0, |g| mvd_bits(g.mvd) + 2) + l1.map_or(0, |g| mvd_bits(g.mvd) + 2)
                }
                PuSyntax::Merge { .. } => 0,
            } as f64;
            let cost = dist as f64 + lam.sse * (res_bits + mrate + 3.0);
            as_leaf(TreeCuKind::Amvp { pu }, tree, y, cb, cr, cost)
        }
        _ => unreachable!("inter winners are merge or AMVP"),
    }
}

/// Write a leaf's reconstruction and decoder-visible state into the
/// picture state (the parent module's commit protocol).
fn commit(
    ctx: &SliceCtx<'_>,
    st: &mut EncState,
    x0: usize,
    y0: usize,
    n: usize,
    depth: u32,
    leaf: &Leaf,
) {
    let w = ctx.width;
    let ly0 = st.recon.ly(y0);
    let maxy = ctx.fmt.max_luma();
    for j in 0..n {
        let row = &mut st.recon.y[(ly0 + j) * w + x0..(ly0 + j) * w + x0 + n];
        for (o, &v) in row.iter_mut().zip(&leaf.y[j * n..j * n + n]) {
            *o = v.clamp(0, maxy) as u16;
        }
    }
    let (nc, cw, cx, cy) = (n / 2, ctx.cw(), x0 / 2, y0 / 2);
    let lcy = st.recon.cy(cy);
    for (plane, src) in [(&mut st.recon.cb, &leaf.cb), (&mut st.recon.cr, &leaf.cr)] {
        for j in 0..nc {
            let row = &mut plane[(lcy + j) * cw + cx..(lcy + j) * cw + cx + nc];
            for (o, &v) in row.iter_mut().zip(&src[j * nc..j * nc + nc]) {
                *o = v as u16;
            }
        }
    }
    let is_skip = matches!(leaf.kind, TreeCuKind::Skip { .. });
    match &leaf.kind {
        TreeCuKind::Intra { modes, .. } => {
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
            st.modes.record_intra_pb(x0, y0, n, modes[0], false);
        }
        _ => {
            let m = &leaf.motions[0];
            let (p0, p1) = cell_pocs(ctx, m);
            st.field.fill_rect(x0, y0, n, n, m.to_cell(p0, p1));
            st.modes.record_non_intra_cu(
                x0,
                y0,
                n,
                if is_skip {
                    CuPredMode::Skip
                } else {
                    CuPredMode::Inter
                },
            );
            if let Some(tree) = &leaf.tree {
                mark_nonzero(&mut st.field, tree, x0, y0, n.trailing_zeros());
            }
        }
    }
    st.fill_cells(x0, y0, n, depth as i8, u8::from(is_skip));
}

/// The coded CU of a committed leaf.
fn to_cu(x0: usize, y0: usize, log2: u32, leaf: Leaf) -> CuCoded {
    CuCoded {
        x0,
        y0,
        log2,
        kind: leaf.kind,
        motions: leaf.motions,
        tree: leaf.tree,
        cost: leaf.cost as u64,
    }
}

/// Decide and code one leaf candidate (no state change).
fn eval_leaf(
    ctx: &SliceCtx<'_>,
    st: &EncState,
    x0: usize,
    y0: usize,
    n: usize,
    qp: i32,
    lam: Lambda,
) -> Leaf {
    if ctx.intra_slice {
        let (mode, _) = intra_mode_search(ctx, st, x0, y0, n, lam);
        code_intra_leaf(ctx, st, x0, y0, n, mode, qp, lam)
    } else {
        inter_leaf(ctx, st, x0, y0, n, qp, lam)
    }
}

/// The fast twin of [`super::code_quadtree`]: decide the node at
/// `(x0, y0)` of size `1 << log2`, committing the winner into `st`.
/// Returns the node and its cost.
pub(super) fn code_quadtree(
    ctx: &SliceCtx<'_>,
    st: &mut EncState,
    x0: usize,
    y0: usize,
    log2: u32,
    depth: u32,
    ctb_qp: i32,
) -> (CuNode, u64) {
    let n = 1usize << log2;
    let lam = Lambda::new(ctx, ctb_qp);
    let fits = x0 + n <= ctx.width && y0 + n <= ctx.height;
    let split = |ctx: &SliceCtx<'_>, st: &mut EncState| -> (CuNode, f64) {
        let half = n / 2;
        let mut children: [Option<CuNode>; 4] = [None, None, None, None];
        let mut cost = 0.0;
        for (k, &(zx, zy)) in Z_OFFSETS.iter().enumerate() {
            let (cx, cy) = (x0 + zx * half, y0 + zy * half);
            if cx < ctx.width && cy < ctx.height {
                let (node, c) = code_quadtree(ctx, st, cx, cy, log2 - 1, depth + 1, ctb_qp);
                children[k] = Some(node);
                cost += c as f64;
            }
        }
        (CuNode::Split(Box::new(children)), cost)
    };
    if !fits || log2 > 5 {
        let (node, cost) = split(ctx, st);
        return (node, cost as u64);
    }
    let leaf = eval_leaf(ctx, st, x0, y0, n, ctb_qp, lam);
    let stop = log2 == ctx.cfg.min_cb_log2()
        || (matches!(leaf.kind, TreeCuKind::Skip { .. })
            && leaf.cost < lam.sse * (n * n) as f64 / 16.0);
    if stop {
        commit(ctx, st, x0, y0, n, depth, &leaf);
        let cost = leaf.cost
            + if log2 > ctx.cfg.min_cb_log2() {
                lam.sse
            } else {
                0.0
            };
        return (
            CuNode::Leaf(Box::new(to_cu(x0, y0, log2, leaf))),
            cost as u64,
        );
    }
    let before = st.snapshot(ctx, x0, y0, n);
    let (node, split_cost) = split(ctx, st);
    if leaf.cost <= split_cost {
        st.restore(ctx, &before);
        commit(ctx, st, x0, y0, n, depth, &leaf);
        let cost = leaf.cost + lam.sse;
        (
            CuNode::Leaf(Box::new(to_cu(x0, y0, log2, leaf))),
            cost as u64,
        )
    } else {
        (node, (split_cost + lam.sse) as u64)
    }
}
