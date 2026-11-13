//! Sample kernels of the speed-first decision path
//! ([`super::TreeCfg::speed`]): block SAD / SATD, the §8.4.4.2 intra
//! sample prediction, the §8.5.3.3.3 fractional sample interpolation
//! with the §8.5.3.3.4.2 default weighted combine, the forward
//! DCT-II / DST-VII with a dead-zone quantizer, and the §8.6.2 scaling
//! plus §8.6.4.2 inverse transform.
//!
//! Every kernel a decoder also runs (prediction, scaling, inverse
//! transform) produces exactly the decode path's samples — the unit
//! tests compare each against [`crate::intra_pred`],
//! [`crate::inter_pred`] and [`crate::transform`] — while working on
//! fixed-size stack buffers instead of per-call allocations.

use crate::transform::{DCT32, DST4, LEVEL_SCALE};

/// Largest block the fast path codes (a 32x32 CU / TU).
pub(super) const MAXB: usize = 32;

/// `intraPredAngle` of modes 2..=34 (Table 8-5), indexed by `mode`.
const ANGLE: [i32; 35] = [
    0, 0, 32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26, -32, -26, -21, -17, -13,
    -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32,
];

/// `invAngle` of modes 11..=25 (Table 8-6), indexed by `mode`.
const INV_ANGLE: [i32; 35] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4096, -1638, -910, -630, -482, -390, -315, -256, -315, -390,
    -482, -630, -910, -1638, -4096, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Luma interpolation filter `fL[ xFrac ][ i ]` (Table 8-11).
const FL: [[i32; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
];

/// Chroma interpolation filter `fC[ xFrac ][ i ]` (Table 8-12).
const FC: [[i32; 4]; 8] = [
    [0, 64, 0, 0],
    [-2, 58, 10, -2],
    [-4, 54, 16, -2],
    [-6, 46, 28, -4],
    [-4, 36, 36, -4],
    [-4, 28, 46, -6],
    [-2, 16, 54, -4],
    [-2, 10, 58, -2],
];

/// Forward quantizer multipliers: `2^20 / LEVEL_SCALE[ k ]` rounded,
/// so that the §8.6.2 scaling inverts them to within rounding.
const QUANT_SCALE: [i64; 6] = [26214, 23302, 20560, 18396, 16384, 14564];

// ---------------------------------------------------------------------
// Distortion kernels
// ---------------------------------------------------------------------

/// SAD of a `w`x`h` source block (plane stride `ss`) against a sample
/// block of a reference plane (stride `rs`).
#[inline]
pub(super) fn sad_plane(src: &[u16], ss: usize, rf: &[u16], rs: usize, w: usize, h: usize) -> u32 {
    let mut acc = 0u32;
    for j in 0..h {
        let a = &src[j * ss..j * ss + w];
        let b = &rf[j * rs..j * rs + w];
        acc += a
            .iter()
            .zip(b)
            .map(|(&x, &y)| u32::from(x.abs_diff(y)))
            .sum::<u32>();
    }
    acc
}

/// Sum of squared differences of a `w`x`h` source block against a
/// `w`-stride block.
#[inline]
pub(super) fn ssd_block(src: &[u16], ss: usize, rec: &[i32], w: usize, h: usize) -> u64 {
    let mut acc = 0u64;
    for j in 0..h {
        let a = &src[j * ss..j * ss + w];
        let b = &rec[j * w..j * w + w];
        acc += a
            .iter()
            .zip(b)
            .map(|(&x, &y)| {
                let d = i64::from(x) - i64::from(y);
                (d * d) as u64
            })
            .sum::<u64>();
    }
    acc
}

/// SATD (4x4 Hadamard) of the 4x4 tile of `src - pred` at `(bx, by)`.
#[inline]
fn hadamard4(src: &[u16], ss: usize, pred: &[i32], ps: usize, bx: usize, by: usize) -> u32 {
    let mut m = [0i32; 16];
    for j in 0..4 {
        let s = &src[(by + j) * ss + bx..(by + j) * ss + bx + 4];
        let p = &pred[(by + j) * ps + bx..(by + j) * ps + bx + 4];
        let r = [
            i32::from(s[0]) - p[0],
            i32::from(s[1]) - p[1],
            i32::from(s[2]) - p[2],
            i32::from(s[3]) - p[3],
        ];
        let (a0, a1) = (r[0] + r[2], r[0] - r[2]);
        let (b0, b1) = (r[1] + r[3], r[1] - r[3]);
        m[j * 4] = a0 + b0;
        m[j * 4 + 1] = a0 - b0;
        m[j * 4 + 2] = a1 + b1;
        m[j * 4 + 3] = a1 - b1;
    }
    let mut acc = 0u32;
    for i in 0..4 {
        let (a0, a1) = (m[i] + m[8 + i], m[i] - m[8 + i]);
        let (b0, b1) = (m[4 + i] + m[12 + i], m[4 + i] - m[12 + i]);
        acc += (a0 + b0).unsigned_abs()
            + (a0 - b0).unsigned_abs()
            + (a1 + b1).unsigned_abs()
            + (a1 - b1).unsigned_abs();
    }
    acc
}

/// One 8-point Walsh-Hadamard transform applied lane-wise to the
/// eight rows of `t` (rows are the transform's inputs).
#[inline(always)]
pub(super) fn butterfly8_rows(t: &mut [[i32; 8]; 8]) {
    let mut u = [[0i32; 8]; 8];
    for l in 0..8 {
        u[0][l] = t[0][l] + t[1][l];
        u[1][l] = t[0][l] - t[1][l];
        u[2][l] = t[2][l] + t[3][l];
        u[3][l] = t[2][l] - t[3][l];
        u[4][l] = t[4][l] + t[5][l];
        u[5][l] = t[4][l] - t[5][l];
        u[6][l] = t[6][l] + t[7][l];
        u[7][l] = t[6][l] - t[7][l];
    }
    for l in 0..8 {
        t[0][l] = u[0][l] + u[2][l];
        t[1][l] = u[1][l] + u[3][l];
        t[2][l] = u[0][l] - u[2][l];
        t[3][l] = u[1][l] - u[3][l];
        t[4][l] = u[4][l] + u[6][l];
        t[5][l] = u[5][l] + u[7][l];
        t[6][l] = u[4][l] - u[6][l];
        t[7][l] = u[5][l] - u[7][l];
    }
    for l in 0..8 {
        u[0][l] = t[0][l] + t[4][l];
        u[1][l] = t[1][l] + t[5][l];
        u[2][l] = t[2][l] + t[6][l];
        u[3][l] = t[3][l] + t[7][l];
        u[4][l] = t[0][l] - t[4][l];
        u[5][l] = t[1][l] - t[5][l];
        u[6][l] = t[2][l] - t[6][l];
        u[7][l] = t[3][l] - t[7][l];
    }
    *t = u;
}

/// SATD (8x8 Hadamard) of the 8x8 tile of `src - pred` at `(bx, by)`.
#[inline]
fn hadamard8(src: &[u16], ss: usize, pred: &[i32], ps: usize, bx: usize, by: usize) -> u32 {
    let mut t = [[0i32; 8]; 8];
    for (j, row) in t.iter_mut().enumerate() {
        let s = &src[(by + j) * ss + bx..(by + j) * ss + bx + 8];
        let p = &pred[(by + j) * ps + bx..(by + j) * ps + bx + 8];
        for l in 0..8 {
            row[l] = i32::from(s[l]) - p[l];
        }
    }
    // Vertical transform (across rows), transpose, vertical again.
    butterfly8_rows(&mut t);
    let mut u = [[0i32; 8]; 8];
    for (j, row) in t.iter().enumerate() {
        for (l, v) in row.iter().enumerate() {
            u[l][j] = *v;
        }
    }
    butterfly8_rows(&mut u);
    let acc: u32 = u.iter().flatten().map(|v| v.unsigned_abs()).sum();
    (acc + 2) >> 2
}

/// SATD of a `w`x`h` source block against a `w`-stride prediction
/// (8x8 transforms when both sides allow, else 4x4).
pub(super) fn satd(src: &[u16], ss: usize, pred: &[i32], w: usize, h: usize) -> u32 {
    let mut acc = 0u32;
    if w >= 8 && h >= 8 {
        for by in (0..h).step_by(8) {
            for bx in (0..w).step_by(8) {
                acc += hadamard8(src, ss, pred, w, bx, by);
            }
        }
    } else {
        for by in (0..h).step_by(4) {
            for bx in (0..w).step_by(4) {
                acc += hadamard4(src, ss, pred, w, bx, by);
            }
        }
        acc = (acc + 1) >> 1;
    }
    acc
}

// ---------------------------------------------------------------------
// §8.4.4.2 intra sample prediction
// ---------------------------------------------------------------------

/// The §8.4.4.2 reference samples of an `n`x`n` TB after §8.4.4.2.2
/// substitution: `left[ 0 ] == top[ 0 ] == p[ −1 ][ −1 ]`,
/// `left[ 1 + y ] = p[ −1 ][ y ]`, `top[ 1 + x ] = p[ x ][ −1 ]`.
#[derive(Clone)]
pub(super) struct Refs {
    pub(super) n: usize,
    pub(super) left: [i32; 2 * MAXB + 1],
    pub(super) top: [i32; 2 * MAXB + 1],
}

/// The mark of a reference sample that is not available for intra
/// prediction (sample values are never negative), before
/// [`Refs::substitute_marked`].
pub(super) const REF_UNAVAILABLE: i32 = -1;

impl Refs {
    /// References of no block (to be filled in place).
    pub(super) fn empty() -> Self {
        Self {
            n: 4,
            left: [0; 2 * MAXB + 1],
            top: [0; 2 * MAXB + 1],
        }
    }

    /// §8.4.4.2.2 substitution in place: `left[ ..= 2n ]` (the corner
    /// at index 0) and `top[ 1 ..= 2n ]` hold sample values or
    /// [`REF_UNAVAILABLE`]; the search runs from `p[ −1 ][ 2n − 1 ]` up
    /// to the corner, then along the top row to `p[ 2n − 1 ][ −1 ]`.
    pub(super) fn substitute_marked(&mut self, bit_depth: u8) {
        let n2 = 2 * self.n;
        let first = self.left[..=n2]
            .iter()
            .rev()
            .chain(&self.top[1..=n2])
            .copied()
            .find(|&v| v >= 0);
        let Some(mut prev) = first else {
            let v = 1i32 << (bit_depth - 1);
            self.left[..=n2].fill(v);
            self.top[..=n2].fill(v);
            return;
        };
        for v in self.left[..=n2]
            .iter_mut()
            .rev()
            .chain(&mut self.top[1..=n2])
        {
            if *v < 0 {
                *v = prev;
            } else {
                prev = *v;
            }
        }
        self.top[0] = self.left[0];
    }

    /// The §8.4.4.2.3 [1 2 1]-filtered twin (strong smoothing off)
    /// written into `f`.
    pub(super) fn filter_into(&self, f: &mut Self) {
        let n2 = 2 * self.n;
        f.n = self.n;
        let c = (self.left[1] + 2 * self.left[0] + self.top[1] + 2) >> 2;
        f.left[0] = c;
        f.top[0] = c;
        for k in 1..n2 {
            f.left[k] = (self.left[k + 1] + 2 * self.left[k] + self.left[k - 1] + 2) >> 2;
            f.top[k] = (self.top[k + 1] + 2 * self.top[k] + self.top[k - 1] + 2) >> 2;
        }
        f.left[n2] = self.left[n2];
        f.top[n2] = self.top[n2];
    }

    /// §8.4.4.2.2 substitution over `(value, available)` marks in the
    /// same layout as [`Refs`].
    pub(super) fn substituted(
        n: usize,
        left: &[(i32, bool); 2 * MAXB + 1],
        top: &[(i32, bool); 2 * MAXB + 1],
        bit_depth: u8,
    ) -> Self {
        let mut r = Self {
            n,
            left: [0; 2 * MAXB + 1],
            top: [0; 2 * MAXB + 1],
        };
        // The search order: p[−1][2n−1] up to p[−1][−1], then
        // p[0][−1] right to p[2n−1][−1].
        let order_len = 4 * n + 1;
        let at = |k: usize| -> (i32, bool) {
            if k < 2 * n + 1 {
                left[2 * n - k]
            } else {
                top[k - 2 * n]
            }
        };
        let first = (0..order_len).find(|&k| at(k).1);
        let Some(first) = first else {
            let v = 1i32 << (bit_depth - 1);
            r.left[..=2 * n].fill(v);
            r.top[..=2 * n].fill(v);
            return r;
        };
        let mut prev = at(first).0;
        for k in 0..order_len {
            let (v, ok) = at(k);
            let val = if ok { v } else { prev };
            prev = val;
            if k < 2 * n + 1 {
                r.left[2 * n - k] = val;
            } else {
                r.top[k - 2 * n] = val;
            }
        }
        r.top[0] = r.left[0];
        r
    }

    /// The §8.4.4.2.3 [1 2 1]-filtered twin (strong smoothing off).
    pub(super) fn filtered(&self) -> Self {
        let n2 = 2 * self.n;
        let mut f = self.clone();
        let c = (self.left[1] + 2 * self.left[0] + self.top[1] + 2) >> 2;
        f.left[0] = c;
        f.top[0] = c;
        for k in 1..n2 {
            f.left[k] = (self.left[k + 1] + 2 * self.left[k] + self.left[k - 1] + 2) >> 2;
            f.top[k] = (self.top[k + 1] + 2 * self.top[k] + self.top[k - 1] + 2) >> 2;
        }
        f
    }
}

/// §8.4.4.2.3 `filterFlag` for a luma TB of size `n` (4:2:0 chroma is
/// never filtered).
pub(super) fn filter_flag(mode: u8, n: usize) -> bool {
    if mode == 1 || n == 4 {
        return false;
    }
    let d = (i32::from(mode) - 26)
        .abs()
        .min((i32::from(mode) - 10).abs());
    let thres = match n {
        8 => 7,
        16 => 1,
        _ => 0,
    };
    d > thres
}

/// §8.4.4.2.4..6 — predict an `n`x`n` block from (already filtered as
/// [`filter_flag`] requires) reference samples into `out` (stride
/// `n`). `luma` enables the DC / horizontal / vertical edge filters.
pub(super) fn predict_intra(r: &Refs, mode: u8, luma: bool, bit_depth: u8, out: &mut [i32]) {
    match r.n {
        4 => predict_intra_n::<4>(r, mode, luma, bit_depth, out),
        8 => predict_intra_n::<8>(r, mode, luma, bit_depth, out),
        16 => predict_intra_n::<16>(r, mode, luma, bit_depth, out),
        _ => predict_intra_n::<32>(r, mode, luma, bit_depth, out),
    }
}

/// The `N` samples of `v` starting at `at`, as an array.
#[inline(always)]
fn run<const N: usize>(v: &[i32], at: usize) -> &[i32; N] {
    v[at..at + N].try_into().expect("an N-sample run")
}

/// [`predict_intra`] for one block size.
#[inline(always)]
fn predict_intra_n<const N: usize>(r: &Refs, mode: u8, luma: bool, bit_depth: u8, out: &mut [i32]) {
    let ni = N as i32;
    let log2 = N.trailing_zeros();
    let max = (1i32 << bit_depth) - 1;
    let out = &mut out[..N * N];
    let top: &[i32; N] = run(&r.top, 1);
    let left: &[i32; N] = run(&r.left, 1);
    match mode {
        0 => {
            let (tr, bl) = (r.top[N + 1], r.left[N + 1]);
            // The x-dependent terms that do not depend on the row.
            let mut ramp = [0i32; N];
            for (x, d) in ramp.iter_mut().enumerate() {
                *d = (x as i32 + 1) * tr + ni;
            }
            for (y, row) in out.as_chunks_mut::<N>().0.iter_mut().enumerate() {
                let (l, yi) = (left[y], y as i32);
                let c = (yi + 1) * bl;
                for (x, ((d, &t), &rp)) in row.iter_mut().zip(top).zip(&ramp).enumerate() {
                    *d = ((ni - 1 - x as i32) * l + rp + (ni - 1 - yi) * t + c) >> (log2 + 1);
                }
            }
        }
        1 => {
            let sum: i32 = top.iter().sum::<i32>() + left.iter().sum::<i32>();
            let dc = (sum + ni) >> (log2 + 1);
            out.fill(dc);
            if luma && N < 32 {
                out[0] = (left[0] + 2 * dc + top[0] + 2) >> 2;
                for (o, &t) in out[1..N].iter_mut().zip(&top[1..]) {
                    *o = (t + 3 * dc + 2) >> 2;
                }
                for (row, &l) in out.as_chunks_mut::<N>().0.iter_mut().zip(left).skip(1) {
                    row[0] = (l + 3 * dc + 2) >> 2;
                }
            }
        }
        _ => {
            let angle = ANGLE[mode as usize];
            let vertical = mode >= 18;
            // ref[ k ] at index k + N (k = −N..=2N).
            let mut refa = [0i32; 3 * MAXB + 1];
            let (main, side) = if vertical {
                (&r.top, &r.left)
            } else {
                (&r.left, &r.top)
            };
            refa[N..=2 * N].copy_from_slice(&main[..=N]);
            if angle < 0 {
                let last = (ni * angle) >> 5;
                if last < -1 {
                    let inv = INV_ANGLE[mode as usize];
                    for k in last..=-1 {
                        let idx = (k * inv + 128) >> 8;
                        refa[(k + ni) as usize] = side[idx as usize];
                    }
                }
            } else {
                refa[2 * N + 1..=3 * N].copy_from_slice(&main[N + 1..=2 * N]);
            }
            // Line `a` (a row of a vertical mode, a column of a
            // horizontal one) is one weighted pair of reference runs.
            let line = |a: usize| -> [i32; N] {
                let pos = (a as i32 + 1) * angle;
                let fact = pos & 31;
                let base = ((pos >> 5) + 1 + ni) as usize;
                let lo: &[i32; N] = run(&refa, base);
                if fact == 0 {
                    return *lo;
                }
                let hi: &[i32; N] = run(&refa, base + 1);
                let mut v = [0i32; N];
                for ((d, &p), &q) in v.iter_mut().zip(lo).zip(hi) {
                    *d = ((32 - fact) * p + fact * q + 16) >> 5;
                }
                v
            };
            if vertical {
                for (a, row) in out.as_chunks_mut::<N>().0.iter_mut().enumerate() {
                    row.copy_from_slice(&line(a));
                }
            } else {
                for a in 0..N {
                    let col = line(a);
                    for (row, &v) in out.as_chunks_mut::<N>().0.iter_mut().zip(&col) {
                        row[a] = v;
                    }
                }
            }
            if luma && N < 32 {
                if mode == 26 {
                    for (row, &l) in out.as_chunks_mut::<N>().0.iter_mut().zip(left) {
                        row[0] = (top[0] + ((l - r.left[0]) >> 1)).clamp(0, max);
                    }
                } else if mode == 10 {
                    for (o, &t) in out[..N].iter_mut().zip(top) {
                        *o = (left[0] + ((t - r.top[0]) >> 1)).clamp(0, max);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// §8.5.3.3.3 fractional sample interpolation
// ---------------------------------------------------------------------

/// Copy the `(w + 7)`x`(h + 7)` (luma, `taps == 8`) or `(w + 3)`x
/// `(h + 3)` (chroma, `taps == 4`) reference window whose top-left
/// integer sample is `(x, y)` minus the filter's leading taps into
/// `win`, clamping coordinates to the plane per eqs 8-228 / 8-229.
#[allow(clippy::too_many_arguments)]
fn window(
    plane: &[u16],
    pw: usize,
    ph: usize,
    x: i32,
    y: i32,
    w: usize,
    h: usize,
    taps: usize,
    win: &mut [u16],
) -> usize {
    let lead = (taps / 2 - 1) as i32;
    let ws = w + taps - 1;
    for j in 0..h + taps - 1 {
        let ry = (y - lead + j as i32).clamp(0, ph as i32 - 1) as usize;
        let row = &plane[ry * pw..ry * pw + pw];
        for i in 0..ws {
            let rx = (x - lead + i as i32).clamp(0, pw as i32 - 1) as usize;
            win[j * ws + i] = row[rx];
        }
    }
    ws
}

/// The separable interpolation of a `w`x`h` block from a reference
/// window `src` (stride `ss`, already offset so `src[0]` is the
/// sample at `(−lead, −lead)` of the block), with `coef_x` / `coef_y`
/// the filter rows (`None` = integer position on that axis).
#[allow(clippy::too_many_arguments)]
fn interp<const T: usize>(
    src: &[u16],
    ss: usize,
    coef_x: Option<&[i32; T]>,
    coef_y: Option<&[i32; T]>,
    w: usize,
    h: usize,
    bit_depth: u8,
    out: &mut [i16],
) {
    let lead = T / 2 - 1;
    let shift1 = u32::from(bit_depth).saturating_sub(8).min(4);
    let shift3 = 14u32.saturating_sub(u32::from(bit_depth)).max(2);
    // One horizontal pass over row `r` of the window into `dst`.
    let hrow = |r: usize, cx: &[i32; T], dst: &mut [i16]| {
        let row = &src[r * ss..r * ss + w + T - 1];
        let mut acc = [0i32; MAXB];
        for (t, &c) in cx.iter().enumerate() {
            for (a, &v) in acc[..w].iter_mut().zip(&row[t..t + w]) {
                *a += c * i32::from(v);
            }
        }
        for (d, &a) in dst[..w].iter_mut().zip(&acc[..w]) {
            *d = (a >> shift1) as i16;
        }
    };
    match (coef_x, coef_y) {
        (None, None) => {
            for j in 0..h {
                let r = &src[(j + lead) * ss + lead..(j + lead) * ss + lead + w];
                for (o, &v) in out[j * w..j * w + w].iter_mut().zip(r) {
                    *o = (i32::from(v) << shift3) as i16;
                }
            }
        }
        (Some(cx), None) => {
            for j in 0..h {
                hrow(j + lead, cx, &mut out[j * w..j * w + w]);
            }
        }
        (None, Some(cy)) => {
            for j in 0..h {
                let mut acc = [0i32; MAXB];
                for (t, &c) in cy.iter().enumerate() {
                    let row = &src[(j + t) * ss + lead..(j + t) * ss + lead + w];
                    for (a, &v) in acc[..w].iter_mut().zip(row) {
                        *a += c * i32::from(v);
                    }
                }
                for (o, &a) in out[j * w..j * w + w].iter_mut().zip(&acc[..w]) {
                    *o = (a >> shift1) as i16;
                }
            }
        }
        (Some(cx), Some(cy)) => {
            let mut tmp = [0i16; (MAXB + 7) * MAXB];
            for j in 0..h + T - 1 {
                hrow(j, cx, &mut tmp[j * w..j * w + w]);
            }
            for j in 0..h {
                let mut acc = [0i32; MAXB];
                for (t, &c) in cy.iter().enumerate() {
                    let row = &tmp[(j + t) * w..(j + t) * w + w];
                    for (a, &v) in acc[..w].iter_mut().zip(row) {
                        *a += c * i32::from(v);
                    }
                }
                for (o, &a) in out[j * w..j * w + w].iter_mut().zip(&acc[..w]) {
                    *o = (a >> 6) as i16;
                }
            }
        }
    }
}

/// §8.5.3.3.3.1 — the `predSamplesLX` (14-bit intermediate) luma
/// block of a `w`x`h` PB at `(x, y)` with quarter-sample `mv`.
#[allow(clippy::too_many_arguments)]
pub(super) fn mc_luma(
    plane: &[u16],
    pw: usize,
    ph: usize,
    x: usize,
    y: usize,
    mv: [i32; 2],
    w: usize,
    h: usize,
    bit_depth: u8,
    out: &mut [i16],
) {
    let xi = x as i32 + (mv[0] >> 2);
    let yi = y as i32 + (mv[1] >> 2);
    let (fx, fy) = ((mv[0] & 3) as usize, (mv[1] & 3) as usize);
    let cx = (fx != 0).then(|| &FL[fx]);
    let cy = (fy != 0).then(|| &FL[fy]);
    if xi >= 3 && yi >= 3 && xi as usize + w + 4 <= pw && yi as usize + h + 4 <= ph {
        let o = (yi as usize - 3) * pw + xi as usize - 3;
        interp::<8>(&plane[o..], pw, cx, cy, w, h, bit_depth, out);
    } else {
        let mut win = [0u16; (MAXB + 7) * (MAXB + 7)];
        let ws = window(plane, pw, ph, xi, yi, w, h, 8, &mut win);
        interp::<8>(&win, ws, cx, cy, w, h, bit_depth, out);
    }
}

/// §8.5.3.3.3.2 — the chroma twin of [`mc_luma`] at 4:2:0: `(x, y)`
/// in chroma samples, `mv` the luma vector (eighth-chroma-sample
/// units at 4:2:0).
#[allow(clippy::too_many_arguments)]
pub(super) fn mc_chroma(
    plane: &[u16],
    pw: usize,
    ph: usize,
    x: usize,
    y: usize,
    mv: [i32; 2],
    w: usize,
    h: usize,
    bit_depth: u8,
    out: &mut [i16],
) {
    let xi = x as i32 + (mv[0] >> 3);
    let yi = y as i32 + (mv[1] >> 3);
    let (fx, fy) = ((mv[0] & 7) as usize, (mv[1] & 7) as usize);
    let cx = (fx != 0).then(|| &FC[fx]);
    let cy = (fy != 0).then(|| &FC[fy]);
    if xi >= 1 && yi >= 1 && xi as usize + w + 2 <= pw && yi as usize + h + 2 <= ph {
        let o = (yi as usize - 1) * pw + xi as usize - 1;
        interp::<4>(&plane[o..], pw, cx, cy, w, h, bit_depth, out);
    } else {
        let mut win = [0u16; (MAXB + 7) * (MAXB + 7)];
        let ws = window(plane, pw, ph, xi, yi, w, h, 4, &mut win);
        interp::<4>(&win, ws, cx, cy, w, h, bit_depth, out);
    }
}

/// §8.5.3.3.4.2 eq. 8-252 — the default uni-prediction samples.
pub(super) fn weight_uni(a: &[i16], len: usize, bit_depth: u8, out: &mut [i32]) {
    let shift = 14 - u32::from(bit_depth);
    let off = 1i32 << (shift - 1);
    let max = (1i32 << bit_depth) - 1;
    for (o, &v) in out[..len].iter_mut().zip(&a[..len]) {
        *o = ((i32::from(v) + off) >> shift).clamp(0, max);
    }
}

/// §8.5.3.3.4.2 eq. 8-254 — the default bi-prediction samples.
pub(super) fn weight_bi(a: &[i16], b: &[i16], len: usize, bit_depth: u8, out: &mut [i32]) {
    let shift = 15 - u32::from(bit_depth);
    let off = 1i32 << (shift - 1);
    let max = (1i32 << bit_depth) - 1;
    for ((o, &x), &y) in out[..len].iter_mut().zip(&a[..len]).zip(&b[..len]) {
        *o = ((i32::from(x) + i32::from(y) + off) >> shift).clamp(0, max);
    }
}

// ---------------------------------------------------------------------
// Forward transform + quantization
// ---------------------------------------------------------------------

/// The `n`-point basis row `k` (DST-VII when `dst`).
#[inline]
fn basis_row(n: usize, k: usize, dst: bool) -> &'static [i32] {
    if dst {
        &DST4[k][..]
    } else {
        &DCT32[k * (32 / n)][..n]
    }
}

/// The `N`-point basis row `k` as a fixed-size array.
#[inline(always)]
fn basis<const N: usize>(k: usize, dst: bool) -> &'static [i32; N] {
    basis_row(N, k, dst)[..N]
        .try_into()
        .expect("an N-point basis row")
}

/// Forward 2-D transform of an `n`x`n` residual (DST-VII when `dst`,
/// `n == 4` only) at the stage shifts `log2 + BitDepth − 9` / `log2 +
/// 6`. The sums run in `i32`: with every residual within `±2^BitDepth`
/// the first stage stays below `2^BitDepth · 32 · 90` and the second
/// below `32 · 90 · ( 90 · 2^9 + 1 )`.
pub(super) fn forward(res: &[i32], n: usize, bit_depth: u8, dst: bool, out: &mut [i32]) {
    match n {
        4 => forward_n::<4, 16>(res, bit_depth, dst, out),
        8 => forward_n::<8, 64>(res, bit_depth, dst, out),
        16 => forward_n::<16, 256>(res, bit_depth, dst, out),
        _ => forward_n::<32, 1024>(res, bit_depth, dst, out),
    }
}

/// [`forward`] for one block size (`NN == N * N`).
#[inline(always)]
fn forward_n<const N: usize, const NN: usize>(
    res: &[i32],
    bit_depth: u8,
    dst: bool,
    out: &mut [i32],
) {
    let log2 = N.trailing_zeros();
    let shift1 = log2 + u32::from(bit_depth) - 9;
    let shift2 = log2 + 6;
    let (r1, r2) = (1i32 << (shift1 - 1), 1i32 << (shift2 - 1));
    let res: &[i32; NN] = res[..NN].try_into().expect("an N x N residual");
    let out: &mut [i32; NN] = (&mut out[..NN]).try_into().expect("an N x N output");
    let mut tmp = [0i32; NN];
    // Rows: tmp[ y ][ k ] = Σ_j basis( k, j ) · res[ y ][ j ].
    for (row, t) in res
        .as_chunks::<N>()
        .0
        .iter()
        .zip(tmp.as_chunks_mut::<N>().0.iter_mut())
    {
        for (k, o) in t.iter_mut().enumerate() {
            let b = basis::<N>(k, dst);
            let acc: i32 = row.iter().zip(b).map(|(&r, &c)| r * c).sum();
            *o = (acc + r1) >> shift1;
        }
    }
    // Columns: out[ k ][ x ] = Σ_y basis( k, y ) · tmp[ y ][ x ].
    for (k, o) in out.as_chunks_mut::<N>().0.iter_mut().enumerate() {
        let b = basis::<N>(k, dst);
        let mut acc = [0i32; N];
        for (y, t) in tmp.as_chunks::<N>().0.iter().enumerate() {
            let c = b[y];
            for (a, &v) in acc.iter_mut().zip(t) {
                *a += c * v;
            }
        }
        for (d, &a) in o.iter_mut().zip(&acc) {
            *d = (a + r2) >> shift2;
        }
    }
}

/// Dead-zone quantization of `n`x`n` coefficients at `qP` (`Qp′`);
/// `intra` selects the 1/3 rounding offset (else 1/6). Returns the
/// number of nonzero levels.
pub(super) fn quantize(
    coef: &[i32],
    n: usize,
    qp: u32,
    bit_depth: u8,
    intra: bool,
    levels: &mut [i32],
) -> u32 {
    let log2 = n.trailing_zeros();
    let shift = 14 + qp / 6 + (15 - u32::from(bit_depth) - log2);
    let scale = QUANT_SCALE[(qp % 6) as usize];
    let offset = (if intra { 171i64 } else { 85i64 }) << (shift - 9);
    let mut nz = 0;
    for (l, &c) in levels[..n * n].iter_mut().zip(&coef[..n * n]) {
        let a = ((i64::from(c.unsigned_abs()) * scale + offset) >> shift).min(32767) as i32;
        *l = if c < 0 { -a } else { a };
        nz += u32::from(a != 0);
    }
    nz
}

/// Rate estimate (bins) of one TB's levels: last-position prefix,
/// significance and magnitude bins.
pub(super) fn levels_bits(levels: &[i32], n: usize) -> u32 {
    let mut bits = 0u32;
    let mut last = 0usize;
    for (i, &l) in levels[..n * n].iter().enumerate() {
        if l != 0 {
            let a = l.unsigned_abs();
            bits += 2 + 2 * (32 - a.leading_zeros());
            last = i;
        }
    }
    if bits == 0 {
        return 1;
    }
    let (lx, ly) = (last % n, last / n);
    bits + (lx.max(ly) as u32).max(1).ilog2() * 2 + 4 + (lx + ly) as u32 / 2
}

// ---------------------------------------------------------------------
// §8.6.2 scaling + §8.6.4.2 inverse transform
// ---------------------------------------------------------------------

/// Reconstruct the residual of an `n`x`n` TB from its levels: §8.6.3
/// flat scaling at `qP`, the two §8.6.4.2 stages with the
/// intermediate clip, the eq. 8-311 `bdShift`.
pub(super) fn dequant_inverse(
    levels: &[i32],
    n: usize,
    qp: u32,
    bit_depth: u8,
    dst: bool,
    out: &mut [i32],
) {
    match n {
        4 => dequant_inverse_n::<4, 16>(levels, qp, bit_depth, dst, out),
        8 => dequant_inverse_n::<8, 64>(levels, qp, bit_depth, dst, out),
        16 => dequant_inverse_n::<16, 256>(levels, qp, bit_depth, dst, out),
        _ => dequant_inverse_n::<32, 1024>(levels, qp, bit_depth, dst, out),
    }
}

/// [`dequant_inverse`] for one block size (`NN == N * N`).
#[inline(always)]
fn dequant_inverse_n<const N: usize, const NN: usize>(
    levels: &[i32],
    qp: u32,
    bit_depth: u8,
    dst: bool,
    out: &mut [i32],
) {
    let log2 = N.trailing_zeros();
    let bd_shift = u32::from(bit_depth) + log2 - 5;
    let scale = i64::from(LEVEL_SCALE[(qp % 6) as usize]) * 16;
    let add = 1i64 << (bd_shift - 1);
    let levels: &[i32; NN] = levels[..NN].try_into().expect("N x N levels");
    let out: &mut [i32; NN] = (&mut out[..NN]).try_into().expect("an N x N output");
    let mut d = [0i32; NN];
    let mut rows_nz = [false; N];
    for ((lrow, drow), nz) in levels
        .as_chunks::<N>()
        .0
        .iter()
        .zip(d.as_chunks_mut::<N>().0.iter_mut())
        .zip(rows_nz.iter_mut())
    {
        for (&l, o) in lrow.iter().zip(drow) {
            if l != 0 {
                *o = ((((i64::from(l) * scale) << (qp / 6)) + add) >> bd_shift).clamp(-32768, 32767)
                    as i32;
                *nz = true;
            }
        }
    }
    // Stage 1 (columns): g[ y ][ x ] = Σ_k basis( k, y ) · d[ k ][ x ].
    let mut g = [0i32; NN];
    for (y, grow) in g.as_chunks_mut::<N>().0.iter_mut().enumerate() {
        let mut acc = [0i32; N];
        for (k, drow) in d.as_chunks::<N>().0.iter().enumerate() {
            if !rows_nz[k] {
                continue;
            }
            let c = basis::<N>(k, dst)[y];
            for (a, &v) in acc.iter_mut().zip(drow) {
                *a += c * v;
            }
        }
        for (d, &a) in grow.iter_mut().zip(&acc) {
            *d = ((a + 64) >> 7).clamp(-32768, 32767);
        }
    }
    // Stage 2 (rows) + bdShift: out[ y ][ x ] = Σ_k basis( k, x ) · g[ y ][ k ].
    let shift2 = 20 - u32::from(bit_depth);
    let add2 = 1i32 << (shift2 - 1);
    for (grow, orow) in g
        .as_chunks::<N>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<N>().0.iter_mut())
    {
        let mut acc = [0i32; N];
        for (k, &v) in grow.iter().enumerate() {
            if v == 0 {
                continue;
            }
            let b = basis::<N>(k, dst);
            for (a, &c) in acc.iter_mut().zip(b) {
                *a += c * v;
            }
        }
        for (d, &a) in orow.iter_mut().zip(&acc) {
            *d = (a + add2) >> shift2;
        }
    }
}

// ---------------------------------------------------------------------
// Half-sample planes for the motion search
// ---------------------------------------------------------------------

/// Padding (luma samples) around each half-sample plane: the motion
/// search keeps candidate blocks inside it.
pub(super) const HP_PAD: usize = 80;

/// A reference picture's luma at the four half-sample phases
/// (integer, x + ½, y + ½, both), each as the uni-prediction sample
/// values of that phase (so a half-sample motion vector's prediction
/// is read, not filtered), edge-padded by [`HP_PAD`].
#[derive(Debug)]
pub(crate) struct HalfPel {
    stride: usize,
    planes: [Vec<u16>; 4],
}

impl HalfPel {
    /// Build the planes of a `w`x`h` luma plane on up to `threads`
    /// workers (bands of 32 rows).
    pub(super) fn build(y: &[u16], w: usize, h: usize, bit_depth: u8, threads: usize) -> Self {
        let stride = w + 2 * HP_PAD;
        let rows = h + 2 * HP_PAD;
        let mut planes: [Vec<u16>; 4] = std::array::from_fn(|_| vec![0u16; stride * rows]);
        let bands = h.div_ceil(MAXB);
        let fill_band = |phase: usize, band: usize, out: &mut [u16]| {
            // `out` holds the plane rows of this band (interior only).
            let y0 = band * MAXB;
            let bh = (h - y0).min(MAXB);
            let mv = [[0, 0], [2, 0], [0, 2], [2, 2]][phase];
            let mut inter = [0i16; MAXB * MAXB];
            let mut pred = [0i32; MAXB * MAXB];
            for x0 in (0..w).step_by(MAXB) {
                let bw = (w - x0).min(MAXB);
                mc_luma(y, w, h, x0, y0, mv, bw, bh, bit_depth, &mut inter);
                weight_uni(&inter, bw * bh, bit_depth, &mut pred);
                for j in 0..bh {
                    let row = &mut out[j * stride + HP_PAD + x0..j * stride + HP_PAD + x0 + bw];
                    for (o, &v) in row.iter_mut().zip(&pred[j * bw..j * bw + bw]) {
                        *o = v as u16;
                    }
                }
            }
        };
        {
            // Interior rows of every phase, banded across workers.
            let mut jobs: Vec<(usize, usize, &mut [u16])> = Vec::new();
            for (phase, plane) in planes.iter_mut().enumerate() {
                let interior = &mut plane[HP_PAD * stride..(HP_PAD + h) * stride];
                for (band, chunk) in interior.chunks_mut(MAXB * stride).enumerate() {
                    jobs.push((phase, band, chunk));
                }
            }
            debug_assert_eq!(jobs.len(), 4 * bands);
            let workers = threads.clamp(1, jobs.len());
            if workers == 1 {
                for (phase, band, chunk) in jobs {
                    fill_band(phase, band, chunk);
                }
            } else {
                let queue = std::sync::Mutex::new(jobs);
                std::thread::scope(|scope| {
                    for _ in 0..workers {
                        scope.spawn(|| {
                            loop {
                                let job = queue.lock().expect("band queue").pop();
                                let Some((phase, band, chunk)) = job else {
                                    break;
                                };
                                fill_band(phase, band, chunk);
                            }
                        });
                    }
                });
            }
        }
        for plane in &mut planes {
            for r in HP_PAD..HP_PAD + h {
                let row = &mut plane[r * stride..(r + 1) * stride];
                let (l, rr) = (row[HP_PAD], row[HP_PAD + w - 1]);
                row[..HP_PAD].fill(l);
                row[HP_PAD + w..].fill(rr);
            }
            let (top, bottom) = (HP_PAD * stride, (HP_PAD + h - 1) * stride);
            for r in 0..HP_PAD {
                plane.copy_within(top..top + stride, r * stride);
                plane.copy_within(bottom..bottom + stride, (HP_PAD + h + r) * stride);
            }
        }
        Self { stride, planes }
    }

    /// The row stride of every plane.
    pub(super) fn stride(&self) -> usize {
        self.stride
    }

    /// The plane of half-sample phase `(px, py)` and the index of its
    /// sample at integer `(x, y)` (which may lie in the padding).
    #[inline]
    pub(super) fn at(&self, px: usize, py: usize, x: i32, y: i32) -> (&[u16], usize) {
        let i = (y + HP_PAD as i32) as usize * self.stride + (x + HP_PAD as i32) as usize;
        (&self.planes[py * 2 + px], i)
    }
}

/// The search-grade prediction of a `w`x`h` block at `(x0, y0)` with
/// quarter-sample `mv`: exact at integer and half-sample positions,
/// the rounded mean of the two nearest half-sample-grid samples at
/// quarter positions. The block must lie inside the padding.
#[allow(clippy::too_many_arguments)]
pub(super) fn approx_pred(
    hp: &HalfPel,
    x0: usize,
    y0: usize,
    mv: [i32; 2],
    w: usize,
    h: usize,
    out: &mut [i32],
) {
    // Half-sample grid coordinates of the block origin (floor).
    let hx = 2 * x0 as i32 + (mv[0] >> 1);
    let hy = 2 * y0 as i32 + (mv[1] >> 1);
    let (ox, oy) = (mv[0] & 1, mv[1] & 1);
    let st = hp.stride();
    let read = |gx: i32, gy: i32| hp.at((gx & 1) as usize, (gy & 1) as usize, gx >> 1, gy >> 1);
    let (pa, ia) = read(hx, hy);
    if ox == 0 && oy == 0 {
        for j in 0..h {
            for (o, &v) in out[j * w..j * w + w]
                .iter_mut()
                .zip(&pa[ia + j * st..ia + j * st + w])
            {
                *o = i32::from(v);
            }
        }
        return;
    }
    let (pb, ib) = read(hx + ox, hy + oy);
    for j in 0..h {
        let ra = &pa[ia + j * st..ia + j * st + w];
        let rb = &pb[ib + j * st..ib + j * st + w];
        for ((o, &a), &b) in out[j * w..j * w + w].iter_mut().zip(ra).zip(rb) {
            *o = (i32::from(a) + i32::from(b) + 1) >> 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intra_pred::{
        Component as PredComponent, IntraPredParams, MarkedReferenceSamples,
        intra_predict_with_substitution,
    };
    use crate::transform::{BlockParams, Component, PredMode, residual_block};

    /// A deterministic pseudo-random sequence.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
    }

    #[test]
    fn intra_matches_decode_path() {
        let mut rng = Lcg(7);
        for bd in [8u8, 10] {
            for n in [4usize, 8, 16, 32] {
                for trial in 0..6 {
                    let mut left = [(0, false); 2 * MAXB + 1];
                    let mut top = [(0, false); 2 * MAXB + 1];
                    let mut mleft = Vec::new();
                    let mut mtop = Vec::new();
                    let max = (1u32 << bd) - 1;
                    let smooth = trial % 2 == 0;
                    let mut v = rng.next() % (max + 1);
                    let mut sample = |rng: &mut Lcg| {
                        if smooth {
                            v = (v as i32 + (rng.next() % 9) as i32 - 4).clamp(0, max as i32)
                                as u32;
                            v as i32
                        } else {
                            (rng.next() % (max + 1)) as i32
                        }
                    };
                    let corner = (sample(&mut rng), trial != 3);
                    left[0] = corner;
                    top[0] = corner;
                    for k in 0..2 * n {
                        let a = (sample(&mut rng), !(trial == 1 && k >= n) && trial != 5);
                        let b = (sample(&mut rng), !(trial == 2 && k < n) && trial != 5);
                        left[1 + k] = a;
                        top[1 + k] = b;
                        mleft.push(a);
                        mtop.push(b);
                    }
                    let marked = MarkedReferenceSamples::new(n, corner, mleft, mtop).unwrap();
                    let r = Refs::substituted(n, &left, &top, bd);
                    let rf = r.filtered();
                    // The in-place twins (sentinel marks, filter into
                    // a reused value) agree sample for sample.
                    let mut marked_r = Refs::empty();
                    marked_r.n = n;
                    for k in 0..=2 * n {
                        let (l, t) = (left[k], top[k]);
                        marked_r.left[k] = if l.1 { l.0 } else { REF_UNAVAILABLE };
                        marked_r.top[k] = if t.1 { t.0 } else { REF_UNAVAILABLE };
                    }
                    marked_r.substitute_marked(bd);
                    assert_eq!(marked_r.left[..=2 * n], r.left[..=2 * n], "n {n} {trial}");
                    assert_eq!(marked_r.top[..=2 * n], r.top[..=2 * n], "n {n} {trial}");
                    let mut filt_r = Refs::empty();
                    marked_r.filter_into(&mut filt_r);
                    assert_eq!(filt_r.left[..=2 * n], rf.left[..=2 * n], "n {n} {trial}");
                    assert_eq!(filt_r.top[..=2 * n], rf.top[..=2 * n], "n {n} {trial}");
                    for mode in 0..=34u8 {
                        for luma in [true, false] {
                            let params = IntraPredParams {
                                pred_mode_intra: mode,
                                cidx: if luma {
                                    PredComponent::Luma
                                } else {
                                    PredComponent::Cb
                                },
                                bit_depth: bd,
                                bit_depth_luma: bd,
                                intra_smoothing_disabled: false,
                                strong_intra_smoothing_enabled: false,
                                chroma_array_type_3: false,
                                disable_boundary_filter: false,
                            };
                            let want = intra_predict_with_substitution(&marked, &params).unwrap();
                            let mut got = [0i32; MAXB * MAXB];
                            let use_f = luma && filter_flag(mode, n);
                            predict_intra(if use_f { &rf } else { &r }, mode, luma, bd, &mut got);
                            assert_eq!(
                                &got[..n * n],
                                &want[..],
                                "bd {bd} n {n} mode {mode} luma {luma} trial {trial}"
                            );
                            // 4:4:4 chroma takes the filtered references
                            // (and no edge filters).
                            if !luma {
                                let params = IntraPredParams {
                                    chroma_array_type_3: true,
                                    ..params
                                };
                                let want =
                                    intra_predict_with_substitution(&marked, &params).unwrap();
                                let use_f = filter_flag(mode, n);
                                predict_intra(
                                    if use_f { &rf } else { &r },
                                    mode,
                                    false,
                                    bd,
                                    &mut got,
                                );
                                assert_eq!(
                                    &got[..n * n],
                                    &want[..],
                                    "4:4:4 chroma bd {bd} n {n} mode {mode} trial {trial}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn inverse_matches_decode_path() {
        let mut rng = Lcg(11);
        for bd in [8u8, 10] {
            for n in [4usize, 8, 16, 32] {
                for dst in [false, true] {
                    if dst && n != 4 {
                        continue;
                    }
                    for qp in [4u32, 22, 37, 51, 63] {
                        let mut levels = vec![0i32; n * n];
                        for (i, l) in levels.iter_mut().enumerate() {
                            if rng.next().is_multiple_of(4) || i == 0 {
                                *l = (rng.next() % 64) as i32 - 32;
                            }
                        }
                        let want = residual_block(
                            &levels,
                            None,
                            BlockParams {
                                n_tbs: n,
                                q_p: qp,
                                component: Component::Luma,
                                pred_mode: if dst {
                                    PredMode::Intra
                                } else {
                                    PredMode::Inter
                                },
                                bit_depth: bd,
                                extended_precision: false,
                                transquant_bypass: false,
                                transform_skip: false,
                                transform_skip_rotation_enabled: false,
                            },
                        )
                        .unwrap();
                        let mut got = [0i32; MAXB * MAXB];
                        dequant_inverse(&levels, n, qp, bd, dst, &mut got);
                        assert_eq!(&got[..n * n], &want[..], "bd {bd} n {n} qp {qp} dst {dst}");
                    }
                }
            }
        }
    }

    /// The `i32` forward transform equals the wide-accumulator one
    /// whenever the residual lies within the sample range — every
    /// depth, size and both bases, extreme residuals included.
    #[test]
    fn forward_matches_the_wide_transform() {
        use crate::encoder::intra::forward_transform_bd;
        let mut rng = Lcg(23);
        for bd in [8u8, 10, 12] {
            let lim = 1i32 << bd;
            for n in [4usize, 8, 16, 32] {
                for trial in 0..6 {
                    let res: Vec<i32> = (0..n * n)
                        .map(|i| match trial {
                            0 => lim,
                            1 => -lim,
                            2 => {
                                if (i + i / n) % 2 == 0 {
                                    lim
                                } else {
                                    -lim
                                }
                            }
                            _ => (rng.next() % (2 * lim as u32 + 1)) as i32 - lim,
                        })
                        .collect();
                    let mut got = [0i32; MAXB * MAXB];
                    forward(&res, n, bd, false, &mut got);
                    assert_eq!(
                        &got[..n * n],
                        &forward_transform_bd(&res, n, bd)[..],
                        "bd {bd} n {n} trial {trial}"
                    );
                    if n == 4 {
                        forward(&res, 4, bd, true, &mut got);
                        assert_eq!(
                            &got[..16],
                            &super::super::forward_transform_dst4(&res, bd)[..],
                            "dst bd {bd} trial {trial}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn quantize_round_trips_near_identity() {
        let mut rng = Lcg(3);
        for n in [4usize, 8, 16, 32] {
            let res: Vec<i32> = (0..n * n).map(|_| (rng.next() % 61) as i32 - 30).collect();
            let mut coef = [0i32; MAXB * MAXB];
            forward(&res, n, 8, false, &mut coef);
            let mut lv = [0i32; MAXB * MAXB];
            quantize(&coef, n, 4, 8, true, &mut lv);
            let mut back = [0i32; MAXB * MAXB];
            dequant_inverse(&lv, n, 4, 8, false, &mut back);
            let err: i32 = res
                .iter()
                .zip(&back)
                .map(|(a, b)| (a - b).abs())
                .max()
                .unwrap();
            assert!(err <= 2, "n {n} max error {err}");
        }
    }

    #[test]
    fn interpolation_matches_decode_path() {
        use crate::encoder::inter::{RefPlanes, predict_block};
        use crate::encoder::sample::SampleFmt;
        use crate::pu_mv::PuMotion;
        let mut rng = Lcg(5);
        for bd in [8u8, 10] {
            let (w, h) = (64usize, 48usize);
            let max = (1u32 << bd) - 1;
            let y: Vec<u16> = (0..w * h)
                .map(|_| (rng.next() % (max + 1)) as u16)
                .collect();
            let cb: Vec<u16> = (0..w * h / 4)
                .map(|_| (rng.next() % (max + 1)) as u16)
                .collect();
            let cr: Vec<u16> = (0..w * h / 4)
                .map(|_| (rng.next() % (max + 1)) as u16)
                .collect();
            let y1: Vec<u16> = (0..w * h)
                .map(|_| (rng.next() % (max + 1)) as u16)
                .collect();
            let fmt = SampleFmt::new(1, bd).unwrap();
            let r0 = RefPlanes {
                halfpel: None,
                y: &y,
                cb: &cb,
                cr: &cr,
                width: w,
                height: h,
                fmt,
            };
            let r1 = RefPlanes {
                halfpel: None,
                y: &y1,
                cb: &cr,
                cr: &cb,
                width: w,
                height: h,
                fmt,
            };
            for trial in 0..200 {
                let bw = [8usize, 16, 32][trial % 3];
                let bh = [8usize, 16, 32][(trial / 3) % 3];
                let px = (rng.next() as usize % (w / 8)) * 8;
                let py = (rng.next() as usize % (h / 8)) * 8;
                let (bw, bh) = (bw.min(w - px), bh.min(h - py));
                let mv0 = [
                    (rng.next() % 161) as i32 - 80,
                    (rng.next() % 161) as i32 - 80,
                ];
                let mv1 = [
                    (rng.next() % 161) as i32 - 80,
                    (rng.next() % 161) as i32 - 80,
                ];
                let bi = trial % 2 == 1;
                let m = PuMotion {
                    pred_flag_l0: true,
                    pred_flag_l1: bi,
                    ref_idx_l0: 0,
                    ref_idx_l1: if bi { 0 } else { -1 },
                    mv_l0: mv0,
                    mv_l1: if bi { mv1 } else { [0, 0] },
                };
                let want = predict_block(
                    std::slice::from_ref(&r0),
                    std::slice::from_ref(&r1),
                    px,
                    py,
                    bw,
                    bh,
                    &m,
                    true,
                );
                let mut a = [0i16; MAXB * MAXB];
                let mut b = [0i16; MAXB * MAXB];
                let mut got = [0i32; MAXB * MAXB];
                mc_luma(&y, w, h, px, py, mv0, bw, bh, bd, &mut a);
                if bi {
                    mc_luma(&y1, w, h, px, py, mv1, bw, bh, bd, &mut b);
                    weight_bi(&a, &b, bw * bh, bd, &mut got);
                } else {
                    weight_uni(&a, bw * bh, bd, &mut got);
                }
                assert_eq!(&got[..bw * bh], &want.luma[..], "luma trial {trial}");
                let (cw, chh) = (bw / 2, bh / 2);
                mc_chroma(&cb, w / 2, h / 2, px / 2, py / 2, mv0, cw, chh, bd, &mut a);
                if bi {
                    mc_chroma(&cr, w / 2, h / 2, px / 2, py / 2, mv1, cw, chh, bd, &mut b);
                    weight_bi(&a, &b, cw * chh, bd, &mut got);
                } else {
                    weight_uni(&a, cw * chh, bd, &mut got);
                }
                assert_eq!(&got[..cw * chh], &want.cb[..], "cb trial {trial}");
            }
        }
    }
}
