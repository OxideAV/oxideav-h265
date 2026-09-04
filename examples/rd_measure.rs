//! Rate-distortion measurement harness for the quadtree encoder.
//!
//! Encodes a deterministic synthetic corpus (or a raw 4:2:0 8-bit
//! file) at the four-point QP ladder {22, 27, 32, 37} under two tool
//! configurations and prints per-point bytes / PSNR plus the
//! Bjøntegaard rate delta (BD-rate, luma PSNR) of `--b` against
//! `--a`. Every stream is additionally decoded through this crate's
//! decoder and checked sample-exact against the encoder
//! reconstruction; `--out <dir>` also writes each `--b` stream as
//! `<dir>/<scene>-qp<N>.hevc` + `.yuv` (display-order reconstruction)
//! for a black-box reference-decoder comparison.
//!
//! ```text
//! cargo run --release --example rd_measure -- \
//!     --mode pyramid --ctb 64 --a "" --b rdoq,sdh --out /tmp/rd
//! ```
//!
//! Tool tokens: `rdoq`, `sdh`, `tu2` (transform hierarchy depth 2),
//! `tu3`, `sl` / `sl2` / `sl3` (default / flattened / steepened
//! scaling lists), `wpp`, `tiles2x2`, `wp`
//! (weighted prediction), `amp`, `tmvp`, `refs2`, `noaq`.

use std::fmt::Write as _;
use std::path::PathBuf;

use oxideav_h265::encoder::ctu::TreeCfg;
use oxideav_h265::encoder::inter::{FrameRecon, LowDelayPEncoder, YuvFrame};
use oxideav_h265::encoder::loopfilter::LoopFilterCfg;
use oxideav_h265::encoder::pyramid::PyramidEncoder;
use oxideav_h265::sequence::decode_annexb_sequence;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Intra,
    LowDelay,
    Pyramid,
}

#[derive(Clone, Default)]
struct Tools {
    rdoq: bool,
    sdh: bool,
    tu_depth: u32,
    scaling_lists: u8,
    wpp: bool,
    tiles: Option<(u32, u32)>,
    weighted_pred: bool,
    amp: bool,
    tmvp: bool,
    refs: usize,
    aq: u8,
}

fn parse_tools(s: &str) -> Result<Tools, String> {
    let mut t = Tools {
        tu_depth: 1,
        refs: 1,
        ..Tools::default()
    };
    for tok in s.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        match tok {
            "rdoq" => t.rdoq = true,
            "sdh" => t.sdh = true,
            "tu2" => t.tu_depth = 2,
            "tu3" => t.tu_depth = 3,
            "sl" => t.scaling_lists = 1,
            "sl2" => t.scaling_lists = 2,
            "sl3" => t.scaling_lists = 3,
            "wpp" => t.wpp = true,
            "tiles2x2" => t.tiles = Some((2, 2)),
            "tiles2x1" => t.tiles = Some((2, 1)),
            "wp" => t.weighted_pred = true,
            "amp" => t.amp = true,
            "tmvp" => t.tmvp = true,
            "refs2" => t.refs = 2,
            "aq1" => t.aq = 1,
            "aq2" => t.aq = 2,
            other => return Err(format!("unknown tool token {other:?}")),
        }
    }
    Ok(t)
}

fn tree_cfg(ctb: usize, t: &Tools) -> TreeCfg {
    let cfg = TreeCfg::new(ctb)
        .expect("ctb 16/32/64")
        .with_tu_depth(t.tu_depth, t.tu_depth)
        .with_rdoq(t.rdoq)
        .with_sign_hiding(t.sdh)
        .with_scaling_lists(t.scaling_lists);
    assert!(
        !(t.wpp || t.weighted_pred || t.tiles.is_some()),
        "tool not wired yet"
    );
    cfg
}

struct Frame {
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
}

/// Deterministic per-pixel hash noise (stable across frames so the
/// texture is trackable by motion search).
fn hash_noise(x: i64, y: i64, seed: u64) -> i32 {
    let mut h = (x as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add(seed);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h & 0xFF) as i32
}

/// A textured world plane sampled at world coordinates.
fn world_luma(x: i64, y: i64, seed: u64) -> i32 {
    let coarse = hash_noise(x >> 4, y >> 4, seed) / 2;
    let fine = hash_noise(x, y, seed ^ 0x55) / 8;
    let stripes = (((x * 3 + y * 2) / 7 % 13) * 3) as i32;
    (60 + coarse + fine + stripes).clamp(0, 255)
}

fn world_chroma(x: i64, y: i64, seed: u64, base: i32) -> i32 {
    let c = hash_noise(x >> 3, y >> 3, seed) / 6;
    (base + c - 20).clamp(0, 255)
}

fn scene(name: &str, w: usize, h: usize, n: usize) -> Vec<Frame> {
    let (cw, ch) = (w / 2, h / 2);
    (0..n)
        .map(|f| {
            let fi = f as i64;
            let mut y = vec![0u8; w * h];
            let mut cb = vec![0u8; cw * ch];
            let mut cr = vec![0u8; cw * ch];
            match name {
                // Textured background panning (3, 1) px/frame with a
                // brighter square moving the other way.
                "pan" => {
                    for j in 0..h {
                        for i in 0..w {
                            let (wx, wy) = (i as i64 + fi * 3, j as i64 + fi);
                            let mut v = world_luma(wx, wy, 1);
                            let sx = 40 + (n as i64 - fi) * 2;
                            if (i as i64) >= sx && (i as i64) < sx + 48 && (24..72).contains(&j) {
                                v = (v + 90).min(255);
                            }
                            y[j * w + i] = v as u8;
                        }
                    }
                    for j in 0..ch {
                        for i in 0..cw {
                            let (wx, wy) = (i as i64 + fi * 3 / 2, j as i64 + fi / 2);
                            cb[j * cw + i] = world_chroma(wx, wy, 2, 128) as u8;
                            cr[j * cw + i] = world_chroma(wx, wy, 3, 120) as u8;
                        }
                    }
                }
                // Smooth gradient scene with sharp edges, a global
                // luminance fade (to 70 %) and a slow drift.
                "fade" => {
                    let gain = 1000 - (fi * 300) / (n as i64 - 1).max(1);
                    for j in 0..h {
                        for i in 0..w {
                            let (wx, wy) = (i as i64 + fi, j as i64);
                            let g = ((wx * 255) / w as i64 + (wy * 128) / h as i64) / 2;
                            let edge = i64::from(((wx / 32) + (wy / 24)) % 2 == 0) * 60;
                            let tex = i64::from(hash_noise(wx, wy, 9) / 16);
                            let v = ((g + edge + tex + 20) * gain) / 1000;
                            y[j * w + i] = v.clamp(0, 255) as u8;
                        }
                    }
                    for j in 0..ch {
                        for i in 0..cw {
                            let v = 128 + ((i as i64 * 40) / cw as i64 - 20) * gain / 1000;
                            cb[j * cw + i] = v.clamp(0, 255) as u8;
                            let v = 128 - ((j as i64 * 40) / ch as i64 - 20) * gain / 1000;
                            cr[j * cw + i] = v.clamp(0, 255) as u8;
                        }
                    }
                }
                // Static high-detail background with a moving noisy
                // occluder: skip-friendly with hard residual pockets.
                "detail" => {
                    let mut rng = Lcg(77 + fi as u64);
                    for j in 0..h {
                        for i in 0..w {
                            let (xi, yj) = (i as i64, j as i64);
                            let mut v = world_luma(xi * 2, yj * 2, 5);
                            let ox = 30 + fi * 5;
                            let oy = 20 + fi * 2;
                            if xi >= ox && xi < ox + 40 && yj >= oy && yj < oy + 40 {
                                v = 100 + (rng.next() % 120) as i32;
                            }
                            y[j * w + i] = v as u8;
                        }
                    }
                    for j in 0..ch {
                        for i in 0..cw {
                            cb[j * cw + i] = world_chroma(i as i64, j as i64, 6, 130) as u8;
                            cr[j * cw + i] = world_chroma(i as i64, j as i64, 7, 126) as u8;
                        }
                    }
                }
                other => panic!("unknown scene {other}"),
            }
            Frame { y, cb, cr }
        })
        .collect()
}

fn load_yuv(path: &str, w: usize, h: usize, max_frames: usize) -> Vec<Frame> {
    let data = std::fs::read(path).expect("read yuv");
    let fsz = w * h * 3 / 2;
    data.chunks_exact(fsz)
        .take(max_frames)
        .map(|c| Frame {
            y: c[..w * h].to_vec(),
            cb: c[w * h..w * h + w * h / 4].to_vec(),
            cr: c[w * h + w * h / 4..].to_vec(),
        })
        .collect()
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let sse: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum();
    if sse == 0.0 {
        return 99.0;
    }
    let mse = sse / a.len() as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

struct Point {
    bytes: usize,
    psnr_y: f64,
    psnr_u: f64,
    psnr_v: f64,
}

#[allow(clippy::too_many_arguments)]
fn encode(
    frames: &[Frame],
    w: usize,
    h: usize,
    qp: i32,
    mode: Mode,
    ctb: usize,
    t: &Tools,
    out: Option<&PathBuf>,
) -> Point {
    let cfg = tree_cfg(ctb, t);
    let mut stream = Vec::new();
    let mut recons: Vec<Option<FrameRecon>> = (0..frames.len()).map(|_| None).collect();
    match mode {
        Mode::Intra | Mode::LowDelay => {
            let gop = if mode == Mode::Intra { 1 } else { 0 };
            let mut enc = LowDelayPEncoder::new(w, h, qp, gop)
                .expect("encoder")
                .with_tree(cfg)
                .with_loop_filters(LoopFilterCfg::all())
                .with_refs(t.refs)
                .with_temporal_mvp(t.tmvp)
                .with_amp(t.amp)
                .with_aq(t.aq);
            for (i, f) in frames.iter().enumerate() {
                let e = enc
                    .encode_frame(&YuvFrame {
                        y: &f.y,
                        cb: &f.cb,
                        cr: &f.cr,
                    })
                    .expect("frame");
                stream.extend_from_slice(&e.au);
                recons[i] = Some(e.recon);
            }
        }
        Mode::Pyramid => {
            let mut enc = PyramidEncoder::new(w, h, qp, 8)
                .expect("encoder")
                .with_tree(cfg)
                .with_loop_filters(LoopFilterCfg::all())
                .with_refs(t.refs)
                .with_temporal_mvp(t.tmvp)
                .with_amp(t.amp)
                .with_aq(t.aq);
            let mut aus = Vec::new();
            for f in frames {
                aus.extend(
                    enc.encode_frame(&YuvFrame {
                        y: &f.y,
                        cb: &f.cb,
                        cr: &f.cr,
                    })
                    .expect("frame"),
                );
            }
            aus.extend(enc.flush());
            for au in aus {
                stream.extend_from_slice(&au.au);
                recons[au.display_order] = Some(au.recon);
            }
        }
    }
    let recons: Vec<FrameRecon> = recons.into_iter().map(|r| r.expect("all coded")).collect();
    // Decoder check: sample-exact in display order.
    let decoded = decode_annexb_sequence(&stream).expect("stream decodes");
    assert_eq!(decoded.len(), frames.len(), "frame count");
    for (i, (d, r)) in decoded.iter().zip(&recons).enumerate() {
        let planar = d.picture.to_planar_u8().expect("8-bit");
        assert!(planar[..w * h] == r.y[..], "frame {i} luma mismatch");
        assert!(
            planar[w * h..w * h + w * h / 4] == r.cb[..],
            "frame {i} cb mismatch"
        );
        assert!(
            planar[w * h + w * h / 4..] == r.cr[..],
            "frame {i} cr mismatch"
        );
    }
    let (mut sy, mut su, mut sv) = (0.0, 0.0, 0.0);
    for (f, r) in frames.iter().zip(&recons) {
        sy += psnr(&f.y, &r.y);
        su += psnr(&f.cb, &r.cb);
        sv += psnr(&f.cr, &r.cr);
    }
    if let Some(dir) = out {
        let mut yuv = Vec::with_capacity(frames.len() * w * h * 3 / 2);
        for r in &recons {
            yuv.extend_from_slice(&r.y);
            yuv.extend_from_slice(&r.cb);
            yuv.extend_from_slice(&r.cr);
        }
        std::fs::write(dir.with_extension("hevc"), &stream).expect("write hevc");
        std::fs::write(dir.with_extension("yuv"), &yuv).expect("write yuv");
    }
    let n = frames.len() as f64;
    Point {
        bytes: stream.len(),
        psnr_y: sy / n,
        psnr_u: su / n,
        psnr_v: sv / n,
    }
}

/// Solve the 4x4 system `m · x = v` (Gaussian elimination, partial
/// pivoting).
fn solve4(mut m: [[f64; 4]; 4], mut v: [f64; 4]) -> [f64; 4] {
    for c in 0..4 {
        let p = (c..4)
            .max_by(|&a, &b| m[a][c].abs().partial_cmp(&m[b][c].abs()).unwrap())
            .unwrap();
        m.swap(c, p);
        v.swap(c, p);
        for r in c + 1..4 {
            let f = m[r][c] / m[c][c];
            let pivot_row = m[c];
            for (k, cell) in m[r].iter_mut().enumerate().skip(c) {
                *cell -= f * pivot_row[k];
            }
            v[r] -= f * v[c];
        }
    }
    let mut x = [0.0; 4];
    for r in (0..4).rev() {
        let mut s = v[r];
        for k in r + 1..4 {
            s -= m[r][k] * x[k];
        }
        x[r] = s / m[r][r];
    }
    x
}

/// Cubic least-squares fit `y = a0 + a1 x + a2 x² + a3 x³`.
fn cubic_fit(xs: &[f64], ys: &[f64]) -> [f64; 4] {
    let mut m = [[0.0; 4]; 4];
    let mut v = [0.0; 4];
    for (&x, &y) in xs.iter().zip(ys) {
        let p = [1.0, x, x * x, x * x * x];
        for i in 0..4 {
            v[i] += p[i] * y;
            for j in 0..4 {
                m[i][j] += p[i] * p[j];
            }
        }
    }
    solve4(m, v)
}

fn poly_integral(c: &[f64; 4], lo: f64, hi: f64) -> f64 {
    let f = |x: f64| {
        c[0] * x + c[1] * x * x / 2.0 + c[2] * x * x * x / 3.0 + c[3] * x * x * x * x / 4.0
    };
    f(hi) - f(lo)
}

/// Bjøntegaard rate delta (percent) of curve `b` against `a`: cubic
/// fits of `log10(rate)` over PSNR, integrated over the common PSNR
/// span.
fn bd_rate(a: &[Point], b: &[Point]) -> f64 {
    let fit = |pts: &[Point]| {
        let xs: Vec<f64> = pts.iter().map(|p| p.psnr_y).collect();
        let ys: Vec<f64> = pts.iter().map(|p| (p.bytes as f64).log10()).collect();
        (cubic_fit(&xs, &ys), xs)
    };
    let (ca, xa) = fit(a);
    let (cb, xb) = fit(b);
    let max_of = |v: &[f64]| v.iter().cloned().fold(f64::MIN, f64::max);
    let min_of = |v: &[f64]| v.iter().cloned().fold(f64::MAX, f64::min);
    let lo_common = min_of(&xa).max(min_of(&xb));
    let hi_common = max_of(&xa).min(max_of(&xb));
    if hi_common <= lo_common {
        return f64::NAN;
    }
    let avg = (poly_integral(&cb, lo_common, hi_common) - poly_integral(&ca, lo_common, hi_common))
        / (hi_common - lo_common);
    (10f64.powf(avg) - 1.0) * 100.0
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut mode = Mode::Pyramid;
    let mut ctb = 64usize;
    let mut a = String::new();
    let mut b = String::new();
    let mut out: Option<String> = None;
    let mut yuv: Option<String> = None;
    let (mut w, mut h) = (192usize, 128usize);
    let mut n_frames = 9usize;
    let mut qps: Vec<i32> = vec![22, 27, 32, 37];
    let mut scenes: Vec<String> = vec!["pan".into(), "fade".into(), "detail".into()];
    let mut i = 0;
    while i < args.len() {
        let v = |i: &mut usize| -> String {
            *i += 1;
            args.get(*i).cloned().unwrap_or_default()
        };
        match args[i].as_str() {
            "--mode" => {
                mode = match v(&mut i).as_str() {
                    "intra" => Mode::Intra,
                    "lowdelay" => Mode::LowDelay,
                    "pyramid" => Mode::Pyramid,
                    m => panic!("unknown mode {m}"),
                }
            }
            "--ctb" => ctb = v(&mut i).parse().expect("ctb"),
            "--a" => a = v(&mut i),
            "--b" => b = v(&mut i),
            "--out" => out = Some(v(&mut i)),
            "--yuv" => yuv = Some(v(&mut i)),
            "--width" => w = v(&mut i).parse().expect("width"),
            "--height" => h = v(&mut i).parse().expect("height"),
            "--frames" => n_frames = v(&mut i).parse().expect("frames"),
            "--qps" => {
                qps = v(&mut i)
                    .split(',')
                    .map(|q| q.parse().expect("qp"))
                    .collect()
            }
            "--scenes" => scenes = v(&mut i).split(',').map(String::from).collect(),
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }
    let ta = parse_tools(&a).expect("--a tools");
    let tb = parse_tools(&b).expect("--b tools");
    if let Some(dir) = &out {
        std::fs::create_dir_all(dir).expect("out dir");
    }
    let corpus: Vec<(String, Vec<Frame>)> = match &yuv {
        Some(p) => vec![("yuv".into(), load_yuv(p, w, h, n_frames))],
        None => scenes
            .iter()
            .map(|s| (s.clone(), scene(s, w, h, n_frames)))
            .collect(),
    };
    let mut report = String::new();
    let mut total_a = 0usize;
    let mut total_b = 0usize;
    let mut bd_sum = 0.0;
    let mut bd_n = 0;
    for (name, frames) in &corpus {
        let mut pa = Vec::new();
        let mut pb = Vec::new();
        for &qp in &qps {
            let start = std::time::Instant::now();
            let p = encode(frames, w, h, qp, mode, ctb, &ta, None);
            let ta_ms = start.elapsed().as_millis();
            let start = std::time::Instant::now();
            let path = out
                .as_ref()
                .map(|d| PathBuf::from(d).join(format!("{name}-qp{qp}")));
            let q = encode(frames, w, h, qp, mode, ctb, &tb, path.as_ref());
            let tb_ms = start.elapsed().as_millis();
            let _ = writeln!(
                report,
                "{name:>7} qp{qp:>2}  A: {:>7} B  {:6.3}/{:6.3}/{:6.3} dB ({ta_ms:>5} ms) | \
                 B: {:>7} B  {:6.3}/{:6.3}/{:6.3} dB ({tb_ms:>5} ms) | bytes {:+6.2} %  dY {:+.3}",
                p.bytes,
                p.psnr_y,
                p.psnr_u,
                p.psnr_v,
                q.bytes,
                q.psnr_y,
                q.psnr_u,
                q.psnr_v,
                (q.bytes as f64 / p.bytes as f64 - 1.0) * 100.0,
                q.psnr_y - p.psnr_y,
            );
            total_a += p.bytes;
            total_b += q.bytes;
            pa.push(p);
            pb.push(q);
        }
        if qps.len() >= 4 {
            let bd = bd_rate(&pa, &pb);
            let _ = writeln!(report, "{name:>7} BD-rate(Y) {bd:+.2} %");
            if bd.is_finite() {
                bd_sum += bd;
                bd_n += 1;
            }
        }
    }
    print!("{report}");
    println!(
        "TOTAL bytes A {total_a}  B {total_b}  ({:+.2} %)",
        (total_b as f64 / total_a as f64 - 1.0) * 100.0
    );
    if bd_n > 0 {
        println!(
            "MEAN BD-rate(Y) {:+.2} % over {bd_n} scenes",
            bd_sum / f64::from(bd_n)
        );
    }
}
