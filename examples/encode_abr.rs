//! Encode raw 4:2:0 frames under average-bitrate rate control.
//!
//! ```text
//! cargo run --example encode_abr -- in.yuv WxH BITRATE FPS out.hevc [recon.yuv] [gop=N] [pyramid=N] [aq=N] [vbv=BITS] [b] [deblock] [sao]
//! ```
//!
//! `BITRATE` is bits per second (optional `k` / `M` suffix); `FPS` is
//! an integer or a `num/den` ratio. `gop=N` re-emits an IDR every `N`
//! frames; `pyramid=N` switches to hierarchical-B mini-GOPs of `N`
//! frames (excludes `gop` / `b`); `aq=N` (1..=3) enables spatial
//! adaptive quantization (per-CTB `cu_qp_delta`); `vbv=BITS`
//! constrains every access unit to a decoder buffer of that size
//! (low-delay path only); `b` codes the
//! non-IDR frames as low-delay B slices; `deblock` / `sao` enable
//! the §8.7 in-loop filters.
//!
//! `in.yuv` is a concatenation of planar 8-bit 4:2:0 frames;
//! dimensions must be multiples of 16. The optional `recon.yuv`
//! receives the encoder's own per-frame reconstruction (what a
//! conforming decoder must output). Per-frame QP elections and sizes
//! are logged to stderr.

use oxideav_h265::encoder::inter::{LowDelayPEncoder, YuvFrame};
use oxideav_h265::encoder::loopfilter::LoopFilterCfg;
use oxideav_h265::encoder::pyramid::{encode_pyramid_with, PyramidEncoder};
use oxideav_h265::encoder::rate::RateControlCfg;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage =
        "usage: encode_abr <in.yuv> <WxH> <BITRATE> <FPS> <out.hevc> [recon.yuv] [gop=N] [b] [deblock] [sao]";
    if args.len() < 5 {
        eprintln!("{usage}");
        std::process::exit(2);
    }
    let (w, h) = args[1]
        .split_once('x')
        .and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?)))
        .expect("WxH");
    let bitrate: u64 = {
        let v = &args[2];
        let (digits, mult) = match v.as_bytes().last() {
            Some(b'k' | b'K') => (&v[..v.len() - 1], 1_000u64),
            Some(b'M') => (&v[..v.len() - 1], 1_000_000),
            _ => (&v[..], 1),
        };
        digits.parse::<u64>().expect("BITRATE") * mult
    };
    let (fps_num, fps_den) = match args[3].split_once('/') {
        None => (args[3].parse::<u32>().expect("FPS"), 1),
        Some((n, d)) => (n.parse().expect("FPS num"), d.parse().expect("FPS den")),
    };
    let data = std::fs::read(&args[0]).expect("read input");
    let frame_len = w * h * 3 / 2;
    assert!(
        !data.is_empty() && data.len() % frame_len == 0,
        "input must be whole planar 4:2:0 frames"
    );
    let frames: Vec<YuvFrame<'_>> = data
        .chunks_exact(frame_len)
        .map(|f| {
            let (y, c) = f.split_at(w * h);
            let (cb, cr) = c.split_at(w * h / 4);
            YuvFrame { y, cb, cr }
        })
        .collect();
    let flag = |name: &str| args.iter().skip(5).any(|a| a == name);
    let gop = args
        .iter()
        .skip(5)
        .find_map(|a| a.strip_prefix("gop=")?.parse::<usize>().ok())
        .unwrap_or(0);
    let pyramid = args
        .iter()
        .skip(5)
        .find_map(|a| a.strip_prefix("pyramid=")?.parse::<usize>().ok());
    let aq = args
        .iter()
        .skip(5)
        .find_map(|a| a.strip_prefix("aq=")?.parse::<u8>().ok())
        .unwrap_or(0);
    let vbv = args
        .iter()
        .skip(5)
        .find_map(|a| a.strip_prefix("vbv=")?.parse::<u64>().ok());
    let mut cfg = RateControlCfg::new(bitrate, fps_num, fps_den);
    cfg.vbv_buffer_bits = vbv;
    let lf = LoopFilterCfg {
        deblocking: flag("deblock"),
        sao_luma: flag("sao"),
        sao_chroma: flag("sao"),
    };
    if let Some(g) = pyramid {
        let enc = PyramidEncoder::new(w, h, 26, g)
            .expect("encoder")
            .with_loop_filters(lf)
            .with_aq(aq)
            .with_frame_rate(fps_num, fps_den)
            .with_rate_control(&cfg);
        let out = encode_pyramid_with(enc, &frames).expect("encode");
        std::fs::write(&args[4], &out.stream).expect("write output");
        let mut recon = Vec::new();
        for rec in &out.recon {
            recon.extend_from_slice(&rec.y);
            recon.extend_from_slice(&rec.cb);
            recon.extend_from_slice(&rec.cr);
        }
        let is_flag = |a: &str| {
            matches!(a, "b" | "deblock" | "sao")
                || a.starts_with("gop=")
                || a.starts_with("pyramid=")
                || a.starts_with("aq=")
                || a.starts_with("vbv=")
        };
        if let Some(recon_path) = args.get(5).filter(|a| !is_flag(a)) {
            std::fs::write(recon_path, &recon).expect("write recon");
        }
        let secs = frames.len() as f64 * f64::from(fps_den) / f64::from(fps_num);
        eprintln!(
            "{w}x{h} pyramid-{g} target {bitrate} b/s @ {fps_num}/{fps_den} fps: {} frames -> {} bytes ({:.0} b/s achieved)",
            frames.len(),
            out.stream.len(),
            out.stream.len() as f64 * 8.0 / secs
        );
        return;
    }
    let mut enc = LowDelayPEncoder::new(w, h, 26, gop)
        .expect("encoder")
        .with_b_slices(flag("b"))
        .with_loop_filters(lf)
        .with_aq(aq)
        .with_frame_rate(fps_num, fps_den)
        .with_rate_control(&cfg);
    let mut stream = Vec::new();
    let mut recon = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        let out = enc.encode_frame(f).expect("encode");
        eprintln!(
            "frame {i}: qp {} -> {} bytes{}",
            out.qp,
            out.au.len(),
            if out.keyframe { " (IDR)" } else { "" }
        );
        stream.extend_from_slice(&out.au);
        recon.extend_from_slice(&out.recon.y);
        recon.extend_from_slice(&out.recon.cb);
        recon.extend_from_slice(&out.recon.cr);
    }
    std::fs::write(&args[4], &stream).expect("write output");
    let is_flag = |a: &str| {
        matches!(a, "b" | "deblock" | "sao")
            || a.starts_with("gop=")
            || a.starts_with("pyramid=")
            || a.starts_with("aq=")
            || a.starts_with("vbv=")
    };
    if let Some(recon_path) = args.get(5).filter(|a| !is_flag(a)) {
        std::fs::write(recon_path, &recon).expect("write recon");
    }
    let secs = frames.len() as f64 * f64::from(fps_den) / f64::from(fps_num);
    eprintln!(
        "{w}x{h} target {bitrate} b/s @ {fps_num}/{fps_den} fps: {} frames -> {} bytes ({:.0} b/s achieved)",
        frames.len(),
        stream.len(),
        stream.len() as f64 * 8.0 / secs
    );
}
