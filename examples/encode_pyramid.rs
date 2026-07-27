//! Encode raw 4:2:0 frames as a hierarchical-B (dyadic pyramid)
//! stream.
//!
//! ```text
//! cargo run --example encode_pyramid -- in.yuv WxH QP GOP out.hevc [recon.yuv] [amp] [deblock] [sao]
//! ```
//!
//! `GOP` is the dyadic mini-GOP length (a power of two in 2..=16).
//! `in.yuv` is a concatenation of planar 8-bit 4:2:0 frames in
//! display order; dimensions must be multiples of 16. The optional
//! `recon.yuv` receives the encoder's per-frame reconstruction in
//! DISPLAY order (what a conforming decoder outputs after §C.5.2.2
//! reordering). Trailing flags: `amp` (asymmetric motion
//! partitions), `deblock` / `sao` (§8.7 in-loop filters).

use oxideav_h265::encoder::inter::YuvFrame;
use oxideav_h265::encoder::loopfilter::LoopFilterCfg;
use oxideav_h265::encoder::pyramid::{encode_pyramid_with, PyramidEncoder};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: encode_pyramid <in.yuv> <WxH> <QP> <GOP> <out.hevc> [recon.yuv] [amp] [deblock] [sao]";
    if args.len() < 5 {
        eprintln!("{usage}");
        std::process::exit(2);
    }
    let (w, h) = args[1]
        .split_once('x')
        .and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?)))
        .expect("WxH");
    let qp: i32 = args[2].parse().expect("QP");
    let gop: usize = args[3].parse().expect("GOP");
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
    let enc = PyramidEncoder::new(w, h, qp, gop)
        .expect("encoder")
        .with_amp(flag("amp"))
        .with_loop_filters(LoopFilterCfg {
            deblocking: flag("deblock"),
            sao_luma: flag("sao"),
            sao_chroma: flag("sao"),
        });
    let out = encode_pyramid_with(enc, &frames).expect("encode");
    std::fs::write(&args[4], &out.stream).expect("write output");
    let is_flag = |a: &str| matches!(a, "amp" | "deblock" | "sao");
    if let Some(recon_path) = args.get(5).filter(|a| !is_flag(a)) {
        let mut recon = Vec::new();
        for r in &out.recon {
            recon.extend_from_slice(&r.y);
            recon.extend_from_slice(&r.cb);
            recon.extend_from_slice(&r.cr);
        }
        std::fs::write(recon_path, &recon).expect("write recon");
    }
    eprintln!(
        "{w}x{h} qp{qp} gop{gop}{}: {} frames -> {} bytes (decode order {:?})",
        if flag("amp") { " (AMP)" } else { "" },
        frames.len(),
        out.stream.len(),
        out.decode_order,
    );
}
