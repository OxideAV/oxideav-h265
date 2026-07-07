//! Encode raw 4:2:0 frames as a low-delay `IDR, P, P, …` stream.
//!
//! ```text
//! cargo run --example encode_low_delay_p -- in.yuv WxH QP out.hevc [recon.yuv] [b]
//! ```
//!
//! A trailing `b` argument codes the non-IDR frames as low-delay B
//! slices (both reference lists resolving to the previous picture).
//!
//! `in.yuv` is a concatenation of planar 8-bit 4:2:0 frames;
//! dimensions must be multiples of 16. The optional `recon.yuv`
//! receives the encoder's own per-frame reconstruction (what a
//! conforming decoder must output).

use oxideav_h265::encoder::inter::{LowDelayPEncoder, YuvFrame};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: encode_low_delay_p <in.yuv> <WxH> <QP> <out.hevc> [recon.yuv]";
    if args.len() < 4 {
        eprintln!("{usage}");
        std::process::exit(2);
    }
    let (w, h) = args[1]
        .split_once('x')
        .and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?)))
        .expect("WxH");
    let qp: i32 = args[2].parse().expect("QP");
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
    let b_slices = args.iter().skip(4).any(|a| a == "b");
    let mut enc = LowDelayPEncoder::new(w, h, qp, 0)
        .expect("encoder")
        .with_b_slices(b_slices);
    let mut stream = Vec::new();
    let mut recon = Vec::new();
    for f in &frames {
        let out = enc.encode_frame(f).expect("encode");
        stream.extend_from_slice(&out.au);
        recon.extend_from_slice(&out.recon.y);
        recon.extend_from_slice(&out.recon.cb);
        recon.extend_from_slice(&out.recon.cr);
    }
    std::fs::write(&args[3], &stream).expect("write output");
    if let Some(recon_path) = args.get(4).filter(|a| a.as_str() != "b") {
        std::fs::write(recon_path, &recon).expect("write recon");
    }
    eprintln!(
        "{w}x{h} qp{qp}{}: {} frames -> {} bytes",
        if b_slices { " (B slices)" } else { "" },
        frames.len(),
        stream.len()
    );
}
