//! Encode raw 4:2:0 frames as a low-delay `IDR, P, P, …` stream.
//!
//! ```text
//! cargo run --example encode_low_delay_p -- in.yuv WxH QP out.hevc [recon.yuv]
//! ```
//!
//! `in.yuv` is a concatenation of planar 8-bit 4:2:0 frames;
//! dimensions must be multiples of 16. The optional `recon.yuv`
//! receives the encoder's own per-frame reconstruction (what a
//! conforming decoder must output).

use oxideav_h265::encoder::inter::{encode_low_delay_p, YuvFrame};

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
    let enc = encode_low_delay_p(&frames, w, h, qp).expect("encode");
    std::fs::write(&args[3], &enc.stream).expect("write output");
    if let Some(recon_path) = args.get(4) {
        let mut recon = Vec::new();
        for r in &enc.recon {
            recon.extend_from_slice(&r.y);
            recon.extend_from_slice(&r.cb);
            recon.extend_from_slice(&r.cr);
        }
        std::fs::write(recon_path, &recon).expect("write recon");
    }
    eprintln!(
        "{w}x{h} qp{qp}: {} frames -> {} bytes",
        frames.len(),
        enc.stream.len()
    );
}
