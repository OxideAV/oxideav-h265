//! Encode one raw 4:2:0 8-bit picture of any size through the registry
//! encoder (the path a HEIF writer takes) and report bytes + wall time.
//!
//! ```text
//! cargo run --release --example encode_still -- in.yuv WxH out.hevc [key=value ...]
//! ```
//!
//! `key=value` pairs are registry codec options (`mode=intra qp=30
//! ctb=64 rdoq=1 sdh=1 tudepth=2 deblock=1 sao=1 still=1 tiles=2x2
//! wpp=1 ...`); the pseudo-option `threads=N` sets the
//! [`oxideav_core::ExecutionContext`] worker budget (tile fan-out).

use std::time::Instant;

use oxideav_core::{CodecParameters, ExecutionContext, Frame, VideoFrame, VideoPlane};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: encode_still <in.yuv> <WxH> <out.hevc> [key=value ...]");
        std::process::exit(2);
    }
    let (w, h) = args[1]
        .split_once('x')
        .and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?)))
        .expect("WxH");
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let data = std::fs::read(&args[0]).expect("read input");
    assert_eq!(data.len(), w * h + 2 * cw * ch, "one planar 4:2:0 picture");
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    let mut threads = 1usize;
    for kv in &args[3..] {
        let (k, v) = kv.split_once('=').expect("key=value");
        if k == "threads" {
            threads = v.parse().expect("threads");
        } else {
            params.options.insert(k, v);
        }
    }
    let mut enc = oxideav_h265::make_encoder(&params).expect("encoder");
    if threads > 1 {
        enc.set_execution_context(&ExecutionContext { threads });
    }
    let (y, c) = data.split_at(w * h);
    let (cb, cr) = c.split_at(cw * ch);
    let frame = Frame::Video(VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w,
                data: y.to_vec(),
            },
            VideoPlane {
                stride: cw,
                data: cb.to_vec(),
            },
            VideoPlane {
                stride: cw,
                data: cr.to_vec(),
            },
        ],
    });
    let t0 = Instant::now();
    enc.send_frame(&frame).expect("encode");
    let pkt = enc.receive_packet().expect("packet");
    let elapsed = t0.elapsed();
    std::fs::write(&args[2], &pkt.data).expect("write output");
    println!(
        "{w}x{h} {} -> {} bytes in {:.3} s",
        args[3..].join(" "),
        pkt.data.len(),
        elapsed.as_secs_f64()
    );
}
