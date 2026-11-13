//! Encode raw planar pictures of any size through the registry
//! encoder (one picture: the path a HEIF writer takes; several
//! concatenated pictures: a sequence, e.g. `mode=inter`) and report
//! bytes + wall time.
//!
//! ```text
//! cargo run --release --example encode_still -- in.yuv WxH out.hevc [key=value ...]
//! ```
//!
//! `key=value` pairs are registry codec options (`mode=intra qp=30
//! ctb=64 rdoq=1 sdh=1 tudepth=2 deblock=1 sao=1 still=1 tiles=2x2
//! wpp=1 ...`); the pseudo-option `threads=N` sets the
//! [`oxideav_core::ExecutionContext`] worker budget (tile fan-out) and
//! `pf=<format>` the input layout (`yuv420p` default, `yuv420p10le`,
//! `yuv420p12le`, `yuv422p`, `yuv422p10le`, `yuv422p12le`, `yuv444p`,
//! `yuv444p10le`, `yuv444p12le`, `gray`, `gray10le`, `gray12le`;
//! little-endian 16-bit samples above 8 bits).

use std::time::Instant;

use oxideav_core::{CodecParameters, ExecutionContext, Frame, PixelFormat, VideoFrame, VideoPlane};

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
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    let mut threads = 1usize;
    let mut pf = "yuv420p".to_string();
    for kv in &args[3..] {
        let (k, v) = kv.split_once('=').expect("key=value");
        match k {
            "threads" => threads = v.parse().expect("threads"),
            "pf" => pf = v.to_string(),
            _ => {
                params.options.insert(k, v);
            }
        }
    }
    // (pixel format, bytes per sample, (SubWidthC, SubHeightC), planes)
    let (format, bps, (sw, sh), n_planes) = match pf.as_str() {
        "yuv420p" => (PixelFormat::Yuv420P, 1, (2, 2), 3),
        "yuv420p10le" => (PixelFormat::Yuv420P10Le, 2, (2, 2), 3),
        "yuv420p12le" => (PixelFormat::Yuv420P12Le, 2, (2, 2), 3),
        "yuv422p" => (PixelFormat::Yuv422P, 1, (2, 1), 3),
        "yuv422p10le" => (PixelFormat::Yuv422P10Le, 2, (2, 1), 3),
        "yuv422p12le" => (PixelFormat::Yuv422P12Le, 2, (2, 1), 3),
        "yuv444p" => (PixelFormat::Yuv444P, 1, (1, 1), 3),
        "yuv444p10le" => (PixelFormat::Yuv444P10Le, 2, (1, 1), 3),
        "yuv444p12le" => (PixelFormat::Yuv444P12Le, 2, (1, 1), 3),
        "gray" => (PixelFormat::Gray8, 1, (1, 1), 1),
        "gray10le" => (PixelFormat::Gray10Le, 2, (1, 1), 1),
        "gray12le" => (PixelFormat::Gray12Le, 2, (1, 1), 1),
        other => panic!("unsupported pf {other}"),
    };
    if pf != "yuv420p" {
        params.pixel_format = Some(format);
    }
    let (cw, ch) = (w.div_ceil(sw), h.div_ceil(sh));
    let data = std::fs::read(&args[0]).expect("read input");
    let luma_len = w * h * bps;
    let chroma_len = cw * ch * bps;
    let frame_len = luma_len + (n_planes - 1) * chroma_len;
    assert!(
        !data.is_empty() && data.len().is_multiple_of(frame_len),
        "whole planar {pf} pictures"
    );
    let mut enc = oxideav_h265::make_encoder(&params).expect("encoder");
    if threads > 1 {
        enc.set_execution_context(&ExecutionContext { threads });
    }
    let frames: Vec<Frame> = data
        .chunks_exact(frame_len)
        .enumerate()
        .map(|(i, pic)| {
            let mut planes = vec![VideoPlane {
                stride: w * bps,
                data: pic[..luma_len].to_vec(),
            }];
            for k in 1..n_planes {
                let off = luma_len + (k - 1) * chroma_len;
                planes.push(VideoPlane {
                    stride: cw * bps,
                    data: pic[off..off + chroma_len].to_vec(),
                });
            }
            Frame::Video(VideoFrame {
                pts: Some(i as i64),
                planes,
            })
        })
        .collect();
    let mut out = Vec::new();
    let t0 = Instant::now();
    for frame in &frames {
        enc.send_frame(frame).expect("encode");
        while let Ok(pkt) = enc.receive_packet() {
            out.extend_from_slice(&pkt.data);
        }
    }
    enc.flush().expect("flush");
    while let Ok(pkt) = enc.receive_packet() {
        out.extend_from_slice(&pkt.data);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    std::fs::write(&args[2], &out).expect("write output");
    if frames.len() == 1 {
        println!(
            "{w}x{h} {} -> {} bytes in {elapsed:.3} s",
            args[3..].join(" "),
            out.len(),
        );
    } else {
        println!(
            "{w}x{h} {} -> {} pictures, {} bytes in {elapsed:.3} s ({:.2} pictures per second)",
            args[3..].join(" "),
            frames.len(),
            out.len(),
            frames.len() as f64 / elapsed
        );
    }
}
