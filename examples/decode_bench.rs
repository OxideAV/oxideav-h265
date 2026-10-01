//! Decode an Annex B HEVC byte stream through the registry decoder
//! (`make_decoder`, the path a HEIF / MP4 consumer takes) and report the
//! wall time plus a digest of every output plane.
//!
//! ```text
//! cargo run --release --example decode_bench -- input.hevc [more.hevc ...] [threads=N] [repeat=N]
//! ```
//!
//! Several inputs decode one after the other (a HEIF grid's tiles);
//! the summary line adds their best times up.
//!
//! `threads=N` hands the decoder an `ExecutionContext` budget of `N`
//! workers (serial when absent); `repeat=N` decodes the stream `N`
//! times with a fresh decoder each (the reported time is the best).
//! The digest is a 64-bit FNV-1a over the planes in output order, so
//! two runs are byte-identical iff their digests agree. Peak memory is
//! the job of the caller (`/usr/bin/time -l` on macOS, `-v` on Linux).

use oxideav_core::{CodecParameters, ExecutionContext, Frame, Packet, TimeBase};
use std::time::Instant;

fn fnv1a(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0100_0000_01b3);
    }
}

fn main() {
    let mut inputs = Vec::new();
    let mut threads = 1usize;
    let mut repeat = 1usize;
    for a in std::env::args().skip(1) {
        if let Some(v) = a.strip_prefix("threads=") {
            threads = v.parse().expect("threads=N");
        } else if let Some(v) = a.strip_prefix("repeat=") {
            repeat = v.parse().expect("repeat=N");
        } else {
            inputs.push(a);
        }
    }
    assert!(
        !inputs.is_empty(),
        "usage: decode_bench <in.hevc> [more.hevc ...] [threads=N] [repeat=N]"
    );
    let mut total = 0.0f64;
    for input in &inputs {
        total += bench(input, threads, repeat);
    }
    if inputs.len() > 1 {
        println!(
            "{} inputs: {:.3} s total (best of {repeat} each)",
            inputs.len(),
            total
        );
    }
}

fn bench(input: &str, threads: usize, repeat: usize) -> f64 {
    let data = std::fs::read(input).expect("read input");
    let params = CodecParameters::video("h265".into());
    let mut best = f64::MAX;
    let mut digest = 0u64;
    let mut frames = 0usize;
    let mut dims = (0usize, 0usize);
    for _ in 0..repeat {
        let t0 = Instant::now();
        let mut dec = oxideav_h265::make_decoder(&params).expect("make_decoder");
        if threads > 1 {
            dec.set_execution_context(&ExecutionContext::with_threads(threads));
        }
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), data.clone()))
            .expect("send_packet");
        dec.flush().expect("flush");
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        frames = 0;
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(v)) => {
                    frames += 1;
                    for p in &v.planes {
                        fnv1a(&mut h, &p.data);
                    }
                    if let Some(p) = v.planes.first() {
                        dims = (p.stride, p.data.len() / p.stride.max(1));
                    }
                }
                Ok(_) => {}
                Err(oxideav_core::Error::NeedMore) => continue,
                Err(oxideav_core::Error::Eof) => break,
                Err(e) => panic!("receive_frame: {e}"),
            }
        }
        drop(dec);
        best = best.min(t0.elapsed().as_secs_f64());
        digest = h;
    }
    println!(
        "{input}: {frames} frame(s) {}x{} (plane 0 bytes), threads={threads}, best {:.3} s, digest {digest:016x}",
        dims.0, dims.1, best
    );
    best
}
