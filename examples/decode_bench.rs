//! Decode an Annex B HEVC byte stream through the registry decoder
//! (`make_decoder`, the path a HEIF / MP4 consumer takes) and report the
//! wall time plus a digest of every output plane.
//!
//! ```text
//! cargo run --release --example decode_bench -- input.hevc [more.hevc ...] [threads=N] [repeat=N] [au] [out=PATH]
//! ```
//!
//! Several inputs decode one after the other (a HEIF grid's tiles);
//! the summary line adds their best times up.
//!
//! `threads=N` hands the decoder an `ExecutionContext` budget of `N`
//! workers (serial when absent; `threads=auto` = the host's available
//! parallelism); `repeat=N` decodes the stream `N`
//! times with a fresh decoder each (the reported time is the best).
//! The digest is a 64-bit FNV-1a (over 8-byte words) of the planes in output order, so
//! two runs are byte-identical iff their digests agree. `au` feeds the
//! stream one access unit per packet (the way a demuxer hands it over)
//! instead of in one packet, draining frames as they come; `out=PATH`
//! writes the output planes as raw YUV (`out=-` for stdout, e.g. piped
//! into a black-box `framemd5` of the same geometry). Peak memory is
//! the job of the caller (`/usr/bin/time -l` on macOS, `-v` on Linux).

use oxideav_core::{CodecParameters, ExecutionContext, Frame, Packet, TimeBase};
use std::io::Write;
use std::time::Instant;

/// FNV-1a over 8-byte little-endian words (the tail zero-padded), so
/// the digest costs little next to the decode it checks.
fn fnv1a(h: &mut u64, bytes: &[u8]) {
    let (words, rest) = bytes.as_chunks::<8>();
    for w in words {
        *h ^= u64::from_le_bytes(*w);
        *h = h.wrapping_mul(0x0100_0000_01b3);
    }
    if !rest.is_empty() {
        let mut last = [0u8; 8];
        last[..rest.len()].copy_from_slice(rest);
        *h ^= u64::from_le_bytes(last);
        *h = h.wrapping_mul(0x0100_0000_01b3);
    }
}

fn main() {
    let mut inputs = Vec::new();
    let mut threads = 1usize;
    let mut repeat = 1usize;
    let mut opts = Opts::default();
    for a in std::env::args().skip(1) {
        if let Some(v) = a.strip_prefix("threads=") {
            threads = if v == "auto" {
                std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
            } else {
                v.parse().expect("threads=N")
            };
        } else if let Some(v) = a.strip_prefix("repeat=") {
            repeat = v.parse().expect("repeat=N");
        } else if a == "au" {
            opts.per_au = true;
        } else if let Some(v) = a.strip_prefix("out=") {
            opts.out = Some(v.to_string());
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
        total += bench(input, threads, repeat, &opts);
    }
    if inputs.len() > 1 {
        eprintln!(
            "{} inputs: {:.3} s total (best of {repeat} each)",
            inputs.len(),
            total
        );
    }
}

#[derive(Default)]
struct Opts {
    per_au: bool,
    out: Option<String>,
}

/// Split an Annex B stream into access units: a new one starts at an
/// AUD, at a VPS / SPS / PPS / prefix SEI following a VCL NAL unit, or
/// at a VCL NAL unit with `first_slice_segment_in_pic_flag` following
/// a VCL NAL unit (§7.4.2.4.4).
fn split_access_units(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            let begin = if i > 0 && data[i - 1] == 0 { i - 1 } else { i };
            starts.push((begin, i + 3));
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut aus = Vec::new();
    let mut au_start = None;
    let mut prev_vcl = false;
    for &(begin, hdr) in &starts {
        let Some(&b0) = data.get(hdr) else { continue };
        let t = (b0 >> 1) & 0x3f;
        let vcl = t < 32;
        let first_slice = vcl && data.get(hdr + 2).is_some_and(|b| b & 0x80 != 0);
        let opens = t == 35 || (prev_vcl && (matches!(t, 32..=34 | 39) || first_slice));
        if opens {
            if let Some(s) = au_start {
                aus.push(&data[s..begin]);
            }
            au_start = Some(begin);
        } else if au_start.is_none() {
            au_start = Some(begin);
        }
        if vcl || opens {
            prev_vcl = vcl;
        }
    }
    if let Some(s) = au_start {
        aus.push(&data[s..]);
    }
    aus
}

fn bench(input: &str, threads: usize, repeat: usize, opts: &Opts) -> f64 {
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
        let mut out: Option<Box<dyn Write>> = opts.out.as_deref().map(|p| -> Box<dyn Write> {
            if p == "-" {
                Box::new(std::io::BufWriter::new(std::io::stdout().lock()))
            } else {
                Box::new(std::io::BufWriter::new(
                    std::fs::File::create(p).expect("create out"),
                ))
            }
        });
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        frames = 0;
        let mut take = |dec: &mut Box<dyn oxideav_core::Decoder>| -> bool {
            loop {
                match dec.receive_frame() {
                    Ok(Frame::Video(v)) => {
                        frames += 1;
                        for p in &v.planes {
                            fnv1a(&mut h, &p.data);
                            if let Some(o) = out.as_mut() {
                                o.write_all(&p.data).expect("write out");
                            }
                        }
                        if let Some(p) = v.planes.first() {
                            dims = (p.stride, p.data.len() / p.stride.max(1));
                        }
                    }
                    Ok(_) => {}
                    Err(oxideav_core::Error::NeedMore) => return false,
                    Err(oxideav_core::Error::Eof) => return true,
                    Err(e) => panic!("receive_frame: {e}"),
                }
            }
        };
        if opts.per_au {
            for (i, au) in split_access_units(&data).into_iter().enumerate() {
                dec.send_packet(
                    &Packet::new(0, TimeBase::new(1, 1), au.to_vec()).with_pts(i as i64),
                )
                .expect("send_packet");
                take(&mut dec);
            }
        } else {
            dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), data.clone()))
                .expect("send_packet");
        }
        dec.flush().expect("flush");
        while !take(&mut dec) {}
        drop(dec);
        best = best.min(t0.elapsed().as_secs_f64());
        digest = h;
    }
    eprintln!(
        "{input}: {frames} frame(s) {}x{} (plane 0 bytes), threads={threads}, best {:.3} s, digest {digest:016x}",
        dims.0, dims.1, best
    );
    best
}
