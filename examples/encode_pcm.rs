//! Encode one raw 4:2:0 frame as a PCM-only IDR Annex B stream.
//!
//! ```text
//! cargo run --example encode_pcm -- in.yuv WxH out.hevc [tiles CxR]
//! ```
//!
//! `in.yuv` is one planar 8-bit 4:2:0 frame (Y then Cb then Cr);
//! dimensions must be multiples of 16. With `tiles CxR` the picture is
//! coded as a single slice segment over a uniform CxR tile grid
//! (`tiles_enabled_flag == 1`, per-tile entry-point offsets).

use oxideav_h265::encoder::pcm::{encode_idr_pcm_au_opts, PcmAuOptions};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: encode_pcm <in.yuv> <WxH> <out.hevc> [tiles <CxR>]";
    let (input, dims, output) = match &args[..] {
        [i, d, o] | [i, d, o, ..] => (i, d, o),
        _ => {
            eprintln!("{usage}");
            std::process::exit(2);
        }
    };
    let parse_pair = |s: &str| -> Option<(usize, usize)> {
        let (a, b) = s.split_once('x')?;
        Some((a.parse().ok()?, b.parse().ok()?))
    };
    let (w, h) = parse_pair(dims).expect("WxH");
    let tiles = match &args[3..] {
        [kw, grid] if kw == "tiles" => {
            let (c, r) = parse_pair(grid).expect("CxR");
            Some((c as u32, r as u32))
        }
        [] => None,
        _ => {
            eprintln!("{usage}");
            std::process::exit(2);
        }
    };
    let data = std::fs::read(input).expect("read input");
    assert_eq!(data.len(), w * h * 3 / 2, "one planar 4:2:0 frame");
    let (y, c) = data.split_at(w * h);
    let (cb, cr) = c.split_at(w * h / 4);
    let au = encode_idr_pcm_au_opts(
        y,
        cb,
        cr,
        w,
        h,
        PcmAuOptions {
            tiles,
            ..PcmAuOptions::default()
        },
    )
    .expect("encode");
    std::fs::write(output, &au).expect("write output");
    eprintln!("{w}x{h} -> {} bytes{}", au.len(), {
        if let Some((c, r)) = tiles {
            format!(" ({c}x{r} tiles)")
        } else {
            String::new()
        }
    });
}
