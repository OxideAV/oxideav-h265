//! Decode an Annex B HEVC byte stream to planar YUV on stdout.
//!
//! ```text
//! cargo run --example decode_annexb -- input.hevc output.yuv
//! ```
//!
//! Frames are written in output order as packed planar samples
//! (8-bit streams: one byte per sample; higher bit depths:
//! little-endian 16-bit per sample).

use std::io::Write;

fn main() {
    let mut args = std::env::args().skip(1);
    let input = args
        .next()
        .expect("usage: decode_annexb <in.hevc> [out.yuv]");
    let output = args.next();
    let data = std::fs::read(&input).expect("read input");
    let frames = match oxideav_h265::decode_annexb_sequence(&data) {
        Ok(frames) => frames,
        Err(e) => {
            eprintln!("decode error: {e}");
            std::process::exit(1);
        }
    };
    let mut out: Box<dyn Write> = match &output {
        Some(path) => Box::new(std::fs::File::create(path).expect("create output")),
        None => Box::new(std::io::sink()),
    };
    for frame in &frames {
        eprintln!(
            "frame cvs={} poc={} output={} {}x{}",
            frame.cvs_index,
            frame.poc,
            frame.output,
            frame.picture.width_luma(),
            frame.picture.height_luma(),
        );
        if !frame.output {
            continue;
        }
        let planar = frame
            .picture
            .to_planar_u8()
            .unwrap_or_else(|| frame.picture.to_planar_le16());
        out.write_all(&planar).expect("write output");
    }
    eprintln!("{} frame(s) decoded", frames.len());
}
