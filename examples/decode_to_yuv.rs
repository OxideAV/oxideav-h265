//! Decode an HEVC Annex-B file via the in-tree decoder and write the
//! 8-bit planar YUV to stdout. Diagnostic helper.
//!
//! Usage: `cargo run --example decode_to_yuv -- input.hevc > out.yuv`

use std::io::Write;

use oxideav_core::Decoder;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h265::decoder::HevcDecoder;
use oxideav_h265::CODEC_ID_STR;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: decode_to_yuv <input.hevc>");
    let data = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("failed to read {path}: {e}");
        std::process::exit(1);
    });
    let mut dec = HevcDecoder::new(CodecId::new(CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 25), data);
    if let Err(e) = dec.send_packet(&pkt) {
        eprintln!("send_packet failed: {e:?}");
        std::process::exit(1);
    }
    let mut stdout = std::io::stdout().lock();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => {
                let h_y = vf.planes[0].data.len() / vf.planes[0].stride;
                let w_y = vf.planes[0].stride;
                let w_c = vf.planes[1].stride;
                let h_c = vf.planes[1].data.len() / w_c;
                // luma
                for y in 0..h_y {
                    let off = y * vf.planes[0].stride;
                    stdout
                        .write_all(&vf.planes[0].data[off..off + w_y])
                        .unwrap();
                }
                // chroma cb
                for y in 0..h_c {
                    let off = y * vf.planes[1].stride;
                    stdout
                        .write_all(&vf.planes[1].data[off..off + w_c])
                        .unwrap();
                }
                // chroma cr
                for y in 0..h_c {
                    let off = y * vf.planes[2].stride;
                    stdout
                        .write_all(&vf.planes[2].data[off..off + w_c])
                        .unwrap();
                }
            }
            Ok(_) => break,
            Err(oxideav_core::Error::NeedMore) => break,
            Err(e) => {
                eprintln!("receive_frame error: {e:?}");
                break;
            }
        }
    }
}
