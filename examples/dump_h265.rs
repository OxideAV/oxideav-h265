//! Tiny helper: write a one-frame 64×64 HEVC file to stdout for inspection.
//!
//! Usage: `cargo run --example dump_h265 > out.h265`

use oxideav_core::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Frame, PixelFormat, Rational, TimeBase, VideoFrame, VideoPlane,
};
use oxideav_h265::encoder::HevcEncoder;

use std::io::Write;

fn main() {
    let w = 64u32;
    let h = 64u32;
    let mut y = vec![0u8; (w * h) as usize];
    for yy in 0..h {
        for xx in 0..w {
            y[(yy * w + xx) as usize] = ((xx * 255) / w.max(1)) as u8;
        }
    }
    let cw = (w / 2) as usize;
    let ch = (h / 2) as usize;
    let frame = VideoFrame {
        format: PixelFormat::Yuv420P,
        width: w,
        height: h,
        pts: Some(0),
        time_base: TimeBase::new(1, 30),
        planes: vec![
            VideoPlane {
                stride: w as usize,
                data: y,
            },
            VideoPlane {
                stride: cw,
                data: vec![128u8; cw * ch],
            },
            VideoPlane {
                stride: cw,
                data: vec![128u8; cw * ch],
            },
        ],
    };
    let mut params = CodecParameters::video(CodecId::new("h265"));
    params.width = Some(w);
    params.height = Some(h);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(30, 1));
    let mut enc = HevcEncoder::from_params(&params).unwrap();
    enc.send_frame(&Frame::Video(frame)).unwrap();
    let pkt = enc.receive_packet().unwrap();
    std::io::stdout().write_all(&pkt.data).unwrap();
}
