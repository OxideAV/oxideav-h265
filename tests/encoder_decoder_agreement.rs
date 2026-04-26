//! Verify that the decoder produces the same reconstruction the encoder
//! locally believes it should. This exercises the CABAC emit path against
//! our own decoder's CABAC consume path for consistent coefficient
//! interpretation.

use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
    VideoPlane,
};
use oxideav_core::{Decoder, Encoder};

const FRAME_W: u32 = 32;
const FRAME_H: u32 = 32;

use oxideav_h265::decoder::HevcDecoder;
use oxideav_h265::encoder::HevcEncoder;

fn make_block_frame(w: u32, h: u32) -> VideoFrame {
    // Diagonal stripes — challenging for DC/planar alone so multiple
    // modes will get tried.
    let mut y = vec![0u8; (w * h) as usize];
    for yy in 0..h {
        for xx in 0..w {
            y[(yy * w + xx) as usize] = (((xx + yy) * 255) / (w + h).max(1)) as u8;
        }
    }
    let cw = (w / 2) as usize;
    let ch = (h / 2) as usize;
    VideoFrame {
        pts: Some(0),
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
    }
}

#[test]
fn decoder_reconstruction_has_reasonable_psnr() {
    let src = make_block_frame(FRAME_W, FRAME_H);
    let mut params = CodecParameters::video(CodecId::new("h265"));
    params.width = Some(FRAME_W);
    params.height = Some(FRAME_H);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(30, 1));
    let mut enc = HevcEncoder::from_params(&params).unwrap();
    enc.send_frame(&Frame::Video(src.clone())).unwrap();
    let pkt = enc.receive_packet().unwrap();

    let mut dec = HevcDecoder::new(CodecId::new("h265"));
    let dpkt = Packet::new(0, TimeBase::new(1, 30), pkt.data);
    dec.send_packet(&dpkt).unwrap();
    let got = match dec.receive_frame() {
        Ok(Frame::Video(v)) => v,
        e => panic!("{:?}", e),
    };

    let mut sse = 0u64;
    let n = (FRAME_W * FRAME_H) as u64;
    for yy in 0..FRAME_H as usize {
        for xx in 0..FRAME_W as usize {
            let s = src.planes[0].data[yy * src.planes[0].stride + xx] as i64;
            let d = got.planes[0].data[yy * got.planes[0].stride + xx] as i64;
            sse += ((s - d) * (s - d)) as u64;
        }
    }
    let mse = sse as f64 / n as f64;
    let psnr = 10.0 * (255.0f64 * 255.0 / mse).log10();
    eprintln!("decoder_agreement 32x32 stripes psnr={psnr:.2}");
    assert!(psnr > 30.0);
}
