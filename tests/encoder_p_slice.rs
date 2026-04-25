//! P-slice round-trip: encode two frames (I + P), decode both through our
//! own decoder, and check that each reconstruction is close enough to the
//! source that the encoder's motion / residual pipeline agrees with the
//! decoder's.

use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
    VideoPlane,
};
use oxideav_core::{Decoder, Encoder};

use oxideav_h265::decoder::HevcDecoder;
use oxideav_h265::encoder::HevcEncoder;

fn make_gradient(w: u32, h: u32, offset: i32) -> VideoFrame {
    let mut y = vec![0u8; (w * h) as usize];
    for yy in 0..h {
        for xx in 0..w {
            // Clamp-shift (not wrap): lets the encoder's ±8 SAD search
            // reconstruct the shift via edge-clamped MC without needing to
            // synthesise wrap-around samples that don't exist in the ref.
            let src_x = (xx as i32 + offset).clamp(0, w as i32 - 1) as u32;
            let v = (src_x * 255) / w.max(1);
            y[(yy * w + xx) as usize] = v as u8;
        }
    }
    let cw = (w / 2) as usize;
    let ch = (h / 2) as usize;
    VideoFrame {
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
    }
}

fn psnr_y(src: &VideoFrame, dec: &VideoFrame) -> f64 {
    let ss = src.planes[0].stride;
    let ds = dec.planes[0].stride;
    let src_y = &src.planes[0].data;
    let dec_y = &dec.planes[0].data;
    let mut sse: u64 = 0;
    let n = (src.width * src.height) as u64;
    for yy in 0..src.height as usize {
        for xx in 0..src.width as usize {
            let s = src_y[yy * ss + xx] as i64;
            let d = dec_y[yy * ds + xx] as i64;
            let diff = s - d;
            sse += (diff * diff) as u64;
        }
    }
    if sse == 0 {
        return 99.0;
    }
    let mse = sse as f64 / n as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

fn encoder_params(w: u32, h: u32) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("h265"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(PixelFormat::Yuv420P);
    p.frame_rate = Some(Rational::new(30, 1));
    p
}

#[test]
fn i_plus_p_roundtrip_through_our_decoder() {
    let w = 64;
    let h = 64;
    let f0 = make_gradient(w, h, 0);
    let f1 = make_gradient(w, h, 1); // Shift by 1 pel.
    let f2 = make_gradient(w, h, 2);

    let mut enc = HevcEncoder::from_params(&encoder_params(w, h)).unwrap();
    enc.send_frame(&Frame::Video(f0.clone())).unwrap();
    enc.send_frame(&Frame::Video(f1.clone())).unwrap();
    enc.send_frame(&Frame::Video(f2.clone())).unwrap();

    let mut all = Vec::new();
    for _ in 0..3 {
        let pkt = enc.receive_packet().unwrap();
        all.extend_from_slice(&pkt.data);
    }

    let mut dec = HevcDecoder::new(CodecId::new("h265"));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), all))
        .expect("send_packet");

    let mut frames = Vec::new();
    for _ in 0..3 {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => frames.push(v),
            Err(e) => panic!("decoder: {e:?}"),
            _ => panic!("non-video frame"),
        }
    }

    let p0 = psnr_y(&f0, &frames[0]);
    let p1 = psnr_y(&f1, &frames[1]);
    let p2 = psnr_y(&f2, &frames[2]);
    eprintln!("IPP psnr: f0={p0:.2} f1={p1:.2} f2={p2:.2}");
    assert!(p0 > 30.0, "I-frame PSNR too low: {p0:.2}");
    assert!(p1 > 25.0, "P-frame 1 PSNR too low: {p1:.2}");
    assert!(p2 > 25.0, "P-frame 2 PSNR too low: {p2:.2}");
}
