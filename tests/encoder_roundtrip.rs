//! End-to-end round-trip for the HEVC encoder.
//!
//! Encodes a synthesised YUV 4:2:0 frame into an HEVC bitstream (single
//! IDR, residual-coded intra CUs), decodes the result through our own
//! `HevcDecoder`, and checks that the reconstruction is within a
//! reasonable PSNR bound and much smaller than the raw sample budget.

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
    VideoPlane,
};
use oxideav_core::{Decoder, Encoder};

use oxideav_h265::decoder::HevcDecoder;
use oxideav_h265::encoder::HevcEncoder;
use oxideav_h265::nal::{iter_annex_b, NalUnitType};

fn make_gradient_frame(w: u32, h: u32) -> VideoFrame {
    // Horizontal gradient on Y, flat chroma.
    let mut y = vec![0u8; (w * h) as usize];
    for yy in 0..h {
        for xx in 0..w {
            y[(yy * w + xx) as usize] = ((xx * 255) / w.max(1)) as u8;
        }
    }
    let cw = (w / 2) as usize;
    let ch = (h / 2) as usize;
    let cb = vec![128u8; cw * ch];
    let cr = vec![128u8; cw * ch];
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w as usize,
                data: y,
            },
            VideoPlane {
                stride: cw,
                data: cb,
            },
            VideoPlane {
                stride: cw,
                data: cr,
            },
        ],
    }
}

fn encoder_params(w: u32, h: u32) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("h265"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(PixelFormat::Yuv420P);
    p.frame_rate = Some(Rational::new(30, 1));
    p
}

fn encode_one(frame: &VideoFrame, w: u32, h: u32) -> Vec<u8> {
    let params = encoder_params(w, h);
    let mut enc = HevcEncoder::from_params(&params).expect("encoder");
    enc.send_frame(&Frame::Video(frame.clone())).expect("send");
    let pkt = enc.receive_packet().expect("packet");
    pkt.data
}

fn psnr_y(src: &VideoFrame, dec: &VideoFrame, w: u32, h: u32) -> f64 {
    let ss = src.planes[0].stride;
    let ds = dec.planes[0].stride;
    let src_y = &src.planes[0].data;
    let dec_y = &dec.planes[0].data;
    let mut sse: u64 = 0;
    let n = (w * h) as u64;
    for yy in 0..h as usize {
        for xx in 0..w as usize {
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

#[test]
fn encoded_stream_contains_vps_sps_pps_and_idr() {
    let frame = make_gradient_frame(64, 64);
    let bytes = encode_one(&frame, 64, 64);
    let nals: Vec<_> = iter_annex_b(&bytes).collect();
    assert_eq!(nals.len(), 4, "expected VPS+SPS+PPS+IDR NALs");
    assert_eq!(nals[0].header.nal_unit_type, NalUnitType::Vps);
    assert_eq!(nals[1].header.nal_unit_type, NalUnitType::Sps);
    assert_eq!(nals[2].header.nal_unit_type, NalUnitType::Pps);
    assert_eq!(nals[3].header.nal_unit_type, NalUnitType::IdrNLp);
}

#[test]
fn self_roundtrip_achieves_reasonable_psnr() {
    let src = make_gradient_frame(64, 64);
    let bytes = encode_one(&src, 64, 64);

    let mut dec = HevcDecoder::new(CodecId::new("h265"));
    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes.clone());
    dec.send_packet(&pkt).expect("decoder send_packet");

    let frame = match dec.receive_frame() {
        Ok(Frame::Video(v)) => v,
        Ok(_) => panic!("non-video frame"),
        Err(Error::NeedMore) => panic!("needed more data"),
        Err(e) => panic!("decoder error: {e:?}"),
    };
    // Frame dimensions live on the stream's CodecParameters now;
    // sanity-check the luma plane size against what we encoded instead.
    assert_eq!(frame.planes[0].data.len(), frame.planes[0].stride * 64);

    // Should be much smaller than raw-samples (64*64 + 2*32*32 = 6144 bytes).
    // With real compression we expect well under 2k.
    let raw_budget = (64 * 64 + 2 * 32 * 32) as usize;
    assert!(
        bytes.len() < raw_budget,
        "compressed {} bytes >= raw budget {}",
        bytes.len(),
        raw_budget
    );

    let psnr = psnr_y(&src, &frame, 64, 64);
    eprintln!(
        "gradient_64x64: bytes={} raw_budget={} psnr_y={:.2}",
        bytes.len(),
        raw_budget,
        psnr
    );
    assert!(psnr > 30.0, "luma PSNR too low: {psnr:.2} dB");
}

#[test]
fn selftest_128x128_multi_ctu_psnr() {
    let src = make_gradient_frame(128, 128);
    let bytes = encode_one(&src, 128, 128);
    let mut dec = HevcDecoder::new(CodecId::new("h265"));
    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes);
    dec.send_packet(&pkt).expect("decoder send_packet");
    let frame = match dec.receive_frame() {
        Ok(Frame::Video(v)) => v,
        Err(e) => panic!("decoder: {e:?}"),
        _ => panic!("not video"),
    };
    let psnr = psnr_y(&src, &frame, 128, 128);
    eprintln!("gradient_128x128: psnr_y={psnr:.2}");
    assert!(psnr > 30.0, "luma PSNR too low: {psnr:.2} dB");
}

#[test]
fn selftest_128x64_two_ctu_rows() {
    let src = make_gradient_frame(128, 64);
    let bytes = encode_one(&src, 128, 64);

    let mut dec = HevcDecoder::new(CodecId::new("h265"));
    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes);
    dec.send_packet(&pkt).expect("decoder send_packet");
    let frame = match dec.receive_frame() {
        Ok(Frame::Video(v)) => v,
        Err(e) => panic!("decoder: {e:?}"),
        _ => panic!("not video"),
    };
    let psnr = psnr_y(&src, &frame, 128, 64);
    eprintln!("gradient_128x64: psnr_y={psnr:.2}");
    assert!(psnr > 30.0, "luma PSNR too low: {psnr:.2} dB");
}
