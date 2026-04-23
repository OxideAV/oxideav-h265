//! End-to-end round-trip for the HEVC encoder MVP.
//!
//! Encodes a synthesised 64×64 YUV 4:2:0 frame into an HEVC bitstream
//! (single IDR, PCM CUs) and decodes the result back through our own
//! `HevcDecoder`. The reconstructed pixels should match the input
//! exactly since PCM CUs carry raw 8-bit samples.

use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
    VideoPlane,
};

use oxideav_h265::decoder::HevcDecoder;
use oxideav_h265::encoder::HevcEncoder;
use oxideav_h265::nal::{iter_annex_b, NalUnitType};

fn make_test_frame(w: u32, h: u32) -> VideoFrame {
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

fn encode_one(frame: &VideoFrame) -> Vec<u8> {
    let params = encoder_params(frame.width, frame.height);
    let mut enc = HevcEncoder::from_params(&params).expect("encoder");
    enc.send_frame(&Frame::Video(frame.clone())).expect("send");
    let pkt = enc.receive_packet().expect("packet");
    pkt.data
}

#[test]
fn encoded_stream_contains_vps_sps_pps_and_idr() {
    let frame = make_test_frame(64, 64);
    let bytes = encode_one(&frame);
    let nals: Vec<_> = iter_annex_b(&bytes).collect();
    assert_eq!(nals.len(), 4, "expected VPS+SPS+PPS+IDR NALs");
    assert_eq!(nals[0].header.nal_unit_type, NalUnitType::Vps);
    assert_eq!(nals[1].header.nal_unit_type, NalUnitType::Sps);
    assert_eq!(nals[2].header.nal_unit_type, NalUnitType::Pps);
    assert_eq!(nals[3].header.nal_unit_type, NalUnitType::IdrNLp);
}

#[test]
fn self_roundtrip_reconstructs_frame() {
    let src = make_test_frame(64, 64);
    let bytes = encode_one(&src);

    let mut dec = HevcDecoder::new(CodecId::new("h265"));
    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes);
    dec.send_packet(&pkt).expect("decoder send_packet");

    let frame = loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => break v,
            Ok(_) => panic!("non-video frame"),
            Err(Error::NeedMore) => panic!("needed more data"),
            Err(e) => panic!("decoder error: {e:?}"),
        }
    };
    assert_eq!(frame.width, 64);
    assert_eq!(frame.height, 64);
    assert_eq!(frame.format, PixelFormat::Yuv420P);

    // Exact match on Y/Cb/Cr because PCM is lossless.
    let src_y = &src.planes[0].data;
    let dec_y = &frame.planes[0].data;
    let src_stride = src.planes[0].stride;
    let dec_stride = frame.planes[0].stride;
    let mut max_err_y = 0i32;
    let mut sse_y: u64 = 0;
    for yy in 0..64usize {
        for xx in 0..64usize {
            let s = src_y[yy * src_stride + xx] as i32;
            let d = dec_y[yy * dec_stride + xx] as i32;
            let diff = (s - d).abs();
            if diff > max_err_y {
                max_err_y = diff;
            }
            sse_y += (diff * diff) as u64;
        }
    }
    assert_eq!(max_err_y, 0, "luma must match exactly for PCM (sse={sse_y})");

    // Chroma — flat 128 in, flat 128 out.
    for plane_idx in 1..=2 {
        let sp = &src.planes[plane_idx].data;
        let dp = &frame.planes[plane_idx].data;
        let ss = src.planes[plane_idx].stride;
        let ds = frame.planes[plane_idx].stride;
        for yy in 0..32usize {
            for xx in 0..32usize {
                let s = sp[yy * ss + xx];
                let d = dp[yy * ds + xx];
                assert_eq!(s, d, "chroma plane {plane_idx} at ({xx},{yy})");
            }
        }
    }
}

fn check_luma_exact(src: &VideoFrame, dec: &VideoFrame) {
    assert_eq!(src.width, dec.width);
    assert_eq!(src.height, dec.height);
    let src_y = &src.planes[0].data;
    let dec_y = &dec.planes[0].data;
    let ss = src.planes[0].stride;
    let ds = dec.planes[0].stride;
    for yy in 0..src.height as usize {
        for xx in 0..src.width as usize {
            let s = src_y[yy * ss + xx];
            let d = dec_y[yy * ds + xx];
            assert_eq!(s, d, "luma mismatch at ({xx},{yy})");
        }
    }
}

#[test]
fn selftest_128x128_four_ctus() {
    let src = make_test_frame(128, 128);
    let bytes = encode_one(&src);
    let mut dec = HevcDecoder::new(CodecId::new("h265"));
    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes);
    dec.send_packet(&pkt).expect("decoder send_packet");
    let frame = match dec.receive_frame() {
        Ok(Frame::Video(v)) => v,
        Err(e) => panic!("decoder: {e:?}"),
        _ => panic!("not video"),
    };
    check_luma_exact(&src, &frame);
}

#[test]
fn selftest_128x64_two_ctus() {
    // Two 64×64 CTUs side-by-side — exercises the multi-CTU CABAC reinit
    // and PCM alignment handling.
    let src = make_test_frame(128, 64);
    let bytes = encode_one(&src);

    let mut dec = HevcDecoder::new(CodecId::new("h265"));
    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes);
    dec.send_packet(&pkt).expect("decoder send_packet");
    let frame = match dec.receive_frame() {
        Ok(Frame::Video(v)) => v,
        Err(e) => panic!("decoder: {e:?}"),
        _ => panic!("not video"),
    };

    let src_y = &src.planes[0].data;
    let dec_y = &frame.planes[0].data;
    let src_stride = src.planes[0].stride;
    let dec_stride = frame.planes[0].stride;
    for yy in 0..64usize {
        for xx in 0..128usize {
            let s = src_y[yy * src_stride + xx];
            let d = dec_y[yy * dec_stride + xx];
            assert_eq!(s, d, "luma mismatch at ({xx},{yy})");
        }
    }
}
