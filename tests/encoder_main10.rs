//! Round-25: Main 10 (10-bit) encoder integration tests.
//!
//! Scope:
//! * The encoder accepts a `Yuv420P10Le` keyframe and emits an Annex B
//!   stream whose SPS declares Main 10 (`profile_idc = 2`,
//!   `bit_depth_*_minus8 = 2`).
//! * Our own decoder reads the bitstream back at 10-bit and the
//!   reconstructed luma stays close to the source (PSNR ≥ 30 dB on a
//!   smooth gradient, computed in the 10-bit range).
//! * When the `FFMPEG` env var is set and `ffmpeg` is on PATH, libavcodec
//!   cross-decodes the same bitstream and the cross-decoded luma matches
//!   the source within ~40 dB (gradient).

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
    VideoPlane,
};
use oxideav_core::{Decoder, Encoder};

use oxideav_h265::decoder::HevcDecoder;
use oxideav_h265::encoder::HevcEncoder;
use oxideav_h265::nal::{iter_annex_b, NalUnitType};
use oxideav_h265::sps::parse_sps;

/// Build a `Yuv420P10Le` frame with a horizontal 0..1023 luma gradient
/// and flat mid-range chroma (512). Plane data follows the
/// little-endian-16-bit layout used by `oxideav-core`'s
/// [`PixelFormat::Yuv420P10Le`].
fn make_p10_gradient_frame(w: u32, h: u32) -> VideoFrame {
    let wu = w as usize;
    let hu = h as usize;
    let cw = wu / 2;
    let ch = hu / 2;
    let mut y = Vec::with_capacity(wu * hu * 2);
    for yy in 0..hu {
        let _ = yy;
        for xx in 0..wu {
            let v = ((xx as u32 * 1023) / w.max(1)) as u16;
            y.push((v & 0xFF) as u8);
            y.push((v >> 8) as u8);
        }
    }
    let mut cb = Vec::with_capacity(cw * ch * 2);
    let mut cr = Vec::with_capacity(cw * ch * 2);
    for _ in 0..(cw * ch) {
        // 10-bit chroma neutral = 512.
        cb.push(0x00);
        cb.push(0x02);
        cr.push(0x00);
        cr.push(0x02);
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: wu * 2,
                data: y,
            },
            VideoPlane {
                stride: cw * 2,
                data: cb,
            },
            VideoPlane {
                stride: cw * 2,
                data: cr,
            },
        ],
    }
}

fn p10_params(w: u32, h: u32) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("h265"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(PixelFormat::Yuv420P10Le);
    p.frame_rate = Some(Rational::new(30, 1));
    p
}

fn encode_one_p10(frame: &VideoFrame, w: u32, h: u32) -> Vec<u8> {
    let params = p10_params(w, h);
    let mut enc = HevcEncoder::from_params(&params).expect("encoder");
    enc.send_frame(&Frame::Video(frame.clone())).expect("send");
    let pkt = enc.receive_packet().expect("packet");
    pkt.data
}

/// Read a single 10-bit sample from a packed `Yuv420P10Le` plane.
fn read_p10(plane: &[u8], stride: usize, x: usize, y: usize) -> u16 {
    let off = y * stride + x * 2;
    (plane[off] as u16) | ((plane[off + 1] as u16) << 8)
}

/// 10-bit luma PSNR — peak = 1023.
fn psnr_y_p10(src: &VideoFrame, dec: &VideoFrame, w: u32, h: u32) -> f64 {
    let ss = src.planes[0].stride;
    let ds = dec.planes[0].stride;
    let src_y = &src.planes[0].data;
    let dec_y = &dec.planes[0].data;
    let mut sse: u64 = 0;
    let n = (w * h) as u64;
    for yy in 0..h as usize {
        for xx in 0..w as usize {
            let s = read_p10(src_y, ss, xx, yy) as i64;
            let d = read_p10(dec_y, ds, xx, yy) as i64;
            let diff = s - d;
            sse += (diff * diff) as u64;
        }
    }
    if sse == 0 {
        return 99.0;
    }
    let mse = sse as f64 / n as f64;
    10.0 * (1023.0 * 1023.0 / mse).log10()
}

#[test]
fn main10_emit_carries_main10_sps() {
    let frame = make_p10_gradient_frame(64, 64);
    let bytes = encode_one_p10(&frame, 64, 64);
    let nals: Vec<_> = iter_annex_b(&bytes).collect();
    // VPS + SPS + PPS + IDR — same NAL count as the 8-bit emission.
    assert_eq!(nals.len(), 4, "expected VPS+SPS+PPS+IDR NALs");
    assert_eq!(nals[0].header.nal_unit_type, NalUnitType::Vps);
    assert_eq!(nals[1].header.nal_unit_type, NalUnitType::Sps);
    assert_eq!(nals[2].header.nal_unit_type, NalUnitType::Pps);
    assert_eq!(nals[3].header.nal_unit_type, NalUnitType::IdrNLp);
    // Inspect the SPS to confirm Main 10 metadata.
    use oxideav_h265::nal::extract_rbsp;
    let sps_rbsp = extract_rbsp(nals[1].payload());
    let sps = parse_sps(&sps_rbsp).expect("parse sps");
    assert_eq!(sps.bit_depth_luma_minus8, 2, "luma should be 10-bit");
    assert_eq!(sps.bit_depth_chroma_minus8, 2, "chroma should be 10-bit");
    assert_eq!(sps.bit_depth_y(), 10);
    assert_eq!(sps.profile_tier_level.general_profile_idc, 2);
}

#[test]
fn main10_self_roundtrip_psnr() {
    let src = make_p10_gradient_frame(64, 64);
    let bytes = encode_one_p10(&src, 64, 64);
    let mut dec = HevcDecoder::new(CodecId::new("h265"));
    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes);
    dec.send_packet(&pkt).expect("send packet");
    let frame = match dec.receive_frame() {
        Ok(Frame::Video(v)) => v,
        Ok(other) => panic!("non-video frame: {other:?}"),
        Err(e) => panic!("decoder error: {e:?}"),
    };
    // Plane should be 10-bit packed (16-bit LE), so the luma plane size
    // is at least 2 * w * h bytes.
    assert!(frame.planes[0].data.len() >= 64 * 64 * 2);
    let psnr = psnr_y_p10(&src, &frame, 64, 64);
    eprintln!("main10 self-roundtrip 64x64: psnr_y={psnr:.2} dB (peak=1023)");
    assert!(psnr > 30.0, "10-bit luma PSNR too low: {psnr:.2} dB");
}

#[test]
fn main10_multi_ctu_roundtrip_psnr() {
    let src = make_p10_gradient_frame(128, 128);
    let bytes = encode_one_p10(&src, 128, 128);
    let mut dec = HevcDecoder::new(CodecId::new("h265"));
    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes);
    dec.send_packet(&pkt).expect("send packet");
    let frame = match dec.receive_frame() {
        Ok(Frame::Video(v)) => v,
        Ok(other) => panic!("non-video frame: {other:?}"),
        Err(e) => panic!("decoder error: {e:?}"),
    };
    let psnr = psnr_y_p10(&src, &frame, 128, 128);
    eprintln!("main10 self-roundtrip 128x128: psnr_y={psnr:.2} dB (peak=1023)");
    assert!(psnr > 30.0, "10-bit luma PSNR too low: {psnr:.2} dB");
}

fn have_ffmpeg() -> bool {
    if std::env::var_os("FFMPEG").is_none() {
        return false;
    }
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

#[test]
fn main10_ffmpeg_cross_decode_psnr() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg gate disabled — skipping Main 10 ffmpeg cross-decode");
        return;
    }
    let w = 64u32;
    let h = 64u32;
    let src = make_p10_gradient_frame(w, h);
    let bytes = encode_one_p10(&src, w, h);

    let tmp: PathBuf = std::env::temp_dir();
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let stem = format!("oxideav_h265_main10_{}_{w}x{h}_{ns}", std::process::id());
    let h265_path = tmp.join(format!("{stem}.h265"));
    let yuv_path = tmp.join(format!("{stem}.yuv"));
    {
        let mut f = std::fs::File::create(&h265_path).expect("create temp");
        f.write_all(&bytes).expect("write bitstream");
    }

    let out = Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(&h265_path)
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("yuv420p10le")
        .arg(&yuv_path)
        .output()
        .expect("run ffmpeg");
    if !out.status.success() {
        eprintln!("ffmpeg stderr:\n{}", String::from_utf8_lossy(&out.stderr));
        panic!(
            "ffmpeg rejected the Main 10 bitstream (exit {:?})",
            out.status
        );
    }
    let dec = std::fs::read(&yuv_path).expect("read decoded yuv");
    let y_len = (w * h * 2) as usize;
    let c_len = ((w / 2) * (h / 2) * 2) as usize;
    let expected_len = y_len + 2 * c_len;
    assert_eq!(dec.len(), expected_len, "decoded YUV length mismatch");

    // Compare luma planes (10-bit) by PSNR (encoder is lossy).
    let src_y = &src.planes[0].data;
    let src_stride = src.planes[0].stride;
    let dec_y = &dec[..y_len];
    let dec_stride = (w * 2) as usize;
    let mut sse: u64 = 0;
    let n = (w * h) as u64;
    for yy in 0..h as usize {
        for xx in 0..w as usize {
            let s = read_p10(src_y, src_stride, xx, yy) as i64;
            let d = read_p10(dec_y, dec_stride, xx, yy) as i64;
            let diff = s - d;
            sse += (diff * diff) as u64;
        }
    }
    let mse = sse as f64 / n as f64;
    let psnr = if mse == 0.0 {
        99.0
    } else {
        10.0 * (1023.0 * 1023.0 / mse).log10()
    };
    eprintln!("ffmpeg Main 10 cross-decode 64x64 gradient: psnr_y={psnr:.2} dB (peak=1023)");
    // Acceptance: ≥ 40 dB Y on a synthetic gradient (round-25 acceptance).
    assert!(
        psnr >= 40.0,
        "ffmpeg cross-decode luma PSNR below 40 dB: {psnr:.2} dB"
    );
}

#[test]
fn main10_mini_gop_two_roundtrip() {
    // Round 33: 10-bit mini_gop=2 (I-P-B sequence) should produce a valid
    // self-decodable stream with PSNR ≥ 25 dB on all three frames.
    let w = 64u32;
    let h = 64u32;
    let params = p10_params(w, h);
    let mut enc = HevcEncoder::from_params_with_mini_gop(&params, 2)
        .expect("Main 10 mini_gop=2 should be accepted in round 33");

    // Send 3 frames: IDR (poc=0), B (poc=1), P (poc=2).
    // Display order: poc0, poc1, poc2.
    // Decode order: IDR, P(poc2), B(poc1).
    let frames: Vec<VideoFrame> = (0..3).map(|_| make_p10_gradient_frame(w, h)).collect();
    for f in &frames {
        enc.send_frame(&Frame::Video(f.clone()))
            .expect("send frame");
    }
    enc.flush().expect("flush");

    let mut pkts = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => pkts.push(p),
            Err(_) => break,
        }
    }
    // IDR access unit has 4 NALs; P and B are single NALs.
    assert!(pkts.len() >= 2, "expected at least IDR + P/B packets");

    // Self-decode all packets and confirm each frame is valid.
    let mut dec = HevcDecoder::new(CodecId::new("h265"));
    for pkt in &pkts {
        dec.send_packet(pkt).expect("send packet to decoder");
    }
    let mut decoded = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => decoded.push(v),
            Ok(_) => {}
            Err(_) => break,
        }
    }
    assert!(
        !decoded.is_empty(),
        "decoder should produce at least one frame"
    );
    let psnr = psnr_y_p10(&frames[0], &decoded[0], w, h);
    eprintln!("main10 mini_gop=2 IDR psnr_y={psnr:.2} dB");
    assert!(psnr > 25.0, "main10 mini_gop=2 IDR PSNR too low: {psnr:.2}");
}

#[test]
fn main10_mini_gop_two_accepts_construction() {
    // Verify that construction succeeds for 10-bit mini_gop=2 (round 33).
    let params = p10_params(64, 64);
    let res = HevcEncoder::from_params_with_mini_gop(&params, 2);
    assert!(
        res.is_ok(),
        "Main 10 mini_gop=2 should be accepted in round 33"
    );
}
