//! Round-30: Main 4:4:4 (8-bit) encoder integration tests.
//!
//! Scope:
//! * The encoder accepts a `Yuv444P` keyframe and emits an Annex B
//!   stream whose SPS declares Main 4:4:4 (`profile_idc = 4`,
//!   `chroma_format_idc = 3`, `bit_depth_*_minus8 = 0`).
//! * Our own decoder reads the bitstream back at 4:4:4 (chroma planes
//!   match the luma resolution) and the reconstructed luma stays close
//!   to the source on a smooth gradient (PSNR ≥ 30 dB at 8-bit).
//! * When the `FFMPEG` env var is set and `ffmpeg` is on PATH,
//!   libavcodec cross-decodes the same bitstream and the cross-decoded
//!   luma matches the source within ≥ 40 dB on the gradient.

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

/// Build a `Yuv444P` frame with a horizontal 0..255 luma gradient and
/// flat mid-range chroma (128). At 4:4:4 the chroma planes are full
/// luma resolution.
fn make_p444_gradient_frame(w: u32, h: u32) -> VideoFrame {
    let wu = w as usize;
    let hu = h as usize;
    let mut y = Vec::with_capacity(wu * hu);
    for _ in 0..hu {
        for xx in 0..wu {
            let v = ((xx as u32 * 255) / w.max(1)) as u8;
            y.push(v);
        }
    }
    let cb = vec![128u8; wu * hu];
    let cr = vec![128u8; wu * hu];
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: wu,
                data: y,
            },
            VideoPlane {
                stride: wu,
                data: cb,
            },
            VideoPlane {
                stride: wu,
                data: cr,
            },
        ],
    }
}

fn p444_params(w: u32, h: u32) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("h265"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(PixelFormat::Yuv444P);
    p.frame_rate = Some(Rational::new(30, 1));
    p
}

fn encode_one_p444(frame: &VideoFrame, w: u32, h: u32) -> Vec<u8> {
    let params = p444_params(w, h);
    let mut enc = HevcEncoder::from_params(&params).expect("encoder");
    enc.send_frame(&Frame::Video(frame.clone())).expect("send");
    let pkt = enc.receive_packet().expect("packet");
    pkt.data
}

/// 8-bit luma PSNR — peak = 255.
fn psnr_y_p444(src: &VideoFrame, dec: &VideoFrame, w: u32, h: u32) -> f64 {
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
fn main444_emit_carries_main444_sps() {
    let frame = make_p444_gradient_frame(64, 64);
    let bytes = encode_one_p444(&frame, 64, 64);
    let nals: Vec<_> = iter_annex_b(&bytes).collect();
    // VPS + SPS + PPS + IDR — same NAL count as the 4:2:0 emissions.
    assert_eq!(nals.len(), 4, "expected VPS+SPS+PPS+IDR NALs");
    assert_eq!(nals[0].header.nal_unit_type, NalUnitType::Vps);
    assert_eq!(nals[1].header.nal_unit_type, NalUnitType::Sps);
    assert_eq!(nals[2].header.nal_unit_type, NalUnitType::Pps);
    assert_eq!(nals[3].header.nal_unit_type, NalUnitType::IdrNLp);
    use oxideav_h265::nal::extract_rbsp;
    let sps_rbsp = extract_rbsp(nals[1].payload());
    let sps = parse_sps(&sps_rbsp).expect("parse sps");
    // 8-bit luma + chroma.
    assert_eq!(sps.bit_depth_luma_minus8, 0, "luma should be 8-bit");
    assert_eq!(sps.bit_depth_chroma_minus8, 0, "chroma should be 8-bit");
    // chroma_format_idc = 3 (4:4:4).
    assert_eq!(sps.chroma_format_idc, 3, "should be 4:4:4");
    // separate_colour_plane_flag must be 0 (interleaved chroma).
    assert!(!sps.separate_colour_plane_flag);
    // Main 4:4:4 lives in the Format Range Extensions profile per §A.3.4,
    // so general_profile_idc = 4.
    assert_eq!(sps.profile_tier_level.general_profile_idc, 4);
    // Sub-sampling factors derived from cfi=3.
    assert_eq!(sps.sub_width_c(), 1);
    assert_eq!(sps.sub_height_c(), 1);
}

#[test]
fn main444_self_roundtrip_psnr() {
    let src = make_p444_gradient_frame(64, 64);
    let bytes = encode_one_p444(&src, 64, 64);
    let mut dec = HevcDecoder::new(CodecId::new("h265"));
    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes);
    dec.send_packet(&pkt).expect("send packet");
    let frame = match dec.receive_frame() {
        Ok(Frame::Video(v)) => v,
        Ok(other) => panic!("non-video frame: {other:?}"),
        Err(e) => panic!("decoder error: {e:?}"),
    };
    // Each plane is 8-bit packed at full luma resolution: every plane
    // should be at least w * h bytes.
    assert!(frame.planes[0].data.len() >= 64 * 64);
    assert!(frame.planes[1].data.len() >= 64 * 64);
    assert!(frame.planes[2].data.len() >= 64 * 64);
    let psnr = psnr_y_p444(&src, &frame, 64, 64);
    eprintln!("main444 self-roundtrip 64x64: psnr_y={psnr:.2} dB (peak=255)");
    assert!(psnr > 30.0, "8-bit 4:4:4 luma PSNR too low: {psnr:.2} dB");
}

#[test]
fn main444_multi_ctu_roundtrip_psnr() {
    let src = make_p444_gradient_frame(128, 128);
    let bytes = encode_one_p444(&src, 128, 128);
    let mut dec = HevcDecoder::new(CodecId::new("h265"));
    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes);
    dec.send_packet(&pkt).expect("send packet");
    let frame = match dec.receive_frame() {
        Ok(Frame::Video(v)) => v,
        Ok(other) => panic!("non-video frame: {other:?}"),
        Err(e) => panic!("decoder error: {e:?}"),
    };
    let psnr = psnr_y_p444(&src, &frame, 128, 128);
    eprintln!("main444 self-roundtrip 128x128: psnr_y={psnr:.2} dB (peak=255)");
    assert!(psnr > 30.0, "8-bit 4:4:4 luma PSNR too low: {psnr:.2} dB");
}

fn have_ffmpeg() -> bool {
    if std::env::var_os("FFMPEG").is_none() {
        return false;
    }
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

#[test]
fn main444_ffmpeg_cross_decode_psnr() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg gate disabled — skipping Main 4:4:4 ffmpeg cross-decode");
        return;
    }
    let w = 64u32;
    let h = 64u32;
    let src = make_p444_gradient_frame(w, h);
    let bytes = encode_one_p444(&src, w, h);

    let tmp: PathBuf = std::env::temp_dir();
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let stem = format!("oxideav_h265_main444_{}_{w}x{h}_{ns}", std::process::id());
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
        .arg("yuv444p")
        .arg(&yuv_path)
        .output()
        .expect("run ffmpeg");
    if !out.status.success() {
        eprintln!("ffmpeg stderr:\n{}", String::from_utf8_lossy(&out.stderr));
        panic!(
            "ffmpeg rejected the Main 4:4:4 bitstream (exit {:?})",
            out.status
        );
    }
    let dec = std::fs::read(&yuv_path).expect("read decoded yuv");
    let plane_len = (w * h) as usize;
    let expected_len = 3 * plane_len;
    assert_eq!(
        dec.len(),
        expected_len,
        "decoded YUV length mismatch (expected {expected_len}, got {})",
        dec.len()
    );

    let src_y = &src.planes[0].data;
    let src_stride = src.planes[0].stride;
    let dec_y = &dec[..plane_len];
    let dec_stride = w as usize;
    let mut sse: u64 = 0;
    let n = (w * h) as u64;
    for yy in 0..h as usize {
        for xx in 0..w as usize {
            let s = src_y[yy * src_stride + xx] as i64;
            let d = dec_y[yy * dec_stride + xx] as i64;
            let diff = s - d;
            sse += (diff * diff) as u64;
        }
    }
    let mse = sse as f64 / n as f64;
    let psnr = if mse == 0.0 {
        99.0
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    };
    eprintln!("ffmpeg Main 4:4:4 cross-decode 64x64 gradient: psnr_y={psnr:.2} dB (peak=255)");
    // Acceptance: ≥ 40 dB Y on a synthetic gradient (round-30 acceptance).
    assert!(
        psnr >= 40.0,
        "ffmpeg cross-decode luma PSNR below 40 dB: {psnr:.2} dB"
    );
}

#[test]
fn main444_rejects_mini_gop_two() {
    // 8-bit 4:4:4 + mini_gop=2 (B slices) should fail at construction
    // time since the B-slice writer only supports 4:2:0.
    let params = p444_params(64, 64);
    let res = HevcEncoder::from_params_with_mini_gop(&params, 2);
    assert!(res.is_err(), "Main 4:4:4 should reject mini_gop=2 for now");
}
