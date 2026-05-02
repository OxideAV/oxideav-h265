//! Round-32: Main 4:4:4 12 (12-bit, 4:4:4) encoder integration tests.
//!
//! Scope:
//! * The encoder accepts a `Yuv444P12Le` keyframe and emits an Annex B
//!   stream whose SPS declares Main 4:4:4 12 (`profile_idc = 4`,
//!   `chroma_format_idc = 3`, `bit_depth_*_minus8 = 4`) with the
//!   §A.3.4 RExt constraint flags row for Main 4:4:4 12
//!   (`max_8bit = 0`, `max_10bit = 0`, `max_12bit = 1`,
//!   `max_422chroma = 0`, `max_420chroma = 0`).
//! * Our own decoder reads the bitstream back at 12-bit 4:4:4 (chroma
//!   planes match luma resolution at 16-bit packed LE) and the
//!   reconstructed luma stays close to the source on a smooth gradient
//!   (PSNR ≥ 30 dB on the 12-bit / peak = 4095 scale).
//! * When the `FFMPEG` env var is set and `ffmpeg` is on PATH,
//!   libavcodec cross-decodes the same bitstream and the cross-decoded
//!   luma matches the source within ≥ 40 dB on the gradient.
//! * `mini_gop > 1` is rejected at construction time (4:4:4 + 12-bit is
//!   IDR-only — same restriction the round-30 / 31 4:4:4 paths apply).

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

/// Build a `Yuv444P12Le` frame with a horizontal 0..4095 luma gradient
/// and flat mid-range chroma (2048). Plane data follows the LE 16-bit
/// layout used by `oxideav-core`'s `PixelFormat::Yuv444P12Le`. At 4:4:4
/// the chroma planes are full luma resolution so each plane is
/// `w * h * 2` bytes.
fn make_p444p12_gradient_frame(w: u32, h: u32) -> VideoFrame {
    let wu = w as usize;
    let hu = h as usize;
    let mut y = Vec::with_capacity(wu * hu * 2);
    for _yy in 0..hu {
        for xx in 0..wu {
            let v = ((xx as u32 * 4095) / w.max(1)) as u16;
            y.push((v & 0xFF) as u8);
            y.push((v >> 8) as u8);
        }
    }
    let mut cb = Vec::with_capacity(wu * hu * 2);
    let mut cr = Vec::with_capacity(wu * hu * 2);
    for _ in 0..(wu * hu) {
        // 12-bit chroma neutral = 2048 = 0x0800 → LE bytes 0x00, 0x08.
        cb.push(0x00);
        cb.push(0x08);
        cr.push(0x00);
        cr.push(0x08);
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: wu * 2,
                data: y,
            },
            VideoPlane {
                stride: wu * 2,
                data: cb,
            },
            VideoPlane {
                stride: wu * 2,
                data: cr,
            },
        ],
    }
}

fn p444p12_params(w: u32, h: u32) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("h265"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(PixelFormat::Yuv444P12Le);
    p.frame_rate = Some(Rational::new(30, 1));
    p
}

fn encode_one_p444p12(frame: &VideoFrame, w: u32, h: u32) -> Vec<u8> {
    let params = p444p12_params(w, h);
    let mut enc = HevcEncoder::from_params(&params).expect("encoder");
    enc.send_frame(&Frame::Video(frame.clone())).expect("send");
    let pkt = enc.receive_packet().expect("packet");
    pkt.data
}

/// Read a single 12-bit sample from a packed `Yuv*P12Le` plane.
fn read_p12(plane: &[u8], stride: usize, x: usize, y: usize) -> u16 {
    let off = y * stride + x * 2;
    (plane[off] as u16) | ((plane[off + 1] as u16) << 8)
}

/// 12-bit luma PSNR — peak = 4095.
fn psnr_y_p12(src: &VideoFrame, dec: &VideoFrame, w: u32, h: u32) -> f64 {
    let ss = src.planes[0].stride;
    let ds = dec.planes[0].stride;
    let src_y = &src.planes[0].data;
    let dec_y = &dec.planes[0].data;
    let mut sse: u64 = 0;
    let n = (w * h) as u64;
    for yy in 0..h as usize {
        for xx in 0..w as usize {
            let s = read_p12(src_y, ss, xx, yy) as i64;
            let d = read_p12(dec_y, ds, xx, yy) as i64;
            let diff = s - d;
            sse += (diff * diff) as u64;
        }
    }
    if sse == 0 {
        return 99.0;
    }
    let mse = sse as f64 / n as f64;
    10.0 * (4095.0 * 4095.0 / mse).log10()
}

#[test]
fn main444_12_emit_carries_main444_12_sps() {
    let frame = make_p444p12_gradient_frame(64, 64);
    let bytes = encode_one_p444p12(&frame, 64, 64);
    let nals: Vec<_> = iter_annex_b(&bytes).collect();
    // VPS + SPS + PPS + IDR — same NAL count as the 4:2:0 / 4:4:4-8 / 4:4:4-10 emissions.
    assert_eq!(nals.len(), 4, "expected VPS+SPS+PPS+IDR NALs");
    assert_eq!(nals[0].header.nal_unit_type, NalUnitType::Vps);
    assert_eq!(nals[1].header.nal_unit_type, NalUnitType::Sps);
    assert_eq!(nals[2].header.nal_unit_type, NalUnitType::Pps);
    assert_eq!(nals[3].header.nal_unit_type, NalUnitType::IdrNLp);
    use oxideav_h265::nal::extract_rbsp;
    let sps_rbsp = extract_rbsp(nals[1].payload());
    let sps = parse_sps(&sps_rbsp).expect("parse sps");
    // 12-bit luma + chroma.
    assert_eq!(sps.bit_depth_luma_minus8, 4, "luma should be 12-bit");
    assert_eq!(sps.bit_depth_chroma_minus8, 4, "chroma should be 12-bit");
    assert_eq!(sps.bit_depth_y(), 12);
    assert_eq!(sps.bit_depth_c(), 12);
    // chroma_format_idc = 3 (4:4:4).
    assert_eq!(sps.chroma_format_idc, 3, "should be 4:4:4");
    // separate_colour_plane_flag must be 0 (interleaved chroma).
    assert!(!sps.separate_colour_plane_flag);
    // Main 4:4:4 12 lives in the Format Range Extensions profile per
    // §A.3.4, so general_profile_idc = 4.
    assert_eq!(sps.profile_tier_level.general_profile_idc, 4);
    // §A.3.4 Table A.2 row for Main 4:4:4 12 — verify the 9 RExt
    // constraint flags directly from the SPS RBSP bitstream. The
    // crate's `parse_profile_tier_level` skips the 43-bit constraint
    // region wholesale, so we walk the bits here to make sure the
    // round-32 emitter writes the exact §A.3.4 signature for Main
    // 4:4:4 12: max_12bit=1, max_10bit=0, max_8bit=0,
    // max_422chroma=0, max_420chroma=0, max_monochrome=0,
    // intra=0, one_picture_only=0, lower_bit_rate=1.
    {
        use oxideav_h265::bitreader::BitReader;
        let mut br = BitReader::new(&sps_rbsp);
        // Skip up to the start of the PTL constraint-flag region:
        // sps_video_parameter_set_id (u4)
        // + sps_max_sub_layers_minus1 (u3)
        // + sps_temporal_id_nesting_flag (u1)
        // + ptl: profile_space(u2) + tier(u1) + profile_idc(u5)
        //        + compat[32] + 4 source/constraint flags
        // = 8 + 8 + 32 + 4 = 52 bits before the 9 RExt flags.
        br.skip(52).expect("skip to RExt flags");
        assert_eq!(br.u1().unwrap(), 1, "max_12bit must be 1");
        assert_eq!(br.u1().unwrap(), 0, "max_10bit must be 0 for Main 4:4:4 12");
        assert_eq!(br.u1().unwrap(), 0, "max_8bit must be 0 for Main 4:4:4 12");
        assert_eq!(br.u1().unwrap(), 0, "max_422chroma must be 0");
        assert_eq!(br.u1().unwrap(), 0, "max_420chroma must be 0");
        assert_eq!(br.u1().unwrap(), 0, "max_monochrome must be 0");
        assert_eq!(br.u1().unwrap(), 0, "intra must be 0");
        assert_eq!(br.u1().unwrap(), 0, "one_picture_only must be 0");
        assert_eq!(br.u1().unwrap(), 1, "lower_bit_rate must be 1");
    }
    // Sub-sampling factors derived from cfi=3.
    assert_eq!(sps.sub_width_c(), 1);
    assert_eq!(sps.sub_height_c(), 1);
}

#[test]
fn main444_12_self_roundtrip_psnr() {
    let src = make_p444p12_gradient_frame(64, 64);
    let bytes = encode_one_p444p12(&src, 64, 64);
    let mut dec = HevcDecoder::new(CodecId::new("h265"));
    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes);
    dec.send_packet(&pkt).expect("send packet");
    let frame = match dec.receive_frame() {
        Ok(Frame::Video(v)) => v,
        Ok(other) => panic!("non-video frame: {other:?}"),
        Err(e) => panic!("decoder error: {e:?}"),
    };
    // At 4:4:4 + 12-bit, every plane is 16-bit packed at full luma
    // resolution: at least 2 * w * h bytes each.
    assert!(frame.planes[0].data.len() >= 64 * 64 * 2);
    assert!(frame.planes[1].data.len() >= 64 * 64 * 2);
    assert!(frame.planes[2].data.len() >= 64 * 64 * 2);
    let psnr = psnr_y_p12(&src, &frame, 64, 64);
    eprintln!("main444_12 self-roundtrip 64x64: psnr_y={psnr:.2} dB (peak=4095)");
    assert!(psnr > 30.0, "12-bit 4:4:4 luma PSNR too low: {psnr:.2} dB");
}

#[test]
fn main444_12_multi_ctu_roundtrip_psnr() {
    let src = make_p444p12_gradient_frame(128, 128);
    let bytes = encode_one_p444p12(&src, 128, 128);
    let mut dec = HevcDecoder::new(CodecId::new("h265"));
    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes);
    dec.send_packet(&pkt).expect("send packet");
    let frame = match dec.receive_frame() {
        Ok(Frame::Video(v)) => v,
        Ok(other) => panic!("non-video frame: {other:?}"),
        Err(e) => panic!("decoder error: {e:?}"),
    };
    let psnr = psnr_y_p12(&src, &frame, 128, 128);
    eprintln!("main444_12 self-roundtrip 128x128: psnr_y={psnr:.2} dB (peak=4095)");
    assert!(psnr > 30.0, "12-bit 4:4:4 luma PSNR too low: {psnr:.2} dB");
}

fn have_ffmpeg() -> bool {
    if std::env::var_os("FFMPEG").is_none() {
        return false;
    }
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

#[test]
fn main444_12_ffmpeg_cross_decode_psnr() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg gate disabled — skipping Main 4:4:4 12 ffmpeg cross-decode");
        return;
    }
    let w = 64u32;
    let h = 64u32;
    let src = make_p444p12_gradient_frame(w, h);
    let bytes = encode_one_p444p12(&src, w, h);

    let tmp: PathBuf = std::env::temp_dir();
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let stem = format!(
        "oxideav_h265_main444_12_{}_{w}x{h}_{ns}",
        std::process::id()
    );
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
        .arg("yuv444p12le")
        .arg(&yuv_path)
        .output()
        .expect("run ffmpeg");
    if !out.status.success() {
        eprintln!("ffmpeg stderr:\n{}", String::from_utf8_lossy(&out.stderr));
        panic!(
            "ffmpeg rejected the Main 4:4:4 12 bitstream (exit {:?})",
            out.status
        );
    }
    let dec = std::fs::read(&yuv_path).expect("read decoded yuv");
    // Each plane is 2*w*h bytes; three planes total.
    let plane_len = (w * h * 2) as usize;
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
    let dec_stride = (w * 2) as usize;
    let mut sse: u64 = 0;
    let n = (w * h) as u64;
    for yy in 0..h as usize {
        for xx in 0..w as usize {
            let s = read_p12(src_y, src_stride, xx, yy) as i64;
            let d = read_p12(dec_y, dec_stride, xx, yy) as i64;
            let diff = s - d;
            sse += (diff * diff) as u64;
        }
    }
    let mse = sse as f64 / n as f64;
    let psnr = if mse == 0.0 {
        99.0
    } else {
        10.0 * (4095.0 * 4095.0 / mse).log10()
    };
    eprintln!("ffmpeg Main 4:4:4 12 cross-decode 64x64 gradient: psnr_y={psnr:.2} dB (peak=4095)");
    // Acceptance: ≥ 40 dB Y on a synthetic gradient (round-32 acceptance).
    assert!(
        psnr >= 40.0,
        "ffmpeg cross-decode luma PSNR below 40 dB: {psnr:.2} dB"
    );
}

#[test]
fn main444_12_rejects_mini_gop_two() {
    // 12-bit + 4:4:4 + mini_gop=2 (B slices) should fail at construction
    // time since both constraints (high-bit-depth and 4:4:4) currently
    // restrict the encoder to keyframe-only.
    let params = p444p12_params(64, 64);
    let res = HevcEncoder::from_params_with_mini_gop(&params, 2);
    assert!(
        res.is_err(),
        "Main 4:4:4 12 should reject mini_gop=2 for now"
    );
}
