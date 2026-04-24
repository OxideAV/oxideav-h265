//! Verify that ffmpeg accepts the HEVC bitstream produced by our encoder.
//!
//! Gated by the `FFMPEG` env var (set to any value to enable) and by the
//! availability of the `ffmpeg` binary on `PATH`. The test writes the
//! encoder output to a temp .h265 file, runs
//! `ffmpeg -i out.h265 -f rawvideo -pix_fmt yuv420p out.yuv`, and checks
//! that ffmpeg exits 0 and decoded luma tracks the source within a
//! reasonable PSNR bound (the encoder is lossy).

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Frame, PixelFormat, Rational, TimeBase, VideoFrame, VideoPlane,
};

use oxideav_h265::encoder::HevcEncoder;

fn have_ffmpeg() -> bool {
    // Respect the environment gate first.
    if std::env::var_os("FFMPEG").is_none() {
        return false;
    }
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

fn make_gradient_frame(w: u32, h: u32) -> VideoFrame {
    let mut y = vec![0u8; (w * h) as usize];
    for yy in 0..h {
        for xx in 0..w {
            y[(yy * w + xx) as usize] = ((xx * 255) / w.max(1)) as u8;
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

fn encode(frame: &VideoFrame) -> Vec<u8> {
    let mut params = CodecParameters::video(CodecId::new("h265"));
    params.width = Some(frame.width);
    params.height = Some(frame.height);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(30, 1));
    let mut enc = HevcEncoder::from_params(&params).unwrap();
    enc.send_frame(&Frame::Video(frame.clone())).unwrap();
    enc.receive_packet().unwrap().data
}

fn run_one(w: u32, h: u32) {
    let src = make_gradient_frame(w, h);
    let bytes = encode(&src);

    let tmp: PathBuf = std::env::temp_dir();
    // Include dimensions + a nanosecond timestamp in the filename so
    // tests running in parallel don't collide on the same temp files.
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let stem = format!(
        "oxideav_h265_{}_{}x{}_{ns}",
        std::process::id(),
        w,
        h
    );
    let h265_path = tmp.join(format!("{stem}.h265"));
    let yuv_path = tmp.join(format!("{stem}.yuv"));
    {
        let mut f = std::fs::File::create(&h265_path).unwrap();
        f.write_all(&bytes).unwrap();
    }

    let out = Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(&h265_path)
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(&yuv_path)
        .output()
        .expect("run ffmpeg");
    if !out.status.success() {
        eprintln!("ffmpeg stderr:\n{}", String::from_utf8_lossy(&out.stderr));
        panic!("ffmpeg rejected the bitstream (exit {:?})", out.status);
    }
    let dec = std::fs::read(&yuv_path).expect("read dec yuv");
    let expected_len = (w * h + 2 * (w / 2) * (h / 2)) as usize;
    assert_eq!(dec.len(), expected_len, "decoded YUV length mismatch");

    // Compare luma planes by PSNR (encoder is lossy).
    let src_y = &src.planes[0].data;
    let dec_y = &dec[..(w * h) as usize];
    let src_stride = src.planes[0].stride;
    let mut sse: u64 = 0;
    let n = (w * h) as u64;
    for yy in 0..h as usize {
        for xx in 0..w as usize {
            let s = src_y[yy * src_stride + xx] as i64;
            let d = dec_y[yy * w as usize + xx] as i64;
            let diff = s - d;
            sse += (diff * diff) as u64;
        }
    }
    let mse = sse as f64 / n as f64;
    let psnr = if mse == 0.0 { 99.0 } else { 10.0 * (255.0f64 * 255.0 / mse).log10() };
    eprintln!("ffmpeg_accepts: {w}x{h} psnr_y={psnr:.2} dB bytes={}", bytes.len());
    // With the CTB-row MPM rule fixed (§8.4.2: candIntraPredModeB forced to
    // INTRA_DC when the B neighbour is in the CTB row above), ffmpeg's
    // reconstruction matches ours within the encoder's QP-26 quantisation
    // budget. Keep a bound that's tight enough to catch regressions but
    // tolerant of small encoder-choice shifts.
    assert!(psnr > 30.0, "ffmpeg-decoded luma PSNR too low: {psnr:.2}");

    // Clean up.
    let _ = std::fs::remove_file(&h265_path);
    let _ = std::fs::remove_file(&yuv_path);
}

#[test]
fn ffmpeg_decodes_our_bitstream() {
    if !have_ffmpeg() {
        eprintln!("[skip] ffmpeg not enabled — set FFMPEG=1 to run");
        return;
    }
    run_one(64, 64);
}

#[test]
fn ffmpeg_decodes_multi_ctu() {
    if !have_ffmpeg() {
        eprintln!("[skip] ffmpeg not enabled — set FFMPEG=1 to run");
        return;
    }
    run_one(128, 64);
    run_one(128, 128);
}
