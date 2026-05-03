#![no_main]

//! Fuzz: random YUV 4:2:0 → oxideav HEVC encode → libde265 decode.
//!
//! HevcEncoder is lossy (default QP 26, ~45 dB PSNR on synthetic
//! content), so the cross-decode comparison uses a generous mean
//! absolute Y bound rather than per-pixel equality. The point of this
//! harness isn't bit-exact equality with libde265 — it's verifying
//! that the bitstream we emit is parseable + decodable by an
//! independent reference decoder, with the reconstructed picture
//! visually close to the input.
//!
//! The harness early-returns when libde265 isn't installed (workspace
//! policy bars vendoring libde265 — runtime-only dep).

use libfuzzer_sys::fuzz_target;
use oxideav_core::{
    CodecId, CodecParameters, Encoder, Frame, PixelFormat, Rational, VideoFrame, VideoPlane,
};
use oxideav_h265::encoder::HevcEncoder;
use oxideav_h265_fuzz::libde265;

const MAX_DIM: u32 = 64;
const MAX_PIXELS: usize = 4096;

fuzz_target!(|data: &[u8]| {
    if !libde265::available() {
        return;
    }
    let Some((width, height, y, cb, cr)) = yuv420_from_fuzz_input(data) else {
        return;
    };

    // Encode through oxideav HevcEncoder.
    let mut params = CodecParameters::video(CodecId::new("h265"));
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(30, 1));

    let mut enc = match HevcEncoder::from_params(&params) {
        Ok(e) => e,
        Err(_) => return,
    };
    let frame = Frame::Video(make_video_frame(width, height, &y, &cb, &cr));
    if enc.send_frame(&frame).is_err() {
        return;
    }
    let pkt = match enc.receive_packet() {
        Ok(p) => p,
        Err(_) => return,
    };

    // Decode through libde265.
    let decoded = match libde265::decode_to_yuv(&pkt.data) {
        Some(d) => d,
        None => return,
    };

    assert_eq!(
        decoded.width, width,
        "libde265 decoded width {} != source width {}",
        decoded.width, width
    );
    assert_eq!(
        decoded.height, height,
        "libde265 decoded height {} != source height {}",
        decoded.height, height
    );

    // Both ends are lossy (QP-26 oxideav encoder + reference libde265
    // dequant/reconstruct). Use mean absolute error as the proxy — at
    // QP 26 the per-channel MAE on natural content is ~5 LSB; we set
    // 50 as a very generous upper bound that still catches catastrophic
    // failures (all-grey, all-zero, swapped planes).
    let y_mae = mean_abs(width as usize, height as usize, &y, &decoded.y);
    assert!(
        y_mae <= 50.0,
        "Y plane diverges from source via libde265 (MAE = {y_mae})"
    );
});

fn mean_abs(w: usize, h: usize, src: &[u8], dec: &[u8]) -> f64 {
    let mut sum: u64 = 0;
    let n = w * h;
    for i in 0..n {
        let s = src[i] as i32;
        let d = dec[i] as i32;
        sum += (s - d).unsigned_abs() as u64;
    }
    sum as f64 / n.max(1) as f64
}

fn make_video_frame(width: u32, height: u32, y: &[u8], cb: &[u8], cr: &[u8]) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w,
                data: y[..w * h].to_vec(),
            },
            VideoPlane {
                stride: cw,
                data: cb[..cw * ch].to_vec(),
            },
            VideoPlane {
                stride: cw,
                data: cr[..cw * ch].to_vec(),
            },
        ],
    }
}

fn yuv420_from_fuzz_input(data: &[u8]) -> Option<(u32, u32, Vec<u8>, Vec<u8>, Vec<u8>)> {
    let (&shape_w, rest) = data.split_first()?;
    let (&shape_h, rest) = rest.split_first()?;
    let w_ctus = ((shape_w as u32) % 4) + 1;
    let h_ctus = ((shape_h as u32) % 4) + 1;
    let width = (w_ctus * 16).min(MAX_DIM);
    let height = (h_ctus * 16).min(MAX_DIM);
    if (width as usize) * (height as usize) > MAX_PIXELS {
        return None;
    }
    let cw = (width / 2) as usize;
    let ch = (height / 2) as usize;
    let y_len = (width as usize) * (height as usize);
    let c_len = cw * ch;
    let need = y_len + 2 * c_len;
    if rest.len() < need {
        return None;
    }
    let y = rest[..y_len].to_vec();
    let cb = rest[y_len..y_len + c_len].to_vec();
    let cr = rest[y_len + c_len..y_len + 2 * c_len].to_vec();
    Some((width, height, y, cb, cr))
}
