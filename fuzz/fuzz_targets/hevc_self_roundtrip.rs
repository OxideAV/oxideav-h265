#![no_main]

//! Fuzz: random YUV 4:2:0 → oxideav HEVC encode → oxideav HEVC decode.
//!
//! HevcEncoder is lossy (default QP 26 with real DCT/DST transforms +
//! flat-list quantisation, ~45 dB PSNR on synthetic content per the
//! existing `encoder_roundtrip` test). We therefore can't assert
//! pixel-equality — instead we assert the decoded planes have the
//! expected size, and that mean absolute Y error is within a generous
//! lossy bound (sanity-check that the encoder didn't emit an
//! all-grey or all-zero picture).
//!
//! The HEVC encoder requires width/height to be multiples of 16 (16×16
//! CTU). With MAX_PIXELS=4096 and MAX_WIDTH=64 the largest feasible
//! frame is 64×64 luma, 32×32 chroma — a sane upper bound for fuzz
//! throughput.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{
    CodecId, CodecParameters, Decoder, Encoder, Frame, Packet, PixelFormat, Rational, TimeBase,
    VideoFrame, VideoPlane,
};
use oxideav_h265::decoder::HevcDecoder;
use oxideav_h265::encoder::HevcEncoder;

/// Maximum frame width/height in luma samples (multiple of 16, and
/// caps total work per fuzz iteration).
const MAX_DIM: u32 = 64;
/// Maximum total pixel count across all planes — keeps fuzz iters fast.
const MAX_PIXELS: usize = 4096;

fuzz_target!(|data: &[u8]| {
    let Some((width, height, y, cb, cr)) = yuv420_from_fuzz_input(data) else {
        return;
    };

    // Encode through HevcEncoder.
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

    // Decode through HevcDecoder.
    let mut dec = HevcDecoder::new(CodecId::new("h265"));
    let in_pkt = Packet::new(0, TimeBase::new(1, 30), pkt.data);
    if dec.send_packet(&in_pkt).is_err() {
        return;
    }
    let out = match dec.receive_frame() {
        Ok(Frame::Video(v)) => v,
        _ => return,
    };

    // Sanity checks: plane lengths match. The encoder is lossy so we
    // cannot pixel-compare — we just confirm the plane shapes and
    // that the decoded content isn't trivially broken (mean absolute
    // Y error within 50 LSBs is loose but catches all-zero / all-255
    // output).
    let cw = (width / 2) as usize;
    let ch = (height / 2) as usize;
    assert_eq!(out.planes.len(), 3, "decoded frame should have 3 planes");
    let dec_y = &out.planes[0];
    let dec_cb = &out.planes[1];
    let dec_cr = &out.planes[2];
    assert!(
        dec_y.data.len() >= dec_y.stride * (height as usize),
        "Y plane too short"
    );
    assert!(
        dec_cb.data.len() >= dec_cb.stride * ch,
        "Cb plane too short"
    );
    assert!(
        dec_cr.data.len() >= dec_cr.stride * ch,
        "Cr plane too short"
    );

    let y_mae = mean_abs_y(width, height, &y, dec_y);
    assert!(
        y_mae <= 50.0,
        "decoded Y plane diverges from source (MAE = {y_mae})"
    );
});

fn mean_abs_y(width: u32, height: u32, src_y: &[u8], dec_y: &VideoPlane) -> f64 {
    let w = width as usize;
    let h = height as usize;
    let mut sum: u64 = 0;
    for row in 0..h {
        for col in 0..w {
            let s = src_y[row * w + col] as i32;
            let d = dec_y.data[row * dec_y.stride + col] as i32;
            sum += (s - d).unsigned_abs() as u64;
        }
    }
    sum as f64 / (w * h) as f64
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

/// Carve fuzz bytes into (width, height, Y, Cb, Cr) for an 8-bit 4:2:0
/// frame. Width/height are forced to a multiple of 16 (CTU size) and
/// capped at MAX_DIM. Returns None if there aren't enough bytes for the
/// luma plane.
fn yuv420_from_fuzz_input(data: &[u8]) -> Option<(u32, u32, Vec<u8>, Vec<u8>, Vec<u8>)> {
    let (&shape_w, rest) = data.split_first()?;
    let (&shape_h, rest) = rest.split_first()?;

    // Width / height in 16-pixel CTUs, [1..=4] CTUs each → 16..=64 px.
    let w_ctus = ((shape_w as u32) % 4) + 1;
    let h_ctus = ((shape_h as u32) % 4) + 1;
    let width = (w_ctus * 16).min(MAX_DIM);
    let height = (h_ctus * 16).min(MAX_DIM);
    let total_pixels = (width as usize) * (height as usize);
    if total_pixels > MAX_PIXELS {
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
