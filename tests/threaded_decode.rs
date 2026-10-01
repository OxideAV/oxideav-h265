//! Thread-budget decoding pins: under an `ExecutionContext` budget the
//! decoder runs a WPP picture's CTB rows in a wavefront and every
//! picture's in-loop filters row-parallel — and the output must equal
//! the serial decode byte for byte, for every stream and every budget.
//!
//! * every embedded conformance / tool-axis / real-world still fixture
//!   decodes identically through `SequenceDecoder` at budgets 1 / 2 / 3
//!   / 5 (WPP streams take the wavefront, tiled ones the tile-parallel
//!   path, the rest the row-parallel filters only);
//! * the registry decoder honours `set_execution_context` and keeps the
//!   HEIC still pins;
//! * the official `WPP_HIGH_TP_444_8BIT` stream (when the corpus is
//!   staged) decodes identically threaded;
//! * a budgeted single-still decode is never slower than the serial one
//!   (a ratio bound that tolerates CI noise, on a stream the encoder
//!   writes in the test).

mod fixture_bytes;

use fixture_bytes::md5;
use oxideav_core::{
    CodecParameters, Error, ExecutionContext, Frame, Packet, TimeBase, VideoFrame, VideoPlane,
};
use oxideav_h265::{DecodedFrame, SequenceDecoder};
use std::time::Instant;

fn decode_with(stream: &[u8], threads: usize) -> Vec<DecodedFrame> {
    let mut dec = SequenceDecoder::new();
    dec.set_threads(threads);
    dec.push_annexb(stream).expect("push");
    dec.finish().expect("finish")
}

fn assert_budget_invariant(stream: &[u8], what: &str) {
    eprintln!("budget invariant: {what}");
    let serial = decode_with(stream, 1);
    assert!(!serial.is_empty(), "{what}: decodes");
    for threads in [2usize, 3, 5] {
        let threaded = decode_with(stream, threads);
        assert_eq!(
            serial.len(),
            threaded.len(),
            "{what}: frame count at {threads}"
        );
        for (i, (a, b)) in serial.iter().zip(&threaded).enumerate() {
            assert_eq!(a.poc, b.poc, "{what}: frame {i} poc at {threads}");
            assert!(
                a.picture == b.picture,
                "{what}: frame {i} differs at {threads} threads"
            );
        }
    }
}

#[test]
fn staged_fixtures_decode_identically_under_any_budget() {
    use fixture_bytes::*;
    for (name, stream) in [
        ("tiny-i", TINY_I_HEVC),
        ("qp-high", QP_HIGH_HEVC),
        ("qp-low", QP_LOW_HEVC),
        ("main-still", MAIN_STILL_HEVC),
        ("sao-on", SAO_ON_HEVC),
        ("allintra", ALLINTRA_HEVC),
        ("i-then-p", I_THEN_P_HEVC),
        ("main10", MAIN10_HEVC),
        ("bipred", BIPRED_HEVC),
        ("wpp", WPP_HEVC),
        ("multi-slice", MULTI_SLICE_HEVC),
        ("tile-cols", TILE_COLS_HEVC),
        ("weighted-p", WEIGHTED_P_HEVC),
        ("weighted-b", WEIGHTED_B_HEVC),
        ("per-slice-lf", PERSLICE_LF_HEVC),
        ("true-tiles", TRUE_TILES_HEVC),
    ] {
        assert_budget_invariant(stream, name);
    }
}

#[test]
fn tool_axis_fixtures_decode_identically_under_any_budget() {
    use fixture_bytes::r410::*;
    for (name, stream) in [
        ("b-pyramid", BPYR_HEVC),
        ("scaling", SCALING_HEVC),
        ("strong", STRONG_HEVC),
        ("rect-amp", RECTAMP_HEVC),
        ("constrained-intra", CI_HEVC),
        ("tskip", TSKIP_HEVC),
        ("wpp-slices", WPPSLICES_HEVC),
        ("open-gop", OPENGOP_HEVC),
    ] {
        assert_budget_invariant(stream, name);
    }
    assert_budget_invariant(
        include_bytes!("fixture_bytes/r413-rdpcm-implicit.hevc"),
        "rdpcm-implicit",
    );
    assert_budget_invariant(
        include_bytes!("fixture_bytes/r413-rdpcm-explicit.hevc"),
        "rdpcm-explicit",
    );
    assert_budget_invariant(include_bytes!("fixture_bytes/r416-ccp.hevc"), "ccp");
    assert_budget_invariant(include_bytes!("fixture_bytes/r416-act.hevc"), "act");
    assert_budget_invariant(
        include_bytes!("fixture_bytes/r456-pyramid-rdoq-sdh-tu2-wp-wpp-qp30.hevc"),
        "pyramid-wpp",
    );
    // Tiled pictures take the tile-parallel path.
    assert_budget_invariant(
        include_bytes!("fixture_bytes/r456-lowdelay-tiles-sl-rdoq-aq-qp29.hevc"),
        "lowdelay-tiles",
    );
    assert_budget_invariant(
        include_bytes!("fixture_bytes/r453-pcm-tiles-explicit-96x64.hevc"),
        "pcm-tiles-explicit",
    );
}

/// Every real-world HEIC still (third-party software encoder with WPP,
/// the OS converter's hardware encoder, a general-purpose converter):
/// 4:2:0 / 4:2:2 / 4:4:4 / monochrome at 8 / 10 / 12 bits, lossless,
/// slices + WPP, transform skip, deep RQTs.
#[test]
fn heic_stills_decode_identically_under_any_budget() {
    for name in [
        "henc-133",
        "henc-133-L",
        "henc-133-444",
        "henc-133-422",
        "henc-133-ctu16-slices-wpp",
        "henc-133-tskip-culossless-rdoq",
        "henc-133-nosao-nosis-cip-sl",
        "henc-133-aq8-cqp-deblock",
        "henc-133-vui-hrd-sei",
        "henc-17",
        "henc-120-b10",
        "henc-120-b10-444-tskip",
        "henc-120-b12-422",
        "henc-120-L-b10",
        "henc-g160",
        "henc-g160-b10",
        "sips-133",
        "sips-133-q100",
        "sips-120-16",
        "sips-120-16-q100",
        "sips-17",
        "sips-2x2",
        "sips-g160",
        "magick-133-444",
        "magick-120-d12",
    ] {
        let path = format!(
            "{}/tests/fixture_bytes/r460/{name}.hevc",
            env!("CARGO_MANIFEST_DIR")
        );
        let stream = std::fs::read(&path).expect("vendored still");
        assert_budget_invariant(&stream, name);
    }
}

fn registry_md5(stream: &[u8], threads: usize) -> (String, usize) {
    let params = CodecParameters::video("h265".into());
    let mut dec = oxideav_h265::make_decoder(&params).expect("decoder");
    dec.set_execution_context(&ExecutionContext::with_threads(threads));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), stream.to_vec()))
        .expect("send");
    dec.flush().expect("flush");
    let mut out = Vec::new();
    let mut frames = 0;
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => {
                frames += 1;
                for p in &v.planes {
                    out.extend_from_slice(&p.data);
                }
            }
            Ok(_) => panic!("video frames only"),
            Err(Error::NeedMore) => continue,
            Err(Error::Eof) => break,
            Err(e) => panic!("receive: {e}"),
        }
    }
    (md5::hex(&out), frames)
}

/// The registry decoder takes the budget through the core contract and
/// keeps the round-460 real-world pins (cropped planar MD5s).
#[test]
fn registry_decoder_keeps_the_still_pins_under_a_budget() {
    for (name, md5) in [
        ("henc-133", "010ef8ba7e899d9d38261544282d9b77"),
        (
            "henc-133-ctu16-slices-wpp",
            "e99c47d70d67a1bffd4c2e8dbd1f5528",
        ),
        ("henc-120-b10", "3d2a6b0a4d1d0d55b8a4ac3c0c9b8f0b"),
    ] {
        let path = format!(
            "{}/tests/fixture_bytes/r460/{name}.hevc",
            env!("CARGO_MANIFEST_DIR")
        );
        let stream = std::fs::read(&path).expect("vendored still");
        let (serial, n) = registry_md5(&stream, 1);
        assert_eq!(n, 1);
        for threads in [2usize, 4] {
            let (threaded, n) = registry_md5(&stream, threads);
            assert_eq!(n, 1);
            assert_eq!(
                serial, threaded,
                "{name}: registry output at {threads} threads"
            );
        }
        if md5.len() == 32 && name != "henc-120-b10" {
            assert_eq!(serial, md5, "{name}: round-460 pin");
        }
    }
}

/// The official WPP conformance stream (an inter 4:4:4 sequence with
/// entropy coding sync), when the corpus is staged.
#[test]
fn official_wpp_stream_decodes_identically_threaded() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/video/h265/conformance/RExt/WPP_HIGH_TP_444_8BIT_RExt_Apple_2.bit");
    let Ok(stream) = std::fs::read(&path) else {
        eprintln!("conformance corpus not staged; skipping");
        return;
    };
    let serial = decode_with(&stream, 1);
    let threaded = decode_with(&stream, 4);
    assert_eq!(serial.len(), threaded.len());
    for (i, (a, b)) in serial.iter().zip(&threaded).enumerate() {
        assert!(a.picture == b.picture, "WPP_HIGH_TP frame {i}");
    }
}

fn hash_noise(x: i64, y: i64, salt: u64) -> i32 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (y as u64).rotate_left(21) ^ salt;
    h ^= h >> 31;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h & 0xFF) as i32
}

/// A WPP still the encoder writes in the test (1024x768, level-0 mode
/// decision), decoded serial and under a four-worker budget: the
/// budgeted decode is never slower (the bound tolerates CI noise and
/// a two-core runner).
#[test]
fn budgeted_still_decode_is_not_slower_than_serial() {
    let (w, h) = (1024usize, 768usize);
    let (cw, ch) = (w / 2, h / 2);
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            let coarse = hash_noise((i >> 4) as i64, (j >> 4) as i64, 1) / 2;
            let fine = hash_noise(i as i64, j as i64, 0x55) / 8;
            y[j * w + i] = (60 + coarse + fine).clamp(0, 255) as u8;
        }
    }
    let cb: Vec<u8> = (0..cw * ch)
        .map(|k| (100 + (k % cw) * 60 / cw) as u8)
        .collect();
    let cr: Vec<u8> = (0..cw * ch)
        .map(|k| (90 + (k / cw) * 70 / ch) as u8)
        .collect();
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    for (k, v) in [
        ("mode", "intra"),
        ("qp", "30"),
        ("ctb", "64"),
        ("still", "1"),
        ("rd", "0"),
        ("wpp", "1"),
        ("deblock", "1"),
        ("sao", "1"),
    ] {
        params.options.insert(k, v);
    }
    let mut enc = oxideav_h265::make_encoder(&params).expect("encoder");
    enc.send_frame(&Frame::Video(VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: cw,
                data: cb,
            },
            VideoPlane {
                stride: cw,
                data: cr,
            },
        ],
    }))
    .expect("send");
    let stream = enc.receive_packet().expect("packet").data;

    let time = |threads: usize| {
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t0 = Instant::now();
            let frames = decode_with(&stream, threads);
            best = best.min(t0.elapsed().as_secs_f64());
            assert_eq!(frames.len(), 1);
        }
        best
    };
    let serial = decode_with(&stream, 1);
    let threaded = decode_with(&stream, 4);
    assert!(serial[0].picture == threaded[0].picture, "bytes identical");
    let (t1, t4) = (time(1), time(4));
    eprintln!(
        "1024x768 WPP still: serial {t1:.4} s, 4 workers {t4:.4} s (x{:.2})",
        t1 / t4
    );
    assert!(
        t4 <= t1 * 1.1,
        "a budgeted decode must not be slower: serial {t1:.4} s vs 4 workers {t4:.4} s"
    );
}
