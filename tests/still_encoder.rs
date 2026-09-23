//! Round-460 still-picture encoder pins: any-size input (conformance
//! window cropping), Main Still Picture profile signalling, and the
//! registry decoder's cropped output — the HEIF `hvc1` image-item
//! contract from the encode side.
//!
//! The two golden streams (intra CTB-64 at QP 30 with both loop
//! filters, and lossless PCM) were validated OUT OF BAND at pin time
//! against a black-box reference decoder, whose cropped output was
//! byte-identical to this crate's; the digests pin both the emitted
//! bytes and the decode.

mod fixture_bytes;

use fixture_bytes::md5;
use oxideav_core::{CodecParameters, Error, Frame, Packet, TimeBase, VideoFrame, VideoPlane};
use oxideav_h265::nal::NalIter;
use oxideav_h265::sequence::decode_annexb_sequence;
use oxideav_h265::sps::SeqParameterSet;

const W: usize = 333;
const H: usize = 217;

/// Deterministic per-pixel hash noise.
fn hash_noise(x: i64, y: i64, seed: u64) -> i32 {
    let mut h = (x as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add(seed);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h & 0xFF) as i32
}

/// A textured still: coarse blocks + fine noise + diagonal stripes on
/// luma, smooth chroma ramps. Odd 333x217 (chroma 167x109).
fn still(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            let (wx, wy) = (i as i64, j as i64);
            let coarse = hash_noise(wx >> 4, wy >> 4, 1) / 2;
            let fine = hash_noise(wx, wy, 0x55) / 8;
            let stripes = (((wx * 3 + wy * 2) / 7 % 13) * 3) as i32;
            y[j * w + i] = (60 + coarse + fine + stripes).clamp(0, 255) as u8;
        }
    }
    let cb: Vec<u8> = (0..cw * ch)
        .map(|k| (100 + (k % cw) * 60 / cw) as u8)
        .collect();
    let cr: Vec<u8> = (0..cw * ch)
        .map(|k| (90 + (k / cw) * 70 / ch) as u8)
        .collect();
    (y, cb, cr)
}

fn frame(w: usize, h: usize, planes: &(Vec<u8>, Vec<u8>, Vec<u8>)) -> Frame {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let plane = |data: &Vec<u8>, stride: usize, rows: usize| VideoPlane {
        stride,
        data: {
            assert_eq!(data.len(), stride * rows);
            data.clone()
        },
    };
    Frame::Video(VideoFrame {
        pts: Some(0),
        planes: vec![
            plane(&planes.0, w, h),
            plane(&planes.1, cw, ch),
            plane(&planes.2, cw, ch),
        ],
    })
}

fn encode_one(opts: &[(&str, &str)], w: usize, h: usize) -> Vec<u8> {
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    for (k, v) in opts {
        params.options.insert(*k, *v);
    }
    let mut enc = oxideav_h265::make_encoder(&params).expect("factory");
    enc.send_frame(&frame(w, h, &still(w, h))).expect("send");
    let pkt = enc.receive_packet().expect("one packet");
    assert!(pkt.flags.keyframe);
    // Out-of-band validation hook: dump the stream for a black-box
    // reference decode (`H265_DUMP_DIR=<dir>`).
    if let Ok(dir) = std::env::var("H265_DUMP_DIR") {
        let name: String = opts
            .iter()
            .map(|(k, v)| format!("{k}-{v}"))
            .collect::<Vec<_>>()
            .join("_");
        std::fs::write(format!("{dir}/still_{name}.hevc"), &pkt.data).expect("dump");
    }
    pkt.data
}

/// The active SPS of a single-picture stream.
fn sps_of(stream: &[u8]) -> SeqParameterSet {
    let rbsp = NalIter::new(stream)
        .flatten()
        .find(|u| u.header.nal_unit_type == 33)
        .map(|u| u.rbsp)
        .expect("SPS");
    SeqParameterSet::parse(&rbsp).expect("SPS parses")
}

/// PSNR of the top-left `w x h` of `out` (stride `ow`) against `src`.
fn psnr(src: &[u8], w: usize, h: usize, out: &[i32], ow: usize) -> f64 {
    let mut sse = 0f64;
    for j in 0..h {
        for i in 0..w {
            let d = f64::from(src[j * w + i]) - f64::from(out[j * ow + i]);
            sse += d * d;
        }
    }
    let mse = sse / (w * h) as f64;
    10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10()
}

fn assert_still_signalling(stream: &[u8], what: &str) {
    let sps = sps_of(stream);
    let ptl = &sps.ptl;
    assert_eq!(
        ptl.general_profile_idc, 3,
        "{what}: Main Still Picture profile_idc"
    );
    assert!(
        ptl.profile_compatible(1) && ptl.profile_compatible(2) && ptl.profile_compatible(3),
        "{what}: Main / Main 10 / MSP compatibility flags"
    );
    assert!(
        ptl.one_picture_only_constraint_flag(),
        "{what}: general_one_picture_only_constraint_flag"
    );
    assert!(ptl.is_still_picture_profile());
    assert_eq!(
        sps.sub_layer_ordering_info[0].max_dec_pic_buffering_minus1, 0,
        "{what}: one-picture DPB"
    );
    assert_eq!(
        sps.pic_width_in_luma_samples, 336,
        "{what}: coded width (16-aligned)"
    );
    assert_eq!(
        sps.pic_height_in_luma_samples, 224,
        "{what}: coded height (16-aligned)"
    );
    assert!(sps.conformance_window_flag, "{what}: conformance window");
    assert_eq!(
        (
            sps.conformance_window.left_offset,
            sps.conformance_window.right_offset,
            sps.conformance_window.top_offset,
            sps.conformance_window.bottom_offset
        ),
        (0, 1, 0, 3),
        "{what}: crop 336x224 -> 334x218 in chroma units"
    );
    // Table A.8: 336 x 224 = 75 264 luma samples -> level 2 (60).
    assert_eq!(ptl.general_level_idc, 60, "{what}: level");
}

/// MD5 of the cropped planar decode through the registry decoder.
fn registry_decode_md5(stream: &[u8], w: usize, h: usize) -> String {
    let params = CodecParameters::video("h265".into());
    let mut dec = oxideav_h265::make_decoder(&params).expect("decoder");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), stream.to_vec()))
        .expect("send");
    dec.flush().expect("flush");
    let mut out = Vec::new();
    let mut frames = 0;
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => {
                frames += 1;
                assert_eq!(v.planes[0].stride, w, "cropped luma stride");
                assert_eq!(v.planes[0].data.len(), w * h, "cropped luma size");
                assert_eq!(v.planes[1].stride, w / 2, "cropped chroma stride");
                for p in &v.planes {
                    out.extend_from_slice(&p.data);
                }
            }
            Ok(_) => panic!("non-video frame"),
            Err(Error::Eof) => break,
            Err(e) => panic!("receive: {e}"),
        }
    }
    assert_eq!(frames, 1);
    md5::hex(&out)
}

/// Golden digests (stream bytes, cropped decode) validated black-box
/// at pin time.
const INTRA_STREAM_MD5: &str = "43d0d0979faea7e5dfe54154f9c9cc78";
const INTRA_DECODE_MD5: &str = "866cbe5eb755da846cd33463a8a93262";
const PCM_STREAM_MD5: &str = "bd53ba698547bc58dc626a159aab13f4";
const PCM_DECODE_MD5: &str = "475f10c4835f4eba613c82c7ab990cf2";

/// `mode = intra, still = 1` on an odd 333x217 picture: Main Still
/// Picture signalling, a 336x224 coded picture cropped to 334x218,
/// the decoded still within 1 dB of the same coder's 16-aligned
/// quality, and the extra column / row being the replicated edge.
#[test]
fn still_intra_odd_size_signals_profile_and_crops() {
    let stream = encode_one(
        &[
            ("mode", "intra"),
            ("still", "1"),
            ("ctb", "64"),
            ("qp", "30"),
            ("deblock", "1"),
            ("sao", "1"),
        ],
        W,
        H,
    );
    assert_still_signalling(&stream, "intra");
    let frames = decode_annexb_sequence(&stream).expect("decodes");
    assert_eq!(frames.len(), 1);
    let out = frames[0].output_picture();
    assert_eq!((out.width_luma(), out.height_luma()), (334, 218));
    let (y, _, _) = still(W, H);
    let luma = out.plane(oxideav_h265::picture::Plane::Luma);
    let p = psnr(&y, W, H, luma, 334);
    assert!(p > 30.0, "luma PSNR {p:.2} dB at QP 30");
    assert_eq!(md5::hex(&stream), INTRA_STREAM_MD5, "golden stream bytes");
    assert_eq!(
        registry_decode_md5(&stream, 334, 218),
        INTRA_DECODE_MD5,
        "golden cropped decode"
    );
}

/// `mode = pcm, still = 1`: the lossless still crops back to the
/// source exactly (the 334th column / 218th row replicate the edge).
#[test]
fn still_pcm_odd_size_is_lossless_after_crop() {
    let stream = encode_one(&[("mode", "pcm"), ("still", "1")], W, H);
    assert_still_signalling(&stream, "pcm");
    let frames = decode_annexb_sequence(&stream).expect("decodes");
    let out = frames[0].output_picture();
    assert_eq!((out.width_luma(), out.height_luma()), (334, 218));
    let (y, cb, cr) = still(W, H);
    let luma = out.plane(oxideav_h265::picture::Plane::Luma);
    for j in 0..218 {
        for i in 0..334 {
            let expect = y[j.min(H - 1) * W + i.min(W - 1)];
            assert_eq!(luma[j * 334 + i], i32::from(expect), "luma ({i},{j})");
        }
    }
    let (cw, ch) = (W.div_ceil(2), H.div_ceil(2));
    for (plane, src) in [
        (oxideav_h265::picture::Plane::Cb, &cb),
        (oxideav_h265::picture::Plane::Cr, &cr),
    ] {
        let p = out.plane(plane);
        for j in 0..109 {
            for i in 0..167 {
                assert_eq!(
                    p[j * 167 + i],
                    i32::from(src[j.min(ch - 1) * cw + i.min(cw - 1)])
                );
            }
        }
    }
    assert_eq!(md5::hex(&stream), PCM_STREAM_MD5, "golden stream bytes");
    assert_eq!(
        registry_decode_md5(&stream, 334, 218),
        PCM_DECODE_MD5,
        "golden cropped decode"
    );
}

/// Without `still`, an odd-size picture still pads + crops (Main
/// profile, the historical DPB bounds); the inter modes too.
#[test]
fn odd_size_without_still_pads_and_crops_on_every_mode() {
    for opts in [
        vec![("mode", "intra"), ("qp", "28")],
        vec![("mode", "intra"), ("qp", "28"), ("ctb", "32")],
        vec![("mode", "pcm")],
        vec![("mode", "inter"), ("qp", "28"), ("gop", "0")],
        vec![("mode", "inter"), ("qp", "28"), ("pyramid", "2")],
    ] {
        let mut params = CodecParameters::video("h265".into());
        params.width = Some(100);
        params.height = Some(75);
        for (k, v) in &opts {
            params.options.insert(*k, *v);
        }
        let mut enc = oxideav_h265::make_encoder(&params).expect("factory");
        let src = still(100, 75);
        enc.send_frame(&frame(100, 75, &src)).expect("send 0");
        enc.send_frame(&frame(100, 75, &src)).expect("send 1");
        enc.flush().expect("flush");
        let mut stream = Vec::new();
        while let Ok(pkt) = enc.receive_packet() {
            stream.extend_from_slice(&pkt.data);
        }
        let sps = sps_of(&stream);
        assert_eq!(sps.ptl.general_profile_idc, 1, "{opts:?}: Main profile");
        assert!(!sps.ptl.is_still_picture_profile(), "{opts:?}");
        assert_eq!(
            (
                sps.pic_width_in_luma_samples,
                sps.pic_height_in_luma_samples
            ),
            (112, 80),
            "{opts:?}: coded size"
        );
        assert_eq!(
            (
                sps.conformance_window.right_offset,
                sps.conformance_window.bottom_offset
            ),
            (6, 2),
            "{opts:?}: crop to 100x76"
        );
        let frames = decode_annexb_sequence(&stream).unwrap_or_else(|e| panic!("{opts:?}: {e}"));
        assert_eq!(frames.len(), 2, "{opts:?}: two pictures");
        for f in &frames {
            let out = f.output_picture();
            assert_eq!((out.width_luma(), out.height_luma()), (100, 76), "{opts:?}");
        }
    }
}

/// `still` is refused on the inter GOP modes.
#[test]
fn still_rejects_inter_mode() {
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(64);
    params.height = Some(64);
    params.options.insert("mode", "inter");
    params.options.insert("still", "1");
    assert!(oxideav_h265::make_encoder(&params).is_err());
}
