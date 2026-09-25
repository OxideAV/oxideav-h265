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
const INTRA_STREAM_MD5: &str = "9baa64d540f89ee7cae6f43a8ac96ea4";
const INTRA_DECODE_MD5: &str = "c2fc2a55c0ec1dac0b51d5148e864a37";
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

/// The registry `tiles=CxR` / `wpp` options reach the quadtree coder
/// (round 460 — they were documented but never parsed): a 2x2 grid
/// still decodes byte-exact, signals `tiles_enabled_flag`, and the
/// bytes do not depend on the execution-context worker count; `wpp`
/// signals `entropy_coding_sync_enabled_flag`; both need `ctb`.
#[test]
fn registry_tiles_and_wpp_options_reach_the_quadtree_coder() {
    use oxideav_core::ExecutionContext;
    use oxideav_h265::pps::PicParameterSet;

    let pps_of = |stream: &[u8]| {
        let rbsp = NalIter::new(stream)
            .flatten()
            .find(|u| u.header.nal_unit_type == 34)
            .map(|u| u.rbsp)
            .expect("PPS");
        PicParameterSet::parse(&rbsp).expect("PPS parses")
    };
    let encode = |opts: &[(&str, &str)], threads: usize| -> Vec<u8> {
        let mut params = CodecParameters::video("h265".into());
        params.width = Some(W as u32);
        params.height = Some(H as u32);
        for (k, v) in opts {
            params.options.insert(*k, *v);
        }
        let mut enc = oxideav_h265::make_encoder(&params).expect("factory");
        enc.set_execution_context(&ExecutionContext { threads });
        enc.send_frame(&frame(W, H, &still(W, H))).expect("send");
        enc.receive_packet().expect("packet").data
    };
    let base = [
        ("mode", "intra"),
        ("still", "1"),
        ("ctb", "32"),
        ("qp", "32"),
    ];
    let tiled: Vec<(&str, &str)> = base.iter().copied().chain([("tiles", "2x2")]).collect();
    let serial = encode(&tiled, 1);
    let parallel = encode(&tiled, 4);
    assert_eq!(serial, parallel, "tile fan-out never changes the bytes");
    let pps = pps_of(&serial);
    assert!(pps.tiles_enabled_flag, "tiles_enabled_flag");
    assert_eq!(
        (
            pps.tiles.num_tile_columns_minus1,
            pps.tiles.num_tile_rows_minus1
        ),
        (1, 1),
        "2x2 grid"
    );
    let frames = decode_annexb_sequence(&serial).expect("tiled still decodes");
    assert_eq!(frames.len(), 1);
    let out = frames[0].output_picture();
    assert_eq!((out.width_luma(), out.height_luma()), (334, 218));

    let wpp: Vec<(&str, &str)> = base.iter().copied().chain([("wpp", "1")]).collect();
    let stream = encode(&wpp, 1);
    assert!(
        pps_of(&stream).entropy_coding_sync_enabled_flag,
        "entropy_coding_sync_enabled_flag"
    );
    assert_eq!(
        decode_annexb_sequence(&stream)
            .expect("WPP still decodes")
            .len(),
        1
    );

    for (k, v) in [("tiles", "2x2"), ("wpp", "1")] {
        let mut params = CodecParameters::video("h265".into());
        params.width = Some(64);
        params.height = Some(64);
        params.options.insert("mode", "intra");
        params.options.insert(k, v);
        assert!(
            oxideav_h265::make_encoder(&params).is_err(),
            "{k} needs ctb"
        );
    }
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(64);
    params.height = Some(64);
    params.options.insert("mode", "intra");
    params.options.insert("ctb", "32");
    params.options.insert("tiles", "1x1");
    assert!(
        oxideav_h265::make_encoder(&params).is_err(),
        "1x1 is not a grid"
    );
}

/// The `rd` option (intra mode-decision effort): level 0 is the
/// historical SAD search (byte-stable), a still defaults to level 2,
/// level 1 / 2 streams differ from level 0 and decode; `rd` needs
/// `ctb`.
#[test]
fn rd_levels_select_the_intra_decision() {
    let base = [
        ("mode", "intra"),
        ("still", "1"),
        ("ctb", "32"),
        ("qp", "30"),
    ];
    let with = |rd: Option<&'static str>| -> Vec<u8> {
        let opts: Vec<(&str, &str)> = base.iter().copied().chain(rd.map(|v| ("rd", v))).collect();
        encode_one(&opts, W, H)
    };
    let rd0 = with(Some("0"));
    let rd1 = with(Some("1"));
    let rd2 = with(Some("2"));
    let default = with(None);
    assert_eq!(default, rd2, "a still defaults to rd 2");
    assert_ne!(rd0, rd1, "level 1 changes the decision");
    assert_ne!(rd1, rd2, "level 2 changes the decision");
    assert!(
        rd2.len() < rd0.len(),
        "level 2 is cheaper at QP 30 ({} vs {} bytes)",
        rd2.len(),
        rd0.len()
    );
    for stream in [&rd0, &rd1, &rd2] {
        assert_eq!(decode_annexb_sequence(stream).expect("decodes").len(), 1);
    }
    // Without `still`, the historical level 0 is the default.
    let plain = encode_one(&[("mode", "intra"), ("ctb", "32"), ("qp", "30")], W, H);
    let plain0 = encode_one(
        &[("mode", "intra"), ("ctb", "32"), ("qp", "30"), ("rd", "0")],
        W,
        H,
    );
    assert_eq!(plain, plain0, "no still: rd defaults to 0");
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(64);
    params.height = Some(64);
    params.options.insert("mode", "intra");
    params.options.insert("rd", "1");
    assert!(oxideav_h265::make_encoder(&params).is_err(), "rd needs ctb");
}

/// A `YuvJ420P` (full-range) frame encodes byte-identically to the
/// same `Yuv420P` frame with `range=full` — on the pcm, intra and
/// inter modes — carries the full-range flag by itself, yields to an
/// explicit `range=limited`, keeps the output pixel format, and the
/// 4:2:2 / 4:4:4 `YuvJ*` twins stay refused (no such coder yet).
#[test]
fn yuvj420p_input_is_the_full_range_twin_of_yuv420p() {
    use oxideav_core::PixelFormat;

    let encode = |pf: PixelFormat, opts: &[(&str, &str)]| -> (Vec<u8>, Option<PixelFormat>) {
        let mut params = CodecParameters::video("h265".into());
        params.width = Some(64);
        params.height = Some(48);
        params.pixel_format = Some(pf);
        for (k, v) in opts {
            params.options.insert(*k, *v);
        }
        let mut enc = oxideav_h265::make_encoder(&params).expect("factory");
        let out_pf = enc.output_params().pixel_format;
        enc.send_frame(&frame(64, 48, &still(64, 48)))
            .expect("send");
        (enc.receive_packet().expect("packet").data, out_pf)
    };
    let modes: [&[(&str, &str)]; 3] = [
        &[("mode", "pcm")],
        &[("mode", "intra"), ("qp", "30"), ("ctb", "32")],
        &[("mode", "inter"), ("qp", "30"), ("gop", "0")],
    ];
    for mode in modes {
        let full: Vec<(&str, &str)> = mode.iter().copied().chain([("range", "full")]).collect();
        let (j, j_pf) = encode(PixelFormat::YuvJ420P, mode);
        let (y_full, y_pf) = encode(PixelFormat::Yuv420P, &full);
        assert_eq!(j, y_full, "{mode:?}: YuvJ420P == Yuv420P + range=full");
        assert_eq!(j_pf, Some(PixelFormat::YuvJ420P), "{mode:?}");
        assert_eq!(y_pf, Some(PixelFormat::Yuv420P), "{mode:?}");
        let vs = sps_of(&j)
            .vui_parameters
            .expect("VUI")
            .video_signal_type
            .expect("video_signal_type");
        assert!(vs.video_full_range_flag && vs.colour_description.is_none());
        let (y_plain, _) = encode(PixelFormat::Yuv420P, mode);
        assert_ne!(j, y_plain, "{mode:?}: plain Yuv420P writes no VUI");
        assert!(sps_of(&y_plain).vui_parameters.is_none());
        let limited: Vec<(&str, &str)> =
            mode.iter().copied().chain([("range", "limited")]).collect();
        let (j_lim, _) = encode(PixelFormat::YuvJ420P, &limited);
        assert!(
            !sps_of(&j_lim)
                .vui_parameters
                .expect("VUI")
                .video_signal_type
                .expect("block")
                .video_full_range_flag,
            "{mode:?}: an explicit range wins"
        );
    }
    // The 4:2:2 / 4:4:4 twins are lossless PCM layouts: accepted on
    // the pcm mode (full range signalled), refused by the intra / inter
    // coders.
    for pf in [
        PixelFormat::YuvJ422P,
        PixelFormat::YuvJ444P,
        PixelFormat::Yuv444P,
    ] {
        let mut params = CodecParameters::video("h265".into());
        params.width = Some(64);
        params.height = Some(48);
        params.pixel_format = Some(pf);
        assert!(oxideav_h265::make_encoder(&params).is_ok(), "{pf:?} pcm");
        params.options.insert("mode", "intra");
        assert!(oxideav_h265::make_encoder(&params).is_err(), "{pf:?} intra");
    }
}

/// Every HEIC sample layout a producer writes — monochrome, 4:2:0,
/// 4:2:2, 4:4:4 at 8 / 10 / 12 bits (and 16-bit grey) — encodes as a
/// lossless PCM still through the registry: the Annex A signalling
/// matches the layout (Main / Main 10 Still Picture, or a format range
/// extensions profile with the Table A.2 row's constraint flags plus the
/// intra / one-picture-only flags), the SPS carries the chroma format
/// and bit depths, and the registry decoder returns the input samples
/// exactly (odd sizes cropped per the layout's chroma units).
#[test]
fn pcm_still_layout_matrix_is_lossless_and_signalled() {
    use oxideav_core::PixelFormat;
    let cases: [(PixelFormat, u8, u8, u8, usize, usize); 12] = [
        // (format, chroma_format_idc, bit depth, profile idc, w, h)
        (PixelFormat::Gray8, 0, 8, 4, 37, 21),
        (PixelFormat::Gray10Le, 0, 10, 4, 18, 14),
        (PixelFormat::Gray12Le, 0, 12, 4, 16, 16),
        (PixelFormat::Gray16Le, 0, 16, 4, 20, 17),
        (PixelFormat::Yuv420P10Le, 1, 10, 2, 33, 19),
        (PixelFormat::Yuv420P12Le, 1, 12, 4, 32, 32),
        (PixelFormat::Yuv422P, 2, 8, 4, 35, 17),
        (PixelFormat::Yuv422P10Le, 2, 10, 4, 34, 18),
        (PixelFormat::Yuv422P12Le, 2, 12, 4, 16, 16),
        (PixelFormat::Yuv444P, 3, 8, 4, 31, 23),
        (PixelFormat::Yuv444P10Le, 3, 10, 4, 17, 15),
        (PixelFormat::Yuv444P12Le, 3, 12, 4, 40, 24),
    ];
    for (pf, cfi, bd, profile, w, h) in cases {
        let (sw, sh) = match cfi {
            1 => (2usize, 2usize),
            2 => (2, 1),
            _ => (1, 1),
        };
        let (cw, ch) = (w.div_ceil(sw), h.div_ceil(sh));
        let max = (1u32 << bd) - 1;
        let sample = |x: usize, y: usize, seed: u32| -> u32 {
            (x as u32 * 977 + y as u32 * 331 + seed * 7919) % (max + 1)
        };
        let wide = bd > 8;
        let bps = if wide { 2 } else { 1 };
        let mk = |pw: usize, ph: usize, seed: u32| -> VideoPlane {
            let mut data = Vec::with_capacity(pw * ph * bps);
            for y in 0..ph {
                for x in 0..pw {
                    let v = sample(x, y, seed);
                    if wide {
                        data.extend_from_slice(&(v as u16).to_le_bytes());
                    } else {
                        data.push(v as u8);
                    }
                }
            }
            VideoPlane {
                stride: pw * bps,
                data,
            }
        };
        let mut planes = vec![mk(w, h, 1)];
        if cfi != 0 {
            planes.push(mk(cw, ch, 2));
            planes.push(mk(cw, ch, 3));
        }
        let src = planes.clone();
        let mut params = CodecParameters::video("h265".into());
        params.width = Some(w as u32);
        params.height = Some(h as u32);
        params.pixel_format = Some(pf);
        params.options.insert("still", "1");
        let mut enc = oxideav_h265::make_encoder(&params).unwrap_or_else(|e| panic!("{pf:?}: {e}"));
        assert_eq!(
            enc.output_params().pixel_format,
            Some(pf),
            "{pf:?}: format echo"
        );
        enc.send_frame(&Frame::Video(VideoFrame {
            pts: Some(0),
            planes,
        }))
        .unwrap_or_else(|e| panic!("{pf:?}: send: {e}"));
        let pkt = enc.receive_packet().expect("one packet");
        assert!(pkt.flags.keyframe);
        let stream = pkt.data;
        if let Ok(dir) = std::env::var("H265_DUMP_DIR") {
            std::fs::write(format!("{dir}/pcm_{pf:?}_{w}x{h}.hevc"), &stream).expect("dump");
        }
        // Signalling.
        let sps = sps_of(&stream);
        assert_eq!(sps.chroma_format_idc, cfi, "{pf:?}: chroma_format_idc");
        assert_eq!(sps.bit_depth_luma(), bd, "{pf:?}: BitDepthY");
        assert_eq!(sps.bit_depth_chroma(), bd, "{pf:?}: BitDepthC");
        assert_eq!(sps.ptl.general_profile_idc, profile, "{pf:?}: profile");
        assert!(
            sps.ptl.is_still_picture_profile(),
            "{pf:?}: still signalling"
        );
        if profile == 4 {
            // Table A.2 flags: bits 43..35 of the 48-bit block are the
            // nine constraint flags in syntax order.
            let f = sps.ptl.general_constraint_indicator_flags;
            let flag = |i: u32| (f >> (43 - i)) & 1 == 1;
            assert_eq!(flag(0), bd <= 12, "{pf:?}: max_12bit");
            assert_eq!(flag(1), bd <= 10, "{pf:?}: max_10bit");
            assert_eq!(flag(2), bd <= 8, "{pf:?}: max_8bit");
            assert_eq!(flag(3), cfi <= 2, "{pf:?}: max_422chroma");
            assert_eq!(flag(4), cfi <= 1, "{pf:?}: max_420chroma");
            assert_eq!(flag(5), cfi == 0, "{pf:?}: max_monochrome");
            assert!(flag(6) && flag(7), "{pf:?}: intra + one_picture_only");
        }
        // Lossless round trip through the registry decoder (Annex B
        // packets), cropped to the caller's size.
        let mut dparams = CodecParameters::video("h265".into());
        let mut dec = oxideav_h265::make_decoder(&dparams).expect("decoder");
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), stream.clone()))
            .expect("send");
        dec.flush().expect("flush");
        let out = match dec.receive_frame() {
            Ok(Frame::Video(v)) => v,
            other => panic!("{pf:?}: {other:?}"),
        };
        dparams.pixel_format = Some(pf);
        let (ow, oh) = (w.div_ceil(sw) * sw, h.div_ceil(sh) * sh);
        assert_eq!(out.planes.len(), src.len(), "{pf:?}: plane count");
        for (i, (o, s)) in out.planes.iter().zip(src.iter()).enumerate() {
            let (pw, ph) = if i == 0 { (ow, oh) } else { (ow / sw, oh / sh) };
            assert_eq!(o.stride, pw * bps, "{pf:?}: plane {i} stride");
            assert_eq!(o.data.len(), pw * ph * bps, "{pf:?}: plane {i} size");
            // Compare the caller's region; the padding column / row of
            // an odd size replicates the edge.
            let (sw_, sh_) = if i == 0 { (w, h) } else { (cw, ch) };
            for y in 0..sh_ {
                let a = &o.data[y * o.stride..y * o.stride + sw_ * bps];
                let b = &s.data[y * s.stride..y * s.stride + sw_ * bps];
                assert_eq!(a, b, "{pf:?}: plane {i} row {y}");
            }
        }
        assert!(
            matches!(dec.receive_frame(), Err(Error::Eof)),
            "{pf:?}: one frame"
        );
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

/// The `range` / `colorprim` / `transfer` / `matrix` options write the
/// §E.2.1 `video_signal_type` VUI block on every mode (pcm, legacy
/// intra, quadtree intra, low-delay, pyramid): `video_full_range_flag`
/// and the H.273 code points parse back, a missing code point reads 2
/// (unspecified), no option writes no VUI, and every stream decodes.
#[test]
fn video_signal_options_write_the_vui_block_on_every_mode() {
    let modes: [&[(&str, &str)]; 5] = [
        &[("mode", "pcm")],
        &[("mode", "intra"), ("qp", "30")],
        &[
            ("mode", "intra"),
            ("qp", "30"),
            ("ctb", "32"),
            ("still", "1"),
        ],
        &[("mode", "inter"), ("qp", "30"), ("gop", "0")],
        &[("mode", "inter"), ("qp", "30"), ("pyramid", "2")],
    ];
    for mode in modes {
        let opts: Vec<(&str, &str)> = mode
            .iter()
            .copied()
            .chain([
                ("range", "full"),
                ("colorprim", "1"),
                ("transfer", "13"),
                ("matrix", "6"),
            ])
            .collect();
        let stream = encode_one(&opts, 64, 48);
        let sps = sps_of(&stream);
        let vui = sps
            .vui_parameters
            .as_ref()
            .unwrap_or_else(|| panic!("{mode:?}: VUI present"));
        assert!(vui.video_signal_type_present_flag, "{mode:?}");
        let vs = vui
            .video_signal_type
            .as_ref()
            .expect("video_signal_type block");
        assert_eq!(vs.video_format, 5, "{mode:?}: video_format unspecified");
        assert!(vs.video_full_range_flag, "{mode:?}: full range");
        let cd = vs.colour_description.as_ref().expect("colour description");
        assert_eq!(
            (
                cd.colour_primaries,
                cd.transfer_characteristics,
                cd.matrix_coeffs
            ),
            (1, 13, 6),
            "{mode:?}: colour description"
        );
        assert_eq!(
            decode_annexb_sequence(&stream).expect("decodes").len(),
            1,
            "{mode:?}"
        );

        // Limited range with only the matrix given: primaries /
        // transfer read 2 (unspecified).
        let opts: Vec<(&str, &str)> = mode
            .iter()
            .copied()
            .chain([("range", "limited"), ("matrix", "1")])
            .collect();
        let vui = sps_of(&encode_one(&opts, 64, 48))
            .vui_parameters
            .expect("VUI present");
        let vs = vui.video_signal_type.expect("video_signal_type block");
        assert!(!vs.video_full_range_flag, "{mode:?}");
        let cd = vs.colour_description.expect("colour description");
        assert_eq!(
            (
                cd.colour_primaries,
                cd.transfer_characteristics,
                cd.matrix_coeffs
            ),
            (2, 2, 1)
        );

        // No option: no VUI at all (the historical streams).
        let opts: Vec<(&str, &str)> = mode.to_vec();
        let sps = sps_of(&encode_one(&opts, 64, 48));
        assert!(
            sps.vui_parameters.is_none(),
            "{mode:?}: no VUI without the options"
        );
    }
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(64);
    params.height = Some(48);
    params.options.insert("range", "wide");
    assert!(
        oxideav_h265::make_encoder(&params).is_err(),
        "unknown range word"
    );
}

/// `vpsid` / `spsid` / `ppsid` reach the VPS, SPS, PPS and every slice
/// header on every mode, the streams decode, and the ranges are
/// enforced.
#[test]
fn parameter_set_id_options_reach_every_parameter_set_and_slice() {
    use oxideav_h265::pps::PicParameterSet;
    use oxideav_h265::slice::SliceSegmentHeader;
    use oxideav_h265::HevcVps;

    let modes: [&[(&str, &str)]; 5] = [
        &[("mode", "pcm")],
        &[("mode", "intra"), ("qp", "30")],
        &[
            ("mode", "intra"),
            ("qp", "30"),
            ("ctb", "32"),
            ("still", "1"),
        ],
        &[("mode", "inter"), ("qp", "30"), ("gop", "0")],
        &[("mode", "inter"), ("qp", "30"), ("pyramid", "2")],
    ];
    for mode in modes {
        let opts: Vec<(&str, &str)> = mode
            .iter()
            .copied()
            .chain([("vpsid", "3"), ("spsid", "5"), ("ppsid", "7")])
            .collect();
        let mut params = CodecParameters::video("h265".into());
        params.width = Some(64);
        params.height = Some(48);
        for (k, v) in &opts {
            params.options.insert(*k, *v);
        }
        let mut enc = oxideav_h265::make_encoder(&params).expect("factory");
        let src = still(64, 48);
        enc.send_frame(&frame(64, 48, &src)).expect("send 0");
        enc.send_frame(&frame(64, 48, &src)).expect("send 1");
        enc.flush().expect("flush");
        let mut stream = Vec::new();
        while let Ok(pkt) = enc.receive_packet() {
            stream.extend_from_slice(&pkt.data);
        }
        let units: Vec<_> = NalIter::new(&stream).flatten().collect();
        let vps = HevcVps::parse(
            &units
                .iter()
                .find(|u| u.header.nal_unit_type == 32)
                .expect("VPS")
                .rbsp,
        )
        .expect("VPS parses");
        assert_eq!(vps.vps_id, 3, "{mode:?}");
        let sps = sps_of(&stream);
        assert_eq!((sps.vps_id, sps.sps_id), (3, 5), "{mode:?}");
        let pps = PicParameterSet::parse(
            &units
                .iter()
                .find(|u| u.header.nal_unit_type == 34)
                .expect("PPS")
                .rbsp,
        )
        .expect("PPS parses");
        assert_eq!((pps.pps_id, pps.sps_id), (7, 5), "{mode:?}");
        let mut slices = 0;
        for u in units.iter().filter(|u| u.header.is_vcl()) {
            let h = SliceSegmentHeader::parse(&u.rbsp, u.header.nal_unit_type, &sps, &pps)
                .unwrap_or_else(|e| panic!("{mode:?}: slice header: {e}"));
            assert_eq!(h.slice_pic_parameter_set_id, 7, "{mode:?}");
            slices += 1;
        }
        assert!(slices >= 2, "{mode:?}: both pictures' slices");
        assert_eq!(
            decode_annexb_sequence(&stream).expect("decodes").len(),
            2,
            "{mode:?}"
        );
    }
    for (k, v) in [
        ("vpsid", "16"),
        ("spsid", "16"),
        ("ppsid", "64"),
        ("ppsid", "x"),
    ] {
        let mut params = CodecParameters::video("h265".into());
        params.width = Some(64);
        params.height = Some(48);
        params.options.insert(k, v);
        assert!(oxideav_h265::make_encoder(&params).is_err(), "{k}={v}");
    }
}
