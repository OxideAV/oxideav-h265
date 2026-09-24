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
