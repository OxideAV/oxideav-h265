//! Registry-contract roundtrip: `oxideav_core::Encoder` (PCM-only IDR
//! bootstrap) → `oxideav_core::Decoder`, bit-exact lossless.

use oxideav_core::{CodecParameters, Error, Frame, PixelFormat, VideoFrame, VideoPlane};

fn planes(w: usize, h: usize, seed: u8) -> VideoFrame {
    let mk = |wp: usize, hp: usize, mul: usize| VideoPlane {
        stride: wp,
        data: (0..wp * hp).map(|i| (i * mul % 253) as u8 ^ seed).collect(),
    };
    VideoFrame {
        pts: None,
        planes: vec![mk(w, h, 7), mk(w / 2, h / 2, 3), mk(w / 2, h / 2, 11)],
    }
}

#[test]
fn encoder_to_decoder_roundtrip_is_lossless() {
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(48);
    params.height = Some(32);
    params.pixel_format = Some(PixelFormat::Yuv420P);

    let mut enc = oxideav_h265::make_encoder(&params).expect("encoder factory");
    let frames: Vec<VideoFrame> = (0..3).map(|i| planes(48, 32, i * 0x35)).collect();
    let mut aus = Vec::new();
    for f in &frames {
        enc.send_frame(&Frame::Video(f.clone())).expect("send");
        let pkt = enc.receive_packet().expect("packet per frame");
        assert!(pkt.flags.keyframe, "every PCM AU is a random access point");
        aus.push(pkt);
    }
    enc.flush().expect("flush");
    assert!(
        matches!(enc.receive_packet(), Err(Error::NeedMore)),
        "nothing buffered"
    );

    // Decode the concatenated stream back through the registry decoder.
    let dec_params = CodecParameters::video("h265".into());
    let mut dec = oxideav_h265::make_decoder(&dec_params).expect("decoder factory");
    for pkt in &aus {
        dec.send_packet(pkt).expect("decode send");
    }
    dec.flush().expect("decode flush");
    for (i, f) in frames.iter().enumerate() {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => {
                assert_eq!(v.planes.len(), 3, "frame {i}");
                for (p, q) in v.planes.iter().zip(f.planes.iter()) {
                    assert_eq!(p.data, q.data, "frame {i} plane bit-exact");
                }
            }
            other => panic!("frame {i}: unexpected {other:?}"),
        }
    }
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
}

#[test]
fn encoder_rejects_unaligned_dimensions() {
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(50);
    params.height = Some(32);
    assert!(oxideav_h265::make_encoder(&params).is_err());
}

/// `mode = "intra"`: the registry encoder runs the real CABAC intra
/// coder. Multi-frame streams decode through the registry decoder to
/// high-fidelity (not bit-exact — it is a lossy transform coder) at
/// low QP, and a bad `qp` / `mode` option is rejected at construction.
#[test]
fn intra_mode_registry_roundtrip_is_high_fidelity() {
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(48);
    params.height = Some(32);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.options.insert("mode", "intra");
    params.options.insert("qp", "8");

    let mut enc = oxideav_h265::make_encoder(&params).expect("encoder factory");
    let frames: Vec<VideoFrame> = (0..2).map(|i| planes(48, 32, i * 0x21)).collect();
    let mut aus = Vec::new();
    for f in &frames {
        enc.send_frame(&Frame::Video(f.clone())).expect("send");
        let pkt = enc.receive_packet().expect("packet per frame");
        assert!(pkt.flags.keyframe, "every intra AU is an IDR");
        aus.push(pkt);
    }

    let dec_params = CodecParameters::video("h265".into());
    let mut dec = oxideav_h265::make_decoder(&dec_params).expect("decoder factory");
    for pkt in &aus {
        dec.send_packet(pkt).expect("decode send");
    }
    dec.flush().expect("decode flush");
    for (i, f) in frames.iter().enumerate() {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => {
                assert_eq!(v.planes.len(), 3, "frame {i}");
                for (pi, (p, q)) in v.planes.iter().zip(f.planes.iter()).enumerate() {
                    let mse: f64 = p
                        .data
                        .iter()
                        .zip(q.data.iter())
                        .map(|(&a, &b)| {
                            let d = f64::from(a) - f64::from(b);
                            d * d
                        })
                        .sum::<f64>()
                        / p.data.len() as f64;
                    let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10();
                    assert!(
                        psnr > 38.0,
                        "frame {i} plane {pi}: PSNR {psnr:.1} dB at qp 8"
                    );
                }
            }
            other => panic!("frame {i}: unexpected {other:?}"),
        }
    }
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
}

#[test]
fn encoder_rejects_bad_options() {
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(16);
    params.height = Some(16);
    params.options.insert("mode", "interpretive-dance");
    assert!(oxideav_h265::make_encoder(&params).is_err());

    let mut params = CodecParameters::video("h265".into());
    params.width = Some(16);
    params.height = Some(16);
    params.options.insert("mode", "intra");
    params.options.insert("qp", "52");
    assert!(oxideav_h265::make_encoder(&params).is_err());
}

#[test]
fn inter_mode_registry_gop_roundtrip() {
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(48);
    params.height = Some(32);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.options.insert("mode", "inter");
    params.options.insert("qp", "12");
    params.options.insert("gop", "3");

    let mut enc = oxideav_h265::make_encoder(&params).expect("encoder factory");
    // Slowly-evolving content so P frames actually reference.
    let frames: Vec<VideoFrame> = (0..5).map(|i| planes(48, 32, i)).collect();
    let mut aus = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        enc.send_frame(&Frame::Video(f.clone())).expect("send");
        let pkt = enc.receive_packet().expect("packet per frame");
        // gop = 3: frames 0 and 3 are IDR, the rest P.
        assert_eq!(
            pkt.flags.keyframe,
            i % 3 == 0,
            "frame {i} keyframe flag (gop 3)"
        );
        aus.push(pkt);
    }

    let dec_params = CodecParameters::video("h265".into());
    let mut dec = oxideav_h265::make_decoder(&dec_params).expect("decoder factory");
    for pkt in &aus {
        dec.send_packet(pkt).expect("decode send");
    }
    dec.flush().expect("decode flush");
    for (i, f) in frames.iter().enumerate() {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => {
                assert_eq!(v.planes.len(), 3, "frame {i}");
                for (pi, (p, q)) in v.planes.iter().zip(f.planes.iter()).enumerate() {
                    let mse: f64 = p
                        .data
                        .iter()
                        .zip(q.data.iter())
                        .map(|(&a, &b)| {
                            let d = f64::from(a) - f64::from(b);
                            d * d
                        })
                        .sum::<f64>()
                        / p.data.len() as f64;
                    let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10();
                    assert!(
                        psnr > 34.0,
                        "frame {i} plane {pi}: PSNR {psnr:.1} dB at qp 12"
                    );
                }
            }
            other => panic!("frame {i}: unexpected {other:?}"),
        }
    }
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
}

#[test]
fn inter_mode_amp_registry_roundtrip() {
    // The `amp` option switches the stream to the AMP configuration
    // (MinCb 8 + amp_enabled_flag + the asymmetric shapes in the CU
    // ladder); the registry wiring and fidelity are exercised here —
    // the bit-exact encoder-recon contract is pinned by the unit
    // tests.
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(48);
    params.height = Some(32);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.options.insert("mode", "inter");
    params.options.insert("qp", "12");
    params.options.insert("amp", "1");

    let mut enc = oxideav_h265::make_encoder(&params).expect("encoder factory");
    let frames: Vec<VideoFrame> = (0..4).map(|i| planes(48, 32, i)).collect();
    let mut aus = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        enc.send_frame(&Frame::Video(f.clone())).expect("send");
        let pkt = enc.receive_packet().expect("packet per frame");
        assert_eq!(pkt.flags.keyframe, i == 0, "frame {i} keyframe flag");
        aus.push(pkt);
    }
    // The SPS signals the AMP geometry.
    let units = oxideav_h265::collect_nal_units(&aus[0].data).expect("walk");
    let sps = units
        .iter()
        .find(|u| u.header.nal_unit_type == 33)
        .expect("SPS in the IDR AU");
    let sps = oxideav_h265::SeqParameterSet::parse(&sps.rbsp).expect("sps parse");
    assert!(sps.amp_enabled_flag);
    assert_eq!(sps.log2_min_luma_coding_block_size_minus3, 0);

    let dec_params = CodecParameters::video("h265".into());
    let mut dec = oxideav_h265::make_decoder(&dec_params).expect("decoder factory");
    for pkt in &aus {
        dec.send_packet(pkt).expect("decode send");
    }
    dec.flush().expect("decode flush");
    for (i, f) in frames.iter().enumerate() {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => {
                for (pi, (p, q)) in v.planes.iter().zip(f.planes.iter()).enumerate() {
                    let mse: f64 = p
                        .data
                        .iter()
                        .zip(q.data.iter())
                        .map(|(&a, &b)| {
                            let d = f64::from(a) - f64::from(b);
                            d * d
                        })
                        .sum::<f64>()
                        / p.data.len() as f64;
                    let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10();
                    assert!(
                        psnr > 34.0,
                        "frame {i} plane {pi}: PSNR {psnr:.1} dB at qp 12 (AMP)"
                    );
                }
            }
            other => panic!("frame {i}: unexpected {other:?}"),
        }
    }
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
}

#[test]
fn amp_option_requires_inter_mode() {
    for mode in [None, Some("pcm"), Some("intra")] {
        let mut params = CodecParameters::video("h265".into());
        params.width = Some(16);
        params.height = Some(16);
        if let Some(m) = mode {
            params.options.insert("mode", m);
        }
        params.options.insert("amp", "1");
        assert!(
            oxideav_h265::make_encoder(&params).is_err(),
            "mode {mode:?} must reject the amp option"
        );
    }
}

#[test]
fn pyramid_registry_roundtrip() {
    // mode=inter + pyramid=4: hierarchical-B coding. Packets arrive
    // in bursts per mini-GOP (decode order), dts trails pts by
    // log2(gop) frames, and flush() emits the low-delay tail. The
    // registry decoder's reorder queue restores display order.
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(48);
    params.height = Some(32);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.options.insert("mode", "inter");
    params.options.insert("qp", "12");
    params.options.insert("pyramid", "4");

    let mut enc = oxideav_h265::make_encoder(&params).expect("encoder factory");
    let frames: Vec<VideoFrame> = (0..6).map(|i| planes(48, 32, i)).collect();
    let mut aus = Vec::new();
    for f in &frames {
        enc.send_frame(&Frame::Video(f.clone())).expect("send");
        while let Ok(pkt) = enc.receive_packet() {
            aus.push(pkt);
        }
    }
    enc.flush().expect("flush");
    while let Ok(pkt) = enc.receive_packet() {
        aus.push(pkt);
    }
    assert_eq!(aus.len(), 6, "one packet per pushed frame after flush");
    // Decode order: IDR 0, then mini-GOP {4, 2, 1, 3}, then tail 5.
    let ptss: Vec<i64> = aus.iter().map(|p| p.pts.expect("pts")).collect();
    assert_eq!(ptss, vec![0, 4, 2, 1, 3, 5], "decode-order pts");
    for (k, pkt) in aus.iter().enumerate() {
        assert_eq!(pkt.flags.keyframe, k == 0, "packet {k} keyframe flag");
        let dts = pkt.dts.expect("dts");
        assert_eq!(dts, k as i64 - 2, "packet {k} dts (delay log2(4) = 2)");
        assert!(dts <= pkt.pts.unwrap(), "packet {k}: dts <= pts");
    }

    let dec_params = CodecParameters::video("h265".into());
    let mut dec = oxideav_h265::make_decoder(&dec_params).expect("decoder factory");
    for pkt in &aus {
        dec.send_packet(pkt).expect("decode send");
    }
    dec.flush().expect("decode flush");
    for (i, f) in frames.iter().enumerate() {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => {
                for (pi, (p, q)) in v.planes.iter().zip(f.planes.iter()).enumerate() {
                    let mse: f64 = p
                        .data
                        .iter()
                        .zip(q.data.iter())
                        .map(|(&a, &b)| {
                            let d = f64::from(a) - f64::from(b);
                            d * d
                        })
                        .sum::<f64>()
                        / p.data.len() as f64;
                    let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10();
                    assert!(
                        psnr > 33.0,
                        "frame {i} plane {pi}: PSNR {psnr:.1} dB at qp 12 (pyramid)"
                    );
                }
            }
            other => panic!("frame {i}: unexpected {other:?}"),
        }
    }
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
}

#[test]
fn pyramid_option_validation() {
    let base = || {
        let mut params = CodecParameters::video("h265".into());
        params.width = Some(16);
        params.height = Some(16);
        params.options.insert("mode", "inter");
        params
    };
    for bad in ["3", "32", "0", "x"] {
        let mut params = base();
        params.options.insert("pyramid", bad);
        assert!(
            oxideav_h265::make_encoder(&params).is_err(),
            "pyramid={bad} must be rejected"
        );
    }
    let mut params = base();
    params.options.insert("pyramid", "4");
    params.options.insert("gop", "3");
    assert!(
        oxideav_h265::make_encoder(&params).is_err(),
        "pyramid + gop must be rejected"
    );
    let mut params = base();
    params.options.insert("pyramid", "4");
    params.options.insert("bslices", "1");
    assert!(
        oxideav_h265::make_encoder(&params).is_err(),
        "pyramid + bslices must be rejected"
    );
    let mut params = base();
    params.options.insert("pyramid", "8");
    assert!(oxideav_h265::make_encoder(&params).is_ok());
}

#[test]
fn inter_mode_rejects_bad_gop_option() {
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(16);
    params.height = Some(16);
    params.options.insert("mode", "inter");
    params.options.insert("gop", "sometimes");
    assert!(oxideav_h265::make_encoder(&params).is_err());
}

#[test]
fn inter_mode_b_slices_registry_roundtrip() {
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(48);
    params.height = Some(32);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.options.insert("mode", "inter");
    params.options.insert("qp", "12");
    params.options.insert("bslices", "1");

    let mut enc = oxideav_h265::make_encoder(&params).expect("encoder factory");
    let frames: Vec<VideoFrame> = (0..4).map(|i| planes(48, 32, i)).collect();
    let mut aus = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        enc.send_frame(&Frame::Video(f.clone())).expect("send");
        let pkt = enc.receive_packet().expect("packet per frame");
        assert_eq!(pkt.flags.keyframe, i == 0, "frame {i} keyframe flag");
        aus.push(pkt);
    }

    let dec_params = CodecParameters::video("h265".into());
    let mut dec = oxideav_h265::make_decoder(&dec_params).expect("decoder factory");
    for pkt in &aus {
        dec.send_packet(pkt).expect("decode send");
    }
    dec.flush().expect("decode flush");
    for (i, f) in frames.iter().enumerate() {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => {
                for (pi, (p, q)) in v.planes.iter().zip(f.planes.iter()).enumerate() {
                    let mse: f64 = p
                        .data
                        .iter()
                        .zip(q.data.iter())
                        .map(|(&a, &b)| {
                            let d = f64::from(a) - f64::from(b);
                            d * d
                        })
                        .sum::<f64>()
                        / p.data.len() as f64;
                    let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10();
                    assert!(
                        psnr > 34.0,
                        "frame {i} plane {pi}: PSNR {psnr:.1} dB at qp 12 (B slices)"
                    );
                }
            }
            other => panic!("frame {i}: unexpected {other:?}"),
        }
    }
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
}

#[test]
fn inter_mode_loop_filters_registry_roundtrip() {
    // The `deblock` / `sao` options turn the §8.7 in-loop filters on:
    // the emitted stream signals them and still decodes cleanly (the
    // bit-exact encoder-recon contract is pinned by the unit tests;
    // here the registry wiring and fidelity are exercised).
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(48);
    params.height = Some(32);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.options.insert("mode", "inter");
    params.options.insert("qp", "18");
    params.options.insert("deblock", "1");
    params.options.insert("sao", "true");

    let mut enc = oxideav_h265::make_encoder(&params).expect("encoder factory");
    let frames: Vec<VideoFrame> = (0..4).map(|i| planes(48, 32, i)).collect();
    let mut aus = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        enc.send_frame(&Frame::Video(f.clone())).expect("send");
        let pkt = enc.receive_packet().expect("packet per frame");
        assert_eq!(pkt.flags.keyframe, i == 0, "frame {i} keyframe flag");
        aus.push(pkt);
    }

    let dec_params = CodecParameters::video("h265".into());
    let mut dec = oxideav_h265::make_decoder(&dec_params).expect("decoder factory");
    for pkt in &aus {
        dec.send_packet(pkt).expect("decode send");
    }
    dec.flush().expect("decode flush");
    for (i, f) in frames.iter().enumerate() {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => {
                for (pi, (p, q)) in v.planes.iter().zip(f.planes.iter()).enumerate() {
                    let mse: f64 = p
                        .data
                        .iter()
                        .zip(q.data.iter())
                        .map(|(&a, &b)| {
                            let d = f64::from(a) - f64::from(b);
                            d * d
                        })
                        .sum::<f64>()
                        / p.data.len() as f64;
                    let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10();
                    assert!(
                        psnr > 32.0,
                        "frame {i} plane {pi}: PSNR {psnr:.1} dB at qp 18 with loop filters"
                    );
                }
            }
            other => panic!("frame {i}: unexpected {other:?}"),
        }
    }
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
}

#[test]
fn intra_mode_loop_filters_registry_roundtrip() {
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(32);
    params.height = Some(32);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.options.insert("mode", "intra");
    params.options.insert("qp", "30");
    params.options.insert("deblock", "true");
    params.options.insert("sao", "1");

    let mut enc = oxideav_h265::make_encoder(&params).expect("encoder factory");
    let f = planes(32, 32, 0x21);
    enc.send_frame(&Frame::Video(f.clone())).expect("send");
    let pkt = enc.receive_packet().expect("packet");
    assert!(pkt.flags.keyframe);

    let dec_params = CodecParameters::video("h265".into());
    let mut dec = oxideav_h265::make_decoder(&dec_params).expect("decoder factory");
    dec.send_packet(&pkt).expect("decode send");
    dec.flush().expect("decode flush");
    assert!(matches!(dec.receive_frame(), Ok(Frame::Video(_))));
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
}

#[test]
fn loop_filter_options_are_validated() {
    // pcm mode has no transform path: the filter options are refused.
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(16);
    params.height = Some(16);
    params.options.insert("deblock", "1");
    assert!(oxideav_h265::make_encoder(&params).is_err());

    // Malformed values are refused in the filtered modes.
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(16);
    params.height = Some(16);
    params.options.insert("mode", "inter");
    params.options.insert("sao", "banana");
    assert!(oxideav_h265::make_encoder(&params).is_err());
}

/// §7.4.2.4.4 — a parameter-set NAL unit following a VCL NAL unit
/// starts a NEW access unit, so the pending picture must be decoded
/// against the parameter sets it was coded with BEFORE the arriving
/// set (legally re-sent with the same id and different content for
/// the next CVS) overwrites them.
#[test]
fn resent_parameter_sets_do_not_retroactively_apply_to_the_pending_picture() {
    let a = planes(16, 16, 0x5a);
    let b = planes(48, 32, 0xa5);
    let mut stream = oxideav_h265::encoder::pcm::encode_idr_pcm_au(
        &a.planes[0].data,
        &a.planes[1].data,
        &a.planes[2].data,
        16,
        16,
    )
    .expect("encode CVS 1");
    // Second CVS with a DIFFERENT geometry under the SAME sps/pps ids.
    stream.extend(
        oxideav_h265::encoder::pcm::encode_idr_pcm_au(
            &b.planes[0].data,
            &b.planes[1].data,
            &b.planes[2].data,
            48,
            32,
        )
        .expect("encode CVS 2"),
    );

    let frames = oxideav_h265::decode_annexb_sequence(&stream).expect("decode both CVSs");
    assert_eq!(frames.len(), 2);
    let dims: Vec<(usize, usize)> = frames
        .iter()
        .map(|f| (f.picture.width_luma(), f.picture.height_luma()))
        .collect();
    assert_eq!(dims, vec![(16, 16), (48, 32)]);
    // PCM is lossless: both pictures reproduce their sources exactly.
    for (frame, (src, w, h)) in frames.iter().zip([(&a, 16usize, 16usize), (&b, 48, 32)]) {
        for (plane, (pw, ph), data) in [
            (
                oxideav_h265::picture::Plane::Luma,
                (w, h),
                &src.planes[0].data,
            ),
            (
                oxideav_h265::picture::Plane::Cb,
                (w / 2, h / 2),
                &src.planes[1].data,
            ),
            (
                oxideav_h265::picture::Plane::Cr,
                (w / 2, h / 2),
                &src.planes[2].data,
            ),
        ] {
            for y in 0..ph {
                for x in 0..pw {
                    assert_eq!(
                        frame.picture.sample(plane, x, y),
                        i32::from(data[y * pw + x]),
                        "plane {plane:?} sample ({x},{y})"
                    );
                }
            }
        }
    }
}
