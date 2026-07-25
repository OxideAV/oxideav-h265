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
