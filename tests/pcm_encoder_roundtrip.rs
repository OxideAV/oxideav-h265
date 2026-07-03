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
