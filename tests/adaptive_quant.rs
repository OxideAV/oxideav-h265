//! Spatial adaptive quantization (per-CTB `cu_qp_delta`) validation.
//!
//! The AQ encoder is the crate's first `cu_qp_delta` *writer*, so the
//! decisive check is differential: every AQ stream must decode
//! byte-exact through this crate's own decoder, whose §8.6.1 QP
//! derivation (`qPY_PREV` threading, neighbour prediction, per-4x4 QP
//! map) and §8.7.2 QP-dependent deblocking run the full
//! specification machinery against the encoder's mirrored emission.

use oxideav_h265::decode_annexb_sequence;
use oxideav_h265::encoder::intra::encode_idr_intra_au_aq;
use oxideav_h265::encoder::loopfilter::LoopFilterCfg;

/// Mixed content: a flat left half (AQ lowers QP), a busy textured
/// right half (AQ raises QP), plus a gradient band so multiple CTB
/// activity classes appear.
fn mixed_planes(w: usize, h: usize, seed: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y: Vec<u8> = (0..w * h)
        .map(|i| {
            let (x, yy) = (i % w, i / w);
            if x < w / 3 {
                90u8.wrapping_add(seed)
            } else if x < 2 * w / 3 {
                ((x * 255) / w) as u8
            } else {
                (x.wrapping_mul(73_856_093)
                    .wrapping_add(yy.wrapping_mul(19_349_663))
                    .wrapping_add(usize::from(seed) * 83) as u32
                    >> 11) as u8
            }
        })
        .collect();
    let cb: Vec<u8> = (0..w * h / 4)
        .map(|i| (100 + (i % (w / 2)) % 60) as u8)
        .collect();
    let cr: Vec<u8> = (0..w * h / 4)
        .map(|i| (170 - (i / (w / 2)) % 50) as u8)
        .collect();
    (y, cb, cr)
}

fn assert_decodes_to_recon(au: &oxideav_h265::encoder::intra::IntraEncodedAu, label: &str) {
    let decoded = decode_annexb_sequence(&au.au).expect("decode");
    assert_eq!(decoded.len(), 1, "{label}");
    let mut expect = au.recon_y.clone();
    expect.extend_from_slice(&au.recon_cb);
    expect.extend_from_slice(&au.recon_cr);
    assert_eq!(
        decoded[0].picture.to_planar_u8().expect("8-bit"),
        expect,
        "{label}: decoder output == encoder reconstruction"
    );
}

#[test]
fn aq_streams_decode_bit_exact_across_strengths_and_qps() {
    let (w, h) = (96usize, 64usize);
    let (y, cb, cr) = mixed_planes(w, h, 0);
    for qp in [17i32, 27, 37] {
        for aq in 0..=3u8 {
            let au = encode_idr_intra_au_aq(&y, &cb, &cr, w, h, qp, &LoopFilterCfg::off(), aq)
                .expect("encode");
            assert_decodes_to_recon(&au, &format!("qp{qp} aq{aq}"));
        }
    }
}

#[test]
fn aq_composes_with_in_loop_filters() {
    // QP-dependent deblocking must consume the per-CTB EFFECTIVE QPs
    // (including inherited values on cbf-less CTBs) on both sides.
    let (w, h) = (96usize, 48usize);
    for seed in [0u8, 7] {
        let (y, cb, cr) = mixed_planes(w, h, seed);
        for (lf, label) in [
            (LoopFilterCfg::all(), "deblock+sao"),
            (
                LoopFilterCfg {
                    deblocking: true,
                    sao_luma: false,
                    sao_chroma: false,
                },
                "deblock",
            ),
        ] {
            for aq in [1u8, 3] {
                let au = encode_idr_intra_au_aq(&y, &cb, &cr, w, h, 33, &lf, aq).expect("encode");
                assert_decodes_to_recon(&au, &format!("seed{seed} {label} aq{aq}"));
            }
        }
    }
}

#[test]
fn aq_actually_moves_per_ctb_qp() {
    let (w, h) = (96usize, 64usize);
    let (y, cb, cr) = mixed_planes(w, h, 0);
    let base = encode_idr_intra_au_aq(&y, &cb, &cr, w, h, 30, &LoopFilterCfg::off(), 0)
        .expect("encode aq0");
    let aq2 = encode_idr_intra_au_aq(&y, &cb, &cr, w, h, 30, &LoopFilterCfg::off(), 2)
        .expect("encode aq2");
    assert_ne!(base.au, aq2.au, "AQ must change the coded stream");
    // AQ spends relatively more bits on the flat third: its luma
    // reconstruction there gets closer to the source (or stays equal),
    // measured as SSD over the flat columns.
    let flat_ssd = |recon: &[u8]| -> u64 {
        (0..w * h)
            .filter(|i| i % w < w / 3)
            .map(|i| {
                let d = i64::from(y[i]) - i64::from(recon[i]);
                (d * d) as u64
            })
            .sum()
    };
    assert!(
        flat_ssd(&aq2.recon_y) <= flat_ssd(&base.recon_y),
        "AQ must not lose quality in the flat region ({} vs {})",
        flat_ssd(&aq2.recon_y),
        flat_ssd(&base.recon_y)
    );
}

#[test]
fn aq_registry_option_roundtrips_and_validates() {
    use oxideav_core::{CodecParameters, Error, Frame, PixelFormat, VideoFrame, VideoPlane};
    let (w, h) = (64usize, 64usize);
    let (y, cb, cr) = mixed_planes(w, h, 3);
    let mut p = CodecParameters::video("h265".into());
    p.width = Some(w as u32);
    p.height = Some(h as u32);
    p.pixel_format = Some(PixelFormat::Yuv420P);
    p.options.insert("mode", "intra");
    p.options.insert("qp", "31");
    p.options.insert("aq", "2");
    let mut enc = oxideav_h265::make_encoder(&p).expect("encoder");
    let plane = |data: &[u8], stride: usize| VideoPlane {
        stride,
        data: data.to_vec(),
    };
    enc.send_frame(&Frame::Video(VideoFrame {
        pts: None,
        planes: vec![plane(&y, w), plane(&cb, w / 2), plane(&cr, w / 2)],
    }))
    .expect("send");
    let pkt = enc.receive_packet().expect("packet");
    let decoded = decode_annexb_sequence(&pkt.data).expect("decode");
    assert_eq!(decoded.len(), 1);

    // Bad strengths and unsupported mode pairings are rejected.
    for (k, v) in [("aq", "4"), ("aq", "x")] {
        let mut bad = p.clone();
        bad.options.insert(k, v);
        assert!(matches!(
            oxideav_h265::make_encoder(&bad),
            Err(Error::InvalidData(_))
        ));
    }
    let mut pcm = p.clone();
    pcm.options.insert("mode", "pcm");
    assert!(matches!(
        oxideav_h265::make_encoder(&pcm),
        Err(Error::InvalidData(_))
    ));
    let mut inter = p.clone();
    inter.options.insert("mode", "inter");
    assert!(matches!(
        oxideav_h265::make_encoder(&inter),
        Err(Error::InvalidData(_))
    ));
}
