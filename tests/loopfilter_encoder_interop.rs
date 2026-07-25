//! Loop-filter encoder interop pins (round 429).
//!
//! Each golden stream below is this crate's deterministic encode with
//! the §8.7 in-loop filters enabled (deblocking + luma/chroma SAO),
//! validated OUT OF BAND against a black-box reference HEVC decoder
//! (which reproduces the encoder's FILTERED reconstruction byte for
//! byte — see `fixture_bytes/r429-generation-notes.md`). Re-encoding
//! the same content must reproduce the golden bytes, and the crate's
//! own decoder must decode them to the encoder's filtered
//! reconstruction — a three-way (encoder recon / this decoder /
//! external reference decoder) bit-exactness pin over the deblocking
//! election, the per-CTB SAO parameters and the filtered reference
//! path. If an intentional encoder change breaks this, regenerate the
//! bytes AND re-run the external cross-decode before updating the
//! pins.

use oxideav_h265::decode_annexb_sequence;
use oxideav_h265::encoder::inter::{LowDelayPEncoder, YuvFrame};
use oxideav_h265::encoder::intra::encode_idr_intra_au_lf;
use oxideav_h265::encoder::loopfilter::LoopFilterCfg;

const GOLDEN_PGOP_QP27: &[u8] = include_bytes!("fixture_bytes/r429-lf-pgop-qp27.hevc");
const GOLDEN_BGOP_QP33: &[u8] = include_bytes!("fixture_bytes/r429-lf-bgop-qp33.hevc");
const GOLDEN_INTRA_QP32: &[u8] = include_bytes!("fixture_bytes/r429-lf-intra-qp32.hevc");

/// The `p_gop_encoder_interop` clip: textured background + a moving
/// bright square, chroma drifting per frame.
fn clip(w: usize, h: usize, n: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    (0..n)
        .map(|t| {
            let y: Vec<u8> = (0..w * h)
                .map(|i| {
                    let (x, yy) = (i % w, i / w);
                    let base = (x * 2 + yy * 3) % 200;
                    let (sx, sy) = (4 + t * 3, 6 + t);
                    if x >= sx && x < sx + 12 && yy >= sy && yy < sy + 12 {
                        (base + 55) as u8
                    } else {
                        base as u8
                    }
                })
                .collect();
            let cb: Vec<u8> = (0..w * h / 4)
                .map(|i| (100 + (i % (w / 2)) * 2 % 60 + t) as u8)
                .collect();
            let cr: Vec<u8> = (0..w * h / 4)
                .map(|i| {
                    (160u32.wrapping_sub((i / (w / 2)) as u32 * 2 % 50) as u8).wrapping_sub(t as u8)
                })
                .collect();
            (y, cb, cr)
        })
        .collect()
}

/// Encode a filtered GOP, assert the golden bytes are reproduced and
/// that the crate's own decoder outputs the encoder's filtered recon
/// for every frame.
fn assert_gop_pin(
    golden: &[u8],
    planes: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
    w: usize,
    h: usize,
    qp: i32,
    b: bool,
    gop: usize,
) {
    let mut enc = LowDelayPEncoder::new(w, h, qp, gop)
        .expect("encoder")
        .with_b_slices(b)
        .with_loop_filters(LoopFilterCfg::all());
    let mut stream = Vec::new();
    let mut recons = Vec::new();
    for (y, cb, cr) in planes {
        let out = enc.encode_frame(&YuvFrame { y, cb, cr }).expect("encode");
        stream.extend_from_slice(&out.au);
        recons.push(out.recon);
    }
    assert_eq!(
        stream, golden,
        "deterministic filtered encode reproduces the externally validated golden stream"
    );
    let decoded = decode_annexb_sequence(golden).expect("decode");
    assert_eq!(decoded.len(), planes.len());
    for (i, (dec, rec)) in decoded.iter().zip(recons.iter()).enumerate() {
        let mut expect = rec.y.clone();
        expect.extend_from_slice(&rec.cb);
        expect.extend_from_slice(&rec.cr);
        assert_eq!(
            dec.picture.to_planar_u8().expect("8-bit"),
            expect,
            "frame {i}: decoder output == encoder's filtered reconstruction"
        );
    }
}

#[test]
fn golden_filtered_p_gop_is_reproduced_and_decodes_to_recon() {
    assert_gop_pin(GOLDEN_PGOP_QP27, &clip(64, 64, 5), 64, 64, 27, false, 0);
}

#[test]
fn golden_filtered_b_gop_with_idr_refresh_is_reproduced_and_decodes_to_recon() {
    assert_gop_pin(GOLDEN_BGOP_QP33, &clip(48, 48, 5), 48, 48, 33, true, 3);
}

#[test]
fn golden_filtered_intra_au_is_reproduced_and_decodes_to_recon() {
    // The `intra_encoder_interop` gradient frame.
    let (w, h) = (64usize, 64usize);
    let y: Vec<u8> = (0..w * h)
        .map(|i| {
            let (x, yy) = (i % w, i / w);
            ((x * 3 + yy * 2 + (x * yy / 7) % 31) % 256) as u8
        })
        .collect();
    let cb: Vec<u8> = (0..w * h / 4)
        .map(|i| ((i % (w / 2)) * 4 % 200 + 20) as u8)
        .collect();
    let cr: Vec<u8> = (0..w * h / 4)
        .map(|i| (240 - (i / (w / 2)) * 3 % 200) as u8)
        .collect();
    let enc =
        encode_idr_intra_au_lf(&y, &cb, &cr, w, h, 32, &LoopFilterCfg::all()).expect("encode");
    assert_eq!(
        enc.au, GOLDEN_INTRA_QP32,
        "deterministic filtered intra encode reproduces the golden AU"
    );
    let frames = decode_annexb_sequence(GOLDEN_INTRA_QP32).expect("decode");
    assert_eq!(frames.len(), 1);
    let mut recon = enc.recon_y.clone();
    recon.extend_from_slice(&enc.recon_cb);
    recon.extend_from_slice(&enc.recon_cr);
    assert_eq!(
        frames[0].picture.to_planar_u8().expect("8-bit"),
        recon,
        "decoder output == encoder's filtered reconstruction"
    );
}
