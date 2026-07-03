//! End-to-end whole-bitstream decode: Annex B in, YUV out, byte-exact.
//!
//! Drives [`oxideav_h265::decode_annexb_sequence`] over embedded
//! public-test-corpus HEVC bitstreams (Main-profile intra pictures at
//! several geometries and QPs) and compares the decoded, in-loop-filtered
//! output against the corresponding expected planar YUV byte for byte.
//! This exercises the full §8.1 chain: NAL demux, parameter-set
//! activation, §7.3.6.1 slice-header parse, the §7.3.8 CABAC slice-data
//! walk (with the §7.4.9.11 mode-dependent scans and §8.4.2 parse-time
//! mode derivation), the §8.4 / §8.6 reconstruction, the §8.6.1 QP
//! derivation, and the §8.7.2 / §8.7.3 in-loop filters.

mod fixture_bytes;

use fixture_bytes::*;
use oxideav_h265::decode_annexb_sequence;

fn assert_decodes_byte_exact(hevc: &[u8], expected_yuv: &[u8], frames_expected: usize, what: &str) {
    let frames = decode_annexb_sequence(hevc).unwrap_or_else(|e| panic!("{what}: decode: {e}"));
    assert_eq!(frames.len(), frames_expected, "{what}: frame count");
    let mut out = Vec::new();
    for f in &frames {
        assert!(f.output, "{what}: every frame is an output frame");
        out.extend(f.picture.to_planar_u8().expect("8-bit planes"));
    }
    assert_eq!(out.len(), expected_yuv.len(), "{what}: output size");
    assert_eq!(out, expected_yuv, "{what}: byte-exact decode");
}

/// 16×16 single-CTU IDR (flat frame): the minimal Main-profile decode.
#[test]
fn tiny_i_decodes_byte_exact() {
    assert_decodes_byte_exact(TINY_I_HEVC, TINY_I_YUV, 1, "tiny-i-only-16x16-main");
}

/// 32×32 IDR at slice QP 45: sparse residual over 4×4 / 8×8 / 16×16
/// intra TBs with directional modes (vertical / horizontal §7.4.9.11
/// scans), SAO edge offsets, and deblocking.
#[test]
fn qp_high_decodes_byte_exact() {
    assert_decodes_byte_exact(QP_HIGH_HEVC, QP_HIGH_YUV, 1, "qp-high");
}

/// 32×32 IDR at slice QP 1: dense residual through every coefficient
/// coding path (multi-sub-block scans, sign hiding, high `coeff_abs`
/// escapes) plus the 4×4 DST-VII.
#[test]
fn qp_low_decodes_byte_exact() {
    assert_decodes_byte_exact(QP_LOW_HEVC, QP_LOW_YUV, 1, "qp-low");
}

/// Main Still Picture profile 32×32 IDR.
#[test]
fn main_still_picture_decodes_byte_exact() {
    assert_decodes_byte_exact(MAIN_STILL_HEVC, MAIN_STILL_YUV, 1, "main-still-picture");
}

/// 64×64 single-CTB IDR with SAO enabled, per-QG `cu_qp_delta` values,
/// and `last_sig_coeff` suffixes (coordinates past 3 in 8×8 TBs).
#[test]
fn sao_on_decodes_byte_exact() {
    assert_decodes_byte_exact(SAO_ON_HEVC, SAO_ON_YUV, 1, "sao-on");
}

/// Four-picture all-intra sequence (every frame IDR): multi-picture
/// output ordering across coded-video-sequence boundaries.
#[test]
fn intra_only_allintra_decodes_byte_exact() {
    assert_decodes_byte_exact(ALLINTRA_HEVC, ALLINTRA_YUV, 4, "intra-only-allintra");
}

/// IDR then TRAIL_R P-picture: the §8.3 reference cycle + §8.5 inter
/// reconstruction (merge / MVP candidates, temporal MVP, quarter-pel
/// interpolation) against the in-DPB reference, byte-exact.
#[test]
fn i_then_p_decodes_byte_exact() {
    assert_decodes_byte_exact(I_THEN_P_HEVC, I_THEN_P_YUV, 2, "i-frame-then-p-frame-main");
}

/// Main10 profile, 10-bit 4:2:0 IDR + P picture (little-endian 16-bit
/// planar): the §8.5 inter path at BitDepth 10.
#[test]
fn main10_decodes_byte_exact() {
    let frames = decode_annexb_sequence(MAIN10_HEVC).expect("main10 decode");
    assert_eq!(frames.len(), 2, "main10: frame count");
    let mut out = Vec::new();
    for f in &frames {
        out.extend(f.picture.to_planar_le16());
    }
    assert_eq!(out, MAIN10_YUV, "main10: byte-exact decode");
}

/// Eight-picture I/P/B sequence with two reference lists: bi-prediction,
/// §8.5.3.2.8 temporal MVP against the collocated picture's motion
/// field, AMVP with signalled MVDs, skip/merge CUs, and POC output
/// reordering — byte-exact across all frames.
#[test]
fn bipred_b_frames_decode_byte_exact() {
    assert_decodes_byte_exact(BIPRED_HEVC, BIPRED_YUV, 8, "bipred-b-frames-main");
}

/// `entropy_coding_sync_enabled_flag == 1` (WPP): an 8×4-CTB picture
/// whose CTB rows are separate entry-point substreams with the §9.3.2.4
/// context storage after each row's second CTB and the §9.3.2.5
/// synchronization at the next row's start, plus per-QG cu_qp_delta
/// chains and per-position deblocking QPs.
#[test]
fn wpp_on_decodes_byte_exact() {
    assert_decodes_byte_exact(WPP_HEVC, WPP_YUV, 1, "wpp-on");
}

/// Four independent slice segments per picture: per-slice CABAC init,
/// cross-slice neighbour denial (§6.4.1), per-slice qPY_PREV reset, and
/// the §8.7.2.1 / §8.7.3.2 loop-filter suppression across slice
/// boundaries (`slice_loop_filter_across_slices_enabled_flag == 0`).
#[test]
fn multi_slice_per_frame_decodes_byte_exact() {
    assert_decodes_byte_exact(
        MULTI_SLICE_HEVC,
        MULTI_SLICE_YUV,
        1,
        "multi-slice-per-frame",
    );
}

/// A 4×2-CTB I+P pair: multi-CTU slices exercising the cross-CTU
/// `split_cu_flag` / `cu_skip_flag` ctxInc neighbour reads.
#[test]
fn tile_cols_2_decodes_byte_exact() {
    assert_decodes_byte_exact(TILE_COLS_HEVC, TILE_COLS_YUV, 2, "tile-cols-2");
}

/// The `oxideav_core::Decoder` contract over the same driver: feed the
/// eight-picture B-pyramid stream as one packet, flush, and pull the
/// frames in output order — byte-exact against the expected YUV.
#[test]
fn registry_decoder_decodes_bipred_byte_exact() {
    use oxideav_core::{CodecParameters, Error, Frame, Packet, TimeBase};

    let params = CodecParameters::video("h265".into());
    let mut dec = oxideav_h265::make_decoder(&params).expect("factory");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), BIPRED_HEVC.to_vec()))
        .expect("send");
    dec.flush().expect("flush");
    let mut out = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => {
                for p in &v.planes {
                    out.extend_from_slice(&p.data);
                }
            }
            Ok(_) => panic!("non-video frame"),
            Err(Error::Eof) => break,
            Err(e) => panic!("receive: {e}"),
        }
    }
    assert_eq!(out, BIPRED_YUV, "registry decoder output byte-exact");
}

/// The ISO-BMFF transport form (`iso-mp4-vs-annexb-pair`): `hvcC`
/// extradata (ISO/IEC 14496-15 §8.3.3.1) activates the out-of-band
/// VPS/SPS/PPS and switches packets to 4-byte length-prefixed NAL
/// runs — the decoded frame is byte-identical to the Annex B decode.
#[test]
fn registry_decoder_decodes_hvcc_mp4_sample_byte_exact() {
    use oxideav_core::{CodecParameters, Error, Frame, Packet, TimeBase};

    let mut params = CodecParameters::video("h265".into());
    params.extradata = ISO_PAIR_HVCC.to_vec();
    let mut dec = oxideav_h265::make_decoder(&params).expect("factory");
    dec.send_packet(&Packet::new(
        0,
        TimeBase::new(1, 25),
        ISO_PAIR_SAMPLE.to_vec(),
    ))
    .expect("send");
    dec.flush().expect("flush");
    let mut out = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => {
                for p in &v.planes {
                    out.extend_from_slice(&p.data);
                }
            }
            Ok(_) => panic!("non-video frame"),
            Err(Error::Eof) => break,
            Err(e) => panic!("receive: {e}"),
        }
    }
    assert_eq!(out, ISO_PAIR_YUV, "hvcC-path decode byte-exact");
}

/// Explicit weighted prediction, P slices (§8.5.3.3.4.3): an 8-frame
/// fade whose seven P slices carry `weighted_pred_flag == 1` with
/// non-default per-slice `pred_weight_table()` denominators, weights
/// and offsets (luma and chroma) — byte-exact through the whole
/// decode chain, including the temporal merge candidates the fade's
/// skip CUs select.
#[test]
fn weighted_pred_p_decodes_byte_exact() {
    assert_decodes_byte_exact(WEIGHTED_P_HEVC, WEIGHTED_P_YUV, 8, "weighted-pred-p");
}

/// Explicit weighted bi-prediction (§8.5.3.3.4.3 equation 8-277): the
/// same fade coded as an I/P/B pyramid with `weighted_bipred_flag == 1`
/// — every B-slice PU combine (uni L0, uni L1 and bi) runs the
/// explicit path with two reference lists' weights.
#[test]
fn weighted_bipred_b_decodes_byte_exact() {
    assert_decodes_byte_exact(WEIGHTED_B_HEVC, WEIGHTED_B_YUV, 8, "weighted-bipred-b");
}

/// Per-slice `slice_loop_filter_across_slices_enabled_flag` (§7.4.7.1 /
/// §8.7.2.1): two IDR pictures, each split into two independent slices
/// with OPPOSITE across-flags (1/0 then 0/1) over an all-PCM picture
/// with deblocking enabled and `pcm_loop_filter_disabled_flag == 0` —
/// the slice-boundary edge filters exactly when the flag of the slice
/// containing the current (q-side) coding block allows it. A decoder
/// using a picture-level flag decodes both pictures identically; the
/// expected YUV distinguishes them.
#[test]
fn per_slice_loop_filter_across_flags_decode_byte_exact() {
    assert_decodes_byte_exact(PERSLICE_LF_HEVC, PERSLICE_LF_YUV, 2, "per-slice-lf-across");
}
