//! Round-416 conformance pins: whole-bitstream byte-exact decodes
//! over the Rext/SCC application-tail axes (cross-component
//! prediction, adaptive colour transform, intra block copy). The
//! bitstreams are self-built by this crate's own deterministic
//! generators; the CCP stream was additionally validated byte-exactly
//! through a black-box reference decoder CLI. Generation commands and
//! SHA-256 sums are recorded in
//! `fixture_bytes/r416-generation-notes.md`.

mod fixture_bytes;

use fixture_bytes::r416::*;
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

/// §8.6.6 cross-component prediction (4:4:4): every legal ResScaleVal
/// magnitude and both signs on Cb and Cr independently, zero-scale
/// controls, and a cbf-clear chroma block reconstructed purely from
/// the scaled luma residual. Black-box-reference-validated.
#[test]
fn ccp_decodes_byte_exact() {
    assert_decodes_byte_exact(CCP_HEVC, CCP_YUV, 1, "r416-ccp");
}

/// §8.6.8 adaptive colour transform (4:4:4): `tu_residual_act_flag`
/// alternating 1 / 0, the §8.6.8.2 lossless lifting inverse restoring
/// the source triples exactly.
#[test]
fn act_decodes_byte_exact() {
    assert_decodes_byte_exact(ACT_HEVC, ACT_YUV, 1, "r416-act");
}

/// Current-picture referencing (intra block copy): an IDR whose P
/// slice lists only the current picture, per-CTB integer motion
/// vectors copying already-decoded CTBs (eqs 8-98..8-101 integer
/// path + §8.5.3.1 pre-filter current-picture prediction).
#[test]
fn ibc_decodes_byte_exact() {
    assert_decodes_byte_exact(IBC_HEVC, IBC_YUV, 1, "r416-ibc");
}
