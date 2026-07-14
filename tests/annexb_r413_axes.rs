//! Round-413 conformance pins: whole-bitstream byte-exact decodes over
//! coding-tool axes that each exposed (and now pin) a decoder fix. The
//! bitstreams were produced by a black-box encoder binary and the
//! expected YUV captured from a black-box reference decode; generation
//! commands and SHA-256 sums are recorded in
//! `fixture_bytes/r413-generation-notes.md`.

mod fixture_bytes;

use fixture_bytes::r413::*;
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

/// 4:4:4 lossless (transquant-bypass) IDR + P run: `ChromaArrayType ==
/// 3` PART_NxN coding units signal FOUR `intra_chroma_pred_mode`
/// elements (§7.3.8.5) and §8.4.3 derives each chroma PB's
/// `IntraPredModeC` from its OWN co-located luma PB's
/// `IntraPredModeY` — not from the CU-corner PB.
#[test]
fn lossless_444_per_pb_chroma_modes_decode_byte_exact() {
    assert_decodes_byte_exact(LL444_HEVC, LL444_YUV, 6, "r413-lossless-444");
}
