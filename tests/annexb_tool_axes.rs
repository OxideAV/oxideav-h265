//! Round-410 tool-axis conformance pins: whole-bitstream byte-exact
//! decodes over coding-tool axes that each exposed (and now pin) a
//! decoder fix. The bitstreams were produced by a black-box encoder
//! binary and the expected YUV captured from a black-box reference
//! decode; generation commands and SHA-256 sums are recorded in
//! `fixture_bytes/r410-generation-notes.md`.

mod fixture_bytes;

use fixture_bytes::r410::*;
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

/// Six-picture B-pyramid GOP (`bframes 4`, pyramid, 3 references): a
/// bi-predicted reference B serves as the collocated picture, so the
/// §8.5.3.2.9 bi-predicted-colPb listCol selection (LN with N =
/// collocated_from_l0_flag under backward prediction, LX under
/// NoBackwardPredFlag) drives the temporal merge/AMVP candidates.
#[test]
fn b_pyramid_tmvp_decodes_byte_exact() {
    assert_decodes_byte_exact(BPYR_HEVC, BPYR_YUV, 6, "r410-b-pyramid");
}

/// `scaling_list_enabled_flag == 1` with the §7.4.5 default lists
/// (Table 7-5 / 7-6, inferred DC 16): the §8.6.3 ScalingFactor
/// matrices apply to intra AND inter dequantization across TB sizes.
#[test]
fn scaling_list_decodes_byte_exact() {
    assert_decodes_byte_exact(SCALING_HEVC, SCALING_YUV, 4, "r410-scaling-list");
}

/// Smooth-gradient 32×32 intra blocks tripping the §8.4.4.2.3
/// `biIntFlag` strong smoothing: the eq 8-37 / 8-39 bilinear
/// interpolation covers reference indices 0..=62 inclusive.
#[test]
fn strong_intra_smoothing_decodes_byte_exact() {
    assert_decodes_byte_exact(STRONG_HEVC, STRONG_YUV, 1, "r410-strong-smoothing");
}

/// Rectangular + asymmetric inter partitions with a deep inter RQT:
/// transform trees splitting to 4×4 luma leaves defer the parent's
/// chroma blocks to the `blkIdx == 3` child (§7.3.8.10), positioned at
/// the parent node in the inter reconstruction path.
#[test]
fn rect_amp_deferred_chroma_decodes_byte_exact() {
    assert_decodes_byte_exact(RECTAMP_HEVC, RECTAMP_YUV, 3, "r410-rect-amp");
}

/// `constrained_intra_pred_flag == 1`: intra prediction blocks inside
/// P/B pictures treat reference samples from non-`MODE_INTRA` coding
/// units as unavailable (§8.4.4.2.1), substituting per §8.4.4.2.2.
#[test]
fn constrained_intra_decodes_byte_exact() {
    assert_decodes_byte_exact(CI_HEVC, CI_YUV, 4, "r410-constrained-intra");
}

/// `transform_skip_enabled_flag == 1`: the §7.3.8.11
/// `transform_skip_flag` element parses ahead of every 4×4 residual
/// block and the §8.6.2 transform-skip (tsShift) path replaces the
/// inverse transform in both intra and inter reconstruction.
#[test]
fn transform_skip_decodes_byte_exact() {
    assert_decodes_byte_exact(TSKIP_HEVC, TSKIP_YUV, 4, "r410-transform-skip");
}

/// WPP (`entropy_coding_sync_enabled_flag == 1`) with TWO independent
/// slices in one picture: slice 2 spans two CTB rows, so it carries an
/// entry point and performs the §9.3.2.5 in-slice row synchronization,
/// while its first row re-initializes (the §9.3.2.1 spatial neighbour
/// T lies in the previous slice).
#[test]
fn wpp_with_slices_decodes_byte_exact() {
    assert_decodes_byte_exact(WPPSLICES_HEVC, WPPSLICES_YUV, 1, "r410-wpp-slices");
}

/// Open-GOP: a mid-stream CRA (keyint 4) with leading B pictures and
/// POC output reordering across the IRAP boundary.
#[test]
fn open_gop_cra_decodes_byte_exact() {
    assert_decodes_byte_exact(OPENGOP_HEVC, OPENGOP_YUV, 8, "r410-open-gop");
}

/// 4:2:2 10-bit I/P/B GOP on textured content: the §7.3.8.8
/// lower-half chroma cbf inheritance into 4x4 leaves and the
/// §7.3.8.10 per-half pairing of the stacked chroma residual blocks
/// (upper/lower coded independently).
#[test]
fn chroma_422_halves_decode_byte_exact() {
    let frames = decode_annexb_sequence(M422_HEVC).expect("4:2:2 decode");
    assert_eq!(frames.len(), 4, "4:2:2: frame count");
    let mut out = Vec::new();
    for f in &frames {
        assert!(f.output, "4:2:2: every frame is an output frame");
        out.extend(f.picture.to_planar_le16());
    }
    assert_eq!(out, M422_YUV, "4:2:2: byte-exact decode");
}
