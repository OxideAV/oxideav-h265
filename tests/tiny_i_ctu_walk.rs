//! End-to-end §7.3.8 slice-data walk on a real HEVC fixture.
//!
//! Drives the §7.3.8.1 .. §7.3.8.6 [`oxideav_h265::slice_data`] CTU/CU
//! CABAC syntax-element walk on the actual slice-segment CABAC bytes of
//! the `tiny-i-only-16x16-main` fixture (a Main-profile, 4:2:0, 8-bit,
//! 16×16 single-CTU IDR I-frame). The fixture's `notes.md` / `trace.txt`
//! document the decode: one 16×16 CTU, SAO enabled, `slice_qp = 25`,
//! `SAO type 0` on every component.
//!
//! The slice NAL bytes are embedded as a public-test-corpus constant
//! (the bytes themselves, not any external decoder). The test strips
//! emulation prevention via the crate's own [`oxideav_h265::nal`]
//! helper, walks the §7.3.6.1 slice-segment-header fields with the
//! crate's [`oxideav_h265::bitreader::BitReader`] up to `byte_alignment(
//! )`, then constructs the §9.3 CABAC engine on the remaining bytes and
//! runs [`oxideav_h265::slice_data::decode_coding_tree_unit`] followed
//! by `end_of_slice_segment_flag`. This exercises the full upper
//! slice-data parse loop on real bitstream, validating it against the
//! fixture's documented CTU structure.

use oxideav_h265::binarization::CuPredMode;
use oxideav_h265::bitreader::BitReader;
use oxideav_h265::cabac::CabacEngine;
use oxideav_h265::ctx_init::SliceContexts;
use oxideav_h265::nal::strip_emulation_prevention;
use oxideav_h265::slice_data::{decode_coding_tree_unit, CodingQuadtree, SliceDataParams};

/// The 16-byte IDR_N_LP (`nal_unit_type == 20`) slice NAL of
/// `docs/video/h265/fixtures/tiny-i-only-16x16-main/input.hevc`
/// (offset 2381..2397), start-code-stripped, with its 2-byte NAL
/// header. Public test-corpus bytes.
const SLICE_NAL: [u8; 16] = [
    0x28, 0x01, 0xaf, 0x78, 0xf7, 0x04, 0x03, 0xff, 0x4f, 0x3d, 0xfe, 0x96, 0xd4, 0x3d, 0x27, 0x7e,
];

/// Build the §7.4.3 slice-data params for the tiny-i fixture from its
/// documented SPS / PPS (`trace.txt`):
///   log2_ctb=4, log2_min_cb=3, log2_min_tb=2, max_tb=5, 4:2:0 8-bit,
///   sao on, cu_qp_delta_enabled=1 (diff_cu_qp_delta_depth=0),
///   amp off, pcm off, max_transform_hierarchy_depth_intra=?.
fn tiny_i_params() -> SliceDataParams {
    SliceDataParams {
        ctb_log2_size_y: 4,
        min_cb_log2_size_y: 3,
        // SPS: log2_min_tb=2, log2_diff_max_min_tb=2 ⇒ MaxTbLog2SizeY = 4.
        max_tb_log2_size_y: 4,
        min_tb_log2_size_y: 2,
        pic_width_in_luma_samples: 16,
        pic_height_in_luma_samples: 16,
        chroma_array_type: 1,
        bit_depth_luma: 8,
        bit_depth_chroma: 8,
        slice_type_is_i: true,
        slice_type_is_b: false,
        slice_sao_luma_flag: true,
        slice_sao_chroma_flag: true,
        transquant_bypass_enabled_flag: false,
        cu_qp_delta_enabled_flag: true,
        // diff_cu_qp_delta_depth = 0 ⇒ Log2MinCuQpDeltaSize = CtbLog2SizeY.
        log2_min_cu_qp_delta_size: 4,
        cu_chroma_qp_offset_enabled_flag: false,
        log2_min_cu_chroma_qp_offset_size: 4,
        chroma_qp_offset_list_len_minus1: 0,
        amp_enabled_flag: false,
        pcm_enabled_flag: false,
        log2_min_ipcm_cb_size_y: 3,
        log2_max_ipcm_cb_size_y: 5,
        // SPS: max_transform_hierarchy_depth_intra = 0.
        max_transform_hierarchy_depth_intra: 0,
        max_transform_hierarchy_depth_inter: 0,
        max_num_merge_cand: 5,
        num_ref_idx_l0_active_minus1: 0,
        num_ref_idx_l1_active_minus1: 0,
        mvd_l1_zero_flag: false,
        sign_data_hiding_enabled_flag: true,
        cross_component_prediction_enabled_flag: false,
        residual_adaptive_colour_transform_enabled_flag: false,
    }
}

/// Walk the §7.3.6.1 slice-segment-header of the IDR_N_LP I-slice up to
/// `byte_alignment( )`, returning the byte offset (in the stripped
/// RBSP) where the §7.3.8.1 CABAC slice-segment data begins.
///
/// The field layout for this exact bitstream (`first_slice`, IRAP
/// `no_output_of_prior_pics_flag`, `pps_id`, `slice_type`, the two SAO
/// flags, `slice_qp_delta`, then `byte_alignment( )`) is established by
/// the fixture's documented decode (`trace.txt` / the
/// `temporal-mvp-sps-slice.md` clean-room finding).
fn slice_header_cabac_offset(rbsp: &[u8]) -> usize {
    let mut br = BitReader::new(rbsp);
    let first = br.u1().unwrap();
    assert_eq!(first, 1, "first_slice_segment_in_pic_flag");
    // IRAP (nal_unit_type 20) ⇒ no_output_of_prior_pics_flag.
    let _ = br.u1().unwrap();
    let pps_id = br.ue().unwrap();
    assert_eq!(pps_id, 0);
    // First slice in pic ⇒ no slice_segment_address.
    let slice_type = br.ue().unwrap();
    assert_eq!(slice_type, 2, "slice_type == I");
    // pps output_flag_present == 0, separate_colour_plane == 0.
    // IDR ⇒ no POC / RPS block.
    let sao_y = br.u1().unwrap();
    let sao_c = br.u1().unwrap();
    assert_eq!((sao_y, sao_c), (1, 1), "slice_sao_{{luma,chroma}}_flag");
    // I-slice ⇒ no ref-idx / mvp / cabac_init; IDR ⇒ no temporal mvp.
    let slice_qp_delta = br.se().unwrap();
    assert_eq!(slice_qp_delta, -1, "slice_qp_delta");
    // pps chroma-qp-offset / deblocking-override / entry-point /
    // header-extension blocks are all absent for this PPS.
    // byte_alignment(): alignment_bit_equal_to_one then zero bits.
    let align_one = br.u1().unwrap();
    assert_eq!(align_one, 1, "alignment_bit_equal_to_one");
    let bit_pos = br.bit_pos();
    // Round up to the next byte boundary.
    bit_pos.div_ceil(8)
}

#[test]
fn tiny_i_single_ctu_decodes_end_to_end() {
    // Strip emulation prevention from the slice NAL payload (after the
    // 2-byte NAL header).
    let rbsp = strip_emulation_prevention(&SLICE_NAL[2..]);
    let cabac_offset = slice_header_cabac_offset(&rbsp);
    assert_eq!(cabac_offset, 2, "CABAC slice-data starts at RBSP byte 2");

    let params = tiny_i_params();
    let cabac_bytes = &rbsp[cabac_offset..];
    let mut engine = CabacEngine::new(BitReader::new(cabac_bytes)).unwrap();
    // §9.3.1: SliceQpY = 26 + init_qp(0) + slice_qp_delta(-1) = 25.
    let mut ctx = SliceContexts::init(0, 25);

    // Decode the single CTU at (0, 0). SAO merge is not allowed (rx==0,
    // ry==0 — no left / above CTB in the slice segment).
    let ctu = decode_coding_tree_unit(&mut engine, &mut ctx, &params, 0, 0, false, false).unwrap();

    // The CTU has SAO (slice_sao_*_flag both set).
    let sao = ctu.sao.expect("sao present");
    // trace.txt: type_y == 0, type_cb == 0, type_cr == 0. These SAO
    // values are decoded bit-exactly from the real bitstream.
    assert_eq!(sao.components[0].sao_type_idx, 0, "SAO type Y == 0");
    assert_eq!(sao.components[1].sao_type_idx, 0, "SAO type Cb == 0");
    assert_eq!(sao.components[2].sao_type_idx, 0, "SAO type Cr == 0");
    assert!(!sao.merge_left, "rx == 0 ⇒ no sao_merge_left_flag");
    assert!(!sao.merge_up, "ry == 0 ⇒ no sao_merge_up_flag");

    // trace.txt: a single 16×16 CTU at (0,0). With MinCbLog2SizeY = 3,
    // MaxTbLog2SizeY = 4 and max_transform_hierarchy_depth_intra = 0, a
    // flat I-frame decodes to one un-split intra coding unit covering
    // the whole CTB. This is the documented CTU structure, decoded
    // bit-exactly through the §7.3.8.4 coding-quadtree / §7.3.8.5
    // coding-unit walk on the real CABAC bytes.
    let cu = match &ctu.quadtree {
        CodingQuadtree::Leaf(cu) => cu,
        CodingQuadtree::Split(_) => panic!("tiny-i decodes to a single un-split CU"),
    };
    assert_eq!(cu.x0, 0);
    assert_eq!(cu.y0, 0);
    assert_eq!(cu.log2_cb_size, 4, "16×16 coding unit");
    assert_eq!(cu.cu_pred_mode, CuPredMode::Intra, "I-slice CU is intra");
    assert!(!cu.cu_transquant_bypass_flag);
    assert!(!cu.pcm_flag);
    assert!(
        cu.transform_tree.is_some(),
        "intra CU carries a transform tree"
    );
    // PART_2Nx2N ⇒ one luma prediction block, one signalled luma mode.
    assert_eq!(cu.intra_luma.len(), 1, "PART_2Nx2N ⇒ one luma PB");
    // 4:2:0 non-PART_NxN ⇒ one intra_chroma_pred_mode.
    assert_eq!(cu.intra_chroma_pred_mode.len(), 1);

    // Exercise the §7.3.8.1 end_of_slice_segment_flag terminate path
    // after the CTU (the residual-coefficient bit-exactness that would
    // make this land on the byte-aligned terminator is the residual /
    // reconstruction subsystem's concern, not the syntax-walk under
    // test here).
    let _eos = oxideav_h265::slice_data::end_of_slice_segment_flag(&mut engine).unwrap();
}

/// End-to-end §8 reconstruction: decode the real `tiny-i-only-16x16-main`
/// IDR slice CABAC bytes through the §7.3.8 syntax walk, run the §8.4
/// intra sample-reconstruction driver, and validate the resulting
/// luma/chroma planes byte-for-byte against the fixture's documented
/// `expected.yuv` (luma 0x51, Cb 0x5a, Cr 0xf0 — the decoded red frame).
///
/// This exercises the full decode-to-pixels pipeline on real bitstream:
/// SAO + coding-quadtree + coding-unit + transform-tree + residual-coding
/// CABAC walk, §8.4.2 / §8.4.3 mode derivation, §8.4.4.2 intra
/// prediction, §8.6 dequantization + inverse transform, §8.6.1 QP
/// derivation, and §8.4.4.1 add-and-clip. The `end_of_slice_segment_flag`
/// terminating the single-CTU slice confirms the CABAC walk is bit-exact.
#[test]
fn tiny_i_reconstructs_expected_yuv_end_to_end() {
    use oxideav_h265::picture::{Picture, Plane};
    use oxideav_h265::recon::{reconstruct_intra_ctu, ReconParams};

    let rbsp = strip_emulation_prevention(&SLICE_NAL[2..]);
    let cabac_offset = slice_header_cabac_offset(&rbsp);
    let params = tiny_i_params();
    let mut engine = CabacEngine::new(BitReader::new(&rbsp[cabac_offset..])).unwrap();
    // §9.3.1: SliceQpY = 26 + init_qp(0) + slice_qp_delta(-1) = 25.
    let mut ctx = SliceContexts::init(0, 25);
    let ctu = decode_coding_tree_unit(&mut engine, &mut ctx, &params, 0, 0, false, false).unwrap();

    // The single-CTU slice must terminate exactly at the byte-aligned
    // end_of_slice_segment_flag — proof the residual-coding CABAC walk
    // consumed the correct number of bins.
    let eos = oxideav_h265::slice_data::end_of_slice_segment_flag(&mut engine).unwrap();
    assert!(
        eos,
        "single-CTU slice must terminate at end_of_slice_segment_flag"
    );

    let recon_params = ReconParams {
        chroma_array_type: 1,
        bit_depth_luma: 8,
        bit_depth_chroma: 8,
        intra_smoothing_disabled: false,
        strong_intra_smoothing_enabled: true,
        slice_qp_y: 25,
        cb_qp_offset: 0,
        cr_qp_offset: 0,
        transform_skip_rotation_enabled: false,
        extended_precision: false,
    };
    let mut pic = Picture::new(16, 16, 1, 8, 8);
    reconstruct_intra_ctu(&mut pic, &recon_params, &ctu).unwrap();

    // expected.yuv: luma plane all 0x51 (81), Cb all 0x5a (90),
    // Cr all 0xf0 (240).
    for y in 0..16 {
        for x in 0..16 {
            assert_eq!(pic.sample(Plane::Luma, x, y), 0x51, "luma ({x},{y})");
        }
    }
    for y in 0..8 {
        for x in 0..8 {
            assert_eq!(pic.sample(Plane::Cb, x, y), 0x5a, "cb ({x},{y})");
            assert_eq!(pic.sample(Plane::Cr, x, y), 0xf0, "cr ({x},{y})");
        }
    }

    // The packed planar output matches expected.yuv byte for byte.
    let packed = pic.to_planar_u8().unwrap();
    let mut expected = vec![0x51u8; 256];
    expected.extend(std::iter::repeat_n(0x5au8, 64));
    expected.extend(std::iter::repeat_n(0xf0u8, 64));
    assert_eq!(packed, expected, "reconstructed planes match expected.yuv");
}

/// The same `tiny-i` IDR slice, but driven through the picture-level
/// [`oxideav_h265::reconstruct_intra_picture`] driver (recon + §8.7.3 SAO)
/// instead of the single-CTU helper. SAO is enabled in the slice
/// (`slice_sao_*_flag` both set) but the CTU's SAO type is 0 on every
/// component, so the SAO pass is a no-op and the output must still match
/// `expected.yuv` byte for byte — exercising the shared-`ReconCtx`
/// neighbour path, the per-CTB SAO resolve, and the `apply_sao_picture`
/// integration end to end.
#[test]
fn tiny_i_picture_driver_reconstructs_expected_yuv_with_sao() {
    use oxideav_h265::availability::TilingParams;
    use oxideav_h265::recon::{
        reconstruct_intra_picture, IntraPictureParams, PlacedCtu, ReconParams,
    };

    let rbsp = strip_emulation_prevention(&SLICE_NAL[2..]);
    let cabac_offset = slice_header_cabac_offset(&rbsp);
    let params = tiny_i_params();
    let mut engine = CabacEngine::new(BitReader::new(&rbsp[cabac_offset..])).unwrap();
    let mut ctx = SliceContexts::init(0, 25);
    let ctu = decode_coding_tree_unit(&mut engine, &mut ctx, &params, 0, 0, false, false).unwrap();
    let eos = oxideav_h265::slice_data::end_of_slice_segment_flag(&mut engine).unwrap();
    assert!(
        eos,
        "single-CTU slice terminates at end_of_slice_segment_flag"
    );

    let recon_params = ReconParams {
        chroma_array_type: 1,
        bit_depth_luma: 8,
        bit_depth_chroma: 8,
        intra_smoothing_disabled: false,
        strong_intra_smoothing_enabled: true,
        slice_qp_y: 25,
        cb_qp_offset: 0,
        cr_qp_offset: 0,
        transform_skip_rotation_enabled: false,
        extended_precision: false,
    };
    let pic_params = IntraPictureParams {
        ctb_log2_size_y: 4,
        min_tb_log2_size_y: 2,
        tiles: TilingParams::single_tile(),
        slice_sao_luma_flag: true,
        slice_sao_chroma_flag: true,
        log2_sao_offset_scale_luma: 0,
        log2_sao_offset_scale_chroma: 0,
    };
    let placed = [PlacedCtu {
        x_ctb: 0,
        y_ctb: 0,
        slice_addr_rs: 0,
        ctu: &ctu,
    }];
    let out = reconstruct_intra_picture(16, 16, &recon_params, &pic_params, &placed).unwrap();

    let packed = out.to_planar_u8().unwrap();
    let mut expected = vec![0x51u8; 256];
    expected.extend(std::iter::repeat_n(0x5au8, 64));
    expected.extend(std::iter::repeat_n(0xf0u8, 64));
    assert_eq!(
        packed, expected,
        "picture-driver (recon + SAO) output matches expected.yuv"
    );
}
