//! §7.3.8.1 .. §7.3.8.6 slice-data walk tests.

use super::*;
use crate::bitreader::BitReader;
use crate::cabac::CabacEngine;

/// A minimal Main-profile (4:2:0, 8-bit) I-slice params block, used by
/// the structural decode tests.
fn i_slice_params(ctb_log2: u32, min_cb_log2: u32) -> SliceDataParams {
    SliceDataParams {
        ctb_log2_size_y: ctb_log2,
        min_cb_log2_size_y: min_cb_log2,
        max_tb_log2_size_y: 5,
        min_tb_log2_size_y: 2,
        pic_width_in_luma_samples: 1 << ctb_log2,
        pic_height_in_luma_samples: 1 << ctb_log2,
        chroma_array_type: 1,
        bit_depth_luma: 8,
        bit_depth_chroma: 8,
        slice_type_is_i: true,
        slice_type_is_b: false,
        slice_sao_luma_flag: false,
        slice_sao_chroma_flag: false,
        transquant_bypass_enabled_flag: false,
        cu_qp_delta_enabled_flag: false,
        log2_min_cu_qp_delta_size: ctb_log2,
        cu_chroma_qp_offset_enabled_flag: false,
        log2_min_cu_chroma_qp_offset_size: ctb_log2,
        chroma_qp_offset_list_len_minus1: 0,
        amp_enabled_flag: false,
        pcm_enabled_flag: false,
        log2_min_ipcm_cb_size_y: 3,
        log2_max_ipcm_cb_size_y: 5,
        max_transform_hierarchy_depth_intra: 0,
        max_transform_hierarchy_depth_inter: 0,
        max_num_merge_cand: 5,
        num_ref_idx_l0_active_minus1: 0,
        num_ref_idx_l1_active_minus1: 0,
        mvd_l1_zero_flag: false,
        sign_data_hiding_enabled_flag: false,
        cross_component_prediction_enabled_flag: false,
        residual_adaptive_colour_transform_enabled_flag: false,
    }
}

fn fresh_contexts() -> SliceContexts {
    // Init every bank at slice_qp_y = 26 with init_type 0 (I slice).
    SliceContexts::init(0, 26)
}

#[test]
fn ctu_grid_left_above_unavailable_at_origin() {
    let params = i_slice_params(4, 3);
    let grid = CtuGrid::new(&params, 0, 0);
    assert_eq!(grid.left_ct_depth(0, 0), (0, false));
    assert_eq!(grid.above_ct_depth(0, 0), (0, false));
    assert_eq!(grid.left_cu_skip(0, 0), (0, false));
    assert_eq!(grid.above_cu_skip(0, 0), (0, false));
}

#[test]
fn ctu_grid_marks_and_reads_back_neighbours() {
    let params = i_slice_params(5, 3); // 32x32 CTB, 8x8 min CB
    let mut grid = CtuGrid::new(&params, 0, 0);
    // Mark an 8x8 block at (0,0) with CtDepth 2.
    grid.mark(0, 0, 3, 2, 0);
    // The block to the right at (8,0) sees (0,0) as its left neighbour.
    assert_eq!(grid.left_ct_depth(8, 0), (2, true));
    // The block below at (0,8) sees (0,0) as its above neighbour.
    assert_eq!(grid.above_ct_depth(0, 8), (2, true));
}

#[test]
fn sao_disabled_yields_no_sao() {
    // slice_sao_*_flag both false → coding_tree_unit has sao == None.
    // An all-zero stream still drives a full intra-CU CABAC walk, so the
    // buffer must be large enough to back every decoded bin.
    let params = i_slice_params(4, 3);
    let buf = [0u8; 4096];
    let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
    let mut ctx = fresh_contexts();
    let ctu = decode_coding_tree_unit(&mut eng, &mut ctx, &params, 0, 0, false, false).unwrap();
    assert!(ctu.sao.is_none());
}

#[test]
fn sao_band_offset_default_is_zero_type() {
    // sao_merge not allowed, both components read; an all-zero CABAC
    // stream drives sao_type_idx bin 0 (MPS-from-init) — the decode
    // simply must not error and must produce a structurally valid
    // SaoCtbParams.
    let mut params = i_slice_params(4, 3);
    params.slice_sao_luma_flag = true;
    params.slice_sao_chroma_flag = true;
    let buf = [0u8; 32];
    let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
    let mut ctx = fresh_contexts();
    let sao = decode_sao(&mut eng, &mut ctx, &params, false, false).unwrap();
    assert!(!sao.merge_left);
    assert!(!sao.merge_up);
    // SaoTypeIdx values are in 0..=2.
    for c in sao.components.iter() {
        assert!(c.sao_type_idx <= 2);
    }
}

#[test]
fn coding_quadtree_at_min_cb_is_a_leaf() {
    // A 16x16 CTB with MinCbLog2SizeY = 4 cannot split (split_cu_flag
    // gate requires log2CbSize > MinCbLog2SizeY) → one coding unit.
    let params = i_slice_params(4, 4);
    let buf = [0u8; 4096];
    let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
    let mut ctx = fresh_contexts();
    let mut grid = CtuGrid::new(&params, 0, 0);
    let mut qg = QuantGroupState::default();
    let qt = decode_coding_quadtree(&mut eng, &mut ctx, &params, &mut grid, &mut qg, 0, 0, 4, 0)
        .unwrap();
    match qt {
        CodingQuadtree::Leaf(cu) => {
            assert_eq!(cu.cu_pred_mode, CuPredMode::Intra);
            assert_eq!(cu.x0, 0);
            assert_eq!(cu.y0, 0);
            assert_eq!(cu.log2_cb_size, 4);
            // An intra CU always carries a transform tree.
            assert!(cu.transform_tree.is_some());
        }
        CodingQuadtree::Split(_) => panic!("min-CB node must be a leaf"),
    }
}

#[test]
fn coding_tree_unit_i_slice_decodes_structurally() {
    // Full CTU walk on an I slice: sao disabled, a 16x16 CTB at
    // MinCbLog2SizeY = 3 may split. The decode must complete without
    // error and produce a quadtree rooted at the CTB.
    let params = i_slice_params(4, 3);
    let buf = [0u8; 4096];
    let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
    let mut ctx = fresh_contexts();
    let ctu = decode_coding_tree_unit(&mut eng, &mut ctx, &params, 0, 0, false, false).unwrap();
    assert!(ctu.sao.is_none());
    // The quadtree is either a leaf or a split; either is valid.
    match ctu.quadtree {
        CodingQuadtree::Leaf(_) | CodingQuadtree::Split(_) => {}
    }
}
