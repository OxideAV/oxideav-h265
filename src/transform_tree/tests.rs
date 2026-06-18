//! §7.3.8.8 `transform_tree( )` recursion tests.

use super::*;
use crate::bitreader::BitReader;
use crate::transform_unit::CuPredMode;

fn template() -> TransformUnitParams {
    TransformUnitParams {
        log2_trafo_size: 3,
        trafo_depth: 0,
        blk_idx: 0,
        cu_pred_mode: CuPredMode::Intra,
        chroma_array_type: 1,
        cbf_luma: false,
        cbf_cb: false,
        cbf_cb_lower: false,
        cbf_cr: false,
        cbf_cr_lower: false,
        intra_pred_mode_y: 0,
        intra_pred_mode_c: 0,
        intra_chroma_pred_mode: 0,
        cu_qp_delta_enabled_flag: false,
        cu_chroma_qp_offset_enabled_flag: false,
        chroma_qp_offset_list_len_minus1: 0,
        cu_transquant_bypass_flag: false,
        sign_data_hiding_enabled_flag: false,
        cross_component_prediction_enabled_flag: false,
        residual_adaptive_colour_transform_enabled_flag: false,
        part_mode_2nx2n: true,
        intra_chroma_pred_mode_corners: [0; 4],
    }
}

fn base_params() -> TransformTreeParams {
    TransformTreeParams {
        max_tb_log2_size_y: 5,
        min_tb_log2_size_y: 2,
        max_trafo_depth: 0,
        intra_split_flag: false,
        inter_split_flag: false,
        cu_pred_mode: CuPredMode::Intra,
        chroma_array_type: 1,
        tu_template: template(),
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_root(
    data: &[u8],
    params: &TransformTreeParams,
    log2: u32,
) -> Result<TransformTree, ResidualCodingError> {
    let mut engine = CabacEngine::new(BitReader::new(data)).unwrap();
    let mut ctx = SliceContexts::init(0, 26);
    let mut qg = QuantGroupState::default();
    decode_transform_tree(
        &mut engine,
        &mut ctx,
        params,
        &mut qg,
        0,
        0,
        0,
        0,
        log2,
        0,
        0,
        false,
        false,
    )
}

/// With `MaxTrafoDepth == 0` and `log2TrafoSize == MinTbLog2SizeY`, the
/// split-flag presence gate is closed at the root, so the node is a leaf
/// with the inferred (0) split. An intra CU always reads `cbf_luma` at a
/// leaf (§7.3.8.8 presence condition). For a 4×4 luma block the chroma
/// block isn't present so no chroma cbf bins are consumed.
#[test]
fn root_4x4_is_leaf_intra_reads_cbf_luma() {
    let data = [0x00u8; 96];
    let mut params = base_params();
    params.max_trafo_depth = 0;
    // 4×4 root: log2 == 2 == MinTbLog2SizeY ⇒ split gate closed.
    let tree = decode_root(&data, &params, 2).unwrap();
    match tree {
        TransformTree::Leaf { .. } => {}
        TransformTree::Split { .. } => panic!("expected a leaf at 4×4 with MaxTrafoDepth 0"),
    }
}

/// `log2TrafoSize > MaxTbLog2SizeY` forces a split (§7.4.9.8 inference)
/// even when the split-flag presence gate is closed. A 64×64 luma node
/// with MaxTbLog2SizeY == 5 is inferred split into four 32×32 children.
#[test]
fn oversize_node_forces_inferred_split() {
    let data = [0x00u8; 256];
    let mut params = base_params();
    params.max_tb_log2_size_y = 5;
    params.min_tb_log2_size_y = 2;
    // 64×64 (log2 == 6) > MaxTbLog2SizeY ⇒ split gate closed, inferred 1.
    let tree = decode_root(&data, &params, 6).unwrap();
    match tree {
        TransformTree::Split { children, .. } => {
            // Four 32×32 children, each a leaf (32×32 == MaxTb so its
            // own split gate is closed at depth 1 < MaxTrafoDepth? no,
            // MaxTrafoDepth 0 closes it ⇒ leaf).
            assert_eq!(children.len(), 4);
        }
        TransformTree::Leaf { .. } => panic!("expected an inferred split at 64×64"),
    }
}

/// `IntraSplitFlag` forces a split at `trafoDepth == 0` (§7.4.9.8) and
/// also closes the split-flag presence gate, so the split is inferred
/// without consuming a bin.
#[test]
fn intra_split_flag_forces_root_split() {
    let data = [0x00u8; 256];
    let mut params = base_params();
    params.intra_split_flag = true;
    params.max_trafo_depth = 1;
    // 16×16 intra CU split into four 8×8 PUs ⇒ forced split at depth 0.
    let tree = decode_root(&data, &params, 4).unwrap();
    match tree {
        TransformTree::Split { children, .. } => assert_eq!(children.len(), 4),
        TransformTree::Leaf { .. } => panic!("IntraSplitFlag must force a root split"),
    }
}

/// The chroma-cbf inheritance gate: at the root (`trafoDepth == 0`) both
/// cbf_cb and cbf_cr are read unconditionally when the chroma block is
/// present (8×8 luma, 4:2:0). The all-zero stream decodes both flags to
/// 0, so the leaf's transform unit has no chroma residuals.
#[test]
fn root_8x8_reads_chroma_cbf() {
    let data = [0x00u8; 96];
    let mut params = base_params();
    params.max_trafo_depth = 0;
    params.chroma_array_type = 1;
    let tree = decode_root(&data, &params, 3).unwrap();
    match tree {
        TransformTree::Leaf { unit, .. } => {
            assert!(unit.residual_cb.is_empty());
            assert!(unit.residual_cr.is_empty());
        }
        TransformTree::Split { .. } => panic!("expected a leaf"),
    }
}

/// Monochrome (`ChromaArrayType == 0`) never reads a chroma cbf; the
/// chroma block presence gate is closed regardless of size.
#[test]
fn monochrome_skips_chroma_cbf() {
    let data = [0x00u8; 96];
    let mut params = base_params();
    params.chroma_array_type = 0;
    params.tu_template.chroma_array_type = 0;
    params.max_trafo_depth = 0;
    let tree = decode_root(&data, &params, 3).unwrap();
    match tree {
        TransformTree::Leaf { unit, .. } => {
            assert!(unit.residual_cb.is_empty());
            assert!(unit.residual_cr.is_empty());
        }
        TransformTree::Split { .. } => panic!("expected a leaf"),
    }
}

/// An inter leaf at `trafoDepth == 0` with no chroma cbf set does NOT
/// read `cbf_luma` (the §7.3.8.8 presence condition is false), so it is
/// inferred to 1.
#[test]
fn inter_root_leaf_infers_cbf_luma() {
    let data = [0x00u8; 96];
    let mut params = base_params();
    params.cu_pred_mode = CuPredMode::Inter;
    params.tu_template.cu_pred_mode = CuPredMode::Inter;
    params.chroma_array_type = 0; // no chroma cbf to set the condition
    params.tu_template.chroma_array_type = 0;
    params.max_trafo_depth = 0;
    let tree = decode_root(&data, &params, 3).unwrap();
    match tree {
        TransformTree::Leaf { cbf_luma, .. } => {
            // cbf_luma not present ⇒ inferred 1.
            assert!(cbf_luma);
        }
        TransformTree::Split { .. } => panic!("expected a leaf"),
    }
}

/// A genuine `split_transform_flag` read: with the presence gate open
/// (16×16 luma, MaxTb 5, MinTb 2, MaxTrafoDepth 2, no IntraSplit), the
/// first context-coded bin is consumed. The result is deterministic for
/// a fixed stream — here we only assert the recursion terminates and
/// returns a well-formed tree.
#[test]
fn open_gate_reads_split_flag_and_recurses() {
    let data = [0x00u8; 256];
    let mut params = base_params();
    params.max_tb_log2_size_y = 5;
    params.min_tb_log2_size_y = 2;
    params.max_trafo_depth = 2;
    let tree = decode_root(&data, &params, 4).unwrap();
    // Whatever the flag decoded to, the tree is one of the two shapes
    // and (if split) carries exactly four children.
    if let TransformTree::Split { children, .. } = &tree {
        assert_eq!(children.len(), 4);
    }
}
