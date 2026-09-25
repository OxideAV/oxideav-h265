//! Annex F `vps_extension( )` — the multi-layer video parameter set
//! extension (F.7.3.2.1.1 syntax, F.7.4.3.1.1 semantics) plus its
//! `rep_format( )` (F.7.3.2.1.2), `dpb_size( )` (F.7.3.2.1.3),
//! `vps_vui( )` (F.7.3.2.1.4), `video_signal_info( )` (F.7.3.2.1.5) and
//! `vps_vui_bsp_hrd_params( )` (F.7.3.2.1.6) sub-structures.
//!
//! The parse materialises every syntax element into typed fields and
//! derives the F.7.4.3.1.1 layer model ([`LayerModel`]): the
//! `LayerIdxInVps` map, the Table F.1 scalability identifiers
//! (`ViewOrderIdx`, `DependencyId`, `AuxId`), `ViewId`, the direct /
//! transitive dependency matrices (`NumDirectRefLayers`,
//! `IdDirectRefLayer`, `IdRefLayer`, `IdPredictedLayer`), the tree
//! partitions, the layer sets (base + additional), the output layer sets
//! (`OlsIdxToLsIdx`, `OutputLayerFlag`, `NecessaryLayerFlag`,
//! `OlsHighestOutputLayerId`) and the eq. F-14 inter-layer sample /
//! motion prediction gates. Those are the inputs the Annex F/G/H slice
//! header parse (`NumDirectRefLayers`, `default_ref_layers_active_flag`,
//! `max_one_active_ref_layer_flag`, `poc_lsb_not_present_flag`,
//! `vps_poc_lsb_aligned_flag`) and the F.8 / G.8 / H.8 decoding
//! processes consume.
//!
//! Every loop is bounded by the F.7.4.3.1.1 value ranges before it is
//! entered (`MaxLayersMinus1 <= 62`, `num_add_layer_sets <= 1023`,
//! `num_add_olss <= 1023`, `vps_num_rep_formats_minus1 <= 255`,
//! `vps_num_profile_tier_level_minus1 <= 63`,
//! `vps_non_vui_extension_length <= 4096`, ...) so a hostile VPS cannot
//! drive an unbounded allocation.

// The parse mirrors the F.7.3.2.1.x index loops verbatim (`[ i ][ j ]`
// matrices addressed by layer index) — iterator rewrites would obscure
// the correspondence with the Recommendation's tables.
#![allow(clippy::needless_range_loop)]

use crate::bitreader::BitReader;
use crate::hrd::HrdParameters;
use crate::vps::{ProfileTierLevel, VpsError};

/// Upper bound on `MaxLayersMinus1` (`Min( 62, vps_max_layers_minus1 )`).
pub const MAX_LAYERS_MINUS1: usize = 62;
/// `nuh_layer_id` values span 0..=63.
const NUH_LAYER_ID_RANGE: usize = 64;

/// `Ceil( Log2( n ) )` for the u(v) widths of the extension; 0 for
/// `n <= 1`.
#[inline]
#[must_use]
pub fn ceil_log2(n: u32) -> u8 {
    if n <= 1 {
        0
    } else {
        (32 - (n - 1).leading_zeros()) as u8
    }
}

/// One `rep_format( )` structure (F.7.3.2.1.2). The chroma / bit-depth
/// block is inherited from the previous entry when
/// `chroma_and_bit_depth_vps_present_flag == 0` (F.7.4.3.1.2); the
/// fields below always hold the effective (inherited) values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RepFormat {
    /// `pic_width_vps_in_luma_samples` (u(16)).
    pub pic_width_vps_in_luma_samples: u16,
    /// `pic_height_vps_in_luma_samples` (u(16)).
    pub pic_height_vps_in_luma_samples: u16,
    /// `chroma_and_bit_depth_vps_present_flag`.
    pub chroma_and_bit_depth_vps_present_flag: bool,
    /// `chroma_format_vps_idc` (effective).
    pub chroma_format_vps_idc: u8,
    /// `separate_colour_plane_vps_flag` (effective).
    pub separate_colour_plane_vps_flag: bool,
    /// `bit_depth_vps_luma_minus8` (effective, 0..=8).
    pub bit_depth_vps_luma_minus8: u8,
    /// `bit_depth_vps_chroma_minus8` (effective, 0..=8).
    pub bit_depth_vps_chroma_minus8: u8,
    /// `conformance_window_vps_flag`.
    pub conformance_window_vps_flag: bool,
    /// `conf_win_vps_left_offset` (0 when absent).
    pub conf_win_vps_left_offset: u32,
    /// `conf_win_vps_right_offset` (0 when absent).
    pub conf_win_vps_right_offset: u32,
    /// `conf_win_vps_top_offset` (0 when absent).
    pub conf_win_vps_top_offset: u32,
    /// `conf_win_vps_bottom_offset` (0 when absent).
    pub conf_win_vps_bottom_offset: u32,
}

impl RepFormat {
    fn parse(br: &mut BitReader<'_>, prev: Option<&RepFormat>) -> Result<Self, VpsError> {
        let pic_width_vps_in_luma_samples = br.u(16)? as u16;
        let pic_height_vps_in_luma_samples = br.u(16)? as u16;
        let chroma_and_bit_depth_vps_present_flag = br.u1()? != 0;
        let (chroma_format_vps_idc, separate_colour_plane_vps_flag, bd_luma, bd_chroma) =
            if chroma_and_bit_depth_vps_present_flag {
                let cf = br.u(2)? as u8;
                let sep = if cf == 3 { br.u1()? != 0 } else { false };
                let bl = br.u(4)? as u8;
                let bc = br.u(4)? as u8;
                if bl > 8 {
                    return Err(VpsError::ValueOutOfRange {
                        field: "bit_depth_vps_luma_minus8",
                        got: u32::from(bl),
                    });
                }
                if bc > 8 {
                    return Err(VpsError::ValueOutOfRange {
                        field: "bit_depth_vps_chroma_minus8",
                        got: u32::from(bc),
                    });
                }
                (cf, sep, bl, bc)
            } else {
                // F.7.4.3.1.2: inferred from the previous rep_format( );
                // the first one shall signal the block.
                let p = prev.ok_or(VpsError::ValueOutOfRange {
                    field: "chroma_and_bit_depth_vps_present_flag",
                    got: 0,
                })?;
                (
                    p.chroma_format_vps_idc,
                    p.separate_colour_plane_vps_flag,
                    p.bit_depth_vps_luma_minus8,
                    p.bit_depth_vps_chroma_minus8,
                )
            };
        let conformance_window_vps_flag = br.u1()? != 0;
        let (l, r, t, b) = if conformance_window_vps_flag {
            (br.ue()?, br.ue()?, br.ue()?, br.ue()?)
        } else {
            (0, 0, 0, 0)
        };
        Ok(Self {
            pic_width_vps_in_luma_samples,
            pic_height_vps_in_luma_samples,
            chroma_and_bit_depth_vps_present_flag,
            chroma_format_vps_idc,
            separate_colour_plane_vps_flag,
            bit_depth_vps_luma_minus8: bd_luma,
            bit_depth_vps_chroma_minus8: bd_chroma,
            conformance_window_vps_flag,
            conf_win_vps_left_offset: l,
            conf_win_vps_right_offset: r,
            conf_win_vps_top_offset: t,
            conf_win_vps_bottom_offset: b,
        })
    }
}

/// One sub-layer row of `dpb_size( )` for one output layer set
/// (F.7.3.2.1.3). Absent rows inherit the previous sub-layer's values
/// (F.7.4.3.1.3); the fields hold the effective values.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DpbSizeSubLayer {
    /// `sub_layer_dpb_info_present_flag[ i ][ j ]` (inferred 1 for
    /// `j == 0`).
    pub sub_layer_dpb_info_present_flag: bool,
    /// `max_vps_dec_pic_buffering_minus1[ i ][ k ][ j ]` per layer `k`
    /// of the OLS's layer set (0 for layers that are not necessary).
    pub max_vps_dec_pic_buffering_minus1: Vec<u32>,
    /// `max_vps_num_reorder_pics[ i ][ j ]`.
    pub max_vps_num_reorder_pics: u32,
    /// `max_vps_latency_increase_plus1[ i ][ j ]`.
    pub max_vps_latency_increase_plus1: u32,
}

/// `dpb_size( )` for one output layer set `i >= 1`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DpbSizeOls {
    /// `sub_layer_flag_info_present_flag[ i ]`.
    pub sub_layer_flag_info_present_flag: bool,
    /// Rows `j = 0..=MaxSubLayersInLayerSetMinus1[ OlsIdxToLsIdx[ i ] ]`.
    pub sub_layers: Vec<DpbSizeSubLayer>,
}

/// `video_signal_info( )` (F.7.3.2.1.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VideoSignalInfo {
    /// `video_vps_format` (u(3)).
    pub video_vps_format: u8,
    /// `video_full_range_vps_flag`.
    pub video_full_range_vps_flag: bool,
    /// `colour_primaries_vps` (u(8)).
    pub colour_primaries_vps: u8,
    /// `transfer_characteristics_vps` (u(8)).
    pub transfer_characteristics_vps: u8,
    /// `matrix_coeffs_vps` (u(8)).
    pub matrix_coeffs_vps: u8,
}

/// One `(layer set, sub-layer)` bit-rate / picture-rate cell of the VPS
/// VUI (F.7.3.2.1.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VpsVuiRateCell {
    /// `bit_rate_present_flag[ i ][ j ]`.
    pub bit_rate_present_flag: bool,
    /// `pic_rate_present_flag[ i ][ j ]`.
    pub pic_rate_present_flag: bool,
    /// `avg_bit_rate[ i ][ j ]`.
    pub avg_bit_rate: u16,
    /// `max_bit_rate[ i ][ j ]`.
    pub max_bit_rate: u16,
    /// `constant_pic_rate_idc[ i ][ j ]`.
    pub constant_pic_rate_idc: u8,
    /// `avg_pic_rate[ i ][ j ]`.
    pub avg_pic_rate: u16,
}

/// The `ilp_restricted_ref_layers_flag` block of the VPS VUI for one
/// `(layer i, direct reference j)` pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IlpRestriction {
    /// `min_spatial_segment_offset_plus1[ i ][ j ]`.
    pub min_spatial_segment_offset_plus1: u32,
    /// `ctu_based_offset_enabled_flag[ i ][ j ]`.
    pub ctu_based_offset_enabled_flag: bool,
    /// `min_horizontal_ctu_offset_plus1[ i ][ j ]`.
    pub min_horizontal_ctu_offset_plus1: u32,
}

/// `vps_vui( )` (F.7.3.2.1.4). Purely informative for the decoding
/// process (F.7.4.3.1.4); the bitstream-partition HRD block is walked
/// (its `hrd_parameters( )` bodies parsed through [`crate::hrd`]) but
/// only its counts are retained.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VpsVui {
    /// `cross_layer_pic_type_aligned_flag`.
    pub cross_layer_pic_type_aligned_flag: bool,
    /// `cross_layer_irap_aligned_flag` (inferred 1 when
    /// `cross_layer_pic_type_aligned_flag == 1`).
    pub cross_layer_irap_aligned_flag: bool,
    /// `all_layers_idr_aligned_flag`.
    pub all_layers_idr_aligned_flag: bool,
    /// `bit_rate_present_vps_flag`.
    pub bit_rate_present_vps_flag: bool,
    /// `pic_rate_present_vps_flag`.
    pub pic_rate_present_vps_flag: bool,
    /// Rate cells `[ layer set ][ sub-layer ]` (empty rows for layer
    /// sets the loop skips).
    pub rates: Vec<Vec<VpsVuiRateCell>>,
    /// `video_signal_info_idx_present_flag`.
    pub video_signal_info_idx_present_flag: bool,
    /// The `video_signal_info( )` list.
    pub video_signal_info: Vec<VideoSignalInfo>,
    /// `vps_video_signal_info_idx[ i ]` per layer index (inferred per
    /// F.7.4.3.1.4 when absent).
    pub vps_video_signal_info_idx: Vec<u8>,
    /// `tiles_not_in_use_flag`.
    pub tiles_not_in_use_flag: bool,
    /// `tiles_in_use_flag[ i ]` per layer index.
    pub tiles_in_use_flag: Vec<bool>,
    /// `loop_filter_not_across_tiles_flag[ i ]` per layer index.
    pub loop_filter_not_across_tiles_flag: Vec<bool>,
    /// `tile_boundaries_aligned_flag[ i ][ j ]` per layer index /
    /// direct reference index.
    pub tile_boundaries_aligned_flag: Vec<Vec<bool>>,
    /// `wpp_not_in_use_flag`.
    pub wpp_not_in_use_flag: bool,
    /// `wpp_in_use_flag[ i ]` per layer index.
    pub wpp_in_use_flag: Vec<bool>,
    /// `single_layer_for_non_irap_flag`.
    pub single_layer_for_non_irap_flag: bool,
    /// `higher_layer_irap_skip_flag`.
    pub higher_layer_irap_skip_flag: bool,
    /// `ilp_restricted_ref_layers_flag`.
    pub ilp_restricted_ref_layers_flag: bool,
    /// The restriction block per layer index / direct reference index.
    pub ilp_restrictions: Vec<Vec<IlpRestriction>>,
    /// `vps_vui_bsp_hrd_present_flag`.
    pub vps_vui_bsp_hrd_present_flag: bool,
    /// `vps_num_add_hrd_params` (0 when the block is absent).
    pub vps_num_add_hrd_params: u32,
    /// `base_layer_parameter_set_compatibility_flag[ i ]` per layer
    /// index (only meaningful for independent layers).
    pub base_layer_parameter_set_compatibility_flag: Vec<bool>,
}

/// The F.7.4.3.1.1 derived layer model of a VPS extension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerModel {
    /// `MaxLayersMinus1 = Min( 62, vps_max_layers_minus1 )`.
    pub max_layers_minus1: u8,
    /// `layer_id_in_nuh[ i ]` for `i = 0..=MaxLayersMinus1`.
    pub layer_id_in_nuh: Vec<u8>,
    /// `LayerIdxInVps[ nuh_layer_id ]` (`None` for ids not in the VPS).
    pub layer_idx_in_vps: [Option<u8>; NUH_LAYER_ID_RANGE],
    /// `ScalabilityId[ i ][ smIdx ]` (Table F.1 index order).
    pub scalability_id: Vec<[u8; 16]>,
    /// `ViewId[ nuh_layer_id ]` (0 for ids not in the VPS).
    pub view_id: [u16; NUH_LAYER_ID_RANGE],
    /// `NumViews`.
    pub num_views: u32,
    /// `DependencyFlag[ i ][ j ]` (transitive).
    pub dependency_flag: Vec<Vec<bool>>,
    /// `IdDirectRefLayer[ nuh_layer_id ]` — the direct reference layers'
    /// `nuh_layer_id` values, in increasing layer-index order.
    pub id_direct_ref_layer: Vec<Vec<u8>>,
    /// `IdRefLayer[ nuh_layer_id ]` (transitive reference layers).
    pub id_ref_layer: Vec<Vec<u8>>,
    /// `IdPredictedLayer[ nuh_layer_id ]`.
    pub id_predicted_layer: Vec<Vec<u8>>,
    /// `TreePartitionLayerIdList[ k ]` (eq. F-6).
    pub tree_partition_layer_id_list: Vec<Vec<u8>>,
    /// `LayerSetLayerIdList[ lsIdx ]` for every layer set (base sets
    /// from the inclusion matrix, then the additional layer sets).
    pub layer_set_layer_id_list: Vec<Vec<u8>>,
    /// `OlsIdxToLsIdx[ olsIdx ]` (eq. F-11).
    pub ols_idx_to_ls_idx: Vec<u16>,
    /// `OutputLayerFlag[ olsIdx ][ j ]`.
    pub output_layer_flag: Vec<Vec<bool>>,
    /// `NecessaryLayerFlag[ olsIdx ][ j ]` (eq. F-13).
    pub necessary_layer_flag: Vec<Vec<bool>>,
    /// `OlsHighestOutputLayerId[ olsIdx ]`.
    pub ols_highest_output_layer_id: Vec<u8>,
    /// `MaxSubLayersInLayerSetMinus1[ lsIdx ]` (eq. F-10).
    pub max_sub_layers_in_layer_set_minus1: Vec<u8>,
    /// `VpsInterLayerSamplePredictionEnabled[ i ][ j ]` (eq. F-14).
    pub inter_layer_sample_prediction_enabled: Vec<Vec<bool>>,
    /// `VpsInterLayerMotionPredictionEnabled[ i ][ j ]` (eq. F-14).
    pub inter_layer_motion_prediction_enabled: Vec<Vec<bool>>,
}

impl Default for LayerModel {
    /// The single-layer model: one layer with `nuh_layer_id == 0`, one
    /// layer set and one output layer set outputting it.
    fn default() -> Self {
        let mut layer_idx_in_vps = [None; NUH_LAYER_ID_RANGE];
        layer_idx_in_vps[0] = Some(0);
        Self {
            max_layers_minus1: 0,
            layer_id_in_nuh: vec![0],
            layer_idx_in_vps,
            scalability_id: vec![[0; 16]],
            view_id: [0; NUH_LAYER_ID_RANGE],
            num_views: 1,
            dependency_flag: vec![vec![false]],
            id_direct_ref_layer: vec![Vec::new(); NUH_LAYER_ID_RANGE],
            id_ref_layer: vec![Vec::new(); NUH_LAYER_ID_RANGE],
            id_predicted_layer: vec![Vec::new(); NUH_LAYER_ID_RANGE],
            tree_partition_layer_id_list: vec![vec![0]],
            layer_set_layer_id_list: vec![vec![0]],
            ols_idx_to_ls_idx: vec![0],
            output_layer_flag: vec![vec![true]],
            necessary_layer_flag: vec![vec![true]],
            ols_highest_output_layer_id: vec![0],
            max_sub_layers_in_layer_set_minus1: vec![0],
            inter_layer_sample_prediction_enabled: vec![vec![false]],
            inter_layer_motion_prediction_enabled: vec![vec![false]],
        }
    }
}

impl LayerModel {
    /// `LayerIdxInVps[ nuh_layer_id ]`.
    #[must_use]
    pub fn layer_idx(&self, nuh_layer_id: u8) -> Option<usize> {
        self.layer_idx_in_vps
            .get(usize::from(nuh_layer_id))
            .copied()
            .flatten()
            .map(usize::from)
    }

    /// `NumDirectRefLayers[ nuh_layer_id ]`.
    #[must_use]
    pub fn num_direct_ref_layers(&self, nuh_layer_id: u8) -> usize {
        self.id_direct_ref_layer
            .get(usize::from(nuh_layer_id))
            .map_or(0, Vec::len)
    }

    /// `IdDirectRefLayer[ nuh_layer_id ][ d ]`.
    #[must_use]
    pub fn direct_ref_layers(&self, nuh_layer_id: u8) -> &[u8] {
        self.id_direct_ref_layer
            .get(usize::from(nuh_layer_id))
            .map_or(&[], Vec::as_slice)
    }

    /// `NumPredictedLayers[ nuh_layer_id ]`.
    #[must_use]
    pub fn num_predicted_layers(&self, nuh_layer_id: u8) -> usize {
        self.id_predicted_layer
            .get(usize::from(nuh_layer_id))
            .map_or(0, Vec::len)
    }

    /// `NumLayerSets`.
    #[must_use]
    pub fn num_layer_sets(&self) -> usize {
        self.layer_set_layer_id_list.len()
    }

    /// `NumOutputLayerSets`.
    #[must_use]
    pub fn num_output_layer_sets(&self) -> usize {
        self.ols_idx_to_ls_idx.len()
    }

    /// `NumLayersInIdList[ OlsIdxToLsIdx[ ols_idx ] ]` list.
    #[must_use]
    pub fn ols_layer_ids(&self, ols_idx: usize) -> &[u8] {
        self.ols_idx_to_ls_idx
            .get(ols_idx)
            .and_then(|&ls| self.layer_set_layer_id_list.get(usize::from(ls)))
            .map_or(&[], Vec::as_slice)
    }

    /// The `nuh_layer_id` values that are output layers of `ols_idx`.
    #[must_use]
    pub fn ols_output_layer_ids(&self, ols_idx: usize) -> Vec<u8> {
        let ids = self.ols_layer_ids(ols_idx);
        self.output_layer_flag
            .get(ols_idx)
            .map(|flags| {
                ids.iter()
                    .zip(flags.iter())
                    .filter(|(_, &f)| f)
                    .map(|(&id, _)| id)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// `ViewId[ nuh_layer_id ]`.
    #[must_use]
    pub fn view_id_of(&self, nuh_layer_id: u8) -> u16 {
        self.view_id
            .get(usize::from(nuh_layer_id))
            .copied()
            .unwrap_or(0)
    }

    /// `ViewOrderIdx[ nuh_layer_id ]` (Table F.1 index 1).
    #[must_use]
    pub fn view_order_idx(&self, nuh_layer_id: u8) -> u8 {
        self.layer_idx(nuh_layer_id)
            .and_then(|i| self.scalability_id.get(i))
            .map_or(0, |s| s[1])
    }

    /// `DependencyId[ nuh_layer_id ]` (Table F.1 index 2).
    #[must_use]
    pub fn dependency_id(&self, nuh_layer_id: u8) -> u8 {
        self.layer_idx(nuh_layer_id)
            .and_then(|i| self.scalability_id.get(i))
            .map_or(0, |s| s[2])
    }

    /// `AuxId[ nuh_layer_id ]` (Table F.1 index 3).
    #[must_use]
    pub fn aux_id(&self, nuh_layer_id: u8) -> u8 {
        self.layer_idx(nuh_layer_id)
            .and_then(|i| self.scalability_id.get(i))
            .map_or(0, |s| s[3])
    }

    /// `VpsInterLayerSamplePredictionEnabled[ LayerIdxInVps[ curr ] ][ LayerIdxInVps[ rl ] ]`.
    #[must_use]
    pub fn sample_prediction_enabled(&self, curr_layer_id: u8, ref_layer_id: u8) -> bool {
        match (self.layer_idx(curr_layer_id), self.layer_idx(ref_layer_id)) {
            (Some(i), Some(j)) => self
                .inter_layer_sample_prediction_enabled
                .get(i)
                .and_then(|r| r.get(j))
                .copied()
                .unwrap_or(false),
            _ => false,
        }
    }

    /// `VpsInterLayerMotionPredictionEnabled[ LayerIdxInVps[ curr ] ][ LayerIdxInVps[ rl ] ]`.
    #[must_use]
    pub fn motion_prediction_enabled(&self, curr_layer_id: u8, ref_layer_id: u8) -> bool {
        match (self.layer_idx(curr_layer_id), self.layer_idx(ref_layer_id)) {
            (Some(i), Some(j)) => self
                .inter_layer_motion_prediction_enabled
                .get(i)
                .and_then(|r| r.get(j))
                .copied()
                .unwrap_or(false),
            _ => false,
        }
    }
}

/// Parsed `vps_extension( )` (F.7.3.2.1.1) with the derived
/// [`LayerModel`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VpsExtension {
    /// `profile_tier_level( 0, vps_max_sub_layers_minus1 )` — present when
    /// `vps_max_layers_minus1 > 0 && vps_base_layer_internal_flag`
    /// (`VpsProfileTierLevel[ 1 ]`, F.7.4.4).
    pub base_layer_ptl: Option<ProfileTierLevel>,
    /// `splitting_flag`.
    pub splitting_flag: bool,
    /// `scalability_mask_flag[ i ]`, Table F.1 index order.
    pub scalability_mask_flag: [bool; 16],
    /// `dimension_id_len_minus1[ j ]` per present scalability type
    /// (the last one inferred under `splitting_flag`).
    pub dimension_id_len_minus1: Vec<u8>,
    /// `vps_nuh_layer_id_present_flag`.
    pub vps_nuh_layer_id_present_flag: bool,
    /// `dimension_id[ i ][ j ]` (inferred under `splitting_flag`).
    pub dimension_id: Vec<Vec<u8>>,
    /// `view_id_len`.
    pub view_id_len: u8,
    /// `view_id_val[ i ]` for `i < NumViews` (0 when absent).
    pub view_id_val: Vec<u16>,
    /// `direct_dependency_flag[ i ][ j ]` (full square, inferred 0).
    pub direct_dependency_flag: Vec<Vec<bool>>,
    /// `num_add_layer_sets`.
    pub num_add_layer_sets: u32,
    /// `highest_layer_idx_plus1[ i ][ j ]` for `j = 1..NumIndependentLayers`
    /// (index `j - 1`).
    pub highest_layer_idx_plus1: Vec<Vec<u32>>,
    /// `vps_sub_layers_max_minus1_present_flag`.
    pub vps_sub_layers_max_minus1_present_flag: bool,
    /// `sub_layers_vps_max_minus1[ i ]` (inferred
    /// `vps_max_sub_layers_minus1`).
    pub sub_layers_vps_max_minus1: Vec<u8>,
    /// `max_tid_ref_present_flag`.
    pub max_tid_ref_present_flag: bool,
    /// `max_tid_il_ref_pics_plus1[ i ][ j ]` (inferred 7).
    pub max_tid_il_ref_pics_plus1: Vec<Vec<u8>>,
    /// `default_ref_layers_active_flag`.
    pub default_ref_layers_active_flag: bool,
    /// `vps_num_profile_tier_level_minus1`.
    pub vps_num_profile_tier_level_minus1: u32,
    /// `vps_profile_present_flag[ i ]` for the signalled entries
    /// (index `i - (base_layer_internal ? 2 : 1)`).
    pub vps_profile_present_flag: Vec<bool>,
    /// `VpsProfileTierLevel[ i ]` for `i = 0..=vps_num_profile_tier_level_minus1`
    /// (F.7.4.4 ordered list: base VPS PTL, then the extension's).
    pub vps_profile_tier_level: Vec<ProfileTierLevel>,
    /// `num_add_olss`.
    pub num_add_olss: u32,
    /// `default_output_layer_idc` (raw; `defaultOutputLayerIdc =
    /// Min( value, 2 )`).
    pub default_output_layer_idc: u8,
    /// `layer_set_idx_for_ols_minus1[ i ]` per OLS (inferred 0).
    pub layer_set_idx_for_ols_minus1: Vec<u32>,
    /// `output_layer_flag[ i ][ j ]` as signalled / inferred.
    pub output_layer_flag: Vec<Vec<bool>>,
    /// `profile_tier_level_idx[ i ][ j ]` per OLS per layer (inferred
    /// per F.7.4.3.1.1 when absent).
    pub profile_tier_level_idx: Vec<Vec<u32>>,
    /// `alt_output_layer_flag[ i ]` per OLS (inferred 0).
    pub alt_output_layer_flag: Vec<bool>,
    /// The `rep_format( )` list.
    pub rep_formats: Vec<RepFormat>,
    /// `rep_format_idx_present_flag`.
    pub rep_format_idx_present_flag: bool,
    /// `vps_rep_format_idx[ i ]` per layer index (inferred
    /// `Min( i, vps_num_rep_formats_minus1 )`).
    pub vps_rep_format_idx: Vec<u32>,
    /// `max_one_active_ref_layer_flag`.
    pub max_one_active_ref_layer_flag: bool,
    /// `vps_poc_lsb_aligned_flag`.
    pub vps_poc_lsb_aligned_flag: bool,
    /// `poc_lsb_not_present_flag[ i ]` per layer index (inferred 0).
    pub poc_lsb_not_present_flag: Vec<bool>,
    /// `dpb_size( )` rows for OLS `1..NumOutputLayerSets` (index `i - 1`).
    pub dpb_size: Vec<DpbSizeOls>,
    /// `direct_dep_type_len_minus2`.
    pub direct_dep_type_len_minus2: u32,
    /// `direct_dependency_all_layers_flag`.
    pub direct_dependency_all_layers_flag: bool,
    /// `direct_dependency_all_layers_type` (0 when absent).
    pub direct_dependency_all_layers_type: u32,
    /// `direct_dependency_type[ i ][ j ]` (full square; inferred per
    /// F.7.4.3.1.1, `u32::MAX` where no dependency exists).
    pub direct_dependency_type: Vec<Vec<u32>>,
    /// `vps_non_vui_extension_length`.
    pub vps_non_vui_extension_length: u32,
    /// `vps_vui_present_flag`.
    pub vps_vui_present_flag: bool,
    /// The parsed `vps_vui( )`. `None` when absent, or when present but
    /// malformed (see [`Self::vps_vui_malformed`]) — the VUI is purely
    /// informative (F.7.4.3.1.4) and sits last in the extension, so a
    /// malformed VUI does not reject the VPS.
    pub vps_vui: Option<VpsVui>,
    /// `true` when `vps_vui_present_flag == 1` but the body failed to
    /// parse.
    pub vps_vui_malformed: bool,
    /// The derived F.7.4.3.1.1 layer model.
    pub layers: LayerModel,
}

/// The base-VPS fields `vps_extension( )` parsing depends on.
#[derive(Debug, Clone, Copy)]
pub struct VpsExtensionInputs<'a> {
    /// `vps_base_layer_internal_flag`.
    pub base_layer_internal_flag: bool,
    /// `vps_base_layer_available_flag`.
    pub base_layer_available_flag: bool,
    /// `vps_max_layers_minus1`.
    pub max_layers_minus1: u8,
    /// `vps_max_sub_layers_minus1`.
    pub max_sub_layers_minus1: u8,
    /// `vps_max_layer_id`.
    pub max_layer_id: u8,
    /// `vps_num_layer_sets_minus1`.
    pub num_layer_sets_minus1: u16,
    /// `layer_id_included_flag[ i ][ j ]` rows for `i = 1..`.
    pub layer_id_included_flag: &'a [Vec<bool>],
    /// `vps_num_hrd_parameters` (0 when no timing info).
    pub num_hrd_parameters: u32,
    /// The base VPS `profile_tier_level( 1, … )` — `VpsProfileTierLevel[ 0 ]`.
    pub base_ptl: &'a ProfileTierLevel,
}

fn oor(field: &'static str, got: u32) -> VpsError {
    VpsError::ValueOutOfRange { field, got }
}

impl VpsExtension {
    /// Parse `vps_extension( )` at the current (byte-aligned) reader
    /// position.
    ///
    /// # Errors
    /// [`VpsError`] on truncation or an F.7.4.3.1.1 range violation.
    pub fn parse(br: &mut BitReader<'_>, inp: &VpsExtensionInputs<'_>) -> Result<Self, VpsError> {
        let max_layers_minus1 = usize::from(inp.max_layers_minus1).min(MAX_LAYERS_MINUS1);
        let num_layers = max_layers_minus1 + 1;
        let base_internal = inp.base_layer_internal_flag;

        let base_layer_ptl = if inp.max_layers_minus1 > 0 && base_internal {
            Some(ProfileTierLevel::parse(
                br,
                false,
                inp.max_sub_layers_minus1,
            )?)
        } else {
            None
        };
        let splitting_flag = br.u1()? != 0;
        let mut scalability_mask_flag = [false; 16];
        let mut num_scalability_types = 0usize;
        for f in &mut scalability_mask_flag {
            *f = br.u1()? != 0;
            num_scalability_types += usize::from(*f);
        }
        let mut dimension_id_len_minus1 = Vec::with_capacity(num_scalability_types);
        for _ in 0..num_scalability_types.saturating_sub(usize::from(splitting_flag)) {
            dimension_id_len_minus1.push(br.u(3)? as u8);
        }
        // F.7.4.3.1.1 (F-2): under splitting the last length is inferred.
        let mut dim_bit_offset = vec![0u32; num_scalability_types + 1];
        if splitting_flag && num_scalability_types > 0 {
            for j in 1..num_scalability_types {
                dim_bit_offset[j] =
                    dim_bit_offset[j - 1] + u32::from(dimension_id_len_minus1[j - 1]) + 1;
            }
            let last = dim_bit_offset[num_scalability_types - 1];
            if last >= 6 {
                return Err(oor("dimension_id_len_minus1", last));
            }
            dimension_id_len_minus1.push((5 - last) as u8);
            dim_bit_offset[num_scalability_types] = 6;
        }
        let vps_nuh_layer_id_present_flag = br.u1()? != 0;
        let mut layer_id_in_nuh = vec![0u8; num_layers];
        let mut dimension_id = vec![vec![0u8; num_scalability_types]; num_layers];
        for i in 1..num_layers {
            layer_id_in_nuh[i] = if vps_nuh_layer_id_present_flag {
                br.u(6)? as u8
            } else {
                i as u8
            };
            if layer_id_in_nuh[i] <= layer_id_in_nuh[i - 1] {
                return Err(oor("layer_id_in_nuh", u32::from(layer_id_in_nuh[i])));
            }
            if !splitting_flag {
                for j in 0..num_scalability_types {
                    dimension_id[i][j] = br.u(dimension_id_len_minus1[j] + 1)? as u8;
                }
            }
        }
        if splitting_flag {
            for (i, row) in dimension_id.iter_mut().enumerate() {
                for (j, d) in row.iter_mut().enumerate() {
                    let mask = (1u32 << dim_bit_offset[j + 1]) - 1;
                    *d = ((u32::from(layer_id_in_nuh[i]) & mask) >> dim_bit_offset[j]) as u8;
                }
            }
        }

        // ---- F-3: ScalabilityId / ViewOrderIdx / NumViews ----
        let mut scalability_id = vec![[0u8; 16]; num_layers];
        let mut layer_idx_in_vps = [None; NUH_LAYER_ID_RANGE];
        let mut num_views = 1u32;
        for i in 0..num_layers {
            layer_idx_in_vps[usize::from(layer_id_in_nuh[i])] = Some(i as u8);
            let mut j = 0;
            for sm in 0..16 {
                if scalability_mask_flag[sm] {
                    scalability_id[i][sm] = dimension_id[i][j];
                    j += 1;
                }
            }
            if i > 0 {
                let new_view = (0..i).all(|k| scalability_id[k][1] != scalability_id[i][1]);
                num_views += u32::from(new_view);
            }
        }

        let view_id_len = br.u(4)? as u8;
        let mut view_id_val = vec![0u16; num_views as usize];
        if view_id_len > 0 {
            for v in &mut view_id_val {
                *v = br.u(view_id_len)? as u16;
            }
        }
        let mut view_id = [0u16; NUH_LAYER_ID_RANGE];
        for i in 0..num_layers {
            view_id[usize::from(layer_id_in_nuh[i])] = view_id_val
                .get(usize::from(scalability_id[i][1]))
                .copied()
                .unwrap_or(0);
        }

        // ---- direct_dependency_flag + F-4 / F-5 / F-6 ----
        let mut direct_dependency_flag = vec![vec![false; num_layers]; num_layers];
        for i in 1..num_layers {
            for j in 0..i {
                direct_dependency_flag[i][j] = br.u1()? != 0;
            }
        }
        let mut dependency_flag = direct_dependency_flag.clone();
        for i in 0..num_layers {
            for j in 0..num_layers {
                for k in 0..i {
                    if direct_dependency_flag[i][k] && dependency_flag[k][j] {
                        dependency_flag[i][j] = true;
                    }
                }
            }
        }
        let mut id_direct_ref_layer = vec![Vec::new(); NUH_LAYER_ID_RANGE];
        let mut id_ref_layer = vec![Vec::new(); NUH_LAYER_ID_RANGE];
        let mut id_predicted_layer = vec![Vec::new(); NUH_LAYER_ID_RANGE];
        for i in 0..num_layers {
            let i_id = usize::from(layer_id_in_nuh[i]);
            for j in 0..num_layers {
                let j_id = layer_id_in_nuh[j];
                if direct_dependency_flag[i][j] {
                    id_direct_ref_layer[i_id].push(j_id);
                }
                if dependency_flag[i][j] {
                    id_ref_layer[i_id].push(j_id);
                }
                if dependency_flag[j][i] {
                    id_predicted_layer[i_id].push(j_id);
                }
            }
        }
        let mut tree_partition_layer_id_list: Vec<Vec<u8>> = Vec::new();
        {
            let mut in_list = [false; NUH_LAYER_ID_RANGE];
            for i in 0..num_layers {
                let i_id = layer_id_in_nuh[i];
                if id_direct_ref_layer[usize::from(i_id)].is_empty() {
                    let mut part = vec![i_id];
                    for &pred in &id_predicted_layer[usize::from(i_id)] {
                        if !in_list[usize::from(pred)] {
                            part.push(pred);
                            in_list[usize::from(pred)] = true;
                        }
                    }
                    tree_partition_layer_id_list.push(part);
                }
            }
        }
        let num_independent_layers = tree_partition_layer_id_list.len();

        // ---- layer sets: base (eq. 7-3) + additional (F-7 .. F-9) ----
        let mut layer_set_layer_id_list: Vec<Vec<u8>> = vec![vec![0]];
        for row in inp.layer_id_included_flag {
            let ids: Vec<u8> = row
                .iter()
                .enumerate()
                .filter(|(_, &f)| f)
                .map(|(m, _)| m as u8)
                .collect();
            layer_set_layer_id_list.push(ids);
        }
        let num_add_layer_sets = if num_independent_layers > 1 {
            let v = br.ue()?;
            if v > 1023 {
                return Err(oor("num_add_layer_sets", v));
            }
            v
        } else {
            0
        };
        if !inp.base_layer_available_flag && num_add_layer_sets == 0 {
            return Err(oor("num_add_layer_sets", 0));
        }
        let mut highest_layer_idx_plus1 = Vec::with_capacity(num_add_layer_sets as usize);
        for _ in 0..num_add_layer_sets {
            let mut row = Vec::with_capacity(num_independent_layers.saturating_sub(1));
            let mut ids = Vec::new();
            for tree in tree_partition_layer_id_list.iter().skip(1) {
                let bits = ceil_log2(tree.len() as u32 + 1);
                let h = br.u(bits)?;
                if h as usize > tree.len() {
                    return Err(oor("highest_layer_idx_plus1", h));
                }
                ids.extend_from_slice(&tree[..h as usize]);
                row.push(h);
            }
            if ids.is_empty() {
                return Err(oor("highest_layer_idx_plus1", 0));
            }
            highest_layer_idx_plus1.push(row);
            layer_set_layer_id_list.push(ids);
        }
        let num_layer_sets = layer_set_layer_id_list.len();

        let vps_sub_layers_max_minus1_present_flag = br.u1()? != 0;
        let mut sub_layers_vps_max_minus1 = vec![inp.max_sub_layers_minus1; num_layers];
        if vps_sub_layers_max_minus1_present_flag {
            for v in &mut sub_layers_vps_max_minus1 {
                *v = br.u(3)? as u8;
                if *v > inp.max_sub_layers_minus1 {
                    return Err(oor("sub_layers_vps_max_minus1", u32::from(*v)));
                }
            }
        }
        // F-10.
        let mut max_sub_layers_in_layer_set_minus1 = Vec::with_capacity(num_layer_sets);
        for ls in &layer_set_layer_id_list {
            let mut m = 0u8;
            for &lid in ls {
                if let Some(idx) = layer_idx_in_vps[usize::from(lid)] {
                    m = m.max(sub_layers_vps_max_minus1[usize::from(idx)]);
                }
            }
            max_sub_layers_in_layer_set_minus1.push(m);
        }

        let max_tid_ref_present_flag = br.u1()? != 0;
        let mut max_tid_il_ref_pics_plus1 = vec![vec![7u8; num_layers]; num_layers];
        if max_tid_ref_present_flag {
            for i in 0..max_layers_minus1 {
                for j in i + 1..num_layers {
                    if direct_dependency_flag[j][i] {
                        max_tid_il_ref_pics_plus1[i][j] = br.u(3)? as u8;
                    }
                }
            }
        }
        let default_ref_layers_active_flag = br.u1()? != 0;
        let vps_num_profile_tier_level_minus1 = br.ue()?;
        if vps_num_profile_tier_level_minus1 > 63 {
            return Err(oor(
                "vps_num_profile_tier_level_minus1",
                vps_num_profile_tier_level_minus1,
            ));
        }
        // F.7.4.4 ordered list: [0] = base VPS PTL, then (when present)
        // the profile_tier_level( 0, … ) above, then the signalled ones.
        let mut vps_profile_tier_level = vec![inp.base_ptl.clone()];
        if let Some(p) = &base_layer_ptl {
            vps_profile_tier_level.push(p.clone());
        }
        let mut vps_profile_present_flag = Vec::new();
        let first = if base_internal { 2 } else { 1 };
        for _ in first..=vps_num_profile_tier_level_minus1 {
            let present = br.u1()? != 0;
            vps_profile_present_flag.push(present);
            let ptl = ProfileTierLevel::parse(br, present, inp.max_sub_layers_minus1)?;
            vps_profile_tier_level.push(ptl);
        }

        // ---- output layer sets ----
        let (num_add_olss, default_output_layer_idc) = if num_layer_sets > 1 {
            let n = br.ue()?;
            if n > 1023 {
                return Err(oor("num_add_olss", n));
            }
            (n, br.u(2)? as u8)
        } else {
            (0, 0)
        };
        let default_ols_idc = default_output_layer_idc.min(2);
        let num_output_layer_sets = num_add_olss as usize + num_layer_sets;
        let mut layer_set_idx_for_ols_minus1 = vec![0u32; num_output_layer_sets];
        let mut ols_idx_to_ls_idx = vec![0u16; num_output_layer_sets];
        let mut output_layer_flag: Vec<Vec<bool>> = Vec::with_capacity(num_output_layer_sets);
        let mut profile_tier_level_idx: Vec<Vec<u32>> = Vec::with_capacity(num_output_layer_sets);
        let mut alt_output_layer_flag = vec![false; num_output_layer_sets];
        let mut necessary_layer_flag: Vec<Vec<bool>> = Vec::with_capacity(num_output_layer_sets);
        let mut ols_highest_output_layer_id = vec![0u8; num_output_layer_sets];
        // OLS 0: layer set 0, output_layer_flag[0][0] = 1.
        output_layer_flag.push(vec![true]);
        necessary_layer_flag.push(vec![true]);
        profile_tier_level_idx.push(vec![u32::from(inp.max_layers_minus1 > 0 && base_internal)]);
        for i in 1..num_output_layer_sets {
            if num_layer_sets > 2 && i >= num_layer_sets {
                let bits = ceil_log2(num_layer_sets as u32 - 1);
                let v = br.u(bits)?;
                if v as usize + 1 >= num_layer_sets {
                    return Err(oor("layer_set_idx_for_ols_minus1", v));
                }
                layer_set_idx_for_ols_minus1[i] = v;
            }
            let ls_idx = if i < num_layer_sets {
                i
            } else {
                layer_set_idx_for_ols_minus1[i] as usize + 1
            };
            ols_idx_to_ls_idx[i] = ls_idx as u16;
            let ids = &layer_set_layer_id_list[ls_idx];
            let n = ids.len();
            let mut flags = vec![false; n];
            if i > usize::from(inp.num_layer_sets_minus1) || default_ols_idc == 2 {
                for f in &mut flags {
                    *f = br.u1()? != 0;
                }
            } else {
                // defaultOutputLayerIdc 0: all layers; 1: highest only.
                let highest = ids.iter().copied().max().unwrap_or(0);
                for (k, f) in flags.iter_mut().enumerate() {
                    *f = default_ols_idc == 0 || ids[k] == highest;
                }
            }
            // F-12 / F-13.
            let mut num_out = 0usize;
            for (k, &f) in flags.iter().enumerate() {
                if f {
                    num_out += 1;
                    ols_highest_output_layer_id[i] = ids[k];
                }
            }
            if num_out == 0 {
                return Err(oor("output_layer_flag", 0));
            }
            let mut necessary = vec![false; n];
            for k in 0..n {
                if flags[k] {
                    necessary[k] = true;
                    let cur = layer_idx_in_vps[usize::from(ids[k])];
                    for r in 0..k {
                        let rf = layer_idx_in_vps[usize::from(ids[r])];
                        if let (Some(c), Some(rr)) = (cur, rf) {
                            if dependency_flag[usize::from(c)][usize::from(rr)] {
                                necessary[r] = true;
                            }
                        }
                    }
                }
            }
            let mut ptl_idx = vec![0u32; n];
            for k in 0..n {
                if necessary[k] && vps_num_profile_tier_level_minus1 > 0 {
                    let bits = ceil_log2(vps_num_profile_tier_level_minus1 + 1);
                    let v = br.u(bits)?;
                    if v > vps_num_profile_tier_level_minus1 {
                        return Err(oor("profile_tier_level_idx", v));
                    }
                    ptl_idx[k] = v;
                }
            }
            if num_out == 1
                && !id_direct_ref_layer[usize::from(ols_highest_output_layer_id[i])].is_empty()
            {
                alt_output_layer_flag[i] = br.u1()? != 0;
            }
            output_layer_flag.push(flags);
            necessary_layer_flag.push(necessary);
            profile_tier_level_idx.push(ptl_idx);
        }

        // ---- rep formats ----
        let vps_num_rep_formats_minus1 = br.ue()?;
        if vps_num_rep_formats_minus1 > 255 {
            return Err(oor(
                "vps_num_rep_formats_minus1",
                vps_num_rep_formats_minus1,
            ));
        }
        let mut rep_formats = Vec::with_capacity(vps_num_rep_formats_minus1 as usize + 1);
        for _ in 0..=vps_num_rep_formats_minus1 {
            let rf = RepFormat::parse(br, rep_formats.last())?;
            rep_formats.push(rf);
        }
        let rep_format_idx_present_flag = if vps_num_rep_formats_minus1 > 0 {
            br.u1()? != 0
        } else {
            false
        };
        let mut vps_rep_format_idx: Vec<u32> = (0..num_layers as u32)
            .map(|i| i.min(vps_num_rep_formats_minus1))
            .collect();
        if rep_format_idx_present_flag {
            let bits = ceil_log2(vps_num_rep_formats_minus1 + 1);
            for i in usize::from(!base_internal)..num_layers {
                let v = br.u(bits)?;
                if v > vps_num_rep_formats_minus1 {
                    return Err(oor("vps_rep_format_idx", v));
                }
                vps_rep_format_idx[i] = v;
            }
        }
        let max_one_active_ref_layer_flag = br.u1()? != 0;
        let vps_poc_lsb_aligned_flag = br.u1()? != 0;
        let mut poc_lsb_not_present_flag = vec![false; num_layers];
        for i in 1..num_layers {
            if id_direct_ref_layer[usize::from(layer_id_in_nuh[i])].is_empty() {
                poc_lsb_not_present_flag[i] = br.u1()? != 0;
            }
        }

        // ---- dpb_size( ) ----
        let mut dpb_size = Vec::with_capacity(num_output_layer_sets.saturating_sub(1));
        for i in 1..num_output_layer_sets {
            let ls_idx = usize::from(ols_idx_to_ls_idx[i]);
            let ids = &layer_set_layer_id_list[ls_idx];
            let sub_layer_flag_info_present_flag = br.u1()? != 0;
            let max_sl = usize::from(max_sub_layers_in_layer_set_minus1[ls_idx]);
            let mut sub_layers: Vec<DpbSizeSubLayer> = Vec::with_capacity(max_sl + 1);
            for j in 0..=max_sl {
                let present = if j > 0 && sub_layer_flag_info_present_flag {
                    br.u1()? != 0
                } else {
                    j == 0
                };
                let mut row = if present {
                    let mut dpb = vec![0u32; ids.len()];
                    for (k, d) in dpb.iter_mut().enumerate() {
                        if necessary_layer_flag[i][k] && (base_internal || ids[k] != 0) {
                            *d = br.ue()?;
                        }
                    }
                    DpbSizeSubLayer {
                        sub_layer_dpb_info_present_flag: true,
                        max_vps_dec_pic_buffering_minus1: dpb,
                        max_vps_num_reorder_pics: br.ue()?,
                        max_vps_latency_increase_plus1: br.ue()?,
                    }
                } else {
                    // F.7.4.3.1.3: inherit the previous sub-layer.
                    sub_layers.last().cloned().unwrap_or_default()
                };
                row.sub_layer_dpb_info_present_flag = present;
                sub_layers.push(row);
            }
            dpb_size.push(DpbSizeOls {
                sub_layer_flag_info_present_flag,
                sub_layers,
            });
        }

        // ---- dependency types (F-14) ----
        let direct_dep_type_len_minus2 = br.ue()?;
        if direct_dep_type_len_minus2 > 30 {
            return Err(oor(
                "direct_dep_type_len_minus2",
                direct_dep_type_len_minus2,
            ));
        }
        let type_bits = (direct_dep_type_len_minus2 + 2) as u8;
        let direct_dependency_all_layers_flag = br.u1()? != 0;
        let mut direct_dependency_type = vec![vec![u32::MAX; num_layers]; num_layers];
        let direct_dependency_all_layers_type = if direct_dependency_all_layers_flag {
            let t = br.u(type_bits)?;
            for i in 0..num_layers {
                for j in 0..num_layers {
                    if direct_dependency_flag[i][j] {
                        direct_dependency_type[i][j] = t;
                    }
                }
            }
            t
        } else {
            for i in usize::from(!base_internal) + 1..num_layers {
                for j in usize::from(!base_internal)..i {
                    if direct_dependency_flag[i][j] {
                        direct_dependency_type[i][j] = br.u(type_bits)?;
                    }
                }
            }
            0
        };
        if !base_internal {
            for (i, row) in direct_dependency_type.iter_mut().enumerate().skip(1) {
                if direct_dependency_flag[i][0] {
                    row[0] = 0;
                }
            }
        }
        let mut inter_layer_sample_prediction_enabled = vec![vec![false; num_layers]; num_layers];
        let mut inter_layer_motion_prediction_enabled = vec![vec![false; num_layers]; num_layers];
        for i in 0..num_layers {
            for j in 0..num_layers {
                if direct_dependency_flag[i][j] {
                    let t = direct_dependency_type[i][j].wrapping_add(1);
                    inter_layer_sample_prediction_enabled[i][j] = t & 1 != 0;
                    inter_layer_motion_prediction_enabled[i][j] = t & 2 != 0;
                }
            }
        }

        let vps_non_vui_extension_length = br.ue()?;
        if vps_non_vui_extension_length > 4096 {
            return Err(oor(
                "vps_non_vui_extension_length",
                vps_non_vui_extension_length,
            ));
        }
        br.skip(8 * vps_non_vui_extension_length as usize)?;
        let vps_vui_present_flag = br.u1()? != 0;

        let layers = LayerModel {
            max_layers_minus1: max_layers_minus1 as u8,
            layer_id_in_nuh: layer_id_in_nuh.clone(),
            layer_idx_in_vps,
            scalability_id,
            view_id,
            num_views,
            dependency_flag,
            id_direct_ref_layer,
            id_ref_layer,
            id_predicted_layer,
            tree_partition_layer_id_list,
            layer_set_layer_id_list,
            ols_idx_to_ls_idx,
            output_layer_flag: output_layer_flag.clone(),
            necessary_layer_flag,
            ols_highest_output_layer_id,
            max_sub_layers_in_layer_set_minus1,
            inter_layer_sample_prediction_enabled,
            inter_layer_motion_prediction_enabled,
        };

        let (vps_vui, vps_vui_malformed) = if vps_vui_present_flag {
            // vps_vui_alignment_bit_equal_to_one until byte aligned.
            while br.bit_pos() % 8 != 0 {
                br.u1()?;
            }
            match VpsVui::parse(br, inp, &layers, &sub_layers_vps_max_minus1) {
                Ok(v) => (Some(v), false),
                Err(_) => (None, true),
            }
        } else {
            (None, false)
        };

        Ok(Self {
            base_layer_ptl,
            splitting_flag,
            scalability_mask_flag,
            dimension_id_len_minus1,
            vps_nuh_layer_id_present_flag,
            dimension_id,
            view_id_len,
            view_id_val,
            direct_dependency_flag,
            num_add_layer_sets,
            highest_layer_idx_plus1,
            vps_sub_layers_max_minus1_present_flag,
            sub_layers_vps_max_minus1,
            max_tid_ref_present_flag,
            max_tid_il_ref_pics_plus1,
            default_ref_layers_active_flag,
            vps_num_profile_tier_level_minus1,
            vps_profile_present_flag,
            vps_profile_tier_level,
            num_add_olss,
            default_output_layer_idc,
            layer_set_idx_for_ols_minus1,
            output_layer_flag,
            profile_tier_level_idx,
            alt_output_layer_flag,
            rep_formats,
            rep_format_idx_present_flag,
            vps_rep_format_idx,
            max_one_active_ref_layer_flag,
            vps_poc_lsb_aligned_flag,
            poc_lsb_not_present_flag,
            dpb_size,
            direct_dep_type_len_minus2,
            direct_dependency_all_layers_flag,
            direct_dependency_all_layers_type,
            direct_dependency_type,
            vps_non_vui_extension_length,
            vps_vui_present_flag,
            vps_vui,
            vps_vui_malformed,
            layers,
        })
    }

    /// The `rep_format( )` that applies to the layer with the given
    /// `nuh_layer_id` (`vps_rep_format_idx[ LayerIdxInVps[ id ] ]`), or
    /// `None` for an id the VPS does not list.
    #[must_use]
    pub fn rep_format_for_layer(&self, nuh_layer_id: u8) -> Option<&RepFormat> {
        let idx = self.layers.layer_idx(nuh_layer_id)?;
        let rf = *self.vps_rep_format_idx.get(idx)?;
        self.rep_formats.get(rf as usize)
    }

    /// `max_vps_dec_pic_buffering_minus1[ ols ][ layer ][ sub_layer ]` for
    /// the layer with `nuh_layer_id` in output layer set `ols_idx`
    /// (OLS 0 has no `dpb_size( )` row: `None`).
    #[must_use]
    pub fn max_dec_pic_buffering_minus1(
        &self,
        ols_idx: usize,
        nuh_layer_id: u8,
        sub_layer: usize,
    ) -> Option<u32> {
        let row = self.dpb_size.get(ols_idx.checked_sub(1)?)?;
        let k = self
            .layers
            .ols_layer_ids(ols_idx)
            .iter()
            .position(|&id| id == nuh_layer_id)?;
        let sl = row
            .sub_layers
            .get(sub_layer.min(row.sub_layers.len().saturating_sub(1)))?;
        sl.max_vps_dec_pic_buffering_minus1.get(k).copied()
    }

    /// `max_vps_num_reorder_pics[ ols ][ sub_layer ]` (OLS 0: `None`).
    #[must_use]
    pub fn max_num_reorder_pics(&self, ols_idx: usize, sub_layer: usize) -> Option<u32> {
        let row = self.dpb_size.get(ols_idx.checked_sub(1)?)?;
        row.sub_layers
            .get(sub_layer.min(row.sub_layers.len().saturating_sub(1)))
            .map(|s| s.max_vps_num_reorder_pics)
    }
}

impl VpsVui {
    fn parse(
        br: &mut BitReader<'_>,
        inp: &VpsExtensionInputs<'_>,
        layers: &LayerModel,
        sub_layers_vps_max_minus1: &[u8],
    ) -> Result<Self, VpsError> {
        let base_internal = inp.base_layer_internal_flag;
        let num_layers = layers.layer_id_in_nuh.len();
        let cross_layer_pic_type_aligned_flag = br.u1()? != 0;
        let cross_layer_irap_aligned_flag = if cross_layer_pic_type_aligned_flag {
            true
        } else {
            br.u1()? != 0
        };
        let all_layers_idr_aligned_flag = if cross_layer_irap_aligned_flag {
            br.u1()? != 0
        } else {
            false
        };
        let bit_rate_present_vps_flag = br.u1()? != 0;
        let pic_rate_present_vps_flag = br.u1()? != 0;
        let num_layer_sets = layers.num_layer_sets();
        let mut rates: Vec<Vec<VpsVuiRateCell>> = vec![Vec::new(); num_layer_sets];
        if bit_rate_present_vps_flag || pic_rate_present_vps_flag {
            for (i, row) in rates
                .iter_mut()
                .enumerate()
                .skip(usize::from(!base_internal))
            {
                let max_sl = usize::from(layers.max_sub_layers_in_layer_set_minus1[i]);
                for _ in 0..=max_sl {
                    let mut cell = VpsVuiRateCell::default();
                    if bit_rate_present_vps_flag {
                        cell.bit_rate_present_flag = br.u1()? != 0;
                    }
                    if pic_rate_present_vps_flag {
                        cell.pic_rate_present_flag = br.u1()? != 0;
                    }
                    if cell.bit_rate_present_flag {
                        cell.avg_bit_rate = br.u(16)? as u16;
                        cell.max_bit_rate = br.u(16)? as u16;
                    }
                    if cell.pic_rate_present_flag {
                        cell.constant_pic_rate_idc = br.u(2)? as u8;
                        cell.avg_pic_rate = br.u(16)? as u16;
                    }
                    row.push(cell);
                }
            }
        }
        let video_signal_info_idx_present_flag = br.u1()? != 0;
        let vps_num_video_signal_info_minus1 = if video_signal_info_idx_present_flag {
            br.u(4)? as usize
        } else {
            0
        };
        let mut video_signal_info = Vec::with_capacity(vps_num_video_signal_info_minus1 + 1);
        for _ in 0..=vps_num_video_signal_info_minus1 {
            video_signal_info.push(VideoSignalInfo {
                video_vps_format: br.u(3)? as u8,
                video_full_range_vps_flag: br.u1()? != 0,
                colour_primaries_vps: br.u(8)? as u8,
                transfer_characteristics_vps: br.u(8)? as u8,
                matrix_coeffs_vps: br.u(8)? as u8,
            });
        }
        // F.7.4.3.1.4 inference: absent -> present_flag ? 0 : i.
        let mut vps_video_signal_info_idx: Vec<u8> = (0..num_layers)
            .map(|i| {
                if video_signal_info_idx_present_flag {
                    0
                } else {
                    (i.min(vps_num_video_signal_info_minus1)) as u8
                }
            })
            .collect();
        if video_signal_info_idx_present_flag && vps_num_video_signal_info_minus1 > 0 {
            for v in vps_video_signal_info_idx
                .iter_mut()
                .skip(usize::from(!base_internal))
            {
                *v = br.u(4)? as u8;
            }
        }
        let tiles_not_in_use_flag = br.u1()? != 0;
        let mut tiles_in_use_flag = vec![false; num_layers];
        let mut loop_filter_not_across_tiles_flag = vec![false; num_layers];
        let mut tile_boundaries_aligned_flag: Vec<Vec<bool>> = vec![Vec::new(); num_layers];
        if !tiles_not_in_use_flag {
            for i in usize::from(!base_internal)..num_layers {
                tiles_in_use_flag[i] = br.u1()? != 0;
                if tiles_in_use_flag[i] {
                    loop_filter_not_across_tiles_flag[i] = br.u1()? != 0;
                }
            }
            for i in usize::from(!base_internal) + 1..num_layers {
                let lid = layers.layer_id_in_nuh[i];
                let refs = layers.direct_ref_layers(lid).to_vec();
                let mut row = vec![false; refs.len()];
                for (j, &r) in refs.iter().enumerate() {
                    let layer_idx = layers.layer_idx(r).unwrap_or(0);
                    if tiles_in_use_flag[i] && tiles_in_use_flag[layer_idx] {
                        row[j] = br.u1()? != 0;
                    }
                }
                tile_boundaries_aligned_flag[i] = row;
            }
        }
        let wpp_not_in_use_flag = br.u1()? != 0;
        let mut wpp_in_use_flag = vec![false; num_layers];
        if !wpp_not_in_use_flag {
            for f in wpp_in_use_flag.iter_mut().skip(usize::from(!base_internal)) {
                *f = br.u1()? != 0;
            }
        }
        let single_layer_for_non_irap_flag = br.u1()? != 0;
        let higher_layer_irap_skip_flag = br.u1()? != 0;
        let ilp_restricted_ref_layers_flag = br.u1()? != 0;
        let mut ilp_restrictions: Vec<Vec<IlpRestriction>> = vec![Vec::new(); num_layers];
        if ilp_restricted_ref_layers_flag {
            for i in 1..num_layers {
                let lid = layers.layer_id_in_nuh[i];
                let refs = layers.direct_ref_layers(lid).to_vec();
                let mut row = vec![IlpRestriction::default(); refs.len()];
                for (j, &r) in refs.iter().enumerate() {
                    if base_internal || r > 0 {
                        let mut e = IlpRestriction {
                            min_spatial_segment_offset_plus1: br.ue()?,
                            ..IlpRestriction::default()
                        };
                        if e.min_spatial_segment_offset_plus1 > 0 {
                            e.ctu_based_offset_enabled_flag = br.u1()? != 0;
                            if e.ctu_based_offset_enabled_flag {
                                e.min_horizontal_ctu_offset_plus1 = br.ue()?;
                            }
                        }
                        row[j] = e;
                    }
                }
                ilp_restrictions[i] = row;
            }
        }
        let vps_vui_bsp_hrd_present_flag = br.u1()? != 0;
        let mut vps_num_add_hrd_params = 0;
        if vps_vui_bsp_hrd_present_flag {
            vps_num_add_hrd_params =
                parse_bsp_hrd_params(br, inp, layers, sub_layers_vps_max_minus1)?;
        }
        let mut base_layer_parameter_set_compatibility_flag = vec![false; num_layers];
        for i in 1..num_layers {
            let lid = layers.layer_id_in_nuh[i];
            if layers.num_direct_ref_layers(lid) == 0 {
                base_layer_parameter_set_compatibility_flag[i] = br.u1()? != 0;
            }
        }
        Ok(Self {
            cross_layer_pic_type_aligned_flag,
            cross_layer_irap_aligned_flag,
            all_layers_idr_aligned_flag,
            bit_rate_present_vps_flag,
            pic_rate_present_vps_flag,
            rates,
            video_signal_info_idx_present_flag,
            video_signal_info,
            vps_video_signal_info_idx,
            tiles_not_in_use_flag,
            tiles_in_use_flag,
            loop_filter_not_across_tiles_flag,
            tile_boundaries_aligned_flag,
            wpp_not_in_use_flag,
            wpp_in_use_flag,
            single_layer_for_non_irap_flag,
            higher_layer_irap_skip_flag,
            ilp_restricted_ref_layers_flag,
            ilp_restrictions,
            vps_vui_bsp_hrd_present_flag,
            vps_num_add_hrd_params,
            base_layer_parameter_set_compatibility_flag,
        })
    }
}

/// Walk `vps_vui_bsp_hrd_params( )` (F.7.3.2.1.6), returning
/// `vps_num_add_hrd_params`. The additional `hrd_parameters( )` bodies
/// are parsed (to advance the reader exactly) and dropped.
fn parse_bsp_hrd_params(
    br: &mut BitReader<'_>,
    inp: &VpsExtensionInputs<'_>,
    layers: &LayerModel,
    _sub_layers_vps_max_minus1: &[u8],
) -> Result<u32, VpsError> {
    let vps_num_add_hrd_params = br.ue()?;
    // F.7.4.3.1.6: 0 .. 1024 − vps_num_hrd_parameters.
    if vps_num_add_hrd_params > 1024u32.saturating_sub(inp.num_hrd_parameters) {
        return Err(oor("vps_num_add_hrd_params", vps_num_add_hrd_params));
    }
    let total = inp.num_hrd_parameters + vps_num_add_hrd_params;
    let mut prev_common = None;
    for i in inp.num_hrd_parameters..total {
        let cprms_add_present_flag = if i > 0 { br.u1()? != 0 } else { true };
        let num_sub_layer_hrd_minus1 = br.ue()?;
        if num_sub_layer_hrd_minus1 > u32::from(inp.max_sub_layers_minus1) {
            return Err(oor("num_sub_layer_hrd_minus1", num_sub_layer_hrd_minus1));
        }
        let hrd = HrdParameters::parse(
            br,
            cprms_add_present_flag,
            num_sub_layer_hrd_minus1 as u8,
            prev_common.as_ref(),
        )
        .map_err(VpsError::Hrd)?;
        prev_common = hrd.common.or(prev_common);
    }
    if total > 0 {
        for h in 1..layers.num_output_layer_sets() {
            let ls_idx = usize::from(layers.ols_idx_to_ls_idx[h]);
            let num_layers_in_ls = layers.layer_set_layer_id_list[ls_idx].len();
            let num_signalled_partitioning_schemes = br.ue()?;
            if num_signalled_partitioning_schemes > 16 {
                return Err(oor(
                    "num_signalled_partitioning_schemes",
                    num_signalled_partitioning_schemes,
                ));
            }
            // num_partitions_in_scheme_minus1[ h ][ 0 ] is inferred 0
            // (one partition holding every layer).
            let mut num_partitions_minus1 =
                vec![0u32; num_signalled_partitioning_schemes as usize + 1];
            for j in 1..=num_signalled_partitioning_schemes as usize {
                let n = br.ue()?;
                if n as usize >= num_layers_in_ls {
                    return Err(oor("num_partitions_in_scheme_minus1", n));
                }
                num_partitions_minus1[j] = n;
                for _ in 0..=n {
                    for _ in 0..num_layers_in_ls {
                        br.u1()?; // layer_included_in_partition_flag
                    }
                }
            }
            let max_sl = usize::from(layers.max_sub_layers_in_layer_set_minus1[ls_idx]);
            for &np in num_partitions_minus1.iter() {
                for _ in 0..=max_sl {
                    let num_bsp_schedules_minus1 = br.ue()?;
                    if num_bsp_schedules_minus1 > 31 {
                        return Err(oor("num_bsp_schedules_minus1", num_bsp_schedules_minus1));
                    }
                    for _ in 0..=num_bsp_schedules_minus1 {
                        for _ in 0..=np {
                            if total > 1 {
                                br.u(ceil_log2(total))?; // bsp_hrd_idx
                            }
                            br.ue()?; // bsp_sched_idx
                        }
                    }
                }
            }
        }
    }
    Ok(vps_num_add_hrd_params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nal::strip_emulation_prevention;
    use crate::vps::HevcVps;

    /// The VPS NAL unit (header included) of a two-view MV-HEVC stream
    /// written by an OS media framework's stereo encoder: two layers,
    /// layer 1 directly dependent on layer 0, a VPS VUI.
    const STEREO_VPS_NAL: &[u8] = &[
        0x40, 0x01, 0x0C, 0x11, 0xFF, 0xFF, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0xB0, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x03, 0x00, 0x78, 0x15, 0xC1, 0x5B, 0x78, 0x20, 0x00, 0x28, 0x24, 0x59,
        0x70, 0x60, 0x20, 0x00, 0x00, 0x0B, 0xF8, 0x00, 0x00, 0x03, 0x00, 0x00, 0x07, 0x88, 0xD0,
        0x3C, 0x00, 0x3C, 0x0A, 0x00, 0xC5, 0x2B, 0xF7, 0x08, 0x50, 0x08, 0x08, 0x08, 0x00, 0x80,
    ];

    fn stereo_vps() -> HevcVps {
        let rbsp = strip_emulation_prevention(STEREO_VPS_NAL);
        HevcVps::parse(&rbsp[2..]).expect("stereo VPS parses")
    }

    #[test]
    fn stereo_vps_extension_layer_model() {
        let vps = stereo_vps();
        assert_eq!(vps.max_layers_minus1, 1);
        assert!(vps.vps_extension_flag);
        // vps_extension2_flag == 0 sits right after the VUI: a VUI
        // parse that ended on the wrong bit would flip it.
        assert!(!vps.vps_extension2_flag);
        assert!(vps.opaque_tail.is_none());
        let ext = vps.extension.as_ref().expect("vps_extension present");
        // profile_tier_level( 0, 0 ) for the base layer, then two more.
        assert!(ext.base_layer_ptl.is_some());
        assert_eq!(ext.vps_num_profile_tier_level_minus1, 2);
        assert_eq!(ext.vps_profile_tier_level.len(), 3);
        assert_eq!(ext.vps_profile_tier_level[0].general_profile_idc, 1);
        // G.11.1.1 Multiview Main for the second view.
        assert_eq!(ext.vps_profile_tier_level[2].general_profile_idc, 6);
        // Table F.1: only the multiview dimension is present.
        assert!(!ext.splitting_flag);
        assert!(ext.scalability_mask_flag[1]);
        assert_eq!(ext.scalability_mask_flag.iter().filter(|&&f| f).count(), 1);
        assert_eq!(ext.dimension_id_len_minus1, vec![2]);
        assert_eq!(ext.dimension_id, vec![vec![0], vec![1]]);
        assert_eq!(ext.view_id_len, 1);
        assert_eq!(ext.view_id_val, vec![0, 1]);
        assert_eq!(
            ext.direct_dependency_flag,
            vec![vec![false, false], vec![true, false]]
        );
        assert!(ext.default_ref_layers_active_flag);
        assert!(ext.max_one_active_ref_layer_flag);
        assert!(ext.vps_poc_lsb_aligned_flag);
        assert_eq!(ext.default_output_layer_idc, 0);
        assert_eq!(ext.num_add_olss, 0);
        assert_eq!(ext.profile_tier_level_idx, vec![vec![1], vec![1, 2]]);
        assert_eq!(ext.rep_formats.len(), 1);
        let rf = ext.rep_formats[0];
        assert_eq!(
            (
                rf.pic_width_vps_in_luma_samples,
                rf.pic_height_vps_in_luma_samples,
                rf.chroma_format_vps_idc,
                rf.bit_depth_vps_luma_minus8
            ),
            (960, 960, 1, 0)
        );
        assert_eq!(ext.vps_rep_format_idx, vec![0, 0]);
        assert_eq!(ext.dpb_size.len(), 1);
        let dpb = &ext.dpb_size[0].sub_layers[0];
        assert_eq!(dpb.max_vps_dec_pic_buffering_minus1, vec![4, 4]);
        assert_eq!(dpb.max_vps_num_reorder_pics, 2);
        assert_eq!(ext.max_dec_pic_buffering_minus1(1, 1, 0), Some(4));
        assert_eq!(ext.max_num_reorder_pics(1, 0), Some(2));
        assert!(ext.direct_dependency_all_layers_flag);
        assert_eq!(ext.direct_dependency_all_layers_type, 2);
        assert!(ext.vps_vui_present_flag);
        assert!(!ext.vps_vui_malformed);
        let vui = ext.vps_vui.as_ref().expect("VUI parsed");
        assert!(vui.video_signal_info_idx_present_flag);
        assert_eq!(vui.video_signal_info.len(), 1);
        assert_eq!(vui.video_signal_info[0].colour_primaries_vps, 1);
        assert!(!vui.vps_vui_bsp_hrd_present_flag);

        // ---- F.7.4.3.1.1 derived model ----
        let m = &ext.layers;
        assert_eq!(m.layer_id_in_nuh, vec![0, 1]);
        assert_eq!(m.layer_idx(1), Some(1));
        assert_eq!(m.layer_idx(2), None);
        assert_eq!(m.num_views, 2);
        assert_eq!((m.view_id_of(0), m.view_id_of(1)), (0, 1));
        assert_eq!((m.view_order_idx(0), m.view_order_idx(1)), (0, 1));
        assert_eq!(m.num_direct_ref_layers(1), 1);
        assert_eq!(m.direct_ref_layers(1), &[0]);
        assert_eq!(m.num_direct_ref_layers(0), 0);
        assert_eq!(m.num_predicted_layers(0), 1);
        assert_eq!(m.tree_partition_layer_id_list, vec![vec![0, 1]]);
        assert_eq!(m.layer_set_layer_id_list, vec![vec![0], vec![0, 1]]);
        assert_eq!(m.num_output_layer_sets(), 2);
        assert_eq!(m.ols_output_layer_ids(1), vec![0, 1]);
        assert_eq!(m.ols_highest_output_layer_id, vec![0, 1]);
        assert!(m.sample_prediction_enabled(1, 0));
        assert!(m.motion_prediction_enabled(1, 0));
        assert!(!m.sample_prediction_enabled(0, 1));
        assert_eq!(
            ext.rep_format_for_layer(1)
                .map(|r| r.pic_width_vps_in_luma_samples),
            Some(960)
        );
    }

    /// The layer-1 SPS (`sps_ext_or_max_sub_layers_minus1 == 7`) and
    /// PPS NAL units of the same stereo stream.
    const STEREO_SPS1_NAL: &[u8] = &[0x42, 0x09, 0x0E, 0x82, 0x2E, 0x45, 0x8A, 0xA0, 0x05, 0x01];
    const STEREO_PPS1_NAL: &[u8] = &[0x44, 0x09, 0x48, 0x02, 0xCB, 0xC1, 0x4D, 0xA8, 0x05];
    const STEREO_SPS0_NAL: &[u8] = &[
        0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0xB0, 0x00, 0x00, 0x03, 0x00, 0x00,
        0x03, 0x00, 0x78, 0xA0, 0x07, 0x82, 0x00, 0xF0, 0x58, 0x81, 0x5E, 0xE4, 0x59, 0x54, 0xD4,
        0x04, 0x04, 0x04, 0x02,
    ];

    #[test]
    fn stereo_layer1_sps_infers_rep_format_from_vps() {
        use crate::pps::PicParameterSet;
        use crate::sps::{SeqParameterSet, SpsError};
        let vps = stereo_vps();
        let rbsp1 = strip_emulation_prevention(STEREO_SPS1_NAL);
        // Without the VPS the multilayer-extension form cannot be
        // completed.
        assert!(matches!(
            SeqParameterSet::parse_layered(&rbsp1[2..], 1, None),
            Err(SpsError::MissingVps)
        ));
        let sps1 = SeqParameterSet::parse_layered(&rbsp1[2..], 1, Some(&vps)).expect("layer-1 SPS");
        assert_eq!(sps1.nuh_layer_id, 1);
        assert!(sps1.multilayer_ext_sps_flag);
        assert_eq!(sps1.sps_ext_or_max_sub_layers_minus1, 7);
        assert_eq!(sps1.max_sub_layers_minus1, 0);
        assert_eq!(sps1.sps_id, 1);
        assert!(!sps1.update_rep_format_flag);
        assert_eq!(
            (
                sps1.pic_width_in_luma_samples,
                sps1.pic_height_in_luma_samples,
                sps1.chroma_format_idc,
                sps1.bit_depth_luma()
            ),
            (960, 960, 1, 8)
        );
        assert_eq!(sps1.ptl.general_profile_idc, 0);
        // The base-layer SPS parsed through the same entry point is the
        // ordinary form and must agree on the geometry.
        let rbsp0 = strip_emulation_prevention(STEREO_SPS0_NAL);
        let sps0 = SeqParameterSet::parse_layered(&rbsp0[2..], 0, Some(&vps)).expect("layer-0 SPS");
        assert!(!sps0.multilayer_ext_sps_flag);
        assert_eq!(sps0.sps_id, 0);
        assert_eq!(sps0.pic_width_in_luma_samples, 960);
        assert_eq!(sps0.log2_ctb_size(), sps1.log2_ctb_size());
        // The layer-1 PPS refers to SPS 1.
        let prbsp = strip_emulation_prevention(STEREO_PPS1_NAL);
        let pps1 = PicParameterSet::parse(&prbsp[2..]).expect("layer-1 PPS");
        assert_eq!((pps1.pps_id, pps1.sps_id), (1, 1));
        // Robustness: every truncation / bit flip of the layer-1 SPS
        // must return, never panic.
        let body = &rbsp1[2..];
        for cut in 0..body.len() {
            let _ = SeqParameterSet::parse_layered(&body[..cut], 1, Some(&vps));
        }
        for bit in 0..body.len() * 8 {
            let mut m = body.to_vec();
            m[bit / 8] ^= 0x80 >> (bit % 8);
            let _ = SeqParameterSet::parse_layered(&m, 1, Some(&vps));
        }
    }

    /// First 40 bytes of the stereo stream's slice NAL units: the
    /// layer-0 IDR_W_RADL, the layer-1 IDR_N_LP of the same access unit,
    /// and the layer-0 / layer-1 TRAIL_R pair of the next one.
    const STEREO_IDR0_HEAD: &[u8] = &[
        0x28, 0x01, 0xAF, 0x87, 0x86, 0x1E, 0x28, 0xFA, 0x97, 0x21, 0x31, 0x8C, 0x49, 0x42, 0x98,
        0x51, 0x30, 0x90, 0xA8, 0x99, 0x0C, 0x38, 0x9F, 0x87, 0xA9, 0x0A, 0x10, 0x20, 0x50, 0x00,
        0x98, 0x09, 0x14, 0xBC, 0x80, 0x3F, 0x7C, 0x02, 0xBF, 0x24, 0x11, 0xBF, 0xAB, 0xFF, 0x13,
        0x20, 0x0C, 0x82, 0x44, 0xF6, 0x22, 0x2D, 0xC7, 0xA2, 0x16, 0x5F, 0xB2, 0x0C, 0x86, 0x79,
        0x49, 0x96, 0x63, 0x3E, 0x70, 0x59, 0x71, 0xF5, 0xAB, 0x2B, 0x7D, 0xDD, 0x02, 0xF7, 0xD4,
        0x1B, 0x9C, 0x3F, 0x3F, 0x26, 0x00, 0x1B, 0x45, 0x45, 0x32, 0x3A, 0x7F, 0xD3, 0x87, 0x71,
        0x39, 0x45, 0xFA, 0x32, 0x5D, 0x70, 0x19, 0x20, 0x8F, 0x2A, 0xBC, 0x6C, 0x82, 0xA0, 0xA7,
        0x4D, 0xC4, 0x57, 0x1C, 0x92, 0x74, 0x65, 0x39, 0x84, 0xD4, 0xF0, 0x4E, 0xC9, 0x56, 0x4C,
        0x05, 0x24, 0xC0, 0x43, 0xB2, 0x5B, 0xE7, 0x03, 0xF6, 0x37, 0x9E, 0x65, 0x17, 0x3E, 0xC1,
        0xEB, 0x3F, 0xF9, 0x3C, 0x00, 0xDF, 0x05, 0x52, 0x02, 0xF4, 0xBF, 0xD8, 0x38, 0x35, 0xD9,
        0xC1, 0x22, 0x28, 0xFC, 0x4B, 0x90, 0xA0, 0xFE, 0xFB, 0x7D,
    ];
    const STEREO_IDR1_HEAD: &[u8] = &[
        0x2A, 0x09, 0x92, 0x00, 0x0D, 0xF8, 0x78, 0x52, 0xD6, 0x6B, 0x94, 0x25, 0x39, 0x2C, 0x5B,
        0x8E, 0x42, 0xF0, 0xCA, 0x3D, 0x10, 0x25, 0x20, 0xF8, 0x39, 0x90, 0xE3, 0xB0, 0x50, 0x12,
        0x0C, 0xB0, 0x0F, 0x63, 0xA2, 0x7E, 0x1D, 0xCF, 0xD9, 0xC5, 0x1F, 0x81, 0x70, 0x75, 0xFE,
        0x0A, 0xF8, 0x9E, 0x41, 0xFB, 0xA3, 0x02, 0x17, 0x7D, 0xC1, 0x68, 0xC4, 0xAC, 0x5D, 0x76,
        0x5D, 0x0C, 0x52, 0xBF, 0x4C, 0xA2, 0xBA, 0xAE, 0x3E, 0x7C, 0xCD, 0x87, 0x44, 0x0F, 0xFB,
        0x24, 0xA0, 0x74, 0x2A, 0xA5, 0xCD, 0x9A, 0x65, 0x72, 0xE2, 0xE9, 0x41, 0xCD, 0xDD, 0x7F,
        0xD0, 0x9F, 0xC0, 0xA7, 0x5F, 0xCA, 0xB5, 0xCA, 0x64, 0x86, 0x7F, 0x4F, 0x95, 0x4C, 0xBC,
        0xC5, 0x52, 0x4C, 0xC1, 0x30, 0x8B, 0x51, 0x24, 0xE2, 0x43, 0x0F, 0xCC, 0x21, 0xE2, 0x21,
        0x5A, 0x5C, 0x1C, 0x2B, 0xA7, 0xFB, 0xF6, 0xCC, 0xDF, 0xF2, 0x5B, 0xE1, 0xAE, 0x36, 0xDA,
        0x08, 0xA0, 0x29, 0x09, 0x88, 0x50, 0x4F, 0xB8, 0xDC, 0x37, 0x1C, 0x18, 0x1D, 0xA3, 0x69,
        0x50, 0x47, 0x75, 0xE9, 0x6F, 0xEA, 0xFB, 0xA8, 0xA7, 0x39,
    ];
    const STEREO_TRAIL0_HEAD: &[u8] = &[
        0x02, 0x01, 0xD0, 0x02, 0x2A, 0xBE, 0xC3, 0xC2, 0xCB, 0xC9, 0x21, 0x2B, 0x23, 0x94, 0x58,
        0x0A, 0xC1, 0x4E, 0x12, 0x81, 0x88, 0x27, 0x06, 0x40, 0xBC, 0x1A, 0x03, 0x70, 0x4E, 0x09,
        0x41, 0x38, 0x1D, 0x48, 0xBC, 0x53, 0x51, 0x6B, 0x9E, 0x1F, 0xB3, 0xB6, 0x16, 0xDC, 0x42,
        0xC0, 0x58, 0x0F, 0xC0, 0xFD, 0x0F, 0x34, 0xFA, 0x26, 0xFF, 0xFC, 0x6A, 0x1E, 0xEA, 0x1F,
        0x11, 0x28, 0x72, 0xBD, 0xBF, 0xC1, 0xDF, 0x1F, 0x38, 0xD9, 0x0A, 0xF9, 0x76, 0x4D, 0x15,
        0xBE, 0xE1, 0xC5, 0x3F, 0x7D, 0x4A, 0xED, 0x3D, 0xCB, 0x37, 0xAD, 0x6B, 0xEC, 0xE4, 0x1C,
        0x25, 0x3C, 0x7C, 0x84, 0xD8, 0xD4, 0x46, 0x0A, 0x0F, 0x36, 0x9C, 0x08, 0x2C, 0x6B, 0x32,
        0x56, 0xCB, 0x37, 0x68, 0xD2, 0x0A, 0xE3, 0x3D, 0x6D, 0x3D, 0x13, 0x44, 0xF1, 0x80, 0x50,
        0x5B, 0x25, 0xEF, 0x31, 0xA5, 0xA1, 0x2A, 0x7B, 0xA1, 0xF1, 0x32, 0x43, 0x32, 0xDE, 0xB8,
        0xEB, 0xC1, 0x72, 0x90, 0x21, 0x31, 0xFA, 0xA6, 0x33, 0x95, 0x8A, 0xAE, 0x53, 0x0F, 0x7C,
        0x96, 0xD2, 0x98, 0x99, 0x31, 0x04, 0xBC, 0x6B, 0x5F, 0xE7,
    ];
    const STEREO_TRAIL1_HEAD: &[u8] = &[
        0x02, 0x09, 0xA8, 0x02, 0x2A, 0xBF, 0x96, 0x1E, 0x16, 0x33, 0x01, 0xF8, 0x71, 0x0A, 0xA1,
        0x84, 0x22, 0x05, 0xA0, 0x86, 0x0D, 0x81, 0x90, 0x2D, 0x06, 0x81, 0x14, 0x1D, 0x82, 0xA0,
        0x56, 0x1F, 0x00, 0x98, 0xB4, 0x7B, 0xF0, 0x4D, 0xF1, 0xC1, 0xD7, 0xE8, 0xFA, 0x1F, 0x99,
        0x0E, 0x01, 0xC0, 0x52, 0xEB, 0xFF, 0x3D, 0x3D, 0x4A, 0x30, 0xA6, 0x76, 0x29, 0x61, 0x0E,
        0x02, 0xD7, 0xD7, 0xDC, 0x19, 0x01, 0x44, 0x22, 0x21, 0xC7, 0x30, 0x04, 0x3D, 0x15, 0x22,
        0xD7, 0x9B, 0x7C, 0xB2, 0x73, 0x0C, 0xDB, 0x9E, 0x3C, 0xD0, 0xD9, 0xB0, 0xD3, 0x98, 0x6D,
        0x18, 0x88, 0x81, 0x21, 0x2C, 0x21, 0x9F, 0xD4, 0x9F, 0xEE, 0x1B, 0xCB, 0x9A, 0x78, 0x11,
        0x46, 0x0A, 0x83, 0x54, 0x08, 0xF1, 0x24, 0x76, 0x3D, 0x40, 0x15, 0x99, 0x44, 0x00, 0xFC,
        0xA3, 0x35, 0xD3, 0xC7, 0x03, 0x57, 0x86, 0x63, 0x0F, 0xBE, 0x3B, 0xA5, 0x20, 0xAD, 0x00,
        0xCC, 0x9B, 0xF4, 0xD7, 0x87, 0x43, 0xE0, 0x93, 0xE1, 0xF3, 0x58, 0x67, 0x79, 0xCC, 0x5C,
        0x61, 0x40, 0xA5, 0x90, 0xA4, 0xC5, 0xDC, 0x5D, 0xF8, 0x36,
    ];

    #[test]
    fn stereo_slice_headers_parse_with_layer_context() {
        use crate::pps::PicParameterSet;
        use crate::slice::{SliceLayerContext, SliceSegmentHeader, SliceType};
        use crate::sps::SeqParameterSet;
        let vps = stereo_vps();
        let sps0 = SeqParameterSet::parse_layered(
            &strip_emulation_prevention(STEREO_SPS0_NAL)[2..],
            0,
            Some(&vps),
        )
        .unwrap();
        let sps1 = SeqParameterSet::parse_layered(
            &strip_emulation_prevention(STEREO_SPS1_NAL)[2..],
            1,
            Some(&vps),
        )
        .unwrap();
        let pps0 = PicParameterSet::parse(
            &strip_emulation_prevention(&[0x44, 0x01, 0xC0, 0x2C, 0xBC, 0x14, 0xC9])[2..],
        )
        .unwrap();
        let pps1 =
            PicParameterSet::parse(&strip_emulation_prevention(STEREO_PPS1_NAL)[2..]).unwrap();
        let ctx0 = SliceLayerContext::from_vps(&vps, 0, 0);
        let ctx1 = SliceLayerContext::from_vps(&vps, 1, 0);
        assert_eq!(ctx0.num_direct_ref_layers(), 0);
        assert_eq!(ctx1.direct_ref_layer_ids, vec![0]);
        assert_eq!(ctx1.ref_layer_pic_idc, vec![0]);
        assert!(ctx1.default_ref_layers_active_flag);
        assert!(ctx1.vps_poc_lsb_aligned_flag);

        let h = |head: &[u8],
                 t: u8,
                 sps: &SeqParameterSet,
                 pps: &PicParameterSet,
                 ctx: &SliceLayerContext| {
            let rbsp = strip_emulation_prevention(head);
            SliceSegmentHeader::parse_layered(&rbsp[2..], t, sps, pps, Some(ctx))
                .expect("slice header")
        };
        let idr0 = h(STEREO_IDR0_HEAD, 19, &sps0, &pps0, &ctx0);
        let idr1 = h(STEREO_IDR1_HEAD, 20, &sps1, &pps1, &ctx1);
        let tr0 = h(STEREO_TRAIL0_HEAD, 1, &sps0, &pps0, &ctx0);
        let tr1 = h(STEREO_TRAIL1_HEAD, 1, &sps1, &pps1, &ctx1);
        // Base layer: an I IDR without POC LSB, then a P TRAIL_R at POC 2.
        assert_eq!(idr0.slice_type, Some(SliceType::I));
        assert_eq!(idr0.slice_pic_order_cnt_lsb, None);
        assert_eq!(idr0.num_active_ref_layer_pics, 0);
        assert_eq!(tr0.slice_type, Some(SliceType::P));
        assert_eq!(tr0.slice_pic_order_cnt_lsb, Some(2));
        // Second view: the IDR_N_LP is a P slice predicted from the base
        // view (F.7.3.6.1 carries its slice_pic_order_cnt_lsb since
        // poc_lsb_not_present_flag == 0), NumActiveRefLayerPics == 1
        // with RefPicLayerId[ 0 ] == 0 through default_ref_layers_
        // active_flag (no inter-layer syntax in the header).
        assert_eq!(idr1.slice_type, Some(SliceType::P));
        assert_eq!(idr1.slice_pic_order_cnt_lsb, Some(0));
        assert_eq!(idr1.inter_layer_pred_enabled_flag, None);
        assert_eq!(idr1.num_active_ref_layer_pics, 1);
        assert_eq!(idr1.ref_pic_layer_id, vec![0]);
        assert_eq!(idr1.num_ref_idx_l0_active_minus1, Some(2));
        assert_eq!(idr1.poc_reset_idc, 0);
        assert!(!idr1.poc_msb_cycle_val_present_flag);
        // The TRAIL_R of the second view is a B slice: one temporal and
        // one inter-layer reference.
        assert_eq!(tr1.slice_type, Some(SliceType::B));
        assert_eq!(tr1.slice_pic_order_cnt_lsb, Some(2));
        assert_eq!(tr1.num_active_ref_layer_pics, 1);
        assert_eq!(tr1.ref_pic_layer_id, vec![0]);
        assert_eq!(
            (
                tr1.num_ref_idx_l0_active_minus1,
                tr1.num_ref_idx_l1_active_minus1
            ),
            (Some(0), Some(0))
        );
        assert!(!tr1.slice_temporal_mvp_enabled_flag);
        // Robustness over the layer-1 headers: truncation / bit flips
        // return, never panic.
        for head in [STEREO_IDR1_HEAD, STEREO_TRAIL1_HEAD] {
            let rbsp = strip_emulation_prevention(head);
            let body = &rbsp[2..];
            for cut in 0..body.len().min(64) {
                let _ =
                    SliceSegmentHeader::parse_layered(&body[..cut], 20, &sps1, &pps1, Some(&ctx1));
            }
            for bit in 0..64 * 8 {
                let mut m = body.to_vec();
                m[bit / 8] ^= 0x80 >> (bit % 8);
                let _ = SliceSegmentHeader::parse_layered(&m, 1, &sps1, &pps1, Some(&ctx1));
            }
        }
    }

    #[test]
    fn single_layer_model_default_is_one_ols() {
        let m = LayerModel::default();
        assert_eq!(m.layer_idx(0), Some(0));
        assert_eq!(m.num_direct_ref_layers(0), 0);
        assert_eq!(m.num_output_layer_sets(), 1);
        assert_eq!(m.ols_output_layer_ids(0), vec![0]);
    }

    #[test]
    fn truncated_stereo_vps_is_rejected_not_panicking() {
        let rbsp = strip_emulation_prevention(STEREO_VPS_NAL);
        let body = &rbsp[2..];
        for cut in 0..body.len() {
            let _ = HevcVps::parse(&body[..cut]);
        }
        // Flipping every bit must never panic either.
        for bit in 0..body.len() * 8 {
            let mut m = body.to_vec();
            m[bit / 8] ^= 0x80 >> (bit % 8);
            let _ = HevcVps::parse(&m);
        }
    }

    #[test]
    fn ceil_log2_matches_spec_widths() {
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(64), 6);
        assert_eq!(ceil_log2(65), 7);
    }
}
