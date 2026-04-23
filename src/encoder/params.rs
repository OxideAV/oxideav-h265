//! VPS / SPS / PPS emission for the MVP encoder.
//!
//! All parameter sets are fixed to the profile envelope needed by the
//! MVP:
//!
//! * Main profile, level 3.0 (bounds: 720×576 @ 30 fps), Tier Main.
//! * 8-bit 4:2:0 luma + chroma.
//! * CTU size = 64, min CU size = 8 (log2=3), transform range 4..16.
//! * `pcm_enabled_flag = 1`, PCM sample bit-depth = 8, PCM CU size 8..64
//!   so a single 64×64 CTU can be emitted as PCM.
//! * No tiles, no wavefront, no SAO, no deblock.

use crate::encoder::bit_writer::{write_rbsp_trailing_bits, BitWriter};
use crate::nal::{NalHeader, NalUnitType};
use crate::encoder::nal_writer::build_annex_b_nal;

/// High-level config used by the emitters. Fixed-profile MVP: most fields
/// are derived from width/height.
#[derive(Clone, Copy, Debug)]
pub struct EncoderConfig {
    pub width: u32,
    pub height: u32,
}

impl EncoderConfig {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

// -----------------------------------------------------------------------
// profile_tier_level()  §7.3.3
// -----------------------------------------------------------------------

fn write_profile_tier_level(bw: &mut BitWriter, max_sub_layers_minus1: u32) {
    // general_profile_space (u2) = 0, general_tier_flag (u1) = 0 (Main),
    // general_profile_idc (u5) = 1 (Main).
    bw.write_bits(0, 2);
    bw.write_u1(0);
    bw.write_bits(1, 5);
    // general_profile_compatibility_flag[32] — only bit 1 is set (Main).
    let compat: u32 = 1 << (31 - 1);
    bw.write_bits(compat, 32);
    // progressive / interlaced / non-packed / frame-only constraint flags (all 0).
    bw.write_u1(1); // progressive_source_flag
    bw.write_u1(0); // interlaced_source_flag
    bw.write_u1(0); // non_packed_constraint_flag
    bw.write_u1(1); // frame_only_constraint_flag
    // general_reserved_zero_43bits.
    bw.write_zero_bits(43);
    // general_inbld_flag = 0.
    bw.write_u1(0);
    // general_level_idc (u8) = 30 (level 1.0 × 30 = 3.0).
    bw.write_bits(90, 8); // level 3.0 = 30 × 3 = 90
    // No sub-layers: sub_layer_profile_present_flag / sub_layer_level_present_flag
    // per sub-layer; if max_sub_layers_minus1 == 0 we still emit the reserved
    // 2-bit padding once for the top level — only if max_sub_layers_minus1 > 0.
    if max_sub_layers_minus1 > 0 {
        for _ in 0..max_sub_layers_minus1 {
            bw.write_u1(0); // sub_layer_profile_present_flag
            bw.write_u1(0); // sub_layer_level_present_flag
        }
        for _ in max_sub_layers_minus1..8 {
            bw.write_bits(0, 2); // reserved_zero_2bits
        }
    }
}

// -----------------------------------------------------------------------
// video_parameter_set_rbsp()  §7.3.2.1
// -----------------------------------------------------------------------

pub fn build_vps_rbsp() -> Vec<u8> {
    let mut bw = BitWriter::new();
    // vps_video_parameter_set_id (u4)
    bw.write_bits(0, 4);
    // vps_base_layer_internal_flag (u1), vps_base_layer_available_flag (u1)
    bw.write_u1(1);
    bw.write_u1(1);
    // vps_max_layers_minus1 (u6)
    bw.write_bits(0, 6);
    // vps_max_sub_layers_minus1 (u3)
    bw.write_bits(0, 3);
    // vps_temporal_id_nesting_flag (u1)
    bw.write_u1(1);
    // vps_reserved_0xffff_16bits
    bw.write_bits(0xFFFF, 16);
    // profile_tier_level(1, vps_max_sub_layers_minus1)
    write_profile_tier_level(&mut bw, 0);
    // vps_sub_layer_ordering_info_present_flag
    bw.write_u1(0);
    // For i = max_sub_layers_minus1..max_sub_layers_minus1 inclusive when
    // present_flag == 0 — single entry.
    bw.write_ue(1); // vps_max_dec_pic_buffering_minus1[0] = 1
    bw.write_ue(0); // vps_max_num_reorder_pics[0] = 0
    bw.write_ue(0); // vps_max_latency_increase_plus1[0] = 0
    // vps_max_layer_id (u6) = 0
    bw.write_bits(0, 6);
    // vps_num_layer_sets_minus1
    bw.write_ue(0);
    // vps_timing_info_present_flag = 0
    bw.write_u1(0);
    // vps_extension_flag = 0
    bw.write_u1(0);
    write_rbsp_trailing_bits(&mut bw);
    bw.finish()
}

// -----------------------------------------------------------------------
// seq_parameter_set_rbsp()  §7.3.2.2
// -----------------------------------------------------------------------

pub fn build_sps_rbsp(cfg: &EncoderConfig) -> Vec<u8> {
    let mut bw = BitWriter::new();
    // sps_video_parameter_set_id (u4)
    bw.write_bits(0, 4);
    // sps_max_sub_layers_minus1 (u3)
    bw.write_bits(0, 3);
    // sps_temporal_id_nesting_flag (u1)
    bw.write_u1(1);
    // profile_tier_level(1, sps_max_sub_layers_minus1)
    write_profile_tier_level(&mut bw, 0);
    // sps_seq_parameter_set_id
    bw.write_ue(0);
    // chroma_format_idc = 1 (4:2:0)
    bw.write_ue(1);
    // pic_width_in_luma_samples, pic_height_in_luma_samples (ue(v))
    bw.write_ue(cfg.width);
    bw.write_ue(cfg.height);
    // conformance_window_flag = 0
    bw.write_u1(0);
    // bit_depth_luma_minus8, bit_depth_chroma_minus8
    bw.write_ue(0);
    bw.write_ue(0);
    // log2_max_pic_order_cnt_lsb_minus4 (ue(v)) — pick 4 → MaxPicOrderCntLsb=256
    bw.write_ue(4);
    // sps_sub_layer_ordering_info_present_flag = 0
    bw.write_u1(0);
    bw.write_ue(1); // max_dec_pic_buffering_minus1[0]
    bw.write_ue(0); // max_num_reorder_pics[0]
    bw.write_ue(0); // max_latency_increase_plus1[0]
    // log2_min_luma_coding_block_size_minus3 = 0 → min CU = 8
    bw.write_ue(0);
    // log2_diff_max_min_luma_coding_block_size = 3 → max CU = 64
    bw.write_ue(3);
    // log2_min_luma_transform_block_size_minus2 = 0 → min TB = 4
    bw.write_ue(0);
    // log2_diff_max_min_luma_transform_block_size = 3 → max TB = 32
    bw.write_ue(3);
    // max_transform_hierarchy_depth_inter = 0
    bw.write_ue(0);
    // max_transform_hierarchy_depth_intra = 0
    bw.write_ue(0);
    // scaling_list_enabled_flag = 0
    bw.write_u1(0);
    // amp_enabled_flag = 0
    bw.write_u1(0);
    // sample_adaptive_offset_enabled_flag = 0
    bw.write_u1(0);
    // pcm_enabled_flag = 1 (we use PCM CUs)
    bw.write_u1(1);
    // pcm_sample_bit_depth_luma_minus1 = 7  (u4)
    bw.write_bits(7, 4);
    // pcm_sample_bit_depth_chroma_minus1 = 7 (u4)
    bw.write_bits(7, 4);
    // log2_min_pcm_luma_coding_block_size_minus3 = 0 → 8
    bw.write_ue(0);
    // log2_diff_max_min_pcm_luma_coding_block_size = 3 → 64
    bw.write_ue(3);
    // pcm_loop_filter_disabled_flag = 1
    bw.write_u1(1);
    // num_short_term_ref_pic_sets = 0
    bw.write_ue(0);
    // long_term_ref_pics_present_flag = 0
    bw.write_u1(0);
    // sps_temporal_mvp_enabled_flag = 0
    bw.write_u1(0);
    // strong_intra_smoothing_enabled_flag = 0
    bw.write_u1(0);
    // vui_parameters_present_flag = 0
    bw.write_u1(0);
    // sps_extension_present_flag = 0
    bw.write_u1(0);
    write_rbsp_trailing_bits(&mut bw);
    bw.finish()
}

// -----------------------------------------------------------------------
// pic_parameter_set_rbsp()  §7.3.2.3
// -----------------------------------------------------------------------

pub fn build_pps_rbsp() -> Vec<u8> {
    let mut bw = BitWriter::new();
    // pps_pic_parameter_set_id
    bw.write_ue(0);
    // pps_seq_parameter_set_id
    bw.write_ue(0);
    // dependent_slice_segments_enabled_flag
    bw.write_u1(0);
    // output_flag_present_flag
    bw.write_u1(0);
    // num_extra_slice_header_bits (u3)
    bw.write_bits(0, 3);
    // sign_data_hiding_enabled_flag
    bw.write_u1(0);
    // cabac_init_present_flag
    bw.write_u1(0);
    // num_ref_idx_l0_default_active_minus1
    bw.write_ue(0);
    // num_ref_idx_l1_default_active_minus1
    bw.write_ue(0);
    // init_qp_minus26 (se(v)) = QP26 - 26 = 0 → QP=26
    bw.write_se(0);
    // constrained_intra_pred_flag
    bw.write_u1(0);
    // transform_skip_enabled_flag
    bw.write_u1(0);
    // cu_qp_delta_enabled_flag
    bw.write_u1(0);
    // pps_cb_qp_offset, pps_cr_qp_offset
    bw.write_se(0);
    bw.write_se(0);
    // pps_slice_chroma_qp_offsets_present_flag
    bw.write_u1(0);
    // weighted_pred_flag / weighted_bipred_flag
    bw.write_u1(0);
    bw.write_u1(0);
    // transquant_bypass_enabled_flag
    bw.write_u1(0);
    // tiles_enabled_flag
    bw.write_u1(0);
    // entropy_coding_sync_enabled_flag
    bw.write_u1(0);
    // pps_loop_filter_across_slices_enabled_flag
    bw.write_u1(1);
    // deblocking_filter_control_present_flag
    bw.write_u1(0);
    // pps_scaling_list_data_present_flag
    bw.write_u1(0);
    // lists_modification_present_flag
    bw.write_u1(0);
    // log2_parallel_merge_level_minus2
    bw.write_ue(0);
    // slice_segment_header_extension_present_flag
    bw.write_u1(0);
    // pps_extension_present_flag
    bw.write_u1(0);
    write_rbsp_trailing_bits(&mut bw);
    bw.finish()
}

/// Build Annex B bytes for a VPS NAL.
pub fn build_vps_nal() -> Vec<u8> {
    let rbsp = build_vps_rbsp();
    build_annex_b_nal(NalHeader::for_type(NalUnitType::Vps), &rbsp)
}

/// Build Annex B bytes for a SPS NAL.
pub fn build_sps_nal(cfg: &EncoderConfig) -> Vec<u8> {
    let rbsp = build_sps_rbsp(cfg);
    build_annex_b_nal(NalHeader::for_type(NalUnitType::Sps), &rbsp)
}

/// Build Annex B bytes for a PPS NAL.
pub fn build_pps_nal() -> Vec<u8> {
    let rbsp = build_pps_rbsp();
    build_annex_b_nal(NalHeader::for_type(NalUnitType::Pps), &rbsp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pps::parse_pps;
    use crate::sps::parse_sps;
    use crate::vps::parse_vps;

    #[test]
    fn vps_roundtrips_through_decoder() {
        let rbsp = build_vps_rbsp();
        let vps = parse_vps(&rbsp).expect("parse vps");
        assert_eq!(vps.vps_video_parameter_set_id, 0);
        assert_eq!(vps.vps_max_sub_layers_minus1, 0);
    }

    #[test]
    fn sps_roundtrips_through_decoder() {
        let rbsp = build_sps_rbsp(&EncoderConfig::new(64, 64));
        let sps = parse_sps(&rbsp).expect("parse sps");
        assert_eq!(sps.pic_width_in_luma_samples, 64);
        assert_eq!(sps.pic_height_in_luma_samples, 64);
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.bit_depth_luma_minus8, 0);
        assert_eq!(sps.bit_depth_chroma_minus8, 0);
        assert!(sps.pcm_enabled_flag);
        assert_eq!(sps.pcm_sample_bit_depth_luma, 8);
        assert_eq!(sps.pcm_sample_bit_depth_chroma, 8);
        assert_eq!(sps.log2_min_pcm_luma_coding_block_size, 3);
        assert_eq!(sps.log2_max_pcm_luma_coding_block_size, 6);
    }

    #[test]
    fn pps_roundtrips_through_decoder() {
        let rbsp = build_pps_rbsp();
        let pps = parse_pps(&rbsp).expect("parse pps");
        assert_eq!(pps.pps_pic_parameter_set_id, 0);
        assert_eq!(pps.pps_seq_parameter_set_id, 0);
        assert!(!pps.tiles_enabled_flag);
    }
}
