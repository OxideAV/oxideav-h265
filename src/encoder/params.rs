//! VPS / SPS / PPS emission for the lossy intra encoder.
//!
//! All parameter sets are fixed to the profile envelope needed by the
//! encoder:
//!
//! * Main profile, level 3.0 (bounds: 720×576 @ 30 fps), Tier Main.
//! * 8-bit 4:2:0 luma + chroma.
//! * CTU size = 16, **min CU size = 8** (log2=3). The encoder still keeps
//!   every CB at 16×16 — it emits `split_cu_flag = 0` at the CTB root —
//!   but lifting `MinCbLog2SizeY` to 3 unblocks AMP for the B-slice
//!   writer (round 24): `amp_enabled_flag` only takes effect when
//!   `log2CbSize > MinCbLog2SizeY` per §7.4.9.5.
//! * Transform range 4..16. `max_transform_hierarchy_depth_intra = 0` so
//!   the 16×16 CU carries one 16×16 luma TB + one 8×8 chroma TB pair.
//! * `amp_enabled_flag = 1` — round 24 (B-slice writer scores 2NxnU /
//!   2NxnD / nLx2N / nRx2N against 2Nx2N and picks the best by Lagrangian
//!   SAD).
//! * `pcm_enabled_flag = 0`, `scaling_list_enabled_flag = 0`,
//!   `sample_adaptive_offset_enabled_flag = 0`.
//! * No tiles, no wavefront, deblock enabled via PPS defaults.

use crate::encoder::bit_writer::{write_rbsp_trailing_bits, BitWriter};
use crate::encoder::nal_writer::build_annex_b_nal;
use crate::nal::{NalHeader, NalUnitType};

/// High-level config used by the emitters. Fixed-profile MVP: most fields
/// are derived from width/height.
///
/// `bit_depth` toggles between Main (8) and Main 10 (10). The 8-bit path
/// is the default and stays byte-for-byte compatible with the round-1..24
/// emissions; the 10-bit path is engaged via [`EncoderConfig::new_main10`]
/// and only used by the I-slice writer for now (see crate README).
#[derive(Clone, Copy, Debug)]
pub struct EncoderConfig {
    pub width: u32,
    pub height: u32,
    /// Component bit depth for both luma and chroma. Encoder supports
    /// 8 (Main profile) and 10 (Main 10 profile).
    pub bit_depth: u32,
}

impl EncoderConfig {
    /// 8-bit Main-profile encoder config.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            bit_depth: 8,
        }
    }

    /// 10-bit Main 10 profile encoder config.
    pub fn new_main10(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            bit_depth: 10,
        }
    }
}

// -----------------------------------------------------------------------
// profile_tier_level()  §7.3.3
// -----------------------------------------------------------------------

fn write_profile_tier_level(bw: &mut BitWriter, max_sub_layers_minus1: u32, bit_depth: u32) {
    // general_profile_space (u2) = 0, general_tier_flag (u1) = 0 (Main).
    // general_profile_idc (u5) = 1 (Main) for 8-bit, 2 (Main 10) for 10-bit.
    bw.write_bits(0, 2);
    bw.write_u1(0);
    let profile_idc: u32 = if bit_depth >= 10 { 2 } else { 1 };
    bw.write_bits(profile_idc, 5);
    // general_profile_compatibility_flag[32]: bit at position
    // `profile_idc` is set; for Main 10 we also set the Main bit per
    // §A.3.5 ("a Main 10 decoder must also accept Main streams"), so
    // both bit 1 and bit 2 are set.
    let compat: u32 = if bit_depth >= 10 {
        (1u32 << (31 - 1)) | (1u32 << (31 - 2))
    } else {
        1u32 << (31 - 1)
    };
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
    build_vps_rbsp_with_bit_depth(8)
}

/// Build the VPS RBSP with an explicit `bit_depth` driving the
/// profile_tier_level emission. `bit_depth = 8` matches the original
/// 8-bit Main emission byte-for-byte; `bit_depth = 10` switches the PTL
/// to Main 10 (profile_idc = 2 + the Main+Main10 compatibility bits).
pub fn build_vps_rbsp_with_bit_depth(bit_depth: u32) -> Vec<u8> {
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
    write_profile_tier_level(&mut bw, 0, bit_depth);
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
    write_profile_tier_level(&mut bw, 0, cfg.bit_depth);
    // sps_seq_parameter_set_id
    bw.write_ue(0);
    // chroma_format_idc = 1 (4:2:0)
    bw.write_ue(1);
    // pic_width_in_luma_samples, pic_height_in_luma_samples (ue(v))
    bw.write_ue(cfg.width);
    bw.write_ue(cfg.height);
    // conformance_window_flag = 0
    bw.write_u1(0);
    // bit_depth_luma_minus8, bit_depth_chroma_minus8 — 0 for Main (8-bit)
    // and 2 for Main 10 (10-bit). The decoder honours both and emits
    // either Yuv420P or Yuv420P10Le accordingly.
    let depth_minus8 = cfg.bit_depth.saturating_sub(8);
    bw.write_ue(depth_minus8);
    bw.write_ue(depth_minus8);
    // log2_max_pic_order_cnt_lsb_minus4 (ue(v)) — pick 4 → MaxPicOrderCntLsb=256
    bw.write_ue(4);
    // sps_sub_layer_ordering_info_present_flag = 0
    bw.write_u1(0);
    bw.write_ue(1); // max_dec_pic_buffering_minus1[0]
    bw.write_ue(0); // max_num_reorder_pics[0]
    bw.write_ue(0); // max_latency_increase_plus1[0]
                    // log2_min_luma_coding_block_size_minus3 = 0 → min CU = 8.
                    // Round 24: lifts MinCbLog2SizeY to 3 so 16×16 CBs sit at
                    // log2CbSize > MinCbLog2SizeY, which is the §7.4.9.5
                    // gate for AMP. We still emit one 16×16 CB per 16×16
                    // CTB (split_cu_flag = 0 at CTB root).
    bw.write_ue(0);
    // log2_diff_max_min_luma_coding_block_size = 1 → max CU = 16 (CTU=16)
    bw.write_ue(1);
    // log2_min_luma_transform_block_size_minus2 = 0 → min TB = 4
    // (Chroma 8×8 TB requires TB = 8 which is min_luma_tb for intra; spec
    // fixes chroma TB = log2_luma_tb - 1 at 4:2:0.)
    bw.write_ue(0);
    // log2_diff_max_min_luma_transform_block_size = 2 → max TB = 16
    bw.write_ue(2);
    // max_transform_hierarchy_depth_inter = 0
    bw.write_ue(0);
    // max_transform_hierarchy_depth_intra = 0
    bw.write_ue(0);
    // scaling_list_enabled_flag = 0
    bw.write_u1(0);
    // amp_enabled_flag = 1 (round 24 — B-slice AMP encode)
    bw.write_u1(1);
    // sample_adaptive_offset_enabled_flag = 0
    bw.write_u1(0);
    // pcm_enabled_flag = 0
    bw.write_u1(0);
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
    bw.write_u1(1);
    // deblocking_filter_override_enabled_flag
    bw.write_u1(0);
    // pps_deblocking_filter_disabled_flag — DISABLE deblocking entirely.
    // Encoder's local reconstruction doesn't run deblock, so matching this
    // on the decode side prevents a subtle drift in neighbour samples.
    bw.write_u1(1);
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

/// Build Annex B bytes for a VPS NAL (8-bit Main).
pub fn build_vps_nal() -> Vec<u8> {
    build_vps_nal_with_bit_depth(8)
}

/// Build Annex B bytes for a VPS NAL with the given component bit depth
/// (8 or 10). At 10 the embedded `profile_tier_level()` switches to
/// Main 10 (profile_idc = 2).
pub fn build_vps_nal_with_bit_depth(bit_depth: u32) -> Vec<u8> {
    let rbsp = build_vps_rbsp_with_bit_depth(bit_depth);
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
        assert!(!sps.pcm_enabled_flag);
        // 16×16 CTU: minCB=8 (lifted in round 24 to unblock AMP),
        // maxCB=16.
        assert_eq!(sps.log2_min_luma_coding_block_size_minus3, 0);
        assert_eq!(sps.log2_diff_max_min_luma_coding_block_size, 1);
        // amp_enabled_flag flipped on in round 24 so the B-slice writer
        // can emit asymmetric partitions.
        assert!(sps.amp_enabled_flag);
    }

    #[test]
    fn pps_roundtrips_through_decoder() {
        let rbsp = build_pps_rbsp();
        let pps = parse_pps(&rbsp).expect("parse pps");
        assert_eq!(pps.pps_pic_parameter_set_id, 0);
        assert_eq!(pps.pps_seq_parameter_set_id, 0);
        assert!(!pps.tiles_enabled_flag);
    }

    #[test]
    fn sps_main10_carries_bit_depth_two() {
        // Round 25: Main 10 SPS must declare 10-bit luma + chroma.
        let rbsp = build_sps_rbsp(&EncoderConfig::new_main10(64, 64));
        let sps = parse_sps(&rbsp).expect("parse main10 sps");
        assert_eq!(sps.bit_depth_luma_minus8, 2);
        assert_eq!(sps.bit_depth_chroma_minus8, 2);
        assert_eq!(sps.bit_depth_y(), 10);
        assert_eq!(sps.bit_depth_c(), 10);
        assert_eq!(sps.profile_tier_level.general_profile_idc, 2);
    }

    #[test]
    fn vps_main10_profile_idc_is_two() {
        // VPS must echo the Main 10 profile so a probe that only inspects
        // the VPS (e.g. extradata sniffers) sees Main 10 immediately.
        let rbsp = build_vps_rbsp_with_bit_depth(10);
        let vps = parse_vps(&rbsp).expect("parse main10 vps");
        assert_eq!(vps.vps_video_parameter_set_id, 0);
        assert_eq!(
            vps.profile_tier_level.general_profile_idc, 2,
            "Main 10 VPS PTL should carry profile_idc = 2"
        );
    }
}
