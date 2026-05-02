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
/// `bit_depth` toggles between Main (8), Main 10 (10), and Main 12 (12).
/// The 8-bit path is the default and stays byte-for-byte compatible with
/// the round-1..24 emissions; the 10-bit path is engaged via
/// [`EncoderConfig::new_main10`] (round 25) and the 12-bit path via
/// [`EncoderConfig::new_main12`] (round 26). Both high-bit-depth paths
/// are I-slice-only for now (see crate README).
///
/// `chroma_format_idc` selects 4:2:0 (1) or 4:4:4 (3). 4:2:2 is not
/// emitted by this encoder. 4:4:4 + 8-bit (Main 4:4:4 — round-30) is
/// engaged via [`EncoderConfig::new_main444_8`]; it lives in the RExt
/// profile (`profile_idc = 4`) per §A.3.4.
#[derive(Clone, Copy, Debug)]
pub struct EncoderConfig {
    pub width: u32,
    pub height: u32,
    /// Component bit depth for both luma and chroma. Encoder supports
    /// 8 (Main profile), 10 (Main 10 profile), and 12 (Main 12 / RExt
    /// profile).
    pub bit_depth: u32,
    /// Chroma format per §6.2 Table 6-1. 1 = 4:2:0, 3 = 4:4:4. The 8-bit
    /// 4:2:0 path is the round-1..24 default; the 8-bit 4:4:4 path
    /// (round 30) lifts SubWidthC = SubHeightC = 1 so chroma blocks live
    /// at full luma resolution.
    pub chroma_format_idc: u32,
}

impl EncoderConfig {
    /// 8-bit Main-profile encoder config (4:2:0).
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            bit_depth: 8,
            chroma_format_idc: 1,
        }
    }

    /// 10-bit Main 10 profile encoder config (4:2:0).
    pub fn new_main10(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            bit_depth: 10,
            chroma_format_idc: 1,
        }
    }

    /// 12-bit Main 12 (RExt) profile encoder config (4:2:0). Round 26:
    /// I-slice only (mirrors the round-25 Main 10 IDR-only restriction).
    pub fn new_main12(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            bit_depth: 12,
            chroma_format_idc: 1,
        }
    }

    /// 8-bit Main 4:4:4 (RExt) profile encoder config. Round 30:
    /// I-slice only — mirrors the Main 10 / Main 12 IDR-only restriction.
    /// SubWidthC = SubHeightC = 1, so chroma planes match luma resolution.
    pub fn new_main444_8(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            bit_depth: 8,
            chroma_format_idc: 3,
        }
    }

    /// 10-bit Main 4:4:4 10 (RExt) profile encoder config. Round 31:
    /// combines the 4:4:4 chroma topology of `new_main444_8` with the
    /// 10-bit precision of `new_main10`. I-slice only.
    /// SubWidthC = SubHeightC = 1 (chroma at full luma resolution) and
    /// `bit_depth = 10` so QpBdOffset = 12 applies on both luma and chroma.
    pub fn new_main444_10(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            bit_depth: 10,
            chroma_format_idc: 3,
        }
    }
}

// -----------------------------------------------------------------------
// profile_tier_level()  §7.3.3
// -----------------------------------------------------------------------

fn write_profile_tier_level(
    bw: &mut BitWriter,
    max_sub_layers_minus1: u32,
    bit_depth: u32,
    chroma_format_idc: u32,
) {
    // general_profile_space (u2) = 0, general_tier_flag (u1) = 0 (Main).
    // general_profile_idc (u5):
    //   * 1 (Main)               for 8-bit 4:2:0
    //   * 2 (Main 10)            for 10-bit 4:2:0
    //   * 4 (Format Range Ext.)  for 12-bit (Main 12) and any 4:4:4
    //     (Main 4:4:4) per §A.3.7 / §A.3.4
    bw.write_bits(0, 2);
    bw.write_u1(0);
    let profile_idc: u32 = if chroma_format_idc == 3 || bit_depth >= 12 {
        // Main 4:4:4 and Main 12 both live in the Format Range Extensions
        // profile (§A.3.4 / §A.3.7); the constraint flags below
        // disambiguate them.
        4
    } else if bit_depth >= 10 {
        2
    } else {
        1
    };
    bw.write_bits(profile_idc, 5);
    // general_profile_compatibility_flag[32]: per §A.3.5 a higher-bit-depth
    // decoder must accept lower-bit-depth streams, so we layer the compat
    // bits cumulatively:
    //   * 8-bit 4:2:0:  Main only             → bit 1
    //   * 10-bit 4:2:0: Main + Main 10        → bits 1, 2
    //   * 12-bit 4:2:0: Main + Main 10 + RExt → bits 1, 2, 4   (RExt sniffer
    //     probes bit 4; backward compat keeps Main / Main 10 sniffers
    //     happy on the same bitstream)
    //   * 4:4:4       : RExt only             → bit 4 (Main / Main 10 do
    //     not cover 4:4:4 chroma format, so cumulative-bit-set is invalid
    //     here per §A.3.5).
    let compat: u32 = if chroma_format_idc == 3 {
        1u32 << (31 - 4)
    } else if bit_depth >= 12 {
        (1u32 << (31 - 1)) | (1u32 << (31 - 2)) | (1u32 << (31 - 4))
    } else if bit_depth >= 10 {
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
                    // general_reserved_zero_43bits, *or* — for profile_idc = 4
                    // (Format Range Extensions) — the 9 RExt constraint flags
                    // (§A.3.5 / §A.3.7) followed by 34 reserved zeros. The
                    // constraint flag values below are the §A.3.7 "Main 12"
                    // signature: 12-bit max, 4:2:0/4:2:2 max, non-intra,
                    // non-still, lower-bit-rate. ffmpeg's
                    // `decode_profile_tier_level` reads these bits to map a
                    // RExt stream to the specific Main 12 profile; without
                    // them the stream is reported as "Rext" with no concrete
                    // profile and several decoders refuse it.
    if profile_idc == 4 {
        if chroma_format_idc == 3 {
            // §A.3.4 Table A.2: at chroma_format_idc == 3 the row to write
            // depends on bit_depth.
            //   * 8-bit (Main 4:4:4)     → max_12bit=1, max_10bit=1, max_8bit=1
            //   * 10-bit (Main 4:4:4 10) → max_12bit=1, max_10bit=1, max_8bit=0
            //   * 12-bit (Main 4:4:4 12) → max_12bit=1, max_10bit=0, max_8bit=0
            // The other six flags are identical across all three rows
            // (max_422 = 0, max_420 = 0, max_mono = 0, intra = 0,
            // one_picture = 0, lower_bit_rate = 1) because Main 4:4:4 in
            // any bit depth supports inter, every chroma format up to
            // 4:4:4, and is not still-picture-only. The encoder currently
            // emits 8-bit and 10-bit 4:4:4; 12-bit 4:4:4 is out of scope
            // (covered by the 8-bit row's `_ =>` fallback should a future
            // round wire it up).
            let max_12bit = 1u32;
            let (max_10bit, max_8bit) = match bit_depth {
                8 => (1u32, 1u32),
                10 => (1u32, 0u32),
                _ => (0u32, 0u32),
            };
            bw.write_u1(max_12bit); // max_12bit_constraint_flag
            bw.write_u1(max_10bit); // max_10bit_constraint_flag
            bw.write_u1(max_8bit); // max_8bit_constraint_flag
            bw.write_u1(0); // max_422chroma_constraint_flag (allow 4:4:4)
            bw.write_u1(0); // max_420chroma_constraint_flag (allow 4:4:4)
            bw.write_u1(0); // max_monochrome_constraint_flag
            bw.write_u1(0); // intra_constraint_flag (Main 4:4:4 supports inter)
            bw.write_u1(0); // one_picture_only_constraint_flag
            bw.write_u1(1); // lower_bit_rate_constraint_flag
            bw.write_zero_bits(34); // reserved
        } else {
            bw.write_u1(1); // max_12bit_constraint_flag
            bw.write_u1(0); // max_10bit_constraint_flag
            bw.write_u1(0); // max_8bit_constraint_flag
            bw.write_u1(1); // max_422chroma_constraint_flag
            bw.write_u1(1); // max_420chroma_constraint_flag
            bw.write_u1(0); // max_monochrome_constraint_flag
            bw.write_u1(0); // intra_constraint_flag (Main 12 supports inter)
            bw.write_u1(0); // one_picture_only_constraint_flag
            bw.write_u1(1); // lower_bit_rate_constraint_flag
            bw.write_zero_bits(34); // reserved
        }
    } else {
        bw.write_zero_bits(43);
    }
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
    build_vps_rbsp_with_profile(8, 1)
}

/// Build the VPS RBSP with an explicit `bit_depth` driving the
/// profile_tier_level emission. Defaults to 4:2:0 chroma format.
/// `bit_depth = 8` matches the original 8-bit Main emission byte-for-byte;
/// `bit_depth = 10` switches the PTL to Main 10 (profile_idc = 2 + the
/// Main+Main10 compatibility bits).
pub fn build_vps_rbsp_with_bit_depth(bit_depth: u32) -> Vec<u8> {
    build_vps_rbsp_with_profile(bit_depth, 1)
}

/// Build the VPS RBSP with both `bit_depth` and `chroma_format_idc`
/// driving the profile_tier_level emission. `chroma_format_idc = 3`
/// engages the Main 4:4:4 signature (round 30).
pub fn build_vps_rbsp_with_profile(bit_depth: u32, chroma_format_idc: u32) -> Vec<u8> {
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
    write_profile_tier_level(&mut bw, 0, bit_depth, chroma_format_idc);
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
    write_profile_tier_level(&mut bw, 0, cfg.bit_depth, cfg.chroma_format_idc);
    // sps_seq_parameter_set_id
    bw.write_ue(0);
    // chroma_format_idc — 1 (4:2:0) by default; 3 (4:4:4) for the Main
    // 4:4:4 path. When chroma_format_idc == 3, the SPS bitstream includes
    // a single separate_colour_plane_flag bit (we emit 0 — no separate
    // colour planes; chroma is interleaved at full luma resolution per
    // §6.2 Table 6-1).
    bw.write_ue(cfg.chroma_format_idc);
    if cfg.chroma_format_idc == 3 {
        bw.write_u1(0); // separate_colour_plane_flag = 0
    }
    // pic_width_in_luma_samples, pic_height_in_luma_samples (ue(v))
    bw.write_ue(cfg.width);
    bw.write_ue(cfg.height);
    // conformance_window_flag = 0
    bw.write_u1(0);
    // bit_depth_luma_minus8, bit_depth_chroma_minus8 — 0 for Main (8-bit),
    // 2 for Main 10 (10-bit), 4 for Main 12 (12-bit). The decoder honours
    // all three and emits either Yuv420P, Yuv420P10Le, or Yuv420P12Le
    // accordingly.
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
/// Main 10 (profile_idc = 2). 4:2:0 is assumed; for 4:4:4 see
/// [`build_vps_nal_with_profile`].
pub fn build_vps_nal_with_bit_depth(bit_depth: u32) -> Vec<u8> {
    let rbsp = build_vps_rbsp_with_bit_depth(bit_depth);
    build_annex_b_nal(NalHeader::for_type(NalUnitType::Vps), &rbsp)
}

/// Build Annex B bytes for a VPS NAL with both `bit_depth` and
/// `chroma_format_idc` driving the embedded `profile_tier_level()`.
/// `chroma_format_idc = 3` engages the Main 4:4:4 RExt signature
/// (round 30).
pub fn build_vps_nal_with_profile(bit_depth: u32, chroma_format_idc: u32) -> Vec<u8> {
    let rbsp = build_vps_rbsp_with_profile(bit_depth, chroma_format_idc);
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

    #[test]
    fn sps_main12_carries_bit_depth_four() {
        // Round 26: Main 12 SPS must declare 12-bit luma + chroma and the
        // RExt profile (general_profile_idc = 4 per §A.3.7).
        let rbsp = build_sps_rbsp(&EncoderConfig::new_main12(64, 64));
        let sps = parse_sps(&rbsp).expect("parse main12 sps");
        assert_eq!(sps.bit_depth_luma_minus8, 4);
        assert_eq!(sps.bit_depth_chroma_minus8, 4);
        assert_eq!(sps.bit_depth_y(), 12);
        assert_eq!(sps.bit_depth_c(), 12);
        assert_eq!(sps.profile_tier_level.general_profile_idc, 4);
        // The Main / Main 10 / RExt compatibility bits are all set per
        // §A.3.5 — a Main 12 stream must be readable by a sniffer that
        // probes any of those bits.
        let compat = sps.profile_tier_level.general_profile_compatibility_flag;
        assert!(compat & (1u32 << (31 - 1)) != 0, "Main compat bit");
        assert!(compat & (1u32 << (31 - 2)) != 0, "Main 10 compat bit");
        assert!(compat & (1u32 << (31 - 4)) != 0, "RExt compat bit");
    }

    #[test]
    fn vps_main12_profile_idc_is_four() {
        // VPS PTL must agree with the SPS PTL on profile_idc — ffmpeg
        // cross-checks the two and refuses streams where they differ.
        let rbsp = build_vps_rbsp_with_bit_depth(12);
        let vps = parse_vps(&rbsp).expect("parse main12 vps");
        assert_eq!(vps.vps_video_parameter_set_id, 0);
        assert_eq!(
            vps.profile_tier_level.general_profile_idc, 4,
            "Main 12 VPS PTL should carry profile_idc = 4 (RExt)"
        );
    }
}
