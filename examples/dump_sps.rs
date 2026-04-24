//! Diagnostic: dump SPS/PPS flags from an Annex B HEVC stream.
//!
//! Used to verify which Rext-family SPS flags libx265 actually emits for
//! its "Main 10 as Rext" output, which drives Main 10 decode validation.
//!
//! Usage: `cargo run --example dump_sps -- path/to/clip.hevc`

use oxideav_h265::nal::{extract_rbsp, iter_annex_b, NalUnitType};
use oxideav_h265::pps::parse_pps;
use oxideav_h265::sps::parse_sps;

fn main() {
    let path = std::env::args().nth(1).expect("need path");
    let data = std::fs::read(&path).expect("read");
    for nal in iter_annex_b(&data) {
        let rbsp = extract_rbsp(nal.payload());
        match nal.header.nal_unit_type {
            NalUnitType::Sps => {
                let sps = parse_sps(&rbsp).expect("parse SPS");
                println!(
                    "SPS {}x{} profile_idc={} bit_depth_y={} bit_depth_c={}",
                    sps.pic_width_in_luma_samples,
                    sps.pic_height_in_luma_samples,
                    sps.profile_tier_level.general_profile_idc,
                    sps.bit_depth_y(),
                    sps.bit_depth_c(),
                );
                println!(
                    "  max_tr_hier_depth_inter={} max_tr_hier_depth_intra={}",
                    sps.max_transform_hierarchy_depth_inter,
                    sps.max_transform_hierarchy_depth_intra,
                );
                println!(
                    "  log2_min_luma_tbsize={} log2_diff_max_min_luma_tbsize={}",
                    sps.log2_min_luma_transform_block_size_minus2 + 2,
                    sps.log2_diff_max_min_luma_transform_block_size,
                );
                println!(
                    "  chroma_format_idc={} amp={} sao={} strong_intra_smoothing={} mvp={} scaling_list={}",
                    sps.chroma_format_idc,
                    sps.amp_enabled_flag,
                    sps.sample_adaptive_offset_enabled_flag,
                    sps.strong_intra_smoothing_enabled_flag,
                    sps.sps_temporal_mvp_enabled_flag,
                    sps.scaling_list_enabled_flag,
                );
                println!("  == RExt flags ==");
                println!(
                    "  transform_skip_rotation={} transform_skip_context={} implicit_rdpcm={} explicit_rdpcm={}",
                    sps.transform_skip_rotation_enabled_flag,
                    sps.transform_skip_context_enabled_flag,
                    sps.implicit_rdpcm_enabled_flag,
                    sps.explicit_rdpcm_enabled_flag,
                );
                println!(
                    "  extended_precision={} intra_smoothing_disabled={} high_precision_offsets={} persistent_rice={} cabac_bypass_alignment={}",
                    sps.extended_precision_processing_flag,
                    sps.intra_smoothing_disabled_flag,
                    sps.high_precision_offsets_flag,
                    sps.persistent_rice_adaptation_enabled_flag,
                    sps.cabac_bypass_alignment_enabled_flag,
                );
            }
            NalUnitType::Pps => {
                let pps = parse_pps(&rbsp).expect("parse PPS");
                println!(
                    "PPS transform_skip={} cu_qp_delta_enabled={} sign_data_hiding={} init_qp_minus26={}",
                    pps.transform_skip_enabled_flag,
                    pps.cu_qp_delta_enabled_flag,
                    pps.sign_data_hiding_enabled_flag,
                    pps.init_qp_minus26,
                );
                println!(
                    "  pps_cb_qp_offset={} pps_cr_qp_offset={} tiles_enabled={} entropy_coding_sync={}",
                    pps.pps_cb_qp_offset,
                    pps.pps_cr_qp_offset,
                    pps.tiles_enabled_flag,
                    pps.entropy_coding_sync_enabled_flag,
                );
            }
            _ => {}
        }
    }
}
