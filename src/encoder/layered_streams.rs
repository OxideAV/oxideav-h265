//! Self-built Annex H (SHVC) conformance stream — the encoder-side
//! generator that pins the multi-layer decode path where no black-box
//! oracle exists on this machine (no third-party tool here writes
//! spatial-scalable HEVC).
//!
//! `build_shvc_2x_stream` writes a two-layer bitstream:
//!
//! * a VPS whose `vps_extension( )` (F.7.3.2.1.1) declares two layers of
//!   the spatial-scalability dimension (Table F.1 index 2), layer 1
//!   directly dependent on layer 0 with `direct_dependency_type == 2`
//!   (sample + motion prediction), two `rep_format( )` structures
//!   (32x32 and 64x64), one layer set holding both layers and its
//!   output layer set outputting both, `default_ref_layers_active_flag`
//!   and `max_one_active_ref_layer_flag` set;
//! * the base layer: the crate's lossless PCM IDR access unit
//!   (`nuh_layer_id` 0, 32x32);
//! * the enhancement layer (`nuh_layer_id` 1, 64x64): its own SPS
//!   (F.7.3.2.2.1 ordinary form, `sps_ext_or_max_sub_layers_minus1 ==
//!   0`) and PPS, and one IDR_N_LP picture coded as a P slice whose
//!   every CTB is a skip CU — `MaxNumMergeCand == 1` makes the merge
//!   candidate the zero motion vector on `RefPicList0[ 0 ]`, which the
//!   F.8.3.4 construction fills with the H.8.1.3 inter-layer reference
//!   picture: the H.8.1.4.2-resampled base picture. With deblocking
//!   and SAO off, the decoded enhancement picture IS that resampled
//!   picture, so the test compares it against an independent
//!   `resample_picture` of the decoded base layer.

use crate::cabac::init_type;
use crate::ctx_init::SliceContexts;
use crate::encoder::bitwriter::BitWriter;
use crate::encoder::cabac::CabacEncoder;
use crate::encoder::nal::{annexb, nal_unit};
use crate::encoder::pcm::{encode_idr_pcm_au_opts, write_ptl_cfg, PcmAuOptions};
use crate::nal::collect_nal_units;

/// CTB and minimum CB size of both layers (one CU per CTB, no
/// `split_cu_flag`).
const CTB: usize = 16;
const LEVEL_IDC: u8 = 30;
const SLICE_QP: i32 = 26;

/// Planar 4:2:0 8-bit source of one layer.
pub(crate) type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

fn source_planes(w: usize, h: usize) -> Planes {
    let y: Vec<u8> = (0..w * h)
        .map(|i| {
            let (x, yy) = (i % w, i / w);
            ((x * 7 + yy * 3) % 200 + 20) as u8
        })
        .collect();
    let cb: Vec<u8> = (0..w * h / 4).map(|i| (60 + (i % 90)) as u8).collect();
    let cr: Vec<u8> = (0..w * h / 4).map(|i| (200 - (i % 70)) as u8).collect();
    (y, cb, cr)
}

/// The VPS with its Annex F extension for two spatial layers.
fn write_vps_two_layers(base_w: u16, base_h: u16, enh_w: u16, enh_h: u16) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(0, 4); // vps_video_parameter_set_id
    w.put_bit(1); // vps_base_layer_internal_flag
    w.put_bit(1); // vps_base_layer_available_flag
    w.put_bits(1, 6); // vps_max_layers_minus1
    w.put_bits(0, 3); // vps_max_sub_layers_minus1
    w.put_bit(1); // vps_temporal_id_nesting_flag
    w.put_bits(0xFFFF, 16); // vps_reserved_0xffff_16bits
    write_ptl_cfg(&mut w, LEVEL_IDC, false); // profile_tier_level( 1, 0 )
    w.put_bit(1); // vps_sub_layer_ordering_info_present_flag
    w.ue(1); // vps_max_dec_pic_buffering_minus1[0]
    w.ue(0); // vps_max_num_reorder_pics[0]
    w.ue(0); // vps_max_latency_increase_plus1[0]
    w.put_bits(1, 6); // vps_max_layer_id
    w.ue(1); // vps_num_layer_sets_minus1
    w.put_bit(1); // layer_id_included_flag[1][0]
    w.put_bit(1); // layer_id_included_flag[1][1]
    w.put_bit(0); // vps_timing_info_present_flag
    w.put_bit(1); // vps_extension_flag
    while w.bit_len() % 8 != 0 {
        w.put_bit(1); // vps_extension_alignment_bit_equal_to_one
    }
    // ---- vps_extension( ) (F.7.3.2.1.1) ----
    w.put_bits(u32::from(LEVEL_IDC), 8); // profile_tier_level( 0, 0 ): general_level_idc
    w.put_bit(0); // splitting_flag
    for i in 0..16 {
        w.put_bit(u8::from(i == 2)); // scalability_mask_flag: spatial
    }
    w.put_bits(0, 3); // dimension_id_len_minus1[0]
    w.put_bit(0); // vps_nuh_layer_id_present_flag
    w.put_bits(1, 1); // dimension_id[1][0] = 1 (DependencyId)
    w.put_bits(0, 4); // view_id_len
    w.put_bit(1); // direct_dependency_flag[1][0]
    w.put_bit(0); // vps_sub_layers_max_minus1_present_flag
    w.put_bit(0); // max_tid_ref_present_flag
    w.put_bit(1); // default_ref_layers_active_flag
    w.ue(1); // vps_num_profile_tier_level_minus1 (no further PTL signalled)
    w.ue(0); // num_add_olss
    w.put_bits(0, 2); // default_output_layer_idc: all layers output
    w.put_bits(1, 1); // profile_tier_level_idx[1][0]
    w.put_bits(1, 1); // profile_tier_level_idx[1][1]
    w.ue(1); // vps_num_rep_formats_minus1
    for (rw, rh) in [(base_w, base_h), (enh_w, enh_h)] {
        w.put_bits(u32::from(rw), 16); // pic_width_vps_in_luma_samples
        w.put_bits(u32::from(rh), 16); // pic_height_vps_in_luma_samples
        w.put_bit(1); // chroma_and_bit_depth_vps_present_flag
        w.put_bits(1, 2); // chroma_format_vps_idc 4:2:0
        w.put_bits(0, 4); // bit_depth_vps_luma_minus8
        w.put_bits(0, 4); // bit_depth_vps_chroma_minus8
        w.put_bit(0); // conformance_window_vps_flag
    }
    w.put_bit(0); // rep_format_idx_present_flag
    w.put_bit(1); // max_one_active_ref_layer_flag
    w.put_bit(0); // vps_poc_lsb_aligned_flag
                  // poc_lsb_not_present_flag[1]: absent (layer 1 has a reference layer)
                  // dpb_size( ): OLS 1
    w.put_bit(0); // sub_layer_flag_info_present_flag[1]
    w.ue(0); // max_vps_dec_pic_buffering_minus1[1][0][0]
    w.ue(0); // max_vps_dec_pic_buffering_minus1[1][1][0]
    w.ue(0); // max_vps_num_reorder_pics[1][0]
    w.ue(0); // max_vps_latency_increase_plus1[1][0]
    w.ue(0); // direct_dep_type_len_minus2
    w.put_bit(1); // direct_dependency_all_layers_flag
    w.put_bits(2, 2); // direct_dependency_all_layers_type: sample + motion
    w.ue(0); // vps_non_vui_extension_length
    w.put_bit(0); // vps_vui_present_flag
    w.put_bit(0); // vps_extension2_flag
    w.rbsp_trailing_bits();
    w.finish()
}

/// The enhancement layer's SPS (ordinary F.7.3.2.2.1 form carried in
/// a `nuh_layer_id == 1` NAL unit).
fn write_sps_layer1(width: usize, height: usize) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(0, 4); // sps_video_parameter_set_id
    w.put_bits(0, 3); // sps_ext_or_max_sub_layers_minus1 (ordinary form)
    w.put_bit(1); // sps_temporal_id_nesting_flag
    write_ptl_cfg(&mut w, LEVEL_IDC, false);
    w.ue(1); // sps_seq_parameter_set_id
    w.ue(1); // chroma_format_idc
    w.ue(width as u32); // pic_width_in_luma_samples
    w.ue(height as u32); // pic_height_in_luma_samples
    w.put_bit(0); // conformance_window_flag
    w.ue(0); // bit_depth_luma_minus8
    w.ue(0); // bit_depth_chroma_minus8
    w.ue(4); // log2_max_pic_order_cnt_lsb_minus4 (8 bits)
    w.put_bit(1); // sps_sub_layer_ordering_info_present_flag
    w.ue(1); // sps_max_dec_pic_buffering_minus1
    w.ue(0); // sps_max_num_reorder_pics
    w.ue(0); // sps_max_latency_increase_plus1
    w.ue(1); // log2_min_luma_coding_block_size_minus3 (16)
    w.ue(0); // log2_diff_max_min_luma_coding_block_size (CTB 16)
    w.ue(0); // log2_min_luma_transform_block_size_minus2 (4)
    w.ue(2); // log2_diff_max_min_luma_transform_block_size (16)
    w.ue(0); // max_transform_hierarchy_depth_inter
    w.ue(0); // max_transform_hierarchy_depth_intra
    w.put_bit(0); // scaling_list_enabled_flag
    w.put_bit(0); // amp_enabled_flag
    w.put_bit(0); // sample_adaptive_offset_enabled_flag
    w.put_bit(0); // pcm_enabled_flag
    w.ue(0); // num_short_term_ref_pic_sets
    w.put_bit(0); // long_term_ref_pics_present_flag
    w.put_bit(0); // sps_temporal_mvp_enabled_flag
    w.put_bit(0); // strong_intra_smoothing_enabled_flag
    w.put_bit(0); // vui_parameters_present_flag
    w.put_bit(0); // sps_extension_present_flag
    w.rbsp_trailing_bits();
    w.finish()
}

/// The enhancement layer's PPS: deblocking disabled, nothing else.
fn write_pps_layer1() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.ue(1); // pps_pic_parameter_set_id
    w.ue(1); // pps_seq_parameter_set_id
    w.put_bit(0); // dependent_slice_segments_enabled_flag
    w.put_bit(0); // output_flag_present_flag
    w.put_bits(0, 3); // num_extra_slice_header_bits
    w.put_bit(0); // sign_data_hiding_enabled_flag
    w.put_bit(0); // cabac_init_present_flag
    w.ue(0); // num_ref_idx_l0_default_active_minus1
    w.ue(0); // num_ref_idx_l1_default_active_minus1
    w.se(0); // init_qp_minus26
    w.put_bit(0); // constrained_intra_pred_flag
    w.put_bit(0); // transform_skip_enabled_flag
    w.put_bit(0); // cu_qp_delta_enabled_flag
    w.se(0); // pps_cb_qp_offset
    w.se(0); // pps_cr_qp_offset
    w.put_bit(0); // pps_slice_chroma_qp_offsets_present_flag
    w.put_bit(0); // weighted_pred_flag
    w.put_bit(0); // weighted_bipred_flag
    w.put_bit(0); // transquant_bypass_enabled_flag
    w.put_bit(0); // tiles_enabled_flag
    w.put_bit(0); // entropy_coding_sync_enabled_flag
    w.put_bit(0); // pps_loop_filter_across_slices_enabled_flag
    w.put_bit(1); // deblocking_filter_control_present_flag
    w.put_bit(0); // deblocking_filter_override_enabled_flag
    w.put_bit(1); // pps_deblocking_filter_disabled_flag
    w.put_bit(0); // pps_scaling_list_data_present_flag
    w.put_bit(0); // lists_modification_present_flag
    w.ue(0); // log2_parallel_merge_level_minus2
    w.put_bit(0); // slice_segment_header_extension_present_flag
    w.put_bit(0); // pps_extension_present_flag
    w.rbsp_trailing_bits();
    w.finish()
}

/// The enhancement picture: an IDR_N_LP P slice (F.7.3.6.1 with
/// `nuh_layer_id == 1`: `slice_pic_order_cnt_lsb` present, no
/// inter-layer syntax under `default_ref_layers_active_flag`) whose
/// every CTB is a skip CU.
fn encode_skip_p_slice_layer1(width: usize, height: usize) -> Vec<u8> {
    let ctbs_x = width / CTB;
    let ctbs_y = height / CTB;
    let mut w = BitWriter::new();
    // ---- slice_segment_header( ) ----
    w.put_bit(1); // first_slice_segment_in_pic_flag
    w.put_bit(0); // no_output_of_prior_pics_flag (IRAP)
    w.ue(1); // slice_pic_parameter_set_id
    w.ue(1); // slice_type = P
    w.put_bits(0, 8); // slice_pic_order_cnt_lsb (nuh_layer_id > 0)
                      // IDR: no reference picture set block.
    w.put_bit(0); // num_ref_idx_active_override_flag
    w.ue(4); // five_minus_max_num_merge_cand (MaxNumMergeCand = 1)
    w.se(0); // slice_qp_delta
    w.rbsp_trailing_bits(); // byte_alignment()
                            // ---- slice_segment_data( ) ----
    let mut cabac = CabacEncoder::new();
    let mut ctxs = SliceContexts::init(init_type(1, false), SLICE_QP);
    for ctb in 0..ctbs_x * ctbs_y {
        let col = ctb % ctbs_x;
        let row = ctb / ctbs_x;
        // cu_skip_flag = 1 with the §9.3.4.2.2 neighbour ctxInc (every
        // decoded neighbour is a skip CU).
        let ctx_inc = usize::from(col > 0) + usize::from(row > 0);
        cabac.encode_decision(&mut w, &mut ctxs.cu_skip_flag[ctx_inc], 1);
        cabac.encode_terminate(&mut w, u8::from(ctb == ctbs_x * ctbs_y - 1));
    }
    w.align_zero();
    w.finish()
}

/// Build the two-layer SHVC stream: `(annex_b_bytes, base_planes)` —
/// the base layer 32x32, the enhancement layer 64x64 (2:1 spatial
/// scalability, no reference layer offsets).
pub(crate) fn build_shvc_2x_stream() -> (Vec<u8>, Planes) {
    let (bw, bh) = (32usize, 32usize);
    let (ew, eh) = (64usize, 64usize);
    let (y, cb, cr) = source_planes(bw, bh);
    let base_au = encode_idr_pcm_au_opts(&y, &cb, &cr, bw, bh, PcmAuOptions::default())
        .expect("PCM base layer");
    // Keep the base layer's SPS / PPS / slice; the VPS is replaced by
    // the two-layer one.
    let base_units = collect_nal_units(&base_au).expect("base NAL walk");
    let mut units: Vec<Vec<u8>> = vec![nal_unit(
        32,
        0,
        0,
        &write_vps_two_layers(bw as u16, bh as u16, ew as u16, eh as u16),
    )];
    for u in base_units.iter().filter(|u| u.header.nal_unit_type != 32) {
        units.push(nal_unit(
            u.header.nal_unit_type,
            0,
            u.header.temporal_id,
            &u.rbsp,
        ));
    }
    // The enhancement layer's parameter sets precede its picture in
    // the same access unit.
    units.push(nal_unit(33, 1, 0, &write_sps_layer1(ew, eh)));
    units.push(nal_unit(34, 1, 0, &write_pps_layer1()));
    units.push(nal_unit(20, 1, 0, &encode_skip_p_slice_layer1(ew, eh)));
    (annexb(&units), (y, cb, cr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ilref::{resample_picture, IlRefGeometry, LayerFormat};
    use crate::sequence::{decode_annexb_sequence, LayerTarget, SequenceDecoder};

    #[test]
    fn shvc_2x_stream_decodes_enhancement_as_resampled_base() {
        let (stream, (y, cb, cr)) = build_shvc_2x_stream();
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixture_bytes/r462-shvc-2x.hevc"
        );
        if std::env::var_os("H265_REGEN_FIXTURES").is_some() {
            std::fs::write(path, &stream).expect("write fixture");
        }
        let pinned: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixture_bytes/r462-shvc-2x.hevc"
        ));
        assert_eq!(stream, pinned, "builder must reproduce the pinned bytes");
        let frames = decode_annexb_sequence(&stream).expect("decode");
        assert_eq!(frames.len(), 2, "one picture per layer");
        let base = &frames[0];
        let enh = &frames[1];
        assert_eq!((base.layer_id, enh.layer_id), (0, 1));
        assert_eq!((base.au_index, enh.au_index), (0, 0));
        assert!(base.output && enh.output);
        // The base layer is lossless PCM.
        let mut expected = y;
        expected.extend(cb);
        expected.extend(cr);
        assert_eq!(base.picture.to_planar_u8().unwrap(), expected);
        // The enhancement picture is the H.8.1.4.2 resampling of it.
        let geom = IlRefGeometry::derive(
            LayerFormat::of(&enh.picture),
            LayerFormat::of(&base.picture),
            None,
        )
        .unwrap();
        assert!(!geom.equal_picture_size_and_offset());
        let resampled = resample_picture(&geom, &base.picture);
        assert_eq!(
            (enh.picture.width_luma(), enh.picture.height_luma()),
            (64, 64)
        );
        assert_eq!(enh.picture, resampled, "skip picture == resampled base");
        // Base-layer-only decode drops the enhancement layer.
        let mut dec = SequenceDecoder::new();
        dec.set_layer_target(LayerTarget::Layer(0));
        dec.push_annexb(&stream).unwrap();
        let only = dec.finish().unwrap();
        assert_eq!(only.len(), 1);
        assert_eq!(only[0].layer_id, 0);
    }
}
