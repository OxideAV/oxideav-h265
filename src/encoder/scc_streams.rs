//! Self-built SCC conformance streams (test-only): §8.6.8 adaptive
//! colour transform and current-picture referencing (intra block
//! copy).
//!
//! No black-box encoder binary in this workspace's toolchain emits the
//! screen-content-coding extensions, so these streams are assembled by
//! this crate's own header writers and CABAC encoder (the round-413
//! RDPCM / round-416 CCP methodology):
//!
//! * an **ACT stream** — 4:4:4 lossless (transquant-bypass) all-intra
//!   IDR whose CUs alternate `tu_residual_act_flag` 1 / 0; act-1 CUs
//!   carry forward-lifted (YCgCo-R-shaped, eqs 8-336..8-339 inverse)
//!   residual triples so the §8.6.8.2 inverse restores the source
//!   exactly, act-0 CUs are plain controls;
//! * an **IBC stream** — a 4:2:0 IDR picture whose slices are
//!   `slice_type == P` with `pps_curr_pic_ref_enabled_flag == 1`
//!   (`RefPicList0 == [ currPic ]`): CTB 0 is an intra CU, every
//!   other CTB an AMVP coding unit whose motion vector points at the
//!   already-decoded CTB to the left (or above, for the first
//!   column), with a bypass residual correcting the copy to the
//!   source — the eqs 8-98..8-101 integer path and the §8.5.3.1
//!   current-picture prediction reconstruct losslessly.
//!
//! The checked-in copies under `tests/fixture_bytes/` are pinned by
//! the tests here: (a) the builders still produce those exact bytes,
//! (b) this crate's decoder reconstructs the procedural source
//! planes. Black-box validation status is recorded in
//! `tests/fixture_bytes/r416-generation-notes.md`.

use crate::binarization::{
    cbf_cb_ctx_inc, cbf_cr_ctx_inc, cbf_luma_ctx_inc, intra_luma_cand_mode_list,
};
use crate::cabac::init_type;
use crate::ctx_init::SliceContexts;
use crate::encoder::bitwriter::BitWriter;
use crate::encoder::cabac::CabacEncoder;
use crate::encoder::ccp_streams::{extract, ip_params_444, source_planes_444, Planes};
use crate::encoder::intra::{gather_refs, zscan_avail};
use crate::encoder::nal::{annexb, nal_unit};
use crate::encoder::residual::encode_residual_coding;
use crate::intra_mode_field::{IntraModeField, Neighbour};
use crate::intra_pred::{intra_predict_with_substitution, Component as PredComponent};
use crate::motion::MotionField;
use crate::pu_mv::{resolve_pu_motion, PartMode, PuGeometry, PuMvContext};
use crate::residual::{residual_coding_scan_idx, ResidualCodingParams};
use crate::slice_data::PredictionUnit;

/// CTB / coding-block log2 size (16x16, `MinCbSizeY == CtbSizeY`).
const CTB_LOG2: u32 = 4;
const CTB: usize = 1 << CTB_LOG2;
/// SliceQpY (immaterial — every CU is transquant-bypassed).
const SLICE_QP: i32 = 26;

/// §7.3.3 — profile tier level for Screen Content Coding extensions
/// (`general_profile_idc == 9`); `chroma444` selects the 4:4:4 vs
/// 4:2:0 constraint-flag pattern. Profile 9 carries
/// `general_max_14bit_constraint_flag` + 33 reserved bits.
fn write_ptl_scc(w: &mut BitWriter, level_idc: u8, chroma444: bool) {
    w.put_bits(0, 2); // general_profile_space
    w.put_bit(0); // general_tier_flag
    w.put_bits(9, 5); // general_profile_idc = 9 (SCC)
    let mut compat: u32 = 0;
    compat |= 1 << (31 - 9); // flag[9] — SCC
    w.put_bits(compat, 32);
    w.put_bit(1); // general_progressive_source_flag
    w.put_bit(0); // general_interlaced_source_flag
    w.put_bit(1); // general_non_packed_constraint_flag
    w.put_bit(1); // general_frame_only_constraint_flag
    w.put_bit(1); // general_max_12bit_constraint_flag
    w.put_bit(1); // general_max_10bit_constraint_flag
    w.put_bit(1); // general_max_8bit_constraint_flag
    w.put_bit(u8::from(!chroma444)); // general_max_422chroma_constraint_flag
    w.put_bit(u8::from(!chroma444)); // general_max_420chroma_constraint_flag
    w.put_bit(0); // general_max_monochrome_constraint_flag
    w.put_bit(0); // general_intra_constraint_flag
    w.put_bit(0); // general_one_picture_only_constraint_flag
    w.put_bit(0); // general_lower_bit_rate_constraint_flag
    w.put_bit(1); // general_max_14bit_constraint_flag
                  // general_reserved_zero_33bits
    w.put_bits(0, 32);
    w.put_bit(0);
    w.put_bit(0); // general_inbld_flag (reserved zero)
    w.put_bits(level_idc as u32, 8); // general_level_idc
}

/// §7.3.2.1 — minimal single-layer VPS carrying the SCC PTL.
fn write_vps_scc(level_idc: u8, chroma444: bool) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(0, 4); // vps_video_parameter_set_id
    w.put_bit(1); // vps_base_layer_internal_flag
    w.put_bit(1); // vps_base_layer_available_flag
    w.put_bits(0, 6); // vps_max_layers_minus1
    w.put_bits(0, 3); // vps_max_sub_layers_minus1
    w.put_bit(1); // vps_temporal_id_nesting_flag
    w.put_bits(0xFFFF, 16); // vps_reserved_0xffff_16bits
    write_ptl_scc(&mut w, level_idc, chroma444);
    w.put_bit(1); // vps_sub_layer_ordering_info_present_flag
    w.ue(1); // vps_max_dec_pic_buffering_minus1[0]
    w.ue(0); // vps_max_num_reorder_pics[0]
    w.ue(0); // vps_max_latency_increase_plus1[0]
    w.put_bits(0, 6); // vps_max_layer_id
    w.ue(0); // vps_num_layer_sets_minus1
    w.put_bit(0); // vps_timing_info_present_flag
    w.put_bit(0); // vps_extension_flag
    w.rbsp_trailing_bits();
    w.finish()
}

/// §7.3.2.2 — SCC SPS (CTB 16, no PCM, SAO off) carrying an
/// `sps_scc_extension()`; `chroma444` selects the chroma format and
/// `curr_pic_ref` sets `sps_curr_pic_ref_enabled_flag`.
fn write_sps_scc(
    width: usize,
    height: usize,
    level_idc: u8,
    chroma444: bool,
    curr_pic_ref: bool,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(0, 4); // sps_video_parameter_set_id
    w.put_bits(0, 3); // sps_max_sub_layers_minus1
    w.put_bit(1); // sps_temporal_id_nesting_flag
    write_ptl_scc(&mut w, level_idc, chroma444);
    w.ue(0); // sps_seq_parameter_set_id
    w.ue(if chroma444 { 3 } else { 1 }); // chroma_format_idc
    if chroma444 {
        w.put_bit(0); // separate_colour_plane_flag
    }
    w.ue(width as u32); // pic_width_in_luma_samples
    w.ue(height as u32); // pic_height_in_luma_samples
    w.put_bit(0); // conformance_window_flag
    w.ue(0); // bit_depth_luma_minus8
    w.ue(0); // bit_depth_chroma_minus8
    w.ue(4); // log2_max_pic_order_cnt_lsb_minus4
    w.put_bit(1); // sps_sub_layer_ordering_info_present_flag
    w.ue(1); // sps_max_dec_pic_buffering_minus1[0]
    w.ue(0); // sps_max_num_reorder_pics[0]
    w.ue(0); // sps_max_latency_increase_plus1[0]
    w.ue(CTB_LOG2 - 3); // log2_min_luma_coding_block_size_minus3 (16)
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
    w.put_bit(1); // sps_extension_present_flag
    w.put_bit(0); // sps_range_extension_flag
    w.put_bit(0); // sps_multilayer_extension_flag
    w.put_bit(0); // sps_3d_extension_flag
    w.put_bit(1); // sps_scc_extension_flag
    w.put_bits(0, 4); // sps_extension_4bits
                      // sps_scc_extension() — §7.3.2.2.3.
    w.put_bit(u8::from(curr_pic_ref)); // sps_curr_pic_ref_enabled_flag
    w.put_bit(0); // palette_mode_enabled_flag
    w.put_bits(0, 2); // motion_vector_resolution_control_idc
    w.put_bit(0); // intra_boundary_filtering_disabled_flag
    w.rbsp_trailing_bits();
    w.finish()
}

/// §7.3.2.3 — bypass PPS carrying a `pps_scc_extension()` with the
/// given `pps_curr_pic_ref_enabled_flag` / ACT flags.
fn write_pps_scc(curr_pic_ref: bool, act: bool, ccp: bool) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.ue(0); // pps_pic_parameter_set_id
    w.ue(0); // pps_seq_parameter_set_id
    w.put_bit(0); // dependent_slice_segments_enabled_flag
    w.put_bit(0); // output_flag_present_flag
    w.put_bits(0, 3); // num_extra_slice_header_bits
    w.put_bit(0); // sign_data_hiding_enabled_flag
    w.put_bit(0); // cabac_init_present_flag
    w.ue(0); // num_ref_idx_l0_default_active_minus1
    w.ue(0); // num_ref_idx_l1_default_active_minus1
    w.se(SLICE_QP - 26); // init_qp_minus26
    w.put_bit(0); // constrained_intra_pred_flag
    w.put_bit(0); // transform_skip_enabled_flag
    w.put_bit(0); // cu_qp_delta_enabled_flag
    w.se(0); // pps_cb_qp_offset
    w.se(0); // pps_cr_qp_offset
    w.put_bit(0); // pps_slice_chroma_qp_offsets_present_flag
    w.put_bit(0); // weighted_pred_flag
    w.put_bit(0); // weighted_bipred_flag
    w.put_bit(1); // transquant_bypass_enabled_flag
    w.put_bit(0); // tiles_enabled_flag
    w.put_bit(0); // entropy_coding_sync_enabled_flag
    w.put_bit(1); // pps_loop_filter_across_slices_enabled_flag
    w.put_bit(1); // deblocking_filter_control_present_flag
    w.put_bit(0); // deblocking_filter_override_enabled_flag
    w.put_bit(1); // pps_deblocking_filter_disabled_flag
    w.put_bit(0); // pps_scaling_list_data_present_flag
    w.put_bit(0); // lists_modification_present_flag
    w.ue(0); // log2_parallel_merge_level_minus2
    w.put_bit(0); // slice_segment_header_extension_present_flag
    w.put_bit(1); // pps_extension_present_flag
    w.put_bit(u8::from(ccp)); // pps_range_extension_flag
    w.put_bit(0); // pps_multilayer_extension_flag
    w.put_bit(0); // pps_3d_extension_flag
    w.put_bit(1); // pps_scc_extension_flag
    w.put_bits(0, 4); // pps_extension_4bits
    if ccp {
        // pps_range_extension() — §7.3.2.3.2 (transform_skip off).
        w.put_bit(1); // cross_component_prediction_enabled_flag
        w.put_bit(0); // chroma_qp_offset_list_enabled_flag
        w.ue(0); // log2_sao_offset_scale_luma
        w.ue(0); // log2_sao_offset_scale_chroma
    }
    // pps_scc_extension() — §7.3.2.3.3.
    w.put_bit(u8::from(curr_pic_ref)); // pps_curr_pic_ref_enabled_flag
    w.put_bit(u8::from(act)); // residual_adaptive_colour_transform_enabled_flag
    if act {
        w.put_bit(0); // pps_slice_act_qp_offsets_present_flag
        w.se(5); // pps_act_y_qp_offset_plus5 (PpsActQpOffsetY = 0)
        w.se(5); // pps_act_cb_qp_offset_plus5 (PpsActQpOffsetCb = 0)
        w.se(3); // pps_act_cr_qp_offset_plus3 (PpsActQpOffsetCr = 0)
    }
    w.put_bit(0); // pps_palette_predictor_initializers_present_flag
    w.rbsp_trailing_bits();
    w.finish()
}

/// The forward lifting matching the §8.6.8.2 lossless inverse: given
/// the wanted post-inverse triple `(c0, c1, c2)` = (rY, rCb, rCr),
/// produce the coded triple.
fn act_forward(c0: i32, c1: i32, c2: i32) -> (i32, i32, i32) {
    let co = c2 - c1;
    let t = c1 + (co >> 1);
    let cg = c0 - t;
    let y = t + (cg >> 1);
    (y, cg, co)
}

/// Encode the luma-mode signalling of one intra 16x16 PART_2Nx2N CU
/// against the §8.4.2 candidate list, recording the mode.
#[allow(clippy::too_many_arguments)]
fn encode_intra_luma_mode(
    w: &mut BitWriter,
    cabac: &mut CabacEncoder,
    ctxs: &mut SliceContexts,
    modes: &mut IntraModeField,
    x0: usize,
    y0: usize,
    mode: u8,
    avail_l: bool,
    avail_a: bool,
) {
    let cand_a = modes.cand_intra_pred_mode(x0, y0, Neighbour::Left, avail_l);
    let cand_b = modes.cand_intra_pred_mode(x0, y0, Neighbour::Above, avail_a);
    let list = intra_luma_cand_mode_list(cand_a, cand_b);
    let sel = list.iter().position(|&m| m == mode);
    cabac.encode_decision(
        w,
        &mut ctxs.prev_intra_luma_pred_flag[0],
        u8::from(sel.is_some()),
    );
    modes.record_intra_pb(x0, y0, CTB, mode, false);
    match sel {
        Some(0) => cabac.encode_bypass(w, 0),
        Some(1) => {
            cabac.encode_bypass(w, 1);
            cabac.encode_bypass(w, 0);
        }
        Some(_) => {
            cabac.encode_bypass(w, 1);
            cabac.encode_bypass(w, 1);
        }
        None => {
            let mut rem = u32::from(mode);
            for &c in &list {
                if u32::from(mode) > u32::from(c) {
                    rem -= 1;
                }
            }
            cabac.encode_bypass_bits(w, rem, 5);
        }
    }
}

// ---------------------------------------------------------------------------
// ACT stream (4:4:4 all-intra)
// ---------------------------------------------------------------------------

/// Per-CTB ACT plan: `(luma_mode, tu_residual_act_flag, cb_scale,
/// cr_scale)` — DM chroma throughout; the act flag alternates so
/// plain CUs interleave as controls, and the cross-component
/// prediction scales cycle so CCP applies BOTH under and without the
/// colour transform (the §8.4.4.1 step-8 before §8.6.8 ordering).
fn act_plan(ctb: usize) -> (u8, bool, i32, i32) {
    const MODES: [u8; 6] = [0, 1, 26, 10, 18, 34];
    const SCALES: [i32; 5] = [0, 1, -2, 4, -8];
    (
        MODES[ctb % MODES.len()],
        ctb % 2 == 0,
        SCALES[ctb % SCALES.len()],
        SCALES[(ctb + 2) % SCALES.len()],
    )
}

/// Encode the all-intra 4:4:4 bypass ACT slice: one 16x16 PART_2Nx2N
/// CU per CTB, DM chroma, `tu_residual_act_flag` per [`act_plan`].
fn encode_act_idr_slice(y: &[u8], cb: &[u8], cr: &[u8], width: usize, height: usize) -> Vec<u8> {
    let ctbs_x = width / CTB;
    let ctbs_y = height / CTB;
    let mut modes = IntraModeField::new(width, height, CTB_LOG2);

    let mut w = BitWriter::new();
    w.put_bit(1); // first_slice_segment_in_pic_flag
    w.put_bit(0); // no_output_of_prior_pics_flag (IRAP NAL)
    w.ue(0); // slice_pic_parameter_set_id
    w.ue(2); // slice_type = I
    w.se(0); // slice_qp_delta
    w.rbsp_trailing_bits(); // byte_alignment()

    let mut cabac = CabacEncoder::new();
    let mut ctxs = SliceContexts::init(init_type(2, false), SLICE_QP);

    for ctb in 0..ctbs_x * ctbs_y {
        let x0 = (ctb % ctbs_x) * CTB;
        let y0 = (ctb / ctbs_x) * CTB;
        let (mode, act, cb_scale, cr_scale) = act_plan(ctb);

        // Component residuals (lossless recon == source).
        let residual = |plane: &[u8], pc: PredComponent| -> Vec<i32> {
            let read = |x: usize, yy: usize| i32::from(plane[yy * width + x]);
            let avail = |nx: i64, ny: i64| zscan_avail(nx, ny, width, height, CTB, ctbs_x, ctb, 0);
            let marked = gather_refs(&read, &avail, x0, y0, CTB);
            let pred = intra_predict_with_substitution(&marked, &ip_params_444(mode, pc))
                .expect("legal prediction");
            let src = extract(plane, width, x0, y0, CTB);
            src.iter().zip(&pred).map(|(s, p)| s - p).collect()
        };
        let r_y = residual(y, PredComponent::Luma);
        let r_cb = residual(cb, PredComponent::Cb);
        let r_cr = residual(cr, PredComponent::Cr);

        // act == 1: code the forward-lifted triples; the decoder
        // applies §8.4.4.1 step-8 CCP BEFORE the §8.6.8 inverse, so
        // the CCP term is subtracted from the post-lifting chroma.
        let (res_l, res_cb, res_cr) = if act {
            let mut cl = Vec::with_capacity(r_y.len());
            let mut ccb = Vec::with_capacity(r_y.len());
            let mut ccr = Vec::with_capacity(r_y.len());
            for k in 0..r_y.len() {
                let (a, b, c) = act_forward(r_y[k], r_cb[k], r_cr[k]);
                cl.push(a);
                ccb.push(b);
                ccr.push(c);
            }
            (cl, ccb, ccr)
        } else {
            (r_y, r_cb, r_cr)
        };
        let cbf_luma_pre = res_l.iter().any(|&v| v != 0);
        // §7.3.8.10: cross_comp_pred needs cbf_luma.
        let (cb_scale, cr_scale) = if cbf_luma_pre {
            (cb_scale, cr_scale)
        } else {
            (0, 0)
        };
        let ccp_sub = |res: &[i32], scale: i32| -> Vec<i32> {
            res.iter()
                .zip(&res_l)
                .map(|(&v, &ry)| v - ((scale * ry) >> 3))
                .collect()
        };
        let res_cb = ccp_sub(&res_cb, cb_scale);
        let res_cr = ccp_sub(&res_cr, cr_scale);

        // ---- coding_unit( ) ----
        cabac.encode_decision(&mut w, &mut ctxs.cu_transquant_bypass_flag[0], 1);
        cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], 1); // PART_2Nx2N
        let avail_l = zscan_avail(x0 as i64 - 1, y0 as i64, width, height, CTB, ctbs_x, ctb, 0);
        let avail_a = zscan_avail(x0 as i64, y0 as i64 - 1, width, height, CTB, ctbs_x, ctb, 0);
        encode_intra_luma_mode(
            &mut w, &mut cabac, &mut ctxs, &mut modes, x0, y0, mode, avail_l, avail_a,
        );
        // intra_chroma_pred_mode = 4 (derived / DM): single ctx bin 0.
        cabac.encode_decision(&mut w, &mut ctxs.intra_chroma_pred_mode[0], 0);

        // ---- transform_tree / transform_unit ----
        let cbf_cb_f = res_cb.iter().any(|&v| v != 0);
        let cbf_cr_f = res_cr.iter().any(|&v| v != 0);
        let cbf_luma = res_l.iter().any(|&v| v != 0);
        cabac.encode_decision(
            &mut w,
            &mut ctxs.cbf_chroma[cbf_cb_ctx_inc(0) as usize],
            u8::from(cbf_cb_f),
        );
        cabac.encode_decision(
            &mut w,
            &mut ctxs.cbf_chroma[cbf_cr_ctx_inc(0) as usize],
            u8::from(cbf_cr_f),
        );
        cabac.encode_decision(
            &mut w,
            &mut ctxs.cbf_luma[cbf_luma_ctx_inc(0) as usize],
            u8::from(cbf_luma),
        );
        // §7.3.8.10: tu_residual_act_flag leads the transform unit
        // (2Nx2N + DM ⇒ the intra gate holds) when any cbf is set.
        if cbf_luma || cbf_cb_f || cbf_cr_f {
            cabac.encode_decision(&mut w, &mut ctxs.tu_residual_act_flag[0], u8::from(act));
        }
        let rc = |is_chroma: bool| ResidualCodingParams {
            log2_trafo_size: 4,
            is_chroma,
            scan_idx: residual_coding_scan_idx(true, 4, u8::from(is_chroma), 3, u32::from(mode)),
            sign_data_hiding_enabled_flag: false,
            sign_hidden_suppressed: false,
            transform_skip_sig_ctx: false,
            persistent_rice_adaptation_enabled_flag: false,
            cabac_bypass_alignment_enabled_flag: false,
            extended_precision_processing_flag: false,
            bit_depth: 8,
            rice_stat_transform_skip: false,
        };
        if cbf_luma {
            encode_residual_coding(&mut w, &mut cabac, &mut ctxs.residual, &rc(false), &res_l)
                .expect("valid luma levels");
        }
        // §7.3.8.10 in-place chroma: cross_comp_pred precedes each
        // chroma residual_coding (gate: enabled && cbf_luma && DM).
        if cbf_luma {
            crate::encoder::ccp_streams::encode_cross_comp_pred(
                &mut w, &mut cabac, &mut ctxs, 0, cb_scale,
            );
        }
        if cbf_cb_f {
            encode_residual_coding(&mut w, &mut cabac, &mut ctxs.residual, &rc(true), &res_cb)
                .expect("valid cb levels");
        }
        if cbf_luma {
            crate::encoder::ccp_streams::encode_cross_comp_pred(
                &mut w, &mut cabac, &mut ctxs, 1, cr_scale,
            );
        }
        if cbf_cr_f {
            encode_residual_coding(&mut w, &mut cabac, &mut ctxs.residual, &rc(true), &res_cr)
                .expect("valid cr levels");
        }

        cabac.encode_terminate(&mut w, u8::from(ctb == ctbs_x * ctbs_y - 1));
    }
    w.align_zero();
    w.finish()
}

/// The ACT conformance stream: VPS + SPS(scc, 4:4:4) + PPS(scc, ACT) +
/// one all-intra bypass IDR picture (64x48).
pub(crate) fn build_act_stream() -> (Vec<u8>, Planes) {
    let (w, h) = (64usize, 48usize);
    let (y, cb, cr) = source_planes_444(w, h);
    let slice = encode_act_idr_slice(&y, &cb, &cr, w, h);
    let units = vec![
        nal_unit(32, 0, 0, &write_vps_scc(30, true)),
        nal_unit(33, 0, 0, &write_sps_scc(w, h, 30, true, false)),
        nal_unit(34, 0, 0, &write_pps_scc(false, true, true)),
        nal_unit(20, 0, 0, &slice), // IDR_N_LP
    ];
    (annexb(&units), (y, cb, cr))
}

// ---------------------------------------------------------------------------
// IBC stream (4:2:0 IDR with P slice referencing the current picture)
// ---------------------------------------------------------------------------

/// Deterministic 4:2:0 source planes for the IBC picture: a repeating
/// texture with per-CTB perturbation (so IBC copies are close but the
/// residual is never all-zero).
fn source_planes_420(w: usize, h: usize) -> Planes {
    let lum = |x: i32, y: i32| -> u8 {
        let g = ((x % 16) * 9 + (y % 16) * 4) % 128;
        let p = (x / 16 + y / 16) * 5 % 31;
        (32 + g + p).clamp(0, 255) as u8
    };
    let chr = |x: i32, y: i32, ph: i32| -> u8 {
        let g = ((x % 8) * 11 + (y % 8) * 6 + ph) % 100;
        let p = (x / 8 + y / 8) * 3 % 17;
        (60 + g + p).clamp(0, 255) as u8
    };
    let mut y_p = Vec::with_capacity(w * h);
    for yy in 0..h as i32 {
        for xx in 0..w as i32 {
            y_p.push(lum(xx, yy));
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    let mut cb_p = Vec::with_capacity(cw * ch);
    let mut cr_p = Vec::with_capacity(cw * ch);
    for yy in 0..ch as i32 {
        for xx in 0..cw as i32 {
            cb_p.push(chr(xx, yy, 0));
            cr_p.push(chr(xx, yy, 47));
        }
    }
    (y_p, cb_p, cr_p)
}

/// Encode the IBC P slice of the IDR picture: CTB 0 intra (DC), every
/// other CTB an AMVP CU referencing the current picture at an
/// integer MV of one CTB left (or one CTB up in column 0), with the
/// bypass residual correcting the copy.
fn encode_ibc_p_slice(y: &[u8], cb: &[u8], cr: &[u8], width: usize, height: usize) -> Vec<u8> {
    let (cw, _ch) = (width / 2, height / 2);
    let ctbs_x = width / CTB;
    let ctbs_y = height / CTB;
    let mut modes = IntraModeField::new(width, height, CTB_LOG2);
    let mut field = MotionField::new(width, height);

    let mut w = BitWriter::new();
    // ---- slice_segment_header( ) ---- (IDR: no POC / RPS fields)
    w.put_bit(1); // first_slice_segment_in_pic_flag
    w.put_bit(0); // no_output_of_prior_pics_flag (IRAP NAL)
    w.ue(0); // slice_pic_parameter_set_id
    w.ue(1); // slice_type = P
    w.put_bit(0); // num_ref_idx_active_override_flag (1 active ref)
    w.ue(4); // five_minus_max_num_merge_cand (MaxNumMergeCand = 1)
             // use_integer_mv_flag absent (motion_vector_resolution_control_idc 0)
    w.se(0); // slice_qp_delta
    w.rbsp_trailing_bits(); // byte_alignment()

    // ---- slice_segment_data( ) ----
    let mut cabac = CabacEncoder::new();
    let mut ctxs = SliceContexts::init(init_type(1, false), SLICE_QP);
    let mut skip_flags = vec![false; ctbs_x * ctbs_y];

    // §8.5.3.2 context mirroring the decoder: RefPicList0 = [currPic]
    // (POC 0, long-term, is_curr_pic).
    let ref_poc = |_l: usize, r: i32| if r == 0 { 0 } else { i32::MIN };
    let ref_long = |_l: usize, r: i32| r == 0;
    let ref_short = |_l: usize, _r: i32| false;
    let col_long = |_p: i32| false;
    let is_curr = |l: usize, r: i32| l == 0 && r == 0;
    let mv_ctx = PuMvContext {
        curr_poc: 0,
        slice_is_b: false,
        ctb_log2_size_y: CTB_LOG2,
        pic_width_luma: width as u32,
        pic_height_luma: height as u32,
        max_num_merge_cand: 1,
        num_ref_idx_l0_active: 1,
        num_ref_idx_l1_active: 0,
        log2_par_mrg_level: 2,
        temporal_mvp_enabled: false,
        collocated_from_l0_flag: true,
        col_poc: 0,
        no_backward_pred: true,
        ref_poc: &ref_poc,
        ref_long_term: &ref_long,
        ref_short_term: &ref_short,
        col_field: None,
        col_ref_long_term: &col_long,
        use_integer_mv: false,
        two_versions_curr_pic: false,
        is_curr_pic: &is_curr,
    };

    for ctb in 0..ctbs_x * ctbs_y {
        let x0 = (ctb % ctbs_x) * CTB;
        let y0 = (ctb / ctbs_x) * CTB;
        let col = ctb % ctbs_x;
        let row = ctb / ctbs_x;

        cabac.encode_decision(&mut w, &mut ctxs.cu_transquant_bypass_flag[0], 1);
        // cu_skip_flag = 0 always (§9.3.4.2.2 neighbour ctxInc).
        let ctx_inc = usize::from(col > 0 && skip_flags[ctb - 1])
            + usize::from(row > 0 && skip_flags[ctb - ctbs_x]);
        cabac.encode_decision(&mut w, &mut ctxs.cu_skip_flag[ctx_inc], 0);
        skip_flags[ctb] = false;

        if ctb == 0 {
            // Intra DC seed CU (no neighbours: prediction = 128).
            let mode = 1u8; // INTRA_DC
            cabac.encode_decision(&mut w, &mut ctxs.pred_mode_flag[0], 1); // MODE_INTRA
            cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], 1); // PART_2Nx2N
            encode_intra_luma_mode(
                &mut w, &mut cabac, &mut ctxs, &mut modes, x0, y0, mode, false, false,
            );
            cabac.encode_decision(&mut w, &mut ctxs.intra_chroma_pred_mode[0], 0); // DM

            let res_from = |plane: &[u8], pw: usize, bx: usize, by: usize, n: usize| -> Vec<i32> {
                let src = extract(plane, pw, bx, by, n);
                src.iter().map(|s| s - 128).collect()
            };
            let res_l = res_from(y, width, x0, y0, CTB);
            let res_cb = res_from(cb, cw, x0 / 2, y0 / 2, 8);
            let res_cr = res_from(cr, cw, x0 / 2, y0 / 2, 8);
            emit_tu_420(
                &mut w, &mut cabac, &mut ctxs, &res_l, &res_cb, &res_cr, true, mode,
            );
            cabac.encode_terminate(&mut w, u8::from(ctb == ctbs_x * ctbs_y - 1));
            continue;
        }

        // IBC AMVP CU: MV one CTB left, or one CTB up for column 0.
        let (dx, dy): (i32, i32) = if col > 0 {
            (-(CTB as i32), 0)
        } else {
            (0, -(CTB as i32))
        };
        let target: [i32; 2] = [dx * 4, dy * 4];

        // Derive both MVP candidates through the decoder's own
        // §8.5.3.2.6 process (mvd = 0 probes), then signal the mvd on
        // the eq. 8-98 integer path: mvd = (target>>2) − (mvp>>2).
        let geom = PuGeometry {
            x_cb: x0,
            y_cb: y0,
            n_cb_s: CTB,
            x_pb: x0,
            y_pb: y0,
            n_pb_w: CTB,
            n_pb_h: CTB,
            part_mode: PartMode::Part2Nx2N,
            part_idx: 0,
        };
        let available = |nx: i32, ny: i32| {
            zscan_avail(
                i64::from(nx),
                i64::from(ny),
                width,
                height,
                CTB,
                ctbs_x,
                ctb,
                0,
            )
        };
        let probe = |mvd: [i32; 2], flag: u8| -> [i32; 2] {
            let pu = PredictionUnit {
                merge_flag: false,
                merge_idx: None,
                inter_pred_idc: Some(crate::binarization::InterPredIdc::PredL0),
                ref_idx_l0: Some(0),
                mvd_l0: Some([mvd_comp(mvd[0]), mvd_comp(mvd[1])]),
                mvp_l0_flag: Some(flag),
                ref_idx_l1: None,
                mvd_l1: None,
                mvp_l1_flag: None,
            };
            resolve_pu_motion(&field, &geom, &pu, &mv_ctx, &available).mv_l0
        };
        let mvp0 = probe([0, 0], 0);
        let mvd = [
            (target[0] >> 2) - (mvp0[0] >> 2),
            (target[1] >> 2) - (mvp0[1] >> 2),
        ];
        debug_assert_eq!(
            probe(mvd, 0),
            target,
            "integer-path mvd must land on target"
        );

        // Residual = source − copied block (lossless).
        let diff = |plane: &[u8], pw: usize, bx: usize, by: usize, n: usize, mx: i32, my: i32| {
            let mut out = Vec::with_capacity(n * n);
            for j in 0..n {
                for i in 0..n {
                    let sx = (bx + i) as i32 + mx;
                    let sy = (by + j) as i32 + my;
                    let refv = i32::from(plane[sy as usize * pw + sx as usize]);
                    out.push(i32::from(plane[(by + j) * pw + bx + i]) - refv);
                }
            }
            out
        };
        let res_l = diff(y, width, x0, y0, CTB, dx, dy);
        let res_cb = diff(cb, cw, x0 / 2, y0 / 2, 8, dx / 2, dy / 2);
        let res_cr = diff(cr, cw, x0 / 2, y0 / 2, 8, dx / 2, dy / 2);
        let any_res = res_l.iter().any(|&v| v != 0)
            || res_cb.iter().any(|&v| v != 0)
            || res_cr.iter().any(|&v| v != 0);

        // Interior CUs whose left neighbour already carries the
        // (−CTB, 0) IBC vector use MERGE (candidate A1, the
        // eqs 8-124/8-125 rounding on the current-picture reference);
        // the rest are AMVP CUs on the eq. 8-98 integer path.
        let use_merge = col >= 2;
        cabac.encode_decision(&mut w, &mut ctxs.pred_mode_flag[0], 0); // MODE_INTER
        cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], 1); // PART_2Nx2N
        if use_merge {
            let merge_probe = resolve_pu_motion(
                &field,
                &geom,
                &PredictionUnit {
                    merge_flag: true,
                    merge_idx: Some(0),
                    inter_pred_idc: None,
                    ref_idx_l0: None,
                    mvd_l0: None,
                    mvp_l0_flag: None,
                    ref_idx_l1: None,
                    mvd_l1: None,
                    mvp_l1_flag: None,
                },
                &mv_ctx,
                &available,
            );
            assert_eq!(merge_probe.mv_l0, target, "A1 merge must copy the IBC MV");
            assert!(any_res, "merge (non-skip) TU needs a coded residual");
            cabac.encode_decision(&mut w, &mut ctxs.merge_flag[0], 1); // merge
                                                                       // merge_idx absent (MaxNumMergeCand == 1); rqt_root_cbf
                                                                       // inferred 1 for a 2Nx2N merge CU.
            emit_tu_420(
                &mut w, &mut cabac, &mut ctxs, &res_l, &res_cb, &res_cr, false, 0,
            );
        } else {
            cabac.encode_decision(&mut w, &mut ctxs.merge_flag[0], 0); // AMVP
                                                                       // ref_idx_l0 absent (one active reference).
            encode_mvd_pair(&mut w, &mut cabac, &mut ctxs, mvd);
            cabac.encode_decision(&mut w, &mut ctxs.mvp_flag[0], 0); // mvp_l0_flag
                                                                     // rqt_root_cbf (non-merge inter CU).
            cabac.encode_decision(&mut w, &mut ctxs.rqt_root_cbf[0], u8::from(any_res));
            if any_res {
                emit_tu_420(
                    &mut w, &mut cabac, &mut ctxs, &res_l, &res_cb, &res_cr, false, 0,
                );
            }
        }

        // Mirror the decoder's motion-field store for later MVPs.
        let motion = probe(mvd, 0);
        let cell = crate::pu_mv::PuMotion {
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_idx_l0: 0,
            ref_idx_l1: -1,
            mv_l0: motion,
            mv_l1: [0, 0],
        }
        .to_cell(0, i32::MIN);
        field.fill_rect(x0, y0, CTB, CTB, cell);

        cabac.encode_terminate(&mut w, u8::from(ctb == ctbs_x * ctbs_y - 1));
    }
    w.align_zero();
    w.finish()
}

/// One §7.3.8.9 `mvd_coding` component wrapper for the probe PUs.
fn mvd_comp(v: i32) -> crate::binarization::MvdComponent {
    crate::binarization::MvdComponent {
        greater0_flag: u8::from(v != 0),
        greater1_flag: None,
        minus2: None,
        sign_flag: None,
        value: v,
    }
}

/// §7.3.8.9 mvd_coding( x0, y0, 0 ) — the two-component encode.
fn encode_mvd_pair(
    w: &mut BitWriter,
    cabac: &mut CabacEncoder,
    ctxs: &mut SliceContexts,
    mvd: [i32; 2],
) {
    // abs_mvd_greater0_flag[ 0 / 1 ] then abs_mvd_greater1_flag pair,
    // then per-component EG1 remainder + sign (§9.3.3.5).
    let gt0 = [mvd[0] != 0, mvd[1] != 0];
    let gt1 = [mvd[0].unsigned_abs() > 1, mvd[1].unsigned_abs() > 1];
    for &g in &gt0 {
        cabac.encode_decision(w, &mut ctxs.abs_mvd_greater0_flag[0], u8::from(g));
    }
    for c in 0..2 {
        if gt0[c] {
            cabac.encode_decision(w, &mut ctxs.abs_mvd_greater1_flag[0], u8::from(gt1[c]));
        }
    }
    for c in 0..2 {
        if gt0[c] {
            let abs = mvd[c].unsigned_abs();
            if gt1[c] {
                // abs_mvd_minus2: EG1 bypass.
                encode_eg1(w, cabac, abs - 2);
            }
            cabac.encode_bypass(w, u8::from(mvd[c] < 0)); // mvd_sign_flag
        }
    }
}

/// EG1 (§9.3.3.3 k = 1) bypass encode.
fn encode_eg1(w: &mut BitWriter, cabac: &mut CabacEncoder, mut value: u32) {
    let mut k = 1u32;
    // Prefix: unary count of exp levels.
    while value >= (1 << k) {
        cabac.encode_bypass(w, 1);
        value -= 1 << k;
        k += 1;
    }
    cabac.encode_bypass(w, 0);
    // Suffix: k fixed bits.
    for bit in (0..k).rev() {
        cabac.encode_bypass(w, ((value >> bit) & 1) as u8);
    }
}

/// Emit one 4:2:0 transform unit (16 luma / two 8x8 chroma) with
/// bypass residuals: cbf flags per §7.3.8.8 order, then the
/// `residual_coding( )` bodies. `intra` selects the scan derivation;
/// `mode` is the intra prediction mode (ignored for inter).
#[allow(clippy::too_many_arguments)]
fn emit_tu_420(
    w: &mut BitWriter,
    cabac: &mut CabacEncoder,
    ctxs: &mut SliceContexts,
    res_l: &[i32],
    res_cb: &[i32],
    res_cr: &[i32],
    intra: bool,
    mode: u8,
) {
    let cbf_cb_f = res_cb.iter().any(|&v| v != 0);
    let cbf_cr_f = res_cr.iter().any(|&v| v != 0);
    let cbf_luma = res_l.iter().any(|&v| v != 0);
    cabac.encode_decision(
        w,
        &mut ctxs.cbf_chroma[cbf_cb_ctx_inc(0) as usize],
        u8::from(cbf_cb_f),
    );
    cabac.encode_decision(
        w,
        &mut ctxs.cbf_chroma[cbf_cr_ctx_inc(0) as usize],
        u8::from(cbf_cr_f),
    );
    // §7.3.8.8: an intra leaf always signals cbf_luma; an inter depth-0
    // leaf with no chroma cbf infers cbf_luma = 1.
    if intra || cbf_cb_f || cbf_cr_f {
        cabac.encode_decision(
            w,
            &mut ctxs.cbf_luma[cbf_luma_ctx_inc(0) as usize],
            u8::from(cbf_luma),
        );
    } else {
        assert!(cbf_luma, "all-zero inter TU must use rqt_root_cbf = 0");
    }
    let rc = |log2: u32, is_chroma: bool| ResidualCodingParams {
        log2_trafo_size: log2,
        is_chroma,
        scan_idx: residual_coding_scan_idx(intra, log2, u8::from(is_chroma), 1, u32::from(mode)),
        sign_data_hiding_enabled_flag: false,
        sign_hidden_suppressed: false,
        transform_skip_sig_ctx: false,
        persistent_rice_adaptation_enabled_flag: false,
        cabac_bypass_alignment_enabled_flag: false,
        extended_precision_processing_flag: false,
        bit_depth: 8,
        rice_stat_transform_skip: false,
    };
    if cbf_luma {
        encode_residual_coding(w, cabac, &mut ctxs.residual, &rc(4, false), res_l)
            .expect("valid luma levels");
    }
    if cbf_cb_f {
        encode_residual_coding(w, cabac, &mut ctxs.residual, &rc(3, true), res_cb)
            .expect("valid cb levels");
    }
    if cbf_cr_f {
        encode_residual_coding(w, cabac, &mut ctxs.residual, &rc(3, true), res_cr)
            .expect("valid cr levels");
    }
}

/// The IBC conformance stream: VPS + SPS(scc, 4:2:0, curr-pic-ref) +
/// PPS(scc, curr-pic-ref) + one IDR picture whose single slice is a P
/// slice referencing only the current picture (64x48).
pub(crate) fn build_ibc_stream() -> (Vec<u8>, Planes) {
    let (w, h) = (64usize, 48usize);
    let (y, cb, cr) = source_planes_420(w, h);
    let slice = encode_ibc_p_slice(&y, &cb, &cr, w, h);
    let units = vec![
        nal_unit(32, 0, 0, &write_vps_scc(30, false)),
        nal_unit(33, 0, 0, &write_sps_scc(w, h, 30, false, true)),
        nal_unit(34, 0, 0, &write_pps_scc(true, false, false)),
        nal_unit(20, 0, 0, &slice), // IDR_N_LP
    ];
    (annexb(&units), (y, cb, cr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::decode_annexb_sequence;

    fn planes_of(frame: &crate::sequence::DecodedFrame) -> Vec<u8> {
        frame.picture.to_planar_u8().expect("8-bit planes")
    }

    /// The checked-in ACT stream bytes are exactly what the builder
    /// produces, and this crate's decoder reconstructs the procedural
    /// source losslessly through the §8.6.8.2 inverse colour
    /// transform (act-1 CUs) with interleaved act-0 controls.
    #[test]
    fn act_stream_decodes_lossless() {
        let (stream, (y, cb, cr)) = build_act_stream();
        let pinned: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixture_bytes/r416-act.hevc"
        ));
        assert_eq!(stream, pinned, "builder must reproduce the validated bytes");
        let frames = decode_annexb_sequence(&stream).expect("decode");
        assert_eq!(frames.len(), 1);
        let mut expected = y;
        expected.extend(cb);
        expected.extend(cr);
        assert_eq!(planes_of(&frames[0]), expected, "lossless ACT decode");
    }

    /// The checked-in IBC stream bytes are exactly what the builder
    /// produces, and this crate's decoder reconstructs the procedural
    /// source losslessly through the current-picture-referencing path
    /// (an IDR whose P slice lists only currPic; per-CTB integer MVs
    /// copying the left / above CTB).
    #[test]
    fn ibc_stream_decodes_lossless() {
        let (stream, (y, cb, cr)) = build_ibc_stream();
        let pinned: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixture_bytes/r416-ibc.hevc"
        ));
        assert_eq!(stream, pinned, "builder must reproduce the validated bytes");
        let frames = decode_annexb_sequence(&stream).expect("decode");
        assert_eq!(frames.len(), 1);
        let mut expected = y;
        expected.extend(cb);
        expected.extend(cr);
        assert_eq!(planes_of(&frames[0]), expected, "lossless IBC decode");
    }
}
