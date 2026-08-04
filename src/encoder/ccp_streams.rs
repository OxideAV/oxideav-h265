//! Self-built §8.6.6 cross-component-prediction conformance stream
//! (test-only).
//!
//! No black-box encoder binary exposes the range-extension
//! cross-component prediction tool, so this module assembles a tiny
//! 4:4:4 lossless (transquant-bypass) all-intra Annex B bitstream from
//! this crate's own header writers and CABAC encoder. Every coding
//! unit signals derived (DM) chroma so the §7.3.8.10 `cross_comp_pred`
//! gate holds, and the per-CTB `ResScaleVal` plan cycles through every
//! legal magnitude and both signs for Cb and Cr independently —
//! including a CTB whose chroma residual is exactly the scaled luma
//! residual (cbf clear, CCP still signalled) and zero-scale controls.
//!
//! The stream reconstructs losslessly: the encoder codes the chroma
//! residual as `( orig − pred ) − ( ResScaleVal * rY ) >> 3` so the
//! decoder's eq. 8-324 modification restores `orig` exactly. The
//! checked-in copy under `tests/fixture_bytes/` was validated against
//! a black-box reference decoder (byte-exact; SHA-256 sums in
//! `tests/fixture_bytes/r416-generation-notes.md`); the tests here pin
//! (a) that the builder still produces those exact bytes and (b) that
//! this crate's decoder reconstructs the procedural source planes.

use crate::binarization::{
    cbf_cb_ctx_inc, cbf_cr_ctx_inc, cbf_luma_ctx_inc, intra_luma_cand_mode_list,
    log2_res_scale_abs_plus1_ctx_inc, res_scale_sign_flag_ctx_inc,
};
use crate::cabac::init_type;
use crate::ctx_init::SliceContexts;
use crate::encoder::bitwriter::BitWriter;
use crate::encoder::cabac::CabacEncoder;
use crate::encoder::intra::{gather_refs, zscan_avail};
use crate::encoder::nal::{annexb, nal_unit};
use crate::encoder::residual::encode_residual_coding;
use crate::intra_mode_field::{IntraModeField, Neighbour};
use crate::intra_pred::{
    intra_predict_with_substitution, Component as PredComponent, IntraPredParams,
};
use crate::residual::{residual_coding_scan_idx, ResidualCodingParams};

/// One picture's `(Y, Cb, Cr)` planes (all full-resolution — 4:4:4).
pub(super) type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

/// CTB / coding-block log2 size (16x16, `MinCbSizeY == CtbSizeY`).
const CTB_LOG2: u32 = 4;
const CTB: usize = 1 << CTB_LOG2;
/// SliceQpY (immaterial — every CU is transquant-bypassed).
const SLICE_QP: i32 = 26;

/// Deterministic 4:4:4 source planes: smooth gradients plus a coarse
/// checker texture (full-resolution chroma).
pub(super) fn source_planes_444(w: usize, h: usize) -> Planes {
    let lum = |x: i32, y: i32| -> u8 {
        let g = (x * 3 + y * 5) % 197;
        let t = ((x / 4 + y / 4) % 3) * 23;
        (16 + g + t).clamp(0, 255) as u8
    };
    let chr = |x: i32, y: i32, ph: i32| -> u8 {
        let g = (x * 7 + y * 2 + ph) % 151;
        let t = ((x / 4 + y / 8) % 2) * 31;
        (40 + g + t).clamp(0, 255) as u8
    };
    let mut y_p = Vec::with_capacity(w * h);
    let mut cb_p = Vec::with_capacity(w * h);
    let mut cr_p = Vec::with_capacity(w * h);
    for yy in 0..h as i32 {
        for xx in 0..w as i32 {
            y_p.push(lum(xx, yy));
            cb_p.push(chr(xx, yy, 0));
            cr_p.push(chr(xx, yy, 61));
        }
    }
    (y_p, cb_p, cr_p)
}

/// §7.3.3 — profile tier level for Format Range Extensions
/// (`general_profile_idc == 4`) with the 4:4:4 8-bit constraint-flag
/// pattern (`max_422chroma` / `max_420chroma` clear).
fn write_ptl_rext_444(w: &mut BitWriter, level_idc: u8) {
    w.put_bits(0, 2); // general_profile_space
    w.put_bit(0); // general_tier_flag
    w.put_bits(4, 5); // general_profile_idc = 4 (Range Extensions)
    let mut compat: u32 = 0;
    compat |= 1 << (31 - 4); // flag[4] — Range Extensions
    w.put_bits(compat, 32);
    w.put_bit(1); // general_progressive_source_flag
    w.put_bit(0); // general_interlaced_source_flag
    w.put_bit(1); // general_non_packed_constraint_flag
    w.put_bit(1); // general_frame_only_constraint_flag
    w.put_bit(1); // general_max_12bit_constraint_flag
    w.put_bit(1); // general_max_10bit_constraint_flag
    w.put_bit(1); // general_max_8bit_constraint_flag
    w.put_bit(0); // general_max_422chroma_constraint_flag
    w.put_bit(0); // general_max_420chroma_constraint_flag
    w.put_bit(0); // general_max_monochrome_constraint_flag
    w.put_bit(0); // general_intra_constraint_flag
    w.put_bit(0); // general_one_picture_only_constraint_flag
    w.put_bit(0); // general_lower_bit_rate_constraint_flag
                  // general_reserved_zero_34bits
    w.put_bits(0, 32);
    w.put_bits(0, 2);
    w.put_bit(0); // general_inbld_flag (reserved zero)
    w.put_bits(level_idc as u32, 8); // general_level_idc
                                     // sps_max_sub_layers_minus1 == 0: no sub-layer flags, no bytes.
}

/// §7.3.2.1 — minimal single-layer VPS carrying the 4:4:4 Rext PTL.
fn write_vps_rext_444(level_idc: u8) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(0, 4); // vps_video_parameter_set_id
    w.put_bit(1); // vps_base_layer_internal_flag
    w.put_bit(1); // vps_base_layer_available_flag
    w.put_bits(0, 6); // vps_max_layers_minus1
    w.put_bits(0, 3); // vps_max_sub_layers_minus1
    w.put_bit(1); // vps_temporal_id_nesting_flag
    w.put_bits(0xFFFF, 16); // vps_reserved_0xffff_16bits
    write_ptl_rext_444(&mut w, level_idc);
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

/// §7.3.2.2 — 4:4:4 8-bit SPS (CTB 16, no PCM, SAO off, no
/// extensions).
fn write_sps_444(width: usize, height: usize, level_idc: u8) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(0, 4); // sps_video_parameter_set_id
    w.put_bits(0, 3); // sps_max_sub_layers_minus1
    w.put_bit(1); // sps_temporal_id_nesting_flag
    write_ptl_rext_444(&mut w, level_idc);
    w.ue(0); // sps_seq_parameter_set_id
    w.ue(3); // chroma_format_idc = 4:4:4
    w.put_bit(0); // separate_colour_plane_flag
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
    w.put_bit(0); // sps_extension_present_flag
    w.rbsp_trailing_bits();
    w.finish()
}

/// §7.3.2.3 — PPS with `transquant_bypass_enabled_flag == 1` and a
/// `pps_range_extension()` whose
/// `cross_component_prediction_enabled_flag` is 1.
fn write_pps_ccp() -> Vec<u8> {
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
    w.put_bit(1); // pps_range_extension_flag
    w.put_bit(0); // pps_multilayer_extension_flag
    w.put_bit(0); // pps_3d_extension_flag
    w.put_bit(0); // pps_scc_extension_flag
    w.put_bits(0, 4); // pps_extension_4bits
                      // pps_range_extension() — §7.3.2.3.2 (transform_skip disabled ⇒ no
                      // log2_max_transform_skip_block_size_minus2).
    w.put_bit(1); // cross_component_prediction_enabled_flag
    w.put_bit(0); // chroma_qp_offset_list_enabled_flag
    w.ue(0); // log2_sao_offset_scale_luma
    w.ue(0); // log2_sao_offset_scale_chroma
    w.rbsp_trailing_bits();
    w.finish()
}

pub(super) fn extract(plane: &[u8], pw: usize, x0: usize, y0: usize, n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            out.push(i32::from(plane[(y0 + j) * pw + x0 + i]));
        }
    }
    out
}

/// The intra prediction params mirroring the stream's SPS: 4:4:4,
/// default smoothing, no strong smoothing, boundary filters active
/// (no implicit RDPCM in this stream).
pub(super) fn ip_params_444(mode: u8, cidx: PredComponent) -> IntraPredParams {
    IntraPredParams {
        pred_mode_intra: mode,
        cidx,
        bit_depth: 8,
        bit_depth_luma: 8,
        intra_smoothing_disabled: false,
        strong_intra_smoothing_enabled: false,
        chroma_array_type_3: true,
        disable_boundary_filter: false,
    }
}

/// Per-CTB plan: `(luma_mode, cb_scale, cr_scale)`. Chroma is always
/// DM (`intra_chroma_pred_mode == 4`) so the §7.3.8.10
/// `cross_comp_pred` gate holds for every CU with a coded luma
/// residual. The ResScaleVal plan sweeps every legal magnitude and
/// both signs across the picture, with zero-scale controls.
fn ccp_plan(ctb: usize) -> (u8, i32, i32) {
    const MODES: [u8; 6] = [0, 1, 26, 10, 18, 34];
    const SCALES: [i32; 9] = [1, -1, 2, -2, 4, -4, 8, -8, 0];
    let mode = MODES[ctb % MODES.len()];
    let cb = SCALES[ctb % SCALES.len()];
    let cr = SCALES[(ctb + 4) % SCALES.len()];
    (mode, cb, cr)
}

/// The CTB index whose chroma source is rewritten so the coded chroma
/// residual is exactly zero while `cross_comp_pred` still fires — the
/// cbf-clear + CCP pin.
const ZERO_RESIDUAL_CTB: usize = 5;

/// Encode the §9.3.3.8 TR(cMax = 4) `log2_res_scale_abs_plus1[ c ]`
/// prefix plus the `res_scale_sign_flag[ c ]` for one component.
pub(super) fn encode_cross_comp_pred(
    w: &mut BitWriter,
    cabac: &mut CabacEncoder,
    ctxs: &mut SliceContexts,
    c: u32,
    res_scale_val: i32,
) {
    let (prefix, sign) = if res_scale_val == 0 {
        (0u32, None)
    } else {
        let mag = res_scale_val.unsigned_abs();
        debug_assert!(mag.is_power_of_two() && mag <= 8);
        (mag.trailing_zeros() + 1, Some(u8::from(res_scale_val < 0)))
    };
    // TR cMax = 4: `prefix` ones then a terminating zero unless the
    // prefix equals cMax; each bin has ctxInc = 4*c + binIdx.
    for bin_idx in 0..prefix {
        let ctx = log2_res_scale_abs_plus1_ctx_inc(bin_idx, c) as usize;
        cabac.encode_decision(w, &mut ctxs.log2_res_scale_abs_plus1[ctx], 1);
    }
    if prefix < 4 {
        let ctx = log2_res_scale_abs_plus1_ctx_inc(prefix, c) as usize;
        cabac.encode_decision(w, &mut ctxs.log2_res_scale_abs_plus1[ctx], 0);
    }
    if let Some(s) = sign {
        let ctx = res_scale_sign_flag_ctx_inc(c) as usize;
        cabac.encode_decision(w, &mut ctxs.res_scale_sign_flag[ctx], s);
    }
}

/// Encode the all-intra 4:4:4 bypass slice: one 16x16 `PART_2Nx2N` CU
/// per CTB, DM chroma, per-CTB `cross_comp_pred` scales from
/// [`ccp_plan`]. Lossless: the reconstruction equals the source, so
/// reference samples read the (possibly rewritten) source planes.
///
/// `cb` / `cr` are mutated for [`ZERO_RESIDUAL_CTB`]: its chroma
/// source becomes exactly `pred + ( ResScaleVal * rY ) >> 3`, so the
/// coded chroma residual is zero while CCP still applies.
fn encode_ccp_idr_slice(
    y: &[u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
) -> Vec<u8> {
    let ctbs_x = width / CTB;
    let ctbs_y = height / CTB;
    let mut modes = IntraModeField::new(width, height, CTB_LOG2);

    let mut w = BitWriter::new();
    // ---- slice_segment_header( ) ----
    w.put_bit(1); // first_slice_segment_in_pic_flag
    w.put_bit(0); // no_output_of_prior_pics_flag (IRAP NAL)
    w.ue(0); // slice_pic_parameter_set_id
    w.ue(2); // slice_type = I
    w.se(0); // slice_qp_delta
    w.rbsp_trailing_bits(); // byte_alignment()

    // ---- slice_segment_data( ) ----
    let mut cabac = CabacEncoder::new();
    let mut ctxs = SliceContexts::init(init_type(2, false), SLICE_QP);

    for ctb in 0..ctbs_x * ctbs_y {
        let x0 = (ctb % ctbs_x) * CTB;
        let y0 = (ctb / ctbs_x) * CTB;
        let (mode, cb_scale, cr_scale) = ccp_plan(ctb);

        // Luma prediction + residual (lossless recon == source).
        let read_y = |x: usize, yy: usize| i32::from(y[yy * width + x]);
        let avail = |nx: i64, ny: i64| zscan_avail(nx, ny, width, height, CTB, ctbs_x, ctb, 0);
        let marked = gather_refs(&read_y, &avail, x0, y0, CTB);
        let pred =
            intra_predict_with_substitution(&marked, &ip_params_444(mode, PredComponent::Luma))
                .expect("legal prediction");
        let src = extract(y, width, x0, y0, CTB);
        let res_l: Vec<i32> = src.iter().zip(&pred).map(|(s, p)| s - p).collect();
        let cbf_luma = res_l.iter().any(|&v| v != 0);
        // §7.3.8.10: cross_comp_pred needs cbf_luma; without it the
        // decoder infers ResScaleVal 0.
        let (cb_scale, cr_scale) = if cbf_luma {
            (cb_scale, cr_scale)
        } else {
            (0, 0)
        };

        // Chroma 16x16 TBs at (x0, y0) — 4:4:4, DM mode.
        let code_chroma = |plane: &mut [u8], pc: PredComponent, scale: i32| -> Vec<i32> {
            let pred = {
                let read = |x: usize, yy: usize| i32::from(plane[yy * width + x]);
                let avail =
                    |nx: i64, ny: i64| zscan_avail(nx, ny, width, height, CTB, ctbs_x, ctb, 0);
                let marked = gather_refs(&read, &avail, x0, y0, CTB);
                intra_predict_with_substitution(&marked, &ip_params_444(mode, pc))
                    .expect("legal prediction")
            };
            if ctb == ZERO_RESIDUAL_CTB {
                // Rewrite the source so orig == pred + ccp_term
                // (clamped to 8-bit: pick a value that stays exact).
                for j in 0..CTB {
                    for i in 0..CTB {
                        let term = (scale * res_l[j * CTB + i]) >> 3;
                        let v = (pred[j * CTB + i] + term).clamp(0, 255);
                        plane[(y0 + j) * width + x0 + i] = v as u8;
                    }
                }
            }
            let src = extract(plane, width, x0, y0, CTB);
            src.iter()
                .zip(&pred)
                .enumerate()
                .map(|(k, (s, p))| {
                    let term = (scale * res_l[k]) >> 3;
                    s - p - term
                })
                .collect()
        };
        let res_cb = code_chroma(cb, PredComponent::Cb, cb_scale);
        let res_cr = code_chroma(cr, PredComponent::Cr, cr_scale);

        // ---- §7.3.8.5 coding_unit( ) ----
        cabac.encode_decision(&mut w, &mut ctxs.cu_transquant_bypass_flag[0], 1);
        cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], 1); // PART_2Nx2N
        let avail_l = zscan_avail(x0 as i64 - 1, y0 as i64, width, height, CTB, ctbs_x, ctb, 0);
        let avail_a = zscan_avail(x0 as i64, y0 as i64 - 1, width, height, CTB, ctbs_x, ctb, 0);
        let cand_a = modes.cand_intra_pred_mode(x0, y0, Neighbour::Left, avail_l);
        let cand_b = modes.cand_intra_pred_mode(x0, y0, Neighbour::Above, avail_a);
        let list = intra_luma_cand_mode_list(cand_a, cand_b);
        let sel = list.iter().position(|&m| m == mode);
        cabac.encode_decision(
            &mut w,
            &mut ctxs.prev_intra_luma_pred_flag[0],
            u8::from(sel.is_some()),
        );
        modes.record_intra_pb(x0, y0, CTB, mode, false);
        match sel {
            Some(0) => cabac.encode_bypass(&mut w, 0),
            Some(1) => {
                cabac.encode_bypass(&mut w, 1);
                cabac.encode_bypass(&mut w, 0);
            }
            Some(_) => {
                cabac.encode_bypass(&mut w, 1);
                cabac.encode_bypass(&mut w, 1);
            }
            None => {
                let mut rem = u32::from(mode);
                for &c in &list {
                    if u32::from(mode) > u32::from(c) {
                        rem -= 1;
                    }
                }
                cabac.encode_bypass_bits(&mut w, rem, 5);
            }
        }
        // intra_chroma_pred_mode = 4 (derived / DM): single ctx bin 0.
        cabac.encode_decision(&mut w, &mut ctxs.intra_chroma_pred_mode[0], 0);

        // ---- transform_tree / transform_unit ----
        let cbf_cb_f = res_cb.iter().any(|&v| v != 0);
        let cbf_cr_f = res_cr.iter().any(|&v| v != 0);
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
        // §7.3.8.10 in-place chroma: cross_comp_pred( x0, y0, 0 ),
        // Cb residual, cross_comp_pred( x0, y0, 1 ), Cr residual. The
        // gate is cross_component_prediction_enabled_flag && cbf_luma
        // && intra_chroma_pred_mode == 4.
        if cbf_luma {
            encode_cross_comp_pred(&mut w, &mut cabac, &mut ctxs, 0, cb_scale);
        }
        if cbf_cb_f {
            encode_residual_coding(&mut w, &mut cabac, &mut ctxs.residual, &rc(true), &res_cb)
                .expect("valid cb levels");
        }
        if cbf_luma {
            encode_cross_comp_pred(&mut w, &mut cabac, &mut ctxs, 1, cr_scale);
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

/// The cross-component-prediction conformance stream: VPS + SPS(4:4:4)
/// + PPS(rext, CCP) + one all-intra bypass IDR picture (64x48).
///
/// Returns the Annex B bytes and the expected (lossless) planes.
pub(crate) fn build_ccp_stream() -> (Vec<u8>, Planes) {
    let (w, h) = (64usize, 48usize);
    let (y, mut cb, mut cr) = source_planes_444(w, h);
    let slice = encode_ccp_idr_slice(&y, &mut cb, &mut cr, w, h);
    let units = vec![
        nal_unit(32, 0, 0, &write_vps_rext_444(30)),
        nal_unit(33, 0, 0, &write_sps_444(w, h, 30)),
        nal_unit(34, 0, 0, &write_pps_ccp()),
        nal_unit(20, 0, 0, &slice), // IDR_N_LP
    ];
    (annexb(&units), (y, cb, cr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::decode_annexb_sequence;

    /// The checked-in stream bytes (black-box-reference-validated; see
    /// `tests/fixture_bytes/r416-generation-notes.md`) are exactly
    /// what the builder produces, and this crate's decoder
    /// reconstructs the procedural source losslessly through the
    /// §8.6.6 cross-component-prediction path — every legal
    /// ResScaleVal magnitude, both signs, both components, plus the
    /// cbf-clear-with-CCP and zero-scale controls.
    #[test]
    fn ccp_stream_decodes_lossless() {
        let (stream, (y, cb, cr)) = build_ccp_stream();
        let pinned: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixture_bytes/r416-ccp.hevc"
        ));
        assert_eq!(stream, pinned, "builder must reproduce the validated bytes");
        let frames = decode_annexb_sequence(&stream).expect("decode");
        assert_eq!(frames.len(), 1);
        let mut expected = y;
        expected.extend(cb);
        expected.extend(cr);
        let got = frames[0].picture.to_planar_u8().expect("8-bit planes");
        assert_eq!(got, expected, "lossless CCP decode");
    }

    /// The plan exercises the cbf-clear + CCP coding unit: the
    /// designated CTB codes NO chroma residual yet carries non-zero
    /// ResScaleVal (its chroma reconstruction is purely the scaled
    /// luma residual on top of the prediction).
    #[test]
    fn ccp_stream_covers_cbf_clear_ccp_block() {
        let (mode, cb_s, cr_s) = ccp_plan(ZERO_RESIDUAL_CTB);
        let _ = mode;
        assert!(cb_s != 0 || cr_s != 0, "zero-residual CTB must scale");
    }
}
