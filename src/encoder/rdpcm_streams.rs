//! Self-built §8.6.5 RDPCM conformance streams (test-only).
//!
//! No black-box encoder binary exposes the range-extension RDPCM
//! tools, so these tests assemble tiny lossless (transquant-bypass)
//! Annex B bitstreams from this crate's own header writers and CABAC
//! encoder:
//!
//! * an all-intra stream whose SPS signals
//!   `implicit_rdpcm_enabled_flag == 1` and whose bypass CUs cycle
//!   through predModeIntra 26 / 10 / DC / PLANAR — the mode-26/10 CUs
//!   carry §8.6.5-DPCM-differenced residuals (and their DM chroma
//!   blocks likewise), the DC / PLANAR CUs raw residuals;
//! * an IDR + P stream whose P-slice merge CUs signal
//!   `explicit_rdpcm_flag` per component (§7.3.8.11) with both
//!   directions, plus flag-0 control CUs and an all-skip CU.
//!
//! Both reconstruct losslessly, so a conforming decoder must emit the
//! procedural source planes exactly. The checked-in copies under
//! `tests/fixture_bytes/` were validated against a black-box reference
//! decoder (byte-exact; SHA-256 sums in
//! `tests/fixture_bytes/r413-generation-notes.md`), and the tests here
//! pin (a) that the builder still produces those exact bytes and
//! (b) that this crate's decoder reconstructs them losslessly.

use crate::binarization::{
    cbf_cb_ctx_inc, cbf_cr_ctx_inc, cbf_luma_ctx_inc, intra_luma_cand_mode_list,
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

/// One picture's `(Y, Cb, Cr)` planes.
type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

/// CTB / coding-block log2 size (16x16, `MinCbSizeY == CtbSizeY`).
const CTB_LOG2: u32 = 4;
const CTB: usize = 1 << CTB_LOG2;
/// SliceQpY (immaterial — every CU is transquant-bypassed).
const SLICE_QP: i32 = 26;

/// Deterministic 4:2:0 source planes: smooth gradients plus a coarse
/// checker texture, phase-shifted per frame.
fn source_planes(w: usize, h: usize, frame: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    // The bottom-right CTB is static across frames so the P slice
    // exercises a genuine skip CU.
    let static_corner =
        |x: i32, y: i32, pw: i32, ph: i32, blk: i32| -> bool { x >= pw - blk && y >= ph - blk };
    let lum = |x: i32, y: i32, f: i32| -> u8 {
        let g = (x * 3 + y * 5 + f * 17) % 197;
        let t = ((x / 4 + y / 4 + f) % 3) * 23;
        (16 + g + t).clamp(0, 255) as u8
    };
    let chr = |x: i32, y: i32, ph: i32, f: i32| -> u8 {
        let g = (x * 7 + y * 2 + f * 29 + ph) % 151;
        let t = ((x / 4 + y / 8) % 2) * 31;
        (40 + g + t).clamp(0, 255) as u8
    };
    let f = frame as i32;
    let mut y_p = Vec::with_capacity(w * h);
    for yy in 0..h as i32 {
        for xx in 0..w as i32 {
            let ef = if static_corner(xx, yy, w as i32, h as i32, 16) {
                0
            } else {
                f
            };
            y_p.push(lum(xx, yy, ef));
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    let mut cb_p = Vec::with_capacity(cw * ch);
    let mut cr_p = Vec::with_capacity(cw * ch);
    for yy in 0..ch as i32 {
        for xx in 0..cw as i32 {
            let ef = if static_corner(xx, yy, cw as i32, ch as i32, 8) {
                0
            } else {
                f
            };
            cb_p.push(chr(xx, yy, 0, ef));
            cr_p.push(chr(xx, yy, 61, ef));
        }
    }
    (y_p, cb_p, cr_p)
}

/// §7.3.3 — profile tier level for Format Range Extensions
/// (`general_profile_idc == 4`) at the given level.
pub(crate) fn write_ptl_rext(w: &mut BitWriter, level_idc: u8) {
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
                  // profile_idc 4: max_12bit / max_10bit / max_8bit /
                  // max_422chroma / max_420chroma constraint flags set
                  // (4:2:0 8-bit), then max_monochrome / intra /
                  // one_picture_only / lossless 0 + 34 reserved bits.
    w.put_bit(1); // general_max_12bit_constraint_flag
    w.put_bit(1); // general_max_10bit_constraint_flag
    w.put_bit(1); // general_max_8bit_constraint_flag
    w.put_bit(1); // general_max_422chroma_constraint_flag
    w.put_bit(1); // general_max_420chroma_constraint_flag
    w.put_bit(0); // general_max_monochrome_constraint_flag
    w.put_bit(0); // general_intra_constraint_flag
    w.put_bit(0); // general_one_picture_only_constraint_flag
    w.put_bit(0); // general_lossless_constraint_flag
    w.put_bits(0, 32); // general_reserved_zero_34bits...
    w.put_bits(0, 2);
    w.put_bit(0); // general_inbld_flag
    w.put_bits(u32::from(level_idc), 8); // general_level_idc
}

/// §7.3.2.1 — single-layer VPS with the Range-Extensions PTL.
fn write_vps_rext(level_idc: u8) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(0, 4); // vps_video_parameter_set_id
    w.put_bit(1); // vps_base_layer_internal_flag
    w.put_bit(1); // vps_base_layer_available_flag
    w.put_bits(0, 6); // vps_max_layers_minus1
    w.put_bits(0, 3); // vps_max_sub_layers_minus1
    w.put_bit(1); // vps_temporal_id_nesting_flag
    w.put_bits(0xFFFF, 16); // vps_reserved_0xffff_16bits
    write_ptl_rext(&mut w, level_idc);
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

/// §7.3.2.2 — 4:2:0 8-bit SPS (CTB 16, no PCM, SAO off) carrying an
/// `sps_range_extension()` with `implicit_rdpcm_enabled_flag` and
/// `explicit_rdpcm_enabled_flag` both set.
fn write_sps_rext(width: usize, height: usize, level_idc: u8) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(0, 4); // sps_video_parameter_set_id
    w.put_bits(0, 3); // sps_max_sub_layers_minus1
    w.put_bit(1); // sps_temporal_id_nesting_flag
    write_ptl_rext(&mut w, level_idc);
    w.ue(0); // sps_seq_parameter_set_id
    w.ue(1); // chroma_format_idc = 4:2:0
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
    w.put_bit(1); // sps_range_extension_flag
    w.put_bit(0); // sps_multilayer_extension_flag
    w.put_bit(0); // sps_3d_extension_flag
    w.put_bit(0); // sps_scc_extension_flag
    w.put_bits(0, 4); // sps_extension_4bits
                      // sps_range_extension() — §7.3.2.2.2.
    w.put_bit(0); // transform_skip_rotation_enabled_flag
    w.put_bit(0); // transform_skip_context_enabled_flag
    w.put_bit(1); // implicit_rdpcm_enabled_flag
    w.put_bit(1); // explicit_rdpcm_enabled_flag
    w.put_bit(0); // extended_precision_processing_flag
    w.put_bit(0); // intra_smoothing_disabled_flag
    w.put_bit(0); // high_precision_offsets_enabled_flag
    w.put_bit(0); // persistent_rice_adaptation_enabled_flag
    w.put_bit(0); // cabac_bypass_alignment_enabled_flag
    w.rbsp_trailing_bits();
    w.finish()
}

/// §7.3.2.3 — PPS with `transquant_bypass_enabled_flag == 1`,
/// deblocking disabled, sign hiding off.
pub(crate) fn write_pps_bypass() -> Vec<u8> {
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
    w.put_bit(0); // pps_extension_present_flag
    w.rbsp_trailing_bits();
    w.finish()
}

/// §8.6.5 in the ENCODE direction: difference the residual along the
/// DPCM direction so the decoder's accumulation restores it exactly.
fn dpcm_difference(res: &[i32], n: usize, vertical: bool) -> Vec<i32> {
    let mut out = res.to_vec();
    if vertical {
        for y in (1..n).rev() {
            for x in 0..n {
                out[y * n + x] -= res[(y - 1) * n + x];
            }
        }
    } else {
        for y in 0..n {
            for x in (1..n).rev() {
                out[y * n + x] -= res[y * n + x - 1];
            }
        }
    }
    out
}

fn extract(plane: &[u8], pw: usize, x0: usize, y0: usize, n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            out.push(i32::from(plane[(y0 + j) * pw + x0 + i]));
        }
    }
    out
}

/// The intra prediction params mirroring the stream's SPS: smoothing
/// on (no rext disable), no strong smoothing, and — because every CU
/// is transquant-bypassed under `implicit_rdpcm_enabled_flag == 1` —
/// the §8.4.4.2.6 `disableIntraBoundaryFilter` (angular 10/26 filters
/// only; the §8.4.4.2.5 DC gate is the SCC flag, which is 0).
fn ip_params(mode: u8, cidx: PredComponent) -> IntraPredParams {
    IntraPredParams {
        pred_mode_intra: mode,
        cidx,
        bit_depth: 8,
        bit_depth_luma: 8,
        intra_smoothing_disabled: false,
        strong_intra_smoothing_enabled: false,
        chroma_array_type_3: false,
        disable_boundary_filter: mode != crate::intra_pred::INTRA_DC,
    }
}

/// Per-CTB intra plan: `(luma_mode, intra_chroma_pred_mode_raw)`.
///
/// The §8.4.4.2.6 mode-26 (left-column) / mode-10 (top-row) boundary
/// filters are disabled by `disableIntraBoundaryFilter` under implicit
/// RDPCM with transquant bypass; a surveyed black-box reference
/// decoder applies them regardless, so luma DPCM modes are placed
/// where the relevant reference edge is a picture boundary (the
/// §8.4.4.2.2 substitution makes the filter a no-op there): mode 10
/// on the top CTB row, mode 26 down the left CTB column. Interior CUs
/// carry DC / PLANAR luma (non-DPCM controls) while their CHROMA
/// blocks — never boundary-filtered — take mode 26 / 10 through
/// `intra_chroma_pred_mode` 1 / 2 (Table 8-3), keeping implicit RDPCM
/// dense across the picture.
fn intra_plan(col: usize, row: usize, ctb: usize) -> (u8, u8) {
    if row == 0 {
        if col == 0 {
            (26, 4) // both edges are picture boundaries
        } else {
            (10, 4) // DM ⇒ chroma 10
        }
    } else if col == 0 {
        (26, 4) // DM ⇒ chroma 26
    } else {
        // Interior: luma control mode (no DPCM), chroma 26 / 10.
        let luma = if ctb % 2 == 0 { 1 } else { 0 };
        let chroma_raw = if ctb % 4 < 2 { 1 } else { 2 };
        (luma, chroma_raw)
    }
}

/// Table 8-3 — `IntraPredModeC` from `intra_chroma_pred_mode` and the
/// luma mode (the collision row maps to 34; the plan never collides).
fn mode_c_of(raw: u8, luma: u8) -> u8 {
    let base = match raw {
        0 => 0u8,
        1 => 26,
        2 => 10,
        3 => 1,
        _ => return luma,
    };
    if base == luma {
        34
    } else {
        base
    }
}

/// Encode the all-intra bypass slice: one 16x16 `PART_2Nx2N` CU per
/// CTB, `cu_transquant_bypass_flag == 1`, modes per [`intra_plan`].
/// Lossless: the reconstruction equals the source, so reference
/// samples read the source planes directly.
fn encode_bypass_idr_slice(y: &[u8], cb: &[u8], cr: &[u8], width: usize, height: usize) -> Vec<u8> {
    let (cw, ch) = (width / 2, height / 2);
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
        let (mode, chroma_raw) = intra_plan(ctb % ctbs_x, ctb / ctbs_x, ctb);
        let mode_c = mode_c_of(chroma_raw, mode);

        // Luma prediction + residual (lossless recon == source).
        let read_y = |x: usize, yy: usize| i32::from(y[yy * width + x]);
        let avail = |nx: i64, ny: i64| zscan_avail(nx, ny, width, height, CTB, ctbs_x, ctb, 0);
        let marked = gather_refs(&read_y, &avail, x0, y0, CTB);
        let pred = intra_predict_with_substitution(&marked, &ip_params(mode, PredComponent::Luma))
            .expect("legal prediction");
        let src = extract(y, width, x0, y0, CTB);
        let mut res_l: Vec<i32> = src.iter().zip(&pred).map(|(s, p)| s - p).collect();
        if mode == 26 || mode == 10 {
            res_l = dpcm_difference(&res_l, CTB, mode == 26);
        }

        // Chroma 8x8 TBs at (x0/2, y0/2), mode `mode_c`.
        let (cx0, cy0) = (x0 / 2, y0 / 2);
        let code_chroma = |plane: &[u8], pc: PredComponent| -> Vec<i32> {
            let read = |x: usize, yy: usize| i32::from(plane[yy * cw + x]);
            let avail = |nx: i64, ny: i64| zscan_avail(nx, ny, cw, ch, CTB / 2, ctbs_x, ctb, 0);
            let marked = gather_refs(&read, &avail, cx0, cy0, 8);
            let pred = intra_predict_with_substitution(&marked, &ip_params(mode_c, pc))
                .expect("legal prediction");
            let src = extract(plane, cw, cx0, cy0, 8);
            let mut res: Vec<i32> = src.iter().zip(&pred).map(|(s, p)| s - p).collect();
            if mode_c == 26 || mode_c == 10 {
                res = dpcm_difference(&res, 8, mode_c == 26);
            }
            res
        };
        let res_cb = code_chroma(cb, PredComponent::Cb);
        let res_cr = code_chroma(cr, PredComponent::Cr);

        // ---- §7.3.8.5 coding_unit( ) ----
        cabac.encode_decision(&mut w, &mut ctxs.cu_transquant_bypass_flag[0], 1);
        // part_mode (intra at MinCb): "1" = PART_2Nx2N.
        cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], 1);
        // Luma mode signalling against the §8.4.2 candidate list.
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
        // intra_chroma_pred_mode: 4 (derived) is the single ctx bin
        // "0"; values 0..3 are ctx bin "1" plus two FL bypass bits.
        if chroma_raw == 4 {
            cabac.encode_decision(&mut w, &mut ctxs.intra_chroma_pred_mode[0], 0);
        } else {
            cabac.encode_decision(&mut w, &mut ctxs.intra_chroma_pred_mode[0], 1);
            cabac.encode_bypass_bits(&mut w, u32::from(chroma_raw), 2);
        }

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
        let rc = |log2: u32, is_chroma: bool| ResidualCodingParams {
            log2_trafo_size: log2,
            is_chroma,
            scan_idx: residual_coding_scan_idx(
                true,
                log2,
                u8::from(is_chroma),
                1,
                u32::from(if is_chroma { mode_c } else { mode }),
            ),
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
            encode_residual_coding(
                &mut w,
                &mut cabac,
                &mut ctxs.residual,
                &rc(4, false),
                &res_l,
            )
            .expect("valid luma levels");
        }
        if cbf_cb_f {
            encode_residual_coding(
                &mut w,
                &mut cabac,
                &mut ctxs.residual,
                &rc(3, true),
                &res_cb,
            )
            .expect("valid cb levels");
        }
        if cbf_cr_f {
            encode_residual_coding(
                &mut w,
                &mut cabac,
                &mut ctxs.residual,
                &rc(3, true),
                &res_cr,
            )
            .expect("valid cr levels");
        }

        cabac.encode_terminate(&mut w, u8::from(ctb == ctbs_x * ctbs_y - 1));
    }
    w.align_zero();
    w.finish()
}

/// Encode the P slice: every CTB one 16x16 CU predicting from the
/// previous picture at MV (0, 0) — `MaxNumMergeCand == 1`, so both
/// merge and skip resolve to the zero candidate. CUs with residual are
/// bypass merge CUs whose per-component `residual_coding( )` signals
/// `explicit_rdpcm_flag` (cycling direction, with flag-0 controls);
/// all-zero CTBs are skip CUs.
#[allow(clippy::too_many_arguments)]
fn encode_bypass_p_slice(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    ry: &[u8],
    rcb: &[u8],
    rcr: &[u8],
    width: usize,
    height: usize,
) -> Vec<u8> {
    let (cw, _ch) = (width / 2, height / 2);
    let ctbs_x = width / CTB;
    let ctbs_y = height / CTB;

    let mut w = BitWriter::new();
    // ---- slice_segment_header( ) ----
    w.put_bit(1); // first_slice_segment_in_pic_flag
    w.ue(0); // slice_pic_parameter_set_id
    w.ue(1); // slice_type = P
    w.put_bits(1, 8); // slice_pic_order_cnt_lsb (log2 max = 8 bits)
    w.put_bit(0); // short_term_ref_pic_set_sps_flag
                  // st_ref_pic_set( 0 ): one negative reference at −1.
    w.ue(1); // num_negative_pics
    w.ue(0); // num_positive_pics
    w.ue(0); // delta_poc_s0_minus1[0]
    w.put_bit(1); // used_by_curr_pic_s0_flag[0]
    w.put_bit(0); // num_ref_idx_active_override_flag
    w.ue(4); // five_minus_max_num_merge_cand (MaxNumMergeCand = 1)
    w.se(0); // slice_qp_delta
    w.rbsp_trailing_bits(); // byte_alignment()

    // ---- slice_segment_data( ) ----
    let mut cabac = CabacEncoder::new();
    let mut ctxs = SliceContexts::init(init_type(1, false), SLICE_QP);
    let mut skip_flags = vec![false; ctbs_x * ctbs_y];

    for ctb in 0..ctbs_x * ctbs_y {
        let x0 = (ctb % ctbs_x) * CTB;
        let y0 = (ctb / ctbs_x) * CTB;
        let col = ctb % ctbs_x;
        let row = ctb / ctbs_x;

        // MV (0, 0) prediction from the reference picture: the
        // residual is the plane difference.
        let diff = |cur: &[u8], refp: &[u8], pw: usize, bx: usize, by: usize, n: usize| {
            let mut out = Vec::with_capacity(n * n);
            for j in 0..n {
                for i in 0..n {
                    let idx = (by + j) * pw + bx + i;
                    out.push(i32::from(cur[idx]) - i32::from(refp[idx]));
                }
            }
            out
        };
        let res_l = diff(y, ry, width, x0, y0, CTB);
        let res_cb = diff(cb, rcb, cw, x0 / 2, y0 / 2, 8);
        let res_cr = diff(cr, rcr, cw, x0 / 2, y0 / 2, 8);
        let all_zero = res_l.iter().all(|&v| v == 0)
            && res_cb.iter().all(|&v| v == 0)
            && res_cr.iter().all(|&v| v == 0);

        // §7.3.8.5: cu_transquant_bypass_flag precedes cu_skip_flag.
        cabac.encode_decision(&mut w, &mut ctxs.cu_transquant_bypass_flag[0], 1);
        // cu_skip_flag with the §9.3.4.2.2 neighbour ctxInc.
        let ctx_inc = usize::from(col > 0 && skip_flags[ctb - 1])
            + usize::from(row > 0 && skip_flags[ctb - ctbs_x]);
        cabac.encode_decision(&mut w, &mut ctxs.cu_skip_flag[ctx_inc], u8::from(all_zero));
        skip_flags[ctb] = all_zero;
        if all_zero {
            // Skip CU: MaxNumMergeCand == 1 ⇒ no merge_idx bins.
            cabac.encode_terminate(&mut w, u8::from(ctb == ctbs_x * ctbs_y - 1));
            continue;
        }

        cabac.encode_decision(&mut w, &mut ctxs.pred_mode_flag[0], 0); // MODE_INTER
        cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], 1); // PART_2Nx2N
        cabac.encode_decision(&mut w, &mut ctxs.merge_flag[0], 1); // merge_flag
                                                                   // merge_idx absent (MaxNumMergeCand == 1);
                                                                   // rqt_root_cbf inferred 1 (2Nx2N merge).

        // Per-CTB explicit-RDPCM plan: two of three CUs signal the
        // flag (alternating direction), the third is a flag-0 control.
        let flag = ctb % 3 != 2;
        let vertical = ctb % 2 == 0;

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
        // §7.3.8.10: inter at depth 0 with no chroma cbf ⇒ cbf_luma
        // inferred 1; otherwise signalled.
        if cbf_cb_f || cbf_cr_f {
            cabac.encode_decision(
                &mut w,
                &mut ctxs.cbf_luma[cbf_luma_ctx_inc(0) as usize],
                u8::from(cbf_luma),
            );
        } else {
            assert!(cbf_luma, "all-zero CU must have been a skip CU");
        }

        let emit = |cabac: &mut CabacEncoder,
                    wtr: &mut BitWriter,
                    ctxs: &mut SliceContexts,
                    res: &[i32],
                    log2: u32,
                    is_chroma: bool| {
            // §7.3.8.11 prelude: explicit_rdpcm_flag / dir (inter +
            // enabled + bypass; Table 9-32 / 9-33 slot by component).
            let slot = usize::from(is_chroma);
            cabac.encode_decision(wtr, &mut ctxs.explicit_rdpcm_flag[slot], u8::from(flag));
            let levels = if flag {
                cabac.encode_decision(
                    wtr,
                    &mut ctxs.explicit_rdpcm_dir_flag[slot],
                    u8::from(vertical),
                );
                dpcm_difference(res, 1 << log2, vertical)
            } else {
                res.to_vec()
            };
            let rc = ResidualCodingParams {
                log2_trafo_size: log2,
                is_chroma,
                scan_idx: residual_coding_scan_idx(false, log2, u8::from(is_chroma), 1, 0),
                sign_data_hiding_enabled_flag: false,
                sign_hidden_suppressed: false,
                transform_skip_sig_ctx: false,
                persistent_rice_adaptation_enabled_flag: false,
                cabac_bypass_alignment_enabled_flag: false,
                extended_precision_processing_flag: false,
                bit_depth: 8,
                rice_stat_transform_skip: false,
            };
            encode_residual_coding(wtr, cabac, &mut ctxs.residual, &rc, &levels)
                .expect("valid inter levels");
        };
        if cbf_luma {
            emit(&mut cabac, &mut w, &mut ctxs, &res_l, 4, false);
        }
        if cbf_cb_f {
            emit(&mut cabac, &mut w, &mut ctxs, &res_cb, 3, true);
        }
        if cbf_cr_f {
            emit(&mut cabac, &mut w, &mut ctxs, &res_cr, 3, true);
        }

        cabac.encode_terminate(&mut w, u8::from(ctb == ctbs_x * ctbs_y - 1));
    }
    w.align_zero();
    w.finish()
}

/// The implicit-RDPCM conformance stream: VPS + SPS(rext) + PPS +
/// one all-intra bypass IDR picture (64x48).
pub(crate) fn build_implicit_stream() -> (Vec<u8>, Planes) {
    let (w, h) = (64usize, 48usize);
    let (y, cb, cr) = source_planes(w, h, 0);
    let slice = encode_bypass_idr_slice(&y, &cb, &cr, w, h);
    let units = vec![
        nal_unit(32, 0, 0, &write_vps_rext(30)),
        nal_unit(33, 0, 0, &write_sps_rext(w, h, 30)),
        nal_unit(34, 0, 0, &write_pps_bypass()),
        nal_unit(20, 0, 0, &slice), // IDR_N_LP
    ];
    (annexb(&units), (y, cb, cr))
}

/// The explicit-RDPCM conformance stream: the implicit IDR picture
/// followed by a P picture whose merge CUs carry per-component
/// explicit-RDPCM residuals (64x48, 2 frames).
pub(crate) fn build_explicit_stream() -> (Vec<u8>, Planes, Planes) {
    let (w, h) = (64usize, 48usize);
    let (y0, cb0, cr0) = source_planes(w, h, 0);
    let (y1, cb1, cr1) = source_planes(w, h, 1);
    let idr = encode_bypass_idr_slice(&y0, &cb0, &cr0, w, h);
    let p = encode_bypass_p_slice(&y1, &cb1, &cr1, &y0, &cb0, &cr0, w, h);
    let units = vec![
        nal_unit(32, 0, 0, &write_vps_rext(30)),
        nal_unit(33, 0, 0, &write_sps_rext(w, h, 30)),
        nal_unit(34, 0, 0, &write_pps_bypass()),
        nal_unit(20, 0, 0, &idr), // IDR_N_LP
        nal_unit(1, 0, 0, &p),    // TRAIL_R
    ];
    (annexb(&units), (y0, cb0, cr0), (y1, cb1, cr1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::decode_annexb_sequence;

    fn planes_of(frame: &crate::sequence::DecodedFrame) -> Vec<u8> {
        frame.picture.to_planar_u8().expect("8-bit planes")
    }

    /// The checked-in stream bytes (black-box-reference-validated;
    /// see `tests/fixture_bytes/r413-generation-notes.md`) are exactly
    /// what the builder produces, and this crate's decoder
    /// reconstructs the procedural source losslessly through the
    /// §8.4.4.1 implicit-RDPCM path.
    #[test]
    fn implicit_rdpcm_stream_decodes_lossless() {
        let (stream, (y, cb, cr)) = build_implicit_stream();
        let pinned: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixture_bytes/r413-rdpcm-implicit.hevc"
        ));
        assert_eq!(stream, pinned, "builder must reproduce the validated bytes");
        let frames = decode_annexb_sequence(&stream).expect("decode");
        assert_eq!(frames.len(), 1);
        let mut expected = y;
        expected.extend(cb);
        expected.extend(cr);
        assert_eq!(planes_of(&frames[0]), expected, "lossless implicit RDPCM");
    }

    /// Same pin for the explicit-RDPCM P picture (§8.5.4.2 step 3 /
    /// §8.5.4.3 step 4 with both mDir values plus flag-0 controls).
    #[test]
    fn explicit_rdpcm_stream_decodes_lossless() {
        let (stream, f0, f1) = build_explicit_stream();
        let pinned: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixture_bytes/r413-rdpcm-explicit.hevc"
        ));
        assert_eq!(stream, pinned, "builder must reproduce the validated bytes");
        let frames = decode_annexb_sequence(&stream).expect("decode");
        assert_eq!(frames.len(), 2);
        for (frame, (y, cb, cr)) in frames.iter().zip([f0, f1]) {
            let mut expected = y;
            expected.extend(cb);
            expected.extend(cr);
            assert_eq!(planes_of(frame), expected, "lossless explicit RDPCM");
        }
    }
}
