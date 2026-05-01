//! P-slice header + CABAC slice-data emission.
//!
//! Scope:
//!
//! * Single-segment `TrailR` slice referencing POC = current - 1.
//! * Per-CU pipeline: INTER 2Nx2N, `cu_skip_flag = 0`, `pred_mode_flag = 0`,
//!   `merge_flag = 0`, implicit `ref_idx_l0 = 0`, a single AMVP candidate
//!   (`mvp_l0_flag = 0`) with integer-pel MVD from a ±8 SAD search.
//! * Integer-pel motion compensation — same pixels the decoder's
//!   `luma_mc` / `chroma_mc` produce for full-pel MVs.
//! * One 2Nx2N transform unit (log2_tb = 4) for luma + two 8×8 chroma TBs.
//!   `rqt_root_cbf = 0` when the residual quantises to zero.
//! * The spec-level SPS flags this encoder relies on are:
//!   `num_short_term_ref_pic_sets = 0` (so each slice emits an inline RPS),
//!   `sps_temporal_mvp_enabled_flag = 0`, `pcm_enabled_flag = 0`,
//!   `amp_enabled_flag = 0`, `scaling_list_enabled_flag = 0`,
//!   `sample_adaptive_offset_enabled_flag = 0`, and
//!   `max_transform_hierarchy_depth_inter = 0`.

use crate::cabac::{
    init_row, CtxState, InitType, ABS_MVD_GREATER_FLAGS_INIT_VALUES, CU_SKIP_FLAG_INIT_VALUES,
    MERGE_FLAG_INIT_VALUES, MVP_LX_FLAG_INIT_VALUES, PART_MODE_INIT_VALUES,
    PRED_MODE_FLAG_INIT_VALUES, RQT_ROOT_CBF_INIT_VALUES,
};
use crate::encoder::bit_writer::{write_rbsp_trailing_bits, BitWriter};
use crate::encoder::cabac_writer::CabacWriter;
use crate::encoder::nal_writer::build_annex_b_nal;
use crate::encoder::params::EncoderConfig;
use crate::encoder::residual_writer::{encode_residual, ResidualCtx};
use crate::nal::{NalHeader, NalUnitType};
use crate::transform::{
    dequantize_flat, forward_transform_2d, inverse_transform_2d, quantize_flat,
};
use oxideav_core::VideoFrame;

/// CTU size = 16. minCU = 16. 2Nx2N inter CU ⇒ one 16×16 luma PB + TB.
const CTU_SIZE: u32 = 16;
/// Slice QP — matches the I-slice emitter so rate/quality stay aligned.
const SLICE_QP_Y: i32 = 26;
/// Integer search window (luma pixels). ±`ME_RANGE` both directions.
const ME_RANGE: i32 = 8;

/// Reconstructed reference frame the encoder uses as the L0 target for
/// P-slice motion compensation + also writes back into for neighbour-CU
/// reuse inside the current frame.
#[derive(Clone)]
pub struct ReferenceFrame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
    pub y_stride: usize,
    pub c_stride: usize,
}

impl ReferenceFrame {
    pub fn from_planes(width: u32, height: u32, y: Vec<u8>, cb: Vec<u8>, cr: Vec<u8>) -> Self {
        let w = width as usize;
        let cw = w / 2;
        Self {
            width,
            height,
            y,
            cb,
            cr,
            y_stride: w,
            c_stride: cw,
        }
    }

    #[inline]
    pub(crate) fn sample_y(&self, x: i32, y: i32) -> u8 {
        let xc = x.clamp(0, self.width as i32 - 1) as usize;
        let yc = y.clamp(0, self.height as i32 - 1) as usize;
        self.y[yc * self.y_stride + xc]
    }

    #[inline]
    pub(crate) fn sample_c(&self, x: i32, y: i32, is_cr: bool) -> u8 {
        let cw = (self.width / 2) as i32;
        let ch = (self.height / 2) as i32;
        let xc = x.clamp(0, cw - 1) as usize;
        let yc = y.clamp(0, ch - 1) as usize;
        let plane = if is_cr { &self.cr } else { &self.cb };
        plane[yc * self.c_stride + xc]
    }
}

/// Build Annex B bytes for a P (TrailR) slice NAL.
pub fn build_p_slice_nal(
    cfg: &EncoderConfig,
    frame: &VideoFrame,
    frame_idx: u32,
    ref_frame: &ReferenceFrame,
) -> Vec<u8> {
    let rbsp = build_p_slice_rbsp(cfg, frame, frame_idx, ref_frame);
    build_annex_b_nal(NalHeader::for_type(NalUnitType::TrailR), &rbsp)
}

/// Build the RBSP payload for a P slice.
pub fn build_p_slice_rbsp(
    cfg: &EncoderConfig,
    frame: &VideoFrame,
    frame_idx: u32,
    ref_frame: &ReferenceFrame,
) -> Vec<u8> {
    let mut bw = BitWriter::new();

    // ---- slice_segment_header() -------------------------------------
    bw.write_u1(1); // first_slice_segment_in_pic_flag
                    // NOT IRAP, so no_output_of_prior_pics_flag is skipped.
    bw.write_ue(0); // slice_pic_parameter_set_id
    bw.write_ue(1); // slice_type = P

    // Not IDR: emit POC LSB + inline RPS.
    let poc_lsb = frame_idx & 0xFF; // log2_max_poc_lsb_minus4 = 4 → 8 bits.
    bw.write_bits(poc_lsb, 8);
    bw.write_u1(0); // short_term_ref_pic_set_sps_flag = 0 → inline.

    // Inline st_ref_pic_set(0): since st_rps_idx == 0 the
    // `inter_ref_pic_set_prediction_flag` is NOT signalled per §7.3.7.
    bw.write_ue(1); // num_negative_pics = 1
    bw.write_ue(0); // num_positive_pics = 0
                    // Entry 0: delta_poc_s0_minus1 = 0 (i.e. delta = -1), used_by_curr_pic_s0 = 1.
    bw.write_ue(0);
    bw.write_u1(1);
    // sps.long_term_ref_pics_present_flag = 0 → skipped.
    // sps.sps_temporal_mvp_enabled_flag = 0 → skipped.

    // SAO disabled → no slice_sao_*_flag.
    // P/B branch.
    bw.write_u1(0); // num_ref_idx_active_override_flag = 0.
                    // pps.lists_modification_present_flag = 0 → skipped.
                    // pps.cabac_init_present_flag = 0 → skipped.
                    // slice_temporal_mvp_enabled_flag = 0 → collocated skipped.
                    // weighted_pred_flag = 0 → skipped.
    bw.write_ue(4); // five_minus_max_num_merge_cand → max_num_merge_cand = 1.

    bw.write_se(0); // slice_qp_delta = 0.
                    // pps_slice_chroma_qp_offsets_present_flag = 0 → skipped.
                    // deblocking_filter_override_enabled_flag = 0 → skipped.
                    // (SAO || !deblock_disabled) is false → slice_loop_filter_across_slices_enabled_flag skipped.
                    // No tiles, no wpp → no entry point offsets.
                    // slice_segment_header_extension_present_flag = 0 → no extension length.
    write_rbsp_trailing_bits(&mut bw);

    // ---- slice_data() ------------------------------------------------
    let mut enc = PEncoderState::new(cfg, ref_frame.clone());
    {
        let mut cabac = CabacWriter::new(&mut bw);
        let pic_w_ctb = cfg.width.div_ceil(CTU_SIZE);
        let pic_h_ctb = cfg.height.div_ceil(CTU_SIZE);
        let total_ctbs = pic_w_ctb * pic_h_ctb;
        for i in 0..total_ctbs {
            let ctb_x = (i % pic_w_ctb) * CTU_SIZE;
            let ctb_y = (i / pic_w_ctb) * CTU_SIZE;
            enc.encode_ctu(&mut cabac, frame, ctb_x, ctb_y);
            let is_last = i + 1 == total_ctbs;
            cabac.encode_terminate(if is_last { 1 } else { 0 });
        }
        cabac.encode_flush();
    }
    bw.align_to_byte_zero();
    bw.finish()
}

/// Return the reconstructed frame the encoder produced while writing the
/// P slice — wrapped in a [`ReferenceFrame`] so the caller can feed it
/// back as the next slice's L0 reference. The L0 reference is assumed
/// to be at POC = frame_idx − 1 (i.e. the immediately preceding picture).
pub fn build_p_slice_with_reconstruction(
    cfg: &EncoderConfig,
    frame: &VideoFrame,
    frame_idx: u32,
    ref_frame: &ReferenceFrame,
) -> (Vec<u8>, ReferenceFrame) {
    build_p_slice_with_reconstruction_delta(
        cfg, frame, frame_idx, /* delta_l0 */ -1, ref_frame,
    )
}

/// Same as [`build_p_slice_with_reconstruction`] but allows the caller to
/// declare a non-default L0 POC delta. Used by the mini-GOP-2 (B-slice)
/// path where the P-anchor at display POC `2k` references the previous
/// anchor at POC `2k − 2`.
pub fn build_p_slice_with_reconstruction_delta(
    cfg: &EncoderConfig,
    frame: &VideoFrame,
    frame_idx: u32,
    delta_l0: i32,
    ref_frame: &ReferenceFrame,
) -> (Vec<u8>, ReferenceFrame) {
    debug_assert!(delta_l0 < 0, "P-slice L0 ref must have negative POC delta");
    let rbsp;
    let recon;
    {
        let mut bw = BitWriter::new();
        bw.write_u1(1);
        bw.write_ue(0);
        bw.write_ue(1);
        let poc_lsb = frame_idx & 0xFF;
        bw.write_bits(poc_lsb, 8);
        bw.write_u1(0);
        bw.write_ue(1);
        bw.write_ue(0);
        bw.write_ue((-delta_l0 - 1) as u32);
        bw.write_u1(1);
        bw.write_u1(0);
        bw.write_ue(4);
        bw.write_se(0);
        write_rbsp_trailing_bits(&mut bw);

        let mut enc = PEncoderState::new(cfg, ref_frame.clone());
        {
            let mut cabac = CabacWriter::new(&mut bw);
            let pic_w_ctb = cfg.width.div_ceil(CTU_SIZE);
            let pic_h_ctb = cfg.height.div_ceil(CTU_SIZE);
            let total_ctbs = pic_w_ctb * pic_h_ctb;
            for i in 0..total_ctbs {
                let ctb_x = (i % pic_w_ctb) * CTU_SIZE;
                let ctb_y = (i / pic_w_ctb) * CTU_SIZE;
                enc.encode_ctu(&mut cabac, frame, ctb_x, ctb_y);
                let is_last = i + 1 == total_ctbs;
                cabac.encode_terminate(if is_last { 1 } else { 0 });
            }
            cabac.encode_flush();
        }
        bw.align_to_byte_zero();
        recon =
            ReferenceFrame::from_planes(cfg.width, cfg.height, enc.rec_y, enc.rec_cb, enc.rec_cr);
        rbsp = bw.finish();
    }
    let nal = build_annex_b_nal(NalHeader::for_type(NalUnitType::TrailR), &rbsp);
    (nal, recon)
}

/// Per-slice P-frame CABAC encoding state.
struct PEncoderState {
    cfg: EncoderConfig,
    ref_pic: ReferenceFrame,
    rec_y: Vec<u8>,
    rec_cb: Vec<u8>,
    rec_cr: Vec<u8>,
    y_stride: usize,
    c_stride: usize,
    // Per-4×4 block: Some(mv_luma_quarter_pel) when a PB at that origin is
    // inter-coded, None when unavailable / outside picture.
    mv_grid: Vec<Option<(i32, i32)>>,
    grid_w4: usize,
    grid_h4: usize,

    // CABAC contexts.
    cu_skip_flag: [CtxState; 3],
    pred_mode_flag: [CtxState; 1],
    part_mode: [CtxState; 4],
    merge_flag: [CtxState; 1],
    mvp_lx_flag: [CtxState; 1],
    abs_mvd_greater: [CtxState; 2],
    rqt_root_cbf: [CtxState; 1],
    residual: ResidualCtx,
}

impl PEncoderState {
    fn new(cfg: &EncoderConfig, ref_pic: ReferenceFrame) -> Self {
        let w = cfg.width as usize;
        let h = cfg.height as usize;
        let cw = w / 2;
        let ch = h / 2;
        let it = InitType::Pa; // cabac_init_flag = 0 for P slices.
        let grid_w4 = w / 4;
        let grid_h4 = h / 4;
        Self {
            cfg: *cfg,
            ref_pic,
            rec_y: vec![128u8; w * h],
            rec_cb: vec![128u8; cw * ch],
            rec_cr: vec![128u8; cw * ch],
            y_stride: w,
            c_stride: cw,
            mv_grid: vec![None; grid_w4 * grid_h4],
            grid_w4,
            grid_h4,
            cu_skip_flag: init_row(&CU_SKIP_FLAG_INIT_VALUES, it, SLICE_QP_Y),
            pred_mode_flag: init_row(&PRED_MODE_FLAG_INIT_VALUES, it, SLICE_QP_Y),
            part_mode: init_row(&PART_MODE_INIT_VALUES, it, SLICE_QP_Y),
            merge_flag: init_row(&MERGE_FLAG_INIT_VALUES, it, SLICE_QP_Y),
            mvp_lx_flag: init_row(&MVP_LX_FLAG_INIT_VALUES, it, SLICE_QP_Y),
            abs_mvd_greater: init_row(&ABS_MVD_GREATER_FLAGS_INIT_VALUES, it, SLICE_QP_Y),
            rqt_root_cbf: init_row(&RQT_ROOT_CBF_INIT_VALUES, it, SLICE_QP_Y),
            residual: ResidualCtx::with_init_type(SLICE_QP_Y, it),
        }
    }

    fn encode_ctu(&mut self, cw: &mut CabacWriter<'_>, src: &VideoFrame, x0: u32, y0: u32) {
        // One 16×16 CU per 16×16 CTU (no split_cu_flag at min CB).
        // cu_skip_flag: emit 0. ctx_inc depends on left / above neighbour's
        // skip status — we never flag skip so all neighbours contribute 0,
        // but we still use the same ctx lookup as the decoder.
        let skip_ctx_inc = self.skip_ctx_inc(x0, y0);
        cw.encode_bin(&mut self.cu_skip_flag[skip_ctx_inc], 0);

        // pred_mode_flag = 0 (INTER). Single context.
        cw.encode_bin(&mut self.pred_mode_flag[0], 0);

        // part_mode bin 0 = 1 (2Nx2N). No further bins for 2Nx2N inter.
        cw.encode_bin(&mut self.part_mode[0], 1);

        // ---- prediction_unit: merge_flag=0, (implicit ref_idx=0), MVD, mvp_flag=0.
        cw.encode_bin(&mut self.merge_flag[0], 0);
        // num_ref_idx_l0_active_minus1 == 0 → ref_idx_l0 is NOT coded.

        // Integer-pel SAD search in luma over ±ME_RANGE around (x0, y0).
        // MV is expressed in 1/4-pel units (shift left by 2 for transmission).
        let (mv_int_x, mv_int_y) = self.estimate_mv_int(src, x0, y0);
        let mv_qp_x = mv_int_x * 4;
        let mv_qp_y = mv_int_y * 4;

        // AMVP predictor. With a single candidate (mvp_flag=0) we take the
        // first entry of build_amvp_list on our local mv_grid. For the
        // first CU in the picture this is always (0, 0).
        let (mvp_x, mvp_y) = self.amvp_predictor(x0, y0, CTU_SIZE, CTU_SIZE);
        let mvd_x = mv_qp_x - mvp_x;
        let mvd_y = mv_qp_y - mvp_y;

        self.encode_mvd(cw, mvd_x, mvd_y);
        cw.encode_bin(&mut self.mvp_lx_flag[0], 0);

        // Publish the selected MV on the 4×4 grid for subsequent neighbours.
        self.store_mv(x0, y0, CTU_SIZE, (mv_qp_x, mv_qp_y));

        // ---- motion-compensate + residual.
        let (luma_pred, cb_pred, cr_pred) = self.motion_compensate(x0, y0, mv_int_x, mv_int_y);
        let log2_tb = 4u32;
        let c_log2 = 3u32;
        let (luma_levels, luma_rec) = self.process_inter_luma(src, x0, y0, &luma_pred);
        let cx = x0 / 2;
        let cy = y0 / 2;
        let (cb_levels, cb_rec) = self.process_inter_chroma(src, cx, cy, &cb_pred, false);
        let (cr_levels, cr_rec) = self.process_inter_chroma(src, cx, cy, &cr_pred, true);

        let cbf_luma = luma_levels.iter().any(|&l| l != 0);
        let cbf_cb = cb_levels.iter().any(|&l| l != 0);
        let cbf_cr = cr_levels.iter().any(|&l| l != 0);
        let any_residual = cbf_luma || cbf_cb || cbf_cr;

        // rqt_root_cbf
        cw.encode_bin(&mut self.rqt_root_cbf[0], any_residual as u32);

        if any_residual {
            // transform_tree_inter at tr_depth=0 with the encoder's SPS
            // (`max_transform_hierarchy_depth_inter == 0`):
            // split_transform_flag is NOT signalled (spec-exact gate in
            // `transform_tree_inter_inner`), so split is inferred 0.
            // cbf_cb / cbf_cr (ctx_inc = 0). cbf_luma is inferred=1 when
            // tr_depth==0 && cbf_cb==0 && cbf_cr==0, otherwise emitted with
            // ctx_inc = 1 (tr_depth==0).
            let _ = log2_tb;
            cw.encode_bin(&mut self.residual.cbf_cb_cr[0], cbf_cb as u32);
            cw.encode_bin(&mut self.residual.cbf_cb_cr[0], cbf_cr as u32);
            let cbf_luma_inferred = !cbf_cb && !cbf_cr; // inferred = 1
            if !cbf_luma_inferred {
                cw.encode_bin(&mut self.residual.cbf_luma[1], cbf_luma as u32);
            }
            // Residual order matches the decoder walk: luma, then Cb, then Cr.
            if cbf_luma {
                encode_residual(cw, &mut self.residual, &luma_levels, log2_tb, true);
            }
            if cbf_cb {
                encode_residual(cw, &mut self.residual, &cb_levels, c_log2, false);
            }
            if cbf_cr {
                encode_residual(cw, &mut self.residual, &cr_levels, c_log2, false);
            }
        }

        // Write reconstructed luma/chroma into rec_* for neighbour CUs.
        self.write_luma_block(x0, y0, CTU_SIZE as usize, &luma_rec);
        self.write_chroma_block(cx, cy, 1 << c_log2, &cb_rec, false);
        self.write_chroma_block(cx, cy, 1 << c_log2, &cr_rec, true);
    }

    /// §9.3.4.2.2 cu_skip_flag ctx_inc: count of `(L.skip + A.skip)` where
    /// L / A are the neighbours' skip flags. The decoder approximates this
    /// by `(L.is_valid_inter + A.is_valid_inter)` since it doesn't track
    /// skip per PB. We mirror that approximation exactly — every emitted
    /// P-slice inter CU counts as "neighbour is valid inter".
    fn skip_ctx_inc(&self, x0: u32, y0: u32) -> usize {
        let left = if x0 == 0 {
            0
        } else {
            let bx = ((x0 - 1) >> 2) as usize;
            let by = (y0 >> 2) as usize;
            if bx < self.grid_w4
                && by < self.grid_h4
                && self.mv_grid[by * self.grid_w4 + bx].is_some()
            {
                1
            } else {
                0
            }
        };
        let above = if y0 == 0 {
            0
        } else {
            let bx = (x0 >> 2) as usize;
            let by = ((y0 - 1) >> 2) as usize;
            if bx < self.grid_w4
                && by < self.grid_h4
                && self.mv_grid[by * self.grid_w4 + bx].is_some()
            {
                1
            } else {
                0
            }
        };
        (left + above).min(2)
    }

    /// Integer-pel motion estimation: SAD search over ±ME_RANGE luma
    /// pixels, clamped so the block stays inside the reference picture.
    /// Steps of 2 luma pel — that way the chroma MV (in 1/8-pel luma units)
    /// stays integer and the encoder's integer-pixel copy matches the
    /// decoder's `chroma_mc` full-pel path bit-exactly.
    fn estimate_mv_int(&self, src: &VideoFrame, x0: u32, y0: u32) -> (i32, i32) {
        let n = CTU_SIZE as i32;
        let src_y = &src.planes[0];
        let pic_w = self.cfg.width as i32;
        let pic_h = self.cfg.height as i32;
        let mut best_sad = u64::MAX;
        let mut best = (0i32, 0i32);
        let mut dy = -ME_RANGE;
        while dy <= ME_RANGE {
            let mut dx = -ME_RANGE;
            while dx <= ME_RANGE {
                let rx = x0 as i32 + dx;
                let ry = y0 as i32 + dy;
                if rx < 0 || ry < 0 || rx + n > pic_w || ry + n > pic_h {
                    dx += 2;
                    continue;
                }
                let mut sad = 0u64;
                for j in 0..n {
                    for i in 0..n {
                        let s = src_y.data
                            [(y0 as i32 + j) as usize * src_y.stride + (x0 as i32 + i) as usize]
                            as i32;
                        let r = self.ref_pic.sample_y(rx + i, ry + j) as i32;
                        sad += (s - r).unsigned_abs() as u64;
                    }
                }
                if sad < best_sad {
                    best_sad = sad;
                    best = (dx, dy);
                }
                dx += 2;
            }
            dy += 2;
        }
        best
    }

    /// Build the one-candidate AMVP predictor used to derive MVD. Matches
    /// `build_amvp_list` in `inter.rs`: the first non-none of A0/A1 at
    /// (x-1, y+h) / (x-1, y+h-1), falling back to B0/B1/B2 at (x+w, y-1) /
    /// (x+w-1, y-1) / (x-1, y-1). No TMVP. No deduplication is needed for
    /// slot 0 alone.
    fn amvp_predictor(&self, x: u32, y: u32, w: u32, h: u32) -> (i32, i32) {
        let fetch = |xi: i32, yi: i32| -> Option<(i32, i32)> {
            if xi < 0 || yi < 0 {
                return None;
            }
            let bx = (xi >> 2) as usize;
            let by = (yi >> 2) as usize;
            if bx >= self.grid_w4 || by >= self.grid_h4 {
                return None;
            }
            self.mv_grid[by * self.grid_w4 + bx]
        };
        let a0 = fetch(x as i32 - 1, y as i32 + h as i32);
        let a1 = fetch(x as i32 - 1, y as i32 + h as i32 - 1);
        if let Some(mv) = a0.or(a1) {
            return mv;
        }
        let b0 = fetch(x as i32 + w as i32, y as i32 - 1);
        let b1 = fetch(x as i32 + w as i32 - 1, y as i32 - 1);
        let b2 = fetch(x as i32 - 1, y as i32 - 1);
        b0.or(b1).or(b2).unwrap_or((0, 0))
    }

    fn store_mv(&mut self, x: u32, y: u32, size: u32, mv: (i32, i32)) {
        let bx0 = (x >> 2) as usize;
        let by0 = (y >> 2) as usize;
        let n4 = (size >> 2) as usize;
        for dy in 0..n4 {
            for dx in 0..n4 {
                let bx = bx0 + dx;
                let by = by0 + dy;
                if bx < self.grid_w4 && by < self.grid_h4 {
                    self.mv_grid[by * self.grid_w4 + bx] = Some(mv);
                }
            }
        }
    }

    /// CABAC-encode an MVD pair (§9.3.4.2.6 / §7.3.8.9). Each component
    /// emits: `abs_mvd_greater0_flag` (ctx 0), then if 1 the
    /// `abs_mvd_greater1_flag` (ctx 1), then if >= 2 the EG1-bypass
    /// remainder, then the sign bypass. X is coded fully before Y.
    fn encode_mvd(&mut self, cw: &mut CabacWriter<'_>, mvd_x: i32, mvd_y: i32) {
        // Note: spec emits both gt0 bins first, then both gt1 bins, then
        // remainders+signs. Our decoder `decode_mvd` reads in that order
        // (gt0_x, gt0_y, gt1_x, gt1_y, [rem_x, sign_x], [rem_y, sign_y]).
        let ax = mvd_x.unsigned_abs();
        let ay = mvd_y.unsigned_abs();
        let gt0_x = (ax > 0) as u32;
        let gt0_y = (ay > 0) as u32;
        cw.encode_bin(&mut self.abs_mvd_greater[0], gt0_x);
        cw.encode_bin(&mut self.abs_mvd_greater[0], gt0_y);
        let gt1_x = (ax > 1) as u32;
        let gt1_y = (ay > 1) as u32;
        if gt0_x == 1 {
            cw.encode_bin(&mut self.abs_mvd_greater[1], gt1_x);
        }
        if gt0_y == 1 {
            cw.encode_bin(&mut self.abs_mvd_greater[1], gt1_y);
        }
        if gt0_x == 1 {
            if gt1_x == 1 {
                // abs >= 2 → emit abs-2 as EG1.
                encode_eg1(cw, ax - 2);
            }
            cw.encode_bypass((mvd_x < 0) as u32);
        }
        if gt0_y == 1 {
            if gt1_y == 1 {
                encode_eg1(cw, ay - 2);
            }
            cw.encode_bypass((mvd_y < 0) as u32);
        }
    }

    /// Integer-pel motion compensation at `(x0, y0)` using MV in luma-pel
    /// units. Chroma is 4:2:0: chroma MV is `mv / 2`, with a `round-to-
    /// integer` floor that matches `luma_mc` for fx=fy=0 paths.
    fn motion_compensate(
        &self,
        x0: u32,
        y0: u32,
        mv_x: i32,
        mv_y: i32,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let n = CTU_SIZE as usize;
        let cn = n / 2;
        let mut luma = vec![0u8; n * n];
        let mut cb = vec![0u8; cn * cn];
        let mut cr = vec![0u8; cn * cn];
        for j in 0..n {
            for i in 0..n {
                luma[j * n + i] = self
                    .ref_pic
                    .sample_y(x0 as i32 + i as i32 + mv_x, y0 as i32 + j as i32 + mv_y);
            }
        }
        // Chroma MV in 1/4-pel is `mv_luma_qp / 2` = `mv_int * 2` integer
        // chroma pel. At full-pel luma (mv_int pixels) the chroma MV is
        // mv_int / 2 (integer) with a half-pel remainder handled by the
        // chroma filter. We encode integer luma MVs; half-pel chroma
        // components (when mv_int is odd) would go through the 4-tap
        // chroma filter in the decoder. To keep encoder / decoder math
        // trivially aligned for this round we round chroma MV to the
        // nearest integer chroma pel — this introduces a chroma drift of
        // up to half a chroma pixel on odd MVs but keeps the encoder's
        // local reconstruction matching what the decoder produces. (The
        // matching happens because our integer chroma MC is also a direct
        // pixel copy, which is exactly what the decoder's chroma_mc does
        // when fx=fy=0.)
        let cx0 = (x0 / 2) as i32;
        let cy0 = (y0 / 2) as i32;
        let mv_cx = mv_x / 2;
        let mv_cy = mv_y / 2;
        for j in 0..cn {
            for i in 0..cn {
                cb[j * cn + i] =
                    self.ref_pic
                        .sample_c(cx0 + i as i32 + mv_cx, cy0 + j as i32 + mv_cy, false);
                cr[j * cn + i] =
                    self.ref_pic
                        .sample_c(cx0 + i as i32 + mv_cx, cy0 + j as i32 + mv_cy, true);
            }
        }
        (luma, cb, cr)
    }

    /// Residual pipeline for an inter luma TB: residual = src - pred,
    /// DCT-II forward, quant, reconstruct via dequant + inverse DCT.
    fn process_inter_luma(
        &self,
        src: &VideoFrame,
        x0: u32,
        y0: u32,
        pred: &[u8],
    ) -> (Vec<i32>, Vec<u8>) {
        let n = CTU_SIZE as usize;
        let src_y = &src.planes[0];
        let mut residual = vec![0i32; n * n];
        for dy in 0..n {
            for dx in 0..n {
                let sx = x0 as usize + dx;
                let sy = y0 as usize + dy;
                let s = src_y.data[sy * src_y.stride + sx] as i32;
                let p = pred[dy * n + dx] as i32;
                residual[dy * n + dx] = s - p;
            }
        }
        let log2_tb = 4u32;
        let mut coeffs = vec![0i32; n * n];
        forward_transform_2d(&residual, &mut coeffs, log2_tb, false, 8);
        let mut levels = vec![0i32; n * n];
        quantize_flat(&coeffs, &mut levels, SLICE_QP_Y, log2_tb, 8);

        let mut deq = vec![0i32; n * n];
        dequantize_flat(&levels, &mut deq, SLICE_QP_Y, log2_tb, 8);
        let mut res_rec = vec![0i32; n * n];
        inverse_transform_2d(&deq, &mut res_rec, log2_tb, false, 8);
        let mut rec = vec![0u8; n * n];
        for i in 0..n * n {
            let v = pred[i] as i32 + res_rec[i];
            rec[i] = v.clamp(0, 255) as u8;
        }
        (levels, rec)
    }

    fn process_inter_chroma(
        &self,
        src: &VideoFrame,
        cx: u32,
        cy: u32,
        pred: &[u8],
        is_cr: bool,
    ) -> (Vec<i32>, Vec<u8>) {
        let n = (CTU_SIZE / 2) as usize;
        let plane = if is_cr {
            &src.planes[2]
        } else {
            &src.planes[1]
        };
        let mut residual = vec![0i32; n * n];
        for dy in 0..n {
            for dx in 0..n {
                let sx = cx as usize + dx;
                let sy = cy as usize + dy;
                let s = plane.data[sy * plane.stride + sx] as i32;
                let p = pred[dy * n + dx] as i32;
                residual[dy * n + dx] = s - p;
            }
        }
        let log2_tb = 3u32;
        let qp = SLICE_QP_Y; // matches chroma_qp_for_slice() for QP=26 with zero offsets
        let mut coeffs = vec![0i32; n * n];
        forward_transform_2d(&residual, &mut coeffs, log2_tb, false, 8);
        let mut levels = vec![0i32; n * n];
        quantize_flat(&coeffs, &mut levels, qp, log2_tb, 8);

        let mut deq = vec![0i32; n * n];
        dequantize_flat(&levels, &mut deq, qp, log2_tb, 8);
        let mut res_rec = vec![0i32; n * n];
        inverse_transform_2d(&deq, &mut res_rec, log2_tb, false, 8);
        let mut rec = vec![0u8; n * n];
        for i in 0..n * n {
            let v = pred[i] as i32 + res_rec[i];
            rec[i] = v.clamp(0, 255) as u8;
        }
        (levels, rec)
    }

    fn write_luma_block(&mut self, x: u32, y: u32, n: usize, block: &[u8]) {
        let x0 = x as usize;
        let y0 = y as usize;
        for dy in 0..n {
            for dx in 0..n {
                let xx = x0 + dx;
                let yy = y0 + dy;
                if xx < self.cfg.width as usize && yy < self.cfg.height as usize {
                    self.rec_y[yy * self.y_stride + xx] = block[dy * n + dx];
                }
            }
        }
    }

    fn write_chroma_block(&mut self, x: u32, y: u32, n: usize, block: &[u8], is_cr: bool) {
        let x0 = x as usize;
        let y0 = y as usize;
        let cw = self.cfg.width as usize / 2;
        let ch = self.cfg.height as usize / 2;
        let plane = if is_cr {
            &mut self.rec_cr
        } else {
            &mut self.rec_cb
        };
        for dy in 0..n {
            for dx in 0..n {
                let xx = x0 + dx;
                let yy = y0 + dy;
                if xx < cw && yy < ch {
                    plane[yy * self.c_stride + xx] = block[dy * n + dx];
                }
            }
        }
    }
}

/// First-order Exp-Golomb bypass encode (§9.3.4.4.3 / §9.3.4.3.4 in reverse).
/// Encoder-side inverse of `decode_eg1`: for a value `v`, k is the smallest
/// integer such that `v + 2 < 2^(k+1)`; emit `k - 1` bypass `1`s, then a `0`
/// terminator, then `k` bits of `(v + 2) - 2^k` MSB-first.
fn encode_eg1(cw: &mut CabacWriter<'_>, v: u32) {
    let value = v + 2; // Matches `decode_eg1`: suf + (1 << k) - 2 == v ⇒ suf = value - (1 << k).
    let k = 32 - value.leading_zeros() - 1; // ≥ 1 since value ≥ 2.
                                            // Prefix: k-1 bypass 1s, then 0. The decoder reads prefix until a 0 is
                                            // seen and sets k = 1 + (number_of_1s_read).
    for _ in 0..(k - 1) {
        cw.encode_bypass(1);
    }
    cw.encode_bypass(0);
    let suf_bits = k;
    let suf = value - (1u32 << k);
    for bit in (0..suf_bits).rev() {
        cw.encode_bypass((suf >> bit) & 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::{init_context, CabacEngine};
    use crate::encoder::bit_writer::BitWriter;

    #[test]
    fn eg1_encode_decode_roundtrip() {
        for v in [0u32, 1, 2, 3, 5, 10, 100, 1000, 65535] {
            let mut bw = BitWriter::new();
            let mut cw = CabacWriter::new(&mut bw);
            encode_eg1(&mut cw, v);
            cw.encode_terminate(1);
            cw.encode_flush();
            let bytes = bw.finish();
            let mut eng = CabacEngine::new(&bytes, 0);
            // Replicate `decode_eg1` inline here — avoids the ctu.rs visibility
            // issue (decode_eg1 is private to that module).
            let mut k = 1u32;
            while eng.decode_bypass() == 1 {
                k += 1;
                if k > 32 {
                    panic!("eg1 prefix overflow on v={v}");
                }
            }
            let mut suf = 0u32;
            for _ in 0..k {
                suf = (suf << 1) | eng.decode_bypass();
            }
            let decoded = suf + (1u32 << k) - 2;
            assert_eq!(decoded, v, "eg1 mismatch for v={v}");
        }
    }

    #[test]
    fn mvd_encode_matches_decoder_pattern() {
        // Encode a few MVD pairs, then decode them with the CabacEngine
        // following the sequence `decode_mvd` in ctu.rs expects.
        let init = 154u8;
        let qp = 26i32;
        let cases = [(0i32, 0i32), (3, 0), (0, -2), (5, -5), (100, -200)];
        for (mvd_x, mvd_y) in cases {
            let mut ctx_gt_enc = [init_context(init, qp), init_context(init, qp)];
            let mut ctx_gt_dec = [init_context(init, qp), init_context(init, qp)];

            let mut bw = BitWriter::new();
            let mut cw = CabacWriter::new(&mut bw);
            // Replicate `encode_mvd` without constructing a full PEncoderState.
            let ax = mvd_x.unsigned_abs();
            let ay = mvd_y.unsigned_abs();
            let gt0_x = (ax > 0) as u32;
            let gt0_y = (ay > 0) as u32;
            cw.encode_bin(&mut ctx_gt_enc[0], gt0_x);
            cw.encode_bin(&mut ctx_gt_enc[0], gt0_y);
            let gt1_x = (ax > 1) as u32;
            let gt1_y = (ay > 1) as u32;
            if gt0_x == 1 {
                cw.encode_bin(&mut ctx_gt_enc[1], gt1_x);
            }
            if gt0_y == 1 {
                cw.encode_bin(&mut ctx_gt_enc[1], gt1_y);
            }
            if gt0_x == 1 {
                if gt1_x == 1 {
                    encode_eg1(&mut cw, ax - 2);
                }
                cw.encode_bypass((mvd_x < 0) as u32);
            }
            if gt0_y == 1 {
                if gt1_y == 1 {
                    encode_eg1(&mut cw, ay - 2);
                }
                cw.encode_bypass((mvd_y < 0) as u32);
            }
            cw.encode_terminate(1);
            cw.encode_flush();
            let bytes = bw.finish();

            let mut eng = CabacEngine::new(&bytes, 0);
            let gt0_x_d = eng.decode_bin(&mut ctx_gt_dec[0]);
            let gt0_y_d = eng.decode_bin(&mut ctx_gt_dec[0]);
            let mut gt1_x_d = 0;
            let mut gt1_y_d = 0;
            if gt0_x_d == 1 {
                gt1_x_d = eng.decode_bin(&mut ctx_gt_dec[1]);
            }
            if gt0_y_d == 1 {
                gt1_y_d = eng.decode_bin(&mut ctx_gt_dec[1]);
            }
            let mut abs_x = 0i32;
            if gt0_x_d == 1 {
                abs_x = 1 + gt1_x_d as i32;
                if gt1_x_d == 1 {
                    // Inline eg1 decode.
                    let mut k = 1u32;
                    while eng.decode_bypass() == 1 {
                        k += 1;
                    }
                    let mut suf = 0u32;
                    for _ in 0..k {
                        suf = (suf << 1) | eng.decode_bypass();
                    }
                    abs_x += (suf + (1u32 << k) - 2) as i32;
                }
                if eng.decode_bypass() == 1 {
                    abs_x = -abs_x;
                }
            }
            let mut abs_y = 0i32;
            if gt0_y_d == 1 {
                abs_y = 1 + gt1_y_d as i32;
                if gt1_y_d == 1 {
                    let mut k = 1u32;
                    while eng.decode_bypass() == 1 {
                        k += 1;
                    }
                    let mut suf = 0u32;
                    for _ in 0..k {
                        suf = (suf << 1) | eng.decode_bypass();
                    }
                    abs_y += (suf + (1u32 << k) - 2) as i32;
                }
                if eng.decode_bypass() == 1 {
                    abs_y = -abs_y;
                }
            }
            assert_eq!((abs_x, abs_y), (mvd_x, mvd_y), "mvd mismatch");
        }
    }
}
