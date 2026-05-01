//! B-slice header + CABAC slice-data emission (round 22).
//!
//! Scope:
//!
//! * Single-segment B (TrailR) slice referencing two pictures: one
//!   earlier-POC L0 reference and one later-POC L1 reference. Both list
//!   sizes are 1 (`num_ref_idx_l{0,1}_active_minus1 = 0`) so `ref_idx_lX`
//!   is implicit per §7.3.8.6.
//! * Per-CU pipeline: INTER 2Nx2N, `cu_skip_flag = 0`, `pred_mode_flag = 0`,
//!   `merge_flag = 0`. Encoder picks the best of L0-only / L1-only / Bi
//!   for every CU by comparing reconstructed-MC SAD against the source
//!   (rate-distortion approximation: pick whichever predictor minimises
//!   absolute-difference; the residual / quant pipeline is identical
//!   across all three so the predictor pick directly controls quality).
//! * `inter_pred_idc` (§9.3.4.2.2 Table 9-32): bin 0 ctxInc =
//!   `CtDepth[x0][y0]` for `(nPbW + nPbH) != 12`. Our CU is 16×16 so
//!   nPbW + nPbH = 32 and CtDepth = 0 → bin0_ctx = 0. We never produce
//!   small (4×8 / 8×4) PBs in this writer so the small-PU path is unused.
//! * Both AMVP candidates per list collapse to a single predictor
//!   (`mvp_lx_flag = 0`); we mirror the P-slice writer's spatial-only
//!   AMVP that walks A0/A1 then B0/B1/B2 on the local 4×4 mv grid.
//! * Integer-pel motion estimation with ±8 pel luma SAD search per list,
//!   step 2 so chroma MV stays integer (matches the decoder's full-pel
//!   chroma_mc path bit-exactly).
//! * `mvd_l1_zero_flag = 0` — both L0 and L1 MVDs are written explicitly
//!   so the decoder reproduces our two MVs without an `MvdL1` zero-out.
//! * SPS / PPS reuse the round-21 envelope: no TMVP, no SAO, no
//!   deblocking, no weighted bi-pred. The new `num_negative_pics = 1`,
//!   `num_positive_pics = 1` RPS is signalled inline per slice
//!   (`short_term_ref_pic_set_sps_flag = 0`).
//!
//! Out of scope for r22 (deferred to r23+): merge encode, B_Skip encode,
//! AMP / Nx2N / 2NxN partitions, MVD-l1-zero optimisation,
//! `weighted_bipred`.

use crate::cabac::{
    init_row, CtxState, InitType, ABS_MVD_GREATER_FLAGS_INIT_VALUES, CU_SKIP_FLAG_INIT_VALUES,
    INTER_PRED_IDC_INIT_VALUES, MERGE_FLAG_INIT_VALUES, MVP_LX_FLAG_INIT_VALUES,
    PART_MODE_INIT_VALUES, PRED_MODE_FLAG_INIT_VALUES, RQT_ROOT_CBF_INIT_VALUES,
};
use crate::encoder::bit_writer::{write_rbsp_trailing_bits, BitWriter};
use crate::encoder::cabac_writer::CabacWriter;
use crate::encoder::nal_writer::build_annex_b_nal;
use crate::encoder::p_slice_writer::ReferenceFrame;
use crate::encoder::params::EncoderConfig;
use crate::encoder::residual_writer::{encode_residual, ResidualCtx};
use crate::nal::{NalHeader, NalUnitType};
use crate::transform::{
    dequantize_flat, forward_transform_2d, inverse_transform_2d, quantize_flat,
};
use oxideav_core::VideoFrame;

/// CTU size = 16. minCU = 16. 2Nx2N inter CU ⇒ one 16×16 luma PB + TB.
const CTU_SIZE: u32 = 16;
/// Slice QP — matches the I/P-slice emitter so rate/quality stay aligned.
const SLICE_QP_Y: i32 = 26;
/// Integer search window (luma pixels). ±`ME_RANGE` both directions.
const ME_RANGE: i32 = 8;

/// Build Annex B bytes for a B (TrailR) slice NAL.
///
/// `delta_l0` and `delta_l1` are the per-list POC deltas relative to the
/// current frame: `delta_l0 < 0` (earlier ref), `delta_l1 > 0` (later ref).
pub fn build_b_slice_with_reconstruction(
    cfg: &EncoderConfig,
    frame: &VideoFrame,
    poc_lsb: u32,
    delta_l0: i32,
    delta_l1: i32,
    ref_l0: &ReferenceFrame,
    ref_l1: &ReferenceFrame,
) -> (Vec<u8>, ReferenceFrame) {
    debug_assert!(delta_l0 < 0, "B-slice L0 ref must have negative POC delta");
    debug_assert!(delta_l1 > 0, "B-slice L1 ref must have positive POC delta");
    let mut bw = BitWriter::new();

    // ---- slice_segment_header() -------------------------------------
    bw.write_u1(1); // first_slice_segment_in_pic_flag
    bw.write_ue(0); // slice_pic_parameter_set_id
    bw.write_ue(0); // slice_type = B (per §7.4.7.1: 0=B, 1=P, 2=I)

    // POC LSB — log2_max_poc_lsb_minus4 = 4 → 8 bits.
    bw.write_bits(poc_lsb & 0xFF, 8);
    bw.write_u1(0); // short_term_ref_pic_set_sps_flag = 0 → inline RPS.

    // Inline st_ref_pic_set(0) — single negative entry + single positive
    // entry. With st_rps_idx == 0 the inter_ref_pic_set_prediction_flag
    // is NOT signalled (§7.3.7).
    bw.write_ue(1); // num_negative_pics = 1
    bw.write_ue(1); // num_positive_pics = 1
                    // negative entry 0: delta_poc_s0_minus1 = -delta_l0 - 1, used = 1.
    bw.write_ue((-delta_l0 - 1) as u32);
    bw.write_u1(1);
    // positive entry 0: delta_poc_s1_minus1 = delta_l1 - 1, used = 1.
    bw.write_ue((delta_l1 - 1) as u32);
    bw.write_u1(1);

    // long_term_ref_pics_present = 0, sps_temporal_mvp = 0 → skipped.
    // SAO disabled → no slice_sao_*_flag.
    // P/B branch:
    bw.write_u1(0); // num_ref_idx_active_override_flag = 0 → use PPS defaults
                    // (both = 0 → num_ref_idx_lX_active_minus1 = 0 → exactly 1
                    // entry per list).
                    // pps.lists_modification_present_flag = 0 → skipped.
    bw.write_u1(0); // mvd_l1_zero_flag = 0
                    // pps.cabac_init_present_flag = 0 → skipped.
                    // slice_temporal_mvp_enabled_flag = 0 → no collocated.
                    // pps.weighted_bipred_flag = 0 → no weighted_pred table.
    bw.write_ue(4); // five_minus_max_num_merge_cand → max_num_merge_cand = 1
    bw.write_se(0); // slice_qp_delta = 0.
    write_rbsp_trailing_bits(&mut bw);

    // ---- slice_data() ------------------------------------------------
    let mut enc = BEncoderState::new(cfg, ref_l0.clone(), ref_l1.clone());
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
    let recon =
        ReferenceFrame::from_planes(cfg.width, cfg.height, enc.rec_y, enc.rec_cb, enc.rec_cr);
    let nal = build_annex_b_nal(NalHeader::for_type(NalUnitType::TrailR), &bw.finish());
    (nal, recon)
}

/// Per-slice B-frame CABAC encoding state.
struct BEncoderState {
    cfg: EncoderConfig,
    ref_l0: ReferenceFrame,
    ref_l1: ReferenceFrame,
    rec_y: Vec<u8>,
    rec_cb: Vec<u8>,
    rec_cr: Vec<u8>,
    y_stride: usize,
    c_stride: usize,
    /// Per-4×4 block: Some((mv_x, mv_y)) for each list when this PB used
    /// that list, None otherwise. Used by AMVP spatial-neighbour fetch.
    mv_grid_l0: Vec<Option<(i32, i32)>>,
    mv_grid_l1: Vec<Option<(i32, i32)>>,
    grid_w4: usize,
    grid_h4: usize,

    // CABAC contexts (B slices use InitType::Pb when cabac_init_flag=0).
    cu_skip_flag: [CtxState; 3],
    pred_mode_flag: [CtxState; 1],
    part_mode: [CtxState; 4],
    merge_flag: [CtxState; 1],
    mvp_lx_flag: [CtxState; 1],
    abs_mvd_greater: [CtxState; 2],
    rqt_root_cbf: [CtxState; 1],
    inter_pred_idc: [CtxState; 5],
    residual: ResidualCtx,
}

impl BEncoderState {
    fn new(cfg: &EncoderConfig, ref_l0: ReferenceFrame, ref_l1: ReferenceFrame) -> Self {
        let w = cfg.width as usize;
        let h = cfg.height as usize;
        let cw = w / 2;
        let ch = h / 2;
        // B slices with cabac_init_flag = 0 use Pb (per InitType::for_slice).
        let it = InitType::Pb;
        let grid_w4 = w / 4;
        let grid_h4 = h / 4;
        Self {
            cfg: *cfg,
            ref_l0,
            ref_l1,
            rec_y: vec![128u8; w * h],
            rec_cb: vec![128u8; cw * ch],
            rec_cr: vec![128u8; cw * ch],
            y_stride: w,
            c_stride: cw,
            mv_grid_l0: vec![None; grid_w4 * grid_h4],
            mv_grid_l1: vec![None; grid_w4 * grid_h4],
            grid_w4,
            grid_h4,
            cu_skip_flag: init_row(&CU_SKIP_FLAG_INIT_VALUES, it, SLICE_QP_Y),
            pred_mode_flag: init_row(&PRED_MODE_FLAG_INIT_VALUES, it, SLICE_QP_Y),
            part_mode: init_row(&PART_MODE_INIT_VALUES, it, SLICE_QP_Y),
            merge_flag: init_row(&MERGE_FLAG_INIT_VALUES, it, SLICE_QP_Y),
            mvp_lx_flag: init_row(&MVP_LX_FLAG_INIT_VALUES, it, SLICE_QP_Y),
            abs_mvd_greater: init_row(&ABS_MVD_GREATER_FLAGS_INIT_VALUES, it, SLICE_QP_Y),
            rqt_root_cbf: init_row(&RQT_ROOT_CBF_INIT_VALUES, it, SLICE_QP_Y),
            inter_pred_idc: init_row(&INTER_PRED_IDC_INIT_VALUES, it, SLICE_QP_Y),
            residual: ResidualCtx::with_init_type(SLICE_QP_Y, it),
        }
    }

    fn encode_ctu(&mut self, cw: &mut CabacWriter<'_>, src: &VideoFrame, x0: u32, y0: u32) {
        // cu_skip_flag = 0, ctx_inc = neighbours-with-skip count (we never
        // emit skip so the count is always derived from "neighbour was an
        // inter PB" — same approximation P-slice writer uses).
        let skip_ctx_inc = self.skip_ctx_inc(x0, y0);
        cw.encode_bin(&mut self.cu_skip_flag[skip_ctx_inc], 0);

        // pred_mode_flag = 0 (INTER).
        cw.encode_bin(&mut self.pred_mode_flag[0], 0);

        // part_mode bin 0 = 1 (2Nx2N). No more bins for 2Nx2N inter.
        cw.encode_bin(&mut self.part_mode[0], 1);

        // merge_flag = 0.
        cw.encode_bin(&mut self.merge_flag[0], 0);

        // ---- candidate MVs per list -----------------------------------
        let (mv_l0_int, sad_l0) = self.estimate_mv_int(src, x0, y0, /* l1 */ false);
        let (mv_l1_int, sad_l1) = self.estimate_mv_int(src, x0, y0, /* l1 */ true);

        // Bi-pred candidate: average the two list predictors and SAD.
        let (luma_pred_l0, cb_pred_l0, cr_pred_l0) =
            motion_compensate(&self.ref_l0, x0, y0, mv_l0_int.0, mv_l0_int.1);
        let (luma_pred_l1, cb_pred_l1, cr_pred_l1) =
            motion_compensate(&self.ref_l1, x0, y0, mv_l1_int.0, mv_l1_int.1);
        let mut luma_pred_bi = vec![0u8; luma_pred_l0.len()];
        for i in 0..luma_pred_bi.len() {
            // Round-half-up average matches §8.5.3.3.3.1 default-bipred
            // weighting `(P0 + P1 + 1) >> 1` (without weighted-pred table).
            luma_pred_bi[i] = ((luma_pred_l0[i] as u32 + luma_pred_l1[i] as u32 + 1) >> 1) as u8;
        }
        let mut cb_pred_bi = vec![0u8; cb_pred_l0.len()];
        for i in 0..cb_pred_bi.len() {
            cb_pred_bi[i] = ((cb_pred_l0[i] as u32 + cb_pred_l1[i] as u32 + 1) >> 1) as u8;
        }
        let mut cr_pred_bi = vec![0u8; cr_pred_l0.len()];
        for i in 0..cr_pred_bi.len() {
            cr_pred_bi[i] = ((cr_pred_l0[i] as u32 + cr_pred_l1[i] as u32 + 1) >> 1) as u8;
        }
        let sad_bi = sad_luma(src, x0, y0, &luma_pred_bi, CTU_SIZE);

        // Pick predictor with minimum luma SAD.
        let (use_l0, use_l1, luma_pred, cb_pred, cr_pred) = if sad_bi <= sad_l0 && sad_bi <= sad_l1
        {
            (true, true, luma_pred_bi, cb_pred_bi, cr_pred_bi)
        } else if sad_l0 <= sad_l1 {
            (true, false, luma_pred_l0, cb_pred_l0, cr_pred_l0)
        } else {
            (false, true, luma_pred_l1, cb_pred_l1, cr_pred_l1)
        };

        // ---- inter_pred_idc (B-slice only; bin0 ctxInc = CtDepth = 0
        //      since CU is at CTB depth 0) ---------------------------------
        // Bin 0: 1 = PRED_BI, 0 = uni (then bin 1 selects L0/L1).
        // Bin 1 ctxInc = 4. (n+w) != 12 so we always take the
        // CtDepth-indexed bin0 path; CtDepth for our 16×16 CU at 16×16 CTB
        // is 0 → bin0_ctx = 0.
        let bin0_ctx = 0usize;
        if use_l0 && use_l1 {
            cw.encode_bin(&mut self.inter_pred_idc[bin0_ctx], 1);
        } else {
            cw.encode_bin(&mut self.inter_pred_idc[bin0_ctx], 0);
            cw.encode_bin(&mut self.inter_pred_idc[4], if use_l1 { 1 } else { 0 });
        }

        // ---- per-list AMVP / MVD ---------------------------------------
        // num_ref_idx_lX_active_minus1 == 0 → ref_idx_lX is NOT coded.
        if use_l0 {
            let mv_qp = (mv_l0_int.0 * 4, mv_l0_int.1 * 4);
            let mvp = self.amvp_predictor(x0, y0, CTU_SIZE, CTU_SIZE, /* l1 */ false);
            self.encode_mvd(cw, mv_qp.0 - mvp.0, mv_qp.1 - mvp.1);
            cw.encode_bin(&mut self.mvp_lx_flag[0], 0);
        }
        if use_l1 {
            let mv_qp = (mv_l1_int.0 * 4, mv_l1_int.1 * 4);
            let mvp = self.amvp_predictor(x0, y0, CTU_SIZE, CTU_SIZE, /* l1 */ true);
            // mvd_l1_zero_flag = 0 → emit the L1 MVD normally.
            self.encode_mvd(cw, mv_qp.0 - mvp.0, mv_qp.1 - mvp.1);
            cw.encode_bin(&mut self.mvp_lx_flag[0], 0);
        }

        // Publish chosen MVs on the per-list 4×4 grid for AMVP neighbour
        // lookups by subsequent CUs in raster scan.
        if use_l0 {
            self.store_mv(x0, y0, CTU_SIZE, (mv_l0_int.0 * 4, mv_l0_int.1 * 4), false);
        }
        if use_l1 {
            self.store_mv(x0, y0, CTU_SIZE, (mv_l1_int.0 * 4, mv_l1_int.1 * 4), true);
        }

        // ---- residual --------------------------------------------------
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
            cw.encode_bin(&mut self.residual.cbf_cb_cr[0], cbf_cb as u32);
            cw.encode_bin(&mut self.residual.cbf_cb_cr[0], cbf_cr as u32);
            let cbf_luma_inferred = !cbf_cb && !cbf_cr; // inferred = 1
            if !cbf_luma_inferred {
                cw.encode_bin(&mut self.residual.cbf_luma[1], cbf_luma as u32);
            }
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

        // Write reconstructed samples for the in-picture neighbour state.
        self.write_luma_block(x0, y0, CTU_SIZE as usize, &luma_rec);
        self.write_chroma_block(cx, cy, 1 << c_log2, &cb_rec, false);
        self.write_chroma_block(cx, cy, 1 << c_log2, &cr_rec, true);
    }

    /// §9.3.4.2.2 cu_skip_flag ctxInc with the spec's `condTermFlagX =
    /// neighbour.is_skip`. Our writer never emits skip CUs so the
    /// neighbour skip flag is always 0 — ctxInc collapses to 0 for every
    /// CU. Mirrors the decoder's r19 fix exactly so the arithmetic stays
    /// in sync.
    fn skip_ctx_inc(&self, _x0: u32, _y0: u32) -> usize {
        0
    }

    /// Integer-pel motion estimation against one of the references.
    fn estimate_mv_int(
        &self,
        src: &VideoFrame,
        x0: u32,
        y0: u32,
        is_l1: bool,
    ) -> ((i32, i32), u64) {
        let n = CTU_SIZE as i32;
        let src_y = &src.planes[0];
        let pic_w = self.cfg.width as i32;
        let pic_h = self.cfg.height as i32;
        let r = if is_l1 { &self.ref_l1 } else { &self.ref_l0 };
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
                        let rr = r.sample_y(rx + i, ry + j) as i32;
                        sad += (s - rr).unsigned_abs() as u64;
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
        (best, best_sad)
    }

    fn amvp_predictor(&self, x: u32, y: u32, w: u32, h: u32, is_l1: bool) -> (i32, i32) {
        let grid = if is_l1 {
            &self.mv_grid_l1
        } else {
            &self.mv_grid_l0
        };
        let fetch = |xi: i32, yi: i32| -> Option<(i32, i32)> {
            if xi < 0 || yi < 0 {
                return None;
            }
            let bx = (xi >> 2) as usize;
            let by = (yi >> 2) as usize;
            if bx >= self.grid_w4 || by >= self.grid_h4 {
                return None;
            }
            grid[by * self.grid_w4 + bx]
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

    fn store_mv(&mut self, x: u32, y: u32, size: u32, mv: (i32, i32), is_l1: bool) {
        let grid = if is_l1 {
            &mut self.mv_grid_l1
        } else {
            &mut self.mv_grid_l0
        };
        let bx0 = (x >> 2) as usize;
        let by0 = (y >> 2) as usize;
        let n4 = (size >> 2) as usize;
        for dy in 0..n4 {
            for dx in 0..n4 {
                let bx = bx0 + dx;
                let by = by0 + dy;
                if bx < self.grid_w4 && by < self.grid_h4 {
                    grid[by * self.grid_w4 + bx] = Some(mv);
                }
            }
        }
    }

    /// CABAC-encode an MVD pair (§9.3.4.2.6 / §7.3.8.9). Same bit
    /// ordering as the P-slice writer's `encode_mvd`.
    fn encode_mvd(&mut self, cw: &mut CabacWriter<'_>, mvd_x: i32, mvd_y: i32) {
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
        let qp = SLICE_QP_Y;
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

/// Integer-pel motion compensation (free-standing helper so we can run
/// it for L0 / L1 candidates without borrowing the encoder state mutably).
fn motion_compensate(
    r: &ReferenceFrame,
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
            luma[j * n + i] = r.sample_y(x0 as i32 + i as i32 + mv_x, y0 as i32 + j as i32 + mv_y);
        }
    }
    let cx0 = (x0 / 2) as i32;
    let cy0 = (y0 / 2) as i32;
    let mv_cx = mv_x / 2;
    let mv_cy = mv_y / 2;
    for j in 0..cn {
        for i in 0..cn {
            cb[j * cn + i] = r.sample_c(cx0 + i as i32 + mv_cx, cy0 + j as i32 + mv_cy, false);
            cr[j * cn + i] = r.sample_c(cx0 + i as i32 + mv_cx, cy0 + j as i32 + mv_cy, true);
        }
    }
    (luma, cb, cr)
}

/// Luma SAD between source CTU and a prediction buffer.
fn sad_luma(src: &VideoFrame, x0: u32, y0: u32, pred: &[u8], size: u32) -> u64 {
    let n = size as usize;
    let src_y = &src.planes[0];
    let mut sad = 0u64;
    for dy in 0..n {
        for dx in 0..n {
            let s = src_y.data[(y0 as usize + dy) * src_y.stride + x0 as usize + dx] as i32;
            let p = pred[dy * n + dx] as i32;
            sad += (s - p).unsigned_abs() as u64;
        }
    }
    sad
}

/// First-order Exp-Golomb bypass encode — duplicate of the helper in
/// `p_slice_writer.rs` to keep the two writers self-contained.
fn encode_eg1(cw: &mut CabacWriter<'_>, v: u32) {
    let value = v + 2;
    let k = 32 - value.leading_zeros() - 1;
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
