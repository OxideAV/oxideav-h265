//! High-bit-depth (10-bit / 12-bit) B-slice emitter for the 4:2:0 path.
//!
//! Mirrors the structure of [`super::b_slice_writer`] but operates on `u16`
//! samples (packed LE-16 in the `VideoFrame` planes) and threads `bit_depth`
//! + `Qp'Y = SliceQpY + QpBdOffsetY` through the residual pipeline.
//!
//! Scope (round 33):
//! * Per-CU mode: explicit AMVP with L0-only / L1-only / Bi pick by SAD.
//!   No merge / B_Skip — that's deferred to a follow-up round.
//! * `five_minus_max_num_merge_cand = 4` → `max_num_merge_cand = 1`.
//! * Integer-pel ±8 luma SAD search per list; bipred = `(P0+P1+1)>>1`.
//! * CABAC: identical context init / syntax to the 8-bit B-slice writer.
//!
//! Non-reference: the B-frame reconstruction is returned (useful for
//! unit tests) but [`super::hevc_encoder`] currently discards it —
//! B-frames don't feed subsequent anchors.

use crate::cabac::{
    init_row, CtxState, InitType, ABS_MVD_GREATER_FLAGS_INIT_VALUES, CU_SKIP_FLAG_INIT_VALUES,
    INTER_PRED_IDC_INIT_VALUES, MERGE_FLAG_INIT_VALUES, MVP_LX_FLAG_INIT_VALUES,
    PART_MODE_INIT_VALUES, PRED_MODE_FLAG_INIT_VALUES, RQT_ROOT_CBF_INIT_VALUES,
    SPLIT_CU_FLAG_INIT_VALUES,
};
use crate::encoder::bit_writer::{write_rbsp_trailing_bits, BitWriter};
use crate::encoder::cabac_writer::CabacWriter;
use crate::encoder::nal_writer::build_annex_b_nal;
use crate::encoder::p_slice_writer_hbd::{read_hbd_sample, ReferenceFrame16};
use crate::encoder::params::EncoderConfig;
use crate::encoder::residual_writer::{encode_residual, ResidualCtx};
use crate::nal::{NalHeader, NalUnitType};
use crate::transform::{
    dequantize_flat, forward_transform_2d, inverse_transform_2d, quantize_flat,
};
use oxideav_core::VideoFrame;

const CTU_SIZE: u32 = 16;
const SLICE_QP_Y: i32 = 26;
const ME_RANGE: i32 = 8;
/// max_num_merge_cand = 1 (five_minus = 4). No merge candidates used
/// in this round — we keep the value minimal so the slice header is
/// valid but AMVP is the only mode we signal.
const FIVE_MINUS_MAX_MERGE: u32 = 4;

/// Build Annex B bytes for an HBD B (TrailR) slice NAL.
///
/// `delta_l0 < 0`, `delta_l1 > 0` (POC deltas relative to current frame).
/// Returns `(NAL bytes, reconstructed frame)`.
pub fn build_b_slice_hbd(
    cfg: &EncoderConfig,
    frame: &VideoFrame,
    poc_lsb: u32,
    delta_l0: i32,
    delta_l1: i32,
    ref_l0: &ReferenceFrame16,
    ref_l1: &ReferenceFrame16,
) -> (Vec<u8>, ReferenceFrame16) {
    debug_assert!(delta_l0 < 0);
    debug_assert!(delta_l1 > 0);

    let bit_depth = cfg.bit_depth;
    let qp_bd_offset = 6 * (bit_depth as i32 - 8);
    let qp_prime = SLICE_QP_Y + qp_bd_offset;

    let mut bw = BitWriter::new();
    // ---- slice_segment_header() ----------------------------------------
    bw.write_u1(1); // first_slice_segment_in_pic_flag
    bw.write_ue(0); // slice_pic_parameter_set_id
    bw.write_ue(0); // slice_type = B
    bw.write_bits(poc_lsb & 0xFF, 8);
    bw.write_u1(0); // short_term_ref_pic_set_sps_flag = 0 → inline
    bw.write_ue(1); // num_negative_pics = 1
    bw.write_ue(1); // num_positive_pics = 1
    bw.write_ue((-delta_l0 - 1) as u32); // delta_poc_s0_minus1
    bw.write_u1(1); // used_by_curr_pic_s0
    bw.write_ue((delta_l1 - 1) as u32); // delta_poc_s1_minus1
    bw.write_u1(1); // used_by_curr_pic_s1
    bw.write_u1(0); // num_ref_idx_active_override_flag = 0
    bw.write_u1(0); // mvd_l1_zero_flag = 0
    bw.write_ue(FIVE_MINUS_MAX_MERGE); // five_minus_max_num_merge_cand
    bw.write_se(0); // slice_qp_delta = 0
    write_rbsp_trailing_bits(&mut bw);

    // ---- slice_data() --------------------------------------------------
    let mut enc = BHBDState::new(cfg, ref_l0.clone(), ref_l1.clone(), qp_prime);
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

    let recon = ReferenceFrame16::new(cfg.width, cfg.height, enc.rec_y, enc.rec_cb, enc.rec_cr);
    let nal = build_annex_b_nal(NalHeader::for_type(NalUnitType::TrailR), &bw.finish());
    (nal, recon)
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct BHBDState {
    cfg: EncoderConfig,
    ref_l0: ReferenceFrame16,
    ref_l1: ReferenceFrame16,
    qp_prime: i32,
    bit_depth: u32,
    max_sample: i32,
    neutral: u16,
    rec_y: Vec<u16>,
    rec_cb: Vec<u16>,
    rec_cr: Vec<u16>,
    y_stride: usize,
    c_stride: usize,
    mv_grid_l0: Vec<Option<(i32, i32)>>,
    mv_grid_l1: Vec<Option<(i32, i32)>>,
    grid_w4: usize,
    grid_h4: usize,

    // CABAC contexts (B slices with cabac_init_flag=0 use InitType::Pb).
    split_cu_flag: [CtxState; 3],
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

impl BHBDState {
    fn new(
        cfg: &EncoderConfig,
        ref_l0: ReferenceFrame16,
        ref_l1: ReferenceFrame16,
        qp_prime: i32,
    ) -> Self {
        let w = cfg.width as usize;
        let h = cfg.height as usize;
        let cw = w / 2;
        let ch = h / 2;
        let it = InitType::Pb;
        let grid_w4 = w / 4;
        let grid_h4 = h / 4;
        let bit_depth = cfg.bit_depth;
        let neutral = 1u16 << (bit_depth - 1);
        let max_sample = (1i32 << bit_depth) - 1;
        Self {
            cfg: *cfg,
            ref_l0,
            ref_l1,
            qp_prime,
            bit_depth,
            max_sample,
            neutral,
            rec_y: vec![neutral; w * h],
            rec_cb: vec![neutral; cw * ch],
            rec_cr: vec![neutral; cw * ch],
            y_stride: w,
            c_stride: cw,
            mv_grid_l0: vec![None; grid_w4 * grid_h4],
            mv_grid_l1: vec![None; grid_w4 * grid_h4],
            grid_w4,
            grid_h4,
            split_cu_flag: init_row(&SPLIT_CU_FLAG_INIT_VALUES, it, SLICE_QP_Y),
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
        // split_cu_flag = 0.
        cw.encode_bin(&mut self.split_cu_flag[0], 0);
        // cu_skip_flag = 0 (no skip — AMVP only in this round).
        cw.encode_bin(&mut self.cu_skip_flag[0], 0);
        // pred_mode_flag = 0 (INTER).
        cw.encode_bin(&mut self.pred_mode_flag[0], 0);
        // part_mode bin 0 = 1 (2Nx2N).
        cw.encode_bin(&mut self.part_mode[0], 1);
        // merge_flag = 0 (explicit AMVP).
        cw.encode_bin(&mut self.merge_flag[0], 0);

        // Motion estimation: per-list SAD search.
        let (mv_l0_int, sad_l0) = self.estimate_mv(src, x0, y0, false);
        let (mv_l1_int, sad_l1) = self.estimate_mv(src, x0, y0, true);
        let (pred_l0_y, pred_l0_cb, pred_l0_cr) = self.mc(x0, y0, mv_l0_int.0, mv_l0_int.1, false);
        let (pred_l1_y, pred_l1_cb, pred_l1_cr) = self.mc(x0, y0, mv_l1_int.0, mv_l1_int.1, true);
        let (pred_bi_y, pred_bi_cb, pred_bi_cr) = bipred_u16(
            &pred_l0_y,
            &pred_l0_cb,
            &pred_l0_cr,
            &pred_l1_y,
            &pred_l1_cb,
            &pred_l1_cr,
        );
        let sad_bi = sad_u16(src, x0, y0, &pred_bi_y, CTU_SIZE);

        let (use_l0, use_l1, pred_y, pred_cb, pred_cr) = if sad_bi <= sad_l0 && sad_bi <= sad_l1 {
            (true, true, pred_bi_y, pred_bi_cb, pred_bi_cr)
        } else if sad_l0 <= sad_l1 {
            (true, false, pred_l0_y, pred_l0_cb, pred_l0_cr)
        } else {
            (false, true, pred_l1_y, pred_l1_cb, pred_l1_cr)
        };

        // inter_pred_idc: ctxInc = CtDepth = 0 (16×16 CB at root depth).
        // Bin 0: 0 = uni (L0 or L1 only), 1 = bi.
        // Bin 1 (only if uni): 0 = L0, 1 = L1.
        if use_l0 && use_l1 {
            cw.encode_bin(&mut self.inter_pred_idc[0], 1); // bi-pred
        } else {
            cw.encode_bin(&mut self.inter_pred_idc[0], 0); // uni-pred
            cw.encode_bin(&mut self.inter_pred_idc[4], if use_l0 { 0 } else { 1 });
        }

        // L0 AMVP
        if use_l0 {
            let mv_qp_l0 = (mv_l0_int.0 * 4, mv_l0_int.1 * 4);
            let mvp_l0 = self.amvp_predictor(x0, y0, false);
            let mvd_l0 = (mv_qp_l0.0 - mvp_l0.0, mv_qp_l0.1 - mvp_l0.1);
            self.encode_mvd(cw, mvd_l0.0, mvd_l0.1);
            cw.encode_bin(&mut self.mvp_lx_flag[0], 0);
            self.store_mv(x0, y0, CTU_SIZE, mv_qp_l0, false);
        }
        // L1 AMVP
        if use_l1 {
            let mv_qp_l1 = (mv_l1_int.0 * 4, mv_l1_int.1 * 4);
            let mvp_l1 = self.amvp_predictor(x0, y0, true);
            let mvd_l1 = (mv_qp_l1.0 - mvp_l1.0, mv_qp_l1.1 - mvp_l1.1);
            self.encode_mvd(cw, mvd_l1.0, mvd_l1.1);
            cw.encode_bin(&mut self.mvp_lx_flag[0], 0);
            self.store_mv(x0, y0, CTU_SIZE, mv_qp_l1, true);
        }

        // Residual
        let log2_tb = 4u32;
        let c_log2 = 3u32;
        let (luma_levels, luma_rec) = self.process_luma(src, x0, y0, &pred_y, log2_tb);
        let cx = x0 / 2;
        let cy = y0 / 2;
        let (cb_levels, cb_rec) = self.process_chroma(src, cx, cy, &pred_cb, false, c_log2);
        let (cr_levels, cr_rec) = self.process_chroma(src, cx, cy, &pred_cr, true, c_log2);

        let cbf_luma = luma_levels.iter().any(|&l| l != 0);
        let cbf_cb = cb_levels.iter().any(|&l| l != 0);
        let cbf_cr = cr_levels.iter().any(|&l| l != 0);
        let any_residual = cbf_luma || cbf_cb || cbf_cr;

        cw.encode_bin(&mut self.rqt_root_cbf[0], any_residual as u32);
        if any_residual {
            cw.encode_bin(&mut self.residual.cbf_cb_cr[0], cbf_cb as u32);
            cw.encode_bin(&mut self.residual.cbf_cb_cr[0], cbf_cr as u32);
            let cbf_luma_inferred = !cbf_cb && !cbf_cr;
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

        self.write_luma_block(x0, y0, CTU_SIZE as usize, &luma_rec);
        self.write_chroma_block(cx, cy, 1 << c_log2, &cb_rec, false);
        self.write_chroma_block(cx, cy, 1 << c_log2, &cr_rec, true);
    }

    fn estimate_mv(&self, src: &VideoFrame, x0: u32, y0: u32, use_l1: bool) -> ((i32, i32), u64) {
        let n = CTU_SIZE as i32;
        let src_y = &src.planes[0];
        let pic_w = self.cfg.width as i32;
        let pic_h = self.cfg.height as i32;
        let ref_pic = if use_l1 { &self.ref_l1 } else { &self.ref_l0 };
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
                        let s = read_hbd_sample(
                            &src_y.data,
                            src_y.stride,
                            (x0 as i32 + i) as usize,
                            (y0 as i32 + j) as usize,
                        ) as i64;
                        let r = ref_pic.sample_y(rx + i, ry + j) as i64;
                        sad += (s - r).unsigned_abs();
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

    fn amvp_predictor(&self, x: u32, y: u32, use_l1: bool) -> (i32, i32) {
        let grid = if use_l1 {
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
        let h = CTU_SIZE;
        let w = CTU_SIZE;
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

    fn store_mv(&mut self, x: u32, y: u32, size: u32, mv: (i32, i32), use_l1: bool) {
        let grid = if use_l1 {
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

    fn mc(
        &self,
        x0: u32,
        y0: u32,
        mv_x: i32,
        mv_y: i32,
        use_l1: bool,
    ) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let n = CTU_SIZE as usize;
        let cn = n / 2;
        let ref_pic = if use_l1 { &self.ref_l1 } else { &self.ref_l0 };
        let mut luma = vec![self.neutral; n * n];
        let mut cb = vec![self.neutral; cn * cn];
        let mut cr = vec![self.neutral; cn * cn];
        for j in 0..n {
            for i in 0..n {
                luma[j * n + i] =
                    ref_pic.sample_y(x0 as i32 + i as i32 + mv_x, y0 as i32 + j as i32 + mv_y);
            }
        }
        let cx0 = (x0 / 2) as i32;
        let cy0 = (y0 / 2) as i32;
        let mv_cx = mv_x / 2;
        let mv_cy = mv_y / 2;
        for j in 0..cn {
            for i in 0..cn {
                cb[j * cn + i] =
                    ref_pic.sample_c(cx0 + i as i32 + mv_cx, cy0 + j as i32 + mv_cy, false);
                cr[j * cn + i] =
                    ref_pic.sample_c(cx0 + i as i32 + mv_cx, cy0 + j as i32 + mv_cy, true);
            }
        }
        (luma, cb, cr)
    }

    fn process_luma(
        &self,
        src: &VideoFrame,
        x0: u32,
        y0: u32,
        pred: &[u16],
        log2_tb: u32,
    ) -> (Vec<i32>, Vec<u16>) {
        let n = 1usize << log2_tb;
        let src_y = &src.planes[0];
        let mut residual = vec![0i32; n * n];
        for dy in 0..n {
            for dx in 0..n {
                let s = read_hbd_sample(
                    &src_y.data,
                    src_y.stride,
                    x0 as usize + dx,
                    y0 as usize + dy,
                ) as i32;
                let p = pred[dy * n + dx] as i32;
                residual[dy * n + dx] = s - p;
            }
        }
        let mut coeffs = vec![0i32; n * n];
        forward_transform_2d(&residual, &mut coeffs, log2_tb, false, self.bit_depth);
        let mut levels = vec![0i32; n * n];
        quantize_flat(&coeffs, &mut levels, self.qp_prime, log2_tb, self.bit_depth);
        let mut deq = vec![0i32; n * n];
        dequantize_flat(&levels, &mut deq, self.qp_prime, log2_tb, self.bit_depth);
        let mut res_rec = vec![0i32; n * n];
        inverse_transform_2d(&deq, &mut res_rec, log2_tb, false, self.bit_depth);
        let mut rec = vec![0u16; n * n];
        for i in 0..n * n {
            let v = pred[i] as i32 + res_rec[i];
            rec[i] = v.clamp(0, self.max_sample) as u16;
        }
        (levels, rec)
    }

    fn process_chroma(
        &self,
        src: &VideoFrame,
        cx: u32,
        cy: u32,
        pred: &[u16],
        is_cr: bool,
        log2_tb: u32,
    ) -> (Vec<i32>, Vec<u16>) {
        let n = 1usize << log2_tb;
        let plane = if is_cr {
            &src.planes[2]
        } else {
            &src.planes[1]
        };
        let mut residual = vec![0i32; n * n];
        for dy in 0..n {
            for dx in 0..n {
                let s = read_hbd_sample(
                    &plane.data,
                    plane.stride,
                    cx as usize + dx,
                    cy as usize + dy,
                ) as i32;
                let p = pred[dy * n + dx] as i32;
                residual[dy * n + dx] = s - p;
            }
        }
        let mut coeffs = vec![0i32; n * n];
        forward_transform_2d(&residual, &mut coeffs, log2_tb, false, self.bit_depth);
        let mut levels = vec![0i32; n * n];
        quantize_flat(&coeffs, &mut levels, self.qp_prime, log2_tb, self.bit_depth);
        let mut deq = vec![0i32; n * n];
        dequantize_flat(&levels, &mut deq, self.qp_prime, log2_tb, self.bit_depth);
        let mut res_rec = vec![0i32; n * n];
        inverse_transform_2d(&deq, &mut res_rec, log2_tb, false, self.bit_depth);
        let mut rec = vec![0u16; n * n];
        for i in 0..n * n {
            let v = pred[i] as i32 + res_rec[i];
            rec[i] = v.clamp(0, self.max_sample) as u16;
        }
        (levels, rec)
    }

    fn write_luma_block(&mut self, x: u32, y: u32, n: usize, block: &[u16]) {
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

    fn write_chroma_block(&mut self, x: u32, y: u32, n: usize, block: &[u16], is_cr: bool) {
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

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

fn bipred_u16(
    y0: &[u16],
    cb0: &[u16],
    cr0: &[u16],
    y1: &[u16],
    cb1: &[u16],
    cr1: &[u16],
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let mut y = vec![0u16; y0.len()];
    let mut cb = vec![0u16; cb0.len()];
    let mut cr = vec![0u16; cr0.len()];
    for i in 0..y0.len() {
        y[i] = ((y0[i] as u32 + y1[i] as u32 + 1) >> 1) as u16;
    }
    for i in 0..cb0.len() {
        cb[i] = ((cb0[i] as u32 + cb1[i] as u32 + 1) >> 1) as u16;
    }
    for i in 0..cr0.len() {
        cr[i] = ((cr0[i] as u32 + cr1[i] as u32 + 1) >> 1) as u16;
    }
    (y, cb, cr)
}

fn sad_u16(src: &VideoFrame, x0: u32, y0: u32, pred: &[u16], size: u32) -> u64 {
    let n = size as usize;
    let src_y = &src.planes[0];
    let mut sum = 0u64;
    for j in 0..n {
        for i in 0..n {
            let s =
                read_hbd_sample(&src_y.data, src_y.stride, x0 as usize + i, y0 as usize + j) as i64;
            let p = pred[j * n + i] as i64;
            sum += (s - p).unsigned_abs();
        }
    }
    sum
}

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
