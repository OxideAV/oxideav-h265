//! High-bit-depth (10-bit / 12-bit) P-slice emitter for the 4:2:0 path.
//!
//! Mirrors [`super::p_slice_writer`] but operates on `u16` samples (packed
//! LE-16 in the `VideoFrame` planes) and threads `bit_depth` + `Qp'Y =
//! SliceQpY + QpBdOffsetY` through the residual pipeline, matching the
//! decoder's inverse-quantiser arithmetic exactly.
//!
//! CABAC syntax, slice-header fields, and motion-estimation strategy are
//! identical to the 8-bit P-slice writer. Only the sample I/O and the
//! QP' value differ.
//!
//! Round 33: lifts the `mini_gop_size = 1` (P-only) gate for `Yuv420P10Le`
//! and `Yuv420P12Le` by providing this writer; `mini_gop = 2` (B slices)
//! at HBD uses `b_slice_writer_hbd.rs`.

use crate::cabac::{
    init_row, CtxState, InitType, ABS_MVD_GREATER_FLAGS_INIT_VALUES, CU_SKIP_FLAG_INIT_VALUES,
    MERGE_FLAG_INIT_VALUES, MVP_LX_FLAG_INIT_VALUES, PART_MODE_INIT_VALUES,
    PRED_MODE_FLAG_INIT_VALUES, RQT_ROOT_CBF_INIT_VALUES, SPLIT_CU_FLAG_INIT_VALUES,
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

const CTU_SIZE: u32 = 16;
const SLICE_QP_Y: i32 = 26;
const ME_RANGE: i32 = 8;

/// High-bit-depth reconstructed reference frame (u16 samples per component).
/// Used for 10-bit and 12-bit P/B encode paths.
#[derive(Clone)]
pub struct ReferenceFrame16 {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u16>,
    pub cb: Vec<u16>,
    pub cr: Vec<u16>,
    pub y_stride: usize,
    pub c_stride: usize,
}

impl ReferenceFrame16 {
    pub fn new(width: u32, height: u32, y: Vec<u16>, cb: Vec<u16>, cr: Vec<u16>) -> Self {
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
    pub fn sample_y(&self, x: i32, y: i32) -> u16 {
        let xc = x.clamp(0, self.width as i32 - 1) as usize;
        let yc = y.clamp(0, self.height as i32 - 1) as usize;
        self.y[yc * self.y_stride + xc]
    }

    #[inline]
    pub fn sample_c(&self, x: i32, y: i32, is_cr: bool) -> u16 {
        let cw = (self.width / 2) as i32;
        let ch = (self.height / 2) as i32;
        let xc = x.clamp(0, cw - 1) as usize;
        let yc = y.clamp(0, ch - 1) as usize;
        let plane = if is_cr { &self.cr } else { &self.cb };
        plane[yc * self.c_stride + xc]
    }
}

/// Build Annex B bytes for an HBD P (TrailR) slice NAL.
///
/// `delta_l0` is the signed POC delta to the L0 reference (must be < 0).
/// `ref_frame` holds the reconstructed L0 reference at `bit_depth`.
/// Returns (NAL bytes, reconstructed reference frame at `bit_depth`).
pub fn build_p_slice_hbd(
    cfg: &EncoderConfig,
    frame: &VideoFrame,
    frame_idx: u32,
    delta_l0: i32,
    ref_frame: &ReferenceFrame16,
) -> (Vec<u8>, ReferenceFrame16) {
    debug_assert!(
        delta_l0 < 0,
        "HBD P-slice L0 ref must have negative POC delta"
    );
    let bit_depth = cfg.bit_depth;
    let qp_bd_offset = 6 * (bit_depth as i32 - 8);
    let qp_prime = SLICE_QP_Y + qp_bd_offset;

    let mut bw = BitWriter::new();
    // ---- slice_segment_header() ----------------------------------------
    bw.write_u1(1); // first_slice_segment_in_pic_flag
    bw.write_ue(0); // slice_pic_parameter_set_id
    bw.write_ue(1); // slice_type = P
    let poc_lsb = frame_idx & 0xFF;
    bw.write_bits(poc_lsb, 8);
    bw.write_u1(0); // short_term_ref_pic_set_sps_flag = 0 → inline RPS
    bw.write_ue(1); // num_negative_pics = 1
    bw.write_ue(0); // num_positive_pics = 0
    bw.write_ue((-delta_l0 - 1) as u32); // delta_poc_s0_minus1
    bw.write_u1(1); // used_by_curr_pic_s0 = 1
    bw.write_u1(0); // num_ref_idx_active_override_flag = 0
    bw.write_ue(4); // five_minus_max_num_merge_cand → max_num_merge_cand = 1
    bw.write_se(0); // slice_qp_delta = 0
    write_rbsp_trailing_bits(&mut bw);

    // ---- slice_data() --------------------------------------------------
    let mut enc = PHBDState::new(cfg, ref_frame.clone(), qp_prime);
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
// Internal encoder state
// ---------------------------------------------------------------------------

struct PHBDState {
    cfg: EncoderConfig,
    ref_pic: ReferenceFrame16,
    qp_prime: i32,
    bit_depth: u32,
    max_sample: i32,
    neutral: u16,
    rec_y: Vec<u16>,
    rec_cb: Vec<u16>,
    rec_cr: Vec<u16>,
    y_stride: usize,
    c_stride: usize,
    mv_grid: Vec<Option<(i32, i32)>>,
    grid_w4: usize,
    grid_h4: usize,

    // CABAC contexts (P slices with cabac_init_flag=0 use InitType::Pa).
    split_cu_flag: [CtxState; 3],
    cu_skip_flag: [CtxState; 3],
    pred_mode_flag: [CtxState; 1],
    part_mode: [CtxState; 4],
    merge_flag: [CtxState; 1],
    mvp_lx_flag: [CtxState; 1],
    abs_mvd_greater: [CtxState; 2],
    rqt_root_cbf: [CtxState; 1],
    residual: ResidualCtx,
}

impl PHBDState {
    fn new(cfg: &EncoderConfig, ref_pic: ReferenceFrame16, qp_prime: i32) -> Self {
        let w = cfg.width as usize;
        let h = cfg.height as usize;
        let cw = w / 2;
        let ch = h / 2;
        let it = InitType::Pa;
        let grid_w4 = w / 4;
        let grid_h4 = h / 4;
        let bit_depth = cfg.bit_depth;
        let neutral = 1u16 << (bit_depth - 1);
        let max_sample = (1i32 << bit_depth) - 1;
        Self {
            cfg: *cfg,
            ref_pic,
            qp_prime,
            bit_depth,
            max_sample,
            neutral,
            rec_y: vec![neutral; w * h],
            rec_cb: vec![neutral; cw * ch],
            rec_cr: vec![neutral; cw * ch],
            y_stride: w,
            c_stride: cw,
            mv_grid: vec![None; grid_w4 * grid_h4],
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
            residual: ResidualCtx::with_init_type(SLICE_QP_Y, it),
        }
    }

    fn encode_ctu(&mut self, cw: &mut CabacWriter<'_>, src: &VideoFrame, x0: u32, y0: u32) {
        cw.encode_bin(&mut self.split_cu_flag[0], 0);
        cw.encode_bin(&mut self.cu_skip_flag[0], 0);
        cw.encode_bin(&mut self.pred_mode_flag[0], 0);
        cw.encode_bin(&mut self.part_mode[0], 1); // 2Nx2N

        // Motion estimation + AMVP
        cw.encode_bin(&mut self.merge_flag[0], 0);
        let (mv_int_x, mv_int_y) = self.estimate_mv(src, x0, y0);
        let mv_qp_x = mv_int_x * 4;
        let mv_qp_y = mv_int_y * 4;
        let (mvp_x, mvp_y) = self.amvp_predictor(x0, y0);
        let mvd_x = mv_qp_x - mvp_x;
        let mvd_y = mv_qp_y - mvp_y;
        self.encode_mvd(cw, mvd_x, mvd_y);
        cw.encode_bin(&mut self.mvp_lx_flag[0], 0);
        self.store_mv(x0, y0, CTU_SIZE, (mv_qp_x, mv_qp_y));

        // Residual
        let (luma_pred, cb_pred, cr_pred) = self.motion_compensate(x0, y0, mv_int_x, mv_int_y);
        let log2_tb = 4u32;
        let c_log2 = 3u32;
        let (luma_levels, luma_rec) = self.process_luma(src, x0, y0, &luma_pred, log2_tb);
        let cx = x0 / 2;
        let cy = y0 / 2;
        let (cb_levels, cb_rec) = self.process_chroma(src, cx, cy, &cb_pred, false, c_log2);
        let (cr_levels, cr_rec) = self.process_chroma(src, cx, cy, &cr_pred, true, c_log2);

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

    /// Integer-pel SAD motion estimation using u16 luma samples.
    fn estimate_mv(&self, src: &VideoFrame, x0: u32, y0: u32) -> (i32, i32) {
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
                        let s = read_hbd_sample(
                            &src_y.data,
                            src_y.stride,
                            (x0 as i32 + i) as usize,
                            (y0 as i32 + j) as usize,
                        ) as i64;
                        let r = self.ref_pic.sample_y(rx + i, ry + j) as i64;
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
        best
    }

    fn amvp_predictor(&self, x: u32, y: u32) -> (i32, i32) {
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

    fn motion_compensate(
        &self,
        x0: u32,
        y0: u32,
        mv_x: i32,
        mv_y: i32,
    ) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let n = CTU_SIZE as usize;
        let cn = n / 2;
        let mut luma = vec![self.neutral; n * n];
        let mut cb = vec![self.neutral; cn * cn];
        let mut cr = vec![self.neutral; cn * cn];
        for j in 0..n {
            for i in 0..n {
                luma[j * n + i] = self
                    .ref_pic
                    .sample_y(x0 as i32 + i as i32 + mv_x, y0 as i32 + j as i32 + mv_y);
            }
        }
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
        // §8.6.1: for QpY=26 with zero offsets, Qp'Cb = Qp'Cr = qp_prime
        // (chroma QP table identity at low QP + QpBdOffset).
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
// Helpers
// ---------------------------------------------------------------------------

/// Read a single HBD sample (10-bit or 12-bit) from a packed LE-16 plane.
/// `stride` is the byte stride. Samples are stored as little-endian u16.
#[inline]
pub fn read_hbd_sample(plane: &[u8], stride: usize, x: usize, y: usize) -> u16 {
    let off = y * stride + x * 2;
    (plane[off] as u16) | ((plane[off + 1] as u16) << 8)
}

/// Exp-Golomb order-1 bypass encode (mirrors `encode_eg1` in `p_slice_writer`).
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
