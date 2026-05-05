//! 8-bit B-slice emitter for the Main 4:4:4 profile.
//!
//! Mirrors [`super::b_slice_writer_hbd`] (AMVP-only, no merge/B_Skip in
//! round 33) but at 8-bit 4:4:4: `SubWidthC = SubHeightC = 1`, so all
//! three planes share the same resolution and the chroma MV equals the
//! luma MV (no `/ 2` shift).
//!
//! Round 33: lifts the `chroma_format_idc == 3 && mini_gop > 1` rejection
//! for 8-bit 4:4:4 (Yuv444P).

use crate::cabac::{
    init_row, CtxState, InitType, ABS_MVD_GREATER_FLAGS_INIT_VALUES, CU_SKIP_FLAG_INIT_VALUES,
    INTER_PRED_IDC_INIT_VALUES, MERGE_FLAG_INIT_VALUES, MVP_LX_FLAG_INIT_VALUES,
    PART_MODE_INIT_VALUES, PRED_MODE_FLAG_INIT_VALUES, RQT_ROOT_CBF_INIT_VALUES,
    SPLIT_CU_FLAG_INIT_VALUES,
};
use crate::encoder::bit_writer::{write_rbsp_trailing_bits, BitWriter};
use crate::encoder::cabac_writer::CabacWriter;
use crate::encoder::nal_writer::build_annex_b_nal;
use crate::encoder::p_slice_writer_444::ReferenceFrame444;
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
const FIVE_MINUS_MAX_MERGE: u32 = 4;

/// Build Annex B bytes for an 8-bit 4:4:4 B (TrailR) slice NAL.
pub fn build_b_slice_444(
    cfg: &EncoderConfig,
    frame: &VideoFrame,
    poc_lsb: u32,
    delta_l0: i32,
    delta_l1: i32,
    ref_l0: &ReferenceFrame444,
    ref_l1: &ReferenceFrame444,
) -> (Vec<u8>, ReferenceFrame444) {
    debug_assert!(delta_l0 < 0);
    debug_assert!(delta_l1 > 0);

    let mut bw = BitWriter::new();
    // ---- slice_segment_header() ----------------------------------------
    bw.write_u1(1);
    bw.write_ue(0);
    bw.write_ue(0); // slice_type = B
    bw.write_bits(poc_lsb & 0xFF, 8);
    bw.write_u1(0); // inline RPS
    bw.write_ue(1); // num_negative_pics = 1
    bw.write_ue(1); // num_positive_pics = 1
    bw.write_ue((-delta_l0 - 1) as u32);
    bw.write_u1(1);
    bw.write_ue((delta_l1 - 1) as u32);
    bw.write_u1(1);
    bw.write_u1(0); // num_ref_idx_active_override_flag = 0
    bw.write_u1(0); // mvd_l1_zero_flag = 0
    bw.write_ue(FIVE_MINUS_MAX_MERGE);
    bw.write_se(0); // slice_qp_delta = 0
    write_rbsp_trailing_bits(&mut bw);

    let mut enc = B444State::new(cfg, ref_l0.clone(), ref_l1.clone());
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

    let recon = ReferenceFrame444 {
        width: cfg.width,
        height: cfg.height,
        y: enc.rec_y,
        cb: enc.rec_cb,
        cr: enc.rec_cr,
        stride: cfg.width as usize,
    };
    let nal = build_annex_b_nal(NalHeader::for_type(NalUnitType::TrailR), &bw.finish());
    (nal, recon)
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct B444State {
    cfg: EncoderConfig,
    ref_l0: ReferenceFrame444,
    ref_l1: ReferenceFrame444,
    rec_y: Vec<u8>,
    rec_cb: Vec<u8>,
    rec_cr: Vec<u8>,
    stride: usize,
    mv_grid_l0: Vec<Option<(i32, i32)>>,
    mv_grid_l1: Vec<Option<(i32, i32)>>,
    grid_w4: usize,
    grid_h4: usize,

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

impl B444State {
    fn new(cfg: &EncoderConfig, ref_l0: ReferenceFrame444, ref_l1: ReferenceFrame444) -> Self {
        let w = cfg.width as usize;
        let h = cfg.height as usize;
        let it = InitType::Pb;
        let grid_w4 = w / 4;
        let grid_h4 = h / 4;
        Self {
            cfg: *cfg,
            ref_l0,
            ref_l1,
            rec_y: vec![128u8; w * h],
            rec_cb: vec![128u8; w * h],
            rec_cr: vec![128u8; w * h],
            stride: w,
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
        cw.encode_bin(&mut self.split_cu_flag[0], 0);
        cw.encode_bin(&mut self.cu_skip_flag[0], 0);
        cw.encode_bin(&mut self.pred_mode_flag[0], 0);
        cw.encode_bin(&mut self.part_mode[0], 1); // 2Nx2N
        cw.encode_bin(&mut self.merge_flag[0], 0);

        let (mv_l0, sad_l0) = self.estimate_mv(src, x0, y0, false);
        let (mv_l1, sad_l1) = self.estimate_mv(src, x0, y0, true);
        let (y_l0, cb_l0, cr_l0) = self.mc(x0, y0, mv_l0.0, mv_l0.1, false);
        let (y_l1, cb_l1, cr_l1) = self.mc(x0, y0, mv_l1.0, mv_l1.1, true);
        let (y_bi, cb_bi, cr_bi) = bipred_8bit(&y_l0, &cb_l0, &cr_l0, &y_l1, &cb_l1, &cr_l1);
        let sad_bi = sad_8bit(src, x0, y0, &y_bi, CTU_SIZE);

        let (use_l0, use_l1, pred_y, pred_cb, pred_cr) = if sad_bi <= sad_l0 && sad_bi <= sad_l1 {
            (true, true, y_bi, cb_bi, cr_bi)
        } else if sad_l0 <= sad_l1 {
            (true, false, y_l0, cb_l0, cr_l0)
        } else {
            (false, true, y_l1, cb_l1, cr_l1)
        };

        // inter_pred_idc
        if use_l0 && use_l1 {
            cw.encode_bin(&mut self.inter_pred_idc[0], 1); // bi
        } else {
            cw.encode_bin(&mut self.inter_pred_idc[0], 0); // uni
            cw.encode_bin(&mut self.inter_pred_idc[4], if use_l0 { 0 } else { 1 });
        }

        if use_l0 {
            let mv_qp = (mv_l0.0 * 4, mv_l0.1 * 4);
            let mvp = self.amvp(x0, y0, false);
            self.encode_mvd(cw, mv_qp.0 - mvp.0, mv_qp.1 - mvp.1);
            cw.encode_bin(&mut self.mvp_lx_flag[0], 0);
            self.store_mv(x0, y0, mv_qp, false);
        }
        if use_l1 {
            let mv_qp = (mv_l1.0 * 4, mv_l1.1 * 4);
            let mvp = self.amvp(x0, y0, true);
            self.encode_mvd(cw, mv_qp.0 - mvp.0, mv_qp.1 - mvp.1);
            cw.encode_bin(&mut self.mvp_lx_flag[0], 0);
            self.store_mv(x0, y0, mv_qp, true);
        }

        let log2_tb = 4u32;
        let (luma_levels, luma_rec) =
            process_tb(&pred_y, src, x0, y0, 0, log2_tb, false, SLICE_QP_Y);
        let (cb_levels, cb_rec) = process_tb(&pred_cb, src, x0, y0, 1, log2_tb, false, SLICE_QP_Y);
        let (cr_levels, cr_rec) = process_tb(&pred_cr, src, x0, y0, 2, log2_tb, false, SLICE_QP_Y);

        let cbf_luma = luma_levels.iter().any(|&l| l != 0);
        let cbf_cb = cb_levels.iter().any(|&l| l != 0);
        let cbf_cr = cr_levels.iter().any(|&l| l != 0);
        let any_residual = cbf_luma || cbf_cb || cbf_cr;

        cw.encode_bin(&mut self.rqt_root_cbf[0], any_residual as u32);
        if any_residual {
            cw.encode_bin(&mut self.residual.cbf_cb_cr[0], cbf_cb as u32);
            cw.encode_bin(&mut self.residual.cbf_cb_cr[0], cbf_cr as u32);
            let inferred = !cbf_cb && !cbf_cr;
            if !inferred {
                cw.encode_bin(&mut self.residual.cbf_luma[1], cbf_luma as u32);
            }
            if cbf_luma {
                encode_residual(cw, &mut self.residual, &luma_levels, log2_tb, true);
            }
            if cbf_cb {
                encode_residual(cw, &mut self.residual, &cb_levels, log2_tb, false);
            }
            if cbf_cr {
                encode_residual(cw, &mut self.residual, &cr_levels, log2_tb, false);
            }
        }

        self.write_block(x0, y0, CTU_SIZE as usize, &luma_rec, 0);
        self.write_block(x0, y0, CTU_SIZE as usize, &cb_rec, 1);
        self.write_block(x0, y0, CTU_SIZE as usize, &cr_rec, 2);
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
                        let s = src_y.data
                            [(y0 as i32 + j) as usize * src_y.stride + (x0 as i32 + i) as usize]
                            as i32;
                        let r = ref_pic.sample_y(rx + i, ry + j) as i32;
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
        (best, best_sad)
    }

    fn amvp(&self, x: u32, y: u32, use_l1: bool) -> (i32, i32) {
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

    fn store_mv(&mut self, x: u32, y: u32, mv: (i32, i32), use_l1: bool) {
        let grid = if use_l1 {
            &mut self.mv_grid_l1
        } else {
            &mut self.mv_grid_l0
        };
        let bx0 = (x >> 2) as usize;
        let by0 = (y >> 2) as usize;
        let n4 = (CTU_SIZE >> 2) as usize;
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
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let n = CTU_SIZE as usize;
        let ref_pic = if use_l1 { &self.ref_l1 } else { &self.ref_l0 };
        let mut luma = vec![128u8; n * n];
        let mut cb = vec![128u8; n * n];
        let mut cr = vec![128u8; n * n];
        for j in 0..n {
            for i in 0..n {
                luma[j * n + i] =
                    ref_pic.sample_y(x0 as i32 + i as i32 + mv_x, y0 as i32 + j as i32 + mv_y);
                cb[j * n + i] = ref_pic.sample_c(
                    x0 as i32 + i as i32 + mv_x,
                    y0 as i32 + j as i32 + mv_y,
                    false,
                );
                cr[j * n + i] = ref_pic.sample_c(
                    x0 as i32 + i as i32 + mv_x,
                    y0 as i32 + j as i32 + mv_y,
                    true,
                );
            }
        }
        (luma, cb, cr)
    }

    fn write_block(&mut self, x: u32, y: u32, n: usize, block: &[u8], plane_idx: usize) {
        let x0 = x as usize;
        let y0 = y as usize;
        let w = self.cfg.width as usize;
        let h = self.cfg.height as usize;
        let plane = match plane_idx {
            0 => &mut self.rec_y,
            1 => &mut self.rec_cb,
            _ => &mut self.rec_cr,
        };
        for dy in 0..n {
            for dx in 0..n {
                let xx = x0 + dx;
                let yy = y0 + dy;
                if xx < w && yy < h {
                    plane[yy * self.stride + xx] = block[dy * n + dx];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

fn bipred_8bit(
    y0: &[u8],
    cb0: &[u8],
    cr0: &[u8],
    y1: &[u8],
    cb1: &[u8],
    cr1: &[u8],
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; y0.len()];
    let mut cb = vec![0u8; cb0.len()];
    let mut cr = vec![0u8; cr0.len()];
    for i in 0..y0.len() {
        y[i] = ((y0[i] as u32 + y1[i] as u32 + 1) >> 1) as u8;
    }
    for i in 0..cb0.len() {
        cb[i] = ((cb0[i] as u32 + cb1[i] as u32 + 1) >> 1) as u8;
    }
    for i in 0..cr0.len() {
        cr[i] = ((cr0[i] as u32 + cr1[i] as u32 + 1) >> 1) as u8;
    }
    (y, cb, cr)
}

fn sad_8bit(src: &VideoFrame, x0: u32, y0: u32, pred: &[u8], size: u32) -> u64 {
    let n = size as usize;
    let src_y = &src.planes[0];
    let mut sum = 0u64;
    for j in 0..n {
        for i in 0..n {
            let s = src_y.data[(y0 as usize + j) * src_y.stride + x0 as usize + i] as i32;
            let p = pred[j * n + i] as i32;
            sum += (s - p).unsigned_abs() as u64;
        }
    }
    sum
}

fn process_tb(
    pred: &[u8],
    src: &VideoFrame,
    x0: u32,
    y0: u32,
    plane_idx: usize,
    log2_tb: u32,
    is_dst: bool,
    qp: i32,
) -> (Vec<i32>, Vec<u8>) {
    let n = 1usize << log2_tb;
    let plane = &src.planes[plane_idx];
    let mut residual = vec![0i32; n * n];
    for dy in 0..n {
        for dx in 0..n {
            let sx = x0 as usize + dx;
            let sy = y0 as usize + dy;
            let s = plane.data[sy * plane.stride + sx] as i32;
            let p = pred[dy * n + dx] as i32;
            residual[dy * n + dx] = s - p;
        }
    }
    let mut coeffs = vec![0i32; n * n];
    forward_transform_2d(&residual, &mut coeffs, log2_tb, is_dst, 8);
    let mut levels = vec![0i32; n * n];
    quantize_flat(&coeffs, &mut levels, qp, log2_tb, 8);
    let mut deq = vec![0i32; n * n];
    dequantize_flat(&levels, &mut deq, qp, log2_tb, 8);
    let mut res_rec = vec![0i32; n * n];
    inverse_transform_2d(&deq, &mut res_rec, log2_tb, is_dst, 8);
    let mut rec = vec![0u8; n * n];
    for i in 0..n * n {
        let v = pred[i] as i32 + res_rec[i];
        rec[i] = v.clamp(0, 255) as u8;
    }
    (levels, rec)
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
