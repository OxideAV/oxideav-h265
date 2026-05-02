//! 8-bit IDR I-slice emitter for the Main 4:4:4 profile (round 30).
//!
//! Mirrors the structure of [`super::slice_writer`] but operates on the
//! 4:4:4 chroma topology: chroma planes are full luma resolution
//! (`SubWidthC = SubHeightC = 1` per §6.2 Table 6-1), so each 16×16
//! luma TB is paired with a co-located 16×16 Cb TB and a 16×16 Cr TB
//! (vs. the 8×8 chroma TBs the 4:2:0 path emits).
//!
//! Like the round-25 Main 10 / round-26 Main 12 writers, this lives as
//! a parallel module rather than templating the existing 4:2:0 paths
//! over `chroma_format_idc`, so the Main (8-bit 4:2:0), Main 10, and
//! Main 12 emissions stay byte-for-byte unchanged at rounds 1..29.
//!
//! The CTU layout, intra mode subset, fixed QP 26, and per-CU CABAC
//! syntax are otherwise identical to `slice_writer`.

use crate::cabac::{
    init_row, CtxState, InitType, INTRA_CHROMA_PRED_MODE_INIT_VALUES,
    PREV_INTRA_LUMA_PRED_FLAG_INIT_VALUES, SPLIT_CU_FLAG_INIT_VALUES,
};
use crate::encoder::bit_writer::{write_rbsp_trailing_bits, BitWriter};
use crate::encoder::cabac_writer::CabacWriter;
use crate::encoder::nal_writer::build_annex_b_nal;
use crate::encoder::params::EncoderConfig;
use crate::encoder::residual_writer::{encode_residual, ResidualCtx};
use crate::intra_pred::{build_ref_samples, filter_decision, filter_ref_samples, predict};
use crate::nal::{NalHeader, NalUnitType};
use crate::transform::{
    dequantize_flat, forward_transform_2d, inverse_transform_2d, quantize_flat,
};
use oxideav_core::VideoFrame;

const CTU_SIZE: u32 = 16;
const SLICE_QP_Y: i32 = 26;
const BIT_DEPTH: u32 = 8;

const LUMA_CANDIDATES: [u32; 7] = [0, 1, 2, 10, 18, 26, 34];

/// Build Annex B bytes for the IDR I-slice NAL using the 8-bit 4:4:4 pipeline.
pub fn build_idr_slice_nal_main444_8(cfg: &EncoderConfig, frame: &VideoFrame) -> Vec<u8> {
    let rbsp = build_idr_slice_rbsp_main444_8(cfg, frame);
    build_annex_b_nal(NalHeader::for_type(NalUnitType::IdrNLp), &rbsp)
}

fn build_idr_slice_rbsp_main444_8(cfg: &EncoderConfig, frame: &VideoFrame) -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.write_u1(1); // first_slice_segment_in_pic_flag
    bw.write_u1(0); // no_output_of_prior_pics_flag
    bw.write_ue(0); // slice_pic_parameter_set_id
    bw.write_ue(2); // slice_type = I
    bw.write_se(0); // slice_qp_delta = 0
    write_rbsp_trailing_bits(&mut bw);

    let mut enc = EncoderState444::new(cfg);
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

struct EncoderState444 {
    cfg: EncoderConfig,
    rec_y: Vec<u8>,
    rec_cb: Vec<u8>,
    rec_cr: Vec<u8>,
    y_stride: usize,
    /// Chroma stride: equal to `y_stride` at 4:4:4 (SubWidthC = 1).
    c_stride: usize,
    intra_mode_grid: Vec<u8>,
    intra_w4: usize,
    intra_h4: usize,
    decoded_4x4: Vec<bool>,

    split_cu_flag: [CtxState; 3],
    prev_intra_luma_pred_flag: [CtxState; 1],
    intra_chroma_pred_mode: [CtxState; 1],
    residual: ResidualCtx,
}

impl EncoderState444 {
    fn new(cfg: &EncoderConfig) -> Self {
        let w = cfg.width as usize;
        let h = cfg.height as usize;
        // 4:4:4 — chroma planes are full luma resolution.
        let it = InitType::I;
        Self {
            cfg: *cfg,
            rec_y: vec![128u8; w * h],
            rec_cb: vec![128u8; w * h],
            rec_cr: vec![128u8; w * h],
            y_stride: w,
            c_stride: w,
            intra_mode_grid: vec![1u8; (w / 4) * (h / 4)],
            intra_w4: w / 4,
            intra_h4: h / 4,
            decoded_4x4: vec![false; (w / 4) * (h / 4)],
            split_cu_flag: init_row(&SPLIT_CU_FLAG_INIT_VALUES, it, SLICE_QP_Y),
            prev_intra_luma_pred_flag: init_row(
                &PREV_INTRA_LUMA_PRED_FLAG_INIT_VALUES,
                it,
                SLICE_QP_Y,
            ),
            intra_chroma_pred_mode: init_row(&INTRA_CHROMA_PRED_MODE_INIT_VALUES, it, SLICE_QP_Y),
            residual: ResidualCtx::new(SLICE_QP_Y),
        }
    }

    fn encode_ctu(&mut self, cw: &mut CabacWriter<'_>, src: &VideoFrame, x0: u32, y0: u32) {
        // split_cu_flag = 0 — the whole 16×16 CTU stays as one CU.
        cw.encode_bin(&mut self.split_cu_flag[0], 0);

        let log2_tb = 4u32; // 16×16 luma TB
        let best_mode = self.choose_luma_mode(src, x0, y0, log2_tb);

        // MPM list + signalling — same as 4:2:0 path.
        let mpm = self.derive_mpm_list(x0, y0);
        let (prev_flag, mpm_idx, rem) = resolve_luma_mode_signalling(&mpm, best_mode);
        cw.encode_bin(&mut self.prev_intra_luma_pred_flag[0], prev_flag);
        if prev_flag == 1 {
            match mpm_idx {
                0 => cw.encode_bypass(0),
                1 => {
                    cw.encode_bypass(1);
                    cw.encode_bypass(0);
                }
                _ => {
                    cw.encode_bypass(1);
                    cw.encode_bypass(1);
                }
            }
        } else {
            for bit in (0..5).rev() {
                cw.encode_bypass((rem >> bit) & 1);
            }
        }
        self.store_luma_mode(x0, y0, CTU_SIZE, best_mode as u8);

        // intra_chroma_pred_mode = 0 (DM / inherit luma).
        cw.encode_bin(&mut self.intra_chroma_pred_mode[0], 0);
        let chroma_mode = resolve_chroma_mode(best_mode, 4);

        // §7.3.8.10 transform_unit at 4:4:4 (ChromaArrayType == 3):
        // log2TrafoSizeC = max(2, log2TrafoSize - 0) = log2TrafoSize, so
        // chroma TBs are 16×16 (matching the luma TB) and co-located at
        // (x0, y0) — *not* (x0/2, y0/2) like 4:2:0.
        let (luma_levels, luma_rec) = self.process_luma_tb(src, x0, y0, log2_tb, best_mode);
        let cbf_luma = luma_levels.iter().any(|&l| l != 0) as u32;

        let c_log2 = log2_tb; // 16×16 chroma TB at 4:4:4
        let cx = x0; // chroma at full luma resolution
        let cy = y0;
        let (cb_levels, cb_rec) = self.process_chroma_tb(src, cx, cy, c_log2, chroma_mode, false);
        let (cr_levels, cr_rec) = self.process_chroma_tb(src, cx, cy, c_log2, chroma_mode, true);
        let cbf_cb = cb_levels.iter().any(|&l| l != 0) as u32;
        let cbf_cr = cr_levels.iter().any(|&l| l != 0) as u32;

        cw.encode_bin(&mut self.residual.cbf_cb_cr[0], cbf_cb);
        cw.encode_bin(&mut self.residual.cbf_cb_cr[0], cbf_cr);
        cw.encode_bin(&mut self.residual.cbf_luma[1], cbf_luma);

        if cbf_luma == 1 {
            encode_residual(cw, &mut self.residual, &luma_levels, log2_tb, true);
        }
        if cbf_cb == 1 {
            encode_residual(cw, &mut self.residual, &cb_levels, c_log2, false);
        }
        if cbf_cr == 1 {
            encode_residual(cw, &mut self.residual, &cr_levels, c_log2, false);
        }

        self.write_luma_block(x0, y0, CTU_SIZE as usize, &luma_rec);
        self.write_chroma_block(cx, cy, 1 << c_log2, &cb_rec, false);
        self.write_chroma_block(cx, cy, 1 << c_log2, &cr_rec, true);
        self.mark_decoded(x0, y0, CTU_SIZE);
    }

    fn choose_luma_mode(&self, src: &VideoFrame, x0: u32, y0: u32, log2_tb: u32) -> u32 {
        let n = 1usize << log2_tb;
        let refs = self.gather_refs_luma(x0, y0, n);
        let src_y = &src.planes[0];
        let mut best_mode = 1u32;
        let mut best_sad = u64::MAX;
        let mut pred = vec![0u16; n * n];
        for &mode in LUMA_CANDIDATES.iter() {
            let mut r = refs.clone();
            let (apply, strong) = filter_decision(log2_tb, mode, false, &r, n, BIT_DEPTH);
            if apply {
                filter_ref_samples(&mut r, n, strong, BIT_DEPTH);
            }
            predict(&r, n, &mut pred, n, mode, true, BIT_DEPTH);
            let sad = sad_block_u16(
                &pred,
                n,
                &src_y.data,
                src_y.stride,
                x0 as usize,
                y0 as usize,
            );
            if sad < best_sad {
                best_sad = sad;
                best_mode = mode;
            }
        }
        best_mode
    }

    fn process_luma_tb(
        &self,
        src: &VideoFrame,
        x0: u32,
        y0: u32,
        log2_tb: u32,
        mode: u32,
    ) -> (Vec<i32>, Vec<u8>) {
        let n = 1usize << log2_tb;
        let mut refs = self.gather_refs_luma(x0, y0, n);
        let (apply, strong) = filter_decision(log2_tb, mode, false, &refs, n, BIT_DEPTH);
        if apply {
            filter_ref_samples(&mut refs, n, strong, BIT_DEPTH);
        }
        let mut pred16 = vec![0u16; n * n];
        predict(&refs, n, &mut pred16, n, mode, true, BIT_DEPTH);
        let pred: Vec<u8> = pred16.iter().map(|&v| v as u8).collect();

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

        let mut coeffs = vec![0i32; n * n];
        let is_dst = log2_tb == 2;
        forward_transform_2d(&residual, &mut coeffs, log2_tb, is_dst, BIT_DEPTH);
        let mut levels = vec![0i32; n * n];
        quantize_flat(&coeffs, &mut levels, SLICE_QP_Y, log2_tb, BIT_DEPTH);

        let mut deq = vec![0i32; n * n];
        dequantize_flat(&levels, &mut deq, SLICE_QP_Y, log2_tb, BIT_DEPTH);
        let mut res_rec = vec![0i32; n * n];
        inverse_transform_2d(&deq, &mut res_rec, log2_tb, is_dst, BIT_DEPTH);
        let mut rec = vec![0u8; n * n];
        for i in 0..n * n {
            let v = pred[i] as i32 + res_rec[i];
            rec[i] = v.clamp(0, 255) as u8;
        }
        (levels, rec)
    }

    fn process_chroma_tb(
        &self,
        src: &VideoFrame,
        cx: u32,
        cy: u32,
        log2_tb: u32,
        mode: u32,
        is_cr: bool,
    ) -> (Vec<i32>, Vec<u8>) {
        let n = 1usize << log2_tb;
        let refs = self.gather_refs_chroma(cx, cy, n, is_cr);
        let mut pred16 = vec![0u16; n * n];
        predict(&refs, n, &mut pred16, n, mode, false, BIT_DEPTH);
        let pred: Vec<u8> = pred16.iter().map(|&v| v as u8).collect();
        let plane = if is_cr {
            &src.planes[2]
        } else {
            &src.planes[1]
        };
        let qp = chroma_qp_for_slice();
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
        let mut coeffs = vec![0i32; n * n];
        forward_transform_2d(&residual, &mut coeffs, log2_tb, false, BIT_DEPTH);
        let mut levels = vec![0i32; n * n];
        quantize_flat(&coeffs, &mut levels, qp, log2_tb, BIT_DEPTH);

        let mut deq = vec![0i32; n * n];
        dequantize_flat(&levels, &mut deq, qp, log2_tb, BIT_DEPTH);
        let mut res_rec = vec![0i32; n * n];
        inverse_transform_2d(&deq, &mut res_rec, log2_tb, false, BIT_DEPTH);
        let mut rec = vec![0u8; n * n];
        for i in 0..n * n {
            let v = pred[i] as i32 + res_rec[i];
            rec[i] = v.clamp(0, 255) as u8;
        }
        (levels, rec)
    }

    fn gather_refs_luma(&self, x0: u32, y0: u32, n: usize) -> Vec<u16> {
        let len = 4 * n + 1;
        let mut samples = vec![128u16; len];
        let mut avail = vec![false; len];
        let stride = self.y_stride;
        let plane = &self.rec_y;
        let w = self.cfg.width as usize;
        let h = self.cfg.height as usize;
        let is_decoded = |lx: usize, ly: usize| -> bool {
            let bx = lx >> 2;
            let by = ly >> 2;
            bx < self.intra_w4 && by < self.intra_h4 && self.decoded_4x4[by * self.intra_w4 + bx]
        };
        let x0 = x0 as usize;
        let y0 = y0 as usize;
        if x0 > 0 && y0 > 0 && is_decoded(x0 - 1, y0 - 1) {
            samples[0] = plane[(y0 - 1) * stride + (x0 - 1)] as u16;
            avail[0] = true;
        }
        if y0 > 0 {
            for i in 0..(2 * n) {
                let xx = x0 + i;
                if xx < w && is_decoded(xx, y0 - 1) {
                    samples[1 + i] = plane[(y0 - 1) * stride + xx] as u16;
                    avail[1 + i] = true;
                }
            }
        }
        if x0 > 0 {
            for i in 0..(2 * n) {
                let yy = y0 + i;
                if yy < h && is_decoded(x0 - 1, yy) {
                    samples[2 * n + 1 + i] = plane[yy * stride + (x0 - 1)] as u16;
                    avail[2 * n + 1 + i] = true;
                }
            }
        }
        build_ref_samples(&samples, &avail, n, BIT_DEPTH)
    }

    /// Gather chroma reference samples. At 4:4:4 the chroma plane is full
    /// luma resolution, so the decoded grid is shared with luma (no shift)
    /// and the picture bounds are `cfg.width` / `cfg.height`.
    fn gather_refs_chroma(&self, cx: u32, cy: u32, n: usize, is_cr: bool) -> Vec<u16> {
        let len = 4 * n + 1;
        let mut samples = vec![128u16; len];
        let mut avail = vec![false; len];
        let stride = self.c_stride;
        let plane = if is_cr { &self.rec_cr } else { &self.rec_cb };
        let cw = self.cfg.width as usize;
        let ch = self.cfg.height as usize;
        let is_decoded = |x: usize, y: usize| -> bool {
            let bx = x >> 2;
            let by = y >> 2;
            bx < self.intra_w4 && by < self.intra_h4 && self.decoded_4x4[by * self.intra_w4 + bx]
        };
        let cx = cx as usize;
        let cy = cy as usize;
        if cx > 0 && cy > 0 && is_decoded(cx - 1, cy - 1) {
            samples[0] = plane[(cy - 1) * stride + (cx - 1)] as u16;
            avail[0] = true;
        }
        if cy > 0 {
            for i in 0..(2 * n) {
                let xx = cx + i;
                if xx < cw && is_decoded(xx, cy - 1) {
                    samples[1 + i] = plane[(cy - 1) * stride + xx] as u16;
                    avail[1 + i] = true;
                }
            }
        }
        if cx > 0 {
            for i in 0..(2 * n) {
                let yy = cy + i;
                if yy < ch && is_decoded(cx - 1, yy) {
                    samples[2 * n + 1 + i] = plane[yy * stride + (cx - 1)] as u16;
                    avail[2 * n + 1 + i] = true;
                }
            }
        }
        build_ref_samples(&samples, &avail, n, BIT_DEPTH)
    }

    fn derive_mpm_list(&self, x0: u32, y0: u32) -> [u32; 3] {
        let get_mode = |x: u32, y: u32| -> Option<u32> {
            if x >= self.cfg.width || y >= self.cfg.height {
                return None;
            }
            let bx = (x >> 2) as usize;
            let by = (y >> 2) as usize;
            Some(self.intra_mode_grid[by * self.intra_w4 + bx] as u32)
        };
        let a = if x0 == 0 { None } else { get_mode(x0 - 1, y0) };
        let ctb_log2: u32 = CTU_SIZE.trailing_zeros();
        let ctb_row_top = (y0 >> ctb_log2) << ctb_log2;
        let b = if y0 == 0 {
            None
        } else if y0 - 1 < ctb_row_top {
            Some(1)
        } else {
            get_mode(x0, y0 - 1)
        };
        let cand_a = a.unwrap_or(1);
        let cand_b = b.unwrap_or(1);
        let mut list = [0u32; 3];
        if cand_a == cand_b {
            if cand_a < 2 {
                list[0] = 0;
                list[1] = 1;
                list[2] = 26;
            } else {
                list[0] = cand_a;
                list[1] = 2 + ((cand_a + 29) % 32);
                list[2] = 2 + ((cand_a - 2 + 1) % 32);
            }
        } else {
            list[0] = cand_a;
            list[1] = cand_b;
            if cand_a != 0 && cand_b != 0 {
                list[2] = 0;
            } else if cand_a != 1 && cand_b != 1 {
                list[2] = 1;
            } else {
                list[2] = 26;
            }
        }
        list
    }

    fn store_luma_mode(&mut self, x: u32, y: u32, size: u32, mode: u8) {
        let bx0 = (x >> 2) as usize;
        let by0 = (y >> 2) as usize;
        let n4 = (size >> 2) as usize;
        for dy in 0..n4 {
            for dx in 0..n4 {
                let bx = bx0 + dx;
                let by = by0 + dy;
                if bx < self.intra_w4 && by < self.intra_h4 {
                    self.intra_mode_grid[by * self.intra_w4 + bx] = mode;
                }
            }
        }
    }

    fn mark_decoded(&mut self, x: u32, y: u32, size: u32) {
        let bx0 = (x >> 2) as usize;
        let by0 = (y >> 2) as usize;
        let n4 = (size >> 2) as usize;
        for dy in 0..n4 {
            for dx in 0..n4 {
                let bx = bx0 + dx;
                let by = by0 + dy;
                if bx < self.intra_w4 && by < self.intra_h4 {
                    self.decoded_4x4[by * self.intra_w4 + bx] = true;
                }
            }
        }
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

    /// Write a chroma block at 4:4:4 — chroma planes are full luma
    /// resolution so the bounds match `cfg.width` / `cfg.height`.
    fn write_chroma_block(&mut self, x: u32, y: u32, n: usize, block: &[u8], is_cr: bool) {
        let x0 = x as usize;
        let y0 = y as usize;
        let cw = self.cfg.width as usize;
        let ch = self.cfg.height as usize;
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

fn sad_block_u16(pred: &[u16], n: usize, src: &[u8], stride: usize, x0: usize, y0: usize) -> u64 {
    let mut sum = 0u64;
    for dy in 0..n {
        for dx in 0..n {
            let s = src[(y0 + dy) * stride + (x0 + dx)] as i32;
            let p = pred[dy * n + dx] as i32;
            sum += (s - p).unsigned_abs() as u64;
        }
    }
    sum
}

fn resolve_chroma_mode(luma_mode: u32, icpm: u32) -> u32 {
    let chosen = match icpm {
        0 => 0,
        1 => 26,
        2 => 10,
        3 => 1,
        _ => luma_mode,
    };
    if chosen == luma_mode && icpm != 4 {
        34
    } else {
        chosen
    }
}

fn resolve_luma_mode_signalling(mpm: &[u32; 3], mode: u32) -> (u32, u32, u32) {
    for (i, &m) in mpm.iter().enumerate() {
        if m == mode {
            return (1, i as u32, 0);
        }
    }
    let mut sorted = *mpm;
    sorted.sort();
    let mut rem = mode;
    if rem >= sorted[2] {
        rem -= 1;
    }
    if rem >= sorted[1] {
        rem -= 1;
    }
    if rem >= sorted[0] {
        rem -= 1;
    }
    (0, 0, rem)
}

fn chroma_qp_for_slice() -> i32 {
    // §8.6.1: at ChromaArrayType == 3 (4:4:4) QpC = qpi (no Table 8-10
    // collapse), so for QpY = 26 with zero offsets QpC = 26.
    26
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::VideoPlane;

    fn make_gradient_yuv444_frame(w: usize, h: usize) -> VideoFrame {
        let mut y = vec![0u8; w * h];
        for yy in 0..h {
            for xx in 0..w {
                y[yy * w + xx] = ((xx * 255) / w.max(1)) as u8;
            }
        }
        let cb = vec![128u8; w * h];
        let cr = vec![128u8; w * h];
        VideoFrame {
            pts: Some(0),
            planes: vec![
                VideoPlane { stride: w, data: y },
                VideoPlane {
                    stride: w,
                    data: cb,
                },
                VideoPlane {
                    stride: w,
                    data: cr,
                },
            ],
        }
    }

    #[test]
    fn local_recon_close_to_input_qp26() {
        let cfg = EncoderConfig::new_main444_8(16, 16);
        let state = EncoderState444::new(&cfg);
        let frame = make_gradient_yuv444_frame(16, 16);
        let (_levels, rec) = state.process_luma_tb(&frame, 0, 0, 4, 1);
        let mut sse = 0u64;
        for (i, &r) in rec.iter().enumerate() {
            let s = frame.planes[0].data[i] as i32;
            let d = s - r as i32;
            sse += (d * d) as u64;
        }
        let mse = sse as f64 / (16.0 * 16.0);
        let psnr = 10.0 * (255.0 * 255.0 / mse).log10();
        eprintln!("main444 local recon psnr={psnr:.2} dB");
        assert!(psnr > 20.0, "Main 4:4:4 local PSNR too low: {psnr:.2} dB");
    }

    #[test]
    fn idr_emits_with_main444_layout() {
        let cfg = EncoderConfig::new_main444_8(16, 16);
        let frame = make_gradient_yuv444_frame(16, 16);
        let nal = build_idr_slice_nal_main444_8(&cfg, &frame);
        assert!(nal.len() > 4, "IDR NAL too short ({} bytes)", nal.len());
    }
}
