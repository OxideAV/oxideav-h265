//! IDR I-slice header + residual-coded CTU payload emitter.
//!
//! Every frame is encoded as a single-slice IDR picture. The CU layout
//! is fixed at 16×16 (matching the 16×16 CTU advertised in the SPS):
//! one 16×16 luma TB plus two 8×8 chroma TBs per CU.
//!
//! The per-CU pipeline mirrors §8.4.4 / §7.3.8:
//!
//! 1. Reconstruct neighbour samples (z-scan available) and derive ref
//!    samples per `intra_pred::build_ref_samples`.
//! 2. Try a small subset of intra modes (DC + planar + cardinals +
//!    diagonals), pick the one with the smallest SAD vs the source.
//! 3. Compute residual = source - prediction.
//! 4. Forward transform (DCT-II) + forward quantise at slice QP.
//! 5. Dequantise + inverse transform to produce the local reconstruction
//!    used as neighbour data for later CUs.
//! 6. Emit the CABAC syntax: `part_mode` (2Nx2N), `prev_intra_luma_pred_flag`
//!    with MPM derivation, `intra_chroma_pred_mode = 4` (DM), `cbf_cb`,
//!    `cbf_cr`, `cbf_luma`, then residual-coding for each non-zero TB.

use crate::cabac::{
    init_row, CtxState, InitType, INTRA_CHROMA_PRED_MODE_INIT_VALUES, PART_MODE_INIT_VALUES,
    PREV_INTRA_LUMA_PRED_FLAG_INIT_VALUES,
};
use crate::encoder::bit_writer::{write_rbsp_trailing_bits, BitWriter};
use crate::encoder::cabac_writer::CabacWriter;
use crate::encoder::nal_writer::build_annex_b_nal;
use crate::encoder::params::EncoderConfig;
use crate::encoder::residual_writer::{encode_residual, ResidualCtx};
use crate::intra_pred::{build_ref_samples, filter_decision, filter_ref_samples, predict};
use crate::nal::{NalHeader, NalUnitType};
use crate::transform::{dequantize_flat, forward_transform_2d, inverse_transform_2d, quantize_flat};
use oxideav_core::VideoFrame;

/// CTU size = 16. minCU = 16. One 16×16 luma TB per CU.
const CTU_SIZE: u32 = 16;
/// Slice QP.
const SLICE_QP_Y: i32 = 26;

/// Candidate luma intra modes tried per CU during mode decision.
const LUMA_CANDIDATES: [u32; 7] = [
    0,  // PLANAR
    1,  // DC
    2,  // diagonal bottom-left
    10, // horizontal
    18, // diagonal
    26, // vertical
    34, // diagonal top-right
];

/// Build Annex B bytes for the IDR I-slice NAL (header + slice_data).
pub fn build_idr_slice_nal(cfg: &EncoderConfig, frame: &VideoFrame) -> Vec<u8> {
    let rbsp = build_idr_slice_rbsp(cfg, frame);
    build_annex_b_nal(NalHeader::for_type(NalUnitType::IdrNLp), &rbsp)
}

/// Build the RBSP payload for the IDR I-slice.
pub fn build_idr_slice_rbsp(cfg: &EncoderConfig, frame: &VideoFrame) -> Vec<u8> {
    let mut bw = BitWriter::new();

    // ---- slice_segment_header() -------------------------------------
    bw.write_u1(1); // first_slice_segment_in_pic_flag
    bw.write_u1(0); // no_output_of_prior_pics_flag
    bw.write_ue(0); // slice_pic_parameter_set_id
    bw.write_ue(2); // slice_type = I
    bw.write_se(0); // slice_qp_delta = 0 (init_qp=26 → SliceQpY=26)
    // PPS disables deblocking and SAO is off, so the gate
    // `(SAO || !deblock_disabled)` is false and the
    // `slice_loop_filter_across_slices_enabled_flag` is NOT emitted.
    write_rbsp_trailing_bits(&mut bw);

    // ---- slice_data() ------------------------------------------------
    // We drive the whole slice off a single CABAC run that never exits
    // until the final `end_of_slice_segment_flag = 1` terminator.
    let mut enc = EncoderState::new(cfg);
    {
        let mut cabac = CabacWriter::new(&mut bw);
        let pic_w_ctb = cfg.width.div_ceil(CTU_SIZE);
        let pic_h_ctb = cfg.height.div_ceil(CTU_SIZE);
        let total_ctbs = pic_w_ctb * pic_h_ctb;
        for i in 0..total_ctbs {
            let ctb_x = (i % pic_w_ctb) * CTU_SIZE;
            let ctb_y = (i / pic_w_ctb) * CTU_SIZE;
            enc.encode_ctu(&mut cabac, frame, ctb_x, ctb_y);
            // end_of_slice_segment_flag: 0 for all but the last CTU.
            let is_last = i + 1 == total_ctbs;
            cabac.encode_terminate(if is_last { 1 } else { 0 });
        }
        cabac.encode_flush();
    }
    bw.align_to_byte_zero();
    bw.finish()
}

/// Per-slice encoding state: contexts, reconstructed picture buffer,
/// neighbour-mode grid for MPM derivation.
struct EncoderState {
    cfg: EncoderConfig,
    // Reconstructed planes — needed for intra reference samples of later CUs.
    rec_y: Vec<u8>,
    rec_cb: Vec<u8>,
    rec_cr: Vec<u8>,
    y_stride: usize,
    c_stride: usize,
    // 4×4 grid of intra luma modes for neighbour-MPM lookups.
    intra_mode_grid: Vec<u8>,
    intra_w4: usize,
    intra_h4: usize,
    // 4×4 grid: true when the block has been reconstructed (neighbour ref
    // samples are usable).
    decoded_4x4: Vec<bool>,

    // CABAC contexts (persist across CTUs within the slice).
    part_mode: [CtxState; 4],
    prev_intra_luma_pred_flag: [CtxState; 1],
    intra_chroma_pred_mode: [CtxState; 1],
    residual: ResidualCtx,
}

impl EncoderState {
    fn new(cfg: &EncoderConfig) -> Self {
        let w = cfg.width as usize;
        let h = cfg.height as usize;
        let cw = w / 2;
        let ch = h / 2;
        let it = InitType::I;
        Self {
            cfg: *cfg,
            rec_y: vec![128u8; w * h],
            rec_cb: vec![128u8; cw * ch],
            rec_cr: vec![128u8; cw * ch],
            y_stride: w,
            c_stride: cw,
            intra_mode_grid: vec![1u8; (w / 4) * (h / 4)],
            intra_w4: w / 4,
            intra_h4: h / 4,
            decoded_4x4: vec![false; (w / 4) * (h / 4)],
            part_mode: init_row(&PART_MODE_INIT_VALUES, it, SLICE_QP_Y),
            prev_intra_luma_pred_flag: init_row(
                &PREV_INTRA_LUMA_PRED_FLAG_INIT_VALUES,
                it,
                SLICE_QP_Y,
            ),
            intra_chroma_pred_mode: init_row(
                &INTRA_CHROMA_PRED_MODE_INIT_VALUES,
                it,
                SLICE_QP_Y,
            ),
            residual: ResidualCtx::new(SLICE_QP_Y),
        }
    }

    fn encode_ctu(&mut self, cw: &mut CabacWriter<'_>, src: &VideoFrame, x0: u32, y0: u32) {
        // At log2_cb == min_cb_log2 (= 4), split_cu_flag is NOT signalled.
        // No skip, no pred_mode, no pcm_flag.
        //
        // part_mode is signalled at min_cb. Emit `1` (2Nx2N).
        cw.encode_bin(&mut self.part_mode[0], 1);

        // --- 16x16 luma intra --------------------------------------------
        // Pick the best luma mode (by SAD vs source).
        let log2_tb = 4u32; // 16x16
        let best_mode = self.choose_luma_mode(src, x0, y0, log2_tb);

        // MPM list for this PB.
        let mpm = self.derive_mpm_list(x0, y0);
        let (prev_flag, mpm_idx, rem) = resolve_luma_mode_signalling(&mpm, best_mode);
        cw.encode_bin(&mut self.prev_intra_luma_pred_flag[0], prev_flag);
        if prev_flag == 1 {
            // mpm_idx truncated-rice cMax=2.
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
            // rem is 5 bypass bits, MSB first.
            for bit in (0..5).rev() {
                cw.encode_bypass((rem >> bit) & 1);
            }
        }
        self.store_luma_mode(x0, y0, CTU_SIZE, best_mode as u8);

        // intra_chroma_pred_mode = 0 (DM / inherit luma) — one context bin = 0.
        cw.encode_bin(&mut self.intra_chroma_pred_mode[0], 0);

        // Resolve chroma mode (ICPM=4 → DM → same as luma).
        let chroma_mode = resolve_chroma_mode(best_mode, 4);

        // --- transform_tree: one 16x16 luma, two 8x8 chroma -------------
        // log2_tb = 4 (16x16) for luma. No split. log2_tb > 2 so cbf_cb/cr signalled.
        // Build prediction + residual + quantised levels.
        let (luma_levels, luma_rec) = self.process_luma_tb(src, x0, y0, log2_tb, best_mode);
        let cbf_luma = luma_levels.iter().any(|&l| l != 0) as u32;
        // Chroma plane size: 8x8, log2_tb=3.
        let c_log2 = 3u32;
        let cx = x0 / 2;
        let cy = y0 / 2;
        let (cb_levels, cb_rec) = self.process_chroma_tb(src, cx, cy, c_log2, chroma_mode, false);
        let (cr_levels, cr_rec) = self.process_chroma_tb(src, cx, cy, c_log2, chroma_mode, true);
        let cbf_cb = cb_levels.iter().any(|&l| l != 0) as u32;
        let cbf_cr = cr_levels.iter().any(|&l| l != 0) as u32;

        // cbf_cb / cbf_cr (ctx_inc = 0 at tr_depth=0).
        cw.encode_bin(&mut self.residual.cbf_cb_cr[0], cbf_cb);
        cw.encode_bin(&mut self.residual.cbf_cb_cr[0], cbf_cr);
        // cbf_luma (ctx_inc = 1 at tr_depth=0).
        cw.encode_bin(&mut self.residual.cbf_luma[1], cbf_luma);

        // Residual coding order: luma first (as decoder does), then Cb, Cr.
        if cbf_luma == 1 {
            encode_residual(cw, &mut self.residual, &luma_levels, log2_tb, true);
        }
        if cbf_cb == 1 {
            encode_residual(cw, &mut self.residual, &cb_levels, c_log2, false);
        }
        if cbf_cr == 1 {
            encode_residual(cw, &mut self.residual, &cr_levels, c_log2, false);
        }

        // Write the locally-reconstructed block back to rec_* planes so
        // later CUs see the same neighbour data the decoder will.
        self.write_luma_block(x0, y0, CTU_SIZE as usize, &luma_rec);
        self.write_chroma_block(cx, cy, 1 << c_log2, &cb_rec, false);
        self.write_chroma_block(cx, cy, 1 << c_log2, &cr_rec, true);
        self.mark_decoded(x0, y0, CTU_SIZE);
    }

    /// Pick the luma intra mode with the lowest SAD against the source.
    fn choose_luma_mode(&self, src: &VideoFrame, x0: u32, y0: u32, log2_tb: u32) -> u32 {
        let n = 1usize << log2_tb;
        let refs = self.gather_refs_luma(x0, y0, n);
        let src_y = &src.planes[0];
        let mut best_mode = 1u32; // DC default
        let mut best_sad = u64::MAX;
        let mut pred = vec![0u8; n * n];
        for &mode in LUMA_CANDIDATES.iter() {
            // Apply MDIS per mode.
            let mut r = refs.clone();
            let (apply, strong) = filter_decision(log2_tb, mode, false, &r, n);
            if apply {
                filter_ref_samples(&mut r, n, strong);
            }
            predict(&r, n, &mut pred, n, mode, true);
            let sad = sad_block(&pred, n, &src_y.data, src_y.stride, x0 as usize, y0 as usize);
            if sad < best_sad {
                best_sad = sad;
                best_mode = mode;
            }
        }
        best_mode
    }

    /// Predict → residual → quant → (dequant+inverse+reconstruct). Returns
    /// `(levels, reconstructed)`.
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
        let (apply, strong) = filter_decision(log2_tb, mode, false, &refs, n);
        if apply {
            filter_ref_samples(&mut refs, n, strong);
        }
        let mut pred = vec![0u8; n * n];
        predict(&refs, n, &mut pred, n, mode, true);

        // Residual = src - pred.
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

        // Forward transform (DCT-II, since 16×16).
        let mut coeffs = vec![0i32; n * n];
        let is_dst = log2_tb == 2;
        forward_transform_2d(&residual, &mut coeffs, log2_tb, is_dst, 8);
        let mut levels = vec![0i32; n * n];
        quantize_flat(&coeffs, &mut levels, SLICE_QP_Y, log2_tb, 8);

        // Dequantise + inverse transform to reconstruct locally.
        let mut deq = vec![0i32; n * n];
        dequantize_flat(&levels, &mut deq, SLICE_QP_Y, log2_tb, 8);
        let mut res_rec = vec![0i32; n * n];
        inverse_transform_2d(&deq, &mut res_rec, log2_tb, is_dst, 8);
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
        // No MDIS for chroma in 4:2:0 Main.
        let mut pred = vec![0u8; n * n];
        predict(&refs, n, &mut pred, n, mode, false);
        let plane = if is_cr { &src.planes[2] } else { &src.planes[1] };
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

    fn gather_refs_luma(&self, x0: u32, y0: u32, n: usize) -> Vec<u8> {
        let len = 4 * n + 1;
        let mut samples = vec![128u8; len];
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
            samples[0] = plane[(y0 - 1) * stride + (x0 - 1)];
            avail[0] = true;
        }
        if y0 > 0 {
            for i in 0..(2 * n) {
                let xx = x0 + i;
                if xx < w && is_decoded(xx, y0 - 1) {
                    samples[1 + i] = plane[(y0 - 1) * stride + xx];
                    avail[1 + i] = true;
                }
            }
        }
        if x0 > 0 {
            for i in 0..(2 * n) {
                let yy = y0 + i;
                if yy < h && is_decoded(x0 - 1, yy) {
                    samples[2 * n + 1 + i] = plane[yy * stride + (x0 - 1)];
                    avail[2 * n + 1 + i] = true;
                }
            }
        }
        build_ref_samples(&samples, &avail, n)
    }

    fn gather_refs_chroma(&self, cx: u32, cy: u32, n: usize, is_cr: bool) -> Vec<u8> {
        let len = 4 * n + 1;
        let mut samples = vec![128u8; len];
        let mut avail = vec![false; len];
        let stride = self.c_stride;
        let plane = if is_cr { &self.rec_cr } else { &self.rec_cb };
        let cw = self.cfg.width as usize / 2;
        let ch = self.cfg.height as usize / 2;
        // Use luma-space decoded grid with shift by 1.
        let is_decoded_c = |x: usize, y: usize| -> bool {
            let lx = x << 1;
            let ly = y << 1;
            let bx = lx >> 2;
            let by = ly >> 2;
            bx < self.intra_w4 && by < self.intra_h4 && self.decoded_4x4[by * self.intra_w4 + bx]
        };
        let cx = cx as usize;
        let cy = cy as usize;
        if cx > 0 && cy > 0 && is_decoded_c(cx - 1, cy - 1) {
            samples[0] = plane[(cy - 1) * stride + (cx - 1)];
            avail[0] = true;
        }
        if cy > 0 {
            for i in 0..(2 * n) {
                let xx = cx + i;
                if xx < cw && is_decoded_c(xx, cy - 1) {
                    samples[1 + i] = plane[(cy - 1) * stride + xx];
                    avail[1 + i] = true;
                }
            }
        }
        if cx > 0 {
            for i in 0..(2 * n) {
                let yy = cy + i;
                if yy < ch && is_decoded_c(cx - 1, yy) {
                    samples[2 * n + 1 + i] = plane[yy * stride + (cx - 1)];
                    avail[2 * n + 1 + i] = true;
                }
            }
        }
        build_ref_samples(&samples, &avail, n)
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
        let b = if y0 == 0 { None } else { get_mode(x0, y0 - 1) };
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

    fn write_chroma_block(&mut self, x: u32, y: u32, n: usize, block: &[u8], is_cr: bool) {
        let x0 = x as usize;
        let y0 = y as usize;
        let cw = self.cfg.width as usize / 2;
        let ch = self.cfg.height as usize / 2;
        let plane = if is_cr { &mut self.rec_cr } else { &mut self.rec_cb };
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

/// Sum-absolute-difference between an n×n prediction and a region of the
/// source plane.
fn sad_block(pred: &[u8], n: usize, src: &[u8], stride: usize, x0: usize, y0: usize) -> u64 {
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
    // Matches decoder's `resolve_chroma_mode`.
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
    // Returns (prev_flag, mpm_idx, rem).
    for (i, &m) in mpm.iter().enumerate() {
        if m == mode {
            return (1, i as u32, 0);
        }
    }
    // Not in MPM → encode as `rem`. Decoder reconstructs:
    //   sorted = sort(mpm ascending); m = rem; for each sorted[i] <= m: m += 1;
    // The "rem" is such that after inserting the MPMs as forbidden positions,
    // we land on `mode`. Invert that.
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
    // QpY = 26, with pps/slice Cb/Cr offsets = 0 → Qp' = qp_y_to_qp_c(26) = 26.
    26
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{PixelFormat, TimeBase, VideoPlane};

    #[test]
    fn encoder_local_reconstruction_matches_input_ballpark() {
        // Encoder's private `process_luma_tb` — the reconstruction it
        // stores as neighbour data — should be close to the source for a
        // smooth gradient at QP 26.
        let cfg = EncoderConfig::new(16, 16);
        let state = EncoderState::new(&cfg);
        let mut y = vec![0u8; 16 * 16];
        for yy in 0..16 {
            for xx in 0..16 {
                y[yy * 16 + xx] = ((xx * 255) / 16) as u8;
            }
        }
        let cb = vec![128u8; 8 * 8];
        let cr = vec![128u8; 8 * 8];
        let src = VideoFrame {
            format: PixelFormat::Yuv420P,
            width: 16,
            height: 16,
            pts: Some(0),
            time_base: TimeBase::new(1, 30),
            planes: vec![
                VideoPlane {
                    stride: 16,
                    data: y.clone(),
                },
                VideoPlane { stride: 8, data: cb },
                VideoPlane { stride: 8, data: cr },
            ],
        };
        // Run the luma pipeline — prediction refs are unavailable (first
        // CTU), so DC defaults to 128 and the residual carries the full
        // gradient. The reconstruction should still be close to the
        // input.
        let (_levels, rec) = state.process_luma_tb(&src, 0, 0, 4, 1);
        let mut sse = 0u64;
        for i in 0..(16 * 16) {
            let d = y[i] as i32 - rec[i] as i32;
            sse += (d * d) as u64;
        }
        let mse = sse as f64 / (16.0 * 16.0);
        let psnr = 10.0 * (255.0f64 * 255.0 / mse).log10();
        eprintln!("local reconstruction psnr={psnr:.2} dB");
        assert!(psnr > 20.0, "local reconstruction PSNR {psnr:.2}");
    }

    #[test]
    fn luma_rem_roundtrip() {
        // For each mpm list + mode, the (prev_flag, mpm_idx, rem) we emit
        // must reconstruct back to the mode via the decoder's reorder rule.
        let lists: [[u32; 3]; 4] = [
            [0, 1, 26],
            [10, 26, 0],
            [18, 0, 1],
            [5, 10, 15],
        ];
        for mpm in &lists {
            for mode in 0u32..=34 {
                let (prev, idx, rem) = resolve_luma_mode_signalling(mpm, mode);
                let decoded = if prev == 1 {
                    mpm[idx as usize]
                } else {
                    let mut sorted = *mpm;
                    sorted.sort();
                    let mut m = rem;
                    if m >= sorted[0] {
                        m += 1;
                    }
                    if m >= sorted[1] {
                        m += 1;
                    }
                    if m >= sorted[2] {
                        m += 1;
                    }
                    m
                };
                assert_eq!(decoded, mode, "mpm={:?} mode={mode}", mpm);
            }
        }
    }
}
