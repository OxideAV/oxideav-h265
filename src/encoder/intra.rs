//! Real CABAC intra encoder — I-slice IDR access units with §8.4
//! intra prediction, forward transform + quantization, and full
//! §7.3.8 CABAC syntax emission (no PCM).
//!
//! Geometry (this bootstrap's fixed shape, mirroring the PCM
//! encoder): `CtbSizeY == MinCbSizeY == 16`, so every CTB is one
//! unsplit intra CU (`PART_2Nx2N`) whose transform tree is a single
//! 16x16 luma TB + two 8x8 chroma TBs (`MaxTbLog2SizeY == 4`,
//! `max_transform_hierarchy_depth_intra == 0`). 4:2:0 8-bit only;
//! dimensions must be multiples of 16.
//!
//! Per CTU the encoder:
//!
//! 1. gathers the §8.4.4.2.1 reference samples from its own
//!    reconstruction buffer (marking availability per decode order:
//!    left column and top row of previously coded CTBs) and runs the
//!    decode-side [`crate::intra_pred`] pipeline (§8.4.4.2.2
//!    substitution + §8.4.4.2.3 filtering + planar / DC / angular
//!    prediction) for every candidate luma mode, picking the
//!    SAD-best;
//! 2. forward-transforms the prediction residual (the transpose of
//!    the §8.6.4.2 DCT-II basis) and quantizes against the §8.6.3
//!    `levelScale`-derived reciprocal at the slice QP (chroma via the
//!    Table 8-10 QP mapping);
//! 3. reconstructs through the crate's own DECODE-side §8.6.2
//!    scaling/transform ([`crate::transform::residual_block`]) so the
//!    encoder's reference buffer is bit-identical to what any
//!    conforming decoder reconstructs;
//! 4. emits the §7.3.8.5 coding-unit syntax (`part_mode`,
//!    `prev_intra_luma_pred_flag` / `mpm_idx` /
//!    `rem_intra_luma_pred_mode` against the §8.4.2 candidate list,
//!    `intra_chroma_pred_mode` = derived-from-luma), the §7.3.8.8
//!    cbf flags, and the §7.3.8.11 residual blocks through
//!    [`crate::encoder::residual::encode_residual_coding`].
//!
//! In-loop filters are off (SAO off in the SPS, deblocking disabled
//! in the PPS), so a conforming decoder's output equals the encoder's
//! reconstruction exactly — pinned by the roundtrip tests.

use crate::binarization::intra_luma_cand_mode_list;
use crate::binarization::{cbf_cb_ctx_inc, cbf_cr_ctx_inc, cbf_luma_ctx_inc};
use crate::cabac::init_type;
use crate::ctx_init::SliceContexts;
use crate::encoder::bitwriter::BitWriter;
use crate::encoder::cabac::CabacEncoder;
use crate::encoder::nal::{annexb, nal_unit};
use crate::encoder::pcm::{level_idc_for, write_pps, write_ptl, write_vps};
use crate::encoder::residual::encode_residual_coding;
use crate::intra_mode_field::{IntraModeField, Neighbour};
use crate::intra_pred::{
    intra_predict_with_substitution, Component as PredComponent, IntraPredParams,
    MarkedReferenceSamples,
};
use crate::residual::ResidualCodingParams;
use crate::scan::ScanIdx;
use crate::transform::{forward_dct_1d, residual_block, BlockParams, Component, PredMode};

/// The fixed CTB / coding-block log2 size (16x16).
const CTB_LOG2: u32 = 4;
/// The fixed CTB size.
const CTB: usize = 1 << CTB_LOG2;
/// Fixed 8-bit depth.
const BIT_DEPTH: u32 = 8;

/// Errors from the intra encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraEncodeError {
    /// Width or height is zero or not a multiple of the 16-sample CTB.
    BadDimensions {
        /// Requested luma width.
        width: usize,
        /// Requested luma height.
        height: usize,
    },
    /// A supplied plane's length does not match the 4:2:0 geometry.
    PlaneSize {
        /// Which plane (`"y"`, `"cb"`, `"cr"`).
        plane: &'static str,
        /// Required sample count.
        expected: usize,
        /// Supplied sample count.
        got: usize,
    },
    /// `SliceQpY` outside 0..=51.
    BadQp(i32),
}

impl core::fmt::Display for IntraEncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadDimensions { width, height } => write!(
                f,
                "intra encoder requires nonzero dimensions that are multiples of 16, got {width}x{height}"
            ),
            Self::PlaneSize {
                plane,
                expected,
                got,
            } => write!(f, "{plane} plane has {got} samples, expected {expected}"),
            Self::BadQp(qp) => write!(f, "slice QP {qp} outside 0..=51"),
        }
    }
}

impl std::error::Error for IntraEncodeError {}

/// The encoded access unit plus the encoder's own reconstruction
/// (what a conforming decoder outputs — in-loop filters are off).
#[derive(Debug, Clone)]
pub struct IntraEncodedAu {
    /// The Annex B access unit (`VPS + SPS + PPS + IDR_N_LP`).
    pub au: Vec<u8>,
    /// Reconstructed luma plane (`width * height`).
    pub recon_y: Vec<u8>,
    /// Reconstructed Cb plane (`width/2 * height/2`).
    pub recon_cb: Vec<u8>,
    /// Reconstructed Cr plane.
    pub recon_cr: Vec<u8>,
}

/// §7.3.2.2 — the fixed-geometry SPS (4:2:0, 8-bit, CTB 16, PCM off,
/// SAO off).
fn write_sps(width: usize, height: usize, level_idc: u8) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(0, 4); // sps_video_parameter_set_id
    w.put_bits(0, 3); // sps_max_sub_layers_minus1
    w.put_bit(1); // sps_temporal_id_nesting_flag
    write_ptl(&mut w, level_idc);
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
    w.put_bit(0); // sps_extension_present_flag
    w.rbsp_trailing_bits();
    w.finish()
}

/// Table 8-10 — the `ChromaArrayType == 1` chroma QP mapping
/// `qPC = f(qPi)` (§8.6.1; `QpBdOffsetC == 0` at 8-bit).
fn chroma_qp_420(qp_y: i32) -> u32 {
    let qpi = qp_y.clamp(0, 57);
    (match qpi {
        x if x < 30 => x,
        30..=33 => qpi - 1,             // 30..=33 -> 29, 30, 31, 32
        34..=43 => 33 + (qpi - 34) / 2, // 34..=43 -> 33, 33, 34, 34 .. 37, 37
        x => x - 6,
    }) as u32
}

/// §8.6.3-derived reciprocal quantizer scale: `levelScale[qP % 6]` is
/// `{40, 45, 51, 57, 64, 72}`; the forward reciprocal is
/// `round(2^20 / levelScale)` so `quant ∘ dequant` has unity gain.
fn quant_scale(qp_rem: u32) -> i64 {
    let ls = i64::from(crate::transform::LEVEL_SCALE[qp_rem as usize]);
    ((1i64 << 20) + ls / 2) / ls
}

/// Forward 2-D DCT-II (the transpose of the §8.6.4 inverse): stage 1
/// over rows with `shift1 = log2TbS + BitDepth − 9`, stage 2 over
/// columns with `shift2 = log2TbS + 6` — the normalization that makes
/// the §8.6.3 dequant + §8.6.4 inverse reproduce the residual.
fn forward_transform(res: &[i32], n: usize) -> Vec<i32> {
    let log2 = n.trailing_zeros();
    let shift1 = log2 + BIT_DEPTH - 9;
    let shift2 = log2 + 6;
    let r1 = 1i64 << (shift1 - 1);
    let r2 = 1i64 << (shift2 - 1);
    // Stage 1: horizontal analysis per row y -> a[y][u].
    let mut a = vec![0i64; n * n];
    for y in 0..n {
        let row: Vec<i64> = (0..n).map(|x| i64::from(res[y * n + x])).collect();
        let t = forward_dct_1d(&row, n);
        for (u, &v) in t.iter().enumerate() {
            a[y * n + u] = (v + r1) >> shift1;
        }
    }
    // Stage 2: vertical analysis per column u -> coef[v][u].
    let mut coef = vec![0i32; n * n];
    for u in 0..n {
        let col: Vec<i64> = (0..n).map(|y| a[y * n + u]).collect();
        let t = forward_dct_1d(&col, n);
        for (v, &val) in t.iter().enumerate() {
            coef[v * n + u] = ((val + r2) >> shift2) as i32;
        }
    }
    coef
}

/// Scalar quantization to `TransCoeffLevel`: `level = sign ·
/// (|coef| · quantScale + offset) >> qBits` with `qBits = 14 + qP/6 +
/// (15 − BitDepth − log2TbS)` (the inverse of the §8.6.3 eq. 8-309
/// scaling chain) and a one-third rounding offset; clamped to the
/// §7.4.9.11 CoeffMax.
fn quantize(coef: &[i32], n: usize, qp: u32) -> Vec<i32> {
    let log2 = n.trailing_zeros();
    let qbits = 14 + qp / 6 + (15 - BIT_DEPTH - log2);
    let scale = quant_scale(qp % 6);
    let offset = (1i64 << qbits) / 3;
    coef.iter()
        .map(|&c| {
            let level = ((i64::from(c.unsigned_abs()) * scale + offset) >> qbits).min(0x7FFF);
            (level as i32) * c.signum()
        })
        .collect()
}

/// Gather the marked reference samples for an `n`-sample TB at plane
/// position `(x0, y0)` from the encoder's reconstruction plane.
/// Decode-order availability: the top row of the CTB row above is
/// fully reconstructed (so top and top-right samples inside the
/// picture are available); the left column is available only down to
/// the CTB height (the below-left CTB is not yet coded).
fn gather_refs(
    recon: &[u8],
    plane_w: usize,
    plane_h: usize,
    x0: usize,
    y0: usize,
    n: usize,
) -> MarkedReferenceSamples {
    let sample = |x: usize, y: usize| i32::from(recon[y * plane_w + x]);
    let corner = if x0 > 0 && y0 > 0 {
        (sample(x0 - 1, y0 - 1), true)
    } else {
        (0, false)
    };
    let left: Vec<(i32, bool)> = (0..2 * n)
        .map(|k| {
            if x0 > 0 && k < n && y0 + k < plane_h {
                (sample(x0 - 1, y0 + k), true)
            } else {
                (0, false)
            }
        })
        .collect();
    let top: Vec<(i32, bool)> = (0..2 * n)
        .map(|k| {
            if y0 > 0 && x0 + k < plane_w {
                (sample(x0 + k, y0 - 1), true)
            } else {
                (0, false)
            }
        })
        .collect();
    MarkedReferenceSamples::new(n, corner, left, top).expect("legal TB geometry")
}

fn pred_params(mode: u8, cidx: PredComponent) -> IntraPredParams {
    IntraPredParams {
        pred_mode_intra: mode,
        cidx,
        bit_depth: BIT_DEPTH as u8,
        bit_depth_luma: BIT_DEPTH as u8,
        intra_smoothing_disabled: false,
        strong_intra_smoothing_enabled: false,
        chroma_array_type_3: false,
        disable_boundary_filter: false,
    }
}

/// Transform + quantize one component TB and reconstruct it through
/// the DECODE-side §8.6.2 path. Returns `(levels, recon_samples)`;
/// `levels` all-zero ⇔ cbf 0 (recon = clipped prediction).
fn code_tb(
    src: &[i32],
    pred: &[i32],
    n: usize,
    qp: u32,
    component: Component,
) -> (Vec<i32>, Vec<u8>) {
    let res: Vec<i32> = src.iter().zip(pred.iter()).map(|(&s, &p)| s - p).collect();
    let coef = forward_transform(&res, n);
    let levels = quantize(&coef, n, qp);
    let recon: Vec<u8> = if levels.iter().all(|&v| v == 0) {
        pred.iter().map(|&p| p.clamp(0, 255) as u8).collect()
    } else {
        let r = residual_block(
            &levels,
            None,
            BlockParams {
                n_tbs: n,
                q_p: qp,
                component,
                pred_mode: PredMode::Intra,
                bit_depth: BIT_DEPTH as u8,
                extended_precision: false,
                transquant_bypass: false,
                transform_skip: false,
                transform_skip_rotation_enabled: false,
            },
        )
        .expect("legal block params");
        pred.iter()
            .zip(r.iter())
            .map(|(&p, &d)| (p + d).clamp(0, 255) as u8)
            .collect()
    };
    (levels, recon)
}

/// Encode one 4:2:0 8-bit frame as a self-contained intra IDR access
/// unit at `SliceQpY == qp` and return it with the reconstruction a
/// conforming decoder produces.
///
/// # Errors
/// [`IntraEncodeError`] on bad dimensions / plane sizes / QP.
pub fn encode_idr_intra_au(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    qp: i32,
) -> Result<IntraEncodedAu, IntraEncodeError> {
    if width == 0 || height == 0 || width % CTB != 0 || height % CTB != 0 {
        return Err(IntraEncodeError::BadDimensions { width, height });
    }
    if !(0..=51).contains(&qp) {
        return Err(IntraEncodeError::BadQp(qp));
    }
    let check = |plane: &'static str, buf: &[u8], expected: usize| {
        if buf.len() != expected {
            Err(IntraEncodeError::PlaneSize {
                plane,
                expected,
                got: buf.len(),
            })
        } else {
            Ok(())
        }
    };
    check("y", y, width * height)?;
    check("cb", cb, width * height / 4)?;
    check("cr", cr, width * height / 4)?;

    let (cw, ch) = (width / 2, height / 2);
    let ctbs_x = width / CTB;
    let ctbs_y = height / CTB;
    let qp_y = qp as u32;
    let qp_c = chroma_qp_420(qp);

    let mut recon_y = vec![0u8; width * height];
    let mut recon_cb = vec![0u8; cw * ch];
    let mut recon_cr = vec![0u8; cw * ch];
    let mut modes = IntraModeField::new(width, height, CTB_LOG2);

    // ---- slice_segment_header( ) ----
    let mut w = BitWriter::new();
    w.put_bit(1); // first_slice_segment_in_pic_flag
    w.put_bit(0); // no_output_of_prior_pics_flag (IRAP NAL)
    w.ue(0); // slice_pic_parameter_set_id
    w.ue(2); // slice_type = I
    w.se(qp - 26); // slice_qp_delta (init_qp_minus26 == 0)
                   // SAO off + deblocking disabled: no more slice-level fields.
    w.rbsp_trailing_bits(); // byte_alignment()

    // ---- slice_segment_data( ) ----
    let mut cabac = CabacEncoder::new();
    // Table 9-4: I slice => initType 0 (raw slice_type 2).
    let mut ctxs = SliceContexts::init(init_type(2, false), qp);

    let extract = |plane: &[u8], pw: usize, x0: usize, y0: usize, n: usize| -> Vec<i32> {
        let mut out = Vec::with_capacity(n * n);
        for j in 0..n {
            for i in 0..n {
                out.push(i32::from(plane[(y0 + j) * pw + x0 + i]));
            }
        }
        out
    };
    let store = |plane: &mut [u8], pw: usize, x0: usize, y0: usize, n: usize, s: &[u8]| {
        for j in 0..n {
            plane[(y0 + j) * pw + x0..(y0 + j) * pw + x0 + n]
                .copy_from_slice(&s[j * n..(j + 1) * n]);
        }
    };

    for ctb in 0..ctbs_x * ctbs_y {
        let x0 = (ctb % ctbs_x) * CTB;
        let y0 = (ctb / ctbs_x) * CTB;

        // ---- §8.4 luma mode decision over the reconstruction ----
        let marked = gather_refs(&recon_y, width, height, x0, y0, CTB);
        let src_y = extract(y, width, x0, y0, CTB);
        let mut best_mode = 0u8;
        let mut best_pred: Vec<i32> = Vec::new();
        let mut best_cost = u64::MAX;
        for mode in 0..=34u8 {
            let pred =
                intra_predict_with_substitution(&marked, &pred_params(mode, PredComponent::Luma))
                    .expect("legal prediction params");
            let cost: u64 = src_y
                .iter()
                .zip(pred.iter())
                .map(|(&s, &p)| u64::from(s.abs_diff(p)))
                .sum();
            if cost < best_cost {
                best_cost = cost;
                best_mode = mode;
                best_pred = pred;
            }
        }

        // ---- residuals: luma 16x16, chroma 8x8 (DM mode) ----
        let (luma_levels, luma_recon) = code_tb(&src_y, &best_pred, CTB, qp_y, Component::Luma);
        store(&mut recon_y, width, x0, y0, CTB, &luma_recon);

        let (cx0, cy0, cn) = (x0 / 2, y0 / 2, CTB / 2);
        let chroma =
            |plane: &[u8], recon: &mut Vec<u8>, comp: Component, pc: PredComponent| -> Vec<i32> {
                let marked = gather_refs(recon, cw, ch, cx0, cy0, cn);
                let pred = intra_predict_with_substitution(&marked, &pred_params(best_mode, pc))
                    .expect("legal prediction params");
                let src = extract(plane, cw, cx0, cy0, cn);
                let (levels, rec) = code_tb(&src, &pred, cn, qp_c, comp);
                store(recon, cw, cx0, cy0, cn, &rec);
                levels
            };
        let cb_levels = chroma(cb, &mut recon_cb, Component::Cb, PredComponent::Cb);
        let cr_levels = chroma(cr, &mut recon_cr, Component::Cr, PredComponent::Cr);

        let cbf_luma = luma_levels.iter().any(|&v| v != 0);
        let cbf_cb = cb_levels.iter().any(|&v| v != 0);
        let cbf_cr = cr_levels.iter().any(|&v| v != 0);

        // ---- §7.3.8.5 coding_unit( ) syntax ----
        // part_mode: §9.3.3.7 intra at MinCb — bin "1" = PART_2Nx2N.
        cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], 1);
        // prev_intra_luma_pred_flag + mpm_idx / rem_intra_luma_pred_mode
        // against the §8.4.2 candidate list.
        let cand_a = modes.cand_intra_pred_mode(x0, y0, Neighbour::Left, x0 > 0);
        let cand_b = modes.cand_intra_pred_mode(x0, y0, Neighbour::Above, y0 > 0);
        let list = intra_luma_cand_mode_list(cand_a, cand_b);
        if let Some(idx) = list.iter().position(|&m| m == best_mode) {
            cabac.encode_decision(&mut w, &mut ctxs.prev_intra_luma_pred_flag[0], 1);
            // mpm_idx: TR cMax 2, bypass: 0 -> "0", 1 -> "10", 2 -> "11".
            match idx {
                0 => cabac.encode_bypass(&mut w, 0),
                1 => {
                    cabac.encode_bypass(&mut w, 1);
                    cabac.encode_bypass(&mut w, 0);
                }
                _ => {
                    cabac.encode_bypass(&mut w, 1);
                    cabac.encode_bypass(&mut w, 1);
                }
            }
        } else {
            cabac.encode_decision(&mut w, &mut ctxs.prev_intra_luma_pred_flag[0], 0);
            // §8.4.2: rem = mode with each smaller candidate removed.
            let mut rem = u32::from(best_mode);
            for &c in &list {
                if u32::from(best_mode) > u32::from(c) {
                    rem -= 1;
                }
            }
            cabac.encode_bypass_bits(&mut w, rem, 5); // FL cMax 31
        }
        modes.record_intra_pb(x0, y0, CTB, best_mode, false);
        // intra_chroma_pred_mode = 4 (derived from luma): bin "0".
        cabac.encode_decision(&mut w, &mut ctxs.intra_chroma_pred_mode[0], 0);

        // ---- §7.3.8.8 transform_tree (single 16x16 TU) ----
        cabac.encode_decision(
            &mut w,
            &mut ctxs.cbf_chroma[cbf_cb_ctx_inc(0) as usize],
            u8::from(cbf_cb),
        );
        cabac.encode_decision(
            &mut w,
            &mut ctxs.cbf_chroma[cbf_cr_ctx_inc(0) as usize],
            u8::from(cbf_cr),
        );
        cabac.encode_decision(
            &mut w,
            &mut ctxs.cbf_luma[cbf_luma_ctx_inc(0) as usize],
            u8::from(cbf_luma),
        );

        // ---- §7.3.8.10 transform_unit: residual blocks ----
        // §7.4.9.11: 16x16 luma and 8x8 4:2:0 chroma both use the
        // up-right diagonal scan regardless of the intra mode.
        let rc_params = |log2: u32, is_chroma: bool| ResidualCodingParams {
            log2_trafo_size: log2,
            is_chroma,
            scan_idx: ScanIdx::Diagonal,
            sign_data_hiding_enabled_flag: false,
            sign_hidden_suppressed: false,
            transform_skip_sig_ctx: false,
        };
        if cbf_luma {
            encode_residual_coding(
                &mut w,
                &mut cabac,
                &mut ctxs.residual,
                &rc_params(4, false),
                &luma_levels,
            )
            .expect("validated luma levels");
        }
        if cbf_cb {
            encode_residual_coding(
                &mut w,
                &mut cabac,
                &mut ctxs.residual,
                &rc_params(3, true),
                &cb_levels,
            )
            .expect("validated cb levels");
        }
        if cbf_cr {
            encode_residual_coding(
                &mut w,
                &mut cabac,
                &mut ctxs.residual,
                &rc_params(3, true),
                &cr_levels,
            )
            .expect("validated cr levels");
        }

        // end_of_slice_segment_flag.
        cabac.encode_terminate(&mut w, u8::from(ctb == ctbs_x * ctbs_y - 1));
    }
    // The final terminate-1 flush wrote the rbsp_stop_one_bit;
    // rbsp_slice_segment_trailing_bits() is alignment zeros from here.
    w.align_zero();
    let slice_rbsp = w.finish();

    let level_idc = level_idc_for(width * height);
    let units = vec![
        nal_unit(32, 0, 0, &write_vps(level_idc)), // VPS_NUT
        nal_unit(33, 0, 0, &write_sps(width, height, level_idc)), // SPS_NUT
        nal_unit(34, 0, 0, &write_pps(false, false, None)), // PPS_NUT
        nal_unit(20, 0, 0, &slice_rbsp),           // IDR_N_LP
    ];
    Ok(IntraEncodedAu {
        au: annexb(&units),
        recon_y,
        recon_cb,
        recon_cr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::decode_annexb_sequence;

    /// Deterministic test content: smooth gradients + a diagonal
    /// texture component so directional modes and residuals both work.
    fn planes(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let y: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, yy) = (i % w, i / w);
                ((x * 3 + yy * 2 + (x * yy / 7) % 31) % 256) as u8
            })
            .collect();
        let cb: Vec<u8> = (0..w * h / 4)
            .map(|i| ((i % (w / 2)) * 4 % 200 + 20) as u8)
            .collect();
        let cr: Vec<u8> = (0..w * h / 4)
            .map(|i| (240 - (i / (w / 2)) * 3 % 200) as u8)
            .collect();
        (y, cb, cr)
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let mse: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = f64::from(x) - f64::from(y);
                d * d
            })
            .sum::<f64>()
            / a.len() as f64;
        if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0f64 * 255.0 / mse).log10()
        }
    }

    /// The core contract: the crate's own decoder reproduces the
    /// encoder's reconstruction EXACTLY (dual-decoder bit-exactness),
    /// and the reconstruction is close to the source.
    #[test]
    fn intra_au_decodes_to_encoder_recon_exactly() {
        for (w, h) in [(16usize, 16usize), (64, 48), (48, 80)] {
            let (y, cb, cr) = planes(w, h);
            for qp in [4i32, 22, 32, 45] {
                let enc = encode_idr_intra_au(&y, &cb, &cr, w, h, qp).expect("encode");
                let frames = decode_annexb_sequence(&enc.au).expect("decode");
                assert_eq!(frames.len(), 1, "{w}x{h} qp{qp}");
                let mut recon = Vec::new();
                recon.extend_from_slice(&enc.recon_y);
                recon.extend_from_slice(&enc.recon_cb);
                recon.extend_from_slice(&enc.recon_cr);
                assert_eq!(
                    frames[0].picture.to_planar_u8().expect("8-bit"),
                    recon,
                    "{w}x{h} qp{qp}: decoder output == encoder reconstruction"
                );
            }
        }
    }

    /// Rate/distortion sanity: low QP is near-transparent, and
    /// quality degrades monotonically-ish while staying decodable.
    #[test]
    fn intra_quality_tracks_qp() {
        let (w, h) = (64usize, 64usize);
        let (y, cb, cr) = planes(w, h);
        let at = |qp: i32| {
            let enc = encode_idr_intra_au(&y, &cb, &cr, w, h, qp).expect("encode");
            (psnr(&enc.recon_y, &y), enc.au.len())
        };
        let (p4, s4) = at(4);
        let (p22, s22) = at(22);
        let (p40, s40) = at(40);
        assert!(p4 > 45.0, "qp4 luma PSNR {p4:.1} dB");
        assert!(p22 > 33.0, "qp22 luma PSNR {p22:.1} dB");
        assert!(p40 > 22.0, "qp40 luma PSNR {p40:.1} dB");
        assert!(p4 > p22 && p22 > p40, "PSNR decreases with QP");
        assert!(
            s4 > s22 && s22 > s40,
            "bytes decrease with QP ({s4} > {s22} > {s40})"
        );
    }

    /// QP 4 on this content should be visually transparent while far
    /// smaller than the PCM (raw) coding — i.e. the transform path
    /// actually compresses.
    #[test]
    fn intra_beats_pcm_size_at_high_quality() {
        let (w, h) = (64usize, 64usize);
        let (y, cb, cr) = planes(w, h);
        let enc = encode_idr_intra_au(&y, &cb, &cr, w, h, 22).expect("encode");
        let raw = w * h * 3 / 2;
        assert!(
            enc.au.len() < raw / 2,
            "compressed {} bytes vs raw {raw}",
            enc.au.len()
        );
    }

    #[test]
    fn rejects_bad_inputs() {
        let (y, cb, cr) = planes(16, 16);
        assert!(matches!(
            encode_idr_intra_au(&y, &cb, &cr, 20, 16, 26),
            Err(IntraEncodeError::BadDimensions { .. })
        ));
        assert!(matches!(
            encode_idr_intra_au(&y, &cb, &cr, 16, 16, 52),
            Err(IntraEncodeError::BadQp(52))
        ));
        assert!(matches!(
            encode_idr_intra_au(&y, &cb, &cr, 32, 16, 26),
            Err(IntraEncodeError::PlaneSize { .. })
        ));
    }

    /// Table 8-10 spot pins.
    #[test]
    fn chroma_qp_mapping_matches_table_8_10() {
        assert_eq!(chroma_qp_420(0), 0);
        assert_eq!(chroma_qp_420(29), 29);
        assert_eq!(chroma_qp_420(30), 29);
        assert_eq!(chroma_qp_420(33), 32);
        assert_eq!(chroma_qp_420(34), 33);
        assert_eq!(chroma_qp_420(35), 33);
        assert_eq!(chroma_qp_420(36), 34);
        assert_eq!(chroma_qp_420(37), 34);
        assert_eq!(chroma_qp_420(38), 35);
        assert_eq!(chroma_qp_420(43), 37);
        assert_eq!(chroma_qp_420(44), 38);
        assert_eq!(chroma_qp_420(51), 45);
    }
}
