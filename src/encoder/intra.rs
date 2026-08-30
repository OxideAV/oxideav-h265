//! Real CABAC intra encoder — I-slice IDR access units with §8.4
//! intra prediction, forward transform + quantization, and full
//! §7.3.8 CABAC syntax emission (no PCM).
//!
//! Geometry (this bootstrap's fixed shape, mirroring the PCM
//! encoder): `CtbSizeY == MinCbSizeY == 16`, so every CTB is one
//! unsplit intra CU. Two partition modes compete per CU
//! (rate-distortion heuristic):
//!
//! * `PART_2Nx2N` — one 16x16 luma PB/TB + two 8x8 chroma TBs;
//! * `PART_NxN` — four 8x8 luma PBs, each its own mode; the §7.4.9.8
//!   `IntraSplitFlag` forces the transform tree to depth 1 (four
//!   8x8 luma TBs + four 4x4 chroma TBs per plane), the 8x8 luma and
//!   4x4 chroma TBs picking their §7.4.9.11 mode-dependent scans.
//!
//! Per CTU the encoder:
//!
//! 1. gathers the §8.4.4.2.1 reference samples from its own
//!    reconstruction buffer, marking availability per the §6.4.1
//!    z-scan decode order (CTB raster + TU z-order within the CTB),
//!    and runs the decode-side [`crate::intra_pred`] pipeline
//!    (§8.4.4.2.2 substitution + §8.4.4.2.3 filtering + planar / DC /
//!    angular prediction) for every candidate mode, picking the
//!    SAD-best per PB;
//! 2. forward-transforms the prediction residual (the transpose of
//!    the §8.6.4.2 DCT-II basis) and quantizes against the §8.6.3
//!    `levelScale`-derived reciprocal at the slice QP (chroma via the
//!    Table 8-10 QP mapping);
//! 3. reconstructs through the crate's own DECODE-side §8.6.2
//!    scaling/transform ([`crate::transform::residual_block`]) so the
//!    encoder's reference buffer is bit-identical to what a
//!    conforming decoder reconstructs, and picks the partition with
//!    the smaller SSD + partition-cost heuristic;
//! 4. emits the §7.3.8.5 coding-unit syntax (`part_mode`, the
//!    §7.3.8.5 two-loop `prev_intra_luma_pred_flag[]` then
//!    `mpm_idx` / `rem_intra_luma_pred_mode` group against the
//!    §8.4.2 candidate lists, `intra_chroma_pred_mode` =
//!    derived-from-luma), the §7.3.8.8 transform tree with its cbf
//!    inheritance, and the §7.3.8.11 residual blocks through
//!    [`crate::encoder::residual::encode_residual_coding`].
//!
//! In-loop filters are off (SAO off in the SPS, deblocking disabled
//! in the PPS), so a conforming decoder's output equals the encoder's
//! reconstruction exactly — pinned by the roundtrip tests.

use crate::binarization::intra_luma_cand_mode_list;
use crate::binarization::PartMode;
use crate::binarization::{cbf_cb_ctx_inc, cbf_cr_ctx_inc, cbf_luma_ctx_inc};
use crate::cabac::init_type;
use crate::ctx_init::SliceContexts;
use crate::encoder::bitwriter::BitWriter;
use crate::encoder::cabac::CabacEncoder;
use crate::encoder::loopfilter::{
    encode_sao_ctb, filter_frame, CtbShape, FilterInput, LoopFilterCfg,
};
use crate::encoder::nal::{annexb, nal_unit};
use crate::encoder::pcm::{level_idc_for, write_pps_full, write_ptl, write_vps, write_vps_cfg};
use crate::encoder::residual::encode_residual_coding;
use crate::intra_mode_field::{IntraModeField, Neighbour};
use crate::intra_pred::{
    intra_predict_with_substitution, Component as PredComponent, IntraPredParams,
    MarkedReferenceSamples,
};
use crate::motion::MotionField;
use crate::residual::{residual_coding_scan_idx, ResidualCodingParams};
use crate::slice_data::SaoCtbParams;
use crate::transform::{forward_dct_1d, residual_block, BlockParams, Component, PredMode};

/// The fixed CTB / coding-block log2 size (16x16).
const CTB_LOG2: u32 = 4;
/// The fixed CTB size.
const CTB: usize = 1 << CTB_LOG2;
/// Fixed 8-bit depth.
const BIT_DEPTH: u32 = 8;
/// The z-order offsets of the four NxN prediction blocks / depth-1
/// transform units within a CTB (§6.5.2 z-scan of the four halves).
const Z_OFFSETS: [(usize, usize); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];

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
    /// Adaptive-quantization strength outside 0..=3.
    BadAq(u8),
    /// HRD signalling enabled without its prerequisites (rate
    /// control with a VBV buffer and an explicit frame rate).
    HrdConfig,
    /// CBR delivery needs a CPB of at least two elemental ticks of
    /// the signalled bit rate (the filler quantum must fit between
    /// the overflow floor and the underflow cap).
    CbrCpbTooSmall,
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
            Self::BadAq(aq) => write!(f, "aq strength {aq} outside 0..=3"),
            Self::HrdConfig => f.write_str(
                "hrd signalling requires rate control with a VBV buffer and an explicit frame rate",
            ),
            Self::CbrCpbTooSmall => f.write_str(
                "cbr delivery requires a VBV buffer of at least two frame intervals of the bitrate",
            ),
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

/// The stream-level geometry / buffering knobs of the shared SPS
/// (everything the intra, low-delay and hierarchical-B encoders vary
/// between them; the rest of the SPS is fixed 4:2:0 8-bit CTB-16).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SpsCfg {
    /// `sps_max_dec_pic_buffering_minus1[0]`.
    pub max_dec_pic_buffering_minus1: u32,
    /// `sps_max_num_reorder_pics[0]` (nonzero for the out-of-order
    /// hierarchical-B GOPs).
    pub max_num_reorder_pics: u32,
    /// `MinCbLog2SizeY` (4 for the legacy CTB == CU geometry; 3 for
    /// the AMP-enabled geometry, where every CTB is an UNSPLIT 16x16
    /// CU with `log2CbSize > MinCbLog2SizeY`).
    pub min_cb_log2: u32,
    /// `amp_enabled_flag` (requires `min_cb_log2 == 3` — Table 9-45
    /// only reaches the AMP bin strings when `log2CbSize >
    /// MinCbLog2SizeY`).
    pub amp: bool,
    /// PPS `cu_qp_delta_enabled_flag` (with `diff_cu_qp_delta_depth
    /// == 0`: one quantization group per CTB) — set by the
    /// adaptive-quantization configurations.
    pub cu_qp_delta: bool,
    /// §E.2.1 VUI timing declaration:
    /// `(vui_num_units_in_tick, vui_time_scale)` — i.e. `(fps_den,
    /// fps_num)`. `None` omits the VUI entirely (the historical
    /// streams).
    pub timing: Option<(u32, u32)>,
    /// §E.2.2 HRD delivery schedule to declare inside the VUI
    /// (requires `timing`); `None` keeps the VUI HRD-free.
    pub hrd: Option<crate::encoder::hrd::HrdSignalCfg>,
    /// The quadtree-coder geometry ([`crate::encoder::ctu::TreeCfg`]):
    /// `Some` routes the picture through the recursive coding-quadtree
    /// coder (CTB 16 / 32 / 64, `MinCbSizeY == 8`, RD-elected splits,
    /// deeper RQTs, 4x4 DST intra TUs). `None` keeps the historical
    /// fixed-geometry coders byte-stable. When set, `min_cb_log2`
    /// must be 3 (the quadtree streams always signal MinCb 8).
    pub tree: Option<crate::encoder::ctu::TreeCfg>,
}

impl SpsCfg {
    /// The legacy fixed geometry: `MinCbSizeY == CtbSizeY == 16`, no
    /// AMP, no output reordering.
    pub(crate) fn legacy(max_dec_pic_buffering_minus1: u32) -> Self {
        Self {
            max_dec_pic_buffering_minus1,
            max_num_reorder_pics: 0,
            min_cb_log2: CTB_LOG2,
            amp: false,
            cu_qp_delta: false,
            timing: None,
            hrd: None,
            tree: None,
        }
    }
}

/// §7.3.2.2 — the fixed-geometry SPS (4:2:0, 8-bit, CTB 16, PCM off,
/// SAO per `sao_enabled`) with the [`SpsCfg`] knob set (coding-block
/// geometry, `amp_enabled_flag`, DPB / reorder bounds). Shared by the
/// intra, low-delay and hierarchical-B encoders (nothing in it is
/// slice-type specific: the P / B slices code their §7.4.8 short-term
/// RPS inline and `sps_temporal_mvp_enabled_flag` is 0).
pub(crate) fn write_sps_cfg(
    width: usize,
    height: usize,
    level_idc: u8,
    cfg: &SpsCfg,
    sao_enabled: bool,
) -> Vec<u8> {
    let ctb_log2 = cfg.tree.map_or(CTB_LOG2, |t| t.ctb_log2);
    let max_tb_log2 = cfg.tree.map_or(CTB_LOG2, |t| t.max_tb_log2());
    let (th_intra, th_inter) = cfg
        .tree
        .map_or((0, 0), |t| (t.th_depth_intra, t.th_depth_inter));
    debug_assert!(
        !cfg.amp || cfg.min_cb_log2 < ctb_log2,
        "AMP requires log2CbSize > MinCbLog2SizeY (Table 9-45)"
    );
    debug_assert!(
        cfg.tree.is_none() || cfg.min_cb_log2 == 3,
        "quadtree streams signal MinCbSizeY == 8"
    );
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
    w.ue(cfg.max_dec_pic_buffering_minus1); // sps_max_dec_pic_buffering_minus1[0]
    w.ue(cfg.max_num_reorder_pics); // sps_max_num_reorder_pics[0]
    w.ue(0); // sps_max_latency_increase_plus1[0]
    w.ue(cfg.min_cb_log2 - 3); // log2_min_luma_coding_block_size_minus3
    w.ue(ctb_log2 - cfg.min_cb_log2); // log2_diff_max_min_luma_coding_block_size
    w.ue(0); // log2_min_luma_transform_block_size_minus2 (4)
    w.ue(max_tb_log2 - 2); // log2_diff_max_min_luma_transform_block_size
    w.ue(th_inter); // max_transform_hierarchy_depth_inter
    w.ue(th_intra); // max_transform_hierarchy_depth_intra
    w.put_bit(0); // scaling_list_enabled_flag
    w.put_bit(u8::from(cfg.amp)); // amp_enabled_flag
    w.put_bit(u8::from(sao_enabled)); // sample_adaptive_offset_enabled_flag
    w.put_bit(0); // pcm_enabled_flag
    w.ue(0); // num_short_term_ref_pic_sets
    w.put_bit(0); // long_term_ref_pics_present_flag
    w.put_bit(0); // sps_temporal_mvp_enabled_flag
    w.put_bit(0); // strong_intra_smoothing_enabled_flag
    match cfg.timing {
        None => w.put_bit(0), // vui_parameters_present_flag
        Some((num_units_in_tick, time_scale)) => {
            // §E.2.1 vui_parameters( ) — everything off except the
            // timing declaration.
            w.put_bit(1); // vui_parameters_present_flag
            w.put_bit(0); // aspect_ratio_info_present_flag
            w.put_bit(0); // overscan_info_present_flag
            w.put_bit(0); // video_signal_type_present_flag
            w.put_bit(0); // chroma_loc_info_present_flag
            w.put_bit(0); // neutral_chroma_indication_flag
            w.put_bit(0); // field_seq_flag
            w.put_bit(0); // frame_field_info_present_flag
            w.put_bit(0); // default_display_window_flag
            w.put_bit(1); // vui_timing_info_present_flag
            w.put_bits(num_units_in_tick, 32); // vui_num_units_in_tick
            w.put_bits(time_scale, 32); // vui_time_scale
            w.put_bit(0); // vui_poc_proportional_to_timing_flag
            match &cfg.hrd {
                None => w.put_bit(0), // vui_hrd_parameters_present_flag
                Some(hrd) => {
                    w.put_bit(1); // vui_hrd_parameters_present_flag
                                  // §E.2.2 hrd_parameters( 1, 0 ).
                    crate::encoder::hrd::write_hrd_parameters(&mut w, hrd);
                }
            }
            w.put_bit(0); // bitstream_restriction_flag
        }
    }
    w.put_bit(0); // sps_extension_present_flag
    w.rbsp_trailing_bits();
    w.finish()
}

/// §7.3.8.14 / §9.3.3.10 — encode one `cu_qp_delta` (`CuQpDeltaVal ==
/// delta`): `cu_qp_delta_abs` as a TR(cMax 5) prefix over the two
/// Table 9-24 contexts (bin 0 its own, bins 1..4 shared) with a
/// bypass EG0 escape suffix, then the bypass `cu_qp_delta_sign_flag`
/// when nonzero. The write-side dual of
/// [`crate::binarization::decode_cu_qp_delta`].
pub(crate) fn encode_cu_qp_delta(
    w: &mut BitWriter,
    cabac: &mut CabacEncoder,
    ctxs: &mut SliceContexts,
    delta: i32,
) {
    const CMAX: u32 = crate::binarization::CU_QP_DELTA_ABS_TR_CMAX;
    let abs = delta.unsigned_abs();
    for i in 0..abs.min(CMAX) {
        let ctx = usize::from(i != 0);
        cabac.encode_decision(w, &mut ctxs.cu_qp_delta_abs[ctx], 1);
    }
    if abs < CMAX {
        let ctx = usize::from(abs != 0);
        cabac.encode_decision(w, &mut ctxs.cu_qp_delta_abs[ctx], 0);
    } else {
        crate::encoder::inter::encode_eg_k(w, cabac, abs - CMAX, 0);
    }
    if abs != 0 {
        cabac.encode_bypass(w, u8::from(delta < 0));
    }
}

/// Table 8-10 — the `ChromaArrayType == 1` chroma QP mapping
/// `qPC = f(qPi)` (§8.6.1; `QpBdOffsetC == 0` at 8-bit).
pub(crate) fn chroma_qp_420(qp_y: i32) -> u32 {
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
pub(crate) fn forward_transform(res: &[i32], n: usize) -> Vec<i32> {
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
pub(crate) fn quantize(coef: &[i32], n: usize, qp: u32) -> Vec<i32> {
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

/// §6.4.1-shaped z-scan availability of the sample at `(nx, ny)`
/// relative to the block being decoded: available iff inside the
/// plane AND its covering coding block precedes in decode order —
/// an earlier CTB (raster), or the same CTB with a smaller depth-1
/// z-order quadrant index (`cur_z`; pass 0 when the current TB is the
/// whole CTB, making every same-CTB neighbour unavailable).
#[allow(clippy::too_many_arguments)]
pub(crate) fn zscan_avail(
    nx: i64,
    ny: i64,
    plane_w: usize,
    plane_h: usize,
    blk: usize,
    ctbs_x: usize,
    cur_ctb: usize,
    cur_z: u32,
) -> bool {
    if nx < 0 || ny < 0 || nx >= plane_w as i64 || ny >= plane_h as i64 {
        return false;
    }
    let (nx, ny) = (nx as usize, ny as usize);
    let nctb = (ny / blk) * ctbs_x + nx / blk;
    match nctb.cmp(&cur_ctb) {
        core::cmp::Ordering::Less => true,
        core::cmp::Ordering::Greater => false,
        core::cmp::Ordering::Equal => {
            let half = blk / 2;
            let z = ((ny % blk) / half) * 2 + ((nx % blk) / half);
            (z as u32) < cur_z
        }
    }
}

/// Gather the §8.4.4.2.1 marked reference array for an `n`-sample TB
/// at `(x0, y0)`: values through `read`, availability through `avail`.
pub(crate) fn gather_refs(
    read: &dyn Fn(usize, usize) -> i32,
    avail: &dyn Fn(i64, i64) -> bool,
    x0: usize,
    y0: usize,
    n: usize,
) -> MarkedReferenceSamples {
    let get = |x: i64, y: i64| -> (i32, bool) {
        if avail(x, y) {
            (read(x as usize, y as usize), true)
        } else {
            (0, false)
        }
    };
    let corner = get(x0 as i64 - 1, y0 as i64 - 1);
    let left: Vec<(i32, bool)> = (0..2 * n)
        .map(|k| get(x0 as i64 - 1, (y0 + k) as i64))
        .collect();
    let top: Vec<(i32, bool)> = (0..2 * n)
        .map(|k| get((x0 + k) as i64, y0 as i64 - 1))
        .collect();
    MarkedReferenceSamples::new(n, corner, left, top).expect("legal TB geometry")
}

pub(crate) fn pred_params(mode: u8, cidx: PredComponent) -> IntraPredParams {
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

/// SAD-search all 35 §8.4.2 modes; returns `(mode, prediction)`.
pub(crate) fn search_best_mode(marked: &MarkedReferenceSamples, src: &[i32]) -> (u8, Vec<i32>) {
    let mut best = (0u8, Vec::new());
    let mut best_cost = u64::MAX;
    for mode in 0..=34u8 {
        let pred = intra_predict_with_substitution(marked, &pred_params(mode, PredComponent::Luma))
            .expect("legal prediction params");
        let cost: u64 = src
            .iter()
            .zip(pred.iter())
            .map(|(&s, &p)| u64::from(s.abs_diff(p)))
            .sum();
        if cost < best_cost {
            best_cost = cost;
            best = (mode, pred);
        }
    }
    best
}

/// Transform + quantize one component TB and reconstruct it through
/// the DECODE-side §8.6.2 path. Returns `(levels, recon_samples)`;
/// `levels` all-zero ⇔ cbf 0 (recon = clipped prediction).
///
/// `pred_mode` selects the §8.6.4 transform family exactly as the
/// decoder does (the intra-luma 4x4 DST case; every TB the intra and
/// low-delay-P encoders emit at other geometries is DCT either way).
pub(crate) fn code_tb(
    src: &[i32],
    pred: &[i32],
    n: usize,
    qp: u32,
    component: Component,
    pred_mode: PredMode,
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
                pred_mode,
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

pub(crate) fn ssd(a: &[u8], b: &[i32]) -> u64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = i64::from(x) - i64::from(y);
            (d * d) as u64
        })
        .sum()
}

/// Crude bit-cost proxy for one TB's quantized levels: each nonzero
/// coefficient costs roughly its magnitude's bit length (sig flag +
/// sign + level bins); a coded-but-empty TB is nearly free, a coded
/// TB pays a small last-sig overhead.
pub(crate) fn rate_proxy(levels: &[i32]) -> u64 {
    let bits: u64 = levels
        .iter()
        .filter(|&&l| l != 0)
        .map(|&l| 3 + 2 * u64::from(32 - l.unsigned_abs().leading_zeros()))
        .sum();
    if bits == 0 {
        1
    } else {
        bits + 8
    }
}

/// One coded luma partition candidate for a CTB.
struct LumaPlan {
    /// `PART_NxN`?
    nxn: bool,
    /// PB modes (1 used for 2Nx2N, 4 for NxN, z-order).
    modes: [u8; 4],
    /// TB level arrays (1 x 16x16 or 4 x 8x8, z-order).
    levels: Vec<Vec<i32>>,
    /// The CTB's 16x16 luma reconstruction, row-major.
    recon: Vec<u8>,
}

/// One CTB's pass-1 coding decisions (pass 2 emits the syntax after
/// the in-loop filter elections are known).
struct CtbPlan {
    /// The elected luma partition + levels.
    plan: LumaPlan,
    /// Chroma Cb TB levels (1 x 8x8 or 4 x 4x4, z-order).
    cb_levels: Vec<Vec<i32>>,
    /// Chroma Cr TB levels.
    cr_levels: Vec<Vec<i32>>,
    /// The `QpY` this CTB was quantized at (slice QP + AQ offset).
    qp: i32,
}

impl CtbPlan {
    /// Any nonzero coded level in the CTB (⇔ some TU signals a cbf,
    /// so a `cu_qp_delta` can be — and is — transmitted).
    fn any_cbf(&self) -> bool {
        self.plan
            .levels
            .iter()
            .chain(self.cb_levels.iter())
            .chain(self.cr_levels.iter())
            .any(|lv| lv.iter().any(|&v| v != 0))
    }
}

/// Encode one 4:2:0 8-bit frame as a self-contained intra IDR access
/// unit at `SliceQpY == qp` and return it with the reconstruction a
/// conforming decoder produces.
///
/// # Errors
/// [`IntraEncodeError`] on bad dimensions / plane sizes / QP.
#[allow(clippy::too_many_lines)]
pub fn encode_idr_intra_au(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    qp: i32,
) -> Result<IntraEncodedAu, IntraEncodeError> {
    // A standalone intra AU needs one reference slot beyond the
    // current picture (`sps_max_dec_pic_buffering_minus1 == 1`).
    encode_idr_intra_au_cfg(y, cb, cr, width, height, qp, 1)
}

/// [`encode_idr_intra_au`] with the §8.7 in-loop filters enabled per
/// `lf`: the reconstruction runs through the crate's decode-side
/// deblocking (per-slice on/off elected against distortion) and SAO
/// (per-CTB §7.3.8.3 parameters from statistics-driven offset
/// estimation), and the returned `recon_*` planes are the *filtered*
/// picture — still exactly what a conforming decoder outputs.
///
/// # Errors
/// [`IntraEncodeError`] on bad dimensions / plane sizes / QP.
pub fn encode_idr_intra_au_lf(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    qp: i32,
    lf: &LoopFilterCfg,
) -> Result<IntraEncodedAu, IntraEncodeError> {
    encode_idr_intra_au_full(y, cb, cr, width, height, qp, &SpsCfg::legacy(1), lf, 0)
}

/// [`encode_idr_intra_au_lf`] with spatial **adaptive quantization**:
/// at `aq` in 1..=3, each CTB's `QpY` moves off the slice QP by `aq`
/// QP per octave of luma-activity ratio against the picture average
/// (clamped ±6), signalled through §7.3.8.10 / §7.4.9.14
/// `cu_qp_delta` (`diff_cu_qp_delta_depth == 0`). `aq == 0` is
/// exactly [`encode_idr_intra_au_lf`].
///
/// # Errors
/// [`IntraEncodeError`] on bad dimensions / plane sizes / QP, or
/// `aq > 3`.
#[allow(clippy::too_many_arguments)]
pub fn encode_idr_intra_au_aq(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    qp: i32,
    lf: &LoopFilterCfg,
    aq: u8,
) -> Result<IntraEncodedAu, IntraEncodeError> {
    let cfg = SpsCfg {
        cu_qp_delta: aq > 0,
        ..SpsCfg::legacy(1)
    };
    encode_idr_intra_au_full(y, cb, cr, width, height, qp, &cfg, lf, aq)
}

/// [`encode_idr_intra_au`] with an explicit
/// `sps_max_dec_pic_buffering_minus1` — the low-delay GOP encoder
/// passes 2 so a conforming decoder keeps BOTH short-term references
/// alive alongside the current picture.
pub(crate) fn encode_idr_intra_au_cfg(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    qp: i32,
    max_dec_pic_buffering_minus1: u32,
) -> Result<IntraEncodedAu, IntraEncodeError> {
    encode_idr_intra_au_full(
        y,
        cb,
        cr,
        width,
        height,
        qp,
        &SpsCfg::legacy(max_dec_pic_buffering_minus1),
        &LoopFilterCfg::off(),
        0,
    )
}

/// The full-configuration intra encode: two passes (per-CTB mode
/// decision + reconstruction, then syntax emission) around the §8.7
/// in-loop filter stage, so the per-CTB §7.3.8.3 SAO parameters and
/// the slice-header filter elections are known before the slice is
/// written.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub(crate) fn encode_idr_intra_au_full(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    qp: i32,
    cfg: &SpsCfg,
    lf: &LoopFilterCfg,
    aq: u8,
) -> Result<IntraEncodedAu, IntraEncodeError> {
    if width == 0 || height == 0 || width % CTB != 0 || height % CTB != 0 {
        return Err(IntraEncodeError::BadDimensions { width, height });
    }
    if !(0..=51).contains(&qp) {
        return Err(IntraEncodeError::BadQp(qp));
    }
    if aq > 3 {
        return Err(IntraEncodeError::BadAq(aq));
    }
    debug_assert!(
        aq == 0 || cfg.cu_qp_delta,
        "AQ needs the PPS cu_qp_delta_enabled_flag"
    );
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

    if cfg.tree.is_some() {
        return crate::encoder::ctu::encode_intra_picture_tree(
            y, cb, cr, width, height, qp, cfg, lf, aq,
        );
    }
    let (cw, ch) = (width / 2, height / 2);
    let ctbs_x = width / CTB;
    let ctbs_y = height / CTB;
    // Rate-distortion tradeoff for the partition decision: SSD per
    // estimated bit, doubling every 3 QP (integer, deterministic).
    let lambda_of = |q: i32| -> u64 { 1u64 << (q.unsigned_abs().saturating_sub(9) / 3) };
    let lambda = lambda_of(qp);
    // Spatial AQ: per-CTB QP offsets against the slice QP (all zero
    // at `aq == 0`).
    let aq_deltas = crate::encoder::aq::ctb_aq_deltas(y, width, height, aq, CTB);

    let mut recon_y = vec![0u8; width * height];
    let mut recon_cb = vec![0u8; cw * ch];
    let mut recon_cr = vec![0u8; cw * ch];
    let mut modes = IntraModeField::new(width, height, CTB_LOG2);
    let mut plans: Vec<CtbPlan> = Vec::with_capacity(ctbs_x * ctbs_y);

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

    for (ctb, &aq_delta) in aq_deltas.iter().enumerate() {
        let x0 = (ctb % ctbs_x) * CTB;
        let y0 = (ctb / ctbs_x) * CTB;
        let ctb_qp = (qp + aq_delta).clamp(0, 51);
        let qp_y = ctb_qp as u32;
        let qp_c = chroma_qp_420(ctb_qp);
        let lambda = lambda_of(ctb_qp);
        let src16 = extract(y, width, x0, y0, CTB);

        // ---- candidate PART_2Nx2N: one 16x16 PB/TB ----
        let plan_2n = {
            let read = |x: usize, yy: usize| i32::from(recon_y[yy * width + x]);
            let avail = |nx: i64, ny: i64| zscan_avail(nx, ny, width, height, CTB, ctbs_x, ctb, 0);
            let marked = gather_refs(&read, &avail, x0, y0, CTB);
            let (mode, pred) = search_best_mode(&marked, &src16);
            let (levels, recon) =
                code_tb(&src16, &pred, CTB, qp_y, Component::Luma, PredMode::Intra);
            LumaPlan {
                nxn: false,
                modes: [mode; 4],
                levels: vec![levels],
                recon,
            }
        };

        // ---- candidate PART_NxN: four 8x8 PBs/TBs, z-order ----
        // Only at the legacy MinCb geometry: at `log2CbSize >
        // MinCbLog2SizeY` an intra CU's `part_mode` is not present
        // (§7.3.8.5), so PART_NxN cannot be signalled.
        let plan_nxn = if cfg.min_cb_log2 < CTB_LOG2 {
            None
        } else {
            let mut scratch = vec![0u8; CTB * CTB]; // in-progress CTB recon
            let mut pb_modes = [0u8; 4];
            let mut pb_levels: Vec<Vec<i32>> = Vec::with_capacity(4);
            for (k, &(zx, zy)) in Z_OFFSETS.iter().enumerate() {
                let (px, py) = (x0 + zx * 8, y0 + zy * 8);
                let read = |x: usize, yy: usize| -> i32 {
                    if (x0..x0 + CTB).contains(&x) && (y0..y0 + CTB).contains(&yy) {
                        i32::from(scratch[(yy - y0) * CTB + (x - x0)])
                    } else {
                        i32::from(recon_y[yy * width + x])
                    }
                };
                let avail = |nx: i64, ny: i64| {
                    zscan_avail(nx, ny, width, height, CTB, ctbs_x, ctb, k as u32)
                };
                let marked = gather_refs(&read, &avail, px, py, 8);
                let src8 = extract(y, width, px, py, 8);
                let (mode, pred) = search_best_mode(&marked, &src8);
                let (levels, recon8) =
                    code_tb(&src8, &pred, 8, qp_y, Component::Luma, PredMode::Intra);
                for j in 0..8 {
                    scratch[(zy * 8 + j) * CTB + zx * 8..(zy * 8 + j) * CTB + zx * 8 + 8]
                        .copy_from_slice(&recon8[j * 8..(j + 1) * 8]);
                }
                pb_modes[k] = mode;
                pb_levels.push(levels);
            }
            Some(LumaPlan {
                nxn: true,
                modes: pb_modes,
                levels: pb_levels,
                recon: scratch,
            })
        };

        // Luma-only rate-distortion comparison: SSD of the coded
        // reconstruction + lambda times a bit proxy (residual levels +
        // mode signalling: ~6 bits per PB).
        let cost = |plan: &LumaPlan| -> u64 {
            let rate: u64 = plan.levels.iter().map(|lv| rate_proxy(lv)).sum::<u64>()
                + plan.levels.len() as u64 * 6;
            ssd(&plan.recon, &src16) + lambda * rate
        };
        let plan = match plan_nxn {
            Some(nxn) if cost(&nxn) < cost(&plan_2n) => nxn,
            _ => plan_2n,
        };
        store(&mut recon_y, width, x0, y0, CTB, &plan.recon);
        // §8.4.3: IntraPredModeC derives from the CU's first PB.
        let mode_c = plan.modes[0];

        // ---- chroma: 8x8 TBs (2Nx2N) or four 4x4 TBs (NxN) ----
        let (cx0, cy0) = (x0 / 2, y0 / 2);
        let code_chroma = |plane: &[u8],
                           recon: &mut Vec<u8>,
                           comp: Component,
                           pc: PredComponent|
         -> Vec<Vec<i32>> {
            if !plan.nxn {
                let read = |x: usize, yy: usize| i32::from(recon[yy * cw + x]);
                let avail = |nx: i64, ny: i64| zscan_avail(nx, ny, cw, ch, CTB / 2, ctbs_x, ctb, 0);
                let marked = gather_refs(&read, &avail, cx0, cy0, 8);
                let pred = intra_predict_with_substitution(&marked, &pred_params(mode_c, pc))
                    .expect("legal prediction params");
                let src = extract(plane, cw, cx0, cy0, 8);
                let (levels, rec) = code_tb(&src, &pred, 8, qp_c, comp, PredMode::Intra);
                store(recon, cw, cx0, cy0, 8, &rec);
                vec![levels]
            } else {
                let mut out = Vec::with_capacity(4);
                let mut scratch = vec![0u8; 64]; // 8x8 chroma CTB recon
                for (k, &(zx, zy)) in Z_OFFSETS.iter().enumerate() {
                    let (px, py) = (cx0 + zx * 4, cy0 + zy * 4);
                    let read = |x: usize, yy: usize| -> i32 {
                        if (cx0..cx0 + 8).contains(&x) && (cy0..cy0 + 8).contains(&yy) {
                            i32::from(scratch[(yy - cy0) * 8 + (x - cx0)])
                        } else {
                            i32::from(recon[yy * cw + x])
                        }
                    };
                    let avail = |nx: i64, ny: i64| {
                        zscan_avail(nx, ny, cw, ch, CTB / 2, ctbs_x, ctb, k as u32)
                    };
                    let marked = gather_refs(&read, &avail, px, py, 4);
                    let pred = intra_predict_with_substitution(&marked, &pred_params(mode_c, pc))
                        .expect("legal prediction params");
                    let src = extract(plane, cw, px, py, 4);
                    let (levels, rec) = code_tb(&src, &pred, 4, qp_c, comp, PredMode::Intra);
                    for j in 0..4 {
                        scratch[(zy * 4 + j) * 8 + zx * 4..(zy * 4 + j) * 8 + zx * 4 + 4]
                            .copy_from_slice(&rec[j * 4..(j + 1) * 4]);
                    }
                    out.push(levels);
                }
                store(recon, cw, cx0, cy0, 8, &scratch);
                out
            }
        };
        let cb_levels = code_chroma(cb, &mut recon_cb, Component::Cb, PredComponent::Cb);
        let cr_levels = code_chroma(cr, &mut recon_cr, Component::Cr, PredComponent::Cr);

        plans.push(CtbPlan {
            plan,
            cb_levels,
            cr_levels,
            qp: ctb_qp,
        });
    }

    // The §8.6.1 effective per-CTB QpY a decoder derives: with one
    // quantization group per CTB the neighbour prediction collapses
    // to qPY_PREV (the previous CU in decode order, SliceQpY at the
    // slice start), so a CTB that transmits no cu_qp_delta (no coded
    // cbf anywhere, or cu_qp_delta off) inherits the running value.
    let eff_qps: Vec<i32> = {
        let mut prev = qp;
        plans
            .iter()
            .map(|p| {
                if cfg.cu_qp_delta && p.any_cbf() {
                    prev = p.qp;
                }
                if cfg.cu_qp_delta {
                    prev
                } else {
                    qp
                }
            })
            .collect()
    };

    // ---- §8.7 in-loop filters (deblocking + SAO) on the recon ----
    let mut deblock_on = false;
    let mut beta_offset_div2 = 0i32;
    let mut tc_offset_div2 = 0i32;
    let mut slice_sao_luma = false;
    let mut slice_sao_chroma = false;
    let mut sao_ctbs: Vec<SaoCtbParams> = Vec::new();
    if lf.any() {
        let shapes: Vec<CtbShape> = plans
            .iter()
            .map(|p| CtbShape {
                part_mode: if p.plan.nxn {
                    PartMode::PartNxN
                } else {
                    PartMode::Part2Nx2N
                },
                split_depth1: p.plan.nxn,
            })
            .collect();
        // Every CU is intra: the fresh motion field's all-intra
        // background is exactly the decoder's state (bS == 2 at every
        // filtered edge).
        let field = MotionField::new(width, height);
        let out = filter_frame(
            &FilterInput {
                width,
                height,
                ctb_qps: &eff_qps,
                lambda,
                recon: [&recon_y, &recon_cb, &recon_cr],
                src: [y, cb, cr],
                field: &field,
                shapes: &shapes,
                ctb_log2: 4,
                tree: None,
            },
            lf,
        );
        deblock_on = out.deblock_on;
        beta_offset_div2 = out.beta_offset_div2;
        tc_offset_div2 = out.tc_offset_div2;
        slice_sao_luma = out.slice_sao_luma;
        slice_sao_chroma = out.slice_sao_chroma;
        sao_ctbs = out.sao_ctbs;
        recon_y = out.y;
        recon_cb = out.cb;
        recon_cr = out.cr;
    }

    // ---- slice_segment_header( ) ----
    let mut w = BitWriter::new();
    w.put_bit(1); // first_slice_segment_in_pic_flag
    w.put_bit(0); // no_output_of_prior_pics_flag (IRAP NAL)
    w.ue(0); // slice_pic_parameter_set_id
    w.ue(2); // slice_type = I
    if lf.sao() {
        // SPS SAO enabled: the per-slice component gates are present.
        w.put_bit(u8::from(slice_sao_luma)); // slice_sao_luma_flag
        w.put_bit(u8::from(slice_sao_chroma)); // slice_sao_chroma_flag
    }
    w.se(qp - 26); // slice_qp_delta (init_qp_minus26 == 0)
    if lf.deblocking {
        // §7.3.6.1 deblocking override group (the PPS signals
        // override-enabled + deblocking disabled; an electing slice
        // overrides to enabled with zero β/tC offsets).
        w.put_bit(u8::from(deblock_on)); // deblocking_filter_override_flag
        if deblock_on {
            w.put_bit(0); // slice_deblocking_filter_disabled_flag
            w.se(beta_offset_div2); // slice_beta_offset_div2
            w.se(tc_offset_div2); // slice_tc_offset_div2
        }
    }
    if slice_sao_luma || slice_sao_chroma || deblock_on {
        // §7.3.6.1: present iff pps_loop_filter_across_slices (1) and
        // some in-loop filter is active for this slice.
        w.put_bit(1); // slice_loop_filter_across_slices_enabled_flag
    }
    w.rbsp_trailing_bits(); // byte_alignment()

    // ---- slice_segment_data( ) — pass 2: syntax emission ----
    let mut cabac = CabacEncoder::new();
    // Table 9-4: I slice => initType 0 (raw slice_type 2).
    let mut ctxs = SliceContexts::init(init_type(2, false), qp);
    // §8.6.1 qPY_PREV thread for the cu_qp_delta emission (one QG per
    // CTB ⇒ qPY_PRED == qPY_PREV; SliceQpY at the slice start).
    let mut qp_prev = qp;
    for (ctb, ctb_plan) in plans.iter().enumerate() {
        let x0 = (ctb % ctbs_x) * CTB;
        let y0 = (ctb / ctbs_x) * CTB;
        if slice_sao_luma || slice_sao_chroma {
            // §7.3.8.3 sao( rx, ry ) ahead of the coding quadtree.
            encode_sao_ctb(
                &mut w,
                &mut cabac,
                &mut ctxs,
                &sao_ctbs[ctb],
                ctb % ctbs_x,
                ctb / ctbs_x,
                slice_sao_luma,
                slice_sao_chroma,
            );
        }
        let plan = &ctb_plan.plan;
        let cb_levels = &ctb_plan.cb_levels;
        let cr_levels = &ctb_plan.cr_levels;
        // §8.4.3: IntraPredModeC derives from the CU's first PB.
        let mode_c = plan.modes[0];

        // ---- §7.3.8.5 coding_unit( ) syntax ----
        if cfg.min_cb_log2 < CTB_LOG2 {
            // §7.3.8.4: log2CbSize (4) > MinCbLog2SizeY — split_cu_flag
            // is coded, and every CTB stays one UNSPLIT 16x16 CU. All
            // CtDepth values are 0, so both §9.3.4.2.2 conds are 0 and
            // the ctxInc is 0. part_mode is then NOT present for an
            // intra CU at log2CbSize > MinCbLog2SizeY (§7.3.8.5;
            // inferred PART_2Nx2N).
            cabac.encode_decision(&mut w, &mut ctxs.split_cu_flag[0], 0);
            debug_assert!(!plan.nxn, "PART_NxN cannot be signalled above MinCb");
        } else {
            // part_mode: §9.3.3.7 intra at MinCb — "1" = PART_2Nx2N,
            // "0" = PART_NxN.
            cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], u8::from(!plan.nxn));
        }
        // §7.3.8.5 two-loop luma mode group: all
        // prev_intra_luma_pred_flag bins first, then the mpm_idx /
        // rem_intra_luma_pred_mode group. The §8.4.2 candidate list of
        // PB k sees the recorded modes of PBs < k, so record as we go
        // through the SECOND loop (derivation order).
        let n_pb = if plan.nxn { 4 } else { 1 };
        let pb_size = if plan.nxn { 8 } else { CTB };
        let pb_pos = |k: usize| (x0 + Z_OFFSETS[k].0 * 8, y0 + Z_OFFSETS[k].1 * 8);
        let mut selections: Vec<Option<usize>> = Vec::with_capacity(n_pb);
        {
            // The candidate list depends only on ALREADY-decoded PBs,
            // which are identical in both loop passes; precompute the
            // per-PB MPM position by simulating the record order.
            for k in 0..n_pb {
                let (px, py) = pb_pos(k);
                let avail_l = zscan_avail(
                    px as i64 - 1,
                    py as i64,
                    width,
                    height,
                    CTB,
                    ctbs_x,
                    ctb,
                    k as u32,
                );
                let avail_a = zscan_avail(
                    px as i64,
                    py as i64 - 1,
                    width,
                    height,
                    CTB,
                    ctbs_x,
                    ctb,
                    k as u32,
                );
                let cand_a = modes.cand_intra_pred_mode(px, py, Neighbour::Left, avail_l);
                let cand_b = modes.cand_intra_pred_mode(px, py, Neighbour::Above, avail_a);
                let list = intra_luma_cand_mode_list(cand_a, cand_b);
                selections.push(list.iter().position(|&m| m == plan.modes[k]));
                // Loop 1: prev_intra_luma_pred_flag[k].
                cabac.encode_decision(
                    &mut w,
                    &mut ctxs.prev_intra_luma_pred_flag[0],
                    u8::from(selections[k].is_some()),
                );
                // Record now: the next PB's candidates must see this
                // one (§8.4.2 derivation order).
                modes.record_intra_pb(px, py, pb_size, plan.modes[k], false);
            }
            // Loop 2: mpm_idx / rem_intra_luma_pred_mode.
            for (k, sel) in selections.iter().enumerate() {
                match *sel {
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
                        // §8.4.2: rem = mode with each smaller
                        // candidate removed. Recompute the list the
                        // same way the decoder will (earlier PBs
                        // recorded).
                        let (px, py) = pb_pos(k);
                        let avail_l = zscan_avail(
                            px as i64 - 1,
                            py as i64,
                            width,
                            height,
                            CTB,
                            ctbs_x,
                            ctb,
                            k as u32,
                        );
                        let avail_a = zscan_avail(
                            px as i64,
                            py as i64 - 1,
                            width,
                            height,
                            CTB,
                            ctbs_x,
                            ctb,
                            k as u32,
                        );
                        let cand_a = modes.cand_intra_pred_mode(px, py, Neighbour::Left, avail_l);
                        let cand_b = modes.cand_intra_pred_mode(px, py, Neighbour::Above, avail_a);
                        let list = intra_luma_cand_mode_list(cand_a, cand_b);
                        let mut rem = u32::from(plan.modes[k]);
                        for &c in &list {
                            if u32::from(plan.modes[k]) > u32::from(c) {
                                rem -= 1;
                            }
                        }
                        cabac.encode_bypass_bits(&mut w, rem, 5); // FL cMax 31
                    }
                }
            }
        }
        // intra_chroma_pred_mode = 4 (derived from luma): bin "0".
        cabac.encode_decision(&mut w, &mut ctxs.intra_chroma_pred_mode[0], 0);

        // ---- §7.3.8.8 transform_tree + §7.3.8.10 transform_unit ----
        let rc_params = |log2: u32, is_chroma: bool, mode: u8| ResidualCodingParams {
            log2_trafo_size: log2,
            is_chroma,
            // §7.4.9.11 mode-dependent scan (only 4x4 / 8x8-luma TBs
            // are eligible; larger TBs come back Diagonal).
            scan_idx: residual_coding_scan_idx(true, log2, u8::from(is_chroma), 1, u32::from(mode)),
            sign_data_hiding_enabled_flag: false,
            sign_hidden_suppressed: false,
            transform_skip_sig_ctx: false,
            persistent_rice_adaptation_enabled_flag: false,
            cabac_bypass_alignment_enabled_flag: false,
            extended_precision_processing_flag: false,
            bit_depth: 8,
            rice_stat_transform_skip: false,
        };
        if !plan.nxn {
            // Single 16x16 TU at depth 0.
            let cbf_cb = cb_levels[0].iter().any(|&v| v != 0);
            let cbf_cr = cr_levels[0].iter().any(|&v| v != 0);
            let cbf_luma = plan.levels[0].iter().any(|&v| v != 0);
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
            // §7.3.8.10 delta_qp( ) in the first (only) TU with a cbf.
            if cfg.cu_qp_delta && (cbf_luma || cbf_cb || cbf_cr) {
                encode_cu_qp_delta(&mut w, &mut cabac, &mut ctxs, ctb_plan.qp - qp_prev);
                qp_prev = ctb_plan.qp;
            }
            if cbf_luma {
                encode_residual_coding(
                    &mut w,
                    &mut cabac,
                    &mut ctxs.residual,
                    &rc_params(4, false, plan.modes[0]),
                    &plan.levels[0],
                )
                .expect("validated luma levels");
            }
            if cbf_cb {
                encode_residual_coding(
                    &mut w,
                    &mut cabac,
                    &mut ctxs.residual,
                    &rc_params(3, true, mode_c),
                    &cb_levels[0],
                )
                .expect("validated cb levels");
            }
            if cbf_cr {
                encode_residual_coding(
                    &mut w,
                    &mut cabac,
                    &mut ctxs.residual,
                    &rc_params(3, true, mode_c),
                    &cr_levels[0],
                )
                .expect("validated cr levels");
            }
        } else {
            // IntraSplitFlag == 1: split_transform_flag inferred 1 at
            // depth 0 (§7.4.9.8); four 8x8 leaves at depth 1. Root
            // cbf_cb / cbf_cr gate the per-leaf chroma flags
            // (§7.3.8.8 inheritance).
            let leaf_cbf = |lv: &Vec<i32>| lv.iter().any(|&v| v != 0);
            let root_cb = cb_levels.iter().any(&leaf_cbf);
            let root_cr = cr_levels.iter().any(&leaf_cbf);
            cabac.encode_decision(
                &mut w,
                &mut ctxs.cbf_chroma[cbf_cb_ctx_inc(0) as usize],
                u8::from(root_cb),
            );
            cabac.encode_decision(
                &mut w,
                &mut ctxs.cbf_chroma[cbf_cr_ctx_inc(0) as usize],
                u8::from(root_cr),
            );
            let mut delta_done = false;
            for k in 0..4 {
                let cbf_cb_k = leaf_cbf(&cb_levels[k]);
                let cbf_cr_k = leaf_cbf(&cr_levels[k]);
                let cbf_luma_k = leaf_cbf(&plan.levels[k]);
                if root_cb {
                    cabac.encode_decision(
                        &mut w,
                        &mut ctxs.cbf_chroma[cbf_cb_ctx_inc(1) as usize],
                        u8::from(cbf_cb_k),
                    );
                }
                if root_cr {
                    cabac.encode_decision(
                        &mut w,
                        &mut ctxs.cbf_chroma[cbf_cr_ctx_inc(1) as usize],
                        u8::from(cbf_cr_k),
                    );
                }
                cabac.encode_decision(
                    &mut w,
                    &mut ctxs.cbf_luma[cbf_luma_ctx_inc(1) as usize],
                    u8::from(cbf_luma_k),
                );
                // §7.3.8.10 delta_qp( ) in the first depth-1 TU with a
                // cbf (`IsCuQpDeltaCoded` then holds for the QG).
                if cfg.cu_qp_delta && !delta_done && (cbf_luma_k || cbf_cb_k || cbf_cr_k) {
                    encode_cu_qp_delta(&mut w, &mut cabac, &mut ctxs, ctb_plan.qp - qp_prev);
                    qp_prev = ctb_plan.qp;
                    delta_done = true;
                }
                if cbf_luma_k {
                    encode_residual_coding(
                        &mut w,
                        &mut cabac,
                        &mut ctxs.residual,
                        &rc_params(3, false, plan.modes[k]),
                        &plan.levels[k],
                    )
                    .expect("validated luma levels");
                }
                if root_cb && cbf_cb_k {
                    encode_residual_coding(
                        &mut w,
                        &mut cabac,
                        &mut ctxs.residual,
                        &rc_params(2, true, mode_c),
                        &cb_levels[k],
                    )
                    .expect("validated cb levels");
                }
                if root_cr && cbf_cr_k {
                    encode_residual_coding(
                        &mut w,
                        &mut cabac,
                        &mut ctxs.residual,
                        &rc_params(2, true, mode_c),
                        &cr_levels[k],
                    )
                    .expect("validated cr levels");
                }
            }
        }

        // end_of_slice_segment_flag.
        cabac.encode_terminate(&mut w, u8::from(ctb == ctbs_x * ctbs_y - 1));
    }
    // The final terminate-1 flush wrote the rbsp_stop_one_bit;
    // rbsp_slice_segment_trailing_bits() is alignment zeros from here.
    w.align_zero();
    let slice_rbsp = w.finish();

    let au = assemble_idr_au(width, height, cfg, lf, &slice_rbsp);
    Ok(IntraEncodedAu {
        au,
        recon_y,
        recon_cb,
        recon_cr,
    })
}

/// Wrap one coded IDR slice RBSP into its Annex B access unit
/// (`VPS + SPS + PPS + IDR_N_LP`), shared by the fixed-geometry and
/// quadtree intra coders.
pub(crate) fn assemble_idr_au(
    width: usize,
    height: usize,
    cfg: &SpsCfg,
    lf: &LoopFilterCfg,
    slice_rbsp: &[u8],
) -> Vec<u8> {
    let level_idc = level_idc_for(width * height);
    // The reorder-free streams keep the historical VPS bounds (1, 0)
    // so every golden pin stays byte-stable; a reordering
    // (hierarchical-B) stream signals its honest DPB bounds.
    let vps = if cfg.max_num_reorder_pics == 0 {
        write_vps(level_idc)
    } else {
        write_vps_cfg(
            level_idc,
            cfg.max_dec_pic_buffering_minus1,
            cfg.max_num_reorder_pics,
        )
    };
    let units = vec![
        nal_unit(32, 0, 0, &vps), // VPS_NUT
        nal_unit(
            33,
            0,
            0,
            &write_sps_cfg(width, height, level_idc, cfg, lf.sao()),
        ), // SPS_NUT
        nal_unit(
            34,
            0,
            0,
            &write_pps_full(false, false, lf.deblocking, None, cfg.cu_qp_delta),
        ), // PPS_NUT
        nal_unit(20, 0, 0, slice_rbsp), // IDR_N_LP
    ];
    annexb(&units)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binarization::PartMode;
    use crate::sequence::{decode_annexb_sequence, decode_annexb_sequence_debug};
    use crate::slice_data::CodingQuadtree;

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

    /// Content with per-8x8 alternating strong directions: drives the
    /// partition decision toward PART_NxN.
    fn blocky_planes(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let y: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, yy) = (i % w, i / w);
                let (bx, by) = (x / 8, yy / 8);
                match (bx + by) % 3 {
                    0 => ((x % 8) * 30) as u8,
                    1 => ((yy % 8) * 30) as u8,
                    _ => (((x + yy) % 16) * 15) as u8,
                }
            })
            .collect();
        let cb = vec![100u8; w * h / 4];
        let cr = vec![160u8; w * h / 4];
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

    fn assert_roundtrip_exact(y: &[u8], cb: &[u8], cr: &[u8], w: usize, h: usize, qp: i32) {
        let enc = encode_idr_intra_au(y, cb, cr, w, h, qp).expect("encode");
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

    /// The core contract: the crate's own decoder reproduces the
    /// encoder's reconstruction EXACTLY (dual-decoder bit-exactness),
    /// on both smooth and NxN-inducing content.
    #[test]
    fn intra_au_decodes_to_encoder_recon_exactly() {
        for (w, h) in [(16usize, 16usize), (64, 48), (48, 80)] {
            let (y, cb, cr) = planes(w, h);
            for qp in [4i32, 22, 32, 45] {
                assert_roundtrip_exact(&y, &cb, &cr, w, h, qp);
            }
            let (y, cb, cr) = blocky_planes(w, h);
            for qp in [10i32, 27, 38] {
                assert_roundtrip_exact(&y, &cb, &cr, w, h, qp);
            }
        }
    }

    /// The partition decision really selects PART_NxN on content with
    /// per-8x8 directional structure (and the stream decodes exactly).
    #[test]
    fn nxn_partition_is_selected_and_decodes() {
        let (w, h) = (64usize, 64usize);
        let (y, cb, cr) = blocky_planes(w, h);
        let enc = encode_idr_intra_au(&y, &cb, &cr, w, h, 27).expect("encode");
        let ctus = decode_annexb_sequence_debug(&enc.au).expect("walk");
        let mut nxn = 0usize;
        let mut two_n = 0usize;
        for (_, _, ctu) in &ctus {
            if let CodingQuadtree::Leaf(cu) = &ctu.quadtree {
                match cu.part_mode {
                    PartMode::PartNxN => nxn += 1,
                    PartMode::Part2Nx2N => two_n += 1,
                    other => panic!("unexpected part mode {other:?}"),
                }
            }
        }
        assert!(nxn > 0, "no PART_NxN CU selected ({two_n} 2Nx2N)");
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

    /// QP 22 on this content should be visually transparent while far
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

    /// Filtered intra AUs (§8.7.2 deblocking / §8.7.3 SAO, alone and
    /// combined) decode bit-exactly to the encoder's filtered
    /// reconstruction through the crate's own decoder, across
    /// geometries and QPs.
    #[test]
    fn filtered_intra_au_decodes_to_encoder_recon_exactly() {
        let cfgs = [
            LoopFilterCfg {
                deblocking: true,
                sao_luma: false,
                sao_chroma: false,
            },
            LoopFilterCfg {
                deblocking: false,
                sao_luma: true,
                sao_chroma: true,
            },
            LoopFilterCfg {
                deblocking: false,
                sao_luma: true,
                sao_chroma: false,
            },
            LoopFilterCfg::all(),
        ];
        for (w, h) in [(32usize, 32usize), (64, 48)] {
            for planes_fn in [planes, blocky_planes] {
                let (y, cb, cr) = planes_fn(w, h);
                for qp in [27i32, 38] {
                    for cfg in &cfgs {
                        let enc =
                            encode_idr_intra_au_lf(&y, &cb, &cr, w, h, qp, cfg).expect("encode");
                        let frames = decode_annexb_sequence(&enc.au).expect("decode");
                        assert_eq!(frames.len(), 1, "{w}x{h} qp{qp} {cfg:?}");
                        let mut recon = enc.recon_y.clone();
                        recon.extend_from_slice(&enc.recon_cb);
                        recon.extend_from_slice(&enc.recon_cr);
                        assert_eq!(
                            frames[0].picture.to_planar_u8().expect("8-bit"),
                            recon,
                            "{w}x{h} qp{qp} {cfg:?}: decoder output == filtered recon"
                        );
                    }
                }
            }
        }
    }

    /// The filter elections are distortion-driven, so a filtered
    /// encode is never further from the source than the unfiltered
    /// one — and on smooth content at high QP (where quantization
    /// manufactures block edges the source never had) the filters
    /// actually engage (the reconstruction changes and the slice
    /// header signals them).
    #[test]
    fn filters_never_hurt_and_engage_on_smooth_content() {
        let (w, h) = (64usize, 64usize);
        let (y, cb, cr) = planes(w, h);
        let qp = 40;
        let plain = encode_idr_intra_au(&y, &cb, &cr, w, h, qp).expect("plain");
        let filtered =
            encode_idr_intra_au_lf(&y, &cb, &cr, w, h, qp, &LoopFilterCfg::all()).expect("lf");
        let ssd = |a: &[u8], b: &[u8]| -> u64 {
            a.iter()
                .zip(b.iter())
                .map(|(&p, &q)| {
                    let d = i64::from(p) - i64::from(q);
                    (d * d) as u64
                })
                .sum()
        };
        let d_plain =
            ssd(&plain.recon_y, &y) + ssd(&plain.recon_cb, &cb) + ssd(&plain.recon_cr, &cr);
        let d_filt = ssd(&filtered.recon_y, &y)
            + ssd(&filtered.recon_cb, &cb)
            + ssd(&filtered.recon_cr, &cr);
        assert!(
            d_filt <= d_plain,
            "filters must not hurt: filtered {d_filt} vs plain {d_plain}"
        );
        assert_ne!(
            filtered.recon_y, plain.recon_y,
            "high-QP blocky content should engage the loop filters"
        );

        // The written headers carry the filter signalling: SAO enabled
        // in the SPS, the deblocking override chain in the PPS, and
        // the elected per-slice flags in the slice header.
        let units = crate::nal::collect_nal_units(&filtered.au).expect("walk");
        let sps = crate::sps::SeqParameterSet::parse(&units[1].rbsp).expect("sps");
        assert!(sps.sample_adaptive_offset_enabled_flag);
        let pps = crate::pps::PicParameterSet::parse(&units[2].rbsp).expect("pps");
        assert!(pps.deblocking_filter_control_present_flag);
        assert!(pps.deblocking.override_enabled_flag);
        assert!(pps.deblocking.disabled_flag, "PPS default stays disabled");
        let header = crate::slice::SliceSegmentHeader::parse(
            &units[3].rbsp,
            units[3].header.nal_unit_type,
            &sps,
            &pps,
        )
        .expect("slice header");
        assert!(
            header.slice_sao_luma_flag
                || header.slice_sao_chroma_flag
                || header.deblocking.is_some_and(|d| !d.disabled_flag),
            "some in-loop filter is elected in the slice header"
        );
    }

    /// `LoopFilterCfg::off()` reproduces the legacy unfiltered stream
    /// byte for byte (headers included) — the golden interop pins stay
    /// valid.
    #[test]
    fn lf_off_is_byte_identical_to_legacy_encode() {
        let (w, h) = (48usize, 32usize);
        let (y, cb, cr) = planes(w, h);
        let plain = encode_idr_intra_au(&y, &cb, &cr, w, h, 24).expect("plain");
        let off =
            encode_idr_intra_au_lf(&y, &cb, &cr, w, h, 24, &LoopFilterCfg::off()).expect("off");
        assert_eq!(plain.au, off.au);
        assert_eq!(plain.recon_y, off.recon_y);
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
