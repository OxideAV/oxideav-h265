//! Low-delay P-GOP encoder — §8.5 inter prediction on the encode side.
//!
//! The sequence shape is `IDR, P, P, …`: the first frame goes through
//! the real CABAC intra encoder ([`crate::encoder::intra`]), every
//! following frame is one P slice (TRAIL_R) whose single active
//! reference (`RefPicList0[0]`) is the previous frame's
//! reconstruction, signalled with an inline §7.4.8 short-term RPS
//! (`num_negative_pics == 1`, `delta_poc == −1`).
//!
//! Geometry mirrors the intra bootstrap: `CtbSizeY == MinCbSizeY ==
//! 16`, so every CTB is one unsplit CU; inter CUs are `PART_2Nx2N`
//! (one 16x16 PU, one depth-0 TU). Per CTU three coding modes compete
//! under an SSD + λ·rate heuristic:
//!
//! * **skip** (`cu_skip_flag == 1`) — the SAD-best §8.5.3.2.2 merge
//!   candidate with no residual;
//! * **merge + residual** (`merge_flag == 1`) — the same candidate
//!   with the transform-coded residual (`rqt_root_cbf` is inferred 1
//!   for a 2Nx2N merge CU, so an all-zero residual falls back to
//!   skip);
//! * **AMVP** — motion-estimated `mvL0` (greedy integer diamond over
//!   seeded predictors, then half- then quarter-pel refinement
//!   against the crate's own §8.5.3.3.3 interpolation), signalled as
//!   `mvp_l0_flag` + §7.3.8.9 `mvd_coding`, with `rqt_root_cbf`
//!   electing the residual.
//!
//! Every candidate's motion is resolved through the DECODE-side
//! §8.5.3.2 derivation ([`crate::pu_mv::resolve_pu_motion`] against
//! the picture's in-progress [`MotionField`] and the §6.4.2
//! availability of [`crate::availability::PictureTiling`]), its
//! prediction through the decode-side §8.5.3.3 driver
//! ([`predict_inter_pu`]), and its reconstruction through the
//! decode-side §8.6 scaling/transform — so the encoder's reference
//! buffer is bit-identical to a conforming decoder's output (in-loop
//! filters are off), which the roundtrip tests pin frame by frame.

use crate::availability::{PictureTiling, TilingParams, MODE_INTRA};
use crate::binarization::{
    cbf_cb_ctx_inc, cbf_cr_ctx_inc, cbf_luma_ctx_inc, cu_skip_flag_ctx_inc,
    intra_luma_cand_mode_list, CuPredMode, InterPredIdc, MvdComponent,
};
use crate::cabac::init_type;
use crate::ctx_init::SliceContexts;
use crate::encoder::bitwriter::BitWriter;
use crate::encoder::cabac::CabacEncoder;
use crate::encoder::intra::{
    chroma_qp_420, code_tb, encode_idr_intra_au, gather_refs, pred_params, search_best_mode, ssd,
    zscan_avail, IntraEncodeError,
};
use crate::encoder::nal::{annexb, nal_unit};
use crate::encoder::residual::encode_residual_coding;
use crate::inter_pred::{
    predict_inter_pu, InterPredGeometry, InterPrediction, ListPrediction, RefPlane,
};
use crate::intra_mode_field::{IntraModeField, Neighbour};
use crate::intra_pred::{intra_predict_with_substitution, Component as PredComponent};
use crate::motion::{derive_chroma_mv, MotionCell, MotionField, Mv};
use crate::pu_mv::{resolve_pu_motion, PartMode, PuGeometry, PuMotion, PuMvContext};
use crate::residual::{residual_coding_scan_idx, ResidualCodingParams};
use crate::slice_data::PredictionUnit;
use crate::transform::{Component, PredMode};

/// The fixed CTB / coding-block log2 size (16x16).
const CTB_LOG2: u32 = 4;
/// The fixed CTB size.
const CTB: usize = 1 << CTB_LOG2;
/// Fixed 8-bit depth.
const BIT_DEPTH: u8 = 8;
/// `MaxNumMergeCand` (§7.4.7.1) — `five_minus_max_num_merge_cand = 0`.
const MAX_MERGE: usize = 5;
/// The greedy integer-search iteration cap (diamond steps).
const ME_MAX_STEPS: usize = 24;

/// One 4:2:0 8-bit input frame (borrowed planes).
#[derive(Debug, Clone, Copy)]
pub struct YuvFrame<'a> {
    /// Luma plane, `width * height` samples.
    pub y: &'a [u8],
    /// Cb plane, `width/2 * height/2` samples.
    pub cb: &'a [u8],
    /// Cr plane.
    pub cr: &'a [u8],
}

/// One frame's reconstruction (what a conforming decoder outputs).
#[derive(Debug, Clone)]
pub struct FrameRecon {
    /// Reconstructed luma plane.
    pub y: Vec<u8>,
    /// Reconstructed Cb plane.
    pub cb: Vec<u8>,
    /// Reconstructed Cr plane.
    pub cr: Vec<u8>,
}

/// The encoded low-delay sequence: the Annex B stream (`VPS + SPS +
/// PPS + IDR, TRAIL_R, TRAIL_R, …`) plus the per-frame encoder
/// reconstructions in output order.
#[derive(Debug, Clone)]
pub struct LowDelayPEncoded {
    /// The whole Annex B byte stream.
    pub stream: Vec<u8>,
    /// Per-frame reconstruction, `frames.len()` entries.
    pub recon: Vec<FrameRecon>,
    /// Per-frame CU mode decisions, `frames.len()` entries (frame 0 —
    /// the IDR — counts every CTB as intra).
    pub stats: Vec<FrameStats>,
}

/// Per-frame CU mode-decision counters (one CU per CTB).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameStats {
    /// `cu_skip_flag == 1` CUs.
    pub skip: usize,
    /// Merge-with-residual CUs.
    pub merge: usize,
    /// AMVP (explicit-MV) CUs.
    pub amvp: usize,
    /// Intra CUs (`pred_mode_flag == 1`, or every CU of the IDR).
    pub intra: usize,
}

/// A resolved-and-coded CU candidate (one of skip / merge / AMVP).
struct CuCandidate {
    /// How the PU is signalled.
    kind: CuKind,
    /// The §8.5.3.2.1-resolved motion (what the decoder will derive).
    motion: PuMotion,
    /// Quantized levels: luma 16x16, cb 8x8, cr 8x8. All-empty ⇔ no
    /// transform tree (skip, or AMVP with `rqt_root_cbf == 0`).
    levels: Option<[Vec<i32>; 3]>,
    /// The CTB reconstruction (y 16x16, cb 8x8, cr 8x8).
    recon: [Vec<u8>; 3],
    /// SSD + λ·rate cost.
    cost: u64,
}

/// The signalling class of a chosen CU.
enum CuKind {
    /// `cu_skip_flag == 1`, `merge_idx`.
    Skip { merge_idx: usize },
    /// `merge_flag == 1`, `merge_idx`, transform tree follows
    /// (`rqt_root_cbf` inferred 1).
    Merge { merge_idx: usize },
    /// AMVP: `mvd_coding` + `mvp_l0_flag`; `rqt_root_cbf` signalled.
    Amvp { mvd: Mv, mvp_flag: u8 },
    /// Intra fallback (`pred_mode_flag == 1`, `PART_2Nx2N`): §8.4
    /// prediction with luma mode `mode`, chroma derived-from-luma.
    Intra { mode: u8 },
}

/// One frame's output from the streaming [`LowDelayPEncoder`].
#[derive(Debug, Clone)]
pub struct EncodedPFrame {
    /// The Annex B access unit (`VPS + SPS + PPS + IDR_N_LP` for a
    /// keyframe, one `TRAIL_R` slice otherwise).
    pub au: Vec<u8>,
    /// `true` for an IDR access unit (a random access point).
    pub keyframe: bool,
    /// The frame's reconstruction (== a conforming decoder's output).
    pub recon: FrameRecon,
    /// The frame's CU mode-decision counters.
    pub stats: FrameStats,
}

/// The streaming low-delay encoder: one frame in, one access unit
/// out. The first frame (and every `gop`-th frame when `gop > 0`)
/// becomes a self-contained IDR access unit; every other frame is a
/// P slice referencing the previous frame's reconstruction.
#[derive(Debug)]
pub struct LowDelayPEncoder {
    width: usize,
    height: usize,
    qp: i32,
    /// GOP length in frames (`0` = a single leading IDR, endless P).
    gop: usize,
    /// POC of the NEXT frame within the current GOP (0 ⇒ IDR next).
    poc: i32,
    prev: Option<FrameRecon>,
}

impl LowDelayPEncoder {
    /// Construct a streaming encoder for `width`x`height` 4:2:0 8-bit
    /// content at constant `SliceQpY == qp`. `gop == 0` emits a
    /// single leading IDR; `gop == n` re-emits an IDR every `n`
    /// frames.
    ///
    /// # Errors
    /// [`IntraEncodeError`] on bad dimensions / QP.
    pub fn new(width: usize, height: usize, qp: i32, gop: usize) -> Result<Self, IntraEncodeError> {
        if width == 0 || height == 0 || width % CTB != 0 || height % CTB != 0 {
            return Err(IntraEncodeError::BadDimensions { width, height });
        }
        if !(0..=51).contains(&qp) {
            return Err(IntraEncodeError::BadQp(qp));
        }
        Ok(Self {
            width,
            height,
            qp,
            gop,
            poc: 0,
            prev: None,
        })
    }

    /// Encode the next frame in display order.
    ///
    /// # Errors
    /// [`IntraEncodeError::PlaneSize`] when a plane length does not
    /// match the 4:2:0 geometry.
    pub fn encode_frame(
        &mut self,
        frame: &YuvFrame<'_>,
    ) -> Result<EncodedPFrame, IntraEncodeError> {
        let (cw, ch) = (self.width / 2, self.height / 2);
        for (plane, buf, expected) in [
            ("y", frame.y, self.width * self.height),
            ("cb", frame.cb, cw * ch),
            ("cr", frame.cr, cw * ch),
        ] {
            if buf.len() != expected {
                return Err(IntraEncodeError::PlaneSize {
                    plane,
                    expected,
                    got: buf.len(),
                });
            }
        }
        let idr_now = self.prev.is_none() || self.poc == 0;
        let out = if idr_now {
            let idr = encode_idr_intra_au(
                frame.y,
                frame.cb,
                frame.cr,
                self.width,
                self.height,
                self.qp,
            )?;
            let recon = FrameRecon {
                y: idr.recon_y,
                cb: idr.recon_cb,
                cr: idr.recon_cr,
            };
            self.prev = Some(recon.clone());
            self.poc = 1;
            EncodedPFrame {
                au: idr.au,
                keyframe: true,
                recon,
                stats: FrameStats {
                    intra: (self.width / CTB) * (self.height / CTB),
                    ..FrameStats::default()
                },
            }
        } else {
            let prev = self.prev.as_ref().expect("IDR precedes every P frame");
            let (rbsp, recon, stats) =
                encode_p_slice(frame, prev, self.poc, self.width, self.height, self.qp);
            let au = annexb(&[nal_unit(1, 0, 0, &rbsp)]); // TRAIL_R
            self.prev = Some(recon.clone());
            self.poc += 1;
            EncodedPFrame {
                au,
                keyframe: false,
                recon,
                stats,
            }
        };
        // GOP wrap: schedule the next IDR.
        if self.gop > 0 && self.poc >= self.gop as i32 {
            self.poc = 0;
        }
        Ok(out)
    }
}

/// Encode a low-delay `IDR, P, P, …` sequence at a constant
/// `SliceQpY == qp` and return the Annex B stream plus the per-frame
/// reconstructions a conforming decoder reproduces exactly.
///
/// # Errors
/// [`IntraEncodeError`] on bad dimensions / plane sizes / QP (the
/// validation contract is shared with the intra encoder).
pub fn encode_low_delay_p(
    frames: &[YuvFrame<'_>],
    width: usize,
    height: usize,
    qp: i32,
) -> Result<LowDelayPEncoded, IntraEncodeError> {
    let mut enc = LowDelayPEncoder::new(width, height, qp, 0)?;
    let mut out = LowDelayPEncoded {
        stream: Vec::new(),
        recon: Vec::new(),
        stats: Vec::new(),
    };
    for frame in frames {
        let f = enc.encode_frame(frame)?;
        out.stream.extend_from_slice(&f.au);
        out.recon.push(f.recon);
        out.stats.push(f.stats);
    }
    Ok(out)
}

/// §7.3.6.1 — the P slice-segment header for POC `poc` (one inline
/// negative-delta-1 short-term RPS, no overrides, `MaxNumMergeCand ==
/// 5`, `slice_qp_delta` against `init_qp == 26`).
fn write_p_slice_header(w: &mut BitWriter, poc: i32, qp: i32) {
    w.put_bit(1); // first_slice_segment_in_pic_flag
    w.ue(0); // slice_pic_parameter_set_id
    w.ue(1); // slice_type = P
    w.put_bits((poc & 0xFF) as u32, 8); // slice_pic_order_cnt_lsb
    w.put_bit(0); // short_term_ref_pic_set_sps_flag
                  // st_ref_pic_set( 0 ) — idx 0 has no
                  // inter_ref_pic_set_prediction_flag (§7.3.7).
    w.ue(1); // num_negative_pics
    w.ue(0); // num_positive_pics
    w.ue(0); // delta_poc_s0_minus1[0] (DeltaPocS0 = −1)
    w.put_bit(1); // used_by_curr_pic_s0_flag[0]
                  // sps_temporal_mvp_enabled_flag == 0, SAO off: nothing.
    w.put_bit(0); // num_ref_idx_active_override_flag (PPS default: 1 active)
                  // lists_modification_present_flag == 0;
                  // cabac_init_present_flag == 0; TMVP off; WP off.
    w.ue((5 - MAX_MERGE) as u32); // five_minus_max_num_merge_cand
    w.se(qp - 26); // slice_qp_delta
                   // Deblocking disabled in the PPS + SAO off: no
                   // loop-filter fields; no tiles / WPP: no entry points.
    w.rbsp_trailing_bits(); // byte_alignment()
}

/// Extract an `n`x`n` block of `plane` at `(x0, y0)` as `i32`s.
fn extract(plane: &[u8], pw: usize, x0: usize, y0: usize, n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            out.push(i32::from(plane[(y0 + j) * pw + x0 + i]));
        }
    }
    out
}

/// Store an `n`x`n` block back into `plane` at `(x0, y0)`.
fn store(plane: &mut [u8], pw: usize, x0: usize, y0: usize, n: usize, s: &[u8]) {
    for j in 0..n {
        plane[(y0 + j) * pw + x0..(y0 + j) * pw + x0 + n].copy_from_slice(&s[j * n..(j + 1) * n]);
    }
}

/// The merge / skip PU syntax for candidate `merge_idx` (the §7.3.8.6
/// fields the §8.5.3.2.2 derivation reads).
fn merge_pu(merge_idx: usize) -> PredictionUnit {
    PredictionUnit {
        merge_flag: true,
        merge_idx: Some(merge_idx as u8),
        inter_pred_idc: None,
        ref_idx_l0: None,
        mvd_l0: None,
        mvp_l0_flag: None,
        ref_idx_l1: None,
        mvd_l1: None,
        mvp_l1_flag: None,
    }
}

/// An AMVP L0 PU with the given mvd pair and `mvp_l0_flag`.
fn amvp_pu(mvd: Mv, mvp_flag: u8) -> PredictionUnit {
    let comp = |v: i32| MvdComponent {
        greater0_flag: u8::from(v != 0),
        greater1_flag: None,
        minus2: None,
        sign_flag: None,
        value: v,
    };
    PredictionUnit {
        merge_flag: false,
        merge_idx: None,
        inter_pred_idc: Some(InterPredIdc::PredL0),
        ref_idx_l0: Some(0),
        mvd_l0: Some([comp(mvd[0]), comp(mvd[1])]),
        mvp_l0_flag: Some(mvp_flag),
        ref_idx_l1: None,
        mvd_l1: None,
        mvp_l1_flag: None,
    }
}

/// §8.5.3.3 — the final clipped prediction for a whole 16x16 CTB PU
/// with uni-L0 motion `mv` (luma quarter-pel). `chroma` selects
/// whether the Cb / Cr planes are predicted too (the SAD search runs
/// luma-only).
fn predict_ctb(refp: &RefPlanes, x0: usize, y0: usize, mv: Mv, chroma: bool) -> InterPrediction {
    let luma = RefPlane::new(&refp.y, refp.width, refp.height).expect("legal ref plane");
    let (cb, cr) = if chroma {
        (
            Some(RefPlane::new(&refp.cb, refp.width / 2, refp.height / 2).expect("legal plane")),
            Some(RefPlane::new(&refp.cr, refp.width / 2, refp.height / 2).expect("legal plane")),
        )
    } else {
        (None, None)
    };
    let l0 = ListPrediction {
        pred_flag: true,
        luma,
        cb,
        cr,
        mv_l: mv,
        // §8.5.3.2.10: 4:2:0 chroma MV = luma MV (eighth-pel units).
        mv_c: derive_chroma_mv(mv, 2, 2),
    };
    // L1 unused (uni-predictive P): pred_flag false, planes ignored.
    let l1 = ListPrediction {
        pred_flag: false,
        ..l0
    };
    let geom = InterPredGeometry {
        x_pb: x0 as i32,
        y_pb: y0 as i32,
        n_pb_w: CTB,
        n_pb_h: CTB,
        chroma_array_type: if chroma { 1 } else { 0 },
        bit_depth_luma: BIT_DEPTH,
        bit_depth_chroma: BIT_DEPTH,
    };
    predict_inter_pu(&l0, &l1, &geom).expect("legal prediction geometry")
}

/// The previous frame's reconstruction as `i32` planes (the
/// §8.5.3.3.3 interpolation input).
struct RefPlanes {
    y: Vec<i32>,
    cb: Vec<i32>,
    cr: Vec<i32>,
    width: usize,
    height: usize,
}

/// Luma SAD between a prediction and the source block.
fn sad(pred: &[i32], src: &[i32]) -> u64 {
    pred.iter()
        .zip(src.iter())
        .map(|(&p, &s)| u64::from(p.abs_diff(s)))
        .sum()
}

/// Rate proxy (bins) for one signed mvd component.
fn mvd_component_bits(v: i32) -> u64 {
    match v.unsigned_abs() {
        0 => 1,
        1 => 3,
        a => 5 + 2 * u64::from(32 - (a - 1).leading_zeros()),
    }
}

/// Rate proxy for a full mvd pair.
fn mvd_bits(mvd: Mv) -> u64 {
    mvd_component_bits(mvd[0]) + mvd_component_bits(mvd[1])
}

/// Crude bit-cost proxy for one TB's quantized levels (mirrors the
/// intra encoder's heuristic).
fn levels_rate(levels: &[i32]) -> u64 {
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

/// Transform-code the three components of a CTB against `pred`,
/// reconstructing through the decode-side path. Returns
/// `(levels, recon, ssd_total, rate)`.
fn code_ctb_residual(
    pred: &InterPrediction,
    src: &[Vec<i32>; 3],
    qp_y: u32,
    qp_c: u32,
) -> ([Vec<i32>; 3], [Vec<u8>; 3], u64, u64) {
    let (ly, ry) = code_tb(
        &src[0],
        &pred.luma,
        CTB,
        qp_y,
        Component::Luma,
        PredMode::Inter,
    );
    let (lcb, rcb) = code_tb(&src[1], &pred.cb, 8, qp_c, Component::Cb, PredMode::Inter);
    let (lcr, rcr) = code_tb(&src[2], &pred.cr, 8, qp_c, Component::Cr, PredMode::Inter);
    let dist = ssd(&ry, &src[0]) + ssd(&rcb, &src[1]) + ssd(&rcr, &src[2]);
    let rate = levels_rate(&ly) + levels_rate(&lcb) + levels_rate(&lcr);
    ([ly, lcb, lcr], [ry, rcb, rcr], dist, rate)
}

/// The prediction-only reconstruction (clip to 8-bit) and its SSD.
fn pred_recon(pred: &InterPrediction, src: &[Vec<i32>; 3]) -> ([Vec<u8>; 3], u64) {
    let clip = |v: &Vec<i32>| -> Vec<u8> { v.iter().map(|&p| p.clamp(0, 255) as u8).collect() };
    let recon = [clip(&pred.luma), clip(&pred.cb), clip(&pred.cr)];
    let dist = ssd(&recon[0], &src[0]) + ssd(&recon[1], &src[1]) + ssd(&recon[2], &src[2]);
    (recon, dist)
}

/// Greedy integer-pel motion search on luma SAD: seeds, then a
/// small-diamond descent (capped). Returns the best full-pel MV in
/// luma samples.
fn integer_me(
    refp: &RefPlanes,
    src_y: &[i32],
    x0: usize,
    y0: usize,
    seeds: &[[i32; 2]],
    lambda_me: u64,
    mvp_q: &[Mv],
) -> [i32; 2] {
    // Full-pel prediction == direct (clamped) reference samples.
    let sad_at = |mx: i32, my: i32| -> u64 {
        let mut acc = 0u64;
        for j in 0..CTB {
            for i in 0..CTB {
                let rx = (x0 as i32 + i as i32 + mx).clamp(0, refp.width as i32 - 1);
                let ry = (y0 as i32 + j as i32 + my).clamp(0, refp.height as i32 - 1);
                let r = refp.y[ry as usize * refp.width + rx as usize];
                acc += u64::from(r.abs_diff(src_y[j * CTB + i]));
            }
        }
        acc
    };
    // Motion-cost proxy at full-pel: cheapest mvd against either MVP.
    let mv_cost = |mx: i32, my: i32| -> u64 {
        mvp_q
            .iter()
            .map(|p| mvd_bits([mx * 4 - p[0], my * 4 - p[1]]))
            .min()
            .unwrap_or(0)
            * lambda_me
    };
    let mut best = seeds[0];
    let mut best_cost = u64::MAX;
    for &s in seeds {
        let c = sad_at(s[0], s[1]) + mv_cost(s[0], s[1]);
        if c < best_cost {
            best_cost = c;
            best = s;
        }
    }
    for _ in 0..ME_MAX_STEPS {
        let mut improved = false;
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (mx, my) = (best[0] + dx, best[1] + dy);
            let c = sad_at(mx, my) + mv_cost(mx, my);
            if c < best_cost {
                best_cost = c;
                best = [mx, my];
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    best
}

/// Half- then quarter-pel refinement around `best` (quarter-pel
/// units), scoring with the real §8.5.3.3.3 interpolation.
fn fractional_me(
    refp: &RefPlanes,
    src_y: &[i32],
    x0: usize,
    y0: usize,
    start: Mv,
    lambda_me: u64,
    mvp_q: &[Mv],
) -> Mv {
    let cost_at = |mv: Mv| -> u64 {
        let pred = predict_ctb(refp, x0, y0, mv, false);
        let mv_rate = mvp_q
            .iter()
            .map(|p| mvd_bits([mv[0] - p[0], mv[1] - p[1]]))
            .min()
            .unwrap_or(0);
        sad(&pred.luma, src_y) + lambda_me * mv_rate
    };
    let mut best = start;
    let mut best_cost = cost_at(best);
    for step in [2i32, 1] {
        let mut improved = true;
        while improved {
            improved = false;
            for (dx, dy) in [
                (step, 0),
                (-step, 0),
                (0, step),
                (0, -step),
                (step, step),
                (step, -step),
                (-step, step),
                (-step, -step),
            ] {
                let mv = [best[0] + dx, best[1] + dy];
                let c = cost_at(mv);
                if c < best_cost {
                    best_cost = c;
                    best = mv;
                    improved = true;
                }
            }
        }
    }
    best
}

/// Encode `merge_idx` (§9.3.3.10 TR with `cMax = MaxNumMergeCand − 1`:
/// bin 0 context-coded, the rest bypass).
fn encode_merge_idx(
    w: &mut BitWriter,
    cabac: &mut CabacEncoder,
    ctxs: &mut SliceContexts,
    idx: usize,
) {
    let c_max = MAX_MERGE - 1;
    cabac.encode_decision(w, &mut ctxs.merge_idx[0], u8::from(idx > 0));
    if idx > 0 {
        for _ in 1..idx {
            cabac.encode_bypass(w, 1);
        }
        if idx < c_max {
            cabac.encode_bypass(w, 0);
        }
    }
}

/// §9.3.3.3 — encode an EGk-coded value (the dual of the decoder's
/// `read_eg_k_with`): a run of `1` prefix bins, a `0`, then
/// `prefix_ones + k` MSB-first suffix bins.
fn encode_eg_k(w: &mut BitWriter, cabac: &mut CabacEncoder, value: u32, k: u32) {
    let mut prefix_ones = 0u32;
    while ((1u64 << (prefix_ones + 1)) - 1) << k <= u64::from(value) {
        prefix_ones += 1;
    }
    for _ in 0..prefix_ones {
        cabac.encode_bypass(w, 1);
    }
    cabac.encode_bypass(w, 0);
    let base = ((1u64 << prefix_ones) - 1) << k;
    let suffix = (u64::from(value) - base) as u32;
    cabac.encode_bypass_bits(w, suffix, (prefix_ones + k) as u8);
}

/// §7.3.8.9 `mvd_coding( )` — the interleaved bin order the decoder's
/// `decode_mvd_pair` reads: both `abs_mvd_greater0_flag`s, both
/// `abs_mvd_greater1_flag`s, then per component the EG1
/// `abs_mvd_minus2` escape and `mvd_sign_flag`.
fn encode_mvd_pair(w: &mut BitWriter, cabac: &mut CabacEncoder, ctxs: &mut SliceContexts, mvd: Mv) {
    let g0 = [mvd[0] != 0, mvd[1] != 0];
    for &g in &g0 {
        cabac.encode_decision(w, &mut ctxs.abs_mvd_greater0_flag[0], u8::from(g));
    }
    for (&g, &v) in g0.iter().zip(mvd.iter()) {
        if g {
            cabac.encode_decision(
                w,
                &mut ctxs.abs_mvd_greater1_flag[0],
                u8::from(v.unsigned_abs() > 1),
            );
        }
    }
    for (&g, &v) in g0.iter().zip(mvd.iter()) {
        if g {
            let a = v.unsigned_abs();
            if a > 1 {
                encode_eg_k(w, cabac, a - 2, 1);
            }
            cabac.encode_bypass(w, u8::from(v < 0));
        }
    }
}

/// Encode one P frame as a single TRAIL_R slice against the previous
/// frame's reconstruction. Returns the slice RBSP, the frame's
/// reconstruction, and its CU mode-decision counters.
#[allow(clippy::too_many_lines)]
fn encode_p_slice(
    frame: &YuvFrame<'_>,
    prev: &FrameRecon,
    poc: i32,
    width: usize,
    height: usize,
    qp: i32,
) -> (Vec<u8>, FrameRecon, FrameStats) {
    let (cw, ch) = (width / 2, height / 2);
    let ctbs_x = width / CTB;
    let ctbs_y = height / CTB;
    let qp_y = qp as u32;
    let qp_c = chroma_qp_420(qp);
    // SSD-per-bit tradeoff, doubling every 3 QP (as the intra path);
    // the SAD-domain motion-search λ uses the same scale.
    let lambda: u64 = 1u64 << (qp.unsigned_abs().saturating_sub(9) / 3);
    let lambda_me: u64 = lambda;

    let to_i32 = |p: &[u8]| -> Vec<i32> { p.iter().map(|&v| i32::from(v)).collect() };
    let refp = RefPlanes {
        y: to_i32(&prev.y),
        cb: to_i32(&prev.cb),
        cr: to_i32(&prev.cr),
        width,
        height,
    };

    let mut recon = FrameRecon {
        y: vec![0u8; width * height],
        cb: vec![0u8; cw * ch],
        cr: vec![0u8; cw * ch],
    };
    let mut field = MotionField::new(width, height);
    // §8.4.2 luma-mode records for the intra-fallback MPM lists (an
    // inter / skip CU records as "not intra" ⇒ candidate INTRA_DC).
    let mut modes = IntraModeField::new(width, height, CTB_LOG2);
    // Per-CTB cu_skip_flag values (CU == CTB) for the §9.3.4.2.2 ctxInc.
    let mut skip_grid = vec![false; ctbs_x * ctbs_y];
    let mut stats = FrameStats::default();

    // §6.4.2 availability plumbing (single slice, single tile).
    let tiling = PictureTiling::new(
        ctbs_x as u32,
        ctbs_y as u32,
        width as u32,
        height as u32,
        CTB_LOG2,
        2,
        &TilingParams::single_tile(),
    )
    .expect("legal single-tile geometry");

    // §8.5.3.2 reference resolvers: one short-term reference, POC − 1.
    let ref_poc = |_list: usize, ref_idx: i32| if ref_idx == 0 { poc - 1 } else { i32::MIN };
    let ref_long_term = |_list: usize, _ref_idx: i32| false;
    let ref_short_term = |_list: usize, ref_idx: i32| ref_idx == 0;
    let col_ref_long_term = |_poc: i32| false;
    let mv_ctx = PuMvContext {
        curr_poc: poc,
        slice_is_b: false,
        ctb_log2_size_y: CTB_LOG2,
        pic_width_luma: width as u32,
        pic_height_luma: height as u32,
        max_num_merge_cand: MAX_MERGE,
        num_ref_idx_l0_active: 1,
        num_ref_idx_l1_active: 0,
        log2_par_mrg_level: 2,
        temporal_mvp_enabled: false,
        collocated_from_l0_flag: true,
        col_poc: 0,
        no_backward_pred: true,
        ref_poc: &ref_poc,
        ref_long_term: &ref_long_term,
        ref_short_term: &ref_short_term,
        col_field: None,
        col_ref_long_term: &col_ref_long_term,
    };

    // ---- slice_segment_header( ) ----
    let mut w = BitWriter::new();
    write_p_slice_header(&mut w, poc, qp);

    // ---- slice_segment_data( ) ----
    let mut cabac = CabacEncoder::new();
    // Table 9-4: P slice, cabac_init_flag 0 ⇒ initType 1.
    let mut ctxs = SliceContexts::init(init_type(1, false), qp);

    for ctb in 0..ctbs_x * ctbs_y {
        let x0 = (ctb % ctbs_x) * CTB;
        let y0 = (ctb / ctbs_x) * CTB;
        let (cx0, cy0) = (x0 / 2, y0 / 2);
        let src = [
            extract(frame.y, width, x0, y0, CTB),
            extract(frame.cb, cw, cx0, cy0, 8),
            extract(frame.cr, cw, cx0, cy0, 8),
        ];

        let geom = PuGeometry {
            x_cb: x0,
            y_cb: y0,
            n_cb_s: CTB,
            x_pb: x0,
            y_pb: y0,
            n_pb_w: CTB,
            n_pb_h: CTB,
            part_mode: PartMode::Part2Nx2N,
            part_idx: 0,
        };

        let chosen = {
            // §6.4.2 prediction-block availability against the
            // in-progress motion field (identical to the decoder's
            // closure in the picture-level inter driver).
            let available = |x_nb: i32, y_nb: i32| -> bool {
                tiling.prediction_block_availability(
                    x0 as u32,
                    y0 as u32,
                    CTB as u32,
                    x0 as u32,
                    y0 as u32,
                    CTB as u32,
                    CTB as u32,
                    0,
                    x_nb,
                    y_nb,
                    |_ctb_rs| 0,
                    |x, y| {
                        if field.cell_at(x as usize, y as usize).is_intra {
                            MODE_INTRA
                        } else {
                            0
                        }
                    },
                )
            };

            // ---- merge / skip candidates (§8.5.3.2.2) ----
            let mut merge_cands: Vec<(usize, PuMotion)> = Vec::with_capacity(MAX_MERGE);
            for idx in 0..MAX_MERGE {
                let m = resolve_pu_motion(&field, &geom, &merge_pu(idx), &mv_ctx, &available);
                // A later duplicate can never beat the earlier index
                // (same samples, more merge_idx bins).
                if !merge_cands.iter().any(|(_, prev)| *prev == m) {
                    merge_cands.push((idx, m));
                }
            }
            let (best_merge_idx, best_merge_motion) = merge_cands
                .iter()
                .map(|&(idx, m)| {
                    let pred = predict_ctb(&refp, x0, y0, m.mv_l0, false);
                    (
                        sad(&pred.luma, &src[0]) + lambda_me * (idx as u64 + 1),
                        idx,
                        m,
                    )
                })
                .min_by_key(|&(cost, idx, _)| (cost, idx))
                .map(|(_, idx, m)| (idx, m))
                .expect("merge list is never empty");

            // ---- AMVP: §8.5.3.2.6 predictors + motion estimation ----
            let mvp = [
                resolve_pu_motion(&field, &geom, &amvp_pu([0, 0], 0), &mv_ctx, &available).mv_l0,
                resolve_pu_motion(&field, &geom, &amvp_pu([0, 0], 1), &mv_ctx, &available).mv_l0,
            ];
            let mut seeds: Vec<[i32; 2]> = vec![[0, 0]];
            for p in &mvp {
                seeds.push([p[0] >> 2, p[1] >> 2]);
            }
            for (_, m) in &merge_cands {
                seeds.push([m.mv_l0[0] >> 2, m.mv_l0[1] >> 2]);
            }
            seeds.dedup();
            let int_mv = integer_me(&refp, &src[0], x0, y0, &seeds, lambda_me, &mvp);
            let me_mv = fractional_me(
                &refp,
                &src[0],
                x0,
                y0,
                [int_mv[0] * 4, int_mv[1] * 4],
                lambda_me,
                &mvp,
            );
            // Choose the cheaper predictor for the found MV.
            let (mvp_flag, mvd) = (0u8..2)
                .map(|f| {
                    let p = mvp[f as usize];
                    (f, [me_mv[0] - p[0], me_mv[1] - p[1]])
                })
                .min_by_key(|&(_, d)| mvd_bits(d))
                .expect("two predictors");
            // Resolve through the decoder's derivation (mv wrap etc.).
            let amvp_motion =
                resolve_pu_motion(&field, &geom, &amvp_pu(mvd, mvp_flag), &mv_ctx, &available);

            // ---- fully code the finalists ----
            let mut cands: Vec<CuCandidate> = Vec::with_capacity(3);

            // Skip: best merge candidate, prediction only.
            let merge_pred = predict_ctb(&refp, x0, y0, best_merge_motion.mv_l0, true);
            let (skip_recon, skip_dist) = pred_recon(&merge_pred, &src);
            cands.push(CuCandidate {
                kind: CuKind::Skip {
                    merge_idx: best_merge_idx,
                },
                motion: best_merge_motion,
                levels: None,
                recon: skip_recon,
                cost: skip_dist + lambda * (best_merge_idx as u64 + 2),
            });

            // Merge + residual (only legal when some level is nonzero:
            // a 2Nx2N merge CU has rqt_root_cbf inferred 1).
            let (m_levels, m_recon, m_dist, m_rate) =
                code_ctb_residual(&merge_pred, &src, qp_y, qp_c);
            if m_levels.iter().any(|l| l.iter().any(|&v| v != 0)) {
                cands.push(CuCandidate {
                    kind: CuKind::Merge {
                        merge_idx: best_merge_idx,
                    },
                    motion: best_merge_motion,
                    levels: Some(m_levels),
                    recon: m_recon,
                    cost: m_dist + lambda * (m_rate + best_merge_idx as u64 + 3),
                });
            }

            // AMVP (with or without residual).
            let amvp_pred = predict_ctb(&refp, x0, y0, amvp_motion.mv_l0, true);
            let (a_levels, a_recon, a_dist, a_rate) =
                code_ctb_residual(&amvp_pred, &src, qp_y, qp_c);
            let motion_rate = mvd_bits(mvd) + 4; // mvd + mvp/merge/skip/rqt flags
            if a_levels.iter().any(|l| l.iter().any(|&v| v != 0)) {
                cands.push(CuCandidate {
                    kind: CuKind::Amvp { mvd, mvp_flag },
                    motion: amvp_motion,
                    levels: Some(a_levels),
                    recon: a_recon,
                    cost: a_dist + lambda * (a_rate + motion_rate),
                });
            } else {
                // All-zero residual: rqt_root_cbf == 0.
                let (pr, pd) = pred_recon(&amvp_pred, &src);
                cands.push(CuCandidate {
                    kind: CuKind::Amvp { mvd, mvp_flag },
                    motion: amvp_motion,
                    levels: None,
                    recon: pr,
                    cost: pd + lambda * motion_rate,
                });
            }

            // ---- intra 2Nx2N fallback (§8.4 against our own recon) ----
            {
                let read_y = |x: usize, yy: usize| i32::from(recon.y[yy * width + x]);
                let avail_y =
                    |nx: i64, ny: i64| zscan_avail(nx, ny, width, height, CTB, ctbs_x, ctb, 0);
                let marked = gather_refs(&read_y, &avail_y, x0, y0, CTB);
                let (mode, pred_y) = search_best_mode(&marked, &src[0]);
                let (ly, ry) = code_tb(
                    &src[0],
                    &pred_y,
                    CTB,
                    qp_y,
                    Component::Luma,
                    PredMode::Intra,
                );
                let code_c = |plane: &[u8], comp: Component, pc: PredComponent| {
                    let read = |x: usize, yy: usize| i32::from(plane[yy * cw + x]);
                    let avail =
                        |nx: i64, ny: i64| zscan_avail(nx, ny, cw, ch, CTB / 2, ctbs_x, ctb, 0);
                    let marked_c = gather_refs(&read, &avail, cx0, cy0, 8);
                    let pred = intra_predict_with_substitution(&marked_c, &pred_params(mode, pc))
                        .expect("legal prediction params");
                    let src_c = match comp {
                        Component::Cb => &src[1],
                        _ => &src[2],
                    };
                    code_tb(src_c, &pred, 8, qp_c, comp, PredMode::Intra)
                };
                let (lcb, rcb) = code_c(&recon.cb, Component::Cb, PredComponent::Cb);
                let (lcr, rcr) = code_c(&recon.cr, Component::Cr, PredComponent::Cr);
                let dist = ssd(&ry, &src[0]) + ssd(&rcb, &src[1]) + ssd(&rcr, &src[2]);
                let rate = levels_rate(&ly) + levels_rate(&lcb) + levels_rate(&lcr) + 8;
                cands.push(CuCandidate {
                    kind: CuKind::Intra { mode },
                    motion: PuMotion::default(),
                    levels: Some([ly, lcb, lcr]),
                    recon: [ry, rcb, rcr],
                    cost: dist + lambda * rate,
                });
            }

            cands
                .into_iter()
                .min_by_key(|c| c.cost)
                .expect("at least the skip candidate")
        };

        // ---- emit the §7.3.8.5 coding_unit( ) syntax ----
        let is_skip = matches!(chosen.kind, CuKind::Skip { .. });
        {
            // cu_skip_flag with the §9.3.4.2.2 left/above ctxInc (CU ==
            // CTB: the neighbour is the raster-preceding CTB, available
            // iff in-picture — single slice, single tile).
            let (l_avail, l_skip) = if x0 > 0 {
                (true, skip_grid[ctb - 1])
            } else {
                (false, false)
            };
            let (a_avail, a_skip) = if y0 > 0 {
                (true, skip_grid[ctb - ctbs_x])
            } else {
                (false, false)
            };
            let inc = cu_skip_flag_ctx_inc(u8::from(l_skip), l_avail, u8::from(a_skip), a_avail);
            cabac.encode_decision(
                &mut w,
                &mut ctxs.cu_skip_flag[inc as usize],
                u8::from(is_skip),
            );
        }
        skip_grid[ctb] = is_skip;

        match &chosen.kind {
            CuKind::Skip { merge_idx } => {
                encode_merge_idx(&mut w, &mut cabac, &mut ctxs, *merge_idx);
            }
            CuKind::Merge { merge_idx } => {
                // pred_mode_flag = 0 (MODE_INTER), part_mode "1" (2Nx2N).
                cabac.encode_decision(&mut w, &mut ctxs.pred_mode_flag[0], 0);
                cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], 1);
                // prediction_unit: merge_flag = 1, merge_idx.
                cabac.encode_decision(&mut w, &mut ctxs.merge_flag[0], 1);
                encode_merge_idx(&mut w, &mut cabac, &mut ctxs, *merge_idx);
                // rqt_root_cbf not present (2Nx2N merge ⇒ inferred 1).
            }
            CuKind::Amvp { mvd, mvp_flag } => {
                cabac.encode_decision(&mut w, &mut ctxs.pred_mode_flag[0], 0);
                cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], 1);
                cabac.encode_decision(&mut w, &mut ctxs.merge_flag[0], 0);
                // P slice ⇒ PRED_L0 inferred; one active ref ⇒ no
                // ref_idx_l0. mvd_coding + mvp_l0_flag.
                encode_mvd_pair(&mut w, &mut cabac, &mut ctxs, *mvd);
                cabac.encode_decision(&mut w, &mut ctxs.mvp_flag[0], *mvp_flag);
                cabac.encode_decision(
                    &mut w,
                    &mut ctxs.rqt_root_cbf[0],
                    u8::from(chosen.levels.is_some()),
                );
            }
            CuKind::Intra { mode } => {
                // pred_mode_flag = 1 (MODE_INTRA), part_mode "1"
                // (2Nx2N at MinCb), PCM disabled in the SPS.
                cabac.encode_decision(&mut w, &mut ctxs.pred_mode_flag[0], 1);
                cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], 1);
                // §7.3.8.5 luma-mode group (single PB): the §8.4.2
                // candidate list against the recorded neighbour modes
                // (inter / skip neighbours contribute INTRA_DC), with
                // the §6.4.1 z-scan availability.
                let avail_l =
                    zscan_avail(x0 as i64 - 1, y0 as i64, width, height, CTB, ctbs_x, ctb, 0);
                let avail_a =
                    zscan_avail(x0 as i64, y0 as i64 - 1, width, height, CTB, ctbs_x, ctb, 0);
                let cand_a = modes.cand_intra_pred_mode(x0, y0, Neighbour::Left, avail_l);
                let cand_b = modes.cand_intra_pred_mode(x0, y0, Neighbour::Above, avail_a);
                let list = intra_luma_cand_mode_list(cand_a, cand_b);
                match list.iter().position(|&m| m == *mode) {
                    Some(k) => {
                        cabac.encode_decision(&mut w, &mut ctxs.prev_intra_luma_pred_flag[0], 1);
                        // mpm_idx: TR cMax 2, all bypass.
                        match k {
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
                    }
                    None => {
                        cabac.encode_decision(&mut w, &mut ctxs.prev_intra_luma_pred_flag[0], 0);
                        // §8.4.2: rem = mode minus each smaller candidate.
                        let mut rem = u32::from(*mode);
                        for &c in &list {
                            if u32::from(*mode) > u32::from(c) {
                                rem -= 1;
                            }
                        }
                        cabac.encode_bypass_bits(&mut w, rem, 5); // FL cMax 31
                    }
                }
                // intra_chroma_pred_mode = 4 (derived from luma).
                cabac.encode_decision(&mut w, &mut ctxs.intra_chroma_pred_mode[0], 0);
            }
        }

        // ---- §7.3.8.8 transform_tree + §7.3.8.10 transform_unit ----
        let cu_is_intra = matches!(chosen.kind, CuKind::Intra { .. });
        if let Some(levels) = &chosen.levels {
            // Depth-0 16x16 TU (split_transform_flag not present,
            // inferred 0: 2Nx2N at MaxTrafoDepth 0).
            let cbf_cb = levels[1].iter().any(|&v| v != 0);
            let cbf_cr = levels[2].iter().any(|&v| v != 0);
            let cbf_luma = levels[0].iter().any(|&v| v != 0);
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
            // §7.3.8.8: an intra CU always signals cbf_luma; an inter
            // depth-0 TU signals it only when a chroma cbf is set
            // (otherwise it is inferred 1).
            if cu_is_intra || cbf_cb || cbf_cr {
                cabac.encode_decision(
                    &mut w,
                    &mut ctxs.cbf_luma[cbf_luma_ctx_inc(0) as usize],
                    u8::from(cbf_luma),
                );
            } else {
                debug_assert!(cbf_luma, "all-zero transform tree must not be coded");
            }
            // §7.4.9.11: inter TBs use the up-right diagonal scan; the
            // 16x16 luma / 8x8 chroma intra TBs are not
            // mode-dependent-scan eligible so they come back diagonal
            // through the same derivation.
            let intra_mode = match &chosen.kind {
                CuKind::Intra { mode } => u32::from(*mode),
                _ => 0,
            };
            let rc_params = |log2: u32, c_idx: u8| ResidualCodingParams {
                log2_trafo_size: log2,
                is_chroma: c_idx != 0,
                scan_idx: residual_coding_scan_idx(cu_is_intra, log2, c_idx, 1, intra_mode),
                sign_data_hiding_enabled_flag: false,
                sign_hidden_suppressed: false,
                transform_skip_sig_ctx: false,
            };
            if cbf_luma {
                encode_residual_coding(
                    &mut w,
                    &mut cabac,
                    &mut ctxs.residual,
                    &rc_params(4, 0),
                    &levels[0],
                )
                .expect("validated luma levels");
            }
            if cbf_cb {
                encode_residual_coding(
                    &mut w,
                    &mut cabac,
                    &mut ctxs.residual,
                    &rc_params(3, 1),
                    &levels[1],
                )
                .expect("validated cb levels");
            }
            if cbf_cr {
                encode_residual_coding(
                    &mut w,
                    &mut cabac,
                    &mut ctxs.residual,
                    &rc_params(3, 2),
                    &levels[2],
                )
                .expect("validated cr levels");
            }
        }

        // ---- state update (eqs 8-80..8-85 + mode fields + recon) ----
        match &chosen.kind {
            CuKind::Intra { mode } => {
                stats.intra += 1;
                // The decoder stamps intra CUs into the motion field so
                // a later CU's §6.4.2 availability denies them.
                field.fill_rect(
                    x0,
                    y0,
                    CTB,
                    CTB,
                    MotionCell {
                        is_intra: true,
                        ..MotionCell::default()
                    },
                );
                modes.record_intra_pb(x0, y0, CTB, *mode, false);
            }
            kind => {
                match kind {
                    CuKind::Skip { .. } => stats.skip += 1,
                    CuKind::Merge { .. } => stats.merge += 1,
                    _ => stats.amvp += 1,
                }
                field.fill_rect(x0, y0, CTB, CTB, chosen.motion.to_cell(poc - 1, i32::MIN));
                let cu_mode = if is_skip {
                    CuPredMode::Skip
                } else {
                    CuPredMode::Inter
                };
                modes.record_non_intra_cu(x0, y0, CTB, cu_mode);
            }
        }
        if let Some(levels) = &chosen.levels {
            if levels[0].iter().any(|&v| v != 0) {
                field.mark_nonzero_coeff(x0, y0, CTB, CTB);
            }
        }
        store(&mut recon.y, width, x0, y0, CTB, &chosen.recon[0]);
        store(&mut recon.cb, cw, cx0, cy0, 8, &chosen.recon[1]);
        store(&mut recon.cr, cw, cx0, cy0, 8, &chosen.recon[2]);

        // end_of_slice_segment_flag.
        cabac.encode_terminate(&mut w, u8::from(ctb == ctbs_x * ctbs_y - 1));
    }
    w.align_zero();
    (w.finish(), recon, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::decode_annexb_sequence;

    /// A deterministic scene: a textured background with a moving
    /// bright square (sub-pel-friendly gradients) — motion vectors and
    /// residuals both get exercised.
    fn scene(w: usize, h: usize, n_frames: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        (0..n_frames)
            .map(|t| {
                let y: Vec<u8> = (0..w * h)
                    .map(|i| {
                        let (x, yy) = (i % w, i / w);
                        let base = (x * 2 + yy * 3) % 200;
                        // A 12x12 square moving 3 px right / 1 px down
                        // per frame.
                        let (sx, sy) = (4 + t * 3, 6 + t);
                        let inside = x >= sx && x < sx + 12 && yy >= sy && yy < sy + 12;
                        if inside {
                            (base + 55) as u8
                        } else {
                            base as u8
                        }
                    })
                    .collect();
                let cb: Vec<u8> = (0..w * h / 4)
                    .map(|i| (100 + (i % (w / 2)) * 2 % 60 + t) as u8)
                    .collect();
                let cr: Vec<u8> = (0..w * h / 4)
                    .map(|i| {
                        (160u32.wrapping_sub((i / (w / 2)) as u32 * 2 % 50) as u8)
                            .wrapping_sub(t as u8)
                    })
                    .collect();
                (y, cb, cr)
            })
            .collect()
    }

    fn as_frames(planes: &[(Vec<u8>, Vec<u8>, Vec<u8>)]) -> Vec<YuvFrame<'_>> {
        planes
            .iter()
            .map(|(y, cb, cr)| YuvFrame { y, cb, cr })
            .collect()
    }

    fn assert_gop_roundtrip(w: usize, h: usize, n: usize, qp: i32) {
        let planes = scene(w, h, n);
        let frames = as_frames(&planes);
        let enc = encode_low_delay_p(&frames, w, h, qp).expect("encode");
        assert_eq!(enc.recon.len(), n);
        let decoded = decode_annexb_sequence(&enc.stream).expect("decode");
        assert_eq!(decoded.len(), n, "{w}x{h} qp{qp}: frame count");
        for (i, (dec, rec)) in decoded.iter().zip(enc.recon.iter()).enumerate() {
            let mut expect = rec.y.clone();
            expect.extend_from_slice(&rec.cb);
            expect.extend_from_slice(&rec.cr);
            assert_eq!(
                dec.picture.to_planar_u8().expect("8-bit"),
                expect,
                "{w}x{h} qp{qp}: frame {i} decoder output == encoder recon"
            );
        }
    }

    /// The core contract: our decoder reproduces every frame of the
    /// low-delay P GOP bit-exactly, across sizes and QPs.
    #[test]
    fn p_gop_decodes_to_encoder_recon_exactly() {
        assert_gop_roundtrip(64, 64, 4, 22);
        assert_gop_roundtrip(64, 48, 3, 32);
        assert_gop_roundtrip(48, 80, 3, 10);
        assert_gop_roundtrip(16, 16, 2, 26);
    }

    /// High QP drives most CTUs to skip; the stream stays decodable
    /// and P frames get much smaller than the IDR.
    #[test]
    fn p_frames_compress_against_reference() {
        let planes = scene(64, 64, 5);
        let frames = as_frames(&planes);
        let enc = encode_low_delay_p(&frames, 64, 64, 30).expect("encode");
        // Rough split: IDR AU ends where the first TRAIL_R start code
        // begins. P payload = total − IDR.
        let idr = encode_idr_intra_au(&planes[0].0, &planes[0].1, &planes[0].2, 64, 64, 30)
            .expect("intra encode");
        let p_bytes = enc.stream.len() - idr.au.len();
        let p_avg = p_bytes / 4;
        assert!(
            p_avg * 3 < idr.au.len(),
            "P frames ({p_avg} B avg) should be much smaller than the IDR ({} B)",
            idr.au.len()
        );
    }

    /// A static scene (all frames identical) should code P frames as
    /// (almost) all-skip: tiny payloads, bit-exact reconstruction.
    #[test]
    fn static_scene_is_skip_coded() {
        let one = scene(64, 64, 1).remove(0);
        let planes = vec![one.clone(), one.clone(), one];
        let frames = as_frames(&planes);
        let enc = encode_low_delay_p(&frames, 64, 64, 26).expect("encode");
        let decoded = decode_annexb_sequence(&enc.stream).expect("decode");
        assert_eq!(decoded.len(), 3);
        // Frames 1 and 2 must reproduce frame 0's reconstruction
        // exactly (skip copies the reference).
        let f0 = decoded[0].picture.to_planar_u8().expect("8-bit");
        for d in &decoded[1..] {
            assert_eq!(d.picture.to_planar_u8().expect("8-bit"), f0);
        }
        // Each all-skip P slice is a handful of bytes.
        let idr_len = encode_idr_intra_au(&planes[0].0, &planes[0].1, &planes[0].2, 64, 64, 26)
            .expect("intra")
            .au
            .len();
        let p_bytes = enc.stream.len() - idr_len;
        assert!(
            p_bytes < 80,
            "two all-skip 64x64 P frames should be tiny, got {p_bytes} B"
        );
    }

    /// Motion estimation actually finds the moving square: at a
    /// moderate QP the P-frame quality stays close to the intra
    /// quality while spending far fewer bits.
    #[test]
    fn motion_estimation_tracks_translation() {
        let planes = scene(64, 64, 4);
        let frames = as_frames(&planes);
        let enc = encode_low_delay_p(&frames, 64, 64, 22).expect("encode");
        for (t, rec) in enc.recon.iter().enumerate() {
            let mse: f64 = rec
                .y
                .iter()
                .zip(planes[t].0.iter())
                .map(|(&a, &b)| {
                    let d = f64::from(a) - f64::from(b);
                    d * d
                })
                .sum::<f64>()
                / rec.y.len() as f64;
            let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10();
            assert!(psnr > 33.0, "frame {t} luma PSNR {psnr:.1} dB too low");
        }
    }

    /// Validation of the shared input contract.
    #[test]
    fn rejects_bad_inputs() {
        let planes = scene(16, 16, 1);
        let frames = as_frames(&planes);
        assert!(matches!(
            encode_low_delay_p(&frames, 20, 16, 26),
            Err(IntraEncodeError::BadDimensions { .. })
        ));
        assert!(matches!(
            encode_low_delay_p(&frames, 16, 16, 52),
            Err(IntraEncodeError::BadQp(52))
        ));
        assert!(matches!(
            encode_low_delay_p(&frames, 32, 32, 26),
            Err(IntraEncodeError::PlaneSize { .. })
        ));
        let empty = encode_low_delay_p(&[], 16, 16, 26).expect("empty ok");
        assert!(empty.stream.is_empty() && empty.recon.is_empty());
    }

    /// A hard scene change (unrelated content mid-GOP) drives the
    /// per-CTU decision to the intra fallback — and the stream still
    /// decodes bit-exactly to the encoder reconstruction.
    #[test]
    fn scene_change_selects_intra_cus_in_p_slice() {
        let (w, h) = (64usize, 64usize);
        // Frames 0..2: the moving-square scene. Frame 2: a completely
        // different high-detail pattern (uncorrelated with frame 1).
        let mut planes = scene(w, h, 2);
        let y2: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, yy) = (i % w, i / w);
                (((x * 13) ^ (yy * 7)) % 251) as u8
            })
            .collect();
        let cb2 = vec![60u8; w * h / 4];
        let cr2 = vec![200u8; w * h / 4];
        planes.push((y2, cb2, cr2));
        let frames = as_frames(&planes);
        let enc = encode_low_delay_p(&frames, w, h, 27).expect("encode");

        // The scene-change P frame elects intra for most CTBs.
        let st = enc.stats[2];
        assert!(
            st.intra > (w / 16) * (h / 16) / 2,
            "scene change should be intra-dominated, got {st:?}"
        );
        // The steady frame before it stays inter-dominated (the CTBs
        // around the moving square may still legitimately elect intra
        // on the covered / uncovered texture).
        let st1 = enc.stats[1];
        assert!(
            st1.intra < (w / 16) * (h / 16) / 2,
            "steady frame should be inter-dominated, got {st1:?}"
        );

        // And the whole stream still decodes to the encoder recon.
        let decoded = decode_annexb_sequence(&enc.stream).expect("decode");
        assert_eq!(decoded.len(), 3);
        for (i, (dec, rec)) in decoded.iter().zip(enc.recon.iter()).enumerate() {
            let mut expect = rec.y.clone();
            expect.extend_from_slice(&rec.cb);
            expect.extend_from_slice(&rec.cr);
            assert_eq!(
                dec.picture.to_planar_u8().expect("8-bit"),
                expect,
                "frame {i}"
            );
        }
    }

    /// The mode-decision stats add up to the CTB count on every frame.
    #[test]
    fn stats_cover_every_ctb() {
        let planes = scene(48, 32, 3);
        let frames = as_frames(&planes);
        let enc = encode_low_delay_p(&frames, 48, 32, 30).expect("encode");
        assert_eq!(enc.stats.len(), 3);
        let ctbs = (48 / 16) * (32 / 16);
        for (i, st) in enc.stats.iter().enumerate() {
            assert_eq!(
                st.skip + st.merge + st.amvp + st.intra,
                ctbs,
                "frame {i}: {st:?}"
            );
        }
        assert_eq!(enc.stats[0].intra, ctbs, "IDR counts as all-intra");
    }

    /// The §7.3.8.9 `mvd_coding( )` encoder is the exact bin-level
    /// dual of the decoder's `decode_mvd_pair` (context flags, the EG1
    /// escape, and signs), across a signed sweep including the EG1
    /// prefix growth points.
    #[test]
    fn mvd_pair_roundtrips_through_cabac() {
        use crate::binarization::decode_mvd_pair;
        use crate::bitreader::BitReader;
        use crate::cabac::CabacEngine;
        let values = [
            0, 1, -1, 2, -2, 3, -3, 4, 5, -6, 7, 9, -12, 17, 33, -64, 129, -300,
        ];
        for &vx in &values {
            for &vy in &values {
                let mut w = BitWriter::new();
                let mut enc = CabacEncoder::new();
                let mut ectx = SliceContexts::init(1, 26);
                encode_mvd_pair(&mut w, &mut enc, &mut ectx, [vx, vy]);
                enc.encode_terminate(&mut w, 1);
                let bytes = w.finish();

                let mut engine = CabacEngine::new(BitReader::new(&bytes)).expect("engine init");
                let mut dctx = SliceContexts::init(1, 26);
                let got = decode_mvd_pair(
                    &mut engine,
                    &mut dctx.abs_mvd_greater0_flag[0],
                    &mut dctx.abs_mvd_greater1_flag[0],
                )
                .expect("decode");
                assert_eq!(got[0].value, vx, "x component of ({vx}, {vy})");
                assert_eq!(got[1].value, vy, "y component of ({vx}, {vy})");
            }
        }
    }
}
