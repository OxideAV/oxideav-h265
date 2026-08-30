//! Low-delay GOP encoder — §8.5 inter prediction on the encode side.
//!
//! The sequence shape is `IDR, P, P, …` (or `IDR, B, B, …` with
//! [`LowDelayPEncoder::with_b_slices`]): the first frame goes through
//! the real CABAC intra encoder ([`crate::encoder::intra`]), every
//! following frame is one P / low-delay-B slice (TRAIL_R) whose
//! active reference(s) — `RefPicList0[0]`, and `RefPicList1[0]` on B
//! slices — resolve to the previous frame's reconstruction, signalled
//! with an inline §7.4.8 short-term RPS (`num_negative_pics == 1`,
//! `delta_poc == −1`). Low-delay B keeps decode order == display
//! order while enabling the bi-predictive §8.5.3.2.2 merge candidates
//! (zero-merge and combined candidates are bi on B slices) and the
//! B-side syntax (`inter_pred_idc`, `mvd_l1_zero_flag`, initType 2
//! contexts).
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
use crate::encoder::hrd::{
    buffering_period_payload, filler_data_nal_framed, pic_timing_payload, sei_prefix_nal, HrdClock,
    HrdSignalCfg, SEI_BUFFERING_PERIOD, SEI_PIC_TIMING,
};
use crate::encoder::intra::{
    chroma_qp_420, code_tb, encode_cu_qp_delta, encode_idr_intra_au_full, gather_refs, pred_params,
    search_best_mode, ssd, zscan_avail, IntraEncodeError, SpsCfg,
};
use crate::encoder::loopfilter::{
    encode_sao_ctb, filter_frame, CtbShape, FilterInput, LoopFilterCfg,
};
use crate::encoder::nal::{annexb, nal_unit};
use crate::encoder::rate::{FrameClass, RateControlCfg, RateController};
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
/// The z-order offsets of the four depth-1 transform units within a
/// CTB (§6.5.2 z-scan of the four halves).
const Z_OFFSETS: [(usize, usize); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];

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
#[derive(Debug, Clone, Default)]
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
    /// Bi-predicted CUs (both `predFlagLX` set on some PU — B slices
    /// only; an overlay of the mode classes above).
    pub bi: usize,
    /// Rectangular-partition CUs (`PART_2NxN` / `PART_Nx2N` — an
    /// overlay counted in `merge` / `amvp` too).
    pub rect: usize,
    /// Asymmetric-partition CUs (`PART_2NxnU` / `PART_2NxnD` /
    /// `PART_nLx2N` / `PART_nRx2N` — an overlay counted in `merge` /
    /// `amvp` too; AMP streams only).
    pub amp: usize,
    /// CUs referencing the second reference picture (POC − 2) on some
    /// PU / list (an overlay of the mode classes above).
    pub ref1: usize,
}

/// A resolved-and-coded CU candidate.
struct CuCandidate {
    /// How the CU is signalled.
    kind: CuKind,
    /// The §8.5.3.2.1-resolved per-PU motion in §7.3.8.6 order (what
    /// the decoder will derive); empty for an intra CU.
    motions: Vec<PuMotion>,
    /// The coded transform tree; `None` ⇔ no tree (skip, or
    /// `rqt_root_cbf == 0`).
    levels: Option<CuLevels>,
    /// The CTB reconstruction (y 16x16, cb 8x8, cr 8x8).
    recon: [Vec<u8>; 3],
    /// SSD + λ·rate cost.
    cost: u64,
}

/// The quantized levels of a CU candidate's transform tree.
enum CuLevels {
    /// One depth-0 TU: luma 16x16, cb 8x8, cr 8x8.
    Depth0([Vec<i32>; 3]),
    /// The §7.4.9.8 `interSplitFlag` forced depth-1 tree (rectangular
    /// inter partitions at `max_transform_hierarchy_depth_inter == 0`):
    /// four z-order 8x8 luma and 4x4 chroma TBs per plane (boxed to
    /// keep the variant sizes comparable).
    Depth1(Box<Depth1Levels>),
}

/// The four z-order leaves of a forced depth-1 transform tree.
struct Depth1Levels {
    luma: [Vec<i32>; 4],
    cb: [Vec<i32>; 4],
    cr: [Vec<i32>; 4],
}

/// One list's non-merge (AMVP) signalling group: `ref_idx_lX` (when
/// that list has more than one active reference), the §7.3.8.9
/// `mvd_coding( )` pair, and `mvp_lX_flag`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AmvpGroup {
    pub(crate) ref_idx: u8,
    pub(crate) mvd: Mv,
    pub(crate) mvp_flag: u8,
}

/// One prediction unit's §7.3.8.6 signalling (merge or AMVP).
#[derive(Clone, Copy, Debug)]
pub(crate) enum PuSyntax {
    /// `merge_flag == 1`, `merge_idx`.
    Merge { merge_idx: usize },
    /// `merge_flag == 0`: `inter_pred_idc` (on B slices) selects the
    /// used list(s) — `PRED_L0`, `PRED_L1` or `PRED_BI` — and each
    /// used list carries its [`AmvpGroup`].
    Amvp {
        l0: Option<AmvpGroup>,
        l1: Option<AmvpGroup>,
    },
}

/// The signalling class of a chosen CU.
enum CuKind {
    /// `cu_skip_flag == 1`, `merge_idx`.
    Skip { merge_idx: usize },
    /// `merge_flag == 1`, `merge_idx`, transform tree follows
    /// (`rqt_root_cbf` inferred 1).
    Merge { merge_idx: usize },
    /// AMVP (one 2Nx2N PU): the carried [`PuSyntax::Amvp`] group;
    /// `rqt_root_cbf` signalled.
    Amvp { pu: PuSyntax },
    /// Two-PU inter partition (`PART_2NxN` / `PART_Nx2N`, or one of
    /// the four AMP shapes on an AMP stream): two prediction units,
    /// the forced depth-1 RQT, `rqt_root_cbf` signalled.
    TwoPu { part: PartMode, pus: [PuSyntax; 2] },
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
    /// The `SliceQpY` this frame was coded at (the constructor QP, or
    /// the rate controller's per-frame election under
    /// [`LowDelayPEncoder::with_rate_control`]).
    pub qp: i32,
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
    /// Code the non-IDR frames as (low-delay) B slices: both
    /// reference lists resolve to the previous picture, enabling
    /// bi-predictive merge candidates and the B-slice syntax.
    b_slices: bool,
    /// The §8.7 in-loop filters this stream signals and applies on
    /// its reconstruction path.
    filters: LoopFilterCfg,
    /// AMP stream configuration: `MinCbSizeY == 8` with every CTB an
    /// unsplit 16x16 CU, `amp_enabled_flag == 1`, and the asymmetric
    /// PART_2NxnU/2NxnD/nLx2N/nRx2N shapes competing in the inter CU
    /// ladder.
    amp: bool,
    /// POC of the NEXT frame within the current GOP (0 ⇒ IDR next).
    poc: i32,
    /// The most recent reconstructions, newest first (up to two — the
    /// active references POC − 1 and POC − 2).
    prevs: Vec<FrameRecon>,
    /// ABR rate controller: when set, it elects each frame's
    /// `SliceQpY` and the constructor `qp` is unused (pass it as
    /// [`RateControlCfg::initial_qp`] to seed the starting point).
    rc: Option<RateController>,
    /// Spatial adaptive-quantization strength (0 = constant slice
    /// QP; 1..=3 = per-CTB `cu_qp_delta` on every frame, IDRs
    /// included).
    aq: u8,
    /// §E.2.1 VUI timing declaration `(num_units_in_tick,
    /// time_scale)` for the stream's SPS (`None` = no VUI).
    timing: Option<(u32, u32)>,
    /// The rate-control configuration behind `rc` (kept for HRD
    /// signalling, which derives its schedule from it).
    rc_cfg: Option<RateControlCfg>,
    /// HRD signalling requested ([`Self::with_hrd`]).
    hrd_on: bool,
    /// CBR delivery requested ([`Self::with_cbr`]; requires
    /// `hrd_on`).
    cbr_on: bool,
    /// The signalled §E.2.2 schedule + Annex C clock, built at the
    /// first frame when `hrd_on`.
    hrd: Option<(HrdSignalCfg, HrdClock)>,
    /// Quadtree-coder geometry ([`Self::with_tree`]).
    tree: Option<crate::encoder::ctu::TreeCfg>,
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
            b_slices: false,
            filters: LoopFilterCfg::off(),
            amp: false,
            poc: 0,
            prevs: Vec::new(),
            rc: None,
            aq: 0,
            timing: None,
            rc_cfg: None,
            hrd_on: false,
            cbr_on: false,
            hrd: None,
            tree: None,
        })
    }

    /// Route every picture through the recursive coding-quadtree
    /// coder ([`crate::encoder::ctu`]) at the given [`TreeCfg`]
    /// geometry: CTB 16 / 32 / 64 with RD-elected `split_cu_flag`
    /// quadtrees down to `MinCbSizeY == 8`, deeper residual
    /// quadtrees (`max_transform_hierarchy_depth_* > 0`), 4x4
    /// DST-VII intra luma TUs and intra `PART_NxN`. Composes with
    /// [`Self::with_amp`] (which then only switches the asymmetric
    /// shapes into the CU ladder).
    #[must_use]
    pub fn with_tree(mut self, tree: crate::encoder::ctu::TreeCfg) -> Self {
        self.tree = Some(tree);
        self
    }

    /// Declare the stream's frame rate (`fps_num / fps_den` frames
    /// per second) in the SPS: the §E.2.1 VUI `vui_timing_info` block
    /// (`vui_num_units_in_tick = fps_den`,
    /// `vui_time_scale = fps_num`). Purely declarative — coding is
    /// unchanged.
    #[must_use]
    pub fn with_frame_rate(mut self, fps_num: u32, fps_den: u32) -> Self {
        self.timing = (fps_num > 0 && fps_den > 0).then_some((fps_den, fps_num));
        self
    }

    /// Enable spatial adaptive quantization at `strength` (clamped to
    /// 0..=3): every frame's CTBs move off the frame QP by `strength`
    /// QP per octave of luma-activity ratio, signalled through
    /// per-CTB `cu_qp_delta` (the PPS gains
    /// `cu_qp_delta_enabled_flag`). Composes with the constant-QP,
    /// GOP, B-slice, filter and rate-control configurations.
    #[must_use]
    pub fn with_aq(mut self, strength: u8) -> Self {
        self.aq = strength.min(3);
        self
    }

    /// Code non-IDR frames as low-delay B slices (both reference
    /// lists resolving to the previous picture) instead of P slices.
    #[must_use]
    pub fn with_b_slices(mut self, on: bool) -> Self {
        self.b_slices = on;
        self
    }

    /// Switch the stream to the AMP configuration: the SPS signals
    /// `MinCbSizeY == 8` + `amp_enabled_flag == 1` (every CTB still
    /// one unsplit 16x16 CU, now with an explicit `split_cu_flag`),
    /// and the inter CU ladder elects the asymmetric
    /// `PART_2NxnU / PART_2NxnD / PART_nLx2N / PART_nRx2N` motion
    /// partitions alongside the symmetric shapes.
    #[must_use]
    pub fn with_amp(mut self, on: bool) -> Self {
        self.amp = on;
        self
    }

    /// Enable the §8.7 in-loop filters (deblocking / SAO) on the
    /// encoder's reconstruction path: every frame's reference — and
    /// the [`EncodedPFrame::recon`] a conforming decoder reproduces —
    /// becomes the FILTERED picture, with the filter elections
    /// signalled in the parameter sets and slice headers.
    #[must_use]
    pub fn with_loop_filters(mut self, filters: LoopFilterCfg) -> Self {
        self.filters = filters;
        self
    }

    /// Switch to average-bitrate coding: a [`RateController`] elects
    /// each frame's `SliceQpY` against `cfg` (target bitrate + frame
    /// rate), replacing the constructor's constant QP. Every stream
    /// stays fully conforming — rate control only moves the per-slice
    /// `slice_qp_delta`.
    #[must_use]
    pub fn with_rate_control(mut self, cfg: &RateControlCfg) -> Self {
        self.rc = Some(RateController::new(cfg, self.width, self.height));
        self.rc_cfg = Some(*cfg);
        self
    }

    /// Declare and enforce HRD conformance: the SPS VUI gains a
    /// §E.2.2 `hrd_parameters( )` schedule (NAL HRD, one CPB at the
    /// rate-control target and VBV size), every IDR access unit
    /// carries a §D.2.2 buffering-period SEI, every access unit a
    /// §D.2.3 pic-timing SEI, and the coded sizes are additionally
    /// capped by an exact Annex C CPB replay (re-encoding at a
    /// higher QP like the VBV path) so the emitted stream satisfies
    /// the §C.4 conformance conditions. Requires
    /// [`Self::with_rate_control`] with [`RateControlCfg::with_vbv`]
    /// AND [`Self::with_frame_rate`]; [`Self::encode_frame`] errors
    /// otherwise.
    #[must_use]
    pub fn with_hrd(mut self, on: bool) -> Self {
        self.hrd_on = on;
        self
    }

    /// Switch the HRD schedule to constant-bit-rate delivery
    /// (`cbr_flag[0] == 1`): the modelled channel feeds the CPB back
    /// to back at exactly the signalled rate (eq. C-3), and the
    /// encoder pads underruns with §7.3.4 filler-data NAL units so
    /// the CPB cannot overflow (§C.4 condition 2). Requires
    /// [`Self::with_hrd`] and a VBV of at least two frame intervals
    /// of the bitrate; [`Self::encode_frame`] errors otherwise.
    #[must_use]
    pub fn with_cbr(mut self, on: bool) -> Self {
        self.cbr_on = on;
        self
    }

    /// Build the HRD schedule + clock on the first frame (validating
    /// the [`Self::with_hrd`] prerequisites).
    fn ensure_hrd(&mut self) -> Result<(), IntraEncodeError> {
        if !self.hrd_on {
            return if self.cbr_on {
                Err(IntraEncodeError::HrdConfig)
            } else {
                Ok(())
            };
        }
        if self.hrd.is_some() {
            return Ok(());
        }
        let (Some(cfg), Some((num_units, time_scale))) = (&self.rc_cfg, self.timing) else {
            return Err(IntraEncodeError::HrdConfig);
        };
        let Some(vbv) = cfg.vbv_buffer_bits else {
            return Err(IntraEncodeError::HrdConfig);
        };
        let signal = HrdSignalCfg::for_rate(cfg.bits_per_second, vbv).with_cbr(self.cbr_on);
        if self.cbr_on {
            // The filler quantum must fit between the CBR overflow
            // floor and the underflow cap (see `cbr_filler_bits`).
            let tick_bits =
                signal.bit_rate.saturating_mul(u64::from(num_units)) / u64::from(time_scale.max(1));
            if signal.cpb_size < tick_bits.saturating_mul(2) {
                return Err(IntraEncodeError::CbrCpbTooSmall);
            }
        }
        self.hrd = Some((signal, HrdClock::new(signal, time_scale, num_units)));
        Ok(())
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
        let idr_now = self.prevs.is_empty() || self.poc == 0;
        let mut frame_qp = match &mut self.rc {
            Some(rc) => rc.pick_qp(if idr_now {
                FrameClass::Intra
            } else {
                FrameClass::Inter
            }),
            None => self.qp,
        };
        // The AU's HRD SEI prefix (buffering period on IDRs, pic
        // timing on every AU), built BEFORE coding: its bits count
        // toward the Annex C / VBV budgets.
        self.ensure_hrd()?;
        let sei = self.hrd.as_mut().map(|(_, clock)| {
            let mut payloads = Vec::with_capacity(2);
            if idr_now {
                let (delay, offset) = clock.begin_buffering_period();
                payloads.push((
                    SEI_BUFFERING_PERIOD,
                    buffering_period_payload(delay, offset),
                ));
            }
            payloads.push((
                SEI_PIC_TIMING,
                // Decode order == display order: no output delay.
                pic_timing_payload(clock.au_cpb_removal_delay_minus1(), 0),
            ));
            let mut framed = vec![0, 0, 0, 1];
            framed.extend(sei_prefix_nal(&payloads));
            framed
        });
        let sei_bits = sei.as_ref().map_or(0, |s| s.len() as u64 * 8);
        // Code the frame at `frame_qp`; under a VBV / HRD constraint,
        // re-encode at a higher QP until the whole access unit (SEI
        // included) fits the modelled decoder buffer AND the Annex C
        // arrival window (or the QP ceiling is reached).
        let cap = match (
            self.rc.as_ref().and_then(RateController::vbv_frame_cap),
            self.hrd.as_ref().map(|(_, clock)| clock.frame_cap()),
        ) {
            (Some(v), Some(h)) => Some(v.min(h)),
            (v, h) => v.or(h),
        }
        // CBR: reserve the filler quantum (a minimum filler NAL is
        // 7 framed bytes) under the cap so the overflow padding can
        // never push the AU past its removal time.
        .map(|c| if self.cbr_on { c.saturating_sub(56) } else { c });
        let mut coded = self.code_frame_at(frame, idr_now, frame_qp)?;
        if let Some(cap) = cap {
            let qp_ceiling = self.rc.as_ref().map_or(51, RateController::max_qp);
            while coded.0.len() as u64 * 8 + sei_bits > cap && frame_qp < qp_ceiling {
                frame_qp = (frame_qp + 3).min(qp_ceiling);
                coded = self.code_frame_at(frame, idr_now, frame_qp)?;
            }
        }
        let (mut au, keyframe, recon, stats) = coded;
        if let Some(sei) = &sei {
            crate::encoder::hrd::splice_sei_before_vcl(&mut au, sei);
        }
        if let Some((_, clock)) = &self.hrd {
            // CBR underrun padding: filler data after the VCL NAL.
            let pad_bits = clock.cbr_filler_bits(au.len() as u64 * 8);
            if pad_bits > 0 {
                au.extend(filler_data_nal_framed((pad_bits as usize).div_ceil(8)));
            }
        }
        // Commit the reference / POC state and the model feedback.
        if idr_now {
            self.prevs = vec![recon.clone()];
            self.poc = 1;
        } else {
            self.prevs.insert(0, recon.clone());
            self.prevs.truncate(2);
            self.poc += 1;
        }
        if let Some(rc) = &mut self.rc {
            rc.update(
                if idr_now {
                    FrameClass::Intra
                } else {
                    FrameClass::Inter
                },
                frame_qp,
                au.len() as u64 * 8,
            );
        }
        if let Some((_, clock)) = &mut self.hrd {
            clock.push_au(au.len() as u64 * 8);
        }
        // GOP wrap: schedule the next IDR.
        if self.gop > 0 && self.poc >= self.gop as i32 {
            self.poc = 0;
        }
        Ok(EncodedPFrame {
            au,
            keyframe,
            recon,
            stats,
            qp: frame_qp,
        })
    }

    /// Code the next frame at `qp` WITHOUT committing any encoder
    /// state (the VBV re-encode loop may call this repeatedly).
    #[allow(clippy::type_complexity)]
    fn code_frame_at(
        &self,
        frame: &YuvFrame<'_>,
        idr_now: bool,
        qp: i32,
    ) -> Result<(Vec<u8>, bool, FrameRecon, FrameStats), IntraEncodeError> {
        if idr_now {
            // sps_max_dec_pic_buffering_minus1 = 2: the current picture
            // plus the two short-term references the P/B slices keep.
            let idr = encode_idr_intra_au_full(
                frame.y,
                frame.cb,
                frame.cr,
                self.width,
                self.height,
                qp,
                &SpsCfg {
                    max_dec_pic_buffering_minus1: 2,
                    max_num_reorder_pics: 0,
                    min_cb_log2: if self.amp || self.tree.is_some() {
                        3
                    } else {
                        4
                    },
                    amp: self.amp,
                    cu_qp_delta: self.aq > 0,
                    timing: self.timing,
                    hrd: self.hrd.as_ref().map(|(signal, _)| *signal),
                    tree: self.tree,
                },
                &self.filters,
                self.aq,
            )?;
            Ok((
                idr.au,
                true,
                FrameRecon {
                    y: idr.recon_y,
                    cb: idr.recon_cb,
                    cr: idr.recon_cr,
                },
                FrameStats {
                    intra: (self.width / CTB) * (self.height / CTB),
                    ..FrameStats::default()
                },
            ))
        } else {
            // The active references: RefPicListX[i] = the
            // reconstruction of POC − 1 − i (newest first, up to two),
            // and a low-delay B slice holds the same set on both lists.
            let l0: Vec<(i32, &FrameRecon)> = self
                .prevs
                .iter()
                .take(2)
                .enumerate()
                .map(|(i, rec)| (self.poc - 1 - i as i32, rec))
                .collect();
            let spec = SliceSpec {
                poc: self.poc,
                qp,
                b_slice: self.b_slices,
                rps_neg: (1..=l0.len() as u32).map(|d| (d, true)).collect(),
                rps_pos: Vec::new(),
                l1: if self.b_slices {
                    l0.clone()
                } else {
                    Vec::new()
                },
                l0,
                lf: &self.filters,
                big_cu: self.amp,
                aq: self.aq,
                tree: self.tree,
            };
            let (rbsp, recon, stats) = encode_inter_slice(frame, &spec, self.width, self.height);
            Ok((annexb(&[nal_unit(1, 0, 0, &rbsp)]), false, recon, stats))
        }
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

/// The slice header's in-loop-filter fields (§7.3.6.1): the enabled
/// filter set plus this slice's elections.
pub(crate) struct SliceLfSignalling<'a> {
    /// The stream's filter configuration (drives field presence: the
    /// SAO gates exist iff the SPS enables SAO, the deblocking
    /// override group iff the PPS enables the override).
    pub(crate) cfg: &'a LoopFilterCfg,
    /// This slice's deblocking election
    /// (`slice_deblocking_filter_disabled_flag == !deblock_on`).
    pub(crate) deblock_on: bool,
    /// Elected `slice_beta_offset_div2`.
    pub(crate) beta_offset_div2: i32,
    /// Elected `slice_tc_offset_div2`.
    pub(crate) tc_offset_div2: i32,
    /// Elected `slice_sao_luma_flag`.
    pub(crate) sao_luma: bool,
    /// Elected `slice_sao_chroma_flag`.
    pub(crate) sao_chroma: bool,
}

/// Everything one P / B slice's coding depends on beyond the frame
/// samples: POC, QP, slice type, the inline §7.4.8 short-term RPS
/// content, and the two active reference lists (POC + reconstruction
/// each, in `RefPicListX` order). The low-delay GOP passes past
/// pictures on both lists; the hierarchical-B GOP passes past
/// pictures on L0 and future pictures on L1.
pub(crate) struct SliceSpec<'a> {
    /// `PicOrderCntVal` (== `slice_pic_order_cnt_lsb`, 8 bits).
    pub poc: i32,
    /// `SliceQpY`.
    pub qp: i32,
    /// B slice (`slice_type == 0`)?
    pub b_slice: bool,
    /// Negative RPS entries closest-first: `(PocDelta, used_by_curr)`
    /// with `PocDelta = poc − entry POC > 0`, strictly increasing.
    pub rps_neg: Vec<(u32, bool)>,
    /// Positive RPS entries closest-first: `(entry POC − poc, used)`.
    pub rps_pos: Vec<(u32, bool)>,
    /// `RefPicList0`: active references, `(POC, reconstruction)`.
    pub l0: Vec<(i32, &'a FrameRecon)>,
    /// `RefPicList1` (B slices; empty on P slices).
    pub l1: Vec<(i32, &'a FrameRecon)>,
    /// The stream's §8.7 in-loop filter configuration.
    pub lf: &'a LoopFilterCfg,
    /// Stream geometry: `log2CbSize > MinCbLog2SizeY` (MinCb 8, one
    /// explicit `split_cu_flag == 0` per CTB, the Table 9-45 big-CU
    /// `part_mode` column, no intra `part_mode`). Implied by the AMP
    /// stream configuration.
    pub big_cu: bool,
    /// Spatial adaptive-quantization strength (0 = constant slice QP;
    /// 1..=3 = per-CTB `cu_qp_delta` signalling — the stream's PPS
    /// must carry `cu_qp_delta_enabled_flag == 1`).
    pub aq: u8,
    /// Quadtree-coder geometry: `Some` routes the slice through
    /// [`crate::encoder::ctu`] (recursive coding quadtrees; `big_cu`
    /// then selects `amp_enabled_flag` only). `None` keeps the
    /// historical fixed-geometry coder byte-stable.
    pub tree: Option<crate::encoder::ctu::TreeCfg>,
}

/// §7.3.6.1 — the P / B slice-segment header: an inline §7.4.8
/// short-term RPS (negative + positive pictures with per-entry used
/// flags), the `num_ref_idx` override when a list keeps more than one
/// active reference, `MaxNumMergeCand == 5`, `slice_qp_delta` against
/// `init_qp == 26`, then the in-loop-filter block (`slice_sao_*`
/// gates when the SPS enables SAO, the deblocking override group when
/// the PPS enables the override, the across-slices flag when any
/// filter is active).
pub(crate) fn write_inter_slice_header(
    w: &mut BitWriter,
    spec: &SliceSpec<'_>,
    lf: &SliceLfSignalling<'_>,
) {
    w.put_bit(1); // first_slice_segment_in_pic_flag
    w.ue(0); // slice_pic_parameter_set_id
    w.ue(u32::from(!spec.b_slice)); // slice_type (0 = B, 1 = P)
    w.put_bits((spec.poc & 0xFF) as u32, 8); // slice_pic_order_cnt_lsb
    w.put_bit(0); // short_term_ref_pic_set_sps_flag
                  // st_ref_pic_set( 0 ) — idx 0 has no
                  // inter_ref_pic_set_prediction_flag (§7.3.7).
    w.ue(spec.rps_neg.len() as u32); // num_negative_pics
    w.ue(spec.rps_pos.len() as u32); // num_positive_pics
    let mut prev = 0u32;
    for &(delta, used) in &spec.rps_neg {
        // §7.4.8: DeltaPocS0[i] = DeltaPocS0[i−1] − (minus1 + 1).
        w.ue(delta - prev - 1); // delta_poc_s0_minus1[i]
        w.put_bit(u8::from(used)); // used_by_curr_pic_s0_flag[i]
        prev = delta;
    }
    let mut prev = 0u32;
    for &(delta, used) in &spec.rps_pos {
        // §7.4.8: DeltaPocS1[i] = DeltaPocS1[i−1] + (minus1 + 1).
        w.ue(delta - prev - 1); // delta_poc_s1_minus1[i]
        w.put_bit(u8::from(used)); // used_by_curr_pic_s1_flag[i]
        prev = delta;
    }
    // sps_temporal_mvp_enabled_flag == 0: nothing.
    if lf.cfg.sao() {
        // SPS SAO enabled: the per-slice component gates are present.
        w.put_bit(u8::from(lf.sao_luma)); // slice_sao_luma_flag
        w.put_bit(u8::from(lf.sao_chroma)); // slice_sao_chroma_flag
    }
    let n_l0 = spec.l0.len() as u32;
    let n_l1 = spec.l1.len() as u32;
    if n_l0 != 1 || (spec.b_slice && n_l1 != 1) {
        // PPS defaults are one active reference per list.
        w.put_bit(1); // num_ref_idx_active_override_flag
        w.ue(n_l0 - 1); // num_ref_idx_l0_active_minus1
        if spec.b_slice {
            w.ue(n_l1 - 1); // num_ref_idx_l1_active_minus1
        }
    } else {
        w.put_bit(0); // num_ref_idx_active_override_flag (default: 1)
    }
    // lists_modification_present_flag == 0.
    if spec.b_slice {
        w.put_bit(0); // mvd_l1_zero_flag
    }
    // cabac_init_present_flag == 0; TMVP off; WP off.
    w.ue((5 - MAX_MERGE) as u32); // five_minus_max_num_merge_cand
    w.se(spec.qp - 26); // slice_qp_delta
    if lf.cfg.deblocking {
        // §7.3.6.1 deblocking override group (the PPS signals
        // override-enabled + deblocking disabled; an electing slice
        // overrides to enabled with zero β/tC offsets).
        w.put_bit(u8::from(lf.deblock_on)); // deblocking_filter_override_flag
        if lf.deblock_on {
            w.put_bit(0); // slice_deblocking_filter_disabled_flag
            w.se(lf.beta_offset_div2); // slice_beta_offset_div2
            w.se(lf.tc_offset_div2); // slice_tc_offset_div2
        }
    }
    if lf.sao_luma || lf.sao_chroma || lf.deblock_on {
        // §7.3.6.1: present iff pps_loop_filter_across_slices (1) and
        // some in-loop filter is active for this slice.
        w.put_bit(1); // slice_loop_filter_across_slices_enabled_flag
    }
    // No tiles / WPP: no entry points.
    w.rbsp_trailing_bits(); // byte_alignment()
}

/// Extract an `n`x`n` block of `plane` at `(x0, y0)` as `i32`s.
pub(crate) fn extract(plane: &[u8], pw: usize, x0: usize, y0: usize, n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            out.push(i32::from(plane[(y0 + j) * pw + x0 + i]));
        }
    }
    out
}

/// Store an `n`x`n` block back into `plane` at `(x0, y0)`.
pub(crate) fn store(plane: &mut [u8], pw: usize, x0: usize, y0: usize, n: usize, s: &[u8]) {
    for j in 0..n {
        plane[(y0 + j) * pw + x0..(y0 + j) * pw + x0 + n].copy_from_slice(&s[j * n..(j + 1) * n]);
    }
}

/// The [`crate::binarization::PartMode`] twin of a [`PartMode`] (the
/// loop-filter descriptors speak the binarization enum).
pub(crate) fn bin_part_mode(p: PartMode) -> crate::binarization::PartMode {
    use crate::binarization::PartMode as B;
    match p {
        PartMode::Part2Nx2N => B::Part2Nx2N,
        PartMode::Part2NxN => B::Part2NxN,
        PartMode::PartNx2N => B::PartNx2N,
        PartMode::PartNxN => B::PartNxN,
        PartMode::Part2NxnU => B::Part2NxnU,
        PartMode::Part2NxnD => B::Part2NxnD,
        PartMode::PartNLx2N => B::PartNLx2N,
        PartMode::PartNRx2N => B::PartNRx2N,
    }
}

/// `true` for the horizontal-split two-PU shapes (`PART_2NxN` /
/// `PART_2NxnU` / `PART_2NxnD`) — Table 9-45 bin 1.
pub(crate) fn part_is_horizontal(p: PartMode) -> bool {
    matches!(
        p,
        PartMode::Part2NxN | PartMode::Part2NxnU | PartMode::Part2NxnD
    )
}

/// `true` for the four asymmetric shapes.
pub(crate) fn part_is_amp(p: PartMode) -> bool {
    matches!(
        p,
        PartMode::Part2NxnU | PartMode::Part2NxnD | PartMode::PartNLx2N | PartMode::PartNRx2N
    )
}

/// The merge / skip PU syntax for candidate `merge_idx` (the §7.3.8.6
/// fields the §8.5.3.2.2 derivation reads).
pub(crate) fn merge_pu(merge_idx: usize) -> PredictionUnit {
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

/// An AMVP PU carrying the given per-list signalling groups (the
/// §7.3.8.6 fields the §8.5.3.2 derivation reads). At least one list
/// must be used; `inter_pred_idc` follows from which are.
fn amvp_pu(l0: Option<AmvpGroup>, l1: Option<AmvpGroup>) -> PredictionUnit {
    let comp = |v: i32| MvdComponent {
        greater0_flag: u8::from(v != 0),
        greater1_flag: None,
        minus2: None,
        sign_flag: None,
        value: v,
    };
    let idc = match (l0.is_some(), l1.is_some()) {
        (true, true) => InterPredIdc::PredBi,
        (false, true) => InterPredIdc::PredL1,
        // An AMVP PU uses at least one list; default to L0.
        _ => InterPredIdc::PredL0,
    };
    PredictionUnit {
        merge_flag: false,
        merge_idx: None,
        inter_pred_idc: Some(idc),
        ref_idx_l0: l0.map(|g| g.ref_idx),
        mvd_l0: l0.map(|g| [comp(g.mvd[0]), comp(g.mvd[1])]),
        mvp_l0_flag: l0.map(|g| g.mvp_flag),
        ref_idx_l1: l1.map(|g| g.ref_idx),
        mvd_l1: l1.map(|g| [comp(g.mvd[0]), comp(g.mvd[1])]),
        mvp_l1_flag: l1.map(|g| g.mvp_flag),
    }
}

/// A single-list [`amvp_pu`] probe (zero or explicit mvd) for the
/// §8.5.3.2.6 predictor derivation and the ME resolution step.
fn amvp_pu_list(list: usize, ref_idx: u8, mvd: Mv, mvp_flag: u8) -> PredictionUnit {
    let g = Some(AmvpGroup {
        ref_idx,
        mvd,
        mvp_flag,
    });
    if list == 0 {
        amvp_pu(g, None)
    } else {
        amvp_pu(None, g)
    }
}

/// Select the plane set a list prediction reads: the used list's
/// `refs[refIdxLX]`, or (for an unused list, whose planes are never
/// read) any legal plane set from either list.
fn pick_ref<'r>(
    refs: &'r [RefPlanes],
    other: &'r [RefPlanes],
    pred_flag: bool,
    ref_idx: i32,
) -> &'r RefPlanes {
    if pred_flag {
        &refs[ref_idx.max(0) as usize]
    } else {
        refs.first().unwrap_or_else(|| &other[0])
    }
}

/// §8.5.3.3 — the final clipped prediction for a `w`x`h` PU at
/// `(x0, y0)` with the resolved motion (uni-L0, uni-L1 or bi). Each
/// used list reads `refs_lX[ refIdxLX ]`; the low-delay GOP passes the
/// same past reconstructions on both lists, the hierarchical-B GOP
/// passes past pictures on L0 and future pictures on L1. `chroma`
/// selects whether the Cb / Cr planes are predicted too (the SAD
/// search runs luma-only).
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_block(
    refs_l0: &[RefPlanes],
    refs_l1: &[RefPlanes],
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    motion: &PuMotion,
    chroma: bool,
) -> InterPrediction {
    fn list(refp: &RefPlanes, pred_flag: bool, mv: Mv, chroma: bool) -> ListPrediction<'_> {
        let luma = RefPlane::new(&refp.y, refp.width, refp.height).expect("legal ref plane");
        let (cb, cr) = if chroma {
            (
                Some(
                    RefPlane::new(&refp.cb, refp.width / 2, refp.height / 2).expect("legal plane"),
                ),
                Some(
                    RefPlane::new(&refp.cr, refp.width / 2, refp.height / 2).expect("legal plane"),
                ),
            )
        } else {
            (None, None)
        };
        ListPrediction {
            pred_flag,
            luma,
            cb,
            cr,
            mv_l: mv,
            // §8.5.3.2.10: 4:2:0 chroma MV = luma MV (eighth-pel units).
            mv_c: derive_chroma_mv(mv, 2, 2),
        }
    }
    let l0 = list(
        pick_ref(refs_l0, refs_l1, motion.pred_flag_l0, motion.ref_idx_l0),
        motion.pred_flag_l0,
        motion.mv_l0,
        chroma,
    );
    let l1 = list(
        pick_ref(refs_l1, refs_l0, motion.pred_flag_l1, motion.ref_idx_l1),
        motion.pred_flag_l1,
        motion.mv_l1,
        chroma,
    );
    let geom = InterPredGeometry {
        x_pb: x0 as i32,
        y_pb: y0 as i32,
        n_pb_w: w,
        n_pb_h: h,
        chroma_array_type: if chroma { 1 } else { 0 },
        bit_depth_luma: BIT_DEPTH,
        bit_depth_chroma: BIT_DEPTH,
    };
    predict_inter_pu(&l0, &l1, &geom).expect("legal prediction geometry")
}

/// Copy a `w`x`h` block into a `bw`-wide buffer at `(bx, by)`.
pub(crate) fn blit(
    dst: &mut [i32],
    bw: usize,
    bx: usize,
    by: usize,
    src: &[i32],
    w: usize,
    h: usize,
) {
    for j in 0..h {
        dst[(by + j) * bw + bx..(by + j) * bw + bx + w].copy_from_slice(&src[j * w..(j + 1) * w]);
    }
}

/// Extract an `n`x`n` sub-block of a `bw`-wide buffer at `(bx, by)`.
pub(crate) fn sub_block(buf: &[i32], bw: usize, bx: usize, by: usize, n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n * n);
    for j in 0..n {
        out.extend_from_slice(&buf[(by + j) * bw + bx..(by + j) * bw + bx + n]);
    }
    out
}

/// A uni-L0 [`PuMotion`] for motion-search probes (list-0 reference
/// index 0 — the probe caller passes the searched reference's planes
/// as the single-entry `refs` slice).
fn uni_l0(mv: Mv) -> PuMotion {
    PuMotion {
        pred_flag_l0: true,
        pred_flag_l1: false,
        ref_idx_l0: 0,
        ref_idx_l1: -1,
        mv_l0: mv,
        mv_l1: [0, 0],
    }
}

/// The previous frame's reconstruction as `i32` planes (the
/// §8.5.3.3.3 interpolation input).
pub(crate) struct RefPlanes {
    pub(crate) y: Vec<i32>,
    pub(crate) cb: Vec<i32>,
    pub(crate) cr: Vec<i32>,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

/// Luma SAD between a prediction and the source block.
pub(crate) fn sad(pred: &[i32], src: &[i32]) -> u64 {
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
pub(crate) fn mvd_bits(mvd: Mv) -> u64 {
    mvd_component_bits(mvd[0]) + mvd_component_bits(mvd[1])
}

/// Crude bit-cost proxy for one TB's quantized levels (mirrors the
/// intra encoder's heuristic).
pub(crate) fn levels_rate(levels: &[i32]) -> u64 {
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
#[allow(clippy::too_many_arguments)]
fn integer_me(
    refp: &RefPlanes,
    src_y: &[i32],
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    seeds: &[[i32; 2]],
    lambda_me: u64,
    mvp_q: &[Mv],
) -> [i32; 2] {
    // Full-pel prediction == direct (clamped) reference samples.
    let sad_at = |mx: i32, my: i32| -> u64 {
        let mut acc = 0u64;
        for j in 0..h {
            for i in 0..w {
                let rx = (x0 as i32 + i as i32 + mx).clamp(0, refp.width as i32 - 1);
                let ry = (y0 as i32 + j as i32 + my).clamp(0, refp.height as i32 - 1);
                let r = refp.y[ry as usize * refp.width + rx as usize];
                acc += u64::from(r.abs_diff(src_y[j * w + i]));
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
#[allow(clippy::too_many_arguments)]
fn fractional_me(
    refp: &RefPlanes,
    src_y: &[i32],
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    start: Mv,
    lambda_me: u64,
    mvp_q: &[Mv],
) -> Mv {
    let cost_at = |mv: Mv| -> u64 {
        let one = std::slice::from_ref(refp);
        let pred = predict_block(one, one, x0, y0, w, h, &uni_l0(mv), false);
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

/// Inputs shared by every per-PU decision of one slice.
pub(crate) struct PuChooseCtx<'a> {
    pub(crate) refs_l0: &'a [RefPlanes],
    pub(crate) refs_l1: &'a [RefPlanes],
    pub(crate) mv_ctx: &'a PuMvContext<'a>,
    pub(crate) lambda_me: u64,
    pub(crate) b_slice: bool,
    /// B slice whose two reference lists differ (hierarchical B): the
    /// AMVP election searches L1 and the bi combination too. A
    /// low-delay B (both lists the same pictures) keeps the L0-only
    /// search — L1 would duplicate it bin for bin at a higher rate.
    pub(crate) two_sided: bool,
}

/// Choose merge-vs-AMVP for one prediction unit against `field` (the
/// motion state the decoder will have when resolving this PU), by
/// luma SAD + λ·signalling-bins. Returns the syntax, the resolved
/// motion, and the motion-rate proxy.
pub(crate) fn choose_pu(
    field: &MotionField,
    geom: &PuGeometry,
    available: &dyn Fn(i32, i32) -> bool,
    src_y: &[i32],
    ctx: &PuChooseCtx<'_>,
) -> (PuSyntax, PuMotion, u64) {
    let (x, y, w, h) = (geom.x_pb, geom.y_pb, geom.n_pb_w, geom.n_pb_h);
    // ---- merge candidates (§8.5.3.2.2) ----
    let mut merge_cands: Vec<(usize, PuMotion)> = Vec::with_capacity(MAX_MERGE);
    for idx in 0..MAX_MERGE {
        let m = resolve_pu_motion(field, geom, &merge_pu(idx), ctx.mv_ctx, available);
        // A later duplicate can never beat the earlier index.
        if !merge_cands.iter().any(|(_, prev)| *prev == m) {
            merge_cands.push((idx, m));
        }
    }
    let (merge_cost, merge_idx, merge_motion) = merge_cands
        .iter()
        .map(|&(idx, m)| {
            let pred = predict_block(ctx.refs_l0, ctx.refs_l1, x, y, w, h, &m, false);
            (
                sad(&pred.luma, src_y) + ctx.lambda_me * (idx as u64 + 2),
                idx,
                m,
            )
        })
        .min_by_key(|&(cost, idx, _)| (cost, idx))
        .expect("merge list is never empty");

    let (amvp_syntax, amvp_motion, best_amvp_rate, best_amvp_cost) =
        amvp_search(field, geom, available, src_y, ctx, &merge_cands);

    if merge_cost <= best_amvp_cost {
        (
            PuSyntax::Merge { merge_idx },
            merge_motion,
            merge_idx as u64 + 2,
        )
    } else {
        (amvp_syntax, amvp_motion, best_amvp_rate)
    }
}

/// §8.5.3.2.6 + motion estimation over every active reference of ONE
/// list: the SAD + λ·bins-best single-list AMVP signalling for one
/// PU. Returns the group, the resolved motion, the rate proxy and the
/// search cost; `None` when the list has no active reference.
#[allow(clippy::too_many_arguments)]
fn amvp_search_list(
    field: &MotionField,
    geom: &PuGeometry,
    available: &dyn Fn(i32, i32) -> bool,
    src_y: &[i32],
    ctx: &PuChooseCtx<'_>,
    merge_cands: &[(usize, PuMotion)],
    list: usize,
) -> Option<(AmvpGroup, PuMotion, u64, u64)> {
    let (x, y, w, h) = (geom.x_pb, geom.y_pb, geom.n_pb_w, geom.n_pb_h);
    let refs = if list == 0 { ctx.refs_l0 } else { ctx.refs_l1 };
    let mv_of = |m: &PuMotion| if list == 0 { m.mv_l0 } else { m.mv_l1 };
    let mut best: Option<(AmvpGroup, PuMotion, u64, u64)> = None;
    for (r, refp) in refs.iter().enumerate() {
        let mvp = [
            mv_of(&resolve_pu_motion(
                field,
                geom,
                &amvp_pu_list(list, r as u8, [0, 0], 0),
                ctx.mv_ctx,
                available,
            )),
            mv_of(&resolve_pu_motion(
                field,
                geom,
                &amvp_pu_list(list, r as u8, [0, 0], 1),
                ctx.mv_ctx,
                available,
            )),
        ];
        let mut seeds: Vec<[i32; 2]> = vec![[0, 0]];
        for p in &mvp {
            seeds.push([p[0] >> 2, p[1] >> 2]);
        }
        for (_, m) in merge_cands {
            let mv = mv_of(m);
            seeds.push([mv[0] >> 2, mv[1] >> 2]);
        }
        seeds.dedup();
        let int_mv = integer_me(refp, src_y, x, y, w, h, &seeds, ctx.lambda_me, &mvp);
        let me_mv = fractional_me(
            refp,
            src_y,
            x,
            y,
            w,
            h,
            [int_mv[0] * 4, int_mv[1] * 4],
            ctx.lambda_me,
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
        let motion = resolve_pu_motion(
            field,
            geom,
            &amvp_pu_list(list, r as u8, mvd, mvp_flag),
            ctx.mv_ctx,
            available,
        );
        // merge_flag + mvp_flag + ref_idx bin (2 refs) + idc (B) + mvd.
        let rate = 2 + mvd_bits(mvd) + u64::from(refs.len() > 1) + u64::from(ctx.b_slice) * 2;
        let pred = predict_block(ctx.refs_l0, ctx.refs_l1, x, y, w, h, &motion, false);
        let cost = sad(&pred.luma, src_y) + ctx.lambda_me * rate;
        let improved = match &best {
            None => true,
            Some((_, _, _, c)) => cost < *c,
        };
        if improved {
            let group = AmvpGroup {
                ref_idx: r as u8,
                mvd,
                mvp_flag,
            };
            best = Some((group, motion, rate, cost));
        }
    }
    best
}

/// The full AMVP election for one PU: the best single-list search on
/// L0 — and, on a two-sided B slice, on L1 plus the bi-predictive
/// combination of the two winners. Returns the syntax, the resolved
/// motion, the rate proxy and the search cost.
pub(crate) fn amvp_search(
    field: &MotionField,
    geom: &PuGeometry,
    available: &dyn Fn(i32, i32) -> bool,
    src_y: &[i32],
    ctx: &PuChooseCtx<'_>,
    merge_cands: &[(usize, PuMotion)],
) -> (PuSyntax, PuMotion, u64, u64) {
    let (x, y, w, h) = (geom.x_pb, geom.y_pb, geom.n_pb_w, geom.n_pb_h);
    let l0 = amvp_search_list(field, geom, available, src_y, ctx, merge_cands, 0);
    if !ctx.two_sided {
        let (g, motion, rate, cost) = l0.expect("at least one L0 reference");
        return (
            PuSyntax::Amvp {
                l0: Some(g),
                l1: None,
            },
            motion,
            rate,
            cost,
        );
    }
    let l1 = amvp_search_list(field, geom, available, src_y, ctx, merge_cands, 1);
    let mut cands: Vec<(PuSyntax, PuMotion, u64, u64)> = Vec::with_capacity(3);
    if let Some((g, motion, rate, cost)) = l0 {
        cands.push((
            PuSyntax::Amvp {
                l0: Some(g),
                l1: None,
            },
            motion,
            rate,
            cost,
        ));
    }
    if let Some((g, motion, rate, cost)) = l1 {
        cands.push((
            PuSyntax::Amvp {
                l0: None,
                l1: Some(g),
            },
            motion,
            rate,
            cost,
        ));
    }
    if let (Some((g0, _, _, _)), Some((g1, _, _, _))) = (&l0, &l1) {
        // Bi combination of the two single-list winners: the AMVP
        // predictor derivation of each list is independent of the
        // other, so the resolved per-list MVs match the uni winners.
        let motion = resolve_pu_motion(
            field,
            geom,
            &amvp_pu(Some(*g0), Some(*g1)),
            ctx.mv_ctx,
            available,
        );
        // merge_flag + inter_pred_idc ("1") + per-list mvd/mvp/ref bins.
        let rate = 2
            + (mvd_bits(g0.mvd) + 1 + u64::from(ctx.refs_l0.len() > 1))
            + (mvd_bits(g1.mvd) + 1 + u64::from(ctx.refs_l1.len() > 1));
        let pred = predict_block(ctx.refs_l0, ctx.refs_l1, x, y, w, h, &motion, false);
        let cost = sad(&pred.luma, src_y) + ctx.lambda_me * rate;
        cands.push((
            PuSyntax::Amvp {
                l0: Some(*g0),
                l1: Some(*g1),
            },
            motion,
            rate,
            cost,
        ));
    }
    cands
        .into_iter()
        .min_by_key(|&(_, _, _, cost)| cost)
        .expect("at least one active reference across the lists")
}

/// Encode `merge_idx` (§9.3.3.10 TR with `cMax = MaxNumMergeCand − 1`:
/// bin 0 context-coded, the rest bypass).
pub(crate) fn encode_merge_idx(
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
pub(crate) fn encode_eg_k(w: &mut BitWriter, cabac: &mut CabacEncoder, value: u32, k: u32) {
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
pub(crate) fn encode_mvd_pair(
    w: &mut BitWriter,
    cabac: &mut CabacEncoder,
    ctxs: &mut SliceContexts,
    mvd: Mv,
) {
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

/// §7.3.8.6 — emit one prediction unit's syntax (merge_flag +
/// merge_idx, or the AMVP groups). On B slices `inter_pred_idc` is
/// coded per Table 9-47 (`PRED_BI` = "1", `PRED_L0` = "00", `PRED_L1`
/// = "01"; bin 0 ctxInc = CtDepth = 0, bin 1 ctxInc = 4 — `nPbW +
/// nPbH != 12` for every PU shape this encoder emits). Each used
/// list codes `ref_idx_lX` (a single context bin — at most two active
/// references per list), `mvd_coding` (`mvd_l1_zero_flag == 0`) and
/// `mvp_lX_flag`.
pub(crate) fn encode_pu_syntax(
    w: &mut BitWriter,
    cabac: &mut CabacEncoder,
    ctxs: &mut SliceContexts,
    pu: &PuSyntax,
    b_slice: bool,
    n_l0: usize,
    n_l1: usize,
) {
    encode_pu_syntax_at(w, cabac, ctxs, pu, b_slice, n_l0, n_l1, 0, (CTB, CTB));
}

/// [`encode_pu_syntax`] for an arbitrary coding-tree position: `ct_depth`
/// selects the `inter_pred_idc` bin-0 context (`ctxInc = CtDepth`), and
/// the PU shape `(nPbW, nPbH)` the `nPbW + nPbH == 12` single-bin form
/// (Table 9-47: "0" = `PRED_L0`, "1" = `PRED_L1`, ctxInc 4 — bi
/// prediction is not signallable there). `ref_idx_lX` is the §9.3.3
/// TR of `cMax = num_ref_idx_lX_active_minus1` with bins 0 / 1
/// context-coded (ctxInc 0 / 1) and the rest bypass.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_pu_syntax_at(
    w: &mut BitWriter,
    cabac: &mut CabacEncoder,
    ctxs: &mut SliceContexts,
    pu: &PuSyntax,
    b_slice: bool,
    n_l0: usize,
    n_l1: usize,
    ct_depth: u32,
    (n_pb_w, n_pb_h): (usize, usize),
) {
    match pu {
        PuSyntax::Merge { merge_idx } => {
            cabac.encode_decision(w, &mut ctxs.merge_flag[0], 1);
            encode_merge_idx(w, cabac, ctxs, *merge_idx);
        }
        PuSyntax::Amvp { l0, l1 } => {
            cabac.encode_decision(w, &mut ctxs.merge_flag[0], 0);
            if b_slice {
                if n_pb_w + n_pb_h == 12 {
                    debug_assert!(
                        !(l0.is_some() && l1.is_some()),
                        "8x4 / 4x8 PUs cannot be bi-predicted (§7.4.9.6)"
                    );
                    cabac.encode_decision(w, &mut ctxs.inter_pred_idc[4], u8::from(l1.is_some()));
                } else {
                    let slot = ct_depth.min(3) as usize;
                    let (head, tail) = ctxs.inter_pred_idc.split_at_mut(4);
                    if l0.is_some() && l1.is_some() {
                        cabac.encode_decision(w, &mut head[slot], 1); // PRED_BI
                    } else {
                        cabac.encode_decision(w, &mut head[slot], 0);
                        cabac.encode_decision(w, &mut tail[0], u8::from(l1.is_some()));
                    }
                }
            }
            for (group, n_active) in [(l0, n_l0), (l1, n_l1)] {
                let Some(g) = group else { continue };
                if n_active > 1 {
                    encode_ref_idx(w, cabac, ctxs, g.ref_idx, n_active as u32 - 1);
                }
                encode_mvd_pair(w, cabac, ctxs, g.mvd);
                cabac.encode_decision(w, &mut ctxs.mvp_flag[0], g.mvp_flag);
            }
        }
    }
}

/// `ref_idx_lX` (§9.3.3 TR, `cMax = num_ref_idx_lX_active_minus1`): bin
/// 0 on ctxInc 0, bin 1 on ctxInc 1, the remaining bins bypass — the
/// write-side dual of [`crate::binarization::decode_ref_idx`].
pub(crate) fn encode_ref_idx(
    w: &mut BitWriter,
    cabac: &mut CabacEncoder,
    ctxs: &mut SliceContexts,
    ref_idx: u8,
    c_max: u32,
) {
    if c_max == 0 {
        return;
    }
    let v = u32::from(ref_idx);
    for bin in 0..c_max {
        let one = v > bin;
        match bin {
            0 => cabac.encode_decision(w, &mut ctxs.ref_idx[0], u8::from(one)),
            1 => cabac.encode_decision(w, &mut ctxs.ref_idx[1], u8::from(one)),
            _ => cabac.encode_bypass(w, u8::from(one)),
        }
        if !one {
            break;
        }
    }
}

/// Encode one P / B frame as a single TRAIL_R slice against the
/// reference lists of `spec`: pass 1 makes the per-CTB mode decisions
/// and builds the pre-filter reconstruction, the §8.7 filter stage
/// elects and applies deblocking + SAO, pass 2 emits the slice header
/// and syntax. Returns the slice RBSP, the frame's (filtered)
/// reconstruction, and its CU mode-decision counters.
#[allow(clippy::too_many_lines)]
pub(crate) fn encode_inter_slice(
    frame: &YuvFrame<'_>,
    spec: &SliceSpec<'_>,
    width: usize,
    height: usize,
) -> (Vec<u8>, FrameRecon, FrameStats) {
    if spec.tree.is_some() {
        return crate::encoder::ctu::encode_inter_slice_tree(frame, spec, width, height);
    }
    let (poc, qp, b_slice, lf) = (spec.poc, spec.qp, spec.b_slice, spec.lf);
    let (cw, ch) = (width / 2, height / 2);
    let ctbs_x = width / CTB;
    let ctbs_y = height / CTB;
    // SSD-per-bit tradeoff, doubling every 3 QP (as the intra path);
    // the SAD-domain motion-search λ uses the same scale.
    let lambda_of = |q: i32| -> u64 { 1u64 << (q.unsigned_abs().saturating_sub(9) / 3) };
    let lambda = lambda_of(qp);
    // Spatial AQ: per-CTB QP offsets against the slice QP (all zero
    // at `spec.aq == 0`); signalled through cu_qp_delta.
    let aq_deltas = crate::encoder::aq::ctb_aq_deltas(frame.y, width, height, spec.aq, CTB);

    let to_i32 = |p: &[u8]| -> Vec<i32> { p.iter().map(|&v| i32::from(v)).collect() };
    let to_planes = |list: &[(i32, &FrameRecon)]| -> Vec<RefPlanes> {
        list.iter()
            .map(|&(_, rec)| RefPlanes {
                y: to_i32(&rec.y),
                cb: to_i32(&rec.cb),
                cr: to_i32(&rec.cr),
                width,
                height,
            })
            .collect()
    };
    let refs_l0: Vec<RefPlanes> = to_planes(&spec.l0);
    let refs_l1: Vec<RefPlanes> = to_planes(&spec.l1);
    let l0_pocs: Vec<i32> = spec.l0.iter().map(|&(p, _)| p).collect();
    let l1_pocs: Vec<i32> = spec.l1.iter().map(|&(p, _)| p).collect();
    let n_l0 = l0_pocs.len() as i32;
    let n_l1 = l1_pocs.len() as i32;
    // Hierarchical B: the two lists differ, so the AMVP election
    // searches L1 / bi as well; a low-delay B keeps the L0-only
    // search (L1 duplicates it at a higher signalling rate).
    let two_sided = b_slice && l0_pocs != l1_pocs;

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

    // §8.5.3.2 reference resolvers over the spec's explicit lists.
    let list_pocs = |list: usize| -> &Vec<i32> {
        if list == 0 {
            &l0_pocs
        } else {
            &l1_pocs
        }
    };
    let ref_poc = |list: usize, ref_idx: i32| -> i32 {
        usize::try_from(ref_idx)
            .ok()
            .and_then(|i| list_pocs(list).get(i).copied())
            .unwrap_or(i32::MIN)
    };
    // The decoder's motion-field cells store the REFERENCED
    // picture's POC per used list (pu_mv fill: `ref_poc(list, idx)`);
    // an unused list's value is ignored by `to_cell`.
    let cell_pocs =
        |m: &PuMotion| -> (i32, i32) { (ref_poc(0, m.ref_idx_l0), ref_poc(1, m.ref_idx_l1)) };
    let ref_long_term = |_list: usize, _ref_idx: i32| false;
    let ref_short_term = |list: usize, ref_idx: i32| {
        usize::try_from(ref_idx).is_ok_and(|i| i < list_pocs(list).len())
    };
    let col_ref_long_term = |_poc: i32| false;
    // §8.5.3.2.2 NoBackwardPredFlag: 1 iff every active reference of
    // both lists precedes the current picture in output order.
    let no_backward_pred = !l0_pocs.iter().chain(l1_pocs.iter()).any(|&p| p > poc);
    let mv_ctx = PuMvContext {
        curr_poc: poc,
        slice_is_b: b_slice,
        ctb_log2_size_y: CTB_LOG2,
        pic_width_luma: width as u32,
        pic_height_luma: height as u32,
        max_num_merge_cand: MAX_MERGE,
        num_ref_idx_l0_active: n_l0,
        num_ref_idx_l1_active: if b_slice { n_l1 } else { 0 },
        log2_par_mrg_level: 2,
        temporal_mvp_enabled: false,
        collocated_from_l0_flag: true,
        col_poc: 0,
        no_backward_pred,
        ref_poc: &ref_poc,
        ref_long_term: &ref_long_term,
        ref_short_term: &ref_short_term,
        col_field: None,
        col_ref_long_term: &col_ref_long_term,
        use_integer_mv: false,
        two_versions_curr_pic: false,
        is_curr_pic: &|_, _| false,
    };

    // ---- pass 1: per-CTB mode decisions + reconstruction ----
    let mut plans: Vec<CuCandidate> = Vec::with_capacity(ctbs_x * ctbs_y);
    // The QP each CTB was quantized at (slice QP + AQ offset).
    let mut plan_qps: Vec<i32> = Vec::with_capacity(ctbs_x * ctbs_y);
    // `ctb` also drives the x0/y0 arithmetic, not just the grid index.
    #[allow(clippy::needless_range_loop)]
    for ctb in 0..ctbs_x * ctbs_y {
        let x0 = (ctb % ctbs_x) * CTB;
        let y0 = (ctb / ctbs_x) * CTB;
        let (cx0, cy0) = (x0 / 2, y0 / 2);
        let ctb_qp = (qp + aq_deltas[ctb]).clamp(0, 51);
        plan_qps.push(ctb_qp);
        let qp_y = ctb_qp as u32;
        let qp_c = chroma_qp_420(ctb_qp);
        let lambda = lambda_of(ctb_qp);
        let lambda_me = crate::encoder::rate::isqrt_u64(lambda);
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
                    let pred = predict_block(&refs_l0, &refs_l1, x0, y0, CTB, CTB, &m, false);
                    (
                        sad(&pred.luma, &src[0]) + lambda_me * (idx as u64 + 1),
                        idx,
                        m,
                    )
                })
                .min_by_key(|&(cost, idx, _)| (cost, idx))
                .map(|(_, idx, m)| (idx, m))
                .expect("merge list is never empty");

            // ---- AMVP: §8.5.3.2.6 predictors + per-reference ME ----
            let choose_ctx = PuChooseCtx {
                refs_l0: &refs_l0,
                refs_l1: &refs_l1,
                mv_ctx: &mv_ctx,
                lambda_me,
                b_slice,
                two_sided,
            };
            let (amvp_syntax, amvp_motion, amvp_rate, _) = amvp_search(
                &field,
                &geom,
                &available,
                &src[0],
                &choose_ctx,
                &merge_cands,
            );

            // ---- fully code the finalists ----
            let mut cands: Vec<CuCandidate> = Vec::with_capacity(6);

            // Skip: best merge candidate, prediction only.
            let merge_pred = predict_block(
                &refs_l0,
                &refs_l1,
                x0,
                y0,
                CTB,
                CTB,
                &best_merge_motion,
                true,
            );
            let (skip_recon, skip_dist) = pred_recon(&merge_pred, &src);
            cands.push(CuCandidate {
                kind: CuKind::Skip {
                    merge_idx: best_merge_idx,
                },
                motions: vec![best_merge_motion],
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
                    motions: vec![best_merge_motion],
                    levels: Some(CuLevels::Depth0(m_levels)),
                    recon: m_recon,
                    cost: m_dist + lambda * (m_rate + best_merge_idx as u64 + 3),
                });
            }

            // AMVP (with or without residual).
            let amvp_pred = predict_block(&refs_l0, &refs_l1, x0, y0, CTB, CTB, &amvp_motion, true);
            let (a_levels, a_recon, a_dist, a_rate) =
                code_ctb_residual(&amvp_pred, &src, qp_y, qp_c);
            let motion_rate = amvp_rate + 2; // + skip / rqt flags
            if a_levels.iter().any(|l| l.iter().any(|&v| v != 0)) {
                cands.push(CuCandidate {
                    kind: CuKind::Amvp { pu: amvp_syntax },
                    motions: vec![amvp_motion],
                    levels: Some(CuLevels::Depth0(a_levels)),
                    recon: a_recon,
                    cost: a_dist + lambda * (a_rate + motion_rate),
                });
            } else {
                // All-zero residual: rqt_root_cbf == 0.
                let (pr, pd) = pred_recon(&amvp_pred, &src);
                cands.push(CuCandidate {
                    kind: CuKind::Amvp { pu: amvp_syntax },
                    motions: vec![amvp_motion],
                    levels: None,
                    recon: pr,
                    cost: pd + lambda * motion_rate,
                });
            }

            // ---- two-PU partitions: the symmetric rectangles, plus
            // the four asymmetric shapes on an AMP stream ----
            let mut two_pu_parts = vec![PartMode::Part2NxN, PartMode::PartNx2N];
            if spec.big_cu {
                two_pu_parts.extend([
                    PartMode::Part2NxnU,
                    PartMode::Part2NxnD,
                    PartMode::PartNLx2N,
                    PartMode::PartNRx2N,
                ]);
            }
            for part in two_pu_parts {
                let rects = crate::pu_mv::pu_partitions(x0, y0, CTB, part);
                // The decoder resolves PU1 with PU0's motion already in
                // the field (§8.5.3.2 order): stage the writes on a copy.
                let mut tmp = field.clone();
                let mut pus = [PuSyntax::Merge { merge_idx: 0 }; 2];
                let mut motions_r = Vec::with_capacity(2);
                let mut pred_y = vec![0i32; CTB * CTB];
                let mut pred_cb = vec![0i32; 64];
                let mut pred_cr = vec![0i32; 64];
                // pred_mode + part_mode bins (the AMP forms carry two
                // extra part_mode bins).
                let mut motion_rate_r = if part_is_amp(part) { 5u64 } else { 3u64 };
                for (k, r) in rects.iter().enumerate() {
                    let g = PuGeometry {
                        x_cb: x0,
                        y_cb: y0,
                        n_cb_s: CTB,
                        x_pb: r.x_pb,
                        y_pb: r.y_pb,
                        n_pb_w: r.n_pb_w,
                        n_pb_h: r.n_pb_h,
                        part_mode: part,
                        part_idx: k as u32,
                    };
                    // §6.4.2 availability against the staged field. The
                    // decoder's closure passes the CU as the PU geometry
                    // (the sameCb NxN clause never fires for 2-PU
                    // shapes) and treats the current CU as MODE_INTER.
                    let avail_k = |x_nb: i32, y_nb: i32| -> bool {
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
                                let (xu, yu) = (x as usize, y as usize);
                                let inside_cu =
                                    (x0..x0 + CTB).contains(&xu) && (y0..y0 + CTB).contains(&yu);
                                if !inside_cu && tmp.cell_at(xu, yu).is_intra {
                                    MODE_INTRA
                                } else {
                                    0
                                }
                            },
                        )
                    };
                    let mut src_pu = Vec::with_capacity(r.n_pb_w * r.n_pb_h);
                    for j in 0..r.n_pb_h {
                        for i in 0..r.n_pb_w {
                            src_pu.push(i32::from(frame.y[(r.y_pb + j) * width + r.x_pb + i]));
                        }
                    }
                    let (pu_syntax, motion, rate_k) =
                        choose_pu(&tmp, &g, &avail_k, &src_pu, &choose_ctx);
                    // eqs 8-80..8-85: PU1's derivation sees PU0's motion.
                    tmp.fill_rect(r.x_pb, r.y_pb, r.n_pb_w, r.n_pb_h, {
                        let (p0, p1) = cell_pocs(&motion);
                        motion.to_cell(p0, p1)
                    });
                    let p = predict_block(
                        &refs_l0, &refs_l1, r.x_pb, r.y_pb, r.n_pb_w, r.n_pb_h, &motion, true,
                    );
                    blit(
                        &mut pred_y,
                        CTB,
                        r.x_pb - x0,
                        r.y_pb - y0,
                        &p.luma,
                        r.n_pb_w,
                        r.n_pb_h,
                    );
                    blit(
                        &mut pred_cb,
                        8,
                        (r.x_pb - x0) / 2,
                        (r.y_pb - y0) / 2,
                        &p.cb,
                        r.n_pb_w / 2,
                        r.n_pb_h / 2,
                    );
                    blit(
                        &mut pred_cr,
                        8,
                        (r.x_pb - x0) / 2,
                        (r.y_pb - y0) / 2,
                        &p.cr,
                        r.n_pb_w / 2,
                        r.n_pb_h / 2,
                    );
                    pus[k] = pu_syntax;
                    motions_r.push(motion);
                    motion_rate_r += rate_k;
                }
                // §7.4.9.8 interSplitFlag: forced depth-1 tree — four
                // z-order 8x8 luma + 4x4 chroma TBs.
                let mut luma_l: [Vec<i32>; 4] = Default::default();
                let mut cb_l: [Vec<i32>; 4] = Default::default();
                let mut cr_l: [Vec<i32>; 4] = Default::default();
                let mut recon_r = [vec![0u8; CTB * CTB], vec![0u8; 64], vec![0u8; 64]];
                let mut rate_r = 0u64;
                for (k, &(zx, zy)) in Z_OFFSETS.iter().enumerate() {
                    let (lx, ly8) = (zx * 8, zy * 8);
                    let s_y = sub_block(&src[0], CTB, lx, ly8, 8);
                    let p_y = sub_block(&pred_y, CTB, lx, ly8, 8);
                    let (lv, rc) = code_tb(&s_y, &p_y, 8, qp_y, Component::Luma, PredMode::Inter);
                    for j in 0..8 {
                        recon_r[0][(ly8 + j) * CTB + lx..(ly8 + j) * CTB + lx + 8]
                            .copy_from_slice(&rc[j * 8..(j + 1) * 8]);
                    }
                    rate_r += levels_rate(&lv);
                    luma_l[k] = lv;
                    for (plane, src_c, pred_c, out) in [
                        (1usize, &src[1], &pred_cb, &mut cb_l),
                        (2usize, &src[2], &pred_cr, &mut cr_l),
                    ] {
                        let s_c = sub_block(src_c, 8, zx * 4, zy * 4, 4);
                        let p_c = sub_block(pred_c, 8, zx * 4, zy * 4, 4);
                        let comp = if plane == 1 {
                            Component::Cb
                        } else {
                            Component::Cr
                        };
                        let (lv_c, rc_c) = code_tb(&s_c, &p_c, 4, qp_c, comp, PredMode::Inter);
                        for j in 0..4 {
                            recon_r[plane]
                                [(zy * 4 + j) * 8 + zx * 4..(zy * 4 + j) * 8 + zx * 4 + 4]
                                .copy_from_slice(&rc_c[j * 4..(j + 1) * 4]);
                        }
                        rate_r += levels_rate(&lv_c);
                        out[k] = lv_c;
                    }
                }
                let any_nonzero = luma_l
                    .iter()
                    .chain(cb_l.iter())
                    .chain(cr_l.iter())
                    .any(|l| l.iter().any(|&v| v != 0));
                let (levels_r, recon_final, dist_r) = if any_nonzero {
                    let d = ssd(&recon_r[0], &src[0])
                        + ssd(&recon_r[1], &src[1])
                        + ssd(&recon_r[2], &src[2]);
                    (
                        Some(CuLevels::Depth1(Box::new(Depth1Levels {
                            luma: luma_l,
                            cb: cb_l,
                            cr: cr_l,
                        }))),
                        recon_r,
                        d + lambda * rate_r,
                    )
                } else {
                    // rqt_root_cbf == 0: prediction-only reconstruction.
                    let clip = |v: &[i32]| -> Vec<u8> {
                        v.iter().map(|&p| p.clamp(0, 255) as u8).collect()
                    };
                    let rec = [clip(&pred_y), clip(&pred_cb), clip(&pred_cr)];
                    let d = ssd(&rec[0], &src[0]) + ssd(&rec[1], &src[1]) + ssd(&rec[2], &src[2]);
                    (None, rec, d)
                };
                cands.push(CuCandidate {
                    kind: CuKind::TwoPu { part, pus },
                    motions: motions_r,
                    levels: levels_r,
                    recon: recon_final,
                    cost: dist_r + lambda * (motion_rate_r + 1),
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
                    motions: Vec::new(),
                    levels: Some(CuLevels::Depth0([ly, lcb, lcr])),
                    recon: [ry, rcb, rcr],
                    cost: dist + lambda * rate,
                });
            }

            cands
                .into_iter()
                .min_by_key(|c| c.cost)
                .expect("at least the skip candidate")
        };

        // ---- state update (eqs 8-80..8-85 + mode fields + recon) ----
        let is_skip = matches!(chosen.kind, CuKind::Skip { .. });
        skip_grid[ctb] = is_skip;
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
                    CuKind::TwoPu { part, pus } => {
                        if pus.iter().any(|p| matches!(p, PuSyntax::Amvp { .. })) {
                            stats.amvp += 1;
                        } else {
                            stats.merge += 1;
                        }
                        if part_is_amp(*part) {
                            stats.amp += 1;
                        } else {
                            stats.rect += 1;
                        }
                    }
                    _ => stats.amvp += 1,
                }
                if chosen
                    .motions
                    .iter()
                    .any(|m| m.pred_flag_l0 && m.pred_flag_l1)
                {
                    stats.bi += 1;
                }
                if chosen.motions.iter().any(|m| {
                    (m.pred_flag_l0 && m.ref_idx_l0 > 0) || (m.pred_flag_l1 && m.ref_idx_l1 > 0)
                }) {
                    stats.ref1 += 1;
                }
                // eqs 8-80..8-85: one motion cell per PU rectangle, the
                // stored identity being the referenced picture's POC.
                let part = match kind {
                    CuKind::TwoPu { part, .. } => *part,
                    _ => PartMode::Part2Nx2N,
                };
                let rects = crate::pu_mv::pu_partitions(x0, y0, CTB, part);
                for (r, m) in rects.iter().zip(chosen.motions.iter()) {
                    field.fill_rect(r.x_pb, r.y_pb, r.n_pb_w, r.n_pb_h, {
                        let (p0, p1) = cell_pocs(m);
                        m.to_cell(p0, p1)
                    });
                }
                let cu_mode = if is_skip {
                    CuPredMode::Skip
                } else {
                    CuPredMode::Inter
                };
                modes.record_non_intra_cu(x0, y0, CTB, cu_mode);
                // §8.7.2.4 — mark each luma TRANSFORM BLOCK carrying a
                // coded coefficient (the decoder marks per tree leaf,
                // not per CU; intra CUs are never marked — their edges
                // are bS 2 regardless).
                match &chosen.levels {
                    Some(CuLevels::Depth0(l)) if l[0].iter().any(|&v| v != 0) => {
                        field.mark_nonzero_coeff(x0, y0, CTB, CTB);
                    }
                    Some(CuLevels::Depth1(d)) => {
                        for (k, &(zx, zy)) in Z_OFFSETS.iter().enumerate() {
                            if d.luma[k].iter().any(|&v| v != 0) {
                                field.mark_nonzero_coeff(x0 + zx * 8, y0 + zy * 8, 8, 8);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        store(&mut recon.y, width, x0, y0, CTB, &chosen.recon[0]);
        store(&mut recon.cb, cw, cx0, cy0, 8, &chosen.recon[1]);
        store(&mut recon.cr, cw, cx0, cy0, 8, &chosen.recon[2]);
        plans.push(chosen);
    }

    // ---- §8.7 in-loop filters (deblocking + SAO) on the recon ----
    let mut lf_sig = SliceLfSignalling {
        cfg: lf,
        deblock_on: false,
        beta_offset_div2: 0,
        tc_offset_div2: 0,
        sao_luma: false,
        sao_chroma: false,
    };
    let mut sao_ctbs: Vec<crate::slice_data::SaoCtbParams> = Vec::new();
    // §8.6.1 effective per-CTB QpY a decoder derives: a CU transmits
    // its cu_qp_delta only when its transform tree codes some cbf
    // (skip / rqt_root_cbf == 0 / all-zero-TU CUs transmit nothing
    // and inherit the predicted QP — with one QG per CTB, the running
    // qPY_PREV). Constant everywhere when AQ is off.
    let any_cbf = |c: &CuCandidate| -> bool {
        match &c.levels {
            Some(CuLevels::Depth0(l)) => l.iter().any(|lv| lv.iter().any(|&v| v != 0)),
            Some(CuLevels::Depth1(d)) => {
                let Depth1Levels { luma, cb, cr } = d.as_ref();
                luma.iter()
                    .chain(cb.iter())
                    .chain(cr.iter())
                    .any(|lv| lv.iter().any(|&v| v != 0))
            }
            None => false,
        }
    };
    let eff_qps: Vec<i32> = {
        let mut prev = qp;
        plans
            .iter()
            .enumerate()
            .map(|(i, c)| {
                if spec.aq > 0 {
                    if any_cbf(c) {
                        prev = plan_qps[i];
                    }
                    prev
                } else {
                    qp
                }
            })
            .collect()
    };
    if lf.any() {
        let shapes: Vec<CtbShape> = plans
            .iter()
            .map(|c| CtbShape {
                part_mode: match &c.kind {
                    CuKind::TwoPu { part, .. } => bin_part_mode(*part),
                    _ => crate::binarization::PartMode::Part2Nx2N,
                },
                split_depth1: matches!(c.levels, Some(CuLevels::Depth1(_))),
            })
            .collect();
        let out = filter_frame(
            &FilterInput {
                width,
                height,
                ctb_qps: &eff_qps,
                lambda,
                recon: [&recon.y, &recon.cb, &recon.cr],
                src: [frame.y, frame.cb, frame.cr],
                field: &field,
                shapes: &shapes,
                ctb_log2: 4,
                tree: None,
            },
            lf,
        );
        lf_sig.deblock_on = out.deblock_on;
        lf_sig.beta_offset_div2 = out.beta_offset_div2;
        lf_sig.tc_offset_div2 = out.tc_offset_div2;
        lf_sig.sao_luma = out.slice_sao_luma;
        lf_sig.sao_chroma = out.slice_sao_chroma;
        sao_ctbs = out.sao_ctbs;
        recon.y = out.y;
        recon.cb = out.cb;
        recon.cr = out.cr;
    }

    // ---- slice_segment_header( ) ----
    let mut w = BitWriter::new();
    write_inter_slice_header(&mut w, spec, &lf_sig);

    // ---- slice_segment_data( ) — pass 2: syntax emission ----
    let mut cabac = CabacEncoder::new();
    // Table 9-4: cabac_init_flag 0 ⇒ initType 1 (P) / 2 (B).
    let raw_slice_type = if b_slice { 0 } else { 1 };
    let mut ctxs = SliceContexts::init(init_type(raw_slice_type, false), qp);
    // §8.6.1 qPY_PREV thread for the cu_qp_delta emission (one QG per
    // CTB ⇒ qPY_PRED == qPY_PREV; SliceQpY at the slice start).
    let mut qp_prev = qp;
    for (ctb, chosen) in plans.iter().enumerate() {
        let x0 = (ctb % ctbs_x) * CTB;
        let y0 = (ctb / ctbs_x) * CTB;
        if lf_sig.sao_luma || lf_sig.sao_chroma {
            // §7.3.8.3 sao( rx, ry ) ahead of the coding quadtree.
            encode_sao_ctb(
                &mut w,
                &mut cabac,
                &mut ctxs,
                &sao_ctbs[ctb],
                ctb % ctbs_x,
                ctb / ctbs_x,
                lf_sig.sao_luma,
                lf_sig.sao_chroma,
            );
        }

        if spec.big_cu {
            // §7.3.8.4: log2CbSize (4) > MinCbLog2SizeY (3) —
            // split_cu_flag is coded and 0 (one unsplit 16x16 CU per
            // CTB; all CtDepth 0 ⇒ both §9.3.4.2.2 conds 0 ⇒ ctxInc 0).
            cabac.encode_decision(&mut w, &mut ctxs.split_cu_flag[0], 0);
        }

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

        match &chosen.kind {
            CuKind::Skip { merge_idx } => {
                encode_merge_idx(&mut w, &mut cabac, &mut ctxs, *merge_idx);
            }
            CuKind::Merge { merge_idx } => {
                // pred_mode_flag = 0 (MODE_INTER), part_mode "1" (2Nx2N).
                cabac.encode_decision(&mut w, &mut ctxs.pred_mode_flag[0], 0);
                cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], 1);
                encode_pu_syntax(
                    &mut w,
                    &mut cabac,
                    &mut ctxs,
                    &PuSyntax::Merge {
                        merge_idx: *merge_idx,
                    },
                    b_slice,
                    n_l0 as usize,
                    n_l1 as usize,
                );
                // rqt_root_cbf not present (2Nx2N merge ⇒ inferred 1).
            }
            CuKind::Amvp { pu } => {
                cabac.encode_decision(&mut w, &mut ctxs.pred_mode_flag[0], 0);
                cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], 1);
                encode_pu_syntax(
                    &mut w,
                    &mut cabac,
                    &mut ctxs,
                    pu,
                    b_slice,
                    n_l0 as usize,
                    n_l1 as usize,
                );
                cabac.encode_decision(
                    &mut w,
                    &mut ctxs.rqt_root_cbf[0],
                    u8::from(chosen.levels.is_some()),
                );
            }
            CuKind::TwoPu { part, pus } => {
                cabac.encode_decision(&mut w, &mut ctxs.pred_mode_flag[0], 0);
                // §9.3.3.7 inter part_mode (bin 0 ctx 0, bin 1 ctx 1):
                // * at MinCb (log2CbSize == MinCbLog2SizeY > 3):
                //   "01" = PART_2NxN, "001" = PART_Nx2N (bin 2 the
                //   MinCb ctx slot 2);
                // * above MinCb with amp_enabled_flag: "011" =
                //   PART_2NxN, "001" = PART_Nx2N (bin 2 = 1 selects
                //   the symmetric form, ctx slot 3), and the AMP
                //   forms "0100" = PART_2NxnU, "0101" = PART_2NxnD,
                //   "0000" = PART_nLx2N, "0001" = PART_nRx2N (bin 2 =
                //   0, bin 3 bypass picking the second-quarter side).
                let horizontal = part_is_horizontal(*part);
                let amp_shape = part_is_amp(*part);
                cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], 0);
                cabac.encode_decision(&mut w, &mut ctxs.part_mode[1], u8::from(horizontal));
                if spec.big_cu {
                    cabac.encode_decision(&mut w, &mut ctxs.part_mode[3], u8::from(!amp_shape));
                    if amp_shape {
                        let second = matches!(part, PartMode::Part2NxnD | PartMode::PartNRx2N);
                        cabac.encode_bypass(&mut w, u8::from(second));
                    }
                } else if !horizontal {
                    cabac.encode_decision(&mut w, &mut ctxs.part_mode[2], 1);
                }
                for pu in pus {
                    encode_pu_syntax(
                        &mut w,
                        &mut cabac,
                        &mut ctxs,
                        pu,
                        b_slice,
                        n_l0 as usize,
                        n_l1 as usize,
                    );
                }
                // rqt_root_cbf: present (PartMode != PART_2Nx2N).
                cabac.encode_decision(
                    &mut w,
                    &mut ctxs.rqt_root_cbf[0],
                    u8::from(chosen.levels.is_some()),
                );
            }
            CuKind::Intra { mode } => {
                // pred_mode_flag = 1 (MODE_INTRA), part_mode "1"
                // (2Nx2N) at MinCb — NOT present above MinCb
                // (§7.3.8.5, inferred PART_2Nx2N); PCM disabled in
                // the SPS.
                cabac.encode_decision(&mut w, &mut ctxs.pred_mode_flag[0], 1);
                if !spec.big_cu {
                    cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], 1);
                }
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
        // §7.4.9.11: inter TBs use the up-right diagonal scan; the
        // TB geometries the intra fallback emits are not
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
            persistent_rice_adaptation_enabled_flag: false,
            cabac_bypass_alignment_enabled_flag: false,
            extended_precision_processing_flag: false,
            bit_depth: 8,
            rice_stat_transform_skip: false,
        };
        match &chosen.levels {
            Some(CuLevels::Depth0(levels)) => {
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
                // §7.3.8.8: an intra CU always signals cbf_luma; an
                // inter depth-0 TU signals it only when a chroma cbf is
                // set (otherwise it is inferred 1).
                if cu_is_intra || cbf_cb || cbf_cr {
                    cabac.encode_decision(
                        &mut w,
                        &mut ctxs.cbf_luma[cbf_luma_ctx_inc(0) as usize],
                        u8::from(cbf_luma),
                    );
                } else {
                    debug_assert!(cbf_luma, "all-zero transform tree must not be coded");
                }
                // §7.3.8.10 delta_qp( ) in the (only) TU when it
                // carries a cbf.
                if spec.aq > 0 && (cbf_luma || cbf_cb || cbf_cr) {
                    encode_cu_qp_delta(&mut w, &mut cabac, &mut ctxs, plan_qps[ctb] - qp_prev);
                    qp_prev = plan_qps[ctb];
                }
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
            Some(CuLevels::Depth1(d)) => {
                let Depth1Levels { luma, cb, cr } = d.as_ref();
                // §7.4.9.8 interSplitFlag == 1: split_transform_flag
                // inferred 1 at depth 0; four 8x8 leaves at depth 1.
                // Root cbf_cb / cbf_cr gate the per-leaf chroma flags
                // (§7.3.8.8 inheritance); cbf_luma is signalled at
                // every depth-1 leaf (trafoDepth != 0).
                let leaf_nz = |lv: &Vec<i32>| lv.iter().any(|&v| v != 0);
                let root_cb = cb.iter().any(&leaf_nz);
                let root_cr = cr.iter().any(&leaf_nz);
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
                    let cbf_cb_k = leaf_nz(&cb[k]);
                    let cbf_cr_k = leaf_nz(&cr[k]);
                    let cbf_luma_k = leaf_nz(&luma[k]);
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
                    // §7.3.8.10 delta_qp( ) in the first depth-1 TU
                    // with a cbf.
                    if spec.aq > 0 && !delta_done && (cbf_luma_k || cbf_cb_k || cbf_cr_k) {
                        encode_cu_qp_delta(&mut w, &mut cabac, &mut ctxs, plan_qps[ctb] - qp_prev);
                        qp_prev = plan_qps[ctb];
                        delta_done = true;
                    }
                    if cbf_luma_k {
                        encode_residual_coding(
                            &mut w,
                            &mut cabac,
                            &mut ctxs.residual,
                            &rc_params(3, 0),
                            &luma[k],
                        )
                        .expect("validated luma levels");
                    }
                    if root_cb && cbf_cb_k {
                        encode_residual_coding(
                            &mut w,
                            &mut cabac,
                            &mut ctxs.residual,
                            &rc_params(2, 1),
                            &cb[k],
                        )
                        .expect("validated cb levels");
                    }
                    if root_cr && cbf_cr_k {
                        encode_residual_coding(
                            &mut w,
                            &mut cabac,
                            &mut ctxs.residual,
                            &rc_params(2, 2),
                            &cr[k],
                        )
                        .expect("validated cr levels");
                    }
                }
            }
            None => {}
        }

        // end_of_slice_segment_flag.
        cabac.encode_terminate(&mut w, u8::from(ctb == ctbs_x * ctbs_y - 1));
    }
    w.align_zero();
    (w.finish(), recon, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::intra::encode_idr_intra_au_cfg;
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
        let idr = encode_idr_intra_au_cfg(&planes[0].0, &planes[0].1, &planes[0].2, 64, 64, 30, 2)
            .expect("intra encode");
        let p_bytes = enc.stream.len() - idr.au.len();
        let p_avg = p_bytes / 4;
        assert!(
            p_avg * 3 < idr.au.len(),
            "P frames ({p_avg} B avg) should be much smaller than the IDR ({} B)",
            idr.au.len()
        );
    }

    /// A static scene (all frames identical) codes its P frames as
    /// (almost) all-skip: tiny payloads, and every decoded P frame is
    /// at least as close to the source as the IDR reconstruction it
    /// predicts from (the SAD-domain motion λ lets a CU re-code the
    /// IDR's quantization ripple when that is the cheaper RD choice,
    /// so the P frames need not be sample-identical to frame 0).
    #[test]
    fn static_scene_is_skip_coded() {
        let one = scene(64, 64, 1).remove(0);
        let planes = vec![one.clone(), one.clone(), one];
        let frames = as_frames(&planes);
        let enc = encode_low_delay_p(&frames, 64, 64, 26).expect("encode");
        let decoded = decode_annexb_sequence(&enc.stream).expect("decode");
        assert_eq!(decoded.len(), 3);
        // Decoder output == encoder reconstruction, frame by frame.
        for (i, (d, rec)) in decoded.iter().zip(enc.recon.iter()).enumerate() {
            let mut expect = rec.y.clone();
            expect.extend_from_slice(&rec.cb);
            expect.extend_from_slice(&rec.cr);
            assert_eq!(
                d.picture.to_planar_u8().expect("8-bit"),
                expect,
                "frame {i}"
            );
        }
        // Nearly every CTB is skip, and no P frame drifts away from
        // the source relative to the IDR.
        let sad = |rec: &FrameRecon| -> u64 {
            rec.y
                .iter()
                .zip(planes[0].0.iter())
                .map(|(&a, &b)| u64::from(a.abs_diff(b)))
                .sum()
        };
        let idr_sad = sad(&enc.recon[0]);
        for (i, (st, rec)) in enc.stats.iter().zip(enc.recon.iter()).enumerate().skip(1) {
            assert!(
                st.skip >= 15,
                "frame {i}: expected >= 15/16 skip CTBs, got {st:?}"
            );
            assert!(
                sad(rec) <= idr_sad,
                "frame {i}: P recon drifted from the source"
            );
        }
        // Each (almost) all-skip P slice is a handful of bytes.
        let idr_len =
            encode_idr_intra_au_cfg(&planes[0].0, &planes[0].1, &planes[0].2, 64, 64, 26, 2)
                .expect("intra")
                .au
                .len();
        let p_bytes = enc.stream.len() - idr_len;
        assert!(
            p_bytes < 128,
            "two near-all-skip P slices should be tiny, got {p_bytes} bytes"
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

        // The scene-change P frame elects intra for a meaningful share
        // of its CTBs (the depth-1 rectangular-inter trees legitimately
        // take the rest — the residual does the work there).
        let st = enc.stats[2];
        assert!(
            st.intra >= (w / 16) * (h / 16) / 4,
            "scene change should elect intra CUs, got {st:?}"
        );
        assert_eq!(st.skip, 0, "nothing to skip across a scene change: {st:?}");
        // The steady frame before it stays inter-dominated (the CTBs
        // around the moving square may still legitimately elect intra
        // on the covered / uncovered texture).
        let st1 = enc.stats[1];
        assert!(
            st1.intra < (w / 16) * (h / 16) / 2,
            "steady frame should be inter-dominated, got {st1:?}"
        );
        assert!(
            st.intra > st1.intra,
            "the scene change should elect more intra than the steady frame: {st:?} vs {st1:?}"
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

    /// Rectangular partitions are actually elected on content with
    /// per-half motion (a horizontal boundary moving through CTBs),
    /// and the streams stay bit-exact through our decoder — including
    /// the §8.5.3.2 second-PU derivation reading the first PU.
    #[test]
    fn rect_partitions_are_selected_and_roundtrip() {
        // Two texture fields moving in opposite directions: the top
        // half scrolls right, the bottom half scrolls left, with the
        // boundary offset from the CTB grid so 2NxN splits pay off.
        let (w, h) = (64usize, 64usize);
        let planes: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..4usize)
            .map(|t| {
                let y: Vec<u8> = (0..w * h)
                    .map(|i| {
                        let (x, yy) = (i % w, i / w);
                        if yy < 24 {
                            (((x + w - 2 * t) * 5 + yy * 3) % 220) as u8
                        } else {
                            (((x + 2 * t) * 7 + yy) % 220) as u8
                        }
                    })
                    .collect();
                let cb = vec![110u8; w * h / 4];
                let cr = vec![150u8; w * h / 4];
                (y, cb, cr)
            })
            .collect();
        let frames = as_frames(&planes);
        for (qp, b) in [(24i32, false), (30, true)] {
            let mut enc = LowDelayPEncoder::new(w, h, qp, 0)
                .expect("encoder")
                .with_b_slices(b);
            let mut stream = Vec::new();
            let mut recons = Vec::new();
            let mut rect_total = 0usize;
            for f in &frames {
                let out = enc.encode_frame(f).expect("encode");
                stream.extend_from_slice(&out.au);
                recons.push(out.recon);
                rect_total += out.stats.rect;
            }
            assert!(
                rect_total > 0,
                "qp{qp} b={b}: expected rectangular partitions on split-motion content"
            );
            let decoded = decode_annexb_sequence(&stream).expect("decode");
            assert_eq!(decoded.len(), 4, "qp{qp} b={b}");
            for (i, (dec, rec)) in decoded.iter().zip(recons.iter()).enumerate() {
                let mut expect = rec.y.clone();
                expect.extend_from_slice(&rec.cb);
                expect.extend_from_slice(&rec.cr);
                assert_eq!(
                    dec.picture.to_planar_u8().expect("8-bit"),
                    expect,
                    "qp{qp} b={b} frame {i}"
                );
            }
        }
    }

    /// Flickering content (even frames match even frames) drives the
    /// per-reference AMVP search to the POC − 2 reference — `ref_idx
    /// == 1` is elected and signalled, and the stream decodes
    /// bit-exactly.
    #[test]
    fn second_reference_is_elected_on_flicker() {
        let (w, h) = (64usize, 64usize);
        // Two alternating texture phases + a moving square so pure
        // skip cannot win everywhere.
        let planes: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..5usize)
            .map(|t| {
                let phase = (t % 2) * 97;
                let y: Vec<u8> = (0..w * h)
                    .map(|i| {
                        let (x, yy) = (i % w, i / w);
                        let base = ((x * 7) ^ (yy * 5)) % 180 + phase;
                        let (sx, sy) = (8 + t * 2, 20);
                        if x >= sx && x < sx + 10 && yy >= sy && yy < sy + 10 {
                            (base + 60).min(255) as u8
                        } else {
                            base as u8
                        }
                    })
                    .collect();
                let cb = vec![(90 + phase / 2) as u8; w * h / 4];
                let cr = vec![(170 - phase / 2) as u8; w * h / 4];
                (y, cb, cr)
            })
            .collect();
        let frames = as_frames(&planes);
        let enc = encode_low_delay_p(&frames, w, h, 24).expect("encode");
        let ref1_total: usize = enc.stats.iter().map(|s| s.ref1).sum();
        assert!(
            ref1_total > 0,
            "flicker content should elect the POC-2 reference: {:?}",
            enc.stats
        );
        let decoded = decode_annexb_sequence(&enc.stream).expect("decode");
        assert_eq!(decoded.len(), 5);
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
        // Every frame ends up visually faithful despite the flicker.
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
            assert!(psnr > 30.0, "frame {t} luma PSNR {psnr:.1} dB");
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

    /// Low-delay B slices: both reference lists resolve to the
    /// previous picture; the whole GOP decodes bit-exactly and the
    /// zero-merge / combined candidates actually exercise
    /// bi-prediction.
    #[test]
    fn b_gop_decodes_to_encoder_recon_exactly() {
        let planes = scene(64, 64, 4);
        let frames = as_frames(&planes);
        for qp in [14i32, 27, 39] {
            let mut enc = LowDelayPEncoder::new(64, 64, qp, 0)
                .expect("encoder")
                .with_b_slices(true);
            let mut stream = Vec::new();
            let mut recons = Vec::new();
            let mut bi_total = 0usize;
            for f in &frames {
                let out = enc.encode_frame(f).expect("encode");
                stream.extend_from_slice(&out.au);
                recons.push(out.recon);
                bi_total += out.stats.bi;
            }
            let decoded = decode_annexb_sequence(&stream).expect("decode");
            assert_eq!(decoded.len(), 4, "qp{qp}");
            for (i, (dec, rec)) in decoded.iter().zip(recons.iter()).enumerate() {
                let mut expect = rec.y.clone();
                expect.extend_from_slice(&rec.cb);
                expect.extend_from_slice(&rec.cr);
                assert_eq!(
                    dec.picture.to_planar_u8().expect("8-bit"),
                    expect,
                    "qp{qp} frame {i}: decoder output == encoder recon"
                );
            }
            assert!(
                bi_total > 0,
                "qp{qp}: expected some bi-predicted CUs in the B GOP"
            );
        }
    }

    /// B slices with an IDR-refreshing GOP also roundtrip (initType 2
    /// contexts + POC reset + both-list handling across the wrap).
    #[test]
    fn b_gop_with_idr_refresh_roundtrips() {
        let planes = scene(48, 48, 5);
        let frames = as_frames(&planes);
        let mut enc = LowDelayPEncoder::new(48, 48, 24, 3)
            .expect("encoder")
            .with_b_slices(true);
        let mut stream = Vec::new();
        let mut recons = Vec::new();
        for (i, f) in frames.iter().enumerate() {
            let out = enc.encode_frame(f).expect("encode");
            assert_eq!(out.keyframe, i % 3 == 0, "frame {i} keyframe");
            stream.extend_from_slice(&out.au);
            recons.push(out.recon);
        }
        let decoded = decode_annexb_sequence(&stream).expect("decode");
        assert_eq!(decoded.len(), 5);
        for (i, (dec, rec)) in decoded.iter().zip(recons.iter()).enumerate() {
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

    /// Encode a whole GOP with a given filter cfg and assert every
    /// frame decodes bit-exactly to the encoder's (filtered) recon
    /// through the crate's own decoder. Returns the per-frame recons.
    fn assert_filtered_gop_roundtrip(
        planes: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
        w: usize,
        h: usize,
        qp: i32,
        b: bool,
        gop: usize,
        cfg: LoopFilterCfg,
    ) -> Vec<FrameRecon> {
        let frames = as_frames(planes);
        let mut enc = LowDelayPEncoder::new(w, h, qp, gop)
            .expect("encoder")
            .with_b_slices(b)
            .with_loop_filters(cfg);
        let mut stream = Vec::new();
        let mut recons = Vec::new();
        for f in &frames {
            let out = enc.encode_frame(f).expect("encode");
            stream.extend_from_slice(&out.au);
            recons.push(out.recon);
        }
        let decoded = decode_annexb_sequence(&stream).expect("decode");
        assert_eq!(decoded.len(), planes.len(), "qp{qp} b={b} {cfg:?}");
        for (i, (dec, rec)) in decoded.iter().zip(recons.iter()).enumerate() {
            let mut expect = rec.y.clone();
            expect.extend_from_slice(&rec.cb);
            expect.extend_from_slice(&rec.cr);
            assert_eq!(
                dec.picture.to_planar_u8().expect("8-bit"),
                expect,
                "qp{qp} b={b} {cfg:?} frame {i}: decoder output == filtered recon"
            );
        }
        recons
    }

    /// Filtered low-delay GOPs (deblocking, SAO, and both — P and B
    /// slices, multiple QPs) stay bit-exact through the crate's own
    /// decoder, with the filtered pictures serving as the references
    /// of every following frame.
    #[test]
    fn filtered_gops_decode_to_encoder_recon_exactly() {
        let planes = scene(64, 64, 4);
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
            LoopFilterCfg::all(),
        ];
        for cfg in cfgs {
            for (qp, b) in [(22i32, false), (32, false), (32, true), (40, true)] {
                assert_filtered_gop_roundtrip(&planes, 64, 64, qp, b, 0, cfg);
            }
        }
        // Non-square geometry.
        let planes = scene(48, 32, 3);
        assert_filtered_gop_roundtrip(&planes, 48, 32, 30, false, 0, LoopFilterCfg::all());
    }

    /// Mid-stream IDR refreshes inside a filtered GOP: the filtered
    /// IDR reconstruction is the reference of the following P frames,
    /// and the whole stream stays bit-exact.
    #[test]
    fn filtered_gop_with_idr_refresh_roundtrips() {
        let planes = scene(48, 48, 5);
        assert_filtered_gop_roundtrip(&planes, 48, 48, 33, true, 3, LoopFilterCfg::all());
    }

    /// The filters actually engage on a real GOP at high QP: the
    /// filtered reconstruction differs from the unfiltered encode and
    /// is never further from the source (the elections are
    /// distortion-driven per frame).
    #[test]
    fn filtered_gop_engages_and_never_hurts() {
        let (w, h, n) = (64usize, 64usize, 4usize);
        let planes = scene(w, h, n);
        let frames = as_frames(&planes);
        let qp = 38;
        let run = |cfg: LoopFilterCfg| -> Vec<FrameRecon> {
            let mut enc = LowDelayPEncoder::new(w, h, qp, 0)
                .expect("encoder")
                .with_loop_filters(cfg);
            frames
                .iter()
                .map(|f| enc.encode_frame(f).expect("encode").recon)
                .collect()
        };
        let plain = run(LoopFilterCfg::off());
        let filtered = run(LoopFilterCfg::all());
        assert!(
            plain
                .iter()
                .zip(filtered.iter())
                .any(|(p, f)| p.y != f.y || p.cb != f.cb || p.cr != f.cr),
            "qp{qp}: the loop filters should engage on some frame"
        );
        // Frame 0 (the IDR) shares its pre-filter reconstruction with
        // the unfiltered run, so the distortion-driven elections can
        // only improve it. (Later frames predict from different
        // references, so only the IDR comparison is deterministic.)
        let ssd = |a: &[u8], b: &[u8]| -> u64 {
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| {
                    let d = i64::from(x) - i64::from(y);
                    (d * d) as u64
                })
                .sum()
        };
        let d = |r: &FrameRecon, t: usize| {
            ssd(&r.y, &planes[t].0) + ssd(&r.cb, &planes[t].1) + ssd(&r.cr, &planes[t].2)
        };
        assert!(
            d(&filtered[0], 0) <= d(&plain[0], 0),
            "IDR: filtered distortion must not exceed unfiltered"
        );
    }

    /// The AMP stream configuration (`MinCbSizeY == 8`, an explicit
    /// `split_cu_flag == 0` per CTB, the Table 9-45 big-CU part_mode
    /// column) roundtrips bit-exactly through the crate's own decoder
    /// on P and B GOPs, with and without the in-loop filters — and the
    /// SPS actually signals the geometry.
    #[test]
    fn amp_geometry_gops_decode_to_encoder_recon_exactly() {
        let planes = scene(64, 64, 4);
        let frames = as_frames(&planes);
        for (qp, b, lf) in [
            (22i32, false, LoopFilterCfg::off()),
            (32, true, LoopFilterCfg::off()),
            (35, true, LoopFilterCfg::all()),
        ] {
            let mut enc = LowDelayPEncoder::new(64, 64, qp, 0)
                .expect("encoder")
                .with_b_slices(b)
                .with_amp(true)
                .with_loop_filters(lf);
            let mut stream = Vec::new();
            let mut recons = Vec::new();
            for f in &frames {
                let out = enc.encode_frame(f).expect("encode");
                stream.extend_from_slice(&out.au);
                recons.push(out.recon);
            }
            // The SPS carries the AMP geometry.
            let units = crate::nal::collect_nal_units(&stream).expect("walk");
            let sps = crate::sps::SeqParameterSet::parse(&units[1].rbsp).expect("sps");
            assert!(sps.amp_enabled_flag, "qp{qp} b={b}");
            assert_eq!(sps.log2_min_luma_coding_block_size_minus3, 0);
            assert_eq!(sps.log2_diff_max_min_luma_coding_block_size, 1);
            let decoded = decode_annexb_sequence(&stream).expect("decode");
            assert_eq!(decoded.len(), 4, "qp{qp} b={b}");
            for (i, (dec, rec)) in decoded.iter().zip(recons.iter()).enumerate() {
                let mut expect = rec.y.clone();
                expect.extend_from_slice(&rec.cb);
                expect.extend_from_slice(&rec.cr);
                assert_eq!(
                    dec.picture.to_planar_u8().expect("8-bit"),
                    expect,
                    "qp{qp} b={b} frame {i}: decoder output == encoder recon"
                );
            }
        }
    }

    /// A scene change inside an AMP-geometry GOP drives the intra
    /// fallback — whose CU syntax above MinCb carries NO part_mode
    /// (§7.3.8.5) — and the stream still decodes bit-exactly.
    #[test]
    fn amp_geometry_intra_fallback_roundtrips() {
        let (w, h) = (64usize, 64usize);
        let mut planes = scene(w, h, 2);
        let y2: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, yy) = (i % w, i / w);
                (((x * 13) ^ (yy * 7)) % 251) as u8
            })
            .collect();
        planes.push((y2, vec![60u8; w * h / 4], vec![200u8; w * h / 4]));
        let frames = as_frames(&planes);
        let mut enc = LowDelayPEncoder::new(w, h, 27, 0)
            .expect("encoder")
            .with_amp(true);
        let mut stream = Vec::new();
        let mut recons = Vec::new();
        let mut intra_total = 0usize;
        for f in &frames {
            let out = enc.encode_frame(f).expect("encode");
            stream.extend_from_slice(&out.au);
            recons.push(out.recon);
            if !out.keyframe {
                intra_total += out.stats.intra;
            }
        }
        assert!(
            intra_total > 0,
            "the scene change should elect intra CUs in the AMP-geometry P slice"
        );
        let decoded = decode_annexb_sequence(&stream).expect("decode");
        assert_eq!(decoded.len(), 3);
        for (i, (dec, rec)) in decoded.iter().zip(recons.iter()).enumerate() {
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

    /// Banded scenes whose motion boundaries sit at QUARTER offsets
    /// inside the CTB rows / columns (where no symmetric split
    /// aligns): the AMP shapes are actually elected, signalled on the
    /// wire, and the streams decode bit-exactly — P and B, plain and
    /// filtered.
    #[test]
    fn amp_shapes_are_elected_and_roundtrip() {
        let (w, h) = (64usize, 64usize);
        // Horizontal bands split at y = 20 (CTB row 1, local offset 4
        // ⇒ PART_2NxnU) and y = 44 (row 2, local offset 12 ⇒
        // PART_2NxnD); `transpose` yields the vertical analogues.
        let banded = |transpose: bool| -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
            (0..4usize)
                .map(|t| {
                    let y: Vec<u8> = (0..w * h)
                        .map(|i| {
                            let (mut x, mut yy) = (i % w, i / w);
                            if transpose {
                                core::mem::swap(&mut x, &mut yy);
                            }
                            let (shift, phase) = if yy < 20 {
                                (2 * t, 0usize)
                            } else if yy < 44 {
                                (w - 2 * t, 60)
                            } else {
                                (2 * t + 1, 120)
                            };
                            (((x + shift) * 7 + yy * 3 + phase) % 240) as u8
                        })
                        .collect();
                    (y, vec![110u8; w * h / 4], vec![140u8; w * h / 4])
                })
                .collect()
        };
        for (transpose, b, lf) in [
            (false, false, LoopFilterCfg::off()),
            (true, false, LoopFilterCfg::off()),
            (false, true, LoopFilterCfg::all()),
        ] {
            let planes = banded(transpose);
            let frames = as_frames(&planes);
            let mut enc = LowDelayPEncoder::new(w, h, 30, 0)
                .expect("encoder")
                .with_b_slices(b)
                .with_amp(true)
                .with_loop_filters(lf);
            let mut stream = Vec::new();
            let mut recons = Vec::new();
            let mut amp_total = 0usize;
            for f in &frames {
                let out = enc.encode_frame(f).expect("encode");
                stream.extend_from_slice(&out.au);
                recons.push(out.recon);
                amp_total += out.stats.amp;
            }
            assert!(
                amp_total > 0,
                "transpose={transpose} b={b}: quarter-offset motion boundaries \
                 should elect AMP shapes"
            );
            // (The bit-exact roundtrip below IS the wire-level truth:
            // a part_mode mismatch would desynchronize CABAC.)
            let decoded = decode_annexb_sequence(&stream).expect("decode");
            assert_eq!(decoded.len(), 4, "transpose={transpose} b={b}");
            for (i, (dec, rec)) in decoded.iter().zip(recons.iter()).enumerate() {
                let mut expect = rec.y.clone();
                expect.extend_from_slice(&rec.cb);
                expect.extend_from_slice(&rec.cr);
                assert_eq!(
                    dec.picture.to_planar_u8().expect("8-bit"),
                    expect,
                    "transpose={transpose} b={b} frame {i}"
                );
            }
        }
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
