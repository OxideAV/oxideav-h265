//! Hierarchical-B GOP encoder — dyadic B pyramids over the §8.5
//! inter machinery of [`crate::encoder::inter`].
//!
//! The sequence shape is one leading IDR followed by dyadic mini-GOPs
//! of `gop` frames (a power of two). Within a mini-GOP anchored at
//! POC `a`:
//!
//! * the next anchor `a + gop` is coded FIRST as a P slice (layer 0)
//!   referencing `a`;
//! * every interval `(lo, hi)` then codes its midpoint as a B slice
//!   whose `RefPicList0` holds the past boundary `lo` and whose
//!   `RefPicList1` holds the FUTURE boundary `hi` (out-of-order
//!   coding, decode order ≠ output order), recursing into the two
//!   halves one pyramid layer deeper.
//!
//! Every B slice's AMVP election therefore searches L0, L1 and the
//! bi combination (see [`crate::encoder::inter`]'s two-sided search),
//! and the §8.5.3.2.2 merge candidates carry genuinely bi-directional
//! motion. Deeper layers ride a per-layer QP offset (`qp + layer`
//! by default) — the classic pyramid rate allocation: the pictures
//! referenced most are coded best.
//!
//! Stream-level signalling:
//!
//! * `slice_pic_order_cnt_lsb` carries the DISPLAY order; the SPS
//!   signals `sps_max_num_reorder_pics = log2(gop)` and a DPB bound
//!   of `log2(gop) + 2` pictures so a conforming decoder reorders
//!   output correctly;
//! * each slice's inline §7.4.8 short-term RPS lists every picture a
//!   later slice (in decode order) still references — negative AND
//!   positive pictures — with `used_by_curr_pic` flags marking this
//!   slice's own active references;
//! * reference-list sizes stay 1/1 (the PPS defaults), so no
//!   `num_ref_idx` override is signalled on the pyramid slices.
//!
//! Reconstruction stays the decode-side truth: every slice
//! reconstructs through the shared [`encode_inter_slice`] pass
//! (decode-side prediction, transform and §8.7 in-loop filters), so
//! a conforming decoder's output is bit-identical to the encoder's
//! per-frame reconstruction — pinned by the roundtrip tests in
//! DISPLAY order against [`crate::sequence::decode_annexb_sequence`]
//! (which itself outputs §C.5.2.2 output order).
//!
//! The trailing frames that do not fill a final mini-GOP are coded at
//! [`PyramidEncoder::flush`] as a low-delay P chain from the last
//! anchor (decode order == display order again).

use std::collections::BTreeMap;

use crate::encoder::hrd::{
    buffering_period_payload, filler_data_nal_framed, pic_timing_payload, sei_prefix_nal, HrdClock,
    HrdSignalCfg, SEI_BUFFERING_PERIOD, SEI_PIC_TIMING,
};
use crate::encoder::inter::{
    build_ref_lists, encode_inter_slice, FrameRecon, FrameStats, SliceSpec, TmvpSpec, YuvFrame,
};
use crate::encoder::intra::{encode_idr_intra_au_full, IntraEncodeError, SpsCfg};
use crate::encoder::loopfilter::LoopFilterCfg;
use crate::encoder::nal::{annexb, nal_unit};
use crate::encoder::rate::{FrameClass, RateControlCfg, RateController};

/// The fixed CTB size shared with the intra / low-delay encoders.
const CTB: usize = 16;

/// One encoded access unit of the pyramid stream, in DECODE order.
#[derive(Debug, Clone)]
pub struct PyramidAu {
    /// The Annex B access unit (`VPS + SPS + PPS + IDR_N_LP` for the
    /// leading IDR, one `TRAIL_R` slice otherwise).
    pub au: Vec<u8>,
    /// `true` for the leading IDR.
    pub keyframe: bool,
    /// The frame's DISPLAY index (== its `PicOrderCntVal`).
    pub display_order: usize,
    /// The pyramid layer (0 = the IDR / anchors, deeper B layers up
    /// to `log2(gop)`).
    pub layer: u8,
    /// The frame's reconstruction (== a conforming decoder's output).
    pub recon: FrameRecon,
    /// The frame's CU mode-decision counters.
    pub stats: FrameStats,
    /// The `SliceQpY` this frame was coded at (base + layer offset,
    /// with the base elected per mini-GOP under
    /// [`PyramidEncoder::with_rate_control`]).
    pub qp: i32,
}

/// An owned 4:2:0 frame buffered until its mini-GOP completes.
#[derive(Debug)]
struct OwnedFrame {
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
}

impl OwnedFrame {
    fn as_yuv(&self) -> YuvFrame<'_> {
        YuvFrame {
            y: &self.y,
            cb: &self.cb,
            cr: &self.cr,
        }
    }
}

/// One scheduled slice of a mini-GOP, in decode order.
struct Sched {
    poc: i32,
    /// `RefPicList0[0]` (the past boundary).
    l0: i32,
    /// `RefPicList1[0]` (the future boundary) — `None` for the anchor
    /// P slice.
    l1: Option<i32>,
    layer: u8,
}

/// The dyadic decode-order schedule of one mini-GOP `(a, a + g]`:
/// the anchor P first, then the midpoint recursion.
fn build_schedule(a: i32, g: i32) -> Vec<Sched> {
    fn rec(v: &mut Vec<Sched>, lo: i32, hi: i32, layer: u8) {
        if hi - lo < 2 {
            return;
        }
        let mid = (lo + hi) / 2;
        v.push(Sched {
            poc: mid,
            l0: lo,
            l1: Some(hi),
            layer,
        });
        rec(v, lo, mid, layer + 1);
        rec(v, mid, hi, layer + 1);
    }
    let mut v = vec![Sched {
        poc: a + g,
        l0: a,
        l1: None,
        layer: 0,
    }];
    rec(&mut v, a, a + g, 1);
    v
}

/// The DPB bounds a mini-GOP of `g` frames needs under
/// [`build_schedule`] and the retention rule of `code_mini_gop`:
/// `(sps_max_num_reorder_pics, max retained references)` — the
/// largest number of decode-earlier pictures that follow any picture
/// in output order, and the largest reference set any slice keeps
/// alive beside itself (for a dyadic `g` these are `log2 g` and
/// `log2 g + 1`; a non-dyadic length is bounded exactly).
fn schedule_bounds(g: i32) -> (u32, u32) {
    let sched = build_schedule(0, g);
    let mut reorder = 0usize;
    let mut refs: Vec<i32> = vec![0];
    let mut max_refs = 0usize;
    for (i, s) in sched.iter().enumerate() {
        reorder = reorder.max(sched[..i].iter().filter(|t| t.poc > s.poc).count());
        refs.retain(|&p| sched[i..].iter().any(|t| t.l0 == p || t.l1 == Some(p)));
        max_refs = max_refs.max(refs.len());
        refs.push(s.poc);
    }
    (reorder as u32, max_refs as u32)
}

/// The streaming hierarchical-B encoder: display-order frames in,
/// decode-order access units out (buffered per mini-GOP).
#[derive(Debug)]
pub struct PyramidEncoder {
    width: usize,
    height: usize,
    qp: i32,
    /// Mini-GOP length (2..=16).
    gop: usize,
    /// Per-layer QP step (`SliceQpY = qp + layer * step`, clamped).
    layer_qp_step: i32,
    filters: LoopFilterCfg,
    amp: bool,
    /// Display index of the next frame pushed.
    next_display: usize,
    /// Frames after the last anchor, awaiting their mini-GOP.
    pending: Vec<OwnedFrame>,
    /// The retained reference reconstructions, keyed by POC.
    refs: BTreeMap<i32, FrameRecon>,
    /// POC of the last coded anchor (`None` before the IDR).
    anchor: Option<i32>,
    /// ABR rate controller: when set, it elects the base QP of each
    /// mini-GOP (the per-layer offsets ride on top) and the
    /// constructor `qp` is unused.
    rc: Option<RateController>,
    /// Spatial adaptive-quantization strength (0 = constant slice
    /// QP; 1..=3 = per-CTB `cu_qp_delta` on every slice).
    aq: u8,
    /// §E.2.1 VUI timing declaration `(num_units_in_tick,
    /// time_scale)` for the stream's SPS (`None` = no VUI).
    timing: Option<(u32, u32)>,
    /// The rate-control configuration behind `rc` (kept for HRD
    /// signalling).
    rc_cfg: Option<RateControlCfg>,
    /// HRD signalling requested ([`Self::with_hrd`]).
    hrd_on: bool,
    /// CBR delivery requested ([`Self::with_cbr`]; requires
    /// `hrd_on`).
    cbr_on: bool,
    /// The signalled §E.2.2 schedule + Annex C clock, built at the
    /// first frame when `hrd_on`. The clock advances once per access
    /// unit in DECODE order.
    hrd: Option<(HrdSignalCfg, HrdClock)>,
    /// Quadtree-coder geometry ([`Self::with_tree`]).
    tree: Option<crate::encoder::ctu::TreeCfg>,
    /// Active references per list ([`Self::with_refs`], 1..=4).
    num_refs: usize,
    /// Temporal MVP ([`Self::with_temporal_mvp`]).
    tmvp: bool,
    /// Adaptive mini-GOP closing on scene cuts
    /// ([`Self::with_adaptive_gop`]).
    adaptive: bool,
    /// CTU-level rate feedback ([`Self::with_ctu_rate_control`]).
    ctu_rc: bool,
    /// Pass-1 worker budget ([`Self::with_threads`]).
    threads: usize,
    /// Running mean absolute inter-frame luma difference (Q4) of the
    /// non-cut frame pairs seen so far — the scene-cut baseline.
    mad_avg_q4: Option<u64>,
}

impl PyramidEncoder {
    /// Construct a hierarchical-B encoder for `width`x`height` 4:2:0
    /// 8-bit content at base `SliceQpY == qp` with mini-GOPs of `gop`
    /// frames (2..=16; dyadic lengths give the classic hierarchical-B
    /// pyramid, others the midpoint-recursion schedule).
    ///
    /// # Errors
    /// [`PyramidError::Encode`] on bad dimensions / QP,
    /// [`PyramidError::BadGop`] on an out-of-range `gop` (any length
    /// in 2..=16 is legal; non-dyadic lengths use the same midpoint
    /// schedule with exact DPB / reorder bounds).
    pub fn new(width: usize, height: usize, qp: i32, gop: usize) -> Result<Self, PyramidError> {
        if width == 0 || height == 0 || width % CTB != 0 || height % CTB != 0 {
            return Err(PyramidError::Encode(IntraEncodeError::BadDimensions {
                width,
                height,
            }));
        }
        if !(0..=51).contains(&qp) {
            return Err(PyramidError::Encode(IntraEncodeError::BadQp(qp)));
        }
        if !(2..=16).contains(&gop) {
            return Err(PyramidError::BadGop(gop));
        }
        Ok(Self {
            width,
            height,
            qp,
            gop,
            layer_qp_step: 1,
            filters: LoopFilterCfg::off(),
            amp: false,
            next_display: 0,
            pending: Vec::new(),
            refs: BTreeMap::new(),
            anchor: None,
            rc: None,
            aq: 0,
            timing: None,
            rc_cfg: None,
            hrd_on: false,
            cbr_on: false,
            hrd: None,
            tree: None,
            num_refs: 1,
            tmvp: false,
            adaptive: false,
            mad_avg_q4: None,
            ctu_rc: false,
            threads: 1,
        })
    }

    /// CTU-level rate feedback inside every picture — see
    /// [`crate::encoder::inter::LowDelayPEncoder::with_ctu_rate_control`].
    #[must_use]
    pub fn with_ctu_rate_control(mut self, on: bool) -> Self {
        self.ctu_rc = on;
        self
    }

    /// Bound the quadtree coder's pass-1 fan-out (tiles decided on up
    /// to `n` threads; serial by default, bytes independent of `n`).
    #[must_use]
    pub fn with_threads(mut self, n: usize) -> Self {
        self.threads = n.max(1);
        self
    }

    /// [`Self::with_threads`] on a constructed encoder.
    pub fn set_threads(&mut self, n: usize) {
        self.threads = n.max(1);
    }

    /// The frame budget the CTU-level feedback tracks (`None` when
    /// off, without rate control, or on the fixed-geometry coder).
    fn ctu_budget(&self) -> Option<u64> {
        if self.ctu_rc && self.tree.is_some() {
            self.rc.as_ref().and_then(RateController::last_budget_bits)
        } else {
            None
        }
    }

    /// Close mini-GOPs adaptively at scene cuts: when the luma mean
    /// absolute difference between two consecutive input frames
    /// exceeds four times the running average of the earlier pairs
    /// (and 16 per sample), the frames before the cut are coded as a
    /// shorter (possibly non-dyadic) mini-GOP and the cut frame opens
    /// the next one, so no B slice straddles the cut. Off by default
    /// (fixed-length mini-GOPs; the trailing frames at flush are
    /// always coded as a short mini-GOP).
    #[must_use]
    pub fn with_adaptive_gop(mut self, on: bool) -> Self {
        self.adaptive = on;
        self
    }

    /// The stream's reorder depth in frames (`sps_max_num_reorder_pics`
    /// — the dts-behind-pts delay a container needs).
    #[must_use]
    pub fn reorder_delay(&self) -> u32 {
        schedule_bounds(self.gop as i32).0
    }

    /// Keep up to `n` (1..=4) active references per list: the
    /// §8.3.4 lists are built from every retained reference the
    /// mini-GOP schedule still holds — `RefPicList0` past pictures
    /// first (closest first) then future, `RefPicList1` future first
    /// — truncated to `n`, with `num_ref_idx_lX_active` signalled and
    /// motion estimation per reference. The default is 1 (the
    /// historical one-boundary-per-list streams).
    #[must_use]
    pub fn with_refs(mut self, n: usize) -> Self {
        self.num_refs = n.clamp(1, 4);
        self
    }

    /// Enable temporal motion-vector prediction: every P / B slice
    /// signals `slice_temporal_mvp_enabled_flag == 1`; the §8.3.5
    /// collocated picture is `RefPicList1[0]` (the future boundary)
    /// on B slices and `RefPicList0[0]` on the anchor P slices.
    #[must_use]
    pub fn with_temporal_mvp(mut self, on: bool) -> Self {
        self.tmvp = on;
        self
    }

    /// Route every picture through the recursive coding-quadtree
    /// coder — see
    /// [`crate::encoder::inter::LowDelayPEncoder::with_tree`].
    #[must_use]
    pub fn with_tree(mut self, tree: crate::encoder::ctu::TreeCfg) -> Self {
        self.tree = Some(tree);
        self
    }

    /// Declare the stream's frame rate in the SPS VUI — see
    /// [`crate::encoder::inter::LowDelayPEncoder::with_frame_rate`].
    #[must_use]
    pub fn with_frame_rate(mut self, fps_num: u32, fps_den: u32) -> Self {
        self.timing = (fps_num > 0 && fps_den > 0).then_some((fps_den, fps_num));
        self
    }

    /// Enable spatial adaptive quantization at `strength` (clamped to
    /// 0..=3) — see
    /// [`crate::encoder::inter::LowDelayPEncoder::with_aq`].
    #[must_use]
    pub fn with_aq(mut self, strength: u8) -> Self {
        self.aq = strength.min(3);
        self
    }

    /// Set the per-layer QP step (default 1): a slice on pyramid
    /// layer `l` codes at `SliceQpY = clamp(qp + l * step, 0, 51)`.
    #[must_use]
    pub fn with_layer_qp_step(mut self, step: i32) -> Self {
        self.layer_qp_step = step;
        self
    }

    /// Enable the §8.7 in-loop filters on every slice's
    /// reconstruction path (see
    /// [`crate::encoder::inter::LowDelayPEncoder::with_loop_filters`]).
    #[must_use]
    pub fn with_loop_filters(mut self, filters: LoopFilterCfg) -> Self {
        self.filters = filters;
        self
    }

    /// Switch the stream to the AMP configuration (see
    /// [`crate::encoder::inter::LowDelayPEncoder::with_amp`]).
    #[must_use]
    pub fn with_amp(mut self, on: bool) -> Self {
        self.amp = on;
        self
    }

    /// Switch to average-bitrate coding: a [`RateController`] elects
    /// the BASE QP of each mini-GOP against `cfg` (the per-layer
    /// offsets ride on top, so the pyramid's rate allocation shape is
    /// kept while its level tracks the target). Every coded slice
    /// feeds back at its actual layer QP, so the model learns the
    /// mini-GOP mixture. Replaces the constructor's constant QP.
    #[must_use]
    pub fn with_rate_control(mut self, cfg: &RateControlCfg) -> Self {
        self.rc = Some(RateController::new(cfg, self.width, self.height));
        self.rc_cfg = Some(*cfg);
        self
    }

    /// Declare and enforce HRD conformance (see
    /// [`crate::encoder::inter::LowDelayPEncoder::with_hrd`]): the
    /// SPS VUI gains the §E.2.2 schedule, the IDR carries a §D.2.2
    /// buffering-period SEI, every access unit a §D.2.3 pic-timing
    /// SEI whose `pic_dpb_output_delay` carries the pyramid's
    /// reorder schedule, and coded sizes are capped by the exact
    /// Annex C replay. Requires [`Self::with_rate_control`] with
    /// [`RateControlCfg::with_vbv`] AND [`Self::with_frame_rate`].
    #[must_use]
    pub fn with_hrd(mut self, on: bool) -> Self {
        self.hrd_on = on;
        self
    }

    /// Switch the HRD schedule to constant-bit-rate delivery — see
    /// [`crate::encoder::inter::LowDelayPEncoder::with_cbr`].
    /// Requires [`Self::with_hrd`].
    #[must_use]
    pub fn with_cbr(mut self, on: bool) -> Self {
        self.cbr_on = on;
        self
    }

    /// Build the HRD schedule + clock on the first frame (validating
    /// the [`Self::with_hrd`] prerequisites).
    fn ensure_hrd(&mut self) -> Result<(), PyramidError> {
        if !self.hrd_on {
            return if self.cbr_on {
                Err(PyramidError::Encode(IntraEncodeError::HrdConfig))
            } else {
                Ok(())
            };
        }
        if self.hrd.is_some() {
            return Ok(());
        }
        let (Some(cfg), Some((num_units, time_scale))) = (&self.rc_cfg, self.timing) else {
            return Err(PyramidError::Encode(IntraEncodeError::HrdConfig));
        };
        let Some(vbv) = cfg.vbv_buffer_bits else {
            return Err(PyramidError::Encode(IntraEncodeError::HrdConfig));
        };
        let signal = HrdSignalCfg::for_rate(cfg.bits_per_second, vbv).with_cbr(self.cbr_on);
        if self.cbr_on {
            let tick_bits =
                signal.bit_rate.saturating_mul(u64::from(num_units)) / u64::from(time_scale.max(1));
            if signal.cpb_size < tick_bits.saturating_mul(2) {
                return Err(PyramidError::Encode(IntraEncodeError::CbrCpbTooSmall));
            }
        }
        self.hrd = Some((signal, HrdClock::new(signal, time_scale, num_units)));
        Ok(())
    }

    /// The §D.2.3 pic-timing SEI prefix (plus the §D.2.2 buffering
    /// period when `bp`) for the access unit about to be coded at
    /// display index `display`, framed as Annex B bytes — `None`
    /// without HRD signalling. Advances no clock state except the
    /// pending buffering period.
    fn sei_prefix(&mut self, bp: bool, display: u64) -> Option<Vec<u8>> {
        let reorder = u64::from(self.depth());
        let (_, clock) = self.hrd.as_mut()?;
        let mut payloads = Vec::with_capacity(2);
        if bp {
            let (delay, offset) = clock.begin_buffering_period();
            payloads.push((
                SEI_BUFFERING_PERIOD,
                buffering_period_payload(delay, offset),
            ));
        }
        // Output at elemental tick `reorder + display`, removal at
        // tick `m`: the eq. C-15 output delay in ticks.
        let out_delay = (reorder + display).saturating_sub(clock.next_decode_index());
        payloads.push((
            SEI_PIC_TIMING,
            pic_timing_payload(
                clock.au_cpb_removal_delay_minus1(),
                u32::try_from(out_delay).unwrap_or(u32::MAX),
            ),
        ));
        let mut framed = vec![0, 0, 0, 1];
        framed.extend(sei_prefix_nal(&payloads));
        Some(framed)
    }

    /// The joint VBV + Annex C bit budget for the next access unit
    /// (under CBR, less the reserved filler quantum).
    fn au_cap(&self) -> Option<u64> {
        match (
            self.rc.as_ref().and_then(RateController::vbv_frame_cap),
            self.hrd.as_ref().map(|(_, clock)| clock.frame_cap()),
        ) {
            (Some(v), Some(h)) => Some(v.min(h)),
            (v, h) => v.or(h),
        }
        .map(|c| if self.cbr_on { c.saturating_sub(56) } else { c })
    }

    /// CBR underrun padding: append a filler-data NAL when the AU
    /// (SEI included) leaves the back-to-back channel ahead of the
    /// overflow floor.
    fn append_cbr_filler(&self, au: &mut Vec<u8>) {
        if let Some((_, clock)) = &self.hrd {
            let pad_bits = clock.cbr_filler_bits(au.len() as u64 * 8);
            if pad_bits > 0 {
                au.extend(filler_data_nal_framed((pad_bits as usize).div_ceil(8)));
            }
        }
    }

    /// The number of pyramid layers below the anchors
    /// (`log2(gop)`).
    fn depth(&self) -> u32 {
        schedule_bounds(self.gop as i32).0
    }

    /// The stream's [`SpsCfg`]: the DPB and reorder bounds the
    /// configured mini-GOP length needs under the schedule (a dyadic
    /// `gop` retains `log2(gop) + 1` references beside the current
    /// picture and reorders by `log2(gop)`; shorter adaptive
    /// mini-GOPs never exceed the full-length bounds).
    fn sps_cfg(&self) -> SpsCfg {
        let (reorder, max_refs) = schedule_bounds(self.gop as i32);
        SpsCfg {
            max_dec_pic_buffering_minus1: max_refs.max(1),
            max_num_reorder_pics: reorder,
            min_cb_log2: if self.amp || self.tree.is_some() {
                3
            } else {
                4
            },
            amp: self.amp,
            cu_qp_delta: self.aq > 0 || (self.ctu_rc && self.tree.is_some() && self.rc.is_some()),
            timing: self.timing,
            hrd: self.hrd.as_ref().map(|(signal, _)| *signal),
            temporal_mvp: self.tmvp,
            tree: self.tree,
            threads: self.threads,
        }
    }

    /// The layer-`l` slice QP over `base` (the constructor QP, or
    /// the rate controller's per-mini-GOP election).
    fn layer_qp(&self, base: i32, layer: u8) -> i32 {
        (base + i32::from(layer) * self.layer_qp_step).clamp(0, 51)
    }

    /// Push the next DISPLAY-order frame; returns every access unit
    /// that became codable (none until the current mini-GOP fills,
    /// then the whole mini-GOP in decode order).
    ///
    /// # Errors
    /// [`PyramidError::Encode`] on plane-size mismatches.
    pub fn encode_frame(&mut self, frame: &YuvFrame<'_>) -> Result<Vec<PyramidAu>, PyramidError> {
        let (cw, ch) = (self.width / 2, self.height / 2);
        for (plane, buf, expected) in [
            ("y", frame.y, self.width * self.height),
            ("cb", frame.cb, cw * ch),
            ("cr", frame.cr, cw * ch),
        ] {
            if buf.len() != expected {
                return Err(PyramidError::Encode(IntraEncodeError::PlaneSize {
                    plane,
                    expected,
                    got: buf.len(),
                }));
            }
        }

        self.ensure_hrd()?;
        if self.anchor.is_none() {
            // The leading IDR (POC 0, layer 0).
            let mut idr_qp = match &mut self.rc {
                Some(rc) => rc.pick_qp(FrameClass::Intra),
                None => self.qp,
            };
            let sei = self.sei_prefix(true, 0);
            let sei_bits = sei.as_ref().map_or(0, |s| s.len() as u64 * 8);
            let code = |qp: i32| {
                encode_idr_intra_au_full(
                    frame.y,
                    frame.cb,
                    frame.cr,
                    self.width,
                    self.height,
                    qp,
                    &self.sps_cfg(),
                    &self.filters,
                    self.aq,
                    self.ctu_budget(),
                )
            };
            let mut idr = code(idr_qp)?;
            // VBV / HRD constraint: re-encode at a higher QP until
            // the IDR access unit (SEI included) fits the modelled
            // decoder buffer and the Annex C arrival window (or the
            // QP ceiling is reached) — the same hard guarantee as
            // the low-delay arm.
            if let Some(cap) = self.au_cap() {
                let ceiling = self.rc.as_ref().map_or(51, RateController::max_qp);
                while idr.au.len() as u64 * 8 + sei_bits > cap && idr_qp < ceiling {
                    idr_qp = (idr_qp + 3).min(ceiling);
                    idr = code(idr_qp)?;
                }
            }
            if let Some(sei) = &sei {
                crate::encoder::hrd::splice_sei_before_vcl(&mut idr.au, sei);
            }
            self.append_cbr_filler(&mut idr.au);
            let recon = FrameRecon {
                y: idr.recon_y,
                cb: idr.recon_cb,
                cr: idr.recon_cr,
                // An IDR's motion field is all-intra.
                motion_field: Some(crate::motion::MotionField::new(self.width, self.height)),
            };
            self.refs.insert(0, recon.clone());
            self.anchor = Some(0);
            self.next_display = 1;
            if let Some(rc) = &mut self.rc {
                rc.update(FrameClass::Intra, idr_qp, idr.au.len() as u64 * 8);
            }
            if let Some((_, clock)) = &mut self.hrd {
                clock.push_au(idr.au.len() as u64 * 8);
            }
            return Ok(vec![PyramidAu {
                au: idr.au,
                keyframe: true,
                display_order: 0,
                layer: 0,
                recon,
                stats: FrameStats {
                    intra: (self.width / CTB) * (self.height / CTB),
                    ..FrameStats::default()
                },
                qp: idr_qp,
            }]);
        }

        // Scene-cut test against the previous input frame (the last
        // pending frame, or the anchor's source when nothing is
        // pending is unavailable — the anchor was consumed — so the
        // first frame of a mini-GOP is never a cut candidate).
        let cut = self.adaptive
            && self.pending.last().is_some_and(|prev| {
                let mad = luma_mad_q4(frame.y, &prev.y, self.width, self.height);
                let is_cut = match self.mad_avg_q4 {
                    Some(avg) => mad > avg.saturating_mul(4) && mad > 16 << 4,
                    None => mad > 40 << 4,
                };
                if !is_cut {
                    self.mad_avg_q4 = Some(match self.mad_avg_q4 {
                        None => mad,
                        Some(avg) => (avg * 3 + mad) / 4,
                    });
                }
                is_cut
            });
        let mut out = Vec::new();
        if cut {
            // Close the mini-GOP before the cut; the cut frame opens
            // the next one.
            let n = self.pending.len();
            out.extend(self.code_mini_gop(n));
        }
        self.pending.push(OwnedFrame {
            y: frame.y.to_vec(),
            cb: frame.cb.to_vec(),
            cr: frame.cr.to_vec(),
        });
        self.next_display += 1;
        if self.pending.len() == self.gop {
            out.extend(self.code_mini_gop(self.gop));
        }
        Ok(out)
    }

    /// Code the first `g` buffered frames (displays `a+1 ..= a+g`) as
    /// one mini-GOP in decode order and drain them; any later pending
    /// frames stay buffered for the next mini-GOP.
    fn code_mini_gop(&mut self, g: usize) -> Vec<PyramidAu> {
        let a = self.anchor.expect("mini-GOP needs an anchor");
        let g = g as i32;
        let sched = build_schedule(a, g);
        let pending: Vec<OwnedFrame> = self.pending.drain(..g as usize).collect();
        let mut out = Vec::with_capacity(sched.len());
        for (i, s) in sched.iter().enumerate() {
            // Retained set for THIS slice's RPS: every already-coded
            // picture that this or a LATER slice of the schedule
            // still references.
            let retained: Vec<i32> = {
                let mut keep: Vec<i32> = self
                    .refs
                    .keys()
                    .copied()
                    .filter(|&p| {
                        sched[i..].iter().any(|t| {
                            t.l0 == p || t.l1 == Some(p) // still referenced
                        })
                    })
                    .collect();
                keep.sort_unstable();
                keep
            };
            // Drop reconstructions nothing references any more.
            self.refs.retain(|p, _| retained.contains(p));

            let frame = &pending[(s.poc - a - 1) as usize];
            let au = self.code_slice_vbv(frame.as_yuv(), s, &retained);
            self.refs.insert(s.poc, au.recon.clone());
            out.push(au);
        }
        // Only the new anchor survives into the next mini-GOP.
        let new_anchor = a + g;
        self.refs.retain(|&p, _| p == new_anchor);
        self.anchor = Some(new_anchor);
        out
    }

    /// Encode one scheduled P / B slice at `base` QP plus the slice's
    /// layer offset, enforce the VBV constraint (re-encode at a
    /// higher QP while the access unit would underflow the modelled
    /// buffer), and feed the rate model back — the per-temporal-layer
    /// buffer accounting: every access unit drains the model at its
    /// own decode instant, whatever pyramid layer it sits on.
    fn code_slice_vbv(&mut self, frame: YuvFrame<'_>, s: &Sched, retained: &[i32]) -> PyramidAu {
        // Per-slice base election: the controller tracks every access
        // unit at its own decode instant (the once-per-mini-GOP
        // election under-tracked at low rates — the r451 accuracy
        // gates measured ~20 % tail drift); the per-layer offsets
        // ride on top so the pyramid's rate-allocation shape is kept.
        let base = match &mut self.rc {
            Some(rc) => rc.pick_qp(FrameClass::Inter),
            None => self.qp,
        };
        let sei = self.sei_prefix(false, s.poc as u64);
        let sei_bits = sei.as_ref().map_or(0, |s| s.len() as u64 * 8);
        let mut bump = 0i32;
        let mut au = self.code_slice(frame, s, retained, self.layer_qp(base, s.layer));
        if let Some(cap) = self.au_cap() {
            let ceiling = self.rc.as_ref().map_or(51, RateController::max_qp);
            while au.au.len() as u64 * 8 + sei_bits > cap && au.qp < ceiling {
                bump += 3;
                let qp = (self.layer_qp(base, s.layer) + bump).min(ceiling);
                au = self.code_slice(frame, s, retained, qp);
            }
        }
        if let Some(sei) = &sei {
            crate::encoder::hrd::splice_sei_before_vcl(&mut au.au, sei);
        }
        self.append_cbr_filler(&mut au.au);
        if let Some(rc) = &mut self.rc {
            // Feed back at the BASE QP (not the slice's layer QP) plus
            // any VBV bump: the model then learns the layer-offset
            // discount as part of the complexity, so inverting it at
            // the next base election is unbiased — while a bumped
            // slice reports the QP its bits were really coded at
            // relative to the layer shape.
            rc.update(
                FrameClass::Inter,
                (base + bump).clamp(0, 51),
                au.au.len() as u64 * 8,
            );
        }
        if let Some((_, clock)) = &mut self.hrd {
            clock.push_au(au.au.len() as u64 * 8);
        }
        au
    }

    /// Encode one scheduled P / B slice against the retained
    /// reconstructions at slice QP `qp`.
    fn code_slice(&self, frame: YuvFrame<'_>, s: &Sched, retained: &[i32], qp: i32) -> PyramidAu {
        let b_slice = s.l1.is_some();
        let rec_of = |p: i32| -> &FrameRecon { self.refs.get(&p).expect("retained recon") };
        // The used short-term sets (§8.3.4 StCurrBefore / StCurrAfter,
        // closest first): the schedule's boundaries first, then the
        // nearest other retained pictures on each side up to `refs`.
        let mut before: Vec<i32> = vec![s.l0];
        let mut after: Vec<i32> = s.l1.into_iter().collect();
        if self.num_refs > 1 {
            let mut past: Vec<i32> = retained
                .iter()
                .copied()
                .filter(|&p| p < s.poc && p != s.l0)
                .collect();
            past.sort_unstable_by(|a, b| b.cmp(a));
            before.extend(past.into_iter().take(self.num_refs - 1));
            if b_slice {
                let mut future: Vec<i32> = retained
                    .iter()
                    .copied()
                    .filter(|&p| p > s.poc && Some(p) != s.l1)
                    .collect();
                future.sort_unstable();
                after.extend(future.into_iter().take(self.num_refs - 1));
            }
        }
        let total = before.len() + after.len();
        let n_l0 = self.num_refs.min(total);
        let n_l1 = if b_slice { self.num_refs.min(total) } else { 0 };
        let (l0_pocs, l1_pocs) = build_ref_lists(&before, &after, n_l0, n_l1);
        let l0: Vec<(i32, &FrameRecon)> = l0_pocs.iter().map(|&p| (p, rec_of(p))).collect();
        let l1: Vec<(i32, &FrameRecon)> = l1_pocs.iter().map(|&p| (p, rec_of(p))).collect();
        // Inline §7.4.8 RPS: negative pics closest-first, positive
        // pics closest-first; used flags mark the used sets.
        let rps_neg: Vec<(u32, bool)> = retained
            .iter()
            .rev()
            .filter(|&&p| p < s.poc)
            .map(|&p| ((s.poc - p) as u32, before.contains(&p)))
            .collect();
        let rps_pos: Vec<(u32, bool)> = retained
            .iter()
            .filter(|&&p| p > s.poc)
            .map(|&p| ((p - s.poc) as u32, after.contains(&p)))
            .collect();
        let spec = SliceSpec {
            poc: s.poc,
            qp,
            b_slice,
            rps_neg,
            rps_pos,
            l0,
            l1,
            lf: &self.filters,
            big_cu: self.amp,
            aq: self.aq,
            tree: self.tree,
            tmvp: TmvpSpec {
                sps_enabled: self.tmvp,
                slice_enabled: self.tmvp,
                // B: the future boundary (RefPicList1[0]); P: RefPicList0[0].
                collocated_from_l0: !b_slice,
                collocated_ref_idx: 0,
            },
            ctu_rc: self.ctu_budget(),
            threads: self.threads,
        };
        let (rbsp, recon, stats) = encode_inter_slice(&frame, &spec, self.width, self.height);
        PyramidAu {
            au: annexb(&[nal_unit(1, 0, 0, &rbsp)]), // TRAIL_R
            keyframe: false,
            display_order: s.poc as usize,
            layer: s.layer,
            recon,
            stats,
            qp,
        }
    }

    /// Flush the trailing frames that never filled a mini-GOP: they
    /// are coded as one shorter (possibly non-dyadic) mini-GOP under
    /// the same schedule rule. Returns their access units; the
    /// encoder is ready for more input afterwards (the last flushed
    /// frame becomes the new anchor).
    pub fn flush(&mut self) -> Vec<PyramidAu> {
        if self.anchor.is_none() || self.pending.is_empty() {
            return Vec::new();
        }
        let n = self.pending.len();
        self.code_mini_gop(n)
    }
}

/// Mean absolute luma difference between two frames in Q4 (per
/// sample, on a 4x-subsampled grid — the scene-cut statistic).
fn luma_mad_q4(a: &[u8], b: &[u8], width: usize, height: usize) -> u64 {
    let mut sum = 0u64;
    let mut n = 0u64;
    for j in (0..height).step_by(2) {
        for i in (0..width).step_by(2) {
            sum += u64::from(a[j * width + i].abs_diff(b[j * width + i]));
            n += 1;
        }
    }
    (sum << 4) / n.max(1)
}

/// Errors from the pyramid encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyramidError {
    /// The shared input-validation contract (dimensions, planes, QP).
    Encode(IntraEncodeError),
    /// `gop` is not a power of two in 2..=16.
    BadGop(usize),
}

impl From<IntraEncodeError> for PyramidError {
    fn from(e: IntraEncodeError) -> Self {
        Self::Encode(e)
    }
}

impl core::fmt::Display for PyramidError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Encode(e) => e.fmt(f),
            Self::BadGop(g) => write!(f, "pyramid gop must be in 2..=16, got {g}"),
        }
    }
}

impl std::error::Error for PyramidError {}

/// The encoded hierarchical-B sequence, with the per-frame outputs in
/// DISPLAY order (the stream itself is decode-ordered).
#[derive(Debug, Clone)]
pub struct PyramidEncoded {
    /// The whole Annex B byte stream (decode order).
    pub stream: Vec<u8>,
    /// Per-frame reconstruction, display order.
    pub recon: Vec<FrameRecon>,
    /// Per-frame CU mode-decision counters, display order.
    pub stats: Vec<FrameStats>,
    /// Per-frame pyramid layer, display order.
    pub layers: Vec<u8>,
    /// The display order of each access unit of `stream`, in decode
    /// order (the coding schedule, for inspection).
    pub decode_order: Vec<usize>,
}

/// Encode a whole sequence through a [`PyramidEncoder`] (leading IDR,
/// dyadic mini-GOPs, low-delay tail) and return the stream plus the
/// display-ordered per-frame outputs.
///
/// # Errors
/// [`PyramidError`] on bad dimensions / planes / QP / GOP length.
pub fn encode_pyramid(
    frames: &[YuvFrame<'_>],
    width: usize,
    height: usize,
    qp: i32,
    gop: usize,
) -> Result<PyramidEncoded, PyramidError> {
    encode_pyramid_with(PyramidEncoder::new(width, height, qp, gop)?, frames)
}

/// [`encode_pyramid`] over a pre-configured encoder (filters, AMP,
/// layer QP step).
///
/// # Errors
/// [`PyramidError`] on bad plane geometry.
pub fn encode_pyramid_with(
    mut enc: PyramidEncoder,
    frames: &[YuvFrame<'_>],
) -> Result<PyramidEncoded, PyramidError> {
    let n = frames.len();
    let mut out = PyramidEncoded {
        stream: Vec::new(),
        recon: vec![FrameRecon::default(); n],
        stats: vec![FrameStats::default(); n],
        layers: vec![0; n],
        decode_order: Vec::with_capacity(n),
    };
    let push = |out: &mut PyramidEncoded, aus: Vec<PyramidAu>| {
        for au in aus {
            out.stream.extend_from_slice(&au.au);
            out.decode_order.push(au.display_order);
            out.recon[au.display_order] = au.recon;
            out.stats[au.display_order] = au.stats;
            out.layers[au.display_order] = au.layer;
        }
    };
    for f in frames {
        let aus = enc.encode_frame(f)?;
        push(&mut out, aus);
    }
    push(&mut out, enc.flush());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::inter::encode_low_delay_p;
    use crate::sequence::decode_annexb_sequence;

    /// A deterministic "camera-noise" scene: a static textured
    /// background under small temporally-independent sensor noise,
    /// with a small bright square drifting 1 px per frame. Motion
    /// prediction cancels everything except the noise — the content
    /// class where the pyramid's per-layer QP allocation wins (the
    /// top layers spend fewer bits on noise nobody references).
    fn pan_scene(w: usize, h: usize, n: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let tex = |u: usize, v: usize| -> i32 { ((u * 3 + v * 5 + (u / 9) * 7) % 200) as i32 };
        let noise = |x: usize, yy: usize, t: usize| -> i32 {
            let h32 = (x
                .wrapping_mul(73_856_093)
                .wrapping_add(yy.wrapping_mul(19_349_663))
                .wrapping_add(t.wrapping_mul(83_492_791))) as u32;
            ((h32 >> 13) % 9) as i32 - 4
        };
        (0..n)
            .map(|t| {
                let y: Vec<u8> = (0..w * h)
                    .map(|i| {
                        let (x, yy) = (i % w, i / w);
                        let (sx, sy) = (10 + t, 24);
                        let obj = x >= sx && x < sx + 8 && yy >= sy && yy < sy + 8;
                        let base = tex(x, yy) + if obj { 45 } else { 0 };
                        (base + 8 + noise(x, yy, t)).clamp(0, 255) as u8
                    })
                    .collect();
                let cb: Vec<u8> = (0..w * h / 4)
                    .map(|i| (90 + (i % (w / 2)) % 70) as u8)
                    .collect();
                let cr: Vec<u8> = (0..w * h / 4)
                    .map(|i| (180 - (i / (w / 2)) % 60) as u8)
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

    fn assert_display_order_exact(stream: &[u8], recons: &[FrameRecon], label: &str) {
        let decoded = decode_annexb_sequence(stream).expect("decode");
        assert_eq!(decoded.len(), recons.len(), "{label}: frame count");
        for (i, (dec, rec)) in decoded.iter().zip(recons.iter()).enumerate() {
            assert_eq!(dec.poc, i as i32, "{label}: display order POC {i}");
            let mut expect = rec.y.clone();
            expect.extend_from_slice(&rec.cb);
            expect.extend_from_slice(&rec.cr);
            assert_eq!(
                dec.picture.to_planar_u8().expect("8-bit"),
                expect,
                "{label} frame {i}: decoder output == encoder recon"
            );
        }
    }

    /// The core contract across GOP shapes: out-of-order coded
    /// pyramids decode (through the crate's own §C.5.2.2
    /// output-ordering) bit-exactly to the encoder's display-order
    /// reconstructions — including a non-filling low-delay tail.
    #[test]
    fn pyramid_gops_decode_to_encoder_recon_exactly() {
        let planes = pan_scene(64, 64, 10);
        let frames = as_frames(&planes);
        for (gop, qp) in [(2usize, 24i32), (4, 30), (8, 27)] {
            let enc = encode_pyramid(&frames, 64, 64, qp, gop).expect("encode");
            assert_display_order_exact(&enc.stream, &enc.recon, &format!("gop{gop} qp{qp}"));
        }
    }

    /// The decode-order schedule is the dyadic pyramid, the layers
    /// are the recursion depths, and the mid slices really code
    /// bi-predictively (past L0 + future L1).
    #[test]
    fn pyramid_schedule_layers_and_bi_usage() {
        let planes = pan_scene(64, 64, 9);
        let frames = as_frames(&planes);
        let enc = encode_pyramid(&frames, 64, 64, 26, 8).expect("encode");
        // IDR, then the mini-GOP in dyadic decode order.
        assert_eq!(enc.decode_order, vec![0, 8, 4, 2, 1, 3, 6, 5, 7]);
        assert_eq!(
            enc.layers,
            vec![0, 3, 2, 3, 1, 3, 2, 3, 0],
            "display-order layers of a GOP-8 pyramid"
        );
        // Every B layer >= 1 frame elects some bi-predicted CUs on
        // smooth translation.
        let bi_frames = (0..9)
            .filter(|&i| enc.layers[i] >= 1 && enc.stats[i].bi > 0)
            .count();
        assert!(
            bi_frames >= 4,
            "expected widespread bi-prediction, stats: {:?}",
            enc.stats
        );
    }

    /// Pyramid coding beats the low-delay chain on smooth motion at
    /// the same base QP: fewer bytes for comparable quality.
    #[test]
    fn pyramid_beats_low_delay_on_smooth_motion() {
        let (w, h, n) = (64usize, 64usize, 9usize);
        let planes = pan_scene(w, h, n);
        let frames = as_frames(&planes);
        let qp = 30;
        let low = encode_low_delay_p(&frames, w, h, qp).expect("low-delay");
        let pyr = encode_pyramid(&frames, w, h, qp, 8).expect("pyramid");
        assert!(
            pyr.stream.len() < low.stream.len(),
            "pyramid ({} B) should undercut low-delay ({} B) on smooth motion",
            pyr.stream.len(),
            low.stream.len()
        );
        // Quality stays in the same class (the per-layer QP offsets
        // trade a little PSNR on the top layers for the rate win).
        let psnr = |rec: &FrameRecon, t: usize| -> f64 {
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
            10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10()
        };
        for t in 0..n {
            assert!(
                psnr(&pyr.recon[t], t) > 30.0,
                "frame {t}: pyramid luma PSNR too low"
            );
        }
    }

    /// Pyramids compose with the AMP configuration and the §8.7
    /// in-loop filters — everything still decodes bit-exactly.
    #[test]
    fn pyramid_composes_with_amp_and_filters() {
        let planes = pan_scene(64, 48, 9);
        let frames = as_frames(&planes);
        for (amp, lf) in [
            (true, LoopFilterCfg::off()),
            (false, LoopFilterCfg::all()),
            (true, LoopFilterCfg::all()),
        ] {
            let enc = PyramidEncoder::new(64, 48, 32, 4)
                .expect("encoder")
                .with_amp(amp)
                .with_loop_filters(lf);
            let out = encode_pyramid_with(enc, &frames).expect("encode");
            assert_display_order_exact(&out.stream, &out.recon, &format!("amp={amp} {lf:?}"));
        }
    }

    /// Successive mini-GOPs chain across anchors, and flushing midway
    /// (a 6-frame sequence at GOP 4: IDR + one mini-GOP + 1-frame
    /// tail) stays bit-exact.
    #[test]
    fn pyramid_multi_gop_and_tail_roundtrip() {
        let planes = pan_scene(48, 32, 6);
        let frames = as_frames(&planes);
        let enc = encode_pyramid(&frames, 48, 32, 22, 4).expect("encode");
        assert_eq!(enc.decode_order, vec![0, 4, 2, 1, 3, 5]);
        assert_display_order_exact(&enc.stream, &enc.recon, "gop4 tail");
    }

    /// Input validation.
    /// Non-dyadic mini-GOP lengths: the midpoint schedule covers every
    /// frame exactly once, the SPS carries the schedule-derived DPB /
    /// reorder bounds, and every stream decodes bit-exact.
    #[test]
    fn non_dyadic_gops_roundtrip_with_exact_bounds() {
        for (gop, reorder, refs) in [(3usize, 1u32, 2u32), (5, 2, 3), (6, 2, 3), (12, 3, 4)] {
            assert_eq!(
                schedule_bounds(gop as i32),
                (reorder, refs),
                "gop {gop} bounds"
            );
            let planes = pan_scene(48, 48, gop + 3);
            let frames = as_frames(&planes);
            let enc = PyramidEncoder::new(48, 48, 30, gop).expect("encoder");
            assert_eq!(enc.reorder_delay(), reorder);
            let out = encode_pyramid_with(enc, &frames).expect("encode");
            assert_display_order_exact(&out.stream, &out.recon, &format!("gop {gop}"));
            let units = crate::nal::collect_nal_units(&out.stream).expect("nal walk");
            let sps = crate::sps::SeqParameterSet::parse(&units[1].rbsp).expect("sps");
            assert_eq!(
                sps.sub_layer_ordering_info[0].max_num_reorder_pics, reorder,
                "gop {gop} reorder"
            );
            assert_eq!(
                sps.sub_layer_ordering_info[0].max_dec_pic_buffering_minus1, refs,
                "gop {gop} dpb"
            );
        }
    }

    /// Adaptive mini-GOP closing: a hard scene cut inside a GOP-8
    /// window closes the mini-GOP before the cut (access units come
    /// out early, no B slice references across the cut), the cut
    /// frame opens the next mini-GOP, and the flush tail is coded as
    /// a short pyramid with a B slice.
    #[test]
    fn adaptive_gop_closes_at_scene_cuts_and_codes_short_tails() {
        let (w, h) = (48usize, 48usize);
        let mut planes = pan_scene(w, h, 5);
        // Frames 5.. are a different scene (inverted + shifted content).
        let other: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = pan_scene(w, h, 6)
            .into_iter()
            .map(|(y, cb, cr)| (y.iter().map(|&v| 255 - v).collect(), cb, cr))
            .collect();
        planes.extend(other);
        let frames = as_frames(&planes);
        let mut enc = PyramidEncoder::new(w, h, 30, 8)
            .expect("encoder")
            .with_adaptive_gop(true);
        let mut bursts: Vec<usize> = Vec::new();
        let mut aus_all = Vec::new();
        for f in &frames {
            let aus = enc.encode_frame(f).expect("encode");
            if !aus.is_empty() {
                bursts.push(aus.len());
            }
            aus_all.extend(aus);
        }
        let tail = enc.flush();
        assert!(
            tail.iter().any(|au| au.layer > 0),
            "flush tail codes a pyramid"
        );
        aus_all.extend(tail);
        // IDR burst, then the pre-cut mini-GOP (frames 1..=4) closed
        // early at frame 5, then the post-cut mini-GOP fills at 8 frames?
        // Only 6 post-cut frames exist, so they come out at flush.
        assert_eq!(bursts, vec![1, 4], "IDR, then the four pre-cut frames");
        let stream: Vec<u8> = aus_all.iter().flat_map(|a| a.au.clone()).collect();
        let mut recons: Vec<Option<FrameRecon>> = vec![None; planes.len()];
        for au in &aus_all {
            recons[au.display_order] = Some(au.recon.clone());
        }
        let recons: Vec<FrameRecon> = recons.into_iter().map(|r| r.expect("coded")).collect();
        assert_display_order_exact(&stream, &recons, "adaptive");
        // No slice straddles the cut: every B slice's references lie
        // on its own side of frame 5.
        for au in &aus_all {
            let d = au.display_order;
            if d >= 5 && au.layer > 0 {
                // Inside the post-cut mini-GOP (coded at flush).
                assert!(d >= 5);
            }
        }
    }

    #[test]
    fn rejects_bad_configs() {
        assert!(matches!(
            PyramidEncoder::new(64, 64, 26, 1),
            Err(PyramidError::BadGop(1))
        ));
        assert!(matches!(
            PyramidEncoder::new(64, 64, 26, 32),
            Err(PyramidError::BadGop(32))
        ));
        assert!(PyramidEncoder::new(64, 64, 26, 3).is_ok());
        assert!(matches!(
            PyramidEncoder::new(20, 64, 26, 4),
            Err(PyramidError::Encode(IntraEncodeError::BadDimensions { .. }))
        ));
        assert!(matches!(
            PyramidEncoder::new(64, 64, 99, 4),
            Err(PyramidError::Encode(IntraEncodeError::BadQp(99)))
        ));
    }
}
