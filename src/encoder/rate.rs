//! Encoder rate control — average-bitrate (ABR) targeting.
//!
//! Rate control is not part of the Recommendation (a conforming
//! bitstream may spend its bits however it likes); this module is the
//! encoder-side policy that picks each frame's `SliceQpY` so the
//! produced stream lands on a caller-supplied average bitrate.
//!
//! The model rides the §8.6.3 quantizer geometry: `levelScale` cycles
//! every 6 QP with a factor-of-two step, so the quantization step is
//! `Qstep ∝ 2^(QP/6)` and, to first order, a frame's coded size is
//!
//! ```text
//! bits(QP) ≈ C / 2^(QP/6)
//! ```
//!
//! for a per-frame-class complexity `C` (intra frames and inter frames
//! track separate estimates — an IDR costs several times a P at equal
//! QP). After each coded frame the observed `C = bits · 2^(QP/6)` is
//! folded into an exponential moving average; before each frame the
//! controller inverts the model against the frame's bit budget (the
//! per-frame target plus a leaky-bucket correction that drains
//! accumulated over/undershoot) and clamps the result to a bounded
//! per-frame QP excursion so quality never jumps.
//!
//! All arithmetic is integer (Q16 fixed point for the `2^(QP/6)`
//! lattice), so a rate-controlled encode is bit-deterministic across
//! platforms.

/// `round(2^16 · 2^(r/6))` for `r = 0..6` — one QP step on the §8.6.3
/// quantizer lattice is a factor of `2^(1/6)`.
const STEP_Q16: [u64; 6] = [65536, 73562, 82570, 92682, 104032, 116772];

/// `2^(qp/6)` in Q16 for `qp` in `0..=51`.
fn pow2_qp_q16(qp: i32) -> u64 {
    debug_assert!((0..=51).contains(&qp));
    STEP_Q16[(qp % 6) as usize] << (qp / 6)
}

/// Average-bitrate rate-control configuration for the coding-mode
/// encoders (`with_rate_control` on the low-delay and hierarchical-B
/// encoders; the `bitrate` / `fps` codec options on the registry
/// encoder).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateControlCfg {
    /// Target average bitrate in bits per second.
    pub bits_per_second: u64,
    /// Frame-rate numerator (frames per `fps_den` seconds).
    pub fps_num: u32,
    /// Frame-rate denominator.
    pub fps_den: u32,
    /// The first frame's QP before any feedback exists. `None` derives
    /// a starting point from the target bits per pixel.
    pub initial_qp: Option<i32>,
    /// Lowest `SliceQpY` the controller may pick (default 4).
    pub min_qp: i32,
    /// Highest `SliceQpY` the controller may pick (default 49).
    pub max_qp: i32,
    /// VBV buffer size in bits (`None` = unconstrained ABR). When
    /// set, the encoder models a decoder buffer of this size filled
    /// at [`Self::bits_per_second`] and drained whole-frame at each
    /// decode instant: the controller aims each frame under the
    /// current fullness, and the low-delay encoder re-encodes at a
    /// higher QP when a frame would still underflow it.
    pub vbv_buffer_bits: Option<u64>,
}

impl RateControlCfg {
    /// An ABR configuration at `bits_per_second` for
    /// `fps_num / fps_den` frames per second, with the default QP
    /// bounds and a bits-per-pixel-derived starting QP.
    #[must_use]
    pub fn new(bits_per_second: u64, fps_num: u32, fps_den: u32) -> Self {
        Self {
            bits_per_second,
            fps_num,
            fps_den,
            initial_qp: None,
            min_qp: 4,
            max_qp: 49,
            vbv_buffer_bits: None,
        }
    }

    /// Add a VBV buffer constraint of `bits` (see
    /// [`Self::vbv_buffer_bits`]).
    #[must_use]
    pub fn with_vbv(mut self, bits: u64) -> Self {
        self.vbv_buffer_bits = Some(bits);
        self
    }
}

/// The two complexity classes the controller models separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameClass {
    /// IDR / all-intra frames.
    Intra,
    /// P / B frames.
    Inter,
}

impl FrameClass {
    fn idx(self) -> usize {
        match self {
            Self::Intra => 0,
            Self::Inter => 1,
        }
    }
}

/// When one class has no history yet, its complexity is seeded from
/// the other class scaled by this ratio (an IDR is worth roughly this
/// many equal-QP inter frames; feedback corrects it after one
/// observation).
const INTRA_OVER_INTER: u64 = 5;

/// Streaming ABR controller: `pick_qp` before each frame,
/// `update` with the coded size after it.
#[derive(Debug, Clone)]
pub(crate) struct RateController {
    /// Per-frame bit budget (`bits_per_second · fps_den / fps_num`).
    target: i64,
    min_qp: i32,
    max_qp: i32,
    /// QP for the very first frame (no history at all).
    initial_qp: i32,
    /// Per-class complexity estimate `C ≈ bits · 2^(QP/6)` (EWMA).
    complexity: [Option<u64>; 2],
    /// Per-class QP of the most recent coded frame.
    last_qp: [Option<i32>; 2],
    /// Frames coded so far.
    frame_count: u64,
    /// Frame index at which each class last coded (drives the
    /// widening excursion window: a class coded every frame moves at
    /// most ±3 per frame, a rare class — IDRs on a long GOP — may
    /// re-anchor faster).
    coded_at: [Option<u64>; 2],
    /// Leaky bucket: accumulated `coded − target` bits, clamped to
    /// ±one second of the target rate.
    buffer: i64,
    buffer_cap: i64,
    /// VBV decoder-buffer model: `(size, fullness)` in bits. Fullness
    /// starts at `size`, refills by [`Self::target`] per frame and
    /// drains whole-frame in `update`.
    vbv: Option<(i64, i64)>,
    /// The bit budget the most recent [`Self::pick_qp`] aimed the
    /// frame at (the CTU-level feedback's per-frame allocation).
    last_budget: Option<u64>,
}

impl RateController {
    /// Build a controller for `width`x`height` luma. The starting QP
    /// is `cfg.initial_qp` when given, else derived from the target
    /// bits per pixel per frame.
    pub(crate) fn new(cfg: &RateControlCfg, width: usize, height: usize) -> Self {
        let (num, den) = (u64::from(cfg.fps_num.max(1)), u64::from(cfg.fps_den.max(1)));
        let target = ((cfg.bits_per_second.saturating_mul(den) / num).max(64)) as i64;
        let (min_qp, max_qp) = (cfg.min_qp.clamp(0, 51), cfg.max_qp.clamp(0, 51));
        let initial_qp = cfg
            .initial_qp
            .unwrap_or_else(|| initial_qp_for_bpp(target as u64, width * height))
            .clamp(min_qp, max_qp);
        Self {
            target,
            min_qp,
            max_qp,
            initial_qp,
            complexity: [None, None],
            last_qp: [None, None],
            frame_count: 0,
            coded_at: [None, None],
            buffer: 0,
            buffer_cap: (cfg.bits_per_second.max(64)).min(i64::MAX as u64) as i64,
            vbv: cfg.vbv_buffer_bits.map(|b| {
                let size = b.clamp(256, i64::MAX as u64) as i64;
                (size, size)
            }),
            last_budget: None,
        }
    }

    /// The bit budget the most recent [`Self::pick_qp`] aimed at
    /// (`None` before the first election) — the frame allocation the
    /// quadtree coder's CTU-level feedback tracks inside the picture.
    pub(crate) fn last_budget_bits(&self) -> Option<u64> {
        self.last_budget
    }

    /// The hard VBV budget for the NEXT frame (its coded size must
    /// stay at or under this many bits, or the modelled decoder
    /// buffer underflows). `None` without a VBV constraint.
    pub(crate) fn vbv_frame_cap(&self) -> Option<u64> {
        self.vbv.map(|(_, fullness)| fullness.max(0) as u64)
    }

    /// The configured QP ceiling (the re-encode loop's last resort).
    pub(crate) fn max_qp(&self) -> i32 {
        self.max_qp
    }

    /// Elect the next frame's `SliceQpY` for a frame of `class`.
    pub(crate) fn pick_qp(&mut self, class: FrameClass) -> i32 {
        // The frame budget: the nominal share minus a drain of the
        // accumulated overshoot spread over the next 8 frames, kept
        // within a sane multiple of the nominal share.
        let mut desired = (self.target - self.buffer / 8)
            .clamp(self.target / 4, self.target.saturating_mul(4))
            .max(64) as u64;
        if let Some((_, fullness)) = self.vbv {
            // Aim comfortably under the hard VBV budget so the
            // re-encode loop stays a rare emergency.
            desired = desired.min((fullness.max(64) as u64).saturating_mul(3) / 4);
        }
        self.last_budget = Some(desired);
        let complexity = self.complexity[class.idx()].or_else(|| {
            // Cold class: borrow the other class through the
            // intra/inter cost ratio.
            let other = self.complexity[1 - class.idx()]?;
            Some(match class {
                FrameClass::Intra => other.saturating_mul(INTRA_OVER_INTER),
                FrameClass::Inter => (other / INTRA_OVER_INTER).max(1),
            })
        });
        let mut qp = match complexity {
            None => self.initial_qp,
            Some(c) => {
                // Smallest QP whose predicted size fits the budget:
                // bits(qp) = c · 2^16 / pow2_qp_q16(qp) ≤ desired.
                let mut pick = self.max_qp;
                for q in self.min_qp..=self.max_qp {
                    let predicted = (c.saturating_mul(1 << 16)) / pow2_qp_q16(q);
                    if predicted <= desired {
                        pick = q;
                        break;
                    }
                }
                pick
            }
        };
        // Bounded excursion against the same class's last frame. The
        // window widens with the frames elapsed since that class last
        // coded (`min(2 + gap, 15)`): a back-to-back class moves at
        // most ±3 per frame, while a rare class (IDRs on a long GOP)
        // may re-anchor within one refresh instead of creeping ±3 per
        // GOP. A class with no history at all stays within ±6 of the
        // other class (the model seed above is only a ratio guess).
        if let (Some(last), Some(at)) = (self.last_qp[class.idx()], self.coded_at[class.idx()]) {
            let gap = self.frame_count.saturating_sub(at);
            let window = (2 + gap).min(15) as i32;
            qp = qp.clamp(last - window, last + window);
        } else if let Some(last) = self.last_qp[1 - class.idx()] {
            qp = qp.clamp(last - 6, last + 6);
        }
        qp.clamp(self.min_qp, self.max_qp)
    }

    /// Fold a coded frame back into the model: `class` at `qp` cost
    /// `bits` bits (the whole access unit, headers included).
    pub(crate) fn update(&mut self, class: FrameClass, qp: i32, bits: u64) {
        let observed = (bits.max(1).saturating_mul(pow2_qp_q16(qp))) >> 16;
        let slot = &mut self.complexity[class.idx()];
        *slot = Some(match *slot {
            None => observed.max(1),
            // EWMA with weight 2/5 on the newest observation.
            Some(c) => ((c.saturating_mul(3) + observed.saturating_mul(2)) / 5).max(1),
        });
        self.last_qp[class.idx()] = Some(qp);
        self.coded_at[class.idx()] = Some(self.frame_count);
        self.frame_count += 1;
        self.buffer = (self.buffer + bits.min(i64::MAX as u64) as i64 - self.target)
            .clamp(-self.buffer_cap, self.buffer_cap);
        if let Some((size, fullness)) = &mut self.vbv {
            // Whole-frame drain, then one frame interval of fill.
            *fullness = (*fullness - bits.min(i64::MAX as u64) as i64 + self.target).min(*size);
        }
    }
}

/// A starting QP from the frame budget in bits per luma sample
/// (coarse on purpose: one coded frame of feedback replaces it).
fn initial_qp_for_bpp(frame_bits: u64, luma_samples: usize) -> i32 {
    let bpp_q16 = frame_bits.saturating_mul(1 << 16) / (luma_samples.max(1) as u64);
    let table: [(u64, i32); 8] = [
        (98304, 12), // ≥ 1.5 bpp
        (49152, 16), // ≥ 0.75
        (26214, 20), // ≥ 0.4
        (13107, 24), // ≥ 0.2
        (6554, 28),  // ≥ 0.1
        (3277, 32),  // ≥ 0.05
        (1638, 36),  // ≥ 0.025
        (786, 40),   // ≥ 0.012
    ];
    for (threshold, qp) in table {
        if bpp_q16 >= threshold {
            return qp;
        }
    }
    44
}

/// The SAD-domain motion-search λ for an SSD-domain mode λ:
/// `3 · isqrt(λ) / 2` (the square-root relation between the two
/// distortion domains, with a 1.5× weight that keeps the motion
/// field coherent once the grid scan can reach far minima — measured
/// on periodic and smooth pans: the 1× weight lost ~8 % rate on the
/// low-delay chain at QP 32, 2× lost ~15 % at QP 27 on smooth
/// content).
pub(crate) fn motion_lambda(lambda: u64) -> u64 {
    (3 * isqrt_u64(lambda)).div_ceil(2)
}

/// Integer square root (floor), deterministic.
pub(crate) fn isqrt_u64(v: u64) -> u64 {
    if v < 2 {
        return v;
    }
    let mut x = 1u64 << ((64 - v.leading_zeros()).div_ceil(2));
    loop {
        let y = (x + v / x) / 2;
        if y >= x {
            return x;
        }
        x = y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pow2_lattice_doubles_every_six_qp() {
        for qp in 0..=45 {
            assert_eq!(
                pow2_qp_q16(qp + 6),
                pow2_qp_q16(qp) * 2,
                "qp {qp}: one octave per 6 QP"
            );
        }
        for qp in 0..51 {
            assert!(pow2_qp_q16(qp) < pow2_qp_q16(qp + 1), "monotonic at {qp}");
        }
        assert_eq!(pow2_qp_q16(0), 1 << 16);
    }

    #[test]
    fn initial_qp_is_monotonic_in_bpp() {
        let mut last = i32::MAX;
        for bits in [1u64, 50, 200, 800, 3000, 12_000, 50_000, 200_000] {
            let qp = initial_qp_for_bpp(bits, 64 * 64);
            assert!(qp <= last, "more bits never raises the starting QP");
            assert!((0..=51).contains(&qp));
            last = qp;
        }
    }

    /// Drive the controller with frames obeying the model exactly
    /// (`bits = C / 2^(qp/6)`, intra worth 5 inter) and return the
    /// per-frame sizes.
    fn simulate(rate: u64, gop: u64, initial_qp: Option<i32>, n: u64) -> Vec<u64> {
        let (c_intra, c_inter) = (2_500_000u64, 500_000u64);
        let mut cfg = RateControlCfg::new(rate, 25, 1);
        cfg.initial_qp = initial_qp;
        let mut rc = RateController::new(&cfg, 64, 64);
        (0..n)
            .map(|i| {
                let (class, c) = if i % gop == 0 {
                    (FrameClass::Intra, c_intra)
                } else {
                    (FrameClass::Inter, c_inter)
                };
                let qp = rc.pick_qp(class);
                assert!((rc.min_qp..=rc.max_qp).contains(&qp));
                let bits = (c.saturating_mul(1 << 16)) / pow2_qp_q16(qp);
                rc.update(class, qp, bits);
                bits
            })
            .collect()
    }

    /// With a sane starting QP the whole run lands on the budget.
    #[test]
    fn controller_hits_target_from_informed_start() {
        for (rate, gop, init) in [(100_000u64, 25u64, 42), (400_000, 25, 30), (128_000, 8, 38)] {
            let bits = simulate(rate, gop, Some(init), 120);
            let total: u64 = bits.iter().sum();
            let wanted = rate * bits.len() as u64 / 25;
            let err_pct = (total as i64 - wanted as i64).unsigned_abs() * 100 / wanted;
            assert!(
                err_pct <= 5,
                "rate {rate} gop {gop}: total {total} vs {wanted} ({err_pct}%)"
            );
        }
    }

    /// From a cold start (bits-per-pixel guess) the tail converges
    /// onto the budget even when the guess was far off — including
    /// the rare-class case where IDRs arrive only every 25 frames and
    /// must re-anchor through the widening excursion window.
    #[test]
    fn controller_converges_from_cold_start() {
        for (rate, gop) in [(100_000u64, 25u64), (400_000, 25), (128_000, 8)] {
            let bits = simulate(rate, gop, None, 120);
            let tail: u64 = bits[90..].iter().sum();
            let wanted = rate * 30 / 25;
            let err_pct = (tail as i64 - wanted as i64).unsigned_abs() * 100 / wanted;
            assert!(
                err_pct <= 8,
                "rate {rate} gop {gop}: tail {tail} vs {wanted} ({err_pct}%)"
            );
        }
    }

    #[test]
    fn vbv_caps_the_frame_budget_and_tracks_fullness() {
        let mut rc = RateController::new(
            &RateControlCfg::new(100_000, 25, 1).with_vbv(20_000),
            64,
            64,
        );
        // Full buffer to start.
        assert_eq!(rc.vbv_frame_cap(), Some(20_000));
        let q0 = rc.pick_qp(FrameClass::Inter);
        rc.update(FrameClass::Inter, q0, 12_000);
        // 20000 - 12000 + 4000 (one frame of fill at 100k/25).
        assert_eq!(rc.vbv_frame_cap(), Some(12_000));
        // Refill never exceeds the buffer size.
        rc.update(FrameClass::Inter, q0, 0);
        assert_eq!(rc.vbv_frame_cap(), Some(16_000));
        for _ in 0..10 {
            rc.update(FrameClass::Inter, q0, 0);
        }
        assert_eq!(rc.vbv_frame_cap(), Some(20_000));
        // A low buffer pushes the next pick's QP up against an
        // unconstrained twin.
        let mut tight = RateController::new(
            &RateControlCfg::new(100_000, 25, 1).with_vbv(20_000),
            64,
            64,
        );
        let mut free = RateController::new(&RateControlCfg::new(100_000, 25, 1), 64, 64);
        for rcx in [&mut tight, &mut free] {
            let q = rcx.pick_qp(FrameClass::Inter);
            rcx.update(FrameClass::Inter, q, 19_000); // nearly drains the VBV
        }
        assert!(tight.pick_qp(FrameClass::Inter) >= free.pick_qp(FrameClass::Inter));
    }

    #[test]
    fn qp_excursion_is_bounded() {
        let mut rc = RateController::new(&RateControlCfg::new(100_000, 25, 1), 64, 64);
        let q0 = rc.pick_qp(FrameClass::Inter);
        // A wildly oversized frame cannot yank the next QP by more
        // than 3.
        rc.update(FrameClass::Inter, q0, 10_000_000);
        let q1 = rc.pick_qp(FrameClass::Inter);
        assert!((q1 - q0) <= 3 && (q1 - q0) >= 0, "{q0} -> {q1}");
        // And a tiny frame at the new point pulls back by at most 3.
        rc.update(FrameClass::Inter, q1, 10);
        let q2 = rc.pick_qp(FrameClass::Inter);
        assert!((q1 - q2) <= 3, "{q1} -> {q2}");
    }

    #[test]
    fn intra_seed_borrows_inter_complexity() {
        let mut rc = RateController::new(&RateControlCfg::new(200_000, 25, 1), 64, 64);
        let qp = rc.pick_qp(FrameClass::Inter);
        rc.update(FrameClass::Inter, qp, 6_000);
        // The first intra pick uses the scaled inter model, not the
        // cold-start heuristic, and stays within the cross-class
        // excursion bound.
        let qi = rc.pick_qp(FrameClass::Intra);
        assert!((qi - qp).abs() <= 6, "inter {qp} -> intra {qi}");
    }
}
