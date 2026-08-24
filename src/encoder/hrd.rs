//! Encoder-side HRD signalling — the §E.2.2 `hrd_parameters( )`
//! writer, the §D.2.2 / §D.2.3 SEI payload builders, and an exact
//! Annex C CPB clock ([`HrdClock`]) the encoders consult so every
//! emitted stream satisfies the §C.4 conformance conditions by
//! construction.
//!
//! The signalled delivery schedule is a single CPB (one
//! `SchedSelIdx`), NAL HRD only, VBR (`cbr_flag == 0`), AU-level
//! (`sub_pic_hrd_params_present_flag == 0`), fixed picture rate (one
//! access unit per §E.3.2 elemental clock tick). The initial CPB
//! removal delay of the stream-opening buffering period is the
//! time-equivalent of the whole CPB, and every later buffering
//! period keeps `delay + offset` constant (§D.3.2) by trading the
//! two against the C-18 bound the clock computes from its own
//! replay.
//!
//! All arithmetic is exact: times are u128 integers in units of
//! `1 / (90000 · fps_num · BitRate)` seconds, in which every Annex C
//! quantity of this schedule — 90 kHz initial delays, `ClockTick`
//! multiples and `bits ÷ BitRate` arrival spans — is an integer, so
//! the clock is bit-deterministic across platforms and free of the
//! rounding the Annex's real-value arithmetic forbids.

use crate::encoder::bitwriter::BitWriter;
use crate::encoder::nal::nal_unit;
use crate::sei::PREFIX_SEI_NUT;

/// `initial_cpb_removal_delay_length_minus1 + 1` this encoder
/// signals: 24-bit 90 kHz initial delays (about 186 s of buffering).
pub const INITIAL_CPB_REMOVAL_DELAY_LENGTH: u8 = 24;
/// `au_cpb_removal_delay_length_minus1 + 1`: 24-bit tick counts
/// between buffering periods.
pub const AU_CPB_REMOVAL_DELAY_LENGTH: u8 = 24;
/// `dpb_output_delay_length_minus1 + 1`: 16-bit output-delay ticks.
pub const DPB_OUTPUT_DELAY_LENGTH: u8 = 16;

/// §D.2.2 `buffering_period` payload type.
pub const SEI_BUFFERING_PERIOD: u8 = 0;
/// §D.2.3 `pic_timing` payload type.
pub const SEI_PIC_TIMING: u8 = 1;

/// The delivery schedule the SPS VUI signals: one CPB entry at
/// `bit_rate_scale == cpb_size_scale == 0`, so `BitRate[0]` is a
/// multiple of 64 b/s (§E.3.3 eq. E-87) and `CpbSize[0]` a multiple
/// of 16 bits (eq. E-88).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HrdSignalCfg {
    /// Signalled `BitRate[0]` in bits per second (multiple of 64).
    pub bit_rate: u64,
    /// Signalled `CpbSize[0]` in bits (multiple of 16).
    pub cpb_size: u64,
    /// `cbr_flag[0]`: constant-bit-rate delivery — the HSS feeds the
    /// CPB back to back at exactly `BitRate` (eq. C-3), so the
    /// encoder must pad underruns with §7.3.4 filler-data NAL units
    /// to keep the CPB from overflowing.
    pub cbr: bool,
}

impl HrdSignalCfg {
    /// The schedule covering an encoder targeting `bits_per_second`
    /// under a VBV buffer of `vbv_buffer_bits`: both values are
    /// rounded UP to the scale-0 granularity (the signalled schedule
    /// is never tighter than the enforced model) and clamped to the
    /// §E.3.3 `ue(v)` value range.
    #[must_use]
    pub fn for_rate(bits_per_second: u64, vbv_buffer_bits: u64) -> Self {
        let bit_rate = bits_per_second
            .clamp(64, u64::from(u32::MAX) * 64)
            .div_ceil(64)
            * 64;
        let cpb_size = vbv_buffer_bits
            .clamp(16, u64::from(u32::MAX) * 16)
            .div_ceil(16)
            * 16;
        Self {
            bit_rate,
            cpb_size,
            cbr: false,
        }
    }

    /// Switch the schedule to constant-bit-rate delivery
    /// (`cbr_flag[0] == 1`; see [`Self::cbr`]).
    #[must_use]
    pub fn with_cbr(mut self, on: bool) -> Self {
        self.cbr = on;
        self
    }

    /// `bit_rate_value_minus1[0]` per eq. E-87 at `bit_rate_scale 0`.
    #[must_use]
    pub fn bit_rate_value_minus1(&self) -> u32 {
        (self.bit_rate / 64 - 1) as u32
    }

    /// `cpb_size_value_minus1[0]` per eq. E-88 at `cpb_size_scale 0`.
    #[must_use]
    pub fn cpb_size_value_minus1(&self) -> u32 {
        (self.cpb_size / 16 - 1) as u32
    }
}

/// Write the §E.2.2 `hrd_parameters( 1, 0 )` body this encoder
/// signals inside the SPS VUI (NAL HRD only, one CPB, AU-level, VBR,
/// fixed picture rate with one AU per elemental tick).
pub(crate) fn write_hrd_parameters(w: &mut BitWriter, cfg: &HrdSignalCfg) {
    w.put_bit(1); // nal_hrd_parameters_present_flag
    w.put_bit(0); // vcl_hrd_parameters_present_flag
    w.put_bit(0); // sub_pic_hrd_params_present_flag
    w.put_bits(0, 4); // bit_rate_scale
    w.put_bits(0, 4); // cpb_size_scale
    w.put_bits(u32::from(INITIAL_CPB_REMOVAL_DELAY_LENGTH) - 1, 5);
    w.put_bits(u32::from(AU_CPB_REMOVAL_DELAY_LENGTH) - 1, 5);
    w.put_bits(u32::from(DPB_OUTPUT_DELAY_LENGTH) - 1, 5);
    // Sub-layer 0 (sps_max_sub_layers_minus1 == 0).
    w.put_bit(1); // fixed_pic_rate_general_flag[0]
    w.ue(0); // elemental_duration_in_tc_minus1[0] — one AU per tick
    w.ue(0); // cpb_cnt_minus1[0]
             // sub_layer_hrd_parameters( 0 ), NAL path.
    w.ue(cfg.bit_rate_value_minus1());
    w.ue(cfg.cpb_size_value_minus1());
    w.put_bit(u8::from(cfg.cbr)); // cbr_flag[0]
}

/// Build one §7.3.4 `filler_data_rbsp( )` NAL unit, framed with its
/// four-byte start code, whose TOTAL framed size is
/// `max(7, min_bytes)` bytes (start code 4 + NAL header 2 + `n`
/// `ff_byte`s + the `rbsp_trailing_bits` `0x80`): the CBR arm's
/// underrun padding. `FD_NUT` follows the VCL NAL unit within the
/// access unit (§7.4.2.4.4).
#[must_use]
pub(crate) fn filler_data_nal_framed(min_bytes: usize) -> Vec<u8> {
    let n_ff = min_bytes.saturating_sub(7);
    let mut rbsp = vec![0xFFu8; n_ff];
    rbsp.push(0x80);
    let mut framed = vec![0, 0, 0, 1];
    framed.extend(nal_unit(38, 0, 0, &rbsp)); // FD_NUT
    framed
}

/// Build the §D.2.2 `buffering_period( )` payload bytes:
/// `bp_seq_parameter_set_id == 0`, no IRAP alternative pairs,
/// `concatenation_flag == 0`, `au_cpb_removal_delay_delta_minus1 ==
/// 0` (every picture is one tick after its decode-order predecessor
/// and all pictures are non-discardable), one NAL CPB pair.
#[must_use]
pub(crate) fn buffering_period_payload(initial_delay: u32, initial_offset: u32) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.ue(0); // bp_seq_parameter_set_id
    w.put_bit(0); // irap_cpb_params_present_flag
    w.put_bit(0); // concatenation_flag
    w.put_bits(0, AU_CPB_REMOVAL_DELAY_LENGTH); // au_cpb_removal_delay_delta_minus1
    w.put_bits(initial_delay, INITIAL_CPB_REMOVAL_DELAY_LENGTH);
    w.put_bits(initial_offset, INITIAL_CPB_REMOVAL_DELAY_LENGTH);
    // §D.3.1 payload alignment (the body is not byte-aligned).
    w.put_bit(1);
    w.align_zero();
    w.finish()
}

/// Build the §D.2.3 `pic_timing( )` payload bytes (the active VUI
/// has `frame_field_info_present_flag == 0`, so the body is the two
/// AU-level delays — 40 bits, exactly byte-aligned).
#[must_use]
pub(crate) fn pic_timing_payload(
    au_cpb_removal_delay_minus1: u32,
    pic_dpb_output_delay: u32,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(au_cpb_removal_delay_minus1, AU_CPB_REMOVAL_DELAY_LENGTH);
    w.put_bits(pic_dpb_output_delay, DPB_OUTPUT_DELAY_LENGTH);
    w.finish()
}

/// Frame SEI payloads into one complete `PREFIX_SEI_NUT` NAL unit
/// (§7.3.5 short/extensible size framing + `rbsp_trailing_bits`).
#[must_use]
pub(crate) fn sei_prefix_nal(payloads: &[(u8, Vec<u8>)]) -> Vec<u8> {
    let mut rbsp = Vec::new();
    for (payload_type, body) in payloads {
        rbsp.push(*payload_type);
        let mut n = body.len();
        while n >= 255 {
            rbsp.push(0xFF);
            n -= 255;
        }
        rbsp.push(n as u8);
        rbsp.extend_from_slice(body);
    }
    rbsp.push(0x80); // rbsp_trailing_bits
    nal_unit(PREFIX_SEI_NUT, 0, 0, &rbsp)
}

/// Splice a framed (start-code-prefixed) SEI NAL unit immediately
/// before the access unit's LAST NAL unit — the VCL slice. Parameter
/// sets stay first, so a decoder has activated the SPS before it
/// meets the context-dependent SEI (§D.3.3 NOTE 1), while the prefix
/// SEI still precedes the coded picture as §7.4.2.4.4 requires.
pub(crate) fn splice_sei_before_vcl(au: &mut Vec<u8>, sei: &[u8]) {
    // Every NAL of this encoder's AUs rides a 4-byte start code, and
    // §7.4.1.1 emulation prevention guarantees none appears inside a
    // NAL — the last match is the VCL slice.
    let pos = au.windows(4).rposition(|w| w == [0, 0, 0, 1]).unwrap_or(0);
    au.splice(pos..pos, sei.iter().copied());
}

/// The exact Annex C CPB clock for the single-schedule, AU-level,
/// VBR, fixed-rate stream this encoder emits.
///
/// Per access unit, in decode order, the encoder asks:
///
/// 1. [`HrdClock::begin_buffering_period`] when the AU carries a
///    §D.2.2 buffering period SEI (IRAP AUs) — the C-18-bounded
///    `(initial_cpb_removal_delay, initial_cpb_removal_offset)` pair;
/// 2. [`HrdClock::au_cpb_removal_delay_minus1`] for the AU's §D.2.3
///    pic-timing field;
/// 3. [`HrdClock::frame_cap`] — the hard bit budget above which the
///    AU's final CPB arrival time would pass its nominal removal
///    time (a §C.4 condition-3 underflow); the encoders re-encode at
///    a higher QP while over it;
/// 4. [`HrdClock::push_au`] with the coded size (every bit of the
///    Type II access unit, SEI and start codes included).
#[derive(Debug, Clone)]
pub(crate) struct HrdClock {
    cfg: HrdSignalCfg,
    fps_num: u32,
    fps_den: u32,
    /// The stream-opening initial CPB removal delay: the
    /// time-equivalent of the full CPB in 90 kHz units (bounded by
    /// the 24-bit field). Every buffering period keeps
    /// `delay + offset == delay0`.
    delay0_90k: u32,
    /// Decode index of the NEXT access unit.
    m: u64,
    /// Decode index of the latest buffering-period AU already
    /// pushed (PT removal delays count from it).
    last_bp: u64,
    /// `AuFinalArrivalTime[m − 1]` in clock units.
    final_arrival: u128,
    /// The buffering period the NEXT AU carries, when
    /// [`Self::begin_buffering_period`] was called: `(delay,
    /// offset)` — switches the earliest-arrival bound from C-6 to
    /// C-7 for that AU.
    pending_bp: Option<(u32, u32)>,
}

impl HrdClock {
    /// A clock over the signalled schedule at `fps_num / fps_den`
    /// access units per second.
    pub(crate) fn new(cfg: HrdSignalCfg, fps_num: u32, fps_den: u32) -> Self {
        let (fps_num, fps_den) = (fps_num.max(1), fps_den.max(1));
        // delay0 = floor(90000 · CpbSize ÷ BitRate), clamped to the
        // signalled field width and nonzero (§D.3.2 forbids 0).
        let delay0 = (90_000u64 * cfg.cpb_size / cfg.bit_rate)
            .clamp(1, (1 << INITIAL_CPB_REMOVAL_DELAY_LENGTH) - 1) as u32;
        Self {
            cfg,
            fps_num,
            fps_den,
            delay0_90k: delay0,
            m: 0,
            last_bp: 0,
            final_arrival: 0,
            pending_bp: None,
        }
    }

    /// One clock unit = `1 / (90000 · fps_num · BitRate)` seconds.
    /// A 90 kHz delay of `d` spans `d · fps_num · BitRate` units.
    fn units_per_90k(&self) -> u128 {
        u128::from(self.fps_num) * u128::from(self.cfg.bit_rate)
    }

    /// `ClockTick` (eq. C-1) in clock units.
    fn tick_units(&self) -> u128 {
        90_000u128 * u128::from(self.fps_den) * u128::from(self.cfg.bit_rate)
    }

    /// `AuNominalRemovalTime[m]` in clock units: the C-9 anchor plus
    /// `m` elemental ticks (C-10/C-11 collapse to this under
    /// `concatenation_flag == 0`, `au_cpb_removal_delay_delta_minus1
    /// == 0` and the PT delays this clock emits).
    fn removal_units(&self, m: u64) -> u128 {
        u128::from(self.delay0_90k) * self.units_per_90k() + u128::from(m) * self.tick_units()
    }

    /// The decode index the next [`Self::push_au`] will occupy.
    pub(crate) fn next_decode_index(&self) -> u64 {
        self.m
    }

    /// Open a buffering period on the NEXT access unit: the §D.2.2
    /// `(initial_cpb_removal_delay, initial_cpb_removal_offset)`
    /// pair. The delay is the full-buffer `delay0` bounded by the
    /// C-18 constraint (`InitCpbRemovalDelay <= Ceil(deltaTime90k)`
    /// against this clock's own replay); the offset keeps
    /// `delay + offset` constant across the CVS (§D.3.2).
    pub(crate) fn begin_buffering_period(&mut self) -> (u32, u32) {
        let delay = if self.m == 0 {
            self.delay0_90k
        } else {
            // deltaTime90k = 90000 · (removal[m] − finalArrival[m−1]);
            // no-underflow at m−1 guarantees this is ≥ one tick.
            let delta_units = self
                .removal_units(self.m)
                .saturating_sub(self.final_arrival);
            let delta_90k = delta_units / self.units_per_90k();
            u32::try_from(delta_90k)
                .unwrap_or(u32::MAX)
                .clamp(1, self.delay0_90k)
        };
        let pair = (delay, self.delay0_90k - delay);
        self.pending_bp = Some(pair);
        pair
    }

    /// The next AU's §D.2.3 `au_cpb_removal_delay_minus1`: elemental
    /// ticks since the latest ALREADY-PUSHED buffering-period AU,
    /// minus one, masked to the signalled field width (AU 0
    /// initializes the HRD and its value is unused — 0 is emitted).
    pub(crate) fn au_cpb_removal_delay_minus1(&self) -> u32 {
        let ticks = self.m.saturating_sub(self.last_bp);
        (ticks.saturating_sub(1) & ((1 << AU_CPB_REMOVAL_DELAY_LENGTH) - 1)) as u32
    }

    /// `initArrivalTime[m]` for the next AU (eqs. C-4..C-7): the
    /// later of the previous AU's final arrival and the
    /// earliest-arrival bound (`removal − delay` for a
    /// buffering-period AU, `removal − (delay + offset) = removal −
    /// delay0` otherwise).
    fn init_arrival_units(&self) -> u128 {
        if self.cfg.cbr && self.m > 0 {
            // Eq. C-3: back-to-back delivery, no earliest-arrival
            // throttle.
            return self.final_arrival;
        }
        let throttle = match self.pending_bp {
            Some((delay, _)) => u128::from(delay) * self.units_per_90k(),
            None => u128::from(self.delay0_90k) * self.units_per_90k(),
        };
        let earliest = self.removal_units(self.m).saturating_sub(throttle);
        self.final_arrival.max(earliest)
    }

    /// The hard §C.4 bit budget for the next AU: the largest Type II
    /// access-unit size whose final arrival time (eq. C-8) does not
    /// pass its nominal removal time. Also never larger than the
    /// signalled CPB (an AU bigger than the buffer could never fit).
    pub(crate) fn frame_cap(&self) -> u64 {
        let window = self
            .removal_units(self.m)
            .saturating_sub(self.init_arrival_units());
        let bits = window / (90_000u128 * u128::from(self.fps_num));
        u64::try_from(bits)
            .unwrap_or(u64::MAX)
            .min(self.cfg.cpb_size)
    }

    /// CBR only: the filler bits the next AU must append (beyond its
    /// `au_bits` coded bits) so the back-to-back arrival cannot
    /// overflow the CPB before the FOLLOWING removal: cumulative
    /// arrival must reach `removal(m + 1) − CpbSize ÷ BitRate` (the
    /// §C.4 condition-2 bound with continuous delivery). Returns 0
    /// under VBR or when the AU is already large enough.
    pub(crate) fn cbr_filler_bits(&self, au_bits: u64) -> u64 {
        if !self.cfg.cbr {
            return 0;
        }
        let bits_to_units = 90_000u128 * u128::from(self.fps_num);
        let need_final = self
            .removal_units(self.m + 1)
            .saturating_sub(u128::from(self.cfg.cpb_size) * bits_to_units);
        let have_final = self.init_arrival_units() + u128::from(au_bits) * bits_to_units;
        let deficit = need_final.saturating_sub(have_final);
        u64::try_from(deficit.div_ceil(bits_to_units)).unwrap_or(u64::MAX)
    }

    /// Commit the next AU at `bits` (all bits of the Type II access
    /// unit): records its arrival span and advances the clock.
    pub(crate) fn push_au(&mut self, bits: u64) {
        let init = self.init_arrival_units();
        self.final_arrival = init + u128::from(bits) * 90_000u128 * u128::from(self.fps_num);
        if self.pending_bp.take().is_some() {
            self.last_bp = self.m;
        }
        self.m += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hrd::HrdCommonInfo;
    use crate::sei::{BufferingPeriodSei, PicTimingSei};

    fn common() -> HrdCommonInfo {
        HrdCommonInfo {
            nal_hrd_parameters_present_flag: true,
            vcl_hrd_parameters_present_flag: false,
            sub_pic_hrd_params_present_flag: false,
            tick_divisor_minus2: 0,
            du_cpb_removal_delay_increment_length_minus1: 0,
            sub_pic_cpb_params_in_pic_timing_sei_flag: false,
            dpb_output_delay_du_length_minus1: 0,
            bit_rate_scale: 0,
            cpb_size_scale: 0,
            cpb_size_du_scale: 0,
            initial_cpb_removal_delay_length_minus1: INITIAL_CPB_REMOVAL_DELAY_LENGTH - 1,
            au_cpb_removal_delay_length_minus1: AU_CPB_REMOVAL_DELAY_LENGTH - 1,
            dpb_output_delay_length_minus1: DPB_OUTPUT_DELAY_LENGTH - 1,
        }
    }

    #[test]
    fn signal_cfg_rounds_up_on_the_scale_lattice() {
        for (rate, vbv) in [(1u64, 1u64), (150_000, 12_000), (63, 15), (64_001, 16_001)] {
            let cfg = HrdSignalCfg::for_rate(rate, vbv);
            assert_eq!(cfg.bit_rate % 64, 0);
            assert_eq!(cfg.cpb_size % 16, 0);
            assert!(cfg.bit_rate >= rate.max(64) || cfg.bit_rate == 64);
            assert!(cfg.cpb_size >= vbv.max(16) || cfg.cpb_size == 16);
            // E-87 / E-88 round-trip.
            assert_eq!(
                (u64::from(cfg.bit_rate_value_minus1()) + 1) * 64,
                cfg.bit_rate
            );
            assert_eq!(
                (u64::from(cfg.cpb_size_value_minus1()) + 1) * 16,
                cfg.cpb_size
            );
        }
    }

    #[test]
    fn sei_payloads_parse_back_through_the_decode_side() {
        let bp = buffering_period_payload(22_500, 100);
        let parsed = BufferingPeriodSei::parse(&bp, &common(), 1).expect("BP parse");
        assert_eq!(parsed.bp_seq_parameter_set_id, 0);
        assert!(!parsed.concatenation_flag);
        assert_eq!(parsed.au_cpb_removal_delay_delta_minus1, 0);
        assert_eq!(parsed.nal_cpb.len(), 1);
        assert_eq!(
            (parsed.nal_cpb[0].delay, parsed.nal_cpb[0].offset),
            (22_500, 100)
        );

        let pt = pic_timing_payload(41, 3);
        assert_eq!(pt.len(), 5, "40-bit body is exactly byte aligned");
        let parsed = PicTimingSei::parse(&pt, &common(), false).expect("PT parse");
        assert_eq!(parsed.au_cpb_removal_delay_minus1, Some(41));
        assert_eq!(parsed.pic_dpb_output_delay, Some(3));
    }

    #[test]
    fn sei_prefix_nal_frames_multiple_payloads() {
        let nal = sei_prefix_nal(&[
            (SEI_BUFFERING_PERIOD, buffering_period_payload(1000, 0)),
            (SEI_PIC_TIMING, pic_timing_payload(0, 0)),
        ]);
        assert_eq!(nal[0] >> 1, PREFIX_SEI_NUT);
        let rbsp = crate::nal::strip_emulation_prevention(&nal[2..]);
        let msgs = crate::sei::parse_sei_rbsp(&rbsp, crate::sei::SeiNalType::Prefix).expect("walk");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].payload_type, u32::from(SEI_BUFFERING_PERIOD));
        assert_eq!(msgs[1].payload_type, u32::from(SEI_PIC_TIMING));
    }

    #[test]
    fn clock_initial_cap_is_the_full_cpb() {
        // 64 kb/s, 16 kbit CPB, 25 fps: delay0 = 90000·16000/64000 =
        // 22500 (250 ms · 90 kHz... exactly the CPB time-equivalent).
        let clock = HrdClock::new(
            HrdSignalCfg {
                bit_rate: 64_000,
                cpb_size: 16_000,
                cbr: false,
            },
            25,
            1,
        );
        assert_eq!(clock.delay0_90k, 22_500);
        assert_eq!(clock.frame_cap(), 16_000);
        assert_eq!(clock.au_cpb_removal_delay_minus1(), 0);
    }

    #[test]
    fn clock_never_allows_an_underflowing_au_and_refills_at_bitrate() {
        let mut clock = HrdClock::new(
            HrdSignalCfg {
                bit_rate: 64_000,
                cpb_size: 16_000,
                cbr: false,
            },
            25,
            1,
        );
        clock.begin_buffering_period();
        // Independent replay: final arrival must never pass nominal
        // removal whatever sizes are pushed, as long as each stays at
        // its cap.
        let sizes = [16_000u64, 2_560, 100, 2_560, 2_560, 2_500, 0, 2_560];
        for (i, &want) in sizes.iter().enumerate() {
            let cap = clock.frame_cap();
            let bits = want.min(cap);
            let removal = clock.removal_units(clock.next_decode_index());
            clock.push_au(bits);
            assert!(
                clock.final_arrival <= removal,
                "AU {i}: arrival past removal"
            );
        }
        // After a max-size AU the cap collapses to one tick of fill.
        let mut clock2 = HrdClock::new(
            HrdSignalCfg {
                bit_rate: 64_000,
                cpb_size: 16_000,
                cbr: false,
            },
            25,
            1,
        );
        clock2.begin_buffering_period();
        clock2.push_au(16_000);
        assert_eq!(clock2.frame_cap(), 64_000 / 25, "one ClockTick of fill");
    }

    #[test]
    fn clock_pt_delays_count_from_the_latest_buffering_period() {
        let mut clock = HrdClock::new(
            HrdSignalCfg {
                bit_rate: 64_000,
                cpb_size: 16_000,
                cbr: false,
            },
            30,
            1,
        );
        clock.begin_buffering_period();
        clock.push_au(4_000); // AU 0 (BP)
        assert_eq!(clock.au_cpb_removal_delay_minus1(), 0); // AU 1
        clock.push_au(2_000);
        assert_eq!(clock.au_cpb_removal_delay_minus1(), 1); // AU 2
                                                            // AU 2 opens a new period: its own PT still counts from AU 0.
        let (delay, offset) = clock.begin_buffering_period();
        assert!(delay >= 1 && delay <= clock.delay0_90k);
        assert_eq!(
            u64::from(delay) + u64::from(offset),
            u64::from(clock.delay0_90k)
        );
        assert_eq!(clock.au_cpb_removal_delay_minus1(), 1);
        clock.push_au(2_000);
        // AU 3 counts from the new period's AU 2.
        assert_eq!(clock.au_cpb_removal_delay_minus1(), 0);
    }

    #[test]
    fn mid_stream_period_delay_respects_c18() {
        let mut clock = HrdClock::new(
            HrdSignalCfg {
                bit_rate: 64_000,
                cpb_size: 16_000,
                cbr: false,
            },
            25,
            1,
        );
        clock.begin_buffering_period();
        // Run the buffer near-empty: a max AU then several cap-sized.
        clock.push_au(16_000);
        for _ in 0..3 {
            let cap = clock.frame_cap();
            clock.push_au(cap);
        }
        // The next period's delay is C-18-bounded: with the bucket
        // running at the wire rate, deltaTime90k is one tick =
        // 90000/25 = 3600, well under delay0 = 22500.
        let (delay, offset) = clock.begin_buffering_period();
        assert_eq!(delay, 3_600);
        assert_eq!(u64::from(delay) + u64::from(offset), 22_500);
    }

    #[test]
    fn cbr_clock_pads_underruns_and_never_underflows() {
        let cfg = HrdSignalCfg {
            bit_rate: 64_000,
            cpb_size: 16_000,
            cbr: true,
        };
        let mut clock = HrdClock::new(cfg, 25, 1);
        clock.begin_buffering_period();
        // A tiny AU underruns the constant channel: the filler floor
        // must ask for enough bits that cumulative arrival reaches
        // removal(m + 1) - CpbSize/BitRate.
        let au = 200u64;
        let pad = clock.cbr_filler_bits(au);
        assert!(pad > 0, "underrun must demand filler");
        // A whole-CPB AU never needs padding.
        assert_eq!(clock.cbr_filler_bits(16_000), 0);
        // Replay: pushing (au + pad)-sized AUs keeps both C.4 sides.
        for i in 0..20u64 {
            let coded = 200 + (i % 3) * 400;
            let cap = clock.frame_cap();
            let bits = coded.min(cap);
            let pad = clock.cbr_filler_bits(bits);
            let total = bits + pad;
            assert!(total <= cap + 56, "filler stays within the cap slack");
            let removal = clock.removal_units(clock.next_decode_index());
            let next_removal = clock.removal_units(clock.next_decode_index() + 1);
            clock.push_au(total);
            // No underflow...
            assert!(clock.final_arrival <= removal, "AU {i}: underflow");
            // ...and the back-to-back channel never runs a full CPB
            // ahead of the next removal (the overflow floor).
            let bits_to_units = 90_000u128 * 25u128;
            assert!(
                clock.final_arrival + u128::from(cfg.cpb_size) * bits_to_units >= next_removal,
                "AU {i}: cumulative arrival fell behind the overflow floor"
            );
        }
    }

    #[test]
    fn cbr_flag_roundtrips_through_hrd_parameters() {
        use crate::bitreader::BitReader;
        use crate::hrd::HrdParameters;
        let cfg = HrdSignalCfg::for_rate(150_000, 12_000).with_cbr(true);
        let mut w = BitWriter::new();
        write_hrd_parameters(&mut w, &cfg);
        w.rbsp_trailing_bits();
        let bytes = w.finish();
        let mut br = BitReader::new(&bytes);
        let hrd = HrdParameters::parse(&mut br, true, 0, None).expect("parse");
        assert!(hrd.sub_layers[0].nal_hrd.as_ref().expect("sched").cpb[0].cbr_flag);
    }

    #[test]
    fn filler_nal_has_requested_framed_size() {
        for want in [0usize, 3, 7, 8, 100] {
            let nal = filler_data_nal_framed(want);
            assert_eq!(nal.len(), want.max(7));
            assert_eq!(&nal[..4], &[0, 0, 0, 1]);
            assert_eq!(nal[4] >> 1, 38, "FD_NUT");
            assert_eq!(*nal.last().unwrap(), 0x80);
            assert!(nal[6..nal.len() - 1].iter().all(|&b| b == 0xFF));
        }
    }

    #[test]
    fn hrd_parameters_write_parses_back() {
        use crate::bitreader::BitReader;
        use crate::hrd::HrdParameters;
        let cfg = HrdSignalCfg::for_rate(150_000, 12_000);
        let mut w = BitWriter::new();
        write_hrd_parameters(&mut w, &cfg);
        w.rbsp_trailing_bits();
        let bytes = w.finish();
        let mut br = BitReader::new(&bytes);
        let hrd = HrdParameters::parse(&mut br, true, 0, None).expect("parse");
        let common = hrd.common.expect("common info");
        assert!(common.nal_hrd_parameters_present_flag);
        assert!(!common.vcl_hrd_parameters_present_flag);
        assert!(!common.sub_pic_hrd_params_present_flag);
        assert_eq!(common.bit_rate_scale, 0);
        assert_eq!(common.cpb_size_scale, 0);
        assert_eq!(
            common.initial_cpb_removal_delay_length_minus1,
            INITIAL_CPB_REMOVAL_DELAY_LENGTH - 1
        );
        assert_eq!(
            common.au_cpb_removal_delay_length_minus1,
            AU_CPB_REMOVAL_DELAY_LENGTH - 1
        );
        assert_eq!(
            common.dpb_output_delay_length_minus1,
            DPB_OUTPUT_DELAY_LENGTH - 1
        );
        let sl = &hrd.sub_layers[0];
        assert!(sl.fixed_pic_rate_general_flag);
        assert_eq!(sl.elemental_duration_in_tc_minus1, Some(0));
        assert_eq!(sl.cpb_cnt_minus1, 0);
        let cpb = &sl.nal_hrd.as_ref().expect("NAL sched").cpb[0];
        assert_eq!(cpb.bit_rate_value_minus1, cfg.bit_rate_value_minus1());
        assert_eq!(cpb.cpb_size_value_minus1, cfg.cpb_size_value_minus1());
        assert!(!cpb.cbr_flag);
    }
}
