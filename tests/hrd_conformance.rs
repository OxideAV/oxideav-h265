//! HRD conformance of the encoder's `with_hrd` streams, self-checked
//! per Annex C from the BITSTREAM alone: the SPS VUI `hrd_parameters`
//! and the per-AU buffering-period / pic-timing SEI are parsed back
//! through the crate's decode side, then the §C.2 CPB arithmetic is
//! replayed exactly (integer arithmetic over a common denominator, no
//! rounding) and the §C.4 bitstream-conformance conditions asserted:
//!
//! * condition 2 — the CPB never overflows (content is checked at
//!   every final-arrival instant, the local maxima of the piecewise
//!   linear fill);
//! * condition 3 — the CPB never underflows (`AuNominalRemovalTime >=
//!   AuFinalArrivalTime` for every AU, `low_delay_hrd_flag == 0`);
//! * eq. C-18 — every mid-stream buffering period's
//!   `InitCpbRemovalDelay <= Ceil(deltaTime90k)`;
//! * §D.3.2 — `delay + offset` constant, delays nonzero and at most
//!   the CPB time-equivalent;
//! * eq. C-15 / §C.4 condition 11 — DPB output times strictly
//!   increase in display (POC) order.

use oxideav_core::{CodecParameters, Error, Frame, PixelFormat, VideoFrame, VideoPlane};
use oxideav_h265::encoder::inter::{LowDelayPEncoder, YuvFrame};
use oxideav_h265::encoder::loopfilter::LoopFilterCfg;
use oxideav_h265::encoder::pyramid::PyramidEncoder;
use oxideav_h265::encoder::rate::RateControlCfg;
use oxideav_h265::hrd::HrdCommonInfo;
use oxideav_h265::sei::{parse_sei_rbsp, BufferingPeriodSei, PicTimingSei, SeiNalType};
use oxideav_h265::sps::SeqParameterSet;
use oxideav_h265::{collect_nal_units, decode_annexb_sequence, make_encoder};

const W: usize = 64;
const H: usize = 48;

/// Deterministic xorshift noise.
fn noise(seed: &mut u32) -> u8 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    (*seed >> 24) as u8
}

/// A moving-square scene over a textured, lightly noisy background.
fn clip(n: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut seed = 0x51D5_EED5u32;
    (0..n)
        .map(|t| {
            let y: Vec<u8> = (0..W * H)
                .map(|i| {
                    let (x, yy) = (i % W, i / W);
                    let base = ((x * 5 + yy * 3) % 170) as i32 + i32::from(noise(&mut seed) % 14);
                    let (sx, sy) = ((3 + t * 2) % (W - 12), (6 + t) % (H - 12));
                    if x >= sx && x < sx + 10 && yy >= sy && yy < sy + 10 {
                        (base + 70).clamp(0, 255) as u8
                    } else {
                        base.clamp(0, 255) as u8
                    }
                })
                .collect();
            let cb: Vec<u8> = (0..W * H / 4).map(|i| (90 + (i + t) % 60) as u8).collect();
            let cr: Vec<u8> = (0..W * H / 4).map(|i| (160 - (i + t) % 50) as u8).collect();
            (y, cb, cr)
        })
        .collect()
}

fn frames(planes: &[(Vec<u8>, Vec<u8>, Vec<u8>)]) -> Vec<YuvFrame<'_>> {
    planes
        .iter()
        .map(|(y, cb, cr)| YuvFrame { y, cb, cr })
        .collect()
}

/// The signalled schedule recovered from the SPS VUI.
struct Schedule {
    bit_rate: u64,
    cpb_size: u64,
    num_units: u64,
    time_scale: u64,
    common: HrdCommonInfo,
}

/// One access unit as carried by the stream.
struct Au {
    /// All bits of the Type II access unit (start codes included).
    bits: u64,
    /// The §D.2.2 message, when the AU carries one.
    bp: Option<BufferingPeriodSei>,
    /// The §D.2.3 message (mandatory under CpbDpbDelaysPresentFlag).
    pt: PicTimingSei,
    /// The VCL NAL type (19/20 = IDR).
    idr: bool,
}

/// Walk the Annex B stream: recover the schedule from the SPS and
/// split the NAL units into access units at each VCL boundary
/// (this encoder emits exactly one VCL NAL per AU).
fn analyze(stream: &[u8]) -> (Schedule, Vec<Au>) {
    let units = collect_nal_units(stream).expect("NAL walk");
    let sps_unit = units
        .iter()
        .find(|u| u.header.nal_unit_type == 33)
        .expect("SPS present");
    let sps = SeqParameterSet::parse(&sps_unit.rbsp).expect("SPS parse");
    let vui = sps.vui_parameters.expect("VUI present");
    let timing = vui.timing_info.expect("vui_timing_info");
    let hrd = timing.hrd_parameters.as_ref().expect("hrd_parameters");
    let common = hrd.common.expect("common info");
    assert!(common.nal_hrd_parameters_present_flag);
    assert!(!common.vcl_hrd_parameters_present_flag);
    assert!(!common.sub_pic_hrd_params_present_flag);
    let sl = &hrd.sub_layers[0];
    assert!(sl.fixed_pic_rate_general_flag, "fixed picture rate");
    assert_eq!(sl.elemental_duration_in_tc_minus1, Some(0));
    assert_eq!(sl.cpb_cnt_minus1, 0, "one delivery schedule");
    let cpb = &sl.nal_hrd.as_ref().expect("NAL schedule").cpb[0];
    assert!(!cpb.cbr_flag, "VBR schedule");
    // Eqs. E-87 / E-88.
    let bit_rate =
        (u64::from(cpb.bit_rate_value_minus1) + 1) << (6 + u32::from(common.bit_rate_scale));
    let cpb_size =
        (u64::from(cpb.cpb_size_value_minus1) + 1) << (4 + u32::from(common.cpb_size_scale));
    let schedule = Schedule {
        bit_rate,
        cpb_size,
        num_units: u64::from(timing.num_units_in_tick),
        time_scale: u64::from(timing.time_scale),
        common,
    };

    let mut aus = Vec::new();
    let mut cur_bits = 0u64;
    let mut cur_bp = None;
    let mut cur_pt: Option<PicTimingSei> = None;
    for unit in &units {
        cur_bits += (4 + 2 + unit.escaped.len()) as u64;
        match unit.header.nal_unit_type {
            39 => {
                for msg in parse_sei_rbsp(&unit.rbsp, SeiNalType::Prefix).expect("SEI walk") {
                    let body = match &msg.payload {
                        oxideav_h265::sei::SeiPayload::Reserved { data, .. } => data.clone(),
                        other => panic!("unexpected typed SEI payload {other:?}"),
                    };
                    match msg.payload_type {
                        0 => {
                            cur_bp = Some(
                                BufferingPeriodSei::parse(&body, &common, 1).expect("BP parse"),
                            )
                        }
                        1 => {
                            cur_pt =
                                Some(PicTimingSei::parse(&body, &common, false).expect("PT parse"))
                        }
                        other => panic!("unexpected SEI payload type {other}"),
                    }
                }
            }
            t if t <= 31 => {
                // VCL: the AU is complete.
                aus.push(Au {
                    bits: cur_bits,
                    bp: cur_bp.take(),
                    pt: cur_pt.take().expect("pic timing on every AU"),
                    idr: (19..=20).contains(&t),
                });
                cur_bits = 0;
            }
            _ => {}
        }
    }
    assert_eq!(cur_bits, 0, "no trailing non-VCL NAL units");
    (schedule, aus)
}

/// The exact §C.2 replay. Times are u128 integers in units of
/// `1 / (90000 · time_scale · BitRate)` seconds. Returns each AU's
/// `(AuNominalRemovalTime, DpbOutputTime)` in those units.
fn replay_annex_c(sch: &Schedule, aus: &[Au]) -> Vec<(u128, u128)> {
    let ts_br = u128::from(sch.time_scale) * u128::from(sch.bit_rate);
    let delay_units = |d90k: u32| u128::from(d90k) * ts_br;
    let tick = 90_000u128 * u128::from(sch.num_units) * u128::from(sch.bit_rate);
    let bits_units = |bits: u64| u128::from(bits) * 90_000u128 * u128::from(sch.time_scale);
    let cpb_time_equiv = 90_000u128 * u128::from(sch.cpb_size) / u128::from(sch.bit_rate);

    let au_len = sch.common.au_cpb_removal_delay_length_minus1 + 1;
    let mut out = Vec::with_capacity(aus.len());
    // D-1 state (every picture is TemporalId 0 and non-discardable,
    // so prevNonDiscardablePic is simply the previous AU).
    let (mut prev_val_field, mut prev_msb, mut prev_bp_reset) = (0u64, 0u64, false);
    // Buffering-period state: nominal removal time of the first AU
    // of the CURRENT buffering period (the C-10 baseTime when the
    // next period opens, the C-11 base within the period).
    let mut first_curr_bp_removal = 0u128;
    let (mut cur_delay, mut cur_offset) = (0u32, 0u32);
    let mut delay_plus_offset: Option<u64> = None;
    let mut final_arrival = 0u128;
    let mut removals: Vec<u128> = Vec::with_capacity(aus.len());
    let mut final_arrivals: Vec<u128> = Vec::with_capacity(aus.len());

    for (n, au) in aus.iter().enumerate() {
        // §D.3.3 (D-1 / D-2): AuCpbRemovalDelayVal.
        let val = if n == 0 {
            0
        } else {
            let field = u64::from(au.pt.au_cpb_removal_delay_minus1.expect("PT delay"));
            let msb = if prev_bp_reset {
                0
            } else if field <= prev_val_field {
                prev_msb + (1u64 << au_len)
            } else {
                prev_msb
            };
            prev_msb = msb;
            prev_val_field = field;
            msb + field + 1
        };
        if n == 0 {
            prev_val_field = 0;
            prev_msb = 0;
        }

        // §C.2.3 nominal removal time.
        let removal = if let Some(bp) = &au.bp {
            let pair = &bp.nal_cpb[0];
            // §D.3.2 sanity on the signalled pair.
            assert!(pair.delay > 0, "AU {n}: initial delay must be nonzero");
            assert!(
                u128::from(pair.delay) <= cpb_time_equiv,
                "AU {n}: initial delay exceeds the CPB time-equivalent"
            );
            assert!(!bp.concatenation_flag);
            let removal = if n == 0 {
                delay_units(pair.delay) // C-9
            } else {
                // C-10 with concatenation_flag == 0: baseTime is the
                // nominal removal of the first AU of the period this
                // one closes (AuCpbRemovalDelayVal counts from it).
                first_curr_bp_removal + tick * u128::from(val)
            };
            // C-18: delay <= Ceil(deltaTime90k) for n > 0.
            if n > 0 {
                let delta_units = removal
                    .checked_sub(final_arrival)
                    .expect("BP AU: removal before the previous AU's final arrival");
                let ceil_90k = delta_units.div_ceil(ts_br);
                assert!(
                    u128::from(pair.delay) <= ceil_90k,
                    "AU {n}: C-18 violated (delay {} > ceil(deltaTime90k) {})",
                    pair.delay,
                    ceil_90k
                );
            }
            // §D.3.2: delay + offset constant over the CVS — this
            // encoder holds it constant over the whole stream.
            let sum = u64::from(pair.delay) + u64::from(pair.offset);
            match delay_plus_offset {
                None => delay_plus_offset = Some(sum),
                Some(prev) => assert_eq!(prev, sum, "AU {n}: delay + offset drifted"),
            }
            first_curr_bp_removal = removal;
            (cur_delay, cur_offset) = (pair.delay, pair.offset);
            removal
        } else {
            // C-11.
            first_curr_bp_removal + tick * u128::from(val)
        };

        // §C.2.2 arrival times (VBR: C-4..C-7).
        let throttle = if au.bp.is_some() {
            delay_units(cur_delay) // C-7
        } else {
            delay_units(cur_delay) + delay_units(cur_offset) // C-6
        };
        let init_arrival = if n == 0 {
            0
        } else {
            final_arrival.max(removal.saturating_sub(throttle))
        };
        final_arrival = init_arrival + bits_units(au.bits); // C-8

        // §C.4 condition 3: no underflow.
        assert!(
            final_arrival <= removal,
            "AU {n}: final arrival past nominal removal (underflow)"
        );

        prev_bp_reset = au.bp.is_some();
        removals.push(removal);
        final_arrivals.push(final_arrival);
        let out_delay = u128::from(au.pt.pic_dpb_output_delay.expect("PT output delay"));
        out.push((removal, removal + tick * out_delay)); // C-15
    }

    // §C.4 condition 2: no overflow — check the CPB content just
    // after every final-arrival instant (the local maxima of the
    // piecewise-linear fill: content only grows while an AU is
    // arriving and drops instantaneously at removals). Arrivals are
    // sequential, so at AU k's final arrival exactly AUs 0..=k have
    // fully arrived; removed are the AUs whose (nominal == actual,
    // `low_delay_hrd_flag == 0`) removal time has passed.
    for k in 0..aus.len() {
        let fa = final_arrivals[k];
        let arrived: u64 = aus[..=k].iter().map(|a| a.bits).sum();
        let removed: u64 = aus
            .iter()
            .zip(&removals)
            .filter(|(_, r)| **r <= fa)
            .map(|(a, _)| a.bits)
            .sum();
        assert!(
            arrived - removed <= sch.cpb_size,
            "AU {k}: CPB content {} exceeds CpbSize {}",
            arrived - removed,
            sch.cpb_size
        );
    }
    out
}

/// Assert conformance of `stream` and return per-AU
/// `(removal, output)` times; `display` maps decode order to display
/// order for the output-order check.
fn assert_conformant(stream: &[u8], display: &[usize], label: &str) {
    let (sch, aus) = analyze(stream);
    assert_eq!(aus.len(), display.len(), "{label}: AU count");
    // BP exactly on IRAP AUs (§D.3.2 presence rule for this encoder).
    for (n, au) in aus.iter().enumerate() {
        assert_eq!(
            au.bp.is_some(),
            au.idr,
            "{label} AU {n}: buffering period iff IRAP"
        );
    }
    let times = replay_annex_c(&sch, &aus);
    // §C.4 condition 11: output times strictly increase in display
    // (POC) order.
    let mut by_display: Vec<(usize, u128)> = display
        .iter()
        .zip(&times)
        .map(|(&d, &(_, out))| (d, out))
        .collect();
    by_display.sort_unstable();
    for pair in by_display.windows(2) {
        assert!(
            pair[0].1 < pair[1].1,
            "{label}: output times not increasing in display order ({pair:?})"
        );
    }
    // Removal cadence: one elemental tick per AU in decode order.
    let tick = 90_000u128 * u128::from(sch.num_units) * u128::from(sch.bit_rate);
    for (n, pair) in times.windows(2).enumerate() {
        assert_eq!(
            pair[1].0 - pair[0].0,
            tick,
            "{label}: AU {} removal cadence",
            n + 1
        );
    }
}

/// One display-indexed reconstruction triple.
type Recon = (usize, Vec<u8>, Vec<u8>, Vec<u8>);

/// Decode through the crate's own decoder and require byte-exact
/// output against the expected reconstructions (SEI must not disturb
/// decode).
fn assert_decodes_exactly(stream: &[u8], recons: &[Recon], label: &str) {
    let decoded = decode_annexb_sequence(stream).expect("decode");
    assert_eq!(decoded.len(), recons.len(), "{label}: frame count");
    let mut sorted: Vec<_> = recons.to_vec();
    sorted.sort_unstable_by_key(|(d, ..)| *d);
    for (dec, (d, y, cb, cr)) in decoded.iter().zip(&sorted) {
        let mut expect = y.clone();
        expect.extend_from_slice(cb);
        expect.extend_from_slice(cr);
        assert_eq!(
            dec.picture.to_planar_u8().expect("8-bit"),
            expect,
            "{label} display {d}"
        );
    }
}

#[test]
fn low_delay_hrd_streams_conform() {
    let planes = clip(30);
    for (rate, bufsize, gop, bslices, lf, aq, label) in [
        (
            150_000u64,
            12_000u64,
            10usize,
            false,
            LoopFilterCfg::off(),
            0u8,
            "gop10 plain",
        ),
        (
            300_000,
            30_000,
            6,
            true,
            LoopFilterCfg::all(),
            2,
            "gop6 b+filters+aq",
        ),
        (
            80_000,
            8_000,
            5,
            false,
            LoopFilterCfg::off(),
            0,
            "tight buffer gop5",
        ),
    ] {
        let mut enc = LowDelayPEncoder::new(W, H, 26, gop)
            .expect("encoder")
            .with_b_slices(bslices)
            .with_loop_filters(lf)
            .with_aq(aq)
            .with_frame_rate(25, 1)
            .with_rate_control(&RateControlCfg::new(rate, 25, 1).with_vbv(bufsize))
            .with_hrd(true);
        let mut stream = Vec::new();
        let mut recons = Vec::new();
        for (d, f) in frames(&planes).iter().enumerate() {
            let out = enc.encode_frame(f).expect("encode");
            stream.extend_from_slice(&out.au);
            recons.push((d, out.recon.y, out.recon.cb, out.recon.cr));
        }
        let display: Vec<usize> = (0..planes.len()).collect();
        assert_conformant(&stream, &display, label);
        assert_decodes_exactly(&stream, &recons, label);
    }
}

#[test]
fn pyramid_hrd_streams_conform() {
    let planes = clip(21); // IDR + mini-GOPs + tail
    for (gop, rate, bufsize, lf, aq, label) in [
        (
            4usize,
            200_000u64,
            20_000u64,
            LoopFilterCfg::off(),
            0u8,
            "pyr4",
        ),
        (
            8,
            150_000,
            18_000,
            LoopFilterCfg::all(),
            2,
            "pyr8 filters+aq",
        ),
    ] {
        let mut enc = PyramidEncoder::new(W, H, 26, gop)
            .expect("encoder")
            .with_loop_filters(lf)
            .with_aq(aq)
            .with_frame_rate(30, 1)
            .with_rate_control(&RateControlCfg::new(rate, 30, 1).with_vbv(bufsize))
            .with_hrd(true);
        let mut stream = Vec::new();
        let mut display = Vec::new();
        let mut recons = Vec::new();
        let mut push = |aus: Vec<oxideav_h265::encoder::pyramid::PyramidAu>| {
            for au in aus {
                stream.extend_from_slice(&au.au);
                display.push(au.display_order);
                recons.push((au.display_order, au.recon.y, au.recon.cb, au.recon.cr));
            }
        };
        for f in frames(&planes) {
            push(enc.encode_frame(&f).expect("encode"));
        }
        push(enc.flush());
        assert_eq!(display.len(), planes.len(), "{label}");
        assert_conformant(&stream, &display, label);
        assert_decodes_exactly(&stream, &recons, label);
        // The pyramid's output schedule: output tick == reorder +
        // display for every AU (eq. C-15 with the emitted delays).
        let (sch, aus) = analyze(&stream);
        let times = replay_annex_c(&sch, &aus);
        let tick = 90_000u128 * u128::from(sch.num_units) * u128::from(sch.bit_rate);
        let reorder = gop.trailing_zeros() as u128;
        for (m, (&d, &(removal, out))) in display.iter().zip(&times).enumerate() {
            assert_eq!(
                (out - times[0].0) / tick,
                reorder + d as u128,
                "{label}: AU {m} output tick"
            );
            assert_eq!((removal - times[0].0) / tick, m as u128);
        }
    }
}

fn video_frame(y: &[u8], cb: &[u8], cr: &[u8]) -> Frame {
    let plane = |data: &[u8], stride: usize| VideoPlane {
        stride,
        data: data.to_vec(),
    };
    Frame::Video(VideoFrame {
        pts: None,
        planes: vec![plane(y, W), plane(cb, W / 2), plane(cr, W / 2)],
    })
}

fn base_params() -> CodecParameters {
    let mut p = CodecParameters::video("h265".into());
    p.width = Some(W as u32);
    p.height = Some(H as u32);
    p.pixel_format = Some(PixelFormat::Yuv420P);
    p
}

/// Registry `hrd` option: the all-intra mode emits a buffering
/// period on EVERY access unit (each frame an IRAP starting its own
/// CVS) — the C-18 bound and the §D.3.2 constant `delay + offset`
/// are exercised on every frame.
#[test]
fn registry_intra_hrd_conforms() {
    let planes = clip(10);
    let mut p = base_params();
    p.options.insert("mode", "intra");
    p.options.insert("bitrate", "400k");
    p.options.insert("bufsize", "60k");
    p.options.insert("fps", "25");
    p.options.insert("hrd", "1");
    let mut enc = make_encoder(&p).expect("encoder");
    let mut stream = Vec::new();
    for (y, cb, cr) in &planes {
        enc.send_frame(&video_frame(y, cb, cr)).expect("send");
        while let Ok(pkt) = enc.receive_packet() {
            assert!(pkt.data.len() as u64 * 8 <= 60_000);
            stream.extend_from_slice(&pkt.data);
        }
    }
    let display: Vec<usize> = (0..planes.len()).collect();
    assert_conformant(&stream, &display, "registry intra hrd");
    assert_eq!(
        decode_annexb_sequence(&stream).expect("decode").len(),
        planes.len()
    );
}

/// Registry `hrd` over the inter and pyramid paths.
#[test]
fn registry_inter_and_pyramid_hrd_conform() {
    let planes = clip(9);
    for extra in [vec![("gop", "4")], vec![("pyramid", "4")]] {
        let mut p = base_params();
        p.options.insert("mode", "inter");
        p.options.insert("bitrate", "250k");
        p.options.insert("bufsize", "35k");
        p.options.insert("fps", "30000/1001");
        p.options.insert("hrd", "true");
        for (k, v) in &extra {
            p.options.insert(*k, *v);
        }
        let mut enc = make_encoder(&p).expect("encoder");
        let mut stream = Vec::new();
        let mut display = Vec::new();
        let drain = |enc: &mut Box<dyn oxideav_core::Encoder>,
                     stream: &mut Vec<u8>,
                     display: &mut Vec<usize>| {
            while let Ok(pkt) = enc.receive_packet() {
                assert!(pkt.data.len() as u64 * 8 <= 35_000, "{extra:?}");
                stream.extend_from_slice(&pkt.data);
                display.push(pkt.pts.expect("pts") as usize);
            }
        };
        for (y, cb, cr) in &planes {
            enc.send_frame(&video_frame(y, cb, cr)).expect("send");
            drain(&mut enc, &mut stream, &mut display);
        }
        enc.flush().expect("flush");
        drain(&mut enc, &mut stream, &mut display);
        assert_eq!(display.len(), planes.len(), "{extra:?}");
        assert_conformant(&stream, &display, &format!("registry {extra:?}"));
        assert_eq!(
            decode_annexb_sequence(&stream).expect("decode").len(),
            planes.len()
        );
    }
}

/// `hrd` prerequisites: the registry rejects partial configurations,
/// and the direct APIs error at the first frame.
#[test]
fn hrd_prerequisites_are_enforced() {
    for missing in [
        vec![("bitrate", "200k"), ("bufsize", "30k")], // no fps
        vec![("bitrate", "200k"), ("fps", "25")],      // no bufsize
        vec![("bufsize", "30k"), ("fps", "25")],       // no bitrate
    ] {
        let mut p = base_params();
        p.options.insert("mode", "inter");
        p.options.insert("hrd", "1");
        for (k, v) in &missing {
            p.options.insert(*k, *v);
        }
        assert!(
            matches!(make_encoder(&p), Err(Error::InvalidData(_))),
            "{missing:?}"
        );
    }
    // Direct API: with_hrd without VBV / frame rate errors on encode.
    let planes = clip(1);
    let (y, cb, cr) = &planes[0];
    let f = YuvFrame { y, cb, cr };
    let mut no_vbv = LowDelayPEncoder::new(W, H, 26, 0)
        .expect("encoder")
        .with_frame_rate(25, 1)
        .with_rate_control(&RateControlCfg::new(200_000, 25, 1))
        .with_hrd(true);
    assert!(no_vbv.encode_frame(&f).is_err(), "missing VBV");
    let mut no_fps = PyramidEncoder::new(W, H, 26, 4)
        .expect("encoder")
        .with_rate_control(&RateControlCfg::new(200_000, 25, 1).with_vbv(30_000))
        .with_hrd(true);
    assert!(no_fps.encode_frame(&f).is_err(), "missing frame rate");
}
