//! ABR rate-control validation: the `bitrate` targeting introduced in
//! `encoder::rate` must (a) land the produced stream on the requested
//! average bitrate, (b) keep every stream bit-exact through the
//! crate's own decoder (rate control only moves per-slice
//! `slice_qp_delta`, so conformance is untouched), and (c) spend a
//! higher budget on measurably lower distortion.

use oxideav_core::{CodecParameters, Error, Frame, PixelFormat, VideoFrame, VideoPlane};
use oxideav_h265::decode_annexb_sequence;
use oxideav_h265::encoder::inter::{LowDelayPEncoder, YuvFrame};
use oxideav_h265::encoder::loopfilter::LoopFilterCfg;
use oxideav_h265::encoder::rate::RateControlCfg;
use oxideav_h265::make_encoder;

const W: usize = 64;
const H: usize = 64;
const FPS: u32 = 25;

/// Deterministic xorshift noise.
fn noise(seed: &mut u32) -> u8 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    (*seed >> 24) as u8
}

/// A moving-square scene over a textured, lightly noisy background —
/// enough temporal churn that the coded size genuinely responds to
/// QP.
fn clip(n: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut seed = 0x1234_5678u32;
    (0..n)
        .map(|t| {
            let y: Vec<u8> = (0..W * H)
                .map(|i| {
                    let (x, yy) = (i % W, i / W);
                    let base = ((x * 3 + yy * 2) % 180) as i32 + i32::from(noise(&mut seed) % 16);
                    let (sx, sy) = ((5 + t * 2) % (W - 14), (8 + t) % (H - 14));
                    if x >= sx && x < sx + 12 && yy >= sy && yy < sy + 12 {
                        (base + 60).clamp(0, 255) as u8
                    } else {
                        base.clamp(0, 255) as u8
                    }
                })
                .collect();
            let cb: Vec<u8> = (0..W * H / 4)
                .map(|i| (96 + (i % (W / 2)) * 3 % 64 + t) as u8)
                .collect();
            let cr: Vec<u8> = (0..W * H / 4)
                .map(|i| (150u8.wrapping_sub((i / (W / 2)) as u8)).wrapping_add(t as u8))
                .collect();
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

/// Encode `planes` through a rate-controlled low-delay encoder and
/// return (per-frame AU sizes in bits, per-frame QPs, recon SSD sum,
/// whole stream).
fn encode_abr(
    planes: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
    bits_per_second: u64,
    gop: usize,
    b_slices: bool,
    lf: LoopFilterCfg,
) -> (Vec<u64>, Vec<i32>, u64, Vec<u8>) {
    let mut enc = LowDelayPEncoder::new(W, H, 26, gop)
        .expect("encoder")
        .with_b_slices(b_slices)
        .with_loop_filters(lf)
        .with_rate_control(&RateControlCfg::new(bits_per_second, FPS, 1));
    let mut bits = Vec::new();
    let mut qps = Vec::new();
    let mut ssd = 0u64;
    let mut stream = Vec::new();
    for (f, (y, _, _)) in frames(planes).iter().zip(planes) {
        let out = enc.encode_frame(f).expect("encode");
        bits.push(out.au.len() as u64 * 8);
        qps.push(out.qp);
        ssd += y
            .iter()
            .zip(&out.recon.y)
            .map(|(&a, &b)| {
                let d = i64::from(a) - i64::from(b);
                (d * d) as u64
            })
            .sum::<u64>();
        stream.extend_from_slice(&out.au);
    }
    (bits, qps, ssd, stream)
}

#[test]
fn abr_hits_target_on_low_delay_gops() {
    let planes = clip(40);
    for target in [100_000u64, 300_000] {
        let (bits, qps, _, _) = encode_abr(&planes, target, 8, false, LoopFilterCfg::off());
        let total: u64 = bits.iter().sum();
        let wanted = target * planes.len() as u64 / u64::from(FPS);
        let err_pct = (total as i64 - wanted as i64).unsigned_abs() * 100 / wanted;
        assert!(
            err_pct <= 20,
            "{target} b/s: coded {total} bits vs budget {wanted} ({err_pct}% off)"
        );
        // Steady state (model converged): the back half alone lands
        // tighter on its share of the budget.
        let back: u64 = bits[20..].iter().sum();
        let back_wanted = target * 20 / u64::from(FPS);
        let back_err = (back as i64 - back_wanted as i64).unsigned_abs() * 100 / back_wanted;
        assert!(
            back_err <= 20,
            "{target} b/s steady state: {back} vs {back_wanted} ({back_err}% off)"
        );
        // Bounded per-class QP excursions: the window is
        // `min(2 + gap, 15)` where `gap` is the frames since that
        // class (IDR every 8 with gop = 8; the rest inter) last
        // coded.
        let mut last: [Option<(i32, usize)>; 2] = [None, None];
        for (i, &qp) in qps.iter().enumerate() {
            let class = usize::from(i % 8 != 0);
            if let Some((prev, at)) = last[class] {
                let window = (2 + (i - at)).min(15) as i32;
                assert!(
                    (qp - prev).abs() <= window,
                    "class {class} QP excursion {prev} -> {qp} at frame {i} (window {window})"
                );
            }
            last[class] = Some((qp, i));
        }
    }
}

#[test]
fn abr_streams_decode_bit_exact_through_own_decoder() {
    let planes = clip(12);
    for (b_slices, lf) in [
        (false, LoopFilterCfg::off()),
        (
            true,
            LoopFilterCfg {
                deblocking: true,
                sao_luma: true,
                sao_chroma: true,
            },
        ),
    ] {
        let mut enc = LowDelayPEncoder::new(W, H, 26, 6)
            .expect("encoder")
            .with_b_slices(b_slices)
            .with_loop_filters(lf)
            .with_rate_control(&RateControlCfg::new(150_000, FPS, 1));
        let mut stream = Vec::new();
        let mut recons = Vec::new();
        for f in frames(&planes) {
            let out = enc.encode_frame(&f).expect("encode");
            stream.extend_from_slice(&out.au);
            recons.push(out.recon);
        }
        let decoded = decode_annexb_sequence(&stream).expect("decode");
        assert_eq!(decoded.len(), planes.len(), "b={b_slices}");
        for (i, (dec, rec)) in decoded.iter().zip(&recons).enumerate() {
            let mut expect = rec.y.clone();
            expect.extend_from_slice(&rec.cb);
            expect.extend_from_slice(&rec.cr);
            assert_eq!(
                dec.picture.to_planar_u8().expect("8-bit"),
                expect,
                "b={b_slices} frame {i}: decoder output == encoder reconstruction"
            );
        }
    }
}

#[test]
fn pyramid_abr_hits_target_and_roundtrips() {
    use oxideav_h265::encoder::pyramid::{encode_pyramid_with, PyramidEncoder};
    let planes = clip(33); // IDR + four GOP-8 mini-GOPs
    let frames = frames(&planes);
    let mut totals = Vec::new();
    for target in [120_000u64, 360_000] {
        let enc = PyramidEncoder::new(W, H, 26, 8)
            .expect("encoder")
            .with_rate_control(&RateControlCfg::new(target, FPS, 1));
        let out = encode_pyramid_with(enc, &frames).expect("encode");
        let total_bits = out.stream.len() as u64 * 8;
        let wanted = target * planes.len() as u64 / u64::from(FPS);
        let err_pct = (total_bits as i64 - wanted as i64).unsigned_abs() * 100 / wanted;
        assert!(
            err_pct <= 25,
            "pyramid {target} b/s: {total_bits} bits vs budget {wanted} ({err_pct}%)"
        );
        totals.push(total_bits);
        // Display-order decode == encoder reconstruction, bit-exact.
        let decoded = decode_annexb_sequence(&out.stream).expect("decode");
        assert_eq!(decoded.len(), planes.len());
        for (i, (dec, rec)) in decoded.iter().zip(&out.recon).enumerate() {
            let mut expect = rec.y.clone();
            expect.extend_from_slice(&rec.cb);
            expect.extend_from_slice(&rec.cr);
            assert_eq!(
                dec.picture.to_planar_u8().expect("8-bit"),
                expect,
                "pyramid {target} frame {i}"
            );
        }
    }
    assert!(totals[1] > totals[0], "3x the budget codes more bits");
}

#[test]
fn registry_pyramid_abr_roundtrips() {
    let planes = clip(9); // IDR + one GOP-4 mini-GOP + 4-frame tail
    let mut p = base_params();
    p.options.insert("mode", "inter");
    p.options.insert("pyramid", "4");
    p.options.insert("bitrate", "250k");
    p.options.insert("fps", "30000/1001");
    let mut enc = make_encoder(&p).expect("encoder");
    let mut stream = Vec::new();
    let mut n_packets = 0usize;
    for (y, cb, cr) in &planes {
        enc.send_frame(&video_frame(y, cb, cr)).expect("send");
        while let Ok(pkt) = enc.receive_packet() {
            stream.extend_from_slice(&pkt.data);
            n_packets += 1;
        }
    }
    enc.flush().expect("flush");
    while let Ok(pkt) = enc.receive_packet() {
        stream.extend_from_slice(&pkt.data);
        n_packets += 1;
    }
    assert_eq!(n_packets, planes.len());
    let decoded = decode_annexb_sequence(&stream).expect("decode");
    assert_eq!(decoded.len(), planes.len());
}

#[test]
fn higher_target_buys_lower_distortion() {
    let planes = clip(24);
    let (bits_lo, _, ssd_lo, _) = encode_abr(&planes, 80_000, 0, false, LoopFilterCfg::off());
    let (bits_hi, _, ssd_hi, _) = encode_abr(&planes, 400_000, 0, false, LoopFilterCfg::off());
    let (lo, hi): (u64, u64) = (bits_lo.iter().sum(), bits_hi.iter().sum());
    assert!(hi > lo, "5x the budget must code more bits ({lo} vs {hi})");
    assert!(
        ssd_hi < ssd_lo,
        "5x the budget must reduce luma SSD ({ssd_lo} -> {ssd_hi})"
    );
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

#[test]
fn registry_abr_options_encode_and_decode() {
    let planes = clip(10);
    let mut p = base_params();
    p.options.insert("mode", "inter");
    p.options.insert("gop", "5");
    p.options.insert("bitrate", "200k");
    p.options.insert("fps", "25");
    let mut enc = make_encoder(&p).expect("encoder");
    let mut stream = Vec::new();
    let mut n_packets = 0usize;
    for (y, cb, cr) in &planes {
        enc.send_frame(&video_frame(y, cb, cr)).expect("send");
        while let Ok(pkt) = enc.receive_packet() {
            stream.extend_from_slice(&pkt.data);
            n_packets += 1;
        }
    }
    assert_eq!(n_packets, planes.len());
    let decoded = decode_annexb_sequence(&stream).expect("decode");
    assert_eq!(decoded.len(), planes.len());
}

#[test]
fn registry_abr_intra_mode_adapts_qp() {
    // All-intra ABR: every frame an IDR, budget spread across them.
    let planes = clip(12);
    let mut p = base_params();
    p.options.insert("mode", "intra");
    p.options.insert("bitrate", "400k");
    let mut enc = make_encoder(&p).expect("encoder");
    let mut sizes = Vec::new();
    for (y, cb, cr) in &planes {
        enc.send_frame(&video_frame(y, cb, cr)).expect("send");
        while let Ok(pkt) = enc.receive_packet() {
            sizes.push(pkt.data.len() as u64 * 8);
        }
    }
    assert_eq!(sizes.len(), planes.len());
    let total: u64 = sizes.iter().sum();
    let wanted = 400_000u64 * planes.len() as u64 / u64::from(FPS);
    let err_pct = (total as i64 - wanted as i64).unsigned_abs() * 100 / wanted;
    assert!(
        err_pct <= 25,
        "all-intra ABR: {total} bits vs budget {wanted} ({err_pct}% off)"
    );
}

/// VBV-constrained ABR: with `RateControlCfg::with_vbv` the encoder
/// guarantees no frame exceeds the modelled decoder buffer — the
/// cold-start IDR (which overshoots a plain ABR run) is re-encoded
/// at a higher QP until it fits, and replaying the leaky bucket over
/// the whole stream never underflows.
#[test]
fn vbv_constrained_stream_never_underflows() {
    let planes = clip(30);
    let (rate, bufsize) = (150_000u64, 12_000u64);
    let per_frame_fill = (rate / u64::from(FPS)) as i64;
    // Unconstrained twin: prove the constraint has something to do.
    let mut free = LowDelayPEncoder::new(W, H, 26, 10)
        .expect("encoder")
        .with_rate_control(&RateControlCfg::new(rate, FPS, 1));
    let free_max = frames(&planes)
        .iter()
        .map(|f| free.encode_frame(f).expect("encode").au.len() as u64 * 8)
        .max()
        .unwrap();
    assert!(
        free_max > bufsize,
        "test premise: the unconstrained run must overshoot the buffer ({free_max} bits)"
    );
    // Constrained run: replay the decoder-buffer model.
    let mut enc = LowDelayPEncoder::new(W, H, 26, 10)
        .expect("encoder")
        .with_rate_control(&RateControlCfg::new(rate, FPS, 1).with_vbv(bufsize));
    let mut stream = Vec::new();
    let mut recons = Vec::new();
    let mut fullness = bufsize as i64;
    for (i, f) in frames(&planes).iter().enumerate() {
        let out = enc.encode_frame(f).expect("encode");
        let bits = out.au.len() as i64 * 8;
        assert!(
            bits <= fullness,
            "frame {i}: {bits} bits would underflow the VBV (fullness {fullness})"
        );
        fullness = (fullness - bits + per_frame_fill).min(bufsize as i64);
        stream.extend_from_slice(&out.au);
        recons.push(out.recon);
    }
    // Still a conforming stream, bit-exact through the own decoder.
    let decoded = decode_annexb_sequence(&stream).expect("decode");
    assert_eq!(decoded.len(), planes.len());
    for (i, (dec, rec)) in decoded.iter().zip(&recons).enumerate() {
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

#[test]
fn registry_vbv_options_validate_and_roundtrip() {
    let planes = clip(6);
    // Happy path: bitrate + bufsize.
    let mut p = base_params();
    p.options.insert("mode", "inter");
    p.options.insert("bitrate", "200k");
    p.options.insert("bufsize", "40k");
    let mut enc = make_encoder(&p).expect("encoder");
    let mut n = 0usize;
    for (y, cb, cr) in &planes {
        enc.send_frame(&video_frame(y, cb, cr)).expect("send");
        while let Ok(pkt) = enc.receive_packet() {
            assert!(
                pkt.data.len() as u64 * 8 <= 40_000,
                "no frame may exceed the VBV buffer"
            );
            n += 1;
        }
    }
    assert_eq!(n, planes.len());
    // bufsize without bitrate / malformed: rejected.
    for opts in [
        vec![("mode", "inter"), ("bufsize", "40k")],
        vec![("mode", "inter"), ("bitrate", "200k"), ("bufsize", "9")],
    ] {
        let mut p = base_params();
        for (k, v) in &opts {
            p.options.insert(*k, *v);
        }
        assert!(
            matches!(make_encoder(&p), Err(Error::InvalidData(_))),
            "{opts:?}"
        );
    }
}

/// VBV over the hierarchical-B pyramid: every access unit — anchor P
/// bursts and every B layer alike — must fit the modelled decoder
/// buffer at ITS OWN decode instant (per-temporal-layer buffer
/// accounting), replay-pinned exactly like the flat-GOP arm: the
/// leaky bucket is replayed over the decode-order access units and
/// may never underflow, while the unconstrained twin provably
/// overshoots the buffer.
#[test]
fn pyramid_vbv_constrained_stream_never_underflows() {
    use oxideav_h265::encoder::pyramid::PyramidEncoder;
    let planes = clip(33); // IDR + four GOP-8 mini-GOPs
    let frames = frames(&planes);
    let (rate, bufsize) = (150_000u64, 14_000u64);
    let per_frame_fill = (rate / u64::from(FPS)) as i64;
    // Decode-order AU sizes through a pyramid encoder.
    let collect = |mut enc: PyramidEncoder| -> Vec<oxideav_h265::encoder::pyramid::PyramidAu> {
        let mut aus = Vec::new();
        for f in &frames {
            aus.extend(enc.encode_frame(f).expect("encode"));
        }
        aus.extend(enc.flush());
        aus
    };
    // Unconstrained twin: prove the constraint has something to do.
    let free = collect(
        PyramidEncoder::new(W, H, 26, 8)
            .expect("encoder")
            .with_rate_control(&RateControlCfg::new(rate, FPS, 1)),
    );
    let free_max = free.iter().map(|au| au.au.len() as u64 * 8).max().unwrap();
    assert!(
        free_max > bufsize,
        "test premise: the unconstrained pyramid must overshoot the buffer ({free_max} bits)"
    );
    // Constrained run: replay the decoder-buffer model in DECODE
    // order (the order the access units hit the buffer).
    let aus = collect(
        PyramidEncoder::new(W, H, 26, 8)
            .expect("encoder")
            .with_rate_control(&RateControlCfg::new(rate, FPS, 1).with_vbv(bufsize)),
    );
    assert_eq!(aus.len(), planes.len());
    let mut fullness = bufsize as i64;
    let mut stream = Vec::new();
    for (i, au) in aus.iter().enumerate() {
        let bits = au.au.len() as i64 * 8;
        assert!(
            bits <= fullness,
            "decode-order AU {i} (display {}, layer {}): {bits} bits would underflow \
             the VBV (fullness {fullness})",
            au.display_order,
            au.layer
        );
        fullness = (fullness - bits + per_frame_fill).min(bufsize as i64);
        stream.extend_from_slice(&au.au);
    }
    // Still a conforming stream: display-order decode == encoder
    // reconstruction, bit-exact.
    let decoded = decode_annexb_sequence(&stream).expect("decode");
    assert_eq!(decoded.len(), planes.len());
    let mut recons: Vec<Option<&oxideav_h265::encoder::inter::FrameRecon>> =
        vec![None; planes.len()];
    for au in &aus {
        recons[au.display_order] = Some(&au.recon);
    }
    for (i, (dec, rec)) in decoded.iter().zip(&recons).enumerate() {
        let rec = rec.expect("every display index coded");
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

/// The registry `bufsize` + `pyramid` combination (rejected before
/// round 451) now encodes with the per-AU VBV guarantee.
#[test]
fn registry_pyramid_vbv_roundtrips() {
    let planes = clip(9); // IDR + one GOP-4 mini-GOP + 4-frame tail
    let mut p = base_params();
    p.options.insert("mode", "inter");
    p.options.insert("pyramid", "4");
    p.options.insert("bitrate", "200k");
    p.options.insert("bufsize", "30k");
    let mut enc = make_encoder(&p).expect("encoder");
    let mut stream = Vec::new();
    let mut n = 0usize;
    for (y, cb, cr) in &planes {
        enc.send_frame(&video_frame(y, cb, cr)).expect("send");
        while let Ok(pkt) = enc.receive_packet() {
            assert!(
                pkt.data.len() as u64 * 8 <= 30_000,
                "no access unit may exceed the VBV buffer"
            );
            stream.extend_from_slice(&pkt.data);
            n += 1;
        }
    }
    enc.flush().expect("flush");
    while let Ok(pkt) = enc.receive_packet() {
        stream.extend_from_slice(&pkt.data);
        n += 1;
    }
    assert_eq!(n, planes.len());
    let decoded = decode_annexb_sequence(&stream).expect("decode");
    assert_eq!(decoded.len(), planes.len());
}

/// An explicit `fps` option (or `with_frame_rate` on the direct
/// APIs) declares the frame rate in the SPS: §E.2.1
/// `vui_timing_info` with `num_units_in_tick == fps_den`,
/// `time_scale == fps_num` — parsed back through the crate's own SPS
/// decoder. Without it the SPS stays VUI-free (historical streams
/// byte-stable).
#[test]
fn fps_option_declares_vui_timing() {
    use oxideav_h265::sps::SeqParameterSet;
    let sps_of = |stream: &[u8]| -> SeqParameterSet {
        let units = oxideav_h265::collect_nal_units(stream).expect("nals");
        let sps = units
            .iter()
            .find(|u| u.header.nal_unit_type == 33)
            .expect("SPS NAL");
        SeqParameterSet::parse(&sps.rbsp).expect("SPS parse")
    };
    let planes = clip(2);
    // Registry inter path with an explicit rational fps.
    let mut p = base_params();
    p.options.insert("mode", "inter");
    p.options.insert("fps", "30000/1001");
    let mut enc = make_encoder(&p).expect("encoder");
    let (y, cb, cr) = &planes[0];
    enc.send_frame(&video_frame(y, cb, cr)).expect("send");
    let pkt = enc.receive_packet().expect("packet");
    let sps = sps_of(&pkt.data);
    let vui = sps.vui_parameters.expect("VUI present");
    let timing = vui.timing_info.expect("timing info present");
    assert_eq!((timing.num_units_in_tick, timing.time_scale), (1001, 30000));
    // No fps option -> no VUI.
    let mut p2 = base_params();
    p2.options.insert("mode", "inter");
    let mut enc2 = make_encoder(&p2).expect("encoder");
    enc2.send_frame(&video_frame(y, cb, cr)).expect("send");
    let pkt2 = enc2.receive_packet().expect("packet");
    assert!(sps_of(&pkt2.data).vui_parameters.is_none());
    // Direct pyramid API + decode still bit-exact with the VUI SPS.
    {
        use oxideav_h265::encoder::pyramid::{encode_pyramid_with, PyramidEncoder};
        let frames = frames(&planes);
        let enc = PyramidEncoder::new(W, H, 30, 2)
            .expect("encoder")
            .with_frame_rate(25, 1);
        let out = encode_pyramid_with(enc, &frames).expect("encode");
        let sps = sps_of(&out.stream);
        let timing = sps
            .vui_parameters
            .expect("VUI")
            .timing_info
            .expect("timing");
        assert_eq!((timing.num_units_in_tick, timing.time_scale), (1, 25));
        let decoded = decode_annexb_sequence(&out.stream).expect("decode");
        assert_eq!(decoded.len(), planes.len());
        for (dec, rec) in decoded.iter().zip(&out.recon) {
            let mut expect = rec.y.clone();
            expect.extend_from_slice(&rec.cb);
            expect.extend_from_slice(&rec.cr);
            assert_eq!(dec.picture.to_planar_u8().expect("8-bit"), expect);
        }
    }
}

#[test]
fn registry_rejects_bad_abr_options() {
    for (opts, needle) in [
        (vec![("bitrate", "200k")], "bitrate"), // pcm default
        (vec![("mode", "inter"), ("bitrate", "12")], "bitrate"),
        (vec![("mode", "inter"), ("bitrate", "abc")], "bitrate"),
        (
            vec![("mode", "inter"), ("bitrate", "200k"), ("fps", "0")],
            "fps",
        ),
        (
            vec![("mode", "inter"), ("bitrate", "200k"), ("fps", "30/0")],
            "fps",
        ),
    ] {
        let mut p = base_params();
        for (k, v) in &opts {
            p.options.insert(*k, *v);
        }
        match make_encoder(&p) {
            Err(Error::InvalidData(msg)) => {
                assert!(msg.contains(needle), "{opts:?}: unexpected message {msg:?}")
            }
            Err(other) => panic!("{opts:?}: expected InvalidData, got {other:?}"),
            Ok(_) => panic!("{opts:?}: expected InvalidData, got an encoder"),
        }
    }
}
