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
        (
            vec![("mode", "inter"), ("pyramid", "4"), ("bitrate", "200k")],
            "pyramid",
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
