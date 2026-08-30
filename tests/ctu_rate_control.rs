//! CTU-level rate feedback (round 453): with rate control on the
//! quadtree coder, every CTB's `QpY` tracks the picture's running
//! coded size (a shadow CABAC count) against the controller's
//! pro-rata frame budget through §7.3.8.14 `cu_qp_delta`. Streams stay
//! conforming and decode bit-exact; the frame-level accuracy gate
//! still holds.

use oxideav_h265::decode_annexb_sequence;
use oxideav_h265::encoder::ctu::TreeCfg;
use oxideav_h265::encoder::inter::{FrameRecon, LowDelayPEncoder, YuvFrame};
use oxideav_h265::encoder::loopfilter::LoopFilterCfg;
use oxideav_h265::encoder::pyramid::PyramidEncoder;
use oxideav_h265::encoder::rate::RateControlCfg;

const W: usize = 64;
const H: usize = 64;

fn clip(n: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut seed = 0x1234_5678u32;
    let mut rnd = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 24) as u8
    };
    (0..n)
        .map(|f| {
            let mut y = vec![0u8; W * H];
            for j in 0..H {
                for i in 0..W {
                    // Flat left half, busy right half, slow pan.
                    let busy = i >= W / 2;
                    let base = ((i + 2 * f) * 3 + (j + f) * 2) % 120 + 60;
                    y[j * W + i] = if busy {
                        (base + usize::from(rnd() % 40)) as u8
                    } else {
                        (base / 2 + 80) as u8
                    };
                }
            }
            let cb: Vec<u8> = (0..W * H / 4).map(|k| ((k % 30) + 100) as u8).collect();
            let cr: Vec<u8> = (0..W * H / 4).map(|k| ((k % 22) + 95) as u8).collect();
            (y, cb, cr)
        })
        .collect()
}

fn assert_exact(stream: &[u8], recons: &[FrameRecon]) {
    let decoded = decode_annexb_sequence(stream).expect("decode");
    assert_eq!(decoded.len(), recons.len());
    for (i, (d, rec)) in decoded.iter().zip(recons.iter()).enumerate() {
        let mut expect = rec.y.clone();
        expect.extend_from_slice(&rec.cb);
        expect.extend_from_slice(&rec.cr);
        assert_eq!(
            d.picture.to_planar_u8().expect("8-bit"),
            expect,
            "frame {i}"
        );
    }
}

fn low_delay(frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)], ctu_rc: bool) -> (Vec<u8>, Vec<FrameRecon>) {
    let mut enc = LowDelayPEncoder::new(W, H, 30, 20)
        .expect("encoder")
        .with_tree(TreeCfg::new(32).expect("ctb 32"))
        .with_rate_control(&RateControlCfg::new(120_000, 25, 1))
        .with_loop_filters(LoopFilterCfg::all())
        .with_ctu_rate_control(ctu_rc);
    let mut stream = Vec::new();
    let mut recons = Vec::new();
    for (y, cb, cr) in frames {
        let f = enc.encode_frame(&YuvFrame { y, cb, cr }).expect("frame");
        stream.extend_from_slice(&f.au);
        recons.push(f.recon);
    }
    (stream, recons)
}

#[test]
fn ctu_rate_feedback_roundtrips_and_holds_the_target() {
    let frames = clip(40);
    let (s_off, r_off) = low_delay(&frames, false);
    let (s_on, r_on) = low_delay(&frames, true);
    assert_exact(&s_off, &r_off);
    assert_exact(&s_on, &r_on);
    assert_ne!(s_off, s_on, "per-CTB cu_qp_delta changes the stream");
    // 40 frames at 25 fps and 120 kb/s: 192 000 bits.
    let target = 120_000u64 * 40 / 25;
    let bits = s_on.len() as u64 * 8;
    let err = (bits as i64 - target as i64).unsigned_abs() * 1000 / target;
    assert!(
        err <= 60,
        "CTU-RC low-delay lands within 6 %: {bits} vs {target} ({err}‰)"
    );
}

#[test]
fn ctu_rate_feedback_composes_with_pyramid_aq_and_tmvp() {
    let frames = clip(13);
    let mut enc = PyramidEncoder::new(W, H, 30, 4)
        .expect("encoder")
        .with_tree(TreeCfg::new(64).expect("ctb 64"))
        .with_rate_control(&RateControlCfg::new(100_000, 25, 1).with_vbv(40_000))
        .with_aq(2)
        .with_temporal_mvp(true)
        .with_ctu_rate_control(true);
    let mut stream = Vec::new();
    let mut recons: Vec<Option<FrameRecon>> = vec![None; frames.len()];
    let push = |aus: Vec<oxideav_h265::encoder::pyramid::PyramidAu>,
                stream: &mut Vec<u8>,
                recons: &mut Vec<Option<FrameRecon>>| {
        for au in aus {
            assert!(
                au.au.len() as u64 * 8 <= 40_000,
                "VBV cap holds under CTU-RC"
            );
            stream.extend_from_slice(&au.au);
            recons[au.display_order] = Some(au.recon);
        }
    };
    for (y, cb, cr) in &frames {
        push(
            enc.encode_frame(&YuvFrame { y, cb, cr }).expect("frame"),
            &mut stream,
            &mut recons,
        );
    }
    push(enc.flush(), &mut stream, &mut recons);
    let recons: Vec<FrameRecon> = recons.into_iter().map(|r| r.expect("coded")).collect();
    assert_exact(&stream, &recons);
}

#[test]
fn ctu_rate_feedback_is_a_no_op_without_the_quadtree_or_rate_control() {
    let frames = clip(4);
    let run = |cfg: fn(LowDelayPEncoder) -> LowDelayPEncoder| -> Vec<u8> {
        let mut enc = cfg(LowDelayPEncoder::new(W, H, 30, 0).expect("encoder"));
        let mut stream = Vec::new();
        for (y, cb, cr) in &frames {
            stream.extend_from_slice(&enc.encode_frame(&YuvFrame { y, cb, cr }).expect("frame").au);
        }
        stream
    };
    // Fixed-geometry coder: the flag cannot engage.
    assert_eq!(
        run(|e| e.with_ctu_rate_control(true)),
        run(|e| e.with_ctu_rate_control(false))
    );
    // Quadtree coder without rate control: no budget, no feedback.
    assert_eq!(
        run(|e| e
            .with_tree(TreeCfg::new(32).unwrap())
            .with_ctu_rate_control(true)),
        run(|e| e.with_tree(TreeCfg::new(32).unwrap()))
    );
}

#[test]
fn registry_cturc_option_validates_and_roundtrips() {
    use oxideav_core::{CodecParameters, Frame, PixelFormat, VideoFrame, VideoPlane};
    let frames = clip(3);
    let mut p = CodecParameters::video("h265".into());
    p.width = Some(W as u32);
    p.height = Some(H as u32);
    p.pixel_format = Some(PixelFormat::Yuv420P);
    p.options.insert("mode", "inter");
    p.options.insert("ctb", "32");
    p.options.insert("bitrate", "100k");
    p.options.insert("cturc", "1");
    let mut enc = oxideav_h265::make_encoder(&p).expect("encoder");
    let plane = |data: &[u8], stride: usize| VideoPlane {
        stride,
        data: data.to_vec(),
    };
    let mut stream = Vec::new();
    for (y, cb, cr) in &frames {
        enc.send_frame(&Frame::Video(VideoFrame {
            pts: None,
            planes: vec![plane(y, W), plane(cb, W / 2), plane(cr, W / 2)],
        }))
        .expect("send");
        stream.extend_from_slice(&enc.receive_packet().expect("packet").data);
    }
    assert_eq!(decode_annexb_sequence(&stream).expect("decodes").len(), 3);
    // Intra mode takes it too.
    let mut ip = p.clone();
    ip.options.insert("mode", "intra");
    assert!(oxideav_h265::make_encoder(&ip).is_ok());
    // Without bitrate or ctb it is rejected.
    for missing in ["bitrate", "ctb"] {
        let mut bad = CodecParameters::video("h265".into());
        bad.width = Some(W as u32);
        bad.height = Some(H as u32);
        bad.pixel_format = Some(PixelFormat::Yuv420P);
        bad.options.insert("mode", "inter");
        bad.options.insert("cturc", "1");
        if missing != "bitrate" {
            bad.options.insert("bitrate", "100k");
        }
        if missing != "ctb" {
            bad.options.insert("ctb", "32");
        }
        assert!(
            oxideav_h265::make_encoder(&bad).is_err(),
            "cturc without {missing} must be rejected"
        );
    }
}

/// Golden pin (black-box validated, see
/// `fixture_bytes/r453-generation-notes.md`): 12 frames of the clip
/// under CTU-level feedback at 120 kb/s.
#[test]
fn golden_ctu_rate_control_pin() {
    let frames = clip(12);
    let (stream, recons) = low_delay(&frames, true);
    let golden: &[u8] = include_bytes!("fixture_bytes/r453-tree-lowdelay-cturc-120k.hevc");
    assert_eq!(stream, golden, "stream drifted off the validated pin");
    assert_exact(golden, &recons);
}
