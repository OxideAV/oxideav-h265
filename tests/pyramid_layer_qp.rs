//! Hierarchical (per-layer) QP offsets: the pyramid's rate-allocation
//! shape — `SliceQpY = base + layer * step` — verified ON THE WIRE
//! (slice-header `slice_qp_delta` against the PPS `init_qp`), its
//! interaction with spatial AQ (per-CTB `cu_qp_delta` riding on the
//! per-layer slice QPs), the rate-shaping effect of the step, and the
//! registry `pyramidstep` exposure.

use oxideav_core::{CodecParameters, Error, Frame, PixelFormat, VideoFrame, VideoPlane};
use oxideav_h265::encoder::inter::YuvFrame;
use oxideav_h265::encoder::loopfilter::LoopFilterCfg;
use oxideav_h265::encoder::pyramid::{PyramidAu, PyramidEncoder};
use oxideav_h265::pps::PicParameterSet;
use oxideav_h265::slice::SliceSegmentHeader;
use oxideav_h265::sps::SeqParameterSet;
use oxideav_h265::{collect_nal_units, decode_annexb_sequence, make_encoder};

const W: usize = 64;
const H: usize = 48;

fn noise(seed: &mut u32) -> u8 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    (*seed >> 24) as u8
}

fn clip(n: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut seed = 0x1A7E_57E9u32;
    (0..n)
        .map(|t| {
            let y: Vec<u8> = (0..W * H)
                .map(|i| {
                    let (x, yy) = (i % W, i / W);
                    let base = ((x * 7 + yy * 3) % 190) as i32 + i32::from(noise(&mut seed) % 10);
                    let (sx, sy) = ((2 + t * 2) % (W - 10), (4 + t) % (H - 10));
                    if x >= sx && x < sx + 8 && yy >= sy && yy < sy + 8 {
                        (base + 60).clamp(0, 255) as u8
                    } else {
                        base.clamp(0, 255) as u8
                    }
                })
                .collect();
            let cb: Vec<u8> = (0..W * H / 4).map(|i| (95 + (i + t) % 62) as u8).collect();
            let cr: Vec<u8> = (0..W * H / 4).map(|i| (155 - (i + t) % 48) as u8).collect();
            (y, cb, cr)
        })
        .collect()
}

/// Encode a whole clip through a configured pyramid encoder; AUs in
/// decode order.
fn encode(enc: &mut PyramidEncoder, planes: &[(Vec<u8>, Vec<u8>, Vec<u8>)]) -> Vec<PyramidAu> {
    let mut aus = Vec::new();
    for (y, cb, cr) in planes {
        aus.extend(enc.encode_frame(&YuvFrame { y, cb, cr }).expect("encode"));
    }
    aus.extend(enc.flush());
    aus
}

/// Parse every VCL slice header of `stream` and return its computed
/// `SliceQpY` (`26 + init_qp_minus26 + slice_qp_delta`), decode order.
fn wire_slice_qps(stream: &[u8]) -> Vec<i32> {
    let units = collect_nal_units(stream).expect("NAL walk");
    let mut sps = None;
    let mut pps: Option<PicParameterSet> = None;
    let mut qps = Vec::new();
    for unit in &units {
        match unit.header.nal_unit_type {
            33 => sps = Some(SeqParameterSet::parse(&unit.rbsp).expect("SPS")),
            34 => pps = Some(PicParameterSet::parse(&unit.rbsp).expect("PPS")),
            t if t <= 31 => {
                let header = SliceSegmentHeader::parse(
                    &unit.rbsp,
                    t,
                    sps.as_ref().expect("SPS before slice"),
                    pps.as_ref().expect("PPS before slice"),
                )
                .expect("slice header");
                let init_qp = 26 + pps.as_ref().unwrap().init_qp_minus26;
                qps.push(init_qp + header.slice_qp_delta.expect("slice_qp_delta parsed"));
            }
            _ => {}
        }
    }
    qps
}

/// The wire carries exactly the hierarchical allocation: at constant
/// base QP, every slice's signalled `SliceQpY` is `base + layer *
/// step` for its pyramid layer — for step 0 (flat), 1 (default) and
/// 3 (steep).
#[test]
fn layer_qp_offsets_are_signalled_on_the_wire() {
    let planes = clip(9); // IDR + one GOP-8 mini-GOP
    for step in [0i32, 1, 3] {
        let mut enc = PyramidEncoder::new(W, H, 28, 8)
            .expect("encoder")
            .with_layer_qp_step(step);
        let aus = encode(&mut enc, &planes);
        let stream: Vec<u8> = aus.iter().flat_map(|au| au.au.iter().copied()).collect();
        let wire = wire_slice_qps(&stream);
        assert_eq!(wire.len(), aus.len(), "step {step}");
        for (au, &wire_qp) in aus.iter().zip(&wire) {
            let expect = (28 + i32::from(au.layer) * step).clamp(0, 51);
            assert_eq!(
                wire_qp, expect,
                "step {step}, display {} (layer {}): wire SliceQpY",
                au.display_order, au.layer
            );
            assert_eq!(au.qp, expect, "encoder-reported QP");
        }
        // And the stream still decodes bit-exactly.
        let decoded = decode_annexb_sequence(&stream).expect("decode");
        assert_eq!(decoded.len(), planes.len());
    }
}

/// The step shapes the allocation: a steeper step spends strictly
/// fewer bits on the deepest layer (relative to the anchors) and
/// fewer bits overall at equal base QP.
#[test]
fn layer_step_shapes_the_rate_allocation() {
    let planes = clip(17); // IDR + two GOP-8 mini-GOPs
    let bits_by = |step: i32| -> (u64, u64, u64) {
        let mut enc = PyramidEncoder::new(W, H, 26, 8)
            .expect("encoder")
            .with_layer_qp_step(step);
        let aus = encode(&mut enc, &planes);
        let total = aus.iter().map(|au| au.au.len() as u64 * 8).sum();
        let deep = aus
            .iter()
            .filter(|au| au.layer == 3)
            .map(|au| au.au.len() as u64 * 8)
            .sum();
        let anchors = aus
            .iter()
            .filter(|au| au.layer == 0 && !au.keyframe)
            .map(|au| au.au.len() as u64 * 8)
            .sum();
        (total, deep, anchors)
    };
    let (t0, d0, a0) = bits_by(0);
    let (t3, d3, a3) = bits_by(3);
    assert!(t3 < t0, "steeper step must cut total bits ({t0} -> {t3})");
    assert!(
        d3 < d0,
        "steeper step must cut deep-layer bits ({d0} -> {d3})"
    );
    // The deep layers' share of the stream shrinks; the anchors are
    // coded at the same QP either way.
    assert!(
        d3 * t0 < d0 * t3,
        "deep-layer SHARE must shrink (d0/t0 {d0}/{t0} vs d3/t3 {d3}/{t3})"
    );
    assert_eq!(a0, a3, "layer-0 anchors are step-independent");
}

/// Hierarchical offsets compose with spatial AQ: the slice header
/// still carries the per-layer QP while every slice's CTBs move off
/// it through `cu_qp_delta` — and the composed stream decodes
/// bit-exactly to the encoder reconstruction through the crate's own
/// decoder (deblocking, `qPY_PREV` chains and mode decisions all
/// consistent at per-CTB, per-layer QPs).
#[test]
fn layer_offsets_compose_with_aq_and_filters() {
    let planes = clip(9);
    let mut enc = PyramidEncoder::new(W, H, 30, 4)
        .expect("encoder")
        .with_layer_qp_step(2)
        .with_aq(3)
        .with_loop_filters(LoopFilterCfg::all());
    let aus = encode(&mut enc, &planes);
    let stream: Vec<u8> = aus.iter().flat_map(|au| au.au.iter().copied()).collect();
    // Wire slice QPs still carry the layer shape under AQ.
    let wire = wire_slice_qps(&stream);
    for (au, &wire_qp) in aus.iter().zip(&wire) {
        assert_eq!(wire_qp, (30 + i32::from(au.layer) * 2).clamp(0, 51));
    }
    // Bit-exact display-order decode.
    let decoded = decode_annexb_sequence(&stream).expect("decode");
    assert_eq!(decoded.len(), planes.len());
    let mut recons: Vec<Option<&PyramidAu>> = vec![None; planes.len()];
    for au in &aus {
        recons[au.display_order] = Some(au);
    }
    for (i, (dec, au)) in decoded.iter().zip(&recons).enumerate() {
        let au = au.expect("coded");
        let mut expect = au.recon.y.clone();
        expect.extend_from_slice(&au.recon.cb);
        expect.extend_from_slice(&au.recon.cr);
        assert_eq!(
            dec.picture.to_planar_u8().expect("8-bit"),
            expect,
            "frame {i}"
        );
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

/// The registry `pyramidstep` option: happy path (wire QPs reflect
/// the step, composing with `aq`), and rejections (without `pyramid`,
/// out of range).
#[test]
fn registry_pyramidstep_option() {
    let planes = clip(5);
    let mut p = base_params();
    p.options.insert("mode", "inter");
    p.options.insert("pyramid", "4");
    p.options.insert("pyramidstep", "2");
    p.options.insert("qp", "30");
    p.options.insert("aq", "1");
    let mut enc = make_encoder(&p).expect("encoder");
    let mut stream = Vec::new();
    for (y, cb, cr) in &planes {
        enc.send_frame(&video_frame(y, cb, cr)).expect("send");
        while let Ok(pkt) = enc.receive_packet() {
            stream.extend_from_slice(&pkt.data);
        }
    }
    enc.flush().expect("flush");
    while let Ok(pkt) = enc.receive_packet() {
        stream.extend_from_slice(&pkt.data);
    }
    let wire = wire_slice_qps(&stream);
    assert_eq!(wire.len(), planes.len());
    // GOP-4 decode order: IDR(0), P(0), B(1), B(2), B(2) layers ->
    // 30, 30, 32, 34, 34.
    assert_eq!(wire, vec![30, 30, 32, 34, 34]);
    assert_eq!(
        decode_annexb_sequence(&stream).expect("decode").len(),
        planes.len()
    );
    for opts in [
        vec![("mode", "inter"), ("pyramidstep", "2")],
        vec![("mode", "inter"), ("pyramid", "4"), ("pyramidstep", "7")],
        vec![("mode", "inter"), ("pyramid", "4"), ("pyramidstep", "-1")],
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
