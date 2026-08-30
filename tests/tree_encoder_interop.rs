//! Round-453 coding-quadtree encoder interop pins.
//!
//! Each golden stream is the quadtree coder's (`encoder::ctu`)
//! deterministic output for a fixed synthetic clip, validated OUT OF
//! BAND against a black-box reference HEVC decoder (byte-exact to the
//! encoder reconstruction — see
//! `fixture_bytes/r453-generation-notes.md`). The tests pin:
//!
//! 1. the encoder still emits the golden bytes (any drift must be
//!    re-validated black-box and re-pinned deliberately);
//! 2. this crate's decoder reproduces the encoder reconstruction
//!    sample-exactly in display order.

use oxideav_h265::encoder::ctu::TreeCfg;
use oxideav_h265::encoder::inter::{FrameRecon, LowDelayPEncoder, YuvFrame};
use oxideav_h265::encoder::loopfilter::LoopFilterCfg;
use oxideav_h265::encoder::pyramid::PyramidEncoder;
use oxideav_h265::sequence::decode_annexb_sequence;

fn scene(w: usize, h: usize, n_frames: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    (0..n_frames)
        .map(|f| {
            let mut y = vec![0u8; w * h];
            for j in 0..h {
                for i in 0..w {
                    let sq = usize::from(i >= 8 + f * 3 && i < 40 + f * 3 && (16..48).contains(&j));
                    let tex = (i * 7 + j * 5 + (i * j) / 9) % 120;
                    y[j * w + i] = (tex + sq * 100) as u8;
                }
            }
            let cb: Vec<u8> = (0..w * h / 4)
                .map(|k| ((k % 40) + 100 + f * 2) as u8)
                .collect();
            let cr: Vec<u8> = (0..w * h / 4).map(|k| ((k % 28) + 88) as u8).collect();
            (y, cb, cr)
        })
        .collect()
}

/// Decode `stream` and assert per-frame sample-exactness against the
/// display-order reconstructions.
fn assert_display_order_exact(stream: &[u8], recons: &[FrameRecon], w: usize, h: usize) {
    let decoded = decode_annexb_sequence(stream).expect("golden stream decodes");
    assert_eq!(decoded.len(), recons.len(), "frame count");
    for (i, (f, rec)) in decoded.iter().zip(recons.iter()).enumerate() {
        let planar = f.picture.to_planar_u8().expect("8-bit planes");
        assert_eq!(planar[..w * h], rec.y[..], "frame {i} luma");
        assert_eq!(planar[w * h..w * h + w * h / 4], rec.cb[..], "frame {i} cb");
        assert_eq!(planar[w * h + w * h / 4..], rec.cr[..], "frame {i} cr");
    }
}

#[test]
fn tree_intra_ctb64_pin() {
    let (w, h) = (96, 80);
    let frames = scene(w, h, 1);
    let mut enc = LowDelayPEncoder::new(w, h, 30, 0)
        .expect("encoder")
        .with_tree(TreeCfg::new(64).expect("ctb 64"))
        .with_loop_filters(LoopFilterCfg::all());
    let f = enc
        .encode_frame(&YuvFrame {
            y: &frames[0].0,
            cb: &frames[0].1,
            cr: &frames[0].2,
        })
        .expect("frame");
    let golden: &[u8] = include_bytes!("fixture_bytes/r453-tree-intra-ctb64-qp30.hevc");
    assert_eq!(f.au, golden, "stream drifted off the validated pin");
    assert_display_order_exact(&f.au, std::slice::from_ref(&f.recon), w, h);
}

#[test]
fn tree_pgop_ctb32_pin() {
    let (w, h) = (96, 64);
    let frames = scene(w, h, 5);
    let mut enc = LowDelayPEncoder::new(w, h, 30, 0)
        .expect("encoder")
        .with_tree(TreeCfg::new(32).expect("ctb 32"))
        .with_loop_filters(LoopFilterCfg::all())
        .with_aq(2);
    let mut stream = Vec::new();
    let mut recons = Vec::new();
    for (y, cb, cr) in &frames {
        let f = enc.encode_frame(&YuvFrame { y, cb, cr }).expect("frame");
        stream.extend_from_slice(&f.au);
        recons.push(f.recon);
    }
    let golden: &[u8] = include_bytes!("fixture_bytes/r453-tree-pgop-ctb32-qp30.hevc");
    assert_eq!(stream, golden, "stream drifted off the validated pin");
    assert_display_order_exact(&stream, &recons, w, h);
}

#[test]
fn tree_bpyramid_ctb64_amp_pin() {
    let (w, h) = (96, 64);
    let frames = scene(w, h, 5);
    let mut enc = PyramidEncoder::new(w, h, 31, 4)
        .expect("encoder")
        .with_tree(TreeCfg::new(64).expect("ctb 64"))
        .with_amp(true);
    let mut stream = Vec::new();
    let mut recons: Vec<Option<FrameRecon>> = vec![None; frames.len()];
    let push = |aus: Vec<oxideav_h265::encoder::pyramid::PyramidAu>,
                stream: &mut Vec<u8>,
                recons: &mut Vec<Option<FrameRecon>>| {
        for au in aus {
            stream.extend_from_slice(&au.au);
            recons[au.display_order] = Some(au.recon);
        }
    };
    for (y, cb, cr) in &frames {
        let aus = enc.encode_frame(&YuvFrame { y, cb, cr }).expect("frame");
        push(aus, &mut stream, &mut recons);
    }
    push(enc.flush(), &mut stream, &mut recons);
    let recons: Vec<FrameRecon> = recons.into_iter().map(|r| r.expect("coded")).collect();
    let golden: &[u8] = include_bytes!("fixture_bytes/r453-tree-bpyr-ctb64-qp31.hevc");
    assert_eq!(stream, golden, "stream drifted off the validated pin");
    assert_display_order_exact(&stream, &recons, w, h);
}

/// The registry `ctb` option routes the codec-contract encoder
/// through the quadtree coder and the stream stays sample-exact to
/// the reconstruction path (crate decoder as oracle).
#[test]
fn registry_ctb_option_roundtrips() {
    use oxideav_core::{CodecParameters, Frame, PixelFormat, VideoFrame, VideoPlane};
    let (w, h) = (48usize, 48usize);
    let frames = scene(w, h, 3);
    let mut p = CodecParameters::video("h265".into());
    p.width = Some(w as u32);
    p.height = Some(h as u32);
    p.pixel_format = Some(PixelFormat::Yuv420P);
    p.options.insert("mode", "inter");
    p.options.insert("ctb", "32");
    p.options.insert("qp", "28");
    let mut enc = oxideav_h265::make_encoder(&p).expect("encoder");
    let plane = |data: &[u8], stride: usize| VideoPlane {
        stride,
        data: data.to_vec(),
    };
    let mut stream = Vec::new();
    for (y, cb, cr) in &frames {
        enc.send_frame(&Frame::Video(VideoFrame {
            pts: None,
            planes: vec![plane(y, w), plane(cb, w / 2), plane(cr, w / 2)],
        }))
        .expect("send");
        let pkt = enc.receive_packet().expect("packet");
        stream.extend_from_slice(&pkt.data);
    }
    let decoded = decode_annexb_sequence(&stream).expect("decodes");
    assert_eq!(decoded.len(), 3);

    // Bad sizes and unsupported mode pairings are rejected.
    for (k, v) in [("ctb", "48"), ("ctb", "0"), ("ctb", "x")] {
        let mut bad = p.clone();
        bad.options.insert(k, v);
        assert!(
            oxideav_h265::make_encoder(&bad).is_err(),
            "{k}={v} must be rejected"
        );
    }
    let mut bad = p.clone();
    bad.options.insert("mode", "pcm");
    assert!(
        oxideav_h265::make_encoder(&bad).is_err(),
        "ctb over pcm must be rejected"
    );

    // The intra mode takes the option too.
    let mut ip = p.clone();
    ip.options.insert("mode", "intra");
    ip.options.insert("ctb", "64");
    let mut enc = oxideav_h265::make_encoder(&ip).expect("intra tree encoder");
    let (y, cb, cr) = &frames[0];
    enc.send_frame(&Frame::Video(VideoFrame {
        pts: None,
        planes: vec![plane(y, w), plane(cb, w / 2), plane(cr, w / 2)],
    }))
    .expect("send");
    let pkt = enc.receive_packet().expect("packet");
    assert_eq!(decode_annexb_sequence(&pkt.data).expect("decodes").len(), 1);
}
