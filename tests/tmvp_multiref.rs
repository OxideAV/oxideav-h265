//! Encoder temporal MVP + multi-reference lists (round 453): every
//! stream decodes bit-exact to the encoder reconstruction through
//! the crate's decoder, which honours `slice_temporal_mvp_enabled_flag`
//! / the collocated selection and the §8.3.4 list construction — so a
//! candidate derived without the temporal input, or a list built in
//! another order, would surface as a reconstruction mismatch.

use oxideav_h265::decode_annexb_sequence;
use oxideav_h265::encoder::ctu::TreeCfg;
use oxideav_h265::encoder::inter::{FrameRecon, LowDelayPEncoder, YuvFrame};
use oxideav_h265::encoder::loopfilter::LoopFilterCfg;
use oxideav_h265::encoder::pyramid::PyramidEncoder;

fn clip(w: usize, h: usize, n: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut seed = 0x5eed_1234u32;
    let mut rnd = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 24) as u8
    };
    let bg: Vec<u8> = (0..(w + 32) * (h + 32))
        .map(|k| {
            let (x, y) = (k % (w + 32), k / (w + 32));
            (((x * 11 + y * 5) % 90) + ((x / 7 + y / 5) % 3) * 30 + usize::from(rnd() % 6)) as u8
        })
        .collect();
    (0..n)
        .map(|f| {
            let mut y = vec![0u8; w * h];
            for j in 0..h {
                for i in 0..w {
                    let mut v = bg[(j + f) * (w + 32) + i + 2 * f];
                    // A flickering block: bright on even frames only,
                    // so the two-frames-back reference wins there.
                    if (8..24).contains(&i) && (8..24).contains(&j) && f % 2 == 0 {
                        v = v / 2 + 100;
                    }
                    y[j * w + i] = v;
                }
            }
            let cb: Vec<u8> = (0..w * h / 4).map(|k| ((k % 30) + 100) as u8).collect();
            let cr: Vec<u8> = (0..w * h / 4).map(|k| ((k % 22) + 95) as u8).collect();
            (y, cb, cr)
        })
        .collect()
}

fn assert_exact(stream: &[u8], recons: &[FrameRecon], w: usize, h: usize) {
    let decoded = decode_annexb_sequence(stream).expect("decode");
    assert_eq!(decoded.len(), recons.len(), "frame count");
    for (i, (d, rec)) in decoded.iter().zip(recons.iter()).enumerate() {
        let planar = d.picture.to_planar_u8().expect("8-bit");
        assert_eq!(planar[..w * h], rec.y[..], "frame {i} luma");
        assert_eq!(planar[w * h..w * h + w * h / 4], rec.cb[..], "frame {i} cb");
        assert_eq!(planar[w * h + w * h / 4..], rec.cr[..], "frame {i} cr");
    }
}

fn low_delay(
    w: usize,
    h: usize,
    n: usize,
    cfg: impl Fn(LowDelayPEncoder) -> LowDelayPEncoder,
) -> (Vec<u8>, Vec<FrameRecon>, usize) {
    let frames = clip(w, h, n);
    let mut enc = cfg(LowDelayPEncoder::new(w, h, 29, 0).expect("encoder"));
    let mut stream = Vec::new();
    let mut recons = Vec::new();
    let mut ref1 = 0;
    for (y, cb, cr) in &frames {
        let f = enc.encode_frame(&YuvFrame { y, cb, cr }).expect("frame");
        stream.extend_from_slice(&f.au);
        ref1 += f.stats.ref1;
        recons.push(f.recon);
    }
    (stream, recons, ref1)
}

fn pyramid(
    w: usize,
    h: usize,
    n: usize,
    gop: usize,
    cfg: impl Fn(PyramidEncoder) -> PyramidEncoder,
) -> (Vec<u8>, Vec<FrameRecon>, usize) {
    let frames = clip(w, h, n);
    let mut enc = cfg(PyramidEncoder::new(w, h, 29, gop).expect("encoder"));
    let mut stream = Vec::new();
    let mut recons: Vec<Option<FrameRecon>> = vec![None; n];
    let mut ref1 = 0;
    let mut push = |aus: Vec<oxideav_h265::encoder::pyramid::PyramidAu>,
                    stream: &mut Vec<u8>,
                    recons: &mut Vec<Option<FrameRecon>>| {
        for au in aus {
            stream.extend_from_slice(&au.au);
            ref1 += au.stats.ref1;
            recons[au.display_order] = Some(au.recon);
        }
    };
    for (y, cb, cr) in &frames {
        let aus = enc.encode_frame(&YuvFrame { y, cb, cr }).expect("frame");
        push(aus, &mut stream, &mut recons);
    }
    push(enc.flush(), &mut stream, &mut recons);
    (
        stream,
        recons.into_iter().map(|r| r.expect("coded")).collect(),
        ref1,
    )
}

#[test]
fn low_delay_p_with_temporal_mvp_roundtrips() {
    let (w, h) = (64, 48);
    let (s_off, r_off, _) = low_delay(w, h, 5, |e| e);
    let (s_on, r_on, _) = low_delay(w, h, 5, |e| e.with_temporal_mvp(true));
    assert_exact(&s_off, &r_off, w, h);
    assert_exact(&s_on, &r_on, w, h);
    assert_ne!(s_off, s_on, "the SPS/slice flags change the stream");
}

#[test]
fn low_delay_b_four_refs_with_temporal_mvp_roundtrips() {
    let (w, h) = (64, 48);
    let (stream, recons, ref1) = low_delay(w, h, 7, |e| {
        e.with_b_slices(true)
            .with_refs(4)
            .with_temporal_mvp(true)
            .with_loop_filters(LoopFilterCfg::all())
    });
    assert_exact(&stream, &recons, w, h);
    assert!(
        ref1 > 0,
        "the flickering block must elect a farther reference"
    );
}

#[test]
fn low_delay_single_ref_roundtrips() {
    let (w, h) = (48, 48);
    let (stream, recons, ref1) = low_delay(w, h, 4, |e| e.with_refs(1));
    assert_exact(&stream, &recons, w, h);
    assert_eq!(ref1, 0, "one active reference: ref_idx never exceeds 0");
}

#[test]
fn pyramid_two_refs_with_temporal_mvp_roundtrips() {
    let (w, h) = (64, 48);
    let (stream, recons, ref1) = pyramid(w, h, 9, 8, |e| e.with_refs(2).with_temporal_mvp(true));
    assert_exact(&stream, &recons, w, h);
    assert!(
        ref1 > 0,
        "second references elected somewhere in the GOP-8 pyramid"
    );
}

#[test]
fn pyramid_three_refs_quadtree_roundtrips() {
    let (w, h) = (64, 48);
    let (stream, recons, ref1) = pyramid(w, h, 10, 8, |e| {
        e.with_refs(3)
            .with_temporal_mvp(true)
            .with_tree(TreeCfg::new(32).expect("ctb 32"))
            .with_loop_filters(LoopFilterCfg::all())
            .with_aq(1)
    });
    assert_exact(&stream, &recons, w, h);
    assert!(ref1 > 0);
}

#[test]
fn pyramid_temporal_mvp_changes_stream_and_roundtrips() {
    let (w, h) = (48, 48);
    let (s_off, r_off, _) = pyramid(w, h, 5, 4, |e| e);
    let (s_on, r_on, _) = pyramid(w, h, 5, 4, |e| e.with_temporal_mvp(true));
    assert_exact(&s_off, &r_off, w, h);
    assert_exact(&s_on, &r_on, w, h);
    assert_ne!(s_off, s_on);
}

// ---- golden pins (black-box validated, see
// fixture_bytes/r453-generation-notes.md) ----

#[test]
fn golden_lowdelay_b_refs4_tmvp_pin() {
    let (w, h) = (64, 48);
    let (stream, recons, _) = low_delay(w, h, 7, |e| {
        e.with_b_slices(true)
            .with_refs(4)
            .with_temporal_mvp(true)
            .with_loop_filters(LoopFilterCfg::all())
    });
    let golden: &[u8] = include_bytes!("fixture_bytes/r453-lowdelay-b-refs4-tmvp-qp29.hevc");
    assert_eq!(stream, golden, "stream drifted off the validated pin");
    assert_exact(golden, &recons, w, h);
}

#[test]
fn golden_pyramid_refs2_tmvp_pin() {
    let (w, h) = (64, 48);
    let (stream, recons, _) = pyramid(w, h, 9, 8, |e| e.with_refs(2).with_temporal_mvp(true));
    let golden: &[u8] = include_bytes!("fixture_bytes/r453-pyramid-refs2-tmvp-qp29.hevc");
    assert_eq!(stream, golden, "stream drifted off the validated pin");
    assert_exact(golden, &recons, w, h);
}

#[test]
fn golden_tree_pyramid_refs3_tmvp_pin() {
    let (w, h) = (64, 48);
    let (stream, recons, _) = pyramid(w, h, 10, 8, |e| {
        e.with_refs(3)
            .with_temporal_mvp(true)
            .with_tree(TreeCfg::new(32).expect("ctb 32"))
            .with_loop_filters(LoopFilterCfg::all())
            .with_aq(1)
    });
    let golden: &[u8] = include_bytes!("fixture_bytes/r453-tree-pyramid-refs3-tmvp-qp29.hevc");
    assert_eq!(stream, golden, "stream drifted off the validated pin");
    assert_exact(golden, &recons, w, h);
}
