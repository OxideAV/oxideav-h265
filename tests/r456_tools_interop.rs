//! Round-456 encoder-tool golden pins.
//!
//! Two composed streams from the quadtree coder's round-456 tool set
//! — RDOQ, sign data hiding, `max_transform_hierarchy_depth_* == 2`,
//! weighted prediction, WPP, tiles, scaling lists — each validated
//! OUT OF BAND against a black-box reference HEVC decoder (byte-exact
//! to the encoder reconstruction; see
//! `fixture_bytes/r456-generation-notes.md`). The tests pin:
//!
//! 1. the encoder still emits the golden bytes (any drift must be
//!    re-validated black-box and re-pinned deliberately);
//! 2. this crate's decoder reproduces the encoder reconstruction
//!    sample-exactly in display order.

use oxideav_h265::encoder::ctu::{TileLayout, TreeCfg};
use oxideav_h265::encoder::inter::{FrameRecon, LowDelayPEncoder, YuvFrame};
use oxideav_h265::encoder::loopfilter::LoopFilterCfg;
use oxideav_h265::encoder::pyramid::PyramidEncoder;
use oxideav_h265::sequence::decode_annexb_sequence;

/// Deterministic per-pixel hash noise (stable across frames).
fn hash_noise(x: i64, y: i64, seed: u64) -> i32 {
    let mut h = (x as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add(seed);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h & 0xFF) as i32
}

/// A textured world panning (2, 1) px/frame under a 100 % -> 60 %
/// luminance fade, with a brighter square drifting the other way.
fn scene(w: usize, h: usize, n: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let (cw, ch) = (w / 2, h / 2);
    (0..n)
        .map(|f| {
            let fi = f as i64;
            let gain = 100 - (fi * 40) / (n as i64 - 1).max(1);
            let mut y = vec![0u8; w * h];
            for j in 0..h {
                for i in 0..w {
                    let (wx, wy) = (i as i64 + fi * 2, j as i64 + fi);
                    let coarse = hash_noise(wx >> 4, wy >> 4, 1) / 2;
                    let fine = hash_noise(wx, wy, 0x55) / 8;
                    let stripes = (((wx * 3 + wy * 2) / 7 % 13) * 3) as i32;
                    let mut v = 60 + coarse + fine + stripes;
                    let sx = 12 + (n as i64 - fi) * 2;
                    if (i as i64) >= sx && (i as i64) < sx + 24 && (8..32).contains(&j) {
                        v += 80;
                    }
                    y[j * w + i] = ((v.clamp(0, 255) as i64 * gain) / 100) as u8;
                }
            }
            let cb: Vec<u8> = (0..cw * ch)
                .map(|k| {
                    let (x, y) = ((k % cw) as i64 + fi, (k / cw) as i64);
                    ((108 + hash_noise(x >> 3, y >> 3, 2) / 6) as i64 * gain / 100) as u8
                })
                .collect();
            let cr: Vec<u8> = (0..cw * ch)
                .map(|k| {
                    let (x, y) = ((k % cw) as i64 + fi, (k / cw) as i64);
                    ((100 + hash_noise(x >> 3, y >> 3, 3) / 6) as i64 * gain / 100) as u8
                })
                .collect();
            (y, cb, cr)
        })
        .collect()
}

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

/// GOP-4 pyramid + tail at CTB 64 with RDOQ, sign hiding, depth-2
/// RQTs, weighted prediction and WPP, deblocking + SAO, QP 30.
#[test]
fn pyramid_rdoq_sdh_tu2_wp_wpp_pin() {
    let (w, h) = (96, 64);
    let frames = scene(w, h, 6);
    let cfg = TreeCfg::new(64)
        .expect("ctb 64")
        .with_rdoq(true)
        .with_sign_hiding(true)
        .with_tu_depth(2, 2)
        .with_weighted_pred(true)
        .with_wpp(true);
    let mut enc = PyramidEncoder::new(w, h, 30, 4)
        .expect("encoder")
        .with_tree(cfg)
        .with_loop_filters(LoopFilterCfg::all());
    let mut aus = Vec::new();
    for (y, cb, cr) in &frames {
        aus.extend(enc.encode_frame(&YuvFrame { y, cb, cr }).expect("frame"));
    }
    aus.extend(enc.flush());
    let mut stream = Vec::new();
    let mut recons: Vec<Option<FrameRecon>> = (0..frames.len()).map(|_| None).collect();
    for au in aus {
        stream.extend_from_slice(&au.au);
        recons[au.display_order] = Some(au.recon);
    }
    let recons: Vec<FrameRecon> = recons.into_iter().map(|r| r.expect("coded")).collect();
    let golden: &[u8] = include_bytes!("fixture_bytes/r456-pyramid-rdoq-sdh-tu2-wp-wpp-qp30.hevc");
    assert_eq!(stream, golden, "stream drifted off the validated pin");
    assert_display_order_exact(&stream, &recons, w, h);
}

/// Low-delay B at CTB 32 over an explicit 2x2 tile grid with RDOQ,
/// default scaling lists and AQ 1, deblocking + SAO, QP 29.
#[test]
fn lowdelay_tiles_sl_rdoq_aq_pin() {
    let (w, h) = (96, 64);
    let frames = scene(w, h, 4);
    let cfg = TreeCfg::new(32)
        .expect("ctb 32")
        .with_rdoq(true)
        .with_scaling_lists(1)
        .with_tiles(TileLayout::explicit(&[1], &[1]));
    let mut enc = LowDelayPEncoder::new(w, h, 29, 0)
        .expect("encoder")
        .with_tree(cfg)
        .with_b_slices(true)
        .with_aq(1)
        .with_loop_filters(LoopFilterCfg::all());
    let mut stream = Vec::new();
    let mut recons = Vec::new();
    for (y, cb, cr) in &frames {
        let f = enc.encode_frame(&YuvFrame { y, cb, cr }).expect("frame");
        stream.extend_from_slice(&f.au);
        recons.push(f.recon);
    }
    let golden: &[u8] = include_bytes!("fixture_bytes/r456-lowdelay-tiles-sl-rdoq-aq-qp29.hevc");
    assert_eq!(stream, golden, "stream drifted off the validated pin");
    assert_display_order_exact(&stream, &recons, w, h);
}
