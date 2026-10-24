//! The speed-first quadtree decisions (`TreeCfg::with_speed`) and the
//! stream-control options (forced keyframes, non-reference pictures,
//! per-layer QP offsets, explicit level / tier): every stream decodes
//! through the crate's decoder to exactly the encoder's
//! reconstruction, in display order.

use oxideav_h265::decode_annexb_sequence;
use oxideav_h265::encoder::ctu::TreeCfg;
use oxideav_h265::encoder::inter::{LowDelayPEncoder, YuvFrame, YuvFrameWide};
use oxideav_h265::encoder::loopfilter::LoopFilterCfg;
use oxideav_h265::encoder::pcm::PcmLayout;
use oxideav_h265::encoder::pyramid::PyramidEncoder;

/// A panning, textured 4:2:0 sequence at `depth` bits (`n` frames of
/// `w`x`h`), as planar `u16` frames.
fn sequence(w: usize, h: usize, n: usize, depth: u8) -> Vec<(Vec<u16>, Vec<u16>, Vec<u16>)> {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut noise = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % 9) as i32 - 4
    };
    let max = (1i32 << depth) - 1;
    let scale = 1i32 << (depth - 8);
    (0..n)
        .map(|f| {
            let pan = (f * 3) as i32;
            let mut y = vec![0u16; w * h];
            for j in 0..h {
                for i in 0..w {
                    let x = i as i32 + pan;
                    let v = 128 + ((x * 7) % 61 - 30) + ((j as i32 * 5 + x / 3) % 37 - 18) + noise();
                    y[j * w + i] = (v * scale).clamp(0, max) as u16;
                }
            }
            let (cw, ch) = (w / 2, h / 2);
            let mut cb = vec![0u16; cw * ch];
            let mut cr = vec![0u16; cw * ch];
            for j in 0..ch {
                for i in 0..cw {
                    let x = i as i32 + pan / 2;
                    cb[j * cw + i] = ((128 + (x % 23) - 11) * scale).clamp(0, max) as u16;
                    cr[j * cw + i] = ((128 + (j as i32 % 19) - 9 + noise()) * scale).clamp(0, max) as u16;
                }
            }
            (y, cb, cr)
        })
        .collect()
}

fn narrow(p: &[u16]) -> Vec<u8> {
    p.iter().map(|&v| v as u8).collect()
}

/// Decode `stream` and compare each output picture (display order)
/// with `recons` (planar planes concatenated per picture).
fn assert_decodes_to(stream: &[u8], recons: &[Vec<u8>], label: &str) {
    let frames = decode_annexb_sequence(stream).expect("decodes");
    let out: Vec<_> = frames.iter().filter(|f| f.output).collect();
    assert_eq!(out.len(), recons.len(), "{label}: picture count");
    for (k, (f, want)) in out.iter().zip(recons).enumerate() {
        let pic = f.output_picture();
        let got = pic.to_planar_u8().unwrap_or_else(|| pic.to_planar_le16());
        assert!(got == *want, "{label}: picture {k} differs from the reconstruction");
    }
}

fn fast_tree(ctb: usize) -> TreeCfg {
    TreeCfg::new(ctb).unwrap().with_speed(1)
}

#[test]
fn fast_low_delay_decodes_to_recon() {
    let (w, h) = (96, 64);
    let seq = sequence(w, h, 6, 8);
    for (ctb, wpp, b) in [(32, false, false), (64, true, true), (16, false, true)] {
        let mut enc = LowDelayPEncoder::new(w, h, 30, 4)
            .unwrap()
            .with_tree(fast_tree(ctb).with_wpp(wpp))
            .with_b_slices(b)
            .with_threads(if wpp { 3 } else { 1 })
            .with_loop_filters(LoopFilterCfg::all());
        let mut stream = Vec::new();
        let mut recons = Vec::new();
        for (k, (y, cb, cr)) in seq.iter().enumerate() {
            if k == 3 {
                enc.force_keyframe();
            }
            let (y, cb, cr) = (narrow(y), narrow(cb), narrow(cr));
            let f = enc.encode_frame(&YuvFrame { y: &y, cb: &cb, cr: &cr }).unwrap();
            assert_eq!(f.keyframe, k == 0 || k == 3, "ctb {ctb}: keyframe {k}");
            stream.extend_from_slice(&f.au);
            recons.push([f.recon.y, f.recon.cb, f.recon.cr].concat());
        }
        assert_decodes_to(&stream, &recons, &format!("low-delay ctb {ctb} wpp {wpp} b {b}"));
    }
}

#[test]
fn fast_pyramid_with_keyframes_and_non_reference_pictures() {
    let (w, h) = (80, 48);
    let seq = sequence(w, h, 13, 8);
    let mut enc = PyramidEncoder::new(w, h, 28, 4)
        .unwrap()
        .with_tree(fast_tree(64).with_wpp(true))
        .with_threads(2)
        .with_loop_filters(LoopFilterCfg::all())
        .with_non_reference_pictures(true)
        .with_layer_qp_offsets(&[0, 2, 5])
        .with_level(123, true);
    let mut stream = Vec::new();
    let mut recons: Vec<Option<Vec<u8>>> = vec![None; seq.len()];
    let mut decode_order = Vec::new();
    let mut nal_types = Vec::new();
    let mut layer_qps = Vec::new();
    let mut take = |aus: Vec<oxideav_h265::encoder::pyramid::PyramidAu>,
                    stream: &mut Vec<u8>,
                    recons: &mut Vec<Option<Vec<u8>>>| {
        for au in aus {
            stream.extend_from_slice(&au.au);
            decode_order.push(au.display_order);
            // The last NAL of the access unit is its slice.
            let pos = au.au.windows(4).rposition(|v| v == [0, 0, 0, 1]).unwrap();
            nal_types.push((au.au[pos + 4] >> 1) & 63);
            layer_qps.push((au.layer, au.qp));
            recons[au.display_order] = Some([au.recon.y, au.recon.cb, au.recon.cr].concat());
        }
    };
    for (k, (y, cb, cr)) in seq.iter().enumerate() {
        if k == 7 {
            enc.force_keyframe();
        }
        let (y, cb, cr) = (narrow(y), narrow(cb), narrow(cr));
        let aus = enc.encode_frame(&YuvFrame { y: &y, cb: &cb, cr: &cr }).unwrap();
        take(aus, &mut stream, &mut recons);
    }
    take(enc.flush(), &mut stream, &mut recons);
    // The frames before the forced keyframe close as a two-frame
    // mini-GOP; the tail after the second IDR as a five-frame one.
    assert_eq!(decode_order, [0, 4, 2, 1, 3, 6, 5, 7, 11, 9, 8, 10, 12]);
    // IDR_N_LP for the keyframes, TRAIL_N for the deepest layers.
    assert_eq!(nal_types, [20, 1, 1, 0, 0, 1, 0, 20, 1, 1, 0, 0, 1]);
    for &(layer, qp) in &layer_qps[1..] {
        assert_eq!(qp, 28 + [0, 2, 5][usize::from(layer).min(2)]);
    }
    // The signalled level and tier.
    let nals = oxideav_h265::nal::collect_nal_units(&stream).unwrap();
    let sps_nal = nals.iter().find(|n| n.header.nal_unit_type == 33).expect("an SPS");
    let sps = oxideav_h265::sps::SeqParameterSet::parse(&sps_nal.rbsp).unwrap();
    assert_eq!(sps.ptl.general_level_idc, 123);
    assert!(sps.ptl.general_tier_flag);
    let recons: Vec<Vec<u8>> = recons.into_iter().map(Option::unwrap).collect();
    assert_decodes_to(&stream, &recons, "pyramid");
}

#[test]
fn fast_main10_pyramid_decodes_to_recon() {
    let (w, h) = (64, 64);
    let seq = sequence(w, h, 5, 10);
    let layout = PcmLayout {
        chroma_format_idc: 1,
        bit_depth: 10,
        ..PcmLayout::default()
    };
    let mut enc = PyramidEncoder::new_fmt(w, h, 30, 4, layout)
        .unwrap()
        .with_tree(fast_tree(32))
        .with_loop_filters(LoopFilterCfg::all());
    let mut stream = Vec::new();
    let mut recons: Vec<Option<Vec<u8>>> = vec![None; seq.len()];
    let le16 = |p: &[u16]| -> Vec<u8> { p.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let mut aus = Vec::new();
    for (y, cb, cr) in &seq {
        aus.extend(enc.encode_frame_wide(&YuvFrameWide::new(y, cb, cr)).unwrap());
    }
    aus.extend(enc.flush_wide());
    for au in aus {
        stream.extend_from_slice(&au.au);
        recons[au.display_order] =
            Some([le16(&au.recon.y), le16(&au.recon.cb), le16(&au.recon.cr)].concat());
    }
    let recons: Vec<Vec<u8>> = recons.into_iter().map(Option::unwrap).collect();
    assert_decodes_to(&stream, &recons, "main10");
}
