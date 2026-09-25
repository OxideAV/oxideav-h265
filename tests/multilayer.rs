//! Annex F / Annex G (MV-HEVC) multi-layer decode pins.
//!
//! `r462-mvhevc-3au.hevc` is a two-view MV-HEVC stream written by an OS
//! media framework's stereo encoder from a side-by-side source: three
//! access units of 960x960 4:2:0 8-bit, the second view (`nuh_layer_id`
//! 1, `ViewId` 1, Multiview Main) predicted from the base view through
//! the G.8.1.3 inter-layer reference picture set — its IDR_N_LP is a
//! P slice from the base view, its TRAIL_R pictures B slices with one
//! temporal and one inter-layer reference. The per-view digests are
//! those of a black-box reference decoder's per-view outputs.

mod fixture_bytes;

use fixture_bytes::md5;
use oxideav_core::{CodecParameters, Decoder, Error, Frame, Packet, TimeBase};
use oxideav_h265::nal::{collect_nal_units, NalUnit};
use oxideav_h265::sequence::{decode_annexb_sequence, LayerTarget, SequenceDecoder};

const STREAM: &[u8] = include_bytes!("fixture_bytes/r462-mvhevc-3au.hevc");
const WIDTH: usize = 960;
const HEIGHT: usize = 960;
/// Black-box per-view digests (three 960x960 4:2:0 frames each).
const VIEW0_MD5: &str = "fb7904a67778df26426b85ccf7d58ef6";
const VIEW1_MD5: &str = "26e6861a5f92798eb35b3e4dc30d9260";

fn planar(pic: &oxideav_h265::picture::Picture) -> Vec<u8> {
    pic.to_planar_u8().unwrap_or_else(|| pic.to_planar_le16())
}

/// Both views decode byte-exact against the black-box reference
/// decoder through the Annex B driver, tagged with their layer / view.
#[test]
fn mvhevc_both_views_decode_byte_exact_annexb() {
    let frames = decode_annexb_sequence(STREAM).expect("decode");
    assert_eq!(frames.len(), 6, "three access units x two views");
    let mut views: [Vec<u8>; 2] = [Vec::new(), Vec::new()];
    for (i, f) in frames.iter().enumerate() {
        assert!(f.output, "frame {i} output");
        assert_eq!(f.layer_id, (i % 2) as u8, "frame {i}: layer order");
        assert_eq!(f.view_id, (i % 2) as u16, "frame {i}: ViewId");
        assert_eq!(
            f.au_index,
            frames[i ^ 1].au_index,
            "frame {i}: both views of one access unit"
        );
        assert_eq!(
            f.poc,
            frames[i / 2 * 2].poc,
            "frame {i}: POC aligned across the AU"
        );
        let pic = f.output_picture();
        assert_eq!((pic.width_luma(), pic.height_luma()), (WIDTH, HEIGHT));
        views[i % 2].extend_from_slice(&planar(&pic));
    }
    assert_eq!(md5::hex(&views[0]), VIEW0_MD5, "base view digest");
    assert_eq!(md5::hex(&views[1]), VIEW1_MD5, "second view digest");
}

/// `LayerTarget::Layer(0)` / `Ols(0)` decode the base view alone;
/// `Layer(1)` / `View(1)` decode both layers but output only the
/// second view.
#[test]
fn mvhevc_layer_targets_select_views() {
    let run = |target: LayerTarget| -> Vec<(u8, Vec<u8>)> {
        let mut dec = SequenceDecoder::new();
        dec.set_layer_target(target);
        dec.push_annexb(STREAM).expect("push");
        dec.finish()
            .expect("finish")
            .into_iter()
            .filter(|f| f.output)
            .map(|f| (f.layer_id, planar(&f.output_picture())))
            .collect()
    };
    for t in [LayerTarget::Layer(0), LayerTarget::Ols(0)] {
        let out = run(t);
        assert_eq!(out.len(), 3, "{t:?}: base view only");
        assert!(out.iter().all(|(l, _)| *l == 0));
        let all: Vec<u8> = out.into_iter().flat_map(|(_, p)| p).collect();
        assert_eq!(md5::hex(&all), VIEW0_MD5, "{t:?}");
    }
    for t in [LayerTarget::Layer(1), LayerTarget::View(1)] {
        let out = run(t);
        assert_eq!(out.len(), 3, "{t:?}: second view only");
        assert!(out.iter().all(|(l, _)| *l == 1));
        let all: Vec<u8> = out.into_iter().flat_map(|(_, p)| p).collect();
        assert_eq!(md5::hex(&all), VIEW1_MD5, "{t:?}");
    }
    let both = run(LayerTarget::View(0));
    assert_eq!(both.len(), 3);
    assert!(both.iter().all(|(l, _)| *l == 0));
}

/// The coded form of a NAL unit (two-byte header + escaped payload).
fn coded(u: &NalUnit) -> Vec<u8> {
    let b0 = (u.header.nal_unit_type << 1) | (u.header.nuh_layer_id >> 5);
    let b1 = (u.header.nuh_layer_id << 3) | (u.header.temporal_id + 1);
    let mut out = vec![b0, b1];
    out.extend_from_slice(&u.escaped);
    out
}

/// Build the layered HEIF extradata — an `hvcC` record with the base
/// layer's parameter sets followed by an `lhvC` record with the
/// non-base layers' — and one length-prefixed sample per access unit.
fn layered_extradata_and_samples(stream: &[u8]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let units = collect_nal_units(stream).expect("nal walk");
    let sps0 = units
        .iter()
        .find(|u| u.header.nal_unit_type == 33 && u.header.nuh_layer_id == 0)
        .expect("base SPS");
    let sps = oxideav_h265::SeqParameterSet::parse(&sps0.rbsp).expect("SPS parse");
    let ptl = &sps.ptl;
    let mut hvcc = vec![
        1u8,
        (ptl.general_profile_space << 6)
            | (u8::from(ptl.general_tier_flag) << 5)
            | ptl.general_profile_idc,
    ];
    hvcc.extend_from_slice(&ptl.general_profile_compatibility_flags.to_be_bytes());
    hvcc.extend_from_slice(&ptl.general_constraint_indicator_flags.to_be_bytes()[2..]);
    hvcc.push(ptl.general_level_idc);
    hvcc.extend_from_slice(&0xF000u16.to_be_bytes());
    hvcc.push(0xFC);
    hvcc.push(0xFC | sps.chroma_format_idc);
    hvcc.push(0xF8 | sps.bit_depth_luma_minus8);
    hvcc.push(0xF8 | sps.bit_depth_chroma_minus8);
    hvcc.extend_from_slice(&0u16.to_be_bytes());
    hvcc.push(0x0F);
    let push_arrays = |rec: &mut Vec<u8>, layer: u8| {
        let arrays: Vec<u8> = [32u8, 33, 34]
            .into_iter()
            .filter(|t| {
                units
                    .iter()
                    .any(|u| u.header.nal_unit_type == *t && u.header.nuh_layer_id == layer)
            })
            .collect();
        rec.push(arrays.len() as u8);
        for t in arrays {
            let nals: Vec<Vec<u8>> = units
                .iter()
                .filter(|u| u.header.nal_unit_type == t && u.header.nuh_layer_id == layer)
                .map(coded)
                .collect();
            rec.push(0x80 | t);
            rec.extend_from_slice(&(nals.len() as u16).to_be_bytes());
            for n in nals {
                rec.extend_from_slice(&(n.len() as u16).to_be_bytes());
                rec.extend_from_slice(&n);
            }
        }
    };
    push_arrays(&mut hvcc, 0);
    // lhvC: version, min_spatial_segmentation (reserved 1111 + 0),
    // parallelismType, numTemporalLayers 1 / nested / lengthSize 4.
    let mut lhvc = vec![1u8, 0xF0, 0x00, 0xFC, 0x0F];
    push_arrays(&mut lhvc, 1);
    hvcc.extend_from_slice(&lhvc);

    // One sample per access unit: the VCL NAL units (and the in-band
    // SEI) split at each base-layer first slice.
    let mut samples: Vec<Vec<u8>> = vec![Vec::new()];
    let mut sample_has_vcl = false;
    for u in units
        .iter()
        .filter(|u| !matches!(u.header.nal_unit_type, 32..=34))
    {
        let is_vcl = u.header.nal_unit_type < 32;
        let first = is_vcl && u.rbsp.first().is_some_and(|b| b & 0x80 != 0);
        if first && u.header.nuh_layer_id == 0 && sample_has_vcl {
            samples.push(Vec::new());
            sample_has_vcl = false;
        }
        sample_has_vcl |= is_vcl;
        let n = coded(u);
        let s = samples.last_mut().unwrap();
        s.extend_from_slice(&(n.len() as u32).to_be_bytes());
        s.extend_from_slice(&n);
    }
    (hvcc, samples)
}

fn collect_frames(dec: &mut Box<dyn Decoder>, samples: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut take = |dec: &mut Box<dyn Decoder>| loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => {
                assert_eq!(v.planes[0].stride, WIDTH);
                let mut f = Vec::new();
                for p in &v.planes {
                    f.extend_from_slice(&p.data);
                }
                out.push(f);
            }
            Ok(_) => panic!("non-video frame"),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("receive: {e}"),
        }
    };
    for (i, s) in samples.iter().enumerate() {
        let mut packet = Packet::new(0, TimeBase::new(1, 24), s.clone());
        packet.pts = Some(i as i64);
        dec.send_packet(&packet).expect("send");
        take(dec);
    }
    dec.flush().expect("flush");
    take(dec);
    out
}

/// The registry decoder with `hvcC` + `lhvC` extradata and
/// length-prefixed access-unit samples emits both views per access
/// unit, base view first, byte-exact against the black-box decoder.
#[test]
fn mvhevc_registry_hvcc_lhvc_extradata_emits_both_views() {
    let (extradata, samples) = layered_extradata_and_samples(STREAM);
    assert_eq!(samples.len(), 3);
    let mut params = CodecParameters::video("h265".into());
    params.extradata = extradata.clone();
    let mut dec = oxideav_h265::make_decoder(&params).expect("decoder");
    let frames = collect_frames(&mut dec, &samples);
    assert_eq!(frames.len(), 6);
    let v0: Vec<u8> = frames.iter().step_by(2).flatten().copied().collect();
    let v1: Vec<u8> = frames
        .iter()
        .skip(1)
        .step_by(2)
        .flatten()
        .copied()
        .collect();
    assert_eq!(md5::hex(&v0), VIEW0_MD5, "base view through the registry");
    assert_eq!(md5::hex(&v1), VIEW1_MD5, "second view through the registry");

    // `layer` / `view` / `ols` options pick one view.
    for (key, val, want) in [
        ("layer", "0", VIEW0_MD5),
        ("layer", "1", VIEW1_MD5),
        ("view", "1", VIEW1_MD5),
        ("ols", "0", VIEW0_MD5),
    ] {
        let mut params = CodecParameters::video("h265".into());
        params.extradata = extradata.clone();
        params.options = params.options.set(key, val);
        let mut dec = oxideav_h265::make_decoder(&params).expect("decoder");
        let frames = collect_frames(&mut dec, &samples);
        assert_eq!(frames.len(), 3, "{key}={val}");
        let all: Vec<u8> = frames.into_iter().flatten().collect();
        assert_eq!(md5::hex(&all), want, "{key}={val}");
    }
    let mut params = CodecParameters::video("h265".into());
    params.options = params.options.set("layer", "64");
    assert!(oxideav_h265::make_decoder(&params).is_err());
}
