//! Round-479 HEIF conformance-suite still pins.
//!
//! The two `iff_hevc_single_item*` items of the HEIF conformance suite
//! (`tests/fixture_bytes/r479/`, generation notes alongside) code a
//! `bitstream_restriction` block whose `max_bytes_per_pic_denom`
//! (26 / 66) breaks its §E.3.1 bound. Annex E §E.1: VUI parameters
//! are not required for constructing the luma or chroma samples, so
//! the block is carried as coded, the violation is reportable, and
//! both pictures decode byte-exact against a black-box decoder —
//! through the Annex B path and the `hvcC` + length-prefixed form a
//! HEIF reader hands the registry decoder.

mod fixture_bytes;

use fixture_bytes::md5;
use oxideav_core::{CodecParameters, Error, Frame, Packet, TimeBase};
use oxideav_h265::nal::NalIter;
use oxideav_h265::sequence::decode_annexb_sequence;
use oxideav_h265::sps::SeqParameterSet;

struct Still {
    name: &'static str,
    hevc: &'static [u8],
    width: usize,
    height: usize,
    bit_depth: u8,
    /// The coded (out-of-range) `max_bytes_per_pic_denom`.
    max_bytes_per_pic_denom: u32,
    /// MD5 of the planar output (bytes at 8 bits, 16-bit LE above).
    md5: &'static str,
}

const STILLS: &[Still] = &[
    Still {
        name: "iff_hevc_single_item",
        hevc: include_bytes!("fixture_bytes/r479/iff_hevc_single_item.hevc"),
        width: 1280,
        height: 720,
        bit_depth: 8,
        max_bytes_per_pic_denom: 26,
        md5: "e0b63e73bac27800c8a2b66ef2583a8e",
    },
    Still {
        name: "iff_hevc_single_item_main10",
        hevc: include_bytes!("fixture_bytes/r479/iff_hevc_single_item_main10.hevc"),
        width: 1920,
        height: 1080,
        bit_depth: 10,
        max_bytes_per_pic_denom: 66,
        md5: "451c58206201b952ee095f3188077871",
    },
];

fn planar(pic: &oxideav_h265::picture::Picture) -> Vec<u8> {
    pic.to_planar_u8().unwrap_or_else(|| pic.to_planar_le16())
}

/// The SPS parses; the out-of-range `bitstream_restriction` value is
/// carried as coded and named by `range_violation`.
#[test]
fn vui_bitstream_restriction_violation_is_carried_and_reported() {
    for s in STILLS {
        let sps_rbsp = NalIter::new(s.hevc)
            .flatten()
            .find(|u| u.header.nal_unit_type == 33)
            .map(|u| u.rbsp)
            .unwrap_or_else(|| panic!("{}: SPS present", s.name));
        let sps =
            SeqParameterSet::parse(&sps_rbsp).unwrap_or_else(|e| panic!("{}: SPS: {e}", s.name));
        assert_eq!(sps.bit_depth_luma_minus8 + 8, s.bit_depth, "{}", s.name);
        let vui = sps
            .vui_parameters
            .as_ref()
            .unwrap_or_else(|| panic!("{}: VUI present", s.name));
        assert!(vui.bitstream_restriction_flag, "{}", s.name);
        let bsr = vui
            .bitstream_restriction
            .unwrap_or_else(|| panic!("{}: block carried", s.name));
        assert_eq!(
            bsr.max_bytes_per_pic_denom, s.max_bytes_per_pic_denom,
            "{}",
            s.name
        );
        assert_eq!(
            bsr.range_violation(),
            Some(("max_bytes_per_pic_denom", s.max_bytes_per_pic_denom)),
            "{}",
            s.name
        );
        assert!(!bsr.is_conformant(), "{}", s.name);
        // The other four fields are inside their bounds.
        assert!(bsr.min_spatial_segmentation_idc <= 4095);
        assert_eq!(bsr.max_bits_per_min_cu_denom, 0);
        assert!(bsr.log2_max_mv_length_horizontal <= 15);
        assert!(bsr.log2_max_mv_length_vertical <= 15);
    }
}

/// Annex B decode: one output picture, byte-exact against the
/// black-box reference digest.
#[test]
fn heif_conformance_stills_decode_byte_exact_annexb() {
    for s in STILLS {
        let frames = decode_annexb_sequence(s.hevc).unwrap_or_else(|e| panic!("{}: {e}", s.name));
        assert_eq!(frames.len(), 1, "{}: one picture", s.name);
        let f = &frames[0];
        assert!(f.output, "{}: output picture", s.name);
        let pic = f.output_picture();
        assert_eq!(
            (pic.width_luma(), pic.height_luma()),
            (s.width, s.height),
            "{}: geometry",
            s.name
        );
        assert_eq!(pic.bit_depth_luma(), s.bit_depth, "{}", s.name);
        assert_eq!(md5::hex(&planar(&pic)), s.md5, "{}: output digest", s.name);
    }
}

/// Build the ISO/IEC 14496-15 §8.3.3.1 `HEVCDecoderConfigurationRecord`
/// a HEIF `hvcC` item property carries for `stream` (parameter sets
/// out of band, slice data re-framed with 4-byte lengths).
fn hvcc_and_sample(stream: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let units: Vec<_> = NalIter::new(stream).flatten().collect();
    let sps_rbsp = units
        .iter()
        .find(|u| u.header.nal_unit_type == 33)
        .map(|u| u.rbsp.clone())
        .expect("SPS");
    let sps = SeqParameterSet::parse(&sps_rbsp).expect("SPS parses");
    let ptl = &sps.ptl;
    let coded = |u: &oxideav_h265::nal::NalUnit| -> Vec<u8> {
        let mut v = vec![
            (u.header.nal_unit_type << 1) | (u.header.nuh_layer_id >> 5),
            ((u.header.nuh_layer_id & 0x1f) << 3) | (u.header.temporal_id + 1),
        ];
        v.extend_from_slice(&u.escaped);
        v
    };
    let mut rec = vec![
        1u8,
        (ptl.general_profile_space << 6)
            | (u8::from(ptl.general_tier_flag) << 5)
            | ptl.general_profile_idc,
    ];
    rec.extend_from_slice(&ptl.general_profile_compatibility_flags.to_be_bytes());
    rec.extend_from_slice(&ptl.general_constraint_indicator_flags.to_be_bytes()[2..]);
    rec.push(ptl.general_level_idc);
    rec.extend_from_slice(&0xF000u16.to_be_bytes()); // min_spatial_segmentation_idc
    rec.push(0xFC); // parallelismType
    rec.push(0xFC | sps.chroma_format_idc);
    rec.push(0xF8 | sps.bit_depth_luma_minus8);
    rec.push(0xF8 | sps.bit_depth_chroma_minus8);
    rec.extend_from_slice(&0u16.to_be_bytes()); // avgFrameRate
    rec.push(0x0F); // constantFrameRate 0, numTemporalLayers 1, temporalIdNested 1, lengthSizeMinusOne 3
    let arrays: Vec<u8> = [32u8, 33, 34, 39]
        .into_iter()
        .filter(|t| units.iter().any(|u| u.header.nal_unit_type == *t))
        .collect();
    rec.push(arrays.len() as u8);
    for t in arrays {
        let nals: Vec<Vec<u8>> = units
            .iter()
            .filter(|u| u.header.nal_unit_type == t)
            .map(coded)
            .collect();
        rec.push(0x80 | t);
        rec.extend_from_slice(&(nals.len() as u16).to_be_bytes());
        for n in nals {
            rec.extend_from_slice(&(n.len() as u16).to_be_bytes());
            rec.extend_from_slice(&n);
        }
    }
    let mut sample = Vec::new();
    for u in units
        .iter()
        .filter(|u| !matches!(u.header.nal_unit_type, 32..=34 | 39))
    {
        let n = coded(u);
        sample.extend_from_slice(&(n.len() as u32).to_be_bytes());
        sample.extend_from_slice(&n);
    }
    (rec, sample)
}

/// The HEIF transport form through the registry decoder: the same
/// digest as the Annex B decode.
#[test]
fn heif_conformance_stills_decode_byte_exact_hvcc() {
    for s in STILLS {
        let (hvcc, sample) = hvcc_and_sample(s.hevc);
        let mut params = CodecParameters::video("h265".into());
        params.extradata = hvcc;
        let mut dec =
            oxideav_h265::make_decoder(&params).unwrap_or_else(|e| panic!("{}: {e}", s.name));
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), sample))
            .unwrap_or_else(|e| panic!("{}: send: {e}", s.name));
        dec.flush()
            .unwrap_or_else(|e| panic!("{}: flush: {e}", s.name));
        let mut frames = 0usize;
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(v)) => {
                    frames += 1;
                    let mut out = Vec::new();
                    for p in &v.planes {
                        out.extend_from_slice(&p.data);
                    }
                    let bps = if s.bit_depth > 8 { 2 } else { 1 };
                    assert_eq!(v.planes[0].stride, s.width * bps, "{}: luma stride", s.name);
                    assert_eq!(md5::hex(&out), s.md5, "{}: hvcC-path digest", s.name);
                }
                Ok(_) => panic!("{}: non-video frame", s.name),
                Err(Error::Eof) => break,
                Err(e) => panic!("{}: receive: {e}", s.name),
            }
        }
        assert_eq!(frames, 1, "{}: one frame", s.name);
    }
}
