//! Round-460 HEIF/HEIC still-picture interop pins.
//!
//! Twenty-five HEVC still items extracted black-box from HEIC files
//! written by three real-world producers (an OS image-conversion
//! tool, a third-party HEVC encoder through a HEIF library CLI, and a
//! general-purpose image converter) — `tests/fixture_bytes/r460/`,
//! generation commands and SHA-256 sums in
//! `fixture_bytes/r460-generation-notes.md`. Every stream decodes
//! byte-exact against a black-box reference decoder; the pins hold
//! the MD5 of the §7.4.3.2.1 conformance-cropped planar output
//! (8-bit samples as bytes, deeper as 16-bit little-endian, planes in
//! Y / Cb / Cr order — the reference decoder's raw-video layout).
//!
//! The axes: Main / Main 10 / Main Still Picture / Main 4:4:4 Still
//! Picture profile signalling, 4:2:0 / 4:2:2 / 4:4:4 / monochrome at
//! 8 / 10 / 12 bits, lossless (`cu_transquant_bypass` pictures and
//! RGB-as-4:4:4), conformance windows (odd source sizes padded by the
//! producer), 2x2 and 18x14 pictures, CTB 16 with multiple slices +
//! WPP, transform skip + `cu_transquant_bypass` + RDOQ, SAO off /
//! strong-intra-smoothing off / constrained intra / default scaling
//! lists / sign hiding off, 8x8 quantization groups with chroma QP
//! offsets and deblocking offsets, VUI timing + HRD + default display
//! window + AUD, and the `hvcC` length-prefixed transport form the
//! HEIF `hvc1` item carries.

mod fixture_bytes;

use fixture_bytes::md5;
use oxideav_core::{CodecParameters, Error, Frame, Packet, TimeBase};
use oxideav_h265::nal::NalIter;
use oxideav_h265::sequence::decode_annexb_sequence;
use oxideav_h265::sps::SeqParameterSet;

struct Still {
    name: &'static str,
    hevc: &'static [u8],
    /// Output (cropped) width x height in luma samples.
    width: usize,
    height: usize,
    /// MD5 of the cropped planar output.
    md5: &'static str,
}

macro_rules! still {
    ($name:literal, $w:expr, $h:expr, $md5:literal) => {
        Still {
            name: $name,
            hevc: include_bytes!(concat!("fixture_bytes/r460/", $name, ".hevc")),
            width: $w,
            height: $h,
            md5: $md5,
        }
    };
}

const STILLS: &[Still] = &[
    // Third-party HEVC encoder through the HEIF library CLI.
    still!("henc-133", 134, 98, "010ef8ba7e899d9d38261544282d9b77"),
    still!("henc-133-L", 134, 98, "c282e3691c547e148e01f509e26e0023"),
    still!("henc-133-444", 134, 98, "b7d88b6d232b5337d97d9b890d54ed91"),
    still!("henc-133-422", 134, 98, "a512c82028696fac044910f2b880aef2"),
    still!(
        "henc-133-ctu16-slices-wpp",
        134,
        98,
        "e99c47d70d67a1bffd4c2e8dbd1f5528"
    ),
    still!(
        "henc-133-tskip-culossless-rdoq",
        134,
        98,
        "6d3f99c0dee29300adb21bad466ce77e"
    ),
    still!(
        "henc-133-nosao-nosis-cip-sl",
        134,
        98,
        "5d7895cf92810d3dfce732f910b21f9a"
    ),
    still!(
        "henc-133-aq8-cqp-deblock",
        134,
        98,
        "b23870b64066df35edec841d1615c885"
    ),
    still!(
        "henc-133-vui-hrd-sei",
        134,
        98,
        "ff433c201d27f0ba47722668d8142688"
    ),
    still!("henc-17", 64, 64, "092725f5fa09cf6b27e18ac81008018f"),
    still!("henc-120-b10", 120, 90, "41700e576494b7fc9ae7610a4d012f89"),
    still!(
        "henc-120-b10-444-tskip",
        120,
        90,
        "5f2fe3088465ad09c2954bbdfb1a21a4"
    ),
    still!(
        "henc-120-b12-422",
        120,
        90,
        "1562f52cecf435d51b6f9f5b22780729"
    ),
    still!(
        "henc-120-L-b10",
        120,
        90,
        "10e9726a9be5b3aae9faf93a5ba153b6"
    ),
    still!("henc-g160", 160, 120, "0c7a3ed3a664763b53f0d1306d807e17"),
    still!(
        "henc-g160-b10",
        160,
        120,
        "c19dc1879127bdbcb1d8618241b7be67"
    ),
    // OS image-conversion tool (hardware / system HEVC encoder).
    still!("sips-133", 134, 98, "dc25546630b4db83b69ce72431f3cb1c"),
    still!("sips-133-q100", 134, 98, "1b8480253a5735e2b777b64994d37f07"),
    still!("sips-120-16", 120, 90, "f3ef2d48c84641d19229ad92aabd0e59"),
    still!(
        "sips-120-16-q100",
        120,
        90,
        "145ee0bd894645ff112a26361784688e"
    ),
    still!("sips-17", 18, 14, "b7e0a28d81f1484caba4e53402f6db35"),
    still!("sips-2x2", 2, 2, "9518ae5d6019f26ed9c61725abbe59d5"),
    still!("sips-g160", 160, 120, "d85a8416b171c164d4721030a0d91149"),
    // General-purpose image converter.
    still!(
        "magick-133-444",
        134,
        98,
        "56fa4ac9c41c3618827dc386b7768dc1"
    ),
    still!(
        "magick-120-d12",
        120,
        90,
        "66357a22620abe911fde3491a251c1ae"
    ),
];

/// The reference decoder's raw planar layout of one output picture.
fn planar(pic: &oxideav_h265::picture::Picture) -> Vec<u8> {
    pic.to_planar_u8().unwrap_or_else(|| pic.to_planar_le16())
}

/// Every vendored still decodes (Annex B path) to exactly one output
/// picture of the cropped geometry with the pinned digest.
#[test]
fn heic_stills_decode_byte_exact_annexb() {
    for s in STILLS {
        let frames = decode_annexb_sequence(s.hevc).unwrap_or_else(|e| panic!("{}: {e}", s.name));
        assert_eq!(frames.len(), 1, "{}: one picture", s.name);
        let f = &frames[0];
        assert!(f.output, "{}: output picture", s.name);
        let pic = f.output_picture();
        assert_eq!(
            (pic.width_luma(), pic.height_luma()),
            (s.width, s.height),
            "{}: cropped geometry",
            s.name
        );
        assert_eq!(
            md5::hex(&planar(&pic)),
            s.md5,
            "{}: cropped output digest",
            s.name
        );
    }
}

/// The SPS of every still is what a HEIF reader hands this crate:
/// most producers signal a still-picture profile (Main Still Picture
/// `general_profile_idc == 3`, or Main 10 with
/// `general_one_picture_only_constraint_flag`), the rest a plain Main
/// / Main 10 / format-range-extensions profile. (The third-party
/// encoder's Main Still Picture streams carry
/// `sps_max_dec_pic_buffering_minus1[0] == 2`, against the A.3.4
/// `== 0` constraint — real-world stills are NOT to be rejected on
/// that.)
#[test]
fn heic_stills_profile_signalling_parses() {
    let mut still_profiles = 0usize;
    for s in STILLS {
        let sps_rbsp = NalIter::new(s.hevc)
            .flatten()
            .find(|u| u.header.nal_unit_type == 33)
            .map(|u| u.rbsp)
            .unwrap_or_else(|| panic!("{}: SPS present", s.name));
        let sps =
            SeqParameterSet::parse(&sps_rbsp).unwrap_or_else(|e| panic!("{}: SPS: {e}", s.name));
        let ptl = &sps.ptl;
        if ptl.is_still_picture_profile() {
            still_profiles += 1;
            assert!(
                ptl.general_profile_idc == 3
                    || ptl.profile_compatible(3)
                    || ptl.one_picture_only_constraint_flag(),
                "{}: still profile indication",
                s.name
            );
        }
        // The 48-bit constraint field starts with the four source /
        // packing flags; progressive stills set the first.
        assert_ne!(
            ptl.general_constraint_indicator_flags >> 44 & 0x8,
            0,
            "{}: general_progressive_source_flag",
            s.name
        );
    }
    assert!(
        still_profiles >= 15,
        "most producers signal a still-picture profile (got {still_profiles})"
    );
}

/// Build the ISO/IEC 14496-15 §8.3.3.1 `HEVCDecoderConfigurationRecord`
/// a HEIF `hvcC` item property carries for `stream`: the parameter
/// sets go out of band (one array per NAL type), the slice data is
/// re-framed with 4-byte big-endian lengths.
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
    let arrays: Vec<u8> = [32u8, 33, 34]
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
        .filter(|u| !matches!(u.header.nal_unit_type, 32..=34))
    {
        let n = coded(u);
        sample.extend_from_slice(&(n.len() as u32).to_be_bytes());
        sample.extend_from_slice(&n);
    }
    (rec, sample)
}

/// The HEIF transport form: `hvcC` extradata + one length-prefixed
/// sample through the registry decoder yields the same cropped frame
/// as the Annex B decode, for every still.
#[test]
fn heic_stills_decode_byte_exact_hvcc_length_prefixed() {
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
                    let wide = v.planes[0].data.len() > s.width * s.height;
                    assert_eq!(
                        v.planes[0].stride,
                        if wide { s.width * 2 } else { s.width },
                        "{}: cropped luma stride",
                        s.name
                    );
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
