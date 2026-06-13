//! Tests for the §7.3.5 / §D.2 SEI parse.

use super::*;

/// Build a single-message SEI RBSP body: the §7.3.5 payloadType /
/// payloadSize framing (short form, both < 255) + `body` + an
/// `rbsp_trailing_bits()` `0x80` stop byte.
fn sei_rbsp_one(payload_type: u8, body: &[u8]) -> Vec<u8> {
    let mut v = vec![payload_type, body.len() as u8];
    v.extend_from_slice(body);
    v.push(0x80);
    v
}

#[test]
fn nal_type_mapping() {
    assert_eq!(SeiNalType::from_nal_unit_type(39), Some(SeiNalType::Prefix));
    assert_eq!(SeiNalType::from_nal_unit_type(40), Some(SeiNalType::Suffix));
    assert_eq!(SeiNalType::from_nal_unit_type(0), None);
    assert_eq!(SeiNalType::from_nal_unit_type(34), None);
    assert_eq!(PREFIX_SEI_NUT, 39);
    assert_eq!(SUFFIX_SEI_NUT, 40);
}

#[test]
fn extensible_byte_run_short_form() {
    let buf = [0x07];
    let mut cur = 0;
    assert_eq!(read_extensible_byte_run(&buf, &mut cur).unwrap(), 7);
    assert_eq!(cur, 1);
}

#[test]
fn extensible_byte_run_ff_accumulation() {
    // 0xFF + 0xFF + 0x03 == 255 + 255 + 3 == 513.
    let buf = [0xFF, 0xFF, 0x03];
    let mut cur = 0;
    assert_eq!(read_extensible_byte_run(&buf, &mut cur).unwrap(), 513);
    assert_eq!(cur, 3);
}

#[test]
fn extensible_byte_run_ff_then_zero() {
    // 0xFF + 0x00 == 255.
    let buf = [0xFF, 0x00];
    let mut cur = 0;
    assert_eq!(read_extensible_byte_run(&buf, &mut cur).unwrap(), 255);
    assert_eq!(cur, 2);
}

#[test]
fn extensible_byte_run_truncated() {
    // A run of only 0xFF with no terminator is truncated.
    let buf = [0xFF, 0xFF];
    let mut cur = 0;
    assert_eq!(
        read_extensible_byte_run(&buf, &mut cur),
        Err(SeiError::TruncatedHeader)
    );
}

#[test]
fn recovery_point_decode() {
    // recovery_poc_cnt = 0 (se -> ue codeNum 0 -> bit '1'), then
    // exact_match_flag = 1, broken_link_flag = 0. Bits: 1 1 0 -> 0b110
    // padded = 0xC0. payloadSize = 1 byte.
    let body = [0b1100_0000];
    let rbsp = sei_rbsp_one(6, &body);
    let msgs = parse_sei_rbsp(&rbsp, SeiNalType::Prefix).unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].payload_type, 6);
    assert_eq!(msgs[0].payload_size, 1);
    match &msgs[0].payload {
        SeiPayload::RecoveryPoint(rp) => {
            assert_eq!(rp.recovery_poc_cnt, 0);
            assert!(rp.exact_match_flag);
            assert!(!rp.broken_link_flag);
        }
        other => panic!("expected RecoveryPoint, got {other:?}"),
    }
}

#[test]
fn recovery_point_negative_poc() {
    // recovery_poc_cnt = -1: se(v) codeNum 2 -> ue '011' (3 bits),
    // then exact=0, broken=1 -> '01'. Bitstring: 011 0 1 -> 0b01101000
    // = 0x68. payloadSize = 1.
    let body = [0b0110_1000];
    let rbsp = sei_rbsp_one(6, &body);
    let msgs = parse_sei_rbsp(&rbsp, SeiNalType::Prefix).unwrap();
    match &msgs[0].payload {
        SeiPayload::RecoveryPoint(rp) => {
            assert_eq!(rp.recovery_poc_cnt, -1);
            assert!(!rp.exact_match_flag);
            assert!(rp.broken_link_flag);
        }
        other => panic!("expected RecoveryPoint, got {other:?}"),
    }
}

#[test]
fn content_light_level_decode() {
    // max_content_light_level = 0x03E8 (1000), max_pic_average = 0x00C8 (200).
    let body = [0x03, 0xE8, 0x00, 0xC8];
    let rbsp = sei_rbsp_one(144, &body);
    let msgs = parse_sei_rbsp(&rbsp, SeiNalType::Prefix).unwrap();
    match &msgs[0].payload {
        SeiPayload::ContentLightLevelInfo(cll) => {
            assert_eq!(cll.max_content_light_level, 1000);
            assert_eq!(cll.max_pic_average_light_level, 200);
        }
        other => panic!("expected ContentLightLevelInfo, got {other:?}"),
    }
}

#[test]
fn content_light_level_truncated() {
    let body = [0x03, 0xE8, 0x00]; // only 3 bytes
    let rbsp = sei_rbsp_one(144, &body);
    assert_eq!(
        parse_sei_rbsp(&rbsp, SeiNalType::Prefix),
        Err(SeiError::TruncatedPayload { payload_type: 144 })
    );
}

#[test]
fn mastering_display_decode() {
    let mut body = Vec::new();
    // primaries (x,y) for c=0,1,2.
    for c in 0u16..3 {
        body.extend_from_slice(&(1000 + c).to_be_bytes()); // x
        body.extend_from_slice(&(2000 + c).to_be_bytes()); // y
    }
    body.extend_from_slice(&15000u16.to_be_bytes()); // white_x
    body.extend_from_slice(&16000u16.to_be_bytes()); // white_y
    body.extend_from_slice(&10_000_000u32.to_be_bytes()); // max lum
    body.extend_from_slice(&50u32.to_be_bytes()); // min lum
    assert_eq!(body.len(), 24);
    let rbsp = sei_rbsp_one(137, &body);
    let msgs = parse_sei_rbsp(&rbsp, SeiNalType::Prefix).unwrap();
    match &msgs[0].payload {
        SeiPayload::MasteringDisplayColourVolume(m) => {
            assert_eq!(m.display_primaries_x, [1000, 1001, 1002]);
            assert_eq!(m.display_primaries_y, [2000, 2001, 2002]);
            assert_eq!(m.white_point_x, 15000);
            assert_eq!(m.white_point_y, 16000);
            assert_eq!(m.max_display_mastering_luminance, 10_000_000);
            assert_eq!(m.min_display_mastering_luminance, 50);
        }
        other => panic!("expected MasteringDisplayColourVolume, got {other:?}"),
    }
}

#[test]
fn mastering_display_truncated() {
    let body = vec![0u8; 23];
    let rbsp = sei_rbsp_one(137, &body);
    assert_eq!(
        parse_sei_rbsp(&rbsp, SeiNalType::Prefix),
        Err(SeiError::TruncatedPayload { payload_type: 137 })
    );
}

#[test]
fn alternative_transfer_characteristics_decode() {
    // preferred_transfer_characteristics = 18 (HLG).
    let body = [18u8];
    let rbsp = sei_rbsp_one(147, &body);
    let msgs = parse_sei_rbsp(&rbsp, SeiNalType::Prefix).unwrap();
    match &msgs[0].payload {
        SeiPayload::AlternativeTransferCharacteristics(a) => {
            assert_eq!(a.preferred_transfer_characteristics, 18);
        }
        other => panic!("expected AlternativeTransferCharacteristics, got {other:?}"),
    }
}

#[test]
fn active_parameter_sets_decode() {
    // active_video_parameter_set_id=0 (u4), self_contained=1 (u1),
    // no_update=1 (u1), num_sps_ids_minus1=0 (ue '1'),
    // active_seq_parameter_set_id[0]=0 (ue '1').
    // Bits: 0000 1 1 1 1 -> 0b0000_1111 = 0x0F.
    let body = [0x0F];
    let rbsp = sei_rbsp_one(129, &body);
    let msgs = parse_sei_rbsp(&rbsp, SeiNalType::Prefix).unwrap();
    match &msgs[0].payload {
        SeiPayload::ActiveParameterSets(a) => {
            assert_eq!(a.active_video_parameter_set_id, 0);
            assert!(a.self_contained_cvs_flag);
            assert!(a.no_parameter_set_update_flag);
            assert_eq!(a.active_seq_parameter_set_ids, vec![0]);
        }
        other => panic!("expected ActiveParameterSets, got {other:?}"),
    }
}

#[test]
fn user_data_unregistered_decode() {
    let mut body = Vec::new();
    let uuid: [u8; 16] = [
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
        0x00,
    ];
    body.extend_from_slice(&uuid);
    body.extend_from_slice(b"hello");
    let rbsp = sei_rbsp_one(5, &body);
    let msgs = parse_sei_rbsp(&rbsp, SeiNalType::Prefix).unwrap();
    match &msgs[0].payload {
        SeiPayload::UserDataUnregistered(u) => {
            assert_eq!(u.uuid_iso_iec_11578, uuid);
            assert_eq!(u.user_data_payload, b"hello");
        }
        other => panic!("expected UserDataUnregistered, got {other:?}"),
    }
}

#[test]
fn user_data_unregistered_too_short() {
    let body = vec![0u8; 15];
    let rbsp = sei_rbsp_one(5, &body);
    assert_eq!(
        parse_sei_rbsp(&rbsp, SeiNalType::Prefix),
        Err(SeiError::TruncatedPayload { payload_type: 5 })
    );
}

#[test]
fn user_data_unregistered_works_in_suffix() {
    // payload 5 is legal in both prefix and suffix branches.
    let mut body = vec![0u8; 16];
    body.push(0x42);
    let rbsp = sei_rbsp_one(5, &body);
    let msgs = parse_sei_rbsp(&rbsp, SeiNalType::Suffix).unwrap();
    match &msgs[0].payload {
        SeiPayload::UserDataUnregistered(u) => {
            assert_eq!(u.user_data_payload, vec![0x42]);
        }
        other => panic!("expected UserDataUnregistered, got {other:?}"),
    }
}

#[test]
fn user_data_registered_country_code() {
    // Single-byte country code != 0xFF, then payload.
    let body = [0x26, 0xAA, 0xBB]; // 0x26 = US country code historically
    let rbsp = sei_rbsp_one(4, &body);
    let msgs = parse_sei_rbsp(&rbsp, SeiNalType::Prefix).unwrap();
    match &msgs[0].payload {
        SeiPayload::UserDataRegisteredItuTT35(u) => {
            assert_eq!(u.country_code, 0x26);
            assert_eq!(u.country_code_extension, None);
            assert_eq!(u.payload, vec![0xAA, 0xBB]);
        }
        other => panic!("expected UserDataRegisteredItuTT35, got {other:?}"),
    }
}

#[test]
fn user_data_registered_country_code_extension() {
    let body = [0xFF, 0x01, 0xCC]; // 0xFF -> extension byte 0x01, payload 0xCC
    let rbsp = sei_rbsp_one(4, &body);
    let msgs = parse_sei_rbsp(&rbsp, SeiNalType::Prefix).unwrap();
    match &msgs[0].payload {
        SeiPayload::UserDataRegisteredItuTT35(u) => {
            assert_eq!(u.country_code, 0xFF);
            assert_eq!(u.country_code_extension, Some(0x01));
            assert_eq!(u.payload, vec![0xCC]);
        }
        other => panic!("expected UserDataRegisteredItuTT35, got {other:?}"),
    }
}

#[test]
fn decoded_picture_hash_md5_suffix() {
    // hash_type = 0 (MD5), 3 components of 16 bytes each.
    let mut body = vec![0u8];
    for c in 0..3u8 {
        for i in 0..16u8 {
            body.push(c.wrapping_mul(16).wrapping_add(i));
        }
    }
    let rbsp = sei_rbsp_one(132, &body);
    let msgs = parse_sei_rbsp(&rbsp, SeiNalType::Suffix).unwrap();
    match &msgs[0].payload {
        SeiPayload::DecodedPictureHash(h) => {
            assert_eq!(h.hash_type, 0);
            assert_eq!(h.component_hashes.len(), 3);
            match h.component_hashes[1] {
                PictureHash::Md5(md5) => assert_eq!(md5[0], 16),
                ref other => panic!("expected Md5, got {other:?}"),
            }
        }
        other => panic!("expected DecodedPictureHash, got {other:?}"),
    }
}

#[test]
fn decoded_picture_hash_crc() {
    // hash_type = 1 (CRC), 1 component (mono) of u16.
    let body = [1u8, 0xAB, 0xCD];
    let rbsp = sei_rbsp_one(132, &body);
    let msgs = parse_sei_rbsp(&rbsp, SeiNalType::Suffix).unwrap();
    match &msgs[0].payload {
        SeiPayload::DecodedPictureHash(h) => {
            assert_eq!(h.hash_type, 1);
            assert_eq!(h.component_hashes, vec![PictureHash::Crc(0xABCD)]);
        }
        other => panic!("expected DecodedPictureHash, got {other:?}"),
    }
}

#[test]
fn decoded_picture_hash_checksum() {
    // hash_type = 2 (checksum), 3 components of u32 each.
    let mut body = vec![2u8];
    for c in 0..3u32 {
        body.extend_from_slice(&(0x0100_0000u32 * (c + 1)).to_be_bytes());
    }
    let rbsp = sei_rbsp_one(132, &body);
    let msgs = parse_sei_rbsp(&rbsp, SeiNalType::Suffix).unwrap();
    match &msgs[0].payload {
        SeiPayload::DecodedPictureHash(h) => {
            assert_eq!(h.hash_type, 2);
            assert_eq!(
                h.component_hashes,
                vec![
                    PictureHash::Checksum(0x0100_0000),
                    PictureHash::Checksum(0x0200_0000),
                    PictureHash::Checksum(0x0300_0000),
                ]
            );
        }
        other => panic!("expected DecodedPictureHash, got {other:?}"),
    }
}

#[test]
fn prefix_only_payload_in_suffix_is_reserved() {
    // recovery_point (6) is prefix-only; in a suffix NAL it must be
    // carried verbatim, not decoded.
    let body = [0b1100_0000];
    let rbsp = sei_rbsp_one(6, &body);
    let msgs = parse_sei_rbsp(&rbsp, SeiNalType::Suffix).unwrap();
    match &msgs[0].payload {
        SeiPayload::Reserved { payload_type, data } => {
            assert_eq!(*payload_type, 6);
            assert_eq!(data, &body);
        }
        other => panic!("expected Reserved, got {other:?}"),
    }
}

#[test]
fn suffix_only_payload_in_prefix_is_reserved() {
    // decoded_picture_hash (132) is suffix-only.
    let body = [1u8, 0xAB, 0xCD];
    let rbsp = sei_rbsp_one(132, &body);
    let msgs = parse_sei_rbsp(&rbsp, SeiNalType::Prefix).unwrap();
    match &msgs[0].payload {
        SeiPayload::Reserved { payload_type, .. } => assert_eq!(*payload_type, 132),
        other => panic!("expected Reserved, got {other:?}"),
    }
}

#[test]
fn unknown_payload_type_is_reserved() {
    let body = [0xDE, 0xAD, 0xBE, 0xEF];
    let rbsp = sei_rbsp_one(99, &body);
    let msgs = parse_sei_rbsp(&rbsp, SeiNalType::Prefix).unwrap();
    assert_eq!(msgs[0].payload_type, 99);
    match &msgs[0].payload {
        SeiPayload::Reserved { payload_type, data } => {
            assert_eq!(*payload_type, 99);
            assert_eq!(data, &body);
        }
        other => panic!("expected Reserved, got {other:?}"),
    }
}

#[test]
fn multiple_messages_in_one_rbsp() {
    // Two messages: CLL then alternative_transfer_characteristics, then
    // the 0x80 trailer.
    let mut rbsp = Vec::new();
    rbsp.push(144);
    rbsp.push(4);
    rbsp.extend_from_slice(&[0x03, 0xE8, 0x00, 0xC8]);
    rbsp.push(147);
    rbsp.push(1);
    rbsp.push(16); // BT.2020-10 transfer
    rbsp.push(0x80);
    let msgs = parse_sei_rbsp(&rbsp, SeiNalType::Prefix).unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0].payload_type, 144);
    assert_eq!(msgs[1].payload_type, 147);
    match &msgs[1].payload {
        SeiPayload::AlternativeTransferCharacteristics(a) => {
            assert_eq!(a.preferred_transfer_characteristics, 16);
        }
        other => panic!("expected ATC, got {other:?}"),
    }
}

#[test]
fn payload_size_overrun() {
    // payloadSize claims 10 bytes but only 1 follows before the buffer
    // ends.
    let rbsp = [147u8, 10, 0x00];
    assert_eq!(
        parse_sei_rbsp(&rbsp, SeiNalType::Prefix),
        Err(SeiError::PayloadSizeOverrun {
            payload_size: 10,
            available: 1,
        })
    );
}

#[test]
fn truncated_header_in_rbsp() {
    // A lone 0xFF run with no terminator: more_sei_messages sees a
    // non-trailer byte, parse attempts the header and finds it
    // truncated.
    let rbsp = [0xFFu8, 0xFF];
    assert_eq!(
        parse_sei_rbsp(&rbsp, SeiNalType::Prefix),
        Err(SeiError::TruncatedHeader)
    );
}

#[test]
fn empty_rbsp_yields_no_messages() {
    assert_eq!(
        parse_sei_rbsp(&[], SeiNalType::Prefix).unwrap(),
        Vec::<SeiMessage>::new()
    );
}

#[test]
fn trailer_only_rbsp_yields_no_messages() {
    // Just the rbsp_trailing_bits() 0x80 (plus zero padding).
    assert_eq!(
        parse_sei_rbsp(&[0x80], SeiNalType::Prefix).unwrap(),
        Vec::<SeiMessage>::new()
    );
    assert_eq!(
        parse_sei_rbsp(&[0x80, 0x00, 0x00], SeiNalType::Prefix).unwrap(),
        Vec::<SeiMessage>::new()
    );
}

#[test]
fn long_payload_type_and_size_round_trip() {
    // payloadType = 256 (0xFF 0x01), payloadSize = 1 (0x01), single
    // byte body, then trailer. Decodes as Reserved.
    let mut rbsp = vec![0xFF, 0x01, 0x01, 0x77, 0x80];
    let msgs = parse_sei_rbsp(&rbsp, SeiNalType::Prefix).unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].payload_type, 256);
    assert_eq!(msgs[0].payload_size, 1);
    match &msgs[0].payload {
        SeiPayload::Reserved { data, .. } => assert_eq!(data, &vec![0x77]),
        other => panic!("expected Reserved, got {other:?}"),
    }
    // mutate to confirm the framing consumed exactly the declared size
    rbsp.clear();
}

#[test]
fn error_display_strings() {
    let e = SeiError::PayloadSizeOverrun {
        payload_size: 5,
        available: 2,
    };
    assert!(format!("{e}").contains("exceeds"));
    assert!(format!("{}", SeiError::TruncatedHeader).contains("truncated"));
    assert!(format!("{}", SeiError::TruncatedPayload { payload_type: 6 }).contains('6'));
}
