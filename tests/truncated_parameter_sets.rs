//! Parameter sets whose RBSP ends before the extension gate.
//!
//! §7.3.2.1 / §7.3.2.2.1 / §7.3.2.3.1 end the VPS / SPS / PPS with
//! `vps_extension_flag` / `sps_extension_present_flag` /
//! `pps_extension_present_flag` and then `rbsp_trailing_bits( )`. Some
//! producers stop right before the gate: the RBSP holds only the
//! trailing bits (or nothing) where the flag belongs. §7.2
//! `more_rbsp_data( )` is FALSE at that position, so the decoder reads
//! no flag and takes it as 0 — every value the decoding process needs
//! has been stated. These tests rewrite the parameter sets of a stream
//! this crate encodes into that truncated shape and require the same
//! decoded pictures.

use oxideav_core::{CodecParameters, Frame, PixelFormat, VideoFrame, VideoPlane};
use oxideav_h265::decode_annexb_sequence;
use oxideav_h265::nal::NalIter;

fn frame(w: usize, h: usize) -> VideoFrame {
    let mk = |wp: usize, hp: usize, mul: usize| VideoPlane {
        stride: wp,
        data: (0..wp * hp).map(|i| (i * mul % 251) as u8).collect(),
    };
    VideoFrame {
        pts: None,
        planes: vec![mk(w, h, 7), mk(w / 2, h / 2, 3), mk(w / 2, h / 2, 11)],
    }
}

fn encode_one() -> Vec<u8> {
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(48);
    params.height = Some(32);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    let mut enc = oxideav_h265::make_encoder(&params).expect("encoder");
    enc.send_frame(&Frame::Video(frame(48, 32))).expect("send");
    enc.receive_packet().expect("packet").data
}

/// §7.4.1.1 emulation prevention for a rewritten RBSP.
fn escape(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + 4);
    let mut zeros = 0;
    for &b in rbsp {
        if zeros >= 2 && b <= 3 {
            out.push(3);
            zeros = 0;
        }
        out.push(b);
        zeros = if b == 0 { zeros + 1 } else { 0 };
    }
    out
}

/// How the bits after the dropped gate flag are written.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Tail {
    /// `rbsp_trailing_bits( )` right where the flag was.
    StopBit,
    /// Nothing: the RBSP is cut at the last whole byte before the flag
    /// (only possible when the flag sits on a byte boundary).
    Exhausted,
}

/// Drop the last syntax bit before the `rbsp_stop_one_bit` (the
/// extension gate, coded 0 by this crate's encoder).
fn drop_gate(rbsp: &[u8], tail: Tail) -> Option<Vec<u8>> {
    let last = rbsp.iter().rposition(|&b| b != 0)?;
    let stop = last * 8 + 7 - rbsp[last].trailing_zeros() as usize;
    let gate = stop - 1;
    assert_eq!(
        (rbsp[gate / 8] >> (7 - gate % 8)) & 1,
        0,
        "the encoder codes the extension gate as 0"
    );
    let mut out = rbsp[..gate.div_ceil(8)].to_vec();
    if !gate.is_multiple_of(8) {
        let keep = gate % 8;
        let idx = out.len() - 1;
        out[idx] &= 0xFFu8 << (8 - keep);
    }
    match tail {
        Tail::StopBit => {
            if gate.is_multiple_of(8) {
                out.push(0x80);
            } else {
                let idx = out.len() - 1;
                out[idx] |= 0x80 >> (gate % 8);
            }
            Some(out)
        }
        Tail::Exhausted => gate.is_multiple_of(8).then_some(out),
    }
}

/// Rewrite the stream with the gate dropped from every NAL unit of
/// the listed types; `None` when the tail shape does not apply.
fn rewrite(stream: &[u8], types: &[u8], tail: Tail) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut touched = false;
    for unit in NalIter::new(stream) {
        let unit = unit.expect("NAL");
        let t = unit.header.nal_unit_type;
        out.extend_from_slice(&[0, 0, 0, 1, t << 1, 1]);
        if types.contains(&t) {
            let rbsp = drop_gate(&unit.rbsp, tail)?;
            out.extend_from_slice(&escape(&rbsp));
            touched = true;
        } else {
            out.extend_from_slice(&unit.escaped);
        }
    }
    assert!(touched, "no parameter set of the requested type");
    Some(out)
}

fn planes(stream: &[u8]) -> Vec<Vec<u8>> {
    decode_annexb_sequence(stream)
        .expect("decode")
        .iter()
        .map(|f| f.output_picture().to_planar_u8().expect("8-bit"))
        .collect()
}

#[test]
fn parameter_sets_ending_before_the_extension_gate_decode() {
    let stream = encode_one();
    let reference = planes(&stream);
    assert_eq!(reference.len(), 1);
    let mut cases = 0;
    for types in [&[32u8][..], &[33], &[34], &[32, 33, 34]] {
        for tail in [Tail::StopBit, Tail::Exhausted] {
            let Some(cut) = rewrite(&stream, types, tail) else {
                continue;
            };
            assert_ne!(cut, stream);
            assert_eq!(planes(&cut), reference, "NAL types {types:?}, {tail:?}");
            cases += 1;
        }
    }
    assert!(cases >= 4, "every stop-bit case ran ({cases})");
}

/// The same shapes through the registry decoder (the path a container
/// hands its parameter sets to).
#[test]
fn registry_decoder_accepts_gateless_parameter_sets() {
    let stream = encode_one();
    let cut = rewrite(&stream, &[32, 33, 34], Tail::StopBit).expect("rewrite");
    let decode = |data: &[u8]| -> Vec<u8> {
        let mut dec =
            oxideav_h265::make_decoder(&CodecParameters::video("h265".into())).expect("decoder");
        let pkt = oxideav_core::Packet::new(0, oxideav_core::TimeBase::new(1, 25), data.to_vec());
        dec.send_packet(&pkt).expect("send");
        dec.flush().expect("flush");
        match dec.receive_frame().expect("frame") {
            Frame::Video(v) => v.planes.iter().flat_map(|p| p.data.clone()).collect(),
            other => panic!("unexpected {other:?}"),
        }
    };
    assert_eq!(decode(&cut), decode(&stream));
}
