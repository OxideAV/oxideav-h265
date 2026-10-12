//! Hostile-input regression pins: the decoder must return an error,
//! never panic, on the arithmetic-coded payloads a container fuzzer
//! hands it.
//!
//! `tests/fixture_bytes/r474/heif-fuzz-a5836b-item{1,2}.{hvcc,au}` are
//! the two `hvc1` items (their `hvcC` records and length-prefixed
//! access units) of a HEIF file a container-level fuzzer produced; the
//! second item's `cu_qp_delta_abs` EGk suffix ran past 32 prefix bins
//! and the `prefix + suffix` sum overflowed `u32` (round 474).

use oxideav_core::{CodecParameters, Error, Packet, TimeBase};
use oxideav_h265::sequence::SequenceDecoder;
use oxideav_h265::{make_decoder, parse_hvcc, split_length_prefixed};

struct Item {
    name: &'static str,
    hvcc: &'static [u8],
    au: &'static [u8],
}

const ITEMS: &[Item] = &[
    Item {
        name: "item1",
        hvcc: include_bytes!("fixture_bytes/r474/heif-fuzz-a5836b-item1.hvcc"),
        au: include_bytes!("fixture_bytes/r474/heif-fuzz-a5836b-item1.au"),
    },
    Item {
        name: "item2",
        hvcc: include_bytes!("fixture_bytes/r474/heif-fuzz-a5836b-item2.hvcc"),
        au: include_bytes!("fixture_bytes/r474/heif-fuzz-a5836b-item2.au"),
    },
];

/// The library path a HEIF reader may take: `hvcC` arrays then the
/// length-prefixed item payload through a `SequenceDecoder`.
#[test]
fn heif_fuzz_items_never_panic_sequence_decoder() {
    for item in ITEMS {
        let rec = parse_hvcc(item.hvcc).unwrap_or_else(|e| panic!("{}: hvcC: {e}", item.name));
        let mut seq = SequenceDecoder::new();
        let mut result = Ok(());
        for unit in rec.nal_units {
            result = result.and(seq.push_nal_unit(unit));
        }
        if let Ok(units) = split_length_prefixed(item.au, rec.length_size) {
            for unit in units {
                result = result.and(seq.push_nal_unit(unit));
            }
        }
        let finished = seq.finish();
        // Either outcome is acceptable; reaching this line is the pin.
        let _ = (result, finished);
    }
}

/// The registry path (`make_decoder` with `hvcC` extradata and a
/// length-prefixed packet) the HEIF crate drives.
#[test]
fn heif_fuzz_items_never_panic_registry_decoder() {
    for item in ITEMS {
        let mut params = CodecParameters::video("h265".into());
        params.extradata = item.hvcc.to_vec();
        let Ok(mut dec) = make_decoder(&params) else {
            continue;
        };
        let pkt = Packet::new(0, TimeBase::new(1, 1), item.au.to_vec());
        let sent = dec.send_packet(&pkt);
        let flushed = dec.flush();
        // After `flush` — whatever it reported — the decoder drains its
        // frames and then answers `Eof`; a decoder that keeps asking
        // for more input would hang every consumer's drain loop.
        let mut calls = 0usize;
        loop {
            calls += 1;
            assert!(
                calls <= 64,
                "{}: receive_frame never reached Eof",
                item.name
            );
            match dec.receive_frame() {
                Ok(_) => {}
                Err(Error::Eof) => break,
                Err(e) => panic!("{}: receive_frame after flush: {e}", item.name),
            }
        }
        // Either outcome is acceptable; reaching this line is the pin.
        let _ = (sent, flushed);
    }
}
