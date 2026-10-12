//! Hostile-input regression pins: the decoder must return an error,
//! never panic, on the arithmetic-coded payloads a container fuzzer
//! hands it.
//!
//! `tests/fixture_bytes/r474/heif-fuzz-a5836b-item{1,2}.{hvcc,au}` are
//! the two `hvc1` items (their `hvcC` records and length-prefixed
//! access units) of a HEIF file a container-level fuzzer produced; the
//! second item's `cu_qp_delta_abs` EGk suffix ran past 32 prefix bins
//! and the `prefix + suffix` sum overflowed `u32` (round 474).
//!
//! `tests/fixture_bytes/r474/*.seed` are inputs of this crate's own
//! `hvcc_decode` fuzz target (a big-endian `u16` record length, the
//! `hvcC` record, then the length-prefixed payload): `hvcc-fuzz-63baae`
//! carries a PPS whose `diff_cu_qp_delta_depth` exceeds the SPS's
//! `log2_diff_max_min_luma_coding_block_size`, so `CtbLog2SizeY −
//! diff_cu_qp_delta_depth` underflowed at activation; `hvcc-fuzz-8e2cdb`
//! an SPS whose picture width is not a multiple of `MinCbSizeY`, which
//! drove the coding quadtree below the minimum coding block (the
//! §7.4.9.4 `split_cu_flag` inference at `MinCbLog2SizeY` is 0, not
//! "split because the block crosses the boundary") and a transform
//! unit to `log2TrafoSize − 1` at size 1; `hvcc-fuzz-4cef6c` a P-slice
//! run whose picture order counts are far enough apart that the
//! §8.5.3.2.7 `DiffPicOrderCnt` (`currPic − neighRefPic`) overflowed
//! `i32` in the temporal motion-vector scaling.

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

/// Inputs in the `hvcc_decode` fuzz target's layout, through the
/// registry decoder exactly as the target drives it.
const SEEDS: &[(&str, &[u8])] = &[
    (
        "hvcc-fuzz-63baae",
        include_bytes!("fixture_bytes/r474/hvcc-fuzz-63baae.seed"),
    ),
    (
        "hvcc-fuzz-8e2cdb",
        include_bytes!("fixture_bytes/r474/hvcc-fuzz-8e2cdb.seed"),
    ),
    (
        "hvcc-fuzz-4cef6c",
        include_bytes!("fixture_bytes/r474/hvcc-fuzz-4cef6c.seed"),
    ),
];

#[test]
fn hvcc_fuzz_seeds_never_panic() {
    for (name, data) in SEEDS {
        let hvcc_len = usize::from(u16::from_be_bytes([data[0], data[1]]));
        let hvcc = &data[2..2 + hvcc_len];
        let payload = &data[2 + hvcc_len..];
        let mut params = CodecParameters::video("h265".into());
        params.extradata = hvcc.to_vec();
        let Ok(mut dec) = make_decoder(&params) else {
            continue;
        };
        let sent = dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), payload.to_vec()));
        let flushed = dec.flush();
        let mut calls = 0usize;
        loop {
            calls += 1;
            assert!(calls <= 64, "{name}: receive_frame never reached Eof");
            match dec.receive_frame() {
                Ok(_) => {}
                Err(Error::Eof) => break,
                Err(e) => panic!("{name}: receive_frame after flush: {e}"),
            }
        }
        let _ = (sent, flushed);
    }
}
