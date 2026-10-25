//! A P picture whose reference was never decoded (the stream entered
//! after its IDR) is refused with an error naming the missing
//! reference-list slot and the picture, not a generic "inter not
//! supported" message.

use oxideav_h265::encoder::inter::{LowDelayPEncoder, YuvFrame};
use oxideav_h265::{ReconError, SequenceDecoder, SequenceError, decode_annexb_sequence};

/// Drop every IDR slice NAL unit (types 19 / 20) from an Annex B stream.
fn drop_idr(stream: &[u8]) -> Vec<u8> {
    let starts: Vec<usize> = (0..stream.len().saturating_sub(3))
        .filter(|&i| stream[i..i + 4] == [0, 0, 0, 1])
        .collect();
    let mut out = Vec::new();
    for (k, &s) in starts.iter().enumerate() {
        let e = starts.get(k + 1).copied().unwrap_or(stream.len());
        let nut = (stream[s + 4] >> 1) & 63;
        if nut != 19 && nut != 20 {
            out.extend_from_slice(&stream[s..e]);
        }
    }
    out
}

/// A three-picture low-delay P stream.
fn p_stream() -> Vec<u8> {
    let (w, h) = (64, 32);
    let mut enc = LowDelayPEncoder::new(w, h, 30, 0).unwrap();
    let mut stream = Vec::new();
    for f in 0..3usize {
        let y: Vec<u8> = (0..w * h).map(|i| ((i * 3 + f * 5) % 251) as u8).collect();
        let cb = vec![120u8; w * h / 4];
        let cr = vec![136u8; w * h / 4];
        let au = enc
            .encode_frame(&YuvFrame {
                y: &y,
                cb: &cb,
                cr: &cr,
            })
            .unwrap();
        stream.extend_from_slice(&au.au);
    }
    stream
}

#[test]
fn p_picture_without_its_idr_names_the_missing_reference() {
    let stream = p_stream();
    assert!(
        decode_annexb_sequence(&stream).is_ok(),
        "intact stream decodes"
    );
    let err = decode_annexb_sequence(&drop_idr(&stream)).expect_err("references missing");
    let SequenceError::Recon(ReconError::MissingReference {
        list, ref_idx, poc, ..
    }) = &err
    else {
        panic!("unexpected error {err}");
    };
    assert_eq!((*list, *ref_idx, *poc), (0, 0, Some(1)));
    let msg = err.to_string();
    assert!(
        msg.contains("picture POC 1: RefPicList0[0] is \"no reference picture\""),
        "{msg}"
    );
}

/// Under a frame-thread budget the worker's error surfaces from the
/// flush with the same first failing picture and slot.
#[test]
fn threaded_decode_reports_the_same_missing_reference() {
    let stream = drop_idr(&p_stream());
    for threads in [2usize, 4] {
        let mut dec = SequenceDecoder::new();
        dec.set_threads(threads);
        let err = dec
            .push_annexb(&stream)
            .and_then(|()| dec.flush())
            .expect_err("references missing");
        let SequenceError::Recon(ReconError::MissingReference {
            list, ref_idx, poc, ..
        }) = &err
        else {
            panic!("unexpected error {err}");
        };
        assert_eq!(
            (*list, *ref_idx, *poc),
            (0, 0, Some(1)),
            "{threads} threads"
        );
    }
}
