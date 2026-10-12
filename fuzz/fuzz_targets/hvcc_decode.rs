//! `hvcC` + length-prefixed access-unit fuzz entry — the carriage a
//! HEIF / ISO-BMFF reader hands the registry decoder.
//!
//! Input layout: a big-endian `u16` byte count, then that many bytes
//! of `HEVCDecoderConfigurationRecord` (ISO/IEC 14496-15 §8.3.3.1.2:
//! the parameter-set arrays and the NAL length-prefix size), then the
//! coded item / sample payload as `lengthSizeMinusOne + 1`-byte
//! length-prefixed NAL units. The record becomes
//! `CodecParameters::extradata`, the payload one `Packet`, and the
//! decoder is driven through `make_decoder` → `send_packet` →
//! `flush` → `receive_frame` — the whole §7 / §8 / §9 decode (slice
//! data, CABAC, residuals, prediction, loop filters) runs on hostile
//! parameter sets and slice payloads.
//!
//! Every path must return `Ok` / `Err` — no panics, no aborts, no
//! unbounded allocation — regardless of input. The §A.4.1 picture
//! dimension ceiling on SPS parse bounds the reconstruction buffers.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecParameters, Error, Packet, TimeBase};

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let hvcc_len = usize::from(u16::from_be_bytes([data[0], data[1]]));
    let Some(hvcc) = data.get(2..2 + hvcc_len) else {
        return;
    };
    let payload = &data[2 + hvcc_len..];

    let mut params = CodecParameters::video("h265".into());
    params.extradata = hvcc.to_vec();
    let Ok(mut dec) = oxideav_h265::make_decoder(&params) else {
        return;
    };
    let _ = dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), payload.to_vec()));
    let _ = dec.flush();
    while let Ok(_) | Err(Error::NeedMore) = dec.receive_frame() {}
});
