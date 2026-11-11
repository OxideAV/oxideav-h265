//! Multi-layer (Annex F / G / H) decode fuzz entry, over the decoder
//! options that select and output layers.
//!
//! Input layout: one option byte, then an Annex B byte stream. The
//! option byte picks the registry decoder's options:
//!
//! * bit 0 — `output_uninitialized_layers` (output the pictures of an
//!   enhancement layer entered at a non-IRAP picture; the F.8.1.7
//!   generated references stand in for the ones it never decoded);
//! * bits 1..=2 — the layer selection: none (the highest output layer
//!   set), `layer=1`, `ols=0`, `view=1`;
//! * bit 3 — feed the stream as one packet per NAL unit instead of a
//!   single packet (access-unit boundaries then fall where the decoder
//!   finds them);
//! * bit 4 — `layerset=<bits 5..=7>` (every layer of a layer set,
//!   auxiliary layers included) when no other selector is set.
//!
//! The stream runs through `make_decoder_with_options` →
//! `send_packet` → `flush` → `receive_frame`: the VPS extension layer
//! model, per-layer parameter-set activation, inter-layer reference
//! derivation (resampling, colour mapping, motion-field mapping) and
//! the per-layer output rules all see hostile input.
//!
//! Every path must return `Ok` / `Err` — no panics, no aborts, no
//! unbounded allocation.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecParameters, Error, Packet, TimeBase};
use oxideav_h265::H265DecoderOptions;

fuzz_target!(|data: &[u8]| {
    let Some((&opt, stream)) = data.split_first() else {
        return;
    };
    let mut options = H265DecoderOptions::new().with_output_uninitialized_layers(opt & 1 != 0);
    options = match (opt >> 1) & 3 {
        1 => options.with_layer(1),
        2 => options.with_ols(0),
        3 => options.with_view(1),
        _ => options,
    };
    if opt & 16 != 0 {
        options = options.with_layer_set(u32::from(opt >> 5));
    }
    let params = CodecParameters::video("h265".into());
    let Ok(mut dec) = oxideav_h265::make_decoder_with_options(&params, &options) else {
        return;
    };
    let mut drain = |dec: &mut Box<dyn oxideav_core::Decoder>| {
        while dec.receive_frame().is_ok() {}
    };
    if opt & 8 != 0 {
        // One packet per start-code-delimited unit.
        let mut starts: Vec<usize> = (0..stream.len().saturating_sub(2))
            .filter(|&i| stream[i..i + 3] == [0, 0, 1])
            .collect();
        starts.push(stream.len());
        for pair in starts.windows(2) {
            let pkt = Packet::new(0, TimeBase::new(1, 1), stream[pair[0]..pair[1]].to_vec());
            let _ = dec.send_packet(&pkt);
            drain(&mut dec);
        }
    } else {
        let _ = dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), stream.to_vec()));
    }
    let _ = dec.flush();
    while let Ok(_) | Err(Error::NeedMore) = dec.receive_frame() {}
});
