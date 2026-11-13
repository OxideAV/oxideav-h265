//! Differential fuzz of the encoder's thread budget: a picture (or a
//! short P / B sequence) built from the input is encoded through the
//! registry encoder on one worker and on several, and the two streams
//! must be the same bytes — the decision pass fans tiles and wavefront
//! rows out to workers, and nothing a worker count changes may reach
//! the bitstream. The stream must then decode to as many pictures as
//! went in.
//!
//! Input layout: six configuration bytes — width and height in units
//! of 16 luma samples (16..=96 x 16..=80), intra / inter and the
//! sequence length, the CTB size and the worker count, the QP, and a
//! tool bit set (WPP, RDOQ, sign hiding, deblocking, SAO, 2x2 tiles,
//! B slices) — then sample data, repeated over the planes on top of a
//! slow ramp so that pictures are neither flat nor noise.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_core::{
    CodecParameters, Error, ExecutionContext, Frame, Packet, TimeBase, VideoFrame, VideoPlane,
};

/// Encode `frames` on `threads` workers; `None` when the configuration
/// is refused.
fn encode(params: &CodecParameters, frames: &[Frame], threads: usize) -> Option<Vec<u8>> {
    let mut enc = oxideav_h265::make_encoder(params).ok()?;
    if threads > 1 {
        enc.set_execution_context(&ExecutionContext { threads });
    }
    let mut out = Vec::new();
    for frame in frames {
        enc.send_frame(frame).ok()?;
        while let Ok(pkt) = enc.receive_packet() {
            out.extend_from_slice(&pkt.data);
        }
    }
    enc.flush().ok()?;
    while let Ok(pkt) = enc.receive_packet() {
        out.extend_from_slice(&pkt.data);
    }
    Some(out)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let (cfg, pix) = data.split_at(6);
    let w = 16 * (1 + usize::from(cfg[0] % 6));
    let h = 16 * (1 + usize::from(cfg[1] % 5));
    let inter = cfg[2] & 1 != 0;
    let n_frames = if inter {
        2 + usize::from((cfg[2] >> 1) & 1)
    } else {
        1
    };
    let ctb = ["16", "32", "64"][usize::from(cfg[3] % 3)];
    let workers = 2 + usize::from((cfg[3] >> 4) & 3);
    let qp = (10 + cfg[4] % 36).to_string();
    let tools = cfg[5];

    let mut params = CodecParameters::video("h265".into());
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params
        .options
        .insert("mode", if inter { "inter" } else { "intra" });
    params.options.insert("ctb", ctb);
    params.options.insert("qp", &qp);
    for (bit, key, value) in [
        (0, "wpp", "1"),
        (1, "rdoq", "1"),
        (2, "sdh", "1"),
        (3, "deblock", "1"),
        (4, "sao", "1"),
        (5, "tiles", "2x2"),
    ] {
        if tools & (1 << bit) != 0 {
            params.options.insert(key, value);
        }
    }
    if inter && tools & 0x40 != 0 {
        params.options.insert("bslices", "1");
    }

    let frames: Vec<Frame> = (0..n_frames)
        .map(|f| {
            let plane = |pw: usize, ph: usize, salt: usize| VideoPlane {
                stride: pw,
                data: (0..pw * ph)
                    .map(|i| {
                        let (x, y) = (i % pw, i / pw);
                        // The picture drifts by two samples per frame.
                        let s = pix[(x + 2 * f + y * pw + salt) % pix.len()];
                        ((x * 3 + y * 2 + salt) as u8).wrapping_add(s / 2)
                    })
                    .collect(),
            };
            Frame::Video(VideoFrame {
                pts: Some(f as i64),
                planes: vec![
                    plane(w, h, 0),
                    plane(w / 2, h / 2, 37),
                    plane(w / 2, h / 2, 91),
                ],
            })
        })
        .collect();

    let serial = encode(&params, &frames, 1);
    let threaded = encode(&params, &frames, workers);
    assert_eq!(
        serial, threaded,
        "{w}x{h} inter={inter} ctb={ctb} qp={qp} tools={tools:#04x}: {workers} workers"
    );
    let Some(stream) = serial else {
        return;
    };

    let mut dec =
        oxideav_h265::make_decoder(&CodecParameters::video("h265".into())).expect("decoder");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), stream))
        .expect("the encoder's own stream decodes");
    dec.flush().expect("flush");
    let mut decoded = 0;
    loop {
        match dec.receive_frame() {
            Ok(_) => decoded += 1,
            Err(Error::NeedMore) => {}
            Err(Error::Eof) => break,
            Err(e) => panic!("decode of the encoder's own stream: {e}"),
        }
    }
    assert_eq!(decoded, n_frames, "decoded picture count");
});
