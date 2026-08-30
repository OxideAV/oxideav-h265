//! Round-453 non-uniform tile-grid encoding pin: a PCM IDR picture
//! coded as ONE slice over an explicit (`uniform_spacing_flag == 0`)
//! 3x2 tile grid — column widths 1 / 3 / 2 CTBs, row heights 3 / 1 —
//! with per-tile §7.3.8.1 subsets and §7.4.7.1 entry points. The
//! golden bytes were validated OUT OF BAND against a black-box
//! reference decoder (lossless to the source; see
//! `fixture_bytes/r453-generation-notes.md`).

use oxideav_h265::decode_annexb_sequence;
use oxideav_h265::encoder::pcm::{encode_idr_pcm_au_opts, PcmAuOptions};

fn planes(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y: Vec<u8> = (0..w * h)
        .map(|k| {
            let (x, yy) = (k % w, k / w);
            ((x * 5 + yy * 3 + (x * yy) % 17) % 256) as u8
        })
        .collect();
    let cb: Vec<u8> = (0..w * h / 4).map(|k| ((k * 7) % 200 + 20) as u8).collect();
    let cr: Vec<u8> = (0..w * h / 4)
        .map(|k| ((k * 11) % 180 + 40) as u8)
        .collect();
    (y, cb, cr)
}

#[test]
fn explicit_tile_grid_pin() {
    let (w, h) = (96, 64);
    let (y, cb, cr) = planes(w, h);
    let opts = PcmAuOptions {
        tile_spans: Some((vec![1, 3, 2], vec![3, 1])),
        ..PcmAuOptions::default()
    };
    let au = encode_idr_pcm_au_opts(&y, &cb, &cr, w, h, opts).expect("encode");
    let golden: &[u8] = include_bytes!("fixture_bytes/r453-pcm-tiles-explicit-96x64.hevc");
    assert_eq!(au, golden, "stream drifted off the validated pin");
    let frames = decode_annexb_sequence(golden).expect("decode");
    assert_eq!(frames.len(), 1);
    let mut expected = y;
    expected.extend_from_slice(&cb);
    expected.extend_from_slice(&cr);
    assert_eq!(frames[0].picture.to_planar_u8().expect("8-bit"), expected);
}
