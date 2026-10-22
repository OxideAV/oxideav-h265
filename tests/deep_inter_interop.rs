//! Round-479 deep-layout inter pins: the low-delay P / low-delay B /
//! hierarchical-B coders on every lossy HEIC sample layout beyond
//! 8-bit 4:2:0 — Main 10 (4:2:0 10-bit), 4:2:0 12-bit, Main 4:2:2 10
//! (8 / 10-bit 4:2:2), Main 4:4:4 (8 / 10 / 12-bit 4:4:4) and
//! monochrome 10-bit.
//!
//! Per cell a five-frame synthetic pan (a textured background with a
//! drifting bright block, chroma drifting the other way) is coded
//! through the direct API (`LowDelayPEncoder::new_fmt` /
//! `PyramidEncoder::new_fmt`, CTB 32) and through the registry
//! encoder with the same options (the two streams are byte-identical);
//! the stream decodes through this crate's decoder to exactly the
//! encoder's reconstruction, and the decoded planar output is pinned
//! by the MD5 a black-box reference decoder produced for the same
//! stream (8-bit samples as bytes, deeper as 16-bit little-endian;
//! monochrome compares the luma plane). `H265_DUMP_DIR` dumps the
//! streams for re-verification.

mod fixture_bytes;

use fixture_bytes::md5;
use oxideav_core::{CodecParameters, Error, Frame, PixelFormat, VideoFrame, VideoPlane};
use oxideav_h265::encoder::ctu::TreeCfg;
use oxideav_h265::encoder::inter::{LowDelayPEncoder, WideRecon, YuvFrameWide};
use oxideav_h265::encoder::pcm::PcmLayout;
use oxideav_h265::encoder::pyramid::PyramidEncoder;
use oxideav_h265::sequence::decode_annexb_sequence;

const QP: i32 = 29;
const FRAMES: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gop {
    /// `IDR, P, P, P, P`.
    LowDelayP,
    /// `IDR, B, B, B, B` (both lists = the past picture).
    LowDelayB,
    /// A four-frame hierarchical-B mini-GOP (`pyramid = 4`).
    Pyramid,
}

struct Cell {
    pf: PixelFormat,
    cfi: u8,
    bd: u8,
    w: usize,
    h: usize,
    gop: Gop,
    /// MD5 of the decoded planar output in output order (black-box
    /// verified).
    md5: &'static str,
}

const CELLS: &[Cell] = &[
    Cell {
        pf: PixelFormat::Yuv420P10Le,
        cfi: 1,
        bd: 10,
        w: 64,
        h: 48,
        gop: Gop::LowDelayP,
        md5: "90e09b4e82fc938f63de6946df2589ff",
    },
    Cell {
        pf: PixelFormat::Yuv420P10Le,
        cfi: 1,
        bd: 10,
        w: 64,
        h: 48,
        gop: Gop::LowDelayB,
        md5: "216553c811d767085488c0cf7af9f891",
    },
    Cell {
        pf: PixelFormat::Yuv420P10Le,
        cfi: 1,
        bd: 10,
        w: 64,
        h: 48,
        gop: Gop::Pyramid,
        md5: "a2bb2a6a0cd1a35a2b975e6f50809e86",
    },
    Cell {
        pf: PixelFormat::Yuv420P12Le,
        cfi: 1,
        bd: 12,
        w: 48,
        h: 32,
        gop: Gop::LowDelayP,
        md5: "178b66f3ee839a40817eb53d5767f796",
    },
    Cell {
        pf: PixelFormat::Yuv420P12Le,
        cfi: 1,
        bd: 12,
        w: 48,
        h: 32,
        gop: Gop::Pyramid,
        md5: "7a893911eff80d8ccdf5769bb1ac6c03",
    },
    Cell {
        pf: PixelFormat::Yuv422P,
        cfi: 2,
        bd: 8,
        w: 64,
        h: 32,
        gop: Gop::LowDelayP,
        md5: "c2b32d7b3ee49e91a2fabc0b94cee90e",
    },
    Cell {
        pf: PixelFormat::Yuv422P,
        cfi: 2,
        bd: 8,
        w: 64,
        h: 32,
        gop: Gop::Pyramid,
        md5: "6cd8e6f70265b69858776e19ac70bd91",
    },
    Cell {
        pf: PixelFormat::Yuv422P10Le,
        cfi: 2,
        bd: 10,
        w: 64,
        h: 48,
        gop: Gop::LowDelayP,
        md5: "93ff59c512b721ac508778384fdf1203",
    },
    Cell {
        pf: PixelFormat::Yuv422P10Le,
        cfi: 2,
        bd: 10,
        w: 64,
        h: 48,
        gop: Gop::LowDelayB,
        md5: "83e1f613419a78b9c3f4e0aab1a012f6",
    },
    Cell {
        pf: PixelFormat::Yuv422P10Le,
        cfi: 2,
        bd: 10,
        w: 64,
        h: 48,
        gop: Gop::Pyramid,
        md5: "b3f28f40c74398c88edf33d6dfcfa7c1",
    },
    Cell {
        pf: PixelFormat::Yuv444P,
        cfi: 3,
        bd: 8,
        w: 64,
        h: 48,
        gop: Gop::LowDelayP,
        md5: "21aa74b0f26401e9b7b83124254aff90",
    },
    Cell {
        pf: PixelFormat::Yuv444P,
        cfi: 3,
        bd: 8,
        w: 64,
        h: 48,
        gop: Gop::Pyramid,
        md5: "311c0bad841a9d32e695bfc8b3afa956",
    },
    Cell {
        pf: PixelFormat::Yuv444P10Le,
        cfi: 3,
        bd: 10,
        w: 64,
        h: 48,
        gop: Gop::LowDelayP,
        md5: "023d8a4cbd6daf285fb07e99bdaacbfa",
    },
    Cell {
        pf: PixelFormat::Yuv444P10Le,
        cfi: 3,
        bd: 10,
        w: 64,
        h: 48,
        gop: Gop::LowDelayB,
        md5: "b3ad2b5c039a77f59e3d46f450bb75d9",
    },
    Cell {
        pf: PixelFormat::Yuv444P10Le,
        cfi: 3,
        bd: 10,
        w: 64,
        h: 48,
        gop: Gop::Pyramid,
        md5: "b6a013a8e7676ddcc6555895cbfa2be1",
    },
    Cell {
        pf: PixelFormat::Yuv444P12Le,
        cfi: 3,
        bd: 12,
        w: 48,
        h: 32,
        gop: Gop::LowDelayP,
        md5: "0a5f476382e0a3272c8850f6efc3f65f",
    },
    Cell {
        pf: PixelFormat::Yuv444P12Le,
        cfi: 3,
        bd: 12,
        w: 48,
        h: 32,
        gop: Gop::Pyramid,
        md5: "0302af743762a4f67d1025b1e081e769",
    },
    Cell {
        pf: PixelFormat::Gray10Le,
        cfi: 0,
        bd: 10,
        w: 64,
        h: 48,
        gop: Gop::LowDelayP,
        md5: "45306a63a893445304e170ee0782cb4d",
    },
    Cell {
        pf: PixelFormat::Gray10Le,
        cfi: 0,
        bd: 10,
        w: 64,
        h: 48,
        gop: Gop::Pyramid,
        md5: "6d35833e4692b235e5186d264604237d",
    },
];

fn sub_wh(cfi: u8) -> (usize, usize) {
    match cfi {
        1 => (2, 2),
        2 => (2, 1),
        _ => (1, 1),
    }
}

/// Frame `t` of the synthetic pan at the cell's layout: `[Y, Cb, Cr]`
/// `u16` planes (chroma empty for monochrome).
fn scene(c: &Cell, t: usize) -> [Vec<u16>; 3] {
    let (sw, sh) = sub_wh(c.cfi);
    let max = (1u32 << c.bd) - 1;
    let scale = |v8: u32| -> u16 { ((v8 * max + 127) / 255) as u16 };
    let y: Vec<u16> = (0..c.w * c.h)
        .map(|i| {
            let (x, yy) = (i % c.w, i / c.w);
            let base = (x * 2 + yy * 3 + (x * yy / 11) % 23) % 200;
            let (sx, sy) = (4 + t * 3, 6 + t * 2);
            let v = if x >= sx && x < sx + 12 && yy >= sy && yy < sy + 10 {
                base + 55
            } else {
                base
            };
            // A little high-frequency texture so the deep samples use
            // their extra bits.
            let fine = ((x * 7 + yy * 5 + t) % 5) as u16;
            let v8 = v as u32;
            (scale(v8) + fine).min(max as u16)
        })
        .collect();
    if c.cfi == 0 {
        return [y, Vec::new(), Vec::new()];
    }
    let (cw, ch) = (c.w / sw, c.h / sh);
    let cb: Vec<u16> = (0..cw * ch)
        .map(|i| {
            let (x, yy) = (i % cw, i / cw);
            scale(((100 + x * 2 + t * 4 + (yy / 3) * 5) % 60 + 90) as u32)
        })
        .collect();
    let cr: Vec<u16> = (0..cw * ch)
        .map(|i| {
            let (x, yy) = (i % cw, i / cw);
            scale(((160 + yy * 3 + (x / 2)).wrapping_sub(t * 3) % 50 + 100) as u32)
        })
        .collect();
    [y, cb, cr]
}

fn planar_of(rec: &WideRecon, bd: u8) -> Vec<u8> {
    let mut out = Vec::new();
    for plane in [&rec.y, &rec.cb, &rec.cr] {
        for &s in plane {
            if bd > 8 {
                out.extend_from_slice(&s.to_le_bytes());
            } else {
                out.push(s as u8);
            }
        }
    }
    out
}

/// Encode through the direct API: `(stream, recon in output order)`.
fn encode_direct(c: &Cell, frames: &[[Vec<u16>; 3]]) -> (Vec<u8>, Vec<WideRecon>) {
    let layout = PcmLayout {
        chroma_format_idc: c.cfi,
        bit_depth: c.bd,
    };
    let tree = TreeCfg::new(32).expect("ctb 32");
    let mut stream = Vec::new();
    match c.gop {
        Gop::LowDelayP | Gop::LowDelayB => {
            let mut enc = LowDelayPEncoder::new_fmt(c.w, c.h, QP, 0, layout)
                .unwrap_or_else(|e| panic!("{:?}: {e}", c.pf))
                .with_tree(tree)
                .with_b_slices(c.gop == Gop::LowDelayB);
            let mut recon = Vec::new();
            for f in frames {
                let out = enc
                    .encode_frame_wide(&YuvFrameWide::new(&f[0], &f[1], &f[2]))
                    .unwrap_or_else(|e| panic!("{:?}: {e}", c.pf));
                stream.extend_from_slice(&out.au);
                recon.push(out.recon);
            }
            (stream, recon)
        }
        Gop::Pyramid => {
            let mut enc = PyramidEncoder::new_fmt(c.w, c.h, QP, 4, layout)
                .unwrap_or_else(|e| panic!("{:?}: {e}", c.pf))
                .with_tree(tree);
            let mut by_display: Vec<Option<WideRecon>> = vec![None; frames.len()];
            let mut push = |aus: Vec<oxideav_h265::encoder::pyramid::PyramidAuWide>| {
                for au in aus {
                    stream.extend_from_slice(&au.au);
                    by_display[au.display_order] = Some(au.recon);
                }
            };
            for f in frames {
                let aus = enc
                    .encode_frame_wide(&YuvFrameWide::new(&f[0], &f[1], &f[2]))
                    .unwrap_or_else(|e| panic!("{:?}: {e}", c.pf));
                push(aus);
            }
            push(enc.flush_wide());
            (
                stream,
                by_display.into_iter().map(|r| r.expect("coded")).collect(),
            )
        }
    }
}

/// Encode through the registry with the matching options.
fn encode_registry(c: &Cell, frames: &[[Vec<u16>; 3]]) -> Vec<u8> {
    let (sw, sh) = sub_wh(c.cfi);
    let mut params = CodecParameters::video("h265".into());
    params.width = Some(c.w as u32);
    params.height = Some(c.h as u32);
    params.pixel_format = Some(c.pf);
    params.options.insert("mode", "inter");
    params.options.insert("qp", "29");
    params.options.insert("ctb", "32");
    match c.gop {
        Gop::LowDelayP => {}
        Gop::LowDelayB => {
            params.options.insert("bslices", "1");
        }
        Gop::Pyramid => {
            params.options.insert("pyramid", "4");
        }
    }
    let mut enc = oxideav_h265::make_encoder(&params).unwrap_or_else(|e| panic!("{:?}: {e}", c.pf));
    let bps = if c.bd > 8 { 2 } else { 1 };
    let plane = |p: &[u16], pw: usize| -> VideoPlane {
        let mut data = Vec::with_capacity(p.len() * bps);
        for &s in p {
            if bps == 2 {
                data.extend_from_slice(&s.to_le_bytes());
            } else {
                data.push(s as u8);
            }
        }
        VideoPlane {
            stride: pw * bps,
            data,
        }
    };
    let mut stream = Vec::new();
    let drain = |enc: &mut Box<dyn oxideav_core::Encoder>, stream: &mut Vec<u8>| loop {
        match enc.receive_packet() {
            Ok(p) => stream.extend_from_slice(&p.data),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("{:?}: receive: {e}", c.pf),
        }
    };
    for (t, f) in frames.iter().enumerate() {
        let mut planes = vec![plane(&f[0], c.w)];
        if c.cfi != 0 {
            planes.push(plane(&f[1], c.w / sw));
            planes.push(plane(&f[2], c.w / sw));
        }
        let _ = sh;
        enc.send_frame(&Frame::Video(VideoFrame {
            pts: Some(t as i64),
            planes,
        }))
        .unwrap_or_else(|e| panic!("{:?}: send: {e}", c.pf));
        drain(&mut enc, &mut stream);
    }
    enc.flush()
        .unwrap_or_else(|e| panic!("{:?}: flush: {e}", c.pf));
    drain(&mut enc, &mut stream);
    stream
}

#[test]
fn deep_layouts_code_inter_byte_exact() {
    let dump = std::env::var("H265_DUMP_DIR").ok();
    let mut unpinned = Vec::new();
    for c in CELLS {
        let label = format!("{:?}-{:?}", c.pf, c.gop);
        let frames: Vec<[Vec<u16>; 3]> = (0..FRAMES).map(|t| scene(c, t)).collect();
        let (stream, recon) = encode_direct(c, &frames);
        let registry = encode_registry(c, &frames);
        assert_eq!(
            registry, stream,
            "{label}: registry stream == direct-API stream"
        );
        if let Some(dir) = &dump {
            std::fs::write(format!("{dir}/deep-{label}.hevc"), &stream).expect("dump");
        }
        let decoded = decode_annexb_sequence(&stream).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(decoded.len(), FRAMES, "{label}: picture count");
        let mut planar = Vec::new();
        for (i, (dec, rec)) in decoded.iter().zip(recon.iter()).enumerate() {
            assert!(dec.output, "{label}: frame {i} output");
            let pic = dec.output_picture();
            assert_eq!(pic.chroma_array_type(), c.cfi, "{label}: chroma format");
            assert_eq!(pic.bit_depth_luma(), c.bd, "{label}: bit depth");
            let got = pic.to_planar_u8().unwrap_or_else(|| pic.to_planar_le16());
            assert_eq!(
                got,
                planar_of(rec, c.bd),
                "{label}: frame {i} decoder output == encoder reconstruction"
            );
            planar.extend_from_slice(&got);
        }
        let digest = md5::hex(&planar);
        if c.md5.is_empty() {
            unpinned.push(format!("{label}: {digest}"));
        } else {
            assert_eq!(digest, c.md5, "{label}: black-box pinned digest");
        }
    }
    assert!(
        unpinned.is_empty(),
        "unpinned cells:\n{}",
        unpinned.join("\n")
    );
}
