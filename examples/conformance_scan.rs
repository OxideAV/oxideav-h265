//! Conformance-corpus triage: decode every staged official bitstream
//! and score the output against its published decoded-YUV MD5.
//!
//! Reads the staged conformance corpus (Annex B elementary streams
//! plus `.md5` sidecars) from `docs/video/h265/conformance/` in the
//! workspace checkout — nothing is copied into this repo. For each
//! `.bit` / `.bin` stream the whole-bitstream driver decodes to
//! output-order frames, the frame planes are serialized both as 8-bit
//! planar and 16-bit little-endian planar, and the MD5 of each
//! serialization is compared against every digest published in the
//! stream's sidecar files. A match on either serialization is a PASS.
//!
//! Usage:
//! ```sh
//! cargo run --release --example conformance_scan -- [CONFORMANCE_DIR] [FILTER]
//! ```

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use oxideav_h265::decode_annexb_sequence;
use oxideav_h265::nal::NalIter;
use oxideav_h265::picture::Plane;
use oxideav_h265::sei::{parse_sei_rbsp, PictureHash, SeiNalType, SeiPayload};

/// RFC 1321 MD5 (self-contained; used only to check decoded output
/// against the published conformance digests).
mod md5 {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];

    pub fn hex(data: &[u8]) -> String {
        let mut a0: u32 = 0x67452301;
        let mut b0: u32 = 0xefcdab89;
        let mut c0: u32 = 0x98badcfe;
        let mut d0: u32 = 0x10325476;

        let bit_len = (data.len() as u64).wrapping_mul(8);
        let mut msg = data.to_vec();
        msg.push(0x80);
        while msg.len() % 64 != 56 {
            msg.push(0);
        }
        msg.extend_from_slice(&bit_len.to_le_bytes());

        for chunk in msg.chunks_exact(64) {
            let mut m = [0u32; 16];
            for (i, w) in m.iter_mut().enumerate() {
                *w = u32::from_le_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
            }
            let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
            for i in 0..64 {
                let (f, g) = match i / 16 {
                    0 => ((b & c) | (!b & d), i),
                    1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                    2 => (b ^ c ^ d, (3 * i + 5) % 16),
                    _ => (c ^ (b | !d), (7 * i) % 16),
                };
                let f = f.wrapping_add(a).wrapping_add(K[i]).wrapping_add(m[g]);
                a = d;
                d = c;
                c = b;
                b = b.wrapping_add(f.rotate_left(S[i]));
            }
            a0 = a0.wrapping_add(a);
            b0 = b0.wrapping_add(b);
            c0 = c0.wrapping_add(c);
            d0 = d0.wrapping_add(d);
        }
        let mut out = String::with_capacity(32);
        for word in [a0, b0, c0, d0] {
            for byte in word.to_le_bytes() {
                out.push_str(&format!("{byte:02x}"));
            }
        }
        out
    }
}

/// Every 32-hex-digit token in the stream's sidecar files.
fn sidecar_digests(dir: &Path, stem: &str) -> Vec<String> {
    let mut digests = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return digests;
    };
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let is_sidecar = name.ends_with(".md5")
            || name.ends_with(".md5sum.txt")
            || name.ends_with("_md5sum.txt");
        // Sidecars whose name starts with the stream stem (upstream
        // also uses `<stem>_yuv.md5` and, for IPCM, a shortened stem).
        let stem_unver = stem
            .trim_end_matches(char::is_numeric)
            .trim_end_matches('_');
        let matches_stem = name.starts_with(stem)
            || name.starts_with(stem_unver)
            || stem.starts_with(name.trim_end_matches(".md5").trim_end_matches("_yuv"));
        if !is_sidecar || !matches_stem {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        for token in text.split(|c: char| !c.is_ascii_hexdigit()) {
            if token.len() == 32 {
                digests.push(token.to_ascii_lowercase());
            }
        }
    }
    digests.sort();
    digests.dedup();
    digests
}

/// All §D.2 decoded-picture-hash suffix-SEI messages in the stream,
/// one `Vec<PictureHash>` (per component) per message, in decode order.
fn picture_hashes(data: &[u8]) -> Vec<Vec<PictureHash>> {
    let mut out = Vec::new();
    for unit in NalIter::new(data).flatten() {
        if unit.header.nal_unit_type != 40 {
            continue;
        }
        let Ok(msgs) = parse_sei_rbsp(&unit.rbsp, SeiNalType::Suffix) else {
            continue;
        };
        for m in msgs {
            if let SeiPayload::DecodedPictureHash(h) = m.payload {
                out.push(h.component_hashes);
            }
        }
    }
    out
}

/// §D.3 `pictureData` serialization of one plane (8-bit samples as
/// single bytes, deeper as 16-bit little-endian).
fn plane_bytes(pic: &oxideav_h265::picture::Picture, plane: Plane) -> Vec<u8> {
    let (w, h) = pic.plane_dims(plane);
    let bd = pic.bit_depth(plane);
    let mut out = Vec::with_capacity(w * h * 2);
    for y in 0..h {
        for x in 0..w {
            let v = pic.sample(plane, x, y);
            if bd <= 8 {
                out.push(v as u8);
            } else {
                out.extend_from_slice(&(v as u16).to_le_bytes());
            }
        }
    }
    out
}

/// The per-plane §D.3.19 hash triplet of one decoded picture,
/// computed once per frame (MD5, CRC-16 and checksum per component).
fn frame_hashes(pic: &oxideav_h265::picture::Picture) -> Vec<([u8; 16], u16, u32)> {
    let planes = [Plane::Luma, Plane::Cb, Plane::Cr];
    let n_comps = if pic.chroma_array_type() == 0 { 1 } else { 3 };
    let mut out = Vec::with_capacity(n_comps);
    for &plane in planes.iter().take(n_comps) {
        let bytes = plane_bytes(pic, plane);
        let hexs = md5::hex(&bytes);
        let mut digest = [0u8; 16];
        for i in 0..16 {
            digest[i] = u8::from_str_radix(&hexs[i * 2..i * 2 + 2], 16).unwrap();
        }
        // §D.3.19 CRC-16 (poly 0x1021, init 0xFFFF, two zero bytes
        // appended, bits msb-first).
        let mut crc: u32 = 0xFFFF;
        for &b in bytes.iter().chain([0u8, 0u8].iter()) {
            for bit in (0..8).rev() {
                let msb = (crc >> 15) & 1;
                let d = u32::from((b >> bit) & 1);
                crc = ((crc << 1) & 0xFFFF) ^ (0x1021 * (msb ^ d));
            }
        }
        // §D.3.19 checksum: byte-wise sum with the coordinate xor mask.
        let (pw, ph) = pic.plane_dims(plane);
        let bd = pic.bit_depth(plane);
        let mut sum: u32 = 0;
        for y in 0..ph {
            for x in 0..pw {
                let v = pic.sample(plane, x, y) as u32;
                let xor_mask = ((x & 0xFF) ^ (y & 0xFF) ^ (x >> 8) ^ (y >> 8)) as u32;
                sum = sum
                    .wrapping_add((v & 0xFF) ^ xor_mask)
                    .wrapping_add(if bd > 8 {
                        ((v >> 8) & 0xFF) ^ xor_mask
                    } else {
                        0
                    });
            }
        }
        out.push((digest, crc as u16, sum));
    }
    out
}

/// Match one frame's precomputed hash triplets against one SEI set.
fn frame_matches(frame: &[([u8; 16], u16, u32)], hashes: &[PictureHash]) -> bool {
    if hashes.len() != frame.len() {
        return false;
    }
    frame
        .iter()
        .zip(hashes)
        .all(|((md5d, crc, sum), h)| match h {
            PictureHash::Md5(e) => md5d == e,
            PictureHash::Crc(e) => crc == e,
            PictureHash::Checksum(e) => sum == e,
        })
}

fn scan_branch(branch_dir: &Path, filter: &str, summary: &mut String) -> (usize, usize) {
    let mut streams: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(branch_dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", branch_dir.display()))
        .flatten()
    {
        let p = entry.path();
        match p.extension().and_then(|e| e.to_str()) {
            Some("bit") | Some("bin") => streams.push(p),
            _ => {}
        }
    }
    streams.sort();

    let (mut pass, mut total) = (0usize, 0usize);
    for stream in &streams {
        let stem = stream.file_stem().unwrap().to_string_lossy().into_owned();
        if !filter.is_empty() && !stem.contains(filter) {
            continue;
        }
        total += 1;
        let data = std::fs::read(stream).expect("read bitstream");
        let digests = sidecar_digests(branch_dir, &stem);
        let bitstream_md5 = md5::hex(&data);

        let outcome = std::panic::catch_unwind(|| decode_annexb_sequence(&data));
        let line = match outcome {
            Err(panic) => {
                let msg = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_else(|| "non-string panic".into());
                format!("PANIC  {stem}: {}", msg.lines().next().unwrap_or(""))
            }
            Ok(Err(e)) => format!("ERROR  {stem}: {e}"),
            Ok(Ok(frames)) => {
                let n = frames.len();
                let mut out8 = Vec::new();
                let mut out16 = Vec::new();
                let mut all8 = true;
                for f in &frames {
                    if let Some(p) = f.picture.to_planar_u8() {
                        out8.extend(p);
                    } else {
                        all8 = false;
                    }
                    out16.extend(f.picture.to_planar_le16());
                }
                let d8 = if all8 { Some(md5::hex(&out8)) } else { None };
                let d16 = md5::hex(&out16);
                let yuv_digests: Vec<&String> =
                    digests.iter().filter(|d| **d != bitstream_md5).collect();
                if d8
                    .as_deref()
                    .is_some_and(|d| digests.contains(&d.to_string()))
                {
                    pass += 1;
                    format!("PASS8  {stem}: {n} frames")
                } else if digests.contains(&d16) {
                    pass += 1;
                    format!("PASS16 {stem}: {n} frames")
                } else {
                    // Frame-level triage against the stream's own
                    // decoded-picture-hash SEI messages (order-free
                    // membership: a frame is OK if ANY SEI hash-set
                    // matches it).
                    let sei = picture_hashes(&data);
                    let (mut ok_frames, mut first_bad) = (0usize, None::<usize>);
                    if !sei.is_empty() {
                        for (i, f) in frames.iter().enumerate() {
                            let fh = frame_hashes(&f.picture);
                            if sei.iter().any(|h| frame_matches(&fh, h)) {
                                ok_frames += 1;
                            } else if first_bad.is_none() {
                                first_bad = Some(i);
                            }
                        }
                    }
                    format!(
                        "MISMATCH {stem}: {n} frames, sei_ok={ok_frames}/{} first_bad={first_bad:?}, ours8={} ours16={d16}, published yuv {:?}",
                        sei.len(),
                        d8.unwrap_or_else(|| "-".into()),
                        yuv_digests
                    )
                }
            }
        };
        println!("{line}");
        let _ = writeln!(summary, "{line}");
    }
    (pass, total)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| {
        format!(
            "{}/../../docs/video/h265/conformance",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let filter = args.next().unwrap_or_default();
    let dir = PathBuf::from(dir);
    assert!(dir.is_dir(), "conformance dir not found: {}", dir.display());

    let mut summary = String::new();
    let mut pass = 0;
    let mut total = 0;
    for branch in ["RExt", "SCC"] {
        let branch_dir = dir.join(branch);
        if !branch_dir.is_dir() {
            continue;
        }
        println!("== {branch} ==");
        let (p, t) = scan_branch(&branch_dir, &filter, &mut summary);
        pass += p;
        total += t;
    }
    println!("\n{pass}/{total} streams byte-exact");
}
