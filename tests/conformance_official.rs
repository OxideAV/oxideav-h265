//! Official JCT-VC RExt / SCC conformance pins.
//!
//! The workspace stages the normative conformance bitstreams under
//! `docs/video/h265/conformance/{RExt,SCC}/` (read in place — nothing
//! is copied into this repo). Every stream named in [`EXPECTED_PASS`]
//! must decode byte-exact: the whole-bitstream decode, serialized as
//! 8-bit planar or 16-bit-LE planar, must reproduce a decoded-output
//! digest published in the stream's own sidecar files.
//!
//! When the staged corpus is not present (CI checks out this crate
//! alone), the test is a no-op — the pins only bind where the data
//! exists.

use std::path::{Path, PathBuf};

use oxideav_h265::decode_annexb_sequence;

/// RFC 1321 MD5 (self-contained; digest comparison only).
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
        let (mut a0, mut b0, mut c0, mut d0) =
            (0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32);
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

/// The streams pinned byte-exact (rounds 437 + 441 + 444). `branch/stem` form.
const EXPECTED_PASS: &[&str] = &[
    "RExt/ADJUST_IPRED_ANGLE_A_RExt_Mitsubishi_2",
    "RExt/Bitdepth_A_RExt_Sony_1",
    "RExt/Bitdepth_B_RExt_Sony_1",
    "RExt/CCP_10bit_RExt_QCOM",
    "RExt/CCP_12bit_RExt_QCOM",
    "RExt/CCP_8bit_RExt_QCOM",
    "RExt/EXTPREC_HIGHTHROUGHPUT_444_16_INTRA_10BIT_RExt_Sony_1",
    "RExt/EXTPREC_HIGHTHROUGHPUT_444_16_INTRA_12BIT_RExt_Sony_1",
    "RExt/EXTPREC_HIGHTHROUGHPUT_444_16_INTRA_16BIT_RExt_Sony_1",
    "RExt/EXTPREC_HIGHTHROUGHPUT_444_16_INTRA_8BIT_RExt_Sony_1",
    "RExt/EXTPREC_MAIN_444_16_INTRA_10BIT_RExt_Sony_1",
    "RExt/EXTPREC_MAIN_444_16_INTRA_12BIT_RExt_Sony_1",
    "RExt/EXTPREC_MAIN_444_16_INTRA_16BIT_RExt_Sony_1",
    "RExt/EXTPREC_MAIN_444_16_INTRA_8BIT_RExt_Sony_1",
    "RExt/ExplicitRdpcm_A_BBC_1",
    "RExt/GENERAL_10b_420_RExt_Sony_1",
    "RExt/GENERAL_10b_422_RExt_Sony_1",
    "RExt/GENERAL_10b_444_RExt_Sony_2",
    "RExt/GENERAL_12b_400_RExt_Sony_1",
    "RExt/GENERAL_12b_420_RExt_Sony_1",
    "RExt/GENERAL_12b_422_RExt_Sony_1",
    "RExt/GENERAL_12b_444_RExt_Sony_2",
    "RExt/GENERAL_16b_400_RExt_Sony_1",
    "RExt/GENERAL_8b_400_RExt_Sony_1",
    "RExt/GENERAL_8b_420_RExt_Sony_1",
    "RExt/GENERAL_8b_444_RExt_Sony_2",
    "RExt/IPCM_A_RExt_NEC_2",
    "RExt/IPCM_B_RExt_NEC",
    "RExt/Main_422_10_A_RExt_Sony_2",
    "RExt/Main_422_10_B_RExt_Sony_2",
    "RExt/PERSIST_RPARAM_A_RExt_Sony_3",
    "RExt/QMATRIX_A_RExt_Sony_1",
    "RExt/SAO_A_RExt_MediaTek_1",
    "RExt/TSCTX_10bit_I_RExt_SHARP_1",
    "RExt/TSCTX_10bit_RExt_SHARP_1",
    "RExt/TSCTX_12bit_I_RExt_SHARP_1",
    "RExt/TSCTX_12bit_RExt_SHARP_1",
    "RExt/TSCTX_8bit_I_RExt_SHARP_1",
    "RExt/TSCTX_8bit_RExt_SHARP_1",
    "RExt/WAVETILES_RExt_Sony_2",
    "RExt/WPP_AND_TILE_10Bit422Test_HIGH_TP_444_10BIT_RExt_Apple_2",
    "RExt/WPP_AND_TILE_AND_CABAC_BYPASS_ALIGN_0_HIGH_TP_444_14BIT_RExt_Apple_2",
    "RExt/WPP_AND_TILE_AND_CABAC_BYPASS_ALIGN_1_HIGH_TP_444_14BIT_RExt_Apple_2",
    "RExt/WPP_AND_TILE_AND_CABAC_EXT_PREC_1_HIGH_TP_444_14BIT_RExt_Apple_2",
    "RExt/WPP_AND_TILE_HIGH_TP_444_8BIT_RExt_Apple_2",
    "RExt/WPP_HIGH_TP_444_8BIT_RExt_Apple_2",
];

fn conformance_dir() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/video/h265/conformance");
    dir.is_dir().then_some(dir)
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
    digests
}

/// Decode one staged stream and check its output digest against the
/// published sidecar digests (8-bit planar or 16-bit-LE planar form).
fn stream_is_byte_exact(dir: &Path, spec: &str) -> bool {
    let (branch, stem) = spec.split_once('/').unwrap();
    let branch_dir = dir.join(branch);
    let stream = ["bit", "bin"]
        .iter()
        .map(|ext| branch_dir.join(format!("{stem}.{ext}")))
        .find(|p| p.is_file())
        .unwrap_or_else(|| panic!("{spec}: staged bitstream not found"));
    let data = std::fs::read(&stream).expect("read bitstream");
    let digests = sidecar_digests(&branch_dir, stem);
    assert!(!digests.is_empty(), "{spec}: no sidecar digests staged");
    let frames = decode_annexb_sequence(&data).unwrap_or_else(|e| panic!("{spec}: decode: {e}"));
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
    (all8 && digests.contains(&md5::hex(&out8))) || digests.contains(&md5::hex(&out16))
}

/// Every stream in [`EXPECTED_PASS`] decodes byte-exact to a digest
/// its own sidecar publishes. Skipped when the corpus is not staged.
#[test]
fn official_conformance_streams_decode_byte_exact() {
    let Some(dir) = conformance_dir() else {
        eprintln!("conformance corpus not staged; skipping");
        return;
    };
    let mut failed = Vec::new();
    let mut checked = 0usize;
    for spec in EXPECTED_PASS {
        // A debug build decodes the multi-megabyte streams orders of
        // magnitude slower than release; keep the routine debug run
        // bounded by pinning only the small streams there. Release
        // runs (and the release-mode conformance sweeps) always cover
        // the full list.
        if cfg!(debug_assertions) {
            let (branch, stem) = spec.split_once('/').unwrap();
            let too_big = ["bit", "bin"]
                .iter()
                .filter_map(|ext| {
                    std::fs::metadata(dir.join(branch).join(format!("{stem}.{ext}"))).ok()
                })
                .all(|m| m.len() > 200_000);
            if too_big {
                continue;
            }
        }
        checked += 1;
        if !stream_is_byte_exact(&dir, spec) {
            failed.push(*spec);
        }
    }
    assert!(checked > 0, "no staged stream matched the pin list");
    assert!(
        failed.is_empty(),
        "official conformance regressions: {failed:?}"
    );
}
