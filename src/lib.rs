//! Pure-Rust HEVC / H.265 (ITU-T H.265 | ISO/IEC 23008-2) bitstream parser
//! plus a decoder scaffold.
//!
//! ## Scope (v1)
//!
//! * **NAL unit framing** — Annex B byte-stream and length-prefixed (HVCC)
//!   modes; emulation-prevention byte removal.
//! * **VPS / SPS / PPS parsers** — every field needed to validate stream
//!   shape (resolution, chroma format, bit depth, CTU size, tiles /
//!   wavefront flags, weighted prediction flags, …) is decoded; fields
//!   not needed by the v1 scaffold are walked past so the bit position
//!   stays correct.
//! * **Slice segment header parser** — first slice flag, PPS id, segment
//!   address, slice type, POC LSB, RPS index. Reference-list modification
//!   and weighted-prediction tables are out of scope until CABAC is wired.
//! * **HEVCDecoderConfigurationRecord (`hvcC`) parser** — used by the
//!   MP4 demuxer to populate `extradata`.
//!
//! ## Out of scope (returns `Error::Unsupported`)
//!
//! * **CTU decoding** — `slice_data()` (§7.3.8) requires CABAC entropy
//!   decode (§9), intra and inter prediction (§8.4 / §8.5), transforms
//!   (§8.6), loop filtering, SAO. CABAC alone is roughly 2 KLOC; the full
//!   pipeline is ~15 KLOC and intentionally deferred.
//! * **Scalable / multiview / 3D extensions.**
//! * **Encoder** — write side is not in scope.
//!
//! ## Crate layout
//!
//! * [`bitreader`] — MSB-first reader, ue(v)/se(v) Exp-Golomb helpers.
//! * [`nal`] — start-code scanner, length-prefix iterator, NAL header
//!   parsing, emulation-prevention removal.
//! * [`ptl`] — `profile_tier_level()` (§7.3.3) shared by VPS/SPS.
//! * [`vps`], [`sps`], [`pps`] — parameter-set parsers.
//! * [`slice`] — slice segment header, including the IDR I-slice
//!   extension (SAO flags, slice_qp_delta, byte-aligned `slice_data()`
//!   offset, `SliceQpY` derivation).
//! * [`cabac`] — CABAC per-context initialisation kernel (§9.3.4.2.1).
//! * [`hvcc`] — HEVCDecoderConfigurationRecord (ISO/IEC 14496-15 §8.3.3).
//! * [`decoder`] — registry factory and `HevcDecoder` (parse-only).

#![allow(clippy::too_many_arguments)]

pub mod bitreader;
pub mod cabac;
pub mod ctu;
pub mod decoder;
pub mod hvcc;
pub mod intra_pred;
pub mod nal;
pub mod pps;
pub mod ptl;
pub mod scan;
pub mod slice;
pub mod sps;
pub mod transform;
pub mod vps;

use oxideav_codec::{CodecInfo, CodecRegistry};
use oxideav_core::{CodecCapabilities, CodecId, CodecTag};

/// Canonical oxideav codec id for H.265 / HEVC.
pub const CODEC_ID_STR: &str = "h265";

/// Register the HEVC implementation with a codec registry.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("h265_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(8192, 8192);
    // AVI FourCC claims — HEVC, H265, HVC1, HEV1, X265 (libx265), DXHE
    // (DivX HEVC). Unambiguous.
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(decoder::make_decoder)
            .tags([
                CodecTag::fourcc(b"HEVC"),
                CodecTag::fourcc(b"H265"),
                CodecTag::fourcc(b"HVC1"),
                CodecTag::fourcc(b"HEV1"),
                CodecTag::fourcc(b"X265"),
                CodecTag::fourcc(b"DXHE"),
            ]),
    );
}
