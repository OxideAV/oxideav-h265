//! HEVC encoder — I-slice MVP.
//!
//! Scope (v0):
//! * **Bit writer + Exp-Golomb encoder** — MSB-first writer that mirrors the
//!   decoder's [`crate::bitreader::BitReader`]; `ue(v)` / `se(v)` helpers per
//!   §9.2.
//! * **NAL framing** — Annex B start codes + §7.4.1.1 emulation-prevention
//!   byte insertion.
//! * **Minimal VPS / SPS / PPS** — Main profile, 8-bit 4:2:0, single-tile,
//!   `pcm_enabled_flag = 1` so the CTU walker can emit raw PCM CUs and skip
//!   the full transform / residual-coding pipeline.
//! * **IDR slice header** (§7.3.6) — first_slice_segment_in_pic + minimal
//!   fields.
//! * **CTU walker** — one 64×64 PCM CU per 64×64 CTU; samples are written
//!   byte-aligned after an `end_of_slice_flag` CABAC bit and before the
//!   next CABAC run.
//! * **CABAC encoder** (§9.3 in emit direction) — full arithmetic engine
//!   with bypass + regular modes. For the PCM path we only need
//!   `end_of_slice_flag`, `split_cu_flag`, and `pcm_flag`; the wider
//!   context-init tables from the decoder are reused for writing.
//!
//! Out of scope (v0):
//! * Transform / quantisation / residual coding — deferred; PCM CUs keep
//!   the MVP round-trippable without touching §8.6 / §7.4.11.8.
//! * P / B slices, tiles, wavefront.
//! * Any rate control — QP is irrelevant for PCM.
//!
//! Wiring: [`register_encoder`] attaches [`make_encoder`] to the codec
//! registry; [`HevcEncoder`] implements `oxideav_codec::Encoder`.

pub mod bit_writer;
pub mod cabac_writer;
pub mod hevc_encoder;
pub mod nal_writer;
pub mod params;
pub mod slice_writer;

pub use hevc_encoder::{make_encoder, HevcEncoder};
