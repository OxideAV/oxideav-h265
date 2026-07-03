//! H.265 / HEVC encoder — write-side building blocks.
//!
//! The encode stack mirrors the decode stack bottom-up:
//!
//! * [`bitwriter::BitWriter`] — the MSB-first RBSP bit sink (`u(n)`,
//!   `ue(v)` / `se(v)`, `rbsp_trailing_bits()`).
//! * [`nal`] — §7.4.1.1 emulation-prevention escaping, the §7.3.1.2
//!   NAL header, and Annex B framing.
//! * [`cabac::CabacEncoder`] — the §9.3.5 arithmetic encoding engine
//!   (the spec's informative encoder that "matches the arithmetic
//!   decoding engine described in clause 9.3.4.3").
//!
//! Every layer is pinned against the crate's own decode side: bit
//! fields re-read through [`crate::bitreader::BitReader`], NAL units
//! re-walked through [`crate::nal`], and CABAC bin streams re-decoded
//! through [`crate::cabac::CabacEngine`].

pub mod bitwriter;
pub mod cabac;
pub mod nal;
