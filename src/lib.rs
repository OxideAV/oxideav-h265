//! # oxideav-h265
//!
//! Pure-Rust H.265 / HEVC (ITU-T H.265 | ISO/IEC 23008-2) parser and
//! decoder, for the [oxideav](https://github.com/OxideAV/oxideav)
//! framework.
//!
//! **Status:** clean-room rebuild in progress (post 2026-05-18 audit).
//! Round 1 lands the Annex B NAL-unit byte-stream walker and the
//! §7.3.1.2 NAL header parse. SPS/PPS/VPS semantic parse, slice
//! decode, and CABAC are *not* implemented yet; the public decoder
//! and encoder entry points still return [`Error::NotImplemented`].
//!
//! ## What works today
//!
//! * Annex B byte-stream splitting (3- and 4-byte start codes,
//!   trailing-zero padding tolerance).
//! * §7.3.1.2 NAL header parse: `forbidden_zero_bit`,
//!   `nal_unit_type`, `nuh_layer_id`, and `TemporalId` (derived
//!   from `nuh_temporal_id_plus1`).
//! * §7.4.1.1 emulation-prevention byte strip (`0x00 0x00 0x03` →
//!   `0x00 0x00`).
//!
//! See [`nal`] for the byte-stream walker entry points.

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

pub mod nal;

pub use nal::{collect_nal_units, NalError, NalHeader, NalIter, NalUnit};

/// Crate-local error type. The decoder and encoder paths still
/// return [`Error::NotImplemented`] while the clean-room rebuild
/// progresses; structural utilities (the NAL walker) surface their
/// own [`NalError`] type directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The crate has been reset to a scaffold pending clean-room
    /// rebuild; no decoder or encoder functionality is wired up yet.
    NotImplemented,
    /// A NAL-walker error surfaced through the top-level entry
    /// points.
    Nal(NalError),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotImplemented => f.write_str("oxideav-h265: decoder/encoder not wired up yet"),
            Self::Nal(e) => write!(f, "oxideav-h265 NAL error: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<NalError> for Error {
    fn from(e: NalError) -> Self {
        Self::Nal(e)
    }
}

/// No-op codec registration — the clean-room rebuild has not yet
/// registered a decoder or encoder factory.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("h265", register);
