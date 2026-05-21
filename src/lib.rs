//! # oxideav-h265
//!
//! Pure-Rust H.265 / HEVC (ITU-T H.265 | ISO/IEC 23008-2) parser and
//! decoder, for the [oxideav](https://github.com/OxideAV/oxideav)
//! framework.
//!
//! **Status:** clean-room rebuild in progress (post 2026-05-18 audit).
//! Rounds 1 + 2 land the Annex B NAL-unit byte-stream walker, the
//! §7.3.1.2 NAL header parse, and the §7.3.2.1 VPS structural parse
//! (including a §7.3.3 profile_tier_level walk). SPS/PPS semantic
//! parse, slice decode, and CABAC are *not* implemented yet; the
//! public decoder and encoder entry points still return
//! [`Error::NotImplemented`].
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
//! * MSB-first bit reader with `u(n)` and 0-th-order
//!   unsigned-Exp-Golomb `ue(v)` (§9.2) descriptors.
//! * §7.3.2.1 [`vps::HevcVps`] — vps_id, base-layer / max-layers /
//!   sub-layers / temporal-nesting flags, reserved-0xFFFF validation,
//!   the §7.3.3 profile_tier_level walk (general profile + level +
//!   per-sub-layer present-flag gates and `sub_layer_level_idc`), and
//!   the per-sub-layer DPB / reorder / latency triple loop.
//!
//! See [`nal`] for the byte-stream walker entry points and [`vps`]
//! for the parsed VPS structure.

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

pub mod bitreader;
pub mod nal;
pub mod vps;

pub use bitreader::{BitReader, BitReaderError};
pub use nal::{collect_nal_units, NalError, NalHeader, NalIter, NalUnit};
pub use vps::{HevcVps, ProfileTierLevel, SubLayerOrderingInfo, VpsError, HEVC_MAX_SUB_LAYERS};

/// Crate-local error type. The decoder and encoder paths still
/// return [`Error::NotImplemented`] while the clean-room rebuild
/// progresses; structural utilities (the NAL walker and parameter-set
/// parsers) surface their own [`NalError`] / [`VpsError`] types
/// directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The crate has been reset to a scaffold pending clean-room
    /// rebuild; no decoder or encoder functionality is wired up yet.
    NotImplemented,
    /// A NAL-walker error surfaced through the top-level entry
    /// points.
    Nal(NalError),
    /// A VPS-parser error surfaced through the top-level entry
    /// points.
    Vps(VpsError),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotImplemented => f.write_str("oxideav-h265: decoder/encoder not wired up yet"),
            Self::Nal(e) => write!(f, "oxideav-h265 NAL error: {e}"),
            Self::Vps(e) => write!(f, "oxideav-h265 VPS error: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<NalError> for Error {
    fn from(e: NalError) -> Self {
        Self::Nal(e)
    }
}

impl From<VpsError> for Error {
    fn from(e: VpsError) -> Self {
        Self::Vps(e)
    }
}

/// No-op codec registration — the clean-room rebuild has not yet
/// registered a decoder or encoder factory.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("h265", register);
