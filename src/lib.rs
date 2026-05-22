//! # oxideav-h265
//!
//! Pure-Rust H.265 / HEVC (ITU-T H.265 | ISO/IEC 23008-2) parser and
//! decoder, for the [oxideav](https://github.com/OxideAV/oxideav)
//! framework.
//!
//! **Status:** clean-room rebuild in progress (post 2026-05-18 audit).
//! Rounds 1 + 2 + 3 + 4 land the Annex B NAL-unit byte-stream walker,
//! the §7.3.1.2 NAL header parse, the §7.3.2.1 VPS structural parse
//! (with a §7.3.3 profile_tier_level walk), and the full §7.3.2.2 SPS
//! parse (through the `vui_parameters_present_flag` /
//! `sps_extension_present_flag` gates, with the VUI body and any
//! extension payload surfaced as an opaque-bytes tail). PPS semantic
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
//! * §7.3.2.2 [`sps::SeqParameterSet`] — vps-id back-reference,
//!   max-sub-layers / nesting flag, the §7.3.3 PTL re-walk,
//!   `chroma_format_idc` / `separate_colour_plane_flag`,
//!   `pic_width_in_luma_samples` / `pic_height_in_luma_samples`,
//!   conformance-window quad, `bit_depth_{luma,chroma}_minus8`,
//!   `log2_max_pic_order_cnt_lsb_minus4`, the per-sub-layer
//!   DPB / reorder / latency triple loop, the four
//!   `log2_*_block_size{_minus_2,_minus_3,_diff_max_min}` fields,
//!   `max_transform_hierarchy_depth_{inter,intra}`,
//!   `scaling_list_enabled_flag`, `amp_enabled_flag`,
//!   `sample_adaptive_offset_enabled_flag`, the [`sps::PcmInfo`] block
//!   gated by `pcm_enabled_flag`, the
//!   `num_short_term_ref_pic_sets` ue(v) + per-set
//!   [`sps::ShortTermRefPicSet`] (§7.3.7, both explicit and
//!   inter-RPS-prediction forms), the
//!   `long_term_ref_pics_present_flag` block plus
//!   [`sps::LongTermRefPicEntry`] table, the
//!   `sps_temporal_mvp_enabled_flag` /
//!   `strong_intra_smoothing_enabled_flag` pair, and the
//!   `vui_parameters_present_flag` / `sps_extension_present_flag`
//!   gates whose bodies are surfaced as [`sps::OpaqueTail`].
//!   Scaling-list data (§7.3.4) is still deferred — the parser
//!   refuses `scaling_list_enabled_flag == 1`.
//!
//! See [`nal`] for the byte-stream walker entry points, [`vps`] for
//! the parsed VPS structure, and [`sps`] for the parsed SPS.

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

pub mod bitreader;
pub mod nal;
pub mod sps;
pub mod vps;

pub use bitreader::{BitReader, BitReaderError};
pub use nal::{collect_nal_units, NalError, NalHeader, NalIter, NalUnit};
pub use sps::{
    ConformanceWindow, LongTermRefPicEntry, OpaqueTail, PcmInfo, SeqParameterSet,
    ShortTermRefPicSet, SpsError, HEVC_MAX_NUM_LONG_TERM_RPS, HEVC_MAX_NUM_SHORT_TERM_RPS,
    HEVC_MAX_RPS_PICS,
};
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
    /// An SPS-parser error surfaced through the top-level entry
    /// points.
    Sps(SpsError),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotImplemented => f.write_str("oxideav-h265: decoder/encoder not wired up yet"),
            Self::Nal(e) => write!(f, "oxideav-h265 NAL error: {e}"),
            Self::Vps(e) => write!(f, "oxideav-h265 VPS error: {e}"),
            Self::Sps(e) => write!(f, "oxideav-h265 SPS error: {e}"),
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

impl From<SpsError> for Error {
    fn from(e: SpsError) -> Self {
        Self::Sps(e)
    }
}

/// No-op codec registration — the clean-room rebuild has not yet
/// registered a decoder or encoder factory.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("h265", register);
