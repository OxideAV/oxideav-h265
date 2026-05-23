//! # oxideav-h265
//!
//! Pure-Rust H.265 / HEVC (ITU-T H.265 | ISO/IEC 23008-2) parser and
//! decoder, for the [oxideav](https://github.com/OxideAV/oxideav)
//! framework.
//!
//! **Status:** clean-room rebuild in progress (post 2026-05-18 audit).
//! Rounds 1 + 2 + 3 + 4 + 5 + 6 land the Annex B NAL-unit byte-stream
//! walker, the §7.3.1.2 NAL header parse, the §7.3.2.1 VPS structural
//! parse (with a §7.3.3 profile_tier_level walk), the full §7.3.2.2
//! SPS parse (through the `vui_parameters_present_flag` /
//! `sps_extension_present_flag` gates, with the VUI body and any
//! extension payload surfaced as an opaque-bytes tail), the
//! §7.3.2.3.1 PPS parse (full general body through
//! `pps_extension_present_flag`, including the tiles and
//! deblocking-control blocks; the PPS extension bodies are surfaced as
//! an opaque tail), and the §7.3.6.1 slice-segment-header parse
//! (independent I-slice IDR segments end to end; the non-IDR POC/RPS
//! block and the P/B reference-list / weighted-prediction
//! sub-structures are surfaced as an opaque tail). Slice data and
//! CABAC are *not* implemented yet; the public decoder and encoder
//! entry points still return [`Error::NotImplemented`].
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
//! * §7.3.2.3.1 [`pps::PicParameterSet`] — the full general
//!   `pic_parameter_set_rbsp()` body: the `pps_*_id` pair, the
//!   slice-header gates, `init_qp_minus26` (`se(v)`), the chroma QP
//!   offsets, the tiles block ([`pps::TileInfo`] — column/row counts
//!   plus the explicit `column_width_minus1[]` / `row_height_minus1[]`
//!   arrays when `uniform_spacing_flag == 0`), the
//!   deblocking-filter-control block ([`pps::DeblockingFilterControl`]),
//!   `lists_modification_present_flag`,
//!   `log2_parallel_merge_level_minus2`, and the
//!   `pps_extension_present_flag` gate (extension bodies surfaced as a
//!   shared [`sps::OpaqueTail`]). `pps_scaling_list_data_present_flag
//!   == 1` is refused alongside the SPS scaling-list deferral. The
//!   §7.4.3.3.1 inference rules are applied so absent conditional
//!   fields carry their effective value.
//! * §7.3.6.1 [`slice::SliceSegmentHeader`] — the
//!   `slice_segment_header()` parse for an independent slice segment,
//!   taking the activated SPS + PPS as context (the
//!   `slice_segment_address` and `slice_pic_order_cnt_lsb` widths plus
//!   the SAO / MVP / tiles gates are SPS/PPS-derived). Independent
//!   I-slice IDR segments parse end to end through `byte_alignment()`;
//!   the non-IDR POC / reference-picture-set block and the P/B
//!   reference-list / weighted-prediction sub-structures are surfaced
//!   as an [`sps::OpaqueTail`]. The §7.4.7.1 inference rules are
//!   applied to absent fields.
//!
//! See [`nal`] for the byte-stream walker entry points, [`vps`] for
//! the parsed VPS structure, [`sps`] for the parsed SPS, [`pps`]
//! for the parsed PPS, and [`crate::slice`] for the parsed slice
//! header.

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

pub mod bitreader;
pub mod nal;
pub mod pps;
pub mod slice;
pub mod sps;
pub mod vps;

pub use bitreader::{BitReader, BitReaderError};
pub use nal::{collect_nal_units, NalError, NalHeader, NalIter, NalUnit};
pub use pps::{DeblockingFilterControl, PicParameterSet, PpsError, TileInfo};
pub use slice::{
    EntryPointOffsets, SliceDeblocking, SliceError, SliceSegmentHeader, SliceType, BLA_W_LP,
    IDR_N_LP, IDR_W_RADL, RSV_IRAP_VCL23,
};
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
    /// A PPS-parser error surfaced through the top-level entry
    /// points.
    Pps(PpsError),
    /// A slice-segment-header-parser error surfaced through the
    /// top-level entry points.
    Slice(SliceError),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotImplemented => f.write_str("oxideav-h265: decoder/encoder not wired up yet"),
            Self::Nal(e) => write!(f, "oxideav-h265 NAL error: {e}"),
            Self::Vps(e) => write!(f, "oxideav-h265 VPS error: {e}"),
            Self::Sps(e) => write!(f, "oxideav-h265 SPS error: {e}"),
            Self::Pps(e) => write!(f, "oxideav-h265 PPS error: {e}"),
            Self::Slice(e) => write!(f, "oxideav-h265 slice header error: {e}"),
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

impl From<PpsError> for Error {
    fn from(e: PpsError) -> Self {
        Self::Pps(e)
    }
}

impl From<SliceError> for Error {
    fn from(e: SliceError) -> Self {
        Self::Slice(e)
    }
}

/// No-op codec registration — the clean-room rebuild has not yet
/// registered a decoder or encoder factory.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("h265", register);
