//! # oxideav-h265
//!
//! Pure-Rust H.265 / HEVC (ITU-T H.265 | ISO/IEC 23008-2) parser and
//! decoder, for the [oxideav](https://github.com/OxideAV/oxideav)
//! framework.
//!
//! **Status:** clean-room rebuild in progress (post 2026-05-18 audit).
//! Round 11 lands the §9.3 CABAC arithmetic decoding engine
//! ([`cabac::CabacEngine`] / [`cabac::ContextModel`] / [`cabac::init_type`]):
//! the §9.3.2.6 engine-register init, the §9.3.2.2 context-variable init
//! (equations 9-4..9-7), the §9.3.4.3.2 DecodeDecision primitive (with
//! the Table 9-52 / Table 9-53 LPS-range / state-transition tables), the
//! §9.3.4.3.3 RenormD loop, the §9.3.4.3.4 DecodeBypass primitive (with
//! an MSB-first `decode_bypass_bits(n)` helper), the §9.3.4.3.5
//! DecodeTerminate primitive, and the §9.3.4.3.6 aligned-bypass
//! alignment hook. The engine ships standalone — independent of the
//! §9.3.4.2 per-syntax-element binarization / context-index derivation
//! that the slice-data parser still needs.
//!
//! Rounds 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 land the Annex B NAL-unit
//! byte-stream walker, the §7.3.1.2 NAL header parse, the §7.3.2.1
//! VPS structural parse (with a §7.3.3 profile_tier_level walk), the
//! full §7.3.2.2 SPS parse (through the `vui_parameters_present_flag`
//! / `sps_extension_present_flag` gates, with the VUI body and any
//! extension payload surfaced as an opaque-bytes tail), the
//! §7.3.2.3.1 PPS parse (full general body through
//! `pps_extension_present_flag`, including the tiles and
//! deblocking-control blocks; the PPS extension bodies are surfaced as
//! an opaque tail), the §7.3.6.1 slice-segment-header parse —
//! independent I-slice IDR segments end to end (round 6), and
//! independent **non-IDR I-slice** segments through the §7.3.6.1 POC +
//! short-term-RPS + long-term-RPS block end to end (round 7) — and
//! now (round 8) the §7.3.4 `scaling_list_data()` parse with the
//! §7.4.5 `ScalingList[sizeId][matrixId][i]` derivation, wired into
//! both the SPS (`sps_scaling_list_data_present_flag`) and PPS
//! (`pps_scaling_list_data_present_flag`) paths. The P/B
//! reference-list / weighted-prediction sub-structures are still
//! surfaced as an opaque tail. Slice data and CABAC are *not*
//! implemented yet; the public decoder and encoder entry points still
//! return [`Error::NotImplemented`].
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
//!   `scaling_list_enabled_flag` (with the nested
//!   `sps_scaling_list_data_present_flag` / [`scaling_list::ScalingListData`]
//!   §7.3.4 block), `amp_enabled_flag`,
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
//!   The §7.3.4 `scaling_list_data()` block — when
//!   `sps_scaling_list_data_present_flag == 1` — is parsed and the
//!   §7.4.5 `ScalingList[sizeId][matrixId][i]` coefficient arrays are
//!   derived (default tables + prediction inference); see
//!   [`scaling_list::ScalingListData`].
//! * §6.5 [`scan`] — all four scan-order initialization processes plus
//!   the §7.4.2 [`scan::scan_order`] `ScanOrder[log2BlockSize][scanIdx]`
//!   accessor: [`scan::up_right_diagonal`] (§6.5.3, equation 6-11),
//!   [`scan::horizontal`] (§6.5.4, equation 6-12),
//!   [`scan::vertical`] (§6.5.5, equation 6-13), and
//!   [`scan::traverse`] (§6.5.6, equation 6-14, the boustrophedon
//!   raster). [`scan::scan_order`] enforces §7.4.2's populated ranges
//!   (`log2BlockSize` 0..=3 for diagonal / horizontal / vertical, 2..=5
//!   for traverse). §7.4.5
//!   [`scaling_list::ScalingListData::scaling_factors`] expands the
//!   flat scaling lists into the two-dimensional
//!   `ScalingFactor[sizeId][matrixId][x][y]` quantization matrices
//!   (equations 7-44..7-51: the diagonal scatter, the 2x / 4x block
//!   replication, the DC `[0][0]` override, and the
//!   `ChromaArrayType == 3` 32x32-chroma derivation).
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
//!   shared [`sps::OpaqueTail`]). When
//!   `pps_scaling_list_data_present_flag == 1` the §7.3.4
//!   `scaling_list_data()` block is parsed into
//!   [`scaling_list::ScalingListData`]. The §7.4.3.3.1 inference rules
//!   are applied so absent conditional fields carry their effective
//!   value.
//! * §7.3.6.1 [`slice::SliceSegmentHeader`] — the
//!   `slice_segment_header()` parse for an independent slice segment,
//!   taking the activated SPS + PPS as context (the
//!   `slice_segment_address` and `slice_pic_order_cnt_lsb` widths plus
//!   the SAO / MVP / tiles gates are SPS/PPS-derived). Independent
//!   **I-slice** segments — both IDR and non-IDR — parse end to end
//!   through `byte_alignment()`, including the §7.3.6.1 non-IDR POC
//!   (`slice_pic_order_cnt_lsb`) + short-term-RPS
//!   (`short_term_ref_pic_set_sps_flag` /
//!   in-line `st_ref_pic_set(num_short_term_ref_pic_sets)` via
//!   [`sps::ShortTermRefPicSet::parse_slice_inline`] /
//!   `short_term_ref_pic_set_idx`) + long-term-RPS block (per-entry
//!   SPS-indexed vs in-slice + `delta_poc_msb_present_flag` /
//!   `delta_poc_msb_cycle_lt`, surfaced as
//!   [`slice::SliceLongTermRefPic`]). The P/B reference-list /
//!   weighted-prediction sub-structures are still surfaced as an
//!   [`sps::OpaqueTail`]. The §7.4.7.1 inference rules are applied to
//!   absent fields.
//!
//! See [`nal`] for the byte-stream walker entry points, [`vps`] for
//! the parsed VPS structure, [`sps`] for the parsed SPS, [`pps`]
//! for the parsed PPS, and [`crate::slice`] for the parsed slice
//! header.

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

pub mod bitreader;
pub mod cabac;
pub mod nal;
pub mod pps;
pub mod scaling_list;
pub mod scan;
pub mod slice;
pub mod sps;
pub mod vps;

pub use bitreader::{BitReader, BitReaderError};
pub use cabac::{init_type, CabacEngine, CabacError, ContextModel};
pub use nal::{collect_nal_units, NalError, NalHeader, NalIter, NalUnit};
pub use pps::{DeblockingFilterControl, PicParameterSet, PpsError, TileInfo};
pub use scaling_list::{
    ScalingFactorMatrix, ScalingFactors, ScalingListData, ScalingListError, ScalingListMatrix,
    MAX_COEF_NUM, NUM_MATRIX_IDS, NUM_SIZE_IDS,
};
pub use scan::{
    horizontal, scan_order, traverse, up_right_diagonal, vertical, ScanIdx, ScanOrderError, ScanPos,
};
pub use slice::{
    EntryPointOffsets, SliceDeblocking, SliceError, SliceLongTermRefPic, SliceLongTermRefPicSource,
    SliceSegmentHeader, SliceType, BLA_W_LP, IDR_N_LP, IDR_W_RADL, RSV_IRAP_VCL23,
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
