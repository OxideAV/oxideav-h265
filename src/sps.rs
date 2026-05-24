//! Sequence Parameter Set (SPS) parser per ITU-T Rec. H.265 §7.3.2.2.
//!
//! Round-4 scope: parse the full SPS RBSP body through the
//! `strong_intra_smoothing_enabled_flag` field and the
//! `vui_parameters_present_flag` / `sps_extension_present_flag` gates.
//! When either gate is set, the trailing bytes (VUI body and/or
//! extension payload and/or RBSP trailing bits) are surfaced as an
//! **opaque tail**: a copy of the still-unparsed RBSP bytes plus the
//! bit offset within the first byte at which the opaque tail begins.
//! The VUI body and the per-extension `sps_*_extension( )` syntax
//! structures are not materialised in this round.
//!
//! When `scaling_list_enabled_flag == 1` and
//! `sps_scaling_list_data_present_flag == 1`, the §7.3.4
//! `scaling_list_data()` block is parsed via the shared
//! [`crate::scaling_list`] module; otherwise the §7.4.5 default lists
//! apply.
//!
//! ## Layout summary
//!
//! ```text
//! sps_video_parameter_set_id                       u(4)
//! sps_max_sub_layers_minus1                        u(3)
//! sps_temporal_id_nesting_flag                     u(1)
//! profile_tier_level( 1, sps_max_sub_layers_minus1 )
//! sps_seq_parameter_set_id                        ue(v)
//! chroma_format_idc                               ue(v)
//! if( chroma_format_idc == 3 )
//!   separate_colour_plane_flag                     u(1)
//! pic_width_in_luma_samples                       ue(v)
//! pic_height_in_luma_samples                      ue(v)
//! conformance_window_flag                          u(1)
//! if( conformance_window_flag ) {
//!   conf_win_left_offset                          ue(v)
//!   conf_win_right_offset                         ue(v)
//!   conf_win_top_offset                           ue(v)
//!   conf_win_bottom_offset                        ue(v)
//! }
//! bit_depth_luma_minus8                           ue(v)
//! bit_depth_chroma_minus8                         ue(v)
//! log2_max_pic_order_cnt_lsb_minus4               ue(v)
//! sps_sub_layer_ordering_info_present_flag         u(1)
//! for( i = (...) ; i <= sps_max_sub_layers_minus1; i++ ) {
//!   sps_max_dec_pic_buffering_minus1[i]           ue(v)
//!   sps_max_num_reorder_pics[i]                   ue(v)
//!   sps_max_latency_increase_plus1[i]             ue(v)
//! }
//! log2_min_luma_coding_block_size_minus3          ue(v)
//! log2_diff_max_min_luma_coding_block_size        ue(v)
//! log2_min_luma_transform_block_size_minus2       ue(v)
//! log2_diff_max_min_luma_transform_block_size     ue(v)
//! max_transform_hierarchy_depth_inter             ue(v)
//! max_transform_hierarchy_depth_intra             ue(v)
//! scaling_list_enabled_flag                        u(1)
//!   if( scaling_list_enabled_flag ) {
//!     sps_scaling_list_data_present_flag            u(1)
//!     if( sps_scaling_list_data_present_flag )
//!       scaling_list_data( )                        /* §7.3.4 */
//!   }
//! amp_enabled_flag                                 u(1)
//! sample_adaptive_offset_enabled_flag              u(1)
//! pcm_enabled_flag                                 u(1)
//! if( pcm_enabled_flag ) {
//!   pcm_sample_bit_depth_luma_minus1               u(4)
//!   pcm_sample_bit_depth_chroma_minus1             u(4)
//!   log2_min_pcm_luma_coding_block_size_minus3    ue(v)
//!   log2_diff_max_min_pcm_luma_coding_block_size  ue(v)
//!   pcm_loop_filter_disabled_flag                  u(1)
//! }
//! num_short_term_ref_pic_sets                     ue(v)
//! for( i = 0; i < num_short_term_ref_pic_sets; i++ )
//!   st_ref_pic_set( i )
//! long_term_ref_pics_present_flag                  u(1)
//! if( long_term_ref_pics_present_flag ) {
//!   num_long_term_ref_pics_sps                    ue(v)
//!   for( i = 0; i < num_long_term_ref_pics_sps; i++ ) {
//!     lt_ref_pic_poc_lsb_sps[i]                    u(v)  /* log2_max_poc_lsb+4 */
//!     used_by_curr_pic_lt_sps_flag[i]              u(1)
//!   }
//! }
//! sps_temporal_mvp_enabled_flag                    u(1)
//! strong_intra_smoothing_enabled_flag              u(1)
//! vui_parameters_present_flag                      u(1)
//!   /* opaque tail begins here when 1 */
//! sps_extension_present_flag                       u(1)
//!   /* opaque tail begins here when 1 (and vui flag was 0) */
//! ```
//!
//! Validity checks performed here, sourced from §7.4.3.2:
//!
//! * `sps_max_sub_layers_minus1` range 0..=6.
//! * `sps_video_parameter_set_id` and `sps_seq_parameter_set_id`
//!   range 0..=15.
//! * `chroma_format_idc` range 0..=3.
//! * `pic_width_in_luma_samples` and `pic_height_in_luma_samples` not
//!   equal to zero (the modulo-`MinCbSizeY` check is deferred — it
//!   depends on the not-yet-validated `log2_min_cb_size`).
//! * `bit_depth_luma_minus8` and `bit_depth_chroma_minus8` range
//!   0..=8 (§7.4.3.2 caps the encoded bit depth at 16).
//! * `log2_max_pic_order_cnt_lsb_minus4` range 0..=12.
//! * `pcm_sample_bit_depth_luma_minus1` / `pcm_sample_bit_depth_chroma_minus1`
//!   must satisfy `PcmBitDepthY <= BitDepthY` /
//!   `PcmBitDepthC <= BitDepthC` per §7.4.3.2 / equation (7-25 / 7-26).
//! * `num_short_term_ref_pic_sets` range 0..=64.
//! * `num_long_term_ref_pics_sps` range 0..=32.
//! * `num_negative_pics` / `num_positive_pics` capped at 16 (the
//!   §7.4.8 bound is
//!   `sps_max_dec_pic_buffering_minus1[sps_max_sub_layers_minus1]`,
//!   itself bounded at 16 because `MaxDpbSize` per §A.4.2 is 16).
//! * `delta_poc_s0_minus1` / `delta_poc_s1_minus1` /
//!   `abs_delta_rps_minus1` range 0..=2^15-1.
//! * `sps_scaling_list_data_present_flag == 1` triggers a §7.3.4
//!   `scaling_list_data()` parse into [`ScalingListData`] via the
//!   shared [`crate::scaling_list`] module.

use crate::bitreader::{BitReader, BitReaderError};
use crate::scaling_list::{ScalingListData, ScalingListError};
use crate::vps::{ProfileTierLevel, SubLayerOrderingInfo, VpsError, HEVC_MAX_SUB_LAYERS};

/// Maximum number of short-term reference picture sets an SPS may
/// carry. Per §7.4.3.2 the field is bounded at 64 inclusive.
pub const HEVC_MAX_NUM_SHORT_TERM_RPS: usize = 64;

/// Maximum number of long-term reference pictures an SPS may carry.
/// Per §7.4.3.2 the field is bounded at 32 inclusive.
pub const HEVC_MAX_NUM_LONG_TERM_RPS: usize = 32;

/// Maximum number of negative / positive entries permitted in one
/// short-term RPS. Per §7.4.8 the bound is
/// `sps_max_dec_pic_buffering_minus1[sps_max_sub_layers_minus1]`
/// which §A.4.2 caps at `MaxDpbSize - 1 = 15` (for typical levels);
/// 16 is a defensive upper bound for the parser.
pub const HEVC_MAX_RPS_PICS: usize = 16;

/// Errors that can arise while parsing an SPS RBSP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpsError {
    /// The RBSP ran out of bits before the SPS was fully parsed.
    Truncated,
    /// A syntax element's parsed value was outside the legal range
    /// specified for it in §7.4.3.2.
    ValueOutOfRange {
        /// Name of the offending syntax element.
        field: &'static str,
        /// The (illegal) value as a `u32` (signed elements are
        /// re-cast at the call site).
        got: u32,
    },
    /// A `scaling_list_data()` parse (§7.3.4) from the SPS failed.
    ScalingList(ScalingListError),
    /// An unexpected bitstream-level error surfaced from the reader.
    Bitstream(BitReaderError),
    /// A `profile_tier_level()` parse from the §7.3.3 walk failed.
    Ptl(VpsError),
}

impl core::fmt::Display for SpsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => f.write_str("SPS RBSP truncated"),
            Self::ValueOutOfRange { field, got } => {
                write!(f, "SPS syntax element {field} out of range: {got}")
            }
            Self::ScalingList(e) => write!(f, "scaling-list error during SPS parse: {e}"),
            Self::Bitstream(e) => write!(f, "bitstream error during SPS parse: {e}"),
            Self::Ptl(e) => write!(f, "profile_tier_level error during SPS parse: {e}"),
        }
    }
}

impl std::error::Error for SpsError {}

impl From<BitReaderError> for SpsError {
    fn from(e: BitReaderError) -> Self {
        match e {
            BitReaderError::EndOfBuffer => Self::Truncated,
            other => Self::Bitstream(other),
        }
    }
}

impl From<VpsError> for SpsError {
    fn from(e: VpsError) -> Self {
        // Surface the inner reader-truncation directly so callers can
        // distinguish "ran off the end mid-PTL" from other PTL faults.
        if matches!(e, VpsError::Truncated) {
            Self::Truncated
        } else {
            Self::Ptl(e)
        }
    }
}

impl From<ScalingListError> for SpsError {
    fn from(e: ScalingListError) -> Self {
        // Flatten truncation / raw-reader faults to the SPS-level
        // equivalents so the public surface stays predictable; carry
        // the structured scaling-list faults through as-is.
        match e {
            ScalingListError::Truncated => Self::Truncated,
            ScalingListError::Bitstream(b) => Self::Bitstream(b),
            other => Self::ScalingList(other),
        }
    }
}

/// Conformance-window offsets, in §6.5.5 SubWidthC/SubHeightC units
/// (not pixel units — the multiplier depends on
/// [`SeqParameterSet::chroma_format_idc`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConformanceWindow {
    /// `conf_win_left_offset` (`ue(v)`).
    pub left_offset: u32,
    /// `conf_win_right_offset` (`ue(v)`).
    pub right_offset: u32,
    /// `conf_win_top_offset` (`ue(v)`).
    pub top_offset: u32,
    /// `conf_win_bottom_offset` (`ue(v)`).
    pub bottom_offset: u32,
}

/// PCM-related SPS block per §7.3.2.2 (`pcm_enabled_flag == 1` only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcmInfo {
    /// `pcm_sample_bit_depth_luma_minus1` (`u(4)`). The §7.4.3.2
    /// `PcmBitDepthY` derivation is `value + 1`.
    pub bit_depth_luma_minus1: u8,
    /// `pcm_sample_bit_depth_chroma_minus1` (`u(4)`). `PcmBitDepthC = value + 1`.
    pub bit_depth_chroma_minus1: u8,
    /// `log2_min_pcm_luma_coding_block_size_minus3` (`ue(v)`).
    /// `Log2MinIpcmCbSizeY = value + 3`.
    pub log2_min_pcm_luma_coding_block_size_minus3: u8,
    /// `log2_diff_max_min_pcm_luma_coding_block_size` (`ue(v)`).
    /// `Log2MaxIpcmCbSizeY = Log2MinIpcmCbSizeY + value`.
    pub log2_diff_max_min_pcm_luma_coding_block_size: u8,
    /// `pcm_loop_filter_disabled_flag` (`u(1)`).
    pub loop_filter_disabled_flag: bool,
}

/// Short-term reference picture set per §7.3.7. Only the explicit
/// (non-inter-predicted) form materialises the per-entry POC and
/// `used_by_curr_pic` arrays; the inter-RPS-prediction form materialises
/// only the inputs to the §7.4.8 derivation (`delta_idx_minus1`,
/// `delta_rps_sign`, `abs_delta_rps_minus1`, and the
/// `used_by_curr_pic_flag` / `use_delta_flag` arrays of length
/// `NumDeltaPocs[RefRpsIdx] + 1`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShortTermRefPicSet {
    /// `inter_ref_pic_set_prediction_flag` — inferred to 0 when not
    /// signalled (i.e. always 0 for `stRpsIdx == 0`).
    pub inter_ref_pic_set_prediction_flag: bool,
    /// `delta_idx_minus1` (only meaningful when
    /// `inter_ref_pic_set_prediction_flag == 1` and the index this RPS
    /// is being constructed at equals `num_short_term_ref_pic_sets`;
    /// otherwise inferred to 0). The §7.4.8 derivation is
    /// `RefRpsIdx = stRpsIdx - (delta_idx_minus1 + 1)`.
    pub delta_idx_minus1: u32,
    /// `delta_rps_sign` (1-bit).
    pub delta_rps_sign: bool,
    /// `abs_delta_rps_minus1` (`ue(v)`, range 0..=2^15-1).
    pub abs_delta_rps_minus1: u32,
    /// Per-entry `used_by_curr_pic_flag[j]` for the inter-RPS-prediction
    /// case. Length is `NumDeltaPocs[RefRpsIdx] + 1` when the form
    /// applies, zero otherwise.
    pub used_by_curr_pic_flag: Vec<bool>,
    /// Per-entry `use_delta_flag[j]` for the inter-RPS-prediction case.
    /// Same length / population rule as
    /// [`used_by_curr_pic_flag`](Self::used_by_curr_pic_flag); per
    /// §7.4.8 the value is inferred to 1 when the
    /// `used_by_curr_pic_flag[j]` was 1.
    pub use_delta_flag: Vec<bool>,
    /// `num_negative_pics` (`ue(v)`), only meaningful in the explicit
    /// form.
    pub num_negative_pics: u32,
    /// `num_positive_pics` (`ue(v)`), only meaningful in the explicit
    /// form.
    pub num_positive_pics: u32,
    /// `delta_poc_s0_minus1[i]` (`ue(v)`), explicit form. Length
    /// `num_negative_pics`.
    pub delta_poc_s0_minus1: Vec<u32>,
    /// `used_by_curr_pic_s0_flag[i]` (`u(1)`), explicit form. Length
    /// `num_negative_pics`.
    pub used_by_curr_pic_s0_flag: Vec<bool>,
    /// `delta_poc_s1_minus1[i]` (`ue(v)`), explicit form. Length
    /// `num_positive_pics`.
    pub delta_poc_s1_minus1: Vec<u32>,
    /// `used_by_curr_pic_s1_flag[i]` (`u(1)`), explicit form. Length
    /// `num_positive_pics`.
    pub used_by_curr_pic_s1_flag: Vec<bool>,
}

impl ShortTermRefPicSet {
    /// `NumDeltaPocs[stRpsIdx]` per §7.4.8: in the explicit form this
    /// is `num_negative_pics + num_positive_pics`; in the inter-RPS
    /// form the same identity holds after the source-RPS derivation
    /// finishes, but this method returns the post-explicit count so
    /// the caller can chain derivations without re-running the §7.4.8
    /// equations.
    pub fn num_delta_pocs(&self) -> u32 {
        if self.inter_ref_pic_set_prediction_flag {
            // The §7.4.8 derivation populates `NumNegativePics[stRpsIdx]`
            // and `NumPositivePics[stRpsIdx]` from the
            // `used_by_curr_pic_flag` / `use_delta_flag` arrays plus the
            // source RPS. Counting the `use_delta_flag == 1` entries
            // gives an upper bound; an exact count requires walking the
            // source RPS chain. The simple count is correct for the
            // typical case where `inter_ref_pic_set_prediction_flag` is
            // used but `use_delta_flag` defaults to 1 for every entry.
            self.use_delta_flag.iter().filter(|&&v| v).count() as u32
        } else {
            self.num_negative_pics + self.num_positive_pics
        }
    }
}

/// Long-term reference picture entry per the `long_term_ref_pics_present_flag`
/// block of §7.3.2.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LongTermRefPicEntry {
    /// `lt_ref_pic_poc_lsb_sps[i]` — `u(v)` whose width is
    /// `log2_max_pic_order_cnt_lsb_minus4 + 4` bits.
    pub poc_lsb: u32,
    /// `used_by_curr_pic_lt_sps_flag[i]`.
    pub used_by_curr_pic: bool,
}

/// Opaque suffix surfaced when the parser hits a tail field whose
/// body it does not yet decode (VUI parameters and/or the SPS
/// extension flag block). The bytes captured are the still-unparsed
/// RBSP body, starting at the byte that contains the next un-read
/// bit. `start_bit_in_first_byte` is the bit offset of that next bit
/// within `bytes[0]` (0 = MSB).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpaqueTail {
    /// Raw RBSP bytes from the first byte containing the next
    /// un-read bit onwards (including the `rbsp_trailing_bits()` byte
    /// at the end).
    pub bytes: Vec<u8>,
    /// Bit offset within `bytes[0]` where the opaque tail begins,
    /// in MSB-first order (0..=7).
    pub start_bit_in_first_byte: u8,
}

/// Parsed Sequence Parameter Set per §7.3.2.2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeqParameterSet {
    /// `sps_video_parameter_set_id` (`u(4)`, range 0..=15).
    pub vps_id: u8,
    /// `sps_max_sub_layers_minus1` (`u(3)`, range 0..=6).
    pub max_sub_layers_minus1: u8,
    /// `sps_temporal_id_nesting_flag`.
    pub temporal_id_nesting_flag: bool,
    /// Parsed `profile_tier_level()` subroutine.
    pub ptl: ProfileTierLevel,
    /// `sps_seq_parameter_set_id` (`ue(v)`, range 0..=15).
    pub sps_id: u8,
    /// `chroma_format_idc` (`ue(v)`, range 0..=3). 0 monochrome, 1
    /// 4:2:0, 2 4:2:2, 3 4:4:4.
    pub chroma_format_idc: u8,
    /// `separate_colour_plane_flag`. Inferred to false when not
    /// signalled (which is whenever `chroma_format_idc != 3`).
    pub separate_colour_plane_flag: bool,
    /// `pic_width_in_luma_samples` (`ue(v)`).
    pub pic_width_in_luma_samples: u32,
    /// `pic_height_in_luma_samples` (`ue(v)`).
    pub pic_height_in_luma_samples: u32,
    /// `conformance_window_flag`.
    pub conformance_window_flag: bool,
    /// `conf_win_*_offset` triple plus bottom — zeroed when
    /// `conformance_window_flag` is false (§7.4.3.2.1).
    pub conformance_window: ConformanceWindow,
    /// `bit_depth_luma_minus8` (`ue(v)`, range 0..=8). The decoded
    /// `BitDepthY` is `8 + value`.
    pub bit_depth_luma_minus8: u8,
    /// `bit_depth_chroma_minus8` (`ue(v)`, range 0..=8).
    pub bit_depth_chroma_minus8: u8,
    /// `log2_max_pic_order_cnt_lsb_minus4` (`ue(v)`, range 0..=12).
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    /// `sps_sub_layer_ordering_info_present_flag`.
    pub sub_layer_ordering_info_present_flag: bool,
    /// Per-sub-layer DPB / reorder / latency triples. Indices outside
    /// `0..=max_sub_layers_minus1` are zero-initialised; when the
    /// present flag was 0 every lower-indexed entry is copied from
    /// the `[max_sub_layers_minus1]` slot (§7.4.3.2.1).
    pub sub_layer_ordering_info: [SubLayerOrderingInfo; HEVC_MAX_SUB_LAYERS],
    /// `log2_min_luma_coding_block_size_minus3` (`ue(v)`).
    pub log2_min_luma_coding_block_size_minus3: u8,
    /// `log2_diff_max_min_luma_coding_block_size` (`ue(v)`). Per the
    /// `log2_ctb_size` derivation in §7.4.3.2.1, the CTU size is
    /// `1 << (3 + log2_min_cb_minus3 + log2_diff_max_min_luma_cb)`.
    pub log2_diff_max_min_luma_coding_block_size: u8,
    /// `log2_min_luma_transform_block_size_minus2` (`ue(v)`).
    pub log2_min_luma_transform_block_size_minus2: u8,
    /// `log2_diff_max_min_luma_transform_block_size` (`ue(v)`).
    pub log2_diff_max_min_luma_transform_block_size: u8,
    /// `max_transform_hierarchy_depth_inter` (`ue(v)`).
    pub max_transform_hierarchy_depth_inter: u8,
    /// `max_transform_hierarchy_depth_intra` (`ue(v)`).
    pub max_transform_hierarchy_depth_intra: u8,
    /// `scaling_list_enabled_flag` (§7.3.2.2). When set, the SPS
    /// either carries an explicit [`Self::scaling_list_data`] (when
    /// `sps_scaling_list_data_present_flag == 1`) or the §7.4.5 default
    /// scaling lists apply.
    pub scaling_list_enabled_flag: bool,
    /// `sps_scaling_list_data_present_flag` (§7.3.2.2). Inferred to
    /// `false` (the §7.4.5 default lists apply) when
    /// `scaling_list_enabled_flag == 0`.
    pub sps_scaling_list_data_present_flag: bool,
    /// The parsed §7.3.4 `scaling_list_data()` structure when
    /// `sps_scaling_list_data_present_flag == 1`; `None` otherwise (the
    /// §7.4.5 default lists apply).
    pub scaling_list_data: Option<ScalingListData>,
    /// `amp_enabled_flag` — asymmetric motion partitions.
    pub amp_enabled_flag: bool,
    /// `sample_adaptive_offset_enabled_flag` — SPS-level gate for
    /// SAO; the per-slice gates are in the slice header.
    pub sample_adaptive_offset_enabled_flag: bool,
    /// `pcm_enabled_flag` — when set, [`Self::pcm`] is populated.
    pub pcm_enabled_flag: bool,
    /// `pcm_*` block per §7.3.2.2 (only meaningful when
    /// [`Self::pcm_enabled_flag`] is true).
    pub pcm: Option<PcmInfo>,
    /// `num_short_term_ref_pic_sets` (`ue(v)`, range 0..=64).
    pub num_short_term_ref_pic_sets: u32,
    /// Parsed `st_ref_pic_set()` entries, length
    /// [`Self::num_short_term_ref_pic_sets`].
    pub short_term_ref_pic_sets: Vec<ShortTermRefPicSet>,
    /// `long_term_ref_pics_present_flag`.
    pub long_term_ref_pics_present_flag: bool,
    /// `num_long_term_ref_pics_sps` (`ue(v)`, range 0..=32). Zero
    /// when [`Self::long_term_ref_pics_present_flag`] is false.
    pub num_long_term_ref_pics_sps: u32,
    /// Per-entry long-term ref pic POC + used-by-curr-pic flag.
    /// Empty when [`Self::long_term_ref_pics_present_flag`] is false.
    pub long_term_ref_pics: Vec<LongTermRefPicEntry>,
    /// `sps_temporal_mvp_enabled_flag`.
    pub sps_temporal_mvp_enabled_flag: bool,
    /// `strong_intra_smoothing_enabled_flag`.
    pub strong_intra_smoothing_enabled_flag: bool,
    /// `vui_parameters_present_flag`. When true, the
    /// `vui_parameters( )` body plus everything after it (including
    /// the `sps_extension_present_flag`, any extension bodies, and
    /// the RBSP trailing bits) is surfaced as
    /// [`Self::opaque_tail`].
    pub vui_parameters_present_flag: bool,
    /// `sps_extension_present_flag`. Only known when
    /// [`Self::vui_parameters_present_flag`] is false; otherwise
    /// inferred to false until the VUI body is parsed. When true,
    /// the extension flags plus their bodies and the RBSP trailing
    /// bits are surfaced as [`Self::opaque_tail`].
    pub sps_extension_present_flag: bool,
    /// Opaque suffix of the SPS RBSP. Populated when a tail body
    /// (VUI or extension) is signalled but not parsed in this round,
    /// or when the parser stops at the end of the structural prefix.
    /// `None` when the SPS ended cleanly after
    /// `sps_extension_present_flag == 0` (in which case only the
    /// `rbsp_trailing_bits()` byte remains and it is consumed
    /// implicitly).
    pub opaque_tail: Option<OpaqueTail>,
}

impl SeqParameterSet {
    /// Parse `seq_parameter_set_rbsp()` starting from the first bit
    /// of the (already-unescaped) RBSP body — i.e. after the two-byte
    /// NAL header has been removed (see [`crate::nal::NalUnit`]).
    pub fn parse(rbsp: &[u8]) -> Result<Self, SpsError> {
        let mut br = BitReader::new(rbsp);
        Self::parse_inner(&mut br, rbsp)
    }

    fn parse_inner(br: &mut BitReader<'_>, rbsp: &[u8]) -> Result<Self, SpsError> {
        let vps_id = br.u(4)? as u8;
        let max_sub_layers_minus1 = br.u(3)? as u8;
        if max_sub_layers_minus1 > 6 {
            return Err(SpsError::ValueOutOfRange {
                field: "sps_max_sub_layers_minus1",
                got: max_sub_layers_minus1 as u32,
            });
        }
        let temporal_id_nesting_flag = br.u1()? != 0;

        // profile_tier_level( 1, sps_max_sub_layers_minus1 )
        let ptl = ProfileTierLevel::parse(br, true, max_sub_layers_minus1)?;

        let sps_id_raw = br.ue()?;
        if sps_id_raw > 15 {
            return Err(SpsError::ValueOutOfRange {
                field: "sps_seq_parameter_set_id",
                got: sps_id_raw,
            });
        }
        let sps_id = sps_id_raw as u8;

        let chroma_format_idc_raw = br.ue()?;
        if chroma_format_idc_raw > 3 {
            return Err(SpsError::ValueOutOfRange {
                field: "chroma_format_idc",
                got: chroma_format_idc_raw,
            });
        }
        let chroma_format_idc = chroma_format_idc_raw as u8;

        let separate_colour_plane_flag = if chroma_format_idc == 3 {
            br.u1()? != 0
        } else {
            false
        };

        let pic_width_in_luma_samples = br.ue()?;
        if pic_width_in_luma_samples == 0 {
            return Err(SpsError::ValueOutOfRange {
                field: "pic_width_in_luma_samples",
                got: 0,
            });
        }
        let pic_height_in_luma_samples = br.ue()?;
        if pic_height_in_luma_samples == 0 {
            return Err(SpsError::ValueOutOfRange {
                field: "pic_height_in_luma_samples",
                got: 0,
            });
        }

        let conformance_window_flag = br.u1()? != 0;
        let conformance_window = if conformance_window_flag {
            ConformanceWindow {
                left_offset: br.ue()?,
                right_offset: br.ue()?,
                top_offset: br.ue()?,
                bottom_offset: br.ue()?,
            }
        } else {
            ConformanceWindow::default()
        };

        let bit_depth_luma_minus8_raw = br.ue()?;
        if bit_depth_luma_minus8_raw > 8 {
            return Err(SpsError::ValueOutOfRange {
                field: "bit_depth_luma_minus8",
                got: bit_depth_luma_minus8_raw,
            });
        }
        let bit_depth_chroma_minus8_raw = br.ue()?;
        if bit_depth_chroma_minus8_raw > 8 {
            return Err(SpsError::ValueOutOfRange {
                field: "bit_depth_chroma_minus8",
                got: bit_depth_chroma_minus8_raw,
            });
        }

        let log2_max_pic_order_cnt_lsb_minus4_raw = br.ue()?;
        if log2_max_pic_order_cnt_lsb_minus4_raw > 12 {
            return Err(SpsError::ValueOutOfRange {
                field: "log2_max_pic_order_cnt_lsb_minus4",
                got: log2_max_pic_order_cnt_lsb_minus4_raw,
            });
        }

        let sub_layer_ordering_info_present_flag = br.u1()? != 0;
        let last = max_sub_layers_minus1 as usize;
        let start = if sub_layer_ordering_info_present_flag {
            0usize
        } else {
            last
        };
        let mut sub_layer_ordering_info = [SubLayerOrderingInfo::default(); HEVC_MAX_SUB_LAYERS];
        for entry in sub_layer_ordering_info
            .iter_mut()
            .take(last + 1)
            .skip(start)
        {
            let max_dpb = br.ue()?;
            let max_reorder = br.ue()?;
            let max_lat = br.ue()?;
            *entry = SubLayerOrderingInfo {
                max_dec_pic_buffering_minus1: max_dpb,
                max_num_reorder_pics: max_reorder,
                max_latency_increase_plus1: max_lat,
            };
        }
        if !sub_layer_ordering_info_present_flag {
            // §7.4.3.2.1: when the present flag is 0, every lower-indexed
            // sub-layer inherits the [max_sub_layers_minus1] triple.
            let copy = sub_layer_ordering_info[last];
            for entry in sub_layer_ordering_info.iter_mut().take(last) {
                *entry = copy;
            }
        }

        let log2_min_luma_coding_block_size_minus3 = br.ue()? as u8;
        let log2_diff_max_min_luma_coding_block_size = br.ue()? as u8;
        let log2_min_luma_transform_block_size_minus2 = br.ue()? as u8;
        let log2_diff_max_min_luma_transform_block_size = br.ue()? as u8;
        let max_transform_hierarchy_depth_inter = br.ue()? as u8;
        let max_transform_hierarchy_depth_intra = br.ue()? as u8;

        let scaling_list_enabled_flag = br.u1()? != 0;
        let mut sps_scaling_list_data_present_flag = false;
        let mut scaling_list_data = None;
        if scaling_list_enabled_flag {
            // §7.3.2.2: when scaling_list_enabled_flag == 1, an inner
            // sps_scaling_list_data_present_flag gates the explicit
            // scaling_list_data() structure (§7.3.4). When the inner
            // flag is 0 the default scaling lists (§7.4.5 Tables 7-5 /
            // 7-6) apply, so the SPS still parses.
            sps_scaling_list_data_present_flag = br.u1()? != 0;
            if sps_scaling_list_data_present_flag {
                scaling_list_data = Some(ScalingListData::parse(br)?);
            }
        }

        let amp_enabled_flag = br.u1()? != 0;
        let sample_adaptive_offset_enabled_flag = br.u1()? != 0;

        let pcm_enabled_flag = br.u1()? != 0;
        let pcm = if pcm_enabled_flag {
            let bit_depth_luma_minus1 = br.u(4)? as u8;
            let pcm_bit_depth_y = bit_depth_luma_minus1 as u32 + 1;
            let bit_depth_y = 8 + bit_depth_luma_minus8_raw;
            if pcm_bit_depth_y > bit_depth_y {
                return Err(SpsError::ValueOutOfRange {
                    field: "pcm_sample_bit_depth_luma_minus1",
                    got: bit_depth_luma_minus1 as u32,
                });
            }
            let bit_depth_chroma_minus1 = br.u(4)? as u8;
            let pcm_bit_depth_c = bit_depth_chroma_minus1 as u32 + 1;
            let bit_depth_c = 8 + bit_depth_chroma_minus8_raw;
            if pcm_bit_depth_c > bit_depth_c {
                return Err(SpsError::ValueOutOfRange {
                    field: "pcm_sample_bit_depth_chroma_minus1",
                    got: bit_depth_chroma_minus1 as u32,
                });
            }
            let log2_min_pcm_luma_coding_block_size_minus3 = br.ue()? as u8;
            let log2_diff_max_min_pcm_luma_coding_block_size = br.ue()? as u8;
            let loop_filter_disabled_flag = br.u1()? != 0;
            Some(PcmInfo {
                bit_depth_luma_minus1,
                bit_depth_chroma_minus1,
                log2_min_pcm_luma_coding_block_size_minus3,
                log2_diff_max_min_pcm_luma_coding_block_size,
                loop_filter_disabled_flag,
            })
        } else {
            None
        };

        let num_short_term_ref_pic_sets_raw = br.ue()?;
        if num_short_term_ref_pic_sets_raw > HEVC_MAX_NUM_SHORT_TERM_RPS as u32 {
            return Err(SpsError::ValueOutOfRange {
                field: "num_short_term_ref_pic_sets",
                got: num_short_term_ref_pic_sets_raw,
            });
        }
        let num_short_term_ref_pic_sets = num_short_term_ref_pic_sets_raw;
        let mut short_term_ref_pic_sets = Vec::with_capacity(num_short_term_ref_pic_sets as usize);
        for st_rps_idx in 0..num_short_term_ref_pic_sets as usize {
            let prev = if st_rps_idx == 0 {
                None
            } else {
                short_term_ref_pic_sets.last()
            };
            let rps = ShortTermRefPicSet::parse(
                br,
                st_rps_idx as u32,
                num_short_term_ref_pic_sets,
                prev,
                &short_term_ref_pic_sets,
            )?;
            short_term_ref_pic_sets.push(rps);
        }

        let long_term_ref_pics_present_flag = br.u1()? != 0;
        let mut num_long_term_ref_pics_sps = 0u32;
        let mut long_term_ref_pics = Vec::new();
        if long_term_ref_pics_present_flag {
            let raw = br.ue()?;
            if raw > HEVC_MAX_NUM_LONG_TERM_RPS as u32 {
                return Err(SpsError::ValueOutOfRange {
                    field: "num_long_term_ref_pics_sps",
                    got: raw,
                });
            }
            num_long_term_ref_pics_sps = raw;
            let poc_lsb_bits = log2_max_pic_order_cnt_lsb_minus4_raw as u8 + 4;
            long_term_ref_pics.reserve(num_long_term_ref_pics_sps as usize);
            for _ in 0..num_long_term_ref_pics_sps {
                let poc_lsb = br.u(poc_lsb_bits)?;
                let used = br.u1()? != 0;
                long_term_ref_pics.push(LongTermRefPicEntry {
                    poc_lsb,
                    used_by_curr_pic: used,
                });
            }
        }

        let sps_temporal_mvp_enabled_flag = br.u1()? != 0;
        let strong_intra_smoothing_enabled_flag = br.u1()? != 0;

        let vui_parameters_present_flag = br.u1()? != 0;

        let (sps_extension_present_flag, opaque_tail) = if vui_parameters_present_flag {
            // VUI body is not yet parsed; everything from here onward
            // (including the eventual sps_extension_present_flag, any
            // extension bodies, and the rbsp_trailing_bits) is surfaced
            // as an opaque tail. `sps_extension_present_flag` is left
            // false in the parsed struct since we genuinely don't know
            // it without decoding the VUI body.
            let tail = OpaqueTail::capture(br, rbsp);
            (false, Some(tail))
        } else if br.bits_left() == 0 {
            // The fixture corpus encoders sometimes elide the
            // sps_extension_present_flag if no extension is signalled
            // and the rbsp_trailing_bits happens to land on a byte
            // boundary; the field is still required, so a buffer with
            // no bits left here is a truncation.
            return Err(SpsError::Truncated);
        } else {
            let flag = br.u1()? != 0;
            if flag {
                // Extension flag block follows (sps_range_extension_flag,
                // sps_multilayer_extension_flag, etc.) plus the various
                // extension bodies and the rbsp_trailing_bits — surface
                // the lot as an opaque tail.
                let tail = OpaqueTail::capture(br, rbsp);
                (true, Some(tail))
            } else {
                // No extension present. Only the rbsp_trailing_bits
                // remain — a single `1` bit followed by zero-padding
                // to a byte boundary. We do not require the caller to
                // have validated it; surface nothing for the opaque tail.
                (false, None)
            }
        };

        Ok(Self {
            vps_id,
            max_sub_layers_minus1,
            temporal_id_nesting_flag,
            ptl,
            sps_id,
            chroma_format_idc,
            separate_colour_plane_flag,
            pic_width_in_luma_samples,
            pic_height_in_luma_samples,
            conformance_window_flag,
            conformance_window,
            bit_depth_luma_minus8: bit_depth_luma_minus8_raw as u8,
            bit_depth_chroma_minus8: bit_depth_chroma_minus8_raw as u8,
            log2_max_pic_order_cnt_lsb_minus4: log2_max_pic_order_cnt_lsb_minus4_raw as u8,
            sub_layer_ordering_info_present_flag,
            sub_layer_ordering_info,
            log2_min_luma_coding_block_size_minus3,
            log2_diff_max_min_luma_coding_block_size,
            log2_min_luma_transform_block_size_minus2,
            log2_diff_max_min_luma_transform_block_size,
            max_transform_hierarchy_depth_inter,
            max_transform_hierarchy_depth_intra,
            scaling_list_enabled_flag,
            sps_scaling_list_data_present_flag,
            scaling_list_data,
            amp_enabled_flag,
            sample_adaptive_offset_enabled_flag,
            pcm_enabled_flag,
            pcm,
            num_short_term_ref_pic_sets,
            short_term_ref_pic_sets,
            long_term_ref_pics_present_flag,
            num_long_term_ref_pics_sps,
            long_term_ref_pics,
            sps_temporal_mvp_enabled_flag,
            strong_intra_smoothing_enabled_flag,
            vui_parameters_present_flag,
            sps_extension_present_flag,
            opaque_tail,
        })
    }

    /// Convenience: derive the SPS-level `BitDepthY` (luma bit depth).
    pub fn bit_depth_luma(&self) -> u8 {
        8 + self.bit_depth_luma_minus8
    }

    /// Derive the SPS-level `BitDepthC` (chroma bit depth).
    pub fn bit_depth_chroma(&self) -> u8 {
        8 + self.bit_depth_chroma_minus8
    }

    /// Derive `MinCbLog2SizeY = log2_min_luma_coding_block_size_minus3 + 3`
    /// (§7.4.3.2.1).
    pub fn log2_min_cb_size(&self) -> u8 {
        self.log2_min_luma_coding_block_size_minus3 + 3
    }

    /// Derive `CtbLog2SizeY = MinCbLog2SizeY
    /// + log2_diff_max_min_luma_coding_block_size` (§7.4.3.2.1).
    pub fn log2_ctb_size(&self) -> u8 {
        self.log2_min_cb_size() + self.log2_diff_max_min_luma_coding_block_size
    }

    /// Derive `MinTbLog2SizeY = log2_min_luma_transform_block_size_minus2 + 2`
    /// (§7.4.3.2.1).
    pub fn log2_min_tb_size(&self) -> u8 {
        self.log2_min_luma_transform_block_size_minus2 + 2
    }

    /// Derive `MaxPicOrderCntLsb = 1 << (log2_max_pic_order_cnt_lsb_minus4 + 4)`
    /// (§7.4.3.2.1).
    pub fn max_pic_order_cnt_lsb(&self) -> u32 {
        1u32 << (self.log2_max_pic_order_cnt_lsb_minus4 + 4)
    }
}

impl OpaqueTail {
    /// Capture all RBSP bytes from the byte holding the next un-read
    /// bit through end-of-buffer.
    fn capture(br: &BitReader<'_>, rbsp: &[u8]) -> Self {
        Self::capture_at(br.bit_pos(), rbsp)
    }

    /// Capture all RBSP bytes from the byte holding the bit at
    /// `bit_pos` (counted MSB-first from the start of `rbsp`) through
    /// end-of-buffer. Used by both the SPS VUI / extension tail and the
    /// [`crate::pps::PicParameterSet`] extension tail.
    pub fn capture_at(bit_pos: usize, rbsp: &[u8]) -> Self {
        let byte_index = bit_pos / 8;
        let bit_in_byte = (bit_pos % 8) as u8;
        Self {
            bytes: rbsp[byte_index..].to_vec(),
            start_bit_in_first_byte: bit_in_byte,
        }
    }
}

impl ShortTermRefPicSet {
    /// Parse the in-line slice-header `st_ref_pic_set( num_short_term_ref_pic_sets )`
    /// per §7.3.6.1 / §7.3.7.
    ///
    /// This is the entry point the slice-header parser uses when
    /// `short_term_ref_pic_set_sps_flag == 0`: the picture's short-term
    /// RPS is constructed *inline* in the slice header at index
    /// `stRpsIdx == num_short_term_ref_pic_sets`, with the SPS's
    /// pre-existing list ([`SeqParameterSet::short_term_ref_pic_sets`])
    /// supplying `all_rps` for the §7.4.8 `RefRpsIdx` derivation.
    ///
    /// `br` must be positioned at the first bit of `st_ref_pic_set()`.
    /// On success the reader is advanced past the structure; on a
    /// truncation or range-check failure the reader state is undefined.
    pub fn parse_slice_inline(
        br: &mut BitReader<'_>,
        sps: &SeqParameterSet,
    ) -> Result<Self, SpsError> {
        Self::parse(
            br,
            sps.num_short_term_ref_pic_sets,
            sps.num_short_term_ref_pic_sets,
            sps.short_term_ref_pic_sets.last(),
            &sps.short_term_ref_pic_sets,
        )
    }

    /// Parse one `st_ref_pic_set( stRpsIdx )` per §7.3.7.
    ///
    /// * `st_rps_idx` is `stRpsIdx`.
    /// * `num_short_term_ref_pic_sets` is the SPS-level count being
    ///   constructed (used to detect when `delta_idx_minus1` is signalled).
    /// * `prev` is the previously-parsed RPS, used when the
    ///   inter-RPS-prediction form is invoked without explicit
    ///   `delta_idx_minus1` (i.e. `stRpsIdx < num_short_term_ref_pic_sets`).
    /// * `all_rps` is the full list of RPSes parsed so far; the
    ///   §7.4.8 `RefRpsIdx` computation indexes into it.
    fn parse(
        br: &mut BitReader<'_>,
        st_rps_idx: u32,
        num_short_term_ref_pic_sets: u32,
        prev: Option<&ShortTermRefPicSet>,
        all_rps: &[ShortTermRefPicSet],
    ) -> Result<Self, SpsError> {
        let inter_ref_pic_set_prediction_flag = if st_rps_idx != 0 {
            br.u1()? != 0
        } else {
            false
        };
        if inter_ref_pic_set_prediction_flag {
            // delta_idx_minus1 is only signalled when the RPS being
            // constructed is the slice-header in-line RPS, i.e.
            // stRpsIdx == num_short_term_ref_pic_sets. For SPS-resident
            // entries the value is inferred to 0 per §7.4.8.
            let delta_idx_minus1 = if st_rps_idx == num_short_term_ref_pic_sets {
                br.ue()?
            } else {
                0
            };
            if delta_idx_minus1 >= st_rps_idx {
                return Err(SpsError::ValueOutOfRange {
                    field: "delta_idx_minus1",
                    got: delta_idx_minus1,
                });
            }
            let delta_rps_sign = br.u1()? != 0;
            let abs_delta_rps_minus1 = br.ue()?;
            if abs_delta_rps_minus1 > (1 << 15) - 1 {
                return Err(SpsError::ValueOutOfRange {
                    field: "abs_delta_rps_minus1",
                    got: abs_delta_rps_minus1,
                });
            }
            // RefRpsIdx = stRpsIdx − (delta_idx_minus1 + 1)
            let ref_rps_idx = (st_rps_idx as i64) - (delta_idx_minus1 as i64 + 1);
            let ref_rps = if ref_rps_idx >= 0 && (ref_rps_idx as usize) < all_rps.len() {
                Some(&all_rps[ref_rps_idx as usize])
            } else {
                // For SPS entries we expect ref_rps_idx in-range; the
                // only legal use of an out-of-range RefRpsIdx is when
                // st_rps_idx == num_short_term_ref_pic_sets, which is
                // the slice-header in-line case (handled elsewhere).
                prev
            };
            let num_delta_pocs = ref_rps.map(|r| r.num_delta_pocs()).unwrap_or(0);
            let entries = num_delta_pocs as usize + 1;
            let mut used_by_curr_pic_flag = Vec::with_capacity(entries);
            let mut use_delta_flag = Vec::with_capacity(entries);
            for _ in 0..entries {
                let used = br.u1()? != 0;
                used_by_curr_pic_flag.push(used);
                if !used {
                    let ud = br.u1()? != 0;
                    use_delta_flag.push(ud);
                } else {
                    // Per §7.4.8: when used_by_curr_pic_flag[j] is 1,
                    // use_delta_flag[j] is inferred to be 1.
                    use_delta_flag.push(true);
                }
            }
            Ok(Self {
                inter_ref_pic_set_prediction_flag,
                delta_idx_minus1,
                delta_rps_sign,
                abs_delta_rps_minus1,
                used_by_curr_pic_flag,
                use_delta_flag,
                num_negative_pics: 0,
                num_positive_pics: 0,
                delta_poc_s0_minus1: Vec::new(),
                used_by_curr_pic_s0_flag: Vec::new(),
                delta_poc_s1_minus1: Vec::new(),
                used_by_curr_pic_s1_flag: Vec::new(),
            })
        } else {
            let num_negative_pics = br.ue()?;
            if num_negative_pics > HEVC_MAX_RPS_PICS as u32 {
                return Err(SpsError::ValueOutOfRange {
                    field: "num_negative_pics",
                    got: num_negative_pics,
                });
            }
            let num_positive_pics = br.ue()?;
            if num_positive_pics > HEVC_MAX_RPS_PICS as u32 {
                return Err(SpsError::ValueOutOfRange {
                    field: "num_positive_pics",
                    got: num_positive_pics,
                });
            }
            let mut delta_poc_s0_minus1 = Vec::with_capacity(num_negative_pics as usize);
            let mut used_by_curr_pic_s0_flag = Vec::with_capacity(num_negative_pics as usize);
            for _ in 0..num_negative_pics {
                let dp = br.ue()?;
                if dp > (1 << 15) - 1 {
                    return Err(SpsError::ValueOutOfRange {
                        field: "delta_poc_s0_minus1",
                        got: dp,
                    });
                }
                delta_poc_s0_minus1.push(dp);
                used_by_curr_pic_s0_flag.push(br.u1()? != 0);
            }
            let mut delta_poc_s1_minus1 = Vec::with_capacity(num_positive_pics as usize);
            let mut used_by_curr_pic_s1_flag = Vec::with_capacity(num_positive_pics as usize);
            for _ in 0..num_positive_pics {
                let dp = br.ue()?;
                if dp > (1 << 15) - 1 {
                    return Err(SpsError::ValueOutOfRange {
                        field: "delta_poc_s1_minus1",
                        got: dp,
                    });
                }
                delta_poc_s1_minus1.push(dp);
                used_by_curr_pic_s1_flag.push(br.u1()? != 0);
            }
            Ok(Self {
                inter_ref_pic_set_prediction_flag,
                delta_idx_minus1: 0,
                delta_rps_sign: false,
                abs_delta_rps_minus1: 0,
                used_by_curr_pic_flag: Vec::new(),
                use_delta_flag: Vec::new(),
                num_negative_pics,
                num_positive_pics,
                delta_poc_s0_minus1,
                used_by_curr_pic_s0_flag,
                delta_poc_s1_minus1,
                used_by_curr_pic_s1_flag,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nal::{collect_nal_units, strip_emulation_prevention};

    /// Helper: convert a bit string (any non-`0`/`1` characters are
    /// ignored, useful for visual spacing) into a packed MSB-first
    /// byte vector with zero-padding up to the next byte boundary.
    fn bits_to_bytes(s: &str) -> Vec<u8> {
        let mut bits: Vec<u8> = Vec::new();
        for c in s.chars() {
            if c == '0' || c == '1' {
                bits.push((c as u8) - b'0');
            }
        }
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut out = Vec::with_capacity(bits.len() / 8);
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for &bit in chunk {
                b = (b << 1) | bit;
            }
            out.push(b);
        }
        out
    }

    /// Prefix of bits that get every SPS hand-assembled fixture through
    /// the structural header up to (but not including) the round-4 tail.
    /// `chroma_format_idc=1, width=16, height=16, conf_win=0, bit_depths=0,
    /// log2_max_poc_lsb_minus4=4, max_sub_layers_minus1=0, ordering_info
    /// present with single triple {dpb=0, reorder=0, latency=0},
    /// log2_min_cb=0, log2_diff=1, log2_min_tb=0, log2_diff_tb=2,
    /// max_transform_depth_*=0, scaling_list=0, amp=0, sao=1`.
    ///
    /// This is the EXACT same prefix the round-3 tests used; the round-4
    /// tail tests then concatenate the tail bits they want to exercise.
    fn synthesised_prefix_bits() -> String {
        let mut s = String::new();
        s += "0000"; // vps_id
        s += "000"; // max_sub_layers_minus1
        s += "1"; // nesting flag
                  // profile_tier_level(1, 0)
        s += "00"; // profile_space
        s += "0"; // tier
        s += "00001"; // profile_idc
        for _ in 0..32 {
            s += "0";
        }
        s += "0000";
        for _ in 0..43 {
            s += "0";
        }
        s += "0";
        s += "00011110"; // level=30
                         // sps_id ue(v)=0
        s += "1";
        // chroma_format_idc ue(v)=1 → '010'
        s += "010";
        // width ue(v)=16 → '000010001'
        s += "000010001";
        // height ue(v)=16
        s += "000010001";
        // conf_win = 0
        s += "0";
        // bd_luma=0, bd_chroma=0
        s += "1";
        s += "1";
        // log2_max_poc_lsb_minus4 = 4 → '00101'
        s += "00101";
        // ordering present = 1, single triple {0,0,0}
        s += "1";
        s += "1";
        s += "1";
        s += "1";
        // log2_min_cb_minus3 = 0
        s += "1";
        // log2_diff = 1 → '010'
        s += "010";
        // log2_min_tb_minus2 = 0
        s += "1";
        // log2_diff_tb = 2 → '011'
        s += "011";
        // max_transform_depth_{inter,intra} = 0
        s += "1";
        s += "1";
        // scaling_list_enabled = 0
        s += "0";
        // amp_enabled = 0
        s += "0";
        // sao_enabled = 1
        s += "1";
        s
    }

    /// SPS RBSP body extracted from
    /// `docs/video/h265/fixtures/tiny-i-only-16x16-main/input.hevc`,
    /// after the Annex B start code and the two-byte NAL header have
    /// been removed and emulation-prevention bytes stripped. The wire
    /// SPS (NAL idx 1, type 33) was 38 bytes including the two-byte
    /// header; after §7.4.1.1 strip (3 escape bytes removed) the body
    /// is 32 bytes.
    const TINY_SPS_RBSP: &[u8] = &[
        0x01, 0x04, 0x08, 0x00, 0x00, 0x00, 0x9F, 0xA8, 0x00, 0x00, 0x00, 0x00, 0x1E, 0xA0, 0x88,
        0x45, 0x96, 0xEA, 0xAF, 0x2B, 0xC0, 0x5A, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x32, 0x10,
    ];

    #[test]
    fn parses_tiny_fixture_sps() {
        let sps = SeqParameterSet::parse(TINY_SPS_RBSP).expect("SPS parse");
        // Trace cross-check (docs/video/h265/fixtures/tiny-i-only-16x16-main/trace.txt):
        //   SPS sps_id=0 vps_id=0 max_sub_layers=1 profile_idc=4 level_idc=30
        //       chroma_format_idc=1 bit_depth=8 bit_depth_chroma=8
        //       width=16 height=16 log2_ctb_size=4 log2_min_cb_size=3
        //       log2_min_tb_size=2 sao_enabled=1 amp_enabled=0 pcm_enabled=0
        //       scaling_list_enabled=0 long_term_ref_pics=0 temporal_mvp=1
        //       strong_intra_smoothing=1
        assert_eq!(sps.vps_id, 0);
        assert_eq!(sps.max_sub_layers_minus1, 0); // max_sub_layers = 1
        assert!(sps.temporal_id_nesting_flag);
        assert_eq!(sps.ptl.general_profile_idc, 4);
        assert_eq!(sps.ptl.general_level_idc, 30);
        assert_eq!(sps.sps_id, 0);
        assert_eq!(sps.chroma_format_idc, 1);
        assert!(!sps.separate_colour_plane_flag);
        assert_eq!(sps.pic_width_in_luma_samples, 16);
        assert_eq!(sps.pic_height_in_luma_samples, 16);
        assert!(!sps.conformance_window_flag);
        assert_eq!(sps.conformance_window, ConformanceWindow::default());
        assert_eq!(sps.bit_depth_luma_minus8, 0);
        assert_eq!(sps.bit_depth_chroma_minus8, 0);
        assert_eq!(sps.bit_depth_luma(), 8);
        assert_eq!(sps.bit_depth_chroma(), 8);
        assert_eq!(sps.log2_max_pic_order_cnt_lsb_minus4, 4); // MaxPicOrderCntLsb = 256
        assert_eq!(sps.max_pic_order_cnt_lsb(), 256);
        assert!(sps.sub_layer_ordering_info_present_flag);
        assert_eq!(
            sps.sub_layer_ordering_info[0].max_dec_pic_buffering_minus1,
            2
        );
        assert_eq!(sps.sub_layer_ordering_info[0].max_num_reorder_pics, 0);
        assert_eq!(sps.sub_layer_ordering_info[0].max_latency_increase_plus1, 1);
        assert_eq!(sps.log2_min_cb_size(), 3); // 8x8 minimum CU
        assert_eq!(sps.log2_ctb_size(), 4); // 16x16 CTU
        assert_eq!(sps.log2_min_tb_size(), 2); // 4x4 minimum TU
        assert_eq!(sps.max_transform_hierarchy_depth_inter, 0);
        assert_eq!(sps.max_transform_hierarchy_depth_intra, 0);
        assert!(!sps.scaling_list_enabled_flag);
        assert!(!sps.amp_enabled_flag);
        assert!(sps.sample_adaptive_offset_enabled_flag);
        // Round-4 tail.
        assert!(!sps.pcm_enabled_flag);
        assert!(sps.pcm.is_none());
        assert_eq!(sps.num_short_term_ref_pic_sets, 0);
        assert!(sps.short_term_ref_pic_sets.is_empty());
        assert!(!sps.long_term_ref_pics_present_flag);
        assert_eq!(sps.num_long_term_ref_pics_sps, 0);
        assert!(sps.long_term_ref_pics.is_empty());
        assert!(sps.sps_temporal_mvp_enabled_flag);
        assert!(sps.strong_intra_smoothing_enabled_flag);
        // The fixture's libx265 encode signals a VUI, so the parser
        // captures everything after `vui_parameters_present_flag` as an
        // opaque tail. We don't introspect its contents here — that is
        // the next round's work — but we do verify the capture happened
        // and starts mid-byte (the VUI body never lands on a clean byte
        // boundary in this fixture).
        assert!(sps.vui_parameters_present_flag);
        let tail = sps.opaque_tail.as_ref().expect("opaque VUI tail");
        assert!(!tail.bytes.is_empty());
    }

    /// End-to-end: pull the SPS NAL out of the raw Annex B stream
    /// through the walker, then parse it. The fixture's full byte
    /// sequence is captured inline so the test does not depend on the
    /// `docs/` tree at run time.
    #[test]
    fn parses_tiny_fixture_sps_via_nal_walker() {
        // VPS NAL + SPS NAL only — the rest of the fixture (PPS / SEI
        // / slice) is unrelated to this test.
        let raw = &[
            // VPS: 00 00 00 01 40 01 ...
            0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0C, 0x01, 0xFF, 0xFF, 0x04, 0x08, 0x00, 0x00,
            0x03, 0x00, 0x9F, 0xA8, 0x00, 0x00, 0x03, 0x00, 0x00, 0x1E, 0xBA, 0x02, 0x40,
            // SPS: 00 00 00 01 42 01 ...
            0x00, 0x00, 0x00, 0x01, 0x42, 0x01, 0x01, 0x04, 0x08, 0x00, 0x00, 0x03, 0x00, 0x9F,
            0xA8, 0x00, 0x00, 0x03, 0x00, 0x00, 0x1E, 0xA0, 0x88, 0x45, 0x96, 0xEA, 0xAF, 0x2B,
            0xC0, 0x5A, 0x02, 0x00, 0x00, 0x03, 0x00, 0x02, 0x00, 0x00, 0x03, 0x00, 0x32, 0x10,
        ];
        let units = collect_nal_units(raw).expect("walker");
        assert_eq!(units.len(), 2);
        assert_eq!(units[1].header.nal_unit_type, 33); // SPS_NUT
        let sps = SeqParameterSet::parse(&units[1].rbsp).expect("SPS parse");
        assert_eq!(sps.sps_id, 0);
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.pic_width_in_luma_samples, 16);
        assert_eq!(sps.pic_height_in_luma_samples, 16);
        assert_eq!(sps.log2_ctb_size(), 4);
        assert!(sps.sample_adaptive_offset_enabled_flag);
        assert!(!sps.pcm_enabled_flag);
        assert_eq!(sps.num_short_term_ref_pic_sets, 0);
        assert!(!sps.long_term_ref_pics_present_flag);
        assert!(sps.sps_temporal_mvp_enabled_flag);
        assert!(sps.strong_intra_smoothing_enabled_flag);
        assert!(sps.vui_parameters_present_flag);
    }

    #[test]
    fn strip_emulation_prevention_then_parse_matches_inline_decode() {
        // The wire SPS body (post NAL header, with the 3 emulation-
        // prevention escapes left in) must, after a §7.4.1.1 strip,
        // match TINY_SPS_RBSP exactly.
        let wire = &[
            0x01, 0x04, 0x08, 0x00, 0x00, 0x03, 0x00, 0x9F, 0xA8, 0x00, 0x00, 0x03, 0x00, 0x00,
            0x1E, 0xA0, 0x88, 0x45, 0x96, 0xEA, 0xAF, 0x2B, 0xC0, 0x5A, 0x02, 0x00, 0x00, 0x03,
            0x00, 0x02, 0x00, 0x00, 0x03, 0x00, 0x32, 0x10,
        ];
        let unesc = strip_emulation_prevention(wire);
        assert_eq!(unesc, TINY_SPS_RBSP);
        let a = SeqParameterSet::parse(&unesc).expect("SPS parse");
        let b = SeqParameterSet::parse(TINY_SPS_RBSP).expect("SPS parse");
        assert_eq!(a, b);
    }

    #[test]
    fn rejects_truncated_rbsp() {
        // Cut the buffer just past the leading `vps_id / max_sub /
        // nesting` byte — well before profile_tier_level finishes.
        let err = SeqParameterSet::parse(&TINY_SPS_RBSP[..3]).unwrap_err();
        assert_eq!(err, SpsError::Truncated);
    }

    /// Hand-assembled SPS exercising the `chroma_format_idc == 3`
    /// path (so `separate_colour_plane_flag` is signalled) plus the
    /// `conformance_window_flag == 1` four-`ue(v)` block. The remaining
    /// fields are kept at minimal values to make the bit string
    /// hand-traceable.
    #[test]
    fn parses_444_with_conformance_window() {
        let mut s = String::new();
        s += "0000"; // vps_id
        s += "000"; // max_sub_layers_minus1
        s += "1"; // nesting
                  // PTL(1,0)
        s += "00";
        s += "0";
        s += "00001";
        for _ in 0..32 {
            s += "0";
        }
        s += "0000";
        for _ in 0..43 {
            s += "0";
        }
        s += "0";
        s += "00011110";
        // sps_id=0
        s += "1";
        // chroma_format_idc=3 → '00100'
        s += "00100";
        // separate_colour_plane_flag=1
        s += "1";
        // width=16, height=16
        s += "000010001";
        s += "000010001";
        // conf_win=1, four ue(v)=0
        s += "1";
        s += "1";
        s += "1";
        s += "1";
        s += "1";
        // bd_luma=2 → '011'
        s += "011";
        s += "011";
        // log2_max_poc_lsb_minus4=4
        s += "00101";
        // ordering present, single triple of zeros
        s += "1";
        s += "1";
        s += "1";
        s += "1";
        // log2_min_cb=0
        s += "1";
        // log2_diff=1
        s += "010";
        s += "1";
        s += "011";
        s += "1";
        s += "1";
        // scaling=0
        s += "0";
        // amp=1
        s += "1";
        // sao=1
        s += "1";
        // pcm_enabled=0
        s += "0";
        // num_short_term_ref_pic_sets=0
        s += "1";
        // long_term_ref_pics_present=0
        s += "0";
        // temporal_mvp=1
        s += "1";
        // strong_intra_smoothing=0
        s += "0";
        // vui=0
        s += "0";
        // sps_extension_present=0
        s += "0";
        // rbsp trailing bits (stop bit then zero pad)
        s += "1";

        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert_eq!(sps.chroma_format_idc, 3);
        assert!(sps.separate_colour_plane_flag);
        assert_eq!(sps.bit_depth_luma(), 10);
        assert!(sps.amp_enabled_flag);
        assert!(sps.sample_adaptive_offset_enabled_flag);
        assert!(!sps.pcm_enabled_flag);
        assert_eq!(sps.num_short_term_ref_pic_sets, 0);
        assert!(!sps.long_term_ref_pics_present_flag);
        assert!(sps.sps_temporal_mvp_enabled_flag);
        assert!(!sps.strong_intra_smoothing_enabled_flag);
        assert!(!sps.vui_parameters_present_flag);
        assert!(!sps.sps_extension_present_flag);
        assert!(sps.opaque_tail.is_none());
    }

    /// Hand-assembled SPS with two sub-layers and the ordering-info
    /// present flag set to 0 — the [0] sub-layer must inherit the
    /// [1] triple per §7.4.3.2.1.
    #[test]
    fn ordering_info_present_flag_zero_propagates() {
        let mut s = String::new();
        s += "0000"; // vps_id
        s += "001"; // max_sub_layers_minus1 = 1
        s += "1";
        // PTL(1, 1):
        s += "00";
        s += "0";
        s += "00001";
        for _ in 0..32 {
            s += "0";
        }
        s += "0000";
        for _ in 0..43 {
            s += "0";
        }
        s += "0";
        s += "00011110"; // level=30
                         // sub_layer_profile_present[0]/level_present[0] = 0,0
        s += "00";
        for _ in 0..14 {
            s += "0";
        }
        // sps_id=0
        s += "1";
        // chroma=1
        s += "010";
        // width=16, height=16
        s += "000010001";
        s += "000010001";
        // conf_win=0
        s += "0";
        // bd_luma=0
        s += "1";
        // bd_chroma=0
        s += "1";
        // log2_max_poc_lsb_minus4=4
        s += "00101";
        // sub_layer_ordering_info_present_flag = 0
        s += "0";
        // single triple at i=max_sub_layers_minus1 (=1): dpb=2 ('011'), reorder=0 ('1'), latency=0 ('1')
        s += "011";
        s += "1";
        s += "1";
        s += "1";
        s += "010";
        s += "1";
        s += "011";
        s += "1";
        s += "1";
        s += "0";
        s += "0";
        s += "1";
        // pcm_enabled=0
        s += "0";
        // num_short_term_ref_pic_sets=0
        s += "1";
        // long_term=0
        s += "0";
        // temporal_mvp=1
        s += "1";
        // strong_intra_smoothing=1
        s += "1";
        // vui=0
        s += "0";
        // sps_extension_present=0
        s += "0";
        // stop bit
        s += "1";

        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert!(!sps.sub_layer_ordering_info_present_flag);
        assert_eq!(sps.max_sub_layers_minus1, 1);
        assert_eq!(
            sps.sub_layer_ordering_info[1].max_dec_pic_buffering_minus1,
            2
        );
        // [0] inherited from [1]
        assert_eq!(
            sps.sub_layer_ordering_info[0].max_dec_pic_buffering_minus1,
            2
        );
        assert!(sps.strong_intra_smoothing_enabled_flag);
    }

    /// SPS prefix bits identical to `synthesised_prefix_bits()` but
    /// stopping just *before* `scaling_list_enabled_flag` (i.e. after
    /// `max_transform_hierarchy_depth_intra`). Round 8 scaling-list
    /// tests append their own scaling_list block + the amp/sao bits +
    /// the SPS tail.
    fn synthesised_prefix_before_scaling_list() -> String {
        let mut s = String::new();
        s += "0000"; // vps_id
        s += "000"; // max_sub_layers_minus1
        s += "1"; // nesting flag
        s += "00"; // profile_space
        s += "0"; // tier
        s += "00001"; // profile_idc
        for _ in 0..32 {
            s += "0";
        }
        s += "0000";
        for _ in 0..43 {
            s += "0";
        }
        s += "0";
        s += "00011110"; // level=30
        s += "1"; // sps_id=0
        s += "010"; // chroma_format_idc=1
        s += "000010001"; // width=16
        s += "000010001"; // height=16
        s += "0"; // conf_win=0
        s += "1"; // bd_luma=0
        s += "1"; // bd_chroma=0
        s += "00101"; // log2_max_poc_lsb_minus4=4
        s += "1"; // ordering present=1
        s += "1"; // dpb=0
        s += "1"; // reorder=0
        s += "1"; // latency=0
        s += "1"; // log2_min_cb_minus3=0
        s += "010"; // log2_diff=1
        s += "1"; // log2_min_tb_minus2=0
        s += "011"; // log2_diff_tb=2
        s += "1"; // max_transform_depth_inter=0
        s += "1"; // max_transform_depth_intra=0
        s
    }

    /// SPS tail bits from `amp_enabled` through the stop bit, matching
    /// the round-4 minimal tail (sao=1, pcm=0, num_short_term_rps=0,
    /// long_term=0, temporal_mvp=1, strong_intra_smoothing=1, vui=0,
    /// extension=0, stop=1).
    fn synthesised_tail_after_scaling_list() -> String {
        let mut s = String::new();
        s += "0"; // amp_enabled=0
        s += "1"; // sao_enabled=1
        s += "0"; // pcm_enabled=0
        s += "1"; // num_short_term_ref_pic_sets ue=0
        s += "0"; // long_term_ref_pics=0
        s += "1"; // temporal_mvp=1
        s += "1"; // strong_intra_smoothing=1
        s += "0"; // vui=0
        s += "0"; // sps_extension_present=0
        s += "1"; // stop bit
        s
    }

    /// `scaling_list_enabled_flag == 1` with
    /// `sps_scaling_list_data_present_flag == 0`: per §7.4.5 the
    /// default scaling lists apply and the SPS parses end to end with
    /// no explicit [`ScalingListData`].
    #[test]
    fn scaling_list_enabled_default_lists() {
        let mut s = synthesised_prefix_before_scaling_list();
        s += "1"; // scaling_list_enabled = 1
        s += "0"; // sps_scaling_list_data_present_flag = 0
        s += &synthesised_tail_after_scaling_list();
        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert!(sps.scaling_list_enabled_flag);
        assert!(!sps.sps_scaling_list_data_present_flag);
        assert!(sps.scaling_list_data.is_none());
    }

    /// `scaling_list_enabled_flag == 1` with
    /// `sps_scaling_list_data_present_flag == 1`: the SPS carries an
    /// explicit `scaling_list_data()` (§7.3.4). Here every one of the
    /// 24 slots signals pred_mode=0, delta=0 (use the default list), so
    /// the parsed lists equal the §7.4.5 default tables.
    #[test]
    fn scaling_list_enabled_explicit_all_default() {
        let mut s = synthesised_prefix_before_scaling_list();
        s += "1"; // scaling_list_enabled = 1
        s += "1"; // sps_scaling_list_data_present_flag = 1
                  // scaling_list_data(): 24 slots, each pred_mode=0
                  // ('0') + scaling_list_pred_matrix_id_delta ue=0 ('1').
                  // sizeId 0/1/2 each 6 slots; sizeId 3 only 2 slots.
        for size_id in 0..4 {
            let step = if size_id == 3 { 3 } else { 1 };
            let mut m = 0;
            while m < 6 {
                s += "0"; // scaling_list_pred_mode_flag = 0
                s += "1"; // scaling_list_pred_matrix_id_delta ue = 0
                m += step;
            }
        }
        s += &synthesised_tail_after_scaling_list();
        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert!(sps.scaling_list_enabled_flag);
        assert!(sps.sps_scaling_list_data_present_flag);
        let data = sps.scaling_list_data.expect("scaling_list_data present");
        // 4x4 default = all 16.
        assert_eq!(data.lists[0][0].coef, vec![16u16; 16]);
        // 8x8 inter default for matrixId 3.
        assert_eq!(data.lists[1][3].coef[63], 91);
    }

    /// `chroma_format_idc` is u-Exp-Golomb so on-wire codeNum 4 would
    /// decode to a value of 4 — outside the legal 0..=3 range. The
    /// parser must reject it.
    #[test]
    fn rejects_chroma_format_idc_out_of_range() {
        let mut s = String::new();
        s += "0000";
        s += "000";
        s += "1";
        s += "00";
        s += "0";
        s += "00001";
        for _ in 0..32 {
            s += "0";
        }
        s += "0000";
        for _ in 0..43 {
            s += "0";
        }
        s += "0";
        s += "00011110";
        // sps_id=0
        s += "1";
        // chroma_format_idc ue=4 → '00101'
        s += "00101";
        let bytes = bits_to_bytes(&s);
        let err = SeqParameterSet::parse(&bytes).unwrap_err();
        assert_eq!(
            err,
            SpsError::ValueOutOfRange {
                field: "chroma_format_idc",
                got: 4
            }
        );
    }

    /// Hand-assembled SPS exercising the `pcm_enabled_flag == 1`
    /// branch (§7.3.2.2 PCM block).
    #[test]
    fn parses_pcm_enabled() {
        let mut s = synthesised_prefix_bits();
        // pcm_enabled_flag = 1
        s += "1";
        // pcm_sample_bit_depth_luma_minus1 = 7 (PcmBitDepthY = 8)
        s += "0111";
        // pcm_sample_bit_depth_chroma_minus1 = 7
        s += "0111";
        // log2_min_pcm_luma_coding_block_size_minus3 = 0 → '1'
        s += "1";
        // log2_diff_max_min_pcm_luma_coding_block_size = 0 → '1'
        s += "1";
        // pcm_loop_filter_disabled_flag = 1
        s += "1";
        // num_short_term_ref_pic_sets = 0
        s += "1";
        // long_term_ref_pics_present_flag = 0
        s += "0";
        // sps_temporal_mvp_enabled_flag = 1
        s += "1";
        // strong_intra_smoothing_enabled_flag = 1
        s += "1";
        // vui_parameters_present_flag = 0
        s += "0";
        // sps_extension_present_flag = 0
        s += "0";
        // stop bit
        s += "1";
        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert!(sps.pcm_enabled_flag);
        let pcm = sps.pcm.expect("pcm block");
        assert_eq!(pcm.bit_depth_luma_minus1, 7);
        assert_eq!(pcm.bit_depth_chroma_minus1, 7);
        assert_eq!(pcm.log2_min_pcm_luma_coding_block_size_minus3, 0);
        assert_eq!(pcm.log2_diff_max_min_pcm_luma_coding_block_size, 0);
        assert!(pcm.loop_filter_disabled_flag);
        assert!(sps.sps_temporal_mvp_enabled_flag);
        assert!(sps.strong_intra_smoothing_enabled_flag);
    }

    /// PCM bit depth above luma bit depth must be rejected per
    /// §7.4.3.2 / equation (7-25).
    #[test]
    fn rejects_pcm_bit_depth_exceeding_luma() {
        let mut s = synthesised_prefix_bits();
        s += "1"; // pcm_enabled_flag
                  // pcm_sample_bit_depth_luma_minus1 = 15 (PcmBitDepthY = 16)
                  // BitDepthY in the synthesised prefix is 8.
        s += "1111";
        let bytes = bits_to_bytes(&s);
        let err = SeqParameterSet::parse(&bytes).unwrap_err();
        assert_eq!(
            err,
            SpsError::ValueOutOfRange {
                field: "pcm_sample_bit_depth_luma_minus1",
                got: 15
            }
        );
    }

    /// Hand-assembled SPS with one explicit short-term RPS (no
    /// inter-RPS-prediction): 1 negative pic, 0 positive pics.
    #[test]
    fn parses_one_short_term_rps_explicit() {
        let mut s = synthesised_prefix_bits();
        // pcm_enabled_flag = 0
        s += "0";
        // num_short_term_ref_pic_sets = 1 → ue(v) codeNum 1 → '010'
        s += "010";
        // st_ref_pic_set(0): inter_ref_pic_set_prediction_flag is NOT
        // signalled (st_rps_idx == 0); implicit 0.
        //   num_negative_pics = 1 → '010'
        s += "010";
        //   num_positive_pics = 0 → '1'
        s += "1";
        //   delta_poc_s0_minus1[0] = 0 → '1'
        s += "1";
        //   used_by_curr_pic_s0_flag[0] = 1
        s += "1";
        // long_term_ref_pics_present_flag = 0
        s += "0";
        // sps_temporal_mvp_enabled_flag = 1
        s += "1";
        // strong_intra_smoothing_enabled_flag = 0
        s += "0";
        // vui_parameters_present_flag = 0
        s += "0";
        // sps_extension_present_flag = 0
        s += "0";
        s += "1"; // stop bit

        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert_eq!(sps.num_short_term_ref_pic_sets, 1);
        let rps = &sps.short_term_ref_pic_sets[0];
        assert!(!rps.inter_ref_pic_set_prediction_flag);
        assert_eq!(rps.num_negative_pics, 1);
        assert_eq!(rps.num_positive_pics, 0);
        assert_eq!(rps.delta_poc_s0_minus1, vec![0]);
        assert_eq!(rps.used_by_curr_pic_s0_flag, vec![true]);
        assert!(rps.delta_poc_s1_minus1.is_empty());
        assert_eq!(rps.num_delta_pocs(), 1);
    }

    /// Hand-assembled SPS with two short-term RPSes, the second one
    /// using inter-RPS-prediction relative to the first. Exercises
    /// the `for( j = 0; j <= NumDeltaPocs[RefRpsIdx]; j++ )` loop in
    /// §7.3.7 with the `use_delta_flag` inference of §7.4.8.
    #[test]
    fn parses_inter_rps_prediction() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm_enabled_flag
                  // num_short_term_ref_pic_sets = 2 → codeNum 2 → '011'
        s += "011";
        // st_ref_pic_set(0): explicit, num_negative_pics=1, num_positive=0,
        // delta_poc_s0_minus1[0]=0, used_by_curr_pic_s0_flag[0]=1
        s += "010"; // num_neg=1
        s += "1"; // num_pos=0
        s += "1"; // dp0=0
        s += "1"; // used=1
                  // st_ref_pic_set(1): inter_ref_pic_set_prediction_flag = 1
        s += "1";
        //   delta_idx_minus1 is NOT signalled (st_rps_idx=1, num=2; only
        //   signalled when st_rps_idx == num). Inferred to 0 →
        //   RefRpsIdx = 1 - (0+1) = 0.
        //   delta_rps_sign = 0
        s += "0";
        //   abs_delta_rps_minus1 = 0 → codeNum 0 → '1'
        s += "1";
        //   NumDeltaPocs[0] = 1, so loop runs j=0..1 (2 entries):
        //     j=0: used_by_curr_pic_flag = 1  → use_delta_flag inferred 1
        s += "1";
        //     j=1: used_by_curr_pic_flag = 0, use_delta_flag = 1
        s += "0";
        s += "1";
        // long_term=0, temporal_mvp=1, strong_intra_smoothing=1, vui=0, ext=0, stop
        s += "0";
        s += "1";
        s += "1";
        s += "0";
        s += "0";
        s += "1";

        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert_eq!(sps.num_short_term_ref_pic_sets, 2);
        let r1 = &sps.short_term_ref_pic_sets[1];
        assert!(r1.inter_ref_pic_set_prediction_flag);
        assert_eq!(r1.delta_idx_minus1, 0);
        assert!(!r1.delta_rps_sign);
        assert_eq!(r1.abs_delta_rps_minus1, 0);
        assert_eq!(r1.used_by_curr_pic_flag, vec![true, false]);
        assert_eq!(r1.use_delta_flag, vec![true, true]);
    }

    /// Hand-assembled SPS exercising the long-term-ref-pic block.
    #[test]
    fn parses_long_term_ref_pics() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm_enabled_flag = 0
        s += "1"; // num_short_term_ref_pic_sets = 0
        s += "1"; // long_term_ref_pics_present_flag = 1
                  // num_long_term_ref_pics_sps = 2 → codeNum 2 → '011'
        s += "011";
        // log2_max_pic_order_cnt_lsb_minus4 = 4 in the synthesised
        // prefix, so lt_ref_pic_poc_lsb_sps[i] is 8 bits wide.
        //   i=0: poc_lsb = 0x10, used_by_curr_pic_lt_sps_flag = 1
        s += "00010000";
        s += "1";
        //   i=1: poc_lsb = 0x20, used = 0
        s += "00100000";
        s += "0";
        // temporal_mvp=1, strong_intra=0, vui=0, ext=0, stop
        s += "1";
        s += "0";
        s += "0";
        s += "0";
        s += "1";

        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert!(sps.long_term_ref_pics_present_flag);
        assert_eq!(sps.num_long_term_ref_pics_sps, 2);
        assert_eq!(sps.long_term_ref_pics.len(), 2);
        assert_eq!(sps.long_term_ref_pics[0].poc_lsb, 0x10);
        assert!(sps.long_term_ref_pics[0].used_by_curr_pic);
        assert_eq!(sps.long_term_ref_pics[1].poc_lsb, 0x20);
        assert!(!sps.long_term_ref_pics[1].used_by_curr_pic);
    }

    /// SPS with `vui_parameters_present_flag == 1`: the parser must
    /// surface everything after the VUI gate as an opaque tail.
    #[test]
    fn captures_vui_opaque_tail() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm_enabled_flag = 0
        s += "1"; // num_short_term_ref_pic_sets = 0
        s += "0"; // long_term = 0
        s += "1"; // temporal_mvp = 1
        s += "1"; // strong_intra_smoothing = 1
        s += "1"; // vui_parameters_present_flag = 1
                  // synthetic opaque body (8 bits) + stop bit + pad
        s += "10101010";
        s += "1"; // trailing stop bit (treated as part of opaque tail)

        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert!(sps.vui_parameters_present_flag);
        assert!(!sps.sps_extension_present_flag);
        let tail = sps.opaque_tail.expect("opaque VUI tail");
        assert!(!tail.bytes.is_empty());
        // start_bit_in_first_byte is within [0, 7]; we don't pin its
        // exact value here because the synthesised prefix varies in
        // length across rounds and the test exists to confirm the tail
        // was captured at all.
        assert!(tail.start_bit_in_first_byte < 8);
    }

    /// SPS with `sps_extension_present_flag == 1`: parser must surface
    /// the extension body + trailer as opaque.
    #[test]
    fn captures_extension_opaque_tail() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm
        s += "1"; // num_short_term=0
        s += "0"; // long_term=0
        s += "1"; // temporal_mvp
        s += "1"; // strong_intra_smoothing
        s += "0"; // vui=0
        s += "1"; // sps_extension_present_flag = 1
                  // some opaque extension bytes + stop bit
        s += "00000000";
        s += "1";
        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert!(!sps.vui_parameters_present_flag);
        assert!(sps.sps_extension_present_flag);
        let tail = sps.opaque_tail.expect("opaque extension tail");
        assert!(!tail.bytes.is_empty());
    }

    /// SPS with both `vui` and `extension` flags off — no opaque
    /// tail is captured (only the rbsp_trailing_bits remain).
    #[test]
    fn no_opaque_tail_when_flags_clear() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm
        s += "1"; // num_short=0
        s += "0"; // long_term=0
        s += "1"; // temporal_mvp
        s += "1"; // strong_intra_smoothing
        s += "0"; // vui=0
        s += "0"; // sps_extension_present=0
        s += "1"; // stop bit
        let bytes = bits_to_bytes(&s);
        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert!(!sps.vui_parameters_present_flag);
        assert!(!sps.sps_extension_present_flag);
        assert!(sps.opaque_tail.is_none());
    }

    /// `num_short_term_ref_pic_sets > 64` is illegal per §7.4.3.2.
    #[test]
    fn rejects_too_many_short_term_rps() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm
                  // num_short_term_ref_pic_sets = 65 → codeNum 65
                  // 65 in 0-th order Exp-Golomb: leadingZeroBits=6,
                  //   suffix = 65 - (2^6 - 1) = 2 = '000010'
                  // so the bit string is '000000 1 000010' = 13 bits.
        s += "0000001000010";
        let bytes = bits_to_bytes(&s);
        let err = SeqParameterSet::parse(&bytes).unwrap_err();
        assert!(matches!(
            err,
            SpsError::ValueOutOfRange {
                field: "num_short_term_ref_pic_sets",
                ..
            }
        ));
    }

    /// `num_long_term_ref_pics_sps > 32` is illegal per §7.4.3.2.
    #[test]
    fn rejects_too_many_long_term_rps() {
        let mut s = synthesised_prefix_bits();
        s += "0"; // pcm
        s += "1"; // num_short_term=0
        s += "1"; // long_term_ref_pics_present
                  // num_long_term_ref_pics_sps = 33: codeNum 33
                  // leadingZeroBits=5, suffix = 33 - (2^5 - 1) = 2 = '00010'
                  // bit string: '00000 1 00010' = 11 bits.
        s += "00000100010";
        let bytes = bits_to_bytes(&s);
        let err = SeqParameterSet::parse(&bytes).unwrap_err();
        assert!(matches!(
            err,
            SpsError::ValueOutOfRange {
                field: "num_long_term_ref_pics_sps",
                ..
            }
        ));
    }
}
