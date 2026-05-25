//! Decoder for the `vui_parameters( )` syntax structure (§E.2.1).
//!
//! The Video Usability Information block is the optional tail of a
//! Sequence Parameter Set, signalled by `vui_parameters_present_flag`
//! (§7.3.2.2). It carries display-oriented metadata that does not
//! affect the decoding process itself: sample aspect ratio, overscan
//! preference, the video-signal / colour-description block, chroma
//! sample location, the field/frame indications, the default display
//! window, the timing (clock-tick) information with an optional nested
//! `hrd_parameters( )` block (§E.2.2, decoded by [`crate::hrd`]), and
//! the bitstream-restriction parameters.
//!
//! The §E.2.1 syntax table reproduced for reference:
//!
//! ```text
//! vui_parameters( ) {                                     Descriptor
//!   aspect_ratio_info_present_flag                            u(1)
//!   if( aspect_ratio_info_present_flag ) {
//!     aspect_ratio_idc                                       u(8)
//!     if( aspect_ratio_idc == EXTENDED_SAR ) {
//!       sar_width                                           u(16)
//!       sar_height                                          u(16)
//!     }
//!   }
//!   overscan_info_present_flag                                u(1)
//!   if( overscan_info_present_flag )
//!     overscan_appropriate_flag                               u(1)
//!   video_signal_type_present_flag                            u(1)
//!   if( video_signal_type_present_flag ) {
//!     video_format                                            u(3)
//!     video_full_range_flag                                   u(1)
//!     colour_description_present_flag                         u(1)
//!     if( colour_description_present_flag ) {
//!       colour_primaries                                      u(8)
//!       transfer_characteristics                              u(8)
//!       matrix_coeffs                                         u(8)
//!     }
//!   }
//!   chroma_loc_info_present_flag                              u(1)
//!   if( chroma_loc_info_present_flag ) {
//!     chroma_sample_loc_type_top_field                       ue(v)
//!     chroma_sample_loc_type_bottom_field                    ue(v)
//!   }
//!   neutral_chroma_indication_flag                            u(1)
//!   field_seq_flag                                            u(1)
//!   frame_field_info_present_flag                             u(1)
//!   default_display_window_flag                               u(1)
//!   if( default_display_window_flag ) {
//!     def_disp_win_left_offset                               ue(v)
//!     def_disp_win_right_offset                              ue(v)
//!     def_disp_win_top_offset                                ue(v)
//!     def_disp_win_bottom_offset                             ue(v)
//!   }
//!   vui_timing_info_present_flag                              u(1)
//!   if( vui_timing_info_present_flag ) {
//!     vui_num_units_in_tick                                  u(32)
//!     vui_time_scale                                         u(32)
//!     vui_poc_proportional_to_timing_flag                     u(1)
//!     if( vui_poc_proportional_to_timing_flag )
//!       vui_num_ticks_poc_diff_one_minus1                    ue(v)
//!     vui_hrd_parameters_present_flag                         u(1)
//!     if( vui_hrd_parameters_present_flag )
//!       hrd_parameters( 1, sps_max_sub_layers_minus1 )
//!   }
//!   bitstream_restriction_flag                                u(1)
//!   if( bitstream_restriction_flag ) {
//!     tiles_fixed_structure_flag                              u(1)
//!     motion_vectors_over_pic_boundaries_flag                 u(1)
//!     restricted_ref_pic_lists_flag                           u(1)
//!     min_spatial_segmentation_idc                           ue(v)
//!     max_bytes_per_pic_denom                                ue(v)
//!     max_bits_per_min_cu_denom                              ue(v)
//!     log2_max_mv_length_horizontal                          ue(v)
//!     log2_max_mv_length_vertical                            ue(v)
//!   }
//! }
//! ```
//!
//! Range / inference rules enforced (§E.2.1 semantics):
//! * `aspect_ratio_idc` 17..=254 reserved → no decode-time error (the
//!   spec says decoders interpret these as 0); the raw value is kept.
//! * `chroma_sample_loc_type_{top,bottom}_field` range 0..=5; inferred 0.
//! * `video_format` inferred 5, `video_full_range_flag` inferred 0,
//!   `colour_primaries` / `transfer_characteristics` / `matrix_coeffs`
//!   inferred 2 when their gates are clear.
//! * `vui_num_units_in_tick` / `vui_time_scale` must be `> 0` (§E.2.1).
//! * `vui_num_ticks_poc_diff_one_minus1` range 0..=2^32 − 2.
//! * `min_spatial_segmentation_idc` range 0..=4095.
//! * `max_bytes_per_pic_denom` / `max_bits_per_min_cu_denom` range 0..=16.
//! * `log2_max_mv_length_{horizontal,vertical}` range 0..=15; inferred 15.

use crate::bitreader::{BitReader, BitReaderError};
use crate::hrd::{HrdError, HrdParameters};

/// `aspect_ratio_idc` value signalling that the sample aspect ratio is
/// carried explicitly by `sar_width` / `sar_height` (§E.2.1, Table E.1).
pub const EXTENDED_SAR: u8 = 255;

/// Upper bound (inclusive) on `chroma_sample_loc_type_{top,bottom}_field`
/// per §E.2.1.
pub const HEVC_MAX_CHROMA_SAMPLE_LOC_TYPE: u32 = 5;

/// Upper bound (inclusive) on `min_spatial_segmentation_idc` per §E.2.1.
pub const HEVC_MAX_MIN_SPATIAL_SEGMENTATION_IDC: u32 = 4095;

/// Upper bound (inclusive) on `max_bytes_per_pic_denom` and
/// `max_bits_per_min_cu_denom` per §E.2.1.
pub const HEVC_MAX_DENOM: u32 = 16;

/// Upper bound (inclusive) on `log2_max_mv_length_{horizontal,vertical}`
/// per §E.2.1.
pub const HEVC_MAX_LOG2_MV_LENGTH: u32 = 15;

/// Errors that can arise while decoding a `vui_parameters( )` body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VuiError {
    /// The RBSP ran out of bits before the VUI body was fully parsed.
    Truncated,
    /// A syntax element's parsed value was outside its §E.2.1 range.
    ValueOutOfRange {
        /// Name of the offending syntax element.
        field: &'static str,
        /// The (illegal) value as a `u64` (32-bit elements re-cast at
        /// the call site).
        got: u64,
    },
    /// The nested `hrd_parameters( )` parse (§E.2.2) failed.
    Hrd(HrdError),
    /// An unexpected bitstream-level error surfaced from the reader.
    Bitstream(BitReaderError),
}

impl core::fmt::Display for VuiError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => f.write_str("vui_parameters() RBSP truncated"),
            Self::ValueOutOfRange { field, got } => {
                write!(f, "VUI syntax element {field} out of range: {got}")
            }
            Self::Hrd(e) => write!(f, "hrd_parameters() error during VUI parse: {e}"),
            Self::Bitstream(e) => write!(f, "bitstream error during VUI parse: {e}"),
        }
    }
}

impl std::error::Error for VuiError {}

impl From<BitReaderError> for VuiError {
    fn from(e: BitReaderError) -> Self {
        match e {
            BitReaderError::EndOfBuffer => Self::Truncated,
            other => Self::Bitstream(other),
        }
    }
}

impl From<HrdError> for VuiError {
    fn from(e: HrdError) -> Self {
        Self::Hrd(e)
    }
}

/// Sample-aspect-ratio block (§E.2.1, gated by
/// `aspect_ratio_info_present_flag`). When `aspect_ratio_idc` is
/// [`EXTENDED_SAR`], `sar_width` / `sar_height` carry the ratio
/// explicitly; otherwise the ratio is given by Table E.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AspectRatioInfo {
    /// `aspect_ratio_idc` (`u(8)`). The raw value is preserved even in
    /// the reserved 17..=254 range; per §E.2.1 decoders treat those as 0.
    pub aspect_ratio_idc: u8,
    /// `sar_width` (`u(16)`); present only when
    /// `aspect_ratio_idc == EXTENDED_SAR`.
    pub sar_width: u16,
    /// `sar_height` (`u(16)`); present only when
    /// `aspect_ratio_idc == EXTENDED_SAR`.
    pub sar_height: u16,
}

/// Video-signal-type block (§E.2.1, gated by
/// `video_signal_type_present_flag`). The colour-description triple is
/// nested under its own present flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoSignalType {
    /// `video_format` (`u(3)`). Inferred to 5 ("unspecified") when the
    /// outer present flag is clear.
    pub video_format: u8,
    /// `video_full_range_flag` (`u(1)`). Inferred to false when absent.
    pub video_full_range_flag: bool,
    /// `colour_description_present_flag` (`u(1)`).
    pub colour_description_present_flag: bool,
    /// `colour_primaries` (`u(8)`). Inferred to 2 when the colour
    /// description is absent.
    pub colour_primaries: u8,
    /// `transfer_characteristics` (`u(8)`). Inferred to 2 when absent.
    pub transfer_characteristics: u8,
    /// `matrix_coeffs` (`u(8)`). Inferred to 2 when absent.
    pub matrix_coeffs: u8,
}

impl Default for VideoSignalType {
    fn default() -> Self {
        // §E.2.1 inference defaults when video_signal_type_present_flag
        // is 0: video_format = 5, video_full_range_flag = 0, and the
        // colour-description triple defaults to 2 ("unspecified").
        Self {
            video_format: 5,
            video_full_range_flag: false,
            colour_description_present_flag: false,
            colour_primaries: 2,
            transfer_characteristics: 2,
            matrix_coeffs: 2,
        }
    }
}

/// Chroma sample location block (§E.2.1, gated by
/// `chroma_loc_info_present_flag`). Both fields are inferred to 0 when
/// absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChromaLocInfo {
    /// `chroma_sample_loc_type_top_field` (`ue(v)`, range 0..=5).
    pub chroma_sample_loc_type_top_field: u32,
    /// `chroma_sample_loc_type_bottom_field` (`ue(v)`, range 0..=5).
    pub chroma_sample_loc_type_bottom_field: u32,
}

/// Default display window block (§E.2.1, gated by
/// `default_display_window_flag`). Offsets are in §6.5.5
/// SubWidthC / SubHeightC units (added to the conformance-window
/// offsets per equations E-68..E-71). All zero when absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DefaultDisplayWindow {
    /// `def_disp_win_left_offset` (`ue(v)`).
    pub left_offset: u32,
    /// `def_disp_win_right_offset` (`ue(v)`).
    pub right_offset: u32,
    /// `def_disp_win_top_offset` (`ue(v)`).
    pub top_offset: u32,
    /// `def_disp_win_bottom_offset` (`ue(v)`).
    pub bottom_offset: u32,
}

/// VUI timing-information block (§E.2.1, gated by
/// `vui_timing_info_present_flag`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VuiTimingInfo {
    /// `vui_num_units_in_tick` (`u(32)`, must be `> 0`).
    pub num_units_in_tick: u32,
    /// `vui_time_scale` (`u(32)`, must be `> 0`).
    pub time_scale: u32,
    /// `vui_poc_proportional_to_timing_flag` (`u(1)`).
    pub poc_proportional_to_timing_flag: bool,
    /// `vui_num_ticks_poc_diff_one_minus1` (`ue(v)`, range 0..=2^32 − 2);
    /// present only when `poc_proportional_to_timing_flag` is set.
    pub num_ticks_poc_diff_one_minus1: Option<u32>,
    /// `vui_hrd_parameters_present_flag` (`u(1)`).
    pub hrd_parameters_present_flag: bool,
    /// The nested `hrd_parameters( 1, sps_max_sub_layers_minus1 )` block
    /// (§E.2.2); present only when `hrd_parameters_present_flag` is set.
    pub hrd_parameters: Option<HrdParameters>,
}

/// Bitstream-restriction block (§E.2.1, gated by
/// `bitstream_restriction_flag`). Each field carries its §E.2.1
/// inferred default when the outer flag is clear (see
/// [`BitstreamRestriction::inferred_absent`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitstreamRestriction {
    /// `tiles_fixed_structure_flag` (`u(1)`). Inferred false when absent.
    pub tiles_fixed_structure_flag: bool,
    /// `motion_vectors_over_pic_boundaries_flag` (`u(1)`). Inferred
    /// **true** when absent.
    pub motion_vectors_over_pic_boundaries_flag: bool,
    /// `restricted_ref_pic_lists_flag` (`u(1)`).
    pub restricted_ref_pic_lists_flag: bool,
    /// `min_spatial_segmentation_idc` (`ue(v)`, range 0..=4095).
    /// Inferred 0 when absent.
    pub min_spatial_segmentation_idc: u32,
    /// `max_bytes_per_pic_denom` (`ue(v)`, range 0..=16).
    pub max_bytes_per_pic_denom: u32,
    /// `max_bits_per_min_cu_denom` (`ue(v)`, range 0..=16). Inferred
    /// **1** when absent.
    pub max_bits_per_min_cu_denom: u32,
    /// `log2_max_mv_length_horizontal` (`ue(v)`, range 0..=15). Inferred
    /// **15** when absent.
    pub log2_max_mv_length_horizontal: u32,
    /// `log2_max_mv_length_vertical` (`ue(v)`, range 0..=15). Inferred
    /// **15** when absent.
    pub log2_max_mv_length_vertical: u32,
}

impl BitstreamRestriction {
    /// The §E.2.1 inferred values applied when
    /// `bitstream_restriction_flag == 0`: `tiles_fixed_structure_flag`
    /// 0, `motion_vectors_over_pic_boundaries_flag` 1,
    /// `min_spatial_segmentation_idc` 0, `max_bits_per_min_cu_denom` 1,
    /// and the two `log2_max_mv_length_*` 15.
    pub fn inferred_absent() -> Self {
        Self {
            tiles_fixed_structure_flag: false,
            motion_vectors_over_pic_boundaries_flag: true,
            restricted_ref_pic_lists_flag: false,
            min_spatial_segmentation_idc: 0,
            max_bytes_per_pic_denom: 0,
            max_bits_per_min_cu_denom: 1,
            log2_max_mv_length_horizontal: 15,
            log2_max_mv_length_vertical: 15,
        }
    }
}

impl Default for BitstreamRestriction {
    fn default() -> Self {
        Self::inferred_absent()
    }
}

/// Parsed `vui_parameters( )` body per §E.2.1.
///
/// The optional sub-blocks are `Some` exactly when their corresponding
/// present flag was set. Fields that have a meaningful §E.2.1 inferred
/// default when their sub-block is absent are surfaced through the
/// sub-block's `Default` impl ([`VideoSignalType`],
/// [`BitstreamRestriction`]).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VuiParameters {
    /// `aspect_ratio_info_present_flag` (`u(1)`).
    pub aspect_ratio_info_present_flag: bool,
    /// The sample-aspect-ratio block; `Some` iff the present flag is set.
    pub aspect_ratio_info: Option<AspectRatioInfo>,
    /// `overscan_info_present_flag` (`u(1)`).
    pub overscan_info_present_flag: bool,
    /// `overscan_appropriate_flag` (`u(1)`); `Some` iff
    /// `overscan_info_present_flag` is set.
    pub overscan_appropriate_flag: Option<bool>,
    /// `video_signal_type_present_flag` (`u(1)`).
    pub video_signal_type_present_flag: bool,
    /// The video-signal-type block; `Some` iff the present flag is set.
    /// When `None`, the §E.2.1 inferred defaults apply (see
    /// [`VideoSignalType::default`]).
    pub video_signal_type: Option<VideoSignalType>,
    /// `chroma_loc_info_present_flag` (`u(1)`).
    pub chroma_loc_info_present_flag: bool,
    /// The chroma sample location block; `Some` iff the present flag is
    /// set (otherwise both fields are inferred to 0).
    pub chroma_loc_info: Option<ChromaLocInfo>,
    /// `neutral_chroma_indication_flag` (`u(1)`).
    pub neutral_chroma_indication_flag: bool,
    /// `field_seq_flag` (`u(1)`).
    pub field_seq_flag: bool,
    /// `frame_field_info_present_flag` (`u(1)`).
    pub frame_field_info_present_flag: bool,
    /// `default_display_window_flag` (`u(1)`).
    pub default_display_window_flag: bool,
    /// The default display window block; `Some` iff the present flag is
    /// set (otherwise all four offsets are inferred to 0).
    pub default_display_window: Option<DefaultDisplayWindow>,
    /// `vui_timing_info_present_flag` (`u(1)`).
    pub vui_timing_info_present_flag: bool,
    /// The timing-information block; `Some` iff the present flag is set.
    pub vui_timing_info: Option<VuiTimingInfo>,
    /// `bitstream_restriction_flag` (`u(1)`).
    pub bitstream_restriction_flag: bool,
    /// The bitstream-restriction block; `Some` iff the present flag is
    /// set (otherwise [`BitstreamRestriction::inferred_absent`] applies).
    pub bitstream_restriction: Option<BitstreamRestriction>,
}

impl VuiParameters {
    /// Parse a `vui_parameters( )` body starting at the current `br`
    /// position. `sps_max_sub_layers_minus1` is the SPS-level value
    /// passed to the nested `hrd_parameters( 1, sps_max_sub_layers_minus1 )`
    /// call (§E.2.1); it is only consulted when
    /// `vui_hrd_parameters_present_flag == 1`.
    pub fn parse(br: &mut BitReader<'_>, sps_max_sub_layers_minus1: u8) -> Result<Self, VuiError> {
        // --- aspect_ratio_info ---
        let mut vui = VuiParameters {
            aspect_ratio_info_present_flag: br.u1()? != 0,
            ..Default::default()
        };
        if vui.aspect_ratio_info_present_flag {
            let aspect_ratio_idc = br.u(8)? as u8;
            let mut info = AspectRatioInfo {
                aspect_ratio_idc,
                ..AspectRatioInfo::default()
            };
            if aspect_ratio_idc == EXTENDED_SAR {
                info.sar_width = br.u(16)? as u16;
                info.sar_height = br.u(16)? as u16;
            }
            vui.aspect_ratio_info = Some(info);
        }

        // --- overscan_info ---
        vui.overscan_info_present_flag = br.u1()? != 0;
        if vui.overscan_info_present_flag {
            vui.overscan_appropriate_flag = Some(br.u1()? != 0);
        }

        // --- video_signal_type (+ colour_description) ---
        vui.video_signal_type_present_flag = br.u1()? != 0;
        if vui.video_signal_type_present_flag {
            let video_format = br.u(3)? as u8;
            let video_full_range_flag = br.u1()? != 0;
            let colour_description_present_flag = br.u1()? != 0;
            let mut vst = VideoSignalType {
                video_format,
                video_full_range_flag,
                colour_description_present_flag,
                // Defaults until the colour-description triple is read.
                colour_primaries: 2,
                transfer_characteristics: 2,
                matrix_coeffs: 2,
            };
            if colour_description_present_flag {
                vst.colour_primaries = br.u(8)? as u8;
                vst.transfer_characteristics = br.u(8)? as u8;
                vst.matrix_coeffs = br.u(8)? as u8;
            }
            vui.video_signal_type = Some(vst);
        }

        // --- chroma_loc_info ---
        vui.chroma_loc_info_present_flag = br.u1()? != 0;
        if vui.chroma_loc_info_present_flag {
            let top = br.ue()?;
            if top > HEVC_MAX_CHROMA_SAMPLE_LOC_TYPE {
                return Err(VuiError::ValueOutOfRange {
                    field: "chroma_sample_loc_type_top_field",
                    got: top as u64,
                });
            }
            let bottom = br.ue()?;
            if bottom > HEVC_MAX_CHROMA_SAMPLE_LOC_TYPE {
                return Err(VuiError::ValueOutOfRange {
                    field: "chroma_sample_loc_type_bottom_field",
                    got: bottom as u64,
                });
            }
            vui.chroma_loc_info = Some(ChromaLocInfo {
                chroma_sample_loc_type_top_field: top,
                chroma_sample_loc_type_bottom_field: bottom,
            });
        }

        // --- single-bit indications ---
        vui.neutral_chroma_indication_flag = br.u1()? != 0;
        vui.field_seq_flag = br.u1()? != 0;
        vui.frame_field_info_present_flag = br.u1()? != 0;

        // --- default_display_window ---
        vui.default_display_window_flag = br.u1()? != 0;
        if vui.default_display_window_flag {
            vui.default_display_window = Some(DefaultDisplayWindow {
                left_offset: br.ue()?,
                right_offset: br.ue()?,
                top_offset: br.ue()?,
                bottom_offset: br.ue()?,
            });
        }

        // --- vui_timing_info (+ nested hrd_parameters) ---
        vui.vui_timing_info_present_flag = br.u1()? != 0;
        if vui.vui_timing_info_present_flag {
            let num_units_in_tick = br.u(32)?;
            // §E.2.1: vui_num_units_in_tick shall be greater than 0.
            if num_units_in_tick == 0 {
                return Err(VuiError::ValueOutOfRange {
                    field: "vui_num_units_in_tick",
                    got: 0,
                });
            }
            let time_scale = br.u(32)?;
            // §E.2.1: vui_time_scale shall be greater than 0.
            if time_scale == 0 {
                return Err(VuiError::ValueOutOfRange {
                    field: "vui_time_scale",
                    got: 0,
                });
            }
            let poc_proportional_to_timing_flag = br.u1()? != 0;
            let num_ticks_poc_diff_one_minus1 = if poc_proportional_to_timing_flag {
                // §E.2.1: range 0..=2^32 − 2. ue(v) cannot exceed
                // 2^32 − 2 anyway (the all-ones 32-bit value is not a
                // valid Exp-Golomb codeNum), so the upper bound is
                // satisfied structurally; we keep the read explicit.
                Some(br.ue()?)
            } else {
                None
            };
            let hrd_parameters_present_flag = br.u1()? != 0;
            let hrd_parameters = if hrd_parameters_present_flag {
                // §E.2.1 call site: hrd_parameters( 1,
                // sps_max_sub_layers_minus1 ). commonInfPresentFlag = 1
                // and there is no inherited common-info block here.
                Some(HrdParameters::parse(
                    br,
                    true,
                    sps_max_sub_layers_minus1,
                    None,
                )?)
            } else {
                None
            };
            vui.vui_timing_info = Some(VuiTimingInfo {
                num_units_in_tick,
                time_scale,
                poc_proportional_to_timing_flag,
                num_ticks_poc_diff_one_minus1,
                hrd_parameters_present_flag,
                hrd_parameters,
            });
        }

        // --- bitstream_restriction ---
        vui.bitstream_restriction_flag = br.u1()? != 0;
        if vui.bitstream_restriction_flag {
            let tiles_fixed_structure_flag = br.u1()? != 0;
            let motion_vectors_over_pic_boundaries_flag = br.u1()? != 0;
            let restricted_ref_pic_lists_flag = br.u1()? != 0;
            let min_spatial_segmentation_idc = br.ue()?;
            if min_spatial_segmentation_idc > HEVC_MAX_MIN_SPATIAL_SEGMENTATION_IDC {
                return Err(VuiError::ValueOutOfRange {
                    field: "min_spatial_segmentation_idc",
                    got: min_spatial_segmentation_idc as u64,
                });
            }
            let max_bytes_per_pic_denom = br.ue()?;
            if max_bytes_per_pic_denom > HEVC_MAX_DENOM {
                return Err(VuiError::ValueOutOfRange {
                    field: "max_bytes_per_pic_denom",
                    got: max_bytes_per_pic_denom as u64,
                });
            }
            let max_bits_per_min_cu_denom = br.ue()?;
            if max_bits_per_min_cu_denom > HEVC_MAX_DENOM {
                return Err(VuiError::ValueOutOfRange {
                    field: "max_bits_per_min_cu_denom",
                    got: max_bits_per_min_cu_denom as u64,
                });
            }
            let log2_max_mv_length_horizontal = br.ue()?;
            if log2_max_mv_length_horizontal > HEVC_MAX_LOG2_MV_LENGTH {
                return Err(VuiError::ValueOutOfRange {
                    field: "log2_max_mv_length_horizontal",
                    got: log2_max_mv_length_horizontal as u64,
                });
            }
            let log2_max_mv_length_vertical = br.ue()?;
            if log2_max_mv_length_vertical > HEVC_MAX_LOG2_MV_LENGTH {
                return Err(VuiError::ValueOutOfRange {
                    field: "log2_max_mv_length_vertical",
                    got: log2_max_mv_length_vertical as u64,
                });
            }
            vui.bitstream_restriction = Some(BitstreamRestriction {
                tiles_fixed_structure_flag,
                motion_vectors_over_pic_boundaries_flag,
                restricted_ref_pic_lists_flag,
                min_spatial_segmentation_idc,
                max_bytes_per_pic_denom,
                max_bits_per_min_cu_denom,
                log2_max_mv_length_horizontal,
                log2_max_mv_length_vertical,
            });
        }

        Ok(vui)
    }

    /// The effective video-signal-type block, applying the §E.2.1
    /// inference defaults ([`VideoSignalType::default`]) when
    /// `video_signal_type_present_flag == 0`.
    pub fn effective_video_signal_type(&self) -> VideoSignalType {
        self.video_signal_type.unwrap_or_default()
    }

    /// The effective chroma sample location, applying the §E.2.1
    /// inference (both fields 0) when `chroma_loc_info_present_flag == 0`.
    pub fn effective_chroma_loc_info(&self) -> ChromaLocInfo {
        self.chroma_loc_info.unwrap_or_default()
    }

    /// The effective bitstream-restriction block, applying the §E.2.1
    /// inference ([`BitstreamRestriction::inferred_absent`]) when
    /// `bitstream_restriction_flag == 0`.
    pub fn effective_bitstream_restriction(&self) -> BitstreamRestriction {
        self.bitstream_restriction.unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convert a bit string (non-`0`/`1` characters ignored) into a
    /// packed MSB-first byte vector, zero-padded to a byte boundary.
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

    /// Encode `v` as a fixed-width `u(n)` bit string (MSB-first).
    fn u_bits(v: u64, n: u32) -> String {
        let mut s = String::with_capacity(n as usize);
        for i in (0..n).rev() {
            s.push(if (v >> i) & 1 == 1 { '1' } else { '0' });
        }
        s
    }

    /// Encode `v` as a 0-th-order unsigned Exp-Golomb `ue(v)` code.
    fn ue_bits(v: u64) -> String {
        let code = v + 1;
        let num_bits = 64 - code.leading_zeros();
        let mut s = String::new();
        for _ in 0..(num_bits - 1) {
            s.push('0');
        }
        s + &u_bits(code, num_bits)
    }

    /// A minimal VUI (every present-flag clear) parses to the §E.2.1
    /// inference defaults.
    #[test]
    fn parse_minimal_vui_defaults() {
        // 10 clear present-flags + padding.
        let bytes = bits_to_bytes("0000000000");
        let mut br = BitReader::new(&bytes);
        let vui = VuiParameters::parse(&mut br, 0).expect("VUI parse");
        assert_eq!(vui, VuiParameters::default());
        assert!(!vui.aspect_ratio_info_present_flag);
        assert!(!vui.vui_timing_info_present_flag);
        assert!(!vui.bitstream_restriction_flag);
        // Effective accessors fall back to the §E.2.1 inferred values.
        assert_eq!(
            vui.effective_video_signal_type(),
            VideoSignalType::default()
        );
        assert_eq!(
            vui.effective_bitstream_restriction(),
            BitstreamRestriction::inferred_absent()
        );
    }

    /// A non-EXTENDED_SAR aspect ratio does not read sar_width/height.
    #[test]
    fn parse_table_aspect_ratio_no_sar() {
        let mut s = String::new();
        s += "1"; // aspect_ratio_info_present_flag
        s += &u_bits(2, 8); // aspect_ratio_idc = 2 (12:11), not EXTENDED_SAR
        s += "0"; // overscan
        s += "0"; // video_signal_type
        s += "0"; // chroma_loc
        s += "000"; // neutral / field_seq / frame_field
        s += "0"; // default_display_window
        s += "0"; // vui_timing_info
        s += "0"; // bitstream_restriction
        let bytes = bits_to_bytes(&s);
        let mut br = BitReader::new(&bytes);
        let vui = VuiParameters::parse(&mut br, 0).expect("VUI parse");
        let ar = vui.aspect_ratio_info.expect("aspect ratio");
        assert_eq!(ar.aspect_ratio_idc, 2);
        assert_eq!(ar.sar_width, 0);
        assert_eq!(ar.sar_height, 0);
    }

    /// `vui_time_scale == 0` is rejected (§E.2.1 "shall be greater than 0").
    #[test]
    fn parse_rejects_zero_time_scale() {
        let mut s = String::new();
        s += "00000000"; // aspect/overscan/vst/chroma/neutral/field/frame/ddw
        s += "1"; // vui_timing_info_present_flag
        s += &u_bits(1, 32); // num_units_in_tick = 1 (> 0, OK)
        s += &u_bits(0, 32); // time_scale = 0 (illegal)
        let bytes = bits_to_bytes(&s);
        let mut br = BitReader::new(&bytes);
        let err = VuiParameters::parse(&mut br, 0).expect_err("must reject");
        assert_eq!(
            err,
            VuiError::ValueOutOfRange {
                field: "vui_time_scale",
                got: 0,
            }
        );
    }

    /// `min_spatial_segmentation_idc > 4095` is rejected (§E.2.1 range).
    #[test]
    fn parse_rejects_min_spatial_segmentation_out_of_range() {
        let mut s = String::new();
        s += "000000000"; // all present-flags clear up to timing (9 bits)
        s += "1"; // bitstream_restriction_flag
        s += "000"; // tiles_fixed / mv_over_bound / restricted_ref
        s += &ue_bits(4096); // min_spatial_segmentation_idc = 4096 (illegal)
        let bytes = bits_to_bytes(&s);
        let mut br = BitReader::new(&bytes);
        let err = VuiParameters::parse(&mut br, 0).expect_err("must reject");
        assert_eq!(
            err,
            VuiError::ValueOutOfRange {
                field: "min_spatial_segmentation_idc",
                got: 4096,
            }
        );
    }

    /// A buffer that ends before the first present flag can be read
    /// surfaces [`VuiError::Truncated`].
    #[test]
    fn parse_truncated_buffer() {
        let empty: Vec<u8> = Vec::new();
        let mut br = BitReader::new(&empty);
        let err = VuiParameters::parse(&mut br, 0).expect_err("must truncate");
        assert_eq!(err, VuiError::Truncated);
    }
}
