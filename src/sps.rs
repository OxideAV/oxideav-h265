//! Sequence Parameter Set (SPS) parser per ITU-T Rec. H.265 §7.3.2.2.
//!
//! Round-3 scope: parse the *structural prefix* of an SPS RBSP up to
//! and including the `sample_adaptive_offset_enabled_flag`. The
//! trailing `pcm_*` / `num_short_term_ref_pic_sets` / `st_ref_pic_set` /
//! `long_term_ref_pics_present_flag` / `sps_temporal_mvp_enabled_flag` /
//! `strong_intra_smoothing_enabled_flag` / VUI / extension tail is
//! **not** materialised yet. The scaling-list path is parsed only to
//! the extent that the gate flag is read; if the flag is set the parser
//! errors out rather than walking `scaling_list_data()` (which is
//! deferred to a future round; the docs round-3 fixture has the flag
//! off).
//!
//! ## Layout summary (structural prefix only)
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
//! amp_enabled_flag                                 u(1)
//! sample_adaptive_offset_enabled_flag              u(1)
//! ...                                              /* tail deferred */
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
//! * `scaling_list_enabled_flag == 1` is **rejected** at round 3 with
//!   [`SpsError::ScalingListUnsupported`]; `scaling_list_data()` is a
//!   deferred parse target.

use crate::bitreader::{BitReader, BitReaderError};
use crate::vps::{ProfileTierLevel, SubLayerOrderingInfo, VpsError, HEVC_MAX_SUB_LAYERS};

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
    /// `scaling_list_enabled_flag == 1` was encountered. The
    /// scaling-list block (§7.3.4) is a deferred parse target; the
    /// fixture corpus this round targets has the flag off.
    ScalingListUnsupported,
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
            Self::ScalingListUnsupported => {
                f.write_str("SPS has scaling_list_enabled_flag == 1; scaling-list parse deferred")
            }
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

/// Parsed Sequence Parameter Set per §7.3.2.2 (structural prefix only).
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
    /// `scaling_list_enabled_flag`. Round 3 errors out when this is
    /// 1; once `scaling_list_data()` is implemented this field will
    /// hold the parsed flag for both states.
    pub scaling_list_enabled_flag: bool,
    /// `amp_enabled_flag` — asymmetric motion partitions.
    pub amp_enabled_flag: bool,
    /// `sample_adaptive_offset_enabled_flag` — SPS-level gate for
    /// SAO; the per-slice gates are in the slice header.
    pub sample_adaptive_offset_enabled_flag: bool,
}

impl SeqParameterSet {
    /// Parse `seq_parameter_set_rbsp()` starting from the first bit
    /// of the (already-unescaped) RBSP body — i.e. after the two-byte
    /// NAL header has been removed (see [`crate::nal::NalUnit`]).
    pub fn parse(rbsp: &[u8]) -> Result<Self, SpsError> {
        let mut br = BitReader::new(rbsp);
        Self::parse_inner(&mut br)
    }

    fn parse_inner(br: &mut BitReader<'_>) -> Result<Self, SpsError> {
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
        if scaling_list_enabled_flag {
            // §7.3.4 scaling_list_data() — deferred to a future round.
            return Err(SpsError::ScalingListUnsupported);
        }

        let amp_enabled_flag = br.u1()? != 0;
        let sample_adaptive_offset_enabled_flag = br.u1()? != 0;

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
            amp_enabled_flag,
            sample_adaptive_offset_enabled_flag,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nal::{collect_nal_units, strip_emulation_prevention};

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
        //       log2_min_tb_size=2 sao_enabled=1 amp_enabled=0
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
        let mut bits = Vec::<u8>::new();
        let mut push = |s: &str| {
            for c in s.chars() {
                if c == '0' || c == '1' {
                    bits.push((c as u8) - b'0');
                }
            }
        };
        // sps_video_parameter_set_id u(4) = 0
        push("0000");
        // sps_max_sub_layers_minus1 u(3) = 0
        push("000");
        // sps_temporal_id_nesting_flag u(1) = 1
        push("1");
        // profile_tier_level(1, 0):
        //   profile_space u(2)=0, tier_flag u(1)=0, profile_idc u(5)=1 (Main)
        push("00");
        push("0");
        push("00001");
        // 32 compat flags = 0
        push(&"0".repeat(32));
        // 4 source flags = 0
        push("0000");
        // 43-bit block = 0
        push(&"0".repeat(43));
        // 1-bit reserved = 0
        push("0");
        // general_level_idc u(8) = 30
        push("00011110");
        // (no per-sub-layer loop — max_sub_layers_minus1 == 0)
        // sps_seq_parameter_set_id ue(v) = 0  -> '1'
        push("1");
        // chroma_format_idc ue(v) = 3 -> codeNum 3 -> '00100'
        push("00100");
        // separate_colour_plane_flag u(1) = 1
        push("1");
        // pic_width_in_luma_samples ue(v) = 16 -> codeNum 16
        //   leadingZeroBits=4, suffix='00001' (1) → 16 = (1<<4)-1+1
        push("000010001");
        // pic_height_in_luma_samples ue(v) = 16 (same encoding)
        push("000010001");
        // conformance_window_flag u(1) = 1
        push("1");
        // conf_win_*_offset ue(v): 0, 0, 0, 0  → '1 1 1 1'
        push("1");
        push("1");
        push("1");
        push("1");
        // bit_depth_luma_minus8 ue(v) = 2 (10-bit) → codeNum 2 → '011'
        push("011");
        // bit_depth_chroma_minus8 ue(v) = 2 → '011'
        push("011");
        // log2_max_pic_order_cnt_lsb_minus4 ue(v) = 4 → codeNum 4 → '00101'
        push("00101");
        // sps_sub_layer_ordering_info_present_flag u(1) = 1
        push("1");
        // one entry (i=0): three ue(v)=0 → '1 1 1'
        push("1");
        push("1");
        push("1");
        // log2_min_luma_coding_block_size_minus3 ue(v) = 0 → '1'
        push("1");
        // log2_diff_max_min_luma_coding_block_size ue(v) = 1 → '010'
        push("010");
        // log2_min_luma_transform_block_size_minus2 ue(v) = 0 → '1'
        push("1");
        // log2_diff_max_min_luma_transform_block_size ue(v) = 2 → '011'
        push("011");
        // max_transform_hierarchy_depth_inter ue(v) = 0 → '1'
        push("1");
        // max_transform_hierarchy_depth_intra ue(v) = 0 → '1'
        push("1");
        // scaling_list_enabled_flag u(1) = 0
        push("0");
        // amp_enabled_flag u(1) = 1
        push("1");
        // sample_adaptive_offset_enabled_flag u(1) = 1
        push("1");

        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut bytes = Vec::with_capacity(bits.len() / 8);
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for &bit in chunk {
                b = (b << 1) | bit;
            }
            bytes.push(b);
        }

        let sps = SeqParameterSet::parse(&bytes).expect("SPS parse");
        assert_eq!(sps.chroma_format_idc, 3);
        assert!(sps.separate_colour_plane_flag);
        assert_eq!(sps.pic_width_in_luma_samples, 16);
        assert_eq!(sps.pic_height_in_luma_samples, 16);
        assert!(sps.conformance_window_flag);
        assert_eq!(sps.conformance_window.left_offset, 0);
        assert_eq!(sps.conformance_window.bottom_offset, 0);
        assert_eq!(sps.bit_depth_luma_minus8, 2);
        assert_eq!(sps.bit_depth_chroma_minus8, 2);
        assert_eq!(sps.bit_depth_luma(), 10);
        assert_eq!(sps.bit_depth_chroma(), 10);
        assert_eq!(sps.log2_max_pic_order_cnt_lsb_minus4, 4);
        assert!(sps.sub_layer_ordering_info_present_flag);
        assert_eq!(sps.log2_min_cb_size(), 3);
        assert_eq!(sps.log2_ctb_size(), 4);
        assert!(sps.amp_enabled_flag);
        assert!(sps.sample_adaptive_offset_enabled_flag);
    }

    /// Hand-assembled SPS with two sub-layers and the ordering-info
    /// present flag set to 0 — the [0] sub-layer must inherit the
    /// [1] triple per §7.4.3.2.1.
    #[test]
    fn ordering_info_present_flag_zero_propagates() {
        let mut bits = Vec::<u8>::new();
        let mut push = |s: &str| {
            for c in s.chars() {
                if c == '0' || c == '1' {
                    bits.push((c as u8) - b'0');
                }
            }
        };
        push("0000"); // vps_id
        push("001"); // max_sub_layers_minus1 = 1
        push("1"); // nesting flag
                   // profile_tier_level(1, 1):
        push("00"); // profile_space
        push("0"); // tier
        push("00001"); // profile_idc
        push(&"0".repeat(32));
        push("0000");
        push(&"0".repeat(43));
        push("0");
        push("00011110"); // level=30
                          // per-sub-layer present flags for i=0..1 (one iter): 2 bits
        push("00"); // sub_layer_profile_present[0]=0 / sub_layer_level_present[0]=0
                    // 8-N reserved 2-bit pads: N=1 → 7 entries × 2 bits = 14 bits, all 0
        push(&"0".repeat(14));

        // sps_id ue=0
        push("1");
        // chroma_format_idc ue=1 → codeNum 1 → '010'
        push("010");
        // pic_width ue=16
        push("000010001");
        // pic_height ue=16
        push("000010001");
        // conf_win=0
        push("0");
        // bit_depths
        push("1"); // 0
        push("1"); // 0
                   // log2_max_poc_lsb_minus4 = 4 → '00101'
        push("00101");
        // sub_layer_ordering_info_present_flag = 0
        push("0");
        // i=1 only: dpb_minus1=2 ('011'), reorder=0 ('1'), latency=0 ('1')
        push("011");
        push("1");
        push("1");
        // log2_min_cb_minus3 = 0
        push("1");
        // log2_diff = 1 → '010'
        push("010");
        // log2_min_tb_minus2 = 0
        push("1");
        // log2_diff_tb = 2 → '011'
        push("011");
        push("1"); // max_transform_depth_inter
        push("1"); // max_transform_depth_intra
        push("0"); // scaling_list_enabled
        push("0"); // amp_enabled
        push("1"); // sao_enabled

        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut bytes = Vec::with_capacity(bits.len() / 8);
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for &bit in chunk {
                b = (b << 1) | bit;
            }
            bytes.push(b);
        }
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
    }

    /// Hand-assembled SPS that signals
    /// `scaling_list_enabled_flag == 1` — the round-3 parser must
    /// refuse this with [`SpsError::ScalingListUnsupported`].
    #[test]
    fn rejects_scaling_list_enabled() {
        let mut bits = Vec::<u8>::new();
        let mut push = |s: &str| {
            for c in s.chars() {
                if c == '0' || c == '1' {
                    bits.push((c as u8) - b'0');
                }
            }
        };
        push("0000"); // vps_id
        push("000"); // max_sub_layers_minus1
        push("1"); // nesting
                   // profile_tier_level(1, 0)
        push("00");
        push("0");
        push("00001");
        push(&"0".repeat(32));
        push("0000");
        push(&"0".repeat(43));
        push("0");
        push("00011110"); // level=30
        push("1"); // sps_id=0
        push("010"); // chroma_format_idc=1
        push("000010001"); // width=16
        push("000010001"); // height=16
        push("0"); // conf_win=0
        push("1"); // bd_luma=0
        push("1"); // bd_chroma=0
        push("00101"); // log2_max_poc_lsb_minus4=4
        push("1"); // ordering present
        push("1");
        push("1");
        push("1"); // sublayer ordering
        push("1"); // log2_min_cb_minus3=0
        push("010"); // diff=1
        push("1");
        push("011");
        push("1");
        push("1");
        push("1"); // scaling_list_enabled = 1  ← must be rejected
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut bytes = Vec::with_capacity(bits.len() / 8);
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for &bit in chunk {
                b = (b << 1) | bit;
            }
            bytes.push(b);
        }
        let err = SeqParameterSet::parse(&bytes).unwrap_err();
        assert_eq!(err, SpsError::ScalingListUnsupported);
    }

    /// `chroma_format_idc` is u-Exp-Golomb so on-wire codeNum 4 would
    /// decode to a value of 4 — outside the legal 0..=3 range. The
    /// parser must reject it.
    #[test]
    fn rejects_chroma_format_idc_out_of_range() {
        let mut bits = Vec::<u8>::new();
        let mut push = |s: &str| {
            for c in s.chars() {
                if c == '0' || c == '1' {
                    bits.push((c as u8) - b'0');
                }
            }
        };
        push("0000");
        push("000");
        push("1");
        push("00");
        push("0");
        push("00001");
        push(&"0".repeat(32));
        push("0000");
        push(&"0".repeat(43));
        push("0");
        push("00011110");
        push("1"); // sps_id=0
                   // chroma_format_idc ue=4 → '00101'
        push("00101");
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut bytes = Vec::with_capacity(bits.len() / 8);
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for &bit in chunk {
                b = (b << 1) | bit;
            }
            bytes.push(b);
        }
        let err = SeqParameterSet::parse(&bytes).unwrap_err();
        assert_eq!(
            err,
            SpsError::ValueOutOfRange {
                field: "chroma_format_idc",
                got: 4
            }
        );
    }
}
