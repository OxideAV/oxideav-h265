//! Picture Parameter Set (PPS) parser per ITU-T Rec. H.265 §7.3.2.3.1.
//!
//! Round-5 scope: parse the general `pic_parameter_set_rbsp()` body
//! through `pps_extension_present_flag`, materialising every field of
//! the §7.3.2.3.1 syntax table including the tiles block
//! (`num_tile_columns_minus1` / `num_tile_rows_minus1` /
//! `uniform_spacing_flag` and, when `uniform_spacing_flag == 0`, the
//! `column_width_minus1[]` / `row_height_minus1[]` arrays) and the
//! deblocking-filter-control block. The §7.4.3.3.1 inference rules for
//! the conditionally-signalled fields are applied so the parsed struct
//! always carries the effective value.
//!
//! Two bodies are deferred and surfaced rather than decoded:
//!
//! * `pps_scaling_list_data_present_flag == 1` is **rejected** with
//!   [`PpsError::ScalingListUnsupported`]; the `scaling_list_data()`
//!   parser (§7.3.4) is a shared follow-up with the SPS path.
//! * When `pps_extension_present_flag == 1` the four extension flags,
//!   their bodies (`pps_range_extension()` etc.), and the trailing
//!   `rbsp_trailing_bits()` are surfaced as a single [`OpaqueTail`]
//!   exactly as the SPS path does for its VUI / extension tail.
//!
//! ## Layout summary
//!
//! ```text
//! pps_pic_parameter_set_id                          ue(v)
//! pps_seq_parameter_set_id                          ue(v)
//! dependent_slice_segments_enabled_flag              u(1)
//! output_flag_present_flag                           u(1)
//! num_extra_slice_header_bits                        u(3)
//! sign_data_hiding_enabled_flag                      u(1)
//! cabac_init_present_flag                            u(1)
//! num_ref_idx_l0_default_active_minus1              ue(v)
//! num_ref_idx_l1_default_active_minus1              ue(v)
//! init_qp_minus26                                   se(v)
//! constrained_intra_pred_flag                        u(1)
//! transform_skip_enabled_flag                        u(1)
//! cu_qp_delta_enabled_flag                           u(1)
//! if( cu_qp_delta_enabled_flag )
//!   diff_cu_qp_delta_depth                          ue(v)
//! pps_cb_qp_offset                                  se(v)
//! pps_cr_qp_offset                                  se(v)
//! pps_slice_chroma_qp_offsets_present_flag           u(1)
//! weighted_pred_flag                                 u(1)
//! weighted_bipred_flag                               u(1)
//! transquant_bypass_enabled_flag                     u(1)
//! tiles_enabled_flag                                 u(1)
//! entropy_coding_sync_enabled_flag                   u(1)
//! if( tiles_enabled_flag ) {
//!   num_tile_columns_minus1                         ue(v)
//!   num_tile_rows_minus1                            ue(v)
//!   uniform_spacing_flag                             u(1)
//!   if( !uniform_spacing_flag ) {
//!     for( i = 0; i < num_tile_columns_minus1; i++ )
//!       column_width_minus1[i]                      ue(v)
//!     for( i = 0; i < num_tile_rows_minus1; i++ )
//!       row_height_minus1[i]                        ue(v)
//!   }
//!   loop_filter_across_tiles_enabled_flag            u(1)
//! }
//! pps_loop_filter_across_slices_enabled_flag         u(1)
//! deblocking_filter_control_present_flag             u(1)
//! if( deblocking_filter_control_present_flag ) {
//!   deblocking_filter_override_enabled_flag          u(1)
//!   pps_deblocking_filter_disabled_flag             u(1)
//!   if( !pps_deblocking_filter_disabled_flag ) {
//!     pps_beta_offset_div2                          se(v)
//!     pps_tc_offset_div2                            se(v)
//!   }
//! }
//! pps_scaling_list_data_present_flag                 u(1)
//!   /* rejected when 1 — scaling_list_data() deferred */
//! lists_modification_present_flag                    u(1)
//! log2_parallel_merge_level_minus2                  ue(v)
//! slice_segment_header_extension_present_flag        u(1)
//! pps_extension_present_flag                         u(1)
//!   /* opaque tail begins here when 1 */
//! ```
//!
//! Validity checks performed here, sourced from §7.4.3.3.1:
//!
//! * `pps_pic_parameter_set_id` range 0..=63.
//! * `pps_seq_parameter_set_id` range 0..=15.
//! * `num_ref_idx_l{0,1}_default_active_minus1` range 0..=14.
//! * `init_qp_minus26` range −( 26 + QpBdOffsetY ) to +25. The
//!   lower bound depends on the bit depth from the active SPS
//!   (`QpBdOffsetY = 6 * bit_depth_luma_minus8`); without an SPS the
//!   parser uses the loosest legal bound, −( 26 + 6*8 ) = −74, and a
//!   caller that has resolved the SPS can re-check against the exact
//!   bound via [`PicParameterSet::init_qp_in_range`].
//! * `pps_cb_qp_offset` / `pps_cr_qp_offset` range −12..=12.
//! * `pps_beta_offset_div2` / `pps_tc_offset_div2` range −6..=6.

use crate::bitreader::{BitReader, BitReaderError};
use crate::sps::OpaqueTail;

/// Errors that can arise while parsing a PPS RBSP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PpsError {
    /// The RBSP ran out of bits before the PPS was fully parsed.
    Truncated,
    /// A syntax element's parsed value was outside the legal range
    /// specified for it in §7.4.3.3.1.
    ValueOutOfRange {
        /// Name of the offending syntax element.
        field: &'static str,
        /// The (illegal) value as an `i64` (covers both `ue(v)` and
        /// `se(v)` elements).
        got: i64,
    },
    /// `pps_scaling_list_data_present_flag == 1` was encountered. The
    /// scaling-list block (§7.3.4) is a deferred parse target shared
    /// with the SPS path; the fixture corpus this round targets has
    /// the flag off.
    ScalingListUnsupported,
    /// An unexpected bitstream-level error surfaced from the reader.
    Bitstream(BitReaderError),
}

impl core::fmt::Display for PpsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => f.write_str("PPS RBSP truncated"),
            Self::ValueOutOfRange { field, got } => {
                write!(f, "PPS syntax element {field} out of range: {got}")
            }
            Self::ScalingListUnsupported => f.write_str(
                "PPS has pps_scaling_list_data_present_flag == 1; scaling-list parse deferred",
            ),
            Self::Bitstream(e) => write!(f, "bitstream error during PPS parse: {e}"),
        }
    }
}

impl std::error::Error for PpsError {}

impl From<BitReaderError> for PpsError {
    fn from(e: BitReaderError) -> Self {
        match e {
            BitReaderError::EndOfBuffer => Self::Truncated,
            other => Self::Bitstream(other),
        }
    }
}

/// Tiles partitioning block per §7.3.2.3.1 (`tiles_enabled_flag == 1`).
///
/// When `tiles_enabled_flag` is 0 this block is absent and the
/// derived values are inferred per §7.4.3.3.1 (one column, one row,
/// uniform spacing, loop filtering across tiles enabled).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileInfo {
    /// `num_tile_columns_minus1` (`ue(v)`). The number of tile columns
    /// is `value + 1`.
    pub num_tile_columns_minus1: u32,
    /// `num_tile_rows_minus1` (`ue(v)`). The number of tile rows is
    /// `value + 1`.
    pub num_tile_rows_minus1: u32,
    /// `uniform_spacing_flag`. When true the explicit
    /// `column_width_minus1[]` / `row_height_minus1[]` arrays are
    /// absent and the partitioning is uniform.
    pub uniform_spacing_flag: bool,
    /// `column_width_minus1[i]` for `i` in
    /// `0..num_tile_columns_minus1`. Empty when `uniform_spacing_flag`.
    pub column_width_minus1: Vec<u32>,
    /// `row_height_minus1[i]` for `i` in `0..num_tile_rows_minus1`.
    /// Empty when `uniform_spacing_flag`.
    pub row_height_minus1: Vec<u32>,
}

/// Deblocking-filter-control block per §7.3.2.3.1
/// (`deblocking_filter_control_present_flag == 1`).
///
/// The [`Default`] value matches the §7.4.3.3.1 inference for the
/// absent-control case: every flag false and both offsets 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeblockingFilterControl {
    /// `deblocking_filter_override_enabled_flag`. Inferred to false
    /// when the control block is absent (§7.4.3.3.1).
    pub override_enabled_flag: bool,
    /// `pps_deblocking_filter_disabled_flag`. Inferred to false when
    /// the control block is absent (§7.4.3.3.1).
    pub disabled_flag: bool,
    /// `pps_beta_offset_div2` (`se(v)`, range −6..=6). Inferred to 0
    /// when absent (the disabled / no-control case).
    pub beta_offset_div2: i8,
    /// `pps_tc_offset_div2` (`se(v)`, range −6..=6). Inferred to 0
    /// when absent.
    pub tc_offset_div2: i8,
}

/// Parsed Picture Parameter Set per §7.3.2.3.1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PicParameterSet {
    /// `pps_pic_parameter_set_id` (`ue(v)`, range 0..=63).
    pub pps_id: u8,
    /// `pps_seq_parameter_set_id` (`ue(v)`, range 0..=15) — the SPS
    /// this PPS activates.
    pub sps_id: u8,
    /// `dependent_slice_segments_enabled_flag`.
    pub dependent_slice_segments_enabled_flag: bool,
    /// `output_flag_present_flag`.
    pub output_flag_present_flag: bool,
    /// `num_extra_slice_header_bits` (`u(3)`). The conformance range
    /// is 0..=2 but decoders must accept any value, so this is not
    /// range-checked.
    pub num_extra_slice_header_bits: u8,
    /// `sign_data_hiding_enabled_flag`.
    pub sign_data_hiding_enabled_flag: bool,
    /// `cabac_init_present_flag`.
    pub cabac_init_present_flag: bool,
    /// `num_ref_idx_l0_default_active_minus1` (`ue(v)`, range 0..=14).
    /// `num_ref_idx_l0_default_active = value + 1`.
    pub num_ref_idx_l0_default_active_minus1: u8,
    /// `num_ref_idx_l1_default_active_minus1` (`ue(v)`, range 0..=14).
    pub num_ref_idx_l1_default_active_minus1: u8,
    /// `init_qp_minus26` (`se(v)`). `init_qp = value + 26`.
    pub init_qp_minus26: i32,
    /// `constrained_intra_pred_flag`.
    pub constrained_intra_pred_flag: bool,
    /// `transform_skip_enabled_flag`.
    pub transform_skip_enabled_flag: bool,
    /// `cu_qp_delta_enabled_flag`. When set, [`Self::diff_cu_qp_delta_depth`]
    /// is signalled; otherwise it is inferred to 0.
    pub cu_qp_delta_enabled_flag: bool,
    /// `diff_cu_qp_delta_depth` (`ue(v)`). Inferred to 0 when
    /// `cu_qp_delta_enabled_flag` is false (§7.4.3.3.1).
    pub diff_cu_qp_delta_depth: u32,
    /// `pps_cb_qp_offset` (`se(v)`, range −12..=12).
    pub pps_cb_qp_offset: i8,
    /// `pps_cr_qp_offset` (`se(v)`, range −12..=12).
    pub pps_cr_qp_offset: i8,
    /// `pps_slice_chroma_qp_offsets_present_flag`.
    pub pps_slice_chroma_qp_offsets_present_flag: bool,
    /// `weighted_pred_flag`.
    pub weighted_pred_flag: bool,
    /// `weighted_bipred_flag`.
    pub weighted_bipred_flag: bool,
    /// `transquant_bypass_enabled_flag`.
    pub transquant_bypass_enabled_flag: bool,
    /// `tiles_enabled_flag`. When set, [`Self::tiles`] carries the
    /// signalled values; otherwise it carries the §7.4.3.3.1 inferred
    /// single-tile values.
    pub tiles_enabled_flag: bool,
    /// `entropy_coding_sync_enabled_flag`.
    pub entropy_coding_sync_enabled_flag: bool,
    /// Tiles block. Present (with inferred single-tile values) even
    /// when `tiles_enabled_flag` is false, so callers always have an
    /// effective tile geometry.
    pub tiles: TileInfo,
    /// `loop_filter_across_tiles_enabled_flag`. Inferred to true when
    /// the tiles block is absent (§7.4.3.3.1).
    pub loop_filter_across_tiles_enabled_flag: bool,
    /// `pps_loop_filter_across_slices_enabled_flag`.
    pub pps_loop_filter_across_slices_enabled_flag: bool,
    /// `deblocking_filter_control_present_flag`.
    pub deblocking_filter_control_present_flag: bool,
    /// Deblocking-filter-control values, carrying §7.4.3.3.1 inferred
    /// defaults when the control block is absent.
    pub deblocking: DeblockingFilterControl,
    /// `pps_scaling_list_data_present_flag`. Round 5 rejects the parse
    /// when this is 1; the `scaling_list_data()` parser is deferred.
    pub pps_scaling_list_data_present_flag: bool,
    /// `lists_modification_present_flag`.
    pub lists_modification_present_flag: bool,
    /// `log2_parallel_merge_level_minus2` (`ue(v)`).
    /// `Log2ParMrgLevel = value + 2`.
    pub log2_parallel_merge_level_minus2: u32,
    /// `slice_segment_header_extension_present_flag`.
    pub slice_segment_header_extension_present_flag: bool,
    /// `pps_extension_present_flag`. When set, the four extension
    /// flags plus their bodies and the RBSP trailing bits are surfaced
    /// as [`Self::opaque_tail`].
    pub pps_extension_present_flag: bool,
    /// Opaque suffix of the PPS RBSP. Populated when
    /// `pps_extension_present_flag == 1` (the extension flag block and
    /// any extension bodies are not parsed this round). `None` when
    /// the PPS ended cleanly after `pps_extension_present_flag == 0`
    /// (only `rbsp_trailing_bits()` remains and it is consumed
    /// implicitly).
    pub opaque_tail: Option<OpaqueTail>,
}

impl PicParameterSet {
    /// Parse `pic_parameter_set_rbsp()` starting from the first bit of
    /// the (already-unescaped) RBSP body — i.e. after the two-byte NAL
    /// header has been removed (see [`crate::nal::NalUnit`]).
    pub fn parse(rbsp: &[u8]) -> Result<Self, PpsError> {
        let mut br = BitReader::new(rbsp);
        Self::parse_inner(&mut br, rbsp)
    }

    fn parse_inner(br: &mut BitReader<'_>, rbsp: &[u8]) -> Result<Self, PpsError> {
        let pps_id_raw = br.ue()?;
        if pps_id_raw > 63 {
            return Err(PpsError::ValueOutOfRange {
                field: "pps_pic_parameter_set_id",
                got: pps_id_raw as i64,
            });
        }
        let sps_id_raw = br.ue()?;
        if sps_id_raw > 15 {
            return Err(PpsError::ValueOutOfRange {
                field: "pps_seq_parameter_set_id",
                got: sps_id_raw as i64,
            });
        }

        let dependent_slice_segments_enabled_flag = br.u1()? != 0;
        let output_flag_present_flag = br.u1()? != 0;
        let num_extra_slice_header_bits = br.u(3)? as u8;
        let sign_data_hiding_enabled_flag = br.u1()? != 0;
        let cabac_init_present_flag = br.u1()? != 0;

        let num_ref_idx_l0_default_active_minus1_raw = br.ue()?;
        if num_ref_idx_l0_default_active_minus1_raw > 14 {
            return Err(PpsError::ValueOutOfRange {
                field: "num_ref_idx_l0_default_active_minus1",
                got: num_ref_idx_l0_default_active_minus1_raw as i64,
            });
        }
        let num_ref_idx_l1_default_active_minus1_raw = br.ue()?;
        if num_ref_idx_l1_default_active_minus1_raw > 14 {
            return Err(PpsError::ValueOutOfRange {
                field: "num_ref_idx_l1_default_active_minus1",
                got: num_ref_idx_l1_default_active_minus1_raw as i64,
            });
        }

        let init_qp_minus26 = br.se()?;
        // §7.4.3.3.1: −( 26 + QpBdOffsetY ) .. +25. The lower bound
        // depends on the active SPS bit depth; QpBdOffsetY is at most
        // 6 * 8 = 48 (the maximum legal bit_depth_luma_minus8), so the
        // loosest legal lower bound is −74. We range-check against the
        // loosest bound here (no SPS context) and provide
        // `init_qp_in_range()` for callers that have resolved the SPS.
        if !(-74..=25).contains(&init_qp_minus26) {
            return Err(PpsError::ValueOutOfRange {
                field: "init_qp_minus26",
                got: init_qp_minus26 as i64,
            });
        }

        let constrained_intra_pred_flag = br.u1()? != 0;
        let transform_skip_enabled_flag = br.u1()? != 0;

        let cu_qp_delta_enabled_flag = br.u1()? != 0;
        let diff_cu_qp_delta_depth = if cu_qp_delta_enabled_flag {
            br.ue()?
        } else {
            // §7.4.3.3.1 inference.
            0
        };

        let pps_cb_qp_offset = br.se()?;
        if !(-12..=12).contains(&pps_cb_qp_offset) {
            return Err(PpsError::ValueOutOfRange {
                field: "pps_cb_qp_offset",
                got: pps_cb_qp_offset as i64,
            });
        }
        let pps_cr_qp_offset = br.se()?;
        if !(-12..=12).contains(&pps_cr_qp_offset) {
            return Err(PpsError::ValueOutOfRange {
                field: "pps_cr_qp_offset",
                got: pps_cr_qp_offset as i64,
            });
        }

        let pps_slice_chroma_qp_offsets_present_flag = br.u1()? != 0;
        let weighted_pred_flag = br.u1()? != 0;
        let weighted_bipred_flag = br.u1()? != 0;
        let transquant_bypass_enabled_flag = br.u1()? != 0;
        let tiles_enabled_flag = br.u1()? != 0;
        let entropy_coding_sync_enabled_flag = br.u1()? != 0;

        let (tiles, loop_filter_across_tiles_enabled_flag) = if tiles_enabled_flag {
            let num_tile_columns_minus1 = br.ue()?;
            let num_tile_rows_minus1 = br.ue()?;
            let uniform_spacing_flag = br.u1()? != 0;
            let (column_width_minus1, row_height_minus1) = if !uniform_spacing_flag {
                let mut cols = Vec::with_capacity(num_tile_columns_minus1 as usize);
                for _ in 0..num_tile_columns_minus1 {
                    cols.push(br.ue()?);
                }
                let mut rows = Vec::with_capacity(num_tile_rows_minus1 as usize);
                for _ in 0..num_tile_rows_minus1 {
                    rows.push(br.ue()?);
                }
                (cols, rows)
            } else {
                (Vec::new(), Vec::new())
            };
            let loop_filter_across_tiles_enabled_flag = br.u1()? != 0;
            (
                TileInfo {
                    num_tile_columns_minus1,
                    num_tile_rows_minus1,
                    uniform_spacing_flag,
                    column_width_minus1,
                    row_height_minus1,
                },
                loop_filter_across_tiles_enabled_flag,
            )
        } else {
            // §7.4.3.3.1 inference: one column, one row, uniform
            // spacing, loop filtering across tiles enabled.
            (
                TileInfo {
                    num_tile_columns_minus1: 0,
                    num_tile_rows_minus1: 0,
                    uniform_spacing_flag: true,
                    column_width_minus1: Vec::new(),
                    row_height_minus1: Vec::new(),
                },
                true,
            )
        };

        let pps_loop_filter_across_slices_enabled_flag = br.u1()? != 0;

        let deblocking_filter_control_present_flag = br.u1()? != 0;
        let deblocking = if deblocking_filter_control_present_flag {
            let override_enabled_flag = br.u1()? != 0;
            let disabled_flag = br.u1()? != 0;
            let (beta_offset_div2, tc_offset_div2) = if !disabled_flag {
                let beta = br.se()?;
                if !(-6..=6).contains(&beta) {
                    return Err(PpsError::ValueOutOfRange {
                        field: "pps_beta_offset_div2",
                        got: beta as i64,
                    });
                }
                let tc = br.se()?;
                if !(-6..=6).contains(&tc) {
                    return Err(PpsError::ValueOutOfRange {
                        field: "pps_tc_offset_div2",
                        got: tc as i64,
                    });
                }
                (beta as i8, tc as i8)
            } else {
                // §7.4.3.3.1: beta/tc offsets inferred to 0 when the
                // deblocking filter is disabled at PPS level.
                (0, 0)
            };
            DeblockingFilterControl {
                override_enabled_flag,
                disabled_flag,
                beta_offset_div2,
                tc_offset_div2,
            }
        } else {
            DeblockingFilterControl::default()
        };

        let pps_scaling_list_data_present_flag = br.u1()? != 0;
        if pps_scaling_list_data_present_flag {
            // scaling_list_data() (§7.3.4) is a deferred parse target
            // shared with the SPS path; reject rather than misalign the
            // subsequent fields.
            return Err(PpsError::ScalingListUnsupported);
        }

        let lists_modification_present_flag = br.u1()? != 0;
        let log2_parallel_merge_level_minus2 = br.ue()?;
        let slice_segment_header_extension_present_flag = br.u1()? != 0;

        let pps_extension_present_flag = br.u1()? != 0;
        let opaque_tail = if pps_extension_present_flag {
            // The four extension flags, their bodies, and the
            // rbsp_trailing_bits are surfaced as an opaque tail; their
            // syntax structures are not materialised this round.
            Some(OpaqueTail::capture_at(br.bit_pos(), rbsp))
        } else {
            // Only rbsp_trailing_bits remains; consumed implicitly.
            None
        };

        Ok(Self {
            pps_id: pps_id_raw as u8,
            sps_id: sps_id_raw as u8,
            dependent_slice_segments_enabled_flag,
            output_flag_present_flag,
            num_extra_slice_header_bits,
            sign_data_hiding_enabled_flag,
            cabac_init_present_flag,
            num_ref_idx_l0_default_active_minus1: num_ref_idx_l0_default_active_minus1_raw as u8,
            num_ref_idx_l1_default_active_minus1: num_ref_idx_l1_default_active_minus1_raw as u8,
            init_qp_minus26,
            constrained_intra_pred_flag,
            transform_skip_enabled_flag,
            cu_qp_delta_enabled_flag,
            diff_cu_qp_delta_depth,
            pps_cb_qp_offset: pps_cb_qp_offset as i8,
            pps_cr_qp_offset: pps_cr_qp_offset as i8,
            pps_slice_chroma_qp_offsets_present_flag,
            weighted_pred_flag,
            weighted_bipred_flag,
            transquant_bypass_enabled_flag,
            tiles_enabled_flag,
            entropy_coding_sync_enabled_flag,
            tiles,
            loop_filter_across_tiles_enabled_flag,
            pps_loop_filter_across_slices_enabled_flag,
            deblocking_filter_control_present_flag,
            deblocking,
            pps_scaling_list_data_present_flag,
            lists_modification_present_flag,
            log2_parallel_merge_level_minus2,
            slice_segment_header_extension_present_flag,
            pps_extension_present_flag,
            opaque_tail,
        })
    }

    /// `init_qp = init_qp_minus26 + 26` per §7.4.3.3.1 — the
    /// per-picture base QP before any slice-header `slice_qp_delta`.
    pub fn init_qp(&self) -> i32 {
        self.init_qp_minus26 + 26
    }

    /// `num_ref_idx_l0_default_active = num_ref_idx_l0_default_active_minus1 + 1`.
    pub fn num_ref_idx_l0_default_active(&self) -> u8 {
        self.num_ref_idx_l0_default_active_minus1 + 1
    }

    /// `num_ref_idx_l1_default_active = num_ref_idx_l1_default_active_minus1 + 1`.
    pub fn num_ref_idx_l1_default_active(&self) -> u8 {
        self.num_ref_idx_l1_default_active_minus1 + 1
    }

    /// Number of tile columns: `num_tile_columns_minus1 + 1`.
    pub fn num_tile_columns(&self) -> u32 {
        self.tiles.num_tile_columns_minus1 + 1
    }

    /// Number of tile rows: `num_tile_rows_minus1 + 1`.
    pub fn num_tile_rows(&self) -> u32 {
        self.tiles.num_tile_rows_minus1 + 1
    }

    /// `Log2ParMrgLevel = log2_parallel_merge_level_minus2 + 2`
    /// (equation 7-37).
    pub fn log2_par_mrg_level(&self) -> u32 {
        self.log2_parallel_merge_level_minus2 + 2
    }

    /// Re-check `init_qp_minus26` against the exact §7.4.3.3.1 lower
    /// bound `−( 26 + QpBdOffsetY )` once the active SPS bit depth is
    /// known. `bit_depth_luma_minus8` is the value carried on the SPS
    /// this PPS activates (`QpBdOffsetY = 6 * bit_depth_luma_minus8`).
    /// The parser already verified the upper bound (+25) and the
    /// loosest lower bound (−74) during the parse.
    pub fn init_qp_in_range(&self, bit_depth_luma_minus8: u8) -> bool {
        let qp_bd_offset_y = 6 * bit_depth_luma_minus8 as i32;
        let lower = -(26 + qp_bd_offset_y);
        (lower..=25).contains(&self.init_qp_minus26)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nal::collect_nal_units;

    /// PPS RBSP body extracted from
    /// `docs/video/h265/fixtures/tiny-i-only-16x16-main/input.hevc`,
    /// after the Annex B start code and the two-byte NAL header have
    /// been removed. The wire PPS (NAL idx 2, type 34) was 6 bytes
    /// including the two-byte header (`44 01`); the body is the
    /// remaining 4 bytes. It carries no emulation-prevention bytes.
    const TINY_PPS_RBSP: &[u8] = &[0xC1, 0x73, 0xC0, 0x89];

    #[test]
    fn parses_tiny_fixture_pps() {
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS parse");
        // Trace cross-check
        // (docs/video/h265/fixtures/tiny-i-only-16x16-main/trace.txt,
        //  line 3 "PPS ..."):
        //   pps_id=0 sps_id=0 dependent_slice_segments=0
        //   output_flag_present=0 num_extra_slice_header_bits=0
        //   sign_data_hiding=1 cabac_init_present=0
        //   num_ref_idx_l0_default=1 num_ref_idx_l1_default=1 init_qp=26
        //   constrained_intra_pred=0 transform_skip_enabled=0
        //   cu_qp_delta_enabled=1 diff_cu_qp_delta_depth=0
        //   weighted_pred=0 weighted_bipred=0 transquant_bypass=0
        //   tiles_enabled=0 num_tile_columns=1 num_tile_rows=1
        //   entropy_coding_sync_enabled=0 loop_filter_across_tiles=1
        //   deblock_filter_override_enabled=0 lists_modification_present=0
        assert_eq!(pps.pps_id, 0);
        assert_eq!(pps.sps_id, 0);
        assert!(!pps.dependent_slice_segments_enabled_flag);
        assert!(!pps.output_flag_present_flag);
        assert_eq!(pps.num_extra_slice_header_bits, 0);
        assert!(pps.sign_data_hiding_enabled_flag);
        assert!(!pps.cabac_init_present_flag);
        assert_eq!(pps.num_ref_idx_l0_default_active_minus1, 0);
        assert_eq!(pps.num_ref_idx_l1_default_active_minus1, 0);
        assert_eq!(pps.num_ref_idx_l0_default_active(), 1);
        assert_eq!(pps.num_ref_idx_l1_default_active(), 1);
        assert_eq!(pps.init_qp_minus26, 0);
        assert_eq!(pps.init_qp(), 26);
        assert!(!pps.constrained_intra_pred_flag);
        assert!(!pps.transform_skip_enabled_flag);
        assert!(pps.cu_qp_delta_enabled_flag);
        assert_eq!(pps.diff_cu_qp_delta_depth, 0);
        assert_eq!(pps.pps_cb_qp_offset, 0);
        assert_eq!(pps.pps_cr_qp_offset, 0);
        assert!(!pps.pps_slice_chroma_qp_offsets_present_flag);
        assert!(!pps.weighted_pred_flag);
        assert!(!pps.weighted_bipred_flag);
        assert!(!pps.transquant_bypass_enabled_flag);
        assert!(!pps.tiles_enabled_flag);
        assert!(!pps.entropy_coding_sync_enabled_flag);
        // tiles_enabled=0 → single tile inferred per §7.4.3.3.1.
        assert_eq!(pps.num_tile_columns(), 1);
        assert_eq!(pps.num_tile_rows(), 1);
        assert!(pps.tiles.uniform_spacing_flag);
        assert!(pps.tiles.column_width_minus1.is_empty());
        assert!(pps.tiles.row_height_minus1.is_empty());
        assert!(pps.loop_filter_across_tiles_enabled_flag); // inferred true
        assert!(pps.pps_loop_filter_across_slices_enabled_flag);
        // deblocking control absent → defaults inferred.
        assert!(!pps.deblocking_filter_control_present_flag);
        assert!(!pps.deblocking.override_enabled_flag);
        assert!(!pps.deblocking.disabled_flag);
        assert_eq!(pps.deblocking.beta_offset_div2, 0);
        assert_eq!(pps.deblocking.tc_offset_div2, 0);
        assert!(!pps.pps_scaling_list_data_present_flag);
        assert!(!pps.lists_modification_present_flag);
        assert_eq!(pps.log2_parallel_merge_level_minus2, 0);
        assert!(!pps.slice_segment_header_extension_present_flag);
        // No PPS extension in this fixture → no opaque tail.
        assert!(!pps.pps_extension_present_flag);
        assert!(pps.opaque_tail.is_none());
    }

    /// End-to-end: pull the PPS NAL out of the raw Annex B stream
    /// through the walker, then parse its (already-unescaped, header-
    /// stripped) RBSP. The fixture's PPS NAL bytes (start code +
    /// `44 01 C1 73 C0 89`) are captured inline so the test does not
    /// depend on the `docs/` tree at run time.
    #[test]
    fn parses_tiny_fixture_pps_via_nal_walker() {
        // 4-byte start code + PPS NAL (type 34: header 0x44 0x01).
        let stream = [0x00, 0x00, 0x00, 0x01, 0x44, 0x01, 0xC1, 0x73, 0xC0, 0x89];
        let nals = collect_nal_units(&stream).expect("collect");
        assert_eq!(nals.len(), 1);
        let pps_nal = &nals[0];
        assert_eq!(pps_nal.header.nal_unit_type, 34);
        // `NalUnit::rbsp` already excludes the two header bytes and has
        // emulation-prevention stripped.
        let pps = PicParameterSet::parse(&pps_nal.rbsp).expect("PPS parse via walker");
        assert_eq!(pps.pps_id, 0);
        assert_eq!(pps.sps_id, 0);
        assert_eq!(pps.init_qp(), 26);
        assert!(pps.cu_qp_delta_enabled_flag);
    }

    /// Hand-assembled PPS exercising the tiles block with explicit
    /// (non-uniform) column/row spacing and the deblocking-control
    /// block with non-zero β / tC offsets.
    #[test]
    fn parses_tiles_and_deblocking_blocks() {
        // Build the RBSP bit-by-bit as a string, then pack MSB-first.
        let mut bits = String::new();
        // pps_pic_parameter_set_id = 0 → ue '1'
        bits += "1";
        // pps_seq_parameter_set_id = 0 → ue '1'
        bits += "1";
        // dependent_slice_segments_enabled_flag = 0
        bits += "0";
        // output_flag_present_flag = 0
        bits += "0";
        // num_extra_slice_header_bits = 0 → u(3)
        bits += "000";
        // sign_data_hiding_enabled_flag = 0
        bits += "0";
        // cabac_init_present_flag = 0
        bits += "0";
        // num_ref_idx_l0_default_active_minus1 = 0 → ue '1'
        bits += "1";
        // num_ref_idx_l1_default_active_minus1 = 0 → ue '1'
        bits += "1";
        // init_qp_minus26 = 0 → se '1'
        bits += "1";
        // constrained_intra_pred_flag = 0
        bits += "0";
        // transform_skip_enabled_flag = 0
        bits += "0";
        // cu_qp_delta_enabled_flag = 0
        bits += "0";
        // pps_cb_qp_offset = 0 → se '1'
        bits += "1";
        // pps_cr_qp_offset = 0 → se '1'
        bits += "1";
        // pps_slice_chroma_qp_offsets_present_flag = 0
        bits += "0";
        // weighted_pred_flag = 0
        bits += "0";
        // weighted_bipred_flag = 0
        bits += "0";
        // transquant_bypass_enabled_flag = 0
        bits += "0";
        // tiles_enabled_flag = 1
        bits += "1";
        // entropy_coding_sync_enabled_flag = 0
        bits += "0";
        //  tiles block:
        //   num_tile_columns_minus1 = 1 → ue codeNum 1 = '010'
        bits += "010";
        //   num_tile_rows_minus1 = 1 → ue '010'
        bits += "010";
        //   uniform_spacing_flag = 0
        bits += "0";
        //   column_width_minus1[0] = 2 → ue codeNum 2 = '011'
        bits += "011";
        //   row_height_minus1[0] = 3 → ue codeNum 3 = '00100'
        bits += "00100";
        //   loop_filter_across_tiles_enabled_flag = 0
        bits += "0";
        // pps_loop_filter_across_slices_enabled_flag = 1
        bits += "1";
        // deblocking_filter_control_present_flag = 1
        bits += "1";
        //   deblocking_filter_override_enabled_flag = 1
        bits += "1";
        //   pps_deblocking_filter_disabled_flag = 0
        bits += "0";
        //   pps_beta_offset_div2 = -2 → se codeNum 4 = '00101'
        bits += "00101";
        //   pps_tc_offset_div2 = 3 → se codeNum 5 = '00110'
        bits += "00110";
        // pps_scaling_list_data_present_flag = 0
        bits += "0";
        // lists_modification_present_flag = 0
        bits += "0";
        // log2_parallel_merge_level_minus2 = 0 → ue '1'
        bits += "1";
        // slice_segment_header_extension_present_flag = 0
        bits += "0";
        // pps_extension_present_flag = 0
        bits += "0";
        // rbsp_trailing_bits(): stop bit '1' then zero-pad to a byte.
        bits += "1";
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);

        let pps = PicParameterSet::parse(&rbsp).expect("tiles/deblock PPS parse");
        assert!(pps.tiles_enabled_flag);
        assert_eq!(pps.tiles.num_tile_columns_minus1, 1);
        assert_eq!(pps.tiles.num_tile_rows_minus1, 1);
        assert_eq!(pps.num_tile_columns(), 2);
        assert_eq!(pps.num_tile_rows(), 2);
        assert!(!pps.tiles.uniform_spacing_flag);
        assert_eq!(pps.tiles.column_width_minus1, vec![2]);
        assert_eq!(pps.tiles.row_height_minus1, vec![3]);
        assert!(!pps.loop_filter_across_tiles_enabled_flag);
        assert!(pps.pps_loop_filter_across_slices_enabled_flag);
        assert!(pps.deblocking_filter_control_present_flag);
        assert!(pps.deblocking.override_enabled_flag);
        assert!(!pps.deblocking.disabled_flag);
        assert_eq!(pps.deblocking.beta_offset_div2, -2);
        assert_eq!(pps.deblocking.tc_offset_div2, 3);
    }

    /// `pps_extension_present_flag == 1` surfaces an opaque tail.
    #[test]
    fn captures_extension_opaque_tail() {
        let mut bits = minimal_pps_prefix_bits();
        // pps_extension_present_flag = 1
        bits += "1";
        // Opaque extension body: 4 extension flags (all zero here) plus
        // rbsp_trailing_bits. We just append some bytes that the parser
        // must not interpret.
        bits += "0000"; // pps_range/multilayer/3d/scc flags
        bits += "0000"; // pps_extension_4bits = 0
        bits += "1"; // rbsp stop bit
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        let pps = PicParameterSet::parse(&rbsp).expect("ext PPS parse");
        assert!(pps.pps_extension_present_flag);
        let tail = pps.opaque_tail.as_ref().expect("opaque extension tail");
        assert!(!tail.bytes.is_empty());
    }

    /// `pps_scaling_list_data_present_flag == 1` is rejected
    /// (scaling_list_data parse deferred).
    #[test]
    fn rejects_scaling_list_present() {
        let mut bits = minimal_pps_prefix_bits_until_scaling_list();
        // pps_scaling_list_data_present_flag = 1
        bits += "1";
        bits += "1"; // a stop bit so the buffer isn't empty
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        assert_eq!(
            PicParameterSet::parse(&rbsp),
            Err(PpsError::ScalingListUnsupported)
        );
    }

    /// `pps_pic_parameter_set_id` out of range (> 63) is rejected.
    #[test]
    fn rejects_pps_id_out_of_range() {
        // ue codeNum 64 = leadingZeroBits 6, suffix = 64+1-(2^6-1)=1 →
        // '0000001000001'. codeNum = 2^6 - 1 + 1 = 64 (> 63).
        let mut bits = String::from("0000001000001");
        bits += "1"; // pad to ensure non-empty buffer for any further read
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        assert_eq!(
            PicParameterSet::parse(&rbsp),
            Err(PpsError::ValueOutOfRange {
                field: "pps_pic_parameter_set_id",
                got: 64,
            })
        );
    }

    /// A truncated PPS RBSP surfaces `Truncated`.
    #[test]
    fn rejects_truncated() {
        // Just the first two ue() ids then end-of-buffer mid-flag-block.
        let rbsp = [0xC0]; // pps_id=0 ('1'), sps_id=0 ('1'), then '000000'
        let err = PicParameterSet::parse(&rbsp).unwrap_err();
        assert_eq!(err, PpsError::Truncated);
    }

    #[test]
    fn init_qp_range_check_uses_sps_bit_depth() {
        // init_qp_minus26 = -70 is within the loosest -74 bound the
        // parser accepts but only legal for high bit depths (where
        // QpBdOffsetY is large enough). Build a complete PPS carrying
        // -70 and exercise the SPS-aware range helper on the result.
        let mut bits = String::new();
        bits += "1"; // pps_id = 0
        bits += "1"; // sps_id = 0
        bits += "0"; // dependent_slice_segments_enabled_flag
        bits += "0"; // output_flag_present_flag
        bits += "000"; // num_extra_slice_header_bits
        bits += "0"; // sign_data_hiding_enabled_flag
        bits += "0"; // cabac_init_present_flag
        bits += "1"; // num_ref_idx_l0_default_active_minus1 = 0
        bits += "1"; // num_ref_idx_l1_default_active_minus1 = 0
                     // init_qp_minus26 = -70 → se value -70: codeNum = 2*70 = 140.
                     // ue(140): leadingZeroBits = 7, suffix = 140+1-128 = 13.
        bits += "0000000"; // 7 leading zeros
        bits += "1"; // terminator
        bits += "0001101"; // 7-bit suffix = 13 → codeNum = 127 + 13 = 140
        bits += "0"; // constrained_intra_pred_flag
        bits += "0"; // transform_skip_enabled_flag
        bits += "0"; // cu_qp_delta_enabled_flag = 0
        bits += "1"; // pps_cb_qp_offset = 0 (se '1')
        bits += "1"; // pps_cr_qp_offset = 0 (se '1')
        bits += "0"; // pps_slice_chroma_qp_offsets_present_flag
        bits += "0"; // weighted_pred_flag
        bits += "0"; // weighted_bipred_flag
        bits += "0"; // transquant_bypass_enabled_flag
        bits += "0"; // tiles_enabled_flag = 0
        bits += "0"; // entropy_coding_sync_enabled_flag
        bits += "1"; // pps_loop_filter_across_slices_enabled_flag
        bits += "0"; // deblocking_filter_control_present_flag = 0
        bits += "0"; // pps_scaling_list_data_present_flag = 0
        bits += "0"; // lists_modification_present_flag
        bits += "1"; // log2_parallel_merge_level_minus2 = 0 (ue '1')
        bits += "0"; // slice_segment_header_extension_present_flag
        bits += "0"; // pps_extension_present_flag = 0
        bits += "1"; // rbsp_trailing_bits stop bit
        while bits.len() % 8 != 0 {
            bits += "0";
        }
        let rbsp = pack_bits(&bits);
        let pps = PicParameterSet::parse(&rbsp).expect("low-init-qp PPS parse");
        assert_eq!(pps.init_qp_minus26, -70);
        assert_eq!(pps.init_qp(), -44);
        // 8-bit (minus8=0): QpBdOffsetY=0, lower bound -26 → out of range.
        assert!(!pps.init_qp_in_range(0));
        // 12-bit (minus8=4): QpBdOffsetY=24, lower bound -50 → out.
        assert!(!pps.init_qp_in_range(4));
        // 16-bit (minus8=8): QpBdOffsetY=48, lower bound -74 → in range.
        assert!(pps.init_qp_in_range(8));
    }

    // --- test helpers --------------------------------------------------

    /// Pack an MSB-first bit string (must be a whole number of bytes)
    /// into a byte vector.
    fn pack_bits(bits: &str) -> Vec<u8> {
        assert_eq!(bits.len() % 8, 0, "bit string must be byte-aligned");
        bits.as_bytes()
            .chunks(8)
            .map(|chunk| chunk.iter().fold(0u8, |acc, &b| (acc << 1) | (b - b'0')))
            .collect()
    }

    /// Minimal PPS prefix bits up to (but not including)
    /// `pps_extension_present_flag`, with every field set to the
    /// simplest legal value (no tiles, no deblocking control, no
    /// scaling list). The caller appends the extension flag + tail.
    fn minimal_pps_prefix_bits() -> String {
        let mut bits = minimal_pps_prefix_bits_until_scaling_list();
        // pps_scaling_list_data_present_flag = 0
        bits += "0";
        // lists_modification_present_flag = 0
        bits += "0";
        // log2_parallel_merge_level_minus2 = 0 → ue '1'
        bits += "1";
        // slice_segment_header_extension_present_flag = 0
        bits += "0";
        bits
    }

    /// Minimal PPS prefix bits up to (but not including)
    /// `pps_scaling_list_data_present_flag`.
    fn minimal_pps_prefix_bits_until_scaling_list() -> String {
        let mut bits = String::new();
        bits += "1"; // pps_id = 0
        bits += "1"; // sps_id = 0
        bits += "0"; // dependent_slice_segments_enabled_flag
        bits += "0"; // output_flag_present_flag
        bits += "000"; // num_extra_slice_header_bits
        bits += "0"; // sign_data_hiding_enabled_flag
        bits += "0"; // cabac_init_present_flag
        bits += "1"; // num_ref_idx_l0_default_active_minus1 = 0
        bits += "1"; // num_ref_idx_l1_default_active_minus1 = 0
        bits += "1"; // init_qp_minus26 = 0 (se '1')
        bits += "0"; // constrained_intra_pred_flag
        bits += "0"; // transform_skip_enabled_flag
        bits += "0"; // cu_qp_delta_enabled_flag = 0 (depth not signalled)
        bits += "1"; // pps_cb_qp_offset = 0 (se '1')
        bits += "1"; // pps_cr_qp_offset = 0 (se '1')
        bits += "0"; // pps_slice_chroma_qp_offsets_present_flag
        bits += "0"; // weighted_pred_flag
        bits += "0"; // weighted_bipred_flag
        bits += "0"; // transquant_bypass_enabled_flag
        bits += "0"; // tiles_enabled_flag = 0
        bits += "0"; // entropy_coding_sync_enabled_flag
        bits += "1"; // pps_loop_filter_across_slices_enabled_flag
        bits += "0"; // deblocking_filter_control_present_flag = 0
        bits
    }
}
