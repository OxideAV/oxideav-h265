//! Slice segment header parser per ITU-T Rec. H.265 §7.3.6.1.
//!
//! Round-6 scope: parse the `slice_segment_header()` syntax structure
//! (§7.3.6.1) for an independent slice segment, materialising every
//! field that does **not** require decoded-picture-buffer state to
//! interpret. The parse takes the activated SPS and PPS as context
//! because several field widths and presence gates are derived from
//! them (the `slice_segment_address` width is
//! `Ceil( Log2( PicSizeInCtbsY ) )`, the `slice_pic_order_cnt_lsb`
//! width is `log2_max_pic_order_cnt_lsb_minus4 + 4`, the SAO / MVP
//! gates come from the SPS, and the tiles / entropy-sync entry-point
//! block comes from the PPS).
//!
//! ## What this round materialises
//!
//! * `first_slice_segment_in_pic_flag`, `no_output_of_prior_pics_flag`
//!   (IRAP only), `slice_pic_parameter_set_id`.
//! * For non-first slice segments: `dependent_slice_segment_flag`
//!   (only when `dependent_slice_segments_enabled_flag`) and
//!   `slice_segment_address` (`u(v)`, width
//!   `Ceil( Log2( PicSizeInCtbsY ) )`).
//! * For independent slice segments (`!dependent_slice_segment_flag`):
//!   the `slice_reserved_flag[]` block, `slice_type`,
//!   `pic_output_flag` (only when `output_flag_present_flag`),
//!   `colour_plane_id` (only when `separate_colour_plane_flag`),
//!   `slice_temporal_mvp_enabled_flag` (only when
//!   `sps_temporal_mvp_enabled_flag`), the SAO luma / chroma gates,
//!   `slice_qp_delta` (`se(v)`), the chroma QP offsets, the
//!   deblocking-filter override block, and
//!   `slice_loop_filter_across_slices_enabled_flag`.
//! * The entry-point-offset block (`num_entry_point_offsets`,
//!   `offset_len_minus1`, `entry_point_offset_minus1[]`) when
//!   `tiles_enabled_flag || entropy_coding_sync_enabled_flag`.
//! * The slice-segment-header extension block when
//!   `slice_segment_header_extension_present_flag`.
//! * `byte_alignment()` consumed to the next byte boundary, so the
//!   reported [`SliceSegmentHeader::byte_offset_to_slice_data`] points
//!   at the first byte of `slice_segment_data()`.
//!
//! ## What this round defers (surfaced, not decoded)
//!
//! Three points need state this round does not carry:
//!
//! * The **non-IDR picture-order-count + reference-picture-set block**
//!   (`slice_pic_order_cnt_lsb`, `short_term_ref_pic_set_sps_flag`,
//!   the inline `st_ref_pic_set()`, the long-term block) needs the
//!   SPS short-term-RPS parser to be re-entered for the in-line
//!   `stRpsIdx == num_short_term_ref_pic_sets` case, which is not yet
//!   exposed publicly. When the current NAL unit is **not** an IDR
//!   (`nal_unit_type != IDR_W_RADL && != IDR_N_LP`), the parser stops
//!   right after `colour_plane_id` and surfaces the remainder as
//!   [`SliceSegmentHeader::opaque_tail`].
//! * The **P / B reference-list / weighted-prediction sub-structures**
//!   (`ref_pic_lists_modification()` §7.3.6.2 and `pred_weight_table()`
//!   §7.3.6.3) need DPB-derived `NumPicTotalCurr` / `RefPicList`
//!   values. When `slice_type` is P or B the parser materialises the
//!   common P/B fields up to (but not including) the point where those
//!   sub-structures would begin, then surfaces the remainder as the
//!   opaque tail. The §7.3.6.2 syntax structure itself is implemented
//!   as a standalone parser ([`RefPicListsModification::parse`]) so a
//!   future round that wires up the §7.4.7.2 `NumPicTotalCurr`
//!   derivation can decode the reference-picture-list-modification
//!   block in place; the implicit `RefPicListTempX` derivation of
//!   §8.3.4 stays the consumer's responsibility.
//!
//! Independent **I-slice IDR** segments — the dominant case for the
//! intra-only fixtures this rebuild targets — are parsed end to end
//! through `byte_alignment()`.

use crate::bitreader::{BitReader, BitReaderError};
use crate::pps::PicParameterSet;
use crate::sps::{OpaqueTail, SeqParameterSet, ShortTermRefPicSet, SpsError};

/// `nal_unit_type` value `BLA_W_LP` (Table 7-1). The IRAP range used by
/// the `no_output_of_prior_pics_flag` gate is `BLA_W_LP..=RSV_IRAP_VCL23`.
pub const BLA_W_LP: u8 = 16;
/// `nal_unit_type` value `IDR_W_RADL` (Table 7-1).
pub const IDR_W_RADL: u8 = 19;
/// `nal_unit_type` value `IDR_N_LP` (Table 7-1).
pub const IDR_N_LP: u8 = 20;
/// `nal_unit_type` value `RSV_IRAP_VCL23` (Table 7-1) — the inclusive
/// upper bound of the IRAP NAL-unit-type range.
pub const RSV_IRAP_VCL23: u8 = 23;

/// `slice_type` enumeration per Table 7-7.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    /// B slice (`slice_type == 0`).
    B,
    /// P slice (`slice_type == 1`).
    P,
    /// I slice (`slice_type == 2`).
    I,
}

impl SliceType {
    /// Map the raw `ue(v)` `slice_type` value to the enum, rejecting
    /// any value outside `0..=2`.
    fn from_raw(v: u32) -> Result<Self, SliceError> {
        match v {
            0 => Ok(Self::B),
            1 => Ok(Self::P),
            2 => Ok(Self::I),
            other => Err(SliceError::ValueOutOfRange {
                field: "slice_type",
                got: other as i64,
            }),
        }
    }

    /// True for P and B slices (the slice types that signal reference
    /// lists, weighted prediction, etc.).
    pub fn is_inter(self) -> bool {
        matches!(self, Self::P | Self::B)
    }
}

/// Errors that can arise while parsing a slice segment header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceError {
    /// The RBSP ran out of bits before the header was fully parsed.
    Truncated,
    /// A syntax element's parsed value was outside the legal range
    /// specified for it in §7.4.7.1.
    ValueOutOfRange {
        /// Name of the offending syntax element.
        field: &'static str,
        /// The (illegal) value as an `i64` (covers both `ue(v)` and
        /// `se(v)` elements).
        got: i64,
    },
    /// An unexpected bitstream-level error surfaced from the reader.
    Bitstream(BitReaderError),
    /// The in-line `st_ref_pic_set(num_short_term_ref_pic_sets)` parse
    /// (invoked from §7.3.6.1 when `short_term_ref_pic_set_sps_flag == 0`)
    /// failed.
    InlineShortTermRpsParse(SpsError),
}

impl core::fmt::Display for SliceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => f.write_str("slice segment header RBSP truncated"),
            Self::ValueOutOfRange { field, got } => {
                write!(f, "slice header syntax element {field} out of range: {got}")
            }
            Self::Bitstream(e) => write!(f, "bitstream error during slice header parse: {e}"),
            Self::InlineShortTermRpsParse(e) => {
                write!(f, "in-line slice-header st_ref_pic_set parse failed: {e}")
            }
        }
    }
}

impl std::error::Error for SliceError {}

impl From<BitReaderError> for SliceError {
    fn from(e: BitReaderError) -> Self {
        match e {
            BitReaderError::EndOfBuffer => Self::Truncated,
            other => Self::Bitstream(other),
        }
    }
}

impl From<SpsError> for SliceError {
    fn from(e: SpsError) -> Self {
        match e {
            SpsError::Truncated => Self::Truncated,
            SpsError::Bitstream(b) => Self::Bitstream(b),
            other => Self::InlineShortTermRpsParse(other),
        }
    }
}

/// Deblocking-filter override block carried in the slice header
/// (§7.3.6.1, gated by `deblocking_filter_override_flag`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceDeblocking {
    /// `slice_deblocking_filter_disabled_flag`. Inferred to
    /// `pps_deblocking_filter_disabled_flag` when the override block is
    /// absent (§7.4.7.1).
    pub disabled_flag: bool,
    /// `slice_beta_offset_div2` (`se(v)`, range −6..=6). Inferred to
    /// `pps_beta_offset_div2` when absent.
    pub beta_offset_div2: i8,
    /// `slice_tc_offset_div2` (`se(v)`, range −6..=6). Inferred to
    /// `pps_tc_offset_div2` when absent.
    pub tc_offset_div2: i8,
}

/// One long-term reference picture entry signalled in the slice
/// header (§7.3.6.1). For the first `num_long_term_sps` entries the
/// `lt_idx_sps` indexes the SPS's long-term-ref-pic table; for the
/// remaining `num_long_term_pics` entries the slice header carries
/// the POC LSB and `used_by_curr_pic_lt_flag` directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceLongTermRefPic {
    /// Source of the entry: SPS table or in-slice signalling.
    pub source: SliceLongTermRefPicSource,
    /// `delta_poc_msb_present_flag[i]`.
    pub delta_poc_msb_present_flag: bool,
    /// `delta_poc_msb_cycle_lt[i]`. Inferred to 0 when
    /// [`Self::delta_poc_msb_present_flag`] is false (§7.4.7.1).
    pub delta_poc_msb_cycle_lt: u32,
}

/// Source of one [`SliceLongTermRefPic`] entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceLongTermRefPicSource {
    /// First `num_long_term_sps` entries: `lt_idx_sps[i]` indexes the
    /// SPS's long-term-ref-pic-poc table. The `u(v)` width is
    /// `Ceil(Log2(num_long_term_ref_pics_sps))` bits; the value 0 is
    /// inferred when `num_long_term_ref_pics_sps == 1`.
    Sps {
        /// `lt_idx_sps[i]` value (or 0 when inferred).
        lt_idx_sps: u32,
    },
    /// Remaining `num_long_term_pics` entries: the POC LSB and
    /// `used_by_curr_pic_lt_flag` are signalled directly in the slice
    /// header.
    InSlice {
        /// `poc_lsb_lt[i]` (`u(v)`, width
        /// `log2_max_pic_order_cnt_lsb_minus4 + 4`).
        poc_lsb_lt: u32,
        /// `used_by_curr_pic_lt_flag[i]`.
        used_by_curr_pic_lt_flag: bool,
    },
}

/// One entry-point offset entry (`tiles_enabled_flag ||
/// entropy_coding_sync_enabled_flag`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntryPointOffsets {
    /// `num_entry_point_offsets` (`ue(v)`). The number of subsets of
    /// slice-segment data is this value plus one.
    pub num_entry_point_offsets: u32,
    /// `offset_len_minus1` (`ue(v)`, range 0..=31). Each
    /// `entry_point_offset_minus1[i]` is `offset_len_minus1 + 1` bits.
    /// Only meaningful when `num_entry_point_offsets > 0`.
    pub offset_len_minus1: u8,
}

/// Parsed `ref_pic_lists_modification()` syntax structure
/// (ITU-T Rec. H.265 §7.3.6.2 / §7.4.7.2).
///
/// The structure is signalled in the slice header when
/// `lists_modification_present_flag == 1 && NumPicTotalCurr > 1`
/// (§7.3.6.1 gate). It carries a per-list "explicit list" override of
/// the implicit `RefPicList0` / `RefPicList1` derivation of §8.3.4: the
/// `list_entry_lX[i]` value is the index of the reference picture in
/// `RefPicListTempX` to place at position `i` of `RefPicListX`. The
/// `RefPicListTempX` derivation itself is part of §8.3.4 (DPB-driven)
/// and is **not** performed by this parser — this struct surfaces only
/// the on-wire syntax elements and applies the §7.4.7.2 inference and
/// range checks.
///
/// Width of each `list_entry_lX[i]` is `Ceil( Log2( NumPicTotalCurr ) )`
/// bits and the value must be in `0 ..= NumPicTotalCurr - 1`
/// (§7.4.7.2). When `ref_pic_list_modification_flag_lX == 0` the
/// corresponding entry list is empty; the implicit derivation of
/// §8.3.4 applies (each `list_entry_lX[i]` is inferred to 0 per the
/// §7.4.7.2 paragraph "When the syntax element list_entry_lX[i] is
/// not present in the slice header, it is inferred to be equal to 0",
/// but the inference is exercised by §8.3.4, not surfaced here).
///
/// The list-1 fields are present only when `slice_type == B`
/// (§7.3.6.2 syntax). For a P slice the `list_entry_l1` vector is
/// always empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefPicListsModification {
    /// `ref_pic_list_modification_flag_l0` (`u(1)`).
    pub ref_pic_list_modification_flag_l0: bool,
    /// `list_entry_l0[i]` for `i = 0 ..= num_ref_idx_l0_active_minus1`.
    /// Empty when `ref_pic_list_modification_flag_l0 == 0`.
    pub list_entry_l0: Vec<u32>,
    /// `ref_pic_list_modification_flag_l1` (`u(1)`). `None` when the
    /// slice is not a B slice (the field is not signalled).
    pub ref_pic_list_modification_flag_l1: Option<bool>,
    /// `list_entry_l1[i]` for `i = 0 ..= num_ref_idx_l1_active_minus1`.
    /// Empty for P slices and when `ref_pic_list_modification_flag_l1
    /// == 0`.
    pub list_entry_l1: Vec<u32>,
}

impl RefPicListsModification {
    /// Parse `ref_pic_lists_modification()` (§7.3.6.2) from the current
    /// bit position of `br`.
    ///
    /// * `slice_type` — the active slice type. Per §7.3.6.2 the L1
    ///   block (`ref_pic_list_modification_flag_l1` /
    ///   `list_entry_l1[]`) is only signalled for B slices. For an
    ///   I slice the structure is never present at all (the §7.3.6.1
    ///   gate `lists_modification_present_flag && NumPicTotalCurr > 1`
    ///   sits inside the inter-slice `slice_type != I` branch), so
    ///   the parser rejects `SliceType::I` up front.
    /// * `num_ref_idx_l0_active_minus1` — the *active* value (after
    ///   the `num_ref_idx_active_override_flag` override, falling back
    ///   to `num_ref_idx_l0_default_active_minus1` from the PPS). Per
    ///   §7.4.7.1 the value is in `0 ..= 14`.
    /// * `num_ref_idx_l1_active_minus1` — same, for L1; ignored for P
    ///   slices.
    /// * `num_pic_total_curr` — `NumPicTotalCurr` (§7.4.7.2 /
    ///   equation 7-57). The caller derives it from the active RPS;
    ///   the §7.3.6.1 gate guarantees `num_pic_total_curr > 1` at the
    ///   point of call. This parser rejects `num_pic_total_curr <= 1`
    ///   (the §7.3.6.1 gate would have prevented the call); a call
    ///   with `num_pic_total_curr == 0` would also imply a bitstream
    ///   conformance failure per §7.4.7.1 ("when the current picture
    ///   contains a P or B slice, the value of NumPicTotalCurr shall
    ///   not be equal to 0"). Each `list_entry_lX[i]` is read as
    ///   `u(v)` of width `Ceil( Log2( num_pic_total_curr ) )` bits
    ///   and range-checked against `num_pic_total_curr - 1`.
    pub fn parse(
        br: &mut BitReader<'_>,
        slice_type: SliceType,
        num_ref_idx_l0_active_minus1: u8,
        num_ref_idx_l1_active_minus1: u8,
        num_pic_total_curr: u32,
    ) -> Result<Self, SliceError> {
        if slice_type == SliceType::I {
            return Err(SliceError::ValueOutOfRange {
                field: "ref_pic_lists_modification/slice_type",
                got: 2,
            });
        }
        if num_pic_total_curr <= 1 {
            return Err(SliceError::ValueOutOfRange {
                field: "ref_pic_lists_modification/NumPicTotalCurr",
                got: num_pic_total_curr as i64,
            });
        }
        // §7.4.7.1 ranges `num_ref_idx_lX_active_minus1` at 0..=14;
        // defensively cap the per-list loop length so a corrupted call
        // can't drive an unbounded allocation. (The cap matches the
        // spec maximum; a value above 14 would be rejected by the
        // §7.4.7.1 slice-header parse before reaching here.)
        if num_ref_idx_l0_active_minus1 > 14 {
            return Err(SliceError::ValueOutOfRange {
                field: "num_ref_idx_l0_active_minus1",
                got: num_ref_idx_l0_active_minus1 as i64,
            });
        }
        if slice_type == SliceType::B && num_ref_idx_l1_active_minus1 > 14 {
            return Err(SliceError::ValueOutOfRange {
                field: "num_ref_idx_l1_active_minus1",
                got: num_ref_idx_l1_active_minus1 as i64,
            });
        }

        let entry_bits = ceil_log2(num_pic_total_curr);
        let max_entry = num_pic_total_curr - 1;

        let ref_pic_list_modification_flag_l0 = br.u1()? != 0;
        let mut list_entry_l0: Vec<u32> = Vec::new();
        if ref_pic_list_modification_flag_l0 {
            let n = num_ref_idx_l0_active_minus1 as u32 + 1;
            list_entry_l0.reserve(n as usize);
            for _ in 0..n {
                let v = br.u(entry_bits)?;
                if v > max_entry {
                    return Err(SliceError::ValueOutOfRange {
                        field: "list_entry_l0",
                        got: v as i64,
                    });
                }
                list_entry_l0.push(v);
            }
        }

        let (ref_pic_list_modification_flag_l1, list_entry_l1) = if slice_type == SliceType::B {
            let flag = br.u1()? != 0;
            let mut entries: Vec<u32> = Vec::new();
            if flag {
                let n = num_ref_idx_l1_active_minus1 as u32 + 1;
                entries.reserve(n as usize);
                for _ in 0..n {
                    let v = br.u(entry_bits)?;
                    if v > max_entry {
                        return Err(SliceError::ValueOutOfRange {
                            field: "list_entry_l1",
                            got: v as i64,
                        });
                    }
                    entries.push(v);
                }
            }
            (Some(flag), entries)
        } else {
            (None, Vec::new())
        };

        Ok(Self {
            ref_pic_list_modification_flag_l0,
            list_entry_l0,
            ref_pic_list_modification_flag_l1,
            list_entry_l1,
        })
    }
}

/// Parsed slice segment header per §7.3.6.1.
///
/// Fields that this round defers (the non-IDR POC/RPS block, the P/B
/// reference-list / weighted-prediction sub-structures) are absent from
/// the materialised struct; when one of those points is reached the
/// remainder of the header is surfaced via [`Self::opaque_tail`] and
/// the corresponding `Option` fields stay `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceSegmentHeader {
    /// `first_slice_segment_in_pic_flag`.
    pub first_slice_segment_in_pic_flag: bool,
    /// `no_output_of_prior_pics_flag`. `None` when not present (the NAL
    /// unit is not an IRAP picture).
    pub no_output_of_prior_pics_flag: Option<bool>,
    /// `slice_pic_parameter_set_id` (`ue(v)`, range 0..=63).
    pub slice_pic_parameter_set_id: u8,
    /// `dependent_slice_segment_flag`. Inferred to false when the slice
    /// is the first segment of the picture or when
    /// `dependent_slice_segments_enabled_flag` is 0 (§7.4.7.1).
    pub dependent_slice_segment_flag: bool,
    /// `slice_segment_address` (`u(v)`). Inferred to 0 when not present
    /// (the first slice segment of the picture).
    pub slice_segment_address: u32,
    /// `slice_reserved_flag[]` — `num_extra_slice_header_bits` raw
    /// flags. Decoders ignore the value; carried for completeness.
    /// Empty for dependent slice segments and when the count is 0.
    pub slice_reserved_flags: Vec<bool>,
    /// `slice_type` per Table 7-7. `None` for dependent slice segments
    /// (the value is inherited from the associated independent slice
    /// segment, which this struct does not resolve).
    pub slice_type: Option<SliceType>,
    /// `pic_output_flag`. Inferred to true when not present (§7.4.7.1).
    pub pic_output_flag: bool,
    /// `colour_plane_id` (`u(2)`). `None` when not present
    /// (`separate_colour_plane_flag == 0`).
    pub colour_plane_id: Option<u8>,
    /// `slice_pic_order_cnt_lsb` (`u(v)`, width
    /// `log2_max_pic_order_cnt_lsb_minus4 + 4` bits). `None` when the
    /// current NAL unit is an IDR — IDR pictures have no slice POC LSB
    /// (the POC is reset to 0 per §8.3.1) — or when the parser stopped
    /// at the deferred P/B body before reaching this point.
    pub slice_pic_order_cnt_lsb: Option<u32>,
    /// `short_term_ref_pic_set_sps_flag`. `None` for IDR slices and
    /// for headers that stopped before this point.
    pub short_term_ref_pic_set_sps_flag: Option<bool>,
    /// In-line `st_ref_pic_set(num_short_term_ref_pic_sets)` parsed from
    /// the slice header itself (only when
    /// `short_term_ref_pic_set_sps_flag == 0`).
    pub inline_short_term_ref_pic_set: Option<ShortTermRefPicSet>,
    /// `short_term_ref_pic_set_idx` (`u(v)`, width
    /// `Ceil(Log2(num_short_term_ref_pic_sets))`). `None` when the SPS
    /// in-line form is used, when `num_short_term_ref_pic_sets <= 1`
    /// (the value is inferred to 0), or for IDR slices.
    pub short_term_ref_pic_set_idx: Option<u32>,
    /// `num_long_term_sps` (`ue(v)`). `None` when the long-term-ref-pic
    /// block is absent (no `long_term_ref_pics_present_flag` on the SPS
    /// or IDR slice); 0 (with the SPS gate satisfied but
    /// `num_long_term_ref_pics_sps == 0`).
    pub num_long_term_sps: Option<u32>,
    /// `num_long_term_pics` (`ue(v)`). `None` when the long-term-ref-pic
    /// block is absent.
    pub num_long_term_pics: Option<u32>,
    /// Per-entry long-term ref pic block (§7.3.6.1), length
    /// `num_long_term_sps + num_long_term_pics`. Empty when the block
    /// is absent.
    pub long_term_ref_pics: Vec<SliceLongTermRefPic>,
    /// `slice_temporal_mvp_enabled_flag`. Inferred to false when not
    /// present (`sps_temporal_mvp_enabled_flag == 0`) (§7.4.7.1).
    pub slice_temporal_mvp_enabled_flag: bool,
    /// `slice_sao_luma_flag`. Inferred to false when not present
    /// (`sample_adaptive_offset_enabled_flag == 0`).
    pub slice_sao_luma_flag: bool,
    /// `slice_sao_chroma_flag`. Inferred to false when not present.
    pub slice_sao_chroma_flag: bool,
    /// `slice_qp_delta` (`se(v)`). `None` when the parser stopped before
    /// reaching it (a deferred non-IDR or P/B body).
    pub slice_qp_delta: Option<i32>,
    /// `slice_cb_qp_offset` (`se(v)`, range −12..=12). Inferred to 0
    /// when not present (`pps_slice_chroma_qp_offsets_present_flag == 0`).
    pub slice_cb_qp_offset: i8,
    /// `slice_cr_qp_offset` (`se(v)`, range −12..=12). Inferred to 0
    /// when not present.
    pub slice_cr_qp_offset: i8,
    /// Deblocking-filter values, carrying the §7.4.7.1 inferred
    /// defaults when the slice override block is absent. `None` when
    /// the parser stopped before this point.
    pub deblocking: Option<SliceDeblocking>,
    /// `slice_loop_filter_across_slices_enabled_flag`. Inferred to
    /// `pps_loop_filter_across_slices_enabled_flag` when not present.
    /// `None` when the parser stopped before this point.
    pub slice_loop_filter_across_slices_enabled_flag: Option<bool>,
    /// Entry-point-offset block. `None` when neither tiles nor
    /// entropy-coding-sync are enabled (the block is absent), or when
    /// the parser stopped before this point.
    pub entry_point_offsets: Option<EntryPointOffsets>,
    /// `slice_segment_header_extension_length` (`ue(v)`). `None` when
    /// `slice_segment_header_extension_present_flag == 0` or the parser
    /// stopped before this point.
    pub slice_segment_header_extension_length: Option<u32>,
    /// Byte offset, from the start of the RBSP, of the first byte of
    /// `slice_segment_data()` — i.e. the position immediately after
    /// `byte_alignment()`. `None` when the header was not parsed all
    /// the way to `byte_alignment()` (a deferred body).
    pub byte_offset_to_slice_data: Option<usize>,
    /// Opaque suffix of the slice-header RBSP. Populated when the
    /// parser reaches a deferred body (the non-IDR POC/RPS block or a
    /// P/B reference-list / weighted-prediction sub-structure); carries
    /// the still-unparsed RBSP bytes and the start-bit offset. `None`
    /// when the header was parsed to completion.
    pub opaque_tail: Option<OpaqueTail>,
}

impl SliceSegmentHeader {
    /// Parse `slice_segment_header()` from the first bit of the
    /// (already-unescaped) slice-segment-layer RBSP body — i.e. after
    /// the two-byte NAL header has been removed (see
    /// [`crate::nal::NalUnit`]).
    ///
    /// * `nal_unit_type` is the value from the NAL header; it gates
    ///   both `no_output_of_prior_pics_flag` (IRAP range) and the
    ///   non-IDR POC/RPS block.
    /// * `sps` is the activated SPS, `pps` the activated PPS. The
    ///   caller resolves `slice_pic_parameter_set_id` to the right PPS
    ///   and that PPS's `pps_seq_parameter_set_id` to the right SPS;
    ///   this parser uses the supplied pair for the field widths and
    ///   presence gates.
    pub fn parse(
        rbsp: &[u8],
        nal_unit_type: u8,
        sps: &SeqParameterSet,
        pps: &PicParameterSet,
    ) -> Result<Self, SliceError> {
        let mut br = BitReader::new(rbsp);

        let first_slice_segment_in_pic_flag = br.u1()? != 0;

        let no_output_of_prior_pics_flag = if (BLA_W_LP..=RSV_IRAP_VCL23).contains(&nal_unit_type) {
            Some(br.u1()? != 0)
        } else {
            None
        };

        let slice_pic_parameter_set_id = br.ue()?;
        if slice_pic_parameter_set_id > 63 {
            return Err(SliceError::ValueOutOfRange {
                field: "slice_pic_parameter_set_id",
                got: slice_pic_parameter_set_id as i64,
            });
        }
        let slice_pic_parameter_set_id = slice_pic_parameter_set_id as u8;

        // §7.3.6.1: dependent_slice_segment_flag / slice_segment_address
        // only appear for non-first slice segments.
        let mut dependent_slice_segment_flag = false;
        let mut slice_segment_address = 0u32;
        if !first_slice_segment_in_pic_flag {
            if pps.dependent_slice_segments_enabled_flag {
                dependent_slice_segment_flag = br.u1()? != 0;
            }
            // slice_segment_address width is Ceil( Log2( PicSizeInCtbsY ) )
            // bits (§7.4.7.1).
            let addr_bits = ceil_log2(pic_size_in_ctbs_y(sps));
            slice_segment_address = br.u(addr_bits)?;
            if slice_segment_address >= pic_size_in_ctbs_y(sps) {
                return Err(SliceError::ValueOutOfRange {
                    field: "slice_segment_address",
                    got: slice_segment_address as i64,
                });
            }
        }

        // Defaults / inferences (§7.4.7.1).
        let mut slice_reserved_flags = Vec::new();
        let mut slice_type = None;
        let mut pic_output_flag = true;
        let mut colour_plane_id = None;
        let mut slice_pic_order_cnt_lsb: Option<u32> = None;
        let mut short_term_ref_pic_set_sps_flag: Option<bool> = None;
        let mut inline_short_term_ref_pic_set: Option<ShortTermRefPicSet> = None;
        let mut short_term_ref_pic_set_idx: Option<u32> = None;
        let mut num_long_term_sps: Option<u32> = None;
        let mut num_long_term_pics: Option<u32> = None;
        let mut long_term_ref_pics: Vec<SliceLongTermRefPic> = Vec::new();
        let mut slice_temporal_mvp_enabled_flag = false;

        if !dependent_slice_segment_flag {
            for _ in 0..pps.num_extra_slice_header_bits {
                slice_reserved_flags.push(br.u1()? != 0);
            }
            let st = SliceType::from_raw(br.ue()?)?;
            slice_type = Some(st);

            if pps.output_flag_present_flag {
                pic_output_flag = br.u1()? != 0;
            }

            if sps.separate_colour_plane_flag {
                let id = br.u(2)? as u8;
                colour_plane_id = Some(id);
            }

            // Non-IDR POC + reference-picture-set block (§7.3.6.1).
            let is_idr = nal_unit_type == IDR_W_RADL || nal_unit_type == IDR_N_LP;
            if !is_idr {
                // slice_pic_order_cnt_lsb u(v), width log2_max_poc_lsb_minus4+4.
                let poc_lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 + 4;
                let poc_lsb = br.u(poc_lsb_bits)?;
                if poc_lsb >= sps.max_pic_order_cnt_lsb() {
                    return Err(SliceError::ValueOutOfRange {
                        field: "slice_pic_order_cnt_lsb",
                        got: poc_lsb as i64,
                    });
                }
                slice_pic_order_cnt_lsb = Some(poc_lsb);

                // short_term_ref_pic_set_sps_flag u(1).
                let st_sps_flag = br.u1()? != 0;
                short_term_ref_pic_set_sps_flag = Some(st_sps_flag);
                if st_sps_flag && sps.num_short_term_ref_pic_sets == 0 {
                    // §7.4.7.1: when num_short_term_ref_pic_sets == 0,
                    // short_term_ref_pic_set_sps_flag shall be 0.
                    return Err(SliceError::ValueOutOfRange {
                        field: "short_term_ref_pic_set_sps_flag",
                        got: 1,
                    });
                }

                if !st_sps_flag {
                    let inline = ShortTermRefPicSet::parse_slice_inline(&mut br, sps)?;
                    inline_short_term_ref_pic_set = Some(inline);
                } else if sps.num_short_term_ref_pic_sets > 1 {
                    let idx_bits = ceil_log2(sps.num_short_term_ref_pic_sets);
                    let idx = br.u(idx_bits)?;
                    if idx >= sps.num_short_term_ref_pic_sets {
                        return Err(SliceError::ValueOutOfRange {
                            field: "short_term_ref_pic_set_idx",
                            got: idx as i64,
                        });
                    }
                    short_term_ref_pic_set_idx = Some(idx);
                }
                // else: short_term_ref_pic_set_idx is inferred to 0
                // (and left as None in the struct to signal "absent").

                if sps.long_term_ref_pics_present_flag {
                    let (nl_sps, nl_pics, entries) = parse_long_term_ref_pic_block(&mut br, sps)?;
                    num_long_term_sps = Some(nl_sps);
                    num_long_term_pics = Some(nl_pics);
                    long_term_ref_pics = entries;
                }
            }

            if sps.sps_temporal_mvp_enabled_flag {
                slice_temporal_mvp_enabled_flag = br.u1()? != 0;
            }
        }

        // SAO block (§7.3.6.1) — outside the !dependent gate.
        let mut slice_sao_luma_flag = false;
        let mut slice_sao_chroma_flag = false;
        if sps.sample_adaptive_offset_enabled_flag {
            slice_sao_luma_flag = br.u1()? != 0;
            if chroma_array_type(sps) != 0 {
                slice_sao_chroma_flag = br.u1()? != 0;
            }
        }

        // The remaining body lives inside the !dependent gate. For a
        // dependent slice segment the header ends after the SAO block,
        // before byte_alignment() (the rest of the header is inherited).
        if dependent_slice_segment_flag {
            let byte_offset = consume_byte_alignment(&mut br)?;
            return Ok(Self {
                first_slice_segment_in_pic_flag,
                no_output_of_prior_pics_flag,
                slice_pic_parameter_set_id,
                dependent_slice_segment_flag,
                slice_segment_address,
                slice_reserved_flags,
                slice_type,
                pic_output_flag,
                colour_plane_id,
                slice_pic_order_cnt_lsb: None,
                short_term_ref_pic_set_sps_flag: None,
                inline_short_term_ref_pic_set: None,
                short_term_ref_pic_set_idx: None,
                num_long_term_sps: None,
                num_long_term_pics: None,
                long_term_ref_pics: Vec::new(),
                slice_temporal_mvp_enabled_flag,
                slice_sao_luma_flag,
                slice_sao_chroma_flag,
                slice_qp_delta: None,
                slice_cb_qp_offset: 0,
                slice_cr_qp_offset: 0,
                deblocking: None,
                slice_loop_filter_across_slices_enabled_flag: None,
                entry_point_offsets: None,
                slice_segment_header_extension_length: None,
                byte_offset_to_slice_data: Some(byte_offset),
                opaque_tail: None,
            });
        }

        // P / B reference-list, mvd, cabac-init, collocated, weighted
        // prediction, merge-cand, integer-mv: deferred this round
        // (needs DPB-derived NumPicTotalCurr / RefPicList). Surface the
        // remainder when the slice is inter.
        let st = slice_type.expect("independent slice has a slice_type");
        if st.is_inter() {
            return Ok(Self {
                first_slice_segment_in_pic_flag,
                no_output_of_prior_pics_flag,
                slice_pic_parameter_set_id,
                dependent_slice_segment_flag,
                slice_segment_address,
                slice_reserved_flags,
                slice_type,
                pic_output_flag,
                colour_plane_id,
                slice_pic_order_cnt_lsb,
                short_term_ref_pic_set_sps_flag,
                inline_short_term_ref_pic_set,
                short_term_ref_pic_set_idx,
                num_long_term_sps,
                num_long_term_pics,
                long_term_ref_pics,
                slice_temporal_mvp_enabled_flag,
                slice_sao_luma_flag,
                slice_sao_chroma_flag,
                slice_qp_delta: None,
                slice_cb_qp_offset: 0,
                slice_cr_qp_offset: 0,
                deblocking: None,
                slice_loop_filter_across_slices_enabled_flag: None,
                entry_point_offsets: None,
                slice_segment_header_extension_length: None,
                byte_offset_to_slice_data: None,
                opaque_tail: Some(OpaqueTail::capture_at(br.bit_pos(), rbsp)),
            });
        }

        // I-slice tail: slice_qp_delta + chroma QP offsets + deblocking
        // override + loop-filter-across-slices.
        let slice_qp_delta = br.se()?;

        let mut slice_cb_qp_offset = 0i8;
        let mut slice_cr_qp_offset = 0i8;
        if pps.pps_slice_chroma_qp_offsets_present_flag {
            slice_cb_qp_offset = parse_qp_offset(&mut br, "slice_cb_qp_offset")?;
            slice_cr_qp_offset = parse_qp_offset(&mut br, "slice_cr_qp_offset")?;
        }

        // Deblocking override (§7.3.6.1). The PPS range-extension
        // `chroma_qp_offset_list_enabled_flag` and the act-QP-offset
        // block are not signalled by a base-profile PPS and are not
        // surfaced by this crate yet, so they are absent here.
        let deblocking = parse_slice_deblocking(&mut br, pps)?;

        // slice_loop_filter_across_slices_enabled_flag gate (§7.3.6.1).
        let slice_loop_filter_across_slices_enabled_flag = if pps
            .pps_loop_filter_across_slices_enabled_flag
            && (slice_sao_luma_flag || slice_sao_chroma_flag || !deblocking.disabled_flag)
        {
            br.u1()? != 0
        } else {
            pps.pps_loop_filter_across_slices_enabled_flag
        };

        // Entry-point-offset block (§7.3.6.1).
        let entry_point_offsets = if pps.tiles_enabled_flag || pps.entropy_coding_sync_enabled_flag
        {
            let num_entry_point_offsets = br.ue()?;
            let offset_len_minus1 = if num_entry_point_offsets > 0 {
                let v = br.ue()?;
                if v > 31 {
                    return Err(SliceError::ValueOutOfRange {
                        field: "offset_len_minus1",
                        got: v as i64,
                    });
                }
                let len = v as u8;
                for _ in 0..num_entry_point_offsets {
                    br.skip((len as usize) + 1)?;
                }
                len
            } else {
                0
            };
            Some(EntryPointOffsets {
                num_entry_point_offsets,
                offset_len_minus1,
            })
        } else {
            None
        };

        // Slice-segment-header extension block (§7.3.6.1).
        let slice_segment_header_extension_length =
            if pps.slice_segment_header_extension_present_flag {
                let len = br.ue()?;
                for _ in 0..len {
                    br.skip(8)?;
                }
                Some(len)
            } else {
                None
            };

        let byte_offset = consume_byte_alignment(&mut br)?;

        Ok(Self {
            first_slice_segment_in_pic_flag,
            no_output_of_prior_pics_flag,
            slice_pic_parameter_set_id,
            dependent_slice_segment_flag,
            slice_segment_address,
            slice_reserved_flags,
            slice_type,
            pic_output_flag,
            colour_plane_id,
            slice_pic_order_cnt_lsb,
            short_term_ref_pic_set_sps_flag,
            inline_short_term_ref_pic_set,
            short_term_ref_pic_set_idx,
            num_long_term_sps,
            num_long_term_pics,
            long_term_ref_pics,
            slice_temporal_mvp_enabled_flag,
            slice_sao_luma_flag,
            slice_sao_chroma_flag,
            slice_qp_delta: Some(slice_qp_delta),
            slice_cb_qp_offset,
            slice_cr_qp_offset,
            deblocking: Some(deblocking),
            slice_loop_filter_across_slices_enabled_flag: Some(
                slice_loop_filter_across_slices_enabled_flag,
            ),
            entry_point_offsets,
            slice_segment_header_extension_length,
            byte_offset_to_slice_data: Some(byte_offset),
            opaque_tail: None,
        })
    }

    /// `SliceQpY = 26 + init_qp_minus26 + slice_qp_delta` (equation
    /// 7-54). Returns `None` when `slice_qp_delta` was not parsed (a
    /// deferred body); `pps` supplies `init_qp_minus26`.
    pub fn slice_qp_y(&self, pps: &PicParameterSet) -> Option<i32> {
        self.slice_qp_delta.map(|d| 26 + pps.init_qp_minus26 + d)
    }
}

/// `ChromaArrayType` per §7.4.2.2: equal to `chroma_format_idc` unless
/// `separate_colour_plane_flag == 1`, in which case it is 0.
fn chroma_array_type(sps: &SeqParameterSet) -> u8 {
    if sps.separate_colour_plane_flag {
        0
    } else {
        sps.chroma_format_idc
    }
}

/// `PicSizeInCtbsY = PicWidthInCtbsY * PicHeightInCtbsY`
/// (equations 7-15/7-17/7-19). `CtbSizeY = 1 << CtbLog2SizeY` and the
/// per-dimension counts use ceiling division.
fn pic_size_in_ctbs_y(sps: &SeqParameterSet) -> u32 {
    let ctb_size = 1u32 << sps.log2_ctb_size();
    let width_in_ctbs = sps.pic_width_in_luma_samples.div_ceil(ctb_size);
    let height_in_ctbs = sps.pic_height_in_luma_samples.div_ceil(ctb_size);
    width_in_ctbs * height_in_ctbs
}

/// `Ceil( Log2( n ) )` — the §7.4.7.1 width formula for
/// `slice_segment_address`. For `n <= 1` the width is 0 bits (a
/// single-CTB picture has no address to signal).
fn ceil_log2(n: u32) -> u8 {
    if n <= 1 {
        0
    } else {
        // Ceil(Log2(n)) = bit-width of (n - 1).
        (32 - (n - 1).leading_zeros()) as u8
    }
}

/// Parse one `se(v)` chroma QP offset, range-checked to −12..=12.
fn parse_qp_offset(br: &mut BitReader<'_>, field: &'static str) -> Result<i8, SliceError> {
    let v = br.se()?;
    if !(-12..=12).contains(&v) {
        return Err(SliceError::ValueOutOfRange {
            field,
            got: v as i64,
        });
    }
    Ok(v as i8)
}

/// Parse the deblocking-filter override block of §7.3.6.1, applying the
/// §7.4.7.1 inference rules for the absent fields.
fn parse_slice_deblocking(
    br: &mut BitReader<'_>,
    pps: &PicParameterSet,
) -> Result<SliceDeblocking, SliceError> {
    // deblocking_filter_override_flag only present when
    // deblocking_filter_override_enabled_flag (PPS).
    let override_flag = if pps.deblocking.override_enabled_flag {
        br.u1()? != 0
    } else {
        false
    };

    if !override_flag {
        // Inferred from the PPS (§7.4.7.1).
        return Ok(SliceDeblocking {
            disabled_flag: pps.deblocking.disabled_flag,
            beta_offset_div2: pps.deblocking.beta_offset_div2,
            tc_offset_div2: pps.deblocking.tc_offset_div2,
        });
    }

    let disabled_flag = br.u1()? != 0;
    let (beta, tc) = if !disabled_flag {
        let beta = br.se()?;
        if !(-6..=6).contains(&beta) {
            return Err(SliceError::ValueOutOfRange {
                field: "slice_beta_offset_div2",
                got: beta as i64,
            });
        }
        let tc = br.se()?;
        if !(-6..=6).contains(&tc) {
            return Err(SliceError::ValueOutOfRange {
                field: "slice_tc_offset_div2",
                got: tc as i64,
            });
        }
        (beta as i8, tc as i8)
    } else {
        // When deblocking is disabled the offsets are not signalled and
        // are inferred to 0 (their effect is moot when disabled).
        (0, 0)
    };

    Ok(SliceDeblocking {
        disabled_flag,
        beta_offset_div2: beta,
        tc_offset_div2: tc,
    })
}

/// Parse the long-term-ref-pic block of §7.3.6.1 (the body gated by
/// `long_term_ref_pics_present_flag` on the SPS), returning the parsed
/// `(num_long_term_sps, num_long_term_pics, entries)` triple.
///
/// The block:
///
/// ```text
///   if( num_long_term_ref_pics_sps > 0 )
///       num_long_term_sps  ue(v)
///   num_long_term_pics      ue(v)
///   for( i = 0; i < num_long_term_sps + num_long_term_pics; i++ ) {
///       if( i < num_long_term_sps ) {
///           if( num_long_term_ref_pics_sps > 1 )
///               lt_idx_sps[i]   u(v) — Ceil(Log2(num_long_term_ref_pics_sps))
///       } else {
///           poc_lsb_lt[i]                    u(v) — log2_max_poc_lsb_minus4+4
///           used_by_curr_pic_lt_flag[i]      u(1)
///       }
///       delta_poc_msb_present_flag[i]        u(1)
///       if( delta_poc_msb_present_flag[i] )
///           delta_poc_msb_cycle_lt[i]        ue(v)
///   }
/// ```
fn parse_long_term_ref_pic_block(
    br: &mut BitReader<'_>,
    sps: &SeqParameterSet,
) -> Result<(u32, u32, Vec<SliceLongTermRefPic>), SliceError> {
    let num_long_term_sps = if sps.num_long_term_ref_pics_sps > 0 {
        let v = br.ue()?;
        if v > sps.num_long_term_ref_pics_sps {
            return Err(SliceError::ValueOutOfRange {
                field: "num_long_term_sps",
                got: v as i64,
            });
        }
        v
    } else {
        0
    };
    let num_long_term_pics = br.ue()?;
    // §7.4.7.1 bounds num_long_term_pics by the SPS DPB capacity; we
    // apply a defensive sanity ceiling instead of computing the full
    // DPB-derived bound (which needs RPS counts not yet wired through
    // here). A pathological encoder could otherwise drive an unbounded
    // allocation.
    if num_long_term_pics > HEVC_MAX_LONG_TERM_PICS_IN_SLICE as u32 {
        return Err(SliceError::ValueOutOfRange {
            field: "num_long_term_pics",
            got: num_long_term_pics as i64,
        });
    }
    let total = num_long_term_sps + num_long_term_pics;
    let lt_idx_bits = if sps.num_long_term_ref_pics_sps > 1 {
        ceil_log2(sps.num_long_term_ref_pics_sps)
    } else {
        0
    };
    let poc_lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 + 4;

    let mut entries = Vec::with_capacity(total as usize);
    for i in 0..total {
        let source = if i < num_long_term_sps {
            let lt_idx_sps = if lt_idx_bits > 0 {
                let v = br.u(lt_idx_bits)?;
                if v >= sps.num_long_term_ref_pics_sps {
                    return Err(SliceError::ValueOutOfRange {
                        field: "lt_idx_sps",
                        got: v as i64,
                    });
                }
                v
            } else {
                0
            };
            SliceLongTermRefPicSource::Sps { lt_idx_sps }
        } else {
            let poc_lsb_lt = br.u(poc_lsb_bits)?;
            let used_by_curr_pic_lt_flag = br.u1()? != 0;
            SliceLongTermRefPicSource::InSlice {
                poc_lsb_lt,
                used_by_curr_pic_lt_flag,
            }
        };
        let delta_poc_msb_present_flag = br.u1()? != 0;
        let delta_poc_msb_cycle_lt = if delta_poc_msb_present_flag {
            br.ue()?
        } else {
            0
        };
        entries.push(SliceLongTermRefPic {
            source,
            delta_poc_msb_present_flag,
            delta_poc_msb_cycle_lt,
        });
    }
    Ok((num_long_term_sps, num_long_term_pics, entries))
}

/// Defensive upper bound on `num_long_term_pics` (§7.4.7.1 bounds the
/// value by `sps_max_dec_pic_buffering_minus1[TemporalId] − …`; an
/// HEVC DPB is bounded by `MaxDpbSize` which is bounded by
/// `sps_max_dec_pic_buffering_minus1` ≤ 15 per §7.4.3.2.1).
const HEVC_MAX_LONG_TERM_PICS_IN_SLICE: usize = 16;

/// Consume `byte_alignment()` (§7.3.2.4): one `alignment_bit_equal_to_one`
/// followed by `alignment_bit_equal_to_zero` bits until the cursor is on
/// a byte boundary. Returns the byte offset (from the start of the RBSP)
/// of the first byte that follows.
fn consume_byte_alignment(br: &mut BitReader<'_>) -> Result<usize, SliceError> {
    // alignment_bit_equal_to_one.
    let _ = br.u1()?;
    while br.bit_pos() % 8 != 0 {
        let _ = br.u1()?;
    }
    Ok(br.bit_pos() / 8)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal SPS for slice-header parsing context. Only the
    /// fields the slice parser reads are populated meaningfully; the
    /// rest carry defaults that do not affect the header parse.
    #[allow(clippy::too_many_arguments)]
    fn ctx_sps(
        chroma_format_idc: u8,
        separate_colour_plane_flag: bool,
        sao: bool,
        mvp: bool,
        width: u32,
        height: u32,
        log2_diff_max_min_cb: u8,
        log2_min_cb_minus3: u8,
        log2_max_poc_lsb_minus4: u8,
    ) -> SeqParameterSet {
        // Hand-assemble the smallest valid SPS RBSP that decodes to the
        // requested gate values, by parsing the tiny fixture's SPS and
        // patching the relevant fields. The slice parser only consults
        // chroma_format_idc, separate_colour_plane_flag,
        // sample_adaptive_offset_enabled_flag,
        // sps_temporal_mvp_enabled_flag, the CTB / picture-size
        // derivations, and log2_max_pic_order_cnt_lsb_minus4, so a
        // patched struct is sufficient for these unit tests.
        let mut sps = SeqParameterSet::parse(TINY_SPS_RBSP).expect("tiny SPS");
        sps.chroma_format_idc = chroma_format_idc;
        sps.separate_colour_plane_flag = separate_colour_plane_flag;
        sps.sample_adaptive_offset_enabled_flag = sao;
        sps.sps_temporal_mvp_enabled_flag = mvp;
        sps.pic_width_in_luma_samples = width;
        sps.pic_height_in_luma_samples = height;
        sps.log2_diff_max_min_luma_coding_block_size = log2_diff_max_min_cb;
        sps.log2_min_luma_coding_block_size_minus3 = log2_min_cb_minus3;
        sps.log2_max_pic_order_cnt_lsb_minus4 = log2_max_poc_lsb_minus4;
        sps
    }

    /// SPS RBSP body from the tiny fixture (see `sps.rs` tests).
    const TINY_SPS_RBSP: &[u8] = &[
        0x01, 0x04, 0x08, 0x00, 0x00, 0x00, 0x9F, 0xA8, 0x00, 0x00, 0x00, 0x00, 0x1E, 0xA0, 0x88,
        0x45, 0x96, 0xEA, 0xAF, 0x2B, 0xC0, 0x5A, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x32, 0x10,
    ];
    /// PPS RBSP body from the tiny fixture (see `pps.rs` tests).
    const TINY_PPS_RBSP: &[u8] = &[0xC1, 0x73, 0xC0, 0x89];

    #[test]
    fn slice_type_table_7_7() {
        assert_eq!(SliceType::from_raw(0).unwrap(), SliceType::B);
        assert_eq!(SliceType::from_raw(1).unwrap(), SliceType::P);
        assert_eq!(SliceType::from_raw(2).unwrap(), SliceType::I);
        assert!(SliceType::from_raw(3).is_err());
        assert!(!SliceType::I.is_inter());
        assert!(SliceType::P.is_inter());
        assert!(SliceType::B.is_inter());
    }

    #[test]
    fn ceil_log2_widths() {
        // §7.4.7.1: slice_segment_address width = Ceil(Log2(N)).
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(8), 3);
        assert_eq!(ceil_log2(9), 4);
    }

    /// Hand-assembled minimal independent I-slice IDR header. With the
    /// fixture SPS/PPS gates (sao=1, mvp=1, chroma=1, single CTB,
    /// loop-filter-across-slices=1) the bit layout is:
    ///   first(1)=1 no_output(1)=0 pps_id ue=1 (->0)
    ///   slice_type ue=011 (->2,I) mvp(1)=0 sao_l(1)=1 sao_c(1)=0
    ///   qp_delta se=011 (->-1) lf_across(1)=1
    ///   byte_alignment: 1 then zeros.
    /// We assemble those bits ourselves so the parse is fully
    /// controlled (the tiny fixture's own slice trace is internally
    /// inconsistent — see the module-level note / docs gap).
    #[test]
    fn parses_hand_assembled_i_idr_header() {
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        // pps from the fixture has loop_filter_across_slices=1,
        // tiles=0, ecs=0, deblock-ctrl-present=0, chroma-qp-off=0.

        // Build the bit string.
        let bits = concat_bits(&[
            (1, 1),     // first_slice_segment_in_pic_flag
            (0, 1),     // no_output_of_prior_pics_flag (IRAP)
            (0b1, 1),   // pps_id ue(v) '1' -> 0
            (0b011, 3), // slice_type ue(v) '011' -> 2 (I)
            (0, 1),     // slice_temporal_mvp_enabled_flag (mvp gate on)
            (1, 1),     // slice_sao_luma_flag
            (0, 1),     // slice_sao_chroma_flag (chroma!=0)
            (0b011, 3), // slice_qp_delta se(v) '011' -> -1
            (1, 1),     // slice_loop_filter_across_slices_enabled_flag
            (1, 1),     // byte_alignment: alignment_bit_equal_to_one
        ]);
        let rbsp = pack_bits(&bits);

        let sh = SliceSegmentHeader::parse(&rbsp, IDR_N_LP, &sps, &pps).expect("slice header");
        assert!(sh.first_slice_segment_in_pic_flag);
        assert_eq!(sh.no_output_of_prior_pics_flag, Some(false));
        assert_eq!(sh.slice_pic_parameter_set_id, 0);
        assert!(!sh.dependent_slice_segment_flag);
        assert_eq!(sh.slice_segment_address, 0);
        assert_eq!(sh.slice_type, Some(SliceType::I));
        assert!(!sh.slice_temporal_mvp_enabled_flag);
        assert!(sh.slice_sao_luma_flag);
        assert!(!sh.slice_sao_chroma_flag);
        assert_eq!(sh.slice_qp_delta, Some(-1));
        assert_eq!(sh.slice_loop_filter_across_slices_enabled_flag, Some(true));
        assert!(sh.opaque_tail.is_none());
        // init_qp from the fixture PPS is 26, so SliceQpY = 26 + -1 = 25.
        assert_eq!(sh.slice_qp_y(&pps), Some(25));
        // The header must byte-align: with 1+1+1+3+1+1+1+3+1=13 bits
        // before alignment, alignment consumes bits 13..16 (one '1'
        // plus zero pad), so slice data begins at byte 2.
        assert_eq!(sh.byte_offset_to_slice_data, Some(2));
    }

    /// Non-IDR **I-slice** (CRA, type 21 — in the IRAP range so
    /// `no_output_of_prior_pics_flag` is present): the POC + RPS block
    /// is now parsed inline (round 105), so the parser reaches
    /// `byte_alignment()` with no opaque tail.
    #[test]
    fn parses_non_idr_i_slice_cra_with_inline_zero_rps() {
        // Tiny SPS context: num_short_term_ref_pic_sets = 0, so
        // short_term_ref_pic_set_sps_flag must be 0 and the in-line RPS
        // (stRpsIdx = 0, no inter-RPS-prediction signal) reads
        // num_negative_pics + num_positive_pics, both 0.
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");

        let bits = concat_bits(&[
            (1, 1),     // first_slice_segment_in_pic_flag
            (0, 1),     // no_output_of_prior_pics_flag (CRA = 21 ∈ IRAP range)
            (0b1, 1),   // pps_id ue '1' -> 0
            (0b011, 3), // slice_type ue '011' -> 2 (I)
            (0, 8),     // slice_pic_order_cnt_lsb u(8) = 0
            (0, 1),     // short_term_ref_pic_set_sps_flag = 0
            // in-line st_ref_pic_set( num_short_term_ref_pic_sets = 0 ):
            // stRpsIdx == 0 so inter_ref_pic_set_prediction_flag absent.
            (0b1, 1), // num_negative_pics ue '1' -> 0
            (0b1, 1), // num_positive_pics ue '1' -> 0
            (0, 1),   // slice_temporal_mvp_enabled_flag = 0
            (1, 1),   // slice_sao_luma_flag = 1
            (0, 1),   // slice_sao_chroma_flag = 0
            (0b1, 1), // slice_qp_delta se '1' -> 0
            // pps.lf_across=1 and sao_luma OR !deblock.disabled => present.
            // PPS deblocking override_enabled_flag=0, disabled_flag=0 by
            // inference => !disabled_flag is true => gate fires.
            (1, 1), // slice_loop_filter_across_slices_enabled_flag = 1
            (1, 1), // byte_alignment one-bit
        ]);
        let rbsp = pack_bits(&bits);
        // CRA_NUT (NAL type 21) is in BLA_W_LP..=RSV_IRAP_VCL23 (16..=23).
        let sh = SliceSegmentHeader::parse(&rbsp, 21, &sps, &pps).expect("slice header");
        assert!(sh.first_slice_segment_in_pic_flag);
        assert_eq!(sh.no_output_of_prior_pics_flag, Some(false));
        assert_eq!(sh.slice_type, Some(SliceType::I));
        assert_eq!(sh.slice_pic_order_cnt_lsb, Some(0));
        assert_eq!(sh.short_term_ref_pic_set_sps_flag, Some(false));
        let inline = sh
            .inline_short_term_ref_pic_set
            .as_ref()
            .expect("inline ST RPS");
        assert!(!inline.inter_ref_pic_set_prediction_flag);
        assert_eq!(inline.num_negative_pics, 0);
        assert_eq!(inline.num_positive_pics, 0);
        assert!(sh.short_term_ref_pic_set_idx.is_none());
        assert_eq!(sh.num_long_term_sps, None); // SPS gate off
        assert_eq!(sh.num_long_term_pics, None);
        assert!(!sh.slice_temporal_mvp_enabled_flag);
        assert!(sh.slice_sao_luma_flag);
        assert!(!sh.slice_sao_chroma_flag);
        assert_eq!(sh.slice_qp_delta, Some(0));
        // Tail consumed, byte_alignment reached.
        assert!(sh.opaque_tail.is_none());
        assert_eq!(sh.byte_offset_to_slice_data, Some(3));
    }

    /// Non-IDR I-slice using the SPS-resident ST RPS (the SPS has
    /// `num_short_term_ref_pic_sets == 1`, so
    /// `short_term_ref_pic_set_idx` is **absent** — its value is
    /// inferred to 0 — and the slice header reads no extra bits for
    /// the RPS).
    #[test]
    fn parses_non_idr_i_slice_with_sps_rps_single_entry() {
        let mut sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        // Forge an SPS with one short-term-RPS entry so the gate path
        // matches the §7.3.6.1 "else if( num_short_term_ref_pic_sets > 1 )"
        // branch FALSE path: short_term_ref_pic_set_idx absent.
        sps.num_short_term_ref_pic_sets = 1;
        sps.short_term_ref_pic_sets = vec![ShortTermRefPicSet {
            inter_ref_pic_set_prediction_flag: false,
            num_negative_pics: 0,
            num_positive_pics: 0,
            ..Default::default()
        }];
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");

        let bits = concat_bits(&[
            (1, 1),     // first
            (0, 1),     // no_output (IRAP)
            (0b1, 1),   // pps_id ue '1' -> 0
            (0b011, 3), // slice_type ue '011' -> 2 (I)
            (0, 8),     // slice_pic_order_cnt_lsb = 0
            (1, 1),     // short_term_ref_pic_set_sps_flag = 1
            // num_short_term_ref_pic_sets == 1 so short_term_ref_pic_set_idx
            // is NOT signalled (inferred 0).
            (0, 1),   // slice_temporal_mvp_enabled_flag = 0
            (1, 1),   // sao_luma = 1
            (0, 1),   // sao_chroma = 0
            (0b1, 1), // slice_qp_delta se '1' -> 0
            (1, 1),   // lf_across_slices = 1
            (1, 1),   // byte_alignment one bit
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, 21, &sps, &pps).expect("slice header");
        assert_eq!(sh.short_term_ref_pic_set_sps_flag, Some(true));
        assert!(sh.inline_short_term_ref_pic_set.is_none());
        assert!(sh.short_term_ref_pic_set_idx.is_none());
        assert_eq!(sh.slice_qp_delta, Some(0));
        assert!(sh.opaque_tail.is_none());
    }

    /// Non-IDR I-slice using the SPS-resident ST RPS with multiple
    /// entries: `short_term_ref_pic_set_idx` is signalled `u(v)` with
    /// width `Ceil(Log2(num_short_term_ref_pic_sets))`.
    #[test]
    fn parses_non_idr_i_slice_with_sps_rps_idx_signalled() {
        let mut sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        sps.num_short_term_ref_pic_sets = 3; // idx width = 2 bits
        sps.short_term_ref_pic_sets = vec![
            ShortTermRefPicSet::default(),
            ShortTermRefPicSet::default(),
            ShortTermRefPicSet::default(),
        ];
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");

        let bits = concat_bits(&[
            (1, 1),
            (0, 1),
            (0b1, 1),
            (0b011, 3),
            (0, 8),    // poc_lsb = 0
            (1, 1),    // short_term_ref_pic_set_sps_flag = 1
            (0b10, 2), // short_term_ref_pic_set_idx u(2) = 2
            (0, 1),    // mvp = 0
            (1, 1),    // sao_luma
            (0, 1),    // sao_chroma
            (0b1, 1),  // slice_qp_delta = 0
            (1, 1),    // lf_across
            (1, 1),    // byte_alignment
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, 21, &sps, &pps).expect("slice header");
        assert_eq!(sh.short_term_ref_pic_set_idx, Some(2));
        assert!(sh.inline_short_term_ref_pic_set.is_none());
        assert!(sh.opaque_tail.is_none());
    }

    /// Long-term-ref-pic block: SPS has
    /// `long_term_ref_pics_present_flag=1, num_long_term_ref_pics_sps=2`.
    /// The slice header carries one SPS-indexed entry plus one in-slice
    /// entry, each with `delta_poc_msb_present_flag` and the cycle.
    #[test]
    fn parses_non_idr_i_slice_with_long_term_block() {
        use crate::sps::LongTermRefPicEntry;
        let mut sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        sps.long_term_ref_pics_present_flag = true;
        sps.num_long_term_ref_pics_sps = 2; // lt_idx_sps width = 1 bit
        sps.long_term_ref_pics = vec![
            LongTermRefPicEntry {
                poc_lsb: 0,
                used_by_curr_pic: true,
            },
            LongTermRefPicEntry {
                poc_lsb: 4,
                used_by_curr_pic: false,
            },
        ];
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");

        let bits = concat_bits(&[
            (1, 1), // first
            (0, 1), // no_output (IRAP)
            (0b1, 1),
            (0b011, 3), // I slice
            (0, 8),     // poc_lsb
            (0, 1),     // st_sps_flag = 0 (num_st_rps=0)
            (0b1, 1),   // num_negative_pics ue '1' -> 0
            (0b1, 1),   // num_positive_pics ue '1' -> 0
            // Long-term block:
            (0b010, 3), // num_long_term_sps ue '010' -> 1
            (0b010, 3), // num_long_term_pics ue '010' -> 1
            // Entry 0: SPS-indexed (i < num_long_term_sps)
            (1, 1), // lt_idx_sps[0] u(1) = 1
            (0, 1), // delta_poc_msb_present_flag[0] = 0
            // Entry 1: in-slice (i >= num_long_term_sps)
            (0b1010, 8), // poc_lsb_lt[1] u(8) = 0xAA bit pattern... use 10
            (1, 1),      // used_by_curr_pic_lt_flag[1] = 1
            (1, 1),      // delta_poc_msb_present_flag[1] = 1
            (0b010, 3),  // delta_poc_msb_cycle_lt[1] ue '010' -> 1
            (0, 1),      // slice_temporal_mvp_enabled_flag = 0
            (1, 1),      // sao_luma
            (0, 1),      // sao_chroma
            (0b1, 1),    // slice_qp_delta se -> 0
            (1, 1),      // lf_across
            (1, 1),      // byte_alignment
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, 21, &sps, &pps).expect("slice header");
        assert_eq!(sh.num_long_term_sps, Some(1));
        assert_eq!(sh.num_long_term_pics, Some(1));
        assert_eq!(sh.long_term_ref_pics.len(), 2);
        match sh.long_term_ref_pics[0].source {
            SliceLongTermRefPicSource::Sps { lt_idx_sps } => assert_eq!(lt_idx_sps, 1),
            other => panic!("entry 0 should be SPS-indexed: {other:?}"),
        }
        assert!(!sh.long_term_ref_pics[0].delta_poc_msb_present_flag);
        match sh.long_term_ref_pics[1].source {
            SliceLongTermRefPicSource::InSlice {
                poc_lsb_lt,
                used_by_curr_pic_lt_flag,
            } => {
                assert_eq!(poc_lsb_lt, 0b1010); // 8-bit u(v) field carrying 10
                assert!(used_by_curr_pic_lt_flag);
            }
            other => panic!("entry 1 should be in-slice: {other:?}"),
        }
        assert!(sh.long_term_ref_pics[1].delta_poc_msb_present_flag);
        assert_eq!(sh.long_term_ref_pics[1].delta_poc_msb_cycle_lt, 1);
        assert!(sh.opaque_tail.is_none());
    }

    /// The §7.4.7.1 cross-check: when `num_short_term_ref_pic_sets ==
    /// 0`, signalling `short_term_ref_pic_set_sps_flag == 1` is illegal.
    #[test]
    fn rejects_st_sps_flag_when_no_sps_rps() {
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        let bits = concat_bits(&[
            (1, 1),
            (0, 1),
            (0b1, 1),
            (0b011, 3), // I
            (0, 8),     // poc_lsb
            (1, 1),     // short_term_ref_pic_set_sps_flag = 1 (illegal: num_st_rps=0)
        ]);
        let rbsp = pack_bits(&bits);
        let err = SliceSegmentHeader::parse(&rbsp, 21, &sps, &pps).unwrap_err();
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "short_term_ref_pic_set_sps_flag",
                got: 1
            }
        );
    }

    /// IDR P-slice: the parser reaches the SAO block then defers the
    /// P/B reference-list / weighted-prediction body.
    #[test]
    fn defers_pb_ref_list_body() {
        let sps = ctx_sps(1, false, true, false, 16, 16, 1, 0, 4);
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        // first=1, no_output(IRAP)=0, pps_id=0, slice_type=P,
        // (mvp gate off), sao_l, sao_c, then num_ref_idx_active_override
        // begins (deferred).
        let bits = concat_bits(&[
            (1, 1),     // first
            (0, 1),     // no_output (IDR is IRAP)
            (0b1, 1),   // pps_id -> 0
            (0b010, 3), // slice_type -> P
            (1, 1),     // sao_luma
            (1, 1),     // sao_chroma
            (0b1, 1),   // first deferred P/B bit, captured opaquely
        ]);
        let rbsp = pack_bits(&bits);
        let sh = SliceSegmentHeader::parse(&rbsp, IDR_W_RADL, &sps, &pps).expect("slice header");
        assert_eq!(sh.slice_type, Some(SliceType::P));
        assert!(sh.slice_sao_luma_flag);
        assert!(sh.slice_sao_chroma_flag);
        assert!(sh.opaque_tail.is_some());
        assert_eq!(sh.slice_qp_delta, None);
        // Opaque tail begins after sao_chroma (bit 8 = byte 1, bit 0).
        let tail = sh.opaque_tail.unwrap();
        assert_eq!(tail.start_bit_in_first_byte, 0);
    }

    /// Non-first dependent slice segment: dependent flag + address are
    /// read, then the body ends after SAO (no slice_type etc.).
    #[test]
    fn parses_dependent_slice_segment() {
        // 4-CTB picture (2x2) so slice_segment_address is 2 bits wide.
        // PPS must have dependent_slice_segments_enabled_flag; the
        // fixture PPS has it 0, so patch a parsed PPS.
        let sps = ctx_sps(1, false, true, true, 32, 32, 1, 0, 4); // 32/16=2 -> 2x2=4 CTBs
        let mut pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        pps.dependent_slice_segments_enabled_flag = true;
        // first=0, dependent=1, slice_segment_address u(2)=2,
        // (dependent: skip slice_type etc.), SAO block: sao_l, sao_c,
        // byte_alignment.
        let bits = concat_bits(&[
            (0, 1),    // first_slice_segment_in_pic_flag = 0
            (0b1, 1),  // pps_id ue -> 0
            (1, 1),    // dependent_slice_segment_flag
            (0b10, 2), // slice_segment_address = 2
            (1, 1),    // sao_luma
            (0, 1),    // sao_chroma
            (1, 1),    // byte_alignment one bit
        ]);
        let rbsp = pack_bits(&bits);
        // Use a non-IRAP type so no_output is absent and the layout
        // above matches.
        let sh =
            SliceSegmentHeader::parse(&rbsp, 0 /* TRAIL_N */, &sps, &pps).expect("slice header");
        assert!(!sh.first_slice_segment_in_pic_flag);
        assert!(sh.dependent_slice_segment_flag);
        assert_eq!(sh.slice_segment_address, 2);
        assert_eq!(sh.slice_type, None);
        assert!(sh.slice_sao_luma_flag);
        assert!(!sh.slice_sao_chroma_flag);
        assert!(sh.opaque_tail.is_none());
        assert_eq!(sh.byte_offset_to_slice_data, Some(1));
    }

    #[test]
    fn rejects_truncated_rbsp() {
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        // Only one byte; first bit reads ok but ue() for pps_id runs off
        // a short enough buffer eventually. Use an empty buffer to force
        // truncation on the very first read.
        let err = SliceSegmentHeader::parse(&[], IDR_N_LP, &sps, &pps).unwrap_err();
        assert_eq!(err, SliceError::Truncated);
    }

    #[test]
    fn rejects_slice_type_out_of_range() {
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        // slice_type ue(v) = 3 ('00100') is out of Table 7-7 range.
        let bits = concat_bits(&[
            (1, 1),       // first
            (0, 1),       // no_output (IDR)
            (0b1, 1),     // pps_id -> 0
            (0b00100, 5), // slice_type ue -> 3 (illegal)
        ]);
        let rbsp = pack_bits(&bits);
        let err = SliceSegmentHeader::parse(&rbsp, IDR_N_LP, &sps, &pps).unwrap_err();
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "slice_type",
                got: 3
            }
        );
    }

    #[test]
    fn end_to_end_via_nal_walker() {
        use crate::nal::collect_nal_units;
        // Build an Annex B stream carrying a single IDR_N_LP slice NAL
        // whose RBSP is the hand-assembled I-slice header from
        // `parses_hand_assembled_i_idr_header`.
        let sps = ctx_sps(1, false, true, true, 16, 16, 1, 0, 4);
        let pps = PicParameterSet::parse(TINY_PPS_RBSP).expect("PPS");
        let bits = concat_bits(&[
            (1, 1),
            (0, 1),
            (0b1, 1),
            (0b011, 3),
            (0, 1),
            (1, 1),
            (0, 1),
            (0b011, 3),
            (1, 1),
            (1, 1),
        ]);
        let body = pack_bits(&bits);
        // NAL header for IDR_N_LP (type 20), layer 0, temporal_id 0.
        let b0 = (IDR_N_LP & 0x3F) << 1;
        let b1 = 0x01; // temporal_id_plus1 = 1
        let mut stream = vec![0x00, 0x00, 0x01, b0, b1];
        stream.extend_from_slice(&body);
        let units = collect_nal_units(&stream).expect("walker");
        assert_eq!(units.len(), 1);
        let u = &units[0];
        assert_eq!(u.header.nal_unit_type, IDR_N_LP);
        let sh = SliceSegmentHeader::parse(&u.rbsp, u.header.nal_unit_type, &sps, &pps)
            .expect("slice header");
        assert_eq!(sh.slice_type, Some(SliceType::I));
        assert_eq!(sh.slice_qp_delta, Some(-1));
        assert_eq!(sh.slice_qp_y(&pps), Some(25));
    }

    // --- §7.3.6.2 ref_pic_lists_modification() ---

    /// Hand-build a `ref_pic_lists_modification()` RBSP and parse it for
    /// a P slice with the modification flag set and a single L0 entry.
    /// `NumPicTotalCurr == 2` so the per-entry width is
    /// `Ceil(Log2(2)) == 1` bit. With `num_ref_idx_l0_active_minus1 ==
    /// 0` the loop reads exactly one `list_entry_l0`. The bits are:
    ///   `ref_pic_list_modification_flag_l0` (1) = 1
    ///   `list_entry_l0[0]` (u(1)) = 1
    /// (B-only L1 fields are absent for a P slice.)
    #[test]
    fn parses_p_slice_l0_only_modification() {
        let bits = concat_bits(&[(1, 1), (1, 1)]);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let m = RefPicListsModification::parse(
            &mut br,
            SliceType::P,
            0, /* num_ref_idx_l0_active_minus1 */
            0, /* num_ref_idx_l1_active_minus1 (ignored for P) */
            2, /* NumPicTotalCurr */
        )
        .expect("ref_pic_lists_modification");

        assert!(m.ref_pic_list_modification_flag_l0);
        assert_eq!(m.list_entry_l0, vec![1]);
        // P slice: list-1 fields absent.
        assert!(m.ref_pic_list_modification_flag_l1.is_none());
        assert!(m.list_entry_l1.is_empty());
        // Exactly 2 bits consumed.
        assert_eq!(br.bit_pos(), 2);
    }

    /// B slice with both list-0 and list-1 modifications active.
    /// `NumPicTotalCurr == 4` so the per-entry width is
    /// `Ceil(Log2(4)) == 2` bits. With both `num_ref_idx_lX_active_minus1
    /// == 1` each list contributes 2 entries.
    ///   flag_l0 (1)=1
    ///   list_entry_l0[0] u(2)=0b10=2
    ///   list_entry_l0[1] u(2)=0b00=0
    ///   flag_l1 (1)=1
    ///   list_entry_l1[0] u(2)=0b01=1
    ///   list_entry_l1[1] u(2)=0b11=3
    /// Total = 1 + 2 + 2 + 1 + 2 + 2 = 10 bits.
    #[test]
    fn parses_b_slice_both_lists_modification() {
        let bits = concat_bits(&[
            (1, 1),    // ref_pic_list_modification_flag_l0
            (0b10, 2), // list_entry_l0[0] = 2
            (0b00, 2), // list_entry_l0[1] = 0
            (1, 1),    // ref_pic_list_modification_flag_l1
            (0b01, 2), // list_entry_l1[0] = 1
            (0b11, 2), // list_entry_l1[1] = 3
        ]);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let m = RefPicListsModification::parse(&mut br, SliceType::B, 1, 1, 4).expect("parse");

        assert!(m.ref_pic_list_modification_flag_l0);
        assert_eq!(m.list_entry_l0, vec![2, 0]);
        assert_eq!(m.ref_pic_list_modification_flag_l1, Some(true));
        assert_eq!(m.list_entry_l1, vec![1, 3]);
        assert_eq!(br.bit_pos(), 10);
    }

    /// B slice where the L0 modification flag is 0 (implicit
    /// derivation): the L0 entry list is empty, and the L1 flag is
    /// read immediately after. `NumPicTotalCurr == 3` so per-entry
    /// width is `Ceil(Log2(3)) == 2` bits.
    ///   flag_l0 (1)=0
    ///   flag_l1 (1)=1
    ///   list_entry_l1[0] u(2)=0b10=2
    /// 4 bits total; the parser must NOT have consumed any L0 entries.
    #[test]
    fn parses_b_slice_l0_implicit_l1_explicit() {
        let bits = concat_bits(&[(0, 1), (1, 1), (0b10, 2)]);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let m = RefPicListsModification::parse(&mut br, SliceType::B, 0, 0, 3).expect("parse");

        assert!(!m.ref_pic_list_modification_flag_l0);
        assert!(m.list_entry_l0.is_empty());
        assert_eq!(m.ref_pic_list_modification_flag_l1, Some(true));
        assert_eq!(m.list_entry_l1, vec![2]);
        assert_eq!(br.bit_pos(), 4);
    }

    /// Both flags zero: a minimal degenerate case where the entire
    /// structure is two bits and produces empty entry lists.
    #[test]
    fn parses_b_slice_both_flags_zero() {
        let bits = concat_bits(&[(0, 1), (0, 1)]);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let m = RefPicListsModification::parse(&mut br, SliceType::B, 0, 0, 2).expect("parse");

        assert!(!m.ref_pic_list_modification_flag_l0);
        assert!(m.list_entry_l0.is_empty());
        assert_eq!(m.ref_pic_list_modification_flag_l1, Some(false));
        assert!(m.list_entry_l1.is_empty());
        assert_eq!(br.bit_pos(), 2);
    }

    /// P slice with `flag_l0 == 0`: exactly one bit consumed, no list
    /// entries, and no L1 fields present.
    #[test]
    fn parses_p_slice_l0_implicit() {
        let bits = concat_bits(&[(0, 1)]);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let m = RefPicListsModification::parse(&mut br, SliceType::P, 14, 14, 5).expect("parse");

        assert!(!m.ref_pic_list_modification_flag_l0);
        assert!(m.list_entry_l0.is_empty());
        assert!(m.ref_pic_list_modification_flag_l1.is_none());
        assert!(m.list_entry_l1.is_empty());
        assert_eq!(br.bit_pos(), 1);
    }

    /// §7.4.7.2 range check: `list_entry_l0[i]` must be
    /// `< NumPicTotalCurr`. With `NumPicTotalCurr == 3` the per-entry
    /// width is 2 bits, so the value `3` (0b11) is legally encodable
    /// but is rejected by the range check.
    #[test]
    fn list_entry_l0_value_must_be_less_than_num_pic_total_curr() {
        let bits = concat_bits(&[
            (1, 1),    // flag_l0
            (0b11, 2), // list_entry_l0[0] = 3 — illegal: must be <= 2
        ]);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let err = RefPicListsModification::parse(&mut br, SliceType::P, 0, 0, 3)
            .expect_err("out-of-range entry must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "list_entry_l0",
                got: 3,
            }
        );
    }

    /// §7.4.7.2 range check for L1: same rule, exercised through the
    /// B-slice branch.
    #[test]
    fn list_entry_l1_value_must_be_less_than_num_pic_total_curr() {
        // NumPicTotalCurr=2 → entry width=1 bit, max value=1. Send a
        // legal L0 (flag=0) then an explicit L1 (flag=1, entry=1 OK,
        // then we cannot encode 2 in 1 bit so use a 3-curr setup
        // instead to exercise the range check.)
        // Use NumPicTotalCurr=3 (2-bit entries) so 0b11=3 is illegal.
        let bits = concat_bits(&[
            (0, 1),    // flag_l0
            (1, 1),    // flag_l1
            (0b11, 2), // list_entry_l1[0] = 3 — illegal
        ]);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let err = RefPicListsModification::parse(&mut br, SliceType::B, 0, 0, 3)
            .expect_err("out-of-range L1 entry must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "list_entry_l1",
                got: 3,
            }
        );
    }

    /// The §7.3.6.2 structure is only signalled for inter slices. The
    /// parser rejects an I-slice call up front rather than reading any
    /// bits (the bitreader position must stay at 0).
    #[test]
    fn rejects_i_slice_call() {
        let rbsp = [0xFFu8; 4];
        let mut br = BitReader::new(&rbsp);
        let err = RefPicListsModification::parse(&mut br, SliceType::I, 0, 0, 2)
            .expect_err("I-slice call must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "ref_pic_lists_modification/slice_type",
                got: 2,
            }
        );
        assert_eq!(br.bit_pos(), 0);
    }

    /// The §7.3.6.1 gate guarantees `NumPicTotalCurr > 1`. The parser
    /// rejects a call with `NumPicTotalCurr <= 1` (a defensive
    /// pre-condition).
    #[test]
    fn rejects_num_pic_total_curr_le_1() {
        let rbsp = [0xFFu8; 4];
        let mut br = BitReader::new(&rbsp);
        let err = RefPicListsModification::parse(&mut br, SliceType::P, 0, 0, 1)
            .expect_err("NumPicTotalCurr <= 1 must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "ref_pic_lists_modification/NumPicTotalCurr",
                got: 1,
            }
        );
        assert_eq!(br.bit_pos(), 0);

        let mut br2 = BitReader::new(&rbsp);
        let err2 = RefPicListsModification::parse(&mut br2, SliceType::B, 0, 0, 0)
            .expect_err("NumPicTotalCurr == 0 must error");
        assert_eq!(
            err2,
            SliceError::ValueOutOfRange {
                field: "ref_pic_lists_modification/NumPicTotalCurr",
                got: 0,
            }
        );
    }

    /// `num_ref_idx_l0_active_minus1` is constrained to 0..=14 by
    /// §7.4.7.1. The parser rejects a call that violates that bound,
    /// matching the precondition documented on
    /// [`RefPicListsModification::parse`].
    #[test]
    fn rejects_num_ref_idx_l0_out_of_range() {
        let rbsp = [0xFFu8; 4];
        let mut br = BitReader::new(&rbsp);
        let err = RefPicListsModification::parse(&mut br, SliceType::P, 15, 0, 2)
            .expect_err("num_ref_idx_l0_active_minus1 > 14 must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "num_ref_idx_l0_active_minus1",
                got: 15,
            }
        );
    }

    /// Same check for L1, exercised through the B-slice branch (the
    /// L1 bound is only validated for B slices).
    #[test]
    fn rejects_num_ref_idx_l1_out_of_range() {
        let rbsp = [0xFFu8; 4];
        let mut br = BitReader::new(&rbsp);
        let err = RefPicListsModification::parse(&mut br, SliceType::B, 0, 15, 2)
            .expect_err("num_ref_idx_l1_active_minus1 > 14 must error");
        assert_eq!(
            err,
            SliceError::ValueOutOfRange {
                field: "num_ref_idx_l1_active_minus1",
                got: 15,
            }
        );
    }

    /// Maximum-active-index case: 15 entries per list (the §7.4.7.1
    /// cap, `num_ref_idx_lX_active_minus1 == 14`), with
    /// `NumPicTotalCurr == 8` (3-bit entries). Bit accounting:
    ///   flag_l0 (1) + 15 * 3 = 46 bits for the L0 portion.
    /// The P slice has no L1 fields, so the test verifies the parser
    /// reads exactly 46 bits.
    #[test]
    fn max_active_minus1_p_slice_l0() {
        let mut fields: Vec<(u32, u8)> = vec![(1, 1)]; // flag_l0 = 1
        for i in 0..15u32 {
            // entry value = i mod 8 ∈ 0..=7, fits in 3 bits, in range.
            fields.push((i % 8, 3));
        }
        let bits = concat_bits(&fields);
        let rbsp = pack_bits(&bits);
        let mut br = BitReader::new(&rbsp);
        let m = RefPicListsModification::parse(&mut br, SliceType::P, 14, 0, 8).expect("parse");
        assert!(m.ref_pic_list_modification_flag_l0);
        assert_eq!(m.list_entry_l0.len(), 15);
        for (i, &v) in m.list_entry_l0.iter().enumerate() {
            assert_eq!(v, (i as u32) % 8);
        }
        assert_eq!(br.bit_pos(), 1 + 15 * 3);
    }

    /// `Ceil(Log2(N))` width: confirm the per-entry width matches the
    /// §7.4.7.2 formula for a representative set of `NumPicTotalCurr`
    /// values by reading exactly the expected bit count from a flag=1
    /// L0 with a single entry.
    #[test]
    fn entry_width_matches_ceil_log2() {
        // (num_pic_total_curr, expected_bits_per_entry)
        let cases: &[(u32, u8)] = &[(2, 1), (3, 2), (4, 2), (5, 3), (8, 3), (9, 4), (16, 4)];
        for &(curr, w) in cases {
            // bit layout: flag_l0=1 then list_entry_l0[0]=0
            let mut bits: Vec<u8> = vec![1];
            bits.resize(1 + w as usize, 0);
            let rbsp = pack_bits(&bits);
            let mut br = BitReader::new(&rbsp);
            let m =
                RefPicListsModification::parse(&mut br, SliceType::P, 0, 0, curr).expect("parse");
            assert_eq!(m.list_entry_l0, vec![0]);
            assert_eq!(br.bit_pos(), 1 + w as usize, "curr={curr} width={w}");
        }
    }

    /// Truncated RBSP: the parser surfaces [`SliceError::Truncated`]
    /// if the buffer runs out mid-element.
    #[test]
    fn truncated_buffer_surfaces_truncated_error() {
        // flag_l0=1 declared but no bits remain for list_entry_l0[0].
        let bits: Vec<u8> = vec![1];
        let rbsp = pack_bits(&bits); // one byte: 0b1000_0000
                                     // Restrict the reader to the first bit only.
        let buf = &rbsp[..0]; // zero bytes; even flag_l0 fails
        let mut br = BitReader::new(buf);
        let err = RefPicListsModification::parse(&mut br, SliceType::P, 0, 0, 4)
            .expect_err("empty buffer must error");
        assert_eq!(err, SliceError::Truncated);
    }

    // --- bit-packing test helpers ---

    /// A `(value, width)` pair to be packed MSB-first.
    type BitField = (u32, u8);

    /// Concatenate `(value, width)` fields into a single bit vector
    /// (each entry's `width` low bits of `value`, MSB-first).
    fn concat_bits(fields: &[BitField]) -> Vec<u8> {
        let mut bits = Vec::new();
        for &(value, width) in fields {
            for i in (0..width).rev() {
                bits.push(((value >> i) & 1) as u8);
            }
        }
        bits
    }

    /// Pack an MSB-first bit vector into bytes, zero-padding the final
    /// byte (matching `byte_alignment()`'s zero pad).
    fn pack_bits(bits: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                out[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        out
    }
}
