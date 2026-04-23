//! HEVC slice segment header parser (§7.3.6).
//!
//! Only the leading "I always need this to identify the slice" portion is
//! exposed in v1. We parse enough to:
//!
//! * Tell whether this is the first slice in the picture and where in the
//!   CTB grid it starts.
//! * Identify the active PPS (and via it, the active SPS).
//! * Read the slice type and the picture-order-count LSB.
//!
//! Reference picture list modifications, weighted prediction tables, and
//! the post-header `slice_data()` are deferred — once we plumb CABAC, those
//! sit naturally on top.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::inter::WeightedPred;
use crate::nal::{NalHeader, NalUnitType};
use crate::pps::PicParameterSet;
use crate::sps::{parse_st_ref_pic_set, SeqParameterSet, ShortTermRps};

/// A single long-term reference picture entry resolved from either the
/// SPS-level candidate list or the slice-level inline signalling
/// (§7.3.6.1). The POC for the LT ref is computed as:
///
/// * If `delta_poc_msb_cycle_lt` is `None`, POC LSB matches on its own
///   (the DPB is searched for a picture whose POC & (MaxPocLsb - 1) ==
///   `poc_lsb_lt`).
/// * Otherwise the full POC is derived as
///   `PocLtCurr = current_poc - (delta_poc_msb_cycle * MaxPocLsb) -
///   ((current_poc & (MaxPocLsb - 1)) - poc_lsb_lt)` (§8.3.2 eq. 8-5).
#[derive(Clone, Copy, Debug)]
pub struct LongTermRef {
    pub poc_lsb_lt: u32,
    pub used_by_curr_pic_lt_flag: bool,
    pub delta_poc_msb_cycle_lt: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceType {
    B,
    P,
    I,
}

impl SliceType {
    pub fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(SliceType::B),
            1 => Ok(SliceType::P),
            2 => Ok(SliceType::I),
            _ => Err(Error::invalid(format!(
                "h265 slice: invalid slice_type {v}"
            ))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SliceSegmentHeader {
    pub first_slice_segment_in_pic_flag: bool,
    pub no_output_of_prior_pics_flag: bool,
    pub slice_pic_parameter_set_id: u32,
    pub dependent_slice_segment_flag: bool,
    pub slice_segment_address: u32,
    pub slice_type: SliceType,
    pub pic_output_flag: bool,
    pub colour_plane_id: u8,
    pub slice_pic_order_cnt_lsb: u32,
    pub short_term_ref_pic_set_sps_flag: bool,
    pub short_term_ref_pic_set_idx: u32,
    /// `slice_sao_luma_flag` (only when SAO is enabled at SPS level). For
    /// the minimal I-slice extension we only fill this for IDR slices.
    pub slice_sao_luma_flag: bool,
    /// `slice_sao_chroma_flag` (only when SAO is enabled and chroma is
    /// present).
    pub slice_sao_chroma_flag: bool,
    /// `cabac_init_flag` (§7.4.7.1). Only signalled for P/B slices when
    /// `cabac_init_present_flag` is set on the PPS; for I slices this is
    /// always false.
    pub cabac_init_flag: bool,
    /// `slice_qp_delta` (se(v)).
    pub slice_qp_delta: i32,
    /// `SliceQpY` derived per §7.4.7.1: `26 + init_qp_minus26 + slice_qp_delta`.
    pub slice_qp_y: i32,
    /// `slice_loop_filter_across_slices_enabled_flag` (defaults from PPS).
    pub slice_loop_filter_across_slices_enabled_flag: bool,
    /// `slice_deblocking_filter_disabled_flag` after PPS default + optional
    /// slice override resolution.
    pub slice_deblocking_filter_disabled_flag: bool,
    /// `slice_beta_offset_div2` (§7.4.7.1). Defaults to PPS β offset; may
    /// be overridden when `deblocking_filter_override_flag` is set.
    pub slice_beta_offset_div2: i32,
    /// `slice_tc_offset_div2` (§7.4.7.1). Defaults to PPS tC offset.
    pub slice_tc_offset_div2: i32,
    /// Bit position (in the RBSP) where the entropy payload (`slice_data()`)
    /// starts after the final `byte_alignment()`. Valid iff `is_full_i_slice`
    /// or `is_full_p_slice`.
    pub slice_data_bit_offset: u64,
    /// Set when the full I-slice extension parse succeeded.
    pub is_full_i_slice: bool,
    /// Set when the full P-slice extension parse succeeded (inter decode).
    pub is_full_p_slice: bool,
    /// Set when the full B-slice extension parse succeeded (inter decode).
    pub is_full_b_slice: bool,
    /// `collocated_ref_idx` (§7.4.7.1). Only meaningful when
    /// `slice_temporal_mvp_enabled_flag` is true. 0 when absent.
    pub collocated_ref_idx: u32,
    /// Parsed `pred_weight_table()` when weighted prediction is enabled
    /// (P-slice with `pps.weighted_pred_flag` or B-slice with
    /// `pps.weighted_bipred_flag`).
    pub weighted_pred: Option<WeightedPred>,
    /// Resolved short-term RPS for the current picture, if any.
    pub current_rps: Option<ShortTermRps>,
    /// Long-term reference picture entries for the current picture, in the
    /// order they appear in the slice header (§7.3.6.1). Each entry carries
    /// the POC LSB, the optional MSB-cycle delta, and the
    /// `used_by_curr_pic_lt_flag`. Empty when no LT refs are signalled.
    pub long_term_refs: Vec<LongTermRef>,
    /// `num_ref_idx_l0_active_minus1` (defaults from PPS). Slices may
    /// override with `num_ref_idx_active_override_flag`.
    pub num_ref_idx_l0_active_minus1: u32,
    /// Similarly for L1.
    pub num_ref_idx_l1_active_minus1: u32,
    /// `mvd_l1_zero_flag` (B slice only; defaults to false).
    pub mvd_l1_zero_flag: bool,
    /// `collocated_from_l0_flag` (§7.4.7.1) — only when TMVP is enabled.
    pub collocated_from_l0_flag: bool,
    /// `five_minus_max_num_merge_cand` (§7.4.7.1) — slice-level override.
    pub five_minus_max_num_merge_cand: u32,
    /// `MaxNumMergeCand = 5 - five_minus_max_num_merge_cand` (§7.4.7.1).
    pub max_num_merge_cand: u32,
    /// `slice_temporal_mvp_enabled_flag` — per-slice override.
    pub slice_temporal_mvp_enabled_flag: bool,
    /// `slice_cb_qp_offset` (defaults to 0).
    pub slice_cb_qp_offset: i32,
    /// `slice_cr_qp_offset` (defaults to 0).
    pub slice_cr_qp_offset: i32,
    /// `ref_pic_list_modification_flag_l0` (§7.3.6.2). `false` when the
    /// explicit reordering syntax is absent.
    pub ref_pic_list_modification_flag_l0: bool,
    /// `list_entry_l0[i]` (§7.3.6.2). Length `num_ref_idx_l0_active_minus1+1`
    /// when `ref_pic_list_modification_flag_l0 == 1`, empty otherwise.
    pub list_entry_l0: Vec<u32>,
    /// `ref_pic_list_modification_flag_l1` (§7.3.6.2). `false` when the
    /// explicit reordering syntax is absent.
    pub ref_pic_list_modification_flag_l1: bool,
    /// `list_entry_l1[i]` (§7.3.6.2).
    pub list_entry_l1: Vec<u32>,
    /// `num_entry_point_offsets` (§7.3.6.3). Present only when tiles or
    /// WPP are enabled. Zero otherwise; also zero for the single-tile /
    /// single-CTU-row WPP degenerate cases.
    pub num_entry_point_offsets: u32,
    /// `offset_len_minus1` — width in bits of each `entry_point_offset`
    /// code when `num_entry_point_offsets > 0`.
    pub offset_len_minus1: u32,
    /// Entry-point offsets in bytes from the end of the slice segment
    /// header, after the final `byte_alignment()`. `entry_point_offsets[i]`
    /// is the cumulative byte offset of the (i+1)-th sub-stream start
    /// (tile or CTU row) — equal to
    /// `sum(entry_point_offset_minus1[0..=i] + 1)`. Index 0 of a slice is
    /// always at offset 0 (the slice_data_bit_offset itself).
    pub entry_point_offsets: Vec<u64>,
}

/// Parse a slice segment header given the active PPS+SPS and the NAL header
/// that introduced this slice. The bitstream `rbsp` is the bytes after the
/// 2-byte NAL header, with emulation-prevention removed.
pub fn parse_slice_segment_header(
    rbsp: &[u8],
    nal: &NalHeader,
    sps: &SeqParameterSet,
    pps: &PicParameterSet,
) -> Result<SliceSegmentHeader> {
    let mut br = BitReader::new(rbsp);
    let first_slice_segment_in_pic_flag = br.u1()? == 1;

    let mut no_output_of_prior_pics_flag = false;
    if nal.nal_unit_type.is_irap() {
        no_output_of_prior_pics_flag = br.u1()? == 1;
    }
    let slice_pic_parameter_set_id = br.ue()?;
    if slice_pic_parameter_set_id != pps.pps_pic_parameter_set_id {
        return Err(Error::invalid(format!(
            "h265 slice: PPS id mismatch (slice {slice_pic_parameter_set_id} vs active {})",
            pps.pps_pic_parameter_set_id
        )));
    }

    let mut dependent_slice_segment_flag = false;
    let mut slice_segment_address: u32 = 0;
    if !first_slice_segment_in_pic_flag {
        if pps.dependent_slice_segments_enabled_flag {
            dependent_slice_segment_flag = br.u1()? == 1;
        }
        let num_ctbs = num_ctbs_in_pic(sps);
        let n = ceil_log2(num_ctbs);
        slice_segment_address = br.u(n)?;
    }

    let mut slice_type = SliceType::I;
    let mut pic_output_flag = true;
    let mut colour_plane_id = 0u8;
    let mut slice_pic_order_cnt_lsb = 0u32;
    let mut short_term_ref_pic_set_sps_flag = false;
    let mut short_term_ref_pic_set_idx = 0u32;
    let mut current_rps: Option<ShortTermRps> = None;
    let mut long_term_refs: Vec<LongTermRef> = Vec::new();
    let mut slice_sao_luma_flag = false;
    let mut slice_sao_chroma_flag = false;
    let mut cabac_init_flag = false;
    let mut slice_qp_delta: i32 = 0;
    let mut slice_cb_qp_offset: i32 = 0;
    let mut slice_cr_qp_offset: i32 = 0;
    let mut slice_loop_filter_across_slices_enabled_flag =
        pps.pps_loop_filter_across_slices_enabled_flag;
    let mut slice_deblocking_filter_disabled_flag = pps.pps_deblocking_filter_disabled_flag;
    let mut slice_beta_offset_div2: i32 = pps.pps_beta_offset_div2;
    let mut slice_tc_offset_div2: i32 = pps.pps_tc_offset_div2;
    let mut num_ref_idx_l0_active_minus1 = pps.num_ref_idx_l0_default_active_minus1;
    let mut num_ref_idx_l1_active_minus1 = pps.num_ref_idx_l1_default_active_minus1;
    let mut mvd_l1_zero_flag = false;
    let mut collocated_from_l0_flag = true;
    let mut five_minus_max_num_merge_cand: u32 = 0;
    let mut max_num_merge_cand: u32 = 5;
    let mut slice_temporal_mvp_enabled_flag = false;

    let is_idr = matches!(
        nal.nal_unit_type,
        NalUnitType::IdrWRadl | NalUnitType::IdrNLp
    );

    if !dependent_slice_segment_flag {
        // Skip num_extra_slice_header_bits flags.
        for _ in 0..pps.num_extra_slice_header_bits {
            br.skip(1)?;
        }
        let st = br.ue()?;
        slice_type = SliceType::from_u32(st)?;
        if pps.output_flag_present_flag {
            pic_output_flag = br.u1()? == 1;
        }
        if sps.separate_colour_plane_flag {
            colour_plane_id = br.u(2)? as u8;
        }
        if !is_idr {
            let lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 + 4;
            slice_pic_order_cnt_lsb = br.u(lsb_bits)?;
            short_term_ref_pic_set_sps_flag = br.u1()? == 1;
            if short_term_ref_pic_set_sps_flag {
                if sps.num_short_term_ref_pic_sets > 1 {
                    let n = ceil_log2(sps.num_short_term_ref_pic_sets);
                    short_term_ref_pic_set_idx = br.u(n)?;
                }
                if (short_term_ref_pic_set_idx as usize) < sps.short_term_ref_pic_sets.len() {
                    current_rps = Some(
                        sps.short_term_ref_pic_sets[short_term_ref_pic_set_idx as usize].clone(),
                    );
                }
            } else {
                // Inline `st_ref_pic_set(num_short_term_ref_pic_sets)`.
                let rps = parse_st_ref_pic_set(
                    &mut br,
                    sps.num_short_term_ref_pic_sets,
                    sps.num_short_term_ref_pic_sets,
                    &sps.short_term_ref_pic_sets,
                )?;
                current_rps = Some(rps);
            }
            if sps.long_term_ref_pics_present_flag {
                // §7.3.6.1 long-term RPS entries. `num_long_term_sps`
                // references the SPS-level candidate list; the remaining
                // `num_long_term_pics` entries are inline.
                let num_long_term_sps = if sps.num_long_term_ref_pics_sps > 0 {
                    br.ue()?
                } else {
                    0
                };
                let num_long_term_pics = br.ue()?;
                let total_lt = num_long_term_sps + num_long_term_pics;
                if total_lt > sps.num_long_term_ref_pics_sps + 32 {
                    return Err(Error::invalid(format!(
                        "h265 slice: num_long_term_{{sps,pics}} out of range ({num_long_term_sps} + {num_long_term_pics})"
                    )));
                }
                let lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 + 4;
                let lt_idx_bits = ceil_log2(sps.num_long_term_ref_pics_sps);
                let mut lts: Vec<LongTermRef> = Vec::with_capacity(total_lt as usize);
                for i in 0..total_lt {
                    let (poc_lsb_lt, used_flag) = if i < num_long_term_sps {
                        let lt_idx_sps = if sps.num_long_term_ref_pics_sps > 1 {
                            br.u(lt_idx_bits)?
                        } else {
                            0
                        };
                        let idx = lt_idx_sps as usize;
                        if idx >= sps.lt_ref_pic_poc_lsb_sps.len() {
                            return Err(Error::invalid(format!(
                                "h265 slice: lt_idx_sps {lt_idx_sps} out of range"
                            )));
                        }
                        (
                            sps.lt_ref_pic_poc_lsb_sps[idx],
                            sps.used_by_curr_pic_lt_sps_flag[idx],
                        )
                    } else {
                        let poc = br.u(lsb_bits)?;
                        let used = br.u1()? == 1;
                        (poc, used)
                    };
                    let delta_poc_msb_present_flag = br.u1()? == 1;
                    let delta_poc_msb_cycle_lt = if delta_poc_msb_present_flag {
                        Some(br.ue()?)
                    } else {
                        None
                    };
                    lts.push(LongTermRef {
                        poc_lsb_lt,
                        used_by_curr_pic_lt_flag: used_flag,
                        delta_poc_msb_cycle_lt,
                    });
                }
                long_term_refs = lts;
            }
            if sps.sps_temporal_mvp_enabled_flag {
                slice_temporal_mvp_enabled_flag = br.u1()? == 1;
            }
        }
    }

    // From here: I-slice extension is handled identically to before; P/B
    // slices get the inter_pred section + ref_idx flags + merge candidates.
    let mut slice_data_bit_offset: u64 = 0;
    let mut is_full_i_slice = false;
    let mut is_full_p_slice = false;
    let mut is_full_b_slice = false;
    let mut collocated_ref_idx: u32 = 0;
    let mut weighted_pred: Option<WeightedPred> = None;
    let mut ref_pic_list_modification_flag_l0 = false;
    let mut ref_pic_list_modification_flag_l1 = false;
    let mut list_entry_l0: Vec<u32> = Vec::new();
    let mut list_entry_l1: Vec<u32> = Vec::new();
    let mut num_entry_point_offsets: u32 = 0;
    let mut offset_len_minus1: u32 = 0;
    let mut entry_point_offsets: Vec<u64> = Vec::new();

    // Only attempt the full parse for single-segment slices (not dependent).
    if !dependent_slice_segment_flag && !sps.separate_colour_plane_flag {
        if sps.sample_adaptive_offset_enabled_flag {
            slice_sao_luma_flag = br.u1()? == 1;
            let chroma_array_type = if sps.separate_colour_plane_flag {
                0
            } else {
                sps.chroma_format_idc
            };
            if chroma_array_type != 0 {
                slice_sao_chroma_flag = br.u1()? == 1;
            }
        }

        if slice_type == SliceType::P || slice_type == SliceType::B {
            let num_ref_idx_active_override_flag = br.u1()? == 1;
            if num_ref_idx_active_override_flag {
                num_ref_idx_l0_active_minus1 = br.ue()?;
                if slice_type == SliceType::B {
                    num_ref_idx_l1_active_minus1 = br.ue()?;
                }
            }
            // `ref_pic_list_modification()` (§7.3.6.2). Present only when
            // `lists_modification_present_flag == 1 && NumPicTotalCurr > 1`.
            // NumPicTotalCurr counts the entries used by the current picture
            // across st_curr_before, st_curr_after, and lt_curr (§7.4.7.2).
            if pps.lists_modification_present_flag {
                let num_pic_total_curr =
                    compute_num_pic_total_curr(current_rps.as_ref(), &long_term_refs);
                if num_pic_total_curr > 1 {
                    let bits = ceil_log2(num_pic_total_curr);
                    ref_pic_list_modification_flag_l0 = br.u1()? == 1;
                    if ref_pic_list_modification_flag_l0 {
                        list_entry_l0.reserve(num_ref_idx_l0_active_minus1 as usize + 1);
                        for _ in 0..=num_ref_idx_l0_active_minus1 {
                            list_entry_l0.push(br.u(bits)?);
                        }
                    }
                    if slice_type == SliceType::B {
                        ref_pic_list_modification_flag_l1 = br.u1()? == 1;
                        if ref_pic_list_modification_flag_l1 {
                            list_entry_l1.reserve(num_ref_idx_l1_active_minus1 as usize + 1);
                            for _ in 0..=num_ref_idx_l1_active_minus1 {
                                list_entry_l1.push(br.u(bits)?);
                            }
                        }
                    }
                }
            }
            if slice_type == SliceType::B {
                mvd_l1_zero_flag = br.u1()? == 1;
            }
            if pps.cabac_init_present_flag {
                cabac_init_flag = br.u1()? == 1;
            }
            if slice_temporal_mvp_enabled_flag {
                if slice_type == SliceType::B {
                    collocated_from_l0_flag = br.u1()? == 1;
                }
                let needs_collocated = (collocated_from_l0_flag
                    && num_ref_idx_l0_active_minus1 > 0)
                    || (!collocated_from_l0_flag && num_ref_idx_l1_active_minus1 > 0);
                if needs_collocated {
                    collocated_ref_idx = br.ue()?;
                }
            }
            if (pps.weighted_pred_flag && slice_type == SliceType::P)
                || (pps.weighted_bipred_flag && slice_type == SliceType::B)
            {
                weighted_pred = Some(parse_pred_weight_table(
                    &mut br,
                    sps,
                    slice_type,
                    num_ref_idx_l0_active_minus1,
                    num_ref_idx_l1_active_minus1,
                )?);
            }
            five_minus_max_num_merge_cand = br.ue()?;
            if five_minus_max_num_merge_cand > 4 {
                return Err(Error::invalid(
                    "h265 slice: five_minus_max_num_merge_cand out of range",
                ));
            }
            max_num_merge_cand = 5 - five_minus_max_num_merge_cand;
        }

        slice_qp_delta = br.se()?;
        if pps.pps_slice_chroma_qp_offsets_present_flag {
            slice_cb_qp_offset = br.se()?;
            slice_cr_qp_offset = br.se()?;
        }
        if pps.deblocking_filter_override_enabled_flag {
            let override_flag = br.u1()? == 1;
            if override_flag {
                slice_deblocking_filter_disabled_flag = br.u1()? == 1;
                if !slice_deblocking_filter_disabled_flag {
                    slice_beta_offset_div2 = br.se()?;
                    slice_tc_offset_div2 = br.se()?;
                }
            }
        }
        if pps.pps_loop_filter_across_slices_enabled_flag
            && (slice_sao_luma_flag
                || slice_sao_chroma_flag
                || !slice_deblocking_filter_disabled_flag)
        {
            slice_loop_filter_across_slices_enabled_flag = br.u1()? == 1;
        }
        if pps.tiles_enabled_flag || pps.entropy_coding_sync_enabled_flag {
            // §7.3.6.3 entry_point_offsets. Each `entry_point_offset_minus1[i]`
            // is coded as `u(offset_len_minus1+1)`. The spec defines
            // `entry_point_offset_minus1[i]` such that the byte distance
            // from the start of the previous sub-stream to the start of
            // sub-stream i+1 is `entry_point_offset_minus1[i] + 1`. We keep
            // cumulative offsets (bytes) so the decoder can jump straight
            // to each sub-stream's CABAC restart point.
            let n = br.ue()?;
            if n > 0 {
                let ol = br.ue()?;
                if ol > 31 {
                    return Err(Error::invalid(format!(
                        "h265 slice: offset_len_minus1 out of range ({ol})"
                    )));
                }
                num_entry_point_offsets = n;
                offset_len_minus1 = ol;
                entry_point_offsets.reserve(n as usize);
                let mut cum: u64 = 0;
                for _ in 0..n {
                    let raw = br.u(ol + 1)? as u64;
                    cum += raw + 1;
                    entry_point_offsets.push(cum);
                }
            }
        }
        if pps.slice_segment_header_extension_present_flag {
            let ext_len = br.ue()?;
            for _ in 0..ext_len {
                br.skip(8)?;
            }
        }
        // byte_alignment(): one '1' bit followed by zero bits to the next byte.
        let stop = br.u1()?;
        if stop != 1 {
            return Err(Error::invalid(
                "h265 slice: byte_alignment() expected stop bit = 1",
            ));
        }
        // Align to byte.
        let pos_before_align = br.bit_position();
        let pad = (8 - (pos_before_align % 8)) % 8;
        if pad > 0 {
            br.skip(pad as u32)?;
        }
        slice_data_bit_offset = br.bit_position();
        match slice_type {
            SliceType::I if is_idr => is_full_i_slice = true,
            SliceType::I => is_full_i_slice = true,
            SliceType::P => is_full_p_slice = true,
            SliceType::B => is_full_b_slice = true,
        }
    }

    Ok(SliceSegmentHeader {
        first_slice_segment_in_pic_flag,
        no_output_of_prior_pics_flag,
        slice_pic_parameter_set_id,
        dependent_slice_segment_flag,
        slice_segment_address,
        slice_type,
        pic_output_flag,
        colour_plane_id,
        slice_pic_order_cnt_lsb,
        short_term_ref_pic_set_sps_flag,
        short_term_ref_pic_set_idx,
        slice_sao_luma_flag,
        slice_sao_chroma_flag,
        cabac_init_flag,
        slice_qp_delta,
        slice_qp_y: 26 + pps.init_qp_minus26 + slice_qp_delta,
        slice_loop_filter_across_slices_enabled_flag,
        slice_deblocking_filter_disabled_flag,
        slice_beta_offset_div2,
        slice_tc_offset_div2,
        slice_data_bit_offset,
        is_full_i_slice,
        is_full_p_slice,
        is_full_b_slice,
        collocated_ref_idx,
        weighted_pred,
        current_rps,
        long_term_refs,
        num_ref_idx_l0_active_minus1,
        num_ref_idx_l1_active_minus1,
        mvd_l1_zero_flag,
        collocated_from_l0_flag,
        five_minus_max_num_merge_cand,
        max_num_merge_cand,
        slice_temporal_mvp_enabled_flag,
        slice_cb_qp_offset,
        slice_cr_qp_offset,
        ref_pic_list_modification_flag_l0,
        list_entry_l0,
        ref_pic_list_modification_flag_l1,
        list_entry_l1,
        num_entry_point_offsets,
        offset_len_minus1,
        entry_point_offsets,
    })
}

/// §7.4.7.2: `NumPicTotalCurr = NumPocStCurrBefore + NumPocStCurrAfter +
/// NumPocLtCurr`. Each count only includes entries with
/// `used_by_curr_pic_*_flag == 1`. Long-term entries with
/// `used_by_curr_pic_lt_flag == 1` also count.
fn compute_num_pic_total_curr(
    current_rps: Option<&ShortTermRps>,
    long_term_refs: &[LongTermRef],
) -> u32 {
    let st = current_rps
        .map(|r| {
            r.used_by_curr_pic_s0.iter().filter(|&&u| u).count()
                + r.used_by_curr_pic_s1.iter().filter(|&&u| u).count()
        })
        .unwrap_or(0);
    let lt = long_term_refs
        .iter()
        .filter(|l| l.used_by_curr_pic_lt_flag)
        .count();
    (st + lt) as u32
}

/// Parse a `pred_weight_table()` body (§7.4.7.3) into a [`WeightedPred`].
/// Weights are stored as spec-signed values; the MC pipeline consumes
/// them via [`WeightedPred::luma_weight_l0`] & friends.
fn parse_pred_weight_table(
    br: &mut BitReader<'_>,
    sps: &SeqParameterSet,
    slice_type: SliceType,
    num_ref_idx_l0_active_minus1: u32,
    num_ref_idx_l1_active_minus1: u32,
) -> Result<WeightedPred> {
    let luma_denom = br.ue()?;
    let chroma_array_type = if sps.separate_colour_plane_flag {
        0
    } else {
        sps.chroma_format_idc
    };
    let mut chroma_denom: u32 = 0;
    if chroma_array_type != 0 {
        let delta = br.se()?;
        chroma_denom = (luma_denom as i32 + delta).max(0) as u32;
    }
    type LumaList = Vec<(bool, i32, i32)>;
    type ChromaList = Vec<[(bool, i32, i32); 2]>;
    let parse_list = |br: &mut BitReader<'_>, count: u32| -> Result<(LumaList, ChromaList)> {
        let mut luma_flags = vec![false; count as usize];
        for f in luma_flags.iter_mut() {
            *f = br.u1()? == 1;
        }
        let mut chroma_flags = vec![false; count as usize];
        if chroma_array_type != 0 {
            for f in chroma_flags.iter_mut() {
                *f = br.u1()? == 1;
            }
        }
        let mut lumas: LumaList = vec![(false, 1 << luma_denom, 0); count as usize];
        let mut chromas: ChromaList = vec![[(false, 1 << chroma_denom, 0); 2]; count as usize];
        for i in 0..count as usize {
            if luma_flags[i] {
                let delta_w = br.se()?;
                let w = (1i32 << luma_denom) + delta_w;
                let o = br.se()?;
                lumas[i] = (true, w, o);
            }
            if chroma_array_type != 0 && chroma_flags[i] {
                for slot in chromas[i].iter_mut() {
                    let delta_w = br.se()?;
                    let w = (1i32 << chroma_denom) + delta_w;
                    let delta_o = br.se()?;
                    *slot = (true, w, delta_o);
                }
            }
        }
        Ok((lumas, chromas))
    };
    let (l0_luma, l0_chroma) = parse_list(br, num_ref_idx_l0_active_minus1 + 1)?;
    let (l1_luma, l1_chroma) = if slice_type == SliceType::B {
        parse_list(br, num_ref_idx_l1_active_minus1 + 1)?
    } else {
        (Vec::new(), Vec::new())
    };
    Ok(WeightedPred {
        luma_denom,
        chroma_denom,
        l0_luma,
        l0_chroma,
        l1_luma,
        l1_chroma,
    })
}

/// PicWidthInCtbsY * PicHeightInCtbsY (§7.4.7.1).
pub fn num_ctbs_in_pic(sps: &SeqParameterSet) -> u32 {
    let ctb = sps.ctb_size().max(1);
    let w = sps.pic_width_in_luma_samples.div_ceil(ctb);
    let h = sps.pic_height_in_luma_samples.div_ceil(ctb);
    w * h
}

/// Bits required to address `n` distinct values: `ceil(log2(n))`. Returns 0
/// for n ≤ 1.
fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        return 0;
    }
    32 - (n - 1).leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_log2_matches() {
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(8), 3);
        assert_eq!(ceil_log2(9), 4);
    }

    #[test]
    fn num_pic_total_curr_sums_used_flags() {
        // Four st entries, two marked "used by curr" + one LT used + one LT
        // unused → NumPicTotalCurr = 3.
        let rps = ShortTermRps {
            delta_poc_s0: vec![-1, -2],
            delta_poc_s1: vec![1, 2],
            used_by_curr_pic_s0: vec![true, false],
            used_by_curr_pic_s1: vec![true, false],
        };
        let lts = vec![
            LongTermRef {
                poc_lsb_lt: 0,
                used_by_curr_pic_lt_flag: true,
                delta_poc_msb_cycle_lt: None,
            },
            LongTermRef {
                poc_lsb_lt: 1,
                used_by_curr_pic_lt_flag: false,
                delta_poc_msb_cycle_lt: None,
            },
        ];
        assert_eq!(compute_num_pic_total_curr(Some(&rps), &lts), 3);
        assert_eq!(compute_num_pic_total_curr(None, &[]), 0);
        assert_eq!(compute_num_pic_total_curr(None, &lts), 1);
    }

    #[test]
    fn long_term_ref_carries_lsb_and_flags() {
        // Sanity: the LongTermRef struct round-trips the three spec fields.
        let lt = LongTermRef {
            poc_lsb_lt: 42,
            used_by_curr_pic_lt_flag: true,
            delta_poc_msb_cycle_lt: Some(2),
        };
        assert_eq!(lt.poc_lsb_lt, 42);
        assert!(lt.used_by_curr_pic_lt_flag);
        assert_eq!(lt.delta_poc_msb_cycle_lt, Some(2));
    }
}
