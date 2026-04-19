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
use crate::nal::{NalHeader, NalUnitType};
use crate::pps::PicParameterSet;
use crate::sps::{parse_st_ref_pic_set, SeqParameterSet, ShortTermRps};

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
    /// Bit position (in the RBSP) where the entropy payload (`slice_data()`)
    /// starts after the final `byte_alignment()`. Valid iff `is_full_i_slice`
    /// or `is_full_p_slice`.
    pub slice_data_bit_offset: u64,
    /// Set when the full I-slice extension parse succeeded.
    pub is_full_i_slice: bool,
    /// Set when the full P-slice extension parse succeeded (inter decode).
    pub is_full_p_slice: bool,
    /// Resolved short-term RPS for the current picture, if any.
    pub current_rps: Option<ShortTermRps>,
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
    let mut slice_sao_luma_flag = false;
    let mut slice_sao_chroma_flag = false;
    let mut cabac_init_flag = false;
    let mut slice_qp_delta: i32 = 0;
    let mut slice_cb_qp_offset: i32 = 0;
    let mut slice_cr_qp_offset: i32 = 0;
    let mut slice_loop_filter_across_slices_enabled_flag =
        pps.pps_loop_filter_across_slices_enabled_flag;
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
                // We don't support long-term refs — bail out early if any are
                // signalled. Parse just enough to detect.
                let num_long_term_sps = 0u32;
                let num_long_term_pics = br.ue()?;
                if num_long_term_sps + num_long_term_pics > 0 {
                    return Ok(partial_header(
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
                        current_rps,
                        pps,
                    ));
                }
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
            if pps.lists_modification_present_flag {
                // Slice-level list modification is not currently supported —
                // bail out of the full-parse path. The caller will surface
                // `Unsupported` on receive_frame.
                return Ok(partial_header(
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
                    current_rps,
                    pps,
                ));
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
                    let _collocated_ref_idx = br.ue()?;
                }
            }
            if (pps.weighted_pred_flag && slice_type == SliceType::P)
                || (pps.weighted_bipred_flag && slice_type == SliceType::B)
            {
                // Walk past pred_weight_table without storing — single-ref
                // weighted prediction path that we currently treat as
                // out-of-scope (the MC pipeline uses unit weights).
                skip_pred_weight_table(
                    &mut br,
                    sps,
                    slice_type,
                    num_ref_idx_l0_active_minus1,
                    num_ref_idx_l1_active_minus1,
                )?;
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
        let mut slice_deblocking_filter_disabled_flag = pps.pps_deblocking_filter_disabled_flag;
        if pps.deblocking_filter_override_enabled_flag {
            let override_flag = br.u1()? == 1;
            if override_flag {
                slice_deblocking_filter_disabled_flag = br.u1()? == 1;
                if !slice_deblocking_filter_disabled_flag {
                    let _slice_beta_offset_div2 = br.se()?;
                    let _slice_tc_offset_div2 = br.se()?;
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
            let num_entry_point_offsets = br.ue()?;
            if num_entry_point_offsets > 0 {
                // Entry-point offsets — needed only for tiles / WPP which is
                // out of scope. Abort the extension parse.
                return Ok(partial_header(
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
                    current_rps,
                    pps,
                ));
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
            // B slice is out of scope.
            SliceType::B => {}
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
        slice_data_bit_offset,
        is_full_i_slice,
        is_full_p_slice,
        current_rps,
        num_ref_idx_l0_active_minus1,
        num_ref_idx_l1_active_minus1,
        mvd_l1_zero_flag,
        collocated_from_l0_flag,
        five_minus_max_num_merge_cand,
        max_num_merge_cand,
        slice_temporal_mvp_enabled_flag,
        slice_cb_qp_offset,
        slice_cr_qp_offset,
    })
}

/// Skip a `pred_weight_table()` body (§7.4.7.3) — we consume but don't
/// expose the weights yet. Returns `Err` if the chroma-format query is
/// inconsistent.
fn skip_pred_weight_table(
    br: &mut BitReader<'_>,
    sps: &SeqParameterSet,
    slice_type: SliceType,
    num_ref_idx_l0_active_minus1: u32,
    num_ref_idx_l1_active_minus1: u32,
) -> Result<()> {
    let _luma_log2_weight_denom = br.ue()?;
    let chroma_array_type = if sps.separate_colour_plane_flag {
        0
    } else {
        sps.chroma_format_idc
    };
    if chroma_array_type != 0 {
        let _delta_chroma_log2_weight_denom = br.se()?;
    }
    let consume = |br: &mut BitReader<'_>, count: u32| -> Result<()> {
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
        for i in 0..count as usize {
            if luma_flags[i] {
                let _delta_luma_weight = br.se()?;
                let _luma_offset = br.se()?;
            }
            if chroma_array_type != 0 && chroma_flags[i] {
                for _ in 0..2 {
                    let _delta_chroma_weight = br.se()?;
                    let _delta_chroma_offset = br.se()?;
                }
            }
        }
        Ok(())
    };
    consume(br, num_ref_idx_l0_active_minus1 + 1)?;
    if slice_type == SliceType::B {
        consume(br, num_ref_idx_l1_active_minus1 + 1)?;
    }
    Ok(())
}

/// Build a `SliceSegmentHeader` for the "abort extension parse" path with
/// whatever we decoded before the unsupported feature.
#[allow(clippy::too_many_arguments)]
fn partial_header(
    first_slice_segment_in_pic_flag: bool,
    no_output_of_prior_pics_flag: bool,
    slice_pic_parameter_set_id: u32,
    dependent_slice_segment_flag: bool,
    slice_segment_address: u32,
    slice_type: SliceType,
    pic_output_flag: bool,
    colour_plane_id: u8,
    slice_pic_order_cnt_lsb: u32,
    short_term_ref_pic_set_sps_flag: bool,
    short_term_ref_pic_set_idx: u32,
    current_rps: Option<ShortTermRps>,
    pps: &PicParameterSet,
) -> SliceSegmentHeader {
    SliceSegmentHeader {
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
        slice_sao_luma_flag: false,
        slice_sao_chroma_flag: false,
        cabac_init_flag: false,
        slice_qp_delta: 0,
        slice_qp_y: 26 + pps.init_qp_minus26,
        slice_loop_filter_across_slices_enabled_flag: pps.pps_loop_filter_across_slices_enabled_flag,
        slice_data_bit_offset: 0,
        is_full_i_slice: false,
        is_full_p_slice: false,
        current_rps,
        num_ref_idx_l0_active_minus1: pps.num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_active_minus1: pps.num_ref_idx_l1_default_active_minus1,
        mvd_l1_zero_flag: false,
        collocated_from_l0_flag: true,
        five_minus_max_num_merge_cand: 0,
        max_num_merge_cand: 5,
        slice_temporal_mvp_enabled_flag: false,
        slice_cb_qp_offset: 0,
        slice_cr_qp_offset: 0,
    }
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
}
