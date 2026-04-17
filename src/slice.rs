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
use crate::sps::SeqParameterSet;

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
    /// starts after the final `byte_alignment()`. Valid iff `is_full_i_slice`.
    pub slice_data_bit_offset: u64,
    /// Set when the full I-slice extension parse succeeded. This is the flag
    /// tests use to confirm the decoder reached the CTU-decode boundary.
    pub is_full_i_slice: bool,
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
        if !matches!(
            nal.nal_unit_type,
            NalUnitType::IdrWRadl | NalUnitType::IdrNLp
        ) {
            let lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 + 4;
            slice_pic_order_cnt_lsb = br.u(lsb_bits)?;
            short_term_ref_pic_set_sps_flag = br.u1()? == 1;
            if short_term_ref_pic_set_sps_flag && sps.num_short_term_ref_pic_sets > 1 {
                let n = ceil_log2(sps.num_short_term_ref_pic_sets);
                short_term_ref_pic_set_idx = br.u(n)?;
            }
            // Inline-RPS (st_ref_pic_set in the slice header) and long-term
            // refs aren't needed for the v1 scaffold acceptance bar — we
            // stop here. The decode-side `Unsupported` error will fire long
            // before any of the deferred fields are consulted.
        }
    }
    // From here on we attempt the I-slice-only extension: SAO flags,
    // slice_qp_delta, and byte_alignment(). If the slice is an IDR I slice
    // under the shapes the v1 scaffold can handle, we fill in the derived
    // fields. Otherwise `is_full_i_slice` stays false and CTU decode will
    // still refuse with `Unsupported`.
    let mut slice_sao_luma_flag = false;
    let mut slice_sao_chroma_flag = false;
    let cabac_init_flag = false;
    let mut slice_qp_delta: i32 = 0;
    let mut slice_loop_filter_across_slices_enabled_flag =
        pps.pps_loop_filter_across_slices_enabled_flag;
    let mut slice_data_bit_offset: u64 = 0;
    let mut is_full_i_slice = false;

    let is_idr = matches!(
        nal.nal_unit_type,
        NalUnitType::IdrWRadl | NalUnitType::IdrNLp
    );

    if !dependent_slice_segment_flag
        && slice_type == SliceType::I
        && is_idr
        && !sps.separate_colour_plane_flag
    {
        // Path for the narrow but common case: IDR I slice, 4:2:0 / 4:2:2 /
        // 4:4:4 (without separate colour planes), no slice-header extension,
        // no PPS-level chroma offsets. Anything we can't handle becomes an
        // early exit with is_full_i_slice=false.
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
        // I-slice skips the inter-prediction and weighted-pred sections.
        slice_qp_delta = br.se()?;
        if pps.pps_slice_chroma_qp_offsets_present_flag {
            let _slice_cb_qp_offset = br.se()?;
            let _slice_cr_qp_offset = br.se()?;
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
                // Entry-point offsets are present; parsing these accurately
                // requires the minus-1-plus-1 bit width dance and they are
                // only needed for multi-tile / wavefront decode. Treat as
                // unsupported for the I-slice extension.
                //
                // Return the headers we have so far without the full
                // extension set.
                return Ok(SliceSegmentHeader {
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
                    slice_data_bit_offset: 0,
                    is_full_i_slice: false,
                });
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
        is_full_i_slice = true;
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
}
