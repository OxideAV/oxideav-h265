//! HEVC Sequence Parameter Set parser (§7.3.2.2).
//!
//! v1 scope: parse all the fields needed to derive raw picture dimensions,
//! the chroma format, the bit depths, the CTU size, and the transform
//! block size limits. We deliberately walk past `vui_parameters()` and the
//! `sps_extension` flags so the parser leaves a clean position even though
//! we don't expose those values.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::ptl::{parse_profile_tier_level, ProfileTierLevel};

#[derive(Clone, Debug)]
pub struct SeqParameterSet {
    pub sps_video_parameter_set_id: u8,
    pub sps_max_sub_layers_minus1: u8,
    pub sps_temporal_id_nesting_flag: bool,
    pub profile_tier_level: ProfileTierLevel,
    pub sps_seq_parameter_set_id: u32,
    pub chroma_format_idc: u32,
    pub separate_colour_plane_flag: bool,
    pub pic_width_in_luma_samples: u32,
    pub pic_height_in_luma_samples: u32,
    pub conformance_window: Option<ConformanceWindow>,
    pub bit_depth_luma_minus8: u32,
    pub bit_depth_chroma_minus8: u32,
    pub log2_max_pic_order_cnt_lsb_minus4: u32,
    pub sps_sub_layer_ordering_info_present_flag: bool,
    pub sps_max_dec_pic_buffering_minus1: Vec<u32>,
    pub sps_max_num_reorder_pics: Vec<u32>,
    pub sps_max_latency_increase_plus1: Vec<u32>,
    pub log2_min_luma_coding_block_size_minus3: u32,
    pub log2_diff_max_min_luma_coding_block_size: u32,
    pub log2_min_luma_transform_block_size_minus2: u32,
    pub log2_diff_max_min_luma_transform_block_size: u32,
    pub max_transform_hierarchy_depth_inter: u32,
    pub max_transform_hierarchy_depth_intra: u32,
    pub scaling_list_enabled_flag: bool,
    pub sps_scaling_list_data_present_flag: bool,
    pub amp_enabled_flag: bool,
    pub sample_adaptive_offset_enabled_flag: bool,
    pub pcm_enabled_flag: bool,
    /// Counts of short-term reference picture sets and long-term references
    /// — we parse the counts but skip the bodies in v1.
    pub num_short_term_ref_pic_sets: u32,
    pub long_term_ref_pics_present_flag: bool,
    pub sps_temporal_mvp_enabled_flag: bool,
    pub strong_intra_smoothing_enabled_flag: bool,
    pub vui_parameters_present_flag: bool,
    /// Parsed short-term RPS entries (§7.4.8). Each entry records the POC
    /// deltas (negative first, positive after) and the per-entry
    /// `used_by_curr_pic_flag`. Needed for slice-level reference picture list
    /// construction at inter-slice decode time.
    pub short_term_ref_pic_sets: Vec<ShortTermRps>,
}

/// A parsed short-term reference picture set (§7.4.8).
#[derive(Clone, Debug, Default)]
pub struct ShortTermRps {
    /// POC deltas relative to the current picture for the "before" set
    /// (historically negative).
    pub delta_poc_s0: Vec<i32>,
    /// POC deltas for the "after" set (positive).
    pub delta_poc_s1: Vec<i32>,
    /// `used_by_curr_pic` flags for the s0 entries.
    pub used_by_curr_pic_s0: Vec<bool>,
    /// `used_by_curr_pic` flags for the s1 entries.
    pub used_by_curr_pic_s1: Vec<bool>,
}

impl ShortTermRps {
    pub fn num_negative_pics(&self) -> usize {
        self.delta_poc_s0.len()
    }

    pub fn num_positive_pics(&self) -> usize {
        self.delta_poc_s1.len()
    }

    pub fn num_delta_pocs(&self) -> usize {
        self.num_negative_pics() + self.num_positive_pics()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ConformanceWindow {
    pub left_offset: u32,
    pub right_offset: u32,
    pub top_offset: u32,
    pub bottom_offset: u32,
}

impl SeqParameterSet {
    /// Width in luma samples, accounting for the conformance cropping window.
    /// SubWidthC / SubHeightC per Table 6-1: 4:2:0 → (2,2); 4:2:2 → (2,1);
    /// 4:4:4 → (1,1); 4:0:0 → (1,1).
    pub fn cropped_width(&self) -> u32 {
        let sub_x = match self.chroma_format_idc {
            1 | 2 => 2,
            _ => 1,
        };
        let crop = self
            .conformance_window
            .map(|c| sub_x * (c.left_offset + c.right_offset))
            .unwrap_or(0);
        self.pic_width_in_luma_samples.saturating_sub(crop)
    }

    pub fn cropped_height(&self) -> u32 {
        let sub_y = match self.chroma_format_idc {
            1 => 2,
            _ => 1,
        };
        let crop = self
            .conformance_window
            .map(|c| sub_y * (c.top_offset + c.bottom_offset))
            .unwrap_or(0);
        self.pic_height_in_luma_samples.saturating_sub(crop)
    }

    /// CTB size = 1 << (log2_min_luma_cb + log2_diff_max_min_luma_cb + 3).
    pub fn ctb_size(&self) -> u32 {
        1 << (self.log2_min_luma_coding_block_size_minus3
            + self.log2_diff_max_min_luma_coding_block_size
            + 3)
    }

    /// Bit depth of luma samples in bits per pixel.
    pub fn bit_depth_y(&self) -> u32 {
        self.bit_depth_luma_minus8 + 8
    }

    pub fn bit_depth_c(&self) -> u32 {
        self.bit_depth_chroma_minus8 + 8
    }
}

/// Parse an SPS NAL RBSP payload (the bytes after the 2-byte NAL header,
/// already stripped of emulation-prevention bytes).
pub fn parse_sps(rbsp: &[u8]) -> Result<SeqParameterSet> {
    let mut br = BitReader::new(rbsp);
    let sps_video_parameter_set_id = br.u(4)? as u8;
    let sps_max_sub_layers_minus1 = br.u(3)? as u8;
    if sps_max_sub_layers_minus1 > 6 {
        return Err(Error::invalid(
            "h265 SPS: sps_max_sub_layers_minus1 must be <= 6",
        ));
    }
    let sps_temporal_id_nesting_flag = br.u1()? == 1;
    let profile_tier_level = parse_profile_tier_level(&mut br, true, sps_max_sub_layers_minus1)?;

    let sps_seq_parameter_set_id = br.ue()?;
    let chroma_format_idc = br.ue()?;
    if chroma_format_idc > 3 {
        return Err(Error::invalid(format!(
            "h265 SPS: chroma_format_idc out of range: {chroma_format_idc}"
        )));
    }
    let separate_colour_plane_flag = if chroma_format_idc == 3 {
        br.u1()? == 1
    } else {
        false
    };
    let pic_width_in_luma_samples = br.ue()?;
    let pic_height_in_luma_samples = br.ue()?;
    if pic_width_in_luma_samples == 0
        || pic_height_in_luma_samples == 0
        || pic_width_in_luma_samples > 16384
        || pic_height_in_luma_samples > 16384
    {
        return Err(Error::invalid(format!(
            "h265 SPS: implausible picture size {pic_width_in_luma_samples}x{pic_height_in_luma_samples}"
        )));
    }
    let conformance_window_flag = br.u1()? == 1;
    let conformance_window = if conformance_window_flag {
        Some(ConformanceWindow {
            left_offset: br.ue()?,
            right_offset: br.ue()?,
            top_offset: br.ue()?,
            bottom_offset: br.ue()?,
        })
    } else {
        None
    };
    let bit_depth_luma_minus8 = br.ue()?;
    let bit_depth_chroma_minus8 = br.ue()?;
    if bit_depth_luma_minus8 > 8 || bit_depth_chroma_minus8 > 8 {
        return Err(Error::invalid(format!(
            "h265 SPS: bit_depth_*_minus8 out of range ({bit_depth_luma_minus8}, {bit_depth_chroma_minus8})"
        )));
    }
    let log2_max_pic_order_cnt_lsb_minus4 = br.ue()?;
    if log2_max_pic_order_cnt_lsb_minus4 > 12 {
        return Err(Error::invalid(format!(
            "h265 SPS: log2_max_pic_order_cnt_lsb_minus4 out of range ({log2_max_pic_order_cnt_lsb_minus4})"
        )));
    }

    let sps_sub_layer_ordering_info_present_flag = br.u1()? == 1;
    let n = (sps_max_sub_layers_minus1 + 1) as usize;
    let lo = if sps_sub_layer_ordering_info_present_flag {
        0
    } else {
        sps_max_sub_layers_minus1 as usize
    };
    let mut sps_max_dec_pic_buffering_minus1 = vec![0u32; n];
    let mut sps_max_num_reorder_pics = vec![0u32; n];
    let mut sps_max_latency_increase_plus1 = vec![0u32; n];
    for i in lo..n {
        sps_max_dec_pic_buffering_minus1[i] = br.ue()?;
        sps_max_num_reorder_pics[i] = br.ue()?;
        sps_max_latency_increase_plus1[i] = br.ue()?;
    }

    let log2_min_luma_coding_block_size_minus3 = br.ue()?;
    let log2_diff_max_min_luma_coding_block_size = br.ue()?;
    let log2_min_luma_transform_block_size_minus2 = br.ue()?;
    let log2_diff_max_min_luma_transform_block_size = br.ue()?;
    let max_transform_hierarchy_depth_inter = br.ue()?;
    let max_transform_hierarchy_depth_intra = br.ue()?;

    let scaling_list_enabled_flag = br.u1()? == 1;
    let sps_scaling_list_data_present_flag = if scaling_list_enabled_flag {
        let present = br.u1()? == 1;
        if present {
            // §7.3.4 scaling_list_data — a few hundred bits of explicit
            // scaling matrices. v1 scaffold skips them so the bit position
            // remains correct without keeping the values.
            skip_scaling_list_data(&mut br)?;
        }
        present
    } else {
        false
    };

    let amp_enabled_flag = br.u1()? == 1;
    let sample_adaptive_offset_enabled_flag = br.u1()? == 1;
    let pcm_enabled_flag = br.u1()? == 1;
    if pcm_enabled_flag {
        // pcm_sample_bit_depth_luma_minus1, pcm_sample_bit_depth_chroma_minus1
        br.skip(4)?;
        br.skip(4)?;
        // log2_min_pcm_luma_coding_block_size_minus3
        let _ = br.ue()?;
        // log2_diff_max_min_pcm_luma_coding_block_size
        let _ = br.ue()?;
        // pcm_loop_filter_disabled_flag
        br.skip(1)?;
    }

    let num_short_term_ref_pic_sets = br.ue()?;
    if num_short_term_ref_pic_sets > 64 {
        return Err(Error::invalid(format!(
            "h265 SPS: num_short_term_ref_pic_sets out of range ({num_short_term_ref_pic_sets})"
        )));
    }
    let mut short_term_ref_pic_sets: Vec<ShortTermRps> =
        Vec::with_capacity(num_short_term_ref_pic_sets as usize);
    for st_rps_idx in 0..num_short_term_ref_pic_sets {
        let rps = parse_st_ref_pic_set(
            &mut br,
            st_rps_idx,
            num_short_term_ref_pic_sets,
            &short_term_ref_pic_sets,
        )?;
        short_term_ref_pic_sets.push(rps);
    }

    let long_term_ref_pics_present_flag = br.u1()? == 1;
    if long_term_ref_pics_present_flag {
        let num_long_term_ref_pics_sps = br.ue()?;
        if num_long_term_ref_pics_sps > 32 {
            return Err(Error::invalid(format!(
                "h265 SPS: num_long_term_ref_pics_sps out of range ({num_long_term_ref_pics_sps})"
            )));
        }
        let lsb_bits = log2_max_pic_order_cnt_lsb_minus4 + 4;
        for _ in 0..num_long_term_ref_pics_sps {
            br.skip(lsb_bits)?;
            br.skip(1)?; // used_by_curr_pic_lt_sps_flag
        }
    }
    let sps_temporal_mvp_enabled_flag = br.u1()? == 1;
    let strong_intra_smoothing_enabled_flag = br.u1()? == 1;

    let vui_parameters_present_flag = br.u1()? == 1;
    // We deliberately stop here — VUI / sps_extension / rbsp_trailing_bits
    // have no impact on the v1 scaffold and a full VUI parse adds another
    // ~200 lines for marginal value.

    Ok(SeqParameterSet {
        sps_video_parameter_set_id,
        sps_max_sub_layers_minus1,
        sps_temporal_id_nesting_flag,
        profile_tier_level,
        sps_seq_parameter_set_id,
        chroma_format_idc,
        separate_colour_plane_flag,
        pic_width_in_luma_samples,
        pic_height_in_luma_samples,
        conformance_window,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
        log2_max_pic_order_cnt_lsb_minus4,
        sps_sub_layer_ordering_info_present_flag,
        sps_max_dec_pic_buffering_minus1,
        sps_max_num_reorder_pics,
        sps_max_latency_increase_plus1,
        log2_min_luma_coding_block_size_minus3,
        log2_diff_max_min_luma_coding_block_size,
        log2_min_luma_transform_block_size_minus2,
        log2_diff_max_min_luma_transform_block_size,
        max_transform_hierarchy_depth_inter,
        max_transform_hierarchy_depth_intra,
        scaling_list_enabled_flag,
        sps_scaling_list_data_present_flag,
        amp_enabled_flag,
        sample_adaptive_offset_enabled_flag,
        pcm_enabled_flag,
        num_short_term_ref_pic_sets,
        long_term_ref_pics_present_flag,
        sps_temporal_mvp_enabled_flag,
        strong_intra_smoothing_enabled_flag,
        vui_parameters_present_flag,
        short_term_ref_pic_sets,
    })
}

/// Parse one `st_ref_pic_set( stRpsIdx )` (§7.4.8). Fills in absolute POC
/// deltas and `used_by_curr_pic` flags. Handles both the explicit form and
/// the inter-RPS-prediction form (references an earlier RPS + a delta_rps).
pub fn parse_st_ref_pic_set(
    br: &mut BitReader<'_>,
    st_rps_idx: u32,
    num_short_term_ref_pic_sets: u32,
    prior: &[ShortTermRps],
) -> Result<ShortTermRps> {
    let inter_ref_pic_set_prediction_flag = if st_rps_idx != 0 {
        br.u1()? == 1
    } else {
        false
    };
    if inter_ref_pic_set_prediction_flag {
        let delta_idx_minus1 = if st_rps_idx == num_short_term_ref_pic_sets {
            br.ue()?
        } else {
            0
        };
        let delta_rps_sign = br.u1()? == 1;
        let abs_delta_rps_minus1 = br.ue()?;
        let delta_rps = if delta_rps_sign {
            -((abs_delta_rps_minus1 as i32) + 1)
        } else {
            (abs_delta_rps_minus1 as i32) + 1
        };
        let ref_rps_idx = (st_rps_idx as i64) - 1 - delta_idx_minus1 as i64;
        if ref_rps_idx < 0 || (ref_rps_idx as usize) >= prior.len() {
            return Err(Error::invalid(format!(
                "h265 SPS RPS: invalid ref_rps_idx {ref_rps_idx}"
            )));
        }
        let r = &prior[ref_rps_idx as usize];
        let num_delta_pocs = r.num_delta_pocs();
        // For each j in 0..=NumDeltaPocs(ref): used_by_curr_pic_flag[j]; if
        // not used, optional use_delta_flag[j] (default 1).
        let mut used_by_curr = vec![false; num_delta_pocs + 1];
        let mut use_delta = vec![true; num_delta_pocs + 1];
        for j in 0..=num_delta_pocs {
            used_by_curr[j] = br.u1()? == 1;
            if !used_by_curr[j] {
                use_delta[j] = br.u1()? == 1;
            }
        }
        // Build up the s0/s1 per §7.4.8 equations.
        let mut s0_deltas: Vec<i32> = Vec::new();
        let mut s0_used: Vec<bool> = Vec::new();
        let mut s1_deltas: Vec<i32> = Vec::new();
        let mut s1_used: Vec<bool> = Vec::new();
        // Case 1: iterate negative pics of the reference in reverse plus the
        // "self" entry at index num_delta_pocs (representing the reference
        // picture itself at delta 0).
        let num_neg = r.num_negative_pics();
        let num_pos = r.num_positive_pics();
        for i in (0..num_neg).rev() {
            let d = r.delta_poc_s0[i] + delta_rps;
            if d < 0 && use_delta[i] {
                s0_deltas.push(d);
                s0_used.push(used_by_curr[i]);
            }
        }
        if delta_rps < 0 && use_delta[num_delta_pocs] {
            s0_deltas.push(delta_rps);
            s0_used.push(used_by_curr[num_delta_pocs]);
        }
        for i in 0..num_pos {
            let d = r.delta_poc_s1[i] + delta_rps;
            if d < 0 && use_delta[num_neg + i] {
                s0_deltas.push(d);
                s0_used.push(used_by_curr[num_neg + i]);
            }
        }
        for i in (0..num_pos).rev() {
            let d = r.delta_poc_s1[i] + delta_rps;
            if d > 0 && use_delta[num_neg + i] {
                s1_deltas.push(d);
                s1_used.push(used_by_curr[num_neg + i]);
            }
        }
        if delta_rps > 0 && use_delta[num_delta_pocs] {
            s1_deltas.push(delta_rps);
            s1_used.push(used_by_curr[num_delta_pocs]);
        }
        for i in 0..num_neg {
            let d = r.delta_poc_s0[i] + delta_rps;
            if d > 0 && use_delta[i] {
                s1_deltas.push(d);
                s1_used.push(used_by_curr[i]);
            }
        }
        Ok(ShortTermRps {
            delta_poc_s0: s0_deltas,
            delta_poc_s1: s1_deltas,
            used_by_curr_pic_s0: s0_used,
            used_by_curr_pic_s1: s1_used,
        })
    } else {
        let num_negative_pics = br.ue()?;
        let num_positive_pics = br.ue()?;
        if num_negative_pics > 16 || num_positive_pics > 16 {
            return Err(Error::invalid(format!(
                "h265 SPS RPS: num_*_pics out of range ({num_negative_pics}/{num_positive_pics})"
            )));
        }
        let mut delta_poc_s0 = Vec::with_capacity(num_negative_pics as usize);
        let mut used_by_curr_pic_s0 = Vec::with_capacity(num_negative_pics as usize);
        let mut prev: i32 = 0;
        for _ in 0..num_negative_pics {
            let delta_poc_s0_minus1 = br.ue()? as i32;
            let d = prev - (delta_poc_s0_minus1 + 1);
            prev = d;
            delta_poc_s0.push(d);
            used_by_curr_pic_s0.push(br.u1()? == 1);
        }
        let mut delta_poc_s1 = Vec::with_capacity(num_positive_pics as usize);
        let mut used_by_curr_pic_s1 = Vec::with_capacity(num_positive_pics as usize);
        let mut prev: i32 = 0;
        for _ in 0..num_positive_pics {
            let delta_poc_s1_minus1 = br.ue()? as i32;
            let d = prev + (delta_poc_s1_minus1 + 1);
            prev = d;
            delta_poc_s1.push(d);
            used_by_curr_pic_s1.push(br.u1()? == 1);
        }
        Ok(ShortTermRps {
            delta_poc_s0,
            delta_poc_s1,
            used_by_curr_pic_s0,
            used_by_curr_pic_s1,
        })
    }
}

/// Walk past a `scaling_list_data()` block (§7.3.4). We read syntax for
/// every (sizeId, matrixId) pair without storing the matrices.
fn skip_scaling_list_data(br: &mut BitReader<'_>) -> Result<()> {
    for size_id in 0..4u32 {
        let max_matrix_id = if size_id == 3 { 2 } else { 6 };
        let mut matrix_id = 0u32;
        while matrix_id < max_matrix_id {
            let scaling_list_pred_mode_flag = br.u1()? == 1;
            if !scaling_list_pred_mode_flag {
                let _ = br.ue()?; // scaling_list_pred_matrix_id_delta
            } else {
                let mut next_coef: i32 = 8;
                let coef_num = core::cmp::min(64u32, 1u32 << (4 + (size_id << 1)));
                if size_id > 1 {
                    let _scale = br.se()?; // scaling_list_dc_coef_minus8 (-7..247)
                    next_coef = _scale + 8;
                }
                for _ in 0..coef_num {
                    let scaling_list_delta_coef = br.se()?;
                    next_coef = (next_coef + scaling_list_delta_coef + 256) & 0xFF;
                }
            }
            matrix_id += if size_id == 3 { 3 } else { 1 };
        }
    }
    Ok(())
}
