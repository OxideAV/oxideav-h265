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
use crate::scaling_list::{parse_scaling_list_data, ScalingListData};

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
    /// Parsed `scaling_list_data()` (§7.3.4) — `Some` iff
    /// `sps_scaling_list_data_present_flag` is set. When
    /// `scaling_list_enabled_flag` is set but the SPS does not carry its
    /// own data, the decoder falls back to Table 7-5 / 7-6 defaults.
    pub scaling_list_data: Option<ScalingListData>,
    pub amp_enabled_flag: bool,
    pub sample_adaptive_offset_enabled_flag: bool,
    pub pcm_enabled_flag: bool,
    /// `pcm_sample_bit_depth_luma_minus1 + 1` (§7.4.3.2.1 eq. 7-25). Only
    /// valid when `pcm_enabled_flag == true`.
    pub pcm_sample_bit_depth_luma: u32,
    /// `pcm_sample_bit_depth_chroma_minus1 + 1` (eq. 7-26).
    pub pcm_sample_bit_depth_chroma: u32,
    /// `log2_min_pcm_luma_coding_block_size_minus3 + 3` (§7.4.3.2.1).
    /// Minimum PCM CU size in luma samples (log2).
    pub log2_min_pcm_luma_coding_block_size: u32,
    /// `log2_min_pcm_luma_coding_block_size + log2_diff_max_min_pcm_luma`.
    /// Maximum PCM CU size in luma samples (log2).
    pub log2_max_pcm_luma_coding_block_size: u32,
    /// `pcm_loop_filter_disabled_flag` — when 1, deblocking + SAO skip
    /// PCM CUs (§7.4.3.2.1). Not yet honoured by our deblock / SAO.
    pub pcm_loop_filter_disabled_flag: bool,
    /// Counts of short-term reference picture sets and long-term references.
    pub num_short_term_ref_pic_sets: u32,
    pub long_term_ref_pics_present_flag: bool,
    /// `num_long_term_ref_pics_sps`. 0 when `long_term_ref_pics_present_flag`
    /// is false.
    pub num_long_term_ref_pics_sps: u32,
    /// `lt_ref_pic_poc_lsb_sps[i]` (§7.4.3.2.1) — candidate LSBs that the
    /// slice header can reference by `lt_idx_sps[i]`.
    pub lt_ref_pic_poc_lsb_sps: Vec<u32>,
    /// `used_by_curr_pic_lt_sps_flag[i]` — whether the SPS-level candidate
    /// is marked as "used for reference by the current picture".
    pub used_by_curr_pic_lt_sps_flag: Vec<bool>,
    pub sps_temporal_mvp_enabled_flag: bool,
    pub strong_intra_smoothing_enabled_flag: bool,
    pub vui_parameters_present_flag: bool,
    /// Parsed short-term RPS entries (§7.4.8). Each entry records the POC
    /// deltas (negative first, positive after) and the per-entry
    /// `used_by_curr_pic_flag`. Needed for slice-level reference picture list
    /// construction at inter-slice decode time.
    pub short_term_ref_pic_sets: Vec<ShortTermRps>,
    /// SPS range extension flags (§7.4.3.2.2). All default `false` when the
    /// range extension is absent. Needed to gate Rext-specific CABAC paths.
    pub transform_skip_rotation_enabled_flag: bool,
    pub transform_skip_context_enabled_flag: bool,
    pub implicit_rdpcm_enabled_flag: bool,
    pub explicit_rdpcm_enabled_flag: bool,
    pub extended_precision_processing_flag: bool,
    pub intra_smoothing_disabled_flag: bool,
    pub high_precision_offsets_flag: bool,
    pub persistent_rice_adaptation_enabled_flag: bool,
    pub cabac_bypass_alignment_enabled_flag: bool,
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
    /// `ChromaArrayType` per §6.2: equal to `chroma_format_idc` unless
    /// `separate_colour_plane_flag == 1` (in which case it is 0). Used as
    /// the dispatch key for every chroma-aware code path.
    pub fn chroma_array_type(&self) -> u32 {
        if self.separate_colour_plane_flag {
            0
        } else {
            self.chroma_format_idc
        }
    }

    /// `SubWidthC` per §6.2 Table 6-1. 4:2:0 / 4:2:2 → 2; 4:4:4 / 4:0:0 → 1.
    pub fn sub_width_c(&self) -> u32 {
        match self.chroma_array_type() {
            1 | 2 => 2,
            _ => 1,
        }
    }

    /// `SubHeightC` per §6.2 Table 6-1. 4:2:0 → 2; 4:2:2 / 4:4:4 → 1; 4:0:0 → 1.
    pub fn sub_height_c(&self) -> u32 {
        match self.chroma_array_type() {
            1 => 2,
            _ => 1,
        }
    }

    /// Picture dimensions in chroma samples (`(width / SubWidthC,
    /// height / SubHeightC)`). Returns `(0, 0)` for 4:0:0.
    pub fn chroma_dims(&self) -> (u32, u32) {
        if self.chroma_array_type() == 0 {
            return (0, 0);
        }
        (
            self.pic_width_in_luma_samples / self.sub_width_c(),
            self.pic_height_in_luma_samples / self.sub_height_c(),
        )
    }

    /// Width in luma samples, accounting for the conformance cropping window.
    /// SubWidthC / SubHeightC per Table 6-1.
    pub fn cropped_width(&self) -> u32 {
        let crop = self
            .conformance_window
            .map(|c| self.sub_width_c() * (c.left_offset + c.right_offset))
            .unwrap_or(0);
        self.pic_width_in_luma_samples.saturating_sub(crop)
    }

    pub fn cropped_height(&self) -> u32 {
        let crop = self
            .conformance_window
            .map(|c| self.sub_height_c() * (c.top_offset + c.bottom_offset))
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
    let mut scaling_list_data: Option<ScalingListData> = None;
    let sps_scaling_list_data_present_flag = if scaling_list_enabled_flag {
        let present = br.u1()? == 1;
        if present {
            // §7.3.4 scaling_list_data — parse the matrices fully so the
            // dequantiser can use them per eq. 8-309 instead of flat 16s.
            scaling_list_data = Some(parse_scaling_list_data(&mut br)?);
        }
        present
    } else {
        false
    };

    let amp_enabled_flag = br.u1()? == 1;
    let sample_adaptive_offset_enabled_flag = br.u1()? == 1;
    let pcm_enabled_flag = br.u1()? == 1;
    let mut pcm_sample_bit_depth_luma: u32 = 0;
    let mut pcm_sample_bit_depth_chroma: u32 = 0;
    let mut log2_min_pcm_luma_coding_block_size: u32 = 0;
    let mut log2_max_pcm_luma_coding_block_size: u32 = 0;
    let mut pcm_loop_filter_disabled_flag = false;
    if pcm_enabled_flag {
        pcm_sample_bit_depth_luma = br.u(4)? + 1;
        pcm_sample_bit_depth_chroma = br.u(4)? + 1;
        let log2_min_pcm_minus3 = br.ue()?;
        let log2_diff_max_min_pcm = br.ue()?;
        log2_min_pcm_luma_coding_block_size = log2_min_pcm_minus3 + 3;
        log2_max_pcm_luma_coding_block_size =
            log2_min_pcm_luma_coding_block_size + log2_diff_max_min_pcm;
        pcm_loop_filter_disabled_flag = br.u1()? == 1;
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
    let mut num_long_term_ref_pics_sps: u32 = 0;
    let mut lt_ref_pic_poc_lsb_sps: Vec<u32> = Vec::new();
    let mut used_by_curr_pic_lt_sps_flag: Vec<bool> = Vec::new();
    if long_term_ref_pics_present_flag {
        num_long_term_ref_pics_sps = br.ue()?;
        if num_long_term_ref_pics_sps > 32 {
            return Err(Error::invalid(format!(
                "h265 SPS: num_long_term_ref_pics_sps out of range ({num_long_term_ref_pics_sps})"
            )));
        }
        let lsb_bits = log2_max_pic_order_cnt_lsb_minus4 + 4;
        lt_ref_pic_poc_lsb_sps.reserve(num_long_term_ref_pics_sps as usize);
        used_by_curr_pic_lt_sps_flag.reserve(num_long_term_ref_pics_sps as usize);
        for _ in 0..num_long_term_ref_pics_sps {
            lt_ref_pic_poc_lsb_sps.push(br.u(lsb_bits)?);
            used_by_curr_pic_lt_sps_flag.push(br.u1()? == 1);
        }
    }
    let sps_temporal_mvp_enabled_flag = br.u1()? == 1;
    let strong_intra_smoothing_enabled_flag = br.u1()? == 1;

    let vui_parameters_present_flag = br.u1()? == 1;
    if vui_parameters_present_flag {
        skip_vui(&mut br, sps_max_sub_layers_minus1)?;
    }

    // SPS range extension flags (§7.4.3.2.2) — only parsed if the stream
    // opted into them. Defaults are all `false` for base-profile streams.
    let mut transform_skip_rotation_enabled_flag = false;
    let mut transform_skip_context_enabled_flag = false;
    let mut implicit_rdpcm_enabled_flag = false;
    let mut explicit_rdpcm_enabled_flag = false;
    let mut extended_precision_processing_flag = false;
    let mut intra_smoothing_disabled_flag = false;
    let mut high_precision_offsets_flag = false;
    let mut persistent_rice_adaptation_enabled_flag = false;
    let mut cabac_bypass_alignment_enabled_flag = false;

    let sps_extension_present_flag = br.u1().unwrap_or(0) == 1;
    if sps_extension_present_flag {
        let sps_range_extension_flag = br.u1().unwrap_or(0) == 1;
        // Skip the four reserved/other extension flags for profiles we do
        // not yet target (multilayer / 3D / SCC / reserved).
        let _ = br.u(4);
        let _sps_extension_4bits = br.u(4).unwrap_or(0);
        if sps_range_extension_flag {
            transform_skip_rotation_enabled_flag = br.u1().unwrap_or(0) == 1;
            transform_skip_context_enabled_flag = br.u1().unwrap_or(0) == 1;
            implicit_rdpcm_enabled_flag = br.u1().unwrap_or(0) == 1;
            explicit_rdpcm_enabled_flag = br.u1().unwrap_or(0) == 1;
            extended_precision_processing_flag = br.u1().unwrap_or(0) == 1;
            intra_smoothing_disabled_flag = br.u1().unwrap_or(0) == 1;
            high_precision_offsets_flag = br.u1().unwrap_or(0) == 1;
            persistent_rice_adaptation_enabled_flag = br.u1().unwrap_or(0) == 1;
            cabac_bypass_alignment_enabled_flag = br.u1().unwrap_or(0) == 1;
        }
    }

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
        scaling_list_data,
        amp_enabled_flag,
        sample_adaptive_offset_enabled_flag,
        pcm_enabled_flag,
        pcm_sample_bit_depth_luma,
        pcm_sample_bit_depth_chroma,
        log2_min_pcm_luma_coding_block_size,
        log2_max_pcm_luma_coding_block_size,
        pcm_loop_filter_disabled_flag,
        num_short_term_ref_pic_sets,
        long_term_ref_pics_present_flag,
        num_long_term_ref_pics_sps,
        lt_ref_pic_poc_lsb_sps,
        used_by_curr_pic_lt_sps_flag,
        sps_temporal_mvp_enabled_flag,
        strong_intra_smoothing_enabled_flag,
        vui_parameters_present_flag,
        short_term_ref_pic_sets,
        transform_skip_rotation_enabled_flag,
        transform_skip_context_enabled_flag,
        implicit_rdpcm_enabled_flag,
        explicit_rdpcm_enabled_flag,
        extended_precision_processing_flag,
        intra_smoothing_disabled_flag,
        high_precision_offsets_flag,
        persistent_rice_adaptation_enabled_flag,
        cabac_bypass_alignment_enabled_flag,
    })
}

/// Best-effort `vui_parameters()` (§E.2.1) skipper. Consumes the bits so
/// that any `sps_extension_present_flag` that follows lands at the right
/// offset. Returns `Err` only for obviously malformed streams.
fn skip_vui(br: &mut BitReader<'_>, sps_max_sub_layers_minus1: u8) -> Result<()> {
    // aspect_ratio_info_present_flag
    if br.u1()? == 1 {
        let aspect_ratio_idc = br.u(8)?;
        if aspect_ratio_idc == 255 {
            br.skip(16)?; // sar_width
            br.skip(16)?; // sar_height
        }
    }
    // overscan_info_present_flag
    if br.u1()? == 1 {
        br.skip(1)?;
    }
    // video_signal_type_present_flag
    if br.u1()? == 1 {
        br.skip(3)?; // video_format
        br.skip(1)?; // video_full_range_flag
        if br.u1()? == 1 {
            br.skip(8)?;
            br.skip(8)?;
            br.skip(8)?;
        }
    }
    // chroma_loc_info_present_flag
    if br.u1()? == 1 {
        let _ = br.ue()?;
        let _ = br.ue()?;
    }
    br.skip(1)?; // neutral_chroma_indication_flag
    br.skip(1)?; // field_seq_flag
    br.skip(1)?; // frame_field_info_present_flag
    if br.u1()? == 1 {
        // default_display_window
        let _ = br.ue()?;
        let _ = br.ue()?;
        let _ = br.ue()?;
        let _ = br.ue()?;
    }
    // vui_timing_info_present_flag
    if br.u1()? == 1 {
        br.skip(32)?; // num_units_in_tick
        br.skip(32)?; // time_scale
        if br.u1()? == 1 {
            let _ = br.ue()?; // vui_num_ticks_poc_diff_one_minus1
        }
        // vui_hrd_parameters_present_flag
        if br.u1()? == 1 {
            skip_hrd_parameters(br, true, sps_max_sub_layers_minus1)?;
        }
    }
    // bitstream_restriction_flag
    if br.u1()? == 1 {
        br.skip(1)?; // tiles_fixed_structure_flag
        br.skip(1)?; // motion_vectors_over_pic_boundaries_flag
        br.skip(1)?; // restricted_ref_pic_lists_flag
        let _ = br.ue()?;
        let _ = br.ue()?;
        let _ = br.ue()?;
        let _ = br.ue()?;
        let _ = br.ue()?;
    }
    Ok(())
}

fn skip_hrd_parameters(
    br: &mut BitReader<'_>,
    common_inf_present: bool,
    max_num_sub_layers_minus1: u8,
) -> Result<()> {
    let mut nal_hrd_parameters_present_flag = false;
    let mut vcl_hrd_parameters_present_flag = false;
    let mut sub_pic_hrd_params_present_flag = false;
    if common_inf_present {
        nal_hrd_parameters_present_flag = br.u1()? == 1;
        vcl_hrd_parameters_present_flag = br.u1()? == 1;
        if nal_hrd_parameters_present_flag || vcl_hrd_parameters_present_flag {
            sub_pic_hrd_params_present_flag = br.u1()? == 1;
            if sub_pic_hrd_params_present_flag {
                br.skip(8)?; // tick_divisor_minus2
                br.skip(5)?; // du_cpb_removal_delay_increment_length_minus1
                br.skip(1)?; // sub_pic_cpb_params_in_pic_timing_sei_flag
                br.skip(5)?; // dpb_output_delay_du_length_minus1
            }
            br.skip(4)?; // bit_rate_scale
            br.skip(4)?; // cpb_size_scale
            if sub_pic_hrd_params_present_flag {
                br.skip(4)?; // cpb_size_du_scale
            }
            br.skip(5)?; // initial_cpb_removal_delay_length_minus1
            br.skip(5)?; // au_cpb_removal_delay_length_minus1
            br.skip(5)?; // dpb_output_delay_length_minus1
        }
    }
    for _ in 0..=max_num_sub_layers_minus1 {
        let fixed_pic_rate_general_flag = br.u1()? == 1;
        let mut fixed_pic_rate_within_cvs_flag = fixed_pic_rate_general_flag;
        if !fixed_pic_rate_general_flag {
            fixed_pic_rate_within_cvs_flag = br.u1()? == 1;
        }
        let mut low_delay_hrd_flag = false;
        let mut cpb_cnt_minus1: u32 = 0;
        if fixed_pic_rate_within_cvs_flag {
            let _ = br.ue()?; // elemental_duration_in_tc_minus1
        } else {
            low_delay_hrd_flag = br.u1()? == 1;
        }
        if !low_delay_hrd_flag {
            cpb_cnt_minus1 = br.ue()?;
        }
        if nal_hrd_parameters_present_flag {
            skip_sub_layer_hrd(br, cpb_cnt_minus1, sub_pic_hrd_params_present_flag)?;
        }
        if vcl_hrd_parameters_present_flag {
            skip_sub_layer_hrd(br, cpb_cnt_minus1, sub_pic_hrd_params_present_flag)?;
        }
    }
    Ok(())
}

fn skip_sub_layer_hrd(
    br: &mut BitReader<'_>,
    cpb_cnt_minus1: u32,
    sub_pic_hrd_params_present_flag: bool,
) -> Result<()> {
    for _ in 0..=cpb_cnt_minus1 {
        let _ = br.ue()?; // bit_rate_value_minus1
        let _ = br.ue()?; // cpb_size_value_minus1
        if sub_pic_hrd_params_present_flag {
            let _ = br.ue()?; // cpb_size_du_value_minus1
            let _ = br.ue()?; // bit_rate_du_value_minus1
        }
        br.skip(1)?; // cbr_flag
    }
    Ok(())
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
