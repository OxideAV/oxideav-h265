//! HEVC Picture Parameter Set parser (§7.3.2.3).
//!
//! v1 scope: every flag the slice-header parser needs to know about plus
//! the tiles / loop-filter geometry. Anything past `pps_extension_flag`
//! is not used in the v1 scaffold.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::scaling_list::{parse_scaling_list_data, ScalingListData};

/// Parsed tile layout (§7.3.2.3). When `tiles_enabled_flag == 0`, the PPS
/// leaves this at `None`. `num_tile_columns_minus1` and
/// `num_tile_rows_minus1` define the tile grid; when
/// `uniform_spacing_flag` is set the rows/columns are sized uniformly at
/// slice_data() derivation time (§6.5.1 eq. 6-3/6-4). Otherwise the
/// explicit `column_width_minus1[]` / `row_height_minus1[]` arrays are
/// kept around so the CTU walker can turn them into CTB counts.
#[derive(Clone, Debug)]
pub struct TileInfo {
    pub num_tile_columns_minus1: u32,
    pub num_tile_rows_minus1: u32,
    pub uniform_spacing_flag: bool,
    /// Explicit column widths in CTBs (minus 1). Empty when
    /// `uniform_spacing_flag == 1`.
    pub column_width_minus1: Vec<u32>,
    /// Explicit row heights in CTBs (minus 1). Empty when
    /// `uniform_spacing_flag == 1`.
    pub row_height_minus1: Vec<u32>,
    /// `loop_filter_across_tiles_enabled_flag`. Defaults to the spec default
    /// of 1 per §7.4.3.3 (present only when tiles are enabled).
    pub loop_filter_across_tiles_enabled_flag: bool,
}

impl TileInfo {
    /// Number of tile columns (`num_tile_columns_minus1 + 1`).
    pub fn num_tile_columns(&self) -> u32 {
        self.num_tile_columns_minus1 + 1
    }

    /// Number of tile rows (`num_tile_rows_minus1 + 1`).
    pub fn num_tile_rows(&self) -> u32 {
        self.num_tile_rows_minus1 + 1
    }

    /// Total number of tiles across the picture.
    pub fn num_tiles(&self) -> u32 {
        self.num_tile_columns() * self.num_tile_rows()
    }

    /// Derive per-tile-column widths in CTBs given the picture's CTB width
    /// (§6.5.1 eq. 6-4). Handles both the uniform-spacing path and the
    /// explicit `column_width_minus1[]` path. The last column soaks up any
    /// remainder in both cases.
    pub fn column_widths_ctb(&self, pic_w_ctb: u32) -> Vec<u32> {
        let n = self.num_tile_columns();
        let mut out = vec![0u32; n as usize];
        if self.uniform_spacing_flag {
            // §6.5.1 eq. 6-4: ColumnWidthInCtbsY[i] =
            //   ((i+1)*PicWidthInCtbsY)/(num_tile_columns_minus1+1)
            //   - (i*PicWidthInCtbsY)/(num_tile_columns_minus1+1).
            for i in 0..n {
                let nxt = ((i + 1) * pic_w_ctb) / n;
                let cur = (i * pic_w_ctb) / n;
                out[i as usize] = nxt - cur;
            }
        } else {
            let mut sum: u32 = 0;
            for i in 0..(n - 1) {
                let w = self.column_width_minus1[i as usize] + 1;
                out[i as usize] = w;
                sum += w;
            }
            out[n as usize - 1] = pic_w_ctb.saturating_sub(sum);
        }
        out
    }

    /// Derive per-tile-row heights in CTBs given the picture's CTB height
    /// (§6.5.1 eq. 6-3). Mirrors `column_widths_ctb`.
    pub fn row_heights_ctb(&self, pic_h_ctb: u32) -> Vec<u32> {
        let n = self.num_tile_rows();
        let mut out = vec![0u32; n as usize];
        if self.uniform_spacing_flag {
            for i in 0..n {
                let nxt = ((i + 1) * pic_h_ctb) / n;
                let cur = (i * pic_h_ctb) / n;
                out[i as usize] = nxt - cur;
            }
        } else {
            let mut sum: u32 = 0;
            for i in 0..(n - 1) {
                let h = self.row_height_minus1[i as usize] + 1;
                out[i as usize] = h;
                sum += h;
            }
            out[n as usize - 1] = pic_h_ctb.saturating_sub(sum);
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct PicParameterSet {
    pub pps_pic_parameter_set_id: u32,
    pub pps_seq_parameter_set_id: u32,
    pub dependent_slice_segments_enabled_flag: bool,
    pub output_flag_present_flag: bool,
    pub num_extra_slice_header_bits: u8,
    pub sign_data_hiding_enabled_flag: bool,
    pub cabac_init_present_flag: bool,
    pub num_ref_idx_l0_default_active_minus1: u32,
    pub num_ref_idx_l1_default_active_minus1: u32,
    pub init_qp_minus26: i32,
    pub constrained_intra_pred_flag: bool,
    pub transform_skip_enabled_flag: bool,
    pub cu_qp_delta_enabled_flag: bool,
    pub diff_cu_qp_delta_depth: u32,
    pub pps_cb_qp_offset: i32,
    pub pps_cr_qp_offset: i32,
    pub pps_slice_chroma_qp_offsets_present_flag: bool,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_flag: bool,
    pub transquant_bypass_enabled_flag: bool,
    pub tiles_enabled_flag: bool,
    pub entropy_coding_sync_enabled_flag: bool,
    /// Tile geometry (§7.3.2.3) — `Some` iff `tiles_enabled_flag == 1`.
    pub tile_info: Option<TileInfo>,
    pub pps_loop_filter_across_slices_enabled_flag: bool,
    pub deblocking_filter_control_present_flag: bool,
    pub deblocking_filter_override_enabled_flag: bool,
    pub pps_deblocking_filter_disabled_flag: bool,
    /// `pps_beta_offset_div2` (§7.4.3.3). Added to slice override β offset
    /// before looking up β in §8.7.2.2 Table 8-11. Range [-6, 6].
    pub pps_beta_offset_div2: i32,
    /// `pps_tc_offset_div2` (§7.4.3.3). Added to slice override tC offset
    /// before looking up tC in §8.7.2.2 Table 8-11. Range [-6, 6].
    pub pps_tc_offset_div2: i32,
    pub lists_modification_present_flag: bool,
    pub log2_parallel_merge_level_minus2: u32,
    pub slice_segment_header_extension_present_flag: bool,
    /// `pps_scaling_list_data_present_flag`.
    pub pps_scaling_list_data_present_flag: bool,
    /// Parsed `scaling_list_data()` from the PPS when present (§7.3.4).
    /// Overrides any SPS-level list per §7.4.3.3.
    pub scaling_list_data: Option<ScalingListData>,
}

/// Parse a PPS NAL RBSP payload (the bytes after the 2-byte NAL header,
/// already stripped of emulation-prevention bytes).
pub fn parse_pps(rbsp: &[u8]) -> Result<PicParameterSet> {
    let mut br = BitReader::new(rbsp);
    let pps_pic_parameter_set_id = br.ue()?;
    let pps_seq_parameter_set_id = br.ue()?;
    let dependent_slice_segments_enabled_flag = br.u1()? == 1;
    let output_flag_present_flag = br.u1()? == 1;
    let num_extra_slice_header_bits = br.u(3)? as u8;
    let sign_data_hiding_enabled_flag = br.u1()? == 1;
    let cabac_init_present_flag = br.u1()? == 1;
    let num_ref_idx_l0_default_active_minus1 = br.ue()?;
    let num_ref_idx_l1_default_active_minus1 = br.ue()?;
    let init_qp_minus26 = br.se()?;
    let constrained_intra_pred_flag = br.u1()? == 1;
    let transform_skip_enabled_flag = br.u1()? == 1;
    let cu_qp_delta_enabled_flag = br.u1()? == 1;
    let diff_cu_qp_delta_depth = if cu_qp_delta_enabled_flag {
        br.ue()?
    } else {
        0
    };
    let pps_cb_qp_offset = br.se()?;
    let pps_cr_qp_offset = br.se()?;
    let pps_slice_chroma_qp_offsets_present_flag = br.u1()? == 1;
    let weighted_pred_flag = br.u1()? == 1;
    let weighted_bipred_flag = br.u1()? == 1;
    let transquant_bypass_enabled_flag = br.u1()? == 1;
    let tiles_enabled_flag = br.u1()? == 1;
    let entropy_coding_sync_enabled_flag = br.u1()? == 1;

    let mut tile_info: Option<TileInfo> = None;
    if tiles_enabled_flag {
        let num_tile_columns_minus1 = br.ue()?;
        let num_tile_rows_minus1 = br.ue()?;
        if num_tile_columns_minus1 > 64 || num_tile_rows_minus1 > 64 {
            return Err(Error::invalid(format!(
                "h265 PPS: tile dims out of range ({num_tile_columns_minus1}+1 x {num_tile_rows_minus1}+1)"
            )));
        }
        let uniform_spacing_flag = br.u1()? == 1;
        let mut column_width_minus1: Vec<u32> = Vec::new();
        let mut row_height_minus1: Vec<u32> = Vec::new();
        if !uniform_spacing_flag {
            column_width_minus1.reserve(num_tile_columns_minus1 as usize);
            for _ in 0..num_tile_columns_minus1 {
                column_width_minus1.push(br.ue()?);
            }
            row_height_minus1.reserve(num_tile_rows_minus1 as usize);
            for _ in 0..num_tile_rows_minus1 {
                row_height_minus1.push(br.ue()?);
            }
        }
        let loop_filter_across_tiles_enabled_flag = br.u1()? == 1;
        tile_info = Some(TileInfo {
            num_tile_columns_minus1,
            num_tile_rows_minus1,
            uniform_spacing_flag,
            column_width_minus1,
            row_height_minus1,
            loop_filter_across_tiles_enabled_flag,
        });
    }
    let pps_loop_filter_across_slices_enabled_flag = br.u1()? == 1;

    let deblocking_filter_control_present_flag = br.u1()? == 1;
    let mut deblocking_filter_override_enabled_flag = false;
    let mut pps_deblocking_filter_disabled_flag = false;
    let mut pps_beta_offset_div2: i32 = 0;
    let mut pps_tc_offset_div2: i32 = 0;
    if deblocking_filter_control_present_flag {
        deblocking_filter_override_enabled_flag = br.u1()? == 1;
        pps_deblocking_filter_disabled_flag = br.u1()? == 1;
        if !pps_deblocking_filter_disabled_flag {
            pps_beta_offset_div2 = br.se()?;
            pps_tc_offset_div2 = br.se()?;
        }
    }
    let pps_scaling_list_data_present_flag = br.u1()? == 1;
    let mut scaling_list_data: Option<ScalingListData> = None;
    if pps_scaling_list_data_present_flag {
        scaling_list_data = Some(parse_scaling_list_data(&mut br)?);
    }
    let lists_modification_present_flag = br.u1()? == 1;
    let log2_parallel_merge_level_minus2 = br.ue()?;
    let slice_segment_header_extension_present_flag = br.u1()? == 1;
    // We deliberately stop before pps_extension_flag — the slice header
    // does not depend on it for the v1 scaffold.

    Ok(PicParameterSet {
        pps_pic_parameter_set_id,
        pps_seq_parameter_set_id,
        dependent_slice_segments_enabled_flag,
        output_flag_present_flag,
        num_extra_slice_header_bits,
        sign_data_hiding_enabled_flag,
        cabac_init_present_flag,
        num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_default_active_minus1,
        init_qp_minus26,
        constrained_intra_pred_flag,
        transform_skip_enabled_flag,
        cu_qp_delta_enabled_flag,
        diff_cu_qp_delta_depth,
        pps_cb_qp_offset,
        pps_cr_qp_offset,
        pps_slice_chroma_qp_offsets_present_flag,
        weighted_pred_flag,
        weighted_bipred_flag,
        transquant_bypass_enabled_flag,
        tiles_enabled_flag,
        entropy_coding_sync_enabled_flag,
        tile_info,
        pps_loop_filter_across_slices_enabled_flag,
        deblocking_filter_control_present_flag,
        deblocking_filter_override_enabled_flag,
        pps_deblocking_filter_disabled_flag,
        pps_beta_offset_div2,
        pps_tc_offset_div2,
        lists_modification_present_flag,
        log2_parallel_merge_level_minus2,
        slice_segment_header_extension_present_flag,
        pps_scaling_list_data_present_flag,
        scaling_list_data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_info_uniform_2x2_widths() {
        // Picture of 20 CTBs wide / 12 CTBs tall, split 2x2 uniformly.
        // Spec §6.5.1 eq. 6-4 gives 10 + 10 columns and 6 + 6 rows.
        let ti = TileInfo {
            num_tile_columns_minus1: 1,
            num_tile_rows_minus1: 1,
            uniform_spacing_flag: true,
            column_width_minus1: Vec::new(),
            row_height_minus1: Vec::new(),
            loop_filter_across_tiles_enabled_flag: true,
        };
        assert_eq!(ti.column_widths_ctb(20), vec![10, 10]);
        assert_eq!(ti.row_heights_ctb(12), vec![6, 6]);

        // Unequal pic width (21): eq. 6-4 ⇒ 10 + 11 (last soaks remainder).
        assert_eq!(ti.column_widths_ctb(21), vec![10, 11]);
    }

    #[test]
    fn tile_info_explicit_column_widths() {
        // 3 columns, 2 rows. Explicit: widths 3, 5 CTBs for first two
        // columns (third takes remainder); heights 4, 2 for first row
        // (second takes remainder).
        let ti = TileInfo {
            num_tile_columns_minus1: 2,
            num_tile_rows_minus1: 1,
            uniform_spacing_flag: false,
            column_width_minus1: vec![2, 4],
            row_height_minus1: vec![3],
            loop_filter_across_tiles_enabled_flag: false,
        };
        assert_eq!(ti.column_widths_ctb(12), vec![3, 5, 4]);
        assert_eq!(ti.row_heights_ctb(10), vec![4, 6]);
    }

    #[test]
    fn tile_info_counts() {
        let ti = TileInfo {
            num_tile_columns_minus1: 1,
            num_tile_rows_minus1: 2,
            uniform_spacing_flag: true,
            column_width_minus1: Vec::new(),
            row_height_minus1: Vec::new(),
            loop_filter_across_tiles_enabled_flag: true,
        };
        assert_eq!(ti.num_tile_columns(), 2);
        assert_eq!(ti.num_tile_rows(), 3);
        assert_eq!(ti.num_tiles(), 6);
    }
}
