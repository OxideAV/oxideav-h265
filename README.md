# oxideav-h265

A pure-Rust H.265 / HEVC video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Clean-room rebuild — round 10 (2026-05-24).** The prior implementation was
retired under the workspace
[clean-room policy](https://github.com/OxideAV/oxideav/blob/master/docs/IMPLEMENTOR_ROUND.md):
a CTU-level source comment cited a specific named variable and line
number in an external library's HEVC decoder — clean-room provenance
for the surrounding code path could not be defended. Master history
was fully erased per the Hat-3 cold-enforcement procedure.

The rebuild is in progress against the published H.265 specification
(ITU-T Recommendation H.265 | ISO/IEC 23008-2). Round 10 completes the
§6.5 scan-order family — the §6.5.4 horizontal (equation 6-12), §6.5.5
vertical (equation 6-13), and §6.5.6 traverse (equation 6-14,
boustrophedon) scans now join round 9's §6.5.3 up-right diagonal — and
adds the §7.4.2 `ScanOrder[log2BlockSize][scanIdx]` accessor
(`scan_order`) with its populated-range checks (`log2BlockSize` 0..=3
for diagonal / horizontal / vertical, 2..=5 for traverse), the form the
residual-coding path (§7.3.8.11 / §9.3.4.2.4) selects per block. Round 9
landed the
§6.5.3 up-right diagonal scan order (equation 6-11) and the §7.4.5
`ScalingFactor[sizeId][matrixId][x][y]` 2-D quantization-matrix
derivation (equations 7-44..7-51), building on round 8's §7.3.4
`scaling_list_data()` parse + flat `ScalingList[sizeId][matrixId][i]`
lists. `ScalingListData::scaling_factors(ChromaArrayType)` scatters
each flat coefficient to the `(x, y)` cell the up-right diagonal scan
maps it to (`ScanOrder[2][0]` for the 4x4 list, `ScanOrder[3][0]` for
every 8x8-based list), replicates each entry into a 2x2 (16x16) or 4x4
(32x32) block, applies the DC-coefficient `[0][0]` override
(equations 7-47 / 7-49 / 7-51), and — only when `ChromaArrayType == 3`
— derives the 32x32 chroma matrices (matrixId 1, 2, 4, 5) from the
16x16 lists (equations 7-50 / 7-51). Round 8 had landed the §7.3.4
parse plus the flat-list derivation (default 4x4 / 8x8 Tables 7-5 /
7-6, the `scaling_list_pred_matrix_id_delta` reference-list copy of
equations 7-42 / 7-43, the explicit delta-coded form's `nextCoef`
accumulator, and the §7.4.5 range checks), wired into both the SPS
(`sps_scaling_list_data_present_flag`) and PPS
(`pps_scaling_list_data_present_flag`) paths. It builds on round 7's
non-IDR POC + reference-picture-set slice sub-block, round 6's
slice-header parse, round 5's §7.3.2.3.1 PPS parse, round 4's complete
SPS RBSP body — PCM block, short-term reference picture sets (§7.3.7),
the long-term reference picture table, the
`sps_temporal_mvp_enabled_flag` / `strong_intra_smoothing_enabled_flag`
pair, and the VUI / extension gates — round 3's structural prefix,
round 2's VPS / profile-tier-level (§7.3.2.1 + §7.3.3), and round 1's
Annex B / NAL-header foundation. All independent I-slice segments (IDR
and non-IDR) parse end to end through `byte_alignment()`; only the P/B
reference-list / weighted-prediction sub-structures (§7.3.6.2 /
§7.3.6.3, both DPB-derived) remain surfaced as an opaque tail. Slice
data and CABAC remain unimplemented.

## Scope so far

* Annex B byte-stream splitting — 3-byte (`00 00 01`) and 4-byte
  (`00 00 00 01`) start codes; multiple NAL units per buffer.
* §7.3.1.2 NAL header parse — `forbidden_zero_bit`,
  `nal_unit_type`, `nuh_layer_id`, and `TemporalId` (derived from
  `nuh_temporal_id_plus1`).
* §7.4.1.1 emulation-prevention strip — every `0x00 0x00 0x03` in
  the on-wire payload is decoded back to `0x00 0x00`.
* MSB-first bit reader with `u(n)` and 0-th-order
  unsigned-Exp-Golomb `ue(v)` (§9.2) descriptors.
* §7.3.2.1 [`HevcVps`] — `vps_video_parameter_set_id`,
  base-layer / max-layers / sub-layers / temporal-nesting flags,
  `vps_reserved_0xffff_16bits` validation, the §7.3.3
  profile-tier-level walk (general profile + level + per-sub-layer
  present-flag gates + `sub_layer_level_idc`), and the per-sub-layer
  DPB / reorder / latency `ue(v)` triple loop with
  ordering-info-present-flag propagation.
* §7.3.2.2 [`SeqParameterSet`] — `sps_video_parameter_set_id`,
  `sps_max_sub_layers_minus1` / `sps_temporal_id_nesting_flag`, the
  §7.3.3 PTL re-walk, `sps_seq_parameter_set_id`, `chroma_format_idc`
  / `separate_colour_plane_flag`, `pic_width_in_luma_samples` /
  `pic_height_in_luma_samples`, `conformance_window_flag` + the four
  `conf_win_*_offset` `ue(v)` values, `bit_depth_luma_minus8` /
  `bit_depth_chroma_minus8`, `log2_max_pic_order_cnt_lsb_minus4`,
  the per-sub-layer DPB / reorder / latency triple loop,
  `log2_min_luma_coding_block_size_minus3`,
  `log2_diff_max_min_luma_coding_block_size`,
  `log2_min_luma_transform_block_size_minus2`,
  `log2_diff_max_min_luma_transform_block_size`,
  `max_transform_hierarchy_depth_{inter,intra}`,
  `scaling_list_enabled_flag`, `amp_enabled_flag`,
  `sample_adaptive_offset_enabled_flag`, the `pcm_*` block gated
  by `pcm_enabled_flag`, `num_short_term_ref_pic_sets` + the per-set
  [`ShortTermRefPicSet`] (§7.3.7 — both the explicit
  `num_negative_pics` / `num_positive_pics` form and the
  inter-RPS-prediction form, with `RefRpsIdx` chained through the
  preceding RPS list per §7.4.8), `long_term_ref_pics_present_flag`
  + the [`LongTermRefPicEntry`] table (`u(v)` POC-LSB width per
  `log2_max_pic_order_cnt_lsb_minus4 + 4`),
  `sps_temporal_mvp_enabled_flag`,
  `strong_intra_smoothing_enabled_flag`, and the
  `vui_parameters_present_flag` / `sps_extension_present_flag`
  gates. The VUI body and any extension payload are surfaced as
  [`OpaqueTail`] (raw RBSP bytes from the cut-off byte through the
  buffer end, with the start-bit offset). When
  `scaling_list_enabled_flag == 1` and
  `sps_scaling_list_data_present_flag == 1`, the §7.3.4
  `scaling_list_data()` block is parsed into [`ScalingListData`]
  (otherwise the §7.4.5 default lists apply).
* §7.3.2.3.1 [`PicParameterSet`] — the full general
  `pic_parameter_set_rbsp()` body: `pps_pic_parameter_set_id` /
  `pps_seq_parameter_set_id`, the slice-header gates
  (`dependent_slice_segments_enabled_flag`, `output_flag_present_flag`,
  `num_extra_slice_header_bits`, `sign_data_hiding_enabled_flag`,
  `cabac_init_present_flag`), `num_ref_idx_l{0,1}_default_active_minus1`,
  `init_qp_minus26` (`se(v)`), `constrained_intra_pred_flag`,
  `transform_skip_enabled_flag`, `cu_qp_delta_enabled_flag` +
  `diff_cu_qp_delta_depth`, `pps_c{b,r}_qp_offset` (`se(v)`),
  `pps_slice_chroma_qp_offsets_present_flag`, `weighted_pred_flag` /
  `weighted_bipred_flag`, `transquant_bypass_enabled_flag`, the tiles
  block ([`TileInfo`] — `num_tile_{columns,rows}_minus1`,
  `uniform_spacing_flag`, and the explicit
  `column_width_minus1[]` / `row_height_minus1[]` arrays when spacing
  is non-uniform, plus `loop_filter_across_tiles_enabled_flag`),
  `entropy_coding_sync_enabled_flag`,
  `pps_loop_filter_across_slices_enabled_flag`, the
  deblocking-filter-control block ([`DeblockingFilterControl`]),
  `lists_modification_present_flag`,
  `log2_parallel_merge_level_minus2`,
  `slice_segment_header_extension_present_flag`, and the
  `pps_extension_present_flag` gate (extension bodies surfaced as a
  shared [`OpaqueTail`]). The §7.4.3.3.1 inference rules are applied so
  absent conditional fields carry their effective value, and when
  `pps_scaling_list_data_present_flag == 1` the §7.3.4
  `scaling_list_data()` block is parsed into [`ScalingListData`]. The
  signed-Exp-Golomb `se(v)` descriptor (§9.2.2) was added to
  [`BitReader`] for the PPS QP / deblocking-offset fields.
* §7.3.4 [`ScalingListData`] — the `scaling_list_data()` syntax
  structure plus the §7.4.5 `ScalingList[sizeId][matrixId][i]`
  derivation: per-slot `scaling_list_pred_mode_flag`, the
  `scaling_list_pred_matrix_id_delta` reference-list / default-list
  selection (equations 7-42 / 7-43), the explicit delta-coded form
  (running `nextCoef` accumulator modulo 256 with the
  `scaling_list_dc_coef_minus8` DC coefficient for `sizeId > 1`), and
  the default 4x4 / 8x8 tables (Tables 7-5 / 7-6). The §7.4.5 range
  checks are enforced ([`ScalingListError`]).
  [`ScalingListData::scaling_factors`] expands the flat lists into the
  two-dimensional `ScalingFactor[sizeId][matrixId][x][y]` quantization
  matrices (equations 7-44..7-51): each flat coefficient is scattered
  to the `(x, y)` cell given by the §6.5.3 up-right diagonal scan, with
  the 2x / 4x block replication for the 16x16 / 32x32 sizes, the
  DC-coefficient `[0][0]` override (equations 7-47 / 7-49 / 7-51), and
  the `ChromaArrayType == 3` 32x32-chroma derivation from the 16x16
  lists (equations 7-50 / 7-51).
* §6.5 [`scan`] — all four scan-order initialization processes plus
  the §7.4.2 [`scan_order`] `ScanOrder[log2BlockSize][scanIdx]`
  accessor: [`up_right_diagonal`] (§6.5.3, equation 6-11),
  [`horizontal`] (§6.5.4, equation 6-12), [`vertical`] (§6.5.5,
  equation 6-13), and [`traverse`] (§6.5.6, equation 6-14 — the
  boustrophedon raster: even rows left-to-right, odd rows
  right-to-left). [`ScanIdx`] is the §7.4.2 selector (0 diagonal /
  1 horizontal / 2 vertical / 3 traverse) and [`scan_order`] enforces
  the table's populated ranges — `log2BlockSize` 0..=3 for the
  diagonal / horizontal / vertical scans, 2..=5 for the traverse scan
  ([`ScanOrderError`]). The §7.4.5 `ScalingFactor` derivation reads
  `ScanOrder[2][0]` (4x4) and `ScanOrder[3][0]` (8x8); the residual
  coding path (§7.3.8.11 / §9.3.4.2.4) reads the full table.
* §7.3.6.1 [`SliceSegmentHeader`] — the `slice_segment_header()` parse
  for an independent slice segment, taking the activated SPS + PPS as
  context: `first_slice_segment_in_pic_flag`,
  `no_output_of_prior_pics_flag` (IRAP-range only),
  `slice_pic_parameter_set_id`; for non-first segments
  `dependent_slice_segment_flag` + `slice_segment_address` (`u(v)`,
  width `Ceil( Log2( PicSizeInCtbsY ) )`); for independent segments the
  `slice_reserved_flag[]` block, [`SliceType`] (Table 7-7),
  `pic_output_flag`, `colour_plane_id`, the non-IDR POC + RPS block
  (`slice_pic_order_cnt_lsb` `u(v)`,
  `short_term_ref_pic_set_sps_flag` `u(1)` with the §7.4.7.1
  `num_short_term_ref_pic_sets == 0` cross-check, the in-line
  `st_ref_pic_set(num_short_term_ref_pic_sets)` parse via
  [`ShortTermRefPicSet::parse_slice_inline`],
  `short_term_ref_pic_set_idx` `u(v)`, and the long-term-ref-pic
  block — `num_long_term_sps` / `num_long_term_pics` / per-entry
  [`SliceLongTermRefPic`] with [`SliceLongTermRefPicSource`]
  discriminating SPS-indexed vs in-slice signalling and
  `delta_poc_msb_present_flag` / `delta_poc_msb_cycle_lt`),
  `slice_temporal_mvp_enabled_flag`, and the SAO luma / chroma gates.
  Independent I-slice segments — IDR and non-IDR alike — parse end to
  end through `byte_alignment()`: `slice_qp_delta` (`se(v)`), the
  chroma QP offsets, the deblocking override block
  ([`SliceDeblocking`]), `slice_loop_filter_across_slices_enabled_flag`,
  the entry-point-offset block ([`EntryPointOffsets`]), and the
  header-extension block;
  [`SliceSegmentHeader::byte_offset_to_slice_data`] reports where
  `slice_segment_data()` begins, and
  [`SliceSegmentHeader::slice_qp_y`] applies equation 7-54. The
  §7.4.7.1 inference rules are applied to absent fields. The P/B
  reference-list / weighted-prediction sub-structures (which need DPB
  state) are surfaced as an [`OpaqueTail`] rather than decoded.

Top-level entry points: [`NalIter`], [`collect_nal_units`],
[`NalHeader::parse`], [`strip_emulation_prevention`],
[`BitReader`], [`HevcVps::parse`], [`ProfileTierLevel::parse`],
[`SeqParameterSet::parse`], [`PicParameterSet::parse`],
[`SliceSegmentHeader::parse`], [`scan_order`].

## Not yet implemented

* VUI parameters (§E.2.1) — currently surfaced as opaque bytes on
  the parsed SPS struct.
* SPS extension bodies (Range Extension, Multilayer Annex F,
  3D Annex I, SCC) — likewise surfaced as opaque bytes.
* PPS extension bodies (`pps_range_extension()`,
  `pps_multilayer_extension()`, `pps_3d_extension()`,
  `pps_scc_extension()`) — surfaced as opaque bytes when
  `pps_extension_present_flag == 1`.
* VPS tail: `vps_max_layer_id`, layer-set inclusion matrix,
  `vps_timing_info_present_flag`, HRD parameters,
  `vps_extension_data_flag`.
* Slice header (§7.3.6.1) deferred body: the P/B
  `ref_pic_lists_modification()` (§7.3.6.2) /
  `pred_weight_table()` (§7.3.6.3) sub-structures — surfaced as an
  opaque tail for P/B slice headers; need DPB-derived
  `NumPicTotalCurr` / `RefPicList` state. The non-IDR POC /
  reference-picture-set block (which previously sat under this bullet)
  is fully decoded as of round 7.
* Slice data (§7.3.8)
* CABAC entropy coding (§9.3) — blocked on the docs `cu_qp_delta`
  + `last_sig_coeff` multi-QG / multi-CTU 4:2:2 trace gap.
* Intra / inter prediction, transform, in-loop filters (deblock /
  SAO), DPB management.
* Encoder.

The runtime registration hook (`register`) is still a no-op.

## License

MIT — see [LICENSE](./LICENSE).
