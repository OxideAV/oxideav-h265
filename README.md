# oxideav-h265

A pure-Rust H.265 / HEVC video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Clean-room rebuild — round 16 (2026-05-26).** The prior implementation was
retired under the workspace
[clean-room policy](https://github.com/OxideAV/oxideav/blob/master/docs/IMPLEMENTOR_ROUND.md):
a CTU-level source comment cited a specific named variable and line
number in an external library's HEVC decoder — clean-room provenance
for the surrounding code path could not be defended. Master history
was fully erased per the Hat-3 cold-enforcement procedure.

The rebuild is in progress against the published H.265 specification
(ITU-T Recommendation H.265 | ISO/IEC 23008-2). Round 16 lands the
§7.4.7.2 `NumPicTotalCurr` derivation (equation 7-57), the explicit
follow-up to round 15's standalone `RefPicListsModification` parser.
The new [`NumPicTotalCurrInputs`] builder takes the per-position
`UsedByCurrPicS0` / `UsedByCurrPicS1` / `UsedByCurrPicLt` flags from
the active short-term RPS plus the slice's long-term ref list, plus
the `pps_curr_pic_ref_enabled_flag` closing clause, and returns the
typed `NumPicTotalCurr: u32`. A
[`NumPicTotalCurrInputs::from_explicit_short_term_rps`] convenience
constructor pulls the `S0` / `S1` slices straight off a
[`ShortTermRefPicSet`] in explicit form (the inter-RPS-predicted
form requires the §7.4.8 derivation to run first; the builder
returns `None` and the caller threads through
[`NumPicTotalCurrInputs::from_used_flags`]). A
[`SliceLongTermRefPic::used_by_curr_pic_lt`] helper resolves each
`UsedByCurrPicLt[i]` per §7.4.7.1 — SPS-table lookup for `Sps
{ lt_idx_sps }` entries, direct flag for `InSlice` entries. The
F.7.4.7.2 multilayer-extension form (equation F-56) is wired through
a `with_multilayer_extension(nal_unit_type, num_active_ref_layer_pics)`
builder method, so when the multilayer profile becomes active the
IDR short-/long-term-loop skip plus the `NumActiveRefLayerPics`
summand land without further surface change. The §7.3.6.1 in-place
wiring of round 15's standalone `RefPicListsModification` parser
remains the next round's target — the derivation itself is no
longer the blocker.
Round 15 had landed the
§7.3.6.2 `ref_pic_lists_modification()` syntax structure as a
standalone parser ([`RefPicListsModification::parse`]): the
`ref_pic_list_modification_flag_l0` `u(1)` gate, the
`list_entry_l0[ 0 .. num_ref_idx_l0_active_minus1 ]` `u(v)` loop with
the per-entry width set to `Ceil( Log2( NumPicTotalCurr ) )` bits
(§7.4.7.2) and each value range-checked to `0 ..=
NumPicTotalCurr - 1`, the B-slice-gated `ref_pic_list_modification_flag_l1`
`u(1)` plus the matching `list_entry_l1[]` loop, and the up-front
preconditions that reject `SliceType::I` calls (the §7.3.6.1 gate
sits inside the inter-slice branch), `NumPicTotalCurr <= 1` (the
§7.3.6.1 gate guarantees `> 1` at the call site), and
`num_ref_idx_lX_active_minus1 > 14` (the §7.4.7.1 cap on
`num_ref_idx_lX_active_minus1`). The implicit `RefPicListTempX`
derivation of §8.3.4 stays the consumer's responsibility; the parser
materialises only the on-wire syntax elements.
[`slice::SliceSegmentHeader::parse`] still surfaces the inter-slice
tail as an [`sps::OpaqueTail`] — the in-place call site is now
unblocked but the §7.3.6.1 inter-slice body parse (pred-weight-table
+ five-or-six-more-flags + slice-data offset) is itself a separate
round's worth of work.
Round 14 had landed the
§E.2.1 `vui_parameters()` body as a typed [`VuiParameters`] (see
"Scope so far" below). Round 13 had landed the
§E.2.2 / §E.2.3 `hrd_parameters()` and `sub_layer_hrd_parameters()`
bodies as a new `hrd` module ([`HrdParameters`] / [`HrdCommonInfo`] /
[`SubLayerHrd`] / [`SubLayerHrdParameters`] / [`CpbEntry`] /
[`VpsHrdEntry`]). The decoder walks the common-info gates
(`nal_hrd_parameters_present_flag` /
`vcl_hrd_parameters_present_flag` /
`sub_pic_hrd_params_present_flag` and the conditional
`tick_divisor_minus2` / `du_cpb_removal_delay_increment_length_minus1`
/ `sub_pic_cpb_params_in_pic_timing_sei_flag` /
`dpb_output_delay_du_length_minus1` / `bit_rate_scale` /
`cpb_size_scale` / `cpb_size_du_scale` /
`initial_cpb_removal_delay_length_minus1` /
`au_cpb_removal_delay_length_minus1` /
`dpb_output_delay_length_minus1` block); the per-sub-layer loop
(`fixed_pic_rate_general_flag[i]` /
`fixed_pic_rate_within_cvs_flag[i]` with the §E.3.2 "general == 1 ⇒
within_cvs := 1" inference, `elemental_duration_in_tc_minus1[i]`
`ue(v)` range-checked at 0..=2047, `low_delay_hrd_flag[i]`, and
`cpb_cnt_minus1[i]` `ue(v)` range-checked at 0..=31); and the §E.2.3
sub-layer body with the §E.3.3 monotonicity constraints
(`bit_rate_value_minus1[i]` strictly increasing,
`cpb_size_value_minus1[i]` non-increasing, sub-pic variants gated on
`sub_pic_hrd_params_present_flag`). The VPS RBSP now decodes the
§7.3.2.1 per-HRD loop inline ([`HevcVps::hrd_parameters`]) with the
`cprms_present_flag[i] == 0` inheritance walked through the prior
entry's [`HrdCommonInfo`]; [`HevcVps::vps_extension_flag`] is now an
unconditional `bool`, and [`HevcVps::opaque_tail`] is populated only
for the extension-data + `rbsp_trailing_bits()` suffix.
Round 12 had finished the
§7.3.2.1 VPS tail through the optional VPS timing-info block: the
`vps_max_layer_id` (`u(6)`) and `vps_num_layer_sets_minus1` (`ue(v)`,
range 0..=1023, capped at `HEVC_VPS_MAX_NUM_LAYER_SETS = 1024`)
fields, the
`layer_id_included_flag[1..=num_layer_sets_minus1][0..=max_layer_id]`
inclusion matrix (one row per signalled layer set; layer set 0 is
implicit per §7.4.3.1), and the `vps_timing_info_present_flag` block
([`VpsTimingInfo`] — `vps_num_units_in_tick` / `vps_time_scale` both
`u(32)` with the §E.2.1 / §7.3.2.1 "shall be > 0" semantics enforced,
the `vps_poc_proportional_to_timing_flag` gate with the optional
`vps_num_ticks_poc_diff_one_minus1` `ue(v)`, and the
`vps_num_hrd_parameters` `ue(v)` count bounded at
`vps_num_layer_sets_minus1 + 1`).
Round 11 landed the
§9.3 CABAC arithmetic decoding engine as a standalone module
([`cabac`]): the §9.3.2.6 engine-register initialization
(`ivlCurrRange = 510`, `ivlOffset = read_bits(9)`, with the spec's
"`ivlOffset` shall not equal 510 or 511" constraint enforced); the
§9.3.2.2 context-variable initialization (equations 9-4..9-6, with
[`ContextModel::init`] taking an 8-bit `initValue` and `SliceQpY`)
plus the §9.3.2.2 `initType` selector (equation 9-7,
[`init_type`]); the §9.3.4.3.2 `DecodeDecision` primitive with the
Table 9-52 `rangeTabLps[64][4]` interval split and the §9.3.4.3.2.2
Table 9-53 (`transIdxLps` / `transIdxMps`) state transition; the
§9.3.4.3.3 renormalization loop (`RenormD`); the §9.3.4.3.4
`DecodeBypass` equal-probability bin plus a `decode_bypass_bits(n)`
MSB-first helper; the §9.3.4.3.5 `DecodeTerminate` decision-before-
termination (`end_of_slice_segment_flag` / `end_of_subset_one_bit` /
`pcm_flag`); and the §9.3.4.3.6 aligned-bypass alignment hook
(`ivlCurrRange = 256` before `coeff_abs_level_remaining[ ]` /
`coeff_sign_flag[ ]`). The engine is the gateway to slice-data
decode — every §7.3.8 syntax element is read through these four
primitives — and ships independently of the §9.3.4.2 per-syntax-
element binarization / context-index derivation, which sits one
layer up in the slice-data parser (still blocked on the docs
trace gap for `cu_qp_delta` + `last_sig_coeff` multi-QG / 4:2:2).
Round 10 completed the
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
  present-flag gates + `sub_layer_level_idc`), the per-sub-layer
  DPB / reorder / latency `ue(v)` triple loop with
  ordering-info-present-flag propagation, `vps_max_layer_id` /
  `vps_num_layer_sets_minus1` + the
  `layer_id_included_flag[i][j]` inclusion matrix (one
  [`LayerIdInclusionRow`] per signalled layer set), the
  `vps_timing_info_present_flag` block ([`VpsTimingInfo`] — the
  `u(32)` `vps_num_units_in_tick` / `vps_time_scale` pair with the
  spec's "shall be > 0" semantics enforced, the
  `vps_poc_proportional_to_timing_flag` gate +
  `vps_num_ticks_poc_diff_one_minus1`, and the
  `vps_num_hrd_parameters` count), the per-HRD loop
  ([`HevcVps::hrd_parameters`] — one [`VpsHrdEntry`] per
  `vps_num_hrd_parameters` with the §7.3.2.1 `hrd_layer_set_idx[i]` +
  `cprms_present_flag[i]` prelude, decoding each
  `hrd_parameters(cprms, vps_max_sub_layers_minus1)` body via the
  shared `hrd` module), and the `vps_extension_flag` gate. When
  `vps_extension_flag == 1` the extension-data run +
  `rbsp_trailing_bits()` are surfaced as an [`OpaqueTail`]
  (`HevcVps::opaque_tail`).
* §E.2.2 / §E.2.3 [`HrdParameters`] / [`SubLayerHrdParameters`] —
  the common-info block ([`HrdCommonInfo`]) with the §E.2.2
  `nal_hrd_parameters_present_flag` / `vcl_hrd_parameters_present_flag`
  / `sub_pic_hrd_params_present_flag` gates and the conditional `u(8)`
  / `u(5)` / `u(4)` length / scale fields; the per-sub-layer loop
  ([`SubLayerHrd`] with the §E.3.2 inference rules for
  `fixed_pic_rate_within_cvs_flag` /  `low_delay_hrd_flag` /
  `cpb_cnt_minus1` plus the `elemental_duration_in_tc_minus1[i]`
  0..=2047 and `cpb_cnt_minus1[i]` 0..=31 range-checks); the §E.2.3
  per-CPB array ([`CpbEntry`]) with the §E.3.3 monotonicity
  constraints enforced inline; and the §7.3.2.1 `cprms_present_flag[i]
  == 0` inheritance of common-info gates from the previous entry.
* §E.2.1 [`VuiParameters`] — the full `vui_parameters()` body:
  aspect-ratio info (`aspect_ratio_idc` + the [`EXTENDED_SAR`]
  `sar_width` / `sar_height` `u(16)` pair), overscan, the
  [`VideoSignalType`] block (`video_format` / `video_full_range_flag`
  + the [`ColourDescription`] `colour_primaries` /
  `transfer_characteristics` / `matrix_coeffs` triple), chroma-loc
  info (`chroma_sample_loc_type_{top,bottom}_field` 0..=5
  range-checked), the neutral-chroma / field-seq / frame-field flags,
  the [`DefaultDisplayWindow`] offset quad, the [`VuiTimingInfo`]
  block (`u(32)` `vui_num_units_in_tick` / `vui_time_scale` enforced
  `> 0` per §E.3.1, POC-proportional flag +
  `vui_num_ticks_poc_diff_one_minus1`, and the nested §E.2.3
  `hrd_parameters( 1, sps_max_sub_layers_minus1 )` call reusing
  [`HrdParameters`]), and the [`BitstreamRestriction`] block (with
  the §E.3.1 `min_spatial_segmentation_idc` 0..=4095,
  `max_bytes_per_pic_denom` / `max_bits_per_min_cu_denom` 0..=16, and
  `log2_max_mv_length_{horizontal,vertical}` 0..=15 range-checks).
  Validated against the libx265 tiny-fixture VUI (1:1 SAR, 25 fps
  timing).
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
  `strong_intra_smoothing_enabled_flag`, the
  `vui_parameters_present_flag` gate whose §E.2.1 `vui_parameters()`
  body is decoded into [`VuiParameters`] (see below), and the
  `sps_extension_present_flag` gate (extension payload surfaced as
  an [`OpaqueTail`] — raw RBSP bytes from the cut-off byte through
  the buffer end, with the start-bit offset). When
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
  state) are surfaced as an [`OpaqueTail`] rather than decoded; the
  §7.3.6.2 syntax structure itself is implemented as a standalone
  parser ([`RefPicListsModification`], see below).

* §7.3.6.2 [`RefPicListsModification`] — the
  `ref_pic_lists_modification()` syntax structure as a standalone
  parser, callable by a future round once the §7.4.7.2
  `NumPicTotalCurr` derivation is wired through the slice parser.
  [`RefPicListsModification::parse`] takes the active
  `slice_type` / `num_ref_idx_l0_active_minus1` /
  `num_ref_idx_l1_active_minus1` / `NumPicTotalCurr` and walks
  the `ref_pic_list_modification_flag_l0` gate + the
  `list_entry_l0[0..=num_ref_idx_l0_active_minus1]` `u(v)` loop
  (each entry `Ceil( Log2( NumPicTotalCurr ) )` bits wide and
  range-checked at `<= NumPicTotalCurr - 1` per §7.4.7.2), then —
  for B slices only — the `ref_pic_list_modification_flag_l1`
  gate + its `list_entry_l1[]` loop. The parser rejects
  preconditions that the §7.3.6.1 call site would have filtered
  (`SliceType::I` and `NumPicTotalCurr <= 1`) and the
  §7.4.7.1 `num_ref_idx_lX_active_minus1 > 14` cap. The implicit
  `RefPicListTempX` derivation of §8.3.4 stays the consumer's
  responsibility; this struct surfaces only the on-wire syntax.

* §7.4.7.2 [`NumPicTotalCurrInputs`] — the `NumPicTotalCurr`
  derivation (equation 7-57) as a small typed builder. The caller
  supplies the per-position `UsedByCurrPicS0` / `UsedByCurrPicS1` /
  `UsedByCurrPicLt` flags from the active short-term RPS plus the
  slice's long-term ref list and the
  `pps_curr_pic_ref_enabled_flag` (inferred to `false` until the SCC
  PPS extension is materialised); [`NumPicTotalCurrInputs::compute`]
  returns the typed `NumPicTotalCurr: u32`.
  [`NumPicTotalCurrInputs::from_explicit_short_term_rps`] pulls the
  `S0` / `S1` slices directly off a [`ShortTermRefPicSet`] in
  explicit form (the inter-RPS-prediction form returns `None`; the
  §7.4.8 derivation must be run first and the result threaded
  through [`NumPicTotalCurrInputs::from_used_flags`]). The
  [`SliceLongTermRefPic::used_by_curr_pic_lt`] helper resolves each
  long-term entry's `UsedByCurrPicLt[i]` per §7.4.7.1: SPS-table
  lookup for `Sps { lt_idx_sps }` entries, direct flag for in-slice
  entries. The F.7.4.7.2 multilayer-extension form (equation F-56)
  is reachable through
  [`NumPicTotalCurrInputs::with_multilayer_extension`] —
  IDR-`nal_unit_type` slices skip the short-term / long-term loops
  and the `NumActiveRefLayerPics` summand is added at the end.

* §7.3.6.3 [`PredWeightTable`] — the `pred_weight_table()` syntax
  structure as a standalone parser, callable by a future round once
  the §7.3.6.1 in-place call site is wired up.
  [`PredWeightTable::parse`] reads `luma_log2_weight_denom`
  (range 0..=7) and, when `ChromaArrayType != 0`,
  `delta_chroma_log2_weight_denom` with the derived
  `ChromaLog2WeightDenom ∈ 0..=7` range check, then performs the two
  flag passes (`luma_weight_lX_flag[i]` and, when chroma is present,
  `chroma_weight_lX_flag[i]`) and the per-reference delta block
  (`delta_luma_weight_lX[i]` / `luma_offset_lX[i]` plus the chroma
  `delta_chroma_weight_lX[i][j]` / `delta_chroma_offset_lX[i][j]`
  pairs), enforcing every §7.4.7.3 range bound including the
  bit-depth-dependent luma/chroma offset bounds parameterised by
  [`PredWeightTableInputs::high_precision_offsets_enabled_flag`] and
  the SPS bit depths. For B slices the L1 block is mirrored after
  L0. The §7.4.7.3 conformance `sumWeightLXFlags ≤ 24` cap is
  enforced. The §7.3.6.3 outer-gate (`pic_layer_id != nuh_layer_id ||
  PicOrderCnt(RefPicListX[i]) != PicOrderCnt(CurrPic)`) per-i
  decision is supplied by the caller through
  [`PredWeightTableInputs::signal_luma_l0`] /
  [`PredWeightTableInputs::signal_chroma_l0`] (mirrored for L1); the
  [`PredWeightTableInputs::base_profile`] constructor leaves all four
  `None` (every position gated `true`), the universal base-profile
  single-layer case. Accessor methods
  [`PredWeightTable::luma_weight_l0`] (mirrored for L1),
  [`PredWeightTable::chroma_weight_l0`] (mirrored) and
  [`PredWeightTable::chroma_offset_l0`] (mirrored, equation 7-58)
  resolve the §7.4.7.3 derived variables `LumaWeightLX[i]`,
  `ChromaWeightLX[i][j]` and `ChromaOffsetLX[i][j]`.

* §9.3 [`cabac`] — the CABAC arithmetic decoding engine as a
  standalone module: [`CabacEngine::new`] initializes the §9.3.2.6
  registers (`ivlCurrRange = 510`, `ivlOffset = read_bits(9)`, with
  the spec's "ivlOffset shall not equal 510 or 511" constraint
  enforced) over a positioned [`BitReader`]; [`ContextModel::init`]
  derives a `(pStateIdx, valMps)` pair from an 8-bit `initValue` and
  `SliceQpY` per equations 9-4..9-6, with the §9.3.2.2 `initType`
  selector (equation 9-7) exposed as the free function [`init_type`];
  [`CabacEngine::decode_decision`] implements §9.3.4.3.2 with the
  Table 9-52 `rangeTabLps[64][4]` quantized-range split, the
  §9.3.4.3.2.2 Table 9-53 state transition, and the §9.3.4.3.3
  `RenormD` loop; [`CabacEngine::decode_bypass`] /
  [`CabacEngine::decode_bypass_bits`] implement §9.3.4.3.4 (the
  helper accumulates `n` bins MSB-first);
  [`CabacEngine::decode_terminate`] implements §9.3.4.3.5
  (`end_of_slice_segment_flag` / `end_of_subset_one_bit` / `pcm_flag`,
  the `ctxTable == 0`, `ctxIdx == 0` decision before termination); and
  [`CabacEngine::align`] implements §9.3.4.3.6 (the `ivlCurrRange =
  256` alignment hook prior to aligned bypass decoding for
  `coeff_abs_level_remaining[ ]` / `coeff_sign_flag[ ]`). The
  Table 9-52 / Table 9-53 values are transcribed directly from the
  H.265 specification.

Top-level entry points: [`NalIter`], [`collect_nal_units`],
[`NalHeader::parse`], [`strip_emulation_prevention`],
[`BitReader`], [`HevcVps::parse`], [`ProfileTierLevel::parse`],
[`SeqParameterSet::parse`], [`PicParameterSet::parse`],
[`SliceSegmentHeader::parse`], [`RefPicListsModification::parse`],
[`scan_order`], [`CabacEngine::new`].

## Not yet implemented

* SPS extension bodies (Range Extension, Multilayer Annex F,
  3D Annex I, SCC) — likewise surfaced as opaque bytes.
* PPS extension bodies (`pps_range_extension()`,
  `pps_multilayer_extension()`, `pps_3d_extension()`,
  `pps_scc_extension()`) — surfaced as opaque bytes when
  `pps_extension_present_flag == 1`.
* VPS `vps_extension_data_flag` extension payload — the §F / §G / §H
  / §I multi-layer / 3D / SCC VPS-extension syntax; surfaced as
  [`HevcVps::opaque_tail`] when `vps_extension_flag == 1`. The §E.2.2
  `hrd_parameters()` bodies are now fully decoded (round 13).
* Slice header (§7.3.6.1) deferred body: the P/B
  `ref_pic_lists_modification()` (§7.3.6.2) /
  `pred_weight_table()` (§7.3.6.3) sub-structures —
  `SliceSegmentHeader::parse` still surfaces these as an opaque tail
  for P/B slice headers because the §7.3.6.1 in-place call site has
  not yet been re-entered. The §7.3.6.2 syntax structure itself is
  decoded by the standalone [`RefPicListsModification::parse`]
  (round 15), the §7.4.7.2 `NumPicTotalCurr` derivation is
  available as [`NumPicTotalCurrInputs::compute`] (round 16), and
  the §7.3.6.3 `pred_weight_table()` syntax structure is decoded by
  the standalone [`PredWeightTable::parse`] (round 17); a future
  round threads all three together at the §7.3.6.1 call site (the
  full inter-slice body also needs the
  `num_ref_idx_active_override_flag` / `num_ref_idx_lX_active_minus1`
  override block, `mvd_l1_zero_flag`, `cabac_init_flag`,
  `collocated_from_l0_flag` / `collocated_ref_idx`,
  `five_minus_max_num_merge_cand`, the `use_integer_mv_flag` SCC
  closing flag, and the QP-offset / deblocking / loop-filter tail).
  The non-IDR POC / reference-picture-set block (which previously sat
  under this bullet) is fully decoded as of round 7.
* Slice data (§7.3.8) — the slice-data syntax-element walk that
  drives the CABAC engine. Needs the §9.3.4.2 per-syntax-element
  binarization / context-index derivation (which selects the
  `ctxTable` / `ctxIdx` for each bin) and the §7.3.8.1..§7.3.8.12
  parse loops.
* §9.3.4.2 per-syntax-element binarization / context-index
  derivation — blocked on the docs `cu_qp_delta` + `last_sig_coeff`
  multi-QG / multi-CTU 4:2:2 trace gap. (The §9.3 arithmetic decode
  engine itself — DecodeDecision / DecodeBypass / DecodeTerminate
  / RenormD / alignment — is implemented as of round 11; it sits
  one layer below the binarization tables and is not affected by
  the trace gap.)
* Intra / inter prediction, transform, in-loop filters (deblock /
  SAO), DPB management.
* Encoder.

The runtime registration hook (`register`) is still a no-op.

## License

MIT — see [LICENSE](./LICENSE).
