# oxideav-h265

A pure-Rust H.265 / HEVC video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Clean-room rebuild — round 5 (2026-05-24).** The prior implementation was
retired under the workspace
[clean-room policy](https://github.com/OxideAV/oxideav/blob/master/docs/IMPLEMENTOR_ROUND.md):
a CTU-level source comment cited a specific named variable and line
number in an external library's HEVC decoder — clean-room provenance
for the surrounding code path could not be defended. Master history
was fully erased per the Hat-3 cold-enforcement procedure.

The rebuild is in progress against the published H.265 specification
(ITU-T Recommendation H.265 | ISO/IEC 23008-2). Round 5 adds the
§7.3.2.3.1 PPS parse — the full general `pic_parameter_set_rbsp()`
body through `pps_extension_present_flag`, including the tiles block
(column/row counts plus explicit `column_width_minus1[]` /
`row_height_minus1[]` arrays) and the deblocking-filter-control block,
with the §7.4.3.3.1 inference rules applied to absent fields and PPS
extension bodies surfaced as an opaque-bytes tail. It builds on
round 4's complete SPS RBSP body — PCM block, short-term reference
picture sets (§7.3.7), the long-term reference picture table, the
`sps_temporal_mvp_enabled_flag` / `strong_intra_smoothing_enabled_flag`
pair, and the VUI / extension gates — round 3's structural prefix,
round 2's VPS / profile-tier-level (§7.3.2.1 + §7.3.3), and round 1's
Annex B / NAL-header foundation. Slice decode, scaling-list data, and
CABAC remain unimplemented.

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
  buffer end, with the start-bit offset). `scaling_list_data()`
  itself is still deferred — the parser refuses
  `scaling_list_enabled_flag == 1`.
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
  absent conditional fields carry their effective value, and
  `pps_scaling_list_data_present_flag == 1` is refused
  ([`PpsError::ScalingListUnsupported`]) alongside the SPS scaling-list
  deferral. The signed-Exp-Golomb `se(v)` descriptor (§9.2.2) was
  added to [`BitReader`] for the PPS QP / deblocking-offset fields.

Top-level entry points: [`NalIter`], [`collect_nal_units`],
[`NalHeader::parse`], [`strip_emulation_prevention`],
[`BitReader`], [`HevcVps::parse`], [`ProfileTierLevel::parse`],
[`SeqParameterSet::parse`], [`PicParameterSet::parse`].

## Not yet implemented

* `scaling_list_data()` (§7.3.4) — both the SPS and PPS paths still
  error out when their scaling-list-present flags are set
  (`scaling_list_enabled_flag == 1` /
  `pps_scaling_list_data_present_flag == 1`).
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
* Slice header parse (§7.3.6) and slice data (§7.3.8)
* CABAC entropy coding (§9.3) — blocked on the docs `cu_qp_delta`
  + `last_sig_coeff` multi-QG / multi-CTU 4:2:2 trace gap.
* Intra / inter prediction, transform, in-loop filters (deblock /
  SAO), DPB management.
* Encoder.

The runtime registration hook (`register`) is still a no-op.

## License

MIT — see [LICENSE](./LICENSE).
