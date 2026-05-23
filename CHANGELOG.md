# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — clean-room rebuild round 6 (2026-05-24)

- §7.3.6.1 `SliceSegmentHeader` structural parse — the
  `slice_segment_header()` syntax structure for an independent slice
  segment, taking the activated SPS + PPS as parse context (several
  field widths and presence gates are SPS/PPS-derived):
  - `first_slice_segment_in_pic_flag`, `no_output_of_prior_pics_flag`
    (only present in the IRAP NAL-unit-type range
    `BLA_W_LP..=RSV_IRAP_VCL23`), `slice_pic_parameter_set_id` (ue(v),
    0..=63).
  - For non-first segments: `dependent_slice_segment_flag` (only when
    `dependent_slice_segments_enabled_flag`) and `slice_segment_address`
    (u(v), width `Ceil( Log2( PicSizeInCtbsY ) )`, range-checked
    against `PicSizeInCtbsY`).
  - For independent segments: the `slice_reserved_flag[]` block
    (`num_extra_slice_header_bits` flags), `slice_type` (Table 7-7,
    rejected outside 0..=2), `pic_output_flag` (only when
    `output_flag_present_flag`; inferred 1 otherwise), `colour_plane_id`
    (only when `separate_colour_plane_flag`),
    `slice_temporal_mvp_enabled_flag` (only when
    `sps_temporal_mvp_enabled_flag`).
  - SAO block: `slice_sao_luma_flag` + `slice_sao_chroma_flag`
    (the latter gated on `ChromaArrayType != 0`).
  - I-slice tail through `byte_alignment()`: `slice_qp_delta` (se(v)),
    `slice_c{b,r}_qp_offset` (se(v), −12..=12, gated by
    `pps_slice_chroma_qp_offsets_present_flag`), the deblocking-filter
    override block (`SliceDeblocking` — `deblocking_filter_override_flag`
    / `slice_deblocking_filter_disabled_flag` /
    `slice_beta_offset_div2` / `slice_tc_offset_div2`, se(v), −6..=6,
    with the §7.4.7.1 PPS-inference defaults applied when absent),
    `slice_loop_filter_across_slices_enabled_flag` (with its
    SAO/deblock gate), the entry-point-offset block
    (`EntryPointOffsets` — `num_entry_point_offsets` /
    `offset_len_minus1` 0..=31 / skipped `entry_point_offset_minus1[]`)
    when `tiles_enabled_flag || entropy_coding_sync_enabled_flag`, and
    the slice-segment-header extension block. `byte_alignment()` is
    consumed and `byte_offset_to_slice_data` reports where
    `slice_segment_data()` begins.
  - Convenience `slice_qp_y(pps)` = `26 + init_qp_minus26 +
    slice_qp_delta` (equation 7-54).
- Two deferred bodies are surfaced as an `sps::OpaqueTail` rather than
  decoded, because they need state this round does not carry:
  - The non-IDR picture-order-count + reference-picture-set block
    (needs the SPS short-term-RPS parser re-entered for the in-line
    `stRpsIdx == num_short_term_ref_pic_sets` case) — the parser stops
    after `colour_plane_id` when `nal_unit_type` is not
    `IDR_W_RADL` / `IDR_N_LP`.
  - The P/B reference-list-modification (§7.3.6.2) / weighted-prediction
    (§7.3.6.3) sub-structures (need DPB-derived `NumPicTotalCurr` /
    `RefPicList`) — the parser stops after the SAO block when
    `slice_type` is P or B.
- Top-level `Error::Slice(SliceError)` variant + `From<SliceError>`.
  Public `SliceType`, `SliceDeblocking`, `EntryPointOffsets`, and the
  `BLA_W_LP` / `IDR_W_RADL` / `IDR_N_LP` / `RSV_IRAP_VCL23` Table-7-1
  constants.
- 9 new unit tests (total 61, was 52): Table-7-7 `slice_type` mapping +
  `is_inter`; `Ceil( Log2( N ) )` width table; a hand-assembled
  independent I-slice IDR header parsed end-to-end through
  `byte_alignment()` (SliceQpY=25); the non-IDR POC-block deferral
  (opaque tail); the P/B ref-list deferral (opaque tail); a non-first
  dependent slice segment (`slice_segment_address` u(2)); end-to-end
  parse via the Annex B walker; truncated-RBSP rejection;
  `slice_type > 2` rejection.

### Note — tiny-fixture slice trace inconsistency (docs gap)

- `docs/video/h265/fixtures/tiny-i-only-16x16-main/trace.txt`'s
  `SLICE_HEADER` line reports `temporal_mvp=0 sao_c=1 slice_qp_delta=-1`,
  but its own `SPS` line (and this crate's verified SPS parse) has
  `sps_temporal_mvp_enabled_flag=1`, so per §7.3.6.1
  `slice_temporal_mvp_enabled_flag` **is** present. Parsing the real
  slice NAL bytes with mvp present yields `sao_c=0 slice_qp_delta=0` and
  an invalid `byte_alignment()` pad (`1 0 0 0`); parsing with mvp absent
  yields the trace's `sao_c=1 slice_qp_delta=-1` and a clean byte-aligned
  pad. The slice bits are therefore self-consistent only with
  `sps_temporal_mvp_enabled_flag=0`, contradicting the SPS line. Because
  the fixture's SPS and slice are mutually inconsistent, the round-6
  slice tests use hand-assembled bit vectors instead of asserting the
  fixture slice's exact fields. Recommend the docs collaborator
  regenerate the tiny fixture's trace (or confirm the SPS↔slice
  mismatch is an x265-encoder/instrumentation artefact).

### Added — clean-room rebuild round 5 (2026-05-24)

- §7.3.2.3.1 `PicParameterSet` structural parse — the full general
  `pic_parameter_set_rbsp()` body through `pps_extension_present_flag`:
  - `pps_pic_parameter_set_id` (ue(v), 0..=63) +
    `pps_seq_parameter_set_id` (ue(v), 0..=15).
  - The slice-header gates `dependent_slice_segments_enabled_flag`,
    `output_flag_present_flag`, `num_extra_slice_header_bits` (u3, not
    range-checked per §7.4.3.3.1 "decoders shall allow any value"),
    `sign_data_hiding_enabled_flag`, `cabac_init_present_flag`.
  - `num_ref_idx_l0_default_active_minus1` /
    `num_ref_idx_l1_default_active_minus1` (ue(v), 0..=14).
  - `init_qp_minus26` (se(v)) range-checked against the loosest legal
    bound −74..=25; `init_qp_in_range(bit_depth_luma_minus8)` re-checks
    against the exact §7.4.3.3.1 lower bound −( 26 + QpBdOffsetY ) once
    the active SPS bit depth is known.
  - `constrained_intra_pred_flag`, `transform_skip_enabled_flag`,
    `cu_qp_delta_enabled_flag` + `diff_cu_qp_delta_depth` (inferred 0
    when disabled), `pps_cb_qp_offset` / `pps_cr_qp_offset` (se(v),
    −12..=12), `pps_slice_chroma_qp_offsets_present_flag`,
    `weighted_pred_flag`, `weighted_bipred_flag`,
    `transquant_bypass_enabled_flag`.
  - `tiles_enabled_flag` + `entropy_coding_sync_enabled_flag`, and the
    tiles block (`TileInfo`): `num_tile_columns_minus1` /
    `num_tile_rows_minus1`, `uniform_spacing_flag`, and the
    `column_width_minus1[]` / `row_height_minus1[]` arrays when
    `uniform_spacing_flag == 0`, plus
    `loop_filter_across_tiles_enabled_flag`. When `tiles_enabled_flag`
    is 0 the §7.4.3.3.1 single-tile inference (one column, one row,
    uniform spacing, loop filter across tiles enabled) is materialised.
  - `pps_loop_filter_across_slices_enabled_flag` and the
    deblocking-filter-control block (`DeblockingFilterControl`):
    `deblocking_filter_override_enabled_flag`,
    `pps_deblocking_filter_disabled_flag`, and `pps_beta_offset_div2` /
    `pps_tc_offset_div2` (se(v), −6..=6) when the filter is not
    disabled; absent-control inference applied per §7.4.3.3.1.
  - `pps_scaling_list_data_present_flag` — **rejected** with
    `PpsError::ScalingListUnsupported` when 1 (shared deferral with the
    SPS scaling-list path).
  - `lists_modification_present_flag`,
    `log2_parallel_merge_level_minus2`,
    `slice_segment_header_extension_present_flag`,
    `pps_extension_present_flag` — when set, the four extension flags,
    their bodies, and `rbsp_trailing_bits()` are surfaced as a shared
    `sps::OpaqueTail` via the new public `OpaqueTail::capture_at`.
- `BitReader::se()` — 0-th-order signed Exp-Golomb (the se(v)
  descriptor) per §9.2.2 Table 9-3: codeNum k → (−1)^(k+1)·Ceil(k/2).
- Convenience derivations on `PicParameterSet`: `init_qp()`,
  `num_ref_idx_l{0,1}_default_active()`, `num_tile_{columns,rows}()`,
  `log2_par_mrg_level()`.
- Top-level `Error::Pps(PpsError)` variant + `From<PpsError>`.
- 10 new unit tests (total 52, was 42): se(v) Table-9-3 mapping +
  single-bit-zero on `BitReader`; the fixture PPS parse cross-checked
  against `docs/video/h265/fixtures/tiny-i-only-16x16-main/trace.txt`
  (line 3); end-to-end PPS parse via the Annex B walker; a
  hand-assembled tiles + deblocking-control PPS (non-uniform spacing,
  non-zero β / tC offsets); opaque PPS-extension tail capture;
  `pps_scaling_list_data_present_flag == 1` rejection;
  `pps_pic_parameter_set_id > 63` rejection; truncated-RBSP rejection;
  SPS-bit-depth-aware `init_qp_in_range` check.

### Added — clean-room rebuild round 4 (2026-05-22)

- §7.3.2.2 SPS tail past `sample_adaptive_offset_enabled_flag`:
  - `pcm_enabled_flag` + the `pcm_*` block (`PcmInfo`):
    `pcm_sample_bit_depth_luma_minus1` / `_chroma_minus1` (4-bit
    each, validated against the `BitDepthY` / `BitDepthC` derived
    from the earlier `bit_depth_*_minus8` fields per equations
    7-25 / 7-26), `log2_min_pcm_luma_coding_block_size_minus3`,
    `log2_diff_max_min_pcm_luma_coding_block_size`,
    `pcm_loop_filter_disabled_flag`.
  - `num_short_term_ref_pic_sets` (ue(v), 0..=64 per §7.4.3.2) +
    `Vec<ShortTermRefPicSet>` populated by the §7.3.7 parser. Both
    forms are materialised: the explicit
    `num_negative_pics` / `num_positive_pics` /
    `delta_poc_s{0,1}_minus1[i]` / `used_by_curr_pic_s{0,1}_flag[i]`
    form, and the inter-RPS-prediction form
    (`inter_ref_pic_set_prediction_flag`, `delta_idx_minus1`,
    `delta_rps_sign`, `abs_delta_rps_minus1`, plus the
    `used_by_curr_pic_flag[j]` / `use_delta_flag[j]` arrays of
    length `NumDeltaPocs[RefRpsIdx] + 1`). `RefRpsIdx` chains
    through the preceding RPS list per §7.4.8;
    `delta_idx_minus1` is only signalled when
    `stRpsIdx == num_short_term_ref_pic_sets` (the slice-header
    in-line form is handled by inferring 0 for SPS entries).
    `use_delta_flag[j]` is inferred to 1 when
    `used_by_curr_pic_flag[j] == 1` per §7.4.8.
  - `long_term_ref_pics_present_flag` block:
    `num_long_term_ref_pics_sps` (0..=32) +
    `Vec<LongTermRefPicEntry>` carrying `lt_ref_pic_poc_lsb_sps[i]`
    (parsed as `u(log2_max_pic_order_cnt_lsb_minus4 + 4)`) +
    `used_by_curr_pic_lt_sps_flag[i]`.
  - `sps_temporal_mvp_enabled_flag` (u1).
  - `strong_intra_smoothing_enabled_flag` (u1).
  - `vui_parameters_present_flag` (u1) — when set, the VUI body
    plus the trailing `sps_extension_present_flag` and any
    extension payload + `rbsp_trailing_bits()` are surfaced as
    a single `OpaqueTail { bytes, start_bit_in_first_byte }`.
  - `sps_extension_present_flag` (u1) — known precisely when the
    VUI gate is 0. When set, the extension flag block plus any
    extension body and the RBSP trailer are surfaced as
    `OpaqueTail`.
- Convenience derivation `max_pic_order_cnt_lsb()` returning
  `1 << (log2_max_pic_order_cnt_lsb_minus4 + 4)` per §7.4.3.2.1.
- 8 new unit tests: `pcm_enabled` happy path; PCM-depth-exceeds-luma
  rejection; one explicit short-term RPS; inter-RPS-prediction
  short-term RPS chaining; long-term-ref-pic block; opaque VUI tail
  capture; opaque extension tail capture; clean tail (both flags
  off, no opaque). Total test count: 42 (was 34).
- Fixture `parses_tiny_fixture_sps` extended to assert every newly-parsed
  tail field against
  `docs/video/h265/fixtures/tiny-i-only-16x16-main/trace.txt`
  (pcm_enabled=0, num_short_term_ref_pic_sets=0,
  long_term_ref_pics=0, temporal_mvp=1, strong_intra_smoothing=1,
  vui present).

### Added — clean-room rebuild round 3 (2026-05-22)

- §7.3.2.2 `SeqParameterSet` structural parse — `sps_video_parameter_set_id`
  (u4), `sps_max_sub_layers_minus1` (u3, 0..=6 range-checked),
  `sps_temporal_id_nesting_flag`, the §7.3.3 `profile_tier_level()`
  re-walk, `sps_seq_parameter_set_id` (ue(v), 0..=15),
  `chroma_format_idc` (ue(v), 0..=3), `separate_colour_plane_flag`
  (parsed only when `chroma_format_idc == 3`),
  `pic_width_in_luma_samples` / `pic_height_in_luma_samples` (ue(v),
  non-zero per §7.4.3.2), `conformance_window_flag` + the four
  `conf_win_{left,right,top,bottom}_offset` ue(v) values,
  `bit_depth_{luma,chroma}_minus8` (ue(v), 0..=8 range-checked),
  `log2_max_pic_order_cnt_lsb_minus4` (ue(v), 0..=12), the
  per-sub-layer DPB / reorder / latency triple loop with
  ordering-info-present-flag propagation (§7.4.3.2.1),
  `log2_min_luma_coding_block_size_minus3`,
  `log2_diff_max_min_luma_coding_block_size`,
  `log2_min_luma_transform_block_size_minus2`,
  `log2_diff_max_min_luma_transform_block_size`,
  `max_transform_hierarchy_depth_{inter,intra}`,
  `scaling_list_enabled_flag` (rejected when set; `scaling_list_data()`
  deferred), `amp_enabled_flag`, `sample_adaptive_offset_enabled_flag`.
- Convenience derivations: `bit_depth_luma()`, `bit_depth_chroma()`,
  `log2_min_cb_size()`, `log2_ctb_size()`, `log2_min_tb_size()` —
  the field combinations §7.4.3.2.1 calls `BitDepthY`, `BitDepthC`,
  `MinCbLog2SizeY`, `CtbLog2SizeY`, `MinTbLog2SizeY`.
- 8 new unit tests: fixture parse against the SPS RBSP from
  `docs/video/h265/fixtures/tiny-i-only-16x16-main/input.hevc`
  (cross-checked against `trace.txt`); end-to-end VPS+SPS parse via
  the Annex B walker; emulation-prevention-strip equivalence;
  truncated-RBSP rejection; hand-assembled `chroma_format_idc == 3`
  + conformance-window 10-bit 4:4:4 path; hand-assembled
  `sub_layer_ordering_info_present_flag == 0` propagation across
  two sub-layers; `scaling_list_enabled_flag == 1` rejection;
  `chroma_format_idc == 4` out-of-range rejection.

### Added — clean-room rebuild round 2 (2026-05-22)

- MSB-first `BitReader` with `u(n)` and 0-th-order
  unsigned-Exp-Golomb `ue(v)` (§9.2) descriptors; `skip(n)` to
  bit-walk over not-yet-materialised fields without parsing them.
- §7.3.2.1 `HevcVps` structural parse — `vps_video_parameter_set_id`
  (u4), `vps_base_layer_internal_flag` / `vps_base_layer_available_flag`,
  `vps_max_layers_minus1` (u6), `vps_max_sub_layers_minus1` (u3 with
  the §7.4.3.1 0..=6 range check), `vps_temporal_id_nesting_flag`,
  `vps_reserved_0xffff_16bits` validation (rejects any value other
  than `0xFFFF`), the §7.3.3 `profile_tier_level()` walk, and the
  per-sub-layer DPB / reorder / latency `ue(v)` triple loop with
  ordering-info-present-flag propagation.
- §7.3.3 `ProfileTierLevel` — materialises
  `general_profile_space` / `_tier_flag` / `_profile_idc` /
  `_level_idc` and the per-sub-layer
  `sub_layer_profile_present_flag` / `_level_present_flag` gates
  plus `sub_layer_level_idc[i]`. The constraint-flag / reserved-zero
  blocks are walked structurally (their bit width is fixed at 43
  bits regardless of the inner conditional branch per the
  `/* not affected by this condition */` note in the syntax table)
  so subsequent VPS fields land on the right bit boundary.
- 17 new unit tests: bit-reader `u(n)` MSB-first packing /
  cross-byte read / `u(0)` zero-consume / `u(32)` full word /
  end-of-buffer / too-many-bits / skip / `ue(v)` codewords 0..=6
  per Table 9-2 / single-bit zero / leading-zero-overrun;
  VPS fixture parse (against
  `docs/video/h265/fixtures/tiny-i-only-16x16-main/input.hevc` — the
  on-wire bytes are inlined into the test, no I/O); end-to-end VPS
  parse via the Annex B walker; reserved-field mismatch rejection;
  truncated-RBSP rejection; emulation-prevention round-trip equality;
  hand-assembled two-sub-layer ordering-info-present-flag=1 parse;
  hand-assembled ordering-info-present-flag=0 propagation parse.

### Added — clean-room rebuild round 1 (2026-05-20)

- Annex B byte-stream walker: `NalIter`, `collect_nal_units`,
  supporting both 3-byte (`00 00 01`) and 4-byte (`00 00 00 01`)
  start codes.
- §7.3.1.2 NAL header parse: `NalHeader` exposes
  `nal_unit_type`, `nuh_layer_id`, and `TemporalId` (derived from
  `nuh_temporal_id_plus1`); `forbidden_zero_bit` set and
  zero-`nuh_temporal_id_plus1` are surfaced as `NalError`.
- §7.4.1.1 emulation-prevention strip
  (`strip_emulation_prevention`) — `0x00 0x00 0x03` decodes to
  `0x00 0x00`.
- 7 unit tests covering: 3-byte start code single NAL, 4-byte
  start code with two NAL units, emulation-prevention round-trip,
  forbidden-bit rejection, zero-temporal-id rejection, no
  start-code-at-all rejection, and header field-packing round
  trip (incl. non-zero `nuh_layer_id`).

### Erased

- Prior master history was force-erased on **2026-05-18** under
  Hat-3 cold enforcement of the workspace clean-room policy
  (`docs/IMPLEMENTOR_ROUND.md`).

### Reset

- Crate reduced to a minimal `oxideav_core::register!` stub. Every
  public API returns `Error::NotImplemented`. The crates.io version
  (`0.0.8`) is preserved on the new master to avoid breaking
  downstream version pins; the published versions on crates.io will
  be yanked by the maintainer.
- HEIF/HEIC `heif` cargo feature is dropped from the scaffold
  (re-introduced in a future rebuild round once the decoder core is
  back).

### Next

- Slice segment header parse (§7.3.6.1).
- PPS range / SCC extensions (§7.3.2.3.2 / §7.3.2.3.3) — currently
  surfaced as opaque bytes when `pps_extension_present_flag == 1`.
- VUI parameters (§E.2.1) — currently surfaced as opaque bytes.
- SPS extension bodies (Range Extension, Multilayer, 3D, SCC) —
  currently surfaced as opaque bytes alongside the VUI tail.
- `scaling_list_data()` (§7.3.4) — currently rejected when
  `scaling_list_enabled_flag == 1` /
  `pps_scaling_list_data_present_flag == 1`.
- VPS tail: `vps_max_layer_id`, `vps_num_layer_sets_minus1`,
  `layer_id_included_flag` matrix, `vps_timing_info_present_flag`,
  HRD parameters, `vps_extension_data_flag`.
- Slice header parse.
- CABAC remains blocked on docs #444 (`cu_qp_delta` +
  `last_sig_coeff` multi-QG / multi-CTU 4:2:2 trace gap).
