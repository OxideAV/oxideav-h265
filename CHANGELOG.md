# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — clean-room rebuild round 15 (2026-05-26)

- §7.3.6.2 `ref_pic_lists_modification()` syntax structure decoded by
  a new standalone parser ([`slice::RefPicListsModification::parse`]):
  - `ref_pic_list_modification_flag_l0` `u(1)` gate +
    `list_entry_l0[ 0 .. num_ref_idx_l0_active_minus1 ]` `u(v)` loop
    (each entry `Ceil( Log2( NumPicTotalCurr ) )` bits wide per
    §7.4.7.2) with each value range-checked to
    `0 ..= NumPicTotalCurr - 1`.
  - B-slice-gated `ref_pic_list_modification_flag_l1` `u(1)` +
    matching `list_entry_l1[ 0 .. num_ref_idx_l1_active_minus1 ]`
    `u(v)` loop (same width / range check).
  - Up-front precondition checks rejecting `SliceType::I` calls (the
    §7.3.6.1 gate sits inside the inter-slice branch),
    `NumPicTotalCurr <= 1` (the §7.3.6.1 gate guarantees `> 1` at the
    in-place call site), and `num_ref_idx_lX_active_minus1 > 14` (the
    §7.4.7.1 cap on the per-list active count).
- `slice::RefPicListsModification` re-exported from the crate root
  for downstream callers.
- `slice::SliceSegmentHeader::parse` still defers the in-place
  call: the §7.3.6.1 invocation `if( lists_modification_present_flag
  && NumPicTotalCurr > 1 ) ref_pic_lists_modification( )` is gated by
  the DPB-derived `NumPicTotalCurr` (§7.4.7.2), which is the next
  round's primitive. The standalone parser unblocks that round.
- 12 spec-pinned unit tests covering: P-slice L0-only and L0-implicit
  cases, B-slice both-lists and L0-implicit-L1-explicit and
  both-flags-zero cases, the `list_entry_lX > NumPicTotalCurr - 1`
  range checks for both L0 and L1, the `SliceType::I`,
  `NumPicTotalCurr <= 1`, and `num_ref_idx_lX_active_minus1 > 14`
  precondition rejections, the maximum-active-index (15-entry) case,
  a `Ceil( Log2( N ) )` per-entry width check across
  `N in { 2, 3, 4, 5, 8, 9, 16 }`, and a truncated-RBSP path.

### Added — clean-room rebuild round 14 (2026-05-25)

- §E.2.1 `vui_parameters()` body decoded as a new `vui` module
  (`vui::VuiParameters`), replacing the opaque SPS VUI tail:
  - aspect-ratio info (`aspect_ratio_idc` `u(8)` + the `EXTENDED_SAR`
    `sar_width` / `sar_height` `u(16)` pair), overscan,
    `video_signal_type` (`video_format` / `video_full_range_flag` +
    the `ColourDescription` `colour_primaries` /
    `transfer_characteristics` / `matrix_coeffs` triple), chroma-loc
    info (`chroma_sample_loc_type_{top,bottom}_field` 0..=5
    range-checked), the neutral-chroma / field-seq / frame-field
    flags, the `DefaultDisplayWindow` offset quad.
  - `VuiTimingInfo`: `u(32)` `vui_num_units_in_tick` /
    `vui_time_scale` enforced `> 0` per §E.3.1, the POC-proportional
    flag + `vui_num_ticks_poc_diff_one_minus1`, and the nested §E.2.3
    `hrd_parameters( 1, sps_max_sub_layers_minus1 )` call reusing the
    existing `HrdParameters` parser.
  - `BitstreamRestriction` with the §E.3.1
    `min_spatial_segmentation_idc` 0..=4095, `max_bytes_per_pic_denom`
    / `max_bits_per_min_cu_denom` 0..=16, and
    `log2_max_mv_length_{horizontal,vertical}` 0..=15 range-checks.
- SPS RBSP parse now decodes the VUI body inline rather than
  capturing it as the opaque tail:
  - `SeqParameterSet::vui_parameters: Option<VuiParameters>` populated
    when `vui_parameters_present_flag == 1`; parsing then continues to
    `sps_extension_present_flag` in both paths.
  - `SeqParameterSet::opaque_tail` now populated only for the
    `sps_extension_present_flag == 1` extension payload +
    `rbsp_trailing_bits()` suffix.
  - `SpsError::Vui` variant added to surface `VuiError` failures while
    preserving the single-pattern `Truncated` handler.
- Public re-exports: `BitstreamRestriction`, `ColourDescription`,
  `DefaultDisplayWindow`, `VideoSignalType`, `VuiError`,
  `VuiParameters`, `VuiTimingInfo`, `EXTENDED_SAR` added to the crate
  root; `Error::Vui` variant added.
- Tests: 18 new `vui` per-field tests plus the SPS-level
  `decodes_vui_then_continues_to_extension_flag` /
  `decodes_vui_then_captures_extension_tail`; the tiny libx265
  fixture test now asserts the decoded VUI (1:1 SAR, video_format 5,
  1/25 timing) instead of an opaque tail. Test count 130 → 146.

### Added — clean-room rebuild round 13 (2026-05-25)

- §E.2.2 / §E.2.3 `hrd_parameters()` and `sub_layer_hrd_parameters()`
  bodies as a new `hrd` module:
  - `hrd::HrdParameters` decoded with full common-info gating
    (`nal_hrd_parameters_present_flag`,
    `vcl_hrd_parameters_present_flag`,
    `sub_pic_hrd_params_present_flag`, the conditional `u(8)` /
    `u(5)` / `u(4)` / `u(5)` length / scale block from §E.2.2),
    `commonInfPresentFlag = 0` inheritance from a previous entry's
    `HrdCommonInfo`, and the per-sub-layer loop
    (`fixed_pic_rate_general_flag[i]`,
    `fixed_pic_rate_within_cvs_flag[i]` with the §E.3.2 "general == 1
    ⇒ within_cvs := 1" inference, `elemental_duration_in_tc_minus1[i]`
    `ue(v)` range-checked at 0..=2047,
    `low_delay_hrd_flag[i]`, and `cpb_cnt_minus1[i]` `ue(v)`
    range-checked at 0..=31 with the §E.3.2 "inferred to 0" path when
    `low_delay_hrd_flag[i] == 1`).
  - `hrd::SubLayerHrdParameters` decoded per §E.2.3 with the
    monotonic-progression constraints from §E.3.3 enforced inline
    (`bit_rate_value_minus1[i]` strictly increasing,
    `cpb_size_value_minus1[i]` monotonic non-increasing, and the
    sub-pic `bit_rate_du_value_minus1[i]` /
    `cpb_size_du_value_minus1[i]` variants gated on
    `sub_pic_hrd_params_present_flag`).
  - `hrd::VpsHrdEntry` wraps each entry of the §7.3.2.1 VPS HRD loop
    (`hrd_layer_set_idx[i]` `ue(v)` with the
    `vps_num_layer_sets_minus1 + 1` ceiling, the per-index
    `cprms_present_flag[i]` `u(1)` for `i > 0` with the implicit `1`
    inference for `i == 0`, and the body itself).
- VPS RBSP parse now decodes the per-HRD bodies inline rather than
  capturing them as the opaque tail:
  - `HevcVps::hrd_parameters: Vec<VpsHrdEntry>` populated when
    `vps_timing_info_present_flag == 1` and `vps_num_hrd_parameters >
    0`, with `cprms_present_flag[i] == 0` inheritance walked through
    the previously-parsed entry's `HrdCommonInfo`.
  - `HevcVps::vps_extension_flag` is now an unconditional `bool` (was
    `Option<bool>`); the parser always reads it after the HRD loop
    completes.
  - `HevcVps::opaque_tail` now populated only for the
    `vps_extension_flag == 1` extension-data + `rbsp_trailing_bits()`
    suffix (the per-HRD-body deferral is gone).
  - `VpsError::Hrd` variant added to surface `HrdError` failures
    inside the VPS HRD loop while preserving the single-pattern
    `Truncated` handler.
- Public re-exports: `CpbEntry`, `HrdCommonInfo`, `HrdError`,
  `HrdParameters`, `SubLayerHrd`, `SubLayerHrdParameters`,
  `VpsHrdEntry`, `HEVC_MAX_CPB_CNT`,
  `HEVC_MAX_ELEMENTAL_DURATION_IN_TC_MINUS1` added to the crate root;
  `Error::Hrd` variant added.
- Tests: 12 new tests
  (`hrd::parses_minimal_common_info_one_sub_layer`,
  `hrd::parses_nal_hrd_with_sub_pic_and_two_cpbs`,
  `hrd::rejects_non_increasing_bit_rate_value`,
  `hrd::rejects_elemental_duration_above_2047`,
  `hrd::rejects_cpb_cnt_above_31`,
  `hrd::low_delay_infers_cpb_cnt_zero`,
  `hrd::cprms_zero_inherits_previous_common_info`,
  `hrd::vps_hrd_entry_skips_cprms_for_index_zero`,
  `hrd::vps_hrd_entry_reads_cprms_for_nonzero_index`,
  `hrd::cprms_zero_without_previous_yields_no_hrd_bodies`,
  `hrd::parses_three_sub_layers`,
  `vps::parses_two_hrd_entries_with_cprms_inheritance`); the round-12
  `captures_hrd_payload_as_opaque_tail` was repurposed into
  `parses_hrd_payload_inline` and now asserts the per-HRD body is
  decoded rather than captured. Test count 118 → 130.

### Added — clean-room rebuild round 12 (2026-05-25)

- §7.3.2.1 VPS tail through the optional VPS timing-info block:
  - `vps_max_layer_id` (`u(6)`) and `vps_num_layer_sets_minus1`
    (`ue(v)`, range 0..=1023, capped at
    `HEVC_VPS_MAX_NUM_LAYER_SETS = 1024` for allocation safety) added
    to `HevcVps`.
  - `layer_id_included_flag[i][j]` inclusion matrix decoded as one
    `LayerIdInclusionRow` per signalled layer set (the spec's
    `i = 1..=vps_num_layer_sets_minus1` loop; layer set 0 is implicit
    per §7.4.3.1, so the matrix has `num_layer_sets_minus1` rows of
    `max_layer_id + 1` flags each).
  - `vps_timing_info_present_flag` block surfaced as
    `Option<VpsTimingInfo>`: `vps_num_units_in_tick` /
    `vps_time_scale` (`u(32)` both, with the §E.2.1 / §7.3.2.1 "shall
    be greater than 0" semantics enforced as
    `VpsError::ValueOutOfRange`), `vps_poc_proportional_to_timing_flag`
    + the optional `vps_num_ticks_poc_diff_one_minus1` `ue(v)`, and
    the `vps_num_hrd_parameters` `ue(v)` count (bounded at
    `vps_num_layer_sets_minus1 + 1` per §7.4.3.1).
  - `vps_extension_flag` decoded into `Option<bool>` (None when the
    parser stopped before reading it because
    `vps_num_hrd_parameters > 0` deferred the rest of the RBSP to the
    opaque tail).
  - `HevcVps::opaque_tail: Option<OpaqueTail>` populated when the
    parser defers HRD bodies (`num_hrd_parameters > 0`) or extension
    data (`vps_extension_flag == 1`); the opaque tail reuses
    `sps::OpaqueTail::capture_at(bit_pos, rbsp)` so the surface
    matches the SPS / PPS opaque-tail convention.
- Public re-exports: `LayerIdInclusionRow`, `VpsTimingInfo`,
  `HEVC_VPS_MAX_NUM_LAYERS`, `HEVC_VPS_MAX_NUM_LAYER_SETS` added to
  the crate root.
- Tests: three new VPS tail tests
  (`parses_layer_set_matrix_and_timing_info`,
  `captures_hrd_payload_as_opaque_tail`,
  `rejects_zero_num_units_in_tick`); two existing handwritten VPS
  tests extended to feed the now-required tail bits. Test count
  115 → 118.

### Added — clean-room rebuild round 11 (2026-05-24)

- §9.3 CABAC arithmetic decoding engine as a new standalone module
  (`cabac`):
  - §9.3.2.6 engine-register initialization: `CabacEngine::new`
    consumes a `BitReader` positioned at the first bit of
    `slice_segment_data()`, sets `ivlCurrRange = 510`, and reads the
    9-bit initial `ivlOffset` — enforcing the spec's "the bitstream
    shall not contain data that result in a value of ivlOffset being
    equal to 510 or 511" constraint as `CabacError::InvalidInitOffset`.
    `CabacEngine::init_engine` re-initializes the registers in place
    (the `pcm_flag == 1` re-init path).
  - §9.3.2.2 context-variable initialization: `ContextModel::init`
    evaluates equations 9-4..9-6 — `slopeIdx` / `offsetIdx`,
    `m = slopeIdx * 5 − 45`, `n = ( offsetIdx << 3 ) − 16`, then
    `preCtxState = Clip3( 1, 126, ( ( m * Clip3( 0, 51, SliceQpY ) ) >> 4 ) + n )`,
    with `valMps` / `pStateIdx` split. The §9.3.2.2 `initType`
    selector (equation 9-7) is exposed as the free function
    `init_type(slice_type, cabac_init_flag)`. `ContextModel::terminate_state`
    yields the §9.3.2.2 NOTE 2 non-adapting `(pStateIdx = 63,
    valMps = 0)` state.
  - §9.3.4.3.2 `DecodeDecision`: `CabacEngine::decode_decision`
    derives `qRangeIdx = ( ivlCurrRange >> 6 ) & 3`, looks up
    `ivlLpsRange` in the Table 9-52 `rangeTabLps[64][4]`, performs
    the LPS / MPS branch on `ivlOffset`, applies the §9.3.4.3.2.2
    state transition (Table 9-53 `transIdxLps` / `transIdxMps`, with
    the `pStateIdx == 0` LPS path flipping `valMps`), and invokes
    `RenormD`. Mutates the supplied `ContextModel` in place.
  - §9.3.4.3.3 `RenormD` renormalization loop, internal to the
    engine: while `ivlCurrRange < 256`, double the range and shift
    one fresh `read_bits(1)` into `ivlOffset`.
  - §9.3.4.3.4 `DecodeBypass`: `CabacEngine::decode_bypass` shifts a
    fresh bit into `ivlOffset` and compares it to `ivlCurrRange`,
    returning the equal-probability bin. `decode_bypass_bits(n)` is a
    convenience wrapper that accumulates `n` bypass bins MSB-first
    into a `u32` (the common fixed-length bypass pattern).
  - §9.3.4.3.5 `DecodeTerminate`: `CabacEngine::decode_terminate`
    decrements `ivlCurrRange` by 2, returns 1 if `ivlOffset >=
    ivlCurrRange` (no renormalization — decoding is terminated) and
    otherwise returns 0 with renormalization. This is the
    `end_of_slice_segment_flag` / `end_of_subset_one_bit` /
    `pcm_flag` decision (ctxTable = 0, ctxIdx = 0).
  - §9.3.4.3.6 alignment process prior to aligned bypass decoding:
    `CabacEngine::align` sets `ivlCurrRange = 256` (the
    pre-`coeff_abs_level_remaining[ ]` / `coeff_sign_flag[ ]` hook);
    `ivlOffset` and the bit reader are untouched.
- 20 new `cabac` unit tests: equation 9-7 truth table; equations
  9-4..9-6 worked examples at boundary `initValue` / `SliceQpY`
  combinations (negative-slope path, high `initValue`, sub-zero QP
  clipping); §9.3.2.2 NOTE 2 terminate-state values; Table 9-52
  corner / monotonicity checks; Table 9-53 transition bounds + LPS /
  MPS monotonicity; §9.3.2.6 engine-init bit consumption and the
  forbidden 510 / 511 rejection; bypass MSB-first bit accumulation
  and the `offset >= range` path; terminate one / zero / no-renorm
  paths; alignment register set; `DecodeDecision` MPS-no-renorm and
  LPS-with-renorm paths (including the `pStateIdx == 0` MPS flip);
  an all-zero-stream MPS-state-walk integration check; and an
  end-of-buffer surfacing test.

### Added — clean-room rebuild round 10 (2026-05-24)

- The remaining three §6.5 scan-order initialization processes, joining
  round 9's §6.5.3 up-right diagonal scan in the `scan` module:
  - §6.5.4 horizontal scan ([`horizontal`], equation 6-12) — a plain
    raster walk, `scanIdx == 1`.
  - §6.5.5 vertical scan ([`vertical`], equation 6-13) — the transpose
    of the horizontal scan (column by column), `scanIdx == 2`.
  - §6.5.6 traverse scan ([`traverse`], equation 6-14) — a
    boustrophedon (serpentine) raster, even rows left-to-right and odd
    rows right-to-left, `scanIdx == 3`.
- The §7.4.2 `ScanOrder[log2BlockSize][scanIdx]` accessor
  ([`scan_order`] / [`ScanIdx`] / [`ScanOrderError`]): dispatches to the
  §6.5.3..§6.5.6 process for the requested block size and scan index,
  enforcing the table's populated ranges — `log2BlockSize` 0..=3 for the
  diagonal / horizontal / vertical scans, 2..=5 for the traverse scan.
  This is the table the residual-coding path (§7.3.8.11 / §9.3.4.2.4)
  selects per transform block.
- 13 new byte-exact `scan` tests: hand-derived 4x4 / 2x2 expected
  coordinate vectors for each new scan, permutation / transpose /
  odd-row-reversal invariants across the populated block sizes, the
  `scan_order` dispatch-vs-builder equivalence, and the
  out-of-range rejection per §7.4.2.

### Added — clean-room rebuild round 9 (2026-05-24)

- §6.5.3 up-right diagonal scan order (equation 6-11), in a new `scan`
  module ([`up_right_diagonal`] / [`ScanPos`]): a direct transcription
  of the 6-11 pseudocode, returning `diagScan[ sPos ]` for a
  `blkSize`x`blkSize` block. This is the `ScanOrder[log2BlockSize][0]`
  entry the §7.4.5 `ScalingFactor` derivation reads (4x4 and 8x8
  blocks). The §6.5.4..§6.5.6 horizontal / vertical / traverse scans
  are deferred to the residual-coding path.
- §7.4.5 `ScalingFactor[sizeId][matrixId][x][y]` 2-D
  quantization-matrix derivation (equations 7-44..7-51), via the new
  [`ScalingListData::scaling_factors`] /
  [`ScalingFactors`] / [`ScalingFactorMatrix`]:
  - 4x4 (equation 7-44) and 8x8 (7-45): each flat
    `ScalingList[sizeId][matrixId][i]` coefficient is placed at the
    `(ScanOrder[·][0][i][0], ScanOrder[·][0][i][1])` cell — `ScanOrder[2][0]`
    (4x4 block, 16 positions) for `sizeId == 0`, `ScanOrder[3][0]`
    (8x8 block, 64 positions) for `sizeId == 1`.
  - 16x16 (7-46): the 8x8-scan placement with each entry replicated
    into a 2x2 block (`x * 2 + k`, `y * 2 + j`), then the DC
    coefficient overrides `[0][0]` (7-47).
  - 32x32 (7-48): the 8x8-scan placement with each entry replicated
    into a 4x4 block (`x * 4 + k`, `y * 4 + j`) for `matrixId` 0 (intra
    Y) and 3 (inter Y) — the only slots the `matrixId += 3` step
    signals — then the DC override (7-49).
  - 32x32 chroma (7-50 / 7-51): when `ChromaArrayType == 3` (4:4:4),
    `matrixId` 1, 2, 4, 5 are derived from the 16x16 (`sizeId == 2`)
    lists of the same `matrixId`, 4x4-replicated, with the sizeId-2 DC
    override. For other chroma formats those matrices are left all-zero
    (they are not used).
  - `ScalingFactorMatrix` is stored row-major (`coef[y * dim + x]`)
    with a `dim` side length (4 / 8 / 16 / 32) and an `at(x, y)`
    accessor.
- 9 new unit tests (total 84, was 75): the §6.5.3 scan for 4x4 / 2x2
  blocks (hand-derived coordinate lists), the permutation invariant for
  the 4x4 / 8x8 blocks, and the 8x8 diagonal-ordering invariant; the
  4x4 all-16 `ScalingFactor`, the 8x8-intra diagonal-scan placement
  against Table 7-6, the 16x16 2x2 replication + isolated DC override,
  the 32x32 4x4 replication with the chroma matrices all-zero for
  non-4:4:4, and the 32x32-chroma derivation for `ChromaArrayType == 3`.

### Added — clean-room rebuild round 8 (2026-05-24)

- §7.3.4 `scaling_list_data()` parse + §7.4.5
  `ScalingList[sizeId][matrixId][i]` derivation, in a new
  `scaling_list` module ([`ScalingListData`]):
  - For each of the 24 (`sizeId`, `matrixId`) slots,
    `scaling_list_pred_mode_flag` (`u(1)`) selects between a predicted
    list and an explicit list.
  - Predicted: `scaling_list_pred_matrix_id_delta` (`ue(v)`) with the
    §7.4.5 range check (`matrixId` for `sizeId <= 2`, `matrixId / 3`
    for `sizeId == 3`). Delta 0 infers the §7.4.5 default list;
    otherwise `refMatrixId = matrixId − delta * (sizeId == 3 ? 3 : 1)`
    (equation 7-42) and the reference list (with its DC coefficient) is
    copied (equation 7-43).
  - Explicit: the running `nextCoef` accumulator, seeded at 8 and
    updated as `(nextCoef + scaling_list_delta_coef + 256) % 256`
    (§7.3.4), with `scaling_list_dc_coef_minus8` (`se(v)`) read first
    for `sizeId > 1` (range −7..=247) supplying the DC coefficient.
  - The default 4x4 / 8x8 intra/inter tables (Tables 7-5 / 7-6) are
    transcribed; `coefNum = Min(64, 1 << (4 + (sizeId << 1)))`.
  - Conformance checks: `scaling_list_pred_matrix_id_delta` bound,
    `scaling_list_dc_coef_minus8 ∈ [−7, 247]`, and derived coefficient
    `> 0`, each surfaced through [`ScalingListError`].
- The block is wired into both the SPS
  (`sps_scaling_list_data_present_flag`, nested under
  `scaling_list_enabled_flag`) and the PPS
  (`pps_scaling_list_data_present_flag`) parse paths, replacing the
  previous outright refusals (`SpsError::ScalingListUnsupported` /
  `PpsError::ScalingListUnsupported` removed in favour of
  `SpsError::ScalingList` / `PpsError::ScalingList`). When
  `scaling_list_enabled_flag == 1` but
  `sps_scaling_list_data_present_flag == 0`, the SPS now parses (the
  §7.4.5 default lists apply) — previously it was rejected.
- `SeqParameterSet` gains `sps_scaling_list_data_present_flag` and
  `scaling_list_data`; `PicParameterSet` gains `scaling_list_data`.
- 10 new unit tests (total 75, was 65): all-default lists matching
  Tables 7-5 / 7-6; `coefNum` per `sizeId`; explicit flat 4x4 list;
  explicit 16x16 list with a DC coefficient; prediction copying a
  reference list; and rejections for out-of-range
  `scaling_list_pred_matrix_id_delta`, non-positive coefficient,
  out-of-range DC coefficient, and truncation — plus the SPS
  default-list / explicit-list and PPS explicit-list integration
  tests. The previous `rejects_scaling_list_enabled` SPS test and
  `rejects_scaling_list_present` PPS test were rewritten in place (not
  ignored) to assert the new parse-through behaviour.

### Added — clean-room rebuild round 7 (2026-05-24)

- §7.3.6.1 non-IDR POC + reference-picture-set block — the
  `slice_segment_header()` sub-block gated by
  `nal_unit_type != IDR_W_RADL && nal_unit_type != IDR_N_LP`, closing
  the opaque tail previously surfaced for non-IDR I-slice segments:
  - `slice_pic_order_cnt_lsb` (`u(v)`, width
    `log2_max_pic_order_cnt_lsb_minus4 + 4`, range
    0..=`MaxPicOrderCntLsb − 1`).
  - `short_term_ref_pic_set_sps_flag` (`u(1)`), with the §7.4.7.1
    constraint that the value shall be 0 when
    `num_short_term_ref_pic_sets == 0`.
  - In-line `st_ref_pic_set(num_short_term_ref_pic_sets)` (§7.3.7)
    when `short_term_ref_pic_set_sps_flag == 0`, exposed via a new
    `ShortTermRefPicSet::parse_slice_inline(&mut BitReader, &SeqParameterSet)`
    public entry point that wraps the existing per-set parser with the
    SPS context (`stRpsIdx == num_short_term_ref_pic_sets`, `all_rps =
    sps.short_term_ref_pic_sets`).
  - `short_term_ref_pic_set_idx` (`u(v)`, width
    `Ceil(Log2(num_short_term_ref_pic_sets))`) when
    `short_term_ref_pic_set_sps_flag == 1 && num_short_term_ref_pic_sets > 1`
    (inferred to 0 otherwise).
  - The long-term-ref-pic block gated by
    `sps.long_term_ref_pics_present_flag`: `num_long_term_sps`
    (`ue(v)`, bounded by `num_long_term_ref_pics_sps`),
    `num_long_term_pics` (`ue(v)`), and the per-entry
    `lt_idx_sps[i]` (`u(v)`, width
    `Ceil(Log2(num_long_term_ref_pics_sps))`) /
    `poc_lsb_lt[i]` + `used_by_curr_pic_lt_flag[i]` /
    `delta_poc_msb_present_flag[i]` /
    `delta_poc_msb_cycle_lt[i]` (`ue(v)`) loop. The §7.4.7.1
    inferences (`num_long_term_sps == 0`, `delta_poc_msb_cycle_lt ==
    0`) are applied for absent fields, and a defensive 16-entry
    ceiling on `num_long_term_pics` (matching the
    §7.4.3.2.1 DPB-size bound) prevents a pathological encoder from
    forcing an unbounded allocation.
- `SliceLongTermRefPic` + `SliceLongTermRefPicSource` (public) carry
  the per-entry source (SPS-indexed vs in-slice signalling) and the
  delta-POC-MSB cycle.
- `SliceSegmentHeader` gains `slice_pic_order_cnt_lsb`,
  `short_term_ref_pic_set_sps_flag`, `inline_short_term_ref_pic_set`,
  `short_term_ref_pic_set_idx`, `num_long_term_sps`,
  `num_long_term_pics`, and `long_term_ref_pics` fields. The opaque
  tail is now populated *only* for the P/B reference-list /
  weighted-prediction body (round 8 target); independent I-slice
  segments — IDR and non-IDR alike — parse all the way through
  `byte_alignment()`.
- `SliceError::InlineShortTermRpsParse(SpsError)` wraps SPS-layer
  failures from the in-line `st_ref_pic_set` parse, with truncation
  and bit-stream errors flattened back into `SliceError::Truncated` /
  `SliceError::Bitstream` so the public surface stays predictable.
- 4 new unit tests (total 65, was 61): hand-assembled non-IDR I-slice
  CRA header with the in-line zero-pic short-term RPS;
  SPS-resident short-term RPS with a single SPS entry (index inferred);
  SPS-resident with multiple entries (`short_term_ref_pic_set_idx`
  `u(v)` width = `Ceil(Log2(N))`); long-term-ref-pic block with one
  SPS-indexed + one in-slice entry plus a `delta_poc_msb_cycle_lt`;
  rejection of `short_term_ref_pic_set_sps_flag == 1` when
  `num_short_term_ref_pic_sets == 0`. The previously-deferred
  `defers_non_idr_poc_block` test was rewritten in place rather than
  ignored — its old premise (non-IDR slice surfaces an opaque tail) is
  exactly what this round eliminates.

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
