# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — clean-room rebuild round 364 (2026-06-24)

- `motion` §8.5.3.2.3 spatial merging candidates — `NeighbourPu` snapshots
  the `(MvLX, RefIdxLX, PredFlagLX)` motion of a neighbour PU;
  `SpatialMergeNeighbours` carries the five §6.4.2-gated neighbour
  positions (A1, B1, B0, A0, B2); `derive_spatial_merge_candidates`
  implements the eq 8-128..8-142 derivation with the full redundancy
  pruning (B1≠A1, B0≠B1, A0≠A1, B2≠A1/B1, B2 dropped when four already
  available), the `PartitionContext` A1/B1 exclusion for the second
  partition of vertical/horizontal-split `PartMode`s, and the
  `Log2ParMrgLevel` same-region forcing. Output `SpatialMergeCandidates`
  appends in the eq 8-119 order.

- `motion` §8.5.3.2.4 combined bi-predictive candidates —
  `append_combined_bi_candidates` pairs each new candidate's L0 motion and
  L1 motion from existing candidates per Table 8-7, gated on the eq 8-143
  `DiffPicOrderCnt(...) != 0 || mvL0 != mvL1` distinctness test (B slices,
  `2 <= numOrig < MaxNumMergeCand`), stopping at
  `combIdx == numOrig*(numOrig−1)` or a full list.

- `motion` §8.5.3.2.2 merge-list driver — `build_merge_candidate`
  assembles the full `mergeCandList` (steps 5–8: spatial, then temporal
  `Col`, then combined bi-pred, then zero-MV padding to `MaxNumMergeCand`)
  and selects `mergeCandList[ merge_idx ]` (step 9) with the step-10
  `(nOrigPbW + nOrigPbH == 12)` bi→uni-L0 reduction. `MergeListParams`
  carries the per-slice inputs. 12 new unit tests cover pruning, partition
  exclusion, same-region forcing, Table 8-7 pairing, the degenerate
  skip, index selection into the zero padding, and the 8x4/4x8 step-10
  reduction. The temporal `Col` candidate is supplied by the caller
  (`None` until the §8.5.3.2.8 collocated-picture path lands).

- `motion` §8.5.3.2.6 / §8.5.3.2.7 luma motion-vector prediction —
  `derive_mvp_candidate` builds the `mvpListLX` (eq 8-170: A, then B when
  `mvLXA != mvLXB`, then the temporal `Col`, then zero padding to two
  entries) and selects `mvpListLX[ mvp_lX_flag ]`. The §8.5.3.2.7 A
  derivation runs its two passes (eqs 8-171/8-172 same-POC, then
  8-173..8-183 long-term-matched + scaling) over A0/A1; the B derivation
  runs the same-POC pass (eqs 8-184/8-185), the `isScaledFlag == 0`
  step-4 B→A promotion (eq 8-186), and the step-5 long-term/scaling
  re-derivation over B0/B1/B2 (eqs 8-187..8-197). `scale_temporal_mv`
  implements the eq 8-179..8-183 / 8-193..8-197 distance scaling.
  `RefPicId` + `MvpContext` carry the per-list reference picture and the
  POC / long-term / short-term resolvers the picture driver supplies. 8
  new unit tests cover zero padding, same-POC pick, B→A promotion, Col
  insertion / suppression, and the short-term scaling arithmetic. The
  §8.5.3.2.8 temporal predictor `mvLXCol` is passed in (`None` until the
  collocated path lands).

### Added — clean-room rebuild round 360 (2026-06-22)

- `intra_mode_field` §8.4.2 most-probable-mode neighbour state — the new
  `IntraModeField` records each decoded luma prediction block's
  `IntraPredModeY` / `CuPredMode` / `pcm_flag` on the 4×4 luma min-block
  grid and implements the §8.4.2 step-1/step-2 `candIntraPredModeX`
  derivation: out-of-picture / `available == FALSE` / non-`MODE_INTRA` /
  `pcm_flag` / above-CTB-row neighbours all map to `INTRA_DC`, otherwise the
  recorded neighbour mode. 9 unit tests cover every branch.

- `recon` §8.4.2 neighbour-aware intra driver — a per-picture `ReconCtx`
  (the `IntraModeField` + the §6.4.1 `PictureTiling`) is shared across CTUs.
  `reconstruct_cu` now derives each luma PB's `IntraPredModeY` from the
  actual left/above neighbours (most-probable-mode) and records it back,
  for both `PART_2Nx2N` (one PB) and `PART_NxN` (four PBs mapped onto the
  four top-level transform-tree children), replacing the flat-single-CU
  `INTRA_DC` hardcode. `gather_reference_samples` switches to the true
  §6.4.1 z-scan availability (mapping chroma plane coords to luma) instead
  of the raster approximation. `reconstruct_intra_ctu_ctx` is the new
  shared-ctx entry; the single-CTU `reconstruct_intra_ctu` keeps its
  signature. Adds `ReconError::Tiling`. Two tests prove a right CU's
  `mpm_idx == 0` inherits the left neighbour's angular mode through
  `candModeList[0]`, whereas an isolated CU derives `INTRA_PLANAR`.

- `recon` picture-level intra driver — `reconstruct_intra_picture` allocates
  the `Picture`, reconstructs each `PlacedCtu` through the shared
  `ReconCtx`, resolves each CTB's §7.4.9.3 `ResolvedSao` with left/above
  merge, then runs the §8.7.3 `apply_sao_picture` in-loop SAO pass.
  `IntraPictureParams` carries the CTB/min-TB log2 sizes, tile layout,
  slice SAO flags, and SAO offset scales. The real `tiny-i` IDR fixture now
  decodes byte-exact to `expected.yuv` through the full recon + SAO path (a
  new integration test), and a unit test confirms a band-offset CTB shifts
  samples by the resolved offset.

### Added — clean-room rebuild round 356 (2026-06-21)

- `deblock` §8.7.2.1 picture-level deblocking driver — `deblock_picture`
  ties the round's pieces into the whole-picture process: it filters all
  vertical edges first, then all horizontal edges (on the
  vertically-filtered samples), walking a `&[DeblockCuDesc]` in coding
  order. Each CU contributes its `TransformSplit` + `PartMode` +
  `filter_left`/`filter_top` boundary flags; per CU per pass the driver
  derives the §8.7.2.2/.3 edge flags, the §8.7.2.4 bS from the
  `MotionField`, and applies `filter_cu_edges`. The caller owns the
  §8.7.2.1 boundary exclusions (picture/tile/slice edges + the
  `slice_deblocking_filter_disabled_flag` skip). 3 new tests: a two-CU
  vertical-seam smooth (luma + 8-aligned 4:2:0 chroma), a single-CU
  no-internal-edge byte-identical no-op, and a one-level-split cross-seam
  smoothed in both passes.

- `deblock` §8.7.2.5.1 / §8.7.2.5.2 CU-level edge-filtering driver —
  `filter_cu_edges` filters every edge of one coding unit in one direction
  directly into a `Picture` (luma + Cb + Cr planes):
  - Luma: the §8.7.2.5.4 block-edge filter at every `bS > 0` sampled
    position (8-stride along the edge axis, 4-stride across).
  - Chroma (`ChromaArrayType != 0`): the §8.7.2.5.5 filter at every
    `bS == 2` position whose chroma edge is on the 8-chroma-sample grid,
    stepping `8 / SubWidthC` (EDGE_VER) / `8 / SubHeightC` (EDGE_HOR) in
    the edge axis, for both Cb and Cr with their `pps_c*_qp_offset`.
  - `DeblockCu` (CU geometry + neighbour p-side `QpY`) + `DeblockCuParams`
    (QpY, slice β/tC offsets, chroma QP offsets, bit depths,
    ChromaArrayType) carry the per-CU context; `Picture::plane_mut`
    exposes a mutable component plane for in-place filtering.
  6 new tests: vertical/horizontal internal luma seam smoothing,
  4:2:0 chroma skipping a non-8-aligned internal edge vs. filtering an
  8-aligned CU-boundary edge, bS=1 luma-only (chroma skipped), and
  monochrome luma-only.

- `deblock` §8.7.2.2 / §8.7.2.3 edge-flag derivation — the missing input
  to the §8.7.2.4 bS stage, produced from the coding block's geometry:
  - `TransformSplit`: the transform-tree split geometry of a coding
    block (`Split` quadrants / `Leaf` transform blocks), with `leaf()` /
    `split_once()` constructors.
  - `transform_block_boundary` (§8.7.2.2): the recursive descent that
    marks each transform-block leading edge into the `edge_flags` +
    `tb_edge` grids, gating the coding-block boundary (xB0/yB0 == 0) by
    `filterEdgeFlag` and marking interior transform splits unconditionally.
  - `prediction_block_boundary` (§8.7.2.3): the `PartMode` internal
    prediction partition column/row (PART_Nx2N/NxN at nCbS/2, AMP
    PART_nLx2N/nRx2N at nCbS/4 · {1,3}, PART_2NxN/2NxnU/2NxnD likewise),
    set in `edge_flags` only (a prediction edge is not a TB edge).
  - `derive_edge_flags` → `EdgeFlags`: the public entry that runs both
    derivations for one CB + edge direction and exposes the
    `edge_flags()` / `tb_edge()` grids that feed `derive_boundary_strength`.
  9 new tests: single-TB left boundary + gating, one-level + nested
  transform splits (EDGE_VER / EDGE_HOR), PART_Nx2N prediction edge (not
  a TB edge), AMP columns/rows, and an end-to-end edge-flags → bS=2 check.

### Added — clean-room rebuild round 350 (2026-06-20)

- `deblock` module — the §8.7.2.5 deblocking edge-filtering process, the
  sample-modification stage that the §8.7.2.4 bS derivation fed into:
  - `beta_prime` / `tc_prime`: Table 8-12 β′/tC′ from input Q.
  - `luma_beta_tc`: §8.7.2.5.3 β/tC derivation (eqs. 8-347..8-351).
  - `luma_sample_decision` (§8.7.2.5.6) + `luma_edge_decision`
    (§8.7.2.5.3 dE/dEp/dEq over the 4-row segment).
  - `filter_luma_sample`: §8.7.2.5.7 strong (eqs. 8-389..8-394) and weak
    (eqs. 8-395..8-402) luma sample filters with ±2·tC clipping.
  - `chroma_qpc_420` (Table 8-10), `chroma_tc` (§8.7.2.5.5) and
    `filter_chroma_sample` (§8.7.2.5.8) for the chroma path.
  - `SamplePlane` + `filter_luma_block_edge` (§8.7.2.5.4) /
    `filter_chroma_block_edge` (§8.7.2.5.5): plane-level drivers that
    gather and apply the primitives in place across a 4-row EDGE_VER /
    EDGE_HOR segment.
  16 new deblock unit/integration tests (Table 8-12 / 8-10 breakpoints,
  β/tC at 8/10-bit, flat/step decision paths, strong/weak/chroma filter
  math, plane-level seam smoothing).

### Added — clean-room rebuild round 341 (2026-06-19)

- `picture` module — the §8 reconstruction target: a `Picture` holding
  the three reconstructed sample planes (`SL` / `SCb` / `SCr`) sized from
  the active geometry + `ChromaArrayType` (Table 6-1 `SubWidthC` /
  `SubHeightC`), per-sample read/write, the §8 `Clip1Y` / `Clip1C`
  sample clip (`clip1`), and an 8-bit planar `Y`→`Cb`→`Cr` packer for
  fixture comparison.
- `recon` module — the §8.4 intra sample-reconstruction driver, the rung
  between the §7.3.8 slice-data syntax walk and the per-block §8.4.4
  intra-prediction + §8.6 dequantization / inverse-transform primitives.
  `reconstruct_intra_ctu` walks a decoded `CodingTreeUnit` and writes
  reconstructed samples into a `Picture`:
  - §8.4.2 `IntraPredModeY` + §8.4.3 `IntraPredModeC` derivation from the
    signalled luma/chroma mode fields.
  - §8.4.4.2.1 reference-sample gathering from the already-reconstructed
    picture (the §6.4.1 raster within-picture availability), then
    §8.4.4.2 prediction.
  - §8.6.2 dequantize + inverse-transform of each coded residual block,
    §8.4.4.1 add-and-clip into the plane.
  - §8.6.1 `Qp′Y` (eq. 8-258) and `Qp′Cb` / `Qp′Cr` (Table 8-10 4:2:0
    chroma-QP mapping, eq. 8-260) derivation.
  - The transform-tree recursion reconstructs each leaf transform block
    luma + (4:2:0 / 4:2:2 / 4:4:4) chroma in §8.4.4.1 decode order.
  - 6 driver unit tests plus an end-to-end fixture test
    (`tiny_i_reconstructs_expected_yuv_end_to_end`) that decodes the real
    `tiny-i-only-16x16-main` IDR slice CABAC bytes through the §7.3.8
    syntax walk and reconstructs the byte-exact `expected.yuv` planes
    (luma 0x51, Cb 0x5a, Cr 0xf0), with the single-CTU slice terminating
    at `end_of_slice_segment_flag`. This is the crate's first
    decode-to-pixels validation on a real bitstream.

### Fixed — round 341

- §9.3.3.3 EGk prefix polarity (two call sites). The k-th-order
  Exp-Golomb prefix decode — both the `coeff_abs_level_remaining`
  escape-path suffix and the shared `read_eg_k_with` helper used by
  `cu_qp_delta_abs`, `palette_escape_val`, and the `abs_mvd_minus2` EG1
  escape — counted leading `0` bins terminated by a `1`, but per
  eq. 9-13 the EGk unary prefix is a run of `1` bins terminated by a
  single `0` (the §9.3.3.3 NOTE: EGk uses 1's and 0's reversed from the
  §9.2 EG0 prefix). The inverted polarity produced wrong escape-path
  magnitudes for every EGk-coded syntax element, mis-decoding
  `cu_qp_delta_abs` and the high-magnitude coefficient levels and
  derailing the residual-coding CABAC alignment. With the fix the
  `tiny-i-only-16x16-main` slice decodes bit-exactly to its
  `end_of_slice_segment_flag` terminator and reconstructs the documented
  `expected.yuv`. Seven unit tests that had encoded the inverted
  zeros-then-one prefix are rewritten to the spec-correct ones-then-zero
  shape.
- §8.6.4 inverse-transform matrix orientation. The §8.6.4.2 1-D transform
  read the in-code DCT base table as `transMatrix[i][j*stride]`, but per
  eqs. 8-318/8-319 the named `transMatrixCol0to15` base table is indexed
  `[column][row]`, so the in-code row-major listing is the transpose of
  `transMatrix` and eq. 8-317 must read it as `DCT32[j*stride][i]`. The
  previous indexing computed the forward (analysis) transform — a DC-only
  coefficient excited a non-constant column instead of the flat row-0 DC
  basis, so a DC-only block reconstructed to a spread of values rather
  than the uniform field the inverse transform must produce. Two unit
  tests that had baked in the transposed behaviour are rewritten to the
  correct constant-field reconstruction.

### Added — clean-room rebuild round 338 (2026-06-19)

- `slice_data` module — the §7.3.8.1 .. §7.3.8.6 slice-data CABAC
  syntax-element walk, the upper rung of the §7.3.8 parse loop that was
  the crate's largest missing subsystem. It drives the CABAC engine
  through the per-CTU syntax structures, composing the leaf decode
  primitives with the existing §7.3.8.8 `transform_tree()` recursion:
  - `decode_coding_tree_unit` (§7.3.8.2) — optional §7.3.8.3 `sao()` +
    the §7.3.8.4 coding-quadtree root, producing `CodingTreeUnit`.
  - `decode_sao` (§7.3.8.3) — the full per-CTB SAO param walk: the two
    merge flags, the per-component `sao_type_idx` / offset / band /
    eo-class reads, the §7.4.9.3 `SaoTypeIdx[2]`/eo-class inheritance,
    into `SaoCtbParams` / `SaoComponent`.
  - `decode_coding_quadtree` (§7.3.8.4) — the recursive `split_cu_flag`
    walk with the §7.4.9.4 boundary inference, the §6.5.1
    quantization-group resets at the `Log2MinCuQpDeltaSize` /
    `Log2MinCuChromaQpOffsetSize` thresholds, and the in-picture
    boundary `if( x1 < … )` child guards, into `CodingQuadtree`.
  - `coding_unit` (§7.3.8.5) — the full CU body: `cu_transquant_bypass`,
    `cu_skip_flag`, `pred_mode_flag`, `part_mode`, the PCM gate, the
    intra luma/chroma mode signalling group, the inter `prediction_unit`
    emission per `PartMode`, `rqt_root_cbf`, and entry into the
    transform tree, into `CodingUnit` / `IntraLumaMode`.
  - `prediction_unit` (§7.3.8.6) — the merge / non-merge inter PU walk
    (`merge_flag` / `merge_idx` / `inter_pred_idc` / `ref_idx` /
    `mvd_coding` / `mvp_lX_flag`, with the `mvd_l1_zero_flag` + PRED_BI
    zero-inference), into `PredictionUnit`.
  - `CtuGrid` — a per-CTU `CtDepth` / `cu_skip_flag` neighbour grid at
    `MinCbSizeY` granularity feeding the §9.3.4.2.2 `split_cu_flag` /
    `cu_skip_flag` left/above ctxInc derivations.
  - 6 driver tests covering the grid neighbour lookups, SAO decode,
    the min-CB leaf case, and the full I-slice CTU walk.
- `tests/tiny_i_ctu_walk.rs` — end-to-end fixture test driving the
  slice-data walk on the real `tiny-i-only-16x16-main` HEVC bitstream
  (embedded slice NAL, emulation-prevention-stripped, slice-header
  walked to `byte_alignment()`, CABAC engine on the remaining bytes).
  The single 16×16 intra CTU decodes bit-exactly through SAO (type 0
  on Y/Cb/Cr, both merge flags absent at rx==ry==0) and the §7.3.8.4 /
  §7.3.8.5 coding-quadtree / coding-unit structure (one un-split 16×16
  intra PART_2Nx2N CU, one luma PB + one chroma mode).

- `binarization` — the remaining §7.3.8.3 / §7.3.8.5 / §7.3.8.6 leaf
  decode primitives the slice-data CTU/CU walk composes:
  - SAO (§7.3.8.3): `decode_sao_merge_flag` (FL `cMax = 1`, ctxInc 0),
    `decode_sao_type_idx` (TR `cMax = 2`, bin 0 context + bin 1 bypass),
    `decode_sao_offset_abs` (TR, bypass, `cMax` per
    `sao_offset_abs_tr_cmax(bitDepth)`), `decode_sao_offset_sign`
    (FL `cMax = 1` bypass), `decode_sao_band_position` (FL `cMax = 31`,
    5 bypass bins), `decode_sao_eo_class` (FL `cMax = 3`, 2 bypass bins).
  - `part_mode` (§7.3.8.5 / §9.3.3.7 / Table 9-45): `decode_part_mode`
    walks the `CuPredMode` + `log2CbSize` + `amp_enabled_flag`-dependent
    bin string (bins 0..=2 context-coded via the `part_mode[ctxInc]`
    bank, bin 3 bypass) into the `PartMode` enum + `IntraSplitFlag`
    (`PartModeResult`); `part_mode_inferred` for the §7.4.9.5
    not-present `PART_2Nx2N` case; `PartMode::is_amp`.
  - `pcm_flag` / `end_of_slice_segment_flag` (§7.3.8.5 / §7.3.8.1):
    `decode_pcm_flag` / `decode_end_of_slice_segment_flag`, both the
    §9.3.4.3.5 *terminate* path (Table 9-48 `terminate` row).
  - prediction_unit (§7.3.8.6): `decode_merge_idx` (TR
    `cMax = MaxNumMergeCand − 1`, bin 0 context + bypass tail),
    `decode_inter_pred_idc` (§9.3.3.9 / Table 9-47 → `InterPredIdc`),
    `decode_ref_idx` (TR `cMax = num_ref_idx_lX_active_minus1`, bins
    0/1 context + bypass tail), `decode_mvp_flag` (FL `cMax = 1`).
  - 20 new unit tests covering the inference paths, AMP classification,
    terminate-path decoders, and the bypass/context bin splits.

### Added — clean-room rebuild round 334 (2026-06-18)

- `binarization` — the §7.3.8.4 `split_cu_flag` and §7.3.8.5
  `cu_skip_flag` single-bin decode primitives, the gateway flags of the
  `coding_quadtree( )` / `coding_unit( )` walk:
  - `decode_split_cu_flag` / `decode_cu_skip_flag` — Table 9-43 FL
    `cMax = 1` (one context-coded bin); the bank slot is selected by the
    existing §9.3.4.2.2 / Table 9-49 `split_cu_flag_ctx_inc` /
    `cu_skip_flag_ctx_inc` left/above neighbour ctxInc helpers, and the
    decode itself is a single §9.3.4.3.2 context decision. The §7.3.8.4
    split-presence gate / §7.4.9.4 boundary inference and the §7.3.8.5
    `slice_type != I` read gate remain the caller's responsibility.
  - `cu_pred_mode_from_skip` — the §7.4.9.5 not-present `CuPredMode`
    derivation from `cu_skip_flag` + slice type: `Some(Intra)` for I
    slices, `Some(Skip)` for a P / B skip CU, and `None` for a P / B
    non-skip CU (signalling a `pred_mode_flag` read is still required).
  - `SPLIT_CU_FLAG_FL_{CMAX,NBITS}` / `CU_SKIP_FLAG_FL_{CMAX,NBITS}`
    binarization-shape constants.
  - 9 new unit tests (zero / one / one-bin-consumed paths for both
    decoders, the FL shapes, and the four `cu_pred_mode_from_skip`
    branches).

### Added — clean-room rebuild round 330 (2026-06-18)

- `transform_tree` module — the §7.3.8.8 `transform_tree( x0, y0,
  xBase, yBase, log2TrafoSize, trafoDepth, blkIdx )` recursion, the
  missing rung between the §7.3.8.5 `coding_unit( )` walk and the
  §7.3.8.10 `transform_unit( )` leaf:
  - `decode_transform_tree` — mirrors the §7.3.8.8 syntax table exactly:
    the `split_transform_flag` presence gate (`log2TrafoSize <=
    MaxTbLog2SizeY && log2TrafoSize > MinTbLog2SizeY && trafoDepth <
    MaxTrafoDepth && !(IntraSplitFlag && trafoDepth == 0)`) with the
    §7.4.9.8 forced-split inference (`log2TrafoSize > MaxTbLog2SizeY`,
    `IntraSplitFlag` at depth 0, `interSplitFlag`); the per-node
    `cbf_cb` / `cbf_cr` reads gated by the inheritance condition
    (`trafoDepth == 0 || cbf_cX[xBase][yBase][trafoDepth − 1]`) with the
    `ChromaArrayType == 2` lower-half companions (`!split_transform_flag
    || log2TrafoSize == 3`); the four-way quarter-size recursion; and at
    each leaf the §7.3.8.8 `cbf_luma` presence condition
    (`CuPredMode == MODE_INTRA || trafoDepth != 0 || cbf_cb || cbf_cr ||
    (ChromaArrayType == 2 && (cbf_cb_lower || cbf_cr_lower))`) before
    invoking `decode_transform_unit`. One `QuantGroupState` is threaded
    through the whole subtree so the `delta_qp()` / `chroma_qp_offset()`
    gates fire once per quantization group.
  - `TransformTreeParams` (the per-CU geometry + context: `MaxTbLog2SizeY`
    / `MinTbLog2SizeY` / `MaxTrafoDepth`, `IntraSplitFlag`,
    `interSplitFlag`, `CuPredMode`, `ChromaArrayType`, and the
    `TransformUnitParams` template) and the `TransformTree`
    `Split { … children } / Leaf { cbf_luma, unit }` decoded result.
- `binarization` — four new §7.3.8.8 single-bin decode primitives the
  recursion composes: `decode_split_transform_flag` /
  `split_transform_flag_inferred`, `decode_cbf_luma` /
  `cbf_luma_inferred`, `decode_cbf_cb` / `decode_cbf_cr` /
  `cbf_chroma_inferred` (each an FL `cMax = 1` context-coded bin per
  §9.3.4.2.1 / Table 9-48, with the §7.4.9.8 not-present inference).

### Added — clean-room rebuild round 325 (2026-06-16)

- `transform_unit` module — the §7.3.8.10 `transform_unit( x0, y0,
  xBase, yBase, log2TrafoSize, trafoDepth, blkIdx )` syntax driver, the
  leaf the §7.3.8.8 `transform_tree()` recursion bottoms out in:
  - `decode_transform_unit` — walks the §7.3.8.10 table exactly: the
    `cbfChroma` derivation (including the `ChromaArrayType == 2`
    lower-half companions), the adaptive-colour-transform predicate
    gating `tu_residual_act_flag`, the `delta_qp()` / `chroma_qp_offset()`
    blocks (each gated by the per-quantization-group `IsCuQpDeltaCoded` /
    `IsCuChromaQpOffsetCoded` state in `QuantGroupState`), the luma
    `residual_coding()`, and the chroma path — both the in-place branch
    (with the §7.3.8.12 `cross_comp_pred()` prelude and the
    `log2TrafoSizeC = Max(2, log2TrafoSize − (ChromaArrayType == 3 ? 0 :
    1))` chroma size, plus the `ChromaArrayType == 2` stacked-sub-block
    pair) and the `blkIdx == 3` deferred-chroma branch where the chroma
    residuals are coded against the parent node at the last luma leaf.
  - `TransformUnitParams` / `TransformUnit` / `QuantGroupState` /
    `CuPredMode` — typed inputs, decoded result, and the
    across-transform-unit quant-group decode state.
- `binarization` — two new §7.3.8.10 / §7.3.8.12 primitive decoders the
  driver composes:
  - `decode_cross_comp_pred` — §7.3.8.12 `cross_comp_pred( x0, y0, c )`:
    the TR(`cMax = 4`) `log2_res_scale_abs_plus1[ c ]` prefix plus the
    conditional `res_scale_sign_flag[ c ]`, with the §7.4.9.12
    `ResScaleVal` derivation (equations 7-79 / 7-80) surfaced on
    `CrossCompPred`.
  - `decode_tu_residual_act_flag` — §7.3.8.10 `tu_residual_act_flag`
    (FL `cMax = 1`, Table 9-39 / Table 9-48 ctxInc 0), plus the §7.4.9.10
    `tu_residual_act_flag_inferred` helper.

### Added — clean-room rebuild round 321 (2026-06-16)

- `inter_pred` module — §8.5.3.3.3 fractional sample interpolation plus
  the §8.5.3.3.4.2 default weighted sample prediction combine, the first
  inter-prediction sample-generation increment:
  - `RefPlane` — a row-major reference-picture sample plane with the
    §8.5.3.3.3 `Clip3( 0, dim − 1, … )` edge extension (equations 8-222 /
    8-223 for luma, 8-239 / 8-240 for chroma) so the filters can index
    with the raw `xInt + i` / `yInt + j` offsets.
  - `interp_luma_block` — §8.5.3.3.3.2 separable 8-tap quarter-pel luma
    interpolation (equations 8-224..8-238), with Table 8-8 phase
    selection. `shift1 = Min(4, BitDepthY − 8)`, `shift2 = 6`,
    `shift3 = Max(2, 14 − BitDepthY)`; full-pel is `A << shift3`.
  - `interp_chroma_block` — §8.5.3.3.3.3 separable 4-tap eighth-pel chroma
    interpolation (equations 8-241..8-261), with Table 8-9 phase
    selection.
  - `default_weighted_pred` — §8.5.3.3.4.2 uni- / bi-predictive combine
    (equations 8-262..8-264, the `weighted_pred_flag == 0` path), with
    `shift1 = Max(2, 14 − bitDepth)`, `shift2 = Max(3, 15 − bitDepth)`,
    clipping to `[0, (1 << bitDepth) − 1]`.
  - `InterPredError` — empty / mismatched plane, empty block, out-of-range
    fraction, out-of-range bit depth, and array-length-mismatch surfaces.

  The interpolation carries the `14 − BitDepth`-bit intermediate precision
  the spec keeps between §8.5.3.3.3 and §8.5.3.3.4; the combine clips to
  the sample range. The §8.5.3.1 / §8.5.3.2 MV / merge derivation, the
  §8.5.3.3.1 block-walk driver, and the §8.5.3.3.4.3 explicit weighted
  path remain follow-ups. 12 unit tests (flat-plane invariants for all
  4×4 luma and 8×8 chroma phases, hand-computed kernel values, edge
  extension, uni- / bi-predictive combine, clipping, 10-bit shift3, and an
  end-to-end interpolate-then-combine pipeline).

### Added — clean-room rebuild round 318 (2026-06-16)

- `availability` module — §6.4 availability processes plus the §6.5.1 /
  §6.5.2 picture-level scanning conversions they depend on. This is the
  neighbour-availability derivation that produces the per-sample
  "available for intra prediction" markings consumed by `intra_pred`:
  - `PictureTiling::new` / `TilingParams` — §6.5.1 CTB raster-scan ↔
    tile-scan conversion. Derives `colWidth` / `rowHeight` (eqs. 6-3 /
    6-4, both the `uniform_spacing_flag` even-split and the explicit
    `column_width_minus1` / `row_height_minus1` forms), `colBd` /
    `rowBd` (eqs. 6-5 / 6-6), `CtbAddrRsToTs` / `CtbAddrTsToRs` (eqs.
    6-7 / 6-8, the tile-scan permutation and its inverse), and `TileId`
    (eq. 6-9). Rejects zero geometry, `CtbLog2SizeY < MinTbLog2SizeY`,
    mis-sized explicit tile arrays, and tile sizes that overflow the
    picture.
  - `PictureTiling::min_tb_addr_zs` — §6.5.2 eq. 6-10. The
    `MinTbAddrZs[ x ][ y ]` z-scan address of a minimum block,
    interleaving the within-CTB Morton (z) order of the block's low
    `( x, y )` bits with the tile-scan CTB address in the high bits.
  - `PictureTiling::z_scan_availability` — §6.4.1. `availableN` from
    the in-picture boundary test (eq. 6-2), the decode-order test
    (`minBlockAddrN > minBlockAddrCurr`), the `SliceAddrRs`
    slice-segment-boundary test, and the `TileId` tile-boundary test.
    Takes a caller-supplied `slice_addr_rs` CTB→`SliceAddrRs` lookup.
  - `PictureTiling::prediction_block_availability` — §6.4.2. Wraps the
    z-scan query with the `sameCb` short-cut (the NxN partIdx-1
    forced-FALSE branch for the not-yet-decoded co-CU region) and the
    final `MODE_INTRA` masking, via a caller-supplied `CuPredMode`
    lookup.
  - 18 unit tests, all expected values hand-derived from the §6.4 /
    §6.5 equations (single-tile identity, 2×2-tile tile-scan reorder,
    explicit-width tiles + overflow rejection, Morton interleave,
    and the availability boundary/slice/tile/sameCb/intra-mask cases).

### Added — clean-room rebuild round 315 (2026-06-15)

- `intra_pred` module — §8.4.4.2 intra sample prediction, the predictor
  core that turns marked neighbour samples into the `(nTbS)x(nTbS)`
  `predSamples` array, consuming the §8.4.2 / §8.4.3 mode derivation
  landed in prior rounds:
  - `substitute_reference_samples` — §8.4.4.2.2 reference-sample
    substitution. The bottom-left→corner→top-right sweep fills every
    sample marked "not available for intra prediction" (step-1 seed of
    `p[ −1 ][ 2*nTbS−1 ]`, then the single forward propagation that
    covers steps 2 and 3); when no neighbour is available, all samples
    take the mid-level `1 << ( bitDepth − 1 )`.
  - `filter_reference_samples` / `reference_filter_flag` — §8.4.4.2.3.
    The Table 8-4 `filterFlag` gate (suppressed for `INTRA_DC` and
    `nTbS == 4`), the `[1 2 1] >> 2` smoothing (eqs. 8-41..8-45), and the
    `nTbS == 32` luma `biIntFlag` bi-linear interpolation (eqs.
    8-36..8-40).
  - `predict_planar` — §8.4.4.2.4 `INTRA_PLANAR` (eq. 8-46).
  - `predict_dc` — §8.4.4.2.5 `INTRA_DC` (`dcVal` eq. 8-47 + the luma
    `nTbS < 32` boundary smoothing eqs. 8-48..8-51).
  - `predict_angular` — §8.4.4.2.6 `INTRA_ANGULAR2..34`. Tables 8-5 /
    8-6 `intraPredAngle` / `invAngle`, the main reference-array
    projection with the negative-angle inverse-angle extension (eqs.
    8-53..8-67), and the mode-26 / mode-10 luma boundary filter (eqs.
    8-60 / 8-68).
  - `intra_predict` / `intra_predict_with_substitution` — §8.4.4.2.1
    steps 1 and 2: the filtering gate
    (`intra_smoothing_disabled_flag == 0 && (cIdx == 0 ||
    ChromaArrayType == 3)`) plus the planar / DC / angular dispatch,
    with the substitution-first variant running the full pipeline from a
    `MarkedReferenceSamples` input.
  - 18 unit tests: substitution (mid-level fallback, sweep propagation,
    step-1 seed, available-value preservation), Table 8-4 boundaries,
    hand-worked planar eq-8-46 / DC eq-8-47..8-50 / angular pure-index
    cells, boundary-filter on/off, and the end-to-end pipeline.

### Added — clean-room rebuild round 311 (2026-06-15)

- `binarization` module — the §8.4.2 derivation process for luma intra
  prediction mode, the process the round-40/41 signalling group
  (`prev_intra_luma_pred_flag`, `mpm_idx`, `rem_intra_luma_pred_mode`)
  feeds, completing the luma intra-mode resolution chain alongside the
  round-42 §8.4.3 chroma derivation:
  - `intra_luma_cand_mode_list` — §8.4.2 step 3: builds the three-entry
    `candModeList[ 0..=2 ]` from the two step-2 candidate neighbour modes
    `candIntraPredModeA` / `candIntraPredModeB`. The equal-candidate
    branch splits on `candA < 2` (eqs. 8-21..8-23 ⇒ `{PLANAR, DC,
    ANGULAR26}`) versus the angular case (eqs. 8-24..8-26 — the candidate
    plus its two mod-32-wrapped neighbouring angular modes); the
    distinct-candidate branch fills slots 0/1 (eqs. 8-27/8-28) and picks
    `candModeList[ 2 ]` as the first of `{PLANAR, DC, ANGULAR26}` not
    already present.
  - `derive_intra_pred_mode_y` — §8.4.2 step 4: on the
    [`LumaIntraModeSource::Mpm`] path `IntraPredModeY =
    candModeList[ mpm_idx ]`; on the [`LumaIntraModeSource::Remaining`]
    path the candidate list is sorted ascending (the eqs. 8-29..8-31
    three-compare-and-swap sort) and `rem_intra_luma_pred_mode` is passed
    through the increment pass (`+1` for every sorted candidate at or
    below the running value), mapping the 31-value remaining field onto
    the 35-mode space exclusive of the three most-probable modes.
  - `INTRA_PLANAR` (0), `INTRA_DC` (1), `INTRA_ANGULAR26` (26) and
    `INTRA_PRED_MODE_MAX` (34) — the Table 8-1 mode-name constants.
  - The §8.4.2-step-2 candidate reduction (§6.4.1 availability,
    `CuPredMode` / `pcm_flag` tests, the CTB-row-boundary B clamp) stays
    the slice-data parser's responsibility, consistent with the
    availability-as-input convention of the §9.3.4.2.2 neighbour ctxInc
    derivations.
- 10 new tests (521 total, was 511): the Table 8-1 mode constants; the
  step-3 equal-low (eqs. 8-21..8-23), equal-angular with mod-32 edge
  wraps at modes 2 and 34 (eqs. 8-24..8-26), and distinct-candidate
  first-missing-default (eqs. 8-27/8-28) branches; the all-candidate-pair
  in-range invariant; the step-4 Mpm direct index; the Remaining
  low-mode anchors, the pre-increment ascending sort, and the
  bijection-onto-`(0..=34) \ candModeList` invariant over the full rem
  range; and an end-to-end candModeList → IntraPredModeY composition on
  both paths.

### Added — clean-room rebuild round 308 (2026-06-15)

- `binarization` module — the §7.3.8.6 `prediction_unit( )`
  `merge_flag` syntax element:
  - `decode_merge_flag` decodes the single context-coded FL `cMax = 1`
    bin (Table 9-43 shape, Table 9-48 bin-0 `ctxInc = 0`) from the
    CABAC engine. Value 1 selects the merge path (inter-prediction
    parameters inferred from a neighbouring inter-predicted partition,
    `merge_idx` follows); value 0 selects the explicit-motion path.
  - `merge_flag_inferred` applies the §7.4.9.6 not-present inference
    (`CuPredMode == MODE_SKIP ⇒ 1`, otherwise `0`) for the §7.3.8.6
    `cu_skip_flag == 1` path, without entering the engine.
  - `merge_flag_ctx_inc` (= 0) plus the `MERGE_FLAG_FL_CMAX` /
    `MERGE_FLAG_FL_NBITS` shape constants. The Table 9-15 init bank
    (`{110, 154}`, initType 0 = `na`) was already wired in `ctx_init`.

### Added — clean-room rebuild round 48 (2026-06-14)

- `binarization` module — the §7.3.8.9 / §7.4.9.9 `mvd_coding( )`
  motion-vector-difference syntax structure:
  - `decode_mvd_component` / `decode_mvd_component_with` decode one
    `mvd_coding( )` component (`compIdx`): the two context-coded
    magnitude flags `abs_mvd_greater0_flag` / `abs_mvd_greater1_flag`
    (Table 9-48 `ctxInc = 0`, Table 9-23 single-context banks via
    `SliceContexts`), the bypass-coded EG1 escape `abs_mvd_minus2`
    (Table 9-43 `EG1`) read only when `abs_mvd_greater1_flag == 1`,
    and the bypass FL sign bit `mvd_sign_flag` (`cMax = 1`) read only
    when `abs_mvd_greater0_flag == 1`.
  - `MvdComponent` carries all four wire fields plus the equation-7-73
    composed signed difference `lMvd`; `mvd_component_value` applies
    the not-present inferences (`abs_mvd_greater1_flag` ⇒ 0,
    `abs_mvd_minus2` ⇒ −1, `mvd_sign_flag` ⇒ 0).
  - `abs_mvd_greater0_flag_ctx_inc` / `abs_mvd_greater1_flag_ctx_inc`
    expose the Table 9-48 `ctxInc`; `ABS_MVD_GREATER_FLAG_FL_CMAX`,
    `MVD_SIGN_FLAG_FL_CMAX`, `ABS_MVD_MINUS2_EG_K` expose the
    Table 9-43 binarization parameters.
  - `read_eg_k_with` factors the §9.3.3.3 k-th-order Exp-Golomb decode
    over a generic bin reader, shared with the engine-driven
    `decode_eg_k`.

### Added — clean-room rebuild round 47 (2026-06-14)

- `transform` module — the §8.6.2 / §8.6.3 / §8.6.4 scaling,
  transformation and residual-array construction process that turns the
  decoded `TransCoeffLevel[ xC ][ yC ]` array of one transform block
  into the `(nTbS)x(nTbS)` array `r` of residual samples:
  - `scale_coefficients` — the §8.6.3 scaling (dequantization) process.
    Each `TransCoeffLevel[ x ][ y ]` is multiplied by `m[ x ][ y ]`
    (a flat 16, or `ScalingFactor[ sizeId ][ matrixId ][ x ][ y ]`), the
    `levelScale[ qP % 6 ]` rational-step list (`{40,45,51,57,64,72}`)
    and `1 << ( qP / 6 )`, then offset-rounded by `bdShift`
    (equation 8-301/8-305) and clipped to `[ coeffMin, coeffMax ]`
    (equations 8-300..8-309), with `i64` product intermediates for the
    extended-precision ranges.
  - `inverse_transform` — the §8.6.4 separable inverse transform: the
    column then row §8.6.4.2 one-dimensional transform, selecting the
    equation-8-316 4x4 DST-VII matrix for `MODE_INTRA` 4x4 luma
    (`trType == 1`) and the equations-8-318..8-321 32x32 DCT-II matrix
    (subsampled at stride `1 << (5 − log2(nTbS))` per equation 8-317)
    for every other block, with the equation-8-314 intermediate
    `(e + 64) >> 7` offset-round and clip.
  - `residual_block` — the §8.6.2 orchestration over
    `cu_transquant_bypass_flag` (the equation-8-297 `rotateCoeffs`
    pass-through), `transform_skip_flag` (the equation-8-298 `tsShift`
    left-shift), and the full scale-then-transform path, applying the
    equation-8-299 final `bdShift` offset-round (equations 8-294..8-296).
  - The §7.4.5 `CoeffMin` / `CoeffMax` derivation (`coeff_range`,
    equations 7-27..7-30) and the `LEVEL_SCALE` / `Component` /
    `PredMode` / `BlockParams` / `TransformError` public surface.
  - 17 tests: hand-computed §8.6.3 single-DC / negative-level /
    saturation scaling, the transquant-bypass verbatim + `rotateCoeffs`
    mirror copies, an exact 4x4 inverse-DCT case, the DST-vs-DCT
    `trType` selection (luma-intra-4x4 only) and the chroma / inter
    exclusions, the `transform_skip` `tsShift` path, the DCT subsample
    stride, matrix-cell pins, and the size / length / bit-depth error
    paths. Total tests 495 (was 478).

### Added — clean-room rebuild round 46 (2026-06-14)

- `sei` module — the §7.3.2.4 / §7.3.5 / §D.2 Supplemental Enhancement
  Information parse:
  - The §7.3.5 `sei_message()` framing with the extensible
    `payloadType` / `payloadSize` byte runs (every `0xFF` adds 255, the
    first non-`0xFF` byte is the final term), and the §7.3.2.4
    `sei_rbsp()` `do … while( more_rbsp_data() )` message loop
    (`parse_sei_rbsp`), terminating positionally at the
    `rbsp_trailing_bits()` `0x80` stop byte.
  - The §D.2 `sei_payload()` dispatch split by `nal_unit_type`
    (`PREFIX_SEI_NUT` 39 / `SUFFIX_SEI_NUT` 40 via `SeiNalType`). Eight
    payload types are decoded into typed structs: `recovery_point` (6),
    `user_data_registered_itu_t_t35` (4), `user_data_unregistered` (5),
    `active_parameter_sets` (129), `decoded_picture_hash` (132,
    suffix-only, MD5 / CRC / checksum variants),
    `mastering_display_colour_volume` (137), `content_light_level_info`
    (144), and `alternative_transfer_characteristics` (147). Every
    other (or branch-illegal) `payloadType` is carried verbatim as
    `SeiPayload::Reserved`, so the framing always advances by the
    declared `payloadSize`.
  - Overrun / truncation are surfaced as `SeiError::PayloadSizeOverrun`
    / `TruncatedHeader` / `TruncatedPayload`. 31 new tests cover the
    extensible byte runs, each typed payload (including negative
    `recovery_poc_cnt` and the three `decoded_picture_hash` variants),
    the prefix/suffix dispatch (prefix-only payload in a suffix NAL ⇒
    Reserved and vice versa), multi-message RBSPs, and the error paths.
    Total tests now 478 (was 447).

### Added — clean-room rebuild round 45 (2026-06-12)

- `ctx_init` module — the complete §9.3.2.2 context-variable
  initialization layer:
  - All 38 `initValue` tables (Tables 9-5..9-42) transcribed from the
    staged specification PDFs, one flat constant per table laid out on
    the printed ctxIdx axis, covering every context-coded syntax
    element of §7.3.8.1..§7.3.8.12 (the §9.3.2.2 exceptions
    `end_of_slice_segment_flag` / `end_of_subset_one_bit` / `pcm_flag`
    keep the NOTE 2 non-adapting `ctxTable == 0` state). Both staged
    PDFs (v8 08/2021 and v11 01/2026) were cross-checked and agree on
    every cell.
  - The Table 9-4 ctxIdx-span selection: `uniform_init_values` for the
    regular contiguous three-`initType` layouts, `inter_init_values`
    for the inter-only two-column tables (returns `None` at
    `initType == 0`), `sig_coeff_flag_init_values` for the Table 9-29
    42-per-type body plus the ctxIdx 126..131 transform-skip tail, and
    dedicated handling for the irregular `part_mode` (1 + 4 + 4),
    `cbf_cb`/`cbf_cr` (4 × 3 + the ctxIdx 12/13/14 fifth context),
    `abs_mvd_greater0_flag`/`greater1_flag` (Table 9-23 interleave),
    `transform_skip_flag` and `explicit_rdpcm_*` (luma + shared-chroma
    pairs) layouts.
  - `SliceContexts` — the whole per-slice context array (185 adapting
    context variables per `initType`; Table 9-4 shared-variable groups
    such as `sao_merge_left/up`, `ref_idx_l0/l1`, `mvp_l0/l1_flag`,
    `cbf_cb/cr` and the palette copy-above pair stored once).
    `SliceContexts::init(initType, SliceQpY)` runs equations 9-4..9-6
    over every bank; `SliceContexts::for_slice(slice_type,
    cabac_init_flag, SliceQpY)` adds the equation 9-7 `initType`
    derivation. Inter-only banks at `initType == 0` take the NOTE 2
    non-adapting placeholder (`pStateIdx = 63`), unreachable from any
    table entry, so an accidental I-slice read is recognisable.
  - `ResidualContexts::init(initType, SliceQpY)` — the Table 9-26..9-31
    per-`initType` bank initialization for the §7.3.8.11
    `residual_coding( )` driver (`init_uniform` stays as the scripted
    bring-up constructor). With this the CABAC engine is
    slice-initialisable end-to-end for `initType` 0 / 1 / 2.
- 11 new tests (447 total, was 436): table-shape pins for all 38
  tables; the Table 9-26 == Table 9-27 printed-value identity;
  hand-evaluated `(pStateIdx, valMps)` pins across QPs 0..51 for
  regular, inter-only and irregular layouts; whole-array smoke tests
  per `initType` (every table-derived state in 0..=62, placeholders
  exactly on the inter-only banks, 185-context count); the
  equation 9-7 routing matrix (P/B `cabac_init_flag` swap, I-slice
  invariance); span-helper slicing pins including the Table 9-29
  transform-skip tail; and the `initType > 2` rejection.

### Added — clean-room rebuild round 44 (2026-06-12)

- cargo-fuzz scaffold under `fuzz/` restoring the scheduled Fuzz
  workflow (the post-rebuild tree had no fuzz targets, so the daily
  run failed at discovery). Two harnesses, each run ≥ 5 minutes
  locally under AddressSanitizer + debug assertions after the fixes
  below:
  - `parse_annexb` — Annex B NAL walk (§B.1) → §7.3.1.2 header →
    VPS (§7.3.2.1) / SPS (§7.3.2.2) / PPS (§7.3.2.3.1) dispatch, plus
    the §7.3.6.1 `slice_segment_header()` parse against the last
    activated SPS + PPS pair. Seeded from the
    `docs/video/h265/fixtures/` Annex B corpus.
  - `decode_residual` — the §7.3.8.11 `residual_coding( )` driver
    through the §9.3 arithmetic engine; the leading input bytes map
    onto the driver configuration (transform size, chroma, the
    §7.4.9.11 scanIdx derivation inputs, sign-data-hiding gates, the
    uniform context init). Seeded with synthetic configurations.
- SPS parse-time validation of the §7.4.3.2.1 block-size derivations:
  `CtbLog2SizeY` (eqs. 7-10 / 7-11) must lie in the Annex A profile
  bound 4..=6, `MinTbLog2SizeY` must be `< MinCbLog2SizeY`,
  `MaxTbLog2SizeY` must be `≤ Min( CtbLog2SizeY, 5 )`, and both
  transform-hierarchy depths `≤ CtbLog2SizeY − MinTbLog2SizeY`; plus
  the §A.4.1 item-b/c picture-dimension ceiling
  (`Sqrt( MaxLumaPs * 8 )` = 33 776 at the largest Table A.8 level).
- PPS parse-time validation of the §A.4.1 item-f tile-grid bounds
  (`num_tile_columns_minus1 < 40`, `num_tile_rows_minus1 < 44`, the
  largest Table A.8 entries), which also stops the explicit
  column/row arrays from pre-allocating off the raw wire count.

### Fixed — clean-room rebuild round 44 (2026-06-12)

- Fuzz finding (`parse_annexb`): an SPS with out-of-range
  coding-block-size fields survived the parse and panicked every
  downstream `CtbSizeY = 1 << CtbLog2SizeY` (eq. 7-13) re-derivation
  (first hit in the slice-header `PicSizeInCtbsY` path). Rejected at
  SPS parse time; regression tests pin all the new bounds.
- Fuzz finding (`decode_residual`): the §9.3.3.11
  `coeff_abs_level_remaining` EGk escape could compose a value past
  u32 on a non-conformant bypass-bin stream (32 leading zeros +
  all-ones suffix) and overflow. The composition now saturates in
  u64, and the §7.3.8.11 driver clamps the level magnitude at the
  widest §7.4.9.11 / eqs. 7-27..7-30 profile bound (±2²²);
  conforming streams are bit-identical. A regression test pins the
  maximal-escape shape.
- Fuzz finding (`parse_annexb`): the §7.3.4 `scaling_list_data()`
  parse accepted an unbounded `scaling_list_delta_coef` and
  overflowed the i32 `nextCoef + scaling_list_delta_coef + 256` sum.
  The §7.4.5 −128..=127 range is now enforced (new
  `ScalingListError::DeltaCoefOutOfRange` variant); a regression
  test pins the one-past-the-bound value.

### Added — clean-room rebuild round 43 (2026-06-12)

- §7.3.8.11 `residual_coding( )` syntax driver — the new [`residual`]
  module composes the rounds-26..35 residual primitives into the full
  coefficient-decode loop reconstructing one transform block's
  `TransCoeffLevel[ ][ ]` array: the do-while locate of
  `(lastSubBlock, lastScanPos)` from `LastSignificantCoeff{X,Y}`
  (with the eq. 7-78 vertical-scan swap), the reverse sub-block scan
  with `coded_sub_block_flag` decode and the §7.4.9.11 not-present
  inferences, the per-sub-block `sig_coeff_flag` loop with the full
  §9.3.4.2.5 sigCtx branch dispatch (eq. 9-40 / Table 9-50 / eq. 9-42
  DC / eqs. 9-43..9-53) and both inference rules (last-significant;
  `inferSbDcSigCoeffFlag` DC), the `coeff_abs_level_greater1_flag`
  pass with the `numGreater1Flag < 8` cap and lazy §9.3.4.2.6
  sub-block entry, the at-most-one `coeff_abs_level_greater2_flag`,
  the `signHidden` derivation + `coeff_sign_flag` gates + odd-parity
  `sumAbsLevel` negation, and the level loop with the §7.3.8.11
  remaining-presence test and the §9.3.3.11 per-sub-block eq.-9-24
  Rice adaptation.
  - [`residual::decode_residual_coding_with`] — bin-source-generic
    driver core; [`residual::ResidualBinSource`] +
    [`residual::ResidualElement`] identify each context-coded request.
  - [`residual::decode_residual_coding`] /
    [`residual::EngineResidualBinSource`] — the §9.3.4.3
    arithmetic-engine binding over [`residual::ResidualContexts`]
    (banks sized 18 / 18 / 4 / 44 / 24 / 6 per the Table 9-26..9-31
    ctxIdx spans, exposed as `*_CTX_COUNT` constants;
    [`residual::ResidualContexts::init_uniform`] is bring-up
    scaffolding until the initValue transcription lands).
  - [`residual::residual_coding_scan_idx`] — the §7.4.9.11 scanIdx
    derivation; [`residual::ResidualBlock`] — the reconstructed
    coefficient array; [`residual::ResidualCodingError`].
- The Table 9-50 `ctxIdxMap` doc-comment on
  [`binarization::SIG_COEFF_FLAG_CTX_IDX_MAP_LOG2_TRAFO_SIZE_2`] now
  cites the staged docs errata entry
  (`docs/video/h265/h265-errata-and-clarifications.md` #93) pinning
  the PDF-truncated `i = 15` cell to 8 (the constant already carried
  that value from the round-32 pair-symmetry reconstruction).
- 14 new tests (427 total, was 413): DC-only luma/chroma context
  routing, 4×4 full sweep with Table 9-50 ctxInc cross-checks, 8×8
  two-sub-block walk (csbf ctxInc, DC inference, eq.-9-58 ctxSet
  bump), sign-data-hiding parity flip + disabled counterpart,
  greater-1 cap with the `numSigCoeff >= 8` threshold, eq.-9-24 Rice
  adaptation, vertical-scan swap, transform-skip sigCtx routing,
  scanIdx derivation matrix, input validation, and two engine-backed
  runs.

### Added — clean-room rebuild round 42 (2026-06-11)

- §9.3.3.8 / Table 9-46 + Table 9-48 entries for `intra_chroma_pred_mode`
  (H.265 §7.3.8.5, §7.4.9.5) — the chroma-mode field that follows the
  round-40/41 luma-mode group, plus the §8.4.3 `IntraPredModeC`
  derivation (Tables 8-2 and 8-3). Unlike the generic Table 9-43
  shapes, this element has its own binarization process: the
  single-bin string `0` carries value 4 and a `1` prefix is followed
  by a two-bit FL bypass suffix carrying values 0..=3. Table 9-48
  marks bin 0 context-coded with `ctxInc = 0` (Table 9-13 supplies
  `initValue = {63, 152, 152}` for initType 0 / 1 / 2 — the element is
  read in I, P and B slices) and bins 1..2 `bypass`. Presence follows
  the §7.3.8.5 `ChromaArrayType` gates: once per CU when
  `ChromaArrayType` is 1 or 2, once per luma prediction block when 3,
  absent when 0 (§8.4.3 is only invoked when `ChromaArrayType != 0`).
  - [`binarization::INTRA_CHROMA_PRED_MODE_SAME_AS_LUMA`] (= 4) — the
    Table 9-46 single-bin value; Table 8-2 row 4 sets `modeIdx` to
    `IntraPredModeY` itself.
  - [`binarization::INTRA_CHROMA_PRED_MODE_SUFFIX_FL_NBITS`] (= 2) —
    the Table 9-46 FL suffix width behind the `1` prefix.
  - [`binarization::intra_chroma_pred_mode_ctx_inc`] — Table 9-48
    bin-0 row: `ctxInc = 0`.
  - [`binarization::decode_intra_chroma_pred_mode`] — engine-driven
    decode: one `decode_decision` for the prefix, then (only on a `1`
    prefix) two MSB-first bypass bins. Output is always in
    `{0, 1, 2, 3, 4}`.
  - [`binarization::intra_pred_mode_c_mode_idx`] — Table 8-2: rows
    0..=3 select the base mode from `{0, 26, 10, 1}` substituting 34
    on a collision with `IntraPredModeY`; row 4 tracks the luma mode.
  - [`binarization::INTRA_PRED_MODE_C_CHROMA_422_MAP`] /
    [`binarization::intra_pred_mode_c_chroma_422`] — the Table 8-3
    35-entry remap applied when `ChromaArrayType == 2`.
  - [`binarization::derive_intra_pred_mode_c`] — the full §8.4.3
    output: Table 8-2 `modeIdx`, then Table 8-3 iff
    `ChromaArrayType == 2`, else pass-through.
- Test count: 403 → 413 (+10 new tests covering the Table 9-46 shape
  constants; the Table 9-48 `ctxInc = 0` anchor; the `0`-prefix ⇒
  value-4 single-bin path (with engine-offset cross-check); the
  `1`-prefix two-suffix-bin wrapper-vs-raw-replay agreement (value +
  engine offset); the `{0..4}` output-domain sweep over both context
  polarities; the full Table 8-2 row/column matrix incl. the row-4
  luma-tracking column; Table 8-3 spot anchors + the X ≤ 2
  pass-through; a Table 8-3 structural cross-check (non-decreasing,
  all entries ≤ 34); and the §8.4.3 combined-derivation 4:2:2 vs
  non-4:2:2 routing).

### Added — clean-room rebuild round 41 (2026-06-10)

- §9.3.4.2 / Table 9-43 + Table 9-48 entries for the two §7.3.8.5
  intra-PB luma-mode fields that follow `prev_intra_luma_pred_flag`:
  `mpm_idx` and `rem_intra_luma_pred_mode` (H.265 §7.4.9.2). Both are
  fully bypass-coded (Table 9-48 marks every bin `bypass`, so neither
  consumes a context model). Presence is exactly the round-40
  [`binarization::LumaIntraModeSource`] selection: `Mpm` (flag == 1) ⇒
  `mpm_idx` present and `IntraPredModeY = candModeList[ mpm_idx ]` per
  §8.4.2; `Remaining` (flag == 0) ⇒ `rem_intra_luma_pred_mode` present
  and seeds the §8.4.2 step-2 `IntraPredModeY` before the sorted
  candModeList increment pass. The two are mutually exclusive per the
  §7.3.8.5 `if( prev_intra_luma_pred_flag ) … else …` syntax.
  - [`binarization::MPM_IDX_TR_CMAX`] (= 2) and
    [`binarization::MPM_IDX_TR_C_RICE_PARAM`] (= 0) — Table 9-43 TR
    shape; with `cRiceParam = 0` the §9.3.3.10 TR collapses to
    truncated-unary (`0` / `10` / `11` for values 0 / 1 / 2).
  - [`binarization::decode_mpm_idx`] — drives the truncated-unary
    prefix through the §9.3.4.3.4 bypass decoder; a `0` bin terminates
    at value 0, otherwise a second bin distinguishes 1 from 2. Output
    is always in `0..=2` (the three-entry §8.4.2 candModeList).
  - [`binarization::REM_INTRA_LUMA_PRED_MODE_FL_CMAX`] (= 31) and
    [`binarization::REM_INTRA_LUMA_PRED_MODE_FL_NBITS`] (= 5) — Table
    9-43 FL shape; §9.3.3.5 `Ceil(Log2(cMax + 1)) = Ceil(Log2(32)) = 5`.
  - [`binarization::decode_rem_intra_luma_pred_mode`] — reads the five
    FL bypass bins MSB-first via
    [`cabac::CabacEngine::decode_bypass_bits`]. Output is always in
    `0..=31`.
- §9.3.4.2.5 Table 9-50 `ctxIdxMap[ 15 ]`: the round-32 `sig_coeff_flag`
  path reconstructed this entry `= 8` by pair-symmetry; the staged docs
  errata #93 now formally confirms it (`= 8`; the PDF truncation at
  `i = 15` is a layout artefact). The existing
  [`binarization::SIG_COEFF_FLAG_CTX_IDX_MAP_LOG2_TRAFO_SIZE_2`] already
  carries the confirmed value — no code change required.
- Test count: 396 → 403 (+7 new tests covering the Table 9-43 TR shape
  for `mpm_idx` (`cMax = 2`, `cRiceParam = 0`); the Table 9-43 FL shape
  for `rem_intra_luma_pred_mode` (`cMax = 31`, `Ceil(Log2(32)) = 5`
  `nBits` cross-check); the `mpm_idx` value-0 first-zero-bin path; the
  `mpm_idx` `0..=2` range invariant; the `mpm_idx` at-most-two-bins
  consumption anchor (value 0 ⇒ one bin, value 1/2 ⇒ two bins, with a
  post-read engine-offset cross-check); the `rem_intra_luma_pred_mode`
  five-bypass-bin wrapper-vs-direct agreement (value + engine offset);
  and the `rem_intra_luma_pred_mode` `0..=31` range invariant).

### Added — clean-room rebuild round 40 (2026-06-10)

- §9.3.4.2 / Table 9-48 entry for `prev_intra_luma_pred_flag` lands in
  the [`binarization`] module — the per-luma-prediction-block bit that
  selects, for an intra CU, whether the luma intra prediction mode is
  taken from the §8.4.2 most-probable-mode list (`mpm_idx` follows) or
  from the remaining-mode field (`rem_intra_luma_pred_mode` follows),
  per H.265 §7.3.8.5, §7.4.9.2. Per Table 9-43 the flag is FL with
  `cMax = 1` (a single context-coded bin); Table 9-48's row lists
  `ctxInc = 0` for bin 0 and `na` for every later binIdx column.
  Table 9-12 supplies three ctxIdx slots with
  `initValue = {184, 154, 183}` for initType 0, 1 and 2 — unlike
  `pred_mode_flag`, this element is read in I, P and B slices (intra
  CUs occur in every slice type), so all three initType slots are
  populated. The flag is always present when the §7.3.8.5 intra-PB
  loop reaches it (no inferred-value rule).
  - [`binarization::PREV_INTRA_LUMA_PRED_FLAG_FL_CMAX`] — Table 9-43
    shape: `cMax = 1`.
  - [`binarization::PREV_INTRA_LUMA_PRED_FLAG_FL_NBITS`] — §9.3.3.5
    `Ceil(Log2(cMax + 1))` collapsed to the `cMax = 1` constant `1`.
  - [`binarization::prev_intra_luma_pred_flag_ctx_inc`] — Table 9-48
    bin-0 row: `ctxInc = 0` (three Table 9-12 ctxIdx slots, selected
    at slice-init scope by the Table 9-4 initType-to-ctxIdx mapping).
  - [`binarization::LumaIntraModeSource`] — two-variant enum capturing
    the §7.4.9.2 selection: `Mpm` (flag == 1, §8.4.2 candidate list) /
    `Remaining` (flag == 0, `rem_intra_luma_pred_mode`).
  - [`binarization::luma_intra_mode_source_from_flag`] — folds a
    decoded flag into the enum: `1 ⇒ Mpm`, `0 ⇒ Remaining`.
  - [`binarization::decode_prev_intra_luma_pred_flag`] — engine-driven
    decode primitive that reads the single FL bin using the
    caller-allocated Table 9-12 context and returns the decoded `u8`.
- Test count: 389 → 396 (+7 new `prev_intra_luma_pred_flag` tests
  covering the Table 9-48 `ctxInc = 0` anchor; the Table 9-43 FL shape
  (`cMax = 1`, `Ceil(Log2(2)) = 1` `nBits` cross-check); the
  `luma_intra_mode_source_from_flag` mapping (`1 ⇒ Mpm`,
  `0 ⇒ Remaining`); the `LumaIntraModeSource` variant distinctness
  anchor; the engine-driven decode for the valMps = 0 / valMps = 1
  contexts (with `LumaIntraModeSource` cross-check); and the
  exactly-one-bin-per-invocation anchor across two back-to-back
  contexts on the same engine).

### Added — clean-room rebuild round 39 (2026-06-09)

- §9.3.4.2 / Table 9-48 entry for `pred_mode_flag` lands in the
  [`binarization`] module — the per-CU bit that selects between
  MODE_INTER (value 0) and MODE_INTRA (value 1) inside a P or B
  slice (H.265 §7.3.8.5, §7.4.9.5). Per Table 9-43 the flag is FL
  with `cMax = 1` (a single context-coded bin); Table 9-48's row
  lists `ctxInc = 0` for bin 0 and `na` for every later binIdx
  column. Table 9-10 supplies two ctxIdx slots with
  `initValue = 149` at initType 1 and `initValue = 134` at
  initType 2 (initType 0 is `na` per Table 9-4 — the §7.3.8.5
  `slice_type != I` guard skips the read for I slices entirely).
  When the §7.3.8.5 guard fails the flag is not coded on the wire
  and §7.4.9.5 derives `CuPredMode` directly: I slice ⇒ MODE_INTRA;
  P or B slice with `cu_skip_flag == 1` ⇒ MODE_SKIP.
  - [`binarization::PRED_MODE_FLAG_FL_CMAX`] — Table 9-43 shape:
    `cMax = 1`.
  - [`binarization::PRED_MODE_FLAG_FL_NBITS`] — §9.3.3.5
    `Ceil(Log2(cMax + 1))` collapsed to the `cMax = 1` constant `1`.
  - [`binarization::pred_mode_flag_ctx_inc`] — Table 9-48 bin-0 row:
    `ctxInc = 0` (two Table 9-10 ctxIdx slots, selected at slice-init
    scope by the Table 9-4 initType-to-ctxIdx mapping).
  - [`binarization::CuPredMode`] — three-variant enum capturing the
    §7.4.9.5 mapping: `Inter`, `Intra`, `Skip`. The `Skip` variant is
    reachable only from the not-present inference path on P/B slices.
  - [`binarization::cu_pred_mode_from_flag`] — present-on-wire
    mapping: `0 ⇒ MODE_INTER`, `1 ⇒ MODE_INTRA`.
  - [`binarization::pred_mode_flag_inferred_cu_pred_mode`] — §7.4.9.5
    not-present derivation: `slice_type == I ⇒ MODE_INTRA`;
    `slice_type ∈ {P, B} && cu_skip_flag == 1 ⇒ MODE_SKIP`.
  - [`binarization::decode_pred_mode_flag`] — engine-driven decode
    primitive that reads the single FL bin using the caller-allocated
    Table 9-10 context and returns the decoded `u8`.

### Added — clean-room rebuild round 38 (2026-06-08)

- §9.3.4.2 / Table 9-48 entry for `rqt_root_cbf` lands in the
  [`binarization`] module — the inter-CU gate that signals whether the
  `transform_tree( )` syntax structure follows the current coding unit
  (H.265 §7.3.8.5, §7.4.9.5). Per Table 9-43 the flag is FL with
  `cMax = 1` (a single context-coded bin); Table 9-48's row lists
  `ctxInc = 0` for bin 0 and `na` for every later binIdx column.
  Table 9-14 supplies a single ctxIdx slot with `initValue = 79` at
  both initType 1 and initType 2 (initType 0 is `na` per Table 9-4 —
  `rqt_root_cbf` is only ever read in inter slices). The §7.3.8.5
  guard is `CuPredMode != MODE_INTRA && !cu_skip_flag`: the flag is
  read only when that guard holds, otherwise §7.4.9.5 (V8 / 2021
  baseline) infers the value to 1 (the `transform_tree( )` syntax
  structure is taken to be present).
  - [`binarization::RQT_ROOT_CBF_FL_CMAX`] — Table 9-43 shape:
    `cMax = 1`.
  - [`binarization::RQT_ROOT_CBF_FL_NBITS`] — §9.3.3.5
    `Ceil(Log2(cMax + 1))` collapsed to the `cMax = 1` constant `1`.
  - [`binarization::rqt_root_cbf_ctx_inc`] — Table 9-48 bin-0 row:
    `ctxInc = 0` (single Table 9-14 ctxIdx slot).
  - [`binarization::rqt_root_cbf_inferred`] — §7.4.9.5 inferred-value
    helper (`1` — the spec-mandated default when the §7.3.8.5 guard
    fails and the element is not present on the wire).
  - [`binarization::decode_rqt_root_cbf`] — engine-driven decode
    primitive that reads the single FL bin using the caller-allocated
    single Table 9-14 context and returns the decoded `u8`.

### Added — clean-room rebuild round 37 (2026-06-08)

- §9.3.4.2 / Table 9-48 entry for `cu_transquant_bypass_flag` lands
  in the [`binarization`] module — the per-CU bypass switch that,
  when set, replaces the §8.6 / §8.7 scaling + transform +
  in-loop-filter path with a verbatim residual passthrough (H.265
  §7.3.8.5, §7.4.9.5). Per Table 9-43 the flag is FL with
  `cMax = 1` (a single context-coded bin); Table 9-48's row lists
  `ctxInc = 0` for bin 0 and `na` for every later binIdx column.
  Table 9-8 supplies the single context's `initValue = 154` at all
  three initType slots. The PPS gate is
  `transquant_bypass_enabled_flag` (§7.4.3.3.1): the flag is read
  only when the PPS field is 1, otherwise §7.4.9.5 infers the value
  to 0 (the normal scaling-and-transform path).
  - [`binarization::CU_TRANSQUANT_BYPASS_FLAG_FL_CMAX`] — Table 9-43
    shape: `cMax = 1`.
  - [`binarization::CU_TRANSQUANT_BYPASS_FLAG_FL_NBITS`] — §9.3.3.5
    `Ceil(Log2(cMax + 1))` collapsed to the `cMax = 1` constant `1`.
  - [`binarization::cu_transquant_bypass_flag_ctx_inc`] — Table 9-48
    bin-0 row: `ctxInc = 0` (single Table 9-8 ctxIdx slot).
  - [`binarization::cu_transquant_bypass_flag_inferred`] — §7.4.9.5
    inferred-value helper (`0` — the spec-mandated default when the
    PPS gate is 0 and the element is not present on the wire).
  - [`binarization::decode_cu_transquant_bypass_flag`] — engine-driven
    decode primitive that reads the single FL bin using the
    caller-allocated single Table 9-8 context and returns the
    decoded `u8`.

### Added — clean-room rebuild round 36 (2026-06-07)

- §9.3.4.2 / Table 9-48 entries for the `cu_chroma_qp_offset_flag` /
  `cu_chroma_qp_offset_idx` transform-unit syntax pair land in the
  [`binarization`] module — the per-TU gate that swaps the picture's
  chroma-QP offset (`pps_cb_qp_offset`, `pps_cr_qp_offset`) for an
  entry from the PPS-signalled `cb_qp_offset_list[ ]` /
  `cr_qp_offset_list[ ]` (§7.3.8.11, §7.4.9.10). Both elements have a
  Table 9-48 row whose every context-coded bin column is
  `ctxInc = 0`: the flag is FL `cMax = 1` (one bin); the idx is TR
  `cMax = chroma_qp_offset_list_len_minus1`, `cRiceParam = 0`
  (binIdx 0..=4 — the §7.4.3.3.1 PPS u(3) field bounds the list
  length at 5).
  - [`binarization::CU_CHROMA_QP_OFFSET_FLAG_FL_CMAX`] — Table 9-43
    shape: `cMax = 1`.
  - [`binarization::CU_CHROMA_QP_OFFSET_FLAG_FL_NBITS`] — §9.3.3.5
    `Ceil(Log2(cMax + 1))` collapsed to the `cMax = 1` constant `1`.
  - [`binarization::cu_chroma_qp_offset_flag_ctx_inc`] — Table 9-48
    bin-0 row for the flag: `ctxInc = 0` (Table 9-34 ctxIdx bank).
  - [`binarization::cu_chroma_qp_offset_idx_ctx_inc`] — Table 9-48
    row for every context-coded bin (binIdx 0..=4) of the TR prefix
    of the idx: `ctxInc = 0` (Table 9-35 ctxIdx bank).
  - [`binarization::cu_chroma_qp_offset_idx_tr_cmax`] — Table 9-43
    cMax pass-through: `cMax = chroma_qp_offset_list_len_minus1`.
  - [`binarization::CuChromaQpOffset`] — typed `(flag, idx)` pair
    with `offset_indices()` surfacing the §7.4.9.10 dereference gate
    (`flag == 0` ⇒ no list dereference; `flag == 1` ⇒ index 0 when
    the idx is not signalled per the cMax == 0 fast path).
  - [`binarization::decode_cu_chroma_qp_offset`] — engine-driven
    decode primitive that reads the FL flag bin, then (when the
    flag is 1 and the list has more than one entry) the TR prefix
    of the idx, returning the typed pair.

### Added — clean-room rebuild round 35 (2026-06-07)

- §9.3.4.2 / Table 9-48 `coeff_sign_flag[ n ]` derivation lands in
  the [`binarization`] module — the per-scan-position sign bit that
  pairs with the round-34 `coeff_abs_level_remaining[ n ]` magnitude
  to form the signed transform-coefficient level per §7.4.9.11. The
  element is fully bypass-coded (Table 9-48 marks bin 0 `bypass`, all
  later bin-index columns `na`) and FL binarized with `cMax = 1`
  (Table 9-43), so the on-wire string is exactly one bin per
  invocation.
  - [`binarization::COEFF_SIGN_FLAG_FL_CMAX`] — Table 9-43 shape:
    `cMax = 1`.
  - [`binarization::COEFF_SIGN_FLAG_FL_NBITS`] — §9.3.3.5
    `fixedLength = Ceil(Log2(cMax + 1))` collapsed to the
    `cMax = 1` constant `1`.
  - [`binarization::decode_coeff_sign_flag`] — reads one
    [`CabacEngine::decode_bypass`] bin from the
    post-§9.3.4.3.6-alignment engine state and returns the
    per-scan-position sign bit (`0` ⇒ positive, `1` ⇒ negative per
    §7.4.9.11).
  - [`binarization::signed_level_from_sign_flag`] — composes the
    §7.4.9.11 signed level via the `(1 − 2 * coeff_sign_flag[n])`
    factor: `sign_flag == 0 ⇒ +abs_level`, `sign_flag == 1 ⇒
    −abs_level`. Returns `i32` so the high-bit-depth `|level|` range
    up to `CoeffMax = (1 << 15) − 1` survives composition before the
    §7.4.9.11 / Annex A `[CoeffMin, CoeffMax]` clip.
- The §9.3.4.3.6 alignment process (`ivlCurrRange := 256`) remains
  a slice-data-loop scope responsibility — the per-flag entry point
  expects [`CabacEngine::align`] to have already been invoked at the
  start of the bypass-coded tail of the current transform block.
- Test count: 350 → 359 (+9 new `coeff_sign_flag` tests covering
  the Table 9-43 FL shape + §9.3.3.5 fixedLength derivation
  cross-check; positive-branch identity sweep (sign_flag = 0 across
  5 anchors `{0, 1, 7, 127, 32_767}`); negative-branch identity
  sweep (sign_flag = 1, same anchors); inverse-identity
  (`signed(abs, 0) + signed(abs, 1) == 0` across 9 levels including
  `0 / 1 / 2 / 5 / 17 / 42 / 255 / 1023 / 65535`); high-bit-depth
  `[CoeffMin, CoeffMax]` round-trip (16-bit `|level| = 32_768`
  recovers under sign_flag = 1); the bypass-bin zero-output anchor
  (post-`align()` all-zero stream ⇒ bin 0); the well-typed-output
  anchor (output always in `{0, 1}` regardless of stream contents);
  the wrapper-vs-direct-`decode_bypass` agreement across 8 bins of
  a `5a a5 3c c3` seed (the wrapper is exactly the underlying
  bypass primitive); and the §7.4.9.11 residual-loop composition
  table (`baseLevel ∈ {1, 2, 3}` × `remaining ∈ {0, 1, 5, 17}` ×
  `sign ∈ {0, 1}`, 10 anchors).

### Added — clean-room rebuild round 34 (2026-06-05)

- §9.3.3.11 `coeff_abs_level_remaining[ n ]` Rice-adaptive
  binarization + bypass decode primitive lands in the
  [`binarization`] module (non-persistent path:
  `persistent_rice_adaptation_enabled_flag == 0`,
  `extended_precision_processing_flag == 0`):
  - [`binarization::coeff_abs_level_remaining_c_rice_param_eq_9_24`]
    — eq. 9-24, the per-coefficient `cRiceParam` adaptation:
    `Min(cLastRiceParam + (cLastAbsLevel > (3 << cLastRiceParam)
    ? 1 : 0), 4)`.
  - [`binarization::coeff_abs_level_remaining_c_max_eq_9_26`] —
    eq. 9-26, `cMax = 4 << cRiceParam`.
  - [`binarization::coeff_abs_level_remaining_prefix_val_eq_9_27`]
    — eq. 9-27, `prefixVal = Min(cMax, level)`.
  - [`binarization::coeff_abs_level_remaining_suffix_val_eq_9_28`]
    — eq. 9-28, `suffixVal = level − cMax`.
  - [`binarization::COEFF_ABS_LEVEL_REMAINING_TR_PREFIX_ESCAPE_LEN`]
    — the §9.3.3.2 TR-prefix length, constant 4 once eq. 9-26 is
    substituted.
  - [`binarization::decode_coeff_abs_level_remaining`] — the
    end-to-end driver that runs against the §9.3.4.3.4
    `CabacEngine` bypass stream.
  - [`binarization::decode_coeff_abs_level_remaining_with`] — the
    bin-source-driven core (the algorithm logic factored from the
    engine wrapper) so the §9.3.3.11 derivation can be exercised
    with a flat bin queue.
- §9.3.3.3 EGk decoder helper generalised to arbitrary `k`; the
  former `decode_eg_k0` is preserved as a named alias for the
  `cu_qp_delta_abs` / `palette_escape_val` `k = 0` callers.
- Test count: 331 → 350 (+19 new `coeff_abs_level_remaining` tests
  covering the eq.-9-24 initial state, no-bump-at-threshold across
  `cRiceParam ∈ {0..=4}`, the +1-above-threshold bump, saturation
  at 4, monotone-in-`cLastAbsLevel`; the eq.-9-26 anchor table; the
  §9.3.3.2 TR-prefix-length-is-4 invariant; the eq.-9-27 clamp at
  `cMax`; the eq.-9-28 subtract; the prefix + suffix round-trip
  recomposition across `r ∈ {0..=4}` and 13 level anchors; the
  bin-source decode on zero / short-prefix `r ∈ {0, 1, 2}` /
  escape-path `r = 0` with and without payload; the TR-only
  round-trip across the full 0..cMax range for `r ∈ {0..=4}`; the
  escape-path round-trip across 5 suffix-value anchors × 5 Rice
  parameters; and the engine-wrapper smoke test).

### Added — clean-room rebuild round 33 (2026-06-04)

- §9.3.4.2.8 `palette_run_prefix` ctxInc derivation lands in the
  [`binarization`] module. The §7.3.8.13 palette-coding (SCC) syntax
  element that signals the unary part of a palette-run length now
  has its `ctxInc` mapped via two cases:
  - [`binarization::palette_run_prefix_ctx_inc_eq_9_63`] — eq. 9-63
    branch (`copy_above_palette_indices_flag == 0 && binIdx == 0`):
    `ctxInc = (palette_idx_idc < 1) ? 0 : ((palette_idx_idc < 3) ?
    1 : 2)`. Returns `{0, 1, 2}`.
  - [`binarization::PALETTE_RUN_PREFIX_CTX_IDX_MAP`] — Table 9-51
    `ctxIdxMap[copy_above_palette_indices_flag][binIdx]` verbatim
    for `binIdx ∈ {1, 2, 3, 4}` when `copy_above == 0` (`3, 3, 4, 4`)
    and for `binIdx ∈ {0..=4}` when `copy_above == 1` (`5, 6, 6, 7,
    7`). The `copy_above == 0, binIdx == 0` cell is held as the
    sentinel [`binarization::PALETTE_RUN_PREFIX_EQ_9_63_DISPATCH`]
    because that cell dispatches to eq. 9-63 above.
  - [`binarization::palette_run_prefix_ctx_inc`] — the public entry
    point that dispatches both branches and returns
    `Option<u32>` (`None` ⇒ bypass, signalled when `binIdx >=
    PALETTE_RUN_PREFIX_FIRST_BYPASS_BIN_IDX = 5` per Table 9-51's
    ">4" column and Table 9-48).
- §9.3.3 / Table 9-43 `palette_run_prefix` binarization shape lands
  as [`binarization::palette_run_prefix_tr_cmax`]
  (`cMax = Floor(Log2(PaletteMaxRunMinus1)) + 1`, `cRiceParam = 0`;
  the degenerate `PaletteMaxRunMinus1 == 0` input collapses to a
  single-bin TR terminator).
- Test count: 317 → 331 (+14 new `palette_run_prefix` tests covering
  the eq.-9-63 three bands, both copy-above branches of Table 9-51,
  the ">4" bypass boundary, the `ctxInc ∈ 0..=7` Table 9-40
  context-bank invariant, the `palette_idx_idc`-irrelevance of the
  non-eq.-9-63 branch, the eq.-9-63 / Table 9-51 disjointness
  invariant, plus TR `cMax` anchors at the
  `Floor(Log2(x)) + 1` power-of-two boundaries up to `u32::MAX` and
  monotonicity in `PaletteMaxRunMinus1`).

### Added — clean-room rebuild round 32 (2026-06-04)

- §9.3.4.2.5 `sig_coeff_flag` ctxInc derivation lands in the
  [`binarization`] module. The per-scan-position significance bin
  the §7.3.8.11 residual-coding loop emits before any greater-1 /
  greater-2 step routes through one of four spec branches:
  - [`binarization::sig_coeff_flag_sig_ctx_transform_skip`] — eq.
    9-40 fast path used when `transform_skip_context_enabled_flag`
    is 1 and either `transform_skip_flag[ x0 ][ y0 ][ cIdx ]` or
    `cu_transquant_bypass_flag` is 1. Returns
    `sigCtx = 42` (luma) / `sigCtx = 16` (chroma), position-
    independent.
  - [`binarization::sig_coeff_flag_sig_ctx_log2_2`] — eq. 9-41 for
    the `log2TrafoSize == 2` (4×4) TB case. Reads `sigCtx` from
    [`binarization::SIG_COEFF_FLAG_CTX_IDX_MAP_LOG2_TRAFO_SIZE_2`],
    the 16-entry Table 9-50 lookup
    `[0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8]` indexed by
    `(yC << 2) + xC`.
  - [`binarization::sig_coeff_flag_sig_ctx_dc`] — eq. 9-42 for the
    `xC + yC == 0` DC coefficient on `log2TrafoSize > 2`. Starts
    `sigCtx` at 0 (the eq.-9-43..9-48 neighbour walk is skipped)
    then applies the eq.-9-49..9-53 colour / size / scan-order
    tail.
  - [`binarization::sig_coeff_flag_sig_ctx_general`] — equations
    9-43..9-53 for the general `log2 > 2`, `xC + yC > 0` case.
    Computes `prevCsbf` from the right / below sub-block-flag
    neighbours (edge-gated by `xS / yS < (1 << (log2TrafoSize − 2))
    − 1`), routes through one of equations 9-45 (`prevCsbf == 0`,
    `(xP + yP == 0) ? 2 : (xP + yP < 3) ? 1 : 0`), 9-46
    (`prevCsbf == 1`, `(yP == 0) ? 2 : (yP == 1) ? 1 : 0`), 9-47
    (`prevCsbf == 2`, `(xP == 0) ? 2 : (xP == 1) ? 1 : 0`), 9-48
    (`prevCsbf == 3`, `sigCtx = 2`), then applies the luma-vs-
    chroma tail (eq. 9-49 luma `(xS + yS > 0) → += 3`; eq. 9-50
    luma `log2 == 3` `(scan_idx == 0 ? 9 : 15)`; eq. 9-51 luma
    other-sizes `+= 21`; eq. 9-52 chroma `log2 == 3` `+= 9`; eq.
    9-53 chroma other-sizes `+= 12`).
  - [`binarization::sig_coeff_flag_ctx_inc_from_sig_ctx`] — eq.
    9-54 luma `ctxInc = sigCtx` / eq. 9-55 chroma
    `ctxInc = 27 + sigCtx`, the final per-component offset
    applied to whichever `sigCtx` derivation the caller dispatched.
  - [`binarization::SIG_COEFF_FLAG_FL_CMAX`] = 1 — the Table 9-43
    binarization shape (FL with one context-coded bin per scan
    position).
- Total tests now 317 (was 294). 23 new tests cover: Table 9-50
  verbatim entries + entry-range invariant; eq. 9-40 luma 42 /
  chroma 16; eq. 9-41 at the (0, 0) DC position; the eq. 9-41
  row-major (yC, xC) indexing sweep over the full 4×4 scan space;
  the `(xc, yc) & 3` defensive masking; eq. 9-45 `prevCsbf = 0`
  luma 8×8 across the three DC / mid / edge positions; eq. 9-50
  scan-idx branch (`scan_idx ∈ {1, 2} → += 15`) vs. `scan_idx == 0
  → += 9`; eq. 9-51 luma 16×16 and 32×32; eq.-9-49 sub-block
  offset bump on `(xS, yS) = (1, 0)` luma vs `(0, 0)`; eq. 9-46
  `prevCsbf = 1` row sweep (yP 0 / 1 / 2); eq. 9-47 `prevCsbf =
  2` row sweep (xP 0 / 1 / 2); eq. 9-48 `prevCsbf = 3` position-
  independent across the 4×4 (xP, yP) product; eq. 9-43 / 9-44
  edge-gating on the right edge (xS = max) and bottom edge (yS =
  max); chroma eq. 9-52 / 9-53 tails (with eq. 9-49 luma bump
  inactive on chroma); eq. 9-52 chroma's scan-idx-irrelevance
  (unlike eq. 9-50 luma); DC eq. 9-42 luma 8×8 `scan_idx ∈ {0,
  1}`; DC luma large sizes; DC chroma 8×8 / 16×16; eq. 9-54 luma
  identity over `sigCtx ∈ 0..=44`; eq. 9-55 chroma `+ 27` over
  `sigCtx ∈ 0..=20` plus the transform-skip chroma anchor (16 → 43)
  and transform-skip luma anchor (42 → 42); the FL `cMax = 1`
  shape assertion; an end-to-end luma compose
  `sig_ctx_general → ctx_inc_from_sig_ctx` on `(log2 = 4, xC =
  4, yC = 0)` → 26; and an end-to-end chroma compose
  `sig_ctx_dc → ctx_inc_from_sig_ctx` on `(log2 = 4, DC)` → 39.

### Added — clean-room rebuild round 31 (2026-06-03)

- §9.3.4.2.6 + §9.3.4.2.7 ctxInc derivations for the absolute-level
  greater-than-1 / greater-than-2 flags land in the
  [`binarization`] module. Both elements are Table 9-43 FL with
  `cMax = 1` (one context-coded bin per invocation), but the bin's
  `ctxInc` is driven by a small sub-block-scoped state machine
  (`ctxSet`, `greater1Ctx`, `lastGreater1Ctx`, `lastGreater1Flag`)
  that the §7.3.8.11 residual loop threads from sub-block to
  sub-block within the same transform block.
  - [`binarization::Greater1State`] — the §9.3.4.2.6 walker the
    slice parser carries across the residual sub-blocks of one
    transform block. Implements equations 9-56 (`i == 0 || cIdx > 0
    ⇒ ctxSet = 0`), 9-57 (luma `i > 0 ⇒ ctxSet = 2`), 9-58 (the
    `lastGreater1Ctx == 0 ⇒ ctxSet += 1` bump after the prior
    sub-block's greater-1-ladder mutation), and 9-59 (`ctxInc =
    (ctxSet * 4) + min(3, greater1Ctx)`) + 9-60 (chroma `+ 16`).
    Public step methods: [`Greater1State::new`],
    [`Greater1State::on_subblock_entry`] (start-of-sub-block init
    of `ctxSet` from `(i, is_chroma)` + prior sub-block's
    `last_greater1_flag`),
    [`Greater1State::on_coeff_abs_level_greater1_flag`] (per-bin
    step applying the `lastGreater1Flag = 1 → 0 / = 0 →
    increment-clamped-by-3` rule),
    [`Greater1State::current_ctx_inc`] (eq. 9-59 + 9-60 read for
    the next bin), and [`Greater1State::ctx_set`] (the §9.3.4.2.7
    read of the same sub-block's `ctxSet`).
  - [`binarization::coeff_abs_level_greater2_flag_ctx_inc`] —
    §9.3.4.2.7 eq. 9-61 / 9-62 `ctxInc = ctxSet` for luma /
    `ctxInc = ctxSet + 4` for chroma. Reads the §9.3.4.2.6 walker's
    current `ctxSet` (the per-sub-block value, not the post-update
    one) via [`Greater1State::ctx_set`].
  - [`binarization::COEFF_ABS_LEVEL_GREATER_X_FL_CMAX`] — Table 9-43
    binarization shape constant `= 1` (one context-coded bin per
    invocation of either flag).
- Total tests now 294 (was 280). 14 new tests cover: the
  first-sub-block init (eq.-9-56 `i == 0 ⇒ ctxSet = 0`,
  `greater1Ctx = 1`, luma first-bin `ctxInc = 1` and chroma `+ 16`);
  eq.-9-57 luma `i > 0 ⇒ ctxSet = 2`; eq.-9-56 chroma always-zero
  across `i ∈ {0, 1, 2, 5, 7}`; the per-bin step
  `lastGreater1Flag = 1 ⇒ greater1Ctx = 0` and `= 0 ⇒
  increment-clamped-at-3`, plus the "once at 0, the guard skips
  later updates" invariant; eq.-9-58 non-bump path (prior sub-block
  decoded a `0`-flag, `lastGreater1Ctx` mutates to a positive
  value, ctxSet stays at eq.-9-57's 2); eq.-9-58 bump path (prior
  sub-block ended at `greater1Ctx = 0`, `lastGreater1Ctx` stays 0,
  ctxSet bumps from 2 to 3); chroma `ctxInc + 16` with eq.-9-58
  bump (chroma starts at 0, bumps to 1, chroma `ctxInc = 1 * 4 + 1
  + 16 = 21`); the eq.-9-59 `Min(3, …)` clamp; eq.-9-61 luma
  identity across `ctxSet ∈ {0..=3}`; eq.-9-62 chroma `+ 4` across
  `ctxSet ∈ {0..=3}`; an end-to-end composition showing
  `coeff_abs_level_greater2_flag_ctx_inc(s.ctx_set(), …)` reads the
  same sub-block's ctxSet as the walker holds; and the FL `cMax = 1`
  shape assertion.

### Added — clean-room rebuild round 30 (2026-06-03)

- §9.3.4.2 / Table 9-48 + §9.3.3 / Table 9-43 derivations for the
  §7.3.4 `sao()` per-CTU syntax-element family land in the
  [`binarization`] module. Every element is either a single
  context-coded bin-0 followed by zero or more bypass-coded bins or
  is fully bypass-coded; no neighbour-table walk is needed at this
  layer. The new public surface:
  - [`binarization::sao_merge_flag_ctx_inc`] — Table 9-48 row for
    `sao_merge_left_flag` and `sao_merge_up_flag`: bin 0
    `ctxInc = 0`. Both merge flags share the Table 9-5 context
    bank (single ctxIdx per initType) per Table 9-4.
    [`binarization::SAO_MERGE_FLAG_FL_CMAX`] = 1 captures the FL
    binarization (single bin) from Table 9-43.
  - [`binarization::sao_type_idx_ctx_inc`] — Table 9-48 row for
    `sao_type_idx_luma` and `sao_type_idx_chroma`: bin 0
    `ctxInc = 0`; bin 1 is bypass per Table 9-48 (not routed
    through a context). The TR(`cMax = 2`, `cRiceParam = 0`)
    binarization caps the prefix at two bins, encoding the §7.4.9.3
    `SaoTypeIdx ∈ {0, 1, 2}` (NOT_APPLIED / BAND / EDGE). The two
    variants share the Table 9-6 context bank per Table 9-4.
    [`binarization::SAO_TYPE_IDX_TR_CMAX`] = 2.
  - [`binarization::sao_offset_abs_tr_cmax`] — Table 9-43 row for
    `sao_offset_abs[ ][ ][ ][ ]`:
    `cMax = (1 << min(bitDepth, 10) − 5) − 1` (Min-clamped to 10 by
    the spec to keep the offset range bounded). All bins of the
    TR(`cRiceParam = 0`) prefix are bypass-coded per Table 9-48.
    bitDepth = 8 → 7; bitDepth = 9 → 15; bitDepth >= 10 → 31.
  - [`binarization::SAO_OFFSET_SIGN_FL_CMAX`] = 1,
    [`binarization::SAO_BAND_POSITION_FL_CMAX`] = 31 +
    [`binarization::SAO_BAND_POSITION_FL_NBITS`] = 5, and
    [`binarization::SAO_EO_CLASS_FL_CMAX`] = 3 +
    [`binarization::SAO_EO_CLASS_FL_NBITS`] = 2 — the Table 9-43
    FL-binarization shapes for the three fully-bypass elements
    (`sao_offset_sign`, `sao_band_position`, `sao_eo_class_{luma,
    chroma}`).
- Eleven new binarization tests cover the SAO row (269 → 280
  total): merge-flag ctxInc identity + FL cMax = 1; type-idx
  bin-0 ctxInc + TR cMax = 2; offset-sign FL cMax = 1;
  band-position FL cMax = 31 + 5-bit consistency; eo-class FL
  cMax = 3 + 2-bit consistency; offset-abs cMax derivation at
  8-bit (7), 9-bit (15), 10-bit (31), and the Min-clamp behaviour
  at 11/12/16-bit (all 31); offset-abs monotonicity in bitDepth.

### Added — clean-room rebuild round 29 (2026-06-03)

- §7.3.2.2.1 SPS extension-flag block now decodes typed: when
  `sps_extension_present_flag == 1` the eight bits of the typed
  block are decoded into a new [`sps::SpsExtensionFlags`] struct
  carrying `sps_range_extension_flag` (§A.3.5 RExt-profile entry
  point), `sps_multilayer_extension_flag` (Annex F),
  `sps_3d_extension_flag` (Annex I), `sps_scc_extension_flag`
  (§A.3.7 Screen Content Coding profiles family), and the
  reserved-for-future-use `sps_extension_4bits` group. The opaque
  tail now starts at the first signalled extension body
  (`sps_range_extension()` / `sps_multilayer_extension()` /
  `sps_3d_extension()` / `sps_scc_extension()`) or at the
  `sps_extension_data_flag` while-loop when only
  `sps_extension_4bits` is non-zero. When every flag in the typed
  block is 0 the SPS ends cleanly without an opaque tail (only
  `rbsp_trailing_bits()` follows). [`sps::SpsExtensionFlags`] is
  re-exported from the crate root alongside the existing
  [`SeqParameterSet`] surface; it mirrors the same shape adopted
  for [`pps::PpsExtensionFlags`] in round 25, so RExt / SCC profile
  detection now reads the same way from both parameter sets.
- Four new SPS unit tests cover the typed block end-to-end (265 →
  269 total, plus two pre-existing opaque-tail tests rewritten to
  match the typed contract):
  - `captures_extension_opaque_tail` — single-flag set
    (`sps_range_extension_flag = 1`) lands the
    `sps_range_extension()` body in the opaque tail at the right
    bit position; the four typed flags + `sps_extension_4bits` are
    asserted.
  - `decodes_extension_flag_block_without_bodies` — typed block with
    every flag 0 decodes cleanly, [`SpsExtensionFlags::has_body`]
    returns false, and no opaque tail is surfaced.
  - `captures_scc_extension_opaque_tail` — single-flag set
    (`sps_scc_extension_flag = 1`) selects the §A.3.7 SCC profile
    family; the typed block decodes and `sps_scc_extension()`
    lands in the opaque tail.
  - `captures_extension_data_flag_tail_when_4bits_nonzero` —
    typed flags all 0 with `sps_extension_4bits = 1` still
    surfaces an opaque tail (the §7.3.2.2.1
    `while( more_rbsp_data() ) sps_extension_data_flag` block).
  - `extension_flags_absent_when_gate_zero` — gate 0 leaves
    `extension_flags = None` and no opaque tail.
  - `decodes_vui_then_captures_extension_tail` (rewritten) now
    drives the typed block through the VUI-present path with
    `sps_range_extension_flag = 1` instead of stuffing the eight
    "typed-flag bits" into the opaque tail.

### Added — clean-room rebuild round 28 (2026-06-02)

- Six more §9.3.4.2 / Table 9-48 closed-form `ctxInc` derivations
  land in the [`binarization`] module. Each is a pure function of
  parameters the slice-data parser already has in hand (no
  neighbour-table walk, no CABAC engine drive at this layer):
  - `split_transform_flag[ ][ ][ ]` (§7.3.8.10 / §7.4.9.10) per
    Table 9-48: [`binarization::split_transform_flag_ctx_inc`]
    returns `ctxInc = 5 − log2TrafoSize` for the legal residual-
    quadtree TB sizes `log2TrafoSize ∈ {2, 3, 4, 5}`, mapping into
    the four-context bank `{0, 1, 2, 3}`.
  - `cbf_luma[ ][ ][ ]` (§7.3.8.10 / §7.4.9.10) per Table 9-48:
    [`binarization::cbf_luma_ctx_inc`] returns
    `ctxInc = (trafoDepth == 0) ? 1 : 0`, mapping into the two-
    context bank `{0, 1}` (root of the residual quadtree on ctx 1,
    deeper depths on ctx 0).
  - `cbf_cb[ ][ ][ ]` and `cbf_cr[ ][ ][ ]` (§7.3.8.10 / §7.4.9.10)
    per Table 9-48: [`binarization::cbf_cb_ctx_inc`] and
    [`binarization::cbf_cr_ctx_inc`] both return
    `ctxInc = trafoDepth` (the shared
    [`binarization::cbf_chroma_ctx_inc`] helper). Cb and Cr each
    have their own `ctxIdxOffset` (Table 9-4); this layer hands
    back the bank-relative `ctxInc` only.
  - `inter_pred_idc[ x0 ][ y0 ]` (§7.3.8.6 / §7.4.9.6) per Table
    9-48: [`binarization::inter_pred_idc_ctx_inc`] returns bin 0
    `ctxInc = (nPbW + nPbH != 12) ? CtDepth[x0][y0] : 4` and bin 1
    `ctxInc = 4`. The `nPbW + nPbH == 12` condition picks out the
    8×4 and 4×8 PUs (luma area 16 samples), which are encoded with
    the bin-0 escape onto the bin-1 context bank.
  - `log2_res_scale_abs_plus1[ c ]` (§7.3.8.13 / §7.4.9.13) per
    Table 9-48:
    [`binarization::log2_res_scale_abs_plus1_ctx_inc`] returns
    `ctxInc = 4*c + binIdx` for `binIdx ∈ {0, 1, 2, 3}` and
    `c ∈ {0, 1}` (Cb / Cr), mapping into per-component banks
    `{0, 1, 2, 3}` and `{4, 5, 6, 7}` of the TR(`cMax = 4`) prefix.
  - `res_scale_sign_flag[ c ]` (§7.3.8.13 / §7.4.9.13) per Table
    9-48: [`binarization::res_scale_sign_flag_ctx_inc`] returns
    `ctxInc = c`, one bit per chroma component each on its own
    context.
- 14 new binarization unit tests (251 → 265 total): Table 9-48 row
  for `split_transform_flag` across log2TrafoSize 2..=5 plus the
  `ctxInc <= 3` bank-bound sweep; `cbf_luma` Table 9-48 row across
  trafoDepth 0..=4 plus the `ctxInc <= 1` bank-bound sweep; shared
  `cbf_chroma_ctx_inc` identity across trafoDepth 0..=4 and the
  Cb/Cr agreement check; `inter_pred_idc` bin 0 with `CtDepth`
  routing (64×64, 16×16 at depth 2, 8×8 at depth 3), bin 0 escape
  on the 8×4 / 4×8 16-sample PUs, bin 1 constant `ctxInc = 4`, and
  the `ctxInc ∈ {0..=4}` bank-bound sweep across eight PU shapes
  and four CtDepths; `log2_res_scale_abs_plus1` Cb bank and Cr bank
  identities (`4*c + binIdx`) plus the Cb/Cr disjoint-bank
  invariant; `res_scale_sign_flag` two-row identity.

### Added — clean-room rebuild round 27 (2026-06-01)

- Three more §9.3.4.2 ctxInc derivations land in the
  [`binarization`] module, all pure-functional given their neighbour /
  sub-block context (no CABAC engine drive at this layer — callers
  compose the engine call themselves):
  - `coded_sub_block_flag` (§7.3.8.11 / §7.4.9.11) per §9.3.4.2.4
    equations 9-35..9-39:
    [`binarization::coded_sub_block_flag_ctx_inc`] takes
    `(is_chroma, right_neighbour, below_neighbour)` and returns
    `ctxInc = Min(csbfCtx, 1)` for luma (bank `{0, 1}`, equation 9-38)
    or `2 + Min(csbfCtx, 1)` for chroma (bank `{2, 3}`, equation
    9-39), where `csbfCtx` is the unsigned sum of the two previously
    decoded sub-block-flag neighbours.
    [`binarization::coded_sub_block_flag_ctx_inc_with_edge`] applies
    the equation 9-36 / 9-37 edge gates `xS < (1 << (log2TrafoSize −
    2)) − 1` / `yS < (1 << (log2TrafoSize − 2)) − 1` (the right /
    bottom sub-block-edge zero-outs) before delegating.
  - `split_cu_flag` (§7.3.8.4 / §7.4.9.4) and `cu_skip_flag` (§7.3.8.5 /
    §7.4.9.5) ctxInc derivations per §9.3.4.2.2 Table 9-49:
    [`binarization::left_above_ctx_inc`] implements the shared row
    shape `ctxInc = (condL && availableL) + (condA && availableA)`;
    [`binarization::split_cu_flag_cond`] returns the `split_cu_flag`
    per-neighbour predicate `CtDepth[xNb][yNb] > cqtDepth`;
    [`binarization::cu_skip_flag_cond`] returns the `cu_skip_flag`
    per-neighbour predicate `cu_skip_flag[xNb][yNb]`; the two row
    specialisations [`binarization::split_cu_flag_ctx_inc`] and
    [`binarization::cu_skip_flag_ctx_inc`] compose the per-neighbour
    cond with the availability AND and produce `ctxInc ∈ {0, 1, 2}`
    directly. Both row specialisations honour the §6.4.1 availability
    contract: an unavailable neighbour contributes 0 to `ctxInc` even
    when its cond would otherwise be true.
- 17 new binarization unit tests (234 → 251 total): §9.3.4.2.4 luma
  no-neighbours / one-neighbour / both-neighbours `Min` clamp; chroma
  `+2` offset across the same four input combinations; high-bit-mask
  defensive input; with-edge gating at 4×4 / 8×8 / 16×16 / 32×32 TBs
  (right and bottom edges drop their neighbours, luma + chroma); the
  §9.3.4.2.2 `(condL && availableL) + (condA && availableA)` truth
  table; the unavailability zero-out branch; `split_cu_flag_cond`
  strict-inequality table; `cu_skip_flag_cond` LSB-mask; four-way
  `split_cu_flag_ctx_inc` table (both deeper / left deeper / left
  unavailable / both unavailable); eight-way `cu_skip_flag_ctx_inc`
  truth table; and a bounded `ctxInc ∈ {0, 1, 2}` invariant sweep
  over a small Cartesian product of inputs.

### Added — clean-room rebuild round 26 (2026-05-31)

- New [`binarization`] module implementing the §9.3.4.2
  per-syntax-element binarization + context-index derivation layer for
  the two CABAC elements unblocked by the clean-room trace
  `docs/video/h265/fixtures/main-422-10bit/cabac-cu-qp-delta-last-sig-trace.md`:
  - `cu_qp_delta_abs` / `cu_qp_delta_sign_flag` (§7.3.8.14 / §7.4.9.14)
    via [`binarization::decode_cu_qp_delta`] and the per-bin ctxInc
    table [`binarization::cu_qp_delta_abs_ctx_inc`] (Table 9-32: bin 0
    → ctx 0, bins 1..=4 → ctx 1). Binarization is §9.3.3.10 TR with
    `cMax = 5`, `cRiceParam = 0`, followed by an EGk(k=0) suffix when
    the prefix is the all-ones escape, plus the bypass-coded sign
    flag. The decoded [`binarization::CuQpDelta`] surfaces the
    §7.4.9.14 `CuQpDeltaVal = cu_qp_delta_abs * (1 − 2 *
    cu_qp_delta_sign_flag)` derivation.
  - `last_sig_coeff_{x,y}_{prefix,suffix}` (§7.3.8.11 / §7.4.9.11) via
    [`binarization::decode_last_sig_coeff`] plus the
    [`binarization::last_sig_coeff_prefix_ctx_offset_shift`] derivation
    (§9.3.4.2.3 luma `ctxOffset = 3*(log2TrafoSize − 2) + ((log2TrafoSize
    − 1) >> 2)`, `ctxShift = (log2TrafoSize + 1) >> 2`; chroma
    `ctxOffset = 15`, `ctxShift = log2TrafoSize − 2`) and
    [`binarization::last_sig_coeff_prefix_ctx_inc`] (`ctxInc =
    (binIdx >> ctxShift) + ctxOffset`). The prefix `cMax =
    (log2TrafoSize << 1) − 1` ([`binarization::last_sig_coeff_prefix_cmax`])
    bounds the TR length; the suffix is a `nBits = (prefix >> 1) − 1`
    bypass-coded fixed-length field present only when `prefix > 3`
    ([`binarization::last_sig_coeff_suffix_n_bits`]). The §7.4.9.11
    equations 7-74..7-77 position derivation lives in
    [`binarization::last_sig_coeff_position`], returning
    `LastSignificantCoeff{X,Y}` from `(prefix, optional suffix)`
    pre-scanIdx-2 swap. A [`binarization::LastSigCoeffBank`] tag (X / Y)
    is exposed for caller-side context-bank routing.
- §9.3.3.10 TR-prefix and §9.3.3.11 EGk(k=0) helpers ship as internal
  building blocks; the module sits one layer above the §9.3 arithmetic
  engine ([`cabac::CabacEngine`], round 11) and consumes context
  variables ([`cabac::ContextModel`]) supplied by the caller (the
  slice-data parser, when it lands).
- 16 new binarization unit tests (218 → 234 total): cu_qp_delta ctxInc
  table (Table 9-32 spot-check); last_sig_coeff offset/shift for luma
  log2 = 2..=5; same for chroma; ctxInc-from-binIdx parametrised over
  the 32×32-luma and 4×4-luma rows; `cMax` per log2 size; equation
  7-74 position derivation across the trace-observed luma 32×32 (px=6,
  LastX=8) + 16×16 chroma rows; suffix-nBits table; TR-prefix
  terminator and all-ones escape; cu_qp_delta_abs = 0 path (no sign
  flag, value = 0); §7.4.9.14 signed-value derivation over the 10
  multi-slice-per-frame trace rows; EGk(k=0) decoding driven through
  a crafted engine offset.

### Added — clean-room rebuild round 25 (2026-05-30)

- §7.3.2.3.1 PPS extension-flag block: new typed
  [`pps::PpsExtensionFlags`] sub-struct exposing
  `pps_range_extension_flag`, `pps_multilayer_extension_flag`,
  `pps_3d_extension_flag`, `pps_scc_extension_flag`, and
  `pps_extension_4bits` decoded from the eight bits that follow
  `pps_extension_present_flag == 1`. The new
  [`PicParameterSet::extension_flags`] field carries it (an
  `Option<PpsExtensionFlags>`; `None` when the gate is absent, every
  flag inferred to 0 per §7.4.3.3.1).
- [`PpsExtensionFlags::has_body`] predicate — true when at least one
  of the four extension flags is set or `pps_extension_4bits != 0`
  (i.e. when an extension body follows in the bit stream and the PPS
  therefore carries an opaque tail starting at the first body's bit
  position).
- Opaque-tail capture for the PPS now starts at the first signalled
  extension body's bit position rather than at the
  `pps_extension_present_flag` boundary; when every flag is zero the
  tail is `None` because only `rbsp_trailing_bits()` remain (consumed
  implicitly). The individual extension-body syntax structures
  (`pps_range_extension()` §7.3.2.3.2, `pps_multilayer_extension()`
  Annex F, `pps_3d_extension()` Annex I, `pps_scc_extension()`, and
  the `pps_extension_data_flag` while-loop) remain inside the opaque
  tail and are not yet decoded.
- Tests: four new pps tests
  (`decodes_extension_flag_block_without_bodies`,
  `captures_range_extension_opaque_tail`,
  `captures_extension_data_flag_tail_when_4bits_nonzero`,
  `extension_flags_absent_when_gate_zero`). The prior
  `captures_extension_opaque_tail` is replaced by the
  no-body-flag-block test since the all-zero flag block no longer
  surfaces an opaque tail. Total test count 218 (was 215).

## [0.0.8](https://github.com/OxideAV/oxideav-h265/releases/tag/v0.0.8) - 2026-05-30

### Other

- §7.4.8 inter-RPS-prediction derivation + in-place wiring
- §7.3.6.2 ref_pic_lists_modification() in-place wiring at the §7.3.6.1 call site
- §7.3.6.1 entry-point-offset per-i values + §7.4.7.1 range check
- §7.3.6.3 pred_weight_table() in-place wiring at the §7.3.6.1 call site
- §7.3.6.1 inter five_minus_max_num_merge_cand + full inter tail walk
- §7.3.6.1 inter mvd / cabac-init / collocated block (no-RPLM path)
- §7.3.6.1 inter-slice num_ref_idx_active_override prelude
- §7.3.6.3 pred_weight_table() standalone parser
- §7.4.7.2 NumPicTotalCurr derivation (round 16)
- §7.3.6.2 ref_pic_lists_modification() standalone parser
- §E.2.1 vui_parameters() typed decode into the SPS
- §E.2.2 / §E.2.3 hrd_parameters() + sub_layer_hrd_parameters() bodies
- §7.3.2.1 VPS tail — layer-set inclusion matrix + timing-info block
- §9.3 CABAC arithmetic decoding engine (DecodeDecision/Bypass/Terminate + context model)
- §6.5.4/6.5.5/6.5.6 horizontal/vertical/traverse scans + §7.4.2 ScanOrder accessor
- §6.5.3 up-right diagonal scan + §7.4.5 ScalingFactor derivation
- §7.3.4 scaling_list_data() parse + §7.4.5 ScalingList derivation
- round 7: §7.3.6.1 non-IDR POC + reference-picture-set block
- round 6: §7.3.6.1 slice-segment-header structural parse
- round 5: §7.3.2.3.1 PPS parse + BitReader::se()
- round 4: §7.3.2.2 SPS tail — PCM / RPS / long-term ref / MVP / smoothing / opaque VUI+ext
- round 3: §7.3.2.2 SPS structural parse up to SAO-enabled flag
- round 2: §7.3.2.1 VPS structural parse + §7.3.3 profile-tier-level walk
- round 1: Annex B NAL walker + §7.3.1.2 header parse
- orphan rebuild: clean-room scaffold post 2026-05-18 audit

### Added — clean-room rebuild round 24 (2026-05-30)

- §7.4.8 inter-RPS-prediction derivation as the new typed builder
  [`ShortTermRefPicSet::materialize`] and the post-derivation form
  [`MaterializedShortTermRefPicSet`]. The explicit-form branch
  implements equations 7-63..7-70: `NumNegativePics =
  num_negative_pics`, `NumPositivePics = num_positive_pics`,
  `UsedByCurrPicS{0,1}[i] = used_by_curr_pic_s{0,1}_flag[i]`, and the
  cumulative `DeltaPocS0[i] = DeltaPocS0[i-1] - (delta_poc_s0_minus1[i]
  + 1)` / `DeltaPocS1[i] = DeltaPocS1[i-1] +
  (delta_poc_s1_minus1[i] + 1)` recurrences with the equation-7-67 /
  7-68 first-element seeds. The inter-RPS-prediction branch implements
  equations 7-60 (`deltaRps = (1 - 2*delta_rps_sign) *
  (abs_delta_rps_minus1 + 1)`), 7-61 (negative-side reconstruction —
  source-positives in reverse, optional `deltaRps` self-term when
  negative, then source-negatives in forward order), and 7-62
  (positive-side, mirrored), running each surviving entry through its
  `use_delta_flag[j]` gate. The per-position
  `used_by_curr_pic_flag` / `use_delta_flag` array lengths are checked
  against the source RPS's `NumDeltaPocs[RefRpsIdx] + 1` and a
  mismatch raises
  [`ShortTermRefPicSetMaterializeError::SourceLengthMismatch`]; an
  absent source for an inter-form RPS raises
  [`ShortTermRefPicSetMaterializeError::MissingSource`].
- [`SeqParameterSet::materialize_short_term_ref_pic_sets`] runs the
  full SPS-level chain, materialising each entry in order and feeding
  inter-form entries their source from prior materialised entries via
  the equation-7-59 `RefRpsIdx = stRpsIdx - (delta_idx_minus1 + 1)`
  lookup. The output is exposed as
  `Vec<MaterializedShortTermRefPicSet>` aligned 1:1 with the
  SPS-resident `short_term_ref_pic_sets[]`.
- §7.3.6.1 slice parser: the previously-deferred SPS / inline
  inter-RPS-prediction branch at the `ref_pic_lists_modification()`
  gate now resolves through the new derivation. The slice parser
  materialises the SPS list once, picks the active RPS (inline source
  via `RefRpsIdx = num_short_term_ref_pic_sets - (delta_idx_minus1 +
  1)` for the inline-inter case), feeds the derived
  `UsedByCurrPicS{0,1}` slices into
  [`NumPicTotalCurrInputs::from_used_flags`] / `compute`, and then
  walks the in-place RPLM gate exactly as the explicit-form path did
  in round 23. Configurations whose materialisation succeeds reach
  `byte_alignment()` end to end; only malformed inter-form chains
  (e.g. on-wire `used_by_curr_pic_flag` length not matching the
  source's `NumDeltaPocs + 1`) defer to an opaque tail at the RPLM
  bit. With this round the only remaining parser-side §7.3.6.1
  deferral is the malformed-inter-RPS-prediction fallback; every
  conformant non-IDR P / B slice — explicit or inter-form short-term
  RPS — parses end to end through `byte_alignment()`.
- New unit tests (5 in `sps`, 1 in `slice`; total 215, was 208):
  `materialize_explicit_form_recurrence` (equations 7-67..7-70 with a
  three-negative / two-positive RPS),
  `materialize_inter_rps_prediction_matches_fixture` (re-uses the
  existing `parses_inter_rps_prediction` wire fixture with a hand-
  traced expected output),
  `materialize_inter_rps_prediction_negative_delta_rps` (deltaRps =
  -2 with a single source positive — exercises the negative-side
  source-positives-reverse + deltaRps-self-term branches of equation
  7-61), `materialize_inter_rps_rejects_missing_source` and
  `materialize_inter_rps_rejects_length_mismatch`,
  `sps_materialize_chains_inter_rps_prediction` (SPS-level chain on
  the same fixture verifying both entries materialise correctly), and
  `parses_p_slice_with_sps_inter_predicted_rps_npc_le_1` (slice-level
  test exercising the new wiring: a P-slice with an SPS-form
  inter-predicted RPS materialises, `NumPicTotalCurr == 0` makes the
  RPLM gate statically false, and the parser walks the inter-slice
  tail to `byte_alignment()` without surfacing an opaque tail). The
  pre-round `defers_rplm_when_active_st_rps_uses_inter_prediction`
  test is preserved with an updated header that describes the
  malformed-array defer path more precisely.

### Added — clean-room rebuild round 23 (2026-05-29)

- §7.3.6.2 `ref_pic_lists_modification()` decoded **in place** at the
  §7.3.6.1 slice-header call site when the §7.4.7.2 `NumPicTotalCurr`
  derivation can be resolved without running the §7.4.8
  inter-RPS-prediction step. The wiring covers two configurations:
  * inline-form short-term RPS (`short_term_ref_pic_set_sps_flag ==
    0`) — per §7.4.8, at `stRpsIdx == num_short_term_ref_pic_sets` the
    `inter_ref_pic_set_prediction_flag` is signalled only when
    `num_short_term_ref_pic_sets > 0`; when it is `0` (or signalled
    `0`) the per-position `used_by_curr_pic_s{0,1}_flag` arrays are
    consumed directly by equation 7-57;
  * SPS-form short-term RPS (`short_term_ref_pic_set_sps_flag == 1`)
    whose picked entry has `inter_ref_pic_set_prediction_flag == 0`
    — the SPS-resident explicit arrays are likewise consumed
    directly.
  The §7.4.7.1 long-term-block resolver
  (`SliceLongTermRefPic::used_by_curr_pic_lt`, round 14) supplies the
  `UsedByCurrPicLt[i]` slice. When `NumPicTotalCurr > 1` the parser
  calls the standalone `RefPicListsModification::parse` (round 15)
  and exposes the result as
  `SliceSegmentHeader::ref_pic_lists_modification` (an
  `Option<RefPicListsModification>`); when `NumPicTotalCurr <= 1` the
  §7.3.6.1 gate is statically false and the parser skips the structure
  and continues into the mvd / cabac-init / collocated block. The
  active short-term RPS in inter-RPS-predicted form still defers — the
  §7.4.8 derivation chain is the next blocker — and the opaque tail
  in that case begins at the `ref_pic_lists_modification()` bit
  position.
- New unit tests covering the new wiring:
  `parses_rplm_in_place_with_explicit_inline_rps_npc_two` (inline
  explicit RPS with `NumPicTotalCurr == 2`, RPLM decoded in place
  with one `list_entry_l0` entry one bit wide),
  `skips_rplm_when_num_pic_total_curr_is_one` (inline explicit RPS
  with `NumPicTotalCurr == 1`, gate statically false, RPLM skipped),
  `defers_rplm_when_active_st_rps_uses_inter_prediction` (SPS-form
  RPS with `inter_ref_pic_set_prediction_flag == 1`, opaque tail
  retained), and an updated
  `skips_rplm_when_num_pic_total_curr_is_zero_idr` that replaces the
  pre-round defer-on-flag test (an IDR slice with
  `lists_modification_present_flag == 1` now walks the full inter
  tail because the non-IDR POC/RPS block is absent and
  `NumPicTotalCurr == 0`).
- New field `SliceSegmentHeader::ref_pic_lists_modification:
  Option<RefPicListsModification>`. `None` for I slices, dependent
  slice segments, headers whose parse stopped before this point, the
  inter-RPS-predicted defer case, and the statically-false-gate case
  (`NumPicTotalCurr <= 1` or `pps.lists_modification_present_flag ==
  0`).

### Added — clean-room rebuild round 22 (2026-05-29)

- §7.3.6.1 entry-point-offset block: the slice-header parser now
  captures the per-i `entry_point_offset_minus1[i]` (`u(offset_len_minus1
  + 1)`) values into [`EntryPointOffsets::entry_point_offset_minus1`]
  (a `Vec<u32>`) instead of skipping them. The per-subset byte length
  of §7.4.7.1 (`entry_point_offset_minus1[i] + 1`) is exposed via
  [`EntryPointOffsets::subset_length`]. The struct loses its `Copy`
  bound (it now owns a `Vec`).
- §7.4.7.1 range check on `num_entry_point_offsets`: the on-wire value
  is now bounded by the active PPS partitioning
  (`NumTileColumns * NumTileRows − 1` for tiles, `PicHeightInCtbsY −
  1` for WPP, with the `tiles + WPP` combination — already barred by
  §7.4.3.3.1 — taking the wider of the two as a defensive cap). A
  breaching wire value raises a `ValueOutOfRange { field:
  "num_entry_point_offsets", got }` (`SliceError`).
- New unit tests: `parses_wpp_entry_point_offsets_in_place` (two
  per-row offsets `{6, 9}` captured verbatim, subset lengths
  `{7, 10}`), `parses_tiles_block_with_single_tile_no_offsets`
  (`num_entry_point_offsets == 0` honored when the §7.4.7.1 bound is
  0), `rejects_wpp_entry_point_offsets_above_pic_height_bound` (16×16
  WPP → bound 0, wire `1` rejected), `rejects_offset_len_minus1_above_31`
  (a wire codeNum 32 rejected).

### Added — clean-room rebuild round 21 (2026-05-27)

- §7.3.6.3 `pred_weight_table()` decoded **in place** at the §7.3.6.1
  slice-header call site (closing the last r20 deferral point for the
  universal base-profile single-layer case). When the §7.3.6.1 outer
  gate is statically present
  (`(pps.weighted_pred_flag && slice_type == P)` or
  `(pps.weighted_bipred_flag && slice_type == B)`), the parser
  constructs a `PredWeightTableInputs::base_profile` from the
  post-override `num_ref_idx_lX_active_minus1`, the SPS-derived
  `ChromaArrayType` (per §7.4.2.2: `chroma_format_idc` unless
  `separate_colour_plane_flag == 1`), and the SPS bit depths, then
  invokes the standalone [`PredWeightTable::parse`] (round 17) and
  continues through the rest of the inter-slice tail to
  `byte_alignment()`. The base-profile constructor treats every per-i
  §7.3.6.3 outer-gate decision
  (`pic_layer_id != nuh_layer_id ||
  PicOrderCnt(RefPicListX[i]) != PicOrderCnt(CurrPic)`) as `true`,
  which is the universal correct value for any single-layer slice:
  every active reference in a single-layer stream is an earlier-POC
  temporal picture (i.e. a different picture). The per-i gate slots
  stay open on `PredWeightTableInputs` for the eventual SCC
  self-reference / inter-layer ref-layer cases, which will be threaded
  through this call site once the SPS multilayer / SCC extensions are
  surfaced (currently they are surfaced as opaque tails). A new
  `SliceSegmentHeader::pred_weight_table: Option<PredWeightTable>`
  field exposes the decoded table (`None` for I slices, dependent
  slice segments, for headers whose parse stopped at a prior
  deferral, and when the gate is statically absent). §7.4.7.3 range
  failures inside the in-place parse propagate directly out of
  `SliceSegmentHeader::parse` as `SliceError::ValueOutOfRange`.

- With the in-place call site wired up, every weighted-pred-gated
  P / B independent slice segment in the crate's currently surfaced
  configuration (no SPS range / multilayer / SCC extensions) now
  parses end to end through `byte_alignment()`. The
  `SliceSegmentHeader::opaque_tail` deferral remains only for the
  `pps.lists_modification_present_flag == 1` path, where the
  §7.3.6.2 `ref_pic_lists_modification()` body still needs the
  §7.4.7.2 `NumPicTotalCurr` derivation threaded through the slice
  parser (the next round's target).

### Changed — clean-room rebuild round 21 (2026-05-27)

- The eight pre-round `slice::tests` units that exercised the
  post-override walk with `pps.weighted_pred_flag = true` or
  `pps.weighted_bipred_flag = true` are updated to consume their
  `pred_weight_table()` bodies in place (minimal "all flags off"
  payloads sized per the active `num_ref_idx_lX_active_minus1`)
  and assert `opaque_tail.is_none()`. The deferred-at-PWT-gate
  scenario no longer exists for these tests.

- Three new `slice::tests` units cover the in-place behaviour: the
  universal base-profile P-slice walk with a non-trivial
  `delta_luma_weight_l0` (verifies the §7.4.7.3 derived
  `LumaWeightL0[0] = (1 << 2) + 5 = 9` via
  `PredWeightTable::luma_weight_l0`); a B-slice walk with an L1
  chroma sub-block + non-trivial `delta_chroma_weight_l1` /
  `delta_chroma_offset_l1` (verifies the equation 7-58
  `ChromaOffsetL1[0][j]` derivation with `WpOffsetHalfRangeC = 128`);
  and a `delta_luma_weight_l0 = 128` range-failure propagation test
  (the in-place call site surfaces the same
  `SliceError::ValueOutOfRange` as the standalone parser).

### Added — clean-room rebuild round 20 (2026-05-27)

- §7.3.6.1 inter-slice `five_minus_max_num_merge_cand` (`ue(v)`) decoded
  in place when the §7.3.6.3 `pred_weight_table()` gate is statically
  absent, i.e. when neither `(pps.weighted_pred_flag && slice_type == P)`
  nor `(pps.weighted_bipred_flag && slice_type == B)` holds. The wire
  value is range-checked at 0..=4 (the derived `MaxNumMergeCand =
  5 - five_minus_max_num_merge_cand` must lie in 1..=5 per §7.4.7.1
  equation 7-53). A new `SliceSegmentHeader::five_minus_max_num_merge_cand`
  field surfaces the raw value, and a new `max_num_merge_cand()`
  accessor returns the derived value. The SCC `use_integer_mv_flag`
  (gated on `motion_vector_resolution_control_idc == 2`) is statically
  absent because the PPS SCC extension is not yet surfaced (§7.4.7.1:
  when not present, `motion_vector_resolution_control_idc` is inferred
  to 0). With the merge-candidate leaf landed and the SCC integer-MV
  bit statically absent, the parser now walks the entire inter-slice
  header through the shared I-slice tail — `slice_qp_delta` (`se(v)`),
  the chroma QP offsets, the deblocking override block, the
  loop-filter-across-slices flag, the entry-point-offset block, the
  slice-segment-header extension block, and `byte_alignment()` — and
  reports a non-`None` `byte_offset_to_slice_data`, with
  `opaque_tail == None`. When the weighted-pred gate IS statically
  present (either of the two conditions above holds), the parser keeps
  deferring at the gate, the four `mvd_l1_zero_flag` /
  `cabac_init_flag` / `collocated_from_l0_flag` / `collocated_ref_idx`
  fields stay populated as in round 19, and `opaque_tail` captures the
  bit position of the `pred_weight_table()` block. Three new
  `slice::tests` units cover the full P-slice walk through
  `byte_alignment()`, the full B-slice walk with temporal MVP +
  collocated_ref_idx, and the `five_minus_max_num_merge_cand > 4` range
  failure. The six pre-round inter-slice tests that exercised the
  mvd / cabac / collocated walk are updated to set
  `pps.weighted_pred_flag = true` (P) or `pps.weighted_bipred_flag =
  true` (B) so they continue to assert the defer-at-weighted-pred
  behaviour now that the no-weighted-pred path walks through.

### Added — clean-room rebuild round 19 (2026-05-26)

- §7.3.6.1 inter-slice `mvd_l1_zero_flag` / `cabac_init_flag` /
  `collocated_from_l0_flag` / `collocated_ref_idx` block decoded
  in-place when `pps.lists_modification_present_flag == 0` (the
  outer `if(... && NumPicTotalCurr > 1)` short-circuit makes the
  `ref_pic_lists_modification()` block statically absent, so the
  DPB-derived §7.4.7.2 `NumPicTotalCurr` is not yet needed). Four
  new `Option`-typed fields on `SliceSegmentHeader` carry the
  values plus the §7.4.7.1 inferences:
  `mvd_l1_zero_flag` (B slices only); `cabac_init_flag` (signalled
  iff `pps.cabac_init_present_flag == 1`, else inferred `false`);
  `collocated_from_l0_flag` (signalled for B + `mvp`, else inferred
  `true` for P + `mvp`); `collocated_ref_idx` (signalled when the
  active list has > 1 entry, range-checked against
  `num_ref_idx_lX_active_minus1`, else inferred `0`). The deferred
  P/B opaque tail now begins at the §7.3.6.1 weighted-pred-table
  gate (`pred_weight_table()` when `weighted_pred_flag` /
  `weighted_bipred_flag` applies). When
  `pps.lists_modification_present_flag == 1` the parser still
  defers at the `ref_pic_lists_modification()` gate (its
  `NumPicTotalCurr` derivation needs §7.4.8 inter-RPS-prediction
  resolution that this round does not wire in). Seven new
  `slice::tests` units cover the no-mvp B-slice mvd walk, the
  P-slice cabac-init walk, the P-slice mvp + inferred
  `collocated_from_l0_flag` paths (single-ref inferred `ref_idx`
  and multi-ref signalled `ref_idx`), the B-slice
  `collocated_from_l0_flag == 0` L1 path, the
  `collocated_ref_idx > num_ref_idx_lX_active_minus1` range
  failure, and the `lists_modification_present_flag == 1` defer
  path. The pre-round `defers_pb_ref_list_body` test is updated
  to assert the §7.4.7.1 `cabac_init_flag` inference (`Some(false)`)
  and the absent-collocated state in addition to the existing
  opaque-tail bit position.

### Added — clean-room rebuild round 18 (2026-05-26)

- §7.3.6.1 inter-slice prelude decoded in place: the
  `num_ref_idx_active_override_flag` `u(1)` and (when set) the
  `num_ref_idx_l0_active_minus1` `ue(v)` and (B slices only)
  `num_ref_idx_l1_active_minus1` `ue(v)` values now sit on the
  `SliceSegmentHeader` as three new `Option`-typed fields. The §7.4.7.1
  inference rule fills the per-list values from the PPS defaults when
  the override flag is 0; explicit values are range-checked at 0..=14.
  The deferred P/B opaque tail now begins immediately after the
  override block (at the `ref_pic_lists_modification()` gate when
  signalled, otherwise at `mvd_l1_zero_flag`), so a future round that
  threads the §7.3.6.2 + §7.4.7.2 + §7.3.6.3 pieces in place starts
  from the right bit position with the correct
  `num_ref_idx_lX_active_minus1` values already in hand. Four new
  unit tests in `slice::tests` cover the inferred-defaults P-slice
  path, the explicit-L0-only P-slice path, the B-slice
  `[L0, L1]` explicit path, the B-slice inferred-defaults path, and
  the `num_ref_idx_l0_active_minus1 > 14` range failure. The pre-round
  `defers_pb_ref_list_body` test is rewritten to encode the new
  override = 0 bit and check the now-correct opaque-tail start.

### Added — clean-room rebuild round 17 (2026-05-26)

- §7.3.6.3 `pred_weight_table()` syntax structure as a new standalone
  parser ([`slice::PredWeightTable`]). The parser takes a
  [`slice::PredWeightTableInputs`] descriptor carrying the active
  `slice_type`, the post-override `num_ref_idx_lX_active_minus1`
  cardinalities, the SPS's `ChromaArrayType` + bit depths and the
  range-extension `high_precision_offsets_enabled_flag`, plus per-i
  override slices for the §7.3.6.3 outer-gate (`pic_layer_id !=
  nuh_layer_id || PicOrderCnt(RefPicListX[i]) != PicOrderCnt(CurrPic)`)
  decision. [`slice::PredWeightTableInputs::base_profile`] covers the
  common single-layer base-profile case (every gate `true`,
  `high_precision_offsets_enabled_flag == false`).
  [`slice::PredWeightTable::parse`] reads
  `luma_log2_weight_denom` (`ue(v)`, range 0..=7) and, when chroma is
  present, `delta_chroma_log2_weight_denom` (`se(v)`) with the derived
  `ChromaLog2WeightDenom ∈ 0..=7` range check; then performs the two
  flag passes (luma and chroma) and the per-reference delta block
  (`delta_luma_weight_lX[i]` ∈ −128..=127, `luma_offset_lX[i]` ∈
  `−WpOffsetHalfRangeY ..= WpOffsetHalfRangeY − 1`,
  `delta_chroma_weight_lX[i][j]` ∈ −128..=127,
  `delta_chroma_offset_lX[i][j]` ∈
  `−4 * WpOffsetHalfRangeC ..= 4 * WpOffsetHalfRangeC − 1`). For B
  slices the L1 block is mirrored after L0. The §7.4.7.3 conformance
  cap `sumWeightLXFlags ≤ 24` is enforced (P: L0 only; B: L0+L1).
  Accessor methods [`slice::PredWeightTable::luma_weight_l0`] (mirrored
  for L1), [`slice::PredWeightTable::chroma_weight_l0`] (mirrored) and
  [`slice::PredWeightTable::chroma_offset_l0`] (mirrored, equation
  7-58) apply the §7.4.7.3 derivations for `LumaWeightLX[i]`,
  `ChromaWeightLX[i][j]` and `ChromaOffsetLX[i][j]` (including the
  §7.4.7.3 inferred values when the per-i flag is `false`).
  [`slice::PredWeightEntry`] groups the per-reference syntax elements
  in their unresolved on-wire form for audit.
- Module-level documentation extended with a §7.3.6.3 bullet covering
  the new `PredWeightTable` parser, the per-i outer-gate threading and
  the §7.4.7.3 derived-variable accessors.
- 11 new unit tests covering: a monochrome (`ChromaArrayType == 0`)
  P-slice single-reference parse with `LumaWeightL0[0]` derivation, a
  4:2:0 P-slice single-reference parse with chroma derivations
  (including equation 7-58 `ChromaOffsetL0[0][j]`), a B-slice
  "all flags zero" minimal-content case with inferred derived
  variables, range failures for `luma_log2_weight_denom > 7`, derived
  `ChromaLog2WeightDenom > 7`, `delta_luma_weight_l0[i] > 127` and
  `luma_offset_l0[i] > 127` at 8-bit, an acceptance test for
  `luma_offset_l0[i] == 200` at `high_precision_offsets_enabled_flag
  == true` + `BitDepthY == 10`, an outer-gate suppression test that
  verifies the gated-off luma-flag bit is not consumed and the
  delta is inferred to 0, a precondition test rejecting an
  `signal_luma_l0` slice with the wrong length, an I-slice rejection
  test, and a `sumWeightL0Flags > 24` conformance test.

### Added — clean-room rebuild round 16 (2026-05-26)

- §7.4.7.2 `NumPicTotalCurr` derivation (equation 7-57) as a new
  typed builder ([`slice::NumPicTotalCurrInputs`]):
  - [`slice::NumPicTotalCurrInputs::from_used_flags`] takes
    pre-resolved per-position `UsedByCurrPicS0` / `UsedByCurrPicS1` /
    `UsedByCurrPicLt` slices; [`slice::NumPicTotalCurrInputs::compute`]
    returns the typed `NumPicTotalCurr: u32`.
  - [`slice::NumPicTotalCurrInputs::from_explicit_short_term_rps`]
    sources the `S0` / `S1` slices straight off an explicit-form
    [`sps::ShortTermRefPicSet`] (equations 7-65 / 7-66); returns
    `None` for inter-RPS-predicted RPS sets (the §7.4.8 derivation
    must run first).
  - Builder methods
    [`slice::NumPicTotalCurrInputs::with_pps_curr_pic_ref_enabled`]
    (the SCC PPS closing-clause flag — inferred `false` until the
    SCC PPS extension is materialised) and
    [`slice::NumPicTotalCurrInputs::with_multilayer_extension`]
    (the F.7.4.7.2 equation `F-56` form — IDR `nal_unit_type` skips
    the short-term / long-term loops; `NumActiveRefLayerPics` is
    added at the end).
- §7.4.7.1 long-term resolution helper
  ([`slice::SliceLongTermRefPic::used_by_curr_pic_lt`]) — looks up
  `used_by_curr_pic_lt_sps_flag[ lt_idx_sps[i] ]` for
  SPS-resident entries against `sps.long_term_ref_pics`, and returns
  `used_by_curr_pic_lt_flag[i]` for in-slice entries. Returns `None`
  when the SPS-table index is out of range (bitstream-conformance
  failure).
- Public re-export: [`slice::NumPicTotalCurrInputs`] added to the
  crate root.
- [`slice::SliceSegmentHeader::parse`] still surfaces the inter-slice
  tail as an opaque tail — the §7.3.6.1 in-place call site is now
  unblocked (round 15's `RefPicListsModification::parse` + this
  round's `NumPicTotalCurr` derivation are both in place) but the
  full inter-slice body parse (the `pred_weight_table()` + the
  remaining handful of post-RPS flags + the slice-data offset) is a
  separate round's worth of work.
- 12 spec-pinned unit tests:
  `num_pic_total_curr_short_term_only`,
  `num_pic_total_curr_mixed_short_and_long_term`,
  `num_pic_total_curr_curr_pic_ref_only`,
  `num_pic_total_curr_all_contributors`,
  `num_pic_total_curr_zero_when_nothing_contributes`,
  `num_pic_total_curr_from_explicit_rps_builder`,
  `num_pic_total_curr_from_explicit_rps_rejects_inter_prediction`,
  `used_by_curr_pic_lt_resolves_sps_table_and_in_slice`,
  `num_pic_total_curr_from_resolved_slice_long_term_list`,
  `num_pic_total_curr_multilayer_skips_temporal_loops_for_idr`,
  `num_pic_total_curr_multilayer_keeps_loops_for_non_idr`,
  `num_pic_total_curr_drives_section_7_3_6_1_gate`. Test count
  160 → 172.

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
  `decodes_vui_then_captures_extension_tail`; the tiny x265-encoded
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
  mismatch is a validator/instrumentation artefact in the source fixture).

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
