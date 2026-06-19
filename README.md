# oxideav-h265

A pure-Rust H.265 / HEVC video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework, built
clean-room against ITU-T Recommendation H.265 | ISO/IEC 23008-2.

## Status

In-progress clean-room rebuild. The crate implements the non-VCL and
entropy-layer foundations of an HEVC decoder — the full parameter-set /
slice-header parser stack, the CABAC arithmetic decoding engine, the
§9.3.4.2 per-syntax-element binarization + context-index primitives, the
§7.3.8 slice-data CABAC syntax walk, the §8.6 scaling / transform path —
and now the **§8.4 intra sample-reconstruction driver**: the `recon`
module walks a decoded `CodingTreeUnit` and writes reconstructed samples
into a `Picture` (§8.4.2 / §8.4.3 mode derivation, §8.4.4.2 reference-
sample prediction, §8.6.2 dequant + inverse transform, §8.6.1 QP
derivation, §8.4.4.1 add-and-clip). A flat 16×16 intra CU reconstructs
to the `tiny-i-only-16x16-main` fixture's documented luma / Cb plane
values from a §8.6 DC residual.

Inter sample prediction (§8.5), the in-loop filters (§8.7 deblocking /
SAO apply), a full DPB, and an encoder are not yet wired. The end-to-end
decode of a real bitstream is blocked on a residual-coefficient CABAC
alignment defect (see below) — the reconstruction driver is validated
against synthesized spec-correct coefficients in the interim.

The runtime registration hook (`register`) is a no-op and every decode
path returns `Error::NotImplemented` until the full pipeline lands. Lower
layers are usable directly through the public parser / engine API.

### Known defect — residual-coding CABAC alignment

The §7.3.8.11 residual-coding walk on the `tiny-i-only-16x16-main`
fixture does not terminate at the byte-aligned `end_of_slice_segment_
flag` (it decodes `false` where `true` is required for the single-CTU
slice), so the decoded `TransCoeffLevel` magnitudes are wrong. The
reconstruction driver, intra prediction, and §8.6 transform are
independently validated; the next round must trace the residual-coding
bin sequence against the spec to locate the divergence.

## What's implemented

* **Annex B + NAL layer** — byte-stream splitting (3-byte and 4-byte
  start codes, multiple NAL units per buffer), §7.3.1.2 NAL header
  parse, §7.4.1.1 emulation-prevention strip, and an MSB-first bit
  reader with `u(n)` / `ue(v)` / `se(v)` descriptors.
* **VPS (§7.3.2.1)** — full video-parameter-set body including the
  §7.3.3 profile-tier-level walk, per-sub-layer DPB / reorder / latency
  loop, the layer-set inclusion matrix, timing info, the per-HRD-set
  `hrd_parameters()` loop, and the opaque extension tail.
* **SPS (§7.3.2.2)** — full sequence-parameter-set body: PTL, geometry
  + conformance window, bit depths, the PCM block, short-term reference
  picture sets (§7.3.7, both explicit and inter-RPS-prediction forms),
  long-term ref-pic table, scaling-list data, and the VUI / extension
  gates. The eight typed SPS-extension flags are decoded; the bodies
  surface as opaque bytes.
* **PPS (§7.3.2.3.1)** — full picture-parameter-set body: slice-header
  gates, QP / chroma-offset fields, the tiles + WPP partitioning block,
  deblocking-filter control, scaling-list data, and the typed
  PPS-extension flags (bodies surfaced as opaque bytes).
* **VUI (§E.2.1) + HRD (§E.2.2 / §E.2.3)** — the complete
  `vui_parameters()` body (aspect-ratio / video-signal / chroma-loc /
  timing / bitstream-restriction blocks with the §E.3 range checks) and
  the shared `hrd_parameters()` / `sub_layer_hrd_parameters()` decode
  with the §E.3 inference + monotonicity constraints.
* **Slice segment header (§7.3.6.1)** — independent I / P / B slice
  segments parse end to end through `byte_alignment()` for every
  conformant single-layer configuration: the POC + reference-picture-set
  block, ref-idx override, `ref_pic_lists_modification()` (§7.3.6.2),
  the mvd / cabac-init / collocated block, `pred_weight_table()`
  (§7.3.6.3) decoded in place, the merge-candidate leaf, and the shared
  QP / deblocking / loop-filter / entry-point / extension tail.
* **Reference-picture-set derivation** — §7.4.7.2 `NumPicTotalCurr`,
  §7.4.8 short-term-RPS materialization (explicit + inter-RPS-prediction
  forms, with the multilayer-extension variant), and the §7.4.7.3
  weighted-prediction derived variables.
* **Scaling lists (§7.3.4 / §7.4.5)** — `scaling_list_data()` parse plus
  the `ScalingFactor[sizeId][matrixId][x][y]` quantization-matrix
  expansion.
* **Scan orders (§6.5 / §7.4.2)** — up-right-diagonal, horizontal,
  vertical, and traverse scans with the `ScanOrder` accessor.
* **Picture-level scanning + availability (§6.4 / §6.5.1 / §6.5.2)** —
  the `availability` module: `PictureTiling` derives the §6.5.1
  `colWidth` / `rowHeight` / `colBd` / `rowBd` / `CtbAddrRsToTs` /
  `CtbAddrTsToRs` / `TileId` tile-scan conversion (eqs. 6-3..6-9) and
  the §6.5.2 `MinTbAddrZs` z-scan address (eq. 6-10), then answers the
  §6.4.1 z-scan block-availability query (in-picture + decode-order +
  slice + tile tests) and the §6.4.2 prediction-block availability
  (the `sameCb` short-cut and the `MODE_INTRA` masking). This is the
  derivation that feeds the per-sample "available for intra prediction"
  markings consumed by `intra_pred`.
* **CABAC engine (§9.3)** — `CabacEngine` (init, DecodeDecision with
  the Table 9-52 / 9-53 range split + state transition, DecodeBypass,
  DecodeTerminate, RenormD, and the aligned-bypass hook) plus
  `ContextModel::init` (the §9.3.2.2 init-type selector + equations
  9-4..9-6).
* **§9.3.4.2 binarization** — a growing set of per-syntax-element
  binarization + context-index primitives, including `cu_qp_delta`,
  `last_sig_coeff_{x,y}_{prefix,suffix}`, `coded_sub_block_flag`,
  the `split_cu_flag` / `cu_skip_flag` single-bin decode primitives
  (§7.3.8.4 / §7.3.8.5, FL `cMax = 1` with the §9.3.4.2.2 left/above
  ctxInc and the §7.4.9.5 `cu_skip_flag` → `CuPredMode::Skip`
  not-present mapping) alongside `pred_mode_flag`,
  the SAO family, `coeff_abs_level_greater{1,2}_flag`,
  `coeff_abs_level_remaining`, `coeff_sign_flag`, `mvd_coding()`,
  `merge_flag`, the luma-intra-mode signalling group
  (`prev_intra_luma_pred_flag` / `mpm_idx` / `rem_intra_luma_pred_mode`),
  and the §8.4.2 / §8.4.3 luma + chroma intra-prediction-mode
  derivation. These compose into the §7.3.8.11 `residual_coding()`
  driver.
* **Transform-tree recursion (§7.3.8.8)** — the `transform_tree` module:
  `decode_transform_tree` walks the §7.3.8.8 `transform_tree( )` syntax
  exactly — the `split_transform_flag` presence gate plus the §7.4.9.8
  forced-split inference (`log2TrafoSize > MaxTbLog2SizeY`, `IntraSplitFlag`
  at depth 0, `interSplitFlag`); the per-node `cbf_cb` / `cbf_cr` reads
  gated by the inheritance condition (`trafoDepth == 0 ||
  cbf_cX[xBase][yBase][trafoDepth − 1]`) with the `ChromaArrayType == 2`
  lower-half companions; the four-way quarter-size recursion; and at each
  leaf the §7.3.8.8 `cbf_luma` presence condition before invoking the
  §7.3.8.10 `decode_transform_unit`. One `QuantGroupState` threads through
  the whole subtree so the `delta_qp()` / `chroma_qp_offset()` gates fire
  once per quantization group. `TransformTreeParams` carries the per-CU
  geometry (`MaxTbLog2SizeY` / `MinTbLog2SizeY` / `MaxTrafoDepth`,
  `IntraSplitFlag`, `interSplitFlag`) and the result is a
  `Split { … } / Leaf { cbf_luma, unit }` tree. Four new §7.3.8.8 single-bin
  decode primitives back it (`decode_split_transform_flag`,
  `decode_cbf_luma`, `decode_cbf_cb` / `decode_cbf_cr`, each with its
  §7.4.9.8 not-present inference helper).
* **Transform-unit driver (§7.3.8.10)** — the `transform_unit` module:
  `decode_transform_unit` walks the §7.3.8.10 `transform_unit( )` syntax
  exactly, composing the existing per-element primitives into the leaf
  the §7.3.8.8 `transform_tree()` recursion bottoms out in. It derives
  `cbfChroma` (with the `ChromaArrayType == 2` lower-half companions),
  evaluates the adaptive-colour-transform predicate gating
  `tu_residual_act_flag`, reads the `delta_qp()` / `chroma_qp_offset()`
  blocks (each gated by the per-quantization-group `IsCuQpDeltaCoded` /
  `IsCuChromaQpOffsetCoded` state carried in `QuantGroupState`), the luma
  `residual_coding()`, and the chroma path — both the in-place branch
  (with the §7.3.8.12 `cross_comp_pred()` prelude, the
  `log2TrafoSizeC = Max(2, log2TrafoSize − (ChromaArrayType == 3 ? 0 : 1))`
  chroma size, and the `ChromaArrayType == 2` stacked-sub-block pair) and
  the `blkIdx == 3` deferred-chroma branch. Two new
  binarization decoders back it: `decode_cross_comp_pred` (§7.3.8.12, the
  TR(`cMax = 4`) `log2_res_scale_abs_plus1` prefix + conditional
  `res_scale_sign_flag`, with the §7.4.9.12 `ResScaleVal` derivation) and
  `decode_tu_residual_act_flag` (§7.3.8.10, FL `cMax = 1`).
* **Slice-data CTU/CU CABAC walk (§7.3.8.1 .. §7.3.8.6)** — the
  `slice_data` module: the upper rung of the §7.3.8 slice-data parse
  loop that drives the CABAC engine through the per-CTU syntax
  structures, composing the leaf decode primitives with the §7.3.8.8
  `transform_tree()` recursion. `decode_coding_tree_unit` (§7.3.8.2)
  pairs an optional §7.3.8.3 `sao()` (the full per-CTB SAO walk —
  merge flags, `sao_type_idx`, the four offsets, signs, band position
  / eo-class, with the §7.4.9.3 `SaoTypeIdx[2]` inheritance) with the
  §7.3.8.4 `decode_coding_quadtree` (`split_cu_flag` recursion + the
  §7.4.9.4 boundary inference, the §6.5.1 quantization-group resets,
  the in-picture child guards). The §7.3.8.5 `coding_unit` body walks
  `cu_transquant_bypass` / `cu_skip_flag` / `pred_mode_flag` /
  `part_mode` (Table 9-45 → `PartMode` + `IntraSplitFlag`), the PCM
  gate, the intra luma/chroma mode signalling group, the §7.3.8.6
  `prediction_unit` emission per `PartMode` (merge / non-merge inter:
  `merge_idx` / `inter_pred_idc` / `ref_idx` / `mvd_coding` /
  `mvp_lX_flag`), `rqt_root_cbf`, and entry into the transform tree —
  into a `CodingTreeUnit` → `CodingQuadtree` → `CodingUnit` parse tree.
  A per-CTU `CtuGrid` threads the `CtDepth` / `cu_skip_flag` neighbour
  state for the §9.3.4.2.2 `split_cu_flag` / `cu_skip_flag` ctxInc.
  The walk runs end to end on real HEVC bitstream: the
  `tiny-i-only-16x16-main` fixture's single intra CTU decodes
  bit-exactly through SAO (type 0 on all components) and the §7.3.8.4 /
  §7.3.8.5 coding-quadtree / coding-unit structure (one un-split 16×16
  intra PART_2Nx2N CU).
* **Scaling + transform (§8.6)** — `scale_coefficients` (§8.6.3
  dequantization), `inverse_transform` (the §8.6.4 separable inverse
  DST-VII / DCT-II), and the `residual_block` orchestrator turning a
  decoded `TransCoeffLevel` array into an `(nTbS)x(nTbS)` residual array.
* **Intra sample prediction (§8.4.4.2)** — the `intra_pred` module: the
  §8.4.4.2.2 reference-sample substitution sweep, the §8.4.4.2.3
  neighbour-sample filtering (Table 8-4 `filterFlag` + the `nTbS == 32`
  luma `biIntFlag` bi-linear path), and the §8.4.4.2.4 / §8.4.4.2.5 /
  §8.4.4.2.6 `INTRA_PLANAR` / `INTRA_DC` / `INTRA_ANGULAR2..34` predictors
  (Tables 8-5 / 8-6 angle + inverse-angle projection, with the vertical /
  horizontal / DC boundary filters). `intra_predict` chains the
  §8.4.4.2.1 filtering-gate + predictor dispatch from a substituted
  reference array; the §6.4.1 availability derivation that marks the
  neighbours remains the caller's responsibility.
* **Inter fractional-sample interpolation + default combine (§8.5.3.3.3 /
  §8.5.3.3.4.2)** — the `inter_pred` module: `RefPlane` (the §8.5.3.3.3
  `Clip3` edge-extended reference plane), `interp_luma_block`
  (the separable 8-tap quarter-pel luma filter of equations 8-224..8-238
  with Table 8-8 phase selection), `interp_chroma_block` (the separable
  4-tap eighth-pel chroma filter of equations 8-241..8-261 with Table 8-9
  phase selection), and `default_weighted_pred` (the §8.5.3.3.4.2
  uni- / bi-predictive combine of equations 8-262..8-264, the
  `weighted_pred_flag == 0` path). The interpolation carries the
  `14 − BitDepth`-bit intermediate precision; the combine clips to the
  sample range. The §8.5.3.1 / §8.5.3.2 motion-vector / merge derivation
  that produces `mvLX` and the §8.5.3.3.1 block-walk driver remain the
  caller's / follow-ups' responsibility.

## Not yet implemented

* Picture reconstruction from the decoded slice-data parse tree: the
  §8.4 intra-block / §8.5 inter-block sample-write passes that turn the
  `slice_data` module's `CodingTreeUnit` → `CodingQuadtree` →
  `CodingUnit` tree (now decoded end to end; see "What's implemented")
  into reconstructed luma / chroma planes, the in-loop deblock / SAO
  application, and DPB management. The §7.3.8 slice-data CABAC
  syntax-element walk itself — the §7.3.8.3 `sao()`, §7.3.8.2
  `coding_tree_unit()`, §7.3.8.4 `coding_quadtree()`, §7.3.8.5
  `coding_unit()` and §7.3.8.6 `prediction_unit()` structures — is
  implemented and runs end to end on real HEVC bitstream (the
  `tiny-i-only-16x16-main` fixture's single intra CTU + SAO decode
  bit-exactly through it).
* The rest of inter prediction (§8.5): the §8.5.3.1 / §8.5.3.2 motion-vector
  / merge-candidate derivation, the §8.5.3.3.1 prediction-block walk that
  splits `mvLX` into its integer / fractional parts and drives
  `inter_pred`, and the §8.5.3.3.4.3 explicit weighted-prediction path.
  In-loop filters (deblock / SAO) and DPB management likewise remain.
  (The §8.5.3.3.3 fractional-sample interpolation and §8.5.3.3.4.2 default
  combine are implemented; see above.)
* SPS / PPS / VPS extension bodies (range / multilayer / 3D / SCC) —
  the typed flags are decoded but the bodies surface as opaque bytes.
* Encoder.

## License

MIT — see [LICENSE](./LICENSE).
