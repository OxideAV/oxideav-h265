# oxideav-h265

A pure-Rust H.265 / HEVC video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework, built
clean-room against ITU-T Recommendation H.265 | ISO/IEC 23008-2.

## Status

In-progress clean-room rebuild. The crate currently implements the
non-VCL and entropy-layer foundations of an HEVC decoder — the full
parameter-set / slice-header parser stack, the CABAC arithmetic
decoding engine, and a growing library of §9.3.4.2 per-syntax-element
binarization + context-index primitives plus the §8.6 scaling /
transform path. The slice-data syntax-element walk, picture
reconstruction (intra / inter prediction, in-loop filters, DPB
management), and an encoder are not yet wired.

The runtime registration hook (`register`) is a no-op and every decode
path returns `Error::NotImplemented` until reconstruction lands. Lower
layers are usable directly through the public parser / engine API.

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
* **CABAC engine (§9.3)** — `CabacEngine` (init, DecodeDecision with
  the Table 9-52 / 9-53 range split + state transition, DecodeBypass,
  DecodeTerminate, RenormD, and the aligned-bypass hook) plus
  `ContextModel::init` (the §9.3.2.2 init-type selector + equations
  9-4..9-6).
* **§9.3.4.2 binarization** — a growing set of per-syntax-element
  binarization + context-index primitives, including `cu_qp_delta`,
  `last_sig_coeff_{x,y}_{prefix,suffix}`, `coded_sub_block_flag`,
  `split_cu_flag` / `cu_skip_flag` / `pred_mode_flag` ctxInc shapes,
  the SAO family, `coeff_abs_level_greater{1,2}_flag`,
  `coeff_abs_level_remaining`, `coeff_sign_flag`, `mvd_coding()`,
  `merge_flag`, the luma-intra-mode signalling group
  (`prev_intra_luma_pred_flag` / `mpm_idx` / `rem_intra_luma_pred_mode`),
  and the §8.4.2 / §8.4.3 luma + chroma intra-prediction-mode
  derivation. These compose into the §7.3.8.11 `residual_coding()`
  driver.
* **Scaling + transform (§8.6)** — `scale_coefficients` (§8.6.3
  dequantization), `inverse_transform` (the §8.6.4 separable inverse
  DST-VII / DCT-II), and the `residual_block` orchestrator turning a
  decoded `TransCoeffLevel` array into an `(nTbS)x(nTbS)` residual array.

## Not yet implemented

* The §7.3.8 slice-data syntax-element walk that drives the CABAC engine
  through the CTU / CU / TU parse loops.
* Intra / inter sample prediction (§8.4.4.2 reference-sample
  substitution + filtering + the DC / planar / angular predictors,
  §8.5 inter prediction), in-loop filters (deblock / SAO), and DPB
  management.
* SPS / PPS / VPS extension bodies (range / multilayer / 3D / SCC) —
  the typed flags are decoded but the bodies surface as opaque bytes.
* Encoder.

## License

MIT — see [LICENSE](./LICENSE).
