# oxideav-h265

[![CI](https://github.com/OxideAV/oxideav-h265/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-h265/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-h265.svg)](https://crates.io/crates/oxideav-h265) [![docs.rs](https://docs.rs/oxideav-h265/badge.svg)](https://docs.rs/oxideav-h265) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A pure-Rust H.265 / HEVC video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework, built
clean-room against ITU-T Recommendation H.265 | ISO/IEC 23008-2.

## Status

**Decoder: end to end.** Every Annex B bitstream in the staged
16-fixture conformance corpus decodes byte-exact to its expected YUV
through the whole-bitstream driver (`decode_annexb_sequence` /
`SequenceDecoder`) and the `oxideav_core::Decoder` registry entry
(`make_decoder`, ids `"h265"` / `"hevc"`, `hvc1` / `hev1` / `HEVC`
FourCCs, MP4 ObjectTypeIndication, Matroska tag) — plus self-built
conformance pins for the features the corpus lacks (rounds 413 / 416:
RDPCM, palette, cross-component prediction, adaptive colour
transform, intra block copy), and a 37-stream black-box tool-axis
sweep (round 410) held byte-exact by nine embedded pins. Against the
staged **official JCT-VC RExt / SCC conformance corpus** (61
decodable bitstreams with published output digests,
`docs/video/h265/conformance/`), 26 streams decode byte-exact
(round 437; docs-gated pins in `tests/conformance_official.rs`) —
including the transform-skip-context, wavefronts-inside-tiles /
aligned-bypass / high-throughput, persistent-Rice-seeded,
Monochrome 8/12-bit, Main 4:2:2 10 and 8-bit
extended-precision-intra families. Coverage:

* intra pictures at every staged geometry / CTB size (16 / 32 / 64)
  and QP extreme (slice QP 1 and 45), with SAO on and off;
* multi-picture sequences: all-intra IDR runs, an IDR+P pair, and an
  eight-picture I/P/B pyramid with two reference lists, temporal MVP,
  AMVP MVDs, skip/merge CUs and POC output reordering;
* Main10 and the 4:2:2 / 4:4:4 10-bit Range-Extensions streams;
* multi-CTU slices, four-independent-slice pictures, dependent slice
  segments (§9.3.2.4/.5 `TableStateIdxDs` context carry + §7.4.7.1
  header inheritance), per-slice loop-filter-across flags, and
  `entropy_coding_sync_enabled_flag` (WPP) streams with per-row
  entry-point substreams;
* **true tiles**: the staged `true-tiles-2x2` fixture (2×2 uniform
  grid, one slice per tile, `loop_filter_across_tiles == 0`) decodes
  byte-exact, and multi-tile SINGLE-slice streams work end to end —
  §7.3.8.1 tile-boundary subsets (`end_of_subset_one_bit` + byte
  alignment), §9.3.2.2 per-tile CABAC re-initialization, tile-relative
  WPP row conditions, §8.6.1 per-tile `qPY_PREV` resets;
* §8.5.3.3.4.3 explicit weighted prediction (P uni and B uni/bi, with
  non-default per-slice weights / offsets / denominators);
* B pyramids / temporal layers / open-GOP CRA + leading pictures /
  RADL streams (a bi-predicted reference B as the §8.5.3.2.9
  collocated picture), §7.4.5 scaling lists (default + explicit, all
  TB sizes, intra + inter), §8.4.4.2.3 strong intra smoothing,
  §8.4.4.2.1 constrained intra prediction, §7.3.8.11 transform skip,
  rectangular/AMP partitions with deep inter RQTs (§7.3.8.10 deferred
  chroma), WPP combined with multiple slices per picture, and
  4:2:2 stacked chroma halves (per-half cbf gating and placement);
* §7.3.8.7 PCM coding units, incl. the §8.7.2.5.4 / §8.7.3.1
  loop-filter suppression (`pcm_loop_filter_disabled_flag`, and
  transquant-bypass CUs);
* range-extension RDPCM (§8.6.5 implicit intra + explicit inter
  directional residual modification) and SCC **palette mode**
  (§7.3.8.13 parse incl. the predictor-palette machinery with WPP /
  dependent-slice synchronization, §8.4.4.2.7 reconstruction with
  transpose and bypass / quantized escapes) — pinned by self-built
  conformance streams (`tests/fixture_bytes/r413-*.hevc`), the
  implicit-RDPCM stream byte-exact against a black-box reference
  decode;
* the Rext/SCC application tail: §8.6.6 **cross-component
  prediction** (eq. 8-324 applied on the intra and inter residual
  paths, cbf-clear chroma blocks included), §8.6.8 **adaptive colour
  transform** (the §8.6.8.2 lifting inverse with ACT-adjusted
  quantization, eqs 8-287/8-288/8-291), and **intra block copy**
  (current-picture referencing: the §8.3.4 currPic list append,
  `use_integer_mv_flag`, the eqs 8-98..8-101 / 8-124..8-125 integer
  MV paths, the eqs 8-102/8-103 reduction, prediction from the
  pre-filter reconstruction) — pinned by self-built conformance
  streams (`tests/fixture_bytes/r416-*.hevc`), the CCP stream
  byte-exact against a black-box reference decode (the surveyed
  reference decoder rejects SCC streams outright, so the ACT / IBC
  pins are decoder-pins);
* both transport forms: Annex B extradata/packets AND `hvcC`
  (`HEVCDecoderConfigurationRecord`, ISO/IEC 14496-15 §8.3.3.1)
  extradata with length-prefixed packets.

**Encoder: intra + low-delay inter + hierarchical-B GOPs with AMP
and in-loop filters, registered.** `make_encoder` / `H265Encoder`
with three modes:

* `mode = "inter"` (`qp` 0..=51, `gop` IDR period, `bslices`, `amp`,
  `pyramid`) — low-delay `IDR, P, P, …` / `IDR, B, B, …` GOPs
  (`encoder::inter::LowDelayPEncoder`, one frame in / one AU out) or
  dyadic **hierarchical-B pyramids**
  (`encoder::pyramid::PyramidEncoder`, `pyramid = 2/4/8/16`):
  per-CTU **skip / merge / AMVP / two-PU-partition / intra**
  decisions under an SSD + λ·rate heuristic, with **two active
  reference pictures** on the low-delay path (POC − 1 / POC − 2,
  `ref_idx_l0` signalled). Motion candidates are resolved through
  the crate's own DECODE-side §8.5.3.2 merge/AMVP derivation against
  the in-progress motion field (§6.4.2 availability included),
  motion estimation is a seeded greedy integer diamond plus
  half-/quarter-pel refinement against the crate's §8.5.3.3.3
  interpolation, two-PU CUs carry the §7.4.9.8 forced depth-1 RQT,
  and a `pred_mode_flag == 1` intra fallback rescues scene changes.
  The **AMP configuration** (`amp` option / `with_amp`) moves the
  stream to `MinCbSizeY == 8` + `amp_enabled_flag == 1` (explicit
  `split_cu_flag`, the Table 9-45 big-CU `part_mode` column) and
  elects `PART_2NxnU / PART_2NxnD / PART_nLx2N / PART_nRx2N`
  alongside the symmetric shapes — on quarter-offset motion-boundary
  content the shapes buy **−20 to −29 % bytes at equal PSNR**. The
  **pyramid** codes each mini-GOP out of display order (next anchor
  as P first, then the midpoint B recursion with the past boundary
  on `RefPicList0` and the FUTURE boundary on `RefPicList1`),
  signals per-slice negative + positive short-term RPS with
  `sps_max_num_reorder_pics = log2(gop)`, allocates rate by
  per-layer QP offsets, and searches uni-L0 / uni-L1 / **bi AMVP**
  per PU (`inter_pred_idc` `PRED_L0/L1/BI` all emitted); on a
  noisy-pan clip the GOP-8 pyramid takes **−11.5 % bytes** vs the
  low-delay chain at qp 27. Every stream decodes **bit-exact** to
  the encoder reconstruction through this crate's decoder AND a
  black-box reference decoder (multi-QP, multi-shape sweeps; golden
  GOP / AMP / pyramid / composition streams CI-pinned). Per-frame
  `FrameStats` expose the skip/merge/AMVP/intra/bi/rect/amp/ref1
  decisions.
* `mode = "intra"` — per-CTU §8.4 intra prediction over the
  encoder's own reconstruction (all 35 modes, per-CTB `PART_2Nx2N`
  vs `PART_NxN` rate-distortion decision with per-PB modes, §8.4.2
  MPM signalling, mode-dependent scans), forward DCT-II + reciprocal
  quantization, full §7.3.8 syntax through the bin-exact §7.3.8.11
  residual encoder (golden interop stream CI-pinned).
* `mode = "pcm"` (default) — the lossless PCM-IDR bootstrap
  (every CTB a 16×16 PCM CU; options for dependent segments,
  multi-slice plans, deblocking, band / edge SAO syntax, and true
  multi-tile single-slice pictures with §7.4.7.1 entry points).

Both coding modes accept the §8.7 **in-loop filters** (`deblock` /
`sao` codec options, `LoopFilterCfg` on the direct APIs): the encoder
reconstructs through its own decode-side §8.7.2 deblocking (per-slice
election over off + a {−2, 0, 2}² β/tC-offset sweep, signalled via
the §7.3.6.1 override group) and §8.7.3 SAO (per-CTB statistics-driven band / edge
offset estimation with merge-left/up pricing, every candidate
measured with the decoder's own apply, `encode_sao_ctb` the bin-exact
§7.3.8.3 dual of the parse) — so the filtered pictures its references
and outputs hold are exactly a conforming decoder's. Filtered P/B
GOPs across a 72-configuration sweep decode byte-exact through a
black-box reference decoder (three golden filtered streams
CI-pinned); on the interop clip the filters buy up to +1.5 dB luma
PSNR at equal rate.

4:2:0 8-bit, dimensions multiples of 16.

## What's implemented

* **Whole-bitstream decode driver** (`sequence`) — Annex B demux →
  SPS/PPS activation → §7.3.6.1 slice headers (independent and
  dependent segments) → the §7.3.8.1 CTU loop (tile-scan addressing,
  per-slice CABAC init, the §9.3.2.4/.5 WPP and dependent-segment
  context storage/sync, WPP substreams via the §7.4.7.1 entry
  points) → picture reconstruction → §8.3.1..§8.3.5 reference
  cycle → output reorder. Tile-scan CTU addressing with §9.3.2.2
  per-tile context re-initialization and entry-point subsets shared
  with WPP.
* **Registry codec** (`decoder` / `encoder`) — the
  `oxideav_core::Decoder` + `Encoder` contracts: Annex B or
  hvcC/length-prefixed packets in, output-order `VideoFrame`s out
  (reorder queue bounded by `sps_max_num_reorder_pics`, packet-PTS
  re-attachment, flush-then-`Eof`); frames in, IDR keyframe packets
  out (`mode = "pcm"` lossless or `mode = "intra"` at a chosen QP).
  `make_decoder` / `make_encoder` are the direct factory endpoints.
* **Headers** — VPS / SPS / PPS (§7.3.2, incl. range + SCC extension
  bodies), VUI + HRD (§E.2), slice segment headers (§7.3.6, all slice
  types, RPS forms, `ref_pic_lists_modification()`,
  `pred_weight_table()`, entry points incl. dependent segments),
  §7.4.8 RPS materialization, scaling lists (§7.3.4 / §7.4.5), and
  the `hvcC` record (`hvcc`, ISO/IEC 14496-15 §8.3.3.1).
* **CABAC, both directions** — the §9.3 decode engine and the §9.3.5
  encode engine (decision / bypass / terminate + flush, PCM
  align-and-reinit), per-syntax-element binarizations (§9.3.4.2), the
  complete §7.3.8 decode syntax tree, and the write-side §7.3.8.11
  `residual_coding( )` dual (`encoder::residual`, differential-tested
  to identical levels + context evolution).
* **Reconstruction** — §8.4 intra prediction (all 35 modes), §8.4.1
  PCM sample write-back, §8.5 inter prediction (merge / MVP /
  temporal candidates with the §8.5.3.2.3 raw-availability redundancy
  gates, §8.5.3.3.3 interpolation, §8.5.3.3.4.2 default AND
  §8.5.3.3.4.3 explicit weighted combines with the §7.4.7.3 table
  resolution), §8.6 dequant / inverse transform, §8.6.1 per-QG QP
  derivation.
* **In-loop filters** — §8.7.2 deblocking and §8.7.3 SAO with
  per-slice `slice_loop_filter_across_slices_enabled_flag` gates
  (deblocking per the current CU's slice; SAO per the §8.7.3.2
  directional later-slice rule) and the PCM / transquant-bypass
  sample suppression (`NoFilterMap`).
* **Reference machinery** — §8.3.1 POC, §8.3.2 RPS marking, §8.3.4
  reference lists, §8.3.5 collocated picture, the DPB, and the
  per-picture decode cycle threading motion fields for temporal MVP.

Thirty-six embedded-fixture regression pins (the 17-stream staged
corpus incl. true tiles + self-built weighted-prediction,
per-slice-loop-filter, hvcC, golden-intra-interop, golden-P-GOP
interop, the round-431 AMP / pyramid / composition interop pins, and
the nine round-410 tool-axis conformance pins), lossless PCM /
exact-reconstruction intra / bit-exact low-delay- and
hierarchical-B-GOP encoder↔decoder roundtrips at multiple
geometries / QPs / partitions / slice types, and ~930 unit tests.

## Not yet implemented

* Larger encoder CTB sizes, deeper encoder RQTs / coding quadtrees
  (the AMP configuration codes 16x16 CUs above an 8x8 MinCb, but
  never splits), 4x4-luma DST TUs, encoder temporal MVP, more than
  one active reference per list on the pyramid path, and adaptive
  (non-dyadic) GOP structures.
* Non-uniform (`uniform_spacing_flag == 0`) tile-grid *encoding*
  (decode side is implemented).
* Known corner: on the §8.7.3.2 SAO cross-slice neighbour rule with
  heterogeneous per-slice flags, a black-box reference decoder
  consults the current sample's slice flag where the spec text (both
  08/2021 and 01/2026 editions) names the later (decode-order)
  slice's flag; this implementation follows the spec text.
* Known corner (RDPCM, spec text followed): §8.4.4.2.6 sets
  `disableIntraBoundaryFilter` when implicit RDPCM combines with
  transquant bypass, suppressing the mode-10/26 edge filters — a
  black-box reference decoder applies those filters regardless.
  (The r413 note claiming a reference deviation on explicit-RDPCM
  *vertical* blocks is retired: the official `ExplicitRdpcm_A`
  conformance stream matches the black-box decode exactly, this
  crate's luma decode agrees byte-for-byte, and the remaining
  divergence is an unrelated ±1 chroma artifact on isolated rows,
  still under investigation.)
* Official-corpus families not yet byte-exact: the 4:4:4
  multi-tool `GENERAL_*_444` / `WAVETILES` / `QMATRIX` parse
  desync (mid-picture, first reproduces in a busy 4:4:4 intra CTB),
  16-bit extended-precision reconstruction (the 8-bit
  extended-precision streams pass; 10/12/16-bit diverge on their
  first picture), unequal luma/chroma bit depths (first divergence
  at frame 4), the official CCP / SAO-offset-scale / persistent-Rice
  single-stream anchors, and most SCC streams (palette-heavy parse
  still desyncs mid-slice; two streams decode all 33 frames with
  subtle reconstruction deltas).

## License

MIT — see [LICENSE](./LICENSE).
