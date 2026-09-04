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
`docs/video/h265/conformance/`), 46 streams decode byte-exact
(round 444; docs-gated pins in `tests/conformance_official.rs`) —
**the complete 46-stream RExt branch**: transform-skip-context,
wavefronts-inside-tiles / aligned-bypass / high-throughput,
persistent-Rice(-seeded and single-stream-anchor), Monochrome
8/12/16-bit, Main 4:2:2 10, the COMPLETE 4:4:4
extended-precision-intra matrix (both profiles at 8/10/12/16-bit),
the 4:4:4 `GENERAL_*` multi-tool family, `WAVETILES` (wavefronts +
tiles + dependent slice segments in one picture), the
cross-component-prediction anchors (`CCP_{8,10,12}bit`), `QMATRIX_A`,
unequal luma/chroma depths (`Bitdepth_A/B`), `SAO_A_RExt`, and
`ExplicitRdpcm_A`. All 15 official SCC bitstreams parse end to end
(round 444 §7.3.8.10 ACT-gate fix); their reconstruction is not yet
byte-exact. Coverage:

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

**Encoder: recursive coding-quadtree I/P/B coding at CTB 16/32/64
with temporal MVP, multi-reference lists, hierarchical GOPs, in-loop
filters and rate control, registered.** `make_encoder` / `H265Encoder`
with three modes:

* `mode = "inter"` (`qp` 0..=51, `gop`, `bslices`, `amp`, `ctb`,
  `refs`, `tmvp`, `tudepth`, `rdoq`, `sdh`, `sl`, `pyramid` /
  `pyramidstep` / `adaptivegop`) —
  low-delay `IDR, P/B, …` GOPs (`encoder::inter::LowDelayPEncoder`)
  or hierarchical-B mini-GOPs of ANY length 2..=16
  (`encoder::pyramid::PyramidEncoder`, dyadic lengths giving the
  classic pyramid, others the same midpoint schedule with
  schedule-exact `sps_max_num_reorder_pics` / DPB bounds;
  `adaptivegop` closes a mini-GOP at scene cuts so no B slice
  straddles one, and flush tails code as short pyramids). Per-CU
  **skip / merge / AMVP / two-PU(+AMP) / intra** decisions under an
  SSD + λ·bins cost at every node of a real §7.3.8.4 **coding
  quadtree** (the `ctb` option: 16 / 32 / 64 with `MinCbSizeY == 8`,
  RD-elected `split_cu_flag` with full encoder-state rollback,
  recursive §7.3.8.8 RQTs to `max_transform_hierarchy_depth_*` 0..=3
  (`tudepth`, RD-elected at every node), 4x4 DST-VII intra luma TUs,
  intra `PART_NxN`, 8x4 / 4x8 inter PUs; without `ctb` the historical
  one-CU-per-CTB coder keeps its streams byte-stable). Motion candidates resolve through the
  crate's own DECODE-side §8.5.3.2 derivation — **temporal MVP**
  included (`tmvp`: `slice_temporal_mvp_enabled_flag`, the §7.3.6.1
  collocated block, per-reference retained motion fields) — over
  §8.3.4-built lists of up to four references per list (`refs`; the
  pyramid's L0/L1 cycle past-then-future / future-then-past across
  every retained picture, TR `ref_idx` coded, ME per reference).
  The two-start integer search (predictor seeds AND a subsampled
  ±24 grid scan, each refined coarse-to-fine then to quarter-pel
  against the crate's §8.5.3.3.3 interpolation) holds up on periodic
  textures at multi-frame distances; the motion λ is the SAD-domain
  `3·isqrt(λ_mode)/2`. On a 9-frame CIF noisy pan at QP 27 / 32 the
  GOP-8 pyramid with TMVP + 2 refs takes **−4.5 % / −16 % bytes vs
  the low-delay chain**, and the CTB-64 quadtree another **−10 % /
  −12 %** on top (+0.2 / +0.55 dB); at CTB 64 the all-intra /
  low-delay legs run **−11 % / −17 % bytes at +0.4..+0.6 dB** vs the
  CTB-16 coder. Every stream decodes **bit-exact** to the encoder
  reconstruction through this crate's decoder AND a black-box
  reference decoder (golden pins across the quadtree / TMVP /
  multi-ref / CTU-RC axes CI-pinned). Per-frame `FrameStats` expose
  the skip/merge/amvp/intra/bi/rect/amp/ref1 decisions.
* `mode = "intra"` — per-CTU §8.4 intra prediction over the
  encoder's own reconstruction (all 35 modes; at `ctb` 16/32/64 the
  full quadtree with per-TU prediction, RD-elected RQT depth and
  DST-VII 4x4 TUs; `PART_2Nx2N` vs `PART_NxN` per MinCb CU, §8.4.2
  MPM signalling, mode-dependent scans), forward DCT-II/DST-VII +
  reciprocal quantization, full §7.3.8 syntax through the bin-exact
  §7.3.8.11 residual encoder.
* `mode = "pcm"` (default) — the lossless PCM-IDR bootstrap
  (every CTB a 16×16 PCM CU; options for dependent segments,
  multi-slice plans, deblocking, band / edge SAO syntax, and true
  multi-tile single-slice pictures — uniform grids OR explicit
  non-uniform spans (`PcmAuOptions::tile_spans`,
  `uniform_spacing_flag == 0`) — with §7.4.7.1 entry points).

Both coding modes accept the §8.7 **in-loop filters** (`deblock` /
`sao` codec options, `LoopFilterCfg` on the direct APIs): the encoder
reconstructs through its own decode-side §8.7.2 deblocking (per-slice
election over off + a {−2, 0, 2}² β/tC-offset sweep) and §8.7.3 SAO
(per-CTB statistics-driven band / edge estimation with
merge-left/up pricing, every candidate measured with the decoder's
own apply) — at any CTB size, over per-CU deblocking descriptors and
a per-4x4 §8.6.1 `QpY` map — so the filtered pictures its references
and outputs hold are exactly a conforming decoder's.

Every coding path also accepts **average-bitrate rate control**
(`bitrate` / `fps`; `with_rate_control` on the low-delay AND pyramid
APIs): a deterministic integer-only controller on the §8.6.3
quantizer lattice elects each frame's `SliceQpY` through
`slice_qp_delta` alone. CI-gated accuracy (`tests/rate_accuracy.rs`):
the low-delay/pyramid × targets × B-slice/filter/AQ × VBV/HRD matrix
lands within 1.6 % of target over 60–65 frames (low-delay within
1 %), with a monotone rate ladder on both paths. **CTU-level rate
feedback** (`cturc`; quadtree coder + rate control) additionally
moves each CTB's `QpY` by up to ±3 through §7.3.8.14 `cu_qp_delta`
against a shadow-CABAC count of the picture's running size vs the
pro-rata frame budget (steadier per-frame sizes; composes with AQ /
TMVP / VBV). A **VBV constraint** (`bufsize` / `with_vbv`) hard-caps
EVERY access unit at its own decode instant on the low-delay,
all-intra AND hierarchical-B paths (re-encode at a higher QP as the
backstop). **HRD conformance** (`hrd` / `with_hrd`): §E.2.2
`hrd_parameters( )` in the SPS VUI, §D.2.2 buffering-period +
§D.2.3 pic-timing SEI on every access unit (the pyramid's
`pic_dpb_output_delay` carries its reorder schedule — non-dyadic
mini-GOPs and short tails included), and an exact integer Annex C
clock capping every AU so the §C.4 conditions hold by construction
— self-checked by a bitstream-only §C.2 replay in CI and validated
black-box. The `cbr` option / `with_cbr` switches to
constant-bit-rate delivery (`cbr_flag == 1`, eq. C-19 bounds,
§7.3.4 filler-data padding). **Spatial adaptive quantization**
(`aq` 1..=3) signals per-CTB activity offsets through `cu_qp_delta`
on every coding mode. An explicit `fps` declares §E.2.1 VUI timing.

**Quantization tools** (quadtree coder, `TreeCfg` builders / registry
options, off by default so the golden pins stay byte-stable): **RDOQ**
(`rdoq`: every TB's levels elected under `D + λ·R` in the decoder's
reverse scan — `sig_coeff_flag` / `greater1` / `greater2` / sign /
Rice-adapted `coeff_abs_level_remaining` bins priced at the running
CABAC context states of a shadow emission through an integer
Table 9-52-derived bin-cost model, then the last significant position
and whole coded sub-blocks re-elected) — **−9.4 % / −5.5 % / −7.9 %
BD-rate** on the pyramid / low-delay / all-intra paths of the
`examples/rd_measure.rs` corpus; and **sign data hiding** (`sdh`: PPS
`sign_data_hiding_enabled_flag`, the §7.3.8.11 `signHidden` sub-blocks
omit their first-in-scan sign and the levels are parity-adjusted by
the cheapest ±1 move under a sample-domain distortion + rate estimate)
— −2.5..−4 % bytes at near-neutral BD-rate; and **scaling lists**
(`sl` 1..=3: the §7.4.5 defaults, or a flattened / steepened custom
family written through §7.3.4 `scaling_list_data( )` — every
quantizer path prices each position at its Table 7-3 / 7-4
`ScalingFactor`; an HVS weighting: the defaults trade −15 % bytes for
−1..−3 dB luma PSNR).

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
  bodies), VUI + HRD (§E.2), SEI (§7.3.5 framing; typed §D.2
  payloads incl. the context-dependent §D.2.2 buffering-period and
  §D.2.3 pic-timing bodies — parsed decode-side AND emitted
  encode-side), slice segment headers (§7.3.6, all slice
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

Forty-four embedded-fixture regression pins (the 17-stream staged
corpus incl. true tiles + self-built weighted-prediction,
per-slice-loop-filter, hvcC, golden-intra-interop, golden-P-GOP
interop, the round-431 AMP / pyramid / composition interop pins, the
nine round-410 tool-axis conformance pins, and the round-453
quadtree / TMVP+multi-ref / CTU-rate-feedback / explicit-tile-grid
pins — every encoder pin byte-exact through a black-box reference
decoder at pin time), lossless PCM / exact-reconstruction intra /
bit-exact low-delay- and hierarchical-B-GOP encoder↔decoder
roundtrips at multiple geometries / QPs / partitions / slice types,
and ~960 unit tests.

## Not yet implemented

* Encoder tools beyond the current set: encoder-side WPP / tile
  parallel emission, weighted prediction estimation, and SCC-tool
  (palette / IBC / ACT) encoding; CTU-level rate feedback and the quantization /
  hierarchy tools ride only the quadtree coder. (Intra `PART_NxN`
  above `MinCbSizeY` is not a gap: §7.3.8.5 codes `part_mode` for
  intra CUs only at `MinCbLog2SizeY`, and the quadtree's split-CU
  path covers that geometry.)
* Known corner: on the §8.7.3.2 SAO cross-slice neighbour rule with
  heterogeneous per-slice flags, a black-box reference decoder
  consults the current sample's slice flag where the spec text (both
  08/2021 and 01/2026 editions) names the later (decode-order)
  slice's flag; this implementation follows the spec text.
* Known corner (RDPCM, spec text followed): §8.4.4.2.6 sets
  `disableIntraBoundaryFilter` when implicit RDPCM combines with
  transquant bypass, suppressing the mode-10/26 edge filters — a
  black-box reference decoder applies those filters regardless.
  (The r413/r437 "±1 chroma artifact under investigation" notes are
  retired: the divergence was the round-444 §8.5.4.3 phantom
  stacked-half bug, and `ExplicitRdpcm_A` is now byte-exact.)
* Official-corpus families not yet byte-exact (the 15 SCC streams):
  every SCC bitstream parses end to end (round 444 closed the
  §7.3.8.10 ACT-gate CABAC desync), and the decoded pictures are
  visually clean, but reconstruction diverges from the published
  per-frame hashes starting at the first picture — before the loop
  filters, in pictures whose only non-validated tools are intra
  block copy and the SCC glue. Round-444 elimination notes: the
  §8.5.3.3.3 MC math, eq. 8-124/8-125 merge rounding, chroma-QP /
  scaling-list / WP paths and the loop-filter stages all self-check
  clean against the Recommendation text; localizing further needs
  per-region ground truth that the corpus sidecars (whole-plane
  MD5s) do not provide.

## License

MIT — see [LICENSE](./LICENSE).
