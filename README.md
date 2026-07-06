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
conformance pins for the features the corpus lacks. Coverage:

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
* §7.3.8.7 PCM coding units, incl. the §8.7.2.5.4 / §8.7.3.1
  loop-filter suppression (`pcm_loop_filter_disabled_flag`, and
  transquant-bypass CUs);
* both transport forms: Annex B extradata/packets AND `hvcC`
  (`HEVCDecoderConfigurationRecord`, ISO/IEC 14496-15 §8.3.3.1)
  extradata with length-prefixed packets.

**Encoder: real CABAC intra coding, registered.** `make_encoder` /
`H265Encoder` (option `mode = "intra"`, `qp` 0..=51) emit Main-profile
Annex B IDR access units with actual compression: per-CTU §8.4 intra
prediction over the encoder's own reconstruction (all 35 modes,
per-CTB `PART_2Nx2N` vs `PART_NxN` rate-distortion decision with
per-PB modes, §8.4.2 MPM signalling, mode-dependent scans), forward
DCT-II + reciprocal quantization, and full §7.3.8 syntax through the
bin-exact §7.3.8.11 residual encoder (`encoder::residual`) — the
crate's decoder AND a black-box reference decoder reproduce the
encoder's reconstruction bit-exactly (golden interop stream CI-pinned).
The default `mode = "pcm"` keeps the lossless PCM-IDR bootstrap
(every CTB a 16×16 PCM CU; options for dependent segments,
multi-slice plans, deblocking, band / edge SAO syntax, and true
multi-tile single-slice pictures with §7.4.7.1 entry points).
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

Twenty-three embedded-fixture regression pins (the 17-stream staged
corpus incl. true tiles + self-built weighted-prediction,
per-slice-loop-filter, hvcC and golden-intra-interop pins), lossless
PCM and exact-reconstruction intra encoder↔decoder roundtrips at
multiple geometries / QPs / partitions, and ~870 unit tests.

## Not yet implemented

* Encoder inter coding (P/B pictures) and encoder-side SAO /
  deblocking-aware reconstruction (the intra encoder disables the
  in-loop filters); larger encoder CTB sizes and 4x4-luma DST TUs.
* Non-uniform (`uniform_spacing_flag == 0`) tile-grid *encoding*
  (decode side is implemented).
* Known corner: on the §8.7.3.2 SAO cross-slice neighbour rule with
  heterogeneous per-slice flags, a black-box reference decoder
  consults the current sample's slice flag where the spec text (both
  08/2021 and 01/2026 editions) names the later (decode-order)
  slice's flag; this implementation follows the spec text.

## License

MIT — see [LICENSE](./LICENSE).
