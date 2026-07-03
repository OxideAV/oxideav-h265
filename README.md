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
* §8.5.3.3.4.3 explicit weighted prediction (P uni and B uni/bi, with
  non-default per-slice weights / offsets / denominators);
* §7.3.8.7 PCM coding units, incl. the §8.7.2.5.4 / §8.7.3.1
  loop-filter suppression (`pcm_loop_filter_disabled_flag`, and
  transquant-bypass CUs);
* both transport forms: Annex B extradata/packets AND `hvcC`
  (`HEVCDecoderConfigurationRecord`, ISO/IEC 14496-15 §8.3.3.1)
  extradata with length-prefixed packets.

**Encoder: PCM-only IDR bootstrap, registered.** `make_encoder` /
`H265PcmEncoder` emit fully conformant Main-profile Annex B IDR
access units in which every CTB is a 16×16 PCM coding unit —
bit-exact lossless, every packet a random access point (4:2:0 8-bit,
dimensions multiples of 16). The write stack underneath is real:
`BitWriter` + §7.4.1.1 NAL encapsulation + the §9.3.5 CABAC
arithmetic encoding engine + VPS/SPS/PPS/slice-header/slice-data
writers, with options for dependent slice segments, independent
multi-slice plans (per-slice loop-filter flags), deblocking, and
band / edge SAO syntax. A black-box reference decoder reproduces the
exact input from every encoded shape.

## What's implemented

* **Whole-bitstream decode driver** (`sequence`) — Annex B demux →
  SPS/PPS activation → §7.3.6.1 slice headers (independent and
  dependent segments) → the §7.3.8.1 CTU loop (tile-scan addressing,
  per-slice CABAC init, the §9.3.2.4/.5 WPP and dependent-segment
  context storage/sync, WPP substreams via the §7.4.7.1 entry
  points) → picture reconstruction → §8.3.1..§8.3.5 reference
  cycle → output reorder.
* **Registry codec** (`decoder` / `encoder`) — the
  `oxideav_core::Decoder` + `Encoder` contracts: Annex B or
  hvcC/length-prefixed packets in, output-order `VideoFrame`s out
  (reorder queue bounded by `sps_max_num_reorder_pics`, packet-PTS
  re-attachment, flush-then-`Eof`); frames in, lossless PCM IDR
  keyframe packets out. `make_decoder` / `make_encoder` are the
  direct factory endpoints.
* **Headers** — VPS / SPS / PPS (§7.3.2, incl. range + SCC extension
  bodies), VUI + HRD (§E.2), slice segment headers (§7.3.6, all slice
  types, RPS forms, `ref_pic_lists_modification()`,
  `pred_weight_table()`, entry points incl. dependent segments),
  §7.4.8 RPS materialization, scaling lists (§7.3.4 / §7.4.5), and
  the `hvcC` record (`hvcc`, ISO/IEC 14496-15 §8.3.3.1).
* **CABAC, both directions** — the §9.3 decode engine and the §9.3.5
  encode engine (decision / bypass / terminate + flush, PCM
  align-and-reinit), per-syntax-element binarizations (§9.3.4.2), and
  the complete §7.3.8 syntax tree incl. §7.3.8.7 PCM sample payloads.
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

Twenty-two embedded-fixture regression pins (the 16-stream staged
corpus + self-built weighted-prediction, per-slice-loop-filter and
hvcC pins), lossless encoder↔decoder roundtrips at multiple
geometries / segmentations / filter shapes, and ~850 unit tests.

## Not yet implemented

* True multi-tile streams (the §6.5.1 tiling machinery and the
  per-tile CABAC re-init points exist, but no tiles-enabled fixture
  pins byte-exactness — the corpus encoder cannot emit tiles; fixture
  ask filed).
* Encoder beyond the PCM bootstrap (intra prediction + residual
  coding write-side).
* Known corner: on the §8.7.3.2 SAO cross-slice neighbour rule with
  heterogeneous per-slice flags, a black-box reference decoder
  consults the current sample's slice flag where the spec text (both
  08/2021 and 01/2026 editions) names the later (decode-order)
  slice's flag; this implementation follows the spec text.

## License

MIT — see [LICENSE](./LICENSE).
