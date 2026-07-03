# oxideav-h265

A pure-Rust H.265 / HEVC video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework, built
clean-room against ITU-T Recommendation H.265 | ISO/IEC 23008-2.

## Status

**The decoder is end-to-end: every Annex B bitstream in the staged
16-fixture conformance corpus decodes byte-exact to its expected YUV**
through the whole-bitstream driver (`decode_annexb_sequence` /
`SequenceDecoder`) and the `oxideav_core::Decoder` registry entry
(`make_decoder`, registered under `"h265"` / `"hevc"` with the
`hvc1` / `hev1` / `HEVC` FourCCs, the MP4 ObjectTypeIndication, and the
Matroska tag). The corpus spans:

* intra pictures at every staged geometry / CTB size (16 / 32 / 64)
  and QP extreme (slice QP 1 and 45), with SAO on and off;
* multi-picture sequences: all-intra IDR runs, an IDR+P pair, and an
  eight-picture I/P/B pyramid with two reference lists, temporal MVP,
  AMVP MVDs, skip/merge CUs and POC output reordering;
* Main10 and the 4:2:2 / 4:4:4 10-bit Range-Extensions streams;
* multi-CTU slices, four-independent-slice pictures (loop filters
  suppressed across slice boundaries), and
  `entropy_coding_sync_enabled_flag` (WPP) streams with per-row
  entry-point substreams and the §9.3.2.4 / §9.3.2.5 context
  storage / synchronization.

The decode stack under the driver: Annex B / NAL demux, the full
parameter-set + slice-header parser, the §9.3 CABAC engine and the
complete §7.3.8 slice-data syntax walk (with the §8.4.2 parse-time
intra-mode derivation feeding the §7.4.9.11 mode-dependent scans and a
picture-level `CtDepth` / `cu_skip_flag` / slice / tile neighbour
state), the §8.4 intra and §8.5 inter reconstruction (merge / MVP /
temporal-MV candidate derivation, quarter-pel interpolation, default
bi-prediction combine), the §8.6 dequant / inverse transform with the
full §8.6.1 per-quantization-group QP prediction, the §8.7.2 deblocking
(per-position QP map, boundary-aware edge gating) and §8.7.3 SAO
in-loop filters, and the §8.3 POC / RPS / reference-list / DPB cycle
with output-order (PicOrderCntVal) frame delivery.

## What's implemented

* **Whole-bitstream decode driver** (`sequence`) — Annex B demux →
  SPS/PPS activation → §7.3.6.1 slice headers → the §7.3.8.1 CTU loop
  (tile-scan addressing, per-slice CABAC init, WPP substreams via the
  §7.4.7.1 entry points with the coded-byte → RBSP boundary mapping,
  `end_of_subset_one_bit`, §9.3.2.4/.5 context storage/sync) →
  picture reconstruction → §8.3.1..§8.3.5 reference cycle → output
  reorder. `decode_annexb_sequence` for one-shot buffers;
  `SequenceDecoder` for streaming (`push_nal_unit` / `flush` /
  `take_decoded`).
* **Registry decoder** (`decoder`) — the `oxideav_core::Decoder`
  contract: Annex B packets in, output-order `VideoFrame`s out (8-bit
  planes byte-per-sample; >8-bit little-endian 16-bit), reorder queue
  bounded by `sps_max_num_reorder_pics`, packet-PTS re-attachment,
  Annex B extradata, flush-then-`Eof` semantics. `make_decoder` is the
  direct factory endpoint.
* **Headers** — VPS / SPS / PPS (§7.3.2, incl. range + SCC extension
  bodies), VUI + HRD (§E.2), slice segment headers (§7.3.6, all slice
  types, RPS forms, `ref_pic_lists_modification()`,
  `pred_weight_table()`, entry points), §7.4.8 RPS materialization,
  scaling lists (§7.3.4 / §7.4.5).
* **CABAC + slice-data walk** — the §9.3 engine, per-syntax-element
  binarizations (§9.3.4.2), and the complete §7.3.8 syntax tree: SAO,
  coding quadtree, coding units (intra / inter / skip / PCM gates),
  prediction units (merge, AMVP with the §7.3.8.9 interleaved
  `mvd_coding` order), transform tree + units, residual coding with
  sign hiding and the §7.4.9.11 mode-dependent scans. Picture-level
  parse state supplies cross-CTU / cross-slice / cross-tile §6.4.1
  neighbour availability for every ctxInc and the §8.4.2 MPM
  derivation.
* **Reconstruction** — §8.4 intra prediction (all 35 modes, reference
  substitution + filtering), §8.5 inter prediction (spatial + temporal
  merge / MVP candidates, §8.5.3.3.3 luma 8-tap / chroma 4-tap
  interpolation, default weighted combine), §8.6 scaling + inverse
  DST-VII / DCT-II, §8.6.1 QP derivation (per-QG `qPY_PRED`
  neighbour/decode-order prediction with slice / WPP-row resets),
  §7.3.8.10 deferred-chroma placement, cbf-clear chroma prediction.
* **In-loop filters** — §8.7.2 deblocking (edge flags, bS, per-position
  QP, luma strong/weak + chroma filters, slice / tile boundary
  gating) and §8.7.3 SAO (band + edge offsets, merge resolution,
  boundary-aware edge classification).
* **Reference machinery** — §8.3.1 POC, §8.3.2 RPS marking, §8.3.4
  reference lists, §8.3.5 collocated picture, the DPB, and the
  per-picture decode cycle threading motion fields for temporal MVP.

Sixteen embedded-fixture regression tests pin the byte-exact decodes
(the fixture bytes are the staged public-test-corpus streams
themselves), on top of ~800 unit tests.

## Not yet implemented

* The §8.5.3.3.4.3 **explicit** weighted-prediction sample combine
  (`weighted_pred_flag == 1` with non-default weights — the staged
  corpus' P slices carry default weights, which the default combine
  reproduces exactly).
* PCM sample reconstruction (§8.4.5.2) — the syntax gate parses, the
  IPCM sample read-back is not wired.
* True multi-tile streams (the §6.5.1 tiling machinery and the
  per-tile CABAC re-init points exist, but the staged corpus has no
  tiles-enabled fixture to pin byte-exactness — see the corpus notes
  on the encoder limitation).
* Dependent slice segments (§9.3.2.4 `TableStateIdxDs` storage and
  SliceAddrRs inheritance).
* hvcC (`HEVCDecoderConfigurationRecord`) extradata form — containers
  re-frame to Annex B for now.
* Encoder.

## License

MIT — see [LICENSE](./LICENSE).
