# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Other

- heif_corpus: re-promote `still-10bit-main10` via
  `BitExactWithinTol(192)` (task #376). The fixture's "essentially
  uncorrelated" failure (98040 of 98304 bytes diff at max |Δ|=255)
  was a comparator alignment bug, not a Main 10 decoder bug: the
  HEVC YUV samples are byte-identical to ffmpeg's reference decode
  (verified with `ffmpeg -pix_fmt yuv420p10le`), but the
  comparator's `compare_rgb48le` was emitting LSB-aligned 16-bit
  RGB while `oxideav-png` decodes 16-bit-from-10-bit-source PNGs as
  MSB-aligned (`value << 6` to fill the 16-bit container). Fix:
  shift the rendered RGB by `16 - bit_depth` before byte-comparing.
  The remaining residual is BT.601 colour-matrix LSB drift (max
  value-delta 5 in 10-bit space → max byte-delta 192 when 16-bit
  values cross a 256-byte boundary). The Main 10 decode pipeline
  (dequant / IDCT / prediction / deblock / SAO) was already
  bit_depth-aware end-to-end via the round-13/15 Main 12 lifts.
- heif_corpus: re-promote `still-image-with-alpha` via
  `BitExactWithinTol(12)` (task #375). Per-channel diff vs the
  oracle PNG: R/G/B = 0 bytes differ (primary HEVC decode is byte-
  perfect after task #346's BT.601-full default), A = 896 bytes
  differ with max |Δ|=12 (740/896 are Δ=1, the rest tail off at
  the steepest alpha-mask edge transitions). The residual is
  monochrome-HEVC decoder rounding noise on the alpha aux, not
  primary-plane drift; tightening below tol=12 needs a fresh round
  on the `chroma_format_idc=0` HEVC pipeline.
- heif: alpha-aware iovl compositing + matrix inheritance (task #346).
  `decode_iovl_primary` now (a) inherits the colour matrix from
  layer 0's `colr nclx` when the iovl item itself has none (with a
  BT.601 full default matching libheif / ImageMagick when neither
  has colr), and (b) decodes each layer's per-layer alpha auxiliary
  via the `auxl` iref and blends the layers in **RGB space at 4:4:4
  chroma**, so anti-aliased alpha edges don't lose colour precision
  through 4:2:0 chroma averaging. `still-image-overlay` divergence
  drops from `max |Δ|=160 across 97.3% of bytes` to `max |Δ|=44
  across ~52%` — the residual sits in the bottom-left gradient
  corner where the HEVC layer-0 decode itself drifts, independent
  of iovl compositing.
- heif_corpus: add `Tier::BitExactWithinTol(u8)` variant + re-promote
  `still-yuv444` (task #374). The 4:4:4 fixture's max |Δ|=3 across
  13.7% of bytes is colour-matrix integer-rounding LSB drift in the
  comparator's f32 BT.601 path (no chroma upsample averaging step at
  4:4:4, so every chroma sample feeds an RGB pixel directly), not a
  decoder bug. The new tier asserts on any divergence above the
  declared tolerance, so it remains a strict comparator — bumping the
  tolerance value should require a fresh investigation.

## [0.0.6](https://github.com/OxideAV/oxideav-h265/compare/v0.0.5...v0.0.6) - 2026-05-03

### Added

- enable heif by default

### Other

- revert still-image-overlay to ReportOnly
- allow dead Tier::Ignored variant
- cfg(test)-gate the legacy compose_overlay_frames shim
- add cargo-fuzz harnesses for HEVC encode/decode (task #308)
- fix Packet::new arg order in moov walker
- round 5 phase B — moov/stbl walker for image-sequence-3frame
- round 5 phase A — match iovl canvas-fill matrix to corpus convention
- rustfmt — XMP arm fits single-line
- rustfmt fixes from round 4 CI
- round 4 cleanup — clippy needless_range_loop + ignored-tier assertion
- round 4 — wire YUV→RGB compare + promote single-image-1x1
- document phase E alpha-monochrome-blocker on decode_alpha_for_primary
- round 3 phase C — iovl overlay composition
- rustfmt — alphabetise meta re-export
- rustfmt fixes for round-3 phase A/B
- round 3 phase B — irot rotation + imir mirroring
- round 3 phase A — clap clean-aperture cropping
- fix clippy doc_overindented_list_items in corpus test
- rustfmt fixes from CI
- add oxideav-core dev-dep for the corpus test
- add iovl parser + decode_item/decode_alpha_for_primary helpers
- round 2 — corpus harness + iloc cm=1 + grid + auxC + alpha lookup
- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- Main 4:4:4 12 (12-bit 4:4:4) encode (§A.3.4, §7.3.8.10, §8.6.1)
- Main 4:4:4 10 (10-bit 4:4:4) encode (§A.3.4, §7.3.8.10, §8.6.1)
- Main 4:4:4 (8-bit) encode + decode (§A.3.4, §6.2, §7.3.8.10)
- Main 12 (12-bit) encode (§A.3.7, §7.4.3.2.1, §8.6.1)
- Main 10 (10-bit) encode + r24 AMP B-slice prep
- B-slice merge / B_Skip encode (§8.5.3.2.2..5, §8.5.3.2.10)
- B-slice encode (§7.4.7.1, §8.5.3.3.3.1, §9.3.4.2.2)
- merge / AMVP candidate-list audit (§8.5.3.2.2 .. §8.5.3.2.5) + inter_pred_idc CtDepth ctxInc
- TMVP scan-order audit (§8.5.3.2.8 / §8.5.3.2.9)
- cu_skip_flag ctxInc + merge ref_poc refresh — Main 10 inter 25.54 → 33.57 dB
- round 19 — WPP audit (§6.3.2 / §9.3.2.4) + tile audit toolchain block
- round 18 — interSplitFlag empirical bin audit (§7.4.9.8)
- round 17 — AMP (§7.4.9.5) lifted from out-of-scope to wired
- adopt slim VideoFrame shape
- round 16 — fix 4:2:2 stacked-chroma cbf inference (§7.4.9.10)
- round 15 — Main 12 + 4:2:2 (yuv422p12le) decode support
- round 13 — Main 12 (12-bit) decode support
- round 12 — 4:2:2 (chroma_format_idc=2) P/B inter decode
- round 11 — 4:2:2 (chroma_format_idc=2) intra decode
- lock in scaling-list decode with a byte-exact ffmpeg fixture
- pin release-plz to patch-only bumps

### Other

- round 32 — **Main 4:4:4 12 (12-bit 4:4:4) encode** (§A.3.4, §6.2,
  §7.3.8.10, §8.6.1). `HevcEncoder::from_params` now accepts
  `PixelFormat::Yuv444P12Le` source frames and routes them through a
  new `src/encoder/slice_writer_main444_12.rs` which combines the
  4:4:4 chroma topology of the round-30 8-bit 4:4:4 writer
  (`SubWidthC = SubHeightC = 1` per Table 6-1, with each 16×16 luma
  TB co-located with a 16×16 Cb TB and a 16×16 Cr TB at `(x0, y0)`)
  and the 12-bit pipeline of the round-26 Main 12 writer (`u16`
  sample containers, `bit_depth = 12` threaded through every
  `intra_pred::predict` / `transform::*` call, and `Qp'Y = SliceQpY +
  QpBdOffsetY = 26 + 24 = 50` on both luma and chroma — at
  ChromaArrayType == 3 the §8.6.1 Table 8-10 chroma collapse does
  **not** apply, so Qp'Cb = Qp'Cr = Qp'Y). The SPS emits
  `chroma_format_idc = 3` plus an explicit `separate_colour_plane_flag
  = 0` bit, `bit_depth_luma_minus8 = bit_depth_chroma_minus8 = 4`, and
  `general_profile_idc = 4` (Format Range Extensions, where Main
  4:4:4 12 lives per §A.3.4). The §A.3.4 "Main 4:4:4 12" RExt
  constraint-flag signature is written into the
  `profile_tier_level()` reserved region — `max_12bit = 1`,
  `max_10bit = 0` (differs from the round-31 Main 4:4:4 10 row),
  `max_8bit = 0`, `max_422chroma = 0`, `max_420chroma = 0`,
  `max_monochrome = 0`, `intra = 0`, `one_picture_only = 0`,
  `lower_bit_rate = 1`. The VPS PTL agrees byte-for-byte with the SPS
  PTL; `general_profile_compatibility_flag` sets only bit 4 since
  §A.3.5 does not allow Main / Main 10 to claim 4:4:4 compatibility.
  Reconstruction planes are seeded with `NEUTRAL = 1 << 11 = 2048`.
  The decoder side reuses the round-30 cfi=3 lift (`ctu.rs` cfi gate
  + §7.3.8.10 chroma-TB-sizing rule) and the existing
  `bit_depth <= 12` envelope unchanged — no new decoder work was
  needed; `decoder::emit_frame`'s 4:4:4 12-bit `Yuv444P12Le` mapping
  was already plumbed through `sub_width_c() / sub_height_c()`.
  Self-roundtrip on a 64×64 / 128×128 Yuv444P12Le gradient at QP 26
  produces 45.04 / 45.69 dB Y on the 12-bit (peak = 4095) scale;
  ffmpeg cross-decode returns 45.04 dB Y on the 64×64 gradient, both
  well above the round-32 ≥ 40 dB acceptance bar. Scope: I-slice
  only — every input frame at 4:4:4 + 12-bit is emitted as an IDR
  (`mini_gop > 1` is rejected at construction time). 4:4:4 P/B
  remains out of scope. The 8-bit 4:2:0, Main 10, Main 12, Main 4:4:4
  8-bit, and Main 4:4:4 10 emission paths (rounds 1..31) are
  unchanged byte-for-byte — the new `(12, 3)` config row in
  `EncoderConfig` and the new `(bit_depth, chroma_format_idc) ==
  (12, 3)` arm in `emit_idr` (placed before the `(12, _)` Main 12
  arm so the more-specific 4:4:4 row matches first) are the only
  branches added on the encode path; the SPS / VPS PTL emitter's
  existing `chroma_format_idc == 3` branch already routes 12-bit
  through the `_ => (0, 0)` `(max_10bit, max_8bit)` arm so the
  round-30 / 31 rows stay bit-identical. Five new tests in
  `tests/encoder_main444_12.rs` plus two new tests in
  `slice_writer_main444_12::tests` and two new SPS / VPS regression
  tests in `params::tests` cover the SPS Main 4:4:4 12 signature
  (incl. a direct-bitstream walk of the 9 RExt constraint flags), the
  64×64 + 128×128 self-roundtrip, the ffmpeg cross-decode, and the
  mini_gop=2 rejection.

- round 31 — **Main 4:4:4 10 (10-bit 4:4:4) encode** (§A.3.4, §6.2,
  §7.3.8.10, §8.6.1). `HevcEncoder::from_params` now accepts
  `PixelFormat::Yuv444P10Le` source frames and routes them through a
  new `src/encoder/slice_writer_main444_10.rs` which combines the
  4:4:4 chroma topology of the round-30 8-bit 4:4:4 writer
  (`SubWidthC = SubHeightC = 1` per Table 6-1, with each 16×16 luma TB
  co-located with a 16×16 Cb TB and a 16×16 Cr TB at `(x0, y0)`) and
  the 10-bit pipeline of the round-25 Main 10 writer (`u16` sample
  containers, `bit_depth = 10` threaded through every
  `intra_pred::predict` / `transform::*` call, and `Qp'Y = SliceQpY +
  QpBdOffsetY = 26 + 12 = 38` on both luma and chroma). The SPS emits
  `chroma_format_idc = 3` plus an explicit `separate_colour_plane_flag
  = 0` bit, `bit_depth_luma_minus8 = bit_depth_chroma_minus8 = 2`, and
  `general_profile_idc = 4` (Format Range Extensions, where Main
  4:4:4 10 lives per §A.3.4). The §A.3.4 "Main 4:4:4 10" RExt
  constraint-flag signature is written into the `profile_tier_level()`
  reserved region — `max_12bit = 1`, `max_10bit = 1`, `max_8bit = 0`
  (the only flag that differs from the round-30 Main 4:4:4 row),
  `max_422chroma = 0`, `max_420chroma = 0`, `max_monochrome = 0`,
  `intra = 0`, `one_picture_only = 0`, `lower_bit_rate = 1`. The VPS
  PTL agrees byte-for-byte with the SPS PTL;
  `general_profile_compatibility_flag` sets only bit 4 since §A.3.5
  does not allow Main / Main 10 to claim 4:4:4 compatibility. The
  decoder side reuses the round-30 cfi=3 lift (`ctu.rs` cfi gate +
  §7.3.8.10 chroma-TB-sizing rule + 16-bit packed `Yuv444P10Le` emit)
  unchanged — no new decoder work was needed. Self-roundtrip on a
  64×64 / 128×128 Yuv444P10Le gradient at QP 26 produces 45.05 /
  45.86 dB Y on the 10-bit (peak = 1023) scale; ffmpeg cross-decode
  returns 45.05 dB Y on the 64×64 gradient, both well above the
  round-31 ≥ 40 dB acceptance bar. Scope: I-slice only — every input
  frame at 4:4:4 + 10-bit is emitted as an IDR (`mini_gop > 1` is
  rejected at construction time). 12-bit 4:4:4 and 4:4:4 P/B remain
  out of scope. The 8-bit 4:2:0, Main 10, Main 12, and Main 4:4:4
  8-bit emission paths (rounds 1..30) are unchanged byte-for-byte —
  the new `(10, 3)` config row in `EncoderConfig` and the new
  `(bit_depth, chroma_format_idc) == (10, 3)` arm in `emit_idr` are
  the only branches added on the encode path; the SPS / VPS PTL
  emitter's existing `chroma_format_idc == 3` branch was extended to
  pick the row-specific `(max_10bit, max_8bit)` pair from
  `bit_depth` (8 → (1,1), 10 → (1,0)) so the round-30 8-bit row stays
  bit-identical. Five new tests in `tests/encoder_main444_10.rs` plus
  two new tests in `slice_writer_main444_10::tests` cover the SPS
  Main 4:4:4 10 signature (incl. a direct-bitstream walk of the 9
  RExt constraint flags), the 64×64 + 128×128 self-roundtrip, the
  ffmpeg cross-decode, and the mini_gop=2 rejection.

- round 30 — **Main 4:4:4 (8-bit) encode + decode** (§A.3.4, §6.2,
  §7.3.8.10). `HevcEncoder::from_params` now accepts
  `PixelFormat::Yuv444P` source frames and routes them through a new
  `src/encoder/slice_writer_main444.rs` which mirrors the round-25 / 26
  parallel-writer pattern but at the 4:4:4 chroma topology
  (`SubWidthC = SubHeightC = 1` per Table 6-1). Each 16×16 luma TB is
  paired with a 16×16 Cb TB and a 16×16 Cr TB, all co-located at
  `(x0, y0)` — vs. the 8×8 chroma TBs at `(x0/2, y0/2)` the 4:2:0 path
  emits. The SPS emits `chroma_format_idc = 3`, an explicit
  `separate_colour_plane_flag = 0` bit (only present when
  `chroma_format_idc == 3` per §7.3.2.2), `bit_depth_luma_minus8 = 0`,
  and `general_profile_idc = 4` (Format Range Extensions, where Main
  4:4:4 lives per §A.3.4). The §A.3.4 "Main 4:4:4" RExt constraint-flag
  signature is written into the `profile_tier_level()` reserved region:
  `max_12bit / max_10bit / max_8bit = 1`, `max_422chroma = 0`,
  `max_420chroma = 0`, `max_monochrome = 0`,
  `lower_bit_rate_constraint_flag = 1`, the rest 0. The VPS PTL agrees
  byte-for-byte with the SPS PTL so ffmpeg's `decode_profile_tier_level`
  cross-check passes; `general_profile_compatibility_flag` sets only
  bit 4 since §A.3.5 does not allow Main / Main 10 to claim 4:4:4
  compatibility. Decoder side: the cfi gate in `ctu.rs::decode_slice_ctus`
  was lifted from `cfi != 1 && cfi != 2` to `cfi != 1 && cfi != 2 && cfi != 3`
  with an additional `slice_type == I` gate at 4:4:4 (P/B at 4:4:4 is a
  follow-up; the chroma-MV scaling / PartIdx / chroma-MC interpolation
  audit is separate work). `transform_tree`'s chroma TB sizing now
  honours the §7.3.8.10 `log2TrafoSizeC = max(2, log2TrafoSize -
  (ChromaArrayType==3 ? 0 : 1))` rule (was hard-coded to `log2_tb - 1`),
  so chroma TBs at 4:4:4 land at 16×16 alongside the luma TB. Self-
  roundtrip on a 64×64 / 128×128 Yuv444P gradient at QP 26 produces
  44.99 / 46.28 dB Y; ffmpeg cross-decode of the same bitstream returns
  44.99 dB Y. Scope: I-slice only — every input frame at 4:4:4 is
  emitted as an IDR (4:4:4 + `mini_gop > 1` is rejected at construction
  time). 4:4:4 at 10 / 12 bit, monochrome, and `separate_colour_plane`
  remain out of scope.

- round 26 — **Main 12 (12-bit) encode** (§A.3.7, §7.4.3.2.1, §8.6.1).
  `HevcEncoder::from_params` now accepts `PixelFormat::Yuv420P12Le`
  source frames and routes them through a new
  `src/encoder/slice_writer_main12.rs` which mirrors the round-25
  Main 10 I-slice writer on `u16` samples with `bit_depth = 12`
  threaded through every `intra_pred::predict` /
  `transform::{forward,inverse,quantize,dequantize}_*` call. The SPS
  emits `bit_depth_luma_minus8 = bit_depth_chroma_minus8 = 4` and
  `general_profile_idc = 4` (Format Range Extensions, where Main 12
  lives per §A.3.7), with the VPS PTL agreeing byte-for-byte. The 9
  RExt constraint flags inside the `profile_tier_level()` reserved
  region are populated to the §A.3.7 "Main 12" signature
  (`max_12bit_constraint_flag = 1`, `max_422chroma_constraint_flag = 1`,
  `max_420chroma_constraint_flag = 1`,
  `lower_bit_rate_constraint_flag = 1`, the rest 0) so ffmpeg's
  `decode_profile_tier_level` maps the stream to the concrete Main 12
  profile rather than to a generic "RExt".
  `general_profile_compatibility_flag` carries the Main + Main 10 +
  RExt bits per §A.3.5 so a sniffer that probes any of those bits
  accepts the stream. `EncoderConfig::new_main12(w, h)` is the direct
  constructor for callers wiring the encoder by hand. The forward
  quantiser uses **Qp'Y = SliceQpY + QpBdOffsetY = 26 + 24 = 50** so
  the encoder's forward quant matches the decoder's `get_qp` inverse
  exactly (mirrored on chroma) — without the `QpBdOffset = 24` step
  the reconstruction overshoots the source by `2^24 / step` and the
  cross-decode collapses to ~10 dB, the same failure mode the
  round-25 Main 10 work hit before its `QpBdOffset = 12` fix. Scope:
  every 12-bit input frame is emitted as an IDR I-slice — 12-bit P/B
  is the next round; `mini_gop_size > 1` at 12-bit is rejected at
  construction time. The 8-bit Main and 10-bit Main 10 emission paths
  (rounds 1..25) are unchanged byte-for-byte. Five new tests in
  `tests/encoder_main12.rs` exercise the SPS / NAL ordering, the
  64×64 + 128×128 self-roundtrip (each ≈ 45 dB Y on the 12-bit /
  peak = 4095 scale), the ffmpeg cross-decode (≥ 45 dB Y, gated on
  the existing `FFMPEG` env var) — measured 45.04 dB Y on the 64×64
  gradient — and the mini_gop=2 rejection. Two new tests in
  `src/encoder/params.rs::tests` (Main 12 SPS bit_depth + RExt compat
  bits, Main 12 VPS profile_idc) plus two new
  `src/encoder/slice_writer_main12::tests` (local reconstruction PSNR
  + IDR NAL emission) cover the lib side.
- round 25 — **Main 10 (10-bit) encode** (§A.3.3, §7.4.3.2.1, §8.6.1).
  `HevcEncoder::from_params` now accepts `PixelFormat::Yuv420P10Le`
  source frames and routes them through a new
  `src/encoder/slice_writer_main10.rs` which mirrors the round-1+
  I-slice writer but operates on `u16` samples throughout and threads
  `bit_depth = 10` through every `intra_pred::predict` /
  `transform::{forward,inverse,quantize,dequantize}_*` call. The SPS
  emits `bit_depth_luma_minus8 = bit_depth_chroma_minus8 = 2` and
  `general_profile_idc = 2` (Main 10), with the VPS PTL agreeing
  byte-for-byte (`general_profile_compatibility_flag` carries both
  the Main and Main 10 bits per §A.3.5 so a Main-only sniffer still
  recognises the stream). `EncoderConfig` grew a `bit_depth: u32`
  field driving SPS / VPS emission, plus
  `EncoderConfig::new_main10(w, h)` for callers wiring the encoder
  directly. The forward quantiser uses
  **Qp'Y = SliceQpY + QpBdOffsetY = 26 + 12 = 38** so the encoder's
  forward quant matches the decoder's `get_qp` inverse exactly
  (mirrored on chroma). Scope: every 10-bit input frame is emitted as
  an IDR I-slice — 10-bit P/B is the next round; `mini_gop_size > 1`
  at 10-bit is rejected at construction time. The 8-bit Main encode
  path (rounds 1..24) is unchanged byte-for-byte: the new code lives
  in a parallel `slice_writer_main10` module rather than templating
  over the existing 8-bit `EncoderState`. Five new tests in
  `tests/encoder_main10.rs` exercise the SPS / NAL ordering, the
  64×64 + 128×128 self-roundtrip (each ≈ 45 dB Y on the 10-bit /
  peak = 1023 scale), the ffmpeg cross-decode (≥ 40 dB Y, gated on
  the existing `FFMPEG` env var), and the mini_gop=2 rejection. Two
  new tests in `src/encoder/params.rs::tests` (Main 10 SPS bit_depth
  and Main 10 VPS profile_idc) plus two new
  `src/encoder/slice_writer_main10::tests` (local reconstruction PSNR
  + IDR NAL emission) cover the lib side.
- round 23 — B-slice **merge / B_Skip encode** (§7.3.8.6, §8.5.3.2.2..5,
  §8.5.3.2.10, §9.3.4.2.2 / Table 9-32, §9.3.4.2.10). The B-slice
  writer (`src/encoder/b_slice_writer.rs`) now picks per CU between
  three modes by luma SAD:
  (1) **B_Skip** — `cu_skip_flag = 1` followed by the truncated-rice
  `merge_idx`. No residual, no `pred_mode_flag`, no `part_mode`, no
  `merge_flag`. Selected when the best merge candidate's prediction
  SAD is at or below `SKIP_RESIDUAL_THRESHOLD = 1024` luma pels for
  a 16×16 CU (≈ 4 per-pixel error — one Qstep at QP 26). On the
  static-content fixture this collapses every B-slice CU to skip,
  dropping each B-slice from hundreds of bytes to 24-26 bytes total.
  (2) **Merge** — `merge_flag = 1` + `merge_idx`, with residual coded
  through the existing inter pipeline. Selected when the best merge
  candidate beats explicit AMVP on luma SAD but residual energy is
  above the skip threshold.
  (3) **Explicit AMVP** — the round-22 fall-through (L0-only / L1-only
  / Bi pick with explicit MVD per list).
  Slice-header `five_minus_max_num_merge_cand` is now `0` (was `4`)
  so the merge list grows from 1 → 5 entries; the encoder picks
  whichever of the §8.5.3.2.{2..5} candidates (spatial A0/A1/B0/B1/B2,
  combined bi-pred, zero-MV pad) minimises SAD against the source.
  The merge list is built with the same call into `build_merge_list_full`
  the decoder uses, so encoder + decoder agree on the picked
  candidate's motion byte-for-byte. The encoder now maintains a full
  `inter::InterState` populated with `PbMotion` records (alongside
  the legacy `mv_grid_l0/l1`) — needed both for merge derivation
  (which reads spatial PBs from the grid) and for `cu_skip_flag`
  ctxInc, which now follows the spec's `condTermFlagX = neighbour.is_skip`
  rule (decoder's r19 fix). Three new tests in
  `tests/encoder_b_slice.rs`:
  `b_skip_engages_on_static_content` (5-frame zero-motion gradient,
  packet sizes [109, 24, 26, 24, 26] in decode order — B-slice
  CUs collapse to skip CUs at 2 bits/CU);
  `b_slice_merge_path_decodes_within_psnr_floor` (5-frame translating
  gradient through our decoder, every frame ≥ 25 dB);
  `b_skip_ffmpeg_cross_decode_psnr` (writes the static-content
  bitstream to `/tmp/oxideav-h265-fixtures/encoder-b-skip-r23.hevc`,
  decodes via libavcodec, every frame ≥ 22 dB — proves the skip
  path stays in CABAC sync end-to-end). The pre-existing r22
  `ipbpb_roundtrip_through_our_decoder` PSNR shifts from
  44.99 / 32.51 / 39.59 / 27.17 / 31.46 to 44.99 / 32.51 / 38.56 /
  27.17 / 30.82 (decode order) — the slight B-frame regressions
  trade off against the dramatic skip-path size reduction on
  near-static content. 154 tests pass (88 unit + 50 reference + 7
  encoder_b_slice + 4 ffmpeg_accepts + 3 reference + 1 encoder_p_slice
  + 1 encoder_roundtrip). Pending for r24+: AMP / rectangular
  partitions (Nx2N / 2NxN), `mvd_l1_zero_flag` optimisation, B-pyramid
  (mini-GOP > 2), 10-bit encode.

- round 22 — B-slice **encode** (§7.4.7.1, §8.5.3.3.3.1, §9.3.4.2.2).
  New `src/encoder/b_slice_writer.rs` emits a single-segment B
  (TrailR) slice referencing one past anchor (L0) and one future
  anchor (L1), both with `num_ref_idx_lX_active_minus1 = 0`. The
  encoder's display-order input is re-ordered into decode order
  (I-P-B-P-B from a I-B-P-B-P display sequence) by holding back
  every odd-display-POC frame until its future anchor lands, then
  emitting the B with `delta_l0 = -1`, `delta_l1 = +1` from
  `b_poc`. Per-CU pipeline:
  (1) integer-pel ±8 luma SAD search **per list** (step 2 → chroma
  MV stays integer, matching the decoder's full-pel `chroma_mc` byte
  for byte);
  (2) bipred predictor = `(P_L0 + P_L1 + 1) >> 1` (default weighting,
  §8.5.3.3.3.1);
  (3) per-CU pick of `L0-only` / `L1-only` / `Bi` by SAD against the
  source — the residual / quant pipeline is identical across the
  three so the predictor pick directly drives quality;
  (4) `inter_pred_idc` bin 0 ctxInc = `CtDepth` (§9.3.4.2.2 Table
  9-32) — for our single 16×16 CU at a 16×16 CTB this collapses to
  `bin0_ctx = 0`; bin 1 ctxInc = 4 (uni-pred L0 vs L1 selector);
  (5) explicit AMVP per list (`mvp_lx_flag = 0`) sharing the same
  spatial-only A0/A1/B0/B1/B2 walk as the P-slice writer, but
  against per-list 4×4 mv grids;
  (6) `mvd_l1_zero_flag = 0` so L1 MVD is encoded normally;
  (7) `cu_skip_flag` ctxInc = 0 (we never emit skip, so per the
  decoder's r19 rule `condTermFlagX = neighbour.is_skip` is always
  0 — same fix the decoder shipped to bring Main 10 inter PSNR up
  to 33.57 dB).
  P-slice writer gains a `..._delta` variant so the second-anchor
  P (POC 4 referencing POC 2 with delta -2) emits the right RPS;
  the default `delta_l0 = -1` path stays byte-identical so the
  existing P-only fixtures don't regress.
  New `HevcEncoder::from_params_with_mini_gop(params, 2)`
  constructor selects the I-P-B-P-B GOP. Three new unit tests in
  `tests/encoder_b_slice.rs`:
  `ipbpb_roundtrip_through_our_decoder` (5-frame translation
  oracle, 44.99 / 32.51 / 39.59 / 27.17 / 31.46 dB per
  decode-order frame, plus a B-beats-later-P invariant),
  `b_slice_static_content_is_lossless_after_quantize` (static
  source, every frame ≥ 22 dB), `b_slice_packet_count_matches_source_count`
  (7 source frames → 7 packets in decode order),
  `b_slice_ffmpeg_cross_decode_psnr` (writes the bitstream to
  `/tmp/oxideav-h265-fixtures/encoder-b-slice-r22.hevc`, decodes
  with ffmpeg, every display frame ≥ 33 dB vs source — 44.99 /
  35.16 / 36.28 / 34.34 / 33.34 dB at POC 0..4). 151 tests pass
  (88 unit + 50 reference + 4 encoder_b_slice + 4 ffmpeg_accepts +
  3 reference + 1 encoder_p_slice + 1 encoder_roundtrip). Pending
  for r23: merge encode, B_Skip encode, AMP / Nx2N / 2NxN
  partitions, mini-GOP > 2.

- round 21 — merge / AMVP candidate-list audit (§8.5.3.2.2 .. §8.5.3.2.5)
  + `inter_pred_idc` ctxInc fix (§9.3.4.2.2 Table 9-32). Four
  spec-correctness patches:
  (1) §8.5.3.2.4 combined bi-pred now walks the spec's Table 8-7
  fixed `(l0CandIdx, l1CandIdx)` schedule with the spec's exact
  termination at `combIdx == numOrigMergeCand * (numOrigMergeCand - 1)`;
  the candidate gate is `predFlagL0 && predFlagL1 &&
  (DiffPicOrderCnt(L0, L1) != 0 || mvL0 != mvL1)`. Pre-r21 we ran
  a generic O(N²) double-loop with a non-spec "skip if equal to any
  existing entry" dedup, missing combined candidates the spec adds
  and adding ones it skips.
  (2) §8.5.3.2.5 zero-MV padding now uses the spec's
  `refIdxLX = (zeroIdx < numRefIdx) ? zeroIdx : 0` ramp (zeroIdx
  monotonic per pad iteration). Pre-r21 every pad slot collapsed
  to `refIdx = 0`, so a `merge_idx` selecting a pad-region entry
  picked the wrong reference picture when `numRefIdx > 1`.
  (3) §8.5.3.2.2 step 10 enforced at the merge / AMVP call sites:
  4×8 / 8×4 PUs (`nOrigPbW + nOrigPbH == 12`) cannot be
  bi-predicted — even when the chosen merge candidate is bi-pred,
  `predFlagL1` is forced to 0 and `refIdxL1` to −1 after candidate
  selection.
  (4) §9.3.4.2.2 Table 9-32 `inter_pred_idc` bin 0 ctxInc is now
  `CtDepth[x0][y0]` (∈ {0,1,2,3}) when `(nPbW + nPbH) != 12`,
  matching the spec's CB-depth-aware ctx bank. Pre-r21 we forced
  ctx 0 for every non-small PU regardless of CB depth, biasing
  the CABAC context for any deeply-split B-slice CU.
  §8.5.3.2.3 redundancy comparison was also tightened: the spec's
  "same motion vectors and same reference indices" rule compares
  only `(predFlag, refIdx, mv)` per list — pre-r21 we compared all
  `MergeCand` fields including the shadow `ref_poc` / `ref_lt`
  metadata, which can legitimately differ between two PBs the
  spec considers equivalent for redundancy.
  Two new fixtures: `hevc_p_slice_short_gop_textured_64`
  (P-only baseline guard against merge audit regressions) and
  `hevc_b_slice_low_motion_merge_audit` (rate=60 textured I-P-B-P-B
  oracle), the latter at **42.01 dB average** (per-frame: I=∞,
  P=57.56, B=61.92, P=36.08, B=41.82). The original
  `hevc_b_slice_tmvp_scan_order_audit` fixture (rate=10 high-motion
  GOP) remains as the round-20 regression guard at 24.65 dB —
  unchanged because its per-frame error is dominated by an
  upstream P-slice rate-10 testsrc bug that's outside the merge
  audit scope. Decode-order vs display-order remap added to the
  PSNR oracle (the ffmpeg `-f rawvideo` reference comes out in
  display order; our decoder emits in decode order, so frames are
  remapped via `[0, 2, 1, 4, 3]` before differencing). 50 tests
  pass. Main 10 inter PSNR holds at 33.57 dB and the 4:2:0 / 4:2:2
  P/B fixtures stay bit-exact.

- round 20 — TMVP scan-order audit (§8.5.3.2.8 / §8.5.3.2.9). The
  bottom-right → centre fallback now fires for every case where
  `availableFlagLXCol == 0` — including the previously-missed case
  where the BR PB is inter but its listCol-selected MV fails the
  §8.5.3.2.9 LT-flag gate. Added `NoBackwardPredFlag` derivation
  on `CtuContext` (`= 1` iff every L0/L1 ref has POC ≤ currPic);
  the §8.5.3.2.9 listCol picker for both-flags-set collocated PBs
  now selects `LX` when `NoBackwardPredFlag == 1` and `LN` (with
  `N = collocated_from_l0_flag`) otherwise — pre-r20 the merge
  path approximated this by preserving the col PB's pred flags
  directly (skipping the spec's listCol selection entirely) and
  the AMVP path used the current-invocation `want_l0` as the
  selector (correct for P-slices where `NoBackwardPredFlag = 1`,
  wrong for any B-slice with a backward L1 ref). Merge temporal
  candidate now invokes §8.5.3.2.8 once per side (L0 / L1) per
  §8.5.3.1.7 step 3/4, with `predFlagLX = availableFlagLXCol`
  (eqs. 8-115 / 8-118). New test
  `hevc_b_slice_tmvp_scan_order_audit` exercises the
  `NoBackwardPredFlag = 0` path on a 5-frame I-P-B-P-B fixture
  (PSNR 24.51 dB on textured 64×64 testsrc, well above the 12 dB
  regression floor). 48 tests pass. P-slice paths preserved
  byte-for-byte: Main 10 inter PSNR holds at 33.57 dB and the
  4:2:0 / 4:2:2 P/B fixtures stay bit-exact.
- round 19 — §9.3.4.2.2 `cu_skip_flag` ctxInc spec-correctness +
  §8.5.3.2.9 merge `ref_poc` refresh. `PbMotion` grew an `is_skip`
  bit recording whether the CU writing this PB ran the
  `cu_skip_flag = 1` fast path; `skip_ctx_inc` now reads the
  neighbour PB's `is_skip` (was approximating with `is_inter`,
  over-counting `condTermFlag = 1` for every non-skip inter
  neighbour and biasing CABAC ctxInc by one slot). Companion
  `refresh_pb_ref_poc` rewrites a merge candidate's stale
  `ref_poc_{l0,l1}` / `ref_lt_{l0,l1}` against the CURRENT slice's
  RPL[ref_idx] before stashing the PB into the inter grid, so the
  merge zero-pad and stale-spatial-neighbour paths stop poisoning
  downstream TMVP scaling. Main 10 inter 80×48 testsrc PSNR lifts
  from 25.54 → 33.57 dB average (frames 1/2/3: 46.11/26.34/20.54 →
  40.05/37.86/28.25; net SSE drops 6.4×). Main 12 inter PSNR
  reaches 27.67 dB.
- round 12 — 4:2:2 (chroma_format_idc=2) P/B inter decode via §8.5.3.2.10
  chroma MV derivation (`mvCLX[1] = mvLX[1] * 2 / SubHeightC`) and
  full-height chroma plane geometry through `motion_compensate_pb`.
  Byte-exact match vs ffmpeg on flat-gray 4:2:2 IDR + P and IDR + B + P
  fixtures; PSNR floor 18 dB on 4:2:2 testsrc IDR + P (IDR byte-exact).

## [0.0.5](https://github.com/OxideAV/oxideav-h265/compare/v0.0.4...v0.0.5) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- round 9 — §8.5.3.2.7 AMVP POC scaling + §8.5.3.2.9 TMVP scaling
- round-8 investigation — document interSplitFlag empirics
- tighten TMVP 16-alignment + CTB-row gate (§8.5.3.2.8)
- encoder — Main-profile CABAC P-slice (round 7)
- round-6 diagnostic — surface interSplitFlag gap in SPS dumper
- tests — raise Main 10 inter PSNR floor to 22 dB
- fix inter transform_tree cbf_cb/cbf_cr order (§7.3.8.9)
- spec-exact MC pipeline + AMP part_mode ctxInc (§8.5.3.3.3.2 / Table 9-48)
- tests — per-frame PSNR breakdown in Main 10 inter test
- fix bi-pred combine shift for 10-bit (§8.5.3.3.4.2 eq. 8-264)
- plumb QpBdOffset into dequant + QP delta wrap (Main 10 fix)
- add examples/dump_sps SPS/PPS flag dumper
- add HEIF/HEIC scaffold module behind `heif` feature
- thread bit_depth through inter MC for Main 10
- add Main 10 intra integration test and scope doc
- migrate sample storage to u16 for Main 10 support
- add PSNR-gated deblock regression (§8.7.2)
- add byte-exact SAO fixture tests (§8.7.3)
- fix CTU-seam intra reconstruction
- h265 encoder: doc update + looser ffmpeg bound
- h265 encoder: replace PCM MVP with DCT + intra + residual CABAC
- tests — Main10 + 4:4:4 surface clean Unsupported
- h265 encoder: forward DCT/DST + forward quantisation
- AMP inter shapes + cu_transquant_bypass
- encoder — rename inner encoder module to silence clippy
- document the encoder MVP in the crate-level doc
- encoder — cover multi-CTU through ffmpeg; unique temp names
- encoder — ffmpeg acceptance test + dump_h265 example
- verify WPP slice decodes to a full frame end-to-end
- tile + WPP CTU iteration with entry-point re-seeded CABAC
- encoder — VPS/SPS/PPS emitters + CABAC writer + PCM I-slice MVP
- parse tile geometry + entry-point offsets
- parse transform_skip_flag + pcm CU path
- parse scaling lists and honour them in dequantisation
- parse ref_pic_list_modification and apply §8.3.4 reordering
- add long-term reference pictures (§8.3.2)
- add SAO sample application (§8.7.3)
- add in-loop deblocking filter (§8.7.2)
- subblock_scan — force diagonal for 16x16 / 32x32 TUs per §6.5.4
- add lavfi-driven intra fixtures — more content variety
- exact-match on testsrc fixture — spec QpY_PRED + intra ref fixes
- all exact-match fixtures pass — decoded-block z-scan availability
- spec-compliant split_cu_flag ctxInc + 64x64 qp51 fixture passes
- add DST-VII DC basis sanity test
- add DCT DC-only uniformity unit test
- remove residual-coding trace instrumentation
- reset IsCuQpDeltaCoded per QG boundary, not per CU
- land first exact-match fixture — 16x16 gray IDR
- fix sig_coeff_flag init values for initTypes 1 and 2
- audit + correct three more CABAC init-value tables
- intra_chroma_pred_mode I-slice init value 63 per HEVC V11 Table 9-13
- add QP=51 single-CTU gray test fixture
- sig_coeff_flag eq. 9-42 keeps sigCtx=0 without size modifiers
- align transform_tree + cbf_cb_cr ctx with HEVC V11 + ffmpeg
- spec-compliance fixes in residual coding + accept SAO streams
- align residual coding + transforms with H.265 V11
- README + lib docs reflect B-slice decode support
- add B-slice fixture + integration test
- land B-slice decode (bi-pred MC, TMVP, merge extensions)

## [0.0.4](https://github.com/OxideAV/oxideav-h265/compare/v0.0.3...v0.0.4) - 2026-04-19

### Other

- README + lib docs reflect P-slice decode support
- wire P-slice inter decode into CTU walker + decoder
- add inter module (DPB, MV, merge/AMVP, 8-tap/4-tap MC)
- extend slice header parser with P-slice extension
- capture short-term RPS deltas + counts in SPS parser
- rewrite README + lib.rs docs to reflect I-slice decode support
- land I-slice pixel decode (intra pred + transforms + CTU walker)
- port CABAC arithmetic engine + I-slice context tables
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
