# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Other

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
