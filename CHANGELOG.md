# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
