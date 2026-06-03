# oxideav-h265

A pure-Rust H.265 / HEVC video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Clean-room rebuild — round 28 (2026-06-02).** The prior implementation was
retired under the workspace
[clean-room policy](https://github.com/OxideAV/oxideav/blob/master/docs/IMPLEMENTOR_ROUND.md):
a CTU-level source comment cited a specific named variable and line
number in an external library's HEVC decoder — clean-room provenance
for the surrounding code path could not be defended. Master history
was fully erased per the Hat-3 cold-enforcement procedure.

The rebuild is in progress against the published H.265 specification
(ITU-T Recommendation H.265 | ISO/IEC 23008-2). Round 28 extends the
[`binarization`] module with six more §9.3.4.2 / Table 9-48 closed-form
`ctxInc` derivations — each a pure function of parameters the
slice-data parser already has in hand (no neighbour-table walk, no
CABAC engine drive at this layer). The new entries:
[`binarization::split_transform_flag_ctx_inc`] (Table 9-48 row
`ctxInc = 5 − log2TrafoSize`, four-context bank `{0..=3}` over the
residual-quadtree TB sizes log2 = 2..=5);
[`binarization::cbf_luma_ctx_inc`] (Table 9-48 row
`ctxInc = (trafoDepth == 0) ? 1 : 0`, two-context bank `{0, 1}`);
[`binarization::cbf_cb_ctx_inc`] and
[`binarization::cbf_cr_ctx_inc`] (Table 9-48 row
`ctxInc = trafoDepth`, shared
[`binarization::cbf_chroma_ctx_inc`] helper; the per-component
`ctxIdxOffset` is the caller's responsibility);
[`binarization::inter_pred_idc_ctx_inc`] (Table 9-48: bin 0
`ctxInc = (nPbW + nPbH != 12) ? CtDepth[x0][y0] : 4` — the
`nPbW + nPbH == 12` condition picks out the 8×4 and 4×8 PUs, luma
area 16 samples, which route bin 0 onto the bin-1 context bank;
bin 1 `ctxInc = 4`);
[`binarization::log2_res_scale_abs_plus1_ctx_inc`] (Table 9-48 row
`ctxInc = 4*c + binIdx` for the TR(`cMax = 4`) prefix bins, per-
component banks `{0..=3}` Cb and `{4..=7}` Cr); and
[`binarization::res_scale_sign_flag_ctx_inc`] (Table 9-48 row
`ctxInc = c`, one bit per chroma component). Total tests now 265
(was 251): 14 new tests cover the Table 9-48 row for
`split_transform_flag` across log2TrafoSize 2..=5 plus the
`ctxInc <= 3` bank-bound sweep; `cbf_luma` Table 9-48 row across
trafoDepth 0..=4 plus the `ctxInc <= 1` bank-bound sweep; shared
`cbf_chroma_ctx_inc` identity and the Cb/Cr agreement check;
`inter_pred_idc` bin 0 with `CtDepth` routing (64×64, 16×16 at
depth 2, 8×8 at depth 3), bin 0 escape on the 8×4 / 4×8
16-sample PUs, bin 1 constant `ctxInc = 4`, and the
`ctxInc ∈ {0..=4}` bank-bound sweep across eight PU shapes and
four CtDepths; `log2_res_scale_abs_plus1` Cb and Cr bank
identities plus the disjoint-bank invariant; `res_scale_sign_flag`
two-row identity.

Round 27 had extended the
[`binarization`] module with three more §9.3.4.2 ctxInc derivations
that are pure-functional given the neighbour / sub-block context
(no CABAC engine drive at this layer; callers compose the engine call
themselves): `coded_sub_block_flag` (§9.3.4.2.4 equations 9-35..9-39)
via [`binarization::coded_sub_block_flag_ctx_inc`] +
[`binarization::coded_sub_block_flag_ctx_inc_with_edge`] (the latter
applies the equation 9-36 / 9-37 edge gates `xS < (1 <<
(log2TrafoSize − 2)) − 1` / `yS < (1 << (log2TrafoSize − 2)) − 1`
before delegating); and the §9.3.4.2.2 Table 9-49 left/above ctxInc
shape `ctxInc = (condL && availableL) + (condA && availableA)` as the
shared [`binarization::left_above_ctx_inc`] helper, with the two row
specialisations [`binarization::split_cu_flag_ctx_inc`] (condL/A =
`CtDepth[xNb][yNb] > cqtDepth`) and
[`binarization::cu_skip_flag_ctx_inc`] (condL/A =
`cu_skip_flag[xNb][yNb]`) on top. The §9.3.4.2.4 luma bank is `{0, 1}`
(equation 9-38 `ctxInc = Min(csbfCtx, 1)`), the chroma bank is `{2, 3}`
(equation 9-39 `ctxInc = 2 + Min(csbfCtx, 1)`). The §9.3.4.2.2 ctxInc
is in `{0, 1, 2}` for both row specialisations. Total tests now 251
(was 234): 17 new tests cover the §9.3.4.2.4 luma + chroma banks (both
neighbours zero / one set / both set with the `Min(csbfCtx, 1)`
clamp), the LSB-masking defensive input, the edge-gating at the
right-edge (4×4 TB single-sub-block / 8×8 TB right-edge / 16×16 TB
bottom-edge / 32×32 TB interior, luma + chroma), the §9.3.4.2.2 Table
9-49 `(condL && availableL) + (condA && availableA)` truth table
(unavailable neighbour zeroes its branch even when its cond is true),
the `split_cu_flag` condX strict inequality `CtDepth > cqtDepth`, the
`cu_skip_flag` condX LSB-mask, the four-way split_cu_flag ctxInc table
(both deeper / left deeper / left unavailable / both unavailable), the
cu_skip_flag eight-way truth table (every combination of left/above
flag-and-availability), and a bounded `ctxInc ∈ {0, 1, 2}` invariant
over a small Cartesian product of inputs.

Round 26 had shipped the §9.3.4.2 per-syntax-element binarization +
context-index derivation layer for the two CABAC elements unblocked
by the clean-room trace at
`docs/video/h265/fixtures/main-422-10bit/cabac-cu-qp-delta-last-sig-trace.md`:
`cu_qp_delta_abs` / `cu_qp_delta_sign_flag` (§7.3.8.14, §7.4.9.14) and
`last_sig_coeff_{x,y}_{prefix,suffix}` (§7.3.8.11, §7.4.9.11). The new
[`binarization`] module exposes [`binarization::decode_cu_qp_delta`]
(§9.3.3.10 TR prefix `cMax = 5` + §9.3.3.11 EGk(k=0) escape suffix +
bypass sign flag; per-bin ctxInc via Table 9-32 — bin 0 → ctx 0, bins
1..=4 → ctx 1) plus [`binarization::decode_last_sig_coeff`] (§9.3.3.10
TR prefix with `cMax = (log2TrafoSize << 1) − 1` and the §9.3.4.2.3
per-bin `ctxInc = (binIdx >> ctxShift) + ctxOffset` via
[`binarization::last_sig_coeff_prefix_ctx_offset_shift`] — luma
`ctxOffset = 3*(log2TrafoSize − 2) + ((log2TrafoSize − 1) >> 2)`,
`ctxShift = (log2TrafoSize + 1) >> 2`; chroma `ctxOffset = 15`,
`ctxShift = log2TrafoSize − 2`). The §7.4.9.11 equations 7-74..7-77
position derivation lives in [`binarization::last_sig_coeff_position`]
(returns the pre-scanIdx-2-swap `LastSignificantCoeff{X,Y}`); the
§7.4.9.14 `CuQpDeltaVal = cu_qp_delta_abs * (1 − 2 *
cu_qp_delta_sign_flag)` derivation lives on the returned
[`binarization::CuQpDelta`]. A [`binarization::LastSigCoeffBank`] tag
(X / Y) is exposed for caller-side context-bank routing. The module
sits one layer above the §9.3 arithmetic engine
([`cabac::CabacEngine`], round 11) and consumes context variables
([`cabac::ContextModel`]) supplied by the eventual slice-data parser.
The §9.3.4.2 surface still has many remaining syntax elements
(`sig_coeff_flag`, `coeff_abs_level_greater{1,2}_flag`,
`coeff_abs_level_remaining`, motion-related elements, split / merge /
prediction-mode flags, …) but the binarization scaffold plus the
first two elements unblock the previously-blocked
trace-dependent path and establish the module shape every future
element plugs into. Total tests now 234 (was 218): 16 new tests cover
the cu_qp_delta_abs ctxInc table (Table 9-32), last_sig_coeff
offset/shift for luma + chroma at log2 = 2..=5, the `(binIdx >>
ctxShift) + ctxOffset` ctxInc identity over the 32×32-luma and
4×4-luma rows, the per-size `cMax`, the §7.4.9.11 position derivation
on the trace-observed luma 32×32 `(prefix=6, LastX=8)` + 16×16 chroma
rows, the suffix-nBits table, the TR-prefix terminator and all-ones
escape paths, the cu_qp_delta_abs = 0 path (no sign flag, value = 0),
the §7.4.9.14 signed derivation across the 10 multi-slice-per-frame
trace rows, and a crafted-engine EGk(k=0) decode. Round 25 had decoded the
§7.3.2.3.1 PPS extension-flag block as a new typed
[`pps::PpsExtensionFlags`] sub-struct: when
`pps_extension_present_flag == 1` the eight bits that follow
(`pps_range_extension_flag`, `pps_multilayer_extension_flag`,
`pps_3d_extension_flag`, `pps_scc_extension_flag`,
`pps_extension_4bits`) are read into typed fields rather than swept
into the opaque tail. The remaining `opaque_tail` capture now starts
at the first bit of the first signalled extension body (the
`pps_range_extension()` body if `pps_range_extension_flag == 1`,
otherwise the next set flag's body, otherwise the
`while( more_rbsp_data() ) pps_extension_data_flag` block gated by
`pps_extension_4bits != 0`); and when all five sub-fields are zero —
the dominant Main / Main 10 case — the tail is `None` because only
`rbsp_trailing_bits()` remain, consumed implicitly. The §7.4.3.3.1
inference rules apply for the absent gate: `extension_flags` is
`None` and every flag is inferred to 0. A small predicate
[`PpsExtensionFlags::has_body`] exposes "any of the four extension
flags is set, or `pps_extension_4bits` is non-zero" so downstream
callers can route to the right decoder once the body parsers land.
Total tests now 218 (was 215): four new pps tests cover the
no-body-flag-block decode (no opaque tail), the range-extension
opaque tail capture, the `pps_extension_4bits != 0` opaque tail
capture, and the gate-zero inference path; the prior
`captures_extension_opaque_tail` test (which assumed every set of
extension bits surfaced an opaque tail) is replaced by the
no-body-flag-block test. Round 24 had landed the
§7.4.8 inter-RPS-prediction derivation as the new
[`ShortTermRefPicSet::materialize`] / [`MaterializedShortTermRefPicSet`]
typed builder and wired it into the §7.3.6.1 slice parser to close
the last remaining round-23 deferral point. The explicit-form branch
implements equations 7-63..7-70 (the per-position `UsedByCurrPicS{0,1}`
arrays direct, the cumulative `DeltaPocS0[i] = DeltaPocS0[i-1] -
(delta_poc_s0_minus1[i] + 1)` / `DeltaPocS1[i] = DeltaPocS1[i-1] +
(delta_poc_s1_minus1[i] + 1)` recurrences with their
equation-7-67 / 7-68 seeds). The inter-RPS-prediction branch
implements equation 7-60 (`deltaRps = (1 - 2*delta_rps_sign) *
(abs_delta_rps_minus1 + 1)`) and equations 7-61 (negative side —
source-positives reverse + optional `deltaRps` self-term + source-
negatives forward) and 7-62 (positive side, mirrored), filtering each
candidate entry by its `dPoc < 0` / `dPoc > 0` sign and the per-entry
`use_delta_flag[j]` gate. The
[`SeqParameterSet::materialize_short_term_ref_pic_sets`] SPS-level
helper runs the full chain, threading inter-form entries through the
equation-7-59 `RefRpsIdx = stRpsIdx - (delta_idx_minus1 + 1)` source
lookup so the materialisation is closed under inter-RPS prediction
across the entire SPS list (and for slice-inline construction at
`stRpsIdx == num_short_term_ref_pic_sets`). The §7.3.6.1 slice parser
now consumes the materialised RPS at the in-place
`ref_pic_lists_modification()` call site, picking the active RPS
(inline source via `RefRpsIdx = num_short_term_ref_pic_sets -
(delta_idx_minus1 + 1)` for the inline-inter case), feeding the
derived `UsedByCurrPicS{0,1}` slices into
[`NumPicTotalCurrInputs::from_used_flags`] / `compute`, and walking
the in-place RPLM gate exactly as the explicit-form path did in
round 23. Configurations whose materialisation succeeds reach
`byte_alignment()` end to end; only malformed inter-form chains
(on-wire `used_by_curr_pic_flag` / `use_delta_flag` length not
matching the source's `NumDeltaPocs + 1`) defer to an opaque tail.
Total tests now 215 (was 208): five new sps tests cover the §7.4.8
derivation directly (explicit-form recurrence, inter-RPS positive and
negative `deltaRps` cases, missing-source rejection, length-mismatch
rejection, and an SPS-level chain test); one new slice test covers a
P-slice with an SPS-form inter-predicted RPS that materialises to
`NumPicTotalCurr == 0` and walks the full inter-slice tail without
deferral; the pre-round
`defers_rplm_when_active_st_rps_uses_inter_prediction` test is
preserved with its prose updated to reflect that the deferral is now
the malformed-array path. With this round, every conformant non-IDR
P / B slice — explicit or inter-form short-term RPS — parses end to
end through `byte_alignment()`. Round 23 had wired the
standalone §7.3.6.2 [`RefPicListsModification::parse`] (round 15)
into the §7.3.6.1 slice-header in-place call site, closing the
`pps.lists_modification_present_flag == 1` deferral point for every
configuration whose §7.4.7.2 `NumPicTotalCurr` derivation does **not**
require the §7.4.8 inter-RPS-prediction step. The slice parser now
resolves the active short-term RPS (inline form via the slice-header
`st_ref_pic_set(...)` body; SPS form via the SPS-resident
`short_term_ref_pic_sets[idx]`), feeds the per-position
`used_by_curr_pic_s{0,1}_flag` arrays (only meaningful when
`inter_ref_pic_set_prediction_flag == 0` — the explicit form) plus
the §7.4.7.1-resolved `UsedByCurrPicLt[i]` slice
(`SliceLongTermRefPic::used_by_curr_pic_lt`, round 14) into
[`NumPicTotalCurrInputs::from_used_flags`] / [`NumPicTotalCurrInputs::compute`]
(round 16) to obtain `NumPicTotalCurr`. When `NumPicTotalCurr > 1`
the parser invokes [`RefPicListsModification::parse`] in place and
exposes the result via the new
[`SliceSegmentHeader::ref_pic_lists_modification`] field (an
`Option<RefPicListsModification>`), then continues into the mvd /
cabac-init / collocated / pred-weight-table / merge-candidate / shared
I-slice tail through `byte_alignment()`. When `NumPicTotalCurr <= 1`
the §7.3.6.1 outer gate is statically false: the structure is not
signalled and the parser advances directly into the mvd block (zero
bits consumed for RPLM). When the picked short-term RPS uses
inter-RPS-prediction (`inter_ref_pic_set_prediction_flag == 1`), the
§7.4.8 derivation chain is the remaining blocker and the parser
continues to surface the opaque tail starting at the
`ref_pic_lists_modification()` bit position. With this round, every
non-IDR P / B slice whose active short-term RPS is in explicit form
parses end to end through `byte_alignment()` in the surfaced
configuration; the §7.4.8 inter-RPS-prediction derivation is the
last parser-side deferral inside the §7.3.6.1 walk. Total tests now
208 (was 205): three new tests cover the inline-explicit RPS
`NumPicTotalCurr == 2` in-place parse, the inline-explicit RPS
`NumPicTotalCurr == 1` static skip, and the SPS-form inter-predicted
RPS opaque-tail defer; the pre-round defer-on-flag test is rewritten
into `skips_rplm_when_num_pic_total_curr_is_zero_idr` (IDR has no
RPS block → `NumPicTotalCurr == 0`, gate statically false, full tail
walked).

Round 22 finishes the
§7.3.6.1 entry-point-offset block: the slice-header parser now
captures every per-i `entry_point_offset_minus1[i]`
(`u(offset_len_minus1 + 1)`) into
[`EntryPointOffsets::entry_point_offset_minus1`] instead of skipping
the bits, and the per-subset byte length of §7.4.7.1
(`entry_point_offset_minus1[i] + 1`) is exposed via
[`EntryPointOffsets::subset_length`]. The on-wire
`num_entry_point_offsets` value is now range-checked against the
§7.4.7.1 upper bound for the active PPS partitioning
(`NumTileColumns * NumTileRows − 1` when `tiles_enabled_flag == 1`,
`PicHeightInCtbsY − 1` when `entropy_coding_sync_enabled_flag == 1`,
with the §7.4.3.3.1-forbidden `tiles + WPP` combination treated as
the wider of the two as a defensive cap); a breaching wire value
raises `SliceError::ValueOutOfRange { field:
"num_entry_point_offsets", got }`. Together this closes the last
parser-side "skip the bits, surface only the count" deferral inside
the §7.3.6.1 slice-segment-header walk; entry-point byte positions
are now first-class output and the downstream CABAC entry-point
seeker can index directly into the slice-data subset cells without
recovering the offsets from the wire a second time.

Round 21 wires the standalone §7.3.6.3 [`PredWeightTable::parse`]
(round 17) into the §7.3.6.1 slice-header in-place call site,
closing the last r20 deferral point for the universal base-profile
single-layer case.
When the §7.3.6.1 gate is statically present
(`(pps.weighted_pred_flag && slice_type == P)` or
`(pps.weighted_bipred_flag && slice_type == B)`) the parser now
constructs a [`PredWeightTableInputs::base_profile`] from the
post-override `num_ref_idx_lX_active_minus1`, `ChromaArrayType` (per
§7.4.2.2: `chroma_format_idc` unless `separate_colour_plane_flag == 1`),
and the SPS bit-depths, then decodes `pred_weight_table()` in place
and continues past the merge-candidate leaf and the shared I-slice
tail through `byte_alignment()` — so `opaque_tail` is `None` and
[`SliceSegmentHeader::byte_offset_to_slice_data`] reports where
`slice_segment_data()` begins for *every* P / B slice in the
crate's currently surfaced configuration (no SPS range / multilayer
/ SCC extensions). The base-profile constructor sets every per-i
§7.3.6.3 outer-gate decision to `true`, which is the universal
correct value for any single-layer slice: in a single-layer stream
every active reference is a different picture (an earlier-POC
temporal reference), so the
`pic_layer_id != nuh_layer_id || PicOrderCnt(RefPicListX[i]) !=
PicOrderCnt(CurrPic)` outer gate of §7.3.6.3 is always satisfied
and every `luma_weight_lX_flag[i]` / `chroma_weight_lX_flag[i]` is
signalled. The per-i gate slots stay open in
[`PredWeightTableInputs`] for the eventual SCC self-reference / IL
ref-layer cases, which will be threaded through here once the SPS
multilayer / SCC extensions surface (currently surfaced as opaque
tails — see "Not yet implemented"). The decoded table is exposed as
[`SliceSegmentHeader::pred_weight_table`] (an `Option<PredWeightTable>`
that is `None` for I slices, for dependent slice segments, when the
gate is statically absent, and for any prior-deferral path).
§7.4.7.3 range failures (e.g. `delta_luma_weight_lX > 127`) raised
inside the in-place parse propagate directly out of
`SliceSegmentHeader::parse` as `SliceError::ValueOutOfRange`. With
this round, every weighted-pred-gated independent slice segment
parses through `byte_alignment()`; only the
`pps.lists_modification_present_flag == 1` path still defers at the
§7.3.6.2 gate (it needs the §7.4.7.2 `NumPicTotalCurr` derivation
threaded through the slice parser — this is the next round's
target). The eight pre-round tests that exercised the post-override
walk are rewritten to consume their PWT bodies in place and assert
`opaque_tail.is_none()`; three new tests cover the universal
base-profile P-slice walk with a non-trivial luma weight (derived
`LumaWeightL0[0] = 9`), the B-slice L1 chroma sub-block + equation
7-58 `ChromaOffsetL1[0][j]` derivation, and the
`delta_luma_weight_l0` range-failure propagation. Total tests now
201 (was 198). Round 20 had landed the
§7.3.6.1 in-place inter-slice `five_minus_max_num_merge_cand` `ue(v)`
leaf (one syntax element after r19's collocated block), with the
§7.4.7.1 equation 7-53 derivation `MaxNumMergeCand = 5 -
five_minus_max_num_merge_cand` exposed as
[`SliceSegmentHeader::max_num_merge_cand`] and the wire value
range-checked at 0..=4. The merge-candidate field is signalled only
when the §7.3.6.3 `pred_weight_table()` gate is statically absent —
i.e. when neither `(pps.weighted_pred_flag && slice_type == P)` nor
`(pps.weighted_bipred_flag && slice_type == B)` holds; with r21 in
place the gate-is-present branch now decodes the table inline
rather than deferring, so the full inter-slice header parses end to
end through `byte_alignment()` in every configuration this crate
surfaces today. Round 19 had landed the
§7.3.6.1 in-place inter-slice
`mvd_l1_zero_flag` / `cabac_init_flag` /
`collocated_from_l0_flag` / `collocated_ref_idx` block (the four
syntax elements that immediately follow `num_ref_idx_lX_active_minus1`
and the `ref_pic_lists_modification()` gate). When
`pps.lists_modification_present_flag == 0` the outer
`if(... && NumPicTotalCurr > 1)` short-circuit makes
`ref_pic_lists_modification()` statically absent, so the parser walks
the next four fields in place without needing the §7.4.7.2
`NumPicTotalCurr` derivation. `mvd_l1_zero_flag` is signalled only
for B slices; `cabac_init_flag` is signalled when
`pps.cabac_init_present_flag == 1` and otherwise inferred to `false`
per §7.4.7.1; `collocated_from_l0_flag` is signalled for B + `mvp`
and otherwise inferred to `true` for P + `mvp`; `collocated_ref_idx`
is signalled when the active list selected by `collocated_from_l0_flag`
has more than one entry (range-checked against the active
`num_ref_idx_lX_active_minus1`), otherwise inferred to `0`. Four new
`Option`-typed accessors —
[`SliceSegmentHeader::mvd_l1_zero_flag`],
[`SliceSegmentHeader::cabac_init_flag`],
[`SliceSegmentHeader::collocated_from_l0_flag`] and
[`SliceSegmentHeader::collocated_ref_idx`] — surface the values.
The deferred P/B opaque tail now begins at the weighted-prediction
table gate (`pred_weight_table()`); when
`pps.lists_modification_present_flag == 1` the parser still defers at
the `ref_pic_lists_modification()` gate because its
`NumPicTotalCurr` derivation needs the §7.4.8 inter-RPS-prediction
step that this round does not wire in. The remaining inter-slice tail
(`ref_pic_lists_modification()` wiring under `NumPicTotalCurr`,
`pred_weight_table()` wiring, `five_minus_max_num_merge_cand`,
`use_integer_mv_flag`, the QP-offset / deblocking / loop-filter tail)
remains the next round's target — the standalone parsers
[`RefPicListsModification::parse`] / [`PredWeightTable::parse`] /
[`NumPicTotalCurrInputs::compute`] are already callable.
Round 18 had landed the
§7.3.6.1 in-place inter-slice prelude — the
`num_ref_idx_active_override_flag` `u(1)` and the
`num_ref_idx_l0_active_minus1` / (B-only) `num_ref_idx_l1_active_minus1`
`ue(v)` block that follows the SAO gates for P / B slices, including
the §7.4.7.1 inference rule (when the override flag is 0 the per-list
values come from
[`PicParameterSet::num_ref_idx_l0_default_active_minus1`] /
[`PicParameterSet::num_ref_idx_l1_default_active_minus1`]) and the
0..=14 range check on the explicit values.
Round 17 had landed the
§7.3.6.3 `pred_weight_table()` syntax structure as a standalone parser
([`PredWeightTable::parse`]). Round 16 lands the
§7.4.7.2 `NumPicTotalCurr` derivation (equation 7-57), the explicit
follow-up to round 15's standalone `RefPicListsModification` parser.
The new [`NumPicTotalCurrInputs`] builder takes the per-position
`UsedByCurrPicS0` / `UsedByCurrPicS1` / `UsedByCurrPicLt` flags from
the active short-term RPS plus the slice's long-term ref list, plus
the `pps_curr_pic_ref_enabled_flag` closing clause, and returns the
typed `NumPicTotalCurr: u32`. A
[`NumPicTotalCurrInputs::from_explicit_short_term_rps`] convenience
constructor pulls the `S0` / `S1` slices straight off a
[`ShortTermRefPicSet`] in explicit form (the inter-RPS-predicted
form requires the §7.4.8 derivation to run first; the builder
returns `None` and the caller threads through
[`NumPicTotalCurrInputs::from_used_flags`]). A
[`SliceLongTermRefPic::used_by_curr_pic_lt`] helper resolves each
`UsedByCurrPicLt[i]` per §7.4.7.1 — SPS-table lookup for `Sps
{ lt_idx_sps }` entries, direct flag for `InSlice` entries. The
F.7.4.7.2 multilayer-extension form (equation F-56) is wired through
a `with_multilayer_extension(nal_unit_type, num_active_ref_layer_pics)`
builder method, so when the multilayer profile becomes active the
IDR short-/long-term-loop skip plus the `NumActiveRefLayerPics`
summand land without further surface change. The §7.3.6.1 in-place
wiring of round 15's standalone `RefPicListsModification` parser
remains the next round's target — the derivation itself is no
longer the blocker.
Round 15 had landed the
§7.3.6.2 `ref_pic_lists_modification()` syntax structure as a
standalone parser ([`RefPicListsModification::parse`]): the
`ref_pic_list_modification_flag_l0` `u(1)` gate, the
`list_entry_l0[ 0 .. num_ref_idx_l0_active_minus1 ]` `u(v)` loop with
the per-entry width set to `Ceil( Log2( NumPicTotalCurr ) )` bits
(§7.4.7.2) and each value range-checked to `0 ..=
NumPicTotalCurr - 1`, the B-slice-gated `ref_pic_list_modification_flag_l1`
`u(1)` plus the matching `list_entry_l1[]` loop, and the up-front
preconditions that reject `SliceType::I` calls (the §7.3.6.1 gate
sits inside the inter-slice branch), `NumPicTotalCurr <= 1` (the
§7.3.6.1 gate guarantees `> 1` at the call site), and
`num_ref_idx_lX_active_minus1 > 14` (the §7.4.7.1 cap on
`num_ref_idx_lX_active_minus1`). The implicit `RefPicListTempX`
derivation of §8.3.4 stays the consumer's responsibility; the parser
materialises only the on-wire syntax elements.
[`slice::SliceSegmentHeader::parse`] still surfaces the inter-slice
tail as an [`sps::OpaqueTail`] — the in-place call site is now
unblocked but the §7.3.6.1 inter-slice body parse (pred-weight-table
+ five-or-six-more-flags + slice-data offset) is itself a separate
round's worth of work.
Round 14 had landed the
§E.2.1 `vui_parameters()` body as a typed [`VuiParameters`] (see
"Scope so far" below). Round 13 had landed the
§E.2.2 / §E.2.3 `hrd_parameters()` and `sub_layer_hrd_parameters()`
bodies as a new `hrd` module ([`HrdParameters`] / [`HrdCommonInfo`] /
[`SubLayerHrd`] / [`SubLayerHrdParameters`] / [`CpbEntry`] /
[`VpsHrdEntry`]). The decoder walks the common-info gates
(`nal_hrd_parameters_present_flag` /
`vcl_hrd_parameters_present_flag` /
`sub_pic_hrd_params_present_flag` and the conditional
`tick_divisor_minus2` / `du_cpb_removal_delay_increment_length_minus1`
/ `sub_pic_cpb_params_in_pic_timing_sei_flag` /
`dpb_output_delay_du_length_minus1` / `bit_rate_scale` /
`cpb_size_scale` / `cpb_size_du_scale` /
`initial_cpb_removal_delay_length_minus1` /
`au_cpb_removal_delay_length_minus1` /
`dpb_output_delay_length_minus1` block); the per-sub-layer loop
(`fixed_pic_rate_general_flag[i]` /
`fixed_pic_rate_within_cvs_flag[i]` with the §E.3.2 "general == 1 ⇒
within_cvs := 1" inference, `elemental_duration_in_tc_minus1[i]`
`ue(v)` range-checked at 0..=2047, `low_delay_hrd_flag[i]`, and
`cpb_cnt_minus1[i]` `ue(v)` range-checked at 0..=31); and the §E.2.3
sub-layer body with the §E.3.3 monotonicity constraints
(`bit_rate_value_minus1[i]` strictly increasing,
`cpb_size_value_minus1[i]` non-increasing, sub-pic variants gated on
`sub_pic_hrd_params_present_flag`). The VPS RBSP now decodes the
§7.3.2.1 per-HRD loop inline ([`HevcVps::hrd_parameters`]) with the
`cprms_present_flag[i] == 0` inheritance walked through the prior
entry's [`HrdCommonInfo`]; [`HevcVps::vps_extension_flag`] is now an
unconditional `bool`, and [`HevcVps::opaque_tail`] is populated only
for the extension-data + `rbsp_trailing_bits()` suffix.
Round 12 had finished the
§7.3.2.1 VPS tail through the optional VPS timing-info block: the
`vps_max_layer_id` (`u(6)`) and `vps_num_layer_sets_minus1` (`ue(v)`,
range 0..=1023, capped at `HEVC_VPS_MAX_NUM_LAYER_SETS = 1024`)
fields, the
`layer_id_included_flag[1..=num_layer_sets_minus1][0..=max_layer_id]`
inclusion matrix (one row per signalled layer set; layer set 0 is
implicit per §7.4.3.1), and the `vps_timing_info_present_flag` block
([`VpsTimingInfo`] — `vps_num_units_in_tick` / `vps_time_scale` both
`u(32)` with the §E.2.1 / §7.3.2.1 "shall be > 0" semantics enforced,
the `vps_poc_proportional_to_timing_flag` gate with the optional
`vps_num_ticks_poc_diff_one_minus1` `ue(v)`, and the
`vps_num_hrd_parameters` `ue(v)` count bounded at
`vps_num_layer_sets_minus1 + 1`).
Round 11 landed the
§9.3 CABAC arithmetic decoding engine as a standalone module
([`cabac`]): the §9.3.2.6 engine-register initialization
(`ivlCurrRange = 510`, `ivlOffset = read_bits(9)`, with the spec's
"`ivlOffset` shall not equal 510 or 511" constraint enforced); the
§9.3.2.2 context-variable initialization (equations 9-4..9-6, with
[`ContextModel::init`] taking an 8-bit `initValue` and `SliceQpY`)
plus the §9.3.2.2 `initType` selector (equation 9-7,
[`init_type`]); the §9.3.4.3.2 `DecodeDecision` primitive with the
Table 9-52 `rangeTabLps[64][4]` interval split and the §9.3.4.3.2.2
Table 9-53 (`transIdxLps` / `transIdxMps`) state transition; the
§9.3.4.3.3 renormalization loop (`RenormD`); the §9.3.4.3.4
`DecodeBypass` equal-probability bin plus a `decode_bypass_bits(n)`
MSB-first helper; the §9.3.4.3.5 `DecodeTerminate` decision-before-
termination (`end_of_slice_segment_flag` / `end_of_subset_one_bit` /
`pcm_flag`); and the §9.3.4.3.6 aligned-bypass alignment hook
(`ivlCurrRange = 256` before `coeff_abs_level_remaining[ ]` /
`coeff_sign_flag[ ]`). The engine is the gateway to slice-data
decode — every §7.3.8 syntax element is read through these four
primitives — and ships independently of the §9.3.4.2 per-syntax-
element binarization / context-index derivation, which sits one
layer up in the slice-data parser (still blocked on the docs
trace gap for `cu_qp_delta` + `last_sig_coeff` multi-QG / 4:2:2).
Round 10 completed the
§6.5 scan-order family — the §6.5.4 horizontal (equation 6-12), §6.5.5
vertical (equation 6-13), and §6.5.6 traverse (equation 6-14,
boustrophedon) scans now join round 9's §6.5.3 up-right diagonal — and
adds the §7.4.2 `ScanOrder[log2BlockSize][scanIdx]` accessor
(`scan_order`) with its populated-range checks (`log2BlockSize` 0..=3
for diagonal / horizontal / vertical, 2..=5 for traverse), the form the
residual-coding path (§7.3.8.11 / §9.3.4.2.4) selects per block. Round 9
landed the
§6.5.3 up-right diagonal scan order (equation 6-11) and the §7.4.5
`ScalingFactor[sizeId][matrixId][x][y]` 2-D quantization-matrix
derivation (equations 7-44..7-51), building on round 8's §7.3.4
`scaling_list_data()` parse + flat `ScalingList[sizeId][matrixId][i]`
lists. `ScalingListData::scaling_factors(ChromaArrayType)` scatters
each flat coefficient to the `(x, y)` cell the up-right diagonal scan
maps it to (`ScanOrder[2][0]` for the 4x4 list, `ScanOrder[3][0]` for
every 8x8-based list), replicates each entry into a 2x2 (16x16) or 4x4
(32x32) block, applies the DC-coefficient `[0][0]` override
(equations 7-47 / 7-49 / 7-51), and — only when `ChromaArrayType == 3`
— derives the 32x32 chroma matrices (matrixId 1, 2, 4, 5) from the
16x16 lists (equations 7-50 / 7-51). Round 8 had landed the §7.3.4
parse plus the flat-list derivation (default 4x4 / 8x8 Tables 7-5 /
7-6, the `scaling_list_pred_matrix_id_delta` reference-list copy of
equations 7-42 / 7-43, the explicit delta-coded form's `nextCoef`
accumulator, and the §7.4.5 range checks), wired into both the SPS
(`sps_scaling_list_data_present_flag`) and PPS
(`pps_scaling_list_data_present_flag`) paths. It builds on round 7's
non-IDR POC + reference-picture-set slice sub-block, round 6's
slice-header parse, round 5's §7.3.2.3.1 PPS parse, round 4's complete
SPS RBSP body — PCM block, short-term reference picture sets (§7.3.7),
the long-term reference picture table, the
`sps_temporal_mvp_enabled_flag` / `strong_intra_smoothing_enabled_flag`
pair, and the VUI / extension gates — round 3's structural prefix,
round 2's VPS / profile-tier-level (§7.3.2.1 + §7.3.3), and round 1's
Annex B / NAL-header foundation. All independent I-slice segments (IDR
and non-IDR) parse end to end through `byte_alignment()`; only the P/B
reference-list / weighted-prediction sub-structures (§7.3.6.2 /
§7.3.6.3, both DPB-derived) remain surfaced as an opaque tail. Slice
data and CABAC remain unimplemented.

## Scope so far

* Annex B byte-stream splitting — 3-byte (`00 00 01`) and 4-byte
  (`00 00 00 01`) start codes; multiple NAL units per buffer.
* §7.3.1.2 NAL header parse — `forbidden_zero_bit`,
  `nal_unit_type`, `nuh_layer_id`, and `TemporalId` (derived from
  `nuh_temporal_id_plus1`).
* §7.4.1.1 emulation-prevention strip — every `0x00 0x00 0x03` in
  the on-wire payload is decoded back to `0x00 0x00`.
* MSB-first bit reader with `u(n)` and 0-th-order
  unsigned-Exp-Golomb `ue(v)` (§9.2) descriptors.
* §7.3.2.1 [`HevcVps`] — `vps_video_parameter_set_id`,
  base-layer / max-layers / sub-layers / temporal-nesting flags,
  `vps_reserved_0xffff_16bits` validation, the §7.3.3
  profile-tier-level walk (general profile + level + per-sub-layer
  present-flag gates + `sub_layer_level_idc`), the per-sub-layer
  DPB / reorder / latency `ue(v)` triple loop with
  ordering-info-present-flag propagation, `vps_max_layer_id` /
  `vps_num_layer_sets_minus1` + the
  `layer_id_included_flag[i][j]` inclusion matrix (one
  [`LayerIdInclusionRow`] per signalled layer set), the
  `vps_timing_info_present_flag` block ([`VpsTimingInfo`] — the
  `u(32)` `vps_num_units_in_tick` / `vps_time_scale` pair with the
  spec's "shall be > 0" semantics enforced, the
  `vps_poc_proportional_to_timing_flag` gate +
  `vps_num_ticks_poc_diff_one_minus1`, and the
  `vps_num_hrd_parameters` count), the per-HRD loop
  ([`HevcVps::hrd_parameters`] — one [`VpsHrdEntry`] per
  `vps_num_hrd_parameters` with the §7.3.2.1 `hrd_layer_set_idx[i]` +
  `cprms_present_flag[i]` prelude, decoding each
  `hrd_parameters(cprms, vps_max_sub_layers_minus1)` body via the
  shared `hrd` module), and the `vps_extension_flag` gate. When
  `vps_extension_flag == 1` the extension-data run +
  `rbsp_trailing_bits()` are surfaced as an [`OpaqueTail`]
  (`HevcVps::opaque_tail`).
* §E.2.2 / §E.2.3 [`HrdParameters`] / [`SubLayerHrdParameters`] —
  the common-info block ([`HrdCommonInfo`]) with the §E.2.2
  `nal_hrd_parameters_present_flag` / `vcl_hrd_parameters_present_flag`
  / `sub_pic_hrd_params_present_flag` gates and the conditional `u(8)`
  / `u(5)` / `u(4)` length / scale fields; the per-sub-layer loop
  ([`SubLayerHrd`] with the §E.3.2 inference rules for
  `fixed_pic_rate_within_cvs_flag` /  `low_delay_hrd_flag` /
  `cpb_cnt_minus1` plus the `elemental_duration_in_tc_minus1[i]`
  0..=2047 and `cpb_cnt_minus1[i]` 0..=31 range-checks); the §E.2.3
  per-CPB array ([`CpbEntry`]) with the §E.3.3 monotonicity
  constraints enforced inline; and the §7.3.2.1 `cprms_present_flag[i]
  == 0` inheritance of common-info gates from the previous entry.
* §E.2.1 [`VuiParameters`] — the full `vui_parameters()` body:
  aspect-ratio info (`aspect_ratio_idc` + the [`EXTENDED_SAR`]
  `sar_width` / `sar_height` `u(16)` pair), overscan, the
  [`VideoSignalType`] block (`video_format` / `video_full_range_flag`
  + the [`ColourDescription`] `colour_primaries` /
  `transfer_characteristics` / `matrix_coeffs` triple), chroma-loc
  info (`chroma_sample_loc_type_{top,bottom}_field` 0..=5
  range-checked), the neutral-chroma / field-seq / frame-field flags,
  the [`DefaultDisplayWindow`] offset quad, the [`VuiTimingInfo`]
  block (`u(32)` `vui_num_units_in_tick` / `vui_time_scale` enforced
  `> 0` per §E.3.1, POC-proportional flag +
  `vui_num_ticks_poc_diff_one_minus1`, and the nested §E.2.3
  `hrd_parameters( 1, sps_max_sub_layers_minus1 )` call reusing
  [`HrdParameters`]), and the [`BitstreamRestriction`] block (with
  the §E.3.1 `min_spatial_segmentation_idc` 0..=4095,
  `max_bytes_per_pic_denom` / `max_bits_per_min_cu_denom` 0..=16, and
  `log2_max_mv_length_{horizontal,vertical}` 0..=15 range-checks).
  Validated against the x265-produced tiny-fixture VUI (1:1 SAR, 25 fps
  timing).
* §7.3.2.2 [`SeqParameterSet`] — `sps_video_parameter_set_id`,
  `sps_max_sub_layers_minus1` / `sps_temporal_id_nesting_flag`, the
  §7.3.3 PTL re-walk, `sps_seq_parameter_set_id`, `chroma_format_idc`
  / `separate_colour_plane_flag`, `pic_width_in_luma_samples` /
  `pic_height_in_luma_samples`, `conformance_window_flag` + the four
  `conf_win_*_offset` `ue(v)` values, `bit_depth_luma_minus8` /
  `bit_depth_chroma_minus8`, `log2_max_pic_order_cnt_lsb_minus4`,
  the per-sub-layer DPB / reorder / latency triple loop,
  `log2_min_luma_coding_block_size_minus3`,
  `log2_diff_max_min_luma_coding_block_size`,
  `log2_min_luma_transform_block_size_minus2`,
  `log2_diff_max_min_luma_transform_block_size`,
  `max_transform_hierarchy_depth_{inter,intra}`,
  `scaling_list_enabled_flag`, `amp_enabled_flag`,
  `sample_adaptive_offset_enabled_flag`, the `pcm_*` block gated
  by `pcm_enabled_flag`, `num_short_term_ref_pic_sets` + the per-set
  [`ShortTermRefPicSet`] (§7.3.7 — both the explicit
  `num_negative_pics` / `num_positive_pics` form and the
  inter-RPS-prediction form, with `RefRpsIdx` chained through the
  preceding RPS list per §7.4.8), `long_term_ref_pics_present_flag`
  + the [`LongTermRefPicEntry`] table (`u(v)` POC-LSB width per
  `log2_max_pic_order_cnt_lsb_minus4 + 4`),
  `sps_temporal_mvp_enabled_flag`,
  `strong_intra_smoothing_enabled_flag`, the
  `vui_parameters_present_flag` gate whose §E.2.1 `vui_parameters()`
  body is decoded into [`VuiParameters`] (see below), and the
  `sps_extension_present_flag` gate (extension payload surfaced as
  an [`OpaqueTail`] — raw RBSP bytes from the cut-off byte through
  the buffer end, with the start-bit offset). When
  `scaling_list_enabled_flag == 1` and
  `sps_scaling_list_data_present_flag == 1`, the §7.3.4
  `scaling_list_data()` block is parsed into [`ScalingListData`]
  (otherwise the §7.4.5 default lists apply).
* §7.3.2.3.1 [`PicParameterSet`] — the full general
  `pic_parameter_set_rbsp()` body: `pps_pic_parameter_set_id` /
  `pps_seq_parameter_set_id`, the slice-header gates
  (`dependent_slice_segments_enabled_flag`, `output_flag_present_flag`,
  `num_extra_slice_header_bits`, `sign_data_hiding_enabled_flag`,
  `cabac_init_present_flag`), `num_ref_idx_l{0,1}_default_active_minus1`,
  `init_qp_minus26` (`se(v)`), `constrained_intra_pred_flag`,
  `transform_skip_enabled_flag`, `cu_qp_delta_enabled_flag` +
  `diff_cu_qp_delta_depth`, `pps_c{b,r}_qp_offset` (`se(v)`),
  `pps_slice_chroma_qp_offsets_present_flag`, `weighted_pred_flag` /
  `weighted_bipred_flag`, `transquant_bypass_enabled_flag`, the tiles
  block ([`TileInfo`] — `num_tile_{columns,rows}_minus1`,
  `uniform_spacing_flag`, and the explicit
  `column_width_minus1[]` / `row_height_minus1[]` arrays when spacing
  is non-uniform, plus `loop_filter_across_tiles_enabled_flag`),
  `entropy_coding_sync_enabled_flag`,
  `pps_loop_filter_across_slices_enabled_flag`, the
  deblocking-filter-control block ([`DeblockingFilterControl`]),
  `lists_modification_present_flag`,
  `log2_parallel_merge_level_minus2`,
  `slice_segment_header_extension_present_flag`, and the
  `pps_extension_present_flag` gate (extension bodies surfaced as a
  shared [`OpaqueTail`]). The §7.4.3.3.1 inference rules are applied so
  absent conditional fields carry their effective value, and when
  `pps_scaling_list_data_present_flag == 1` the §7.3.4
  `scaling_list_data()` block is parsed into [`ScalingListData`]. The
  signed-Exp-Golomb `se(v)` descriptor (§9.2.2) was added to
  [`BitReader`] for the PPS QP / deblocking-offset fields.
* §7.3.4 [`ScalingListData`] — the `scaling_list_data()` syntax
  structure plus the §7.4.5 `ScalingList[sizeId][matrixId][i]`
  derivation: per-slot `scaling_list_pred_mode_flag`, the
  `scaling_list_pred_matrix_id_delta` reference-list / default-list
  selection (equations 7-42 / 7-43), the explicit delta-coded form
  (running `nextCoef` accumulator modulo 256 with the
  `scaling_list_dc_coef_minus8` DC coefficient for `sizeId > 1`), and
  the default 4x4 / 8x8 tables (Tables 7-5 / 7-6). The §7.4.5 range
  checks are enforced ([`ScalingListError`]).
  [`ScalingListData::scaling_factors`] expands the flat lists into the
  two-dimensional `ScalingFactor[sizeId][matrixId][x][y]` quantization
  matrices (equations 7-44..7-51): each flat coefficient is scattered
  to the `(x, y)` cell given by the §6.5.3 up-right diagonal scan, with
  the 2x / 4x block replication for the 16x16 / 32x32 sizes, the
  DC-coefficient `[0][0]` override (equations 7-47 / 7-49 / 7-51), and
  the `ChromaArrayType == 3` 32x32-chroma derivation from the 16x16
  lists (equations 7-50 / 7-51).
* §6.5 [`scan`] — all four scan-order initialization processes plus
  the §7.4.2 [`scan_order`] `ScanOrder[log2BlockSize][scanIdx]`
  accessor: [`up_right_diagonal`] (§6.5.3, equation 6-11),
  [`horizontal`] (§6.5.4, equation 6-12), [`vertical`] (§6.5.5,
  equation 6-13), and [`traverse`] (§6.5.6, equation 6-14 — the
  boustrophedon raster: even rows left-to-right, odd rows
  right-to-left). [`ScanIdx`] is the §7.4.2 selector (0 diagonal /
  1 horizontal / 2 vertical / 3 traverse) and [`scan_order`] enforces
  the table's populated ranges — `log2BlockSize` 0..=3 for the
  diagonal / horizontal / vertical scans, 2..=5 for the traverse scan
  ([`ScanOrderError`]). The §7.4.5 `ScalingFactor` derivation reads
  `ScanOrder[2][0]` (4x4) and `ScanOrder[3][0]` (8x8); the residual
  coding path (§7.3.8.11 / §9.3.4.2.4) reads the full table.
* §7.3.6.1 [`SliceSegmentHeader`] — the `slice_segment_header()` parse
  for an independent slice segment, taking the activated SPS + PPS as
  context: `first_slice_segment_in_pic_flag`,
  `no_output_of_prior_pics_flag` (IRAP-range only),
  `slice_pic_parameter_set_id`; for non-first segments
  `dependent_slice_segment_flag` + `slice_segment_address` (`u(v)`,
  width `Ceil( Log2( PicSizeInCtbsY ) )`); for independent segments the
  `slice_reserved_flag[]` block, [`SliceType`] (Table 7-7),
  `pic_output_flag`, `colour_plane_id`, the non-IDR POC + RPS block
  (`slice_pic_order_cnt_lsb` `u(v)`,
  `short_term_ref_pic_set_sps_flag` `u(1)` with the §7.4.7.1
  `num_short_term_ref_pic_sets == 0` cross-check, the in-line
  `st_ref_pic_set(num_short_term_ref_pic_sets)` parse via
  [`ShortTermRefPicSet::parse_slice_inline`],
  `short_term_ref_pic_set_idx` `u(v)`, and the long-term-ref-pic
  block — `num_long_term_sps` / `num_long_term_pics` / per-entry
  [`SliceLongTermRefPic`] with [`SliceLongTermRefPicSource`]
  discriminating SPS-indexed vs in-slice signalling and
  `delta_poc_msb_present_flag` / `delta_poc_msb_cycle_lt`),
  `slice_temporal_mvp_enabled_flag`, and the SAO luma / chroma gates.
  For P / B slices the SAO block is followed in-place by the
  `num_ref_idx_active_override_flag` `u(1)` and the
  `num_ref_idx_l0_active_minus1` (P / B) /
  `num_ref_idx_l1_active_minus1` (B only) `ue(v)` block (§7.3.6.1),
  with the §7.4.7.1 inference rule filling both values from the PPS
  defaults when the override flag is 0; values are range-checked at
  0..=14. When `pps.lists_modification_present_flag == 0` the parser
  continues in-place into the §7.3.6.1 mvd / cabac-init / collocated
  block: `mvd_l1_zero_flag` (B only), `cabac_init_flag` (signalled
  iff `pps.cabac_init_present_flag == 1`, else inferred `false`),
  `collocated_from_l0_flag` (signalled iff `mvp && slice_type == B`,
  else inferred `true`), and `collocated_ref_idx` (signalled iff
  `mvp` and the active list has > 1 entry, else inferred `0`,
  range-checked against `num_ref_idx_lX_active_minus1`). The
  §7.3.6.3 `pred_weight_table()` is then decoded **in place** when
  its outer gate is statically present
  (`pps.weighted_pred_flag && slice_type == P` or
  `pps.weighted_bipred_flag && slice_type == B`), via
  [`PredWeightTable::parse`] with
  [`PredWeightTableInputs::base_profile`] driven by the
  post-override `num_ref_idx_lX_active_minus1`, the SPS-derived
  `ChromaArrayType`, and the SPS bit-depths; the result is exposed
  on [`SliceSegmentHeader::pred_weight_table`]. When the gate is
  statically absent the table is skipped (the field stays `None`).
  Either way the parser walks past the (possibly-decoded) table
  into `five_minus_max_num_merge_cand` `ue(v)` (range 0..=4, with
  §7.4.7.1 equation 7-53 yielding `MaxNumMergeCand =
  5 - five_minus_max_num_merge_cand` accessible via
  [`SliceSegmentHeader::max_num_merge_cand`]), past the SCC
  `use_integer_mv_flag` (statically absent because the PPS SCC
  extension is not surfaced yet — §7.4.7.1 infers
  `motion_vector_resolution_control_idc` to 0), and through the
  shared I-slice tail (slice_qp_delta + chroma QP offsets +
  deblocking override + loop-filter-across-slices + entry-points +
  slice-header extension + `byte_alignment()`); `opaque_tail` is
  `None` and [`SliceSegmentHeader::byte_offset_to_slice_data`]
  reports where `slice_segment_data()` begins. The remaining
  inter-slice work (`ref_pic_lists_modification()` in-place wiring
  under `NumPicTotalCurr`) stays the next round's target — the gate
  is statically present only when `pps.lists_modification_present_flag
  == 1`, at which point the parser still defers at the §7.3.6.2
  block.
  Independent I-slice segments — IDR and non-IDR alike — parse end to
  end through `byte_alignment()`: `slice_qp_delta` (`se(v)`), the
  chroma QP offsets, the deblocking override block
  ([`SliceDeblocking`]), `slice_loop_filter_across_slices_enabled_flag`,
  the entry-point-offset block ([`EntryPointOffsets`]) — including the
  per-i `entry_point_offset_minus1[i]` `u(offset_len_minus1 + 1)`
  values captured into `Vec<u32>` and the per-subset byte length
  exposed via [`EntryPointOffsets::subset_length`], with
  `num_entry_point_offsets` range-checked against the §7.4.7.1
  partitioning bound — and the
  header-extension block;
  [`SliceSegmentHeader::byte_offset_to_slice_data`] reports where
  `slice_segment_data()` begins, and
  [`SliceSegmentHeader::slice_qp_y`] applies equation 7-54. The
  §7.4.7.1 inference rules are applied to absent fields.

* §7.3.6.2 [`RefPicListsModification`] — the
  `ref_pic_lists_modification()` syntax structure as a standalone
  parser, callable by a future round once the §7.4.7.2
  `NumPicTotalCurr` derivation is wired through the slice parser.
  [`RefPicListsModification::parse`] takes the active
  `slice_type` / `num_ref_idx_l0_active_minus1` /
  `num_ref_idx_l1_active_minus1` / `NumPicTotalCurr` and walks
  the `ref_pic_list_modification_flag_l0` gate + the
  `list_entry_l0[0..=num_ref_idx_l0_active_minus1]` `u(v)` loop
  (each entry `Ceil( Log2( NumPicTotalCurr ) )` bits wide and
  range-checked at `<= NumPicTotalCurr - 1` per §7.4.7.2), then —
  for B slices only — the `ref_pic_list_modification_flag_l1`
  gate + its `list_entry_l1[]` loop. The parser rejects
  preconditions that the §7.3.6.1 call site would have filtered
  (`SliceType::I` and `NumPicTotalCurr <= 1`) and the
  §7.4.7.1 `num_ref_idx_lX_active_minus1 > 14` cap. The implicit
  `RefPicListTempX` derivation of §8.3.4 stays the consumer's
  responsibility; this struct surfaces only the on-wire syntax.

* §7.4.7.2 [`NumPicTotalCurrInputs`] — the `NumPicTotalCurr`
  derivation (equation 7-57) as a small typed builder. The caller
  supplies the per-position `UsedByCurrPicS0` / `UsedByCurrPicS1` /
  `UsedByCurrPicLt` flags from the active short-term RPS plus the
  slice's long-term ref list and the
  `pps_curr_pic_ref_enabled_flag` (inferred to `false` until the SCC
  PPS extension is materialised); [`NumPicTotalCurrInputs::compute`]
  returns the typed `NumPicTotalCurr: u32`.
  [`NumPicTotalCurrInputs::from_explicit_short_term_rps`] pulls the
  `S0` / `S1` slices directly off a [`ShortTermRefPicSet`] in
  explicit form (the inter-RPS-prediction form returns `None`; the
  §7.4.8 derivation must be run first and the result threaded
  through [`NumPicTotalCurrInputs::from_used_flags`]). The
  [`SliceLongTermRefPic::used_by_curr_pic_lt`] helper resolves each
  long-term entry's `UsedByCurrPicLt[i]` per §7.4.7.1: SPS-table
  lookup for `Sps { lt_idx_sps }` entries, direct flag for in-slice
  entries. The F.7.4.7.2 multilayer-extension form (equation F-56)
  is reachable through
  [`NumPicTotalCurrInputs::with_multilayer_extension`] —
  IDR-`nal_unit_type` slices skip the short-term / long-term loops
  and the `NumActiveRefLayerPics` summand is added at the end.

* §7.4.8 [`MaterializedShortTermRefPicSet`] /
  [`ShortTermRefPicSet::materialize`] — the post-derivation form of a
  short-term RPS, exposing the per-position `DeltaPocS0[]`,
  `UsedByCurrPicS0[]`, `DeltaPocS1[]`, `UsedByCurrPicS1[]` arrays as
  signed POC deltas; `NumNegativePics` / `NumPositivePics` /
  `NumDeltaPocs` are exposed as the array lengths
  (equation 7-71). The explicit-form branch implements equations
  7-63..7-70 directly (the per-position `UsedByCurrPicS{0,1}` arrays
  pass through, the cumulative `DeltaPocS0[i] = DeltaPocS0[i-1] -
  (delta_poc_s0_minus1[i] + 1)` / `DeltaPocS1[i] = DeltaPocS1[i-1] +
  (delta_poc_s1_minus1[i] + 1)` recurrences with their
  equation-7-67 / 7-68 first-element seeds). The
  inter-RPS-prediction branch implements equation 7-60 (`deltaRps =
  (1 - 2*delta_rps_sign) * (abs_delta_rps_minus1 + 1)`) and equations
  7-61 / 7-62 — the negative side iterates source positives in
  reverse, then optionally the `deltaRps` self-term, then source
  negatives in forward order; the positive side mirrors the same
  pattern with sources swapped. Each candidate entry is gated by its
  sign test (`dPoc < 0` for the negative side, `dPoc > 0` for the
  positive side) and the per-entry `use_delta_flag[j]` flag. Per-array
  length is checked against the source's
  `NumDeltaPocs[RefRpsIdx] + 1`; mismatches raise
  [`ShortTermRefPicSetMaterializeError::SourceLengthMismatch`].
  [`SeqParameterSet::materialize_short_term_ref_pic_sets`] runs the
  full SPS-level chain, threading inter-form entries through the
  equation-7-59 `RefRpsIdx = stRpsIdx - (delta_idx_minus1 + 1)` source
  lookup. The slice parser uses both: the SPS list is materialised
  once at the §7.3.6.1 `ref_pic_lists_modification()` gate, the
  active RPS is picked (or constructed from the slice-inline form),
  and the derived `UsedByCurrPicS{0,1}` slices feed
  [`NumPicTotalCurrInputs::from_used_flags`] / `compute` so the
  in-place RPLM gate is reached for every conformant slice — explicit
  or inter-form.

* §7.3.6.3 [`PredWeightTable`] — the `pred_weight_table()` syntax
  structure, wired in-place at the §7.3.6.1 call site (round 21).
  When the §7.3.6.1 outer gate is statically present
  (`weighted_pred_flag && slice_type == P` or
  `weighted_bipred_flag && slice_type == B`), the slice parser
  constructs a [`PredWeightTableInputs::base_profile`] from the
  post-override `num_ref_idx_lX_active_minus1`, the SPS-derived
  `ChromaArrayType`, and the SPS bit-depths, then decodes the table
  in place and continues through the inter-slice tail.
  [`SliceSegmentHeader::pred_weight_table`] surfaces the parsed
  table (`None` when the gate is statically absent, for I slices,
  for dependent slice segments, and for prior-deferral paths).
  [`PredWeightTable::parse`] reads `luma_log2_weight_denom`
  (range 0..=7) and, when `ChromaArrayType != 0`,
  `delta_chroma_log2_weight_denom` with the derived
  `ChromaLog2WeightDenom ∈ 0..=7` range check, then performs the two
  flag passes (`luma_weight_lX_flag[i]` and, when chroma is present,
  `chroma_weight_lX_flag[i]`) and the per-reference delta block
  (`delta_luma_weight_lX[i]` / `luma_offset_lX[i]` plus the chroma
  `delta_chroma_weight_lX[i][j]` / `delta_chroma_offset_lX[i][j]`
  pairs), enforcing every §7.4.7.3 range bound including the
  bit-depth-dependent luma/chroma offset bounds parameterised by
  [`PredWeightTableInputs::high_precision_offsets_enabled_flag`] and
  the SPS bit depths. For B slices the L1 block is mirrored after
  L0. The §7.4.7.3 conformance `sumWeightLXFlags ≤ 24` cap is
  enforced. The §7.3.6.3 outer-gate (`pic_layer_id != nuh_layer_id ||
  PicOrderCnt(RefPicListX[i]) != PicOrderCnt(CurrPic)`) per-i
  decision is supplied by the caller through
  [`PredWeightTableInputs::signal_luma_l0`] /
  [`PredWeightTableInputs::signal_chroma_l0`] (mirrored for L1); the
  [`PredWeightTableInputs::base_profile`] constructor leaves all four
  `None` (every position gated `true`), the universal base-profile
  single-layer case. Accessor methods
  [`PredWeightTable::luma_weight_l0`] (mirrored for L1),
  [`PredWeightTable::chroma_weight_l0`] (mirrored) and
  [`PredWeightTable::chroma_offset_l0`] (mirrored, equation 7-58)
  resolve the §7.4.7.3 derived variables `LumaWeightLX[i]`,
  `ChromaWeightLX[i][j]` and `ChromaOffsetLX[i][j]`.

* §9.3 [`cabac`] — the CABAC arithmetic decoding engine as a
  standalone module: [`CabacEngine::new`] initializes the §9.3.2.6
  registers (`ivlCurrRange = 510`, `ivlOffset = read_bits(9)`, with
  the spec's "ivlOffset shall not equal 510 or 511" constraint
  enforced) over a positioned [`BitReader`]; [`ContextModel::init`]
  derives a `(pStateIdx, valMps)` pair from an 8-bit `initValue` and
  `SliceQpY` per equations 9-4..9-6, with the §9.3.2.2 `initType`
  selector (equation 9-7) exposed as the free function [`init_type`];
  [`CabacEngine::decode_decision`] implements §9.3.4.3.2 with the
  Table 9-52 `rangeTabLps[64][4]` quantized-range split, the
  §9.3.4.3.2.2 Table 9-53 state transition, and the §9.3.4.3.3
  `RenormD` loop; [`CabacEngine::decode_bypass`] /
  [`CabacEngine::decode_bypass_bits`] implement §9.3.4.3.4 (the
  helper accumulates `n` bins MSB-first);
  [`CabacEngine::decode_terminate`] implements §9.3.4.3.5
  (`end_of_slice_segment_flag` / `end_of_subset_one_bit` / `pcm_flag`,
  the `ctxTable == 0`, `ctxIdx == 0` decision before termination); and
  [`CabacEngine::align`] implements §9.3.4.3.6 (the `ivlCurrRange =
  256` alignment hook prior to aligned bypass decoding for
  `coeff_abs_level_remaining[ ]` / `coeff_sign_flag[ ]`). The
  Table 9-52 / Table 9-53 values are transcribed directly from the
  H.265 specification.

* §9.3.4.2 [`binarization`] — per-syntax-element binarization +
  context-index derivation, the layer that drives
  [`cabac::CabacEngine`] with concrete `(ctxTable, ctxIdx)` selections
  for each entropy-coded syntax element. As of round 27 the module
  ships two CABAC-engine-coupled elements (rounds 26) plus three
  pure-functional ctxInc derivations (round 27):
  - **(round 27)** [`binarization::coded_sub_block_flag_ctx_inc`] +
    [`binarization::coded_sub_block_flag_ctx_inc_with_edge`] handle
    `coded_sub_block_flag` ctxInc derivation per §9.3.4.2.4 equations
    9-35..9-39: `csbfCtx` is the sum of the two previously decoded
    neighbour bins (gated by equations 9-36 / 9-37 against the TB's
    sub-block edges `xS < (1 << (log2TrafoSize − 2)) − 1` /
    `yS < (1 << (log2TrafoSize − 2)) − 1`), then `ctxInc =
    Min(csbfCtx, 1)` for luma (bank `{0, 1}`) or `2 + Min(csbfCtx, 1)`
    for chroma (bank `{2, 3}`).
  - **(round 27)** [`binarization::left_above_ctx_inc`] +
    [`binarization::split_cu_flag_ctx_inc`] +
    [`binarization::cu_skip_flag_ctx_inc`] handle the
    §9.3.4.2.2 Table 9-49 row shape `ctxInc = (condL && availableL) +
    (condA && availableA)`: `split_cu_flag` reads each neighbour's
    `CtDepth[xNb][yNb] > cqtDepth` via
    [`binarization::split_cu_flag_cond`]; `cu_skip_flag` reads each
    neighbour's `cu_skip_flag[xNb][yNb]` via
    [`binarization::cu_skip_flag_cond`]. Both rows produce `ctxInc ∈
    {0, 1, 2}`.
  - [`binarization::decode_cu_qp_delta`] handles
    `cu_qp_delta_abs` / `cu_qp_delta_sign_flag` (§7.3.8.14, §7.4.9.14):
    §9.3.3.10 TR prefix (`cMax = 5`, `cRiceParam = 0`), §9.3.3.11
    EGk(k=0) suffix when the prefix is the all-ones escape, bypass-coded
    sign flag (§7.4.9.14 implies the flag is absent when `abs == 0`),
    and the §7.4.9.14 `CuQpDeltaVal = abs * (1 − 2 * sign_flag)`
    derivation. Per-bin ctxInc (Table 9-32) lives in
    [`binarization::cu_qp_delta_abs_ctx_inc`]: bin 0 → ctx 0, bins 1..=4
    → ctx 1.
  - [`binarization::decode_last_sig_coeff`] handles
    `last_sig_coeff_{x,y}_{prefix,suffix}` (§7.3.8.11, §7.4.9.11):
    §9.3.3.10 TR prefix with `cMax = (log2TrafoSize << 1) − 1`, a
    `nBits = (prefix >> 1) − 1` bypass-coded suffix when `prefix > 3`,
    and the §7.4.9.11 equations 7-74..7-77 position derivation
    ([`binarization::last_sig_coeff_position`] returns the
    pre-scanIdx-2-swap `LastSignificantCoeff{X,Y}`). The §9.3.4.2.3
    `(ctxOffset, ctxShift)` derivation is in
    [`binarization::last_sig_coeff_prefix_ctx_offset_shift`] (luma /
    chroma per Tables 9-26 / 9-27), and the per-bin
    `ctxInc = (binIdx >> ctxShift) + ctxOffset` in
    [`binarization::last_sig_coeff_prefix_ctx_inc`]. The X and Y
    prefix bins live in separate context banks; the
    [`binarization::LastSigCoeffBank`] tag exposes the routing
    decision.

Top-level entry points: [`NalIter`], [`collect_nal_units`],
[`NalHeader::parse`], [`strip_emulation_prevention`],
[`BitReader`], [`HevcVps::parse`], [`ProfileTierLevel::parse`],
[`SeqParameterSet::parse`],
[`SeqParameterSet::materialize_short_term_ref_pic_sets`],
[`PicParameterSet::parse`], [`SliceSegmentHeader::parse`],
[`RefPicListsModification::parse`], [`PredWeightTable::parse`],
[`ShortTermRefPicSet::materialize`], [`scan_order`],
[`CabacEngine::new`],
[`binarization::decode_cu_qp_delta`],
[`binarization::decode_last_sig_coeff`],
[`binarization::coded_sub_block_flag_ctx_inc`],
[`binarization::coded_sub_block_flag_ctx_inc_with_edge`],
[`binarization::split_cu_flag_ctx_inc`],
[`binarization::cu_skip_flag_ctx_inc`],
[`binarization::left_above_ctx_inc`].

## Not yet implemented

* SPS extension bodies (Range Extension, Multilayer Annex F,
  3D Annex I, SCC) — likewise surfaced as opaque bytes.
* PPS extension bodies (`pps_range_extension()`,
  `pps_multilayer_extension()`, `pps_3d_extension()`,
  `pps_scc_extension()`, and the `pps_extension_data_flag` while-loop
  gated by `pps_extension_4bits != 0`) — surfaced as opaque bytes
  starting at the first signalled body when
  `pps_extension_present_flag == 1`. The eight bits of typed
  extension flags themselves are decoded as of round 25
  ([`pps::PpsExtensionFlags`]); only the bodies still defer.
* VPS `vps_extension_data_flag` extension payload — the §F / §G / §H
  / §I multi-layer / 3D / SCC VPS-extension syntax; surfaced as
  [`HevcVps::opaque_tail`] when `vps_extension_flag == 1`. The §E.2.2
  `hrd_parameters()` bodies are now fully decoded (round 13).
* Slice header (§7.3.6.1) deferred body: the active short-term RPS
  is in inter-RPS-prediction form (`inter_ref_pic_set_prediction_flag
  == 1`) **and** the on-wire `used_by_curr_pic_flag` /
  `use_delta_flag` array lengths do not match the source's
  `NumDeltaPocs[RefRpsIdx] + 1`. This is the only remaining
  parser-side §7.3.6.1 deferral path: every conformant slice
  configuration parses through `byte_alignment()` as of round 24
  (round 23 wired the explicit-form RPLM; round 24 wires the §7.4.8
  inter-RPS-prediction derivation via
  [`ShortTermRefPicSet::materialize`] /
  [`SeqParameterSet::materialize_short_term_ref_pic_sets`] so the
  derived `UsedByCurrPicS{0,1}` arrays feed
  [`NumPicTotalCurrInputs::from_used_flags`] and the inter-form RPS
  path matches the explicit-form behaviour). The §7.3.6.3
  `pred_weight_table()` is now
  decoded **in place** at the §7.3.6.1 call site (round 21, this
  round) using [`PredWeightTableInputs::base_profile`] — the
  universal base-profile single-layer case where every per-i
  §7.3.6.3 outer gate (`pic_layer_id != nuh_layer_id ||
  PicOrderCnt(...) != PicOrderCnt(CurrPic)`) is `true` because every
  active reference in a single-layer slice is a different temporal
  picture. The
  `num_ref_idx_active_override_flag` / `num_ref_idx_lX_active_minus1`
  override block (round 18), the
  `mvd_l1_zero_flag` / `cabac_init_flag` / `collocated_from_l0_flag`
  / `collocated_ref_idx` block (round 19), and the
  `five_minus_max_num_merge_cand` leaf (round 20) are decoded in
  place; the SCC `use_integer_mv_flag` closing flag is
  statically absent until the PPS SCC extension is surfaced (§7.4.7.1
  infers `motion_vector_resolution_control_idc` to 0). The non-IDR
  POC / reference-picture-set block (which previously sat under this
  bullet) is fully decoded as of round 7. The QP-offset / deblocking
  / loop-filter / entry-points / extension / `byte_alignment()` tail
  is now shared between I and P/B slices.
* Slice data (§7.3.8) — the slice-data syntax-element walk that
  drives the CABAC engine. Needs the §9.3.4.2 per-syntax-element
  binarization / context-index derivation (which selects the
  `ctxTable` / `ctxIdx` for each bin) and the §7.3.8.1..§7.3.8.12
  parse loops.
* §9.3.4.2 per-syntax-element binarization / context-index
  derivation — the scaffold module ([`binarization`]) landed in
  round 26 with the first two CABAC-engine-coupled elements
  (`cu_qp_delta_abs` / `cu_qp_delta_sign_flag` and
  `last_sig_coeff_{x,y}_{prefix,suffix}`) unblocked by the docs CABAC
  trace
  (`docs/video/h265/fixtures/main-422-10bit/cabac-cu-qp-delta-last-sig-trace.md`).
  Round 27 adds three more pure-functional ctxInc derivations:
  `coded_sub_block_flag` (§9.3.4.2.4 equations 9-35..9-39), and the
  §9.3.4.2.2 Table 9-49 row shape for `split_cu_flag` and
  `cu_skip_flag` (`ctxInc = (condL && availableL) + (condA &&
  availableA)`). Still to land — every other §9.3.4.2 syntax element:
  `sig_coeff_flag` (§9.3.4.2.5), `coeff_abs_level_greater1_flag` /
  `_greater2_flag` / `coeff_abs_level_remaining` / `coeff_sign_flag`
  (§9.3.4.2.6 + bypass), prediction-mode / part-mode / merge / merge-idx
  / inter-pred-idc flags, motion-vector binarization (`mvd_lX[]` EGk +
  sign), `sao_*` elements, etc. (The §9.3 arithmetic decode engine
  itself — DecodeDecision / DecodeBypass / DecodeTerminate / RenormD /
  alignment — is implemented as of round 11; it sits one layer below
  the binarization tables.)
* Intra / inter prediction, transform, in-loop filters (deblock /
  SAO), DPB management.
* Encoder.

The runtime registration hook (`register`) is still a no-op.

## License

MIT — see [LICENSE](./LICENSE).
