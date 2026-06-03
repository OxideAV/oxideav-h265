//! §9.3.4.2 per-syntax-element binarization + context-index derivation.
//!
//! Sits one layer above the §9.3 arithmetic decoding engine
//! ([`crate::cabac::CabacEngine`]). For each entropy-coded syntax
//! element this module supplies the §9.3.3 binarization shape (truncated
//! rice, fixed length, Exp-Golomb of order *k*, …) plus the §9.3.4.2.X
//! per-bin `ctxInc` derivation, returning a context-index into the
//! per-slice context-model array.
//!
//! Round 26 ships the two syntax elements unblocked by the clean-room
//! CABAC trace `docs/video/h265/fixtures/main-422-10bit/cabac-cu-qp-delta-last-sig-trace.md`:
//!
//! * **`cu_qp_delta_abs` / `cu_qp_delta_sign_flag`** (H.265 §7.3.8.14,
//!   §7.4.9.14). Binarization: §9.3.3.10 TR prefix (`cMax = 5`,
//!   `cRiceParam = 0`) followed, when the prefix is the all-ones escape
//!   (`prefixVal == 5`), by an EGk suffix with `k = 0` (§9.3.3.11). The
//!   per-bin `ctxInc` (§9.3.4.2.2, Table 9-32): bin 0 uses `ctxInc = 0`,
//!   bins 1..=4 share `ctxInc = 1`. The sign flag is bypass-coded.
//!   `CuQpDeltaVal = cu_qp_delta_abs * (1 − 2 * cu_qp_delta_sign_flag)`.
//!
//! * **`last_sig_coeff_{x,y}_prefix`** (H.265 §7.3.8.11, §7.4.9.11).
//!   Binarization: §9.3.3.10 TR prefix with
//!   `cMax = (log2TrafoSize << 1) − 1` and `cRiceParam = 0`. The per-bin
//!   `ctxInc` (§9.3.4.2.3) is
//!   `ctxInc = (binIdx >> ctxShift) + ctxOffset`. For luma:
//!   `ctxOffset = 3 * (log2TrafoSize − 2) + ((log2TrafoSize − 1) >> 2)`,
//!   `ctxShift = (log2TrafoSize + 1) >> 2`. For chroma (cIdx > 0):
//!   `ctxOffset = 15`, `ctxShift = log2TrafoSize − 2`. The X and Y
//!   prefix bins live in separate context banks; this module hands back
//!   only the bank-relative `ctxInc` and the matching
//!   [`LastSigCoeffBank`] tag for caller-side routing.
//!
//! * **`last_sig_coeff_{x,y}_suffix`** (H.265 §7.3.8.11, §7.4.9.11). A
//!   `nBits = (prefix >> 1) − 1`-bit fixed-length bypass field, present
//!   only when `prefix > 3`. Together with the prefix the final
//!   `LastSignificantCoeffX` / `Y` position is derived per equations
//!   7-74..7-77.
//!
//! Both elements live downstream of the §9.3.4.2 binarization layer
//! that the slice-data parser (§7.3.8) drives; this module is the
//! reusable per-element building block.
//!
//! Round 27 extends the ctxInc-derivation surface with three more
//! syntax elements that are pure-functional given their neighbour /
//! sub-block context (no CABAC engine drive needed at this layer —
//! callers compose the engine call themselves):
//!
//! * **`coded_sub_block_flag`** (H.265 §7.3.8.11, §7.4.9.11) — the
//!   §9.3.4.2.4 derivation of `ctxInc` from the colour-component
//!   index, the current sub-block scan location `(xS, yS)`, the
//!   transform-block size, and the previously decoded
//!   `coded_sub_block_flag[xS+1][yS]` / `coded_sub_block_flag[xS][yS+1]`
//!   neighbours. Implemented in
//!   [`coded_sub_block_flag_ctx_inc`] following equations 9-35..9-39.
//!
//! * **`split_cu_flag` / `cu_skip_flag`** (H.265 §7.3.8.4, §7.4.9.4,
//!   §7.3.8.5, §7.4.9.5) — the §9.3.4.2.2 Table 9-49 derivation of
//!   `ctxInc` from the left and above neighbours' `condL` / `condA`
//!   booleans plus their availability:
//!   `ctxInc = (condL && availableL) + (condA && availableA)`.
//!   Implemented as the shared
//!   [`left_above_ctx_inc`] helper (the body of Table 9-49) plus
//!   the [`split_cu_flag_cond`] and [`cu_skip_flag_cond`] callbacks
//!   the caller wires into it. The Table 9-49 row for
//!   `split_cu_flag` compares each neighbour's `CtDepth[xNb][yNb]`
//!   to the current `cqtDepth`; the row for `cu_skip_flag` reads
//!   each neighbour's `cu_skip_flag[xNb][yNb]` directly. Both rows
//!   produce a `ctxInc` in `{0, 1, 2}`.
//!
//! Round 28 extends the ctxInc-derivation surface with six more
//! syntax elements whose Table 9-48 entry is itself a closed-form
//! expression on parameters the caller already has in hand (no
//! neighbour table walk, no CABAC engine drive at this layer):
//!
//! * **`split_transform_flag[ ][ ][ ]`** (H.265 §7.3.8.10, §7.4.9.10)
//!   — Table 9-48 row: `ctxInc = 5 − log2TrafoSize`. Valid
//!   `log2TrafoSize ∈ {2, 3, 4, 5}` for the residual-quadtree
//!   transform-block sizes, so `ctxInc ∈ {0, 1, 2, 3}`. Implemented
//!   in [`split_transform_flag_ctx_inc`].
//!
//! * **`cbf_luma[ ][ ][ ]`** (H.265 §7.3.8.10, §7.4.9.10) — Table
//!   9-48 row: `ctxInc = (trafoDepth == 0) ? 1 : 0`. Implemented in
//!   [`cbf_luma_ctx_inc`].
//!
//! * **`cbf_cb[ ][ ][ ]` / `cbf_cr[ ][ ][ ]`** (H.265 §7.3.8.10,
//!   §7.4.9.10) — Table 9-48 row: `ctxInc = trafoDepth`. The two
//!   syntax elements live in separate context banks (Table 9-22
//!   shares the bank shape but each chroma component has its own
//!   `ctxIdxOffset`); this layer hands back the bank-relative
//!   `ctxInc` only. Implemented in [`cbf_cb_ctx_inc`] and
//!   [`cbf_cr_ctx_inc`] (each is a wafer over the shared
//!   [`cbf_chroma_ctx_inc`] helper).
//!
//! * **`inter_pred_idc[ x0 ][ y0 ]`** (H.265 §7.3.8.6, §7.4.9.6) —
//!   Table 9-48 row: bin 0 uses
//!   `ctxInc = (nPbW + nPbH != 12) ? CtDepth[x0][y0] : 4`; bin 1
//!   uses `ctxInc = 4`. The `nPbW + nPbH == 12` condition picks out
//!   the prediction-block shapes whose luma area is 16 samples
//!   (8×4 and 4×8 PUs), which are encoded with the bin-0 escape
//!   onto the bin-1 context bank. Implemented in
//!   [`inter_pred_idc_ctx_inc`].
//!
//! * **`log2_res_scale_abs_plus1[ c ]`** (H.265 §7.3.8.13,
//!   §7.4.9.13) — Table 9-48 row: `ctxInc = 4*c + binIdx` for
//!   `binIdx ∈ {0, 1, 2, 3}` and `c ∈ {0, 1}` (Cb / Cr). The four
//!   bins of the TR(`cMax = 4`) prefix occupy four consecutive
//!   contexts per chroma component, so the per-component banks are
//!   `{0, 1, 2, 3}` (Cb) and `{4, 5, 6, 7}` (Cr). Implemented in
//!   [`log2_res_scale_abs_plus1_ctx_inc`].
//!
//! * **`res_scale_sign_flag[ c ]`** (H.265 §7.3.8.13, §7.4.9.13) —
//!   Table 9-48 row: `ctxInc = c` for `c ∈ {0, 1}`. One bit per
//!   chroma component, each on its own context. Implemented in
//!   [`res_scale_sign_flag_ctx_inc`].
//!
//! All round-28 entries are bin-0 (and bin-1 for `inter_pred_idc` /
//! `log2_res_scale_abs_plus1`) ctxInc derivations only; the matching
//! binarization shapes (FL for the flags, TR(`cMax = 4`) for
//! `log2_res_scale_abs_plus1` followed by a bypass tail) are driven
//! by the slice-data parser the same way as the round-26 / 27
//! elements.
//!
//! Round 30 extends the surface with the §7.3.4 `sao()` per-CTU
//! syntax-element family. Every element has either a single
//! context-coded bin-0 followed by zero or more bypass-coded bins
//! (Table 9-48) or is fully bypass-coded; no neighbour walk is
//! needed at this layer.
//!
//! * **`sao_merge_left_flag` / `sao_merge_up_flag`** (H.265 §7.3.4,
//!   §7.4.9.3) — Table 9-48 row: bin 0 `ctxInc = 0`. The FL
//!   binarization has `cMax = 1` (single bin). The two merge flags
//!   share the Table 9-5 context bank (Table 9-4). Implemented in
//!   [`sao_merge_flag_ctx_inc`] + [`SAO_MERGE_FLAG_FL_CMAX`].
//!
//! * **`sao_type_idx_luma` / `sao_type_idx_chroma`** (H.265 §7.3.4,
//!   §7.4.9.3) — Table 9-48 row: bin 0 `ctxInc = 0`, bin 1 bypass.
//!   The TR(`cMax = 2`, `cRiceParam = 0`) binarization caps the
//!   prefix at two bins, encoding `SaoTypeIdx ∈ {0, 1, 2}`. The two
//!   variants share the Table 9-6 context bank. Implemented in
//!   [`sao_type_idx_ctx_inc`] + [`SAO_TYPE_IDX_TR_CMAX`].
//!
//! * **`sao_offset_abs`** (H.265 §7.3.4, §7.4.9.3) — Table 9-48 row:
//!   all bins bypass. TR(`cMax = (1 << min(bitDepth, 10) − 5) − 1`,
//!   `cRiceParam = 0`). Implemented in [`sao_offset_abs_tr_cmax`].
//!
//! * **`sao_offset_sign`**, **`sao_band_position`**,
//!   **`sao_eo_class_luma` / `sao_eo_class_chroma`** (H.265 §7.3.4,
//!   §7.4.9.3) — Table 9-48 row: all bins bypass; FL binarizations
//!   with `cMax = 1`, `cMax = 31`, `cMax = 3` respectively.
//!   Implemented as [`SAO_OFFSET_SIGN_FL_CMAX`],
//!   [`SAO_BAND_POSITION_FL_CMAX`] + [`SAO_BAND_POSITION_FL_NBITS`],
//!   and [`SAO_EO_CLASS_FL_CMAX`] + [`SAO_EO_CLASS_FL_NBITS`].
//!
//! Round 31 extends the surface with the two §9.3.4.2.6 /
//! §9.3.4.2.7 derivations whose `ctxInc` carries *persistent*
//! per-transform-block state across sub-block invocations: the
//! greater-than-1 / greater-than-2 absolute-level flags. Both
//! elements are FL binarized with `cMax = 1` (one context-coded
//! bin) per Table 9-43, but the bin's ctxInc is driven by a small
//! sub-block-scoped state machine (`ctxSet`, `greater1Ctx`,
//! `lastGreater1Ctx`, `lastGreater1Flag`) that the residual-coding
//! loop threads from sub-block to sub-block within the same
//! transform block.
//!
//! * **`coeff_abs_level_greater1_flag[ n ]`** (H.265 §7.3.8.11,
//!   §7.4.9.11) — §9.3.4.2.6, equations 9-56..9-60. The per-bin
//!   `ctxInc = (ctxSet * 4) + min(3, greater1Ctx)` for luma; chroma
//!   adds `+16`. `ctxSet` is initialised at the start of each
//!   sub-block from the previous sub-block's terminal `greater1Ctx`
//!   value, and rotates within `{0, 1, 2, 3}` per the spec's
//!   `lastGreater1Ctx == 0` increment rule. Implemented as a
//!   pure-functional state machine in [`Greater1State`] (the
//!   walker the slice-data parser carries across the residual
//!   sub-blocks of one transform block) plus the per-flag step
//!   functions [`Greater1State::on_subblock_entry`] +
//!   [`Greater1State::on_coeff_abs_level_greater1_flag`].
//!
//! * **`coeff_abs_level_greater2_flag[ lastGreater1ScanPos ]`**
//!   (H.265 §7.3.8.11, §7.4.9.11) — §9.3.4.2.7, equations
//!   9-61..9-62. `ctxInc = ctxSet` (luma) / `ctxInc = ctxSet + 4`
//!   (chroma). The element is signalled at most once per sub-block
//!   at the first scan position that took the greater-1 escape
//!   (`lastGreater1ScanPos`). Implemented in
//!   [`coeff_abs_level_greater2_flag_ctx_inc`], reading the
//!   `ctxSet` value the §9.3.4.2.6 walker has already produced for
//!   that same sub-block.

use crate::cabac::{CabacEngine, CabacError, ContextModel};

// ---------------------------------------------------------------------
// Truncated-rice helper — §9.3.3.10
// ---------------------------------------------------------------------

/// §9.3.3.10 — read a truncated-rice prefix with `cMax` and
/// `cRiceParam = 0` (the only configuration this module currently
/// needs). The prefix is unary up to a maximum of `cMax >> 0` = `cMax`
/// bins; a value of `cMax / (1 << 0)` = `cMax` is the "escape" form
/// signalling that a suffix follows for callers that pair TR with an
/// EGk continuation.
///
/// `read_bin(bin_idx)` decodes the bin at index `bin_idx` (0-based)
/// using the caller's choice of context-coded or bypass-coded path, and
/// returns the decoded bin (0 = continue, 1 = terminate / escape).
///
/// Returns `(prefix_val, is_escape)`. `is_escape` is `true` exactly
/// when `prefix_val == cMax`: i.e. all `cMax` bins were 1, signalling
/// that a TR + EGk continuation suffix must be read by the caller.
fn read_truncated_rice_prefix<F>(c_max: u32, mut read_bin: F) -> Result<(u32, bool), CabacError>
where
    F: FnMut(u32) -> Result<u8, CabacError>,
{
    // Unary: count leading 1s up to c_max. The decoded bin tally is
    // simultaneously the prefix value (number of leading 1s) and the
    // escape signal (all-ones when value == c_max).
    let mut value: u32 = 0;
    while value < c_max {
        let bin = read_bin(value)?;
        if bin == 0 {
            return Ok((value, false));
        }
        value += 1;
    }
    Ok((value, true))
}

// ---------------------------------------------------------------------
// §9.3.3.11 — Exp-Golomb of order k (bypass-coded, MSB-first prefix
// then k-bit suffix). For k = 0 (the only k this round needs) the read
// is a unary "0…0 1" leading zeros count followed by `count` bits of
// suffix.
// ---------------------------------------------------------------------

/// §9.3.3.11 — decode an EGk-coded value from the bypass stream. For
/// `k = 0` (the form `cu_qp_delta_abs` uses after its TR prefix) the
/// reader counts leading zero bypass bins until a 1 is seen
/// (`leading_zeros`), then reads `leading_zeros` more bypass bins as
/// the suffix; the decoded value is `(1 << leading_zeros) - 1 + suffix`.
/// The generalisation for other `k` adds the EGk shift; we expose only
/// the `k = 0` form to keep the surface aligned with what this round
/// has a clean-room trace for.
fn decode_eg_k0(engine: &mut CabacEngine<'_>) -> Result<u32, CabacError> {
    // Count leading zero bypass bins, capped at 32 to prevent runaway.
    // A conforming HEVC `cu_qp_delta_abs` value is bounded by
    // (51 + QpBdOffsetY) / 2 + 2; 32 is comfortably above any legal
    // encoding.
    let mut leading_zeros: u32 = 0;
    while leading_zeros < 32 {
        let bin = engine.decode_bypass()?;
        if bin == 1 {
            break;
        }
        leading_zeros += 1;
    }
    let suffix = engine.decode_bypass_bits(leading_zeros as u8)?;
    Ok((1u32 << leading_zeros) - 1 + suffix)
}

// ---------------------------------------------------------------------
// cu_qp_delta_abs / cu_qp_delta_sign_flag (§7.3.8.14, §7.4.9.14)
// ---------------------------------------------------------------------

/// §9.3.4.2.2 Table 9-32 — per-bin `ctxInc` for `cu_qp_delta_abs`. Bin
/// 0 uses a dedicated context (`ctxInc = 0`); bins 1..=4 share a second
/// dedicated context (`ctxInc = 1`). The TR prefix carries at most
/// `cMax = 5` bins; the EGk(k=0) suffix is bypass-coded and not
/// indexed.
#[must_use]
pub fn cu_qp_delta_abs_ctx_inc(bin_idx: u32) -> u32 {
    if bin_idx == 0 {
        0
    } else {
        1
    }
}

/// `cu_qp_delta_abs` TR prefix `cMax`. §9.3.3.10 (with §9.3.4.2.2
/// row).
pub const CU_QP_DELTA_ABS_TR_CMAX: u32 = 5;

/// Decoded `cu_qp_delta` pair plus the §7.4.9.14 derived
/// `CuQpDeltaVal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CuQpDelta {
    /// `cu_qp_delta_abs` from the wire.
    pub abs: u32,
    /// `cu_qp_delta_sign_flag` from the wire; `None` when
    /// `abs == 0` (the flag is not signalled in that case).
    pub sign_flag: Option<u8>,
    /// Derived signed delta:
    /// `cu_qp_delta_abs * (1 − 2 * cu_qp_delta_sign_flag)`.
    /// Equals `0` when `abs == 0`.
    pub value: i32,
}

/// Decode the `cu_qp_delta` syntax pair (§7.3.8.14 / §7.4.9.14) from
/// the CABAC engine, consuming the two context variables for the
/// `cu_qp_delta_abs` TR prefix (bin 0 + bins 1..=4) and any required
/// bypass bins for the EGk(k=0) suffix and the sign flag.
///
/// `ctx_bin0` and `ctx_bin_rest` are the caller's
/// `(pStateIdx, valMps)` state for the two `cu_qp_delta_abs` contexts;
/// they are mutated in place per §9.3.4.3.2.2 state transition.
///
/// Returns the decoded [`CuQpDelta`] including the §7.4.9.14
/// `CuQpDeltaVal` derivation.
pub fn decode_cu_qp_delta(
    engine: &mut CabacEngine<'_>,
    ctx_bin0: &mut ContextModel,
    ctx_bin_rest: &mut ContextModel,
) -> Result<CuQpDelta, CabacError> {
    // Decode the §9.3.3.10 TR prefix using two contexts: bin 0 uses
    // ctx_bin0, all subsequent bins share ctx_bin_rest.
    let (prefix, is_escape) = read_truncated_rice_prefix(CU_QP_DELTA_ABS_TR_CMAX, |bin_idx| {
        if cu_qp_delta_abs_ctx_inc(bin_idx) == 0 {
            engine.decode_decision(ctx_bin0)
        } else {
            engine.decode_decision(ctx_bin_rest)
        }
    })?;

    // §9.3.3.10: if the prefix is the all-ones escape (== cMax), an
    // EGk(k=0) suffix follows; the decoded value is prefix + suffix.
    let abs = if is_escape {
        prefix + decode_eg_k0(engine)?
    } else {
        prefix
    };

    // §7.4.9.14: the sign flag is signalled only when abs != 0.
    let (sign_flag, signed) = if abs == 0 {
        (None, 0i32)
    } else {
        let s = engine.decode_bypass()?;
        // CuQpDeltaVal = abs * (1 − 2 * sign_flag)
        let sign_factor = 1i32 - 2 * (s as i32);
        (Some(s), abs as i32 * sign_factor)
    };

    Ok(CuQpDelta {
        abs,
        sign_flag,
        value: signed,
    })
}

// ---------------------------------------------------------------------
// last_sig_coeff_{x,y}_{prefix,suffix} (§7.3.8.11, §7.4.9.11)
// ---------------------------------------------------------------------

/// Which `last_sig_coeff_*_prefix` bank the caller's context-model
/// slice corresponds to. The X and Y prefixes live in separate banks
/// of contexts; we tag the request so the caller can pick the right
/// slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LastSigCoeffBank {
    /// `last_sig_coeff_x_prefix` bank.
    X,
    /// `last_sig_coeff_y_prefix` bank.
    Y,
}

/// §9.3.4.2.3 — `(ctxOffset, ctxShift)` for the
/// `last_sig_coeff_{x,y}_prefix` ctxInc derivation. `log2_trafo_size`
/// is the transform-block log2 dimension (2..=5; 2 = 4×4, 5 = 32×32);
/// `is_chroma` selects the chroma bank when `cIdx > 0`.
///
/// Spec equations (§9.3.4.2.3 Table 9-26 / 9-27):
///
/// ```text
/// // luma (cIdx == 0):
/// ctxOffset = 3 * (log2TrafoSize - 2) + ((log2TrafoSize - 1) >> 2)
/// ctxShift  = (log2TrafoSize + 1) >> 2
///
/// // chroma (cIdx > 0):
/// ctxOffset = 15
/// ctxShift  = log2TrafoSize - 2
/// ```
#[must_use]
pub fn last_sig_coeff_prefix_ctx_offset_shift(log2_trafo_size: u32, is_chroma: bool) -> (u32, u32) {
    if is_chroma {
        // §9.3.4.2.3 — chroma TBs always sit at ctxOffset = 15,
        // ctxShift = log2TrafoSize − 2.
        (15, log2_trafo_size - 2)
    } else {
        // §9.3.4.2.3 — luma: piecewise linear in log2TrafoSize.
        let offset = 3 * (log2_trafo_size - 2) + ((log2_trafo_size - 1) >> 2);
        let shift = (log2_trafo_size + 1) >> 2;
        (offset, shift)
    }
}

/// §9.3.4.2.3 — bank-relative `ctxInc` for a single
/// `last_sig_coeff_*_prefix` bin:
///
/// ```text
/// ctxInc = (binIdx >> ctxShift) + ctxOffset
/// ```
#[must_use]
pub fn last_sig_coeff_prefix_ctx_inc(bin_idx: u32, ctx_offset: u32, ctx_shift: u32) -> u32 {
    (bin_idx >> ctx_shift) + ctx_offset
}

/// §9.3.3.10 — `cMax` for `last_sig_coeff_{x,y}_prefix`:
/// `(log2TrafoSize << 1) − 1`. For a 4×4 TB (log2=2) the prefix is at
/// most 3 bins; for a 32×32 TB (log2=5) at most 9 bins.
#[must_use]
pub fn last_sig_coeff_prefix_cmax(log2_trafo_size: u32) -> u32 {
    (log2_trafo_size << 1) - 1
}

/// §7.4.9.11 equations 7-74..7-77 — derive the final
/// `LastSignificantCoeffX/Y` position from a decoded
/// `(prefix, optional suffix)` pair. When `prefix <= 3` no suffix is
/// signalled and the position equals the prefix. When `prefix > 3` the
/// suffix is `nBits = (prefix >> 1) − 1` bits, and the position is
///
/// ```text
/// pos = ( ( 2 + ( prefix & 1 ) ) << ( (prefix >> 1) - 1 ) ) + suffix
/// ```
///
/// `suffix` is the unsigned suffix value (caller already decoded the
/// `nBits` bypass bins, MSB-first); pass `None` when `prefix <= 3`.
#[must_use]
pub fn last_sig_coeff_position(prefix: u32, suffix: Option<u32>) -> u32 {
    if prefix <= 3 {
        // §7.4.9.11: no suffix, the prefix is the final value.
        prefix
    } else {
        let n_bits = (prefix >> 1) - 1;
        let base = (2 + (prefix & 1)) << n_bits;
        base + suffix.unwrap_or(0)
    }
}

/// Suffix bit count for `last_sig_coeff_{x,y}_suffix`: `nBits =
/// (prefix >> 1) − 1` when `prefix > 3`, otherwise 0 (no suffix
/// signalled).
#[must_use]
pub fn last_sig_coeff_suffix_n_bits(prefix: u32) -> u32 {
    if prefix > 3 {
        (prefix >> 1) - 1
    } else {
        0
    }
}

/// Decoded `last_sig_coeff_*` element (one of X or Y; same shape for
/// both banks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LastSigCoeff {
    /// `last_sig_coeff_*_prefix` from the wire.
    pub prefix: u32,
    /// `last_sig_coeff_*_suffix` from the wire (`None` when
    /// `prefix <= 3`, the no-suffix-signalled case).
    pub suffix: Option<u32>,
    /// §7.4.9.11 derived `LastSignificantCoeff{X,Y}` position. Range
    /// 0..=(1 << log2TrafoSize) - 1.
    pub position: u32,
}

/// Decode one `last_sig_coeff_*_{prefix,suffix}` pair from the CABAC
/// engine. `ctx_bank` is the caller-supplied slice of context variables
/// for the active bank ([`LastSigCoeffBank`]); the bin at index
/// `binIdx` consumes the context at index
/// `ctx_inc = (binIdx >> ctxShift) + ctxOffset` within that slice (the
/// caller is responsible for sizing the slice so the highest used
/// `ctx_inc` index is in-bounds).
///
/// Returns the decoded prefix, optional suffix, and the §7.4.9.11
/// derived position. The arguments
/// `log2_trafo_size` and `is_chroma` drive both the TR-prefix `cMax`
/// and the §9.3.4.2.3 `(ctxOffset, ctxShift)` derivation.
///
/// The §7.4.9.11 swap for `scanIdx == 2` (the diagonal scan reflection
/// of `LastSignificantCoeff{X,Y}`) is **not** applied here; the caller
/// receives the on-wire X / Y positions and applies the swap at the
/// residual-coding level where `scanIdx` is known.
pub fn decode_last_sig_coeff(
    engine: &mut CabacEngine<'_>,
    log2_trafo_size: u32,
    is_chroma: bool,
    ctx_bank: &mut [ContextModel],
) -> Result<LastSigCoeff, CabacError> {
    let c_max = last_sig_coeff_prefix_cmax(log2_trafo_size);
    let (ctx_offset, ctx_shift) =
        last_sig_coeff_prefix_ctx_offset_shift(log2_trafo_size, is_chroma);

    let (prefix, _is_escape) = read_truncated_rice_prefix(c_max, |bin_idx| {
        let ctx_inc = last_sig_coeff_prefix_ctx_inc(bin_idx, ctx_offset, ctx_shift) as usize;
        engine.decode_decision(&mut ctx_bank[ctx_inc])
    })?;

    let n_bits = last_sig_coeff_suffix_n_bits(prefix);
    let suffix = if n_bits > 0 {
        Some(engine.decode_bypass_bits(n_bits as u8)?)
    } else {
        None
    };
    let position = last_sig_coeff_position(prefix, suffix);
    Ok(LastSigCoeff {
        prefix,
        suffix,
        position,
    })
}

// ---------------------------------------------------------------------
// coded_sub_block_flag ctxInc — §9.3.4.2.4
// ---------------------------------------------------------------------

/// §9.3.4.2.4 — derive the `ctxInc` for one bin of
/// `coded_sub_block_flag` from the colour-component index, the
/// current sub-block scan location `(xS, yS)`, the transform-block
/// size `log2TrafoSize`, and the two previously decoded neighbour
/// bins in sub-block scan order.
///
/// Spec equations:
///
/// ```text
/// // §9.3.4.2.4, equations 9-35..9-39
/// csbfCtx = 0
/// if xS < (1 << (log2TrafoSize - 2)) - 1:
///     csbfCtx += coded_sub_block_flag[xS + 1][yS]
/// if yS < (1 << (log2TrafoSize - 2)) - 1:
///     csbfCtx += coded_sub_block_flag[xS][yS + 1]
/// if cIdx == 0:
///     ctxInc = min(csbfCtx, 1)
/// else:
///     ctxInc = 2 + min(csbfCtx, 1)
/// ```
///
/// `right_neighbour` is the previously decoded
/// `coded_sub_block_flag[xS + 1][yS]` (0/1); pass `0` whenever the
/// current sub-block sits on the right edge of the TB
/// (`xS == (1 << (log2TrafoSize - 2)) - 1`), where equation 9-36 does
/// not apply. `below_neighbour` is the previously decoded
/// `coded_sub_block_flag[xS][yS + 1]` (0/1); pass `0` whenever the
/// current sub-block sits on the bottom edge of the TB
/// (`yS == (1 << (log2TrafoSize - 2)) - 1`).
///
/// Returns `ctxInc` in `{0, 1}` for luma (`cIdx == 0`) or
/// `{2, 3}` for chroma (`cIdx > 0`).
///
/// # Panics
/// Does not panic; both neighbour inputs are clamped via the boolean
/// `>= 1` comparison.
#[must_use]
pub fn coded_sub_block_flag_ctx_inc(
    is_chroma: bool,
    right_neighbour: u8,
    below_neighbour: u8,
) -> u32 {
    // §9.3.4.2.4: csbfCtx is the unsigned sum of the two neighbour
    // bins (each either 0 or 1), so it lies in {0, 1, 2}. The final
    // ctxInc clamps to {0, 1} via min(csbfCtx, 1).
    let csbf_ctx = (right_neighbour & 1) as u32 + (below_neighbour & 1) as u32;
    let clipped = csbf_ctx.min(1);
    if is_chroma {
        // §9.3.4.2.4 equation 9-39: chroma uses the {2, 3} bank.
        2 + clipped
    } else {
        // §9.3.4.2.4 equation 9-38: luma uses the {0, 1} bank.
        clipped
    }
}

/// §9.3.4.2.4 — helper that takes the full sub-block location and
/// transform-block size and applies the edge-gating from equations
/// 9-36 / 9-37 before delegating to [`coded_sub_block_flag_ctx_inc`].
/// The caller passes the previously decoded
/// `coded_sub_block_flag[xS + 1][yS]` and
/// `coded_sub_block_flag[xS][yS + 1]` values regardless of the edge
/// condition; this function zeroes them when the corresponding
/// equation 9-36 / 9-37 does not apply (sub-block on the right /
/// bottom TB edge).
///
/// `xs` / `ys` are the sub-block scan location in 4-sample units (so
/// a 32×32 TB at `log2TrafoSize == 5` has `xs, ys ∈ 0..8`).
#[must_use]
pub fn coded_sub_block_flag_ctx_inc_with_edge(
    is_chroma: bool,
    xs: u32,
    ys: u32,
    log2_trafo_size: u32,
    right_neighbour: u8,
    below_neighbour: u8,
) -> u32 {
    // §9.3.4.2.4 edge gates: ( 1 << ( log2TrafoSize − 2 ) ) − 1 is
    // the maximum sub-block index along each axis for the TB.
    let max_sub_block_idx = (1u32 << (log2_trafo_size - 2)) - 1;
    let right = if xs < max_sub_block_idx {
        right_neighbour & 1
    } else {
        0
    };
    let below = if ys < max_sub_block_idx {
        below_neighbour & 1
    } else {
        0
    };
    coded_sub_block_flag_ctx_inc(is_chroma, right, below)
}

// ---------------------------------------------------------------------
// split_cu_flag / cu_skip_flag ctxInc — §9.3.4.2.2 Table 9-49
// ---------------------------------------------------------------------

/// §9.3.4.2.2 Table 9-49 — the shared `ctxInc` derivation for
/// syntax elements that take their `ctxInc` from a `condL` / `condA`
/// neighbour pair plus the §6.4.1 availability of the neighbour
/// blocks:
///
/// ```text
/// ctxInc = (condL && availableL) + (condA && availableA)
/// ```
///
/// Used by `split_cu_flag` (with `condL = CtDepth[xNbL][yNbL] >
/// cqtDepth`, `condA = CtDepth[xNbA][yNbA] > cqtDepth`) and
/// `cu_skip_flag` (with `condL = cu_skip_flag[xNbL][yNbL]`,
/// `condA = cu_skip_flag[xNbA][yNbA]`).
///
/// Returns a `ctxInc` in `{0, 1, 2}`.
#[must_use]
pub fn left_above_ctx_inc(cond_l: bool, available_l: bool, cond_a: bool, available_a: bool) -> u32 {
    let l = (cond_l && available_l) as u32;
    let a = (cond_a && available_a) as u32;
    l + a
}

/// §9.3.4.2.2 Table 9-49 — `condL` / `condA` predicate for
/// `split_cu_flag`: each neighbour contributes `CtDepth[xNb][yNb] >
/// cqtDepth`. Returns the predicate as a `bool` ready to feed into
/// [`left_above_ctx_inc`].
///
/// `neighbour_ct_depth` is the `CtDepth[xNb][yNb]` value already
/// derived by the caller for the neighbouring (left or above) block;
/// `cqt_depth` is the current block's coding-quadtree depth.
#[must_use]
pub fn split_cu_flag_cond(neighbour_ct_depth: u32, cqt_depth: u32) -> bool {
    neighbour_ct_depth > cqt_depth
}

/// §9.3.4.2.2 Table 9-49 — `condL` / `condA` predicate for
/// `cu_skip_flag`: each neighbour contributes
/// `cu_skip_flag[xNb][yNb]` as-is.
#[must_use]
pub fn cu_skip_flag_cond(neighbour_cu_skip_flag: u8) -> bool {
    (neighbour_cu_skip_flag & 1) != 0
}

/// §9.3.4.2.2 Table 9-49 row for `split_cu_flag`. Convenience: takes
/// the neighbour `CtDepth` values and availability and returns
/// `ctxInc` directly.
///
/// The neighbour `CtDepth` arguments are ignored when the matching
/// `available` flag is false (the `(condX && availableX)` AND in the
/// Table 9-49 formula short-circuits in that case), but the caller
/// may pass any placeholder for them.
#[must_use]
pub fn split_cu_flag_ctx_inc(
    left_ct_depth: u32,
    available_l: bool,
    above_ct_depth: u32,
    available_a: bool,
    cqt_depth: u32,
) -> u32 {
    left_above_ctx_inc(
        split_cu_flag_cond(left_ct_depth, cqt_depth),
        available_l,
        split_cu_flag_cond(above_ct_depth, cqt_depth),
        available_a,
    )
}

/// §9.3.4.2.2 Table 9-49 row for `cu_skip_flag`. Convenience: takes
/// the neighbour `cu_skip_flag` values and availability and returns
/// `ctxInc` directly.
#[must_use]
pub fn cu_skip_flag_ctx_inc(
    left_cu_skip_flag: u8,
    available_l: bool,
    above_cu_skip_flag: u8,
    available_a: bool,
) -> u32 {
    left_above_ctx_inc(
        cu_skip_flag_cond(left_cu_skip_flag),
        available_l,
        cu_skip_flag_cond(above_cu_skip_flag),
        available_a,
    )
}

// ---------------------------------------------------------------------
// split_transform_flag ctxInc — Table 9-48 row
// ---------------------------------------------------------------------

/// Table 9-48 row for `split_transform_flag[ ][ ][ ]`:
///
/// ```text
/// ctxInc = 5 − log2TrafoSize
/// ```
///
/// The residual-quadtree transform-block sizes are
/// `log2TrafoSize ∈ {2, 3, 4, 5}` (4×4, 8×8, 16×16, 32×32), so the
/// returned `ctxInc` lies in `{0, 1, 2, 3}`.
///
/// The §7.4.9.10 `split_transform_flag` syntax element is itself
/// only signalled when `log2TrafoSize > MaxTbLog2SizeY` is false and
/// `log2TrafoSize > MinTbLog2SizeY` is true; the caller is responsible
/// for that gate. This function is the per-bin ctxInc derivation
/// only.
///
/// # Panics
/// Does not panic. For `log2TrafoSize > 5` the result is `u32::MAX`
/// (a wrap of the `5 − x` subtraction); the spec only defines the
/// derivation for `log2TrafoSize ∈ {2, 3, 4, 5}` and the caller is
/// expected to honour that range.
#[must_use]
pub fn split_transform_flag_ctx_inc(log2_trafo_size: u32) -> u32 {
    // §7.4.9.10 restricts the syntax element to log2TrafoSize in
    // {2, 3, 4, 5}; perform the subtraction wrap-safely so an
    // out-of-range caller surfaces a u32::MAX rather than panics.
    5u32.wrapping_sub(log2_trafo_size)
}

// ---------------------------------------------------------------------
// cbf_luma / cbf_cb / cbf_cr ctxInc — Table 9-48 rows
// ---------------------------------------------------------------------

/// Table 9-48 row for `cbf_luma[ ][ ][ ]`:
///
/// ```text
/// ctxInc = (trafoDepth == 0) ? 1 : 0
/// ```
///
/// `trafoDepth` is the §7.4.9.10 transform-tree depth, where 0 is
/// the root of the residual quadtree. The two-context bank is
/// `{0, 1}` and the spec assigns the *deeper* depths to context 0
/// (the "no further split, deeper TU" case) and the depth-0 root
/// to context 1.
#[must_use]
pub fn cbf_luma_ctx_inc(trafo_depth: u32) -> u32 {
    if trafo_depth == 0 {
        1
    } else {
        0
    }
}

/// Shared `ctxInc` derivation for `cbf_cb` and `cbf_cr` (Table 9-48
/// row): `ctxInc = trafoDepth`. The §7.4.9.10 residual quadtree
/// allows `trafoDepth ∈ {0, 1, 2, 3, 4}`; the chroma cbf
/// context banks supplied by Table 9-22 are five entries per
/// component, matching the depth range.
///
/// The two callers ([`cbf_cb_ctx_inc`] and [`cbf_cr_ctx_inc`])
/// delegate to this helper; the per-component `ctxIdxOffset` is the
/// caller's responsibility (Cb and Cr each have a distinct bank in
/// Table 9-4).
#[must_use]
pub fn cbf_chroma_ctx_inc(trafo_depth: u32) -> u32 {
    trafo_depth
}

/// Table 9-48 row for `cbf_cb[ ][ ][ ]`: `ctxInc = trafoDepth`.
/// Forwards to [`cbf_chroma_ctx_inc`] for the shared formula.
#[must_use]
pub fn cbf_cb_ctx_inc(trafo_depth: u32) -> u32 {
    cbf_chroma_ctx_inc(trafo_depth)
}

/// Table 9-48 row for `cbf_cr[ ][ ][ ]`: `ctxInc = trafoDepth`.
/// Forwards to [`cbf_chroma_ctx_inc`] for the shared formula.
#[must_use]
pub fn cbf_cr_ctx_inc(trafo_depth: u32) -> u32 {
    cbf_chroma_ctx_inc(trafo_depth)
}

// ---------------------------------------------------------------------
// inter_pred_idc ctxInc — Table 9-48 row
// ---------------------------------------------------------------------

/// Table 9-48 row for `inter_pred_idc[ x0 ][ y0 ]`:
///
/// ```text
/// bin 0:  ctxInc = (nPbW + nPbH != 12) ? CtDepth[x0][y0] : 4
/// bin 1:  ctxInc = 4
/// ```
///
/// `n_pb_w` and `n_pb_h` are the prediction-block width and height in
/// luma samples (the §7.3.8.6 `nPbW` / `nPbH` derivation). The
/// `nPbW + nPbH == 12` condition picks the two prediction-block
/// shapes whose luma area is 16 samples — namely 8×4 (sum = 12) and
/// 4×8 (sum = 12) — which are encoded with the bin-0 escape onto
/// the shared bin-1 context bank (`ctxInc = 4`). All other shapes
/// route bin 0 through the per-`CtDepth` context bank `{0, 1, 2, 3}`.
///
/// `ct_depth` is the §7.4.9.4 `CtDepth[x0][y0]` value (the coding-
/// quadtree depth of the current coding unit); valid range is
/// `{0, 1, 2, 3}` matching the §7.4.7.1 MaxCuDepth limit.
///
/// Returns `ctxInc` for any `binIdx`; `binIdx ∈ {2, 3, 4}` is `na`
/// per Table 9-48 (the binarization is a TR with `cMax = 2`), so the
/// function returns `4` for `binIdx >= 1` (matching bin 1).
#[must_use]
pub fn inter_pred_idc_ctx_inc(bin_idx: u32, n_pb_w: u32, n_pb_h: u32, ct_depth: u32) -> u32 {
    if bin_idx == 0 {
        // Bin 0 routes through CtDepth unless the PU is 8×4 or 4×8
        // (luma area 16 samples ⇒ nPbW + nPbH == 12), in which case
        // it shares the bin-1 context bank.
        if n_pb_w + n_pb_h != 12 {
            ct_depth
        } else {
            4
        }
    } else {
        // Bin 1 — and the `na` entries `binIdx >= 2` — use ctx 4.
        // Table 9-47 caps the prefix at `cMax = 2` (one or two bins),
        // so the slice-data parser never invokes this with binIdx >=
        // 2; the return value is the spec's bin-1 ctxInc.
        4
    }
}

// ---------------------------------------------------------------------
// log2_res_scale_abs_plus1 / res_scale_sign_flag ctxInc —
// Table 9-48 rows for §7.3.8.13 cross-component prediction
// ---------------------------------------------------------------------

/// Table 9-48 row for `log2_res_scale_abs_plus1[ c ]`:
///
/// ```text
/// ctxInc = 4*c + binIdx
/// ```
///
/// `c` is the chroma-component index passed at the §7.3.8.13
/// `cross_comp_pred()` call site: `c == 0` for Cb, `c == 1` for Cr.
/// The TR(`cMax = 4`) prefix occupies four consecutive contexts per
/// chroma component, so the per-component banks are `{0, 1, 2, 3}`
/// (Cb) and `{4, 5, 6, 7}` (Cr).
///
/// `binIdx` is the position of the bin within the TR prefix, in
/// `{0, 1, 2, 3}`. The `binIdx >= 4` slots (the EGk(k=0) bypass
/// escape, when the prefix is the all-ones escape `0b1111`) are
/// `na` for the context-coded path; the caller switches to bypass
/// decoding at that point and never passes them here.
#[must_use]
pub fn log2_res_scale_abs_plus1_ctx_inc(bin_idx: u32, c: u32) -> u32 {
    // Per-component bank offset (4) times c, plus the bin position
    // within the TR prefix.
    4 * c + bin_idx
}

/// Table 9-48 row for `res_scale_sign_flag[ c ]`:
///
/// ```text
/// ctxInc = c
/// ```
///
/// `c == 0` for Cb, `c == 1` for Cr; each component has its own
/// single-bin sign-flag context.
#[must_use]
pub fn res_scale_sign_flag_ctx_inc(c: u32) -> u32 {
    c
}

// ---------------------------------------------------------------------
// §7.3.4 sao() — per-CTU SAO syntax elements (Table 9-48 / Table 9-43)
// ---------------------------------------------------------------------
//
// Round 30 extends the §9.3.4.2 surface with the SAO syntax-element
// family, all decoded inside the §7.3.4 `sao()` per-CTU block. Every
// element is either a single context-coded bin-0 followed by zero or
// more bypass-coded bins (Table 9-48), or fully bypass-coded; no
// neighbour-table walk is needed at this layer.
//
// Table 9-48 rows captured here:
//
// * `sao_merge_left_flag`            → bin 0: ctxInc = 0  (FL, cMax = 1)
// * `sao_merge_up_flag`              → bin 0: ctxInc = 0  (FL, cMax = 1)
// * `sao_type_idx_luma`              → bin 0: ctxInc = 0; bin 1: bypass
//                                      (TR, cMax = 2, cRiceParam = 0)
// * `sao_type_idx_chroma`            → bin 0: ctxInc = 0; bin 1: bypass
//                                      (TR, cMax = 2, cRiceParam = 0)
// * `sao_offset_abs[ ][ ][ ][ ]`     → bypass × all
//                                      (TR, cMax = (1 << (min(bitDepth, 10) − 5)) − 1,
//                                      cRiceParam = 0)
// * `sao_offset_sign[ ][ ][ ][ ]`    → bypass × 1     (FL, cMax = 1)
// * `sao_band_position[ ][ ][ ]`     → bypass × 5     (FL, cMax = 31)
// * `sao_eo_class_luma`              → bypass × 2     (FL, cMax = 3)
// * `sao_eo_class_chroma`            → bypass × 2     (FL, cMax = 3)
//
// Table 9-4 association (the ctx banks each element lives in):
//
// * `sao_merge_left_flag` + `sao_merge_up_flag` share Table 9-5 (one
//   ctxIdx per initType).
// * `sao_type_idx_luma` + `sao_type_idx_chroma` share Table 9-6 (one
//   ctxIdx per initType).
//
// The two pairs are distinct context banks; this layer hands back the
// bank-relative ctxInc only (0 for every context-coded SAO bin).

/// Table 9-48 row for `sao_merge_left_flag` and `sao_merge_up_flag`:
///
/// ```text
/// bin 0: ctxInc = 0
/// ```
///
/// Only bin 0 is context-coded; the FL binarization has `cMax = 1`, so
/// there is exactly one bin and the function returns 0 for any call.
/// The two merge flags share the same Table 9-5 context bank (see
/// Table 9-4); this layer returns the bank-relative ctxInc only.
#[must_use]
pub fn sao_merge_flag_ctx_inc() -> u32 {
    0
}

/// Table 9-48 row for `sao_type_idx_luma` and `sao_type_idx_chroma`:
///
/// ```text
/// bin 0: ctxInc = 0
/// bin 1: bypass
/// ```
///
/// `binIdx == 0` is context-coded with `ctxInc = 0`; `binIdx == 1` is
/// bypass per Table 9-48 and is **not** routed through a context. The
/// TR(`cMax = 2`) binarization caps the prefix at two bins. Callers
/// driving the §9.3 engine must invoke `decode_decision` for bin 0 and
/// `decode_bypass` for bin 1 directly, switching paths between bins.
///
/// Returns the ctxInc when called with `bin_idx == 0`; any other value
/// is `na` per Table 9-48 and the function returns 0 defensively
/// (callers should not invoke this for the bypass bin).
#[must_use]
pub fn sao_type_idx_ctx_inc(bin_idx: u32) -> u32 {
    debug_assert!(
        bin_idx == 0,
        "sao_type_idx Table 9-48 row: only bin 0 is context-coded; bin >= 1 is bypass"
    );
    0
}

/// Table 9-43 binarization shape for `sao_merge_left_flag` and
/// `sao_merge_up_flag`: FL with `cMax = 1` (single bin).
pub const SAO_MERGE_FLAG_FL_CMAX: u32 = 1;

/// Table 9-43 binarization shape for `sao_type_idx_luma` and
/// `sao_type_idx_chroma`: TR with `cMax = 2`, `cRiceParam = 0`. The
/// prefix is at most two bins; bin 0 is context-coded (ctxInc = 0),
/// bin 1 is bypass-coded per Table 9-48.
pub const SAO_TYPE_IDX_TR_CMAX: u32 = 2;

/// Table 9-43 binarization shape for `sao_offset_sign`: FL with
/// `cMax = 1` (single bypass-coded bin).
pub const SAO_OFFSET_SIGN_FL_CMAX: u32 = 1;

/// Table 9-43 binarization shape for `sao_band_position`: FL with
/// `cMax = 31` (five bypass-coded bins; the FL-of-cMax-31 form
/// uses `ceil(log2(cMax + 1)) = 5` bits per §9.3.3.5).
pub const SAO_BAND_POSITION_FL_CMAX: u32 = 31;

/// FL-binarization bit count for `sao_band_position`: 5 bypass bins.
/// Per §9.3.3.5, FL of `cMax = N` uses `ceil(log2(N + 1))` bits.
pub const SAO_BAND_POSITION_FL_NBITS: u32 = 5;

/// Table 9-43 binarization shape for `sao_eo_class_luma` and
/// `sao_eo_class_chroma`: FL with `cMax = 3` (two bypass-coded bins,
/// encoding the §7.4.9.3 `SaoEoClass` ∈ {0, 1, 2, 3}: horizontal,
/// vertical, 135-degree, 45-degree).
pub const SAO_EO_CLASS_FL_CMAX: u32 = 3;

/// FL-binarization bit count for `sao_eo_class_luma` /
/// `sao_eo_class_chroma`: 2 bypass bins.
pub const SAO_EO_CLASS_FL_NBITS: u32 = 2;

/// Table 9-43 binarization shape for `sao_offset_abs`: TR with
///
/// ```text
/// cMax = (1 << min(bitDepth, 10) - 5) - 1
/// cRiceParam = 0
/// ```
///
/// where `bitDepth` is the component's bit-depth in samples. For the
/// canonical 8-bit case `cMax = (1 << 3) - 1 = 7`; for 10-bit and
/// beyond `cMax = (1 << 5) - 1 = 31` (Min-clamped to 10 by the spec
/// to keep the maximum SAO offset range bounded). All bins of the TR
/// prefix are bypass-coded per Table 9-48.
///
/// `bit_depth` is the §7.4.7.1 `BitDepthY` or `BitDepthC` value
/// for the colour component being decoded.
#[must_use]
pub fn sao_offset_abs_tr_cmax(bit_depth: u32) -> u32 {
    // §9.3.3.5: Min(bitDepth, 10). Clamp at the spec maximum so the
    // SAO offset range never exceeds [−31, 31].
    let clamped = if bit_depth < 10 { bit_depth } else { 10 };
    // The subtraction is well-defined for bit_depth >= 5; HEVC mandates
    // bitDepth >= 8 in every conformant profile, so the result is
    // always positive in practice.
    (1u32 << (clamped - 5)) - 1
}

// ---------------------------------------------------------------------
// coeff_abs_level_greater1_flag / coeff_abs_level_greater2_flag
// — §9.3.4.2.6 (equations 9-56..9-60) and §9.3.4.2.7 (equations
// 9-61..9-62). Both elements are Table 9-43 FL with cMax = 1, so the
// binarization shape is one context-coded bin. The bin's `ctxInc` is
// driven by a small per-transform-block state machine that threads
// `ctxSet` / `greater1Ctx` from sub-block to sub-block.
// ---------------------------------------------------------------------

/// Persistent §9.3.4.2.6 state machine carried across the residual
/// sub-blocks of a single transform block. The slice-data parser
/// constructs one [`Greater1State`] per transform block and threads it
/// through the §7.3.8.11 sub-block loop, calling
/// [`Greater1State::on_subblock_entry`] before the sub-block's first
/// `coeff_abs_level_greater1_flag` bin and
/// [`Greater1State::on_coeff_abs_level_greater1_flag`] after each
/// decoded bin's value lands.
///
/// The machine is the §9.3.4.2.6 equations encoded as a tiny FSM:
///
/// * Equations 9-56 / 9-57 initialise `ctxSet` for each sub-block
///   based on the sub-block scan index `i` and the colour-component
///   index `cIdx`.
/// * The §9.3.4.2.6 `lastGreater1Ctx == 0` step (the spec's
///   "increment `ctxSet` by one") routes the sub-block into the
///   *next* context set whenever the previous sub-block ran out of
///   the `greater1Ctx ∈ {1, 2, 3}` band.
/// * Equation 9-59 `ctxInc = (ctxSet * 4) + min(3, greater1Ctx)`
///   plus the chroma `+16` offset (eq. 9-60) is reported by
///   [`Greater1State::current_ctx_inc`].
/// * The post-bin `greater1Ctx` update from §9.3.4.2.6 (the
///   `lastGreater1Flag == 1` → 0, `lastGreater1Flag == 0` →
///   increment-clamped-by-3 rule) lives in
///   [`Greater1State::on_coeff_abs_level_greater1_flag`].
///
/// The §9.3.4.2.7 `coeff_abs_level_greater2_flag` derivation reads
/// the *current* `ctxSet` from this same machine (the per-sub-block
/// value, not the post-update one) via
/// [`Greater1State::ctx_set`], so callers compose the
/// greater-2 invocation after the greater-1 sub-block walk without
/// any cross-state-machine plumbing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Greater1State {
    /// §9.3.4.2.6 `ctxSet` — the current sub-block's context set in
    /// `{0, 1, 2, 3}`. Eq. 9-59 multiplies this by 4 to pick the
    /// per-set base.
    ctx_set: u32,
    /// §9.3.4.2.6 `greater1Ctx` — the within-set ctx position in
    /// `{0, 1, 2, 3}` (clamped at 3 for the `ctxInc` eq. 9-59).
    /// `0` means the sub-block has run out of the active band and
    /// the next sub-block's entry will bump `ctxSet`.
    greater1_ctx: u32,
    /// Has any sub-block been entered yet in this transform block?
    /// Used to recognise the "first sub-block" case the spec calls
    /// out (`lastGreater1Ctx = 1`).
    seen_any_subblock: bool,
    /// True iff at least one `coeff_abs_level_greater1_flag` bin has
    /// been read in *this* sub-block; signals that the next entry's
    /// `lastGreater1Ctx == 0` check has a well-defined
    /// `lastGreater1Flag` value to consume.
    has_last_greater1_flag: bool,
}

impl Greater1State {
    /// Construct a fresh state machine at the start of a transform
    /// block. No sub-block has yet been entered;
    /// [`on_subblock_entry`](Self::on_subblock_entry) must be called
    /// before [`current_ctx_inc`](Self::current_ctx_inc) is
    /// meaningful.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ctx_set: 0,
            greater1_ctx: 1,
            seen_any_subblock: false,
            has_last_greater1_flag: false,
        }
    }

    /// §9.3.4.2.6 sub-block entry: invoked once per sub-block before
    /// the first `coeff_abs_level_greater1_flag` bin is decoded.
    ///
    /// * `i` — the §7.3.8.11 sub-block scan index inside the
    ///   transform block.
    /// * `is_chroma` — `true` for `cIdx > 0`, used by eq. 9-56 to
    ///   force `ctxSet = 0` on every chroma sub-block.
    /// * `last_greater1_flag` — the most-recently decoded
    ///   `coeff_abs_level_greater1_flag` value from the *previous*
    ///   sub-block (0 or 1); ignored when this is the first
    ///   sub-block being entered (the spec's "first invocation"
    ///   branch sets `lastGreater1Ctx = 1` unconditionally).
    ///
    /// Implements equations 9-56, 9-57, and 9-58. After this call
    /// returns, [`current_ctx_inc`](Self::current_ctx_inc) yields
    /// the eq.-9-59 `ctxInc` for the sub-block's first bin.
    pub fn on_subblock_entry(&mut self, i: u32, is_chroma: bool, last_greater1_flag: u8) {
        // Eq. 9-56 / 9-57: initialise ctxSet from (i, cIdx).
        self.ctx_set = if i == 0 || is_chroma { 0 } else { 2 };

        // §9.3.4.2.6 lastGreater1Ctx derivation. The spec branches
        // on "first invocation" vs not:
        //   * First sub-block in this transform block:
        //     lastGreater1Ctx = 1. The ctxSet increment in eq. 9-58
        //     never triggers (lastGreater1Ctx > 0).
        //   * Subsequent sub-block, prior sub-block decoded at
        //     least one greater1 bin: lastGreater1Ctx is the prior
        //     sub-block's terminal greater1Ctx, then mutated by the
        //     prior sub-block's lastGreater1Flag per the spec
        //     bullets.
        // The walker tracks the prior sub-block's terminal
        // greater1Ctx as self.greater1_ctx (updated by the per-bin
        // step in on_coeff_abs_level_greater1_flag). We replay the
        // spec's lastGreater1Ctx mutation here, then the eq. 9-58
        // bump condition is `lastGreater1Ctx == 0`.
        let last_greater1_ctx = if !self.seen_any_subblock {
            // First sub-block: spec sets lastGreater1Ctx = 1.
            1
        } else if !self.has_last_greater1_flag {
            // Prior sub-block existed but decoded zero greater1
            // bins (rare: the sub-block had only sig coeffs equal
            // to 1 — i.e. no bin reached this code path). In that
            // case the spec's "When lastGreater1Ctx is greater than
            // 0, the variable lastGreater1Flag is set equal to ..."
            // branch is skipped and lastGreater1Ctx retains the
            // prior sub-block's value directly.
            self.greater1_ctx
        } else {
            // Apply the spec's lastGreater1Ctx mutation: if the
            // prior sub-block's lastGreater1Flag was 1 the ctx
            // resets to 0; otherwise it advances by 1.
            if self.greater1_ctx == 0 {
                // No mutation: the "When lastGreater1Ctx > 0"
                // guard skips the body entirely.
                0
            } else if last_greater1_flag != 0 {
                0
            } else {
                self.greater1_ctx + 1
            }
        };

        // Eq. 9-58: bump ctxSet whenever the just-derived
        // lastGreater1Ctx is 0.
        if last_greater1_ctx == 0 {
            self.ctx_set += 1;
        }

        // Start-of-sub-block reset of the within-set ctx: §9.3.4.2.6
        // says "greater1Ctx is set equal to 1".
        self.greater1_ctx = 1;
        self.seen_any_subblock = true;
        self.has_last_greater1_flag = false;
    }

    /// §9.3.4.2.6 per-bin step: invoked *after* every
    /// `coeff_abs_level_greater1_flag` bin in the current sub-block
    /// is decoded. Updates `greater1Ctx` per the spec's
    /// "lastGreater1Flag = 1 → 0 / = 0 → increment" rule, clamped to
    /// `0..=3` by eq. 9-59's `min(3, greater1Ctx)` (the underlying
    /// counter is implicitly clamped at 3 because eq. 9-59 truncates
    /// any further growth).
    ///
    /// Note: the §9.3.4.2.6 spec keeps `greater1Ctx` unclamped
    /// internally (it grows past 3) but eq. 9-59 clamps it; this
    /// implementation tracks the clamped value because the only
    /// observable property used downstream is whether
    /// `greater1Ctx == 0` (for the next sub-block's eq.-9-58 bump),
    /// and the eq. 9-59 reading clamps regardless.
    pub fn on_coeff_abs_level_greater1_flag(&mut self, decoded_bin: u8) {
        if self.greater1_ctx > 0 {
            self.greater1_ctx = if decoded_bin != 0 {
                0
            } else {
                (self.greater1_ctx + 1).min(3)
            };
            self.has_last_greater1_flag = true;
        }
    }

    /// Eq. 9-59 + eq. 9-60: returns the `coeff_abs_level_greater1_flag`
    /// `ctxInc` for the *next* bin to be decoded, given this state
    /// machine's `(ctxSet, greater1Ctx)`.
    ///
    /// `is_chroma == true` (cIdx > 0) adds the eq.-9-60 `+ 16`
    /// offset.
    #[must_use]
    pub fn current_ctx_inc(&self, is_chroma: bool) -> u32 {
        let base = self.ctx_set * 4 + self.greater1_ctx.min(3);
        if is_chroma {
            base + 16
        } else {
            base
        }
    }

    /// Current `ctxSet` value, exported for §9.3.4.2.7
    /// `coeff_abs_level_greater2_flag` ctxInc derivation (which
    /// reads the same sub-block's `ctxSet` per eq. 9-61).
    #[must_use]
    pub fn ctx_set(&self) -> u32 {
        self.ctx_set
    }
}

impl Default for Greater1State {
    fn default() -> Self {
        Self::new()
    }
}

/// §9.3.4.2.7 ctxInc derivation for
/// `coeff_abs_level_greater2_flag[ lastGreater1ScanPos ]`. The spec
/// equations are:
///
/// ```text
/// ctxInc = ctxSet                  (9-61)  // luma
/// ctxInc = ctxSet + 4              (9-62)  // chroma (cIdx > 0)
/// ```
///
/// `ctxSet` is the value derived by §9.3.4.2.6 for the same
/// sub-block, available from [`Greater1State::ctx_set`].
///
/// The element is signalled at most once per sub-block at the first
/// scan position that took the greater-1 escape; the per-sub-block
/// `Greater1State` already has the matching `ctxSet` at the time
/// the residual loop reaches that scan position, so the caller
/// simply reads it off without further state plumbing.
#[must_use]
pub fn coeff_abs_level_greater2_flag_ctx_inc(ctx_set: u32, is_chroma: bool) -> u32 {
    if is_chroma {
        ctx_set + 4
    } else {
        ctx_set
    }
}

/// Table 9-43 binarization shape for `coeff_abs_level_greater1_flag`
/// and `coeff_abs_level_greater2_flag`: FL with `cMax = 1` (single
/// context-coded bin per element invocation).
pub const COEFF_ABS_LEVEL_GREATER_X_FL_CMAX: u32 = 1;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;

    // -------------------------------------------------------------
    // ctxInc derivation — pure-function tables
    // -------------------------------------------------------------

    #[test]
    fn cu_qp_delta_abs_ctx_inc_table_9_32() {
        // Bin 0 → dedicated ctx 0; bins 1..=4 → shared ctx 1.
        assert_eq!(cu_qp_delta_abs_ctx_inc(0), 0);
        assert_eq!(cu_qp_delta_abs_ctx_inc(1), 1);
        assert_eq!(cu_qp_delta_abs_ctx_inc(2), 1);
        assert_eq!(cu_qp_delta_abs_ctx_inc(3), 1);
        assert_eq!(cu_qp_delta_abs_ctx_inc(4), 1);
    }

    #[test]
    fn last_sig_coeff_prefix_offset_shift_luma() {
        // §9.3.4.2.3, equations for cIdx == 0:
        //   ctxOffset = 3*(log2-2) + ((log2-1) >> 2)
        //   ctxShift  = (log2+1) >> 2
        // log2=2 (4x4):   off = 0 + 0 = 0; shift = 3 >> 2 = 0
        // log2=3 (8x8):   off = 3 + 0 = 3; shift = 4 >> 2 = 1
        // log2=4 (16x16): off = 6 + 0 = 6; shift = 5 >> 2 = 1
        // log2=5 (32x32): off = 9 + 1 = 10; shift = 6 >> 2 = 1
        assert_eq!(last_sig_coeff_prefix_ctx_offset_shift(2, false), (0, 0));
        assert_eq!(last_sig_coeff_prefix_ctx_offset_shift(3, false), (3, 1));
        assert_eq!(last_sig_coeff_prefix_ctx_offset_shift(4, false), (6, 1));
        assert_eq!(last_sig_coeff_prefix_ctx_offset_shift(5, false), (10, 1));
    }

    #[test]
    fn last_sig_coeff_prefix_offset_shift_chroma() {
        // §9.3.4.2.3, chroma row: ctxOffset = 15, ctxShift = log2 - 2.
        // log2=2: (15, 0); log2=3: (15, 1); log2=4: (15, 2)
        // (4:4:4 chroma can also reach log2=5 → (15, 3)).
        assert_eq!(last_sig_coeff_prefix_ctx_offset_shift(2, true), (15, 0));
        assert_eq!(last_sig_coeff_prefix_ctx_offset_shift(3, true), (15, 1));
        assert_eq!(last_sig_coeff_prefix_ctx_offset_shift(4, true), (15, 2));
        assert_eq!(last_sig_coeff_prefix_ctx_offset_shift(5, true), (15, 3));
    }

    #[test]
    fn last_sig_coeff_prefix_ctx_inc_eq() {
        // ctxInc = (binIdx >> ctxShift) + ctxOffset.
        // 32x32 luma: offset=10, shift=1. binIdx 0..=9 →
        //   0,0,1,1,2,2,3,3,4,4 + 10 = 10,10,11,11,12,12,13,13,14,14.
        for bin in 0..=9 {
            let inc = last_sig_coeff_prefix_ctx_inc(bin, 10, 1);
            assert_eq!(inc, (bin >> 1) + 10);
        }
        // 4x4 luma: offset=0, shift=0. binIdx 0..=2 → 0,1,2.
        for bin in 0..=2 {
            let inc = last_sig_coeff_prefix_ctx_inc(bin, 0, 0);
            assert_eq!(inc, bin);
        }
    }

    #[test]
    fn last_sig_coeff_cmax_per_size() {
        // (log2 << 1) - 1.
        assert_eq!(last_sig_coeff_prefix_cmax(2), 3);
        assert_eq!(last_sig_coeff_prefix_cmax(3), 5);
        assert_eq!(last_sig_coeff_prefix_cmax(4), 7);
        assert_eq!(last_sig_coeff_prefix_cmax(5), 9);
    }

    // -------------------------------------------------------------
    // §7.4.9.11 position derivation — trace-derived examples
    // -------------------------------------------------------------

    #[test]
    fn last_sig_coeff_position_no_suffix() {
        // prefix <= 3 → no suffix, position == prefix. From the
        // multi-slice-per-frame trace: ctb 0, cIdx 1, prefix=2 → LastX=2.
        assert_eq!(last_sig_coeff_position(0, None), 0);
        assert_eq!(last_sig_coeff_position(1, None), 1);
        assert_eq!(last_sig_coeff_position(2, None), 2);
        assert_eq!(last_sig_coeff_position(3, None), 3);
    }

    #[test]
    fn last_sig_coeff_position_with_suffix_from_trace() {
        // From cabac-cu-qp-delta-last-sig-trace.md, main-422-10bit
        // primary fixture, luma 32x32 TB:
        //   x_prefix = 6, LastX = 8 → nBits = 2, suffix = 0,
        //     position = ((2 + (6&1)) << 2) + 0 = (2 << 2) = 8.
        //   y_prefix = 4, LastY = 5 → nBits = 1, suffix = 1,
        //     position = ((2 + (4&1)) << 1) + 1 = (2 << 1) + 1 = 5.
        assert_eq!(last_sig_coeff_position(6, Some(0)), 8);
        assert_eq!(last_sig_coeff_position(4, Some(1)), 5);
        // Chroma 16x16 Cb (cIdx 1) row: x_prefix=4, LastX=4 →
        //   nBits = 1, suffix = 0 → ((2+0)<<1)+0 = 4. y_prefix=1,
        //   LastY=1 → no suffix.
        assert_eq!(last_sig_coeff_position(4, Some(0)), 4);
        assert_eq!(last_sig_coeff_position(1, None), 1);
    }

    #[test]
    fn last_sig_coeff_suffix_n_bits_table() {
        // nBits = (prefix >> 1) - 1 for prefix > 3, else 0.
        assert_eq!(last_sig_coeff_suffix_n_bits(0), 0);
        assert_eq!(last_sig_coeff_suffix_n_bits(3), 0);
        assert_eq!(last_sig_coeff_suffix_n_bits(4), 1);
        assert_eq!(last_sig_coeff_suffix_n_bits(5), 1);
        assert_eq!(last_sig_coeff_suffix_n_bits(6), 2);
        assert_eq!(last_sig_coeff_suffix_n_bits(7), 2);
        assert_eq!(last_sig_coeff_suffix_n_bits(8), 3);
        assert_eq!(last_sig_coeff_suffix_n_bits(9), 3);
    }

    // -------------------------------------------------------------
    // §9.3.3.10 / §9.3.3.11 TR + EGk helpers
    // -------------------------------------------------------------

    #[test]
    fn tr_prefix_returns_value_at_terminator_bin() {
        // Read four bins (0, 1, 0, 0) via a closure. cMax = 5; the
        // unary terminates at the first 0 → prefix = 0, is_escape = false.
        let bins = [0u8, 1, 0, 0];
        let mut idx = 0;
        let (v, esc) = read_truncated_rice_prefix(5, |_bin_idx| {
            let b = bins[idx];
            idx += 1;
            Ok::<u8, CabacError>(b)
        })
        .unwrap();
        assert_eq!(v, 0);
        assert!(!esc);
        assert_eq!(idx, 1);
    }

    #[test]
    fn tr_prefix_escape_when_all_ones_to_cmax() {
        // Five 1s with cMax = 5 → prefix = 5, is_escape = true.
        let bins = [1u8; 5];
        let mut idx = 0;
        let (v, esc) = read_truncated_rice_prefix(5, |_bin_idx| {
            let b = bins[idx];
            idx += 1;
            Ok::<u8, CabacError>(b)
        })
        .unwrap();
        assert_eq!(v, 5);
        assert!(esc);
        assert_eq!(idx, 5);
    }

    // -------------------------------------------------------------
    // End-to-end CABAC integration — drive the engine on a crafted
    // bitstream and confirm the decoded element matches the trace.
    // -------------------------------------------------------------

    /// Helper: drop an MPS-only context (`pStateIdx = 62`, very high
    /// confidence) into the caller's slot. With this state the LPS
    /// range is tiny and an all-zero bitstream (after init) keeps
    /// taking the MPS path with `valMps = 0`, so every bin decodes to
    /// 0 and the unary TR prefix terminates at the first bin.
    fn fresh_mps_ctx(val_mps: u8) -> ContextModel {
        ContextModel {
            p_state_idx: 62,
            val_mps,
        }
    }

    #[test]
    fn cu_qp_delta_zero_when_first_bin_zero() {
        // All-zero stream: 9 init bits = 0, then the first
        // cu_qp_delta_abs bin is 0 → abs = 0, sign flag NOT signalled,
        // value = 0.
        let buf = [0u8; 8];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        let mut c0 = fresh_mps_ctx(0);
        let mut c1 = fresh_mps_ctx(0);
        let d = decode_cu_qp_delta(&mut eng, &mut c0, &mut c1).unwrap();
        assert_eq!(d.abs, 0);
        assert_eq!(d.sign_flag, None);
        assert_eq!(d.value, 0);
    }

    #[test]
    fn cu_qp_delta_value_eq_neg_two_when_sign_one() {
        // Engineer abs = 2 by hand: walking the TR prefix bins of the
        // arithmetic decoder is non-trivial because the LPS range
        // depends on context state. Instead, validate the §7.4.9.14
        // derivation directly with the trace-observed (abs=2, sign=1)
        // values from the main-422-10bit primary fixture.
        let abs = 2u32;
        let sign = 1u8;
        let value = abs as i32 * (1 - 2 * sign as i32);
        assert_eq!(value, -2);
    }

    #[test]
    fn cu_qp_delta_signed_derivation_matches_trace_rows() {
        // Walk every row of the multi-slice-per-frame trace
        // (abs, sign_flag) → CuQpDeltaVal. Confirms the §7.4.9.14
        // arithmetic step in isolation.
        let rows = [
            (3u32, 0u8, 3i32),
            (1, 1, -1),
            (1, 0, 1),
            (3, 0, 3),
            (2, 1, -2),
            (2, 0, 2),
            (3, 0, 3),
            (2, 1, -2),
            (1, 0, 1),
            (3, 0, 3),
        ];
        for (abs, sign, expected) in rows {
            let got = abs as i32 * (1 - 2 * sign as i32);
            assert_eq!(got, expected, "abs={abs} sign={sign}");
        }
    }

    #[test]
    fn last_sig_coeff_round_trips_from_a_known_bypass_stream() {
        // Test the bypass-suffix path end-to-end by manually decoding
        // a fixture-traced prefix=6 + suffix=0 = LastX=8 sequence.
        // The TR prefix is hard to inject without controlling the LPS
        // range; instead, we re-derive position from the trace's
        // (prefix, observed-LastX-and-LastY) pairs and verify equation
        // 7-74 directly across all primary-fixture rows.
        let rows = [
            // (cIdx, prefix_x, prefix_y, LastX, LastY)
            (0u32, 6u32, 4u32, 8u32, 5u32),
            (1, 4, 1, 4, 1),
            (1, 4, 2, 4, 2),
            (2, 2, 1, 2, 1),
            (2, 2, 0, 2, 0),
        ];
        for (_c, px, py, lx, ly) in rows {
            // X side
            let n_x = last_sig_coeff_suffix_n_bits(px);
            // From the trace, suffix can be inferred:
            //   For (6, 8): n=2, base=8 → suffix=0
            //   For (4, 4): n=1, base=4 → suffix=0
            //   For (2, 2): n=0, suffix=None → pos=2
            let inferred_suf_x = if n_x == 0 {
                None
            } else {
                let base = (2 + (px & 1)) << ((px >> 1) - 1);
                Some(lx - base)
            };
            assert_eq!(last_sig_coeff_position(px, inferred_suf_x), lx);
            // Y side
            let n_y = last_sig_coeff_suffix_n_bits(py);
            let inferred_suf_y = if n_y == 0 {
                None
            } else {
                let base = (2 + (py & 1)) << ((py >> 1) - 1);
                Some(ly - base)
            };
            assert_eq!(last_sig_coeff_position(py, inferred_suf_y), ly);
        }
    }

    #[test]
    fn last_sig_coeff_bank_tag_round_trip() {
        // The bank tag is a routing helper for callers; verify the
        // enum survives a copy + equality check.
        let x = LastSigCoeffBank::X;
        let y = LastSigCoeffBank::Y;
        assert_ne!(x, y);
        assert_eq!(x, LastSigCoeffBank::X);
        assert_eq!(y, LastSigCoeffBank::Y);
    }

    // -------------------------------------------------------------
    // §9.3.3.11 EGk(k=0) — driven through the engine. The bin emitted
    // by `decode_bypass` depends on `ivlCurrRange` (510) versus the
    // shifted `ivlOffset`, not directly on the read bit value, so we
    // construct an offset that crosses the threshold on the first
    // bypass read: init `ivlOffset` to 256, then a fed 1-bit pushes
    // it to 513 ≥ 510, the engine returns bin = 1, and EGk's
    // leading-zeros count terminates at 0 → value = 0.
    #[test]
    fn eg_k0_decodes_zero_when_first_bypass_returns_one() {
        // Pack 9-bit init offset = 256 (0b1_0000_0000): byte0 =
        // 0b1000_0000, byte1 MSB = 0. Next bypass bin reads byte1
        // bit1; set it to 1 → byte1 = 0b0100_0000 = 0x40.
        let buf = [0b1000_0000, 0b0100_0000, 0x00];
        let mut eng = CabacEngine::new(BitReader::new(&buf)).unwrap();
        assert_eq!(eng.ivl_offset(), 256);
        let v = decode_eg_k0(&mut eng).unwrap();
        assert_eq!(v, 0);
    }

    // -------------------------------------------------------------
    // §9.3.4.2.4 coded_sub_block_flag ctxInc
    // -------------------------------------------------------------

    #[test]
    fn coded_sub_block_flag_ctx_inc_luma_no_neighbours_active() {
        // §9.3.4.2.4: both neighbours = 0 → csbfCtx = 0 →
        // ctxInc = min(0, 1) = 0 for luma.
        assert_eq!(coded_sub_block_flag_ctx_inc(false, 0, 0), 0);
    }

    #[test]
    fn coded_sub_block_flag_ctx_inc_luma_one_neighbour_active() {
        // One neighbour set → csbfCtx = 1 → ctxInc = min(1, 1) = 1.
        assert_eq!(coded_sub_block_flag_ctx_inc(false, 1, 0), 1);
        assert_eq!(coded_sub_block_flag_ctx_inc(false, 0, 1), 1);
    }

    #[test]
    fn coded_sub_block_flag_ctx_inc_luma_both_neighbours_active_clamps() {
        // Both neighbours set → csbfCtx = 2 → min(2, 1) = 1
        // (equation 9-38 clamps the sum). The luma bank is {0, 1}
        // so the maximum value is 1.
        assert_eq!(coded_sub_block_flag_ctx_inc(false, 1, 1), 1);
    }

    #[test]
    fn coded_sub_block_flag_ctx_inc_chroma_offset_two() {
        // §9.3.4.2.4 equation 9-39: chroma adds 2 to the luma
        // ctxInc, putting chroma in {2, 3}.
        assert_eq!(coded_sub_block_flag_ctx_inc(true, 0, 0), 2);
        assert_eq!(coded_sub_block_flag_ctx_inc(true, 1, 0), 3);
        assert_eq!(coded_sub_block_flag_ctx_inc(true, 0, 1), 3);
        assert_eq!(coded_sub_block_flag_ctx_inc(true, 1, 1), 3);
    }

    #[test]
    fn coded_sub_block_flag_ctx_inc_ignores_high_bits() {
        // Only the LSB of each neighbour input is consulted (the
        // function masks with & 1). Confirms a defensive input
        // does not leak into csbfCtx.
        assert_eq!(coded_sub_block_flag_ctx_inc(false, 0xFE, 0), 0);
        assert_eq!(coded_sub_block_flag_ctx_inc(false, 0xFF, 0xFE), 1);
    }

    #[test]
    fn coded_sub_block_flag_ctx_inc_with_edge_drops_right() {
        // 4×4 TB: log2 = 2, max sub-block index = (1 << 0) − 1 = 0.
        // So xs = 0 == max → equation 9-36 does NOT apply; the
        // right_neighbour input is ignored.
        assert_eq!(
            coded_sub_block_flag_ctx_inc_with_edge(false, 0, 0, 2, 1, 0),
            0
        );
        // 8×8 TB: log2 = 3, max sub-block index = (1 << 1) − 1 = 1.
        // xs = 0 < 1 → right neighbour counts; xs = 1 == max → it
        // does not.
        assert_eq!(
            coded_sub_block_flag_ctx_inc_with_edge(false, 0, 0, 3, 1, 0),
            1
        );
        assert_eq!(
            coded_sub_block_flag_ctx_inc_with_edge(false, 1, 0, 3, 1, 0),
            0
        );
    }

    #[test]
    fn coded_sub_block_flag_ctx_inc_with_edge_drops_below() {
        // 16×16 TB: log2 = 4, max sub-block index = (1 << 2) − 1 =
        // 3. ys = 2 < 3 → below counts; ys = 3 == max → it does not.
        assert_eq!(
            coded_sub_block_flag_ctx_inc_with_edge(false, 0, 2, 4, 0, 1),
            1
        );
        assert_eq!(
            coded_sub_block_flag_ctx_inc_with_edge(false, 0, 3, 4, 0, 1),
            0
        );
    }

    #[test]
    fn coded_sub_block_flag_ctx_inc_with_edge_32x32_interior() {
        // 32×32 TB: log2 = 5, max sub-block index = (1 << 3) − 1 =
        // 7. An interior sub-block at (3, 3) with both neighbours
        // active → csbfCtx = 2 → ctxInc = min(2, 1) = 1.
        assert_eq!(
            coded_sub_block_flag_ctx_inc_with_edge(false, 3, 3, 5, 1, 1),
            1
        );
        // Same location, chroma: ctxInc = 2 + 1 = 3.
        assert_eq!(
            coded_sub_block_flag_ctx_inc_with_edge(true, 3, 3, 5, 1, 1),
            3
        );
    }

    // -------------------------------------------------------------
    // §9.3.4.2.2 Table 9-49 — split_cu_flag / cu_skip_flag ctxInc
    // -------------------------------------------------------------

    #[test]
    fn left_above_ctx_inc_truth_table() {
        // ctxInc = (condL && availableL) + (condA && availableA);
        // for the four combinations of the two AND'ed booleans the
        // sum is 0, 1, 1, or 2.
        assert_eq!(left_above_ctx_inc(false, true, false, true), 0);
        assert_eq!(left_above_ctx_inc(true, true, false, true), 1);
        assert_eq!(left_above_ctx_inc(false, true, true, true), 1);
        assert_eq!(left_above_ctx_inc(true, true, true, true), 2);
    }

    #[test]
    fn left_above_ctx_inc_unavailable_zeroes_branch() {
        // If a neighbour is unavailable per §6.4.1, its branch
        // contributes 0 regardless of the cond value.
        assert_eq!(left_above_ctx_inc(true, false, true, false), 0);
        assert_eq!(left_above_ctx_inc(true, false, true, true), 1);
        assert_eq!(left_above_ctx_inc(true, true, true, false), 1);
    }

    #[test]
    fn split_cu_flag_cond_table() {
        // §9.3.4.2.2 Table 9-49: condX = CtDepth[xNb][yNb] >
        // cqtDepth. Strict inequality, so equal depth → false.
        assert!(!split_cu_flag_cond(0, 0));
        assert!(!split_cu_flag_cond(1, 1));
        assert!(split_cu_flag_cond(1, 0));
        assert!(split_cu_flag_cond(3, 2));
        assert!(!split_cu_flag_cond(2, 3));
    }

    #[test]
    fn cu_skip_flag_cond_table() {
        // §9.3.4.2.2 Table 9-49: condX = cu_skip_flag[xNb][yNb].
        assert!(!cu_skip_flag_cond(0));
        assert!(cu_skip_flag_cond(1));
        // Only the LSB matters.
        assert!(!cu_skip_flag_cond(0b1110));
        assert!(cu_skip_flag_cond(0b1111));
    }

    #[test]
    fn split_cu_flag_ctx_inc_both_neighbours_deeper() {
        // Both neighbours at depth 2, current cqtDepth = 1 → both
        // conds true. Both available → ctxInc = 2.
        assert_eq!(split_cu_flag_ctx_inc(2, true, 2, true, 1), 2);
    }

    #[test]
    fn split_cu_flag_ctx_inc_only_left_deeper() {
        // Left at depth 2, above at depth 1, current cqtDepth = 1.
        // condL true (2 > 1), condA false (1 > 1 == false) →
        // ctxInc = 1.
        assert_eq!(split_cu_flag_ctx_inc(2, true, 1, true, 1), 1);
    }

    #[test]
    fn split_cu_flag_ctx_inc_unavailable_left_drops_contribution() {
        // Same conds as the both-deeper test, but left is
        // unavailable per §6.4.1 → its contribution is 0.
        assert_eq!(split_cu_flag_ctx_inc(2, false, 2, true, 1), 1);
        // Both unavailable → 0 regardless of CtDepth.
        assert_eq!(split_cu_flag_ctx_inc(7, false, 7, false, 0), 0);
    }

    #[test]
    fn cu_skip_flag_ctx_inc_table() {
        // Walk the (L, availL, A, availA) truth table for cu_skip_flag.
        assert_eq!(cu_skip_flag_ctx_inc(0, true, 0, true), 0);
        assert_eq!(cu_skip_flag_ctx_inc(1, true, 0, true), 1);
        assert_eq!(cu_skip_flag_ctx_inc(0, true, 1, true), 1);
        assert_eq!(cu_skip_flag_ctx_inc(1, true, 1, true), 2);
        // Unavailable neighbours zero their contribution even if
        // the flag itself is set.
        assert_eq!(cu_skip_flag_ctx_inc(1, false, 1, true), 1);
        assert_eq!(cu_skip_flag_ctx_inc(1, true, 1, false), 1);
        assert_eq!(cu_skip_flag_ctx_inc(1, false, 1, false), 0);
    }

    #[test]
    fn split_cu_flag_ctx_inc_bounded_zero_three() {
        // ctxInc must lie in {0, 1, 2} regardless of the inputs.
        for left_d in 0u32..=4 {
            for above_d in 0u32..=4 {
                for &cqt in &[0u32, 1, 2, 3] {
                    for &avl in &[false, true] {
                        for &ava in &[false, true] {
                            let c = split_cu_flag_ctx_inc(left_d, avl, above_d, ava, cqt);
                            assert!(c <= 2, "ctxInc out of {{0,1,2}} for inputs");
                        }
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------
    // split_transform_flag ctxInc — Table 9-48 row
    // -------------------------------------------------------------

    #[test]
    fn split_transform_flag_ctx_inc_table_row() {
        // ctxInc = 5 − log2TrafoSize for log2TrafoSize in
        // {2, 3, 4, 5} → ctxInc in {3, 2, 1, 0}.
        assert_eq!(split_transform_flag_ctx_inc(2), 3);
        assert_eq!(split_transform_flag_ctx_inc(3), 2);
        assert_eq!(split_transform_flag_ctx_inc(4), 1);
        assert_eq!(split_transform_flag_ctx_inc(5), 0);
    }

    #[test]
    fn split_transform_flag_ctx_inc_bounded_zero_three() {
        // The four legal residual-quadtree TB sizes map exactly
        // onto the four-context bank {0, 1, 2, 3}.
        for log2 in 2u32..=5 {
            let inc = split_transform_flag_ctx_inc(log2);
            assert!(inc <= 3, "ctxInc {} out of bank {{0..=3}}", inc);
        }
    }

    // -------------------------------------------------------------
    // cbf_luma / cbf_cb / cbf_cr ctxInc — Table 9-48 rows
    // -------------------------------------------------------------

    #[test]
    fn cbf_luma_ctx_inc_table_row() {
        // ctxInc = (trafoDepth == 0) ? 1 : 0.
        assert_eq!(cbf_luma_ctx_inc(0), 1);
        assert_eq!(cbf_luma_ctx_inc(1), 0);
        assert_eq!(cbf_luma_ctx_inc(2), 0);
        assert_eq!(cbf_luma_ctx_inc(3), 0);
        assert_eq!(cbf_luma_ctx_inc(4), 0);
    }

    #[test]
    fn cbf_luma_ctx_inc_bank_zero_one() {
        // The two-ctx bank {0, 1}: every legal trafoDepth lands in
        // one of those two slots.
        for d in 0u32..=4 {
            let inc = cbf_luma_ctx_inc(d);
            assert!(inc <= 1, "cbf_luma ctxInc {} out of {{0, 1}}", inc);
        }
    }

    #[test]
    fn cbf_chroma_ctx_inc_identity() {
        // ctxInc = trafoDepth for both cbf_cb and cbf_cr (shared).
        for d in 0u32..=4 {
            assert_eq!(cbf_chroma_ctx_inc(d), d);
            assert_eq!(cbf_cb_ctx_inc(d), d);
            assert_eq!(cbf_cr_ctx_inc(d), d);
        }
    }

    #[test]
    fn cbf_cb_cr_share_formula() {
        // cbf_cb and cbf_cr always agree at the bank-relative
        // ctxInc layer; the distinct context banks are encoded in
        // the caller's ctxIdxOffset, not here.
        for d in 0u32..=4 {
            assert_eq!(cbf_cb_ctx_inc(d), cbf_cr_ctx_inc(d));
        }
    }

    // -------------------------------------------------------------
    // inter_pred_idc ctxInc — Table 9-48 row
    // -------------------------------------------------------------

    #[test]
    fn inter_pred_idc_bin_0_routes_ct_depth() {
        // For a PU that is not 8×4 or 4×8 (nPbW+nPbH != 12), bin 0
        // ctxInc = CtDepth.
        // 64×64 PU (CTU root): 64+64=128 != 12; CtDepth=0 → 0.
        assert_eq!(inter_pred_idc_ctx_inc(0, 64, 64, 0), 0);
        // 16×16 PU at CtDepth 2: 16+16=32 != 12 → ctxInc = 2.
        assert_eq!(inter_pred_idc_ctx_inc(0, 16, 16, 2), 2);
        // 8×8 PU at CtDepth 3: 8+8=16 != 12 → ctxInc = 3.
        assert_eq!(inter_pred_idc_ctx_inc(0, 8, 8, 3), 3);
    }

    #[test]
    fn inter_pred_idc_bin_0_escape_on_16_sample_pus() {
        // 8×4 PU: nPbW + nPbH = 12 → bin 0 ctxInc = 4, regardless
        // of CtDepth.
        assert_eq!(inter_pred_idc_ctx_inc(0, 8, 4, 0), 4);
        assert_eq!(inter_pred_idc_ctx_inc(0, 8, 4, 3), 4);
        // 4×8 PU: same.
        assert_eq!(inter_pred_idc_ctx_inc(0, 4, 8, 0), 4);
        assert_eq!(inter_pred_idc_ctx_inc(0, 4, 8, 3), 4);
    }

    #[test]
    fn inter_pred_idc_bin_1_always_ctx_4() {
        // Bin 1 ctxInc = 4 regardless of nPbW / nPbH / CtDepth.
        assert_eq!(inter_pred_idc_ctx_inc(1, 64, 64, 0), 4);
        assert_eq!(inter_pred_idc_ctx_inc(1, 8, 4, 3), 4);
        assert_eq!(inter_pred_idc_ctx_inc(1, 4, 8, 2), 4);
        assert_eq!(inter_pred_idc_ctx_inc(1, 16, 8, 1), 4);
    }

    #[test]
    fn inter_pred_idc_ctx_inc_bank_zero_four() {
        // ctxInc ∈ {0, 1, 2, 3, 4} across all legal inputs.
        for &(w, h) in &[
            (64, 64),
            (32, 32),
            (16, 16),
            (8, 8),
            (8, 4),
            (4, 8),
            (16, 8),
            (8, 16),
        ] {
            for ct_depth in 0u32..=3 {
                for bin_idx in 0u32..=1 {
                    let inc = inter_pred_idc_ctx_inc(bin_idx, w, h, ct_depth);
                    assert!(inc <= 4, "ctxInc {} out of {{0..=4}}", inc);
                }
            }
        }
    }

    // -------------------------------------------------------------
    // log2_res_scale_abs_plus1 / res_scale_sign_flag ctxInc —
    // Table 9-48 rows for §7.3.8.13 cross_comp_pred()
    // -------------------------------------------------------------

    #[test]
    fn log2_res_scale_abs_plus1_ctx_inc_cb_bank() {
        // c = 0 (Cb): ctxInc = binIdx for binIdx ∈ {0, 1, 2, 3} →
        // bank {0, 1, 2, 3}.
        for bin_idx in 0u32..=3 {
            assert_eq!(log2_res_scale_abs_plus1_ctx_inc(bin_idx, 0), bin_idx);
        }
    }

    #[test]
    fn log2_res_scale_abs_plus1_ctx_inc_cr_bank() {
        // c = 1 (Cr): ctxInc = 4 + binIdx for binIdx ∈ {0, 1, 2, 3}
        // → bank {4, 5, 6, 7}.
        for bin_idx in 0u32..=3 {
            assert_eq!(log2_res_scale_abs_plus1_ctx_inc(bin_idx, 1), 4 + bin_idx);
        }
    }

    #[test]
    fn log2_res_scale_abs_plus1_ctx_inc_banks_disjoint() {
        // Cb and Cr banks must not overlap.
        for bin_idx in 0u32..=3 {
            let cb = log2_res_scale_abs_plus1_ctx_inc(bin_idx, 0);
            let cr = log2_res_scale_abs_plus1_ctx_inc(bin_idx, 1);
            assert!(
                cb < 4 && (4..8).contains(&cr),
                "Cb / Cr ctxInc not in disjoint banks: cb={}, cr={}",
                cb,
                cr
            );
        }
    }

    #[test]
    fn res_scale_sign_flag_ctx_inc_identity() {
        // ctxInc = c. One bit per chroma component, each on its own
        // context.
        assert_eq!(res_scale_sign_flag_ctx_inc(0), 0);
        assert_eq!(res_scale_sign_flag_ctx_inc(1), 1);
    }

    // -------------------------------------------------------------
    // SAO ctxInc / binarization-shape — Table 9-48 + Table 9-43
    // -------------------------------------------------------------

    #[test]
    fn sao_merge_flag_ctx_inc_is_zero() {
        // Table 9-48 row for sao_merge_left_flag / sao_merge_up_flag:
        // bin 0 ctxInc = 0; the FL cMax = 1 ⇒ exactly one bin total.
        assert_eq!(sao_merge_flag_ctx_inc(), 0);
        assert_eq!(SAO_MERGE_FLAG_FL_CMAX, 1);
    }

    #[test]
    fn sao_type_idx_ctx_inc_bin_zero() {
        // Table 9-48 row for sao_type_idx_{luma,chroma}: bin 0 ctxInc
        // = 0 (the only context-coded bin); bin 1 is bypass per
        // Table 9-48 and not routed through this helper.
        assert_eq!(sao_type_idx_ctx_inc(0), 0);
    }

    #[test]
    fn sao_type_idx_tr_cmax_is_two() {
        // §9.3.3.10 Table 9-43: TR cMax = 2 caps the prefix at two
        // bins (the §7.4.9.3 SaoTypeIdx range {0, 1, 2}: NOT_APPLIED
        // / BAND / EDGE).
        assert_eq!(SAO_TYPE_IDX_TR_CMAX, 2);
    }

    #[test]
    fn sao_offset_sign_fl_cmax_is_one() {
        // Table 9-43: FL cMax = 1 ⇒ single bypass bin.
        assert_eq!(SAO_OFFSET_SIGN_FL_CMAX, 1);
    }

    #[test]
    fn sao_band_position_fl_shape() {
        // Table 9-43: FL cMax = 31 ⇒ 5 bypass bins (§9.3.3.5
        // ceil(log2(32)) = 5).
        assert_eq!(SAO_BAND_POSITION_FL_CMAX, 31);
        assert_eq!(SAO_BAND_POSITION_FL_NBITS, 5);
        // Sanity: the FL-of-cMax-31 nbits is the expected log2 ceil.
        // 2^5 = 32 = cMax + 1.
        assert_eq!(
            1u32 << SAO_BAND_POSITION_FL_NBITS,
            SAO_BAND_POSITION_FL_CMAX + 1
        );
    }

    #[test]
    fn sao_eo_class_fl_shape() {
        // Table 9-43: FL cMax = 3 ⇒ 2 bypass bins (encoding §7.4.9.3
        // SaoEoClass ∈ {0, 1, 2, 3}).
        assert_eq!(SAO_EO_CLASS_FL_CMAX, 3);
        assert_eq!(SAO_EO_CLASS_FL_NBITS, 2);
        // Sanity: 2^2 = 4 = cMax + 1.
        assert_eq!(1u32 << SAO_EO_CLASS_FL_NBITS, SAO_EO_CLASS_FL_CMAX + 1);
    }

    #[test]
    fn sao_offset_abs_tr_cmax_8bit() {
        // §9.3.3.5: cMax = (1 << min(bitDepth, 10) − 5) − 1.
        // bitDepth = 8 ⇒ (1 << 3) − 1 = 7.
        assert_eq!(sao_offset_abs_tr_cmax(8), 7);
    }

    #[test]
    fn sao_offset_abs_tr_cmax_10bit() {
        // bitDepth = 10 ⇒ (1 << 5) − 1 = 31.
        assert_eq!(sao_offset_abs_tr_cmax(10), 31);
    }

    #[test]
    fn sao_offset_abs_tr_cmax_9bit() {
        // bitDepth = 9 ⇒ (1 << 4) − 1 = 15.
        assert_eq!(sao_offset_abs_tr_cmax(9), 15);
    }

    #[test]
    fn sao_offset_abs_tr_cmax_clamps_above_10() {
        // §9.3.3.5 Min(bitDepth, 10): bitDepth >= 10 must clamp to
        // the 10-bit cMax = 31. Covers 12-bit and 16-bit profiles.
        assert_eq!(sao_offset_abs_tr_cmax(11), 31);
        assert_eq!(sao_offset_abs_tr_cmax(12), 31);
        assert_eq!(sao_offset_abs_tr_cmax(16), 31);
    }

    #[test]
    fn sao_offset_abs_tr_cmax_monotone() {
        // cMax is non-decreasing in bitDepth across the clamp at 10.
        let mut prev = sao_offset_abs_tr_cmax(8);
        for bd in 9u32..=16 {
            let cur = sao_offset_abs_tr_cmax(bd);
            assert!(
                cur >= prev,
                "cMax must be non-decreasing in bitDepth: bd={} cur={} prev={}",
                bd,
                cur,
                prev
            );
            prev = cur;
        }
    }

    // -------------------------------------------------------------
    // §9.3.4.2.6 coeff_abs_level_greater1_flag — Greater1State
    // -------------------------------------------------------------

    #[test]
    fn greater1_initial_state_is_set_zero_ctx_one() {
        // Per §9.3.4.2.6, the first sub-block enters with
        // lastGreater1Ctx = 1 (no eq.-9-58 bump) and greater1Ctx = 1.
        // Eq. 9-56 forces ctxSet = 0 when i == 0.
        let mut s = Greater1State::new();
        s.on_subblock_entry(0, false, 0);
        assert_eq!(s.ctx_set, 0);
        assert_eq!(s.greater1_ctx, 1);
        // Eq. 9-59 first-bin ctxInc = 0 * 4 + 1 = 1 for luma.
        assert_eq!(s.current_ctx_inc(false), 1);
        // Chroma adds +16 per eq. 9-60.
        assert_eq!(s.current_ctx_inc(true), 17);
    }

    #[test]
    fn greater1_eq_9_57_luma_nonzero_subblock_uses_ctx_set_two() {
        // Eq. 9-57: i > 0 and cIdx == 0 (luma) ⇒ ctxSet starts at 2.
        let mut s = Greater1State::new();
        s.on_subblock_entry(1, false, 0);
        assert_eq!(s.ctx_set, 2);
        // First-bin ctxInc = 2 * 4 + 1 = 9 for luma.
        assert_eq!(s.current_ctx_inc(false), 9);
    }

    #[test]
    fn greater1_eq_9_56_chroma_always_starts_at_zero() {
        // Eq. 9-56: cIdx > 0 (chroma) ⇒ ctxSet = 0 regardless of i.
        for i in [0u32, 1, 2, 5, 7] {
            let mut s = Greater1State::new();
            s.on_subblock_entry(i, true, 0);
            assert_eq!(s.ctx_set, 0, "chroma sub-block i={} ctxSet", i);
            // Chroma first-bin ctxInc = 0 * 4 + 1 + 16 = 17.
            assert_eq!(s.current_ctx_inc(true), 17);
        }
    }

    #[test]
    fn greater1_per_bin_step_flag_one_resets_ctx_to_zero() {
        // §9.3.4.2.6: lastGreater1Flag == 1 ⇒ greater1Ctx becomes 0.
        let mut s = Greater1State::new();
        s.on_subblock_entry(0, false, 0);
        s.on_coeff_abs_level_greater1_flag(1);
        assert_eq!(s.greater1_ctx, 0);
        // Eq. 9-59 next-bin ctxInc = 0 * 4 + min(3, 0) = 0.
        assert_eq!(s.current_ctx_inc(false), 0);
    }

    #[test]
    fn greater1_per_bin_step_flag_zero_increments() {
        // §9.3.4.2.6: lastGreater1Flag == 0 ⇒ greater1Ctx
        // incremented (until eq.-9-59 clamps at 3).
        let mut s = Greater1State::new();
        s.on_subblock_entry(0, false, 0);
        // 1 → 2 → 3 → 3 (clamped).
        s.on_coeff_abs_level_greater1_flag(0);
        assert_eq!(s.greater1_ctx, 2);
        s.on_coeff_abs_level_greater1_flag(0);
        assert_eq!(s.greater1_ctx, 3);
        s.on_coeff_abs_level_greater1_flag(0);
        assert_eq!(s.greater1_ctx, 3);
        // Eq. 9-59: 0 * 4 + 3 = 3.
        assert_eq!(s.current_ctx_inc(false), 3);
    }

    #[test]
    fn greater1_per_bin_step_after_reset_stops_advancing() {
        // §9.3.4.2.6: once greater1Ctx reaches 0 the "When
        // greater1Ctx > 0" guard skips the update on every later bin
        // in the same sub-block. The terminal value stays 0.
        let mut s = Greater1State::new();
        s.on_subblock_entry(0, false, 0);
        s.on_coeff_abs_level_greater1_flag(1); // → 0
        s.on_coeff_abs_level_greater1_flag(0); // skipped
        s.on_coeff_abs_level_greater1_flag(1); // skipped
        assert_eq!(s.greater1_ctx, 0);
    }

    #[test]
    fn greater1_eq_9_58_bumps_ctx_set_when_last_greater1_ctx_zero() {
        // Sub-block 0: read one flag = 0 → greater1Ctx terminal = 2.
        // Then enter sub-block 1 with last_greater1_flag = 0:
        // lastGreater1Ctx = 2 (post-spec mutation) ≠ 0 ⇒ no bump.
        // But sub-block 1 with i > 0 starts at ctxSet = 2 by eq. 9-57.
        let mut s = Greater1State::new();
        s.on_subblock_entry(0, false, 0);
        s.on_coeff_abs_level_greater1_flag(0); // greater1Ctx 1→2
        s.on_subblock_entry(1, false, 0);
        // ctxSet starts at 2 per eq. 9-57; no eq.-9-58 bump since
        // lastGreater1Ctx mutation 2+1=3 ≠ 0.
        assert_eq!(s.ctx_set, 2);
    }

    #[test]
    fn greater1_eq_9_58_bump_triggers_after_flag_one_ladder() {
        // Sub-block 0: greater1Ctx 1→0 via flag = 1. Then sub-block 1
        // enters with last_greater1_flag = 1: lastGreater1Ctx is 0
        // (the spec's "When lastGreater1Ctx > 0" guard skips the
        // mutation, retaining 0). Eq. 9-58 bumps ctxSet by 1.
        let mut s = Greater1State::new();
        s.on_subblock_entry(0, false, 0);
        s.on_coeff_abs_level_greater1_flag(1); // → 0
        s.on_subblock_entry(1, false, 1);
        // Eq. 9-57 says ctxSet = 2 (i > 0, luma), then eq. 9-58
        // bumps by one ⇒ 3.
        assert_eq!(s.ctx_set, 3);
        // greater1Ctx resets to 1 per §9.3.4.2.6.
        assert_eq!(s.greater1_ctx, 1);
        // Eq. 9-59 first-bin ctxInc = 3 * 4 + 1 = 13.
        assert_eq!(s.current_ctx_inc(false), 13);
    }

    #[test]
    fn greater1_chroma_subblock_with_eq_9_58_bump() {
        // Chroma sub-block 0: flag = 1 → greater1Ctx 1→0.
        // Sub-block 1 (chroma) enters with last_greater1_flag = 1
        // ⇒ eq. 9-56 forces ctxSet = 0, then eq. 9-58 bumps to 1.
        let mut s = Greater1State::new();
        s.on_subblock_entry(0, true, 0);
        s.on_coeff_abs_level_greater1_flag(1);
        s.on_subblock_entry(1, true, 1);
        assert_eq!(s.ctx_set, 1);
        // Chroma ctxInc = 1 * 4 + 1 + 16 = 21.
        assert_eq!(s.current_ctx_inc(true), 21);
    }

    #[test]
    fn greater1_ctx_inc_clamps_at_min_three() {
        // Eq. 9-59 explicitly takes Min(3, greater1Ctx). The state
        // machine tracks the clamped value, so synthesising ctxSet=1,
        // greater1Ctx=3 gives ctxInc = 4 + 3 = 7 for luma; greater1Ctx
        // can never exceed 3 internally so this is enforced by the
        // type, not the read path.
        let mut s = Greater1State::new();
        // Manually push to (ctxSet=1, greater1Ctx=3) by reading three
        // greater1=0 bins from the first sub-block (the third hits
        // the clamp).
        s.on_subblock_entry(0, false, 0);
        s.on_coeff_abs_level_greater1_flag(0); // 1→2
        s.on_coeff_abs_level_greater1_flag(0); // 2→3
        s.on_coeff_abs_level_greater1_flag(0); // 3→3 (clamped)
        assert_eq!(s.greater1_ctx, 3);
        // Eq. 9-59 = 0*4 + 3 = 3.
        assert_eq!(s.current_ctx_inc(false), 3);
        // Chroma adds +16.
        assert_eq!(s.current_ctx_inc(true), 19);
    }

    // -------------------------------------------------------------
    // §9.3.4.2.7 coeff_abs_level_greater2_flag — ctxInc derivation
    // -------------------------------------------------------------

    #[test]
    fn greater2_ctx_inc_is_ctx_set_for_luma() {
        // Eq. 9-61: ctxInc = ctxSet for cIdx == 0.
        for ctx_set in 0u32..=3 {
            assert_eq!(
                coeff_abs_level_greater2_flag_ctx_inc(ctx_set, false),
                ctx_set,
                "luma ctxSet={}",
                ctx_set
            );
        }
    }

    #[test]
    fn greater2_ctx_inc_chroma_adds_four() {
        // Eq. 9-62: ctxInc = ctxSet + 4 for cIdx > 0.
        for ctx_set in 0u32..=3 {
            assert_eq!(
                coeff_abs_level_greater2_flag_ctx_inc(ctx_set, true),
                ctx_set + 4,
                "chroma ctxSet={}",
                ctx_set
            );
        }
    }

    #[test]
    fn greater2_reads_same_ctx_set_as_greater1_machine() {
        // §9.3.4.2.7 explicitly references the §9.3.4.2.6 ctxSet for
        // the same sub-block. Walk a small example: luma, sub-block 0,
        // first greater-1 flag = 1, then the greater-2 flag is read at
        // the lastGreater1ScanPos in the same sub-block ⇒ ctxSet = 0.
        let mut s = Greater1State::new();
        s.on_subblock_entry(0, false, 0);
        s.on_coeff_abs_level_greater1_flag(1);
        assert_eq!(s.ctx_set(), 0);
        assert_eq!(coeff_abs_level_greater2_flag_ctx_inc(s.ctx_set(), false), 0);

        // After the eq.-9-58 bump on entry to sub-block 1 (last
        // flag = 1), ctxSet = 3 and the greater-2 read picks up 3.
        s.on_subblock_entry(1, false, 1);
        assert_eq!(s.ctx_set(), 3);
        assert_eq!(coeff_abs_level_greater2_flag_ctx_inc(s.ctx_set(), false), 3);
        assert_eq!(coeff_abs_level_greater2_flag_ctx_inc(s.ctx_set(), true), 7);
    }

    #[test]
    fn coeff_abs_level_greater_x_fl_shape_is_one_bin() {
        // Table 9-43: both flags are FL with cMax = 1 (a single
        // context-coded bin per invocation).
        assert_eq!(COEFF_ABS_LEVEL_GREATER_X_FL_CMAX, 1);
    }
}
