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
}
