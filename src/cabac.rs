//! HEVC CABAC initialisation (§9.3.4.2).
//!
//! A full CABAC entropy decoder (arithmetic engine + context-managed binary
//! symbol decode) is ~2 KLOC and not yet implemented. This module lands
//! the piece that is self-contained and testable in isolation: the
//! *per-context initial state* derived from the `initValue` constants in
//! Table 9-4 through Table 9-32.
//!
//! ### Derivation (§9.3.4.2.1)
//!
//! For every context variable and every possible slice_qp_y / initType
//! combination, HEVC derives a pair `(pStateIdx, valMps)` from an 8-bit
//! initValue byte as follows:
//!
//! ```text
//!   slopeIdx     = initValue >> 4
//!   offsetIdx    = initValue & 15
//!   m            = slopeIdx * 5 - 45
//!   n            = (offsetIdx << 3) - 16
//!   preCtxState  = clip3(1, 126, ((m * clip3(0, 51, SliceQpY)) >> 4) + n)
//!   if preCtxState <= 63:
//!     pStateIdx = 63 - preCtxState
//!     valMps    = 0
//!   else:
//!     pStateIdx = preCtxState - 64
//!     valMps    = 1
//! ```
//!
//! This is a tiny but load-bearing arithmetic kernel: every HEVC CABAC
//! implementation runs it once per slice for every context variable in the
//! slice's initType set, and a bug here mis-seeds every subsequent binary
//! decode. Having it in isolation + unit-tested gives the future CABAC
//! engine a trusted foundation.
//!
//! ### What is *not* in this module yet
//!
//! * The full initValue tables (there are ~50 of them spanning hundreds of
//!   bytes). We include one exemplar — `split_cu_flag` (Table 9-11) — so we
//!   can exercise the derivation end-to-end. Adding the remaining tables is
//!   straightforward mechanical transcription from the spec; it's deferred
//!   until the arithmetic decode engine that consumes them lands.
//! * The arithmetic decode engine itself (ivlCurrRange / ivlOffset /
//!   renormalisation — §9.3.4.3).
//! * Context selection per syntax element (ctxIdx derivation tables in
//!   §9.3.4.2.X). Those are also mechanical once the engine exists.

/// Clip an integer value into the inclusive range `[lo, hi]`.
#[inline]
fn clip3(lo: i32, hi: i32, v: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// A CABAC context variable's internal state: an LPS-probability index and
/// the most-probable-symbol value. See §9.3.4.2.1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CtxState {
    pub p_state_idx: u8,
    pub val_mps: u8,
}

/// Derive the initial `(pStateIdx, valMps)` for one context variable from
/// its 8-bit `initValue` byte and the slice's `SliceQpY`. Implements
/// §9.3.4.2.1 verbatim.
pub fn init_context(init_value: u8, slice_qp_y: i32) -> CtxState {
    let slope_idx = (init_value >> 4) as i32;
    let offset_idx = (init_value & 0x0F) as i32;
    let m = slope_idx * 5 - 45;
    let n = (offset_idx << 3) - 16;
    let qp_clipped = clip3(0, 51, slice_qp_y);
    let pre = clip3(1, 126, ((m * qp_clipped) >> 4) + n);
    if pre <= 63 {
        CtxState {
            p_state_idx: (63 - pre) as u8,
            val_mps: 0,
        }
    } else {
        CtxState {
            p_state_idx: (pre - 64) as u8,
            val_mps: 1,
        }
    }
}

/// `initType` choice per slice (§9.3.4.2, step 2):
///
/// * I slice → 0
/// * P slice → `cabac_init_flag ? 2 : 1`
/// * B slice → `cabac_init_flag ? 1 : 2`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitType {
    I = 0,
    Pa = 1,
    Pb = 2,
}

impl InitType {
    /// Resolve the initType for a slice given its slice_type and the PPS-level
    /// `cabac_init_flag`.
    pub fn for_slice(is_i: bool, is_b: bool, cabac_init_flag: bool) -> Self {
        if is_i {
            InitType::I
        } else if is_b {
            if cabac_init_flag {
                InitType::Pa
            } else {
                InitType::Pb
            }
        } else if cabac_init_flag {
            InitType::Pb
        } else {
            InitType::Pa
        }
    }
}

/// `split_cu_flag` initValue constants (Table 9-11). Indexed as
/// `[initType][ctxIdx]` with three context indices (0..=2) per initType.
/// We expose these as an exemplar so the init kernel can be exercised
/// end-to-end; the remaining context tables (≈50 of them) will be added
/// alongside the arithmetic engine.
pub const SPLIT_CU_FLAG_INIT_VALUES: [[u8; 3]; 3] = [
    // initType 0 (I slice)
    [139, 141, 157],
    // initType 1
    [107, 139, 126],
    // initType 2
    [107, 139, 126],
];

/// Initialise the three `split_cu_flag` contexts for a given slice.
pub fn split_cu_flag_contexts(init_type: InitType, slice_qp_y: i32) -> [CtxState; 3] {
    let row = SPLIT_CU_FLAG_INIT_VALUES[init_type as usize];
    [
        init_context(row[0], slice_qp_y),
        init_context(row[1], slice_qp_y),
        init_context(row[2], slice_qp_y),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip3_bounds() {
        assert_eq!(clip3(1, 126, -5), 1);
        assert_eq!(clip3(1, 126, 200), 126);
        assert_eq!(clip3(1, 126, 50), 50);
    }

    #[test]
    fn init_context_matches_spec_corner_cases() {
        // initValue = 154 (slopeIdx=9, offsetIdx=10), qp=26 (mid-range).
        //   m = 9*5-45 = 0
        //   n = (10<<3)-16 = 64
        //   pre = clip3(1,126, (0*26>>4)+64) = 64
        //   pre > 63 => pStateIdx = 0, valMps = 1
        let s = init_context(154, 26);
        assert_eq!(s.p_state_idx, 0);
        assert_eq!(s.val_mps, 1);

        // initValue = 0 (slopeIdx=0, offsetIdx=0), qp=0 (min).
        //   m = -45, n = -16, pre = clip3(1,126, (0>>4)+-16) = 1
        //   pre <= 63 => pStateIdx = 62, valMps = 0
        let s = init_context(0, 0);
        assert_eq!(s.p_state_idx, 62);
        assert_eq!(s.val_mps, 0);

        // initValue=255, qp=51 (max positive slope * max qp).
        //   slope=15, off=15; m=30, n=104; (30*51)>>4 = 95; 95+104=199
        //   pre=clip3(1,126,199)=126 => pStateIdx=62, valMps=1
        let s = init_context(255, 51);
        assert_eq!(s.p_state_idx, 62);
        assert_eq!(s.val_mps, 1);
    }

    #[test]
    fn split_cu_flag_i_slice_qp26() {
        // For an I slice at QP=26 the three split_cu_flag contexts derive
        // from initValues {139, 141, 157}. Recompute by hand:
        //   139: slope=8, off=11 -> m=-5, n=72; pre=(-5*26>>4)+72 = -8-1+72=... be careful.
        //     (-5*26) = -130; -130>>4 arithmetic-shift = -9 (rounding toward -inf).
        //     pre = clip3(1,126, -9 + 72) = 63 -> pStateIdx=0, valMps=0
        let ctx = split_cu_flag_contexts(InitType::I, 26);
        assert_eq!(ctx[0].val_mps, 0);
        assert_eq!(ctx[0].p_state_idx, 0);
        // initValue 141: slope=8, off=13; m=-5, n=88;
        //   -130>>4=-9; pre=clip3(1,126,-9+88)=79; pre>63 -> pStateIdx=15, valMps=1
        assert_eq!(ctx[1].val_mps, 1);
        assert_eq!(ctx[1].p_state_idx, 15);
        // initValue 157: slope=9, off=13; m=0, n=88;
        //   pre=clip3(1,126,88) = 88 > 63 -> pStateIdx=24, valMps=1
        assert_eq!(ctx[2].val_mps, 1);
        assert_eq!(ctx[2].p_state_idx, 24);
    }

    #[test]
    fn init_type_for_slice() {
        assert_eq!(InitType::for_slice(true, false, false), InitType::I);
        assert_eq!(InitType::for_slice(true, false, true), InitType::I);
        // P slice (is_i=false, is_b=false)
        assert_eq!(InitType::for_slice(false, false, false), InitType::Pa);
        assert_eq!(InitType::for_slice(false, false, true), InitType::Pb);
        // B slice (is_i=false, is_b=true)
        assert_eq!(InitType::for_slice(false, true, false), InitType::Pb);
        assert_eq!(InitType::for_slice(false, true, true), InitType::Pa);
    }
}
