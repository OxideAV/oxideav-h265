//! HEVC CABAC entropy engine (§9.3.4).
//!
//! Implements:
//!
//! * **Per-context initialisation** (§9.3.4.2.1) — the `(pStateIdx, valMps)`
//!   kernel shared by every context variable.
//! * **Arithmetic decode engine** (§9.3.4.3) — `ivlCurrRange` /
//!   `ivlOffset`, regular-mode [`decode_bin`], bypass-mode [`decode_bypass`],
//!   and [`decode_terminate`] for end-of-slice detection.
//! * **State transition and rangeTabLps tables** from Tables 9-41 and 9-42.
//!
//! The engine is driven directly off a byte-aligned slice of the slice's
//! RBSP starting at the `slice_data()` byte position.

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

/// rangeTabLps[pStateIdx][σ], Table 9-52. Used by the regular decode path.
#[rustfmt::skip]
pub const RANGE_TAB_LPS: [[u8; 4]; 64] = [
    [128, 176, 208, 240], [128, 167, 197, 227], [128, 158, 187, 216], [123, 150, 178, 205],
    [116, 142, 169, 195], [111, 135, 160, 185], [105, 128, 152, 175], [100, 122, 144, 166],
    [ 95, 116, 137, 158], [ 90, 110, 130, 150], [ 85, 104, 123, 142], [ 81,  99, 117, 135],
    [ 77,  94, 111, 128], [ 73,  89, 105, 122], [ 69,  85, 100, 116], [ 66,  80,  95, 110],
    [ 62,  76,  90, 104], [ 59,  72,  86,  99], [ 56,  69,  81,  94], [ 53,  65,  77,  89],
    [ 51,  62,  73,  85], [ 48,  59,  69,  80], [ 46,  56,  66,  76], [ 43,  53,  63,  72],
    [ 41,  50,  59,  69], [ 39,  48,  56,  65], [ 37,  45,  54,  62], [ 35,  43,  51,  59],
    [ 33,  41,  48,  56], [ 32,  39,  46,  53], [ 30,  37,  43,  50], [ 29,  35,  41,  48],
    [ 27,  33,  39,  45], [ 26,  31,  37,  43], [ 24,  30,  35,  41], [ 23,  28,  33,  39],
    [ 22,  27,  32,  37], [ 21,  26,  30,  35], [ 20,  24,  29,  33], [ 19,  23,  27,  31],
    [ 18,  22,  26,  30], [ 17,  21,  25,  28], [ 16,  20,  23,  27], [ 15,  19,  22,  25],
    [ 14,  18,  21,  24], [ 14,  17,  20,  23], [ 13,  16,  19,  22], [ 12,  15,  18,  21],
    [ 12,  14,  17,  20], [ 11,  14,  16,  19], [ 11,  13,  15,  18], [ 10,  12,  15,  17],
    [ 10,  12,  14,  16], [  9,  11,  13,  15], [  9,  11,  12,  14], [  8,  10,  12,  14],
    [  8,   9,  11,  13], [  7,   9,  11,  12], [  7,   9,  10,  12], [  7,   8,  10,  11],
    [  6,   8,   9,  11], [  6,   7,   9,  10], [  6,   7,   8,   9], [  2,   2,   2,   2],
];

/// transIdxMps[pStateIdx], Table 9-53 (MPS branch). 63 has no successor; we
/// keep the fixed point at 62 for safety even though HEVC never uses
/// pStateIdx=63 in the MPS branch (it becomes the sticky terminator).
#[rustfmt::skip]
pub const TRANS_IDX_MPS: [u8; 64] = [
     1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
];

/// transIdxLps[pStateIdx], Table 9-53 (LPS branch).
#[rustfmt::skip]
pub const TRANS_IDX_LPS: [u8; 64] = [
     0,  0,  1,  2,  2,  4,  4,  5,  6,  7,  8,  9,  9, 11, 11, 12,
    13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21, 21, 22, 22, 23, 24,
    24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33,
    33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

/// The CABAC arithmetic decode engine (§9.3.4.3).
///
/// Consumes bits byte-MSB-first from `data` starting at `byte_off`. The
/// `ivlCurrRange` / `ivlOffset` pair and the residual-bits counter are
/// maintained per §9.3.4.3.
pub struct CabacEngine<'a> {
    data: &'a [u8],
    /// Next byte to consume from `data`.
    byte_pos: usize,
    /// Residual bits left in `bits_buf` (MSB-first).
    bits_in_buf: u32,
    bits_buf: u32,

    /// Current interval width, 9 bits effective (§9.3.4.3.2 step 2).
    ivl_curr_range: u32,
    /// Current offset into the interval, 9 bits effective.
    ivl_offset: u32,
}

impl<'a> CabacEngine<'a> {
    /// Initialise the arithmetic engine at a given byte offset (§9.3.4.1.1).
    /// `data` must be the slice's RBSP (emulation-prevention-stripped) and
    /// `byte_off` the byte-aligned start of `slice_data()`.
    pub fn new(data: &'a [u8], byte_off: usize) -> Self {
        let mut e = CabacEngine {
            data,
            byte_pos: byte_off,
            bits_in_buf: 0,
            bits_buf: 0,
            ivl_curr_range: 510,
            ivl_offset: 0,
        };
        // Pre-load 9 bits into ivl_offset.
        for _ in 0..9 {
            let b = e.read_bit();
            e.ivl_offset = (e.ivl_offset << 1) | b;
        }
        e
    }

    /// Read one bit from the underlying byte stream, MSB-first. Returns 0
    /// beyond EOF, which mirrors how the rbsp_trailing_bits tail lets the
    /// engine drain to the stop bit without erroring out.
    fn read_bit(&mut self) -> u32 {
        if self.bits_in_buf == 0 {
            if self.byte_pos < self.data.len() {
                self.bits_buf = self.data[self.byte_pos] as u32;
                self.byte_pos += 1;
                self.bits_in_buf = 8;
            } else {
                return 0;
            }
        }
        self.bits_in_buf -= 1;
        (self.bits_buf >> self.bits_in_buf) & 1
    }

    /// Current byte position (useful for tests / diagnostics).
    pub fn byte_pos(&self) -> usize {
        self.byte_pos
    }

    /// Re-initialise the arithmetic engine at a new byte position
    /// (§9.3.2.6). Called after a PCM CU consumes its payload: the
    /// engine's `ivl_curr_range` and `ivl_offset` must be re-seeded
    /// from the new byte position. Context variables are preserved
    /// per §9.3.1.2.
    pub fn reinit_at_byte(&mut self, byte_pos: usize) {
        self.byte_pos = byte_pos;
        self.bits_in_buf = 0;
        self.bits_buf = 0;
        self.ivl_curr_range = 510;
        self.ivl_offset = 0;
        for _ in 0..9 {
            let b = self.read_bit();
            self.ivl_offset = (self.ivl_offset << 1) | b;
        }
    }

    /// Raw byte slice of the engine's payload — needed by the PCM path
    /// to read `pcm_sample_luma/chroma` directly from the bitstream
    /// without going through the arithmetic decoder.
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Number of bits already drawn from the underlying bitstream into
    /// either the arithmetic interval or the `bits_buf` cache. The PCM
    /// path (§7.3.8.5) needs this so it can advance past
    /// `pcm_alignment_zero_bit` without double-counting buffered bits.
    pub fn bits_consumed(&self) -> u64 {
        (self.byte_pos as u64) * 8 - (self.bits_in_buf as u64)
    }

    /// Decode one regular (context-modelled) bin. Mutates `ctx` to advance
    /// the probability state (§9.3.4.3.2).
    pub fn decode_bin(&mut self, ctx: &mut CtxState) -> u32 {
        let rang_lps_idx = ((self.ivl_curr_range >> 6) & 3) as usize;
        let ivl_lps_range = RANGE_TAB_LPS[ctx.p_state_idx as usize][rang_lps_idx] as u32;
        self.ivl_curr_range -= ivl_lps_range;
        let bin_val;
        if self.ivl_offset >= self.ivl_curr_range {
            // LPS path.
            bin_val = (ctx.val_mps ^ 1) as u32;
            self.ivl_offset -= self.ivl_curr_range;
            self.ivl_curr_range = ivl_lps_range;
            if ctx.p_state_idx == 0 {
                ctx.val_mps ^= 1;
            }
            ctx.p_state_idx = TRANS_IDX_LPS[ctx.p_state_idx as usize];
        } else {
            // MPS path.
            bin_val = ctx.val_mps as u32;
            ctx.p_state_idx = TRANS_IDX_MPS[ctx.p_state_idx as usize];
        }
        // Renormalise (§9.3.4.3.5).
        while self.ivl_curr_range < 256 {
            self.ivl_curr_range <<= 1;
            self.ivl_offset = (self.ivl_offset << 1) | self.read_bit();
        }
        bin_val
    }

    /// Decode one bypass (equiprobable) bin (§9.3.4.3.4).
    pub fn decode_bypass(&mut self) -> u32 {
        self.ivl_offset = (self.ivl_offset << 1) | self.read_bit();
        if self.ivl_offset >= self.ivl_curr_range {
            self.ivl_offset -= self.ivl_curr_range;
            1
        } else {
            0
        }
    }

    /// Decode the terminating bin (§9.3.4.3.5). Returns 1 when the
    /// end-of-slice stop bit has been reached.
    pub fn decode_terminate(&mut self) -> u32 {
        self.ivl_curr_range -= 2;
        if self.ivl_offset >= self.ivl_curr_range {
            1
        } else {
            // Renormalise like the regular branch.
            while self.ivl_curr_range < 256 {
                self.ivl_curr_range <<= 1;
                self.ivl_offset = (self.ivl_offset << 1) | self.read_bit();
            }
            0
        }
    }
}

// ---------------------------------------------------------------------------
//  Context tables
//
//  Every context variable used by the I-slice pipeline is declared below.
//  The numbering matches H.265 §9.3.4.2.2 tables 9-11 … 9-32. Each table is
//  `[[u8; N]; 3]` indexed by initType (I=0, P=1, B=2) — for I-slice-only
//  decode we only consult the first row but we keep all three so the
//  declaration lines up with the spec page for easy audit.
//
//  Only the tables actually consumed by the I-slice CTU walk are present.
//  Inter-slice-only tables (merge_flag, inter_pred_idc, etc.) are
//  intentionally omitted — the decoder rejects inter slices up-front.
// ---------------------------------------------------------------------------

/// `split_cu_flag` (Table 9-11). Three contexts, one per depth class.
pub const SPLIT_CU_FLAG_INIT_VALUES: [[u8; 3]; 3] =
    [[139, 141, 157], [107, 139, 126], [107, 139, 126]];

/// `cu_transquant_bypass_flag` (Table 9-12). One context.
pub const CU_TRANSQUANT_BYPASS_FLAG_INIT_VALUES: [[u8; 1]; 3] = [[154], [154], [154]];

/// `cu_skip_flag` (Table 9-13). Three contexts — I-slice never signals
/// this syntax but we keep it to avoid a separate branch in init code.
pub const CU_SKIP_FLAG_INIT_VALUES: [[u8; 3]; 3] = [
    [154, 154, 154], // I: unused, kept at CNU per ffmpeg.
    [197, 185, 201],
    [197, 185, 201],
];

/// `pred_mode_flag` (Table 9-15). One context; I-slice doesn't read it
/// because CuPredMode is inferred to INTRA, but kept for completeness.
pub const PRED_MODE_FLAG_INIT_VALUES: [[u8; 1]; 3] = [[0], [149], [134]];

/// `part_mode` (Table 9-16). Four contexts.
pub const PART_MODE_INIT_VALUES: [[u8; 4]; 3] = [
    [184, 154, 154, 154],
    [154, 139, 154, 154],
    [154, 139, 154, 154],
];

/// `prev_intra_luma_pred_flag` (Table 9-17).
pub const PREV_INTRA_LUMA_PRED_FLAG_INIT_VALUES: [[u8; 1]; 3] = [[184], [154], [183]];

/// `intra_chroma_pred_mode` (Table 9-13). Single context.
pub const INTRA_CHROMA_PRED_MODE_INIT_VALUES: [[u8; 1]; 3] = [[63], [152], [152]];

/// `rqt_root_cbf` (Table 9-19) — I-slice doesn't read it but included.
pub const RQT_ROOT_CBF_INIT_VALUES: [[u8; 1]; 3] = [[0], [79], [79]];

/// `split_transform_flag` (Table 9-20). Three contexts.
pub const SPLIT_TRANSFORM_FLAG_INIT_VALUES: [[u8; 3]; 3] =
    [[153, 138, 138], [124, 138, 94], [224, 167, 122]];

/// `cbf_luma` (Table 9-21). Two contexts.
pub const CBF_LUMA_INIT_VALUES: [[u8; 2]; 3] = [[111, 141], [153, 111], [153, 111]];

/// `cbf_cb` / `cbf_cr` (Table 9-22). Five contexts per initType. `ctxInc ==
/// trafoDepth` per §9.3.4.2.1. Entries 4..4 extend the range required by
/// range-extension profiles; base-profile streams only use 0..3.
pub const CBF_CB_CR_INIT_VALUES: [[u8; 5]; 3] = [
    [94, 138, 182, 154, 154],
    [149, 107, 167, 154, 154],
    [149, 92, 167, 154, 154],
];

/// `cu_qp_delta_abs` (Table 9-22). Two contexts.
pub const CU_QP_DELTA_ABS_INIT_VALUES: [[u8; 2]; 3] = [[154, 154], [154, 154], [154, 154]];

/// `transform_skip_flag` for luma (Table 9-23).
pub const TRANSFORM_SKIP_FLAG_LUMA_INIT_VALUES: [[u8; 1]; 3] = [[139], [139], [139]];

/// `transform_skip_flag` for chroma (Table 9-23).
pub const TRANSFORM_SKIP_FLAG_CHROMA_INIT_VALUES: [[u8; 1]; 3] = [[139], [139], [139]];

/// `last_significant_coeff_x_prefix` (Table 9-25). 18 contexts.
#[rustfmt::skip]
pub const LAST_SIG_COEFF_X_PREFIX_INIT_VALUES: [[u8; 18]; 3] = [
    [110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111,  79, 108, 123,  63],
    [125, 110,  94, 110, 95,  79, 125, 111, 110,  78, 110, 111, 111,  95,  94, 108, 123, 108],
    [125, 110, 124, 110,  95,  94, 125, 111, 111,  79, 125, 126, 111, 111,  79, 108, 123,  93],
];

/// `last_significant_coeff_y_prefix` (Table 9-25). Same init-values layout.
#[rustfmt::skip]
pub const LAST_SIG_COEFF_Y_PREFIX_INIT_VALUES: [[u8; 18]; 3] = [
    [110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111,  79, 108, 123,  63],
    [125, 110,  94, 110, 95,  79, 125, 111, 110,  78, 110, 111, 111,  95,  94, 108, 123, 108],
    [125, 110, 124, 110,  95,  94, 125, 111, 111,  79, 125, 126, 111, 111,  79, 108, 123,  93],
];

/// `sig_coeff_flag` (Table 9-29). Per HEVC V11 the element has 42 main
/// contexts (ctxIdx 0..41) per initType plus 2 extras (ctxIdx 126..127
/// for initType 0 etc.). We store 44 per initType with the main block at
/// indices 0..41 matching spec positions {0..41} / {42..83} / {84..125}
/// and the extras at indices 42..43 matching {126..127} / {128..129} /
/// {130..131}. Only indices 0..41 are consulted by the current decoder.
#[rustfmt::skip]
pub const SIG_COEFF_FLAG_INIT_VALUES: [[u8; 44]; 3] = [
    [
        // initType 0 main (spec ctxIdx 0..41)
        111, 111, 125, 110, 110,  94, 124, 108, 124, 107, 125, 141,
        179, 153, 125, 107, 125, 141, 179, 153, 125, 107, 125, 141,
        179, 153, 125, 140, 139, 182, 182, 152, 136, 152, 136, 153,
        136, 139, 111, 136, 139, 111,
        // initType 0 extras (spec 126..127)
        141, 111,
    ],
    [
        // initType 1 main (spec ctxIdx 0..41 at flat positions 42..83)
        155, 154, 139, 153, 139, 123, 123,  63, 153, 166, 183, 140,
        136, 153, 154, 166, 183, 140, 136, 153, 154, 166, 183, 140,
        136, 153, 154, 170, 153, 123, 123, 107, 121, 107, 121, 167,
        151, 183, 140, 151, 183, 140,
        // initType 1 extras (spec 128..129)
        140, 140,
    ],
    [
        // initType 2 main (spec ctxIdx 0..41 at flat positions 84..125)
        170, 154, 139, 153, 139, 123, 123,  63, 124, 166, 183, 140,
        136, 153, 154, 166, 183, 140, 136, 153, 154, 166, 183, 140,
        136, 153, 154, 170, 153, 138, 138, 122, 121, 122, 121, 167,
        151, 183, 140, 151, 183, 140,
        // initType 2 extras (spec 130..131)
        140, 140,
    ],
];

/// `coeff_abs_level_greater1_flag` (Table 9-30). 24 contexts per initType.
#[rustfmt::skip]
pub const COEFF_ABS_GT1_INIT_VALUES: [[u8; 24]; 3] = [
    [140,  92, 137, 138, 140, 152, 138, 139, 153,  74, 149,  92,
     139, 107, 122, 152, 140, 179, 166, 182, 140, 227, 122, 197],
    [154, 196, 196, 167, 154, 152, 167, 182, 182, 134, 149, 136,
     153, 121, 136, 137, 169, 194, 166, 167, 154, 167, 137, 182],
    [154, 196, 167, 167, 154, 152, 167, 182, 182, 134, 149, 136,
     153, 121, 136, 122, 169, 208, 166, 167, 154, 152, 167, 182],
];

/// `coeff_abs_level_greater2_flag` (Table 9-28). 6 contexts.
pub const COEFF_ABS_GT2_INIT_VALUES: [[u8; 6]; 3] = [
    [138, 153, 136, 167, 152, 152],
    [107, 167, 91, 122, 107, 167],
    [107, 167, 91, 107, 107, 167],
];

/// `coded_sub_block_flag` (Table 9-29). 4 contexts.
pub const CODED_SUB_BLOCK_FLAG_INIT_VALUES: [[u8; 4]; 3] = [
    [91, 171, 134, 141],
    [121, 140, 61, 154],
    [121, 140, 61, 154],
];

/// `sao_merge_left_flag` / `sao_merge_up_flag` (Table 9-32) — 1 context, same
/// init across initTypes.
pub const SAO_MERGE_FLAG_INIT_VALUES: [[u8; 1]; 3] = [[153], [153], [153]];

/// `merge_flag` (Table 9-14). One context for initType=1/2; I-slice unused.
pub const MERGE_FLAG_INIT_VALUES: [[u8; 1]; 3] = [[0], [110], [154]];

/// `merge_idx` (Table 9-14). One context.
pub const MERGE_IDX_INIT_VALUES: [[u8; 1]; 3] = [[0], [122], [137]];

/// `inter_pred_idc` (Table 9-14). Five contexts.
pub const INTER_PRED_IDC_INIT_VALUES: [[u8; 5]; 3] =
    [[0, 0, 0, 0, 0], [95, 79, 63, 31, 31], [95, 79, 63, 31, 31]];

/// `ref_idx_l0` / `ref_idx_l1` (Table 9-14). Two contexts.
pub const REF_IDX_INIT_VALUES: [[u8; 2]; 3] = [[0, 0], [153, 153], [153, 153]];

/// `abs_mvd_greater0_flag` / `abs_mvd_greater1_flag` (Table 9-14). Two
/// contexts (one per greater-flag type).
pub const ABS_MVD_GREATER_FLAGS_INIT_VALUES: [[u8; 2]; 3] = [[0, 0], [140, 198], [169, 198]];

/// `mvp_l0_flag` / `mvp_l1_flag` (Table 9-14). One context.
pub const MVP_LX_FLAG_INIT_VALUES: [[u8; 1]; 3] = [[0], [168], [168]];

/// `sao_type_idx_luma` / `sao_type_idx_chroma` (Table 9-32) — 1 context.
pub const SAO_TYPE_IDX_INIT_VALUES: [[u8; 1]; 3] = [[200], [185], [160]];

/// Build a vector of [`CtxState`] from an initValues row and the slice
/// QP. Convenience wrapper that hides the per-initType indexing.
pub fn init_row<const N: usize>(
    table: &[[u8; N]; 3],
    init_type: InitType,
    slice_qp_y: i32,
) -> [CtxState; N] {
    let row = table[init_type as usize];
    let mut out = [CtxState::default(); N];
    for i in 0..N {
        out[i] = init_context(row[i], slice_qp_y);
    }
    out
}

/// Initialise the three `split_cu_flag` contexts for a given slice.
pub fn split_cu_flag_contexts(init_type: InitType, slice_qp_y: i32) -> [CtxState; 3] {
    init_row(&SPLIT_CU_FLAG_INIT_VALUES, init_type, slice_qp_y)
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
        let s = init_context(154, 26);
        assert_eq!(s.p_state_idx, 0);
        assert_eq!(s.val_mps, 1);

        let s = init_context(0, 0);
        assert_eq!(s.p_state_idx, 62);
        assert_eq!(s.val_mps, 0);

        let s = init_context(255, 51);
        assert_eq!(s.p_state_idx, 62);
        assert_eq!(s.val_mps, 1);
    }

    #[test]
    fn split_cu_flag_i_slice_qp26() {
        let ctx = split_cu_flag_contexts(InitType::I, 26);
        assert_eq!(ctx[0].val_mps, 0);
        assert_eq!(ctx[0].p_state_idx, 0);
        assert_eq!(ctx[1].val_mps, 1);
        assert_eq!(ctx[1].p_state_idx, 15);
        assert_eq!(ctx[2].val_mps, 1);
        assert_eq!(ctx[2].p_state_idx, 24);
    }

    #[test]
    fn init_type_for_slice() {
        assert_eq!(InitType::for_slice(true, false, false), InitType::I);
        assert_eq!(InitType::for_slice(true, false, true), InitType::I);
        assert_eq!(InitType::for_slice(false, false, false), InitType::Pa);
        assert_eq!(InitType::for_slice(false, false, true), InitType::Pb);
        assert_eq!(InitType::for_slice(false, true, false), InitType::Pb);
        assert_eq!(InitType::for_slice(false, true, true), InitType::Pa);
    }

    #[test]
    fn engine_bypass_matches_msb_read() {
        // For a brand-new engine (ivlCurrRange=510, ivlOffset = first 9 bits),
        // the bypass decode effectively reads one bit and compares against
        // ivl_curr_range. Construct a byte stream that feeds known bits.
        //
        // bytes: 0b11111111 0b00000000 ...
        let bytes = [0xFFu8, 0x00, 0x00, 0x00, 0x00];
        let mut e = CabacEngine::new(&bytes, 0);
        // Initial ivl_offset = top 9 bits = 0b1_1111_1111 = 511 > range=510.
        // First bypass shifts in bit 0 (next MSB of second byte) -> offset
        // becomes (511 << 1) | 0 = 1022; 1022 >= range*2? Actually, bypass
        // doubles the range implicitly: if offset >= range, output 1 and
        // subtract range. Here offset=1022 >= 510 -> 1, offset = 512.
        assert_eq!(e.decode_bypass(), 1);
    }
}
