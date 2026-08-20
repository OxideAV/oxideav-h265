//! §7.3.8.11 `residual_coding()` + §9.3 CABAC fuzz entry.
//!
//! The first three input bytes select the driver configuration; the
//! remainder feeds the §9.3.2.6/§9.3.4.3 arithmetic-decoding engine
//! whose bins drive the full coefficient-decode loop (last-significant
//! position, `coded_sub_block_flag`, `sig_coeff_flag`, the greater-1 /
//! greater-2 passes, sign bins, and the §9.3.3.11
//! `coeff_abs_level_remaining` bypass decode with Rice adaptation).
//!
//! Configuration mapping:
//!
//! * byte 0 bits 0..=1 — `log2TrafoSize - 2` (so 2..=5),
//! * byte 0 bit 2 — `cIdx > 0` (chroma),
//! * byte 0 bit 3 — `CuPredMode == MODE_INTRA` (feeds the §7.4.9.11
//!   scan-order derivation),
//! * byte 0 bit 4 — PPS `sign_data_hiding_enabled_flag`,
//! * byte 0 bit 5 — the §7.3.8.11 `signHidden = 0` force condition,
//! * byte 0 bits 6..=7 — `ChromaArrayType`,
//! * byte 1 bits 0..=5 — `predModeIntra` (0..=34 reachable) and,
//!   clamped to 51, `SliceQpY` for the context init,
//! * byte 1 bit 6 — the §9.3.4.2.5 transform-skip `sigCtx` gate,
//! * byte 2 — the uniform context `initValue`.
//!
//! The decode must return `Ok`/`Err` without panicking for every
//! input, conformant or not.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_h265::residual::{
    decode_residual_coding, residual_coding_scan_idx, ResidualCodingParams, ResidualContexts,
};
use oxideav_h265::{BitReader, CabacEngine};

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let cfg = data[0];
    let log2_trafo_size = 2 + u32::from(cfg & 0x03);
    let is_chroma = cfg & 0x04 != 0;
    let cu_pred_mode_is_intra = cfg & 0x08 != 0;
    let chroma_array_type = cfg >> 6;
    let pred_mode_intra = u32::from(data[1] & 0x3F);

    // §7.4.9.11 scan selection through the real derivation helper so
    // all three reachable scan orders are exercised.
    let scan_idx = residual_coding_scan_idx(
        cu_pred_mode_is_intra,
        log2_trafo_size,
        u8::from(is_chroma),
        chroma_array_type,
        pred_mode_intra,
    );

    let params = ResidualCodingParams {
        log2_trafo_size,
        is_chroma,
        scan_idx,
        sign_data_hiding_enabled_flag: cfg & 0x10 != 0,
        sign_hidden_suppressed: cfg & 0x20 != 0,
        transform_skip_sig_ctx: data[1] & 0x40 != 0,
        // Range-extension knobs driven from the same config bytes so
        // the persistent-Rice / alignment / extended-precision paths
        // are reachable too.
        persistent_rice_adaptation_enabled_flag: data[1] & 0x80 != 0,
        cabac_bypass_alignment_enabled_flag: data[2] & 0x01 != 0,
        extended_precision_processing_flag: data[2] & 0x02 != 0,
        bit_depth: if data[2] & 0x04 != 0 { 12 } else { 8 },
        rice_stat_transform_skip: data[2] & 0x08 != 0,
    };

    let slice_qp_y = i32::from(data[1] & 0x3F).min(51);
    let mut contexts = ResidualContexts::init_uniform(data[2], slice_qp_y);
    let Ok(mut engine) = CabacEngine::new(BitReader::new(&data[3..])) else {
        return;
    };
    let _ = decode_residual_coding(&mut engine, &mut contexts, &params);
});
