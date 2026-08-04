//! §7.3.8.11 `residual_coding( )` *encoding* — the bin-exact dual of
//! [`crate::residual::decode_residual_coding_with`].
//!
//! Walks the coefficient array in the same reverse §6.5 scan the
//! decoder uses, emitting `last_sig_coeff_{x,y}_{prefix,suffix}`,
//! `coded_sub_block_flag`, `sig_coeff_flag`,
//! `coeff_abs_level_greater1_flag`, `coeff_abs_level_greater2_flag`,
//! `coeff_sign_flag` and `coeff_abs_level_remaining` through the
//! §9.3.5 arithmetic encoding engine, using the *same* §9.3.4.2
//! ctxInc derivation helpers as the decoder — so any block this
//! module emits decodes back to the identical `TransCoeffLevel`
//! array with identical context-state evolution (pinned by the
//! differential tests below).
//!
//! Scope bounds (the emitting encoder's fixed configuration, not
//! §7.3.8.11 limits): `sign_data_hiding_enabled_flag == 0` (a hidden
//! sign would require parity-forcing the quantized levels) and no
//! transform-skip / transquant-bypass residual rewrites.

use crate::binarization::{
    coded_sub_block_flag_ctx_inc_with_edge, coeff_abs_level_greater2_flag_ctx_inc,
    coeff_abs_level_remaining_c_max_eq_9_26, coeff_abs_level_remaining_c_rice_param_eq_9_24,
    last_sig_coeff_position, last_sig_coeff_prefix_cmax, last_sig_coeff_prefix_ctx_inc,
    last_sig_coeff_prefix_ctx_offset_shift, last_sig_coeff_suffix_n_bits,
    sig_coeff_flag_ctx_inc_from_sig_ctx, sig_coeff_flag_sig_ctx_dc, sig_coeff_flag_sig_ctx_general,
    sig_coeff_flag_sig_ctx_log2_2, sig_coeff_flag_sig_ctx_transform_skip, Greater1State,
};
use crate::cabac::ContextModel;
use crate::encoder::bitwriter::BitWriter;
use crate::encoder::cabac::CabacEncoder;
use crate::residual::{ResidualCodingParams, ResidualContexts, ResidualElement};
use crate::scan::{scan_order, ScanIdx};

/// Errors from [`encode_residual_coding`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualEncodeError {
    /// `log2TrafoSize` outside 2..=5.
    UnsupportedLog2TrafoSize(u32),
    /// A scan §7.4.9.11 never selects.
    UnsupportedScanIdx(ScanIdx),
    /// `levels.len() != (1 << log2TrafoSize)²`.
    LengthMismatch {
        /// Required level count.
        expected: usize,
        /// Supplied level count.
        got: usize,
    },
    /// Every level is zero — the caller must signal `cbf == 0`
    /// instead of invoking `residual_coding( )`.
    AllZero,
    /// A |level| exceeds the §7.4.9.11 CoeffMax bound for the
    /// non-extended-precision profiles (`2^15 − 1`).
    LevelOutOfRange(i32),
    /// `sign_data_hiding_enabled_flag == 1` — this encoder does not
    /// parity-force levels for hidden signs.
    SignHidingUnsupported,
}

impl core::fmt::Display for ResidualEncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedLog2TrafoSize(v) => {
                write!(f, "residual encode: log2TrafoSize {v} outside 2..=5")
            }
            Self::UnsupportedScanIdx(s) => {
                write!(f, "residual encode: scanIdx {s:?} not reachable")
            }
            Self::LengthMismatch { expected, got } => {
                write!(f, "residual encode: {got} levels, expected {expected}")
            }
            Self::AllZero => write!(f, "residual encode: all-zero block (cbf must be 0)"),
            Self::LevelOutOfRange(v) => write!(f, "residual encode: level {v} out of range"),
            Self::SignHidingUnsupported => {
                write!(f, "residual encode: sign data hiding not supported")
            }
        }
    }
}

impl std::error::Error for ResidualEncodeError {}

/// Bank dispatch — the encode dual of the decoder's
/// `EngineResidualBinSource` element→bank mapping.
fn bank(contexts: &mut ResidualContexts, element: ResidualElement) -> &mut [ContextModel] {
    match element {
        ResidualElement::LastSigCoeffXPrefix => &mut contexts.last_sig_coeff_x_prefix,
        ResidualElement::LastSigCoeffYPrefix => &mut contexts.last_sig_coeff_y_prefix,
        ResidualElement::CodedSubBlockFlag => &mut contexts.coded_sub_block_flag,
        ResidualElement::SigCoeffFlag => &mut contexts.sig_coeff_flag,
        ResidualElement::CoeffAbsLevelGreater1Flag => &mut contexts.coeff_abs_level_greater1_flag,
        ResidualElement::CoeffAbsLevelGreater2Flag => &mut contexts.coeff_abs_level_greater2_flag,
    }
}

/// Invert §7.4.9.11 eqs. 7-74..7-77: split a `LastSignificantCoeff*`
/// position into its `(prefix, suffix)` wire pair.
fn split_last_sig_position(v: u32, c_max: u32) -> (u32, Option<(u32, u32)>) {
    for prefix in 0..=c_max {
        let n_bits = last_sig_coeff_suffix_n_bits(prefix);
        if n_bits == 0 {
            if last_sig_coeff_position(prefix, None) == v {
                return (prefix, None);
            }
        } else {
            let base = last_sig_coeff_position(prefix, Some(0));
            if v >= base && v < base + (1u32 << n_bits) {
                return (prefix, Some((v - base, n_bits)));
            }
        }
    }
    unreachable!("last-sig position exceeds the TB (caller bounds it)")
}

/// Emit one `last_sig_coeff_*_prefix` as its §9.3.3.2 TR bin string
/// (unary ones, a terminating zero unless `prefix == cMax`), each bin
/// context-coded with the §9.3.4.2.3 ctxInc.
fn encode_last_sig_prefix(
    w: &mut BitWriter,
    cabac: &mut CabacEncoder,
    contexts: &mut ResidualContexts,
    element: ResidualElement,
    log2_trafo_size: u32,
    is_chroma: bool,
    prefix: u32,
) {
    let c_max = last_sig_coeff_prefix_cmax(log2_trafo_size);
    let (ctx_offset, ctx_shift) =
        last_sig_coeff_prefix_ctx_offset_shift(log2_trafo_size, is_chroma);
    for bin_idx in 0..prefix {
        let inc = last_sig_coeff_prefix_ctx_inc(bin_idx, ctx_offset, ctx_shift);
        cabac.encode_decision(w, &mut bank(contexts, element)[inc as usize], 1);
    }
    if prefix < c_max {
        let inc = last_sig_coeff_prefix_ctx_inc(prefix, ctx_offset, ctx_shift);
        cabac.encode_decision(w, &mut bank(contexts, element)[inc as usize], 0);
    }
}

/// §9.3.3.11 — emit one `coeff_abs_level_remaining` value as the TR
/// prefix (escape at 4 ones) + conditional EGk(`cRiceParam + 1`)
/// suffix, all bypass-coded. The bin-exact dual of
/// [`crate::binarization::decode_coeff_abs_level_remaining_with`].
fn encode_coeff_abs_level_remaining(
    w: &mut BitWriter,
    cabac: &mut CabacEncoder,
    c_rice_param: u32,
    value: u32,
) {
    let c_max = coeff_abs_level_remaining_c_max_eq_9_26(c_rice_param);
    if value < c_max {
        // TR shape: `value >> cRiceParam` ones + terminating zero +
        // `cRiceParam` suffix bits.
        let prefix_len = value >> c_rice_param;
        for _ in 0..prefix_len {
            cabac.encode_bypass(w, 1);
        }
        cabac.encode_bypass(w, 0);
        if c_rice_param > 0 {
            cabac.encode_bypass_bits(w, value & ((1 << c_rice_param) - 1), c_rice_param as u8);
        }
        return;
    }
    // Escape: 4 ones (no terminator), then EGk with k = cRiceParam+1
    // over `value − cMax`: `prefix_ones` ones + one zero +
    // `prefix_ones + k` suffix bits of `v' − ((1 << prefix_ones) − 1)
    // << k`.
    for _ in 0..4 {
        cabac.encode_bypass(w, 1);
    }
    let k = c_rice_param + 1;
    let v = value - c_max;
    let mut prefix_ones = 0u32;
    while (((1u64 << (prefix_ones + 1)) - 1) << k) <= u64::from(v) {
        prefix_ones += 1;
    }
    let base = ((1u64 << prefix_ones) - 1) << k;
    let suffix = u64::from(v) - base;
    for _ in 0..prefix_ones {
        cabac.encode_bypass(w, 1);
    }
    cabac.encode_bypass(w, 0);
    let n_bits = prefix_ones + k;
    for i in (0..n_bits).rev() {
        cabac.encode_bypass(w, ((suffix >> i) & 1) as u8);
    }
}

/// §7.3.8.11 — encode one `residual_coding( )` body from the
/// row-major `TransCoeffLevel` array (`levels[yC * size + xC]`),
/// following the decoder's control flow exactly (same inference
/// rules, same ctxInc derivations, same §9.3.4.2.6 greater-1 state
/// machine).
///
/// # Errors
/// [`ResidualEncodeError`] on out-of-scope parameters or an all-zero
/// / out-of-range level array.
pub fn encode_residual_coding(
    w: &mut BitWriter,
    cabac: &mut CabacEncoder,
    contexts: &mut ResidualContexts,
    params: &ResidualCodingParams,
    levels: &[i32],
) -> Result<(), ResidualEncodeError> {
    let log2 = params.log2_trafo_size;
    if !(2..=5).contains(&log2) {
        return Err(ResidualEncodeError::UnsupportedLog2TrafoSize(log2));
    }
    let scan_idx_num = u32::from(params.scan_idx.index());
    if scan_idx_num > 2 {
        return Err(ResidualEncodeError::UnsupportedScanIdx(params.scan_idx));
    }
    if params.sign_data_hiding_enabled_flag {
        return Err(ResidualEncodeError::SignHidingUnsupported);
    }
    let size = 1usize << log2;
    if levels.len() != size * size {
        return Err(ResidualEncodeError::LengthMismatch {
            expected: size * size,
            got: levels.len(),
        });
    }
    if let Some(&bad) = levels.iter().find(|&&v| v.unsigned_abs() > 0x7FFF) {
        return Err(ResidualEncodeError::LevelOutOfRange(bad));
    }
    let is_chroma = params.is_chroma;

    let pos_scan = scan_order(2, params.scan_idx)
        .map_err(|_| ResidualEncodeError::UnsupportedScanIdx(params.scan_idx))?;
    let sub_scan = scan_order((log2 - 2) as u8, params.scan_idx)
        .map_err(|_| ResidualEncodeError::UnsupportedScanIdx(params.scan_idx))?;
    let num_sb_1d = 1usize << (log2 - 2);

    // Locate the last significant coefficient in scan order.
    let coeff_at = |sb_i: usize, n: usize| -> i32 {
        let sb = sub_scan[sb_i];
        let xc = ((sb.x as usize) << 2) + pos_scan[n].x as usize;
        let yc = ((sb.y as usize) << 2) + pos_scan[n].y as usize;
        levels[yc * size + xc]
    };
    let mut last_sub_block: i32 = -1;
    let mut last_scan_pos: i32 = -1;
    for i in 0..num_sb_1d * num_sb_1d {
        for n in 0..16 {
            if coeff_at(i, n) != 0 {
                last_sub_block = i as i32;
                last_scan_pos = n as i32;
            }
        }
    }
    if last_sub_block < 0 {
        return Err(ResidualEncodeError::AllZero);
    }

    // §7.4.9.11 LastSignificantCoeff{X,Y} (+ the eq.-7-78 swap for the
    // vertical scan), split into prefix/suffix wire pairs.
    let sb = sub_scan[last_sub_block as usize];
    let last_x = ((sb.x as u32) << 2) + u32::from(pos_scan[last_scan_pos as usize].x);
    let last_y = ((sb.y as u32) << 2) + u32::from(pos_scan[last_scan_pos as usize].y);
    let (wire_x, wire_y) = if params.scan_idx == ScanIdx::Vertical {
        (last_y, last_x)
    } else {
        (last_x, last_y)
    };
    let c_max = last_sig_coeff_prefix_cmax(log2);
    let (prefix_x, suffix_x) = split_last_sig_position(wire_x, c_max);
    let (prefix_y, suffix_y) = split_last_sig_position(wire_y, c_max);
    // §7.3.8.11 bin order: both context-coded prefixes, then both
    // bypass suffixes.
    encode_last_sig_prefix(
        w,
        cabac,
        contexts,
        ResidualElement::LastSigCoeffXPrefix,
        log2,
        is_chroma,
        prefix_x,
    );
    encode_last_sig_prefix(
        w,
        cabac,
        contexts,
        ResidualElement::LastSigCoeffYPrefix,
        log2,
        is_chroma,
        prefix_y,
    );
    if let Some((suffix, n_bits)) = suffix_x {
        cabac.encode_bypass_bits(w, suffix, n_bits as u8);
    }
    if let Some((suffix, n_bits)) = suffix_y {
        cabac.encode_bypass_bits(w, suffix, n_bits as u8);
    }

    // coded_sub_block_flag grid, filled progressively in reverse scan
    // order exactly as the decoder builds it.
    let mut csbf = vec![0u8; num_sb_1d * num_sb_1d];
    let csbf_at = |grid: &[u8], xs: usize, ys: usize| -> u8 {
        if xs < num_sb_1d && ys < num_sb_1d {
            grid[ys * num_sb_1d + xs]
        } else {
            0
        }
    };

    let mut g1_state = Greater1State::new();
    let mut last_g1_bin: u8 = 0;

    for i in (0..=last_sub_block).rev() {
        let sb = sub_scan[i as usize];
        let (xs, ys) = (u32::from(sb.x), u32::from(sb.y));
        let is_last_sb = i == last_sub_block;
        let any_nonzero = (0..16).any(|n| coeff_at(i as usize, n) != 0);

        // coded_sub_block_flag: coded for 0 < i < lastSubBlock,
        // inferred 1 otherwise.
        let mut infer_sb_dc_sig = false;
        let sb_coded: u8 = if i < last_sub_block && i > 0 {
            let right = csbf_at(&csbf, xs as usize + 1, ys as usize);
            let below = csbf_at(&csbf, xs as usize, ys as usize + 1);
            let ctx_inc =
                coded_sub_block_flag_ctx_inc_with_edge(is_chroma, xs, ys, log2, right, below);
            let bin = u8::from(any_nonzero);
            cabac.encode_decision(
                w,
                &mut bank(contexts, ResidualElement::CodedSubBlockFlag)[ctx_inc as usize],
                bin,
            );
            infer_sb_dc_sig = true;
            bin
        } else {
            1
        };
        csbf[ys as usize * num_sb_1d + xs as usize] = sb_coded;
        if sb_coded == 0 {
            continue;
        }

        // sig_coeff_flag pass (same coding/inference split as decode).
        let mut sig = [0u8; 16];
        if is_last_sb {
            sig[last_scan_pos as usize] = 1;
        }
        let start_n: i32 = if is_last_sb { last_scan_pos - 1 } else { 15 };
        for n in (0..=start_n).rev() {
            let significant = u8::from(coeff_at(i as usize, n as usize) != 0);
            let xc = (xs << 2) + u32::from(pos_scan[n as usize].x);
            let yc = (ys << 2) + u32::from(pos_scan[n as usize].y);
            if n > 0 || !infer_sb_dc_sig {
                let sig_ctx = if params.transform_skip_sig_ctx {
                    sig_coeff_flag_sig_ctx_transform_skip(is_chroma)
                } else if log2 == 2 {
                    sig_coeff_flag_sig_ctx_log2_2(xc & 3, yc & 3)
                } else if xc + yc == 0 {
                    sig_coeff_flag_sig_ctx_dc(is_chroma, log2, scan_idx_num)
                } else {
                    let right = csbf_at(&csbf, xs as usize + 1, ys as usize);
                    let below = csbf_at(&csbf, xs as usize, ys as usize + 1);
                    sig_coeff_flag_sig_ctx_general(
                        is_chroma,
                        log2,
                        xc,
                        yc,
                        xs,
                        ys,
                        right,
                        below,
                        scan_idx_num,
                    )
                };
                let ctx_inc = sig_coeff_flag_ctx_inc_from_sig_ctx(sig_ctx, is_chroma);
                cabac.encode_decision(
                    w,
                    &mut bank(contexts, ResidualElement::SigCoeffFlag)[ctx_inc as usize],
                    significant,
                );
                sig[n as usize] = significant;
                if significant == 1 {
                    infer_sb_dc_sig = false;
                }
            } else {
                // n == 0 with the DC inference alive: the decoder
                // infers sig[0] = 1; a conforming encoder only reaches
                // here when the DC coefficient IS significant (a coded
                // sub-block cannot be all-zero).
                debug_assert_eq!(significant, 1, "coded sub-block with only a zero DC");
                sig[0] = 1;
            }
        }

        // greater-1 pass: the first 8 significant positions carry a
        // coded flag; the §9.3.4.2.6 state machine advances on the
        // encoder's own bins.
        let mut first_sig_scan_pos: i32 = 16;
        let mut last_sig_scan_pos: i32 = -1;
        let mut num_greater1: u32 = 0;
        let mut last_greater1_scan_pos: i32 = -1;
        let mut g1 = [0u8; 16];
        let mut entered_subblock = false;
        for n in (0..16usize).rev() {
            if sig[n] == 1 {
                let abs = coeff_at(i as usize, n).unsigned_abs();
                if num_greater1 < 8 {
                    if !entered_subblock {
                        g1_state.on_subblock_entry(i as u32, is_chroma, last_g1_bin);
                        entered_subblock = true;
                    }
                    let ctx_inc = g1_state.current_ctx_inc(is_chroma);
                    let bin = u8::from(abs > 1);
                    cabac.encode_decision(
                        w,
                        &mut bank(contexts, ResidualElement::CoeffAbsLevelGreater1Flag)
                            [ctx_inc as usize],
                        bin,
                    );
                    g1_state.on_coeff_abs_level_greater1_flag(bin);
                    last_g1_bin = bin;
                    g1[n] = bin;
                    num_greater1 += 1;
                    if bin == 1 && last_greater1_scan_pos == -1 {
                        last_greater1_scan_pos = n as i32;
                    }
                }
                if last_sig_scan_pos == -1 {
                    last_sig_scan_pos = n as i32;
                }
                first_sig_scan_pos = n as i32;
            }
        }
        let _ = (first_sig_scan_pos, last_sig_scan_pos); // sign hiding off

        // greater-2 flag — at most once per sub-block.
        let mut g2 = [0u8; 16];
        if last_greater1_scan_pos != -1 {
            let ctx_inc = coeff_abs_level_greater2_flag_ctx_inc(g1_state.ctx_set(), is_chroma);
            let abs = coeff_at(i as usize, last_greater1_scan_pos as usize).unsigned_abs();
            let bin = u8::from(abs > 2);
            cabac.encode_decision(
                w,
                &mut bank(contexts, ResidualElement::CoeffAbsLevelGreater2Flag)[ctx_inc as usize],
                bin,
            );
            g2[last_greater1_scan_pos as usize] = bin;
        }

        // coeff_sign_flag pass (bypass; sign hiding disabled).
        for n in (0..16usize).rev() {
            if sig[n] == 1 {
                let sign = u8::from(coeff_at(i as usize, n) < 0);
                cabac.encode_bypass(w, sign);
            }
        }

        // level pass: coeff_abs_level_remaining with the §9.3.3.11
        // per-sub-block Rice adaptation.
        let mut num_sig_coeff: u32 = 0;
        let mut c_last_abs_level: u32 = 0;
        let mut c_last_rice_param: u32 = 0;
        for n in (0..16usize).rev() {
            if sig[n] != 1 {
                continue;
            }
            let abs = coeff_at(i as usize, n).unsigned_abs();
            let base_level = 1 + u32::from(g1[n]) + u32::from(g2[n]);
            let threshold = if num_sig_coeff < 8 {
                if n as i32 == last_greater1_scan_pos {
                    3
                } else {
                    2
                }
            } else {
                1
            };
            if base_level == threshold {
                let remaining = abs - base_level;
                let c_rice_param = coeff_abs_level_remaining_c_rice_param_eq_9_24(
                    c_last_abs_level,
                    c_last_rice_param,
                );
                encode_coeff_abs_level_remaining(w, cabac, c_rice_param, remaining);
                c_last_abs_level = abs;
                c_last_rice_param = c_rice_param;
            } else {
                // The flag pass fully determines the level here.
                debug_assert_eq!(abs, base_level, "level not representable without remaining");
            }
            num_sig_coeff += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;
    use crate::cabac::CabacEngine;
    use crate::residual::{decode_residual_coding, ResidualContexts};

    /// Differential harness: encode `levels` (row-major), terminate,
    /// then decode with the crate's §7.3.8.11 decoder from identically
    /// initialized contexts — levels AND context states must match.
    fn roundtrip(params: &ResidualCodingParams, levels: &[i32]) {
        let mut w = BitWriter::new();
        let mut cabac = CabacEncoder::new();
        let mut enc_ctx = ResidualContexts::init(0, 26);
        encode_residual_coding(&mut w, &mut cabac, &mut enc_ctx, params, levels).expect("encode");
        cabac.encode_terminate(&mut w, 1);
        w.align_zero();
        let bytes = w.finish();

        let mut engine = CabacEngine::new(BitReader::new(&bytes)).expect("init");
        let mut dec_ctx = ResidualContexts::init(0, 26);
        let block = decode_residual_coding(&mut engine, &mut dec_ctx, params).expect("decode");
        assert_eq!(block.levels, levels, "decoded levels");
        assert_eq!(engine.decode_terminate().unwrap(), 1, "terminator");
        assert_eq!(enc_ctx, dec_ctx, "context state evolution");
    }

    fn params(log2: u32, is_chroma: bool, scan: ScanIdx) -> ResidualCodingParams {
        ResidualCodingParams {
            log2_trafo_size: log2,
            is_chroma,
            scan_idx: scan,
            sign_data_hiding_enabled_flag: false,
            sign_hidden_suppressed: false,
            transform_skip_sig_ctx: false,
            persistent_rice_adaptation_enabled_flag: false,
            cabac_bypass_alignment_enabled_flag: false,
            extended_precision_processing_flag: false,
            bit_depth: 8,
            rice_stat_transform_skip: false,
        }
    }

    #[test]
    fn single_dc_coefficient_roundtrips() {
        for log2 in 2..=5u32 {
            let size = 1usize << log2;
            let mut levels = vec![0i32; size * size];
            levels[0] = -3;
            roundtrip(&params(log2, false, ScanIdx::Diagonal), &levels);
            roundtrip(&params(log2, true, ScanIdx::Diagonal), &levels);
        }
    }

    #[test]
    fn single_far_coefficient_exercises_last_sig_suffix() {
        // 32x32: position (25, 30) needs prefix > 3 + suffix bits.
        let size = 32usize;
        let mut levels = vec![0i32; size * size];
        levels[30 * size + 25] = 7;
        levels[0] = 1;
        roundtrip(&params(5, false, ScanIdx::Diagonal), &levels);
    }

    #[test]
    fn dense_blocks_roundtrip_all_sizes_and_scans() {
        let mut x = 0x2468_ACE1u32;
        let mut rand = move || {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            x >> 16
        };
        for log2 in 2..=5u32 {
            let size = 1usize << log2;
            let scans: &[ScanIdx] = if log2 <= 3 {
                &[ScanIdx::Diagonal, ScanIdx::Horizontal, ScanIdx::Vertical]
            } else {
                &[ScanIdx::Diagonal]
            };
            for &scan in scans {
                for density in [2u32, 6, 16] {
                    for chroma in [false, true] {
                        let levels: Vec<i32> = (0..size * size)
                            .map(|_| {
                                let r = rand();
                                if r % 16 < density {
                                    let mag = (r >> 8) % 200;
                                    let v = mag as i32 + 1;
                                    if r & 1 == 0 {
                                        v
                                    } else {
                                        -v
                                    }
                                } else {
                                    0
                                }
                            })
                            .collect();
                        if levels.iter().all(|&v| v == 0) {
                            continue;
                        }
                        roundtrip(&params(log2, chroma, scan), &levels);
                    }
                }
            }
        }
    }

    #[test]
    fn large_escape_levels_roundtrip() {
        // Rice escapes deep into EGk territory + the CoeffMax edge.
        let size = 8usize;
        let mut levels = vec![0i32; size * size];
        levels[0] = 32767;
        levels[1] = -32768 + 1; // -32767
        levels[8] = 1000;
        levels[9] = -1;
        roundtrip(&params(3, false, ScanIdx::Diagonal), &levels);
    }

    #[test]
    fn dc_only_middle_subblock_uses_inferred_sig() {
        // A middle sub-block whose only significant coefficient is its
        // DC cell: the decoder infers sig[0] without a bin — the
        // encoder must skip it symmetrically.
        let size = 16usize;
        let mut levels = vec![0i32; size * size];
        levels[15 * size + 15] = 2; // last sub-block (3,3)
        levels[4 * size + 4] = 5; // sub-block (1,1) DC only
        levels[0] = -9;
        roundtrip(&params(4, false, ScanIdx::Diagonal), &levels);
    }

    #[test]
    fn rejects_all_zero_and_out_of_range() {
        let mut w = BitWriter::new();
        let mut cabac = CabacEncoder::new();
        let mut ctx = ResidualContexts::init(0, 26);
        let p = params(2, false, ScanIdx::Diagonal);
        assert_eq!(
            encode_residual_coding(&mut w, &mut cabac, &mut ctx, &p, &[0i32; 16]),
            Err(ResidualEncodeError::AllZero)
        );
        let mut levels = [0i32; 16];
        levels[0] = 40000;
        assert_eq!(
            encode_residual_coding(&mut w, &mut cabac, &mut ctx, &p, &levels),
            Err(ResidualEncodeError::LevelOutOfRange(40000))
        );
    }
}
