//! Quantization tools of the quadtree coder: the §7.3.8.11 sign-data-
//! hiding parity adjustment over the deadzone quantizer.
//!
//! **Sign data hiding** ([`apply_sign_hiding`]): for every 4x4
//! sub-block where `signHidden` holds (`lastSigScanPos −
//! firstSigScanPos > 3`), the decoder infers the sign of the
//! first-in-scan-order coefficient from the parity of `sumAbsLevel`;
//! when the quantized parity disagrees, the cheapest ±1 level
//! adjustment (quantization residue distortion mapped to the sample
//! domain through the transform gain, plus a rate estimate; sub-block
//! consistency re-checked after the change) is applied.

use crate::scan::{scan_order, ScanIdx};
use crate::transform::LEVEL_SCALE;

/// Fixed 8-bit depth.
const BIT_DEPTH: u32 = 8;
/// §7.4.9.11 CoeffMax for the non-extended-precision profiles.
const COEFF_MAX: i64 = 0x7FFF;

/// One transform block's quantization request.
#[derive(Debug, Clone, Copy)]
pub struct TbQuant {
    /// `log2TrafoSize` (2..=5).
    pub log2: u32,
    /// The block's `qP` (luma `QpY` or the Table 8-10-mapped chroma QP).
    pub qp: u32,
    /// `cIdx > 0`.
    pub is_chroma: bool,
    /// The §7.4.9.11 scan.
    pub scan: ScanIdx,
    /// Sample-domain Lagrangian (SSD per bin).
    pub lambda: u64,
    /// Apply the sign-data-hiding parity adjustment.
    pub sign_hiding: bool,
}

/// §8.6.3-derived reciprocal quantizer scale (`round( 2^20 /
/// levelScale[ qP % 6 ] )`).
fn quant_scale(qp_rem: u32) -> u64 {
    let ls = u64::from(LEVEL_SCALE[qp_rem as usize] as u32);
    ((1u64 << 20) + ls / 2) / ls
}

/// `qBits = 14 + qP / 6 + ( 15 − BitDepth − log2TbS )`.
fn q_bits(log2: u32, qp: u32) -> u32 {
    14 + qp / 6 + (15 - BIT_DEPTH - log2)
}

/// Coefficient-domain squared error of coding the scaled magnitude
/// `q` (`|coef| · scale`) as `level`.
fn coef_sq_err(q: u64, level: u64, qbits: u32, scale: u64) -> u64 {
    let e = q.abs_diff(level << qbits);
    ((u128::from(e) * u128::from(e)) / (u128::from(scale) * u128::from(scale))) as u64
}

/// Plain deadzone quantizer (one-third rounding offset) on one
/// coefficient magnitude.
fn deadzone_level(q: u64, qbits: u32) -> u64 {
    ((q + (1u64 << qbits) / 3) >> qbits).min(COEFF_MAX as u64)
}

/// Quantize one TB's forward-transform coefficients (row-major) to
/// `TransCoeffLevel` under `req`: the deadzone quantizer, then the
/// sign-hiding parity adjustment when requested.
#[must_use]
pub fn quantize_tb(coef: &[i32], req: &TbQuant) -> Vec<i32> {
    let qbits = q_bits(req.log2, req.qp);
    let scale = quant_scale(req.qp % 6);
    let q: Vec<u64> = coef
        .iter()
        .map(|&c| u64::from(c.unsigned_abs()) * scale)
        .collect();
    let mut levels: Vec<i32> = q
        .iter()
        .zip(coef)
        .map(|(&qv, &c)| deadzone_level(qv, qbits) as i32 * c.signum())
        .collect();
    if req.sign_hiding && levels.iter().any(|&l| l != 0) {
        apply_sign_hiding(&mut levels, &q, coef, req, qbits, scale);
    }
    levels
}

/// Sample-domain λ lifted to the coefficient-squared-error domain
/// (`coef = 2^(7 − log2) · orthonormal`), in 1/256-bit-rate units.
fn lambda_coef(lambda: u64, log2: u32) -> u64 {
    lambda << (2 * (7 - log2))
}

/// Rough bit estimate of one |level| (sig + sign + greater flags +
/// Rice bins) for the sign-hiding rate term.
fn level_bits_estimate(level: u64) -> u64 {
    if level == 0 {
        0
    } else {
        3 + 2 * u64::from(64 - level.leading_zeros())
    }
}

/// The §7.3.8.11 sign-hiding parity adjustment (see the module docs).
/// `levels` is row-major and is modified in place.
fn apply_sign_hiding(
    levels: &mut [i32],
    q: &[u64],
    coef: &[i32],
    req: &TbQuant,
    qbits: u32,
    scale: u64,
) {
    let log2 = req.log2;
    let size = 1usize << log2;
    let lam = lambda_coef(req.lambda, log2);
    let pos_scan = scan_order(2, req.scan).expect("4x4 scan");
    let sub_scan = scan_order((log2 - 2) as u8, req.scan).expect("sub-block scan");
    let num_sb_1d = 1usize << (log2 - 2);
    let n_sb = num_sb_1d * num_sb_1d;

    // Sub-block descriptor: (firstSigScanPos, lastSigScanPos, sumAbs,
    // first-coefficient index) in the §7.3.8.11 sense.
    let describe = |levels: &[i32], sb_i: usize| -> Option<(usize, usize, u64, usize)> {
        let sb = sub_scan[sb_i];
        let mut first = 16usize;
        let mut last: Option<usize> = None;
        let mut sum = 0u64;
        let mut first_idx = 0usize;
        for n in (0..16).rev() {
            let xc = ((sb.x as usize) << 2) + pos_scan[n].x as usize;
            let yc = ((sb.y as usize) << 2) + pos_scan[n].y as usize;
            let idx = yc * size + xc;
            let l = levels[idx];
            if l != 0 {
                if last.is_none() {
                    last = Some(n);
                }
                first = n;
                first_idx = idx;
                sum += u64::from(l.unsigned_abs());
            }
        }
        last.map(|l| (first, l, sum, first_idx))
    };
    // Whether the sub-block is consistent: not hidden, or parity
    // matches the first coefficient's sign.
    let consistent = |levels: &[i32], sb_i: usize| -> bool {
        match describe(levels, sb_i) {
            None => true,
            Some((first, last, sum, first_idx)) => {
                if last - first <= 3 {
                    true
                } else {
                    let negative = levels[first_idx] < 0;
                    (sum % 2 == 1) == negative
                }
            }
        }
    };

    for (sb_i, sb) in sub_scan.iter().enumerate().take(n_sb) {
        if consistent(levels, sb_i) {
            continue;
        }
        let mut best: Option<(i128, usize, i32)> = None;
        for pos in pos_scan.iter().take(16) {
            let xc = ((sb.x as usize) << 2) + pos.x as usize;
            let yc = ((sb.y as usize) << 2) + pos.y as usize;
            let idx = yc * size + xc;
            let cur = i64::from(levels[idx]);
            let cur_abs = cur.unsigned_abs();
            let sign: i64 = if cur != 0 {
                cur.signum()
            } else if coef[idx] < 0 {
                -1
            } else {
                1
            };
            let rate_of = |l: u64| -> u64 { lam * 256 * level_bits_estimate(l) };
            let d_cur = coef_sq_err(q[idx], cur_abs, qbits, scale) * 256 + rate_of(cur_abs);
            for delta in [1i64, -1] {
                let new_abs = cur_abs as i64 + delta;
                if !(0..=COEFF_MAX).contains(&new_abs) {
                    continue;
                }
                let new_abs = new_abs as u64;
                let d_new = coef_sq_err(q[idx], new_abs, qbits, scale) * 256 + rate_of(new_abs);
                let cost = d_new as i128 - d_cur as i128;
                let candidate = (sign * new_abs as i64) as i32;
                let old = levels[idx];
                levels[idx] = candidate;
                let ok = consistent(levels, sb_i);
                levels[idx] = old;
                if ok && best.map_or(true, |(c, _, _)| cost < c) {
                    best = Some((cost, idx, candidate));
                }
            }
        }
        if let Some((_, idx, candidate)) = best {
            levels[idx] = candidate;
        }
        debug_assert!(consistent(levels, sb_i), "sign-hiding parity unresolved");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;
    use crate::cabac::CabacEngine;
    use crate::encoder::bitwriter::BitWriter;
    use crate::encoder::cabac::CabacEncoder;
    use crate::encoder::residual::encode_residual_coding;
    use crate::residual::{decode_residual_coding, ResidualCodingParams, ResidualContexts};

    fn params(log2: u32, scan: ScanIdx, sdh: bool) -> ResidualCodingParams {
        ResidualCodingParams {
            log2_trafo_size: log2,
            is_chroma: false,
            scan_idx: scan,
            sign_data_hiding_enabled_flag: sdh,
            sign_hidden_suppressed: false,
            transform_skip_sig_ctx: false,
            persistent_rice_adaptation_enabled_flag: false,
            cabac_bypass_alignment_enabled_flag: false,
            extended_precision_processing_flag: false,
            bit_depth: 8,
            rice_stat_transform_skip: false,
        }
    }

    fn roundtrip(p: &ResidualCodingParams, levels: &[i32]) {
        let mut w = BitWriter::new();
        let mut cabac = CabacEncoder::new();
        let mut enc_ctx = ResidualContexts::init(0, 26);
        encode_residual_coding(&mut w, &mut cabac, &mut enc_ctx, p, levels).expect("encode");
        cabac.encode_terminate(&mut w, 1);
        w.align_zero();
        let bytes = w.finish();
        let mut engine = CabacEngine::new(BitReader::new(&bytes)).expect("init");
        let mut dec_ctx = ResidualContexts::init(0, 26);
        let block = decode_residual_coding(&mut engine, &mut dec_ctx, p).expect("decode");
        assert_eq!(block.levels, levels);
    }

    fn synthetic_coefs(log2: u32, seed: u32, amp: i32) -> Vec<i32> {
        let size = 1usize << log2;
        let mut x = seed;
        (0..size * size)
            .map(|i| {
                x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let (u, v) = (i % size, i / size);
                let decay = (u + v + 1) as i32;
                let r = ((x >> 16) % (2 * amp as u32 + 1)) as i32 - amp;
                r * 8 / decay
            })
            .collect()
    }

    #[test]
    fn sign_hiding_parity_holds_and_roundtrips() {
        for log2 in 2..=5u32 {
            for scan in [ScanIdx::Diagonal, ScanIdx::Horizontal, ScanIdx::Vertical] {
                if log2 > 3 && scan != ScanIdx::Diagonal {
                    continue;
                }
                for seed in 1..12u32 {
                    let coef = synthetic_coefs(log2, seed, 900);
                    let req = TbQuant {
                        log2,
                        qp: 27,
                        is_chroma: false,
                        scan,
                        lambda: 64,
                        sign_hiding: true,
                    };
                    let levels = quantize_tb(&coef, &req);
                    if levels.iter().any(|&l| l != 0) {
                        roundtrip(&params(log2, scan, true), &levels);
                    }
                }
            }
        }
    }

    #[test]
    fn deadzone_matches_legacy_quantizer() {
        for log2 in 2..=5u32 {
            let coef = synthetic_coefs(log2, 3, 700);
            for qp in [4u32, 22, 37, 51] {
                let req = TbQuant {
                    log2,
                    qp,
                    is_chroma: false,
                    scan: ScanIdx::Diagonal,
                    lambda: 1,
                    sign_hiding: false,
                };
                assert_eq!(
                    quantize_tb(&coef, &req),
                    crate::encoder::intra::quantize(&coef, 1 << log2, qp)
                );
            }
        }
    }
}
