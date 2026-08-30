//! Spatial adaptive quantization — per-CTB QP offsets from source
//! activity.
//!
//! Not part of the Recommendation: AQ is encoder policy. The classic
//! observation is that flat regions show quantization error much more
//! than busy ones, so the encoder spends relatively more bits there:
//! each CTB's luma *activity* (mean absolute deviation from the CTB
//! mean) is compared against the picture's average activity on a log
//! scale, and the CTB's QP moves by `strength` QP per octave of
//! activity ratio, clamped to ±6. The offsets are signalled through
//! §7.3.8.10 / §7.4.9.14 `cu_qp_delta`, so the stream stays fully
//! conforming.
//!
//! All arithmetic is integer (log2 in Q3 with a linear mantissa
//! approximation), so AQ is bit-deterministic across platforms.

/// `log2(x)` in Q3 (eighths), linear-in-mantissa approximation.
/// `x` must be nonzero.
fn log2_q3(x: u64) -> i64 {
    debug_assert!(x > 0);
    let il = 63 - i64::from(x.leading_zeros());
    // The three mantissa bits right below the leading one.
    let frac = if il >= 3 {
        ((x >> (il - 3)) & 7) as i64
    } else {
        ((x << (3 - il)) & 7) as i64
    };
    8 * il + frac
}

/// Per-CTB (`ctb`x`ctb`, raster order, edge CTBs clipped by the
/// picture) QP offsets for the `width`x`height` luma plane `y` at AQ
/// `strength` (0 = all zeros; 1..=3 = QP per octave of activity ratio,
/// clamped to ±6).
pub(crate) fn ctb_aq_deltas(
    y: &[u8],
    width: usize,
    height: usize,
    strength: u8,
    ctb: usize,
) -> Vec<i32> {
    let (ctbs_x, ctbs_y) = (width.div_ceil(ctb), height.div_ceil(ctb));
    let n = ctbs_x * ctbs_y;
    if strength == 0 {
        return vec![0; n];
    }
    // Activity per CTB: 1 + mean-absolute-deviation sum, normalized to
    // the 16x16 sample count so the octave scale is size-independent
    // (the +1 keeps the log finite on perfectly flat blocks).
    let mut log_act = Vec::with_capacity(n);
    for idx in 0..n {
        let x0 = (idx % ctbs_x) * ctb;
        let y0 = (idx / ctbs_x) * ctb;
        let (w, h) = ((width - x0).min(ctb), (height - y0).min(ctb));
        let mut sum = 0u64;
        for j in 0..h {
            for i in 0..w {
                sum += u64::from(y[(y0 + j) * width + x0 + i]);
            }
        }
        let count = (w * h) as u64;
        let mean = sum / count;
        let mut dev = 0u64;
        for j in 0..h {
            for i in 0..w {
                dev += u64::from(y[(y0 + j) * width + x0 + i]).abs_diff(mean);
            }
        }
        log_act.push(log2_q3(1 + dev * 256 / count));
    }
    let avg: i64 = log_act.iter().sum::<i64>() / n as i64;
    log_act
        .iter()
        .map(|&la| {
            // `strength` QP per octave (8 Q3 units), rounded to
            // nearest, clamped to ±6.
            let num = i64::from(strength) * (la - avg);
            let delta = if num >= 0 {
                (num + 4) / 8
            } else {
                (num - 4) / 8
            };
            delta.clamp(-6, 6) as i32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log2_q3_is_monotonic_and_exact_on_powers() {
        for e in 0..40u32 {
            assert_eq!(log2_q3(1u64 << e), 8 * i64::from(e));
        }
        let mut last = i64::MIN;
        for x in 1..5000u64 {
            let l = log2_q3(x);
            assert!(l >= last, "monotonic at {x}");
            last = l;
        }
    }

    #[test]
    fn strength_zero_is_all_zeros() {
        let y = vec![128u8; 64 * 32];
        assert_eq!(ctb_aq_deltas(&y, 64, 32, 0, 16), vec![0; 8]);
    }

    #[test]
    fn flat_blocks_get_lower_qp_than_busy_blocks() {
        // Left half flat, right half checkerboard.
        let (w, h) = (64usize, 32usize);
        let y: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, yy) = (i % w, i / w);
                if x < w / 2 {
                    100
                } else if (x + yy) % 2 == 0 {
                    30
                } else {
                    220
                }
            })
            .collect();
        for strength in 1..=3u8 {
            let d = ctb_aq_deltas(&y, w, h, strength, 16);
            assert_eq!(d.len(), 8);
            for (ctb, &delta) in d.iter().enumerate() {
                let flat = ctb % 4 < 2;
                assert!((-6..=6).contains(&delta));
                if flat {
                    assert!(delta < 0, "flat CTB {ctb} strength {strength}: {delta}");
                } else {
                    assert!(delta > 0, "busy CTB {ctb} strength {strength}: {delta}");
                }
            }
            // Stronger AQ spreads at least as wide.
            if strength > 1 {
                let d1 = ctb_aq_deltas(&y, w, h, 1, 16);
                let spread = |v: &[i32]| v.iter().max().unwrap() - v.iter().min().unwrap();
                assert!(spread(&d) >= spread(&d1));
            }
        }
    }
}
