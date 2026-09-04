//! Encoder-side explicit weighted prediction: fade detection, the
//! §7.3.6.3 `pred_weight_table( )` writer, and the §8.5.3.3.4.3
//! weight tables the prediction path applies.
//!
//! For every active reference of every list the luma and both chroma
//! planes are fitted by integer moment matching against the source
//! picture (`src ≈ w · ref / 2^6 + o` with the weight from the
//! standard-deviation ratio and the offset from the means —
//! statistics that motion leaves alone — `luma_log2_weight_denom ==
//! ChromaLog2WeightDenom == 6`, on a 2:1 sub-sampled grid); a weight
//! pair is signalled only when the gain differs from unity by at
//! least 4/64 or the mean shifts by at least two levels (a fade, not
//! noise or motion).
//! Unsignalled references keep the §7.4.7.3 inferred identity — and
//! the explicit combine with identity weights reproduces the default
//! §8.5.3.3.4.2 combine bit for bit, so a `weighted_pred_flag` stream
//! never costs prediction quality. Motion search runs on
//! pre-weighted reference copies ([`weighted_ref_planes`]) so a fade
//! does not bias the SAD; the exact prediction and reconstruction
//! take the decoder's own [`SliceWpTables::resolve_pu`] combine.

use crate::encoder::bitwriter::BitWriter;
use crate::encoder::inter::{FrameRecon, RefPlanes, YuvFrame};
use crate::inter_pred::WpListWeights;
use crate::inter_recon::SliceWpTables;

/// `luma_log2_weight_denom` (and, with `delta_chroma_log2_weight_denom
/// == 0`, `ChromaLog2WeightDenom`).
pub const LOG2_WEIGHT_DENOM: u8 = 6;
/// `WpOffsetHalfRangeC` with `high_precision_offsets_enabled_flag == 0`.
const WP_OFFSET_HALF_RANGE_C: i32 = 128;

/// One reference's signalled `pred_weight_table( )` entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WpRefEntry {
    /// `luma_weight_lX_flag == 1`: `(delta_luma_weight, luma_offset)`.
    pub luma: Option<(i32, i32)>,
    /// `chroma_weight_lX_flag == 1`: per `j` (Cb, Cr)
    /// `(delta_chroma_weight, delta_chroma_offset)`.
    pub chroma: Option<[(i32, i32); 2]>,
}

/// The slice's weighted-prediction decision: the syntax entries per
/// list plus the resolved tables the prediction reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WpTables {
    /// `RefPicList0` entries.
    pub l0: Vec<WpRefEntry>,
    /// `RefPicList1` entries (empty on P slices).
    pub l1: Vec<WpRefEntry>,
    /// The §7.4.7.3-resolved tables.
    pub resolved: SliceWpTables,
}

impl WpTables {
    /// Whether any reference carries a non-identity weight.
    #[must_use]
    pub fn any(&self) -> bool {
        self.l0
            .iter()
            .chain(&self.l1)
            .any(|e| e.luma.is_some() || e.chroma.is_some())
    }
}

/// `round( num / den )` for `den > 0` (ties away from zero).
fn div_round(num: i128, den: i128) -> i128 {
    if num >= 0 {
        (num + den / 2) / den
    } else {
        -((-num + den / 2) / den)
    }
}

/// Integer square root (floor).
fn isqrt(v: u128) -> u128 {
    if v < 2 {
        return v;
    }
    let mut x = 1u128 << (v.ilog2() / 2 + 1);
    loop {
        let y = (x + v / x) / 2;
        if y >= x {
            return x;
        }
        x = y;
    }
}

/// Moment-matching fit of `src ≈ w · reference / 64 + o` over a 2:1
/// sub-sampled grid — the weight from the standard-deviation ratio,
/// the offset from the means — which motion between the two pictures
/// leaves untouched (a pixelwise regression would read misalignment
/// as a contrast loss). `Some((w, o))` when the pair moves the picture
/// meaningfully: the gain differs from unity by >= 4/64 or the mean
/// shifts by >= 2 levels.
fn fit_plane(src: &[u8], reference: &[u8], width: usize, height: usize) -> Option<(i32, i32)> {
    let (mut n, mut ss, mut sr, mut sss, mut srr) = (0i128, 0i128, 0i128, 0i128, 0i128);
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            let s = i128::from(src[y * width + x]);
            let r = i128::from(reference[y * width + x]);
            n += 1;
            ss += s;
            sr += r;
            sss += s * s;
            srr += r * r;
        }
    }
    let var_s = n * sss - ss * ss;
    let var_r = n * srr - sr * sr;
    if var_r <= 0 || n == 0 {
        return None;
    }
    // w = round( 64 · sqrt( var_s / var_r ) ).
    let ratio_q = (4096u128 * 4 * var_s.max(0) as u128) / var_r as u128;
    let w = isqrt(ratio_q).div_ceil(2) as i128;
    let w = w.clamp(64 - 128, 64 + 127) as i32;
    let o = div_round(64 * ss - i128::from(w) * sr, 64 * n).clamp(-128, 127) as i32;
    let mean_shift = div_round(ss - sr, n);
    if (w - 64).abs() < 4 && mean_shift.abs() < 2 {
        return None;
    }
    ((w, o) != (64, 0)).then_some((w, o))
}

/// Fit one reference picture: the luma and chroma entries.
fn fit_reference(frame: &YuvFrame<'_>, r: &FrameRecon, width: usize, height: usize) -> WpRefEntry {
    let luma = fit_plane(frame.y, &r.y, width, height).map(|(w, o)| (w - 64, o));
    let (cw, ch) = (width / 2, height / 2);
    let cb = fit_plane(frame.cb, &r.cb, cw, ch);
    let cr = fit_plane(frame.cr, &r.cr, cw, ch);
    let chroma = (cb.is_some() || cr.is_some()).then(|| {
        [cb, cr].map(|c| {
            let (w, o) = c.unwrap_or((64, 0));
            // eq. 7-58 inverted: delta_chroma_offset = ChromaOffset −
            // WpOffsetHalfRangeC + ( ( WpOffsetHalfRangeC · w ) >> denom ).
            let delta_o =
                o - WP_OFFSET_HALF_RANGE_C + ((WP_OFFSET_HALF_RANGE_C * w) >> LOG2_WEIGHT_DENOM);
            (w - 64, delta_o)
        })
    });
    WpRefEntry { luma, chroma }
}

/// The §7.4.7.3-resolved [`WpListWeights`] of one entry.
fn resolve_entry(e: &WpRefEntry) -> WpListWeights {
    let (dw, o) = e.luma.unwrap_or((0, 0));
    let [(dwb, dob), (dwr, dor)] = e.chroma.unwrap_or([(0, 0), (0, 0)]);
    let chroma_o = |dw: i32, delta_o: i32, present: bool| -> i32 {
        if !present {
            return 0;
        }
        let w = (1 << LOG2_WEIGHT_DENOM) + dw;
        (WP_OFFSET_HALF_RANGE_C + delta_o - ((WP_OFFSET_HALF_RANGE_C * w) >> LOG2_WEIGHT_DENOM))
            .clamp(-WP_OFFSET_HALF_RANGE_C, WP_OFFSET_HALF_RANGE_C - 1)
    };
    // 8-bit: WpOffsetBdShiftY == WpOffsetBdShiftC == 0.
    WpListWeights {
        w_luma: (1 << LOG2_WEIGHT_DENOM) + dw,
        o_luma: o,
        w_cb: (1 << LOG2_WEIGHT_DENOM) + dwb,
        o_cb: chroma_o(dwb, dob, e.chroma.is_some()),
        w_cr: (1 << LOG2_WEIGHT_DENOM) + dwr,
        o_cr: chroma_o(dwr, dor, e.chroma.is_some()),
    }
}

/// Estimate the slice's weighted-prediction tables against its
/// reference lists.
#[must_use]
pub fn estimate(
    frame: &YuvFrame<'_>,
    l0: &[&FrameRecon],
    l1: &[&FrameRecon],
    width: usize,
    height: usize,
) -> WpTables {
    let fit = |list: &[&FrameRecon]| -> Vec<WpRefEntry> {
        list.iter()
            .map(|r| fit_reference(frame, r, width, height))
            .collect()
    };
    let l0 = fit(l0);
    let l1 = fit(l1);
    let resolved = SliceWpTables {
        luma_log2_weight_denom: LOG2_WEIGHT_DENOM,
        chroma_log2_weight_denom: LOG2_WEIGHT_DENOM,
        l0: l0.iter().map(resolve_entry).collect(),
        l1: l1.iter().map(resolve_entry).collect(),
    };
    WpTables { l0, l1, resolved }
}

/// Reference planes with the luma weights applied (motion-search
/// inputs); references without a luma entry are copied unchanged.
#[must_use]
pub(crate) fn weighted_ref_planes(refs: &[RefPlanes], entries: &[WpRefEntry]) -> Vec<RefPlanes> {
    refs.iter()
        .zip(entries)
        .map(|(r, e)| {
            let y = match e.luma {
                Some((dw, o)) => {
                    let w = (1 << LOG2_WEIGHT_DENOM) + dw;
                    r.y.iter()
                        .map(|&v| (((v * w + 32) >> LOG2_WEIGHT_DENOM) + o).clamp(0, 255))
                        .collect()
                }
                None => r.y.clone(),
            };
            RefPlanes {
                y,
                cb: r.cb.clone(),
                cr: r.cr.clone(),
                width: r.width,
                height: r.height,
            }
        })
        .collect()
}

/// §7.3.6.3 `pred_weight_table( )` (4:2:0, single layer: every
/// reference differs from the current picture, so every per-i flag
/// is present).
pub fn write_pred_weight_table(w: &mut BitWriter, t: &WpTables, b_slice: bool) {
    w.ue(u32::from(LOG2_WEIGHT_DENOM)); // luma_log2_weight_denom
    w.se(0); // delta_chroma_log2_weight_denom
    let write_list = |w: &mut BitWriter, list: &[WpRefEntry]| {
        for e in list {
            w.put_bit(u8::from(e.luma.is_some())); // luma_weight_lX_flag[i]
        }
        for e in list {
            w.put_bit(u8::from(e.chroma.is_some())); // chroma_weight_lX_flag[i]
        }
        for e in list {
            if let Some((dw, o)) = e.luma {
                w.se(dw); // delta_luma_weight_lX[i]
                w.se(o); // luma_offset_lX[i]
            }
            if let Some(c) = e.chroma {
                for (dw, delta_o) in c {
                    w.se(dw); // delta_chroma_weight_lX[i][j]
                    w.se(delta_o); // delta_chroma_offset_lX[i][j]
                }
            }
        }
    };
    write_list(w, &t.l0);
    if b_slice {
        write_list(w, &t.l1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: usize, h: usize, f: impl Fn(usize, usize) -> u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let y: Vec<u8> = (0..w * h).map(|i| f(i % w, i / w)).collect();
        let cb: Vec<u8> = (0..w * h / 4)
            .map(|i| f(i % (w / 2), i / (w / 2)) / 2 + 64)
            .collect();
        let cr = cb.clone();
        (y, cb, cr)
    }

    #[test]
    fn fade_is_detected_and_identity_is_not() {
        let (w, h) = (64, 48);
        let (ry, rcb, rcr) = frame(w, h, |x, y| (40 + (x * 3 + y * 2) % 150) as u8);
        let reference = FrameRecon {
            y: ry.clone(),
            cb: rcb.clone(),
            cr: rcr.clone(),
            motion_field: None,
        };
        // 70 % brightness fade.
        let y: Vec<u8> = ry.iter().map(|&v| (u32::from(v) * 7 / 10) as u8).collect();
        let cb: Vec<u8> = rcb.iter().map(|&v| (u32::from(v) * 7 / 10) as u8).collect();
        let cr = cb.clone();
        let src = YuvFrame {
            y: &y,
            cb: &cb,
            cr: &cr,
        };
        let t = estimate(&src, &[&reference], &[], w, h);
        assert!(t.any());
        let (dw, o) = t.l0[0].luma.expect("luma weight");
        assert!(
            (dw + 64 - 45).abs() <= 2,
            "weight ~0.7·64 = 45, got {}",
            dw + 64
        );
        assert!(o.abs() <= 3, "offset ~0, got {o}");
        assert!(t.l0[0].chroma.is_some());
        // Same picture: identity.
        let same = YuvFrame {
            y: &ry,
            cb: &rcb,
            cr: &rcr,
        };
        assert!(!estimate(&same, &[&reference], &[&reference], w, h).any());
    }

    #[test]
    fn chroma_offset_round_trips_eq_7_58() {
        for w in [-64i32, 10, 64, 100, 191] {
            for o in [-128i32, -3, 0, 5, 127] {
                let delta_o = o - WP_OFFSET_HALF_RANGE_C
                    + ((WP_OFFSET_HALF_RANGE_C * w) >> LOG2_WEIGHT_DENOM);
                assert!((-4 * 128..4 * 128).contains(&delta_o));
                let e = WpRefEntry {
                    luma: None,
                    chroma: Some([(w - 64, delta_o); 2]),
                };
                let r = resolve_entry(&e);
                assert_eq!((r.w_cb, r.o_cb), (w, o));
            }
        }
    }
}
