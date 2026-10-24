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
use crate::encoder::inter::{FrameRecon, RefPlanes, WideRecon, YuvFrame, YuvFrameWide};
use crate::encoder::sample::SampleFmt;
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
/// `shift` is `BitDepth − 8`: the fit runs in 8-bit sample units
/// (the weight is scale-free; the coded offset is the sample-domain
/// offset divided by `1 << WpOffsetBdShift`, §7.4.7.3).
fn fit_plane(
    src: &[u16],
    reference: &[u16],
    width: usize,
    height: usize,
    shift: u32,
) -> Option<(i32, i32)> {
    let (mut n, mut ss, mut sr, mut sss, mut srr) = (0i128, 0i128, 0i128, 0i128, 0i128);
    let half = if shift > 0 { 1i128 << (shift - 1) } else { 0 };
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            let s = (i128::from(src[y * width + x]) + half) >> shift;
            let r = (i128::from(reference[y * width + x]) + half) >> shift;
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

fn fit_reference(
    frame: &YuvFrameWide<'_>,
    r: &WideRecon,
    width: usize,
    height: usize,
    fmt: SampleFmt,
) -> WpRefEntry {
    let shift_y = u32::from(fmt.bit_depth_luma.saturating_sub(8));
    let shift_c = u32::from(fmt.bit_depth_chroma.saturating_sub(8));
    let luma = fit_plane(frame.y, &r.y, width, height, shift_y).map(|(w, o)| (w - 64, o));
    let (cw, ch) = fmt.chroma_dims(width, height);
    let (cb, cr) = if fmt.has_chroma() {
        (
            fit_plane(frame.cb, &r.cb, cw, ch, shift_c),
            fit_plane(frame.cr, &r.cr, cw, ch, shift_c),
        )
    } else {
        (None, None)
    };
    let chroma = (cb.is_some() || cr.is_some()).then(|| {
        [cb, cr].map(|c| {
            let (w, o) = c.unwrap_or((64, 0));
            let delta_o =
                o - WP_OFFSET_HALF_RANGE_C + ((WP_OFFSET_HALF_RANGE_C * w) >> LOG2_WEIGHT_DENOM);
            (w - 64, delta_o)
        })
    });
    WpRefEntry { luma, chroma }
}

/// The decode-side resolution: §7.4.7.3 eqs 7-56 .. 7-59 for the
/// chroma offset, then the `<< WpOffsetBdShiftY` / `<<
/// WpOffsetBdShiftC` of eqs 8-268 / 8-273 that the §8.5.3.3.4.3
/// combine reads (identity at 8 bits).
fn resolve_entry(e: &WpRefEntry, fmt: SampleFmt) -> WpListWeights {
    let shift_y = fmt.bit_depth_luma.saturating_sub(8);
    let shift_c = fmt.bit_depth_chroma.saturating_sub(8);
    let (dw, o) = e.luma.unwrap_or((0, 0));
    let [(dwb, dob), (dwr, dor)] = e.chroma.unwrap_or([(0, 0), (0, 0)]);
    let chroma_o = |dw: i32, delta_o: i32, present: bool| -> i32 {
        if !present {
            return 0;
        }
        let w = (1 << LOG2_WEIGHT_DENOM) + dw;
        (WP_OFFSET_HALF_RANGE_C + delta_o - ((WP_OFFSET_HALF_RANGE_C * w) >> LOG2_WEIGHT_DENOM))
            .clamp(-WP_OFFSET_HALF_RANGE_C, WP_OFFSET_HALF_RANGE_C - 1)
            << shift_c
    };
    WpListWeights {
        w_luma: (1 << LOG2_WEIGHT_DENOM) + dw,
        o_luma: o << shift_y,
        w_cb: (1 << LOG2_WEIGHT_DENOM) + dwb,
        o_cb: chroma_o(dwb, dob, e.chroma.is_some()),
        w_cr: (1 << LOG2_WEIGHT_DENOM) + dwr,
        o_cr: chroma_o(dwr, dor, e.chroma.is_some()),
    }
}

/// Fit one `pred_weight_table( )` for a frame against its 8-bit
/// 4:2:0 references: per reference and component a least-squares
/// weight / offset over the 2x2-subsampled picture, kept only when
/// it differs materially from identity. See [`estimate_wide`].
#[must_use]
pub fn estimate(
    frame: &YuvFrame<'_>,
    l0: &[&FrameRecon],
    l1: &[&FrameRecon],
    width: usize,
    height: usize,
) -> WpTables {
    let (y, cb, cr) = crate::encoder::inter::widen_frame(frame);
    let widen = |r: &FrameRecon| WideRecon::from_narrow(r.y.clone(), r.cb.clone(), r.cr.clone());
    let l0w: Vec<WideRecon> = l0.iter().map(|r| widen(r)).collect();
    let l1w: Vec<WideRecon> = l1.iter().map(|r| widen(r)).collect();
    estimate_wide(
        &YuvFrameWide::new(&y, &cb, &cr),
        &l0w.iter().collect::<Vec<_>>(),
        &l1w.iter().collect::<Vec<_>>(),
        width,
        height,
        SampleFmt::YUV420_8,
    )
}

/// [`estimate`] at any sample format: the fit runs in 8-bit units
/// (the coded `luma_offset_lX` / `delta_chroma_offset_lX` are 8-bit
/// quantities under `high_precision_offsets_enabled_flag == 0`), the
/// resolved tables carry the `WpOffsetBdShift`-scaled offsets.
#[must_use]
pub(crate) fn estimate_wide(
    frame: &YuvFrameWide<'_>,
    l0: &[&WideRecon],
    l1: &[&WideRecon],
    width: usize,
    height: usize,
    fmt: SampleFmt,
) -> WpTables {
    let fit = |list: &[&WideRecon]| -> Vec<WpRefEntry> {
        list.iter()
            .map(|r| fit_reference(frame, r, width, height, fmt))
            .collect()
    };
    let l0 = fit(l0);
    let l1 = fit(l1);
    let resolved = SliceWpTables {
        luma_log2_weight_denom: LOG2_WEIGHT_DENOM,
        chroma_log2_weight_denom: LOG2_WEIGHT_DENOM,
        l0: l0.iter().map(|e| resolve_entry(e, fmt)).collect(),
        l1: l1.iter().map(|e| resolve_entry(e, fmt)).collect(),
    };
    WpTables { l0, l1, resolved }
}

/// The luma planes the motion search reads under explicit weighted
/// prediction: each reference's luma through its fitted weight /
/// offset (the §8.5.3.3.4.3 uni-prediction formula at full-sample
/// positions, the offset in sample units), clipped to the picture's
/// depth; a reference without a luma entry is copied.
#[must_use]
pub(crate) fn weighted_luma_planes(
    refs: &[RefPlanes<'_>],
    entries: &[WpRefEntry],
) -> Vec<Vec<u16>> {
    refs.iter()
        .zip(entries)
        .map(|(r, e)| match e.luma {
            Some((dw, o)) => {
                let w = (1 << LOG2_WEIGHT_DENOM) + dw;
                let o = o << r.fmt.bit_depth_luma.saturating_sub(8);
                let max = r.fmt.max_luma();
                r.y.iter()
                    .map(|&v| {
                        (((i32::from(v) * w + 32) >> LOG2_WEIGHT_DENOM) + o).clamp(0, max) as u16
                    })
                    .collect()
            }
            None => r.y.to_vec(),
        })
        .collect()
}

/// Reference planes whose luma is replaced by the weighted copies of
/// [`weighted_luma_planes`] (chroma stays the reference's own).
pub(crate) fn with_luma<'r>(planes: &'r [Vec<u16>], refs: &[RefPlanes<'r>]) -> Vec<RefPlanes<'r>> {
    planes
        .iter()
        .zip(refs)
        .map(|(y, r)| RefPlanes {
            halfpel: None,
            y,
            cb: r.cb,
            cr: r.cr,
            width: r.width,
            height: r.height,
            fmt: r.fmt,
        })
        .collect()
}

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
                let r = resolve_entry(&e, SampleFmt::YUV420_8);
                assert_eq!((r.w_cb, r.o_cb), (w, o));
            }
        }
    }
}
