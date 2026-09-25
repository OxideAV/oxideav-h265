//! Annex H (SHVC) inter-layer reference picture derivation — H.8.1.4.
//!
//! Given a decoded direct reference layer picture `rlPic` (its samples
//! and per-4x4 motion field) and the current layer's picture format, the
//! process yields the inter-layer reference picture `ilRefPic` the
//! F.8.3.4 reference lists hold:
//!
//! * H.8.1.4.1 derives the reference / scaled reference regions from the
//!   PPS `ref_region_*` / `scaled_ref_layer_*` offsets, the 16.16
//!   spatial scale factors (H-18 .. H-21) and the resampling phases
//!   (H-22 .. H-25, with the F.7.4.3.3.4 inference of an absent vertical
//!   chroma phase); when the sizes, offsets, phases, bit depths and
//!   chroma formats all match, `ilRefPic` IS `rlPic`
//!   (`equalPictureSizeAndOffsetFlag`) and nothing is generated.
//! * H.8.1.4.2 resamples the sample arrays: the 16-phase 8-tap luma
//!   filter (Table H.1) and 4-tap chroma filter (Table H.2) applied
//!   separably — the horizontal pass over every reference row, the
//!   vertical pass over the current picture — exactly as eqs H-38 / H-39
//!   and H-50 / H-51 combine them, with the reference-layer sample
//!   location in 1/16 units from H.8.1.4.2.4 (H-53 .. H-64).
//! * H.8.1.4.3 resamples the motion and mode parameters per 16x16 block
//!   of the current picture from the rounded reference-layer location
//!   (H-65 .. H-70), scaling the motion vectors by the region ratio
//!   (H-74 .. H-79).
//!
//! The H.8.1.4.4 colour mapping process (colour-gamut scalability) is
//! not implemented; a PPS enabling it for the reference layer is
//! reported as unsupported by the driver.

use crate::motion::{MotionCell, MotionField};
use crate::picture::{sub_wh_c, Picture, Plane};
use crate::pps::RefLocOffset;

/// Table H.1 — 16-phase luma resampling filter `fL[ p ][ x ]`.
pub const LUMA_FILTER: [[i32; 8]; 16] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [0, 1, -3, 63, 4, -2, 1, 0],
    [-1, 2, -5, 62, 8, -3, 1, 0],
    [-1, 3, -8, 60, 13, -4, 1, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 52, 26, -8, 3, -1],
    [-1, 3, -9, 47, 31, -10, 4, -1],
    [-1, 4, -11, 45, 34, -10, 4, -1],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [-1, 4, -10, 34, 45, -11, 4, -1],
    [-1, 4, -10, 31, 47, -9, 3, -1],
    [-1, 3, -8, 26, 52, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
    [0, 1, -4, 13, 60, -8, 3, -1],
    [0, 1, -3, 8, 62, -5, 2, -1],
    [0, 1, -2, 4, 63, -3, 1, 0],
];

/// Table H.2 — 16-phase chroma resampling filter `fC[ p ][ x ]`.
pub const CHROMA_FILTER: [[i32; 4]; 16] = [
    [0, 64, 0, 0],
    [-2, 62, 4, 0],
    [-2, 58, 10, -2],
    [-4, 56, 14, -2],
    [-4, 54, 16, -2],
    [-6, 52, 20, -2],
    [-6, 46, 28, -4],
    [-4, 42, 30, -4],
    [-4, 36, 36, -4],
    [-4, 30, 42, -4],
    [-4, 28, 46, -6],
    [-2, 20, 52, -6],
    [-2, 16, 54, -4],
    [-2, 14, 56, -4],
    [-2, 10, 58, -2],
    [0, 4, 62, -2],
];

/// The current layer's picture format (H.8.1.4.1 `*Curr*` variables).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerFormat {
    /// `pic_width_in_luma_samples`.
    pub width: u32,
    /// `pic_height_in_luma_samples`.
    pub height: u32,
    /// `ChromaArrayType`.
    pub chroma_array_type: u8,
    /// `BitDepthY`.
    pub bit_depth_luma: u8,
    /// `BitDepthC`.
    pub bit_depth_chroma: u8,
}

impl LayerFormat {
    /// The format of a decoded picture.
    #[must_use]
    pub fn of(pic: &Picture) -> Self {
        Self {
            width: pic.width_luma() as u32,
            height: pic.height_luma() as u32,
            chroma_array_type: pic.chroma_array_type(),
            bit_depth_luma: pic.bit_depth_luma(),
            bit_depth_chroma: pic.bit_depth_chroma(),
        }
    }
}

/// The H.8.1.4.1 derived geometry of one inter-layer reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IlRefGeometry {
    cur: LayerFormat,
    rl: LayerFormat,
    /// `RefLayerRegion{Left,Top,Right,Bottom}Offset` (luma samples).
    ref_region: [i32; 4],
    /// `ScaledRefLayer{Left,Top,Right,Bottom}Offset` (luma samples).
    scaled: [i32; 4],
    /// `RefLayerRegionWidthInSamplesY` / `...HeightInSamplesY`.
    ref_region_wh: (i32, i32),
    /// `ScaledRefRegionWidthInSamplesY` / `...HeightInSamplesY`.
    scaled_wh: (i32, i32),
    /// `SpatialScaleFactor{Hor,Ver}Y`, `{Hor,Ver}C` (16.16).
    scale_y: (i64, i64),
    scale_c: (i64, i64),
    /// `Phase{Hor,Ver}Y`, `Phase{Hor,Ver}C`.
    phase_y: (i32, i32),
    phase_c: (i32, i32),
}

/// Why an inter-layer reference cannot be derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IlRefError {
    /// A reference / scaled region has a non-positive size, or the
    /// scaled region is smaller than the reference region (H.8.1.4.1
    /// conformance requirements).
    BadRegion,
    /// The reference layer's bit depth exceeds the current layer's.
    BitDepth,
}

impl core::fmt::Display for IlRefError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadRegion => f.write_str("inter-layer reference region geometry is invalid"),
            Self::BitDepth => f.write_str(
                "inter-layer reference layer has a higher bit depth than the current layer",
            ),
        }
    }
}

impl std::error::Error for IlRefError {}

impl IlRefGeometry {
    /// H.8.1.4.1 — derive the geometry of the reference `rl` used by a
    /// picture of format `cur`, with the PPS reference layer location
    /// offsets `offsets` for that layer (`None` = all inferred 0).
    ///
    /// # Errors
    /// [`IlRefError`] when the conformance requirements on the regions
    /// or bit depths are violated.
    pub fn derive(
        cur: LayerFormat,
        rl: LayerFormat,
        offsets: Option<&RefLocOffset>,
    ) -> Result<Self, IlRefError> {
        let (sub_w_cur, sub_h_cur) = sub_wh_c(cur.chroma_array_type);
        let (sub_w_rl, sub_h_rl) = sub_wh_c(rl.chroma_array_type);
        let (sub_w_cur, sub_h_cur) = (sub_w_cur as i32, sub_h_cur as i32);
        let (sub_w_rl, sub_h_rl) = (sub_w_rl as i32, sub_h_rl as i32);
        let o = offsets.copied().unwrap_or(RefLocOffset {
            phase_hor_chroma_plus8: 8,
            ..RefLocOffset::default()
        });
        // H-2 .. H-5.
        let ref_region = [
            o.ref_region_left_offset * sub_w_rl,
            o.ref_region_top_offset * sub_h_rl,
            o.ref_region_right_offset * sub_w_rl,
            o.ref_region_bottom_offset * sub_h_rl,
        ];
        // H-6 / H-7.
        let ref_w = rl.width as i32 - ref_region[0] - ref_region[2];
        let ref_h = rl.height as i32 - ref_region[1] - ref_region[3];
        // H-12 .. H-15.
        let scaled = [
            o.scaled_ref_layer_left_offset * sub_w_cur,
            o.scaled_ref_layer_top_offset * sub_h_cur,
            o.scaled_ref_layer_right_offset * sub_w_cur,
            o.scaled_ref_layer_bottom_offset * sub_h_cur,
        ];
        // H-16 / H-17.
        let sc_w = cur.width as i32 - scaled[0] - scaled[2];
        let sc_h = cur.height as i32 - scaled[1] - scaled[3];
        if ref_w <= 0 || ref_h <= 0 || sc_w <= 0 || sc_h <= 0 || sc_w < ref_w || sc_h < ref_h {
            return Err(IlRefError::BadRegion);
        }
        if rl.bit_depth_luma > cur.bit_depth_luma || rl.bit_depth_chroma > cur.bit_depth_chroma {
            return Err(IlRefError::BitDepth);
        }
        // H-18 .. H-21.
        let scale = |ref_n: i64, sc_n: i64| ((ref_n << 16) + (sc_n >> 1)) / sc_n;
        let scale_y = (
            scale(i64::from(ref_w), i64::from(sc_w)),
            scale(i64::from(ref_h), i64::from(sc_h)),
        );
        let scale_c = (
            scale(
                i64::from(ref_w / sub_w_rl),
                i64::from(sc_w / sub_w_cur).max(1),
            ),
            scale(
                i64::from(ref_h / sub_h_rl),
                i64::from(sc_h / sub_h_cur).max(1),
            ),
        );
        // H-22 .. H-25, with the F.7.4.3.3.4 inference of an absent
        // phase_ver_chroma_plus8.
        let phase_ver_chroma_plus8 = match o.phase_ver_chroma_plus8 {
            Some(v) => i32::from(v),
            None if cur.chroma_array_type == 3 => 8,
            None => (4 * sc_h + ref_h / 2) / ref_h + 4,
        };
        Ok(Self {
            cur,
            rl,
            ref_region,
            scaled,
            ref_region_wh: (ref_w, ref_h),
            scaled_wh: (sc_w, sc_h),
            scale_y,
            scale_c,
            phase_y: (i32::from(o.phase_hor_luma), i32::from(o.phase_ver_luma)),
            phase_c: (
                i32::from(o.phase_hor_chroma_plus8) - 8,
                phase_ver_chroma_plus8 - 8,
            ),
        })
    }

    /// `equalPictureSizeAndOffsetFlag` (H.8.1.4.1).
    #[must_use]
    pub fn equal_picture_size_and_offset(&self) -> bool {
        self.cur.width == self.rl.width
            && self.cur.height == self.rl.height
            && self.scaled == self.ref_region
            && self.phase_y == (0, 0)
            && self.phase_c == (0, 0)
    }

    /// `true` when `ilRefPic` is `rlPic` itself: equal size / offsets /
    /// phases, bit depths and chroma format (no resampling needed).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.equal_picture_size_and_offset()
            && self.cur.bit_depth_luma == self.rl.bit_depth_luma
            && self.cur.bit_depth_chroma == self.rl.bit_depth_chroma
            && self.cur.chroma_array_type == self.rl.chroma_array_type
    }

    /// H.8.1.4.2.4 — the reference layer location of current sample
    /// `(x, y)` in 1/16 units: `(xRef16, yRef16)`.
    #[inline]
    fn ref16(&self, chroma: bool, x: i32, y: i32) -> (i64, i64) {
        let (sub_w_cur, sub_h_cur) = sub_wh_c(self.cur.chroma_array_type);
        let (sub_w_rl, sub_h_rl) = sub_wh_c(self.rl.chroma_array_type);
        let (dw_cur, dh_cur, dw_rl, dh_rl) = if chroma {
            (
                sub_w_cur as i32,
                sub_h_cur as i32,
                sub_w_rl as i32,
                sub_h_rl as i32,
            )
        } else {
            (1, 1, 1, 1)
        };
        // H-53 .. H-56.
        let cur_off_left = self.scaled[0] / dw_cur;
        let cur_off_top = self.scaled[1] / dh_cur;
        let ref_off_left = i64::from(self.ref_region[0] / dw_rl) << 4;
        let ref_off_top = i64::from(self.ref_region[1] / dh_rl) << 4;
        // H-57 .. H-60.
        let (phase_hor, phase_ver) = if chroma { self.phase_c } else { self.phase_y };
        let (scale_hor, scale_ver) = if chroma { self.scale_c } else { self.scale_y };
        // H-61 / H-62.
        let add_hor = -((scale_hor * i64::from(phase_hor) + 8) >> 4);
        let add_ver = -((scale_ver * i64::from(phase_ver) + 8) >> 4);
        // H-63 / H-64.
        let x_ref16 =
            ((i64::from(x - cur_off_left) * scale_hor + add_hor + (1 << 11)) >> 12) + ref_off_left;
        let y_ref16 =
            ((i64::from(y - cur_off_top) * scale_ver + add_ver + (1 << 11)) >> 12) + ref_off_top;
        (x_ref16, y_ref16)
    }
}

/// One resampled plane: the separable H.8.1.4.2.2 / H.8.1.4.2.3
/// process over the whole current plane.
#[allow(clippy::too_many_arguments)]
fn resample_plane(
    src: &[i32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    bd_ref: u8,
    bd_cur: u8,
    chroma: bool,
    geom: &IlRefGeometry,
) -> Vec<i32> {
    let taps: usize = if chroma { 4 } else { 8 };
    let half: i32 = if chroma { 1 } else { 3 };
    let coef = |phase: usize, k: usize| -> i32 {
        if chroma {
            CHROMA_FILTER[phase][k]
        } else {
            LUMA_FILTER[phase][k]
        }
    };
    // H-33 .. H-35 / H-45 .. H-47.
    let shift1 = i64::from(bd_ref) - 8;
    let shift2 = 20 - i64::from(bd_cur);
    let offset = 1i64 << (shift2 - 1);
    let max_val = (1i64 << bd_cur) - 1;
    let sw = src_w as i32;
    let sh = src_h as i32;
    // Per current column: xRef / xPhase (the same for every row).
    let cols: Vec<(i32, usize)> = (0..dst_w as i32)
        .map(|x| {
            let (x16, _) = geom.ref16(chroma, x, 0);
            ((x16 >> 4) as i32, (x16 & 15) as usize)
        })
        .collect();
    // Horizontal pass over every reference row (H-38 / H-50 per row).
    let mut temp = vec![0i64; src_h * dst_w];
    for y in 0..src_h {
        let row = &src[y * src_w..(y + 1) * src_w];
        let trow = &mut temp[y * dst_w..(y + 1) * dst_w];
        for (x, &(x_ref, phase)) in cols.iter().enumerate() {
            let mut acc = 0i64;
            for k in 0..taps {
                let xs = (x_ref - half + k as i32).clamp(0, sw - 1) as usize;
                acc += i64::from(coef(phase, k)) * i64::from(row[xs]);
            }
            trow[x] = acc >> shift1;
        }
    }
    // Vertical pass (H-39 / H-51).
    let mut out = vec![0i32; dst_w * dst_h];
    for y in 0..dst_h as i32 {
        let (_, y16) = geom.ref16(chroma, 0, y);
        let y_ref = (y16 >> 4) as i32;
        let phase = (y16 & 15) as usize;
        let orow = &mut out[y as usize * dst_w..(y as usize + 1) * dst_w];
        for x in 0..dst_w {
            let mut acc = offset;
            for n in 0..taps {
                let ys = (y_ref + n as i32 - half).clamp(0, sh - 1) as usize;
                acc += i64::from(coef(phase, n)) * temp[ys * dst_w + x];
            }
            orow[x] = (acc >> shift2).clamp(0, max_val) as i32;
        }
    }
    out
}

/// H.8.1.4.2 — the resampled sample arrays of `rl` at the current
/// layer's format.
#[must_use]
pub fn resample_picture(geom: &IlRefGeometry, rl: &Picture) -> Picture {
    let cur = geom.cur;
    let mut out = Picture::new(
        cur.width as usize,
        cur.height as usize,
        cur.chroma_array_type,
        cur.bit_depth_luma,
        cur.bit_depth_chroma,
    );
    let (src_w, src_h) = rl.plane_dims(Plane::Luma);
    let luma = resample_plane(
        rl.plane(Plane::Luma),
        src_w,
        src_h,
        cur.width as usize,
        cur.height as usize,
        geom.rl.bit_depth_luma,
        cur.bit_depth_luma,
        false,
        geom,
    );
    out.plane_mut(Plane::Luma).0.copy_from_slice(&luma);
    if cur.chroma_array_type != 0 {
        let (dst_cw, dst_ch) = out.plane_dims(Plane::Cb);
        for plane in [Plane::Cb, Plane::Cr] {
            let (scw, sch) = rl.plane_dims(plane);
            let res = if geom.rl.chroma_array_type == 0 {
                // A monochrome reference: chroma at mid-grey.
                vec![1i32 << (cur.bit_depth_chroma - 1); dst_cw * dst_ch]
            } else {
                resample_plane(
                    rl.plane(plane),
                    scw,
                    sch,
                    dst_cw,
                    dst_ch,
                    geom.rl.bit_depth_chroma,
                    cur.bit_depth_chroma,
                    true,
                    geom,
                )
            };
            out.plane_mut(plane).0.copy_from_slice(&res);
        }
    }
    out
}

/// H.8.1.4.3 — the resampled motion and mode parameters of `rl_motion`
/// (the reference layer picture's field) at the current layer's
/// geometry, per 16x16 block.
#[must_use]
pub fn resample_motion(geom: &IlRefGeometry, rl_motion: &MotionField) -> MotionField {
    let (cur_w, cur_h) = (geom.cur.width as usize, geom.cur.height as usize);
    let mut out = MotionField::new(cur_w, cur_h);
    let (ref_w, ref_h) = geom.ref_region_wh;
    let (sc_w, sc_h) = geom.scaled_wh;
    // H-74 / H-77.
    let scale_mv = |sc: i32, rf: i32| -> Option<i64> {
        (sc != rf).then(|| {
            ((((i64::from(sc)) << 8) + i64::from(rf >> 1)) / i64::from(rf)).clamp(-4096, 4095)
        })
    };
    let scale_x = scale_mv(sc_w, ref_w);
    let scale_y = scale_mv(sc_h, ref_h);
    // H-75 / H-78: Sign( scale * mv ) * ( ( Abs( scale * mv ) + 127 ) >> 8 ).
    let apply = |scale: Option<i64>, v: i32| -> i32 {
        match scale {
            None => v,
            Some(s) => {
                let p = s * i64::from(v);
                let m = p.signum() * ((p.abs() + 127) >> 8);
                m.clamp(-32768, 32767) as i32
            }
        }
    };
    for yb in 0..cur_h.div_ceil(16) {
        for xb in 0..cur_w.div_ceil(16) {
            let (xp, yp) = ((xb * 16) as i32, (yb * 16) as i32);
            // H-65 .. H-68.
            let x_ref = ((i64::from(xp + 8 - geom.scaled[0]) * geom.scale_y.0 + (1 << 15)) >> 16)
                as i32
                + geom.ref_region[0];
            let y_ref = ((i64::from(yp + 8 - geom.scaled[1]) * geom.scale_y.1 + (1 << 15)) >> 16)
                as i32
                + geom.ref_region[1];
            // H-69 / H-70.
            let x_rl = ((x_ref + 4) >> 4) << 4;
            let y_rl = ((y_ref + 4) >> 4) << 4;
            let cell = if x_rl < 0
                || x_rl >= geom.rl.width as i32
                || y_rl < 0
                || y_rl >= geom.rl.height as i32
            {
                MotionCell {
                    is_intra: true,
                    ..MotionCell::default()
                }
            } else {
                let c = rl_motion.cell_at(x_rl as usize, y_rl as usize);
                if c.is_intra {
                    MotionCell {
                        is_intra: true,
                        ..MotionCell::default()
                    }
                } else {
                    MotionCell {
                        mv_l0: [apply(scale_x, c.mv_l0[0]), apply(scale_y, c.mv_l0[1])],
                        mv_l1: [apply(scale_x, c.mv_l1[0]), apply(scale_y, c.mv_l1[1])],
                        ..c
                    }
                }
            };
            out.fill_rect(xb * 16, yb * 16, 16, 16, cell);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::Mv;

    fn fmt(w: u32, h: u32, cat: u8, bd: u8) -> LayerFormat {
        LayerFormat {
            width: w,
            height: h,
            chroma_array_type: cat,
            bit_depth_luma: bd,
            bit_depth_chroma: bd,
        }
    }

    #[test]
    fn identity_geometry_needs_no_resampling() {
        let g = IlRefGeometry::derive(fmt(64, 64, 1, 8), fmt(64, 64, 1, 8), None).unwrap();
        assert!(g.equal_picture_size_and_offset());
        assert!(g.is_identity());
        assert_eq!(g.scale_y, (1 << 16, 1 << 16));
    }

    #[test]
    fn two_to_one_scale_factors_and_phase_inference() {
        let g = IlRefGeometry::derive(fmt(128, 96, 1, 8), fmt(64, 48, 1, 8), None).unwrap();
        assert!(!g.equal_picture_size_and_offset());
        assert_eq!(g.scale_y, (1 << 15, 1 << 15));
        assert_eq!(g.scale_c, (1 << 15, 1 << 15));
        // F.7.4.3.3.4: ( 4 * 96 + 24 ) / 48 + 4 = 12 → PhaseVerC = 4.
        assert_eq!(g.phase_c, (0, 4));
        // A 4:4:4 current picture infers 8 → PhaseVerC = 0.
        let g444 = IlRefGeometry::derive(fmt(128, 96, 3, 8), fmt(64, 48, 3, 8), None).unwrap();
        assert_eq!(g444.phase_c, (0, 0));
    }

    #[test]
    fn invalid_regions_are_rejected() {
        assert_eq!(
            IlRefGeometry::derive(fmt(32, 32, 1, 8), fmt(64, 64, 1, 8), None),
            Err(IlRefError::BadRegion)
        );
        assert_eq!(
            IlRefGeometry::derive(fmt(64, 64, 1, 8), fmt(64, 64, 1, 10), None),
            Err(IlRefError::BitDepth)
        );
    }

    #[test]
    fn constant_picture_resamples_to_the_same_constant() {
        let mut rl = Picture::new(32, 32, 1, 8, 8);
        rl.plane_mut(Plane::Luma).0.fill(100);
        rl.plane_mut(Plane::Cb).0.fill(60);
        rl.plane_mut(Plane::Cr).0.fill(200);
        let g = IlRefGeometry::derive(fmt(64, 64, 1, 8), LayerFormat::of(&rl), None).unwrap();
        let out = resample_picture(&g, &rl);
        assert!(out.plane(Plane::Luma).iter().all(|&v| v == 100));
        assert!(out.plane(Plane::Cb).iter().all(|&v| v == 60));
        assert!(out.plane(Plane::Cr).iter().all(|&v| v == 200));
    }

    #[test]
    fn bit_depth_only_resampling_is_a_left_shift() {
        let mut rl = Picture::new(16, 16, 1, 8, 8);
        for (i, v) in rl.plane_mut(Plane::Luma).0.iter_mut().enumerate() {
            *v = (i % 256) as i32;
        }
        let g = IlRefGeometry::derive(fmt(16, 16, 1, 10), LayerFormat::of(&rl), None).unwrap();
        assert!(g.equal_picture_size_and_offset());
        assert!(!g.is_identity());
        let out = resample_picture(&g, &rl);
        for (o, r) in out.plane(Plane::Luma).iter().zip(rl.plane(Plane::Luma)) {
            assert_eq!(*o, r << 2);
        }
    }

    #[test]
    fn horizontal_ramp_upsamples_monotonically_with_exact_midpoints() {
        // A linear ramp 0, 8, 16, .. through the phase-8 luma filter
        // (symmetric taps summing to 64) lands exactly on the midpoints.
        let mut rl = Picture::new(16, 4, 0, 8, 8);
        {
            let (p, stride) = rl.plane_mut(Plane::Luma);
            for y in 0..4 {
                for x in 0..16 {
                    p[y * stride + x] = (x * 8) as i32;
                }
            }
        }
        let g = IlRefGeometry::derive(fmt(32, 4, 0, 8), LayerFormat::of(&rl), None).unwrap();
        let out = resample_picture(&g, &rl);
        let row: Vec<i32> = out.plane(Plane::Luma)[..32].to_vec();
        assert!(row.windows(2).all(|w| w[0] <= w[1]), "{row:?}");
        // 2:1 with phase 0: eq. H-63 gives xRef16 = 8 * x, so even
        // outputs reproduce the reference samples and odd ones (phase
        // 8, a symmetric filter) the exact midpoints of the ramp.
        assert_eq!(row[16], 64, "{row:?}");
        assert_eq!(row[15], 60, "{row:?}");
        assert_eq!(row[17], 68, "{row:?}");
    }

    #[test]
    fn motion_resampling_scales_vectors_and_marks_outside_intra() {
        let mut rl = MotionField::new(32, 32);
        rl.fill_rect(
            0,
            0,
            32,
            32,
            MotionCell {
                is_intra: false,
                pred_flag_l0: true,
                ref_poc_l0: 3,
                mv_l0: [16, -8],
                ..MotionCell::default()
            },
        );
        let g = IlRefGeometry::derive(fmt(64, 64, 1, 8), fmt(32, 32, 1, 8), None).unwrap();
        let out = resample_motion(&g, &rl);
        let c = out.cell_at(0, 0);
        assert!(!c.is_intra);
        assert_eq!(c.ref_poc_l0, 3);
        assert_eq!(c.mv_l0, [32, -16]);
        // The identity geometry copies the field.
        let gi = IlRefGeometry::derive(fmt(32, 32, 1, 8), fmt(32, 32, 1, 8), None).unwrap();
        let same = resample_motion(&gi, &rl);
        let expect: Mv = [16, -8];
        assert_eq!(same.cell_at(20, 20).mv_l0, expect);
    }
}
