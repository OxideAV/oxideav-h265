//! Decoded-picture sample storage — the §8 reconstruction target.
//!
//! A [`Picture`] holds the three reconstructed sample planes (luma `SL`
//! and the two chroma planes `SCb` / `SCr`) sized from the active SPS
//! geometry and `ChromaArrayType`. Reconstruction (§8.4 intra / §8.5
//! inter sample prediction, §8.6 residual add, §8.7 in-loop filters)
//! writes into these planes; the DPB and any output cropping read out of
//! them.
//!
//! Samples are stored one `u16` per pixel — every bit depth the
//! Recommendation allows (8..=16) fits, and the stored values are
//! always already clipped to `[0, (1 << bitDepth) − 1]` by the
//! §8.6.7 / §8.7 clips. The sample accessors widen to `i32` so the
//! prediction + residual arithmetic of §8.4.4 / §8.6.2 (which works in
//! the full `i32` range before the final `Clip1Y` / `Clip1C` clip)
//! stays in one type.
//!
//! Each plane lives behind a shared, copy-on-write handle: cloning a
//! `Picture` shares the sample buffers (the DPB's reference copy and
//! the output frame of the same decoded picture are one allocation),
//! and the first mutation of a shared plane copies it. A plane handed
//! out through [`Picture::plane_mut`] is therefore unique for the
//! duration of the borrow; hot reconstruction loops take the plane
//! once and index it rather than calling [`Picture::set_sample`] per
//! sample.

use std::sync::Arc;

/// One reconstructed picture: the luma plane and (unless monochrome) the
/// two chroma planes, each row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Picture {
    /// `pic_width_in_luma_samples`.
    width_luma: usize,
    /// `pic_height_in_luma_samples`.
    height_luma: usize,
    /// Chroma plane width (`width_luma >> SubWidthC`), 0 if monochrome.
    width_chroma: usize,
    /// Chroma plane height (`height_luma >> SubHeightC`), 0 if monochrome.
    height_chroma: usize,
    /// `ChromaArrayType` (0 = monochrome, 1 = 4:2:0, 2 = 4:2:2,
    /// 3 = 4:4:4).
    chroma_array_type: u8,
    /// `BitDepthY`.
    bit_depth_luma: u8,
    /// `BitDepthC`.
    bit_depth_chroma: u8,
    /// The stored rectangle of each plane (`(x0, y0, width)` in plane
    /// samples): the whole plane for a picture, a CTB row or tile for a
    /// band of the parallel decoders. Sample `(x, y)` of a plane lives at
    /// `(y - y0) * width + (x - x0)` of its buffer.
    band_luma: (usize, usize, usize),
    /// See [`Self::band_luma`].
    band_chroma: (usize, usize, usize),
    /// `SL[ x ][ y ]`, row-major.
    luma: Arc<Vec<u16>>,
    /// `SCb[ x ][ y ]`, row-major; empty when monochrome.
    cb: Arc<Vec<u16>>,
    /// `SCr[ x ][ y ]`, row-major; empty when monochrome.
    cr: Arc<Vec<u16>>,
}

/// The three colour components addressable in a [`Picture`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plane {
    /// The luma plane `SL`.
    Luma,
    /// The first chroma plane `SCb`.
    Cb,
    /// The second chroma plane `SCr`.
    Cr,
}

/// `(SubWidthC, SubHeightC)` from Table 6-1 for a `ChromaArrayType`.
/// `ChromaArrayType == 0` (monochrome) has no chroma planes; this
/// returns `(2, 2)` for it purely so callers that never touch chroma do
/// not special-case it (the chroma planes are zero-sized in that case).
#[must_use]
pub fn sub_wh_c(chroma_array_type: u8) -> (usize, usize) {
    match chroma_array_type {
        // 4:2:0.
        1 => (2, 2),
        // 4:2:2.
        2 => (2, 1),
        // 4:4:4.
        3 => (1, 1),
        // monochrome (no chroma) — never indexed.
        _ => (2, 2),
    }
}

impl Picture {
    /// Allocate a black (all-zero) picture of the given geometry. The
    /// chroma planes are sized by `chroma_array_type` per Table 6-1; a
    /// monochrome picture (`chroma_array_type == 0`) carries no chroma
    /// samples.
    #[must_use]
    pub fn new(
        width_luma: usize,
        height_luma: usize,
        chroma_array_type: u8,
        bit_depth_luma: u8,
        bit_depth_chroma: u8,
    ) -> Self {
        let (width_chroma, height_chroma) = if chroma_array_type == 0 {
            (0, 0)
        } else {
            let (sw, sh) = sub_wh_c(chroma_array_type);
            (width_luma / sw, height_luma / sh)
        };
        let luma = Arc::new(vec![0u16; width_luma * height_luma]);
        let cb = Arc::new(vec![0u16; width_chroma * height_chroma]);
        let cr = Arc::new(vec![0u16; width_chroma * height_chroma]);
        Self {
            width_luma,
            height_luma,
            width_chroma,
            height_chroma,
            chroma_array_type,
            bit_depth_luma,
            bit_depth_chroma,
            band_luma: (0, 0, width_luma),
            band_chroma: (0, 0, width_chroma),
            luma,
            cb,
            cr,
        }
    }

    /// A horizontal **band** of a `width_luma × height_luma` picture:
    /// only luma rows `y_origin_luma .. y_origin_luma + band_rows_luma`
    /// (and the chroma rows covering them) are stored, but the picture
    /// reports the full geometry and every accessor takes absolute
    /// picture coordinates — the wavefront decoder reconstructs one CTB
    /// row (plus the line above it) into such a band. `plane` /
    /// `plane_mut` hand out the band's storage, whose first row is
    /// picture row `y_origin`.
    ///
    /// # Panics
    /// Panics if the band does not lie inside the picture or
    /// `y_origin_luma` is not a multiple of `SubHeightC`.
    #[must_use]
    pub fn new_band(
        width_luma: usize,
        height_luma: usize,
        chroma_array_type: u8,
        bit_depth_luma: u8,
        bit_depth_chroma: u8,
        y_origin_luma: usize,
        band_rows_luma: usize,
    ) -> Self {
        Self::new_rect_band(
            width_luma,
            height_luma,
            chroma_array_type,
            bit_depth_luma,
            bit_depth_chroma,
            (0, y_origin_luma, width_luma, band_rows_luma),
        )
    }

    /// A rectangular **band**: only the luma rectangle `rect = (x0, y0,
    /// w, h)` (and the chroma samples covering it) is stored — a tile of
    /// the tile-parallel decoder. As [`Self::new_band`], every accessor
    /// takes absolute picture coordinates; `plane` / `plane_mut` hand
    /// out the band's `w`-wide storage.
    ///
    /// # Panics
    /// Panics if the rectangle does not lie inside the picture or its
    /// origin is not on a chroma sample.
    #[must_use]
    pub fn new_rect_band(
        width_luma: usize,
        height_luma: usize,
        chroma_array_type: u8,
        bit_depth_luma: u8,
        bit_depth_chroma: u8,
        rect: (usize, usize, usize, usize),
    ) -> Self {
        let (x0, y0, w, h) = rect;
        assert!(
            x0 + w <= width_luma && y0 + h <= height_luma,
            "band past the picture"
        );
        let (sw, sh) = sub_wh_c(chroma_array_type);
        assert!(
            x0 % sw == 0 && y0 % sh == 0,
            "band origin on a chroma sample"
        );
        let (width_chroma, height_chroma, band_chroma, rows_chroma) = if chroma_array_type == 0 {
            (0, 0, (0, 0, 0), 0)
        } else {
            (
                width_luma / sw,
                height_luma / sh,
                (x0 / sw, y0 / sh, w.div_ceil(sw)),
                h.div_ceil(sh),
            )
        };
        Self {
            width_luma,
            height_luma,
            width_chroma,
            height_chroma,
            chroma_array_type,
            bit_depth_luma,
            bit_depth_chroma,
            band_luma: (x0, y0, w),
            band_chroma,
            luma: Arc::new(vec![0u16; w * h]),
            cb: Arc::new(vec![0u16; band_chroma.2 * rows_chroma]),
            cr: Arc::new(vec![0u16; band_chroma.2 * rows_chroma]),
        }
    }

    /// All three plane buffers for writing at once (each copied first
    /// if shared) — the row-parallel paths split them into disjoint row
    /// chunks.
    pub fn planes_mut(&mut self) -> (&mut [u16], &mut [u16], &mut [u16]) {
        (
            Arc::make_mut(&mut self.luma).as_mut_slice(),
            Arc::make_mut(&mut self.cb).as_mut_slice(),
            Arc::make_mut(&mut self.cr).as_mut_slice(),
        )
    }

    /// The stored rectangle of `plane` as `(x0, y0, width)` (the whole
    /// plane unless this is a band).
    #[inline]
    #[must_use]
    pub fn band(&self, plane: Plane) -> (usize, usize, usize) {
        match plane {
            Plane::Luma => self.band_luma,
            Plane::Cb | Plane::Cr => self.band_chroma,
        }
    }

    /// The first stored row of `plane` (0 unless this is a band).
    #[inline]
    #[must_use]
    pub fn y_origin(&self, plane: Plane) -> usize {
        self.band(plane).1
    }

    /// The first stored column of `plane` (0 unless this is a
    /// rectangular band).
    #[inline]
    #[must_use]
    pub fn x_origin(&self, plane: Plane) -> usize {
        self.band(plane).0
    }

    /// Number of stored rows of `plane` (the plane height unless this
    /// is a band).
    #[inline]
    #[must_use]
    pub fn stored_rows(&self, plane: Plane) -> usize {
        let (buf, stride) = self.plane_slice(plane);
        buf.len().checked_div(stride).unwrap_or(0)
    }

    /// The stored part of row `y` (absolute picture row) of `plane`:
    /// the samples from column [`Self::x_origin`] on.
    ///
    /// # Panics
    /// Panics if the row is not stored.
    #[inline]
    #[must_use]
    pub fn row(&self, plane: Plane, y: usize) -> &[u16] {
        let (buf, stride) = self.plane_slice(plane);
        let r = y - self.y_origin(plane);
        &buf[r * stride..(r + 1) * stride]
    }

    /// The plane's buffer for writing (copied first if shared) with its
    /// stride and the stored rectangle's origin `(x0, y0)` — block
    /// writers take this once and index as `(y - y0) * stride + (x - x0)`.
    #[inline]
    pub fn plane_mut_origin(&mut self, plane: Plane) -> (&mut [u16], usize, (usize, usize)) {
        let (x0, y0, _) = self.band(plane);
        let (buf, stride) = self.plane_slice_mut(plane);
        (buf, stride, (x0, y0))
    }

    /// The stored part of row `y` (absolute picture row) of `plane`, for
    /// writing (from column [`Self::x_origin`] on).
    ///
    /// # Panics
    /// Panics if the row is not stored.
    #[inline]
    pub fn row_mut(&mut self, plane: Plane, y: usize) -> &mut [u16] {
        let r = y - self.y_origin(plane);
        let (buf, stride) = self.plane_slice_mut(plane);
        &mut buf[r * stride..(r + 1) * stride]
    }

    /// Build a picture around already-reconstructed planes (row-major,
    /// `width * height` samples each; the chroma vectors are ignored
    /// for a monochrome picture).
    ///
    /// # Panics
    /// Panics if a plane's length does not match the geometry.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn from_planes(
        width_luma: usize,
        height_luma: usize,
        chroma_array_type: u8,
        bit_depth_luma: u8,
        bit_depth_chroma: u8,
        luma: Vec<u16>,
        cb: Vec<u16>,
        cr: Vec<u16>,
    ) -> Self {
        let (width_chroma, height_chroma) = if chroma_array_type == 0 {
            (0, 0)
        } else {
            let (sw, sh) = sub_wh_c(chroma_array_type);
            (width_luma / sw, height_luma / sh)
        };
        assert_eq!(luma.len(), width_luma * height_luma, "luma plane length");
        let (cb, cr) = if chroma_array_type == 0 {
            (Vec::new(), Vec::new())
        } else {
            assert_eq!(cb.len(), width_chroma * height_chroma, "cb plane length");
            assert_eq!(cr.len(), width_chroma * height_chroma, "cr plane length");
            (cb, cr)
        };
        Self {
            width_luma,
            height_luma,
            width_chroma,
            height_chroma,
            chroma_array_type,
            bit_depth_luma,
            bit_depth_chroma,
            band_luma: (0, 0, width_luma),
            band_chroma: (0, 0, width_chroma),
            luma: Arc::new(luma),
            cb: Arc::new(cb),
            cr: Arc::new(cr),
        }
    }

    /// Take the three planes out (row-major `u16`, `Y` / `Cb` / `Cr`;
    /// the chroma vectors are empty for monochrome). A plane still
    /// shared with another `Picture` is copied; a uniquely held one is
    /// moved without a copy — the ownership hand-off a consumer uses
    /// to wrap the decoded samples in its own frame type.
    #[must_use]
    pub fn into_planes(self) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let take = |a: Arc<Vec<u16>>| Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone());
        (take(self.luma), take(self.cb), take(self.cr))
    }

    /// Whether the sample planes are shared with another `Picture`
    /// (a clone that has not been written since).
    #[must_use]
    pub fn is_shared(&self) -> bool {
        Arc::strong_count(&self.luma) > 1
            || Arc::strong_count(&self.cb) > 1
            || Arc::strong_count(&self.cr) > 1
    }

    /// The §7.4.3.2.1 output-cropped copy of this picture: the luma
    /// rectangle starting at `(x0, y0)` of `width x height` samples,
    /// with the chroma planes cut at the matching `SubWidthC` /
    /// `SubHeightC`-scaled positions (Table 6-1). `x0 + width` /
    /// `y0 + height` are clamped to the coded picture; a window
    /// covering the whole picture returns an identical copy.
    #[must_use]
    pub fn cropped(&self, x0: usize, y0: usize, width: usize, height: usize) -> Self {
        assert_eq!(
            self.band_luma,
            (0, 0, self.width_luma),
            "cropping a band picture"
        );
        let x0 = x0.min(self.width_luma);
        let y0 = y0.min(self.height_luma);
        let width = width.min(self.width_luma - x0);
        let height = height.min(self.height_luma - y0);
        if x0 == 0 && y0 == 0 && width == self.width_luma && height == self.height_luma {
            return self.clone();
        }
        let (sw, sh) = sub_wh_c(self.chroma_array_type);
        let (cx0, cy0, cw, ch) = if self.chroma_array_type == 0 {
            (0, 0, 0, 0)
        } else {
            (x0 / sw, y0 / sh, width / sw, height / sh)
        };
        let cut = |src: &[u16], stride: usize, x0: usize, y0: usize, w: usize, h: usize| {
            let mut out = Vec::with_capacity(w * h);
            for y in y0..y0 + h {
                out.extend_from_slice(&src[y * stride + x0..y * stride + x0 + w]);
            }
            Arc::new(out)
        };
        Self {
            width_luma: width,
            height_luma: height,
            width_chroma: cw,
            height_chroma: ch,
            chroma_array_type: self.chroma_array_type,
            bit_depth_luma: self.bit_depth_luma,
            bit_depth_chroma: self.bit_depth_chroma,
            band_luma: (0, 0, width),
            band_chroma: (0, 0, cw),
            luma: cut(&self.luma, self.width_luma, x0, y0, width, height),
            cb: cut(&self.cb, self.width_chroma, cx0, cy0, cw, ch),
            cr: cut(&self.cr, self.width_chroma, cx0, cy0, cw, ch),
        }
    }

    /// `pic_width_in_luma_samples`.
    #[inline]
    #[must_use]
    pub fn width_luma(&self) -> usize {
        self.width_luma
    }

    /// `pic_height_in_luma_samples`.
    #[inline]
    #[must_use]
    pub fn height_luma(&self) -> usize {
        self.height_luma
    }

    /// `ChromaArrayType`.
    #[inline]
    #[must_use]
    pub fn chroma_array_type(&self) -> u8 {
        self.chroma_array_type
    }

    /// `BitDepthY`.
    #[inline]
    #[must_use]
    pub fn bit_depth_luma(&self) -> u8 {
        self.bit_depth_luma
    }

    /// `BitDepthC`.
    #[inline]
    #[must_use]
    pub fn bit_depth_chroma(&self) -> u8 {
        self.bit_depth_chroma
    }

    /// Plane dimensions `(width, height)` in samples for `plane`.
    #[must_use]
    pub fn plane_dims(&self, plane: Plane) -> (usize, usize) {
        match plane {
            Plane::Luma => (self.width_luma, self.height_luma),
            Plane::Cb | Plane::Cr => (self.width_chroma, self.height_chroma),
        }
    }

    /// `BitDepth` of `plane`.
    #[must_use]
    pub fn bit_depth(&self, plane: Plane) -> u8 {
        match plane {
            Plane::Luma => self.bit_depth_luma,
            Plane::Cb | Plane::Cr => self.bit_depth_chroma,
        }
    }

    #[inline]
    fn plane_slice(&self, plane: Plane) -> (&[u16], usize) {
        match plane {
            Plane::Luma => (&self.luma, self.band_luma.2),
            Plane::Cb => (&self.cb, self.band_chroma.2),
            Plane::Cr => (&self.cr, self.band_chroma.2),
        }
    }

    /// The plane's buffer for writing: copies it first when another
    /// `Picture` still shares it (copy-on-write).
    #[inline]
    fn plane_slice_mut(&mut self, plane: Plane) -> (&mut [u16], usize) {
        match plane {
            Plane::Luma => (
                Arc::make_mut(&mut self.luma).as_mut_slice(),
                self.band_luma.2,
            ),
            Plane::Cb => (
                Arc::make_mut(&mut self.cb).as_mut_slice(),
                self.band_chroma.2,
            ),
            Plane::Cr => (
                Arc::make_mut(&mut self.cr).as_mut_slice(),
                self.band_chroma.2,
            ),
        }
    }

    /// Read one sample at `(x, y)` of `plane`.
    ///
    /// # Panics
    /// Panics if `(x, y)` lies outside the plane.
    #[inline]
    #[must_use]
    pub fn sample(&self, plane: Plane, x: usize, y: usize) -> i32 {
        let (x0, y0, _) = self.band(plane);
        let (buf, stride) = self.plane_slice(plane);
        i32::from(buf[(y - y0) * stride + (x - x0)])
    }

    /// Write one sample at `(x, y)` of `plane`. `v` must already be
    /// clipped to the plane's bit depth (it is stored as `u16`).
    ///
    /// This is the convenience path: every call checks the plane's
    /// sharing state. Block writers take [`Self::plane_mut`] once.
    ///
    /// # Panics
    /// Panics if `(x, y)` lies outside the plane.
    pub fn set_sample(&mut self, plane: Plane, x: usize, y: usize, v: i32) {
        let (x0, y0, _) = self.band(plane);
        let (buf, stride) = self.plane_slice_mut(plane);
        buf[(y - y0) * stride + (x - x0)] = v as u16;
    }

    /// Borrow the raw row-major plane buffer (read-only). For a band
    /// (see [`Self::new_band`]) the first stored row is picture row
    /// [`Self::y_origin`].
    #[inline]
    #[must_use]
    pub fn plane(&self, plane: Plane) -> &[u16] {
        self.plane_slice(plane).0
    }

    /// Borrow the raw row-major plane buffer + its row stride (mutable;
    /// a plane shared with another `Picture` is copied first).
    ///
    /// Used by the §8.7.2 deblocking driver to wrap a component plane in a
    /// [`crate::deblock::SamplePlane`] for in-place edge filtering, and by
    /// every block-writing reconstruction step.
    pub fn plane_mut(&mut self, plane: Plane) -> (&mut [u16], usize) {
        self.plane_slice_mut(plane)
    }

    /// Pack the three planes into a single planar 8-bit buffer in
    /// `Y` then `Cb` then `Cr` order, each plane row-major. Only valid
    /// for `BitDepth == 8` planes (the common `yuv420p` / `yuv444p`
    /// fixture layout); samples are already clipped to `[0, 255]` by the
    /// reconstruction step so the cast is exact.
    ///
    /// Returns `None` if any plane has a bit depth other than 8.
    #[must_use]
    pub fn to_planar_u8(&self) -> Option<Vec<u8>> {
        if self.bit_depth_luma != 8 {
            return None;
        }
        if self.chroma_array_type != 0 && self.bit_depth_chroma != 8 {
            return None;
        }
        let mut out = Vec::with_capacity(self.luma.len() + self.cb.len() + self.cr.len());
        for plane in [&self.luma, &self.cb, &self.cr] {
            out.extend(plane.iter().map(|&v| v as u8));
        }
        Some(out)
    }

    /// Pack the three planes into a single planar little-endian 16-bit
    /// buffer in `Y` then `Cb` then `Cr` order, each plane row-major —
    /// the `yuv420p10le` / `yuv422p10le` / `yuv444p10le` fixture layout
    /// for bit depths above 8. Samples are already clipped to
    /// `[0, (1 << BitDepth) − 1]` by the reconstruction step so the
    /// `u16` cast is exact.
    #[must_use]
    pub fn to_planar_le16(&self) -> Vec<u8> {
        let n = self.luma.len() + self.cb.len() + self.cr.len();
        let mut out = Vec::with_capacity(n * 2);
        for plane in [&self.luma, &self.cb, &self.cr] {
            for &v in plane.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }
}

/// `Clip3( 0, (1 << bitDepth) − 1, x )` — the §8 `Clip1Y` / `Clip1C`
/// sample clip.
#[inline]
#[must_use]
pub fn clip1(x: i32, bit_depth: u8) -> i32 {
    let max = (1i32 << bit_depth) - 1;
    x.clamp(0, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_420_planes_by_table_6_1() {
        let p = Picture::new(16, 16, 1, 8, 8);
        assert_eq!(p.plane_dims(Plane::Luma), (16, 16));
        assert_eq!(p.plane_dims(Plane::Cb), (8, 8));
        assert_eq!(p.plane_dims(Plane::Cr), (8, 8));
    }

    #[test]
    fn allocates_422_planes_by_table_6_1() {
        let p = Picture::new(16, 16, 2, 8, 8);
        // SubWidthC = 2, SubHeightC = 1.
        assert_eq!(p.plane_dims(Plane::Cb), (8, 16));
    }

    #[test]
    fn allocates_444_planes_by_table_6_1() {
        let p = Picture::new(16, 16, 3, 8, 8);
        assert_eq!(p.plane_dims(Plane::Cb), (16, 16));
    }

    #[test]
    fn monochrome_has_no_chroma() {
        let p = Picture::new(16, 16, 0, 8, 8);
        assert_eq!(p.plane_dims(Plane::Cb), (0, 0));
        assert!(p.plane(Plane::Cb).is_empty());
    }

    #[test]
    fn sample_roundtrips() {
        let mut p = Picture::new(8, 8, 1, 8, 8);
        p.set_sample(Plane::Luma, 3, 4, 200);
        assert_eq!(p.sample(Plane::Luma, 3, 4), 200);
        assert_eq!(p.sample(Plane::Luma, 0, 0), 0);
    }

    #[test]
    fn clone_shares_planes_until_written() {
        let mut a = Picture::new(8, 8, 1, 8, 8);
        a.set_sample(Plane::Luma, 1, 1, 7);
        let b = a.clone();
        assert!(a.is_shared() && b.is_shared());
        // Writing one copy detaches it; the other keeps its samples.
        a.set_sample(Plane::Luma, 1, 1, 9);
        assert_eq!(a.sample(Plane::Luma, 1, 1), 9);
        assert_eq!(b.sample(Plane::Luma, 1, 1), 7);
        assert!(!b.is_shared() || a.is_shared());
        let (y, cb, cr) = b.into_planes();
        assert_eq!((y.len(), cb.len(), cr.len()), (64, 16, 16));
        assert_eq!(y[9], 7);
    }

    #[test]
    fn band_addresses_absolute_rows() {
        let mut b = Picture::new_band(16, 64, 1, 8, 8, 16, 17);
        assert_eq!(b.plane_dims(Plane::Luma), (16, 64));
        assert_eq!(b.plane_dims(Plane::Cb), (8, 32));
        assert_eq!(b.y_origin(Plane::Luma), 16);
        assert_eq!(b.y_origin(Plane::Cb), 8);
        assert_eq!(b.stored_rows(Plane::Luma), 17);
        assert_eq!(b.stored_rows(Plane::Cb), 9);
        b.set_sample(Plane::Luma, 3, 16, 77);
        b.row_mut(Plane::Cb, 8)[2] = 5;
        assert_eq!(b.plane(Plane::Luma)[3], 77);
        assert_eq!(b.sample(Plane::Cb, 2, 8), 5);
        assert_eq!(b.row(Plane::Luma, 32)[3], 0);
        // A tile band: columns from 32 on.
        let mut t = Picture::new_rect_band(64, 64, 1, 8, 8, (32, 16, 32, 48));
        assert_eq!(t.band(Plane::Luma), (32, 16, 32));
        assert_eq!(t.band(Plane::Cb), (16, 8, 16));
        t.set_sample(Plane::Luma, 40, 20, 9);
        assert_eq!(t.row(Plane::Luma, 20)[8], 9);
        assert_eq!(t.sample(Plane::Luma, 40, 20), 9);
        assert_eq!(t.stored_rows(Plane::Cb), 24);
    }

    #[test]
    fn from_planes_wraps_without_copying() {
        let luma: Vec<u16> = (0..16).collect();
        let p = Picture::from_planes(4, 4, 0, 8, 8, luma, Vec::new(), Vec::new());
        assert_eq!(p.sample(Plane::Luma, 3, 3), 15);
        assert!(p.plane(Plane::Cb).is_empty());
        let (y, _, _) = p.into_planes();
        assert_eq!(y.len(), 16);
    }

    #[test]
    fn clip1_clamps_to_bit_depth_range() {
        assert_eq!(clip1(-5, 8), 0);
        assert_eq!(clip1(300, 8), 255);
        assert_eq!(clip1(81, 8), 81);
        assert_eq!(clip1(2000, 10), 1023);
    }

    #[test]
    fn planar_u8_packs_y_cb_cr() {
        let mut p = Picture::new(2, 2, 1, 8, 8);
        for y in 0..2 {
            for x in 0..2 {
                p.set_sample(Plane::Luma, x, y, 0x51);
            }
        }
        p.set_sample(Plane::Cb, 0, 0, 0x5a);
        p.set_sample(Plane::Cr, 0, 0, 0xf0);
        let packed = p.to_planar_u8().unwrap();
        // 4 luma + 1 cb + 1 cr.
        assert_eq!(packed.len(), 6);
        assert_eq!(&packed[0..4], &[0x51, 0x51, 0x51, 0x51]);
        assert_eq!(packed[4], 0x5a);
        assert_eq!(packed[5], 0xf0);
    }
}
