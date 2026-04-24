//! HEVC Sample Adaptive Offset (SAO) filter — §8.7.3.
//!
//! SAO is the in-loop post-deblocking filter that adds a small signed
//! offset to individual samples based either on the sample's value (the
//! "band offset" mode) or on its local edge pattern (the "edge offset"
//! mode).
//!
//! ## Modes (§8.7.3.1)
//!
//! * `SaoType = 0` — SAO is off for this CTB / component.
//! * `SaoType = 1` — Band offset. The sample range `[0, 2^bitDepth)` is
//!   split into 32 equal-width bands; a 4-band contiguous window starting
//!   at `sao_band_position` receives four signed offsets, one per band.
//!   All samples in the CTB that fall inside the 4-band window are
//!   shifted; samples outside the window are left unchanged.
//! * `SaoType = 2` — Edge offset. One of four edge directions
//!   (`sao_eo_class`: 0 = horizontal, 1 = vertical, 2 = 135°, 3 = 45°) is
//!   selected. Each sample is classified into one of five edge categories
//!   (§8.7.3.3 Table 8-12) based on the comparison with its two
//!   neighbours along that direction. Categories 1..4 receive an offset;
//!   category 0 is a flat region and is left alone.
//!
//! ## Shape
//!
//! SAO runs **after deblocking** and **before the DPB write** on the
//! reconstructed picture buffers. Samples are processed CTB-by-CTB, and
//! the spec notes that edge-offset classification reads the *already
//! deblocked but not yet SAO-filtered* neighbours, so we always compare
//! against the buffer as it stood on entry to the CTB. For a single-pass
//! in-place implementation that's only an issue across CTB seams; we
//! replicate the conservative behaviour by writing outputs into a copy
//! and swapping at the end of the pass.
//!
//! ## Chroma sizing (§8.7.3.2)
//!
//! The CTB's chroma dimensions follow `ChromaArrayType`:
//! * `ChromaArrayType == 1` (4:2:0) — chroma CTB = luma CTB >> (1, 1).
//! * `ChromaArrayType == 2` (4:2:2) — chroma CTB = luma CTB >> (1, 0).
//! * `ChromaArrayType == 3` (4:4:4) — chroma CTB = luma CTB.
//!
//! ## Tables from the spec
//!
//! Only two lookup tables are used: the four edge-offset neighbour
//! deltas (per `sao_eo_class`) and the five-way category derivation
//! (§8.7.3.3 eq. 8-304..8-307). Both are small compile-time constants.

use crate::ctu::Picture;
use crate::slice::SliceSegmentHeader;
use crate::sps::SeqParameterSet;

/// Per-CTB SAO parameters as parsed from the bitstream (§7.3.8.3 + §7.4.9.3).
///
/// Components are indexed `[luma=0, Cb=1, Cr=2]`. `type_idx` 0 = off,
/// 1 = band, 2 = edge. `eo_class` is the 2-bit edge direction and is only
/// meaningful when `type_idx == 2`. `band_position` is the 5-bit starting
/// band index and only meaningful when `type_idx == 1`.
#[derive(Clone, Debug, Default)]
pub struct CtuSaoParams {
    pub type_idx: [u8; 3],
    pub eo_class: [u8; 3],
    pub band_position: [u8; 3],
    /// Signed offsets, four per component. For edge offset, indices 0..3 map
    /// to categories 1..4 (with fixed sign per §7.4.9.3: +, +, -, -).
    /// For band offset, indices 0..3 are the four signed offsets for the
    /// four contiguous bands.
    pub offsets: [[i8; 4]; 3],
}

impl CtuSaoParams {
    pub fn is_noop(&self) -> bool {
        self.type_idx.iter().all(|&t| t == 0)
    }
}

/// Edge-offset neighbour deltas per `sao_eo_class` (§8.7.3.3 Table 8-12).
///
/// For a sample `p[x][y]`, the two neighbours `p0` and `p1` along the edge
/// direction are at `(x + dx0, y + dy0)` and `(x + dx1, y + dy1)`.
/// Returned as `[(dx0, dy0), (dx1, dy1)]`.
const EO_OFFSETS: [[(i32, i32); 2]; 4] = [
    // 0: horizontal — neighbours at (−1, 0) and (+1, 0).
    [(-1, 0), (1, 0)],
    // 1: vertical — neighbours at (0, −1) and (0, +1).
    [(0, -1), (0, 1)],
    // 2: 135° — neighbours at (−1, −1) and (+1, +1).
    [(-1, -1), (1, 1)],
    // 3: 45° — neighbours at (+1, −1) and (−1, +1).
    [(1, -1), (-1, 1)],
];

/// Sign function (§8.7.3.3 eq. 8-304): sign(a - b) ∈ {-1, 0, +1}.
#[inline]
fn sign_diff(a: i32, b: i32) -> i32 {
    (a - b).signum()
}

/// Classify a single sample into one of the five edge-offset categories
/// (§8.7.3.3 Table 8-12). Returns 0..4 where 0 = no offset, 1..4 = category.
///
/// `c` is the current sample value, `n0` and `n1` are its two neighbours
/// along the selected edge direction. The classification uses:
///
/// * `edgeIdx = 2 + sign(c - n0) + sign(c - n1)` → integer ∈ {0, 1, 2, 3, 4}.
/// * `edgeIdx` 0, 1, 2, 3, 4 maps to category 1, 2, 0, 3, 4 respectively.
#[inline]
fn eo_category(c: i32, n0: i32, n1: i32) -> u8 {
    let edge_idx = 2 + sign_diff(c, n0) + sign_diff(c, n1);
    // Remap per Table 8-12: edge_idx 0..4 → cat 1, 2, 0, 3, 4.
    match edge_idx {
        0 => 1,
        1 => 2,
        2 => 0,
        3 => 3,
        4 => 4,
        _ => 0, // unreachable; guard against numerical surprises.
    }
}

/// Per-slice SAO state: grid of per-CTB parameters for a reconstructed
/// picture. Lives on `Picture` so the CTU walker can populate entries as
/// it parses, and the post-decode filter can read them back.
#[derive(Clone, Debug)]
pub struct SaoGrid {
    pub ctbs_x: u32,
    pub ctbs_y: u32,
    pub ctb_log2: u32,
    pub params: Vec<CtuSaoParams>,
}

impl SaoGrid {
    pub fn new(ctbs_x: u32, ctbs_y: u32, ctb_log2: u32) -> Self {
        let n = (ctbs_x as usize) * (ctbs_y as usize);
        Self {
            ctbs_x,
            ctbs_y,
            ctb_log2,
            params: vec![CtuSaoParams::default(); n],
        }
    }

    pub fn get(&self, cx: u32, cy: u32) -> &CtuSaoParams {
        let idx = (cy as usize) * (self.ctbs_x as usize) + cx as usize;
        &self.params[idx]
    }

    pub fn set(&mut self, cx: u32, cy: u32, p: CtuSaoParams) {
        let idx = (cy as usize) * (self.ctbs_x as usize) + cx as usize;
        self.params[idx] = p;
    }
}

/// Apply SAO to the reconstructed picture's luma + chroma planes.
/// Must be called after deblocking and before the DPB write (§8.7.3).
pub fn apply_sao(pic: &mut Picture, sps: &SeqParameterSet, slice: &SliceSegmentHeader) {
    let bit_depth_y = sps.bit_depth_y();
    let bit_depth_c = sps.bit_depth_c();
    let grid = match pic.sao_grid.take() {
        Some(g) => g,
        None => return,
    };
    // Short-circuit a slice that has no SAO signalled at all — avoids the
    // plane copy below.
    if !slice.slice_sao_luma_flag && !slice.slice_sao_chroma_flag {
        pic.sao_grid = Some(grid);
        return;
    }
    let chroma_array_type = if sps.separate_colour_plane_flag {
        0
    } else {
        sps.chroma_format_idc
    };
    let ctb_log2 = grid.ctb_log2;
    let ctb_size = 1u32 << ctb_log2;
    let pic_w = pic.width;
    let pic_h = pic.height;

    // SAO is defined as operating on the pre-filter reconstruction of the
    // CTB. §8.7.3 is explicit that edge-offset classification uses
    // neighbours from the post-deblock-but-pre-SAO picture, so we filter
    // from a snapshot and write into the live buffer. The snapshot is only
    // the input for the classify step — outputs land in `pic.luma` /
    // `pic.cb` / `pic.cr` directly.
    if slice.slice_sao_luma_flag {
        let snapshot = pic.luma.clone();
        let stride = pic.luma_stride;
        for cy in 0..grid.ctbs_y {
            for cx in 0..grid.ctbs_x {
                let p = grid.get(cx, cy);
                let t = p.type_idx[0];
                if t == 0 {
                    continue;
                }
                let x0 = cx * ctb_size;
                let y0 = cy * ctb_size;
                let w = ctb_size.min(pic_w - x0);
                let h = ctb_size.min(pic_h - y0);
                apply_ctb(
                    &snapshot,
                    &mut pic.luma,
                    stride,
                    pic_w as usize,
                    pic_h as usize,
                    x0 as usize,
                    y0 as usize,
                    w as usize,
                    h as usize,
                    t,
                    p.eo_class[0],
                    p.band_position[0],
                    &p.offsets[0],
                    bit_depth_y,
                );
            }
        }
    }
    if slice.slice_sao_chroma_flag && chroma_array_type != 0 {
        let (sub_x, sub_y): (u32, u32) = match chroma_array_type {
            1 => (1, 1),
            2 => (1, 0),
            3 => (0, 0),
            _ => (1, 1),
        };
        let cw = pic_w >> sub_x;
        let ch = pic_h >> sub_y;
        let ctb_w_c = ctb_size >> sub_x;
        let ctb_h_c = ctb_size >> sub_y;
        let cb_snapshot = pic.cb.clone();
        let cr_snapshot = pic.cr.clone();
        let cstride = pic.chroma_stride;
        for cy in 0..grid.ctbs_y {
            for cx in 0..grid.ctbs_x {
                let p = grid.get(cx, cy);
                let x0 = cx * ctb_w_c;
                let y0 = cy * ctb_h_c;
                let w = ctb_w_c.min(cw - x0);
                let h = ctb_h_c.min(ch - y0);
                for comp in 0..2usize {
                    let t = p.type_idx[comp + 1];
                    if t == 0 {
                        continue;
                    }
                    let (src, dst) = if comp == 0 {
                        (&cb_snapshot, &mut pic.cb)
                    } else {
                        (&cr_snapshot, &mut pic.cr)
                    };
                    apply_ctb(
                        src,
                        dst,
                        cstride,
                        cw as usize,
                        ch as usize,
                        x0 as usize,
                        y0 as usize,
                        w as usize,
                        h as usize,
                        t,
                        p.eo_class[comp + 1],
                        p.band_position[comp + 1],
                        &p.offsets[comp + 1],
                        bit_depth_c,
                    );
                }
            }
        }
    }
    pic.sao_grid = Some(grid);
}

#[allow(clippy::too_many_arguments)]
fn apply_ctb(
    src: &[u16],
    dst: &mut [u16],
    stride: usize,
    plane_w: usize,
    plane_h: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    type_idx: u8,
    eo_class: u8,
    band_pos: u8,
    offsets: &[i8; 4],
    bit_depth: u32,
) {
    // §8.7.3.4: 32 bands regardless of bit depth, so the band shift is
    // `bit_depth - 5`.
    let band_shift = bit_depth.saturating_sub(5) as i32;
    if type_idx == 1 {
        // Band offset (§8.7.3.4). The band index is `sample >> (bitDepth - 5)`
        // (32 bands). Four contiguous bands starting at `band_pos` receive the
        // four signed offsets. Samples falling outside those four bands are
        // unchanged.
        //
        // The 4-band window wraps modulo 32 per the spec; a window
        // starting at band 30 covers bands 30, 31, 0, 1.
        for yy in 0..h {
            for xx in 0..w {
                let idx = (y0 + yy) * stride + (x0 + xx);
                let s = src[idx] as i32;
                let band = (s >> band_shift) as u8;
                let rel = band.wrapping_sub(band_pos) & 0x1F;
                if rel < 4 {
                    let off = offsets[rel as usize] as i32;
                    dst[idx] = clip_bd(s + off, bit_depth);
                }
            }
        }
        return;
    }
    // Edge offset (§8.7.3.3). Classify every sample, then add the category
    // offset. Samples whose neighbour is outside the picture fall into
    // category 0 (no filter) since their sign-diff cannot be evaluated.
    let (d0, d1) = (EO_OFFSETS[eo_class as usize][0], EO_OFFSETS[eo_class as usize][1]);
    for yy in 0..h {
        for xx in 0..w {
            let x = (x0 + xx) as i32;
            let y = (y0 + yy) as i32;
            let nx0 = x + d0.0;
            let ny0 = y + d0.1;
            let nx1 = x + d1.0;
            let ny1 = y + d1.1;
            if nx0 < 0
                || ny0 < 0
                || nx1 < 0
                || ny1 < 0
                || nx0 as usize >= plane_w
                || ny0 as usize >= plane_h
                || nx1 as usize >= plane_w
                || ny1 as usize >= plane_h
            {
                continue;
            }
            let idx = (y as usize) * stride + x as usize;
            let c = src[idx] as i32;
            let n0 = src[(ny0 as usize) * stride + nx0 as usize] as i32;
            let n1 = src[(ny1 as usize) * stride + nx1 as usize] as i32;
            let cat = eo_category(c, n0, n1);
            if cat == 0 {
                continue;
            }
            // offsets[i] is the stored signed offset for category i+1
            // (spec signs: + + - -, already baked into the stored value
            // by the parser).
            let off = offsets[(cat - 1) as usize] as i32;
            dst[idx] = clip_bd(c + off, bit_depth);
        }
    }
}

#[inline]
fn clip_bd(v: i32, bit_depth: u32) -> u16 {
    let max = ((1i32 << bit_depth) - 1).max(0);
    v.clamp(0, max) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eo_category_matches_spec_table() {
        // Spec Table 8-12 categories:
        //   c < n0 && c < n1           → cat 1 (local minimum)
        //   (c < n0 && c == n1) || (c == n0 && c < n1) → cat 2
        //   (c > n0 && c == n1) || (c == n0 && c > n1) → cat 3
        //   c > n0 && c > n1           → cat 4 (local maximum)
        //   otherwise                   → cat 0
        assert_eq!(eo_category(5, 10, 20), 1); // local min
        assert_eq!(eo_category(5, 10, 5), 2); // equal to one, less than other
        assert_eq!(eo_category(5, 5, 10), 2);
        assert_eq!(eo_category(10, 5, 10), 3);
        assert_eq!(eo_category(10, 10, 5), 3);
        assert_eq!(eo_category(20, 10, 5), 4); // local max
        assert_eq!(eo_category(5, 5, 5), 0); // flat
        assert_eq!(eo_category(5, 3, 10), 0); // monotone
    }

    #[test]
    fn band_offset_wraps_at_32() {
        // Window starting at band 30 should cover 30, 31, 0, 1.
        // Sample value 248 lives in band 31 (248 >> 3 = 31) → rel = 1.
        // Sample value 8 lives in band 1 (8 >> 3 = 1) → rel = 3.
        let band_pos: u8 = 30;
        let band31 = 31u8;
        let band1 = 1u8;
        let rel31 = band31.wrapping_sub(band_pos) & 0x1F;
        let rel1 = band1.wrapping_sub(band_pos) & 0x1F;
        assert_eq!(rel31, 1);
        assert_eq!(rel1, 3);
    }

    #[test]
    fn apply_ctb_band_offset_adds_only_in_window() {
        // Tiny 4x1 CTB, band_pos=0, offsets +2, +4, +6, +8 for bands 0..3.
        // Input samples: 0 (band 0), 8 (band 1), 40 (band 5 — outside),
        // 24 (band 3 — inside).
        let src = vec![0u16, 8, 40, 24];
        let mut dst = src.clone();
        let offsets = [2i8, 4, 6, 8];
        apply_ctb(&src, &mut dst, 4, 4, 1, 0, 0, 4, 1, 1, 0, 0, &offsets, 8);
        assert_eq!(dst[0], 2); // band 0 → +2
        assert_eq!(dst[1], 12); // band 1 → +4
        assert_eq!(dst[2], 40); // outside → untouched
        assert_eq!(dst[3], 32); // band 3 → +8
    }

    #[test]
    fn apply_ctb_edge_offset_only_touches_nonzero_categories() {
        // 3x3 luma patch, all samples = 100 except the centre = 50.
        // With horizontal EO (class 0), neighbours of the centre are both
        // 100, so centre is cat 1 (local min). Edges are skipped (no
        // neighbour on one side).
        #[rustfmt::skip]
        let src: Vec<u16> = vec![
            100, 100, 100,
            100,  50, 100,
            100, 100, 100,
        ];
        let mut dst = src.clone();
        let offsets = [7i8, 0, 0, 0]; // only cat 1 active.
        apply_ctb(&src, &mut dst, 3, 3, 3, 0, 0, 3, 3, 2, 0, 0, &offsets, 8);
        assert_eq!(dst[4], 57, "centre classified as cat 1 and offset by +7");
        // Corner samples have no left or right neighbour inside; they stay
        // put.
        assert_eq!(dst[0], 100);
        assert_eq!(dst[2], 100);
    }
}
