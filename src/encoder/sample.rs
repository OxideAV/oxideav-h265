//! Sample-format plumbing shared by the encoders: the chroma format
//! and bit depths a picture is coded in ([`SampleFmt`]), and the
//! [`Sample`] storage trait the widened coders read their planes
//! through (`u8` for the historical 8-bit paths, `u16` for the
//! bit-depth-general quadtree coder).

use crate::encoder::pcm::PcmLayout;

/// A stored picture sample: `u8` (8-bit planes) or `u16` (any depth).
pub(crate) trait Sample: Copy + Send + Sync + 'static {
    /// The sample value.
    fn to_i32(self) -> i32;
    /// Clip `v` into `0..=max` and store it.
    fn clipped(v: i32, max: i32) -> Self;
}

impl Sample for u8 {
    #[inline]
    fn to_i32(self) -> i32 {
        i32::from(self)
    }

    #[inline]
    fn clipped(v: i32, max: i32) -> Self {
        v.clamp(0, max) as u8
    }
}

impl Sample for u16 {
    #[inline]
    fn to_i32(self) -> i32 {
        i32::from(self)
    }

    #[inline]
    fn clipped(v: i32, max: i32) -> Self {
        v.clamp(0, max) as u16
    }
}

/// The sample format of a coded picture: `chroma_format_idc`
/// (Table 6-1; `separate_colour_plane_flag == 0`, so it is also
/// `ChromaArrayType`) and the luma / chroma bit depths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleFmt {
    /// `chroma_format_idc` / `ChromaArrayType` (0 monochrome, 1 4:2:0,
    /// 2 4:2:2, 3 4:4:4).
    pub chroma_format_idc: u8,
    /// `BitDepthY` (8..=16).
    pub bit_depth_luma: u8,
    /// `BitDepthC` (8..=16; irrelevant for monochrome).
    pub bit_depth_chroma: u8,
}

impl Default for SampleFmt {
    fn default() -> Self {
        Self::YUV420_8
    }
}

impl SampleFmt {
    /// The historical coders' format: 4:2:0 at 8 bits.
    pub const YUV420_8: Self = Self {
        chroma_format_idc: 1,
        bit_depth_luma: 8,
        bit_depth_chroma: 8,
    };

    /// A format with equal luma / chroma depth, validated
    /// (`chroma_format_idc <= 3`, depth 8..=16).
    #[must_use]
    #[allow(dead_code)] // the registry's deep-layout intra path (next commit)
    pub fn new(chroma_format_idc: u8, bit_depth: u8) -> Option<Self> {
        (chroma_format_idc <= 3 && (8..=16).contains(&bit_depth)).then_some(Self {
            chroma_format_idc,
            bit_depth_luma: bit_depth,
            bit_depth_chroma: bit_depth,
        })
    }

    /// The format of a PCM layout (equal depths).
    #[must_use]
    #[allow(dead_code)] // the registry's deep-layout intra path (next commit)
    pub fn from_layout(layout: PcmLayout) -> Self {
        Self {
            chroma_format_idc: layout.chroma_format_idc,
            bit_depth_luma: layout.bit_depth,
            bit_depth_chroma: layout.bit_depth,
        }
    }

    /// The PCM layout twin (the Annex A profile row writer speaks it;
    /// the chroma depth is what it carries when the two differ).
    #[must_use]
    pub fn layout(&self) -> PcmLayout {
        PcmLayout {
            chroma_format_idc: self.chroma_format_idc,
            bit_depth: self.bit_depth_luma.max(self.bit_depth_chroma),
        }
    }

    /// `true` for the format every historical (fixed-`u8`) path codes.
    #[must_use]
    pub fn is_yuv420_8(&self) -> bool {
        *self == Self::YUV420_8
    }

    /// `( SubWidthC, SubHeightC )` (Table 6-1); `(1, 1)` for monochrome,
    /// whose chroma planes are empty.
    #[must_use]
    pub fn sub_wh(&self) -> (usize, usize) {
        match self.chroma_format_idc {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        }
    }

    /// Whether chroma planes exist (`ChromaArrayType != 0`).
    #[must_use]
    pub fn has_chroma(&self) -> bool {
        self.chroma_format_idc != 0
    }

    /// The chroma plane dimensions of a `width x height` picture
    /// (`(0, 0)` for monochrome).
    #[must_use]
    pub fn chroma_dims(&self, width: usize, height: usize) -> (usize, usize) {
        if !self.has_chroma() {
            return (0, 0);
        }
        let (sw, sh) = self.sub_wh();
        (width / sw, height / sh)
    }

    /// `QpBdOffsetY = 6 * bit_depth_luma_minus8`.
    #[must_use]
    pub fn qp_bd_offset_y(&self) -> i32 {
        6 * (i32::from(self.bit_depth_luma) - 8)
    }

    /// `QpBdOffsetC = 6 * bit_depth_chroma_minus8`.
    #[must_use]
    pub fn qp_bd_offset_c(&self) -> i32 {
        6 * (i32::from(self.bit_depth_chroma) - 8)
    }

    /// The legal `SliceQpY` / `QpY` range: `−QpBdOffsetY ..= 51`.
    #[must_use]
    pub fn qp_range(&self) -> core::ops::RangeInclusive<i32> {
        -self.qp_bd_offset_y()..=51
    }

    /// `( 1 << BitDepthY ) − 1`.
    #[must_use]
    pub fn max_luma(&self) -> i32 {
        (1i32 << self.bit_depth_luma) - 1
    }

    /// `( 1 << BitDepthC ) − 1`.
    #[must_use]
    pub fn max_chroma(&self) -> i32 {
        (1i32 << self.bit_depth_chroma) - 1
    }

    /// The component's bit depth (`c_idx` 0 luma, 1 / 2 chroma).
    #[must_use]
    pub fn bit_depth(&self, c_idx: u8) -> u8 {
        if c_idx == 0 {
            self.bit_depth_luma
        } else {
            self.bit_depth_chroma
        }
    }

    /// §8.6.1 eq. 8-284: `Qp′Y = QpY + QpBdOffsetY` — the luma `qP`
    /// of the §8.6.2 scaling process.
    #[must_use]
    pub fn luma_qp_prime(&self, qp_y: i32) -> u32 {
        (qp_y + self.qp_bd_offset_y()) as u32
    }

    /// §8.6.1 eqs 8-285..8-287: the chroma `qP` (`Qp′Cb` / `Qp′Cr`)
    /// for `QpY` and the PPS + slice chroma QP offset — `qPi` clipped
    /// to `−QpBdOffsetC ..= 57`, mapped through Table 8-10 for
    /// `ChromaArrayType == 1` and `Min( qPi, 51 )` otherwise, plus
    /// `QpBdOffsetC`.
    #[must_use]
    pub fn chroma_qp_prime(&self, qp_y: i32, offset: i32) -> u32 {
        let bd_c = self.qp_bd_offset_c();
        let qpi = (qp_y + offset).clamp(-bd_c, 57);
        let qpc = if self.chroma_format_idc == 1 {
            qpc_table_8_10(qpi)
        } else {
            qpi.min(51)
        };
        (qpc + bd_c) as u32
    }

    /// Chroma transform blocks per luma transform block: two square
    /// blocks stacked vertically for 4:2:2, one otherwise (0 for
    /// monochrome).
    #[must_use]
    pub fn chroma_blocks(&self) -> usize {
        match self.chroma_format_idc {
            0 => 0,
            2 => 2,
            _ => 1,
        }
    }

    /// §7.3.8.10 `log2TrafoSizeC = Max( 2, log2TrafoSize −
    /// ( ChromaArrayType == 3 ? 0 : 1 ) )`.
    #[must_use]
    pub fn log2_chroma_tb(&self, log2_trafo_size: u32) -> u32 {
        if self.chroma_format_idc == 3 {
            log2_trafo_size
        } else {
            (log2_trafo_size - 1).max(2)
        }
    }

    /// Whether a transform-tree node of luma size `log2TrafoSize`
    /// carries its chroma blocks in place (§7.3.8.10: `log2TrafoSize`
    /// above 2, or `ChromaArrayType == 3`); otherwise a 4x4 luma leaf
    /// defers chroma to its parent's `blkIdx == 3` child. Always false
    /// for monochrome.
    #[must_use]
    pub fn chroma_in_place(&self, log2_trafo_size: u32) -> bool {
        self.has_chroma() && (log2_trafo_size > 2 || self.chroma_format_idc == 3)
    }

    /// Whether the §7.3.8.8 chroma-cbf block is present at a node:
    /// `( log2TrafoSize > 2 && ChromaArrayType != 0 ) ||
    /// ChromaArrayType == 3`.
    #[must_use]
    pub fn chroma_cbf_present(&self, log2_trafo_size: u32) -> bool {
        self.chroma_in_place(log2_trafo_size)
    }

    /// Table 9-52 `sao_offset_abs` `cMax = ( 1 << ( Min( bitDepth, 10 )
    /// − 5 ) ) − 1` for the component.
    #[must_use]
    pub fn sao_offset_max(&self, c_idx: u8) -> i32 {
        (1i32 << (i32::from(self.bit_depth(c_idx).min(10)) - 5)) - 1
    }
}

/// Table 8-10 — `QpC` as a function of `qPi` for `ChromaArrayType == 1`.
fn qpc_table_8_10(qpi: i32) -> i32 {
    match qpi {
        x if x < 30 => x,
        30..=33 => qpi - 1,
        34..=43 => 33 + (qpi - 34) / 2,
        x => x - 6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chroma_qp_matches_the_8bit_table_and_offsets_at_depth() {
        let f8 = SampleFmt::YUV420_8;
        for qp in 0..=51 {
            assert_eq!(
                f8.chroma_qp_prime(qp, 0),
                crate::encoder::intra::chroma_qp_420(qp)
            );
        }
        let f10 = SampleFmt::new(1, 10).unwrap();
        assert_eq!(f10.qp_bd_offset_y(), 12);
        // qPi = 30 (clipped range −12..57) → QpC 29, + 12.
        assert_eq!(f10.chroma_qp_prime(30, 0), 41);
        // Negative QpY is legal at depth: qPi = −12 → QpC −12 → 0.
        assert_eq!(f10.chroma_qp_prime(-12, 0), 0);
        assert_eq!(f10.luma_qp_prime(-12), 0);
        // 4:4:4 / 4:2:2: Min( qPi, 51 ).
        let f444 = SampleFmt::new(3, 8).unwrap();
        assert_eq!(f444.chroma_qp_prime(40, 0), 40);
        assert_eq!(f444.chroma_qp_prime(51, 6), 51);
    }

    #[test]
    fn geometry_helpers_follow_table_6_1() {
        assert_eq!(SampleFmt::new(0, 8).unwrap().chroma_dims(64, 32), (0, 0));
        assert_eq!(SampleFmt::new(1, 8).unwrap().chroma_dims(64, 32), (32, 16));
        assert_eq!(SampleFmt::new(2, 8).unwrap().chroma_dims(64, 32), (32, 32));
        assert_eq!(SampleFmt::new(3, 8).unwrap().chroma_dims(64, 32), (64, 32));
        let f422 = SampleFmt::new(2, 10).unwrap();
        assert_eq!(f422.chroma_blocks(), 2);
        assert_eq!(f422.log2_chroma_tb(3), 2);
        assert!(!f422.chroma_in_place(2));
        let f444 = SampleFmt::new(3, 12).unwrap();
        assert_eq!(f444.log2_chroma_tb(2), 2);
        assert!(f444.chroma_in_place(2));
        assert_eq!(f444.sao_offset_max(0), 31);
        assert_eq!(SampleFmt::YUV420_8.sao_offset_max(1), 7);
        assert!(SampleFmt::new(4, 8).is_none());
        assert!(SampleFmt::new(1, 7).is_none());
    }
}
