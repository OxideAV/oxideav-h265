//! The stream-level description of the frames the registry decoder
//! emits — the answer to `Decoder::output_params` (oxideav-core
//! 0.1.42). A [`VideoFrame`](oxideav_core::VideoFrame) carries only
//! `pts` and planes; its size, layout, rate and colour signal are
//! stated once per stream and re-stated when they change.

use oxideav_core::{CodecId, CodecParameters, ColorSignal, PixelFormat, Rational};

use crate::sequence::CropWindow;
use crate::sps::SeqParameterSet;

/// What a picture's SPS fixes about its output beyond its samples:
/// the §E.2.1 VUI timing and colour signal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ShapeTraits {
    pub frame_rate: Option<Rational>,
    pub color: ColorSignal,
}

impl ShapeTraits {
    pub(crate) fn of_sps(sps: &SeqParameterSet) -> Self {
        let vui = sps.vui_parameters.as_ref();
        // §E.3.1: a picture lasts num_units_in_tick / time_scale
        // seconds (each emitted frame is one picture, a field picture
        // included when field_seq_flag is set).
        let frame_rate = vui.and_then(|v| {
            let t = v.timing_info.as_ref()?;
            if t.num_units_in_tick == 0 || t.time_scale == 0 {
                return None;
            }
            Some(reduce(
                i64::from(t.time_scale),
                i64::from(t.num_units_in_tick),
            ))
        });
        // §E.3.1: an absent colour description infers 2 (unspecified)
        // for all three code points; an absent video_signal_type
        // leaves the range unsignalled.
        let color = match vui.and_then(|v| v.video_signal_type.as_ref()) {
            None => ColorSignal::unspecified(),
            Some(vst) => {
                let (p, t, m) = vst.colour_description.map_or((2, 2, 2), |c| {
                    (
                        c.colour_primaries,
                        c.transfer_characteristics,
                        c.matrix_coeffs,
                    )
                });
                ColorSignal::from_code_points(p, t, m, vst.video_full_range_flag)
            }
        };
        Self { frame_rate, color }
    }
}

impl ShapeTraits {
    /// `self`, with every field it leaves unsignalled taken from
    /// `fallback`.
    pub(crate) fn or(self, fallback: &Self) -> Self {
        use oxideav_core::{
            ColorPrimaries, ColorRange, MatrixCoefficients, TransferCharacteristics,
        };
        let (c, f) = (self.color, fallback.color);
        let mut color = c;
        if c.range == ColorRange::Unspecified {
            color.range = f.range;
        }
        if c.primaries == ColorPrimaries::UNSPECIFIED {
            color.primaries = f.primaries;
        }
        if c.transfer == TransferCharacteristics::UNSPECIFIED {
            color.transfer = f.transfer;
        }
        if c.matrix == MatrixCoefficients::UNSPECIFIED {
            color.matrix = f.matrix;
        }
        Self {
            frame_rate: self.frame_rate.or(fallback.frame_rate),
            color,
        }
    }
}

fn reduce(a: i64, b: i64) -> Rational {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    Rational::new(a / x, b / x)
}

/// The planar layout `video_frame_owned` packs a picture of this
/// `ChromaArrayType` / bit depth into: one byte per sample when both
/// depths are 8, else little-endian 16-bit for every plane. `None` for
/// depth combinations no [`PixelFormat`] names (9 / 11 / 13 / 14 / 15
/// bits, or luma and chroma at different depths).
pub(crate) fn pixel_format(
    chroma_array_type: u8,
    depth_luma: u8,
    depth_chroma: u8,
) -> Option<PixelFormat> {
    use PixelFormat::*;
    if chroma_array_type != 0 && depth_chroma != depth_luma {
        return None;
    }
    Some(match (chroma_array_type, depth_luma) {
        (0, 8) => Gray8,
        (0, 10) => Gray10Le,
        (0, 12) => Gray12Le,
        (0, 16) => Gray16Le,
        (1, 8) => Yuv420P,
        (2, 8) => Yuv422P,
        (3, 8) => Yuv444P,
        (1, 10) => Yuv420P10Le,
        (2, 10) => Yuv422P10Le,
        (3, 10) => Yuv444P10Le,
        (1, 12) => Yuv420P12Le,
        (2, 12) => Yuv422P12Le,
        (3, 12) => Yuv444P12Le,
        (1, 16) => Yuv420P16Le,
        (2, 16) => Yuv422P16Le,
        (3, 16) => Yuv444P16Le,
        _ => return None,
    })
}

/// The description of one emitted frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OutShape {
    pub width: u32,
    pub height: u32,
    pub pixel_format: Option<PixelFormat>,
    pub traits: ShapeTraits,
}

impl OutShape {
    /// The shape a picture of `sps` is output with, before any has
    /// been decoded (§7.4.3.2.1 conformance window, §7.4.3.2.1
    /// ChromaArrayType and bit depths).
    pub(crate) fn of_sps(sps: &SeqParameterSet) -> Self {
        let crop = CropWindow::from_sps(sps);
        let cat = if sps.separate_colour_plane_flag {
            0
        } else {
            sps.chroma_format_idc
        };
        Self {
            width: crop.width as u32,
            height: crop.height as u32,
            pixel_format: pixel_format(
                cat,
                sps.bit_depth_luma_minus8 + 8,
                sps.bit_depth_chroma_minus8 + 8,
            ),
            traits: ShapeTraits::of_sps(sps),
        }
    }

    /// As the `CodecParameters` `output_params` answers.
    pub(crate) fn to_params(self, codec_id: &CodecId) -> CodecParameters {
        let mut p = CodecParameters::video(codec_id.clone());
        p.width = Some(self.width);
        p.height = Some(self.height);
        p.pixel_format = self.pixel_format;
        p.frame_rate = self.traits.frame_rate;
        p.color_signal = self.traits.color;
        p
    }
}
