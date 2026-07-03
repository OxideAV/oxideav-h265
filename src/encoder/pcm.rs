//! PCM-only IDR encoder — the crate's encoder bootstrap.
//!
//! Emits fully conformant Main-profile Annex B access units in which
//! every coding unit is a `pcm_flag == 1` PCM block (§7.3.8.7): the
//! samples ride uncompressed inside the slice data, so the decode is
//! exactly lossless. The geometry is fixed at the simplest legal
//! shape: `CtbSizeY == MinCbSizeY == 16` with the PCM size range
//! pinned to 16 (`Log2MinIpcmCbSizeY == Log2MaxIpcmCbSizeY == 4`), so
//! every CTB is one unsplit coding unit and the whole §7.3.8 syntax
//! walk per CTB is: `part_mode` (one context-coded bin, `PART_2Nx2N`),
//! `pcm_flag` (a §9.3.5.6 terminate bin), the `pcm_alignment_zero_bit`
//! run, the raw §7.3.8.7 sample payload, the §9.3.5.2 engine re-init,
//! and `end_of_slice_segment_flag`.
//!
//! Every picture is an IDR (`IDR_N_LP`) with in-band VPS/SPS/PPS, so
//! the stream is seekable anywhere and decoder-stateless. 4:2:0 8-bit
//! only; picture dimensions must be multiples of 16 (the CTB size —
//! this bootstrap writes no conformance-window cropping).
//!
//! In-loop filters are neutralized the conformant way: SAO off in the
//! SPS, deblocking disabled in the PPS, and
//! `pcm_loop_filter_disabled_flag == 1` — a PCM-only picture
//! reconstructs to the raw samples bit for bit.

use crate::cabac::init_type;
use crate::ctx_init::SliceContexts;
use crate::encoder::bitwriter::BitWriter;
use crate::encoder::cabac::CabacEncoder;
use crate::encoder::nal::{annexb, nal_unit};

/// The fixed CTB / coding-block / PCM-block log2 size of this encoder.
const CTB_LOG2: u32 = 4;
/// The fixed CTB size (16).
const CTB: usize = 1 << CTB_LOG2;
/// `SliceQpY` written in every slice header (`slice_qp_delta == 0`
/// over `init_qp_minus26 == 0`). PCM blocks carry no residual, so the
/// QP only seeds context initialization.
const SLICE_QP: i32 = 26;

/// Errors from the PCM encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcmEncodeError {
    /// Width or height is zero or not a multiple of the 16-sample CTB.
    BadDimensions {
        /// Requested luma width.
        width: usize,
        /// Requested luma height.
        height: usize,
    },
    /// A supplied plane's length does not match the 4:2:0 geometry.
    PlaneSize {
        /// Which plane (`"y"`, `"cb"`, `"cr"`).
        plane: &'static str,
        /// Required sample count.
        expected: usize,
        /// Supplied sample count.
        got: usize,
    },
}

impl core::fmt::Display for PcmEncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadDimensions { width, height } => write!(
                f,
                "PCM encoder requires nonzero dimensions that are multiples of 16, got {width}x{height}"
            ),
            Self::PlaneSize {
                plane,
                expected,
                got,
            } => write!(f, "{plane} plane has {got} samples, expected {expected}"),
        }
    }
}

impl std::error::Error for PcmEncodeError {}

/// Table A.8 — pick the smallest `general_level_idc` whose `MaxLumaPs`
/// covers the picture. (A PCM payload can exceed a level's bit-rate
/// ceilings for large / high-rate content; this bootstrap levels by
/// luma picture size, the binding constraint for its intended
/// small-picture use.)
fn level_idc_for(luma_ps: usize) -> u8 {
    const LEVELS: [(usize, u8); 9] = [
        (36_864, 30),      // 1
        (122_880, 60),     // 2
        (245_760, 63),     // 2.1
        (552_960, 90),     // 3
        (983_040, 93),     // 3.1
        (2_228_224, 120),  // 4
        (8_912_896, 150),  // 5
        (35_651_584, 180), // 6
        (usize::MAX, 186), // 6.2
    ];
    LEVELS
        .iter()
        .find(|(max_ps, _)| luma_ps <= *max_ps)
        .map(|&(_, idc)| idc)
        .unwrap_or(186)
}

/// §7.3.3 `profile_tier_level( 1, 0 )` — Main profile, Main tier.
fn write_ptl(w: &mut BitWriter, level_idc: u8) {
    w.put_bits(0, 2); // general_profile_space
    w.put_bit(0); // general_tier_flag
    w.put_bits(1, 5); // general_profile_idc = 1 (Main)
                      // general_profile_compatibility_flag[0..32]: Main (1) is also
                      // decodable by Main 10 (2) decoders.
    let mut compat: u32 = 0;
    compat |= 1 << (31 - 1); // flag[1] — Main
    compat |= 1 << (31 - 2); // flag[2] — Main 10
    w.put_bits(compat, 32);
    w.put_bit(1); // general_progressive_source_flag
    w.put_bit(0); // general_interlaced_source_flag
    w.put_bit(1); // general_non_packed_constraint_flag
    w.put_bit(1); // general_frame_only_constraint_flag
                  // Main profile: general_reserved_zero_43bits.
    w.put_bits(0, 32);
    w.put_bits(0, 11);
    // Profile 1: general_inbld_flag = 0.
    w.put_bit(0);
    w.put_bits(u32::from(level_idc), 8); // general_level_idc
                                         // max_sub_layers_minus1 == 0: no sub-layer PTL syntax.
}

/// §7.3.2.1 — the minimal single-layer VPS.
fn write_vps(level_idc: u8) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(0, 4); // vps_video_parameter_set_id
    w.put_bit(1); // vps_base_layer_internal_flag
    w.put_bit(1); // vps_base_layer_available_flag
    w.put_bits(0, 6); // vps_max_layers_minus1
    w.put_bits(0, 3); // vps_max_sub_layers_minus1
    w.put_bit(1); // vps_temporal_id_nesting_flag
    w.put_bits(0xFFFF, 16); // vps_reserved_0xffff_16bits
    write_ptl(&mut w, level_idc);
    w.put_bit(1); // vps_sub_layer_ordering_info_present_flag
    w.ue(1); // vps_max_dec_pic_buffering_minus1[0]
    w.ue(0); // vps_max_num_reorder_pics[0]
    w.ue(0); // vps_max_latency_increase_plus1[0]
    w.put_bits(0, 6); // vps_max_layer_id
    w.ue(0); // vps_num_layer_sets_minus1
    w.put_bit(0); // vps_timing_info_present_flag
    w.put_bit(0); // vps_extension_flag
    w.rbsp_trailing_bits();
    w.finish()
}

/// §7.3.2.2 — the fixed-geometry SPS (4:2:0, 8-bit, CTB 16, PCM on).
fn write_sps(width: usize, height: usize, level_idc: u8) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(0, 4); // sps_video_parameter_set_id
    w.put_bits(0, 3); // sps_max_sub_layers_minus1
    w.put_bit(1); // sps_temporal_id_nesting_flag
    write_ptl(&mut w, level_idc);
    w.ue(0); // sps_seq_parameter_set_id
    w.ue(1); // chroma_format_idc = 4:2:0
    w.ue(width as u32); // pic_width_in_luma_samples
    w.ue(height as u32); // pic_height_in_luma_samples
    w.put_bit(0); // conformance_window_flag
    w.ue(0); // bit_depth_luma_minus8
    w.ue(0); // bit_depth_chroma_minus8
    w.ue(4); // log2_max_pic_order_cnt_lsb_minus4
    w.put_bit(1); // sps_sub_layer_ordering_info_present_flag
    w.ue(1); // sps_max_dec_pic_buffering_minus1[0]
    w.ue(0); // sps_max_num_reorder_pics[0]
    w.ue(0); // sps_max_latency_increase_plus1[0]
    w.ue(CTB_LOG2 - 3); // log2_min_luma_coding_block_size_minus3 (16)
    w.ue(0); // log2_diff_max_min_luma_coding_block_size (CTB 16)
    w.ue(0); // log2_min_luma_transform_block_size_minus2 (4)
    w.ue(2); // log2_diff_max_min_luma_transform_block_size (16)
    w.ue(0); // max_transform_hierarchy_depth_inter
    w.ue(0); // max_transform_hierarchy_depth_intra
    w.put_bit(0); // scaling_list_enabled_flag
    w.put_bit(0); // amp_enabled_flag
    w.put_bit(0); // sample_adaptive_offset_enabled_flag
    w.put_bit(1); // pcm_enabled_flag
    w.put_bits(7, 4); // pcm_sample_bit_depth_luma_minus1 (8-bit)
    w.put_bits(7, 4); // pcm_sample_bit_depth_chroma_minus1
    w.ue(CTB_LOG2 - 3); // log2_min_pcm_luma_coding_block_size_minus3 (16)
    w.ue(0); // log2_diff_max_min_pcm_luma_coding_block_size
    w.put_bit(1); // pcm_loop_filter_disabled_flag
    w.ue(0); // num_short_term_ref_pic_sets
    w.put_bit(0); // long_term_ref_pics_present_flag
    w.put_bit(0); // sps_temporal_mvp_enabled_flag
    w.put_bit(0); // strong_intra_smoothing_enabled_flag
    w.put_bit(0); // vui_parameters_present_flag
    w.put_bit(0); // sps_extension_present_flag
    w.rbsp_trailing_bits();
    w.finish()
}

/// §7.3.2.3 — the all-defaults PPS (deblocking disabled).
fn write_pps(dependent_slice_segments_enabled: bool) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.ue(0); // pps_pic_parameter_set_id
    w.ue(0); // pps_seq_parameter_set_id
    w.put_bit(u8::from(dependent_slice_segments_enabled)); // dependent_slice_segments_enabled_flag
    w.put_bit(0); // output_flag_present_flag
    w.put_bits(0, 3); // num_extra_slice_header_bits
    w.put_bit(0); // sign_data_hiding_enabled_flag
    w.put_bit(0); // cabac_init_present_flag
    w.ue(0); // num_ref_idx_l0_default_active_minus1
    w.ue(0); // num_ref_idx_l1_default_active_minus1
    w.se(SLICE_QP - 26); // init_qp_minus26
    w.put_bit(0); // constrained_intra_pred_flag
    w.put_bit(0); // transform_skip_enabled_flag
    w.put_bit(0); // cu_qp_delta_enabled_flag
    w.se(0); // pps_cb_qp_offset
    w.se(0); // pps_cr_qp_offset
    w.put_bit(0); // pps_slice_chroma_qp_offsets_present_flag
    w.put_bit(0); // weighted_pred_flag
    w.put_bit(0); // weighted_bipred_flag
    w.put_bit(0); // transquant_bypass_enabled_flag
    w.put_bit(0); // tiles_enabled_flag
    w.put_bit(0); // entropy_coding_sync_enabled_flag
    w.put_bit(1); // pps_loop_filter_across_slices_enabled_flag
    w.put_bit(1); // deblocking_filter_control_present_flag
    w.put_bit(0); // deblocking_filter_override_enabled_flag
    w.put_bit(1); // pps_deblocking_filter_disabled_flag
    w.put_bit(0); // pps_scaling_list_data_present_flag
    w.put_bit(0); // lists_modification_present_flag
    w.ue(0); // log2_parallel_merge_level_minus2
    w.put_bit(0); // slice_segment_header_extension_present_flag
    w.put_bit(0); // pps_extension_present_flag
    w.rbsp_trailing_bits();
    w.finish()
}

/// §7.3.6.1 + §7.3.8.1 — the picture's slice segments: one
/// independent I-slice segment followed by `segments − 1` dependent
/// slice segments (§7.4.7.1 inheritance), every CTB a PCM coding
/// unit. Returns one RBSP per segment.
///
/// The CABAC context variables carry across the segment boundary per
/// §9.3.1 / §9.3.2.4 (`TableStateIdxDs` storage at each segment's
/// `end_of_slice_segment_flag == 1`, §9.3.2.5 synchronization at the
/// next dependent segment's start) — each segment gets a fresh
/// §9.3.5.2 arithmetic engine over the shared context state.
fn write_idr_slice_segments(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    segments: usize,
) -> Vec<Vec<u8>> {
    let ctbs_x = width / CTB;
    let ctbs_y = height / CTB;
    let total = ctbs_x * ctbs_y;
    let cw = width / 2;
    // Ceil(Log2(PicSizeInCtbsY)) — the slice_segment_address width.
    let addr_bits = (usize::BITS - (total - 1).leading_zeros()).max(1) as u8;

    // Segment boundaries: split the CTB raster as evenly as possible.
    let seg_len = total.div_ceil(segments);
    let starts: Vec<usize> = (0..segments).map(|i| i * seg_len).collect();

    // initType 0 (I slice, equation 9-7), SliceQpY = 26. Shared across
    // the picture's segments (§9.3.2.4 / §9.3.2.5).
    let mut ctxs = SliceContexts::init(init_type(2, false), SLICE_QP);

    let mut out = Vec::with_capacity(segments);
    for (seg_idx, &start) in starts.iter().enumerate() {
        let end = (start + seg_len).min(total);
        let mut w = BitWriter::new();

        // ---- slice_segment_header() ----
        let first = seg_idx == 0;
        w.put_bit(u8::from(first)); // first_slice_segment_in_pic_flag
        w.put_bit(0); // no_output_of_prior_pics_flag (IRAP NAL)
        w.ue(0); // slice_pic_parameter_set_id
        if !first {
            // dependent_slice_segments_enabled_flag == 1 in the PPS.
            w.put_bit(1); // dependent_slice_segment_flag
            w.put_bits(start as u32, addr_bits); // slice_segment_address
        } else if segments > 1 {
            // (first segment of a multi-segment picture: the address
            // syntax is absent for first_slice_segment_in_pic_flag.)
        }
        if first {
            w.ue(2); // slice_type = I
            w.se(SLICE_QP - 26); // slice_qp_delta
        }
        // No tiles / WPP: no entry points. No header extension.
        w.rbsp_trailing_bits(); // byte_alignment() before slice data

        // ---- slice_segment_data() ----
        let mut cabac = CabacEncoder::new();
        for addr in start..end {
            let x0 = (addr % ctbs_x) * CTB;
            let y0 = (addr / ctbs_x) * CTB;
            // coding_quadtree: CtbLog2SizeY == MinCbLog2SizeY, so no
            // split_cu_flag. coding_unit( x0, y0, 4 ):
            // part_mode — §9.3.3.7 intra at MinCb: bin "1" = PART_2Nx2N.
            cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], 1);
            // pcm_flag — §9.3.5.6 terminate bin, value 1: flush.
            cabac.encode_terminate(&mut w, 1);
            // pcm_alignment_zero_bit run.
            w.align_zero();
            // pcm_sample( x0, y0, 4 ): 256 luma then 64 Cb + 64 Cr.
            for j in 0..CTB {
                for i in 0..CTB {
                    w.put_bits(u32::from(y[(y0 + j) * width + x0 + i]), 8);
                }
            }
            let (cx, cy) = (x0 / 2, y0 / 2);
            for plane in [cb, cr] {
                for j in 0..CTB / 2 {
                    for i in 0..CTB / 2 {
                        w.put_bits(u32::from(plane[(cy + j) * cw + cx + i]), 8);
                    }
                }
            }
            // §9.3.5.2: re-init the arithmetic engine after PCM data.
            cabac.reinit();
            // end_of_slice_segment_flag: 1 only at the segment's last CTB.
            cabac.encode_terminate(&mut w, u8::from(addr == end - 1));
        }
        // The final terminate-1 flush wrote the rbsp_stop_one_bit;
        // rbsp_slice_segment_trailing_bits() is alignment zeros from here.
        w.align_zero();
        out.push(w.finish());
    }
    out
}

/// Encode one 4:2:0 8-bit frame as a self-contained IDR access unit
/// (`VPS + SPS + PPS + IDR_N_LP` in Annex B form). The decode is
/// bit-exact lossless: every coding unit is a §7.3.8.7 PCM block.
///
/// # Errors
/// [`PcmEncodeError`] when the dimensions are not nonzero multiples of
/// 16 or a plane buffer has the wrong length.
pub fn encode_idr_pcm_au(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, PcmEncodeError> {
    encode_au(y, cb, cr, width, height, 1)
}

/// As [`encode_idr_pcm_au`], with the picture split into `segments`
/// slice segments: the first independent, the rest §7.3.6.1
/// *dependent* slice segments (`dependent_slice_segment_flag == 1`,
/// header inherited per §7.4.7.1, CABAC contexts carried across the
/// boundary per §9.3.2.4 / §9.3.2.5).
///
/// # Errors
/// [`PcmEncodeError`] as for [`encode_idr_pcm_au`], or
/// [`PcmEncodeError::BadDimensions`] when `segments` is 0 or exceeds
/// the picture's CTB count.
pub fn encode_idr_pcm_au_segmented(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    segments: usize,
) -> Result<Vec<u8>, PcmEncodeError> {
    encode_au(y, cb, cr, width, height, segments)
}

fn encode_au(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    segments: usize,
) -> Result<Vec<u8>, PcmEncodeError> {
    if width == 0 || height == 0 || width % CTB != 0 || height % CTB != 0 {
        return Err(PcmEncodeError::BadDimensions { width, height });
    }
    if segments == 0 || segments > (width / CTB) * (height / CTB) {
        return Err(PcmEncodeError::BadDimensions { width, height });
    }
    let check = |plane: &'static str, buf: &[u8], expected: usize| {
        if buf.len() != expected {
            Err(PcmEncodeError::PlaneSize {
                plane,
                expected,
                got: buf.len(),
            })
        } else {
            Ok(())
        }
    };
    check("y", y, width * height)?;
    check("cb", cb, width * height / 4)?;
    check("cr", cr, width * height / 4)?;

    let level_idc = level_idc_for(width * height);
    let mut units = vec![
        nal_unit(32, 0, 0, &write_vps(level_idc)), // VPS_NUT
        nal_unit(33, 0, 0, &write_sps(width, height, level_idc)), // SPS_NUT
        nal_unit(34, 0, 0, &write_pps(segments > 1)), // PPS_NUT
    ];
    for rbsp in write_idr_slice_segments(y, cb, cr, width, height, segments) {
        units.push(nal_unit(20, 0, 0, &rbsp)); // IDR_N_LP
    }
    Ok(annexb(&units))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pps::PicParameterSet;
    use crate::sps::SeqParameterSet;

    fn gradient_planes(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let y: Vec<u8> = (0..w * h).map(|i| (i * 7 % 251) as u8).collect();
        let cb: Vec<u8> = (0..w * h / 4).map(|i| (i * 3 % 240 + 8) as u8).collect();
        let cr: Vec<u8> = (0..w * h / 4).map(|i| (255 - i * 5 % 250) as u8).collect();
        (y, cb, cr)
    }

    #[test]
    fn rejects_bad_geometry() {
        let (y, cb, cr) = gradient_planes(16, 16);
        assert!(matches!(
            encode_idr_pcm_au(&y, &cb, &cr, 20, 16),
            Err(PcmEncodeError::BadDimensions { .. })
        ));
        assert!(matches!(
            encode_idr_pcm_au(&y, &cb, &cr, 32, 16),
            Err(PcmEncodeError::PlaneSize { .. })
        ));
    }

    /// The whole bootstrap contract: encode → decode with the crate's
    /// own end-to-end driver → bit-exact planes (PCM is lossless).
    #[test]
    fn pcm_au_roundtrips_losslessly_through_the_decoder() {
        for (w, h) in [(16usize, 16usize), (48, 32), (32, 64)] {
            let (y, cb, cr) = gradient_planes(w, h);
            let au = encode_idr_pcm_au(&y, &cb, &cr, w, h).expect("encode");
            let frames = crate::sequence::decode_annexb_sequence(&au).expect("decode");
            assert_eq!(frames.len(), 1, "{w}x{h}: one IDR frame");
            let mut expected = Vec::new();
            expected.extend_from_slice(&y);
            expected.extend_from_slice(&cb);
            expected.extend_from_slice(&cr);
            assert_eq!(
                frames[0].picture.to_planar_u8().expect("8-bit"),
                expected,
                "{w}x{h}: lossless PCM roundtrip"
            );
        }
    }

    /// Two frames form two self-contained IDR AUs; both decode losslessly
    /// and in order.
    #[test]
    fn multi_frame_pcm_stream_decodes_in_order() {
        let (y0, cb0, cr0) = gradient_planes(32, 32);
        let y1: Vec<u8> = y0.iter().map(|&v| v ^ 0x5A).collect();
        let mut stream = encode_idr_pcm_au(&y0, &cb0, &cr0, 32, 32).expect("au0");
        stream.extend(encode_idr_pcm_au(&y1, &cb0, &cr0, 32, 32).expect("au1"));
        let frames = crate::sequence::decode_annexb_sequence(&stream).expect("decode");
        assert_eq!(frames.len(), 2);
        let p0 = frames[0].picture.to_planar_u8().unwrap();
        let p1 = frames[1].picture.to_planar_u8().unwrap();
        assert_eq!(&p0[..y0.len()], &y0[..]);
        assert_eq!(&p1[..y1.len()], &y1[..]);
        assert_ne!(p0, p1);
    }

    /// Dependent slice segments: the picture splits into an independent
    /// segment + dependent segments whose CABAC contexts continue
    /// across the NAL boundary (§9.3.2.4 / §9.3.2.5) and whose header
    /// values inherit per §7.4.7.1 — still bit-exact lossless.
    #[test]
    fn dependent_slice_segments_roundtrip_losslessly() {
        let (w, h) = (48usize, 48usize); // 9 CTBs
        let (y, cb, cr) = gradient_planes(w, h);
        for segments in [2usize, 3, 9] {
            let au = encode_idr_pcm_au_segmented(&y, &cb, &cr, w, h, segments).expect("encode");
            // One IDR NAL per segment behind the parameter sets.
            let units = crate::nal::collect_nal_units(&au).expect("walk");
            assert_eq!(units.len(), 3 + segments, "{segments} segments");
            assert!(units[4..].iter().all(|u| u.header.nal_unit_type == 20));
            let frames = crate::sequence::decode_annexb_sequence(&au).expect("decode");
            assert_eq!(frames.len(), 1);
            let mut expected = Vec::new();
            expected.extend_from_slice(&y);
            expected.extend_from_slice(&cb);
            expected.extend_from_slice(&cr);
            assert_eq!(
                frames[0].picture.to_planar_u8().expect("8-bit"),
                expected,
                "{segments}-segment lossless roundtrip"
            );
        }
    }

    /// A dependent segment decoded WITHOUT the preceding segment's
    /// stored context state is rejected, not misdecoded.
    #[test]
    fn dependent_segment_without_predecessor_is_rejected() {
        let (y, cb, cr) = gradient_planes(32, 32);
        let au = encode_idr_pcm_au_segmented(&y, &cb, &cr, 32, 32, 2).expect("encode");
        let units = crate::nal::collect_nal_units(&au).expect("walk");
        // Drop the independent slice segment (unit 3), keep the
        // dependent one: the driver must refuse.
        let mut broken = Vec::new();
        for (i, u) in units.iter().enumerate() {
            if i == 3 {
                continue;
            }
            let mut coded = vec![0, 0, 0, 1];
            coded.extend(crate::encoder::nal::nal_unit(
                u.header.nal_unit_type,
                u.header.nuh_layer_id,
                u.header.temporal_id,
                &u.rbsp,
            ));
            broken.extend(coded);
        }
        assert!(crate::sequence::decode_annexb_sequence(&broken).is_err());
    }

    #[test]
    fn written_sps_pps_parse_back() {
        let (y, cb, cr) = gradient_planes(48, 32);
        let au = encode_idr_pcm_au(&y, &cb, &cr, 48, 32).expect("encode");
        let units = crate::nal::collect_nal_units(&au).expect("walk");
        assert_eq!(units.len(), 4);
        let sps = SeqParameterSet::parse(&units[1].rbsp).expect("sps parses");
        assert_eq!(sps.pic_width_in_luma_samples, 48);
        assert_eq!(sps.pic_height_in_luma_samples, 32);
        assert_eq!(sps.chroma_format_idc, 1);
        let pcm = sps.pcm.as_ref().expect("pcm block present");
        assert_eq!(pcm.bit_depth_luma_minus1, 7);
        assert!(pcm.loop_filter_disabled_flag);
        let pps = PicParameterSet::parse(&units[2].rbsp).expect("pps parses");
        assert!(pps.deblocking_filter_control_present_flag);
        assert!(pps.deblocking.disabled_flag);
    }
}
