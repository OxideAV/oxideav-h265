//! IDR I-slice header + PCM CTU payload emitter.
//!
//! Emits the minimal §7.3.6 `slice_segment_header()` followed by the
//! matching `slice_data()` body for a picture made of 64×64 PCM CUs.
//! The only CABAC elements consumed are:
//!
//! * `split_cu_flag` = 0 (per CTU at depth 0) — regular, context-modelled.
//! * `pcm_flag` = 1 (per CTU) — decoded via `decode_terminate`.
//! * `end_of_slice_flag` = 1 (at the very end) — also via `decode_terminate`.
//!
//! Per §9.3.2.6, after a PCM CU the arithmetic engine is re-initialised
//! at the next byte boundary; we match that by ending the CABAC run with
//! `encode_flush()`, byte-aligning, writing the raw samples, then starting
//! a new [`CabacWriter`] for the next CTU.

use crate::cabac::{init_context, InitType, SPLIT_CU_FLAG_INIT_VALUES};
use crate::encoder::bit_writer::{write_rbsp_trailing_bits, BitWriter};
use crate::encoder::cabac_writer::CabacWriter;
use crate::encoder::nal_writer::build_annex_b_nal;
use crate::encoder::params::EncoderConfig;
use crate::nal::{NalHeader, NalUnitType};
use oxideav_core::VideoFrame;

/// Build Annex B bytes for the IDR I-slice NAL (header + slice_data).
pub fn build_idr_slice_nal(cfg: &EncoderConfig, frame: &VideoFrame) -> Vec<u8> {
    let rbsp = build_idr_slice_rbsp(cfg, frame);
    build_annex_b_nal(NalHeader::for_type(NalUnitType::IdrNLp), &rbsp)
}

/// Build the RBSP payload (no 2-byte NAL header, no EBSP'ing) for the
/// IDR I-slice.
pub fn build_idr_slice_rbsp(cfg: &EncoderConfig, frame: &VideoFrame) -> Vec<u8> {
    let mut bw = BitWriter::new();

    // ---- slice_segment_header() -------------------------------------
    bw.write_u1(1); // first_slice_segment_in_pic_flag
    bw.write_u1(0); // no_output_of_prior_pics_flag (IDR only)
    bw.write_ue(0); // slice_pic_parameter_set_id

    // no extra header bits (num_extra_slice_header_bits = 0 in PPS).
    bw.write_ue(2); // slice_type = I

    // IDR → no slice_pic_order_cnt_lsb / short_term_ref_pic_set section.
    // SAO disabled in SPS.
    // Not P/B → skip the inter block.

    bw.write_se(0); // slice_qp_delta = 0 (init QP = 26)

    // pps_loop_filter_across_slices_enabled_flag = 1 in our PPS, SAO off,
    // slice_deblocking_filter_disabled_flag defaults to 0 (deblock on):
    // condition `(SAO || !deblock_disabled)` is true, so the flag is
    // signalled explicitly.
    bw.write_u1(1); // slice_loop_filter_across_slices_enabled_flag

    // No tiles, no wavefront — skip entry_point_offsets section.
    // No slice_segment_header_extension.

    // byte_alignment(): stop bit + zero bits.
    write_rbsp_trailing_bits(&mut bw);

    // ---- slice_data() -----------------------------------------------
    let ctu_size: u32 = 64;
    let pic_w_ctb = cfg.width.div_ceil(ctu_size);
    let pic_h_ctb = cfg.height.div_ceil(ctu_size);
    let total_ctbs = pic_w_ctb * pic_h_ctb;
    let slice_qp_y = 26;

    // Initialise the split_cu_flag contexts for an I slice.
    let init_type = InitType::I;
    let split_row = &SPLIT_CU_FLAG_INIT_VALUES[init_type as usize];
    let mut split_ctx = [
        init_context(split_row[0], slice_qp_y),
        init_context(split_row[1], slice_qp_y),
        init_context(split_row[2], slice_qp_y),
    ];

    // Walker structure (matches the decoder's CTU loop):
    //
    //   for each ctu:
    //       coding_quadtree(...)            // split_cu_flag + pcm_flag + samples
    //       engine.decode_terminate()        // end_of_slice_segment_flag (0/1)
    //
    // A PCM CU's pcm_flag terminator re-initialises the arithmetic
    // engine at the next byte boundary (§9.3.2.6). The
    // end_of_slice_segment_flag that follows is emitted under that
    // re-init'd engine; for a non-last CTU it is `0`, and it shares
    // the CABAC run with the NEXT CTU's split_cu_flag + pcm_flag.
    //
    // Concretely, every CABAC run emits one of:
    //   first CTU:   split_cu=0, pcm_flag=1, flush
    //   middle CTU:  end_of_slice=0, split_cu=0, pcm_flag=1, flush
    //   last CTU:    end_of_slice=1, flush                     (emitted
    //                 after the previous CTU's PCM body)

    for i in 0..total_ctbs {
        let ctb_x = (i % pic_w_ctb) * ctu_size;
        let ctb_y = (i / pic_w_ctb) * ctu_size;

        {
            let mut cabac = CabacWriter::new(&mut bw);
            if i > 0 {
                // Previous CTU's end_of_slice_segment_flag.
                cabac.encode_terminate(0);
            }
            cabac.encode_bin(&mut split_ctx[0], 0); // split_cu_flag = 0
            cabac.encode_terminate(1); // pcm_flag = 1
            cabac.encode_flush();
        }

        // pcm_alignment_zero_bit → next byte boundary.
        bw.align_to_byte_zero();
        // Raw PCM samples for this CTU.
        write_pcm_ctu_samples(&mut bw, frame, ctb_x, ctb_y, ctu_size);
    }

    // Final end_of_slice_segment_flag = 1 (after the last CTU's PCM body).
    {
        let mut cabac = CabacWriter::new(&mut bw);
        cabac.encode_terminate(1);
        cabac.encode_flush();
    }

    // slice_segment_data() rbsp_slice_segment_trailing_bits(): after the
    // CABAC flush the bitstream must be byte-aligned and a zero-bit pad
    // satisfies the trailing-bits contract. The encode_flush() already
    // emits bits + bypass_terminate; pad any residual to byte alignment.
    bw.align_to_byte_zero();

    bw.finish()
}

/// Copy an n×n luma block + matching n/2 × n/2 chroma blocks from the
/// frame into the bitstream as 8-bit raw PCM samples. Clamp to the
/// picture extent so encoding a crop-sized input (e.g. 64×64) doesn't
/// read past the buffers.
fn write_pcm_ctu_samples(bw: &mut BitWriter, frame: &VideoFrame, x0: u32, y0: u32, n: u32) {
    let y_plane = &frame.planes[0];
    let cb_plane = &frame.planes[1];
    let cr_plane = &frame.planes[2];
    let y_stride = y_plane.stride;
    let c_stride = cb_plane.stride;
    let w = frame.width;
    let h = frame.height;

    // Luma: n × n.
    for py in 0..n {
        for px in 0..n {
            let xx = (x0 + px).min(w.saturating_sub(1));
            let yy = (y0 + py).min(h.saturating_sub(1));
            let v = y_plane.data[yy as usize * y_stride + xx as usize];
            bw.write_bytes(&[v]);
        }
    }
    let cn = n / 2;
    let cw = w / 2;
    let ch = h / 2;
    let cx0 = x0 / 2;
    let cy0 = y0 / 2;
    // Cb.
    for py in 0..cn {
        for px in 0..cn {
            let xx = (cx0 + px).min(cw.saturating_sub(1));
            let yy = (cy0 + py).min(ch.saturating_sub(1));
            let v = cb_plane.data[yy as usize * c_stride + xx as usize];
            bw.write_bytes(&[v]);
        }
    }
    // Cr.
    for py in 0..cn {
        for px in 0..cn {
            let xx = (cx0 + px).min(cw.saturating_sub(1));
            let yy = (cy0 + py).min(ch.saturating_sub(1));
            let v = cr_plane.data[yy as usize * c_stride + xx as usize];
            bw.write_bytes(&[v]);
        }
    }
}
