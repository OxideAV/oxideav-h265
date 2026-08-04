//! Self-built §7.3.8.13 palette-mode conformance stream (test-only).
//!
//! No black-box encoder binary exposes the Screen Content Coding
//! tools, so this module assembles a tiny all-palette Annex B
//! bitstream from this crate's own header writers and CABAC encoder:
//! every 16x16 CTB is one `palette_mode_flag == 1` coding unit, and
//! the per-CTB plans exercise new-entry signalling, predictor reuse
//! (`palette_predictor_run`), explicit-index and copy-above runs, the
//! run-to-end inference, `MaxPaletteIndex == 0` degenerate blocks,
//! `palette_transpose_flag`, and escape samples in BOTH forms —
//! transquant-bypass (FL) and quantized (EG3, at `SliceQpY == 4`
//! where the eq. 8-77 dequantization is exact, keeping the whole
//! stream lossless).
//!
//! The checked-in copy under `tests/fixture_bytes/` is pinned by the
//! tests here: the builder must reproduce its exact bytes and this
//! crate's decoder must reconstruct the planned source planes
//! losslessly (reference-decoder validation notes live in
//! `tests/fixture_bytes/r413-generation-notes.md`).

use crate::cabac::init_type;
use crate::ctx_init::SliceContexts;
use crate::encoder::bitwriter::BitWriter;
use crate::encoder::cabac::CabacEncoder;
use crate::encoder::nal::{annexb, nal_unit};
use crate::encoder::rdpcm_streams::{write_pps_bypass, write_ptl_rext};
use crate::palette::{PaletteCu, PalettePredictor};

/// One picture's `(Y, Cb, Cr)` planes.
type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);
use crate::scan::traverse;

/// CTB / coding-block size (16x16, `MinCbSizeY == CtbSizeY`).
const CTB: usize = 16;
/// `SliceQpY` — 4, where escape dequantization (eq. 8-77) is exact:
/// `levelScale[4 % 6] = 64`, shift `4 / 6 = 0`, so
/// `((v * 64) + 32) >> 6 == v`.
const SLICE_QP: i32 = 4;
/// `palette_max_size` signalled in the SPS SCC extension.
const PALETTE_MAX_SIZE: u32 = 7;
/// `delta_palette_max_predictor_size`.
const DELTA_MAX_PREDICTOR: u32 = 9;

/// §7.3.2.2 — 4:2:0 8-bit SPS (CTB 16, no PCM, SAO off) with an
/// `sps_scc_extension()` enabling palette mode.
fn write_sps_scc(width: usize, height: usize, level_idc: u8, initializers: &[[u16; 3]]) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(0, 4); // sps_video_parameter_set_id
    w.put_bits(0, 3); // sps_max_sub_layers_minus1
    w.put_bit(1); // sps_temporal_id_nesting_flag
    write_ptl_rext(&mut w, level_idc);
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
    w.ue(1); // log2_min_luma_coding_block_size_minus3 (16)
    w.ue(0); // log2_diff_max_min_luma_coding_block_size (CTB 16)
    w.ue(0); // log2_min_luma_transform_block_size_minus2 (4)
    w.ue(2); // log2_diff_max_min_luma_transform_block_size (16)
    w.ue(0); // max_transform_hierarchy_depth_inter
    w.ue(0); // max_transform_hierarchy_depth_intra
    w.put_bit(0); // scaling_list_enabled_flag
    w.put_bit(0); // amp_enabled_flag
    w.put_bit(0); // sample_adaptive_offset_enabled_flag
    w.put_bit(0); // pcm_enabled_flag
    w.ue(0); // num_short_term_ref_pic_sets
    w.put_bit(0); // long_term_ref_pics_present_flag
    w.put_bit(0); // sps_temporal_mvp_enabled_flag
    w.put_bit(0); // strong_intra_smoothing_enabled_flag
    w.put_bit(0); // vui_parameters_present_flag
    w.put_bit(1); // sps_extension_present_flag
    w.put_bit(0); // sps_range_extension_flag
    w.put_bit(0); // sps_multilayer_extension_flag
    w.put_bit(0); // sps_3d_extension_flag
    w.put_bit(1); // sps_scc_extension_flag
    w.put_bits(0, 4); // sps_extension_4bits
                      // sps_scc_extension() — §7.3.2.2.3.
    w.put_bit(0); // sps_curr_pic_ref_enabled_flag
    w.put_bit(1); // palette_mode_enabled_flag
    w.ue(PALETTE_MAX_SIZE); // palette_max_size
    w.ue(DELTA_MAX_PREDICTOR); // delta_palette_max_predictor_size
    if initializers.is_empty() {
        w.put_bit(0); // sps_palette_predictor_initializers_present_flag
    } else {
        w.put_bit(1); // sps_palette_predictor_initializers_present_flag
        w.ue(initializers.len() as u32 - 1); // sps_num_..._minus1
                                             // §7.3.2.2.3: component-major, u(BitDepth) each (8 bits here).
        for c in 0..3 {
            for e in initializers {
                w.put_bits(u32::from(e[c]), 8);
            }
        }
    }
    w.put_bits(0, 2); // motion_vector_resolution_control_idc
    w.put_bit(0); // intra_boundary_filtering_disabled_flag
    w.rbsp_trailing_bits();
    w.finish()
}

/// One planned palette CU: the palette (per component), the target
/// index map (coded coordinates), transpose, bypass and the escape
/// samples (`Some(values per component)` at positions whose index is
/// `MaxPaletteIndex`).
struct CuPlan {
    /// Reuse flags into the CURRENT predictor (length =
    /// `PredictorPaletteSize` at this CU).
    reuse: Vec<bool>,
    /// Explicitly signalled new entries (per component).
    new_entries: [Vec<u16>; 3],
    /// Escape samples present.
    escape_present: bool,
    /// `palette_transpose_flag`.
    transpose: bool,
    /// `cu_transquant_bypass_flag`.
    bypass: bool,
    /// Target `PaletteIndexMap` (coded coords, `CTB * CTB`).
    index_map: Vec<u8>,
    /// Escape values per component at coded luma coords (only
    /// positions whose index equals `MaxPaletteIndex` and pass the
    /// §7.3.8.13 4:2:0 phase gate are consulted).
    escape_vals: [Vec<u16>; 3],
}

/// Bypass-encode an EG0 value.
fn encode_eg0(cabac: &mut CabacEncoder, w: &mut BitWriter, v: u32) {
    encode_eg_k(cabac, w, v, 0);
}

/// §9.3.3.3 dual of `decode_eg_k`.
fn encode_eg_k(cabac: &mut CabacEncoder, w: &mut BitWriter, v: u32, k: u32) {
    let mut prefix_ones = 0u32;
    while (((1u64 << (prefix_ones + 1)) - 1) << k) <= u64::from(v) {
        prefix_ones += 1;
    }
    for _ in 0..prefix_ones {
        cabac.encode_bypass(w, 1);
    }
    cabac.encode_bypass(w, 0);
    let base = ((1u64 << prefix_ones) - 1) << k;
    let suffix = u64::from(v) - base;
    let bits = prefix_ones + k;
    for i in (0..bits).rev() {
        cabac.encode_bypass(w, ((suffix >> i) & 1) as u8);
    }
}

/// §9.3.3.6 dual of `decode_tb`.
fn encode_tb(cabac: &mut CabacEncoder, w: &mut BitWriter, v: u32, c_max: u32) {
    if c_max == 0 {
        return;
    }
    let n = c_max + 1;
    let k = 31 - n.leading_zeros();
    let u = (1u32 << (k + 1)) - n;
    if v < u {
        for i in (0..k).rev() {
            cabac.encode_bypass(w, ((v >> i) & 1) as u8);
        }
    } else {
        let val = v + u;
        for i in (0..=k).rev() {
            cabac.encode_bypass(w, ((val >> i) & 1) as u8);
        }
    }
}

/// §9.3.3.14 dual.
fn encode_num_palette_indices_minus1(
    cabac: &mut CabacEncoder,
    w: &mut BitWriter,
    v: u32,
    max_palette_index: u32,
) {
    let k = 3 + ((max_palette_index + 1) >> 3);
    let c_max = 4u32 << k;
    if v < c_max {
        let q = v >> k;
        for _ in 0..q {
            cabac.encode_bypass(w, 1);
        }
        cabac.encode_bypass(w, 0);
        for i in (0..k).rev() {
            cabac.encode_bypass(w, ((v >> i) & 1) as u8);
        }
    } else {
        for _ in 0..4 {
            cabac.encode_bypass(w, 1);
        }
        encode_eg_k(cabac, w, v - c_max, k + 1);
    }
}

/// The run segmentation of a target index map: `(copy_above, explicit
/// index or 0, run_minus1)` per run, in scan order.
struct Run {
    copy: bool,
    index: u8,
    len: usize, // PaletteRunMinus1 + 1
}

/// Greedy §7.3.8.13-compatible segmentation: maximal runs, preferring
/// copy-above when it is at least as long as the explicit run (and
/// syntactically expressible).
fn segment_runs(index_map: &[u8], n: usize) -> Vec<Run> {
    let scan = traverse(n);
    let area = n * n;
    let mut runs: Vec<Run> = Vec::new();
    let mut pos = 0usize;
    while pos < area {
        let at = |p: usize| {
            let s = &scan[p];
            index_map[s.y as usize * n + s.x as usize]
        };
        let above = |p: usize| {
            let s = &scan[p];
            debug_assert!(s.y > 0);
            index_map[(s.y as usize - 1) * n + s.x as usize]
        };
        let prev_copy = runs.last().is_some_and(|r| r.copy);
        // Maximal explicit run of the current index.
        let idx = at(pos);
        let mut exp_len = 1usize;
        while pos + exp_len < area && at(pos + exp_len) == idx {
            exp_len += 1;
        }
        // Maximal copy-above run (when expressible here).
        let mut copy_len = 0usize;
        if pos >= n && !prev_copy {
            while pos + copy_len < area
                && scan[pos + copy_len].y > 0
                && at(pos + copy_len) == above(pos + copy_len)
            {
                copy_len += 1;
            }
        }
        if copy_len >= exp_len && copy_len > 0 {
            runs.push(Run {
                copy: true,
                index: 0,
                len: copy_len,
            });
            pos += copy_len;
        } else {
            runs.push(Run {
                copy: false,
                index: idx,
                len: exp_len,
            });
            pos += exp_len;
        }
    }
    runs
}

/// Encode one `palette_coding( )` body for `plan`, mirroring the
/// §7.3.8.13 parse gate-for-gate (panicking if the plan is not
/// syntactically representable), and apply the eq. 8-79 predictor
/// update to `predictor`.
#[allow(clippy::too_many_lines)]
fn encode_palette_cu(
    cabac: &mut CabacEncoder,
    w: &mut BitWriter,
    ctx: &mut SliceContexts,
    predictor: &mut PalettePredictor,
    plan: &CuPlan,
) -> PaletteCu {
    let n = CTB;
    let area = n * n;
    let predictor_size = predictor.size();
    assert_eq!(plan.reuse.len(), predictor_size);

    // ---- palette_predictor_run ----
    let num_predicted = plan.reuse.iter().filter(|&&r| r).count();
    {
        let mut last = -1i64;
        let mut emitted = 0usize;
        for (i, &r) in plan.reuse.iter().enumerate() {
            if !r {
                continue;
            }
            let run = if emitted == 0 {
                i as u32
            } else {
                (i as i64 - last) as u32
            };
            // run semantics: 0 = reuse at current cursor; >1 = skip
            // run-1 then reuse; the value 1 is the terminator, so a
            // gap of exactly one entry is encoded as run = 2, etc.
            let wire = if emitted == 0 {
                if i == 0 {
                    0
                } else {
                    i as u32 + 1
                }
            } else {
                let gap = (i as i64 - last - 1) as u32;
                if gap == 0 {
                    0
                } else {
                    gap + 1
                }
            };
            let _ = run;
            encode_eg0(cabac, w, wire);
            emitted += 1;
            last = i as i64;
        }
        // Terminator when the loop would otherwise continue.
        if predictor_size > 0
            && num_predicted < PALETTE_MAX_SIZE as usize
            && (emitted == 0 || (last as usize) < predictor_size - 1)
        {
            encode_eg0(cabac, w, 1);
        }
    }

    // ---- num_signalled_palette_entries + new entries ----
    let num_signalled = plan.new_entries[0].len();
    if num_predicted < PALETTE_MAX_SIZE as usize {
        encode_eg0(cabac, w, num_signalled as u32);
    } else {
        assert_eq!(num_signalled, 0);
    }
    for c in 0..3 {
        for &v in &plan.new_entries[c] {
            cabac.encode_bypass_bits(w, u32::from(v), 8);
        }
    }

    // ---- CurrentPaletteEntries (eq. 7-82) ----
    let mut palette: [Vec<u16>; 3] = Default::default();
    for (c, pal) in palette.iter_mut().enumerate() {
        for (i, &r) in plan.reuse.iter().enumerate() {
            if r {
                pal.push(predictor.entries[c][i]);
            }
        }
        pal.extend_from_slice(&plan.new_entries[c]);
    }
    let current_size = palette[0].len();

    // ---- palette_escape_val_present_flag ----
    if current_size != 0 {
        cabac.encode_bypass(w, u8::from(plan.escape_present));
    } else {
        assert!(plan.escape_present, "empty palette requires escapes");
    }
    let max_index = current_size as u32 + u32::from(plan.escape_present) - 1;

    // ---- run segmentation, explicit index list ----
    let runs = segment_runs(&plan.index_map, n);
    let explicit: Vec<&Run> = runs.iter().filter(|r| !r.copy).collect();
    let num_indices = explicit.len();
    let final_copy = runs.last().map(|r| r.copy).unwrap_or(false);
    // Per-run RAW `palette_idx_idc` values (0 for copy-above runs):
    // §9.3.4.2.8 derives the `palette_run_prefix` ctxInc from the
    // signalled syntax element, so the run loop below needs the
    // pre-eq.-7-84 value, not the plan's final index.
    let mut idc_by_run: Vec<u32> = vec![0; runs.len()];

    if max_index > 0 {
        encode_num_palette_indices_minus1(cabac, w, num_indices as u32 - 1, max_index);
        // The idc list needs the eq. 7-83/7-84 INVERSE adjustment,
        // which depends on scan state; walk the runs to derive each
        // adjusted idc.
        let scan = traverse(n);
        let mut adjust = 0u32;
        let mut pos = 0usize;
        let mut prev_copy = false;
        for (run_i, r) in runs.iter().enumerate() {
            if !r.copy {
                let adjusted_ref = if pos > 0 {
                    let p = &scan[pos - 1];
                    if !prev_copy {
                        u32::from(plan.index_map[p.y as usize * n + p.x as usize])
                    } else {
                        let c = &scan[pos];
                        u32::from(plan.index_map[(c.y as usize - 1) * n + c.x as usize])
                    }
                } else {
                    max_index + 1
                };
                let target = u32::from(r.index);
                assert_ne!(
                    target, adjusted_ref,
                    "explicit index equal to adjustedRefPaletteIndex is not encodable"
                );
                let idc = if target > adjusted_ref {
                    target - 1
                } else {
                    target
                };
                idc_by_run[run_i] = idc;
                let c_max = max_index - adjust;
                if c_max > 0 {
                    encode_tb(cabac, w, idc, c_max);
                } else {
                    assert_eq!(idc, 0);
                }
                adjust = 1;
            }
            prev_copy = r.copy;
            pos += r.len;
        }
        // copy_above_indices_for_final_run_flag + transpose.
        let ctxm = &mut ctx.palette_copy_above_flag[0];
        cabac.encode_decision(w, ctxm, u8::from(final_copy));
        cabac.encode_decision(
            w,
            &mut ctx.palette_transpose_flag[0],
            u8::from(plan.transpose),
        );
    } else {
        assert_eq!(num_indices, 1, "MaxPaletteIndex 0 has one inferred index");
        assert!(!plan.transpose);
    }

    // (delta_qp: cu_qp_delta_enabled == 0 in this stream ⇒ no bins.)

    // ---- run loop ----
    let mut remaining = num_indices;
    let mut pos = 0usize;
    let mut prev_copy = false;
    for (run_i, r) in runs.iter().enumerate() {
        // Mirror the parse-side flag presence gates.
        if max_index > 0 && pos >= n && !prev_copy {
            if remaining > 0 && pos < area - 1 {
                cabac.encode_decision(w, &mut ctx.palette_copy_above_flag[0], u8::from(r.copy));
            } else {
                let inferred = !(pos == area - 1 && remaining > 0);
                assert_eq!(r.copy, inferred, "inferred copy flag must match the plan");
            }
        } else {
            assert!(!r.copy, "copy-above not expressible at pos {pos}");
        }
        // §9.3.4.2.8: the run-prefix ctxInc consumes the RAW
        // signalled `palette_idx_idc`, not the final palette index.
        let idc_for_ctx = if r.copy { 0 } else { idc_by_run[run_i] };
        if max_index > 0 {
            if !r.copy {
                remaining -= 1;
            }
            if remaining > 0 || r.copy != final_copy {
                let max_run_minus1 =
                    area as i64 - pos as i64 - 1 - remaining as i64 - i64::from(final_copy);
                assert!(max_run_minus1 >= 0);
                let run_minus1 = (r.len - 1) as u32;
                assert!(i64::from(run_minus1) <= max_run_minus1);
                if max_run_minus1 > 0 {
                    // palette_run_prefix / suffix.
                    let c_max =
                        crate::binarization::palette_run_prefix_tr_cmax(max_run_minus1 as u32);
                    let (prefix, suffix) = if run_minus1 < 2 {
                        (run_minus1, None)
                    } else {
                        let p = 32 - run_minus1.leading_zeros(); // floor(log2(v)) + 1
                        let prefix_offset = 1u32 << (p - 1);
                        debug_assert!(prefix_offset <= run_minus1);
                        (p, Some(run_minus1 - prefix_offset))
                    };
                    // TR prefix: `prefix` ones then a terminating zero
                    // (omitted at cMax).
                    for bin_idx in 0..prefix {
                        emit_run_prefix_bin(cabac, w, ctx, bin_idx, r.copy, idc_for_ctx, 1);
                    }
                    if prefix < c_max {
                        emit_run_prefix_bin(cabac, w, ctx, prefix, r.copy, idc_for_ctx, 0);
                    }
                    if let Some(sfx) = suffix {
                        let prefix_offset = 1u32 << (prefix - 1);
                        if max_run_minus1 as u32 != prefix_offset {
                            let c_max_tb = if (prefix_offset << 1) > max_run_minus1 as u32 {
                                max_run_minus1 as u32 - prefix_offset
                            } else {
                                prefix_offset - 1
                            };
                            encode_tb(cabac, w, sfx, c_max_tb);
                        } else {
                            assert_eq!(sfx, 0);
                        }
                    }
                } else {
                    assert_eq!(run_minus1, 0);
                }
            } else {
                // RunToEnd inference: the run must actually reach the
                // block end.
                assert_eq!(pos + r.len, area, "final run must cover the tail");
            }
        }
        prev_copy = r.copy;
        pos += r.len;
    }

    // ---- escape values ----
    if plan.escape_present {
        let scan = traverse(n);
        for c in 0..3usize {
            for p in &scan {
                let (x, y) = (p.x as usize, p.y as usize);
                if u32::from(plan.index_map[y * n + x]) != max_index {
                    continue;
                }
                let present = c == 0 || (x % 2 == 0 && y % 2 == 0);
                if !present {
                    continue;
                }
                let v = u32::from(plan.escape_vals[c][y * n + x]);
                if plan.bypass {
                    cabac.encode_bypass_bits(w, v, 8);
                } else {
                    encode_eg_k(cabac, w, v, 3);
                }
            }
        }
    }

    // ---- eq. 8-79 predictor update ----
    let max_pred = (PALETTE_MAX_SIZE + DELTA_MAX_PREDICTOR) as usize;
    let mut new_pred: [Vec<u16>; 3] = palette.clone();
    let mut new_size = current_size;
    for (i, &r) in plan.reuse.iter().enumerate() {
        if new_size >= max_pred {
            break;
        }
        if !r {
            for (c, np) in new_pred.iter_mut().enumerate() {
                np.push(predictor.entries[c][i]);
            }
            new_size += 1;
        }
    }
    *predictor = PalettePredictor { entries: new_pred };

    PaletteCu {
        n_cbs: n,
        palette,
        escape_present: plan.escape_present,
        transpose: plan.transpose,
        index_map: plan.index_map.clone(),
        escape_vals: plan.escape_vals.clone(),
        cu_qp_delta: None,
        cu_chroma_qp_offset: None,
    }
}

fn emit_run_prefix_bin(
    cabac: &mut CabacEncoder,
    w: &mut BitWriter,
    ctx: &mut SliceContexts,
    bin_idx: u32,
    copy: bool,
    idc: u32,
    bin: u8,
) {
    match crate::binarization::palette_run_prefix_ctx_inc(bin_idx, copy, idc) {
        Some(inc) => cabac.encode_decision(w, &mut ctx.palette_run_prefix[inc as usize], bin),
        None => cabac.encode_bypass(w, bin),
    }
}

/// The per-CTB plans for a 64x48 picture (4x3 CTBs).
fn cu_plans(predictor_snapshots: &mut Vec<PalettePredictor>) -> Vec<CuPlan> {
    let n = CTB;
    let _ = predictor_snapshots;
    let mut plans = Vec::new();

    // CTB 0: four new colours, horizontal bands (explicit runs only).
    plans.push(CuPlan {
        reuse: vec![],
        new_entries: [
            vec![30, 90, 160, 220],
            vec![60, 100, 150, 200],
            vec![200, 150, 100, 50],
        ],
        escape_present: false,
        transpose: false,
        bypass: true,
        index_map: {
            let mut m = vec![0u8; n * n];
            for y in 0..n {
                for x in 0..n {
                    m[y * n + x] = (y / 4) as u8;
                }
            }
            m
        },
        escape_vals: Default::default(),
    });

    // CTB 1: reuse predictor entries 0 and 2, one new colour;
    // vertical stripes (row 0 explicit, the rest copy-above runs).
    plans.push(CuPlan {
        reuse: vec![true, false, true, false],
        new_entries: [vec![120], vec![130], vec![140]],
        escape_present: false,
        transpose: false,
        bypass: true,
        index_map: {
            let mut m = vec![0u8; n * n];
            for y in 0..n {
                for x in 0..n {
                    m[y * n + x] = ((x / 2) % 3) as u8;
                }
            }
            m
        },
        escape_vals: Default::default(),
    });

    // CTB 2: single colour (MaxPaletteIndex == 0 degenerate: no index
    // list, no run coding).
    plans.push(CuPlan {
        reuse: vec![false, false, true, false, false],
        new_entries: [vec![], vec![], vec![]],
        escape_present: false,
        transpose: false,
        bypass: true,
        index_map: vec![0u8; n * n],
        escape_vals: Default::default(),
    });

    // CTB 3: two colours + BYPASS escapes on a sparse grid.
    plans.push(CuPlan {
        reuse: vec![true, true, false, false, false],
        new_entries: [vec![], vec![], vec![]],
        escape_present: true,
        transpose: false,
        bypass: true,
        index_map: {
            let mut m = vec![0u8; n * n];
            for y in 0..n {
                for x in 0..n {
                    m[y * n + x] = u8::from(y >= 8);
                }
            }
            // Escape index = MaxPaletteIndex = 2, at even positions.
            m[2 * n + 2] = 2;
            m[2 * n + 3] = 2; // odd x: luma escape only (4:2:0 gate)
            m[10 * n + 6] = 2;
            m
        },
        escape_vals: {
            let mut y_v = vec![0u16; n * n];
            let mut cb_v = vec![0u16; n * n];
            let mut cr_v = vec![0u16; n * n];
            y_v[2 * n + 2] = 250;
            y_v[2 * n + 3] = 17;
            y_v[10 * n + 6] = 3;
            cb_v[2 * n + 2] = 77;
            cr_v[2 * n + 2] = 99;
            cb_v[10 * n + 6] = 5;
            cr_v[10 * n + 6] = 240;
            [y_v, cb_v, cr_v]
        },
    });

    // CTB 4: transpose stripes (coded rows become picture columns).
    plans.push(CuPlan {
        reuse: vec![true, true, true, false, false],
        new_entries: [vec![], vec![], vec![]],
        escape_present: false,
        transpose: true,
        bypass: true,
        index_map: {
            let mut m = vec![0u8; n * n];
            for y in 0..n {
                for x in 0..n {
                    m[y * n + x] = ((y / 3) % 3) as u8;
                }
            }
            m
        },
        escape_vals: Default::default(),
    });

    // CTB 5: QUANTIZED escapes (non-bypass CU at SliceQpY 4 — exact).
    plans.push(CuPlan {
        reuse: vec![true, false, false, false, false],
        new_entries: [vec![10], vec![20], vec![30]],
        escape_present: true,
        transpose: false,
        bypass: false,
        index_map: {
            let mut m = vec![0u8; n * n];
            for y in 0..n {
                for x in 0..n {
                    m[y * n + x] = u8::from((x + y) % 4 == 0);
                }
            }
            // A 2x2-aligned escape block.
            m[4 * n + 4] = 2;
            m[12 * n + 8] = 2;
            m
        },
        escape_vals: {
            let mut y_v = vec![0u16; n * n];
            let mut cb_v = vec![0u16; n * n];
            let mut cr_v = vec![0u16; n * n];
            y_v[4 * n + 4] = 133;
            y_v[12 * n + 8] = 255;
            cb_v[4 * n + 4] = 44;
            cr_v[4 * n + 4] = 211;
            cb_v[12 * n + 8] = 0;
            cr_v[12 * n + 8] = 128;
            [y_v, cb_v, cr_v]
        },
    });

    // CTBs 6..11: diagonal two-colour texture cycling reuse patterns
    // (keeps the predictor evolving).
    for k in 0..6usize {
        plans.push(CuPlan {
            reuse: vec![], // filled at build time (reuse first two)
            new_entries: Default::default(),
            escape_present: false,
            transpose: k % 2 == 1,
            bypass: true,
            index_map: {
                let mut m = vec![0u8; n * n];
                for y in 0..n {
                    for x in 0..n {
                        m[y * n + x] = u8::from((x + y + k) % 8 < 4);
                    }
                }
                m
            },
            escape_vals: Default::default(),
        });
    }

    plans
}

/// Build the palette conformance stream: VPS + SPS(SCC) + PPS + one
/// all-palette IDR picture (64x48). Returns the Annex B bytes and the
/// expected (lossless) decoded planes.
pub(crate) fn build_palette_stream() -> (Vec<u8>, Planes) {
    let (width, height) = (64usize, 48usize);
    let (cw, ch) = (width / 2, height / 2);
    let ctbs_x = width / CTB;

    let mut w = BitWriter::new();
    // ---- slice_segment_header( ) ----
    w.put_bit(1); // first_slice_segment_in_pic_flag
    w.put_bit(0); // no_output_of_prior_pics_flag (IRAP NAL)
    w.ue(0); // slice_pic_parameter_set_id
    w.ue(2); // slice_type = I
    w.se(SLICE_QP - 26); // slice_qp_delta
    w.rbsp_trailing_bits();

    let mut cabac = CabacEncoder::new();
    let mut ctx = SliceContexts::init(init_type(2, false), SLICE_QP);
    let mut predictor = PalettePredictor::default();

    let mut y_plane = vec![0u8; width * height];
    let mut cb_plane = vec![0u8; cw * ch];
    let mut cr_plane = vec![0u8; cw * ch];

    let mut snapshots = Vec::new();
    let mut plans = cu_plans(&mut snapshots);
    for (ctb, plan) in plans.iter_mut().enumerate() {
        let x0 = (ctb % ctbs_x) * CTB;
        let y0 = (ctb / ctbs_x) * CTB;

        // Late-bound reuse vectors for the tail CTBs: reuse the first
        // two predictor entries.
        if ctb >= 6 {
            let mut reuse = vec![false; predictor.size()];
            if reuse.len() >= 2 {
                reuse[0] = true;
                reuse[1] = true;
            }
            plan.reuse = reuse;
        }

        // ---- coding_unit( ) prelude ----
        cabac.encode_decision(
            &mut w,
            &mut ctx.cu_transquant_bypass_flag[0],
            u8::from(plan.bypass),
        );
        // palette_mode_flag = 1.
        cabac.encode_decision(&mut w, &mut ctx.palette_mode_flag[0], 1);

        let cu = encode_palette_cu(&mut cabac, &mut w, &mut ctx, &mut predictor, plan);

        // Reference reconstruction into the expected planes (§8.4.4.2.7
        // with the exact-at-QP-4 escape dequantization).
        crate::palette::reconstruct_palette_component(
            &cu,
            0,
            1,
            1,
            SLICE_QP,
            8,
            plan.bypass,
            |x, y, v| {
                y_plane[(y0 + y) * width + x0 + x] = v as u8;
            },
        );
        crate::palette::reconstruct_palette_component(
            &cu,
            1,
            2,
            2,
            SLICE_QP,
            8,
            plan.bypass,
            |x, y, v| {
                cb_plane[(y0 / 2 + y) * cw + x0 / 2 + x] = v as u8;
            },
        );
        crate::palette::reconstruct_palette_component(
            &cu,
            2,
            2,
            2,
            SLICE_QP,
            8,
            plan.bypass,
            |x, y, v| {
                cr_plane[(y0 / 2 + y) * cw + x0 / 2 + x] = v as u8;
            },
        );

        cabac.encode_terminate(&mut w, u8::from(ctb == 11));
    }
    w.align_zero();
    let slice = w.finish();

    let units = vec![
        nal_unit(32, 0, 0, &{
            // Reuse the Rext VPS shape (PTL profile bits are opaque to
            // the decode path).
            let mut vw = BitWriter::new();
            vw.put_bits(0, 4);
            vw.put_bit(1);
            vw.put_bit(1);
            vw.put_bits(0, 6);
            vw.put_bits(0, 3);
            vw.put_bit(1);
            vw.put_bits(0xFFFF, 16);
            write_ptl_rext(&mut vw, 30);
            vw.put_bit(1);
            vw.ue(1);
            vw.ue(0);
            vw.ue(0);
            vw.put_bits(0, 6);
            vw.ue(0);
            vw.put_bit(0);
            vw.put_bit(0);
            vw.rbsp_trailing_bits();
            vw.finish()
        }),
        nal_unit(33, 0, 0, &write_sps_scc(width, height, 30, &[])),
        nal_unit(34, 0, 0, &write_pps_bypass()),
        nal_unit(20, 0, 0, &slice), // IDR_N_LP
    ];
    (annexb(&units), (y_plane, cb_plane, cr_plane))
}

/// Second pin: §9.3.2.3 SPS palette predictor INITIALIZERS + per-
/// independent-slice predictor re-initialization. A 32x32 picture
/// (2x2 CTBs) in TWO independent slices; the SPS carries four
/// predictor initializer entries, so every slice starts with a
/// four-entry predictor. CU plans: reuse-only palettes from the
/// initializers, a MaxPaletteIndex-0 block at the second slice's
/// start (proving the re-init), and a 256-run diagonal block whose
/// index count exercises the §9.3.3.14 all-ones escape suffix.
pub(crate) fn build_palette_init_stream() -> (Vec<u8>, Planes) {
    const INIT: [[u16; 3]; 4] = [[25, 60, 190], [80, 90, 100], [140, 120, 70], [200, 180, 40]];
    let (width, height) = (32usize, 32usize);
    let (cw, ch) = (width / 2, height / 2);
    let ctbs_x = width / CTB;
    let n = CTB;

    let base_predictor = PalettePredictor {
        entries: [
            INIT.iter().map(|e| e[0]).collect(),
            INIT.iter().map(|e| e[1]).collect(),
            INIT.iter().map(|e| e[2]).collect(),
        ],
    };

    let plans: Vec<CuPlan> = vec![
        // CTB 0 (slice 1): reuse initializer entries 0 and 2; stripes.
        CuPlan {
            reuse: vec![true, false, true, false],
            new_entries: Default::default(),
            escape_present: false,
            transpose: false,
            bypass: true,
            index_map: {
                let mut m = vec![0u8; n * n];
                for y in 0..n {
                    for x in 0..n {
                        m[y * n + x] = u8::from((x / 4) % 2 == 1);
                    }
                }
                m
            },
            escape_vals: Default::default(),
        },
        // CTB 1 (slice 1): predictor evolved to [i0, i2, i1, i3];
        // reuse the outer two; horizontal bands.
        CuPlan {
            reuse: vec![true, false, false, true],
            new_entries: Default::default(),
            escape_present: false,
            transpose: false,
            bypass: true,
            index_map: {
                let mut m = vec![0u8; n * n];
                for y in 0..n {
                    for x in 0..n {
                        m[y * n + x] = u8::from(y >= 8);
                    }
                }
                m
            },
            escape_vals: Default::default(),
        },
        // CTB 2 (slice 2, FIRST CU): the predictor must be the
        // re-initialized four initializer entries again — reuse entry
        // 1 only (MaxPaletteIndex == 0 degenerate block).
        CuPlan {
            reuse: vec![false, true, false, false],
            new_entries: Default::default(),
            escape_present: false,
            transpose: false,
            bypass: true,
            index_map: vec![0u8; n * n],
            escape_vals: Default::default(),
        },
        // CTB 3 (slice 2): reuse all four evolved entries; diagonal
        // (x + y) % 4 texture — 256 single-sample explicit runs, so
        // num_palette_indices_minus1 = 255 takes the §9.3.3.14
        // all-ones prefix + EGk escape.
        CuPlan {
            reuse: vec![true, true, true, true],
            new_entries: Default::default(),
            escape_present: false,
            transpose: false,
            bypass: true,
            index_map: {
                let mut m = vec![0u8; n * n];
                for y in 0..n {
                    for x in 0..n {
                        m[y * n + x] = ((x + y) % 4) as u8;
                    }
                }
                m
            },
            escape_vals: Default::default(),
        },
    ];

    let mut y_plane = vec![0u8; width * height];
    let mut cb_plane = vec![0u8; cw * ch];
    let mut cr_plane = vec![0u8; cw * ch];

    // Two independent slices: CTBs [0, 1] and [2, 3].
    let mut slice_rbsps: Vec<Vec<u8>> = Vec::new();
    for (slice_idx, ctbs) in [[0usize, 1], [2, 3]].iter().enumerate() {
        let mut w = BitWriter::new();
        let first = slice_idx == 0;
        w.put_bit(u8::from(first)); // first_slice_segment_in_pic_flag
        w.put_bit(0); // no_output_of_prior_pics_flag (IRAP NAL)
        w.ue(0); // slice_pic_parameter_set_id
        if !first {
            // slice_segment_address: Ceil(Log2(4)) = 2 bits.
            w.put_bits(ctbs[0] as u32, 2);
        }
        w.ue(2); // slice_type = I
        w.se(SLICE_QP - 26); // slice_qp_delta
        w.rbsp_trailing_bits();

        let mut cabac = CabacEncoder::new();
        let mut ctx = SliceContexts::init(init_type(2, false), SLICE_QP);
        // §9.3.2.3 — every independent slice re-initializes the
        // predictor from the SPS initializers.
        let mut predictor = base_predictor.clone();

        for &ctb in ctbs {
            let x0 = (ctb % ctbs_x) * CTB;
            let y0 = (ctb / ctbs_x) * CTB;
            let plan = &plans[ctb];
            cabac.encode_decision(
                &mut w,
                &mut ctx.cu_transquant_bypass_flag[0],
                u8::from(plan.bypass),
            );
            cabac.encode_decision(&mut w, &mut ctx.palette_mode_flag[0], 1);
            let cu = encode_palette_cu(&mut cabac, &mut w, &mut ctx, &mut predictor, plan);
            crate::palette::reconstruct_palette_component(
                &cu,
                0,
                1,
                1,
                SLICE_QP,
                8,
                plan.bypass,
                |x, y, v| {
                    y_plane[(y0 + y) * width + x0 + x] = v as u8;
                },
            );
            crate::palette::reconstruct_palette_component(
                &cu,
                1,
                2,
                2,
                SLICE_QP,
                8,
                plan.bypass,
                |x, y, v| {
                    cb_plane[(y0 / 2 + y) * cw + x0 / 2 + x] = v as u8;
                },
            );
            crate::palette::reconstruct_palette_component(
                &cu,
                2,
                2,
                2,
                SLICE_QP,
                8,
                plan.bypass,
                |x, y, v| {
                    cr_plane[(y0 / 2 + y) * cw + x0 / 2 + x] = v as u8;
                },
            );
            cabac.encode_terminate(&mut w, u8::from(ctb == ctbs[1]));
        }
        w.align_zero();
        slice_rbsps.push(w.finish());
    }

    let mut units = vec![
        nal_unit(32, 0, 0, &{
            let mut vw = BitWriter::new();
            vw.put_bits(0, 4);
            vw.put_bit(1);
            vw.put_bit(1);
            vw.put_bits(0, 6);
            vw.put_bits(0, 3);
            vw.put_bit(1);
            vw.put_bits(0xFFFF, 16);
            write_ptl_rext(&mut vw, 30);
            vw.put_bit(1);
            vw.ue(1);
            vw.ue(0);
            vw.ue(0);
            vw.put_bits(0, 6);
            vw.ue(0);
            vw.put_bit(0);
            vw.put_bit(0);
            vw.rbsp_trailing_bits();
            vw.finish()
        }),
        nal_unit(33, 0, 0, &write_sps_scc(width, height, 30, &INIT)),
        nal_unit(34, 0, 0, &write_pps_bypass()),
    ];
    for rbsp in &slice_rbsps {
        units.push(nal_unit(20, 0, 0, rbsp)); // IDR_N_LP
    }
    (annexb(&units), (y_plane, cb_plane, cr_plane))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::decode_annexb_sequence;

    /// SPS predictor initializers + two independent slices: the
    /// second slice's first CU must see the re-initialized
    /// four-entry predictor (§9.3.2.3), not the first slice's
    /// evolved one.
    #[test]
    fn palette_initializer_stream_decodes_lossless() {
        let (stream, (y, cb, cr)) = build_palette_init_stream();
        let pinned: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixture_bytes/r413-palette-init.hevc"
        ));
        assert_eq!(stream, pinned, "builder must reproduce the validated bytes");
        let frames = decode_annexb_sequence(&stream).expect("decode");
        assert_eq!(frames.len(), 1);
        let mut expected = y;
        expected.extend(cb);
        expected.extend(cr);
        assert_eq!(
            frames[0].picture.to_planar_u8().expect("8-bit planes"),
            expected,
            "lossless palette decode with SPS initializers"
        );
    }

    /// The checked-in stream bytes are exactly what the builder
    /// produces, and this crate's decoder reconstructs the planned
    /// palette content losslessly (predictor reuse, explicit /
    /// copy-above runs, run-to-end, MaxPaletteIndex 0, transpose, and
    /// bypass + quantized escapes).
    #[test]
    fn palette_stream_decodes_lossless() {
        let (stream, (y, cb, cr)) = build_palette_stream();
        let pinned: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixture_bytes/r413-palette.hevc"
        ));
        assert_eq!(stream, pinned, "builder must reproduce the validated bytes");
        let frames = decode_annexb_sequence(&stream).expect("decode");
        assert_eq!(frames.len(), 1);
        let mut expected = y;
        expected.extend(cb);
        expected.extend(cr);
        assert_eq!(
            frames[0].picture.to_planar_u8().expect("8-bit planes"),
            expected,
            "lossless palette decode"
        );
    }
}
