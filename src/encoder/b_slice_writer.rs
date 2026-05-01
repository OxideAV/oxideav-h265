//! B-slice header + CABAC slice-data emission (rounds 22–23).
//!
//! Scope (round 23):
//!
//! * Single-segment B (TrailR) slice referencing two pictures: one
//!   earlier-POC L0 reference and one later-POC L1 reference. Both list
//!   sizes are 1 (`num_ref_idx_l{0,1}_active_minus1 = 0`) so `ref_idx_lX`
//!   is implicit per §7.3.8.6.
//! * Per-CU pipeline picks among three modes:
//!     - **B_Skip** (round 23): `cu_skip_flag = 1` + `merge_idx`. No
//!       residual, no `pred_mode`, no `part_mode`, no explicit AMVP. The
//!       CU inherits the chosen merge candidate's motion in full
//!       (§8.5.3.2.10). Selected when the best merge candidate's SAD
//!       residual is below `SKIP_RESIDUAL_THRESHOLD`.
//!     - **Merge** (round 23): `cu_skip_flag = 0`, `pred_mode_flag = 0`,
//!       `part_mode = 2Nx2N`, `merge_flag = 1`, `merge_idx`. Residual
//!       coded normally. Selected when the best merge SAD beats the
//!       best AMVP SAD.
//!     - **Explicit AMVP** (round 22): `merge_flag = 0` with explicit
//!       `inter_pred_idc`, per-list `ref_idx`/`mvd`/`mvp_lx_flag = 0`.
//! * Merge candidate list: full §8.5.3.2.{2..5} pipeline via
//!   [`build_merge_list_full`] — spatial A0/A1/B0/B1/B2, temporal (we
//!   don't enable TMVP so it's skipped), combined bi-pred, then zero-MV
//!   pad. With `max_num_merge_cand = 5` we get up to five candidates.
//! * Slice header now emits `five_minus_max_num_merge_cand = 0` (vs r22's
//!   `4`) so the merge list grows from 1 → 5 entries. The unused entries
//!   stay zero-MV pads when the SAD search picks the explicit-AMVP path
//!   instead.
//! * `inter_pred_idc` (§9.3.4.2.2 Table 9-32): bin 0 ctxInc =
//!   `CtDepth[x0][y0]` for `(nPbW + nPbH) != 12`. Our CU is 16×16 so
//!   nPbW + nPbH = 32 and CtDepth = 0 → bin0_ctx = 0.
//! * Both AMVP candidates per list collapse to a single predictor
//!   (`mvp_lx_flag = 0`); we mirror the P-slice writer's spatial-only
//!   AMVP that walks A0/A1 then B0/B1/B2 on the local 4×4 mv grid.
//! * Integer-pel motion estimation with ±8 pel luma SAD search per list,
//!   step 2 so chroma MV stays integer (matches the decoder's full-pel
//!   chroma_mc path bit-exactly).
//! * `mvd_l1_zero_flag = 0` — both L0 and L1 MVDs are written explicitly
//!   so the decoder reproduces our two MVs without an `MvdL1` zero-out.
//! * SPS / PPS reuse the round-21 envelope: no TMVP, no SAO, no
//!   deblocking, no weighted bi-pred. The new `num_negative_pics = 1`,
//!   `num_positive_pics = 1` RPS is signalled inline per slice
//!   (`short_term_ref_pic_set_sps_flag = 0`).
//!
//! `cu_skip_flag` ctxInc: §9.3.4.2.2 Table 9-32 — `ctxInc =
//! condTermFlagL + condTermFlagA` where `condTermFlagX = neighbour.is_skip`
//! (decoder's r19 fix). The encoder maintains a per-4×4 `is_skip` grid
//! (populated alongside the MV grids) so the ctxInc derivation stays in
//! lock-step with the decoder.
//!
//! Out of scope for r23 (deferred to r24+): AMP / Nx2N / 2NxN partitions,
//! `mvd_l1_zero_flag` optimisation, B-pyramid (mini-GOP > 2), 10-bit.

use crate::cabac::{
    init_row, CtxState, InitType, ABS_MVD_GREATER_FLAGS_INIT_VALUES, CU_SKIP_FLAG_INIT_VALUES,
    INTER_PRED_IDC_INIT_VALUES, MERGE_FLAG_INIT_VALUES, MERGE_IDX_INIT_VALUES,
    MVP_LX_FLAG_INIT_VALUES, PART_MODE_INIT_VALUES, PRED_MODE_FLAG_INIT_VALUES,
    RQT_ROOT_CBF_INIT_VALUES, SPLIT_CU_FLAG_INIT_VALUES, SPLIT_TRANSFORM_FLAG_INIT_VALUES,
};
use crate::encoder::bit_writer::{write_rbsp_trailing_bits, BitWriter};
use crate::encoder::cabac_writer::CabacWriter;
use crate::encoder::nal_writer::build_annex_b_nal;
use crate::encoder::p_slice_writer::ReferenceFrame;
use crate::encoder::params::EncoderConfig;
use crate::encoder::residual_writer::{encode_residual, ResidualCtx};
use crate::inter::{
    build_merge_list_full, InterState, MergeCand, MergeCombinedCfg, MergeZeroPad, MotionVector,
    NeighbourContext, PbMotion, RefPocEntry,
};
use crate::nal::{NalHeader, NalUnitType};
use crate::transform::{
    dequantize_flat, forward_transform_2d, inverse_transform_2d, quantize_flat,
};
use oxideav_core::VideoFrame;

/// CTU size = 16. minCU = 8 (round 24 — lifted to unblock AMP). The
/// encoder still places one 16×16 CB per CTB, but the SPS now allows
/// further splits down to 8×8 (we never use them) and AMP partitions
/// inside the 16×16 CB (which we do use here).
const CTU_SIZE: u32 = 16;
/// Slice QP — matches the I/P-slice emitter so rate/quality stay aligned.
const SLICE_QP_Y: i32 = 26;
/// Integer search window (luma pixels). ±`ME_RANGE` both directions.
const ME_RANGE: i32 = 8;
/// Lagrangian λ for the AMP-vs-2Nx2N rate gate. Roughly `0.85 *
/// 2^((QP-12)/3)` per the HEVC reference's standard mapping; at QP 26
/// that's ≈ 21.5. We round up so the gate is slightly conservative
/// (prefers 2Nx2N when AMP doesn't beat it by a clear margin).
const AMP_LAMBDA: u64 = 22;
/// Estimated extra bits paid for an AMP partition (vs the 1-bit `part_mode
/// = 1` for 2Nx2N). Spec: `0` (NOT 2Nx2N) + dir bit + `0` (AMP) + bypass
/// position (=4 bins for part_mode) + the second PB's merge_flag +
/// inter_pred_idc + MVD pair + mvp_lx_flag. With small / near-zero MVDs
/// the typical extra is ~12 bins. `AMP_LAMBDA × 12 ≈ 264` SAD units is
/// the minimum margin AMP must beat 2Nx2N by.
const AMP_BIT_COST: u64 = 12;
/// `MaxNumMergeCand` advertised in the slice header. Round 23 lifts this
/// from 1 (round 22) to 5 — the spec's maximum. Larger lists are a
/// strict super-set: with `max == 1` only the first spatial candidate is
/// reachable via `merge_idx`, with `max == 5` the SAD search can pick
/// any of A0/A1/B0/B1/B2 + (eventually) combined bi-pred + zero-MV pads.
const MAX_NUM_MERGE_CAND: u32 = 5;
/// `five_minus_max_num_merge_cand` slice-header field. With the current
/// `MAX_NUM_MERGE_CAND = 5`, this is `5 - 5 = 0`.
const FIVE_MINUS_MAX_MERGE: u32 = 5 - MAX_NUM_MERGE_CAND;
/// Skip threshold: when the best merge candidate's residual SAD is at or
/// below this many luma pels of distortion, the encoder emits the CU as
/// `B_Skip` (no residual, no AMVP, no `merge_flag`). The threshold is
/// generous so static / near-static CUs collapse to a 2-bit skip CU
/// (`cu_skip_flag = 1`, `merge_idx = 0`) while preserving a clean
/// fall-through to the merge / AMVP residual path for moving content.
/// Value chosen as one quantisation step (≈ Qstep at QP 26) per pel
/// over a 16×16 CU = 4 × 256 ≈ 1024.
const SKIP_RESIDUAL_THRESHOLD: u64 = 1024;

/// Build Annex B bytes for a B (TrailR) slice NAL.
///
/// `delta_l0` and `delta_l1` are the per-list POC deltas relative to the
/// current frame: `delta_l0 < 0` (earlier ref), `delta_l1 > 0` (later ref).
pub fn build_b_slice_with_reconstruction(
    cfg: &EncoderConfig,
    frame: &VideoFrame,
    poc_lsb: u32,
    delta_l0: i32,
    delta_l1: i32,
    ref_l0: &ReferenceFrame,
    ref_l1: &ReferenceFrame,
) -> (Vec<u8>, ReferenceFrame) {
    debug_assert!(delta_l0 < 0, "B-slice L0 ref must have negative POC delta");
    debug_assert!(delta_l1 > 0, "B-slice L1 ref must have positive POC delta");
    let mut bw = BitWriter::new();

    // ---- slice_segment_header() -------------------------------------
    bw.write_u1(1); // first_slice_segment_in_pic_flag
    bw.write_ue(0); // slice_pic_parameter_set_id
    bw.write_ue(0); // slice_type = B (per §7.4.7.1: 0=B, 1=P, 2=I)

    // POC LSB — log2_max_poc_lsb_minus4 = 4 → 8 bits.
    bw.write_bits(poc_lsb & 0xFF, 8);
    bw.write_u1(0); // short_term_ref_pic_set_sps_flag = 0 → inline RPS.

    // Inline st_ref_pic_set(0) — single negative entry + single positive
    // entry. With st_rps_idx == 0 the inter_ref_pic_set_prediction_flag
    // is NOT signalled (§7.3.7).
    bw.write_ue(1); // num_negative_pics = 1
    bw.write_ue(1); // num_positive_pics = 1
                    // negative entry 0: delta_poc_s0_minus1 = -delta_l0 - 1, used = 1.
    bw.write_ue((-delta_l0 - 1) as u32);
    bw.write_u1(1);
    // positive entry 0: delta_poc_s1_minus1 = delta_l1 - 1, used = 1.
    bw.write_ue((delta_l1 - 1) as u32);
    bw.write_u1(1);

    // long_term_ref_pics_present = 0, sps_temporal_mvp = 0 → skipped.
    // SAO disabled → no slice_sao_*_flag.
    // P/B branch:
    bw.write_u1(0); // num_ref_idx_active_override_flag = 0 → use PPS defaults
                    // (both = 0 → num_ref_idx_lX_active_minus1 = 0 → exactly 1
                    // entry per list).
                    // pps.lists_modification_present_flag = 0 → skipped.
    bw.write_u1(0); // mvd_l1_zero_flag = 0
                    // pps.cabac_init_present_flag = 0 → skipped.
                    // slice_temporal_mvp_enabled_flag = 0 → no collocated.
                    // pps.weighted_bipred_flag = 0 → no weighted_pred table.
    bw.write_ue(FIVE_MINUS_MAX_MERGE); // max_num_merge_cand = MAX_NUM_MERGE_CAND
    bw.write_se(0); // slice_qp_delta = 0.
    write_rbsp_trailing_bits(&mut bw);

    // ---- slice_data() ------------------------------------------------
    let mut enc = BEncoderState::new(cfg, ref_l0.clone(), ref_l1.clone(), delta_l0, delta_l1);
    {
        let mut cabac = CabacWriter::new(&mut bw);
        let pic_w_ctb = cfg.width.div_ceil(CTU_SIZE);
        let pic_h_ctb = cfg.height.div_ceil(CTU_SIZE);
        let total_ctbs = pic_w_ctb * pic_h_ctb;
        for i in 0..total_ctbs {
            let ctb_x = (i % pic_w_ctb) * CTU_SIZE;
            let ctb_y = (i / pic_w_ctb) * CTU_SIZE;
            enc.encode_ctu(&mut cabac, frame, ctb_x, ctb_y);
            let is_last = i + 1 == total_ctbs;
            cabac.encode_terminate(if is_last { 1 } else { 0 });
        }
        cabac.encode_flush();
    }
    bw.align_to_byte_zero();
    let recon =
        ReferenceFrame::from_planes(cfg.width, cfg.height, enc.rec_y, enc.rec_cb, enc.rec_cr);
    let nal = build_annex_b_nal(NalHeader::for_type(NalUnitType::TrailR), &bw.finish());
    (nal, recon)
}

/// Per-slice B-frame CABAC encoding state.
struct BEncoderState {
    cfg: EncoderConfig,
    ref_l0: ReferenceFrame,
    ref_l1: ReferenceFrame,
    /// POC of the L0 reference relative to the current picture (< 0).
    /// Recorded so spatial neighbours pulled from `inter_state` carry
    /// matching `ref_poc_l0` for `build_merge_list_full`'s combined-bi
    /// gate.
    delta_l0: i32,
    /// POC of the L1 reference relative to the current picture (> 0).
    delta_l1: i32,
    rec_y: Vec<u8>,
    rec_cb: Vec<u8>,
    rec_cr: Vec<u8>,
    y_stride: usize,
    c_stride: usize,
    /// Per-PB motion grid — drives the merge candidate derivation
    /// (`build_merge_list_full`). Mirrors `decoder.pic.inter`.
    inter_state: InterState,
    /// Per-4×4 block: Some((mv_x, mv_y)) for each list when this PB used
    /// that list, None otherwise. Used by the AMVP spatial-neighbour
    /// fetch which scans the encoder's local grid (the decoder runs the
    /// same scan on its `pic.inter` PB grid; both produce the same MV
    /// when the MV records line up).
    mv_grid_l0: Vec<Option<(i32, i32)>>,
    mv_grid_l1: Vec<Option<(i32, i32)>>,
    grid_w4: usize,
    grid_h4: usize,

    // CABAC contexts (B slices use InitType::Pb when cabac_init_flag=0).
    split_cu_flag: [CtxState; 3],
    cu_skip_flag: [CtxState; 3],
    pred_mode_flag: [CtxState; 1],
    part_mode: [CtxState; 4],
    merge_flag: [CtxState; 1],
    merge_idx: [CtxState; 1],
    mvp_lx_flag: [CtxState; 1],
    abs_mvd_greater: [CtxState; 2],
    rqt_root_cbf: [CtxState; 1],
    inter_pred_idc: [CtxState; 5],
    /// Round 24 — needed for AMP CUs because the decoder reads the bin at
    /// the inter root TU when `part_mode != Mode2Nx2N` (see decoder's
    /// `transform_tree_inter_inner` § 7.3.8.9 path). For 2Nx2N we still
    /// take the `skip_for_2nx2n` shortcut and don't emit anything here.
    split_transform_flag: [CtxState; 3],
    residual: ResidualCtx,
}

/// AMP partition shapes (§7.4.9.5 Table 7-10) the round-24 encoder
/// considers as candidates against 2Nx2N. Each shape produces two PBs
/// over a 16×16 CB; the "quarter" stripe is 4 luma pels thick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AmpShape {
    /// `2NxnU` — top stripe 16×4, bottom stripe 16×12.
    NxnU,
    /// `2NxnD` — top stripe 16×12, bottom stripe 16×4.
    NxnD,
    /// `nLx2N` — left stripe 4×16, right stripe 12×16.
    NLx2N,
    /// `nRx2N` — left stripe 12×16, right stripe 4×16.
    NRx2N,
}

impl AmpShape {
    /// Return the two PB rectangles `(x, y, w, h)` for this shape rooted
    /// at `(x0, y0)` over a `cb_size × cb_size` CB.
    pub fn pbs(self, x0: u32, y0: u32, cb_size: u32) -> [(u32, u32, u32, u32); 2] {
        let q = cb_size / 4;
        match self {
            // §7.4.9.5: "U" / "L" → quarter stripe FIRST.
            AmpShape::NxnU => [(x0, y0, cb_size, q), (x0, y0 + q, cb_size, cb_size - q)],
            // "D" / "R" → quarter stripe LAST.
            AmpShape::NxnD => [
                (x0, y0, cb_size, cb_size - q),
                (x0, y0 + cb_size - q, cb_size, q),
            ],
            AmpShape::NLx2N => [(x0, y0, q, cb_size), (x0 + q, y0, cb_size - q, cb_size)],
            AmpShape::NRx2N => [
                (x0, y0, cb_size - q, cb_size),
                (x0 + cb_size - q, y0, q, cb_size),
            ],
        }
    }

    /// Horizontal family (split along Y) → bin1 = 1; vertical family
    /// (split along X) → bin1 = 0. Mirrors the decoder's part_mode
    /// derivation in `coding_quadtree`.
    pub fn dir_bin(self) -> u32 {
        match self {
            AmpShape::NxnU | AmpShape::NxnD => 1,
            AmpShape::NLx2N | AmpShape::NRx2N => 0,
        }
    }

    /// AMP position bypass bit per §9.3.4.2: `0` = quarter on the
    /// top/left, `1` = quarter on the bottom/right.
    pub fn pos_bypass(self) -> u32 {
        match self {
            AmpShape::NxnU | AmpShape::NLx2N => 0,
            AmpShape::NxnD | AmpShape::NRx2N => 1,
        }
    }
}

const AMP_CANDIDATES: [AmpShape; 4] = [
    AmpShape::NxnU,
    AmpShape::NxnD,
    AmpShape::NLx2N,
    AmpShape::NRx2N,
];

/// Per-PB motion pick used by the AMP path.
#[derive(Clone, Copy, Debug, Default)]
struct PbPick {
    use_l0: bool,
    use_l1: bool,
    mv_l0: (i32, i32),
    mv_l1: (i32, i32),
}

/// Per-CU mode pick.
#[derive(Clone, Copy, Debug)]
enum BMode {
    /// `cu_skip_flag = 1` + `merge_idx`. Best merge candidate, no residual.
    Skip { merge_idx: u32 },
    /// `merge_flag = 1` + `merge_idx`. Best merge candidate, residual coded.
    Merge { merge_idx: u32 },
    /// Explicit AMVP, 2Nx2N. `use_l0`/`use_l1` indicate which lists are
    /// active; `mv_l0`/`mv_l1` are the integer-pel selected MVs.
    Amvp {
        use_l0: bool,
        use_l1: bool,
        mv_l0: (i32, i32),
        mv_l1: (i32, i32),
    },
    /// Explicit AMVP, AMP partition. Per-PB list selection + integer-pel MV.
    AmvpAmp { shape: AmpShape, picks: [PbPick; 2] },
}

impl BEncoderState {
    fn new(
        cfg: &EncoderConfig,
        ref_l0: ReferenceFrame,
        ref_l1: ReferenceFrame,
        delta_l0: i32,
        delta_l1: i32,
    ) -> Self {
        let w = cfg.width as usize;
        let h = cfg.height as usize;
        let cw = w / 2;
        let ch = h / 2;
        // B slices with cabac_init_flag = 0 use Pb (per InitType::for_slice).
        let it = InitType::Pb;
        let grid_w4 = w / 4;
        let grid_h4 = h / 4;
        Self {
            cfg: *cfg,
            ref_l0,
            ref_l1,
            delta_l0,
            delta_l1,
            rec_y: vec![128u8; w * h],
            rec_cb: vec![128u8; cw * ch],
            rec_cr: vec![128u8; cw * ch],
            y_stride: w,
            c_stride: cw,
            inter_state: InterState::new(cfg.width, cfg.height),
            mv_grid_l0: vec![None; grid_w4 * grid_h4],
            mv_grid_l1: vec![None; grid_w4 * grid_h4],
            grid_w4,
            grid_h4,
            split_cu_flag: init_row(&SPLIT_CU_FLAG_INIT_VALUES, it, SLICE_QP_Y),
            cu_skip_flag: init_row(&CU_SKIP_FLAG_INIT_VALUES, it, SLICE_QP_Y),
            pred_mode_flag: init_row(&PRED_MODE_FLAG_INIT_VALUES, it, SLICE_QP_Y),
            part_mode: init_row(&PART_MODE_INIT_VALUES, it, SLICE_QP_Y),
            merge_flag: init_row(&MERGE_FLAG_INIT_VALUES, it, SLICE_QP_Y),
            merge_idx: init_row(&MERGE_IDX_INIT_VALUES, it, SLICE_QP_Y),
            mvp_lx_flag: init_row(&MVP_LX_FLAG_INIT_VALUES, it, SLICE_QP_Y),
            abs_mvd_greater: init_row(&ABS_MVD_GREATER_FLAGS_INIT_VALUES, it, SLICE_QP_Y),
            rqt_root_cbf: init_row(&RQT_ROOT_CBF_INIT_VALUES, it, SLICE_QP_Y),
            inter_pred_idc: init_row(&INTER_PRED_IDC_INIT_VALUES, it, SLICE_QP_Y),
            split_transform_flag: init_row(&SPLIT_TRANSFORM_FLAG_INIT_VALUES, it, SLICE_QP_Y),
            residual: ResidualCtx::with_init_type(SLICE_QP_Y, it),
        }
    }

    fn encode_ctu(&mut self, cw: &mut CabacWriter<'_>, src: &VideoFrame, x0: u32, y0: u32) {
        // Round 24: split_cu_flag = 0 at CTB root (log2_cb=4 > minCb=3,
        // depth-0 neighbours never carry cqtDepth > 0 → ctxInc = 0).
        cw.encode_bin(&mut self.split_cu_flag[0], 0);

        // ---- evaluate merge candidates (driven SAD pick) ---------------
        let merge_cands = self.build_merge_cands(x0, y0);
        let (merge_idx_best, merge_pb, merge_sad, merge_pred_y, merge_pred_cb, merge_pred_cr) =
            self.pick_merge_candidate(src, x0, y0, &merge_cands);

        // ---- evaluate explicit AMVP: per-list integer-pel SAD search ---
        let (mv_l0_int, sad_l0) = self.estimate_mv_int(src, x0, y0, /* l1 */ false);
        let (mv_l1_int, sad_l1) = self.estimate_mv_int(src, x0, y0, /* l1 */ true);
        let (luma_pred_l0, cb_pred_l0, cr_pred_l0) =
            motion_compensate(&self.ref_l0, x0, y0, mv_l0_int.0, mv_l0_int.1);
        let (luma_pred_l1, cb_pred_l1, cr_pred_l1) =
            motion_compensate(&self.ref_l1, x0, y0, mv_l1_int.0, mv_l1_int.1);
        let (luma_pred_bi, cb_pred_bi, cr_pred_bi) = make_bipred(
            &luma_pred_l0,
            &cb_pred_l0,
            &cr_pred_l0,
            &luma_pred_l1,
            &cb_pred_l1,
            &cr_pred_l1,
        );
        let sad_bi = sad_luma(src, x0, y0, &luma_pred_bi, CTU_SIZE);
        let (amvp_use_l0, amvp_use_l1, amvp_luma, amvp_cb, amvp_cr, amvp_sad) =
            if sad_bi <= sad_l0 && sad_bi <= sad_l1 {
                (true, true, luma_pred_bi, cb_pred_bi, cr_pred_bi, sad_bi)
            } else if sad_l0 <= sad_l1 {
                (true, false, luma_pred_l0, cb_pred_l0, cr_pred_l0, sad_l0)
            } else {
                (false, true, luma_pred_l1, cb_pred_l1, cr_pred_l1, sad_l1)
            };

        // ---- evaluate AMP candidates -----------------------------------
        // For each of the 4 AMP shapes, run a per-stripe integer ME on
        // the better of L0 / L1, build the composite prediction, and
        // score. Winner has to beat 2Nx2N by at least
        // `AMP_LAMBDA × AMP_BIT_COST` (~352 SAD units at QP 26) to be
        // picked — the rate gate for a partition that pays the second
        // PB's ~16-bit overhead.
        let mut best_amp: Option<(AmpShape, [PbPick; 2], u64)> = None;
        if std::env::var_os("H265_ENC_AMP_OFF").is_none() {
            for shape in AMP_CANDIDATES {
                let (picks, sad) = self.evaluate_amp_shape(src, x0, y0, shape);
                let is_better = best_amp.as_ref().map(|(_, _, s)| sad < *s).unwrap_or(true);
                if is_better {
                    best_amp = Some((shape, picks, sad));
                }
            }
        }
        let amp_winner = best_amp
            .as_ref()
            .filter(|(_, _, sad)| *sad + AMP_LAMBDA * AMP_BIT_COST < amvp_sad);

        // ---- pick mode -------------------------------------------------
        // Skip when the best merge SAD is below the threshold AND merge
        // beats AMVP (otherwise we'd lose quality on a moving CU just to
        // save 2 bits of header).
        let merge_wins = merge_sad < amvp_sad;
        let mode = if merge_wins && merge_sad <= SKIP_RESIDUAL_THRESHOLD {
            BMode::Skip {
                merge_idx: merge_idx_best,
            }
        } else if merge_wins {
            BMode::Merge {
                merge_idx: merge_idx_best,
            }
        } else if let Some((shape, picks, _)) = amp_winner {
            BMode::AmvpAmp {
                shape: *shape,
                picks: *picks,
            }
        } else {
            BMode::Amvp {
                use_l0: amvp_use_l0,
                use_l1: amvp_use_l1,
                mv_l0: mv_l0_int,
                mv_l1: mv_l1_int,
            }
        };

        // ---- emit ------------------------------------------------------
        let skip_ctx_inc = self.skip_ctx_inc(x0, y0);
        match mode {
            BMode::Skip { merge_idx } => {
                cw.encode_bin(&mut self.cu_skip_flag[skip_ctx_inc], 1);
                self.emit_merge_idx(cw, merge_idx);
                // Merge MC + state update — no residual.
                let pb = self.materialise_merge_pb(merge_pb, /* is_skip */ true);
                self.apply_prediction_no_residual(
                    x0,
                    y0,
                    &merge_pred_y,
                    &merge_pred_cb,
                    &merge_pred_cr,
                );
                self.publish_pb(x0, y0, CTU_SIZE, pb);
            }
            BMode::Merge { merge_idx } => {
                cw.encode_bin(&mut self.cu_skip_flag[skip_ctx_inc], 0);
                cw.encode_bin(&mut self.pred_mode_flag[0], 0); // INTER
                cw.encode_bin(&mut self.part_mode[0], 1); // 2Nx2N
                cw.encode_bin(&mut self.merge_flag[0], 1);
                self.emit_merge_idx(cw, merge_idx);
                let pb = self.materialise_merge_pb(merge_pb, /* is_skip */ false);
                self.code_residual(
                    cw,
                    src,
                    x0,
                    y0,
                    &merge_pred_y,
                    &merge_pred_cb,
                    &merge_pred_cr,
                );
                self.publish_pb(x0, y0, CTU_SIZE, pb);
            }
            BMode::Amvp {
                use_l0,
                use_l1,
                mv_l0,
                mv_l1,
            } => {
                cw.encode_bin(&mut self.cu_skip_flag[skip_ctx_inc], 0);
                cw.encode_bin(&mut self.pred_mode_flag[0], 0); // INTER
                cw.encode_bin(&mut self.part_mode[0], 1); // 2Nx2N
                cw.encode_bin(&mut self.merge_flag[0], 0);

                // inter_pred_idc bin0 ctxInc = CtDepth = 0.
                let bin0_ctx = 0usize;
                if use_l0 && use_l1 {
                    cw.encode_bin(&mut self.inter_pred_idc[bin0_ctx], 1);
                } else {
                    cw.encode_bin(&mut self.inter_pred_idc[bin0_ctx], 0);
                    cw.encode_bin(&mut self.inter_pred_idc[4], if use_l1 { 1 } else { 0 });
                }
                if use_l0 {
                    let mv_qp = (mv_l0.0 * 4, mv_l0.1 * 4);
                    let mvp = self.amvp_predictor(x0, y0, CTU_SIZE, CTU_SIZE, /* l1 */ false);
                    self.encode_mvd(cw, mv_qp.0 - mvp.0, mv_qp.1 - mvp.1);
                    cw.encode_bin(&mut self.mvp_lx_flag[0], 0);
                }
                if use_l1 {
                    let mv_qp = (mv_l1.0 * 4, mv_l1.1 * 4);
                    let mvp = self.amvp_predictor(x0, y0, CTU_SIZE, CTU_SIZE, /* l1 */ true);
                    self.encode_mvd(cw, mv_qp.0 - mvp.0, mv_qp.1 - mvp.1);
                    cw.encode_bin(&mut self.mvp_lx_flag[0], 0);
                }

                self.code_residual(cw, src, x0, y0, &amvp_luma, &amvp_cb, &amvp_cr);

                // Build the PB record published to the inter grid.
                let mv_l0_qp = MotionVector::new(mv_l0.0 * 4, mv_l0.1 * 4);
                let mv_l1_qp = MotionVector::new(mv_l1.0 * 4, mv_l1.1 * 4);
                let pb = PbMotion {
                    valid: true,
                    is_intra: false,
                    is_skip: false,
                    pred_l0: use_l0,
                    pred_l1: use_l1,
                    ref_idx_l0: if use_l0 { 0 } else { -1 },
                    ref_idx_l1: if use_l1 { 0 } else { -1 },
                    mv_l0: if use_l0 {
                        mv_l0_qp
                    } else {
                        MotionVector::default()
                    },
                    mv_l1: if use_l1 {
                        mv_l1_qp
                    } else {
                        MotionVector::default()
                    },
                    ref_poc_l0: if use_l0 { self.delta_l0 } else { 0 },
                    ref_poc_l1: if use_l1 { self.delta_l1 } else { 0 },
                    ref_lt_l0: false,
                    ref_lt_l1: false,
                };
                self.publish_pb(x0, y0, CTU_SIZE, pb);
            }
            BMode::AmvpAmp { shape, picks } => {
                self.emit_amvp_amp(cw, src, x0, y0, skip_ctx_inc, shape, picks);
            }
        }
    }

    /// Score one AMP shape: per-stripe integer ME on the better of
    /// L0 / L1, build the full composite luma / chroma prediction, return
    /// `(picks, sad)`.
    fn evaluate_amp_shape(
        &self,
        src: &VideoFrame,
        x0: u32,
        y0: u32,
        shape: AmpShape,
    ) -> ([PbPick; 2], u64) {
        let pbs = shape.pbs(x0, y0, CTU_SIZE);
        let mut picks = [PbPick::default(); 2];
        for (idx, (pb_x, pb_y, pb_w, pb_h)) in pbs.iter().enumerate() {
            // Per-stripe SAD: pick L0-only or L1-only, smaller wins.
            let (mv0, sad0) = self.estimate_mv_int_rect(src, *pb_x, *pb_y, *pb_w, *pb_h, false);
            let (mv1, sad1) = self.estimate_mv_int_rect(src, *pb_x, *pb_y, *pb_w, *pb_h, true);
            picks[idx] = if sad0 <= sad1 {
                PbPick {
                    use_l0: true,
                    use_l1: false,
                    mv_l0: mv0,
                    mv_l1: (0, 0),
                }
            } else {
                PbPick {
                    use_l0: false,
                    use_l1: true,
                    mv_l0: (0, 0),
                    mv_l1: mv1,
                }
            };
        }
        let (luma, _, _) = self.amp_composite(x0, y0, shape, &picks);
        let sad = sad_luma(src, x0, y0, &luma, CTU_SIZE);
        (picks, sad)
    }

    /// Build the AMP composite prediction (per-stripe MC, glued together)
    /// for one CB. Returns `(luma_16x16, cb_8x8, cr_8x8)`.
    fn amp_composite(
        &self,
        x0: u32,
        y0: u32,
        shape: AmpShape,
        picks: &[PbPick; 2],
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let n = CTU_SIZE as usize;
        let cn = n / 2;
        let mut luma = vec![0u8; n * n];
        let mut cb = vec![0u8; cn * cn];
        let mut cr = vec![0u8; cn * cn];
        let pbs = shape.pbs(x0, y0, CTU_SIZE);
        for (idx, (pb_x, pb_y, pb_w, pb_h)) in pbs.iter().enumerate() {
            let pick = &picks[idx];
            let (mv_x, mv_y) = if pick.use_l0 { pick.mv_l0 } else { pick.mv_l1 };
            let r = if pick.use_l0 {
                &self.ref_l0
            } else {
                &self.ref_l1
            };
            // Luma stripe.
            for j in 0..(*pb_h as i32) {
                for i in 0..(*pb_w as i32) {
                    let abs_x = *pb_x as i32 + i;
                    let abs_y = *pb_y as i32 + j;
                    let lx = (abs_x - x0 as i32) as usize;
                    let ly = (abs_y - y0 as i32) as usize;
                    luma[ly * n + lx] = r.sample_y(abs_x + mv_x, abs_y + mv_y);
                }
            }
            // Chroma stripe — 4:2:0 means the chroma rect is half the
            // luma rect; AMP stripes (4 luma) divide cleanly by 2.
            let cx0 = (*pb_x / 2) as i32;
            let cy0 = (*pb_y / 2) as i32;
            let cw_pb = (*pb_w / 2) as i32;
            let ch_pb = (*pb_h / 2) as i32;
            let mv_cx = mv_x / 2;
            let mv_cy = mv_y / 2;
            let cb0_cx = (x0 / 2) as i32;
            let cb0_cy = (y0 / 2) as i32;
            for j in 0..ch_pb {
                for i in 0..cw_pb {
                    let abs_cx = cx0 + i;
                    let abs_cy = cy0 + j;
                    let lx = (abs_cx - cb0_cx) as usize;
                    let ly = (abs_cy - cb0_cy) as usize;
                    cb[ly * cn + lx] = r.sample_c(abs_cx + mv_cx, abs_cy + mv_cy, false);
                    cr[ly * cn + lx] = r.sample_c(abs_cx + mv_cx, abs_cy + mv_cy, true);
                }
            }
        }
        (luma, cb, cr)
    }

    /// Emit the full CABAC payload for an AMP-partitioned CU. Order
    /// follows §7.3.8.5: `cu_skip_flag = 0`, `pred_mode_flag = 0`,
    /// `part_mode` (4 bins for AMP per Table 9-45), then per-PB
    /// `prediction_unit()`, then `rqt_root_cbf`,
    /// `split_transform_flag = 0`, then the residual.
    fn emit_amvp_amp(
        &mut self,
        cw: &mut CabacWriter<'_>,
        src: &VideoFrame,
        x0: u32,
        y0: u32,
        skip_ctx_inc: usize,
        shape: AmpShape,
        picks: [PbPick; 2],
    ) {
        cw.encode_bin(&mut self.cu_skip_flag[skip_ctx_inc], 0);
        cw.encode_bin(&mut self.pred_mode_flag[0], 0); // INTER

        // part_mode bins per Table 9-45 (AMP path, log2CbSize > MinCbLog2SizeY):
        //   bin0 (ctx 0) = 0   — NOT 2Nx2N
        //   bin1 (ctx 1) = dir — 1 horizontal family (2NxnU/D), 0 vertical
        //   bin2 (ctx 3) = 0   — AMP (vs symmetric 2NxN/Nx2N which is 1)
        //   bin3 bypass  = pos — 0 = U/L (quarter first), 1 = D/R (quarter last)
        cw.encode_bin(&mut self.part_mode[0], 0);
        cw.encode_bin(&mut self.part_mode[1], shape.dir_bin());
        cw.encode_bin(&mut self.part_mode[3], 0);
        cw.encode_bypass(shape.pos_bypass());

        // Build the composite prediction so we can both publish the MC
        // result and run residual coding against it.
        let pbs = shape.pbs(x0, y0, CTU_SIZE);
        let (composite_y, composite_cb, composite_cr) = self.amp_composite(x0, y0, shape, &picks);

        // Per-PB prediction_unit syntax in raster order.
        for (idx, (pb_x, pb_y, pb_w, pb_h)) in pbs.iter().enumerate() {
            let pick = &picks[idx];
            cw.encode_bin(&mut self.merge_flag[0], 0);

            // inter_pred_idc bin 0: ctxInc = CtDepth = 0 (depth-0 16×16 CB).
            // AMP "quarter" stripes are 16×4 / 4×16 (sum 20) and the
            // "3·n" stripes sum to 28 — neither hits 12 so the small-PU
            // ctxInc = 4 path is not taken.
            let bin0_ctx = 0usize;
            if pick.use_l0 && pick.use_l1 {
                cw.encode_bin(&mut self.inter_pred_idc[bin0_ctx], 1);
            } else {
                cw.encode_bin(&mut self.inter_pred_idc[bin0_ctx], 0);
                cw.encode_bin(&mut self.inter_pred_idc[4], if pick.use_l1 { 1 } else { 0 });
            }
            if pick.use_l0 {
                let mv_qp = (pick.mv_l0.0 * 4, pick.mv_l0.1 * 4);
                let mvp = self.amvp_predictor(*pb_x, *pb_y, *pb_w, *pb_h, /* l1 */ false);
                self.encode_mvd(cw, mv_qp.0 - mvp.0, mv_qp.1 - mvp.1);
                cw.encode_bin(&mut self.mvp_lx_flag[0], 0);
            }
            if pick.use_l1 {
                let mv_qp = (pick.mv_l1.0 * 4, pick.mv_l1.1 * 4);
                let mvp = self.amvp_predictor(*pb_x, *pb_y, *pb_w, *pb_h, /* l1 */ true);
                self.encode_mvd(cw, mv_qp.0 - mvp.0, mv_qp.1 - mvp.1);
                cw.encode_bin(&mut self.mvp_lx_flag[0], 0);
            }
            // Publish PB-0 BEFORE PB-1 so PB-1's AMVP sees the right
            // neighbour. (PB-1 published below.)
            if idx == 0 {
                self.publish_pb_pick(*pb_x, *pb_y, *pb_w, *pb_h, pick);
            }
        }
        let (pb1_x, pb1_y, pb1_w, pb1_h) = pbs[1];
        self.publish_pb_pick(pb1_x, pb1_y, pb1_w, pb1_h, &picks[1]);

        // Code the composite residual. Differs from the 2Nx2N path in
        // one bin: the decoder reads `split_transform_flag` at the root
        // because the CU is not 2Nx2N (the `skip_for_2nx2n` shortcut in
        // `transform_tree_inter_inner` is gated on
        // `part_mode == Mode2Nx2N`). Emit `0` so the root stays a single
        // 16×16 luma TB.
        self.code_residual_amp(cw, src, x0, y0, &composite_y, &composite_cb, &composite_cr);
    }

    /// Code the composite residual for an AMP CU. Differs from
    /// [`Self::code_residual`] in one bin: `split_transform_flag = 0`
    /// at the root for a non-2Nx2N inter CU.
    fn code_residual_amp(
        &mut self,
        cw: &mut CabacWriter<'_>,
        src: &VideoFrame,
        x0: u32,
        y0: u32,
        luma_pred: &[u8],
        cb_pred: &[u8],
        cr_pred: &[u8],
    ) {
        let log2_tb = 4u32;
        let c_log2 = 3u32;
        let (luma_levels, luma_rec) = self.process_inter_luma(src, x0, y0, luma_pred);
        let cx = x0 / 2;
        let cy = y0 / 2;
        let (cb_levels, cb_rec) = self.process_inter_chroma(src, cx, cy, cb_pred, false);
        let (cr_levels, cr_rec) = self.process_inter_chroma(src, cx, cy, cr_pred, true);

        let cbf_luma = luma_levels.iter().any(|&l| l != 0);
        let cbf_cb = cb_levels.iter().any(|&l| l != 0);
        let cbf_cr = cr_levels.iter().any(|&l| l != 0);
        let any_residual = cbf_luma || cbf_cb || cbf_cr;

        cw.encode_bin(&mut self.rqt_root_cbf[0], any_residual as u32);
        if !any_residual {
            self.write_luma_block(x0, y0, CTU_SIZE as usize, luma_pred);
            self.write_chroma_block(cx, cy, 1 << c_log2, cb_pred, false);
            self.write_chroma_block(cx, cy, 1 << c_log2, cr_pred, true);
            return;
        }

        // §7.3.8.9 transform_tree at the root (tr_depth = 0) for an AMP
        // CU. The decoder reads cbf_cb / cbf_cr BEFORE the split bin
        // (chroma_here = log2_tb > 2 = true; tr_depth == 0 ⇒
        // parent_cbf treated as 1; ctxInc = 0).
        cw.encode_bin(&mut self.residual.cbf_cb_cr[0], cbf_cb as u32);
        cw.encode_bin(&mut self.residual.cbf_cb_cr[0], cbf_cr as u32);
        // split_transform_flag at root: ctxInc = min((5 - log2_tb), 2)
        // = min(1, 2) = 1. Emit 0 → single 16×16 root luma TB.
        cw.encode_bin(&mut self.split_transform_flag[1], 0);
        // cbf_luma at the leaf (tr_depth still 0). ctxInc = 1 (tr_depth
        // == 0). cbf_luma is inferred = 1 when `cbf_cb == 0 && cbf_cr
        // == 0`, otherwise emitted.
        let cbf_luma_inferred = !cbf_cb && !cbf_cr;
        if !cbf_luma_inferred {
            cw.encode_bin(&mut self.residual.cbf_luma[1], cbf_luma as u32);
        }
        if cbf_luma {
            encode_residual(cw, &mut self.residual, &luma_levels, log2_tb, true);
        }
        if cbf_cb {
            encode_residual(cw, &mut self.residual, &cb_levels, c_log2, false);
        }
        if cbf_cr {
            encode_residual(cw, &mut self.residual, &cr_levels, c_log2, false);
        }

        self.write_luma_block(x0, y0, CTU_SIZE as usize, &luma_rec);
        self.write_chroma_block(cx, cy, 1 << c_log2, &cb_rec, false);
        self.write_chroma_block(cx, cy, 1 << c_log2, &cr_rec, true);
    }

    /// Build & publish a [`PbMotion`] for one AMP PB.
    fn publish_pb_pick(&mut self, x: u32, y: u32, w: u32, h: u32, pick: &PbPick) {
        let mv_l0_qp = MotionVector::new(pick.mv_l0.0 * 4, pick.mv_l0.1 * 4);
        let mv_l1_qp = MotionVector::new(pick.mv_l1.0 * 4, pick.mv_l1.1 * 4);
        let pb = PbMotion {
            valid: true,
            is_intra: false,
            is_skip: false,
            pred_l0: pick.use_l0,
            pred_l1: pick.use_l1,
            ref_idx_l0: if pick.use_l0 { 0 } else { -1 },
            ref_idx_l1: if pick.use_l1 { 0 } else { -1 },
            mv_l0: if pick.use_l0 {
                mv_l0_qp
            } else {
                MotionVector::default()
            },
            mv_l1: if pick.use_l1 {
                mv_l1_qp
            } else {
                MotionVector::default()
            },
            ref_poc_l0: if pick.use_l0 { self.delta_l0 } else { 0 },
            ref_poc_l1: if pick.use_l1 { self.delta_l1 } else { 0 },
            ref_lt_l0: false,
            ref_lt_l1: false,
        };
        self.inter_state.set_rect(x, y, w, h, pb);
        if pb.pred_l0 {
            self.store_mv_rect(x, y, w, h, (pb.mv_l0.x, pb.mv_l0.y), false);
        }
        if pb.pred_l1 {
            self.store_mv_rect(x, y, w, h, (pb.mv_l1.x, pb.mv_l1.y), true);
        }
    }

    /// Per-rectangle integer-pel ME (equivalent to `estimate_mv_int` but
    /// over a `(w, h)` PB rather than the full CTU). Steps of 2 keep
    /// chroma MVs at integer pel.
    fn estimate_mv_int_rect(
        &self,
        src: &VideoFrame,
        x0: u32,
        y0: u32,
        w: u32,
        h: u32,
        is_l1: bool,
    ) -> ((i32, i32), u64) {
        let nw = w as i32;
        let nh = h as i32;
        let src_y = &src.planes[0];
        let pic_w = self.cfg.width as i32;
        let pic_h = self.cfg.height as i32;
        let r = if is_l1 { &self.ref_l1 } else { &self.ref_l0 };
        let mut best_sad = u64::MAX;
        let mut best = (0i32, 0i32);
        let mut dy = -ME_RANGE;
        while dy <= ME_RANGE {
            let mut dx = -ME_RANGE;
            while dx <= ME_RANGE {
                let rx = x0 as i32 + dx;
                let ry = y0 as i32 + dy;
                if rx < 0 || ry < 0 || rx + nw > pic_w || ry + nh > pic_h {
                    dx += 2;
                    continue;
                }
                let mut sad = 0u64;
                for j in 0..nh {
                    for i in 0..nw {
                        let s = src_y.data
                            [(y0 as i32 + j) as usize * src_y.stride + (x0 as i32 + i) as usize]
                            as i32;
                        let rr = r.sample_y(rx + i, ry + j) as i32;
                        sad += (s - rr).unsigned_abs() as u64;
                    }
                }
                if sad < best_sad {
                    best_sad = sad;
                    best = (dx, dy);
                }
                dx += 2;
            }
            dy += 2;
        }
        (best, best_sad)
    }

    /// Rectangle variant of [`Self::store_mv`] for AMP PB shapes.
    fn store_mv_rect(&mut self, x: u32, y: u32, w: u32, h: u32, mv: (i32, i32), is_l1: bool) {
        let grid = if is_l1 {
            &mut self.mv_grid_l1
        } else {
            &mut self.mv_grid_l0
        };
        let bx0 = (x >> 2) as usize;
        let by0 = (y >> 2) as usize;
        let nw4 = (w >> 2) as usize;
        let nh4 = (h >> 2) as usize;
        for dy in 0..nh4 {
            for dx in 0..nw4 {
                let bx = bx0 + dx;
                let by = by0 + dy;
                if bx < self.grid_w4 && by < self.grid_h4 {
                    grid[by * self.grid_w4 + bx] = Some(mv);
                }
            }
        }
    }

    /// Build the merge candidate list for the CU at `(x0, y0)`. Mirrors
    /// the decoder's call into `build_merge_list_full` exactly so the
    /// candidate at index `merge_idx` here matches the candidate the
    /// decoder will materialise.
    fn build_merge_cands(&self, x0: u32, y0: u32) -> Vec<MergeCand> {
        let nb = NeighbourContext::default();
        let rpl0 = vec![RefPocEntry {
            poc: self.delta_l0,
            is_long_term: false,
        }];
        let rpl1 = vec![RefPocEntry {
            poc: self.delta_l1,
            is_long_term: false,
        }];
        let comb_cfg = MergeCombinedCfg {
            rpl0: &rpl0,
            rpl1: &rpl1,
        };
        let zero_cfg = MergeZeroPad {
            num_ref_idx: 1, // both lists have exactly 1 entry
            rpl0: &rpl0,
            rpl1: &rpl1,
        };
        // No TMVP — `slice_temporal_mvp_enabled_flag = 0`.
        build_merge_list_full(
            &self.inter_state,
            x0,
            y0,
            CTU_SIZE,
            CTU_SIZE,
            MAX_NUM_MERGE_CAND,
            true, // is_b_slice
            None, // tmvp
            nb,
            comb_cfg,
            zero_cfg,
        )
    }

    /// Score every merge candidate by SAD against the source and return
    /// `(idx, candidate, sad, luma_pred, cb_pred, cr_pred)` for the best.
    fn pick_merge_candidate(
        &self,
        src: &VideoFrame,
        x0: u32,
        y0: u32,
        cands: &[MergeCand],
    ) -> (u32, MergeCand, u64, Vec<u8>, Vec<u8>, Vec<u8>) {
        debug_assert!(!cands.is_empty(), "merge list must have ≥ 1 entry");
        let mut best_idx = 0u32;
        let mut best_sad = u64::MAX;
        let mut best_y = Vec::new();
        let mut best_cb = Vec::new();
        let mut best_cr = Vec::new();
        let mut best_cand = cands[0];
        for (i, cand) in cands.iter().enumerate() {
            // Convert candidate quarter-pel MV → integer luma pel for our
            // integer-only MC path. Anything off-grid is approximated;
            // the decoder reads the same MV bits so its MC will match.
            let (luma, cb, cr) = self.merge_motion_compensate(x0, y0, cand);
            let sad = sad_luma(src, x0, y0, &luma, CTU_SIZE);
            if sad < best_sad {
                best_sad = sad;
                best_idx = i as u32;
                best_y = luma;
                best_cb = cb;
                best_cr = cr;
                best_cand = *cand;
            }
        }
        (best_idx, best_cand, best_sad, best_y, best_cb, best_cr)
    }

    /// Motion-compensate a merge candidate. Bipred candidates average the
    /// two list predictors with `(P0 + P1 + 1) >> 1` (default-bipred
    /// weighting per §8.5.3.3.3.1).
    fn merge_motion_compensate(
        &self,
        x0: u32,
        y0: u32,
        cand: &MergeCand,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        // Quarter-pel MV → integer luma pel via floor-div by 4. The
        // decoder runs a full sub-pel interpolation but in this writer's
        // configuration every MV that the encoder ever stores is a
        // multiple of 4 (ME runs at integer pel, store_mv multiplies by
        // 4), so the chosen merge candidate's MV is also a multiple of
        // 4. The arithmetic-shift below is the inverse of that store.
        let mv0 = (cand.mv_l0.x >> 2, cand.mv_l0.y >> 2);
        let mv1 = (cand.mv_l1.x >> 2, cand.mv_l1.y >> 2);
        match (cand.pred_l0, cand.pred_l1) {
            (true, false) => motion_compensate(&self.ref_l0, x0, y0, mv0.0, mv0.1),
            (false, true) => motion_compensate(&self.ref_l1, x0, y0, mv1.0, mv1.1),
            (true, true) => {
                let (y0p, cb0p, cr0p) = motion_compensate(&self.ref_l0, x0, y0, mv0.0, mv0.1);
                let (y1p, cb1p, cr1p) = motion_compensate(&self.ref_l1, x0, y0, mv1.0, mv1.1);
                make_bipred(&y0p, &cb0p, &cr0p, &y1p, &cb1p, &cr1p)
            }
            (false, false) => {
                // Should never happen — the merge derivation always
                // produces at least one prediction list. Fall back to L0
                // zero-MV to keep the encoder safe.
                motion_compensate(&self.ref_l0, x0, y0, 0, 0)
            }
        }
    }

    /// Convert a [`MergeCand`] into a [`PbMotion`] suitable for the inter
    /// grid. Patches `ref_poc_*` from the slice's RPL so the grid carries
    /// spec-correct shadow metadata for future neighbour lookups.
    fn materialise_merge_pb(&self, cand: MergeCand, is_skip: bool) -> PbMotion {
        let mut pb = cand.to_pb();
        pb.is_skip = is_skip;
        if pb.pred_l0 {
            pb.ref_poc_l0 = self.delta_l0;
            pb.ref_lt_l0 = false;
        }
        if pb.pred_l1 {
            pb.ref_poc_l1 = self.delta_l1;
            pb.ref_lt_l1 = false;
        }
        pb
    }

    /// Write the merge candidate's prediction into the local
    /// reconstruction buffers without coding any residual (skip path).
    fn apply_prediction_no_residual(
        &mut self,
        x0: u32,
        y0: u32,
        luma_pred: &[u8],
        cb_pred: &[u8],
        cr_pred: &[u8],
    ) {
        let cx = x0 / 2;
        let cy = y0 / 2;
        let n = CTU_SIZE as usize;
        let cn = n / 2;
        self.write_luma_block(x0, y0, n, luma_pred);
        self.write_chroma_block(cx, cy, cn, cb_pred, false);
        self.write_chroma_block(cx, cy, cn, cr_pred, true);
    }

    /// Code the residual for a CU given its prediction. Updates the local
    /// reconstruction buffers with `pred + dequant(quant(residual))`.
    fn code_residual(
        &mut self,
        cw: &mut CabacWriter<'_>,
        src: &VideoFrame,
        x0: u32,
        y0: u32,
        luma_pred: &[u8],
        cb_pred: &[u8],
        cr_pred: &[u8],
    ) {
        let log2_tb = 4u32;
        let c_log2 = 3u32;
        let (luma_levels, luma_rec) = self.process_inter_luma(src, x0, y0, luma_pred);
        let cx = x0 / 2;
        let cy = y0 / 2;
        let (cb_levels, cb_rec) = self.process_inter_chroma(src, cx, cy, cb_pred, false);
        let (cr_levels, cr_rec) = self.process_inter_chroma(src, cx, cy, cr_pred, true);

        let cbf_luma = luma_levels.iter().any(|&l| l != 0);
        let cbf_cb = cb_levels.iter().any(|&l| l != 0);
        let cbf_cr = cr_levels.iter().any(|&l| l != 0);
        let any_residual = cbf_luma || cbf_cb || cbf_cr;

        cw.encode_bin(&mut self.rqt_root_cbf[0], any_residual as u32);

        if any_residual {
            cw.encode_bin(&mut self.residual.cbf_cb_cr[0], cbf_cb as u32);
            cw.encode_bin(&mut self.residual.cbf_cb_cr[0], cbf_cr as u32);
            let cbf_luma_inferred = !cbf_cb && !cbf_cr; // inferred = 1
            if !cbf_luma_inferred {
                cw.encode_bin(&mut self.residual.cbf_luma[1], cbf_luma as u32);
            }
            if cbf_luma {
                encode_residual(cw, &mut self.residual, &luma_levels, log2_tb, true);
            }
            if cbf_cb {
                encode_residual(cw, &mut self.residual, &cb_levels, c_log2, false);
            }
            if cbf_cr {
                encode_residual(cw, &mut self.residual, &cr_levels, c_log2, false);
            }
        }

        self.write_luma_block(x0, y0, CTU_SIZE as usize, &luma_rec);
        self.write_chroma_block(cx, cy, 1 << c_log2, &cb_rec, false);
        self.write_chroma_block(cx, cy, 1 << c_log2, &cr_rec, true);
    }

    /// Publish a CU's PB onto the per-4×4 inter / mv grids so subsequent
    /// CUs see the right neighbour state for AMVP / merge / skip-ctx
    /// derivation.
    fn publish_pb(&mut self, x0: u32, y0: u32, size: u32, pb: PbMotion) {
        self.inter_state.set_rect(x0, y0, size, size, pb);
        if pb.pred_l0 {
            self.store_mv(x0, y0, size, (pb.mv_l0.x, pb.mv_l0.y), false);
        }
        if pb.pred_l1 {
            self.store_mv(x0, y0, size, (pb.mv_l1.x, pb.mv_l1.y), true);
        }
    }

    /// Emit `merge_idx` per §9.3.4.2.10 — bin 0 context-coded, remainder
    /// bypass-unary up to `MAX_NUM_MERGE_CAND - 1`.
    fn emit_merge_idx(&mut self, cw: &mut CabacWriter<'_>, idx: u32) {
        if MAX_NUM_MERGE_CAND <= 1 {
            return;
        }
        if idx == 0 {
            cw.encode_bin(&mut self.merge_idx[0], 0);
            return;
        }
        cw.encode_bin(&mut self.merge_idx[0], 1);
        let max = MAX_NUM_MERGE_CAND - 1; // cMax in the TR binarisation
        let mut v = 1u32;
        while v < max {
            if v == idx {
                cw.encode_bypass(0);
                return;
            }
            cw.encode_bypass(1);
            v += 1;
        }
        // idx == max: ran out of bins, no terminating 0 needed.
    }

    /// §9.3.4.2.2 cu_skip_flag ctxInc = `condTermFlagL + condTermFlagA`
    /// where `condTermFlagX = neighbour.is_skip`. Mirrors the decoder's
    /// r19 fix exactly so encoder and decoder stay in CABAC sync. This
    /// is the lookup that tripped Main 10 inter to 25 dB before r19.
    fn skip_ctx_inc(&self, x0: u32, y0: u32) -> usize {
        let left = if x0 == 0 {
            0
        } else {
            let bx = ((x0 - 1) >> 2) as usize;
            let by = (y0 >> 2) as usize;
            self.inter_state
                .get(bx, by)
                .map(|p| p.valid && p.is_skip)
                .unwrap_or(false) as usize
        };
        let above = if y0 == 0 {
            0
        } else {
            let bx = (x0 >> 2) as usize;
            let by = ((y0 - 1) >> 2) as usize;
            self.inter_state
                .get(bx, by)
                .map(|p| p.valid && p.is_skip)
                .unwrap_or(false) as usize
        };
        (left + above).min(2)
    }

    /// Integer-pel motion estimation against one of the references.
    fn estimate_mv_int(
        &self,
        src: &VideoFrame,
        x0: u32,
        y0: u32,
        is_l1: bool,
    ) -> ((i32, i32), u64) {
        let n = CTU_SIZE as i32;
        let src_y = &src.planes[0];
        let pic_w = self.cfg.width as i32;
        let pic_h = self.cfg.height as i32;
        let r = if is_l1 { &self.ref_l1 } else { &self.ref_l0 };
        let mut best_sad = u64::MAX;
        let mut best = (0i32, 0i32);
        let mut dy = -ME_RANGE;
        while dy <= ME_RANGE {
            let mut dx = -ME_RANGE;
            while dx <= ME_RANGE {
                let rx = x0 as i32 + dx;
                let ry = y0 as i32 + dy;
                if rx < 0 || ry < 0 || rx + n > pic_w || ry + n > pic_h {
                    dx += 2;
                    continue;
                }
                let mut sad = 0u64;
                for j in 0..n {
                    for i in 0..n {
                        let s = src_y.data
                            [(y0 as i32 + j) as usize * src_y.stride + (x0 as i32 + i) as usize]
                            as i32;
                        let rr = r.sample_y(rx + i, ry + j) as i32;
                        sad += (s - rr).unsigned_abs() as u64;
                    }
                }
                if sad < best_sad {
                    best_sad = sad;
                    best = (dx, dy);
                }
                dx += 2;
            }
            dy += 2;
        }
        (best, best_sad)
    }

    fn amvp_predictor(&self, x: u32, y: u32, w: u32, h: u32, is_l1: bool) -> (i32, i32) {
        let grid = if is_l1 {
            &self.mv_grid_l1
        } else {
            &self.mv_grid_l0
        };
        let fetch = |xi: i32, yi: i32| -> Option<(i32, i32)> {
            if xi < 0 || yi < 0 {
                return None;
            }
            let bx = (xi >> 2) as usize;
            let by = (yi >> 2) as usize;
            if bx >= self.grid_w4 || by >= self.grid_h4 {
                return None;
            }
            grid[by * self.grid_w4 + bx]
        };
        let a0 = fetch(x as i32 - 1, y as i32 + h as i32);
        let a1 = fetch(x as i32 - 1, y as i32 + h as i32 - 1);
        if let Some(mv) = a0.or(a1) {
            return mv;
        }
        let b0 = fetch(x as i32 + w as i32, y as i32 - 1);
        let b1 = fetch(x as i32 + w as i32 - 1, y as i32 - 1);
        let b2 = fetch(x as i32 - 1, y as i32 - 1);
        b0.or(b1).or(b2).unwrap_or((0, 0))
    }

    fn store_mv(&mut self, x: u32, y: u32, size: u32, mv: (i32, i32), is_l1: bool) {
        let grid = if is_l1 {
            &mut self.mv_grid_l1
        } else {
            &mut self.mv_grid_l0
        };
        let bx0 = (x >> 2) as usize;
        let by0 = (y >> 2) as usize;
        let n4 = (size >> 2) as usize;
        for dy in 0..n4 {
            for dx in 0..n4 {
                let bx = bx0 + dx;
                let by = by0 + dy;
                if bx < self.grid_w4 && by < self.grid_h4 {
                    grid[by * self.grid_w4 + bx] = Some(mv);
                }
            }
        }
    }

    /// CABAC-encode an MVD pair (§9.3.4.2.6 / §7.3.8.9). Same bit
    /// ordering as the P-slice writer's `encode_mvd`.
    fn encode_mvd(&mut self, cw: &mut CabacWriter<'_>, mvd_x: i32, mvd_y: i32) {
        let ax = mvd_x.unsigned_abs();
        let ay = mvd_y.unsigned_abs();
        let gt0_x = (ax > 0) as u32;
        let gt0_y = (ay > 0) as u32;
        cw.encode_bin(&mut self.abs_mvd_greater[0], gt0_x);
        cw.encode_bin(&mut self.abs_mvd_greater[0], gt0_y);
        let gt1_x = (ax > 1) as u32;
        let gt1_y = (ay > 1) as u32;
        if gt0_x == 1 {
            cw.encode_bin(&mut self.abs_mvd_greater[1], gt1_x);
        }
        if gt0_y == 1 {
            cw.encode_bin(&mut self.abs_mvd_greater[1], gt1_y);
        }
        if gt0_x == 1 {
            if gt1_x == 1 {
                encode_eg1(cw, ax - 2);
            }
            cw.encode_bypass((mvd_x < 0) as u32);
        }
        if gt0_y == 1 {
            if gt1_y == 1 {
                encode_eg1(cw, ay - 2);
            }
            cw.encode_bypass((mvd_y < 0) as u32);
        }
    }

    fn process_inter_luma(
        &self,
        src: &VideoFrame,
        x0: u32,
        y0: u32,
        pred: &[u8],
    ) -> (Vec<i32>, Vec<u8>) {
        let n = CTU_SIZE as usize;
        let src_y = &src.planes[0];
        let mut residual = vec![0i32; n * n];
        for dy in 0..n {
            for dx in 0..n {
                let sx = x0 as usize + dx;
                let sy = y0 as usize + dy;
                let s = src_y.data[sy * src_y.stride + sx] as i32;
                let p = pred[dy * n + dx] as i32;
                residual[dy * n + dx] = s - p;
            }
        }
        let log2_tb = 4u32;
        let mut coeffs = vec![0i32; n * n];
        forward_transform_2d(&residual, &mut coeffs, log2_tb, false, 8);
        let mut levels = vec![0i32; n * n];
        quantize_flat(&coeffs, &mut levels, SLICE_QP_Y, log2_tb, 8);

        let mut deq = vec![0i32; n * n];
        dequantize_flat(&levels, &mut deq, SLICE_QP_Y, log2_tb, 8);
        let mut res_rec = vec![0i32; n * n];
        inverse_transform_2d(&deq, &mut res_rec, log2_tb, false, 8);
        let mut rec = vec![0u8; n * n];
        for i in 0..n * n {
            let v = pred[i] as i32 + res_rec[i];
            rec[i] = v.clamp(0, 255) as u8;
        }
        (levels, rec)
    }

    fn process_inter_chroma(
        &self,
        src: &VideoFrame,
        cx: u32,
        cy: u32,
        pred: &[u8],
        is_cr: bool,
    ) -> (Vec<i32>, Vec<u8>) {
        let n = (CTU_SIZE / 2) as usize;
        let plane = if is_cr {
            &src.planes[2]
        } else {
            &src.planes[1]
        };
        let mut residual = vec![0i32; n * n];
        for dy in 0..n {
            for dx in 0..n {
                let sx = cx as usize + dx;
                let sy = cy as usize + dy;
                let s = plane.data[sy * plane.stride + sx] as i32;
                let p = pred[dy * n + dx] as i32;
                residual[dy * n + dx] = s - p;
            }
        }
        let log2_tb = 3u32;
        let qp = SLICE_QP_Y;
        let mut coeffs = vec![0i32; n * n];
        forward_transform_2d(&residual, &mut coeffs, log2_tb, false, 8);
        let mut levels = vec![0i32; n * n];
        quantize_flat(&coeffs, &mut levels, qp, log2_tb, 8);

        let mut deq = vec![0i32; n * n];
        dequantize_flat(&levels, &mut deq, qp, log2_tb, 8);
        let mut res_rec = vec![0i32; n * n];
        inverse_transform_2d(&deq, &mut res_rec, log2_tb, false, 8);
        let mut rec = vec![0u8; n * n];
        for i in 0..n * n {
            let v = pred[i] as i32 + res_rec[i];
            rec[i] = v.clamp(0, 255) as u8;
        }
        (levels, rec)
    }

    fn write_luma_block(&mut self, x: u32, y: u32, n: usize, block: &[u8]) {
        let x0 = x as usize;
        let y0 = y as usize;
        for dy in 0..n {
            for dx in 0..n {
                let xx = x0 + dx;
                let yy = y0 + dy;
                if xx < self.cfg.width as usize && yy < self.cfg.height as usize {
                    self.rec_y[yy * self.y_stride + xx] = block[dy * n + dx];
                }
            }
        }
    }

    fn write_chroma_block(&mut self, x: u32, y: u32, n: usize, block: &[u8], is_cr: bool) {
        let x0 = x as usize;
        let y0 = y as usize;
        let cw = self.cfg.width as usize / 2;
        let ch = self.cfg.height as usize / 2;
        let plane = if is_cr {
            &mut self.rec_cr
        } else {
            &mut self.rec_cb
        };
        for dy in 0..n {
            for dx in 0..n {
                let xx = x0 + dx;
                let yy = y0 + dy;
                if xx < cw && yy < ch {
                    plane[yy * self.c_stride + xx] = block[dy * n + dx];
                }
            }
        }
    }
}

/// Integer-pel motion compensation (free-standing helper so we can run
/// it for L0 / L1 candidates without borrowing the encoder state mutably).
fn motion_compensate(
    r: &ReferenceFrame,
    x0: u32,
    y0: u32,
    mv_x: i32,
    mv_y: i32,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let n = CTU_SIZE as usize;
    let cn = n / 2;
    let mut luma = vec![0u8; n * n];
    let mut cb = vec![0u8; cn * cn];
    let mut cr = vec![0u8; cn * cn];
    for j in 0..n {
        for i in 0..n {
            luma[j * n + i] = r.sample_y(x0 as i32 + i as i32 + mv_x, y0 as i32 + j as i32 + mv_y);
        }
    }
    let cx0 = (x0 / 2) as i32;
    let cy0 = (y0 / 2) as i32;
    let mv_cx = mv_x / 2;
    let mv_cy = mv_y / 2;
    for j in 0..cn {
        for i in 0..cn {
            cb[j * cn + i] = r.sample_c(cx0 + i as i32 + mv_cx, cy0 + j as i32 + mv_cy, false);
            cr[j * cn + i] = r.sample_c(cx0 + i as i32 + mv_cx, cy0 + j as i32 + mv_cy, true);
        }
    }
    (luma, cb, cr)
}

/// Default-bipred average `(P0 + P1 + 1) >> 1` (§8.5.3.3.3.1) for the
/// three planes.
fn make_bipred(
    y0: &[u8],
    cb0: &[u8],
    cr0: &[u8],
    y1: &[u8],
    cb1: &[u8],
    cr1: &[u8],
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; y0.len()];
    let mut cb = vec![0u8; cb0.len()];
    let mut cr = vec![0u8; cr0.len()];
    for i in 0..y.len() {
        y[i] = ((y0[i] as u32 + y1[i] as u32 + 1) >> 1) as u8;
    }
    for i in 0..cb.len() {
        cb[i] = ((cb0[i] as u32 + cb1[i] as u32 + 1) >> 1) as u8;
    }
    for i in 0..cr.len() {
        cr[i] = ((cr0[i] as u32 + cr1[i] as u32 + 1) >> 1) as u8;
    }
    (y, cb, cr)
}

/// Luma SAD between source CTU and a prediction buffer.
fn sad_luma(src: &VideoFrame, x0: u32, y0: u32, pred: &[u8], size: u32) -> u64 {
    let n = size as usize;
    let src_y = &src.planes[0];
    let mut sad = 0u64;
    for dy in 0..n {
        for dx in 0..n {
            let s = src_y.data[(y0 as usize + dy) * src_y.stride + x0 as usize + dx] as i32;
            let p = pred[dy * n + dx] as i32;
            sad += (s - p).unsigned_abs() as u64;
        }
    }
    sad
}

/// First-order Exp-Golomb bypass encode — duplicate of the helper in
/// `p_slice_writer.rs` to keep the two writers self-contained.
fn encode_eg1(cw: &mut CabacWriter<'_>, v: u32) {
    let value = v + 2;
    let k = 32 - value.leading_zeros() - 1;
    for _ in 0..(k - 1) {
        cw.encode_bypass(1);
    }
    cw.encode_bypass(0);
    let suf_bits = k;
    let suf = value - (1u32 << k);
    for bit in (0..suf_bits).rev() {
        cw.encode_bypass((suf >> bit) & 1);
    }
}
