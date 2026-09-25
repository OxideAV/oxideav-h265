//! Picture-sequence decode state machine: the §8.3.1 → §8.3.2 → §8.3.4 →
//! §8.3.5 per-picture chain that ties the [`crate::poc`] POC derivation
//! and the [`crate::dpb`] decoded-picture buffer together.
//!
//! [`PictureSequenceState`] threads the cross-picture state — the
//! [`crate::poc::PocState`] (`prevTid0Pic`) and the [`crate::dpb::Dpb`] —
//! across a coded video sequence. For each coded picture the driver:
//!
//! 1. derives `PicOrderCntVal` (§8.3.1) from the picture's `NalKind`,
//!    `TemporalId`, `NoRaslOutputFlag` and `slice_pic_order_cnt_lsb`;
//! 2. builds the five §8.3.2 RPS POC lists and applies the §8.3.2 marking
//!    to the DPB, resolving the RPS to concrete DPB entries;
//! 3. for a P / B slice, constructs `RefPicList0` / `RefPicList1` (§8.3.4)
//!    and selects the §8.3.5 `ColPic` + `NoBackwardPredFlag` for temporal
//!    MV prediction.
//!
//! After the picture's samples + per-PU motion field are reconstructed,
//! [`PictureSequenceState::store_picture`] inserts it into the DPB as a
//! short-term reference, completing the per-picture cycle.

use crate::dpb::{
    build_rps_poc_lists, no_backward_pred_flag, select_col_pic, Dpb, DpbEntry, LongTermEntry,
    Marking, RefPicListParams, RefPicLists, ResolvedRps,
};
use crate::motion::MotionField;
use crate::picture::Picture;
use crate::poc::{NalKind, PicOrderCnt, PocState};
use crate::sps::MaterializedShortTermRefPicSet;

/// The per-picture §8.3 inputs the [`PictureSequenceState`] driver needs,
/// resolved from the picture's NAL header + activated SPS + parsed slice
/// segment header.
#[derive(Debug, Clone)]
pub struct PictureHeaderInfo {
    /// The picture's `nal_unit_type` classification.
    pub nal_kind: NalKind,
    /// `TemporalId`.
    pub temporal_id: u8,
    /// `nuh_layer_id`.
    pub layer_id: u8,
    /// `NoRaslOutputFlag`.
    pub no_rasl_output: bool,
    /// `slice_pic_order_cnt_lsb` (0 for an IDR).
    pub poc_lsb: u32,
    /// `MaxPicOrderCntLsb` from the active SPS.
    pub max_poc_lsb: u32,
    /// The current picture's materialized short-term RPS
    /// (`DeltaPocS0` / `UsedByCurrPicS0` / …).
    pub short_term_rps: MaterializedShortTermRefPicSet,
    /// The current picture's long-term RPS entries (§7.4.7.1-resolved).
    pub long_term: Vec<LongTermEntry>,
}

/// The §8.3 per-picture decode outputs the slice / CU decode then uses.
#[derive(Debug, Clone)]
pub struct PictureRefState {
    /// The derived `PicOrderCntVal` (§8.3.1).
    pub poc: PicOrderCnt,
    /// The §8.3.2 RPS resolved to DPB entry indices.
    pub rps: ResolvedRps,
    /// `RefPicList0` / `RefPicList1` (§8.3.4), `None` for an I slice.
    pub ref_pic_lists: Option<RefPicLists>,
    /// `ColPic` (§8.3.5) — DPB index of the collocated picture, when
    /// `slice_temporal_mvp_enabled_flag` and a P / B slice selected one.
    pub col_pic: Option<usize>,
    /// `NoBackwardPredFlag` (§8.3.5).
    pub no_backward_pred: bool,
}

/// The per-slice §8.3.4 / §8.3.5 inputs (the reference-list sizing + the
/// collocated-picture selectors from the slice segment header).
#[derive(Debug, Clone, Default)]
pub struct SliceRefParams {
    /// `true` for a P or B slice (builds reference lists).
    pub is_inter: bool,
    /// `true` for a B slice (builds `RefPicList1`).
    pub is_b: bool,
    /// `num_ref_idx_l0_active_minus1`.
    pub num_ref_idx_l0_active_minus1: u32,
    /// `num_ref_idx_l1_active_minus1`.
    pub num_ref_idx_l1_active_minus1: u32,
    /// `NumPicTotalCurr` (§7.4.7.2).
    pub num_pic_total_curr: u32,
    /// `slice_temporal_mvp_enabled_flag`.
    pub temporal_mvp_enabled: bool,
    /// `collocated_from_l0_flag` (inferred 1 for a P slice).
    pub collocated_from_l0_flag: bool,
    /// `collocated_ref_idx` (inferred 0 when absent).
    pub collocated_ref_idx: u32,
    /// `pps_curr_pic_ref_enabled_flag` — the §8.3.4 currPic append
    /// (intra block copy).
    pub curr_pic_ref_enabled: bool,
    /// `list_entry_l0[ ]` when `ref_pic_list_modification_flag_l0`
    /// (§7.3.6.2) — reorders `RefPicListTemp0` (eq. 8-9 / F-66).
    pub list_entry_l0: Option<Vec<u32>>,
    /// `list_entry_l1[ ]` when `ref_pic_list_modification_flag_l1`.
    pub list_entry_l1: Option<Vec<u32>>,
}

/// The Annex F per-picture inputs of a `nuh_layer_id > 0` picture (or
/// of any picture of a multi-layer bitstream): the F.8.3.1 POC form
/// and the G.8.1.3 / H.8.1.3 inter-layer reference picture sets.
#[derive(Debug, Clone, Default)]
pub struct LayeredPictureInputs {
    /// `true` when the F.8.3.1 POC derivation applies (a multi-layer
    /// bitstream); `false` keeps the §8.3.1 single-layer form.
    pub annex_f_poc: bool,
    /// `FirstPicInLayerDecodedFlag[ nuh_layer_id ]` BEFORE this picture.
    pub first_pic_in_layer_decoded: bool,
    /// `poc_msb_cycle_val` when `poc_msb_cycle_val_present_flag`.
    pub poc_msb_cycle_val: Option<u32>,
    /// `PicOrderCntVal` computed by the driver for a POC resetting
    /// picture (F.8.3.1 eq. F-61); overrides the eq. F-62 derivation.
    pub poc_override: Option<i32>,
    /// `true` when the picture updates `PrevPicOrderCnt[ nuh_layer_id ]`
    /// (F.8.3.1: `TemporalId == 0`, not RASL / RADL / SLNR,
    /// `discardable_flag == 0`).
    pub prev_poc_anchor: bool,
    /// `RefPicSetInterLayer0` as DPB entry indices (already marked
    /// "used for long-term reference" by the driver).
    pub inter_layer0: Vec<Option<usize>>,
    /// `RefPicSetInterLayer1`.
    pub inter_layer1: Vec<Option<usize>>,
}

/// The cross-picture decode state for one coded video sequence.
#[derive(Debug, Default)]
pub struct PictureSequenceState {
    /// `PrevPicOrderCnt[ nuh_layer_id ]` — one §8.3.1 / F.8.3.1 state
    /// per layer.
    poc_states: std::collections::BTreeMap<u8, PocState>,
    dpb: Dpb,
}

impl PictureSequenceState {
    /// A fresh state at the start of a coded video sequence.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrow the decoded-picture buffer (reference-picture inspection).
    #[must_use]
    pub fn dpb(&self) -> &Dpb {
        &self.dpb
    }

    /// Mutably borrow the decoded-picture buffer (the Annex F/G/H
    /// driver's inter-layer marking, POC resets and temporary
    /// resampled references).
    pub fn dpb_mut(&mut self) -> &mut Dpb {
        &mut self.dpb
    }

    /// The per-layer POC state (`PrevPicOrderCnt[ layer_id ]`).
    pub fn poc_state_mut(&mut self, layer_id: u8) -> &mut PocState {
        self.poc_states.entry(layer_id).or_default()
    }

    /// §8.3.1 → §8.3.2 → §8.3.4 → §8.3.5 — run the per-picture reference
    /// derivation for a picture about to be decoded.
    ///
    /// `header` carries the POC + RPS inputs; `slice` carries the
    /// reference-list / collocated-picture selectors. The returned
    /// [`PictureRefState`] is consumed by the slice / CU decode (its
    /// `RefPicList0` / `ColPic` resolve `RefPicListX[ refIdx ]` and the
    /// temporal-MV collocated picture).
    pub fn begin_picture(
        &mut self,
        header: &PictureHeaderInfo,
        slice: &SliceRefParams,
    ) -> PictureRefState {
        self.begin_picture_layered(header, slice, &LayeredPictureInputs::default())
    }

    /// [`Self::begin_picture`] with the Annex F inputs: the F.8.3.1 POC
    /// form (per-layer `PrevPicOrderCnt`, eq. F-62) and the inter-layer
    /// reference picture sets spliced into `RefPicListTemp0/1` per
    /// eqs F-65 / F-67.
    pub fn begin_picture_layered(
        &mut self,
        header: &PictureHeaderInfo,
        slice: &SliceRefParams,
        layered: &LayeredPictureInputs,
    ) -> PictureRefState {
        // Step 1 — §8.3.1 / F.8.3.1 POC, per layer.
        let poc_state = self.poc_states.entry(header.layer_id).or_default();
        let poc = if layered.annex_f_poc {
            let poc = match layered.poc_override {
                Some(val) => PicOrderCnt {
                    msb: val.wrapping_sub(header.poc_lsb as i32),
                    lsb: header.poc_lsb,
                    val,
                },
                None => poc_state.derive_f62(
                    header.nal_kind.is_idr(),
                    layered.first_pic_in_layer_decoded,
                    header.poc_lsb,
                    header.max_poc_lsb,
                    layered.poc_msb_cycle_val,
                ),
            };
            // F.8.3.1: PrevPicOrderCnt[ lId ] follows a non-RASL /
            // RADL / SLNR picture with TemporalId 0 and
            // discardable_flag 0.
            if layered.prev_poc_anchor {
                poc_state.update_prev_tid0(poc);
            }
            poc
        } else {
            poc_state.decode_picture_poc(
                header.nal_kind,
                header.temporal_id,
                header.no_rasl_output,
                header.poc_lsb,
                header.max_poc_lsb,
            )
        };

        // Step 2 — §8.3.2 RPS POC lists + DPB marking.
        let lists = build_rps_poc_lists(
            header.nal_kind.is_idr(),
            poc.val,
            header.max_poc_lsb,
            &header.short_term_rps,
            &header.long_term,
        );
        let is_irap_no_rasl = header.nal_kind.is_irap() && header.no_rasl_output;
        let rps = self
            .dpb
            .apply_rps(is_irap_no_rasl, header.layer_id, &lists, header.max_poc_lsb);

        // Step 3 — §8.3.4 reference picture lists (P / B only).
        let ref_pic_lists = if slice.is_inter {
            Some(self.dpb.build_ref_pic_lists(
                &rps,
                &RefPicListParams {
                    num_ref_idx_l0_active_minus1: slice.num_ref_idx_l0_active_minus1,
                    num_ref_idx_l1_active_minus1: slice.num_ref_idx_l1_active_minus1,
                    num_pic_total_curr: slice.num_pic_total_curr,
                    is_b: slice.is_b,
                    list_entry_l0: slice.list_entry_l0.as_deref(),
                    list_entry_l1: slice.list_entry_l1.as_deref(),
                    curr_pic_ref_enabled: slice.curr_pic_ref_enabled,
                    inter_layer0: &layered.inter_layer0,
                    inter_layer1: &layered.inter_layer1,
                },
            ))
        } else {
            None
        };

        // Step 4 — §8.3.5 ColPic + NoBackwardPredFlag.
        let (col_pic, no_backward_pred) = match &ref_pic_lists {
            Some(rpl) if slice.temporal_mvp_enabled => (
                select_col_pic(
                    rpl,
                    slice.is_b,
                    slice.collocated_from_l0_flag,
                    slice.collocated_ref_idx,
                ),
                no_backward_pred_flag(&self.dpb, rpl, poc.val),
            ),
            Some(rpl) => (None, no_backward_pred_flag(&self.dpb, rpl, poc.val)),
            None => (None, true),
        };

        PictureRefState {
            poc,
            rps,
            ref_pic_lists,
            col_pic,
            no_backward_pred,
        }
    }

    /// Insert a freshly-decoded picture into the DPB as a short-term
    /// reference, completing the per-picture cycle. `picture` are the
    /// reconstructed (in-loop-filtered) samples; `motion` is the per-PU
    /// motion field for the §8.5.3.2.9 collocated-MV derivation of future
    /// pictures.
    pub fn store_picture(
        &mut self,
        poc: PicOrderCnt,
        layer_id: u8,
        picture: Picture,
        motion: MotionField,
    ) {
        self.dpb.insert(DpbEntry {
            poc: poc.val,
            layer_id,
            marking: Marking::ShortTerm,
            picture,
            motion,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_rps() -> MaterializedShortTermRefPicSet {
        MaterializedShortTermRefPicSet {
            delta_poc_s0: vec![],
            used_by_curr_pic_s0: vec![],
            delta_poc_s1: vec![],
            used_by_curr_pic_s1: vec![],
        }
    }

    fn idr_header() -> PictureHeaderInfo {
        PictureHeaderInfo {
            nal_kind: NalKind::new(NalKind::IDR_N_LP),
            temporal_id: 0,
            layer_id: 0,
            no_rasl_output: true,
            poc_lsb: 0,
            max_poc_lsb: 256,
            short_term_rps: empty_rps(),
            long_term: vec![],
        }
    }

    fn i_slice() -> SliceRefParams {
        SliceRefParams {
            is_inter: false,
            is_b: false,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            num_pic_total_curr: 0,
            temporal_mvp_enabled: false,
            collocated_from_l0_flag: true,
            collocated_ref_idx: 0,
            curr_pic_ref_enabled: false,
            list_entry_l0: None,
            list_entry_l1: None,
        }
    }

    #[test]
    fn idr_then_p_frame_resolves_ref_pic_list() {
        let mut st = PictureSequenceState::new();
        // IDR (POC 0), I slice, no references.
        let idr = st.begin_picture(&idr_header(), &i_slice());
        assert_eq!(idr.poc.val, 0);
        assert!(idr.ref_pic_lists.is_none());
        st.store_picture(
            idr.poc,
            0,
            Picture::new(16, 16, 1, 8, 8),
            MotionField::new(16, 16),
        );

        // P frame (POC 1), one short-term-before ref at POC 0.
        let p_header = PictureHeaderInfo {
            nal_kind: NalKind::new(NalKind::TRAIL_R),
            no_rasl_output: false,
            poc_lsb: 1,
            short_term_rps: MaterializedShortTermRefPicSet {
                delta_poc_s0: vec![-1],
                used_by_curr_pic_s0: vec![true],
                delta_poc_s1: vec![],
                used_by_curr_pic_s1: vec![],
            },
            ..idr_header()
        };
        let p_slice = SliceRefParams {
            is_inter: true,
            num_pic_total_curr: 1,
            ..i_slice()
        };
        let p = st.begin_picture(&p_header, &p_slice);
        assert_eq!(p.poc.val, 1);
        let rpl = p.ref_pic_lists.expect("P slice builds RefPicList0");
        // RefPicList0[0] points at the IDR (DPB index 0).
        assert_eq!(rpl.list0, vec![Some(0)]);
        // The IDR stays short-term (it is in the P frame's RPS).
        assert_eq!(st.dpb().entries()[0].marking, Marking::ShortTerm);
        // NoBackwardPredFlag: the only ref (POC 0) is in the past of POC 1.
        assert!(p.no_backward_pred);
    }

    #[test]
    fn temporal_mvp_selects_col_pic() {
        let mut st = PictureSequenceState::new();
        let idr = st.begin_picture(&idr_header(), &i_slice());
        st.store_picture(
            idr.poc,
            0,
            Picture::new(16, 16, 1, 8, 8),
            MotionField::new(16, 16),
        );

        let p_header = PictureHeaderInfo {
            nal_kind: NalKind::new(NalKind::TRAIL_R),
            no_rasl_output: false,
            poc_lsb: 1,
            short_term_rps: MaterializedShortTermRefPicSet {
                delta_poc_s0: vec![-1],
                used_by_curr_pic_s0: vec![true],
                delta_poc_s1: vec![],
                used_by_curr_pic_s1: vec![],
            },
            ..idr_header()
        };
        let p_slice = SliceRefParams {
            is_inter: true,
            num_pic_total_curr: 1,
            temporal_mvp_enabled: true,
            collocated_from_l0_flag: true,
            collocated_ref_idx: 0,
            curr_pic_ref_enabled: false,
            ..i_slice()
        };
        let p = st.begin_picture(&p_header, &p_slice);
        // ColPic = RefPicList0[0] = the IDR at DPB index 0.
        assert_eq!(p.col_pic, Some(0));
    }

    #[test]
    fn second_idr_unmarks_prior_references() {
        let mut st = PictureSequenceState::new();
        let idr = st.begin_picture(&idr_header(), &i_slice());
        st.store_picture(
            idr.poc,
            0,
            Picture::new(16, 16, 1, 8, 8),
            MotionField::new(16, 16),
        );
        // A second IDR with NoRaslOutputFlag == 1 marks the prior IDR
        // "unused for reference".
        let _ = st.begin_picture(&idr_header(), &i_slice());
        assert_eq!(st.dpb().entries()[0].marking, Marking::Unused);
    }
}
