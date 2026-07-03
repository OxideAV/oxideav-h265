//! Whole-bitstream decode driver — Annex B byte stream to output
//! pictures.
//!
//! This is the §8.1 general decoding process expressed over the crate's
//! subsystems: the [`crate::nal`] Annex B demux feeds parameter-set
//! activation ([`crate::sps`] / [`crate::pps`]), each coded picture's
//! slice segments are parsed ([`crate::slice`]) and their
//! `slice_segment_data()` CABAC-decoded through the §7.3.8 syntax walk
//! ([`crate::slice_data`]), and the decoded coding tree units are handed
//! to the picture-level reconstruction + in-loop-filter driver
//! ([`crate::inter_recon::reconstruct_inter_picture`]) with the §8.3
//! POC / RPS / reference-list cycle threaded by
//! [`crate::decode::PictureSequenceState`]. Decoded pictures are
//! returned in output order (§8.3.1 `PicOrderCntVal` order within each
//! coded video sequence).

use std::collections::BTreeMap;

use crate::availability::{PictureTiling, TilingParams};
use crate::bitreader::BitReader;
use crate::cabac::{init_type, CabacEngine};
use crate::ctx_init::SliceContexts;
use crate::decode::{PictureHeaderInfo, PictureSequenceState, SliceRefParams};
use crate::dpb::{LongTermEntry, RefPicLists};
use crate::inter_recon::{
    reconstruct_inter_picture, InterSliceContext, PlacedInterCtu, RefListAccess,
};
use crate::nal::{NalError, NalIter, NalUnit};
use crate::picture::Picture;
use crate::poc::NalKind;
use crate::pps::{PicParameterSet, PpsError};
use crate::recon::{ReconError, ReconParams};
use crate::residual::ResidualCodingError;
use crate::slice::{SliceError, SliceLongTermRefPicSource, SliceSegmentHeader, SliceType};
use crate::slice_data::{
    decode_coding_tree_unit_in_picture, end_of_slice_segment_flag, CodingTreeUnit,
    PictureParseState, SliceDataParams,
};
use crate::sps::{
    MaterializedShortTermRefPicSet, SeqParameterSet, ShortTermRefPicSetMaterializeError, SpsError,
};

/// NAL unit type: video parameter set (Table 7-1).
const NAL_VPS: u8 = 32;
/// NAL unit type: sequence parameter set.
const NAL_SPS: u8 = 33;
/// NAL unit type: picture parameter set.
const NAL_PPS: u8 = 34;

/// Errors from the whole-bitstream decode driver.
#[derive(Debug)]
pub enum SequenceError {
    /// Annex B demux / NAL header error.
    Nal(NalError),
    /// SPS parse error.
    Sps(SpsError),
    /// PPS parse error.
    Pps(PpsError),
    /// Slice-segment-header parse error.
    Slice(SliceError),
    /// §7.3.8 slice-data CABAC walk error.
    SliceData(ResidualCodingError),
    /// Picture reconstruction error.
    Recon(ReconError),
    /// §7.4.8 short-term-RPS materialization error.
    Rps(ShortTermRefPicSetMaterializeError),
    /// A referenced parameter set was never activated.
    MissingParameterSet {
        /// `"sps"` or `"pps"`.
        kind: &'static str,
        /// The referenced parameter-set id.
        id: u8,
    },
    /// A structural bitstream-conformance failure.
    Malformed(&'static str),
    /// A conformant configuration this driver does not decode yet.
    Unsupported(&'static str),
}

impl core::fmt::Display for SequenceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Nal(e) => write!(f, "NAL demux error: {e}"),
            Self::Sps(e) => write!(f, "SPS parse error: {e}"),
            Self::Pps(e) => write!(f, "PPS parse error: {e}"),
            Self::Slice(e) => write!(f, "slice header parse error: {e}"),
            Self::SliceData(e) => write!(f, "slice data decode error: {e}"),
            Self::Recon(e) => write!(f, "picture reconstruction error: {e}"),
            Self::Rps(e) => write!(f, "short-term RPS materialization error: {e:?}"),
            Self::MissingParameterSet { kind, id } => {
                write!(f, "referenced {kind} id {id} was never activated")
            }
            Self::Malformed(what) => write!(f, "malformed bitstream: {what}"),
            Self::Unsupported(what) => write!(f, "unsupported configuration: {what}"),
        }
    }
}

impl std::error::Error for SequenceError {}

impl From<NalError> for SequenceError {
    fn from(e: NalError) -> Self {
        Self::Nal(e)
    }
}
impl From<SpsError> for SequenceError {
    fn from(e: SpsError) -> Self {
        Self::Sps(e)
    }
}
impl From<PpsError> for SequenceError {
    fn from(e: PpsError) -> Self {
        Self::Pps(e)
    }
}
impl From<SliceError> for SequenceError {
    fn from(e: SliceError) -> Self {
        Self::Slice(e)
    }
}
impl From<ResidualCodingError> for SequenceError {
    fn from(e: ResidualCodingError) -> Self {
        Self::SliceData(e)
    }
}
impl From<ReconError> for SequenceError {
    fn from(e: ReconError) -> Self {
        Self::Recon(e)
    }
}
impl From<ShortTermRefPicSetMaterializeError> for SequenceError {
    fn from(e: ShortTermRefPicSetMaterializeError) -> Self {
        Self::Rps(e)
    }
}

/// One decoded picture with its output-ordering keys.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// Index of the coded video sequence this picture belongs to
    /// (incremented at each IRAP with `NoRaslOutputFlag == 1`).
    pub cvs_index: u32,
    /// `PicOrderCntVal` (§8.3.1).
    pub poc: i32,
    /// `pic_output_flag` — `false` pictures are decoded (they may be
    /// referenced) but not output.
    pub output: bool,
    /// The reconstructed, in-loop-filtered picture.
    pub picture: Picture,
}

/// One slice segment of the picture being assembled.
#[derive(Debug)]
struct SegmentData {
    nal_type: u8,
    temporal_id: u8,
    layer_id: u8,
    rbsp: Vec<u8>,
    /// Coded (escaped) payload — the §7.4.7.1 entry-point offsets are
    /// expressed in this byte space.
    escaped: Vec<u8>,
    header: SliceSegmentHeader,
}

/// The whole-bitstream decoder: parameter-set activation + per-picture
/// slice-data decode + the §8.3 reference cycle.
#[derive(Debug, Default)]
pub struct SequenceDecoder {
    sps: BTreeMap<u8, SeqParameterSet>,
    pps: BTreeMap<u8, PicParameterSet>,
    state: PictureSequenceState,
    frames: Vec<DecodedFrame>,
    pending: Vec<SegmentData>,
    cvs_index: u32,
    seen_picture: bool,
    /// Debug: tolerate an end_of_slice_segment_flag mismatch (decode
    /// as much as possible instead of erroring).
    tolerant: bool,
}

impl SequenceDecoder {
    /// A fresh decoder with no activated parameter sets.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Debug: keep decoding past an `end_of_slice_segment_flag`
    /// mismatch. Not part of the stable API.
    #[doc(hidden)]
    pub fn set_tolerant(&mut self, tolerant: bool) {
        self.tolerant = tolerant;
    }

    /// Feed a whole Annex B byte stream, decoding every access unit.
    ///
    /// # Errors
    /// Any demux / parse / decode error; the decoder state is
    /// unspecified after an error.
    pub fn push_annexb(&mut self, data: &[u8]) -> Result<(), SequenceError> {
        for unit in NalIter::new(data) {
            self.push_nal_unit(unit?)?;
        }
        Ok(())
    }

    /// Feed one demuxed NAL unit.
    ///
    /// # Errors
    /// Any parse / decode error.
    pub fn push_nal_unit(&mut self, unit: NalUnit) -> Result<(), SequenceError> {
        let NalUnit {
            header,
            rbsp,
            escaped,
        } = unit;
        if header.is_vcl() {
            // §7.4.2.4.4: a VCL NAL with first_slice_segment_in_pic_flag
            // set starts a new access unit.
            let first_in_pic = rbsp.first().is_some_and(|b| b & 0x80 != 0);
            if first_in_pic {
                self.finish_picture()?;
            }
            let pps_id = peek_slice_pps_id(&rbsp, header.nal_unit_type)?;
            let pps = self
                .pps
                .get(&pps_id)
                .ok_or(SequenceError::MissingParameterSet {
                    kind: "pps",
                    id: pps_id,
                })?;
            let sps = self
                .sps
                .get(&pps.sps_id)
                .ok_or(SequenceError::MissingParameterSet {
                    kind: "sps",
                    id: pps.sps_id,
                })?;
            let parsed = SliceSegmentHeader::parse(&rbsp, header.nal_unit_type, sps, pps)?;
            self.pending.push(SegmentData {
                nal_type: header.nal_unit_type,
                temporal_id: header.temporal_id,
                layer_id: header.nuh_layer_id,
                rbsp,
                escaped,
                header: parsed,
            });
            return Ok(());
        }
        match header.nal_unit_type {
            NAL_VPS => {
                // The VPS carries no fields the single-layer decode
                // needs; activation is a no-op.
            }
            NAL_SPS => {
                let sps = SeqParameterSet::parse(&rbsp)?;
                self.sps.insert(sps.sps_id, sps);
            }
            NAL_PPS => {
                let pps = PicParameterSet::parse(&rbsp)?;
                self.pps.insert(pps.pps_id, pps);
            }
            // AUD / EOS / EOB / FD / SEI: nothing to activate.
            _ => {}
        }
        Ok(())
    }

    /// Decode any picture still being assembled (a flush point — call
    /// when the input stream ends but the decoder object lives on).
    ///
    /// # Errors
    /// Any decode error from the pending picture.
    pub fn flush(&mut self) -> Result<(), SequenceError> {
        self.finish_picture()
    }

    /// Drain the pictures decoded so far, in decode order. The caller
    /// owns output reordering (the streaming [`crate::decoder`] holds a
    /// `sps_max_num_reorder_pics`-deep queue; [`Self::finish`] sorts a
    /// whole sequence at once).
    pub fn take_decoded(&mut self) -> Vec<DecodedFrame> {
        std::mem::take(&mut self.frames)
    }

    /// `sps_max_num_reorder_pics` of the highest sub-layer of the most
    /// recently activated SPS (`None` before any SPS).
    #[must_use]
    pub fn max_num_reorder_pics(&self) -> Option<u32> {
        self.sps.values().next_back().map(|sps| {
            let idx =
                usize::from(sps.max_sub_layers_minus1).min(sps.sub_layer_ordering_info.len() - 1);
            sps.sub_layer_ordering_info[idx].max_num_reorder_pics
        })
    }

    /// Decode any picture still being assembled and return every
    /// decoded frame in output order.
    ///
    /// # Errors
    /// Any decode error from the final pending picture.
    pub fn finish(mut self) -> Result<Vec<DecodedFrame>, SequenceError> {
        self.finish_picture()?;
        // §C.5.2.2 output order: `PicOrderCntVal` order within each
        // coded video sequence, sequences in decode order.
        let mut frames = self.frames;
        frames.sort_by_key(|f| (f.cvs_index, f.poc));
        Ok(frames)
    }

    /// Decode the pending picture's slice segments into a picture.
    fn finish_picture(&mut self) -> Result<(), SequenceError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let segs = std::mem::take(&mut self.pending);
        self.decode_picture(&segs)
    }

    fn decode_picture(&mut self, segs: &[SegmentData]) -> Result<(), SequenceError> {
        let indep = &segs[0];
        if indep.header.dependent_slice_segment_flag {
            return Err(SequenceError::Malformed(
                "first slice segment of a picture is dependent",
            ));
        }
        let pps = self
            .pps
            .get(&indep.header.slice_pic_parameter_set_id)
            .ok_or(SequenceError::MissingParameterSet {
                kind: "pps",
                id: indep.header.slice_pic_parameter_set_id,
            })?;
        let sps = self
            .sps
            .get(&pps.sps_id)
            .ok_or(SequenceError::MissingParameterSet {
                kind: "sps",
                id: pps.sps_id,
            })?;

        let geom = Geometry::derive(sps, pps)?;

        // §7.4.2.4.4 CVS bookkeeping: an IRAP with NoRaslOutputFlag
        // starts a new coded video sequence (for output ordering).
        let nal_kind = NalKind::new(indep.nal_type);
        let no_rasl_output =
            nal_kind.is_idr() || nal_kind.is_bla() || (nal_kind.is_irap() && !self.seen_picture);
        if nal_kind.is_irap() && no_rasl_output && self.seen_picture {
            self.cvs_index += 1;
        }
        self.seen_picture = true;

        // ---- §7.3.8 slice-data CABAC decode of every slice segment ----
        let pic_size_in_ctbs = (geom.pic_w_ctbs * geom.pic_h_ctbs) as usize;
        let mut decoded: Vec<(u32, u32, CodingTreeUnit)> = Vec::new();
        let mut slice_addr_of: Vec<Option<u32>> = vec![None; pic_size_in_ctbs];
        let first_slice_type = segs[0]
            .header
            .slice_type
            .ok_or(SequenceError::Malformed("independent slice without type"))?;
        let mut parse_state = PictureParseState::new(&build_slice_data_params(
            &segs[0].header,
            sps,
            pps,
            &geom,
            first_slice_type,
        ));

        for seg in segs {
            if seg.header.dependent_slice_segment_flag {
                return Err(SequenceError::Unsupported(
                    "dependent slice segments (SliceAddrRs inheritance) are not decoded yet",
                ));
            }
            decode_slice_segment_data(
                seg,
                sps,
                pps,
                &geom,
                &mut parse_state,
                &mut decoded,
                &mut slice_addr_of,
                self.tolerant,
            )?;
        }

        // ---- §8.3 reference cycle + §8.4/§8.5/§8.7 reconstruction ----
        let slice_type = indep
            .header
            .slice_type
            .ok_or(SequenceError::Malformed("independent slice without type"))?;
        let header_info = self.build_header_info(indep, sps, nal_kind, no_rasl_output)?;
        let slice_ref = build_slice_ref_params(&indep.header, pps, slice_type, &header_info);

        let ref_state = self.state.begin_picture(&header_info, &slice_ref);
        let lists = ref_state.ref_pic_lists.clone().unwrap_or(RefPicLists {
            list0: Vec::new(),
            list1: None,
        });
        let entries = self.state.dpb().entries();
        let col_field = ref_state.col_pic.map(|idx| &entries[idx].motion);
        let col_poc = ref_state
            .col_pic
            .map(|idx| entries[idx].poc)
            .unwrap_or_default();
        let refs = RefListAccess {
            lists: &lists,
            entries,
        };

        let recon_params = build_recon_params(&indep.header, sps, pps, &geom)?;
        let slice_ctx = build_inter_slice_context(
            &indep.header,
            sps,
            pps,
            &geom,
            &recon_params,
            ref_state.poc.val,
            col_poc,
            ref_state.no_backward_pred,
            slice_type,
        );

        let placed: Vec<PlacedInterCtu<'_>> = decoded
            .iter()
            .map(|(x, y, ctu)| {
                let rs = (y >> geom.ctb_log2) * geom.pic_w_ctbs + (x >> geom.ctb_log2);
                PlacedInterCtu {
                    x_ctb: *x,
                    y_ctb: *y,
                    slice_addr_rs: slice_addr_of[rs as usize].unwrap_or(0),
                    ctu,
                }
            })
            .collect();

        let (picture, motion) = reconstruct_inter_picture(
            geom.width as usize,
            geom.height as usize,
            &recon_params,
            &slice_ctx,
            &geom.tiles,
            &placed,
            &refs,
            col_field,
        )?;

        let output = indep.header.pic_output_flag;
        let poc = ref_state.poc;
        self.frames.push(DecodedFrame {
            cvs_index: self.cvs_index,
            poc: poc.val,
            output,
            picture: picture.clone(),
        });
        self.state
            .store_picture(poc, indep.layer_id, picture, motion);
        Ok(())
    }

    /// Assemble the §8.3 [`PictureHeaderInfo`] from the independent
    /// slice segment header.
    fn build_header_info(
        &self,
        seg: &SegmentData,
        sps: &SeqParameterSet,
        nal_kind: NalKind,
        no_rasl_output: bool,
    ) -> Result<PictureHeaderInfo, SequenceError> {
        let max_poc_lsb = 1u32 << (sps.log2_max_pic_order_cnt_lsb_minus4 + 4);
        let short_term_rps = materialize_slice_rps(&seg.header, sps)?;
        let mut long_term = Vec::new();
        for lt in &seg.header.long_term_ref_pics {
            let poc_lsb_lt = match lt.source {
                SliceLongTermRefPicSource::Sps { lt_idx_sps } => {
                    sps.long_term_ref_pics
                        .get(lt_idx_sps as usize)
                        .ok_or(SequenceError::Malformed(
                            "lt_idx_sps out of range of the SPS long-term table",
                        ))?
                        .poc_lsb
                }
                SliceLongTermRefPicSource::InSlice { poc_lsb_lt, .. } => poc_lsb_lt,
            };
            let used = lt.used_by_curr_pic_lt(sps).ok_or(SequenceError::Malformed(
                "lt_idx_sps out of range of the SPS long-term table",
            ))?;
            long_term.push(LongTermEntry {
                poc_lsb_lt,
                used_by_curr_pic_lt: used,
                delta_poc_msb_present: lt.delta_poc_msb_present_flag,
                delta_poc_msb_cycle_lt: lt.delta_poc_msb_cycle_lt,
            });
        }
        Ok(PictureHeaderInfo {
            nal_kind,
            temporal_id: seg.temporal_id,
            layer_id: seg.layer_id,
            no_rasl_output,
            poc_lsb: seg.header.slice_pic_order_cnt_lsb.unwrap_or(0),
            max_poc_lsb,
            short_term_rps,
            long_term,
        })
    }
}

/// Decode a whole Annex B byte stream to its output-order frames.
///
/// # Errors
/// Any demux / parse / decode error.
pub fn decode_annexb_sequence(data: &[u8]) -> Result<Vec<DecodedFrame>, SequenceError> {
    let mut dec = SequenceDecoder::new();
    dec.push_annexb(data)?;
    dec.finish()
}

/// Debug helper: CABAC-decode the FIRST picture's slice-segment data and
/// return whatever CTUs were decoded, even when the walk diverges (the
/// `end_of_slice_segment_flag` never fires). Not part of the stable API.
#[doc(hidden)]
pub fn decode_annexb_sequence_debug(
    data: &[u8],
) -> Result<Vec<(u32, u32, CodingTreeUnit)>, SequenceError> {
    // Which picture (1-based) to CABAC-decode; earlier pictures are
    // fully decoded through the normal driver so the DPB and parse
    // state are real.
    let target: usize = std::env::var("H265_DEBUG_PIC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let mut dec = SequenceDecoder::new();
    let mut segs: Vec<SegmentData> = Vec::new();
    let mut pic_no = 0usize;
    for unit in NalIter::new(data) {
        let unit = unit?;
        if unit.header.is_vcl() {
            let first = unit.rbsp.first().is_some_and(|b| b & 0x80 != 0);
            if first {
                pic_no += 1;
                if pic_no > target && !segs.is_empty() {
                    break;
                }
            }
            if pic_no < target {
                dec.push_nal_unit(unit)?;
                continue;
            }
            let pps_id = peek_slice_pps_id(&unit.rbsp, unit.header.nal_unit_type)?;
            let pps = dec.pps.get(&pps_id).unwrap();
            let sps = dec.sps.get(&pps.sps_id).unwrap();
            let parsed =
                SliceSegmentHeader::parse(&unit.rbsp, unit.header.nal_unit_type, sps, pps)?;
            segs.push(SegmentData {
                nal_type: unit.header.nal_unit_type,
                temporal_id: unit.header.temporal_id,
                layer_id: unit.header.nuh_layer_id,
                rbsp: unit.rbsp,
                escaped: unit.escaped,
                header: parsed,
            });
        } else {
            dec.push_nal_unit(unit)?;
        }
    }
    let seg = &segs[0];
    let pps = dec.pps.get(&seg.header.slice_pic_parameter_set_id).unwrap();
    let sps = dec.sps.get(&pps.sps_id).unwrap();
    let geom = Geometry::derive(sps, pps)?;
    let pic_size = (geom.pic_w_ctbs * geom.pic_h_ctbs) as usize;
    let mut decoded = Vec::new();
    let mut slice_addr_of = vec![None; pic_size];
    let st = seg.header.slice_type.unwrap();
    let mut parse_state =
        PictureParseState::new(&build_slice_data_params(&seg.header, sps, pps, &geom, st));
    let res = decode_slice_segment_data(
        seg,
        sps,
        pps,
        &geom,
        &mut parse_state,
        &mut decoded,
        &mut slice_addr_of,
        false,
    );
    if let Err(e) = res {
        eprintln!("(walk error: {e})");
    }
    Ok(decoded)
}

/// Debug helper: reconstruct the FIRST picture even when the CABAC walk
/// diverges, returning the in-loop-filtered picture. Not stable API.
#[doc(hidden)]
pub fn decode_annexb_first_picture_tolerant(data: &[u8]) -> Result<Picture, SequenceError> {
    let mut dec = SequenceDecoder::new();
    let mut segs: Vec<SegmentData> = Vec::new();
    for unit in NalIter::new(data) {
        let unit = unit?;
        if unit.header.is_vcl() {
            let first = unit.rbsp.first().is_some_and(|b| b & 0x80 != 0);
            if first && !segs.is_empty() {
                break;
            }
            let pps_id = peek_slice_pps_id(&unit.rbsp, unit.header.nal_unit_type)?;
            let pps = dec.pps.get(&pps_id).unwrap();
            let sps = dec.sps.get(&pps.sps_id).unwrap();
            let parsed =
                SliceSegmentHeader::parse(&unit.rbsp, unit.header.nal_unit_type, sps, pps)?;
            segs.push(SegmentData {
                nal_type: unit.header.nal_unit_type,
                temporal_id: unit.header.temporal_id,
                layer_id: unit.header.nuh_layer_id,
                rbsp: unit.rbsp,
                escaped: unit.escaped,
                header: parsed,
            });
        } else {
            dec.push_nal_unit(unit)?;
        }
    }
    let seg = &segs[0];
    let pps = dec
        .pps
        .get(&seg.header.slice_pic_parameter_set_id)
        .unwrap()
        .clone();
    let sps = dec.sps.get(&pps.sps_id).unwrap().clone();
    let geom = Geometry::derive(&sps, &pps)?;
    let pic_size = (geom.pic_w_ctbs * geom.pic_h_ctbs) as usize;
    let mut decoded = Vec::new();
    let mut slice_addr_of = vec![None; pic_size];
    let st0 = seg.header.slice_type.unwrap();
    let mut parse_state = PictureParseState::new(&build_slice_data_params(
        &seg.header,
        &sps,
        &pps,
        &geom,
        st0,
    ));
    if let Err(e) = decode_slice_segment_data(
        seg,
        &sps,
        &pps,
        &geom,
        &mut parse_state,
        &mut decoded,
        &mut slice_addr_of,
        true,
    ) {
        eprintln!("(walk error: {e})");
    }
    let slice_type = seg.header.slice_type.unwrap();
    let recon_params = build_recon_params(&seg.header, &sps, &pps, &geom)?;
    let slice_ctx = build_inter_slice_context(
        &seg.header,
        &sps,
        &pps,
        &geom,
        &recon_params,
        0,
        0,
        true,
        slice_type,
    );
    let placed: Vec<PlacedInterCtu<'_>> = decoded
        .iter()
        .map(|(x, y, ctu)| PlacedInterCtu {
            x_ctb: *x,
            y_ctb: *y,
            slice_addr_rs: 0,
            ctu,
        })
        .collect();
    let lists = RefPicLists {
        list0: Vec::new(),
        list1: None,
    };
    let refs = RefListAccess {
        lists: &lists,
        entries: &[],
    };
    let (picture, _) = reconstruct_inter_picture(
        geom.width as usize,
        geom.height as usize,
        &recon_params,
        &slice_ctx,
        &geom.tiles,
        &placed,
        &refs,
        None,
    )?;
    Ok(picture)
}

/// The per-SPS geometry constants (§7.4.3.2.1 derived variables).
struct Geometry {
    width: u32,
    height: u32,
    ctb_log2: u32,
    min_cb_log2: u32,
    min_tb_log2: u32,
    max_tb_log2: u32,
    chroma_array_type: u8,
    pic_w_ctbs: u32,
    pic_h_ctbs: u32,
    tiles: TilingParams,
}

impl Geometry {
    fn derive(sps: &SeqParameterSet, pps: &PicParameterSet) -> Result<Self, SequenceError> {
        let min_cb_log2 = u32::from(sps.log2_min_luma_coding_block_size_minus3) + 3;
        let ctb_log2 = min_cb_log2 + u32::from(sps.log2_diff_max_min_luma_coding_block_size);
        let min_tb_log2 = u32::from(sps.log2_min_luma_transform_block_size_minus2) + 2;
        let max_tb_log2 = min_tb_log2 + u32::from(sps.log2_diff_max_min_luma_transform_block_size);
        let width = sps.pic_width_in_luma_samples;
        let height = sps.pic_height_in_luma_samples;
        if width == 0 || height == 0 {
            return Err(SequenceError::Malformed("zero picture dimensions"));
        }
        let ctb = 1u32 << ctb_log2;
        let chroma_array_type = if sps.separate_colour_plane_flag {
            0
        } else {
            sps.chroma_format_idc
        };
        let tiles = if pps.tiles_enabled_flag {
            TilingParams {
                num_tile_columns_minus1: pps.tiles.num_tile_columns_minus1,
                num_tile_rows_minus1: pps.tiles.num_tile_rows_minus1,
                uniform_spacing_flag: pps.tiles.uniform_spacing_flag,
                column_width_minus1: pps.tiles.column_width_minus1.clone(),
                row_height_minus1: pps.tiles.row_height_minus1.clone(),
            }
        } else {
            TilingParams::single_tile()
        };
        Ok(Self {
            width,
            height,
            ctb_log2,
            min_cb_log2,
            min_tb_log2,
            max_tb_log2,
            chroma_array_type,
            pic_w_ctbs: width.div_ceil(ctb),
            pic_h_ctbs: height.div_ceil(ctb),
            tiles,
        })
    }

    fn tiling(&self) -> Result<PictureTiling, SequenceError> {
        PictureTiling::new(
            self.pic_w_ctbs,
            self.pic_h_ctbs,
            self.width,
            self.height,
            self.ctb_log2,
            self.min_tb_log2,
            &self.tiles,
        )
        .map_err(|_| SequenceError::Malformed("invalid tile geometry"))
    }
}

/// Pre-read `slice_pic_parameter_set_id` from a slice-segment RBSP (the
/// two leading fields before it are fixed-width).
fn peek_slice_pps_id(rbsp: &[u8], nal_unit_type: u8) -> Result<u8, SequenceError> {
    let mut br = BitReader::new(rbsp);
    let _first = br.u1().map_err(|_| {
        SequenceError::Malformed("slice header truncated before first_slice_segment_in_pic_flag")
    })?;
    if (NalKind::BLA_W_LP..=NalKind::RSV_IRAP_VCL23).contains(&nal_unit_type) {
        let _ = br.u1().map_err(|_| {
            SequenceError::Malformed("slice header truncated at no_output_of_prior_pics_flag")
        })?;
    }
    let pps_id = br.ue().map_err(|_| {
        SequenceError::Malformed("slice header truncated at slice_pic_parameter_set_id")
    })?;
    if pps_id > 63 {
        return Err(SequenceError::Malformed(
            "slice_pic_parameter_set_id out of range",
        ));
    }
    Ok(pps_id as u8)
}

/// §7.4.8 — materialize the slice's short-term RPS (SPS-indexed or
/// slice-inline, explicit or inter-RPS-predicted).
fn materialize_slice_rps(
    header: &SliceSegmentHeader,
    sps: &SeqParameterSet,
) -> Result<MaterializedShortTermRefPicSet, SequenceError> {
    // IDR: no RPS block — empty set.
    let Some(sps_flag) = header.short_term_ref_pic_set_sps_flag else {
        return Ok(MaterializedShortTermRefPicSet {
            delta_poc_s0: Vec::new(),
            used_by_curr_pic_s0: Vec::new(),
            delta_poc_s1: Vec::new(),
            used_by_curr_pic_s1: Vec::new(),
        });
    };
    // Materialize the SPS chain (set i may inter-predict from an
    // earlier set).
    let mut chain: Vec<MaterializedShortTermRefPicSet> =
        Vec::with_capacity(sps.short_term_ref_pic_sets.len());
    for (idx, set) in sps.short_term_ref_pic_sets.iter().enumerate() {
        let source = if set.inter_ref_pic_set_prediction_flag {
            let ref_idx = idx
                .checked_sub(set.delta_idx_minus1 as usize + 1)
                .ok_or(SequenceError::Malformed("RefRpsIdx underflow"))?;
            Some(&chain[ref_idx])
        } else {
            None
        };
        chain.push(set.materialize(source)?);
    }
    if sps_flag {
        let idx = header.short_term_ref_pic_set_idx.unwrap_or(0) as usize;
        chain
            .into_iter()
            .nth(idx)
            .ok_or(SequenceError::Malformed("short_term_ref_pic_set_idx OOR"))
    } else {
        let set = header
            .inline_short_term_ref_pic_set
            .as_ref()
            .ok_or(SequenceError::Malformed("missing inline st_ref_pic_set"))?;
        let source = if set.inter_ref_pic_set_prediction_flag {
            // stRpsIdx == num_short_term_ref_pic_sets for the inline set.
            let ref_idx = chain
                .len()
                .checked_sub(set.delta_idx_minus1 as usize + 1)
                .ok_or(SequenceError::Malformed("RefRpsIdx underflow"))?;
            Some(&chain[ref_idx])
        } else {
            None
        };
        Ok(set.materialize(source)?)
    }
}

/// `NumPicTotalCurr` (§7.4.7.2) from the already-resolved picture
/// header info (single-layer, `pps_curr_pic_ref_enabled_flag == 0`).
fn num_pic_total_curr(info: &PictureHeaderInfo) -> u32 {
    let st = info
        .short_term_rps
        .used_by_curr_pic_s0
        .iter()
        .chain(info.short_term_rps.used_by_curr_pic_s1.iter())
        .filter(|&&u| u)
        .count();
    let lt = info
        .long_term
        .iter()
        .filter(|e| e.used_by_curr_pic_lt)
        .count();
    (st + lt) as u32
}

fn build_slice_ref_params(
    header: &SliceSegmentHeader,
    pps: &PicParameterSet,
    slice_type: SliceType,
    info: &PictureHeaderInfo,
) -> SliceRefParams {
    let is_b = slice_type == SliceType::B;
    let is_inter = slice_type != SliceType::I;
    SliceRefParams {
        is_inter,
        is_b,
        num_ref_idx_l0_active_minus1: u32::from(
            header
                .num_ref_idx_l0_active_minus1
                .unwrap_or(pps.num_ref_idx_l0_default_active_minus1),
        ),
        num_ref_idx_l1_active_minus1: u32::from(
            header
                .num_ref_idx_l1_active_minus1
                .unwrap_or(pps.num_ref_idx_l1_default_active_minus1),
        ),
        num_pic_total_curr: num_pic_total_curr(info),
        temporal_mvp_enabled: header.slice_temporal_mvp_enabled_flag,
        collocated_from_l0_flag: header.collocated_from_l0_flag.unwrap_or(true),
        collocated_ref_idx: header.collocated_ref_idx.unwrap_or(0),
    }
}

fn build_recon_params(
    header: &SliceSegmentHeader,
    sps: &SeqParameterSet,
    pps: &PicParameterSet,
    geom: &Geometry,
) -> Result<ReconParams, SequenceError> {
    let slice_qp_y = header
        .slice_qp_y(pps)
        .ok_or(SequenceError::Malformed("slice header without slice_qp"))?;
    let range = sps.sps_range_extension.as_ref();
    Ok(ReconParams {
        chroma_array_type: geom.chroma_array_type,
        bit_depth_luma: sps.bit_depth_luma_minus8 + 8,
        bit_depth_chroma: sps.bit_depth_chroma_minus8 + 8,
        intra_smoothing_disabled: range.is_some_and(|r| r.intra_smoothing_disabled_flag),
        strong_intra_smoothing_enabled: sps.strong_intra_smoothing_enabled_flag,
        slice_qp_y,
        cb_qp_offset: i32::from(pps.pps_cb_qp_offset) + i32::from(header.slice_cb_qp_offset),
        cr_qp_offset: i32::from(pps.pps_cr_qp_offset) + i32::from(header.slice_cr_qp_offset),
        transform_skip_rotation_enabled: range
            .is_some_and(|r| r.transform_skip_rotation_enabled_flag),
        extended_precision: range.is_some_and(|r| r.extended_precision_processing_flag),
    })
}

#[allow(clippy::too_many_arguments)]
fn build_inter_slice_context(
    header: &SliceSegmentHeader,
    sps: &SeqParameterSet,
    pps: &PicParameterSet,
    geom: &Geometry,
    recon: &ReconParams,
    curr_poc: i32,
    col_poc: i32,
    no_backward_pred: bool,
    slice_type: SliceType,
) -> InterSliceContext {
    let pps_range = pps.pps_range_extension.as_ref();
    let deblock = header.deblocking.as_ref();
    let _ = sps;
    InterSliceContext {
        curr_poc,
        slice_is_b: slice_type == SliceType::B,
        ctb_log2_size_y: geom.ctb_log2,
        pic_width_luma: geom.width,
        pic_height_luma: geom.height,
        max_num_merge_cand: usize::from(header.max_num_merge_cand().unwrap_or(5)),
        num_ref_idx_l0_active: i32::from(
            header
                .num_ref_idx_l0_active_minus1
                .unwrap_or(pps.num_ref_idx_l0_default_active_minus1),
        ) + 1,
        num_ref_idx_l1_active: i32::from(
            header
                .num_ref_idx_l1_active_minus1
                .unwrap_or(pps.num_ref_idx_l1_default_active_minus1),
        ) + 1,
        log2_par_mrg_level: pps.log2_parallel_merge_level_minus2 + 2,
        temporal_mvp_enabled: header.slice_temporal_mvp_enabled_flag,
        collocated_from_l0_flag: header.collocated_from_l0_flag.unwrap_or(true),
        col_poc,
        no_backward_pred,
        min_tb_log2_size_y: geom.min_tb_log2,
        log2_min_cu_qp_delta_size: geom.ctb_log2 - pps.diff_cu_qp_delta_depth,
        wpp_qp_row_reset: pps.entropy_coding_sync_enabled_flag,
        filter_across_slices: header
            .slice_loop_filter_across_slices_enabled_flag
            .unwrap_or(pps.pps_loop_filter_across_slices_enabled_flag),
        filter_across_tiles: pps.loop_filter_across_tiles_enabled_flag,
        deblock_enabled: deblock.map_or(true, |d| !d.disabled_flag),
        beta_offset_div2: deblock.map_or(0, |d| i32::from(d.beta_offset_div2)),
        tc_offset_div2: deblock.map_or(0, |d| i32::from(d.tc_offset_div2)),
        slice_qp_y: recon.slice_qp_y,
        cb_qp_offset: recon.cb_qp_offset,
        cr_qp_offset: recon.cr_qp_offset,
        slice_sao_luma_flag: header.slice_sao_luma_flag,
        slice_sao_chroma_flag: header.slice_sao_chroma_flag,
        log2_sao_offset_scale_luma: pps_range.map_or(0, |r| r.log2_sao_offset_scale_luma as u8),
        log2_sao_offset_scale_chroma: pps_range.map_or(0, |r| r.log2_sao_offset_scale_chroma as u8),
    }
}

/// §7.4.3 — derive the [`SliceDataParams`] for one slice segment.
fn build_slice_data_params(
    header: &SliceSegmentHeader,
    sps: &SeqParameterSet,
    pps: &PicParameterSet,
    geom: &Geometry,
    slice_type: SliceType,
) -> SliceDataParams {
    let pps_range = pps.pps_range_extension.as_ref();
    let (log2_min_ipcm, log2_max_ipcm) = sps.pcm.as_ref().map_or((3, 5), |p| {
        let min = u32::from(p.log2_min_pcm_luma_coding_block_size_minus3) + 3;
        (
            min,
            min + u32::from(p.log2_diff_max_min_pcm_luma_coding_block_size),
        )
    });
    let cu_chroma_qp_offset_enabled = header.cu_chroma_qp_offset_enabled_flag;
    let log2_min_cu_chroma_qp_offset_size =
        geom.ctb_log2 - pps_range.map_or(0, |r| r.diff_cu_chroma_qp_offset_depth);
    SliceDataParams {
        ctb_log2_size_y: geom.ctb_log2,
        min_cb_log2_size_y: geom.min_cb_log2,
        max_tb_log2_size_y: geom.max_tb_log2,
        min_tb_log2_size_y: geom.min_tb_log2,
        pic_width_in_luma_samples: geom.width,
        pic_height_in_luma_samples: geom.height,
        chroma_array_type: geom.chroma_array_type,
        bit_depth_luma: u32::from(sps.bit_depth_luma_minus8) + 8,
        bit_depth_chroma: u32::from(sps.bit_depth_chroma_minus8) + 8,
        slice_type_is_i: slice_type == SliceType::I,
        slice_type_is_b: slice_type == SliceType::B,
        slice_sao_luma_flag: header.slice_sao_luma_flag,
        slice_sao_chroma_flag: header.slice_sao_chroma_flag,
        transquant_bypass_enabled_flag: pps.transquant_bypass_enabled_flag,
        cu_qp_delta_enabled_flag: pps.cu_qp_delta_enabled_flag,
        log2_min_cu_qp_delta_size: geom.ctb_log2 - pps.diff_cu_qp_delta_depth,
        cu_chroma_qp_offset_enabled_flag: cu_chroma_qp_offset_enabled,
        log2_min_cu_chroma_qp_offset_size,
        chroma_qp_offset_list_len_minus1: pps_range
            .map_or(0, |r| r.chroma_qp_offset_list_len_minus1),
        amp_enabled_flag: sps.amp_enabled_flag,
        pcm_enabled_flag: sps.pcm_enabled_flag,
        log2_min_ipcm_cb_size_y: log2_min_ipcm,
        log2_max_ipcm_cb_size_y: log2_max_ipcm,
        max_transform_hierarchy_depth_intra: u32::from(sps.max_transform_hierarchy_depth_intra),
        max_transform_hierarchy_depth_inter: u32::from(sps.max_transform_hierarchy_depth_inter),
        max_num_merge_cand: u32::from(header.max_num_merge_cand().unwrap_or(5)),
        num_ref_idx_l0_active_minus1: u32::from(
            header
                .num_ref_idx_l0_active_minus1
                .unwrap_or(pps.num_ref_idx_l0_default_active_minus1),
        ),
        num_ref_idx_l1_active_minus1: u32::from(
            header
                .num_ref_idx_l1_active_minus1
                .unwrap_or(pps.num_ref_idx_l1_default_active_minus1),
        ),
        mvd_l1_zero_flag: header.mvd_l1_zero_flag.unwrap_or(false),
        sign_data_hiding_enabled_flag: pps.sign_data_hiding_enabled_flag,
        cross_component_prediction_enabled_flag: pps_range
            .is_some_and(|r| r.cross_component_prediction_enabled_flag),
        residual_adaptive_colour_transform_enabled_flag: false,
    }
}

/// §7.3.8.1 — CABAC-decode one slice segment's `slice_segment_data()`,
/// appending its CTUs (in tile-scan order) to `decoded` and recording
/// each CTB's `SliceAddrRs` in `slice_addr_of`.
#[allow(clippy::too_many_arguments)]
fn decode_slice_segment_data(
    seg: &SegmentData,
    sps: &SeqParameterSet,
    pps: &PicParameterSet,
    geom: &Geometry,
    state: &mut PictureParseState,
    decoded: &mut Vec<(u32, u32, CodingTreeUnit)>,
    slice_addr_of: &mut [Option<u32>],
    tolerant: bool,
) -> Result<(), SequenceError> {
    let header = &seg.header;
    let slice_type = header
        .slice_type
        .ok_or(SequenceError::Malformed("independent slice without type"))?;
    let params = build_slice_data_params(header, sps, pps, geom, slice_type);
    let slice_qp_y = header
        .slice_qp_y(pps)
        .ok_or(SequenceError::Malformed("slice header without slice_qp"))?;

    let data_offset = header
        .byte_offset_to_slice_data
        .ok_or(SequenceError::Malformed("slice header without data offset"))?;
    if data_offset >= seg.rbsp.len() {
        return Err(SequenceError::Malformed("slice data offset out of range"));
    }

    // §7.4.7.1 — split `slice_segment_data( )` into its subsets. The
    // entry-point offsets count CODED bytes (emulation-prevention bytes
    // included), so map each escaped boundary onto the stripped RBSP.
    let substreams = split_substreams(
        &seg.escaped,
        seg.rbsp.len(),
        data_offset,
        header.entry_point_offsets.as_ref(),
    )?;

    // Table 9-4 initType: I => 0; P => cabac_init ? 2 : 1;
    // B => cabac_init ? 1 : 2 (crate::cabac::init_type on the raw
    // slice_type value).
    let raw_slice_type = match slice_type {
        SliceType::B => 0,
        SliceType::P => 1,
        SliceType::I => 2,
    };
    let it = init_type(raw_slice_type, header.cabac_init_flag.unwrap_or(false));

    let tiling = geom.tiling()?;
    let wpp = pps.entropy_coding_sync_enabled_flag;
    let slice_addr_rs = header.slice_segment_address;
    let mut ctb_addr_ts = tiling.ctb_addr_rs_to_ts(slice_addr_rs);
    let pic_size_in_ctbs = (geom.pic_w_ctbs * geom.pic_h_ctbs) as usize;

    let sub_range = |idx: usize| -> Result<&[u8], SequenceError> {
        let &(a, b) = substreams
            .get(idx)
            .ok_or(SequenceError::Malformed("more CTB rows than substreams"))?;
        seg.rbsp
            .get(a..b)
            .ok_or(SequenceError::Malformed("substream range out of RBSP"))
    };
    let mut sub_idx = 0usize;
    let mut engine = CabacEngine::new(BitReader::new(sub_range(0)?))
        .map_err(|_| SequenceError::Malformed("slice data too short for CABAC init"))?;
    let mut ctx = SliceContexts::init(it, slice_qp_y);
    // §9.3.2.4 / §9.3.2.5 — the WPP context storage (stored after the
    // second CTB of a row; synchronized at the next row's start when
    // the above-right CTB is available).
    let mut wpp_stored: Option<SliceContexts> = None;
    let mut first_ctu = true;
    // Set after the row-final CTU's end_of_subset_one_bit: the next CTU
    // starts a new substream.
    let mut advance_substream = false;

    loop {
        if (ctb_addr_ts as usize) >= pic_size_in_ctbs {
            return Err(SequenceError::Malformed(
                "slice segment runs past the last CTB of the picture",
            ));
        }
        let ctb_addr_rs = tiling.ctb_addr_ts_to_rs(ctb_addr_ts);
        let rx = ctb_addr_rs % geom.pic_w_ctbs;
        let ry = ctb_addr_rs / geom.pic_w_ctbs;
        let x_ctb = rx << geom.ctb_log2;
        let y_ctb = ry << geom.ctb_log2;
        slice_addr_of[ctb_addr_rs as usize] = Some(slice_addr_rs);

        // §9.3.1 / §9.3.2.1 — at the first CTB of a CTB row (WPP), start
        // the next substream and either synchronize the context
        // variables from the stored above-right state (§9.3.2.5) or
        // re-initialize (§9.3.2.2). The first CTU of the slice keeps the
        // §9.3.1-item-1 slice initialization done above.
        if wpp && !first_ctu && rx == 0 {
            if !advance_substream {
                return Err(SequenceError::Malformed(
                    "CTB row start without end_of_subset_one_bit",
                ));
            }
            sub_idx += 1;
            engine = CabacEngine::new(BitReader::new(sub_range(sub_idx)?))
                .map_err(|_| SequenceError::Malformed("substream too short for CABAC init"))?;
            // Spatial neighbour T = the CTB at ( x0 + CtbSizeY,
            // y0 − CtbSizeY ) (eq. 9-3), §6.4.1-gated.
            let t_avail = ry > 0 && geom.pic_w_ctbs > 1 && {
                let t_rs = (ry - 1) * geom.pic_w_ctbs + 1;
                slice_addr_of[t_rs as usize] == Some(slice_addr_rs)
                    && tiling.tile_id(tiling.ctb_addr_rs_to_ts(t_rs)) == tiling.tile_id(ctb_addr_ts)
            };
            ctx = match (&wpp_stored, t_avail) {
                (Some(stored), true) => stored.clone(),
                _ => SliceContexts::init(it, slice_qp_y),
            };
        }
        advance_substream = false;
        first_ctu = false;

        // §7.3.8.3 SAO merge-candidate availability: the left / above
        // CTB must exist, lie in the same slice segment sequence
        // (same SliceAddrRs) and the same tile.
        let tile_here = tiling.tile_id(ctb_addr_ts);
        let merge_left = rx > 0 && {
            let left_rs = ctb_addr_rs - 1;
            slice_addr_of[left_rs as usize] == Some(slice_addr_rs)
                && tiling.tile_id(tiling.ctb_addr_rs_to_ts(left_rs)) == tile_here
        };
        let merge_up = ry > 0 && {
            let up_rs = ctb_addr_rs - geom.pic_w_ctbs;
            slice_addr_of[up_rs as usize] == Some(slice_addr_rs)
                && tiling.tile_id(tiling.ctb_addr_rs_to_ts(up_rs)) == tile_here
        };

        let ctu = decode_coding_tree_unit_in_picture(
            &mut engine,
            &mut ctx,
            &params,
            state,
            x_ctb,
            y_ctb,
            slice_addr_rs,
            tile_here,
            merge_left,
            merge_up,
        )?;
        decoded.push((x_ctb, y_ctb, ctu));

        // §9.3.2.4 — store the context state after the SECOND CTB of a
        // row (CtbAddrInRs % PicWidthInCtbsY == 1).
        if wpp && rx == 1 {
            wpp_stored = Some(ctx.clone());
        }

        let eos = end_of_slice_segment_flag(&mut engine)
            .map_err(|_| SequenceError::Malformed("CABAC underrun at end_of_slice_segment"))?;
        ctb_addr_ts += 1;
        if eos {
            break;
        }
        if (ctb_addr_ts as usize) >= pic_size_in_ctbs {
            if tolerant {
                eprintln!("(tolerant: end_of_slice_segment_flag not set on the last CTB)");
                break;
            }
            return Err(SequenceError::Malformed(
                "end_of_slice_segment_flag not set on the last CTB",
            ));
        }
        // §7.3.8.1 — end_of_subset_one_bit + byte alignment after the
        // last CTB of a WPP row that does not end the slice; the next
        // CTU reads from the following substream.
        if wpp && rx == geom.pic_w_ctbs - 1 {
            let one = end_of_slice_segment_flag(&mut engine)
                .map_err(|_| SequenceError::Malformed("CABAC underrun at end_of_subset_one_bit"))?;
            if !one && !tolerant {
                return Err(SequenceError::Malformed("end_of_subset_one_bit not set"));
            }
            advance_substream = true;
        }
    }
    Ok(())
}

/// §7.4.7.1 — the stripped-RBSP byte ranges of the
/// `num_entry_point_offsets + 1` subsets of `slice_segment_data( )`.
///
/// The wire offsets count coded (escaped) bytes from the first byte of
/// the slice segment data, so walk the escaped payload with the
/// §7.4.1.1 emulation state machine and translate each boundary into
/// the stripped-RBSP index space.
fn split_substreams(
    escaped: &[u8],
    rbsp_len: usize,
    stripped_data_offset: usize,
    entry_points: Option<&crate::slice::EntryPointOffsets>,
) -> Result<Vec<(usize, usize)>, SequenceError> {
    let n_offsets = entry_points.map_or(0, |e| e.entry_point_offset_minus1.len());
    if n_offsets == 0 {
        return Ok(vec![(stripped_data_offset, rbsp_len)]);
    }
    // stripped index -> escaped index of the slice-data start.
    let mut stripped_of_escaped = vec![0usize; escaped.len() + 1];
    let mut zeros = 0u32;
    let mut stripped = 0usize;
    for (i, &b) in escaped.iter().enumerate() {
        stripped_of_escaped[i] = stripped;
        if zeros >= 2 && b == 0x03 {
            // Emulation-prevention byte: consumed, not emitted.
            zeros = 0;
            continue;
        }
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        stripped += 1;
    }
    stripped_of_escaped[escaped.len()] = stripped;
    // Escaped index of the slice-data start.
    let escaped_start = stripped_of_escaped
        .iter()
        .position(|&sidx| sidx == stripped_data_offset)
        .ok_or(SequenceError::Malformed("slice data offset unmappable"))?;

    let entry_points = entry_points.expect("checked above");
    let mut ranges = Vec::with_capacity(n_offsets + 1);
    let mut first_escaped = escaped_start;
    let mut first_stripped = stripped_data_offset;
    for &off_m1 in &entry_points.entry_point_offset_minus1 {
        let len = off_m1 as usize + 1;
        let last_escaped = first_escaped
            .checked_add(len)
            .filter(|&e| e <= escaped.len())
            .ok_or(SequenceError::Malformed("entry point past slice data"))?;
        let last_stripped = stripped_of_escaped[last_escaped];
        ranges.push((first_stripped, last_stripped));
        first_escaped = last_escaped;
        first_stripped = last_stripped;
    }
    ranges.push((first_stripped, rbsp_len));
    Ok(ranges)
}
