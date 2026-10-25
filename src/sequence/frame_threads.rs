//! Frame-level threading of the [`super::SequenceDecoder`].
//!
//! Under an [`oxideav_core::ExecutionContext`] budget above one, the
//! decoder keeps the §8.3 picture-order / RPS / DPB cycle on the
//! calling thread, in decode order, exactly as the serial decoder runs
//! it, and hands each picture's slice-segment data (§7.3.8 CABAC parse,
//! §8.4 / §8.5 reconstruction, §8.7 in-loop filters) to a worker
//! thread. The DPB holds a stand-in for every picture still being
//! decoded; a later picture's worker reads its samples and collocated
//! motion through the published CTB rows of
//! [`crate::progress::InFlightPicture`]. Each worker reconstructs its
//! picture in CTB raster order and runs the filters one CTB row behind
//! (row `r`'s deblocking, then row `r − 1`'s SAO), publishing every
//! row as it becomes final — so the samples any reader sees are the
//! samples the serial decode produces, and the output is bit-identical
//! for every budget.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::{
    Geometry, SegmentData, SequenceError, build_slice_data_params, decode_slice_segment_data,
};
use crate::ctx_init::SliceContexts;
use crate::dpb::{DpbEntry, RefPicLists};
use crate::inter_recon::{
    InterSliceContext, PictureReconstructor, PlacedInterCtu, RefListAccess, RowFilterState,
};
use crate::motion::MotionField;
use crate::picture::Picture;
use crate::pps::PicParameterSet;
use crate::progress::InFlightPicture;
use crate::recon::ReconParams;
use crate::slice_data::{CodingTreeUnit, PictureParseState};
use crate::sps::SeqParameterSet;

/// Everything one picture's worker needs, owned.
pub(super) struct PictureJob {
    pub(super) segs: Vec<SegmentData>,
    pub(super) sps: SeqParameterSet,
    pub(super) pps: PicParameterSet,
    pub(super) geom: Geometry,
    pub(super) recon_params: ReconParams,
    pub(super) slice_ctx: InterSliceContext,
    pub(super) lists: RefPicLists,
    /// A snapshot of the DPB (stand-ins for the pictures still in
    /// flight) the reference lists index.
    pub(super) entries: Vec<DpbEntry>,
    pub(super) col_pic: Option<usize>,
    pub(super) across_of_slice: BTreeMap<u32, bool>,
    pub(super) progress: Arc<InFlightPicture>,
}

/// A worker's result: the picture (partial when decoding failed) and
/// its motion field.
#[derive(Debug)]
pub(super) struct JobOutput {
    pub(super) result: Result<(), SequenceError>,
    pub(super) picture: Picture,
    pub(super) motion: MotionField,
}

/// Publishes whatever is left when the worker ends without publishing
/// every row (an error, or a panic unwinding), so no reader waits
/// forever.
struct AbandonGuard(Arc<InFlightPicture>);

impl Drop for AbandonGuard {
    fn drop(&mut self) {
        if self.0.published_lines() < self.0.height() {
            self.0.abandon();
        }
    }
}

/// Decode one picture's slice data (the worker body).
pub(super) fn run(job: PictureJob) -> JobOutput {
    let PictureJob {
        segs,
        sps,
        pps,
        geom,
        recon_params,
        slice_ctx,
        lists,
        entries,
        col_pic,
        across_of_slice,
        progress,
    } = job;
    let _guard = AbandonGuard(progress.clone());
    let refs = RefListAccess {
        lists: &lists,
        entries: &entries,
    };
    let col_field = col_pic.map(|idx| &*entries[idx].motion);
    let blank = || {
        (
            Picture::new(
                geom.width as usize,
                geom.height as usize,
                geom.chroma_array_type,
                sps.bit_depth_luma(),
                sps.bit_depth_chroma(),
            ),
            MotionField::new(geom.width as usize, geom.height as usize),
        )
    };
    let mut reconstructor = match PictureReconstructor::new(
        geom.width as usize,
        geom.height as usize,
        &recon_params,
        &slice_ctx,
        &geom.tiles,
        &refs,
        col_field,
    ) {
        Ok(r) => r,
        Err(e) => {
            let (picture, motion) = blank();
            return JobOutput {
                result: Err(e.into()),
                picture,
                motion,
            };
        }
    };
    // §8.5.3.1: a picture that references itself (intra block copy)
    // predicts from its samples before in-loop filtering — its filters
    // run once every CTU is reconstructed.
    let curr_ref = lists.list0.contains(&Some(crate::dpb::CURR_PIC))
        || lists
            .list1
            .as_ref()
            .is_some_and(|l| l.contains(&Some(crate::dpb::CURR_PIC)));
    let result = decode_rows(
        &segs,
        &sps,
        &pps,
        &geom,
        &across_of_slice,
        &mut reconstructor,
        (!curr_ref).then_some(&*progress),
    );
    let (picture, motion) = reconstructor.into_parts();
    if result.is_ok() {
        progress.publish_motion(usize::MAX, &motion);
        progress.publish_lines(progress.height(), &picture);
    }
    JobOutput {
        result,
        picture,
        motion,
    }
}

/// The serial per-picture walk of [`super::SequenceDecoder`]'s
/// `decode_picture_serial` with the §8.7 filters run row by row: with
/// `publish`, CTB row `r`'s deblocking (and row `r − 1`'s SAO) runs as
/// soon as row `r` is reconstructed — its unfiltered last line, which
/// row `r + 1`'s intra prediction reads, is kept aside first — and
/// the final rows are published as they complete. Without, the
/// filters run row by row after the last CTU.
fn decode_rows(
    segs: &[SegmentData],
    sps: &SeqParameterSet,
    pps: &PicParameterSet,
    geom: &Geometry,
    across_of_slice: &BTreeMap<u32, bool>,
    reconstructor: &mut PictureReconstructor<'_>,
    publish: Option<&InFlightPicture>,
) -> Result<(), SequenceError> {
    let (w_ctbs, h_ctbs) = (geom.pic_w_ctbs as usize, geom.pic_h_ctbs as usize);
    let mut filter = RowFilterState::default();
    let mut ctus_in_row = vec![0usize; h_ctbs];
    // CTB rows fully reconstructed / deblocked so far.
    let mut complete = 0usize;
    let mut filtered = 0usize;
    let mut slice_addr_of: Vec<Option<u32>> = vec![None; w_ctbs * h_ctbs];
    {
        let mut sink = |x_ctb: u32, y_ctb: u32, ctu: CodingTreeUnit, slice_addr_rs: u32| {
            let placed = PlacedInterCtu {
                x_ctb,
                y_ctb,
                slice_addr_rs,
                filter_across_slices: across_of_slice
                    .get(&slice_addr_rs)
                    .copied()
                    .unwrap_or(pps.pps_loop_filter_across_slices_enabled_flag),
                ctu: &ctu,
            };
            reconstructor.push_ctu(&placed)?;
            let ry = (y_ctb >> geom.ctb_log2) as usize;
            if let Some(n) = ctus_in_row.get_mut(ry) {
                *n += 1;
            }
            if let Some(progress) = publish {
                // Row `k` complete: its motion is final; keep its
                // unfiltered last line for row `k + 1`'s intra
                // prediction, then deblock it and SAO what that makes
                // final.
                while complete < h_ctbs && ctus_in_row[complete] == w_ctbs {
                    progress.publish_motion(complete + 1, reconstructor.field());
                    if complete + 1 < h_ctbs {
                        reconstructor.save_unfiltered_last_line(complete);
                    }
                    let lines = reconstructor.filter_ctb_row(complete, &mut filter);
                    complete += 1;
                    filtered = complete;
                    progress.publish_lines(lines, reconstructor.picture());
                }
            }
            Ok(())
        };
        let first_slice_type = segs[0]
            .header
            .slice_type
            .ok_or(SequenceError::Malformed("independent slice without type"))?;
        let mut parse_state = PictureParseState::new(&build_slice_data_params(
            &segs[0].header,
            sps,
            pps,
            geom,
            first_slice_type,
        ));
        let mut cur_indep: &SegmentData = &segs[0];
        let mut ds_stored: Option<SliceContexts> = None;
        let mut wpp_stored: Option<SliceContexts> = None;
        for seg in segs {
            if seg.header.dependent_slice_segment_flag {
                if ds_stored.is_none() {
                    return Err(SequenceError::Malformed(
                        "dependent slice segment without a preceding segment's context state",
                    ));
                }
            } else {
                cur_indep = seg;
            }
            decode_slice_segment_data(
                seg,
                &cur_indep.header,
                sps,
                pps,
                geom,
                &mut parse_state,
                &mut sink,
                &mut slice_addr_of,
                &mut ds_stored,
                &mut wpp_stored,
                false,
            )?;
        }
    }
    // The rows not filtered yet (all of them without `publish`, or a
    // picture whose CTUs do not cover every row), then the last SAO.
    while filtered < h_ctbs {
        reconstructor.filter_ctb_row(filtered, &mut filter);
        filtered += 1;
    }
    Ok(())
}
