//! Wavefront-parallel decoding of one picture under an
//! [`oxideav_core::ExecutionContext`] thread budget.
//!
//! A picture coded with `entropy_coding_sync_enabled_flag` (and no
//! tiles) carries one CABAC substream per CTB row, each row's contexts
//! synchronized from the row above after its second CTB (§9.3.2.4 /
//! §9.3.2.5). Every other dependency of a CTB on its neighbours — the
//! §8.4.4.2 reference samples, the §8.5.3.2 motion candidates, the
//! §9.3.4.2.2 context increments, the §6.4.1 availability — reaches at
//! most the CTB above-right. So the CTB at column `c` of row `r` may
//! start once row `r − 1` has finished column `c + 1`, and the rows run
//! on the workers in a wavefront.
//!
//! Each row is parsed **and** reconstructed by one worker into its own
//! band structures ([`PictureReconstructor::new_band`] and a band
//! [`PictureParseState`]): a band stores the CTB row plus a four-line
//! halo above it. What a CTB hands to the row below — the samples of
//! its bottom line, the cells of its bottom cell row, its slice
//! identity and SAO parameters, the context storage — is published as a
//! [`CtbHalo`] in a per-CTB slot; the row below imports a slot once the
//! progress counter of the row above says it is complete. Nothing is
//! shared mutably between workers: the finished rows are copied into
//! disjoint row chunks of the whole-picture structures, and the §8.7
//! in-loop filters then run on the merged picture. The bytes equal the
//! serial decode for any worker count.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use super::{build_slice_data_params, split_substreams, Geometry, SegmentData, SequenceError};
use crate::bitreader::BitReader;
use crate::cabac::{init_type, CabacEngine};
use crate::ctx_init::SliceContexts;
use crate::deblock::SamplePlane;
use crate::inter_recon::{
    BandOutput, CtbHalo, InterSliceContext, PictureReconstructor, PlacedInterCtu, RefListAccess,
};
use crate::motion::MotionField;
use crate::picture::{Picture, Plane};
use crate::pps::PicParameterSet;
use crate::recon::ReconParams;
use crate::slice::SliceType;
use crate::slice_data::{
    decode_coding_tree_unit_in_picture, end_of_slice_segment_flag, PictureParseState,
};
use crate::sps::SeqParameterSet;

/// One run of CTBs of a row that one slice segment's substream codes.
#[derive(Debug, Clone, Copy)]
struct RowPart {
    /// Index into the picture's segments.
    seg: usize,
    /// Substream index within that segment.
    sub: usize,
    /// First CTB raster address (inclusive).
    start: u32,
    /// Last CTB raster address (exclusive).
    end: u32,
}

/// The per-picture plan: which segment / substream codes each CTB run.
pub(super) struct WavefrontPlan {
    /// Per CTB row, the parts in raster order.
    rows: Vec<Vec<RowPart>>,
    /// Per segment: `(first CTB, one past the last CTB)`.
    seg_range: Vec<(u32, u32)>,
    /// Per segment: the stripped-RBSP byte ranges of its substreams.
    substreams: Vec<Vec<(usize, usize)>>,
    /// Per segment: the index of the independent segment whose header
    /// governs it.
    indep_of: Vec<usize>,
}

impl WavefrontPlan {
    /// Lay the segments out over the CTB rows. `None` when the picture
    /// is not a wavefront candidate: tiles, no `entropy_coding_sync`,
    /// a single CTB row, SCC / chroma-QP-offset-list tools whose
    /// decode-order state crosses rows, or a segment layout that is
    /// not a monotone partition of the picture.
    pub(super) fn build(
        segs: &[SegmentData],
        sps: &SeqParameterSet,
        pps: &PicParameterSet,
        geom: &Geometry,
    ) -> Option<Self> {
        if !pps.entropy_coding_sync_enabled_flag
            || pps.tiles_enabled_flag
            || geom.pic_h_ctbs < 2
            || pps
                .pps_range_extension
                .as_ref()
                .is_some_and(|e| e.chroma_qp_offset_list_enabled_flag)
            || pps
                .pps_scc_extension
                .as_ref()
                .is_some_and(|e| e.pps_curr_pic_ref_enabled_flag)
            || sps
                .sps_scc_extension
                .as_ref()
                .is_some_and(|e| e.palette_mode_enabled_flag)
        {
            return None;
        }
        let w = geom.pic_w_ctbs;
        let pic_size = w * geom.pic_h_ctbs;
        let mut seg_range = Vec::with_capacity(segs.len());
        let mut indep_of = Vec::with_capacity(segs.len());
        let mut substreams = Vec::with_capacity(segs.len());
        let mut last_indep = None;
        for (k, seg) in segs.iter().enumerate() {
            let start = seg.header.slice_segment_address;
            let end = segs
                .get(k + 1)
                .map_or(pic_size, |n| n.header.slice_segment_address);
            if (k == 0 && start != 0) || end <= start || end > pic_size {
                return None;
            }
            if seg.header.dependent_slice_segment_flag {
                indep_of.push(last_indep?);
            } else {
                last_indep = Some(k);
                indep_of.push(k);
            }
            seg_range.push((start, end));
            let data_offset = seg.header.byte_offset_to_slice_data?;
            if data_offset >= seg.rbsp.len() {
                return None;
            }
            let subs = split_substreams(
                &seg.escaped,
                seg.rbsp.len(),
                data_offset,
                seg.header.entry_point_offsets.as_ref(),
            )
            .ok()?;
            // One substream per CTB row the segment touches.
            let rows_touched = (end - 1) / w - start / w + 1;
            if subs.len() != rows_touched as usize {
                return None;
            }
            substreams.push(subs);
        }
        let mut rows = vec![Vec::new(); geom.pic_h_ctbs as usize];
        for (k, &(start, end)) in seg_range.iter().enumerate() {
            let r0 = start / w;
            let r1 = (end - 1) / w;
            for r in r0..=r1 {
                rows[r as usize].push(RowPart {
                    seg: k,
                    sub: (r - r0) as usize,
                    start: start.max(r * w),
                    end: end.min((r + 1) * w),
                });
            }
        }
        Some(Self {
            rows,
            seg_range,
            substreams,
            indep_of,
        })
    }
}

/// Everything a row worker reads (shared, immutable).
pub(super) struct WavefrontInputs<'a> {
    pub segs: &'a [SegmentData],
    pub sps: &'a SeqParameterSet,
    pub pps: &'a PicParameterSet,
    pub geom: &'a Geometry,
    pub slice_ctx: &'a InterSliceContext,
    pub refs: &'a RefListAccess<'a>,
    pub col_field: Option<&'a MotionField>,
    /// Per independent slice address, its loop-filter-across flag.
    pub across_of_slice: &'a std::collections::BTreeMap<u32, bool>,
    pub tolerant: bool,
}

/// Row progress: per row, the number of CTBs published.
struct Progress {
    done: Mutex<Vec<usize>>,
    changed: Condvar,
    abort: AtomicBool,
}

impl Progress {
    /// Wait until row `r` has published at least `n` CTBs (or the
    /// picture was aborted — `false`).
    fn wait(&self, r: usize, n: usize) -> bool {
        let mut g = self.done.lock().unwrap_or_else(|e| e.into_inner());
        while g[r] < n {
            if self.abort.load(Ordering::Acquire) {
                return false;
            }
            g = self.changed.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        true
    }

    fn publish(&self, r: usize) {
        let mut g = self.done.lock().unwrap_or_else(|e| e.into_inner());
        g[r] += 1;
        drop(g);
        self.changed.notify_all();
    }

    fn fail(&self) {
        self.abort.store(true, Ordering::Release);
        let _g = self.done.lock().unwrap_or_else(|e| e.into_inner());
        self.changed.notify_all();
    }
}

/// The whole-picture row chunks one finished band is copied into.
struct RowSink<'a> {
    luma: &'a mut [u16],
    cb: &'a mut [u16],
    cr: &'a mut [u16],
    qp: &'a mut [i8],
    no_filter: &'a mut [bool],
    motion_flags: &'a mut [u8],
    motion: Option<&'a mut [crate::motion::CellMotion]>,
    sao: &'a mut [crate::sao::ResolvedSao],
    slice_addr: &'a mut [u32],
    filter_across: &'a mut [bool],
}

/// The merged whole-picture structures the §8.7 filters consume.
pub(super) struct MergedPicture {
    pub pic: Picture,
    pub field: MotionField,
    pub qp_cells: Vec<i8>,
    pub no_filter: Vec<bool>,
    pub edges: crate::deblock::DeblockEdgeMap,
    pub sao: Vec<crate::sao::ResolvedSao>,
    pub slice_addr: Vec<u32>,
    pub filter_across: Vec<bool>,
}

/// Decode the picture's CTB rows in a wavefront on up to `workers`
/// threads and merge them (pre-filter). Bytes equal the serial decode.
pub(super) fn decode_rows(
    plan: &WavefrontPlan,
    inputs: &WavefrontInputs<'_>,
    params: &ReconParams,
    workers: usize,
) -> Result<MergedPicture, SequenceError> {
    let geom = inputs.geom;
    let (width, height) = (geom.width as usize, geom.height as usize);
    let w_ctbs = geom.pic_w_ctbs as usize;
    let h_ctbs = geom.pic_h_ctbs as usize;
    let ctb = 1usize << geom.ctb_log2;
    let w4 = width.div_ceil(4);
    let with_motion = inputs
        .segs
        .iter()
        .any(|s| !matches!(s.header.slice_type, Some(SliceType::I)));

    // Whole-picture structures, split into per-row chunks up front so
    // every worker writes its own rows only.
    let mut pic = Picture::new(
        width,
        height,
        params.chroma_array_type,
        params.bit_depth_luma,
        params.bit_depth_chroma,
    );
    let mut field = MotionField::new(width, height);
    let mut qp_cells = vec![0i8; w4 * height.div_ceil(4)];
    let mut no_filter = vec![false; w4 * height.div_ceil(4)];
    let edges = Mutex::new(crate::deblock::DeblockEdgeMap::new(width, height));
    let mut sao = vec![crate::sao::ResolvedSao::off(); w_ctbs * h_ctbs];
    let mut slice_addr = vec![0u32; w_ctbs * h_ctbs];
    let mut filter_across = vec![true; w_ctbs * h_ctbs];
    let (sub_w, sub_h) = if params.chroma_array_type == 0 {
        (1, 1)
    } else {
        crate::picture::sub_wh_c(params.chroma_array_type)
    };
    let (cw, _) = pic.plane_dims(Plane::Cb);

    let error: Mutex<Option<SequenceError>> = Mutex::new(None);
    let progress = Progress {
        done: Mutex::new(vec![0usize; h_ctbs]),
        changed: Condvar::new(),
        abort: AtomicBool::new(false),
    };
    let halos: Vec<OnceLock<CtbHalo>> = (0..w_ctbs * h_ctbs).map(|_| OnceLock::new()).collect();
    let next_row = AtomicUsize::new(0);
    {
        let (luma, cbp, crp) = pic.planes_mut();
        let (mflags, mmotion) = field.storage_mut(with_motion);
        let mut sinks: Vec<Option<RowSink<'_>>> = Vec::with_capacity(h_ctbs);
        let mut luma_rest = luma;
        let mut cb_rest = cbp;
        let mut cr_rest = crp;
        let mut qp_rest = qp_cells.as_mut_slice();
        let mut nf_rest = no_filter.as_mut_slice();
        let mut mf_rest = mflags;
        let mut mm_rest = mmotion;
        let mut sao_rest = sao.as_mut_slice();
        let mut sa_rest = slice_addr.as_mut_slice();
        let mut fa_rest = filter_across.as_mut_slice();
        for r in 0..h_ctbs {
            let y0 = r * ctb;
            let rows = ctb.min(height - y0);
            let crows = rows.div_ceil(sub_h);
            let cells = rows.div_ceil(4);
            let (l, lr) = std::mem::take(&mut luma_rest).split_at_mut(rows * width);
            luma_rest = lr;
            let (c1, c1r) = std::mem::take(&mut cb_rest).split_at_mut(crows * cw);
            cb_rest = c1r;
            let (c2, c2r) = std::mem::take(&mut cr_rest).split_at_mut(crows * cw);
            cr_rest = c2r;
            let (q, qr) = std::mem::take(&mut qp_rest).split_at_mut(cells * w4);
            qp_rest = qr;
            let (n, nr) = std::mem::take(&mut nf_rest).split_at_mut(cells * w4);
            nf_rest = nr;
            let (mf, mfr) = std::mem::take(&mut mf_rest).split_at_mut(cells * w4);
            mf_rest = mfr;
            let mm = mm_rest.take().map(|m| {
                let (a, b) = m.split_at_mut(cells * w4);
                mm_rest = Some(b);
                a
            });
            let (s, sr) = std::mem::take(&mut sao_rest).split_at_mut(w_ctbs);
            sao_rest = sr;
            let (sa, sar) = std::mem::take(&mut sa_rest).split_at_mut(w_ctbs);
            sa_rest = sar;
            let (fa, far) = std::mem::take(&mut fa_rest).split_at_mut(w_ctbs);
            fa_rest = far;
            sinks.push(Some(RowSink {
                luma: l,
                cb: c1,
                cr: c2,
                qp: q,
                no_filter: n,
                motion_flags: mf,
                motion: mm,
                sao: s,
                slice_addr: sa,
                filter_across: fa,
            }));
        }
        let sinks = Mutex::new(sinks);
        let _ = sub_w;

        std::thread::scope(|scope| {
            for _ in 0..workers.min(h_ctbs) {
                // Each worker carries its own copy of the per-slice
                // reconstruction parameters (their chroma QP offset
                // state is a per-slice `Cell`, which cannot be shared).
                let my_params = params.clone();
                let (sinks, halos, progress, error, edges, next_row) =
                    (&sinks, &halos, &progress, &error, &edges, &next_row);
                scope.spawn(move || {
                    // A panicking worker must still release the rows
                    // waiting on it (the scope re-raises the panic).
                    struct Unblock<'p>(&'p Progress);
                    impl Drop for Unblock<'_> {
                        fn drop(&mut self) {
                            if std::thread::panicking() {
                                self.0.fail();
                            }
                        }
                    }
                    let _unblock = Unblock(progress);
                    loop {
                        let r = next_row.fetch_add(1, Ordering::AcqRel);
                        if r >= h_ctbs || progress.abort.load(Ordering::Acquire) {
                            break;
                        }
                        let sink = sinks.lock().unwrap_or_else(|e| e.into_inner())[r]
                            .take()
                            .expect("row sink taken once");
                        match decode_row(plan, inputs, &my_params, r, halos, progress) {
                            Ok(band) => {
                                merge_band(band, sink, edges, with_motion);
                            }
                            Err(e) => {
                                let mut slot = error.lock().unwrap_or_else(|e| e.into_inner());
                                if slot.is_none() {
                                    *slot = Some(e);
                                }
                                progress.fail();
                                break;
                            }
                        }
                    }
                });
            }
        });
    }
    if let Some(e) = error.into_inner().unwrap_or_else(|e| e.into_inner()) {
        return Err(e);
    }
    Ok(MergedPicture {
        pic,
        field,
        qp_cells,
        no_filter,
        edges: edges.into_inner().unwrap_or_else(|e| e.into_inner()),
        sao,
        slice_addr,
        filter_across,
    })
}

/// Copy a finished band into its whole-picture row chunks.
fn merge_band(
    band: BandOutput,
    sink: RowSink<'_>,
    edges: &Mutex<crate::deblock::DeblockEdgeMap>,
    with_motion: bool,
) {
    let BandOutput {
        y0,
        rows,
        pic,
        field,
        qp_cells,
        no_filter,
        edges: band_edges,
        sao,
        slice_addr,
        filter_across,
    } = band;
    let (pw, ph) = pic.plane_dims(Plane::Luma);
    let rows = rows.min(ph - y0);
    for y in 0..rows {
        sink.luma[y * pw..(y + 1) * pw].copy_from_slice(pic.row(Plane::Luma, y0 + y));
    }
    if pic.chroma_array_type() != 0 {
        let (cw, ch) = pic.plane_dims(Plane::Cb);
        let (_, sh) = crate::picture::sub_wh_c(pic.chroma_array_type());
        let cy0 = y0 / sh;
        let crows = rows.div_ceil(sh).min(ch - cy0);
        for y in 0..crows {
            sink.cb[y * cw..(y + 1) * cw].copy_from_slice(pic.row(Plane::Cb, cy0 + y));
            sink.cr[y * cw..(y + 1) * cw].copy_from_slice(pic.row(Plane::Cr, cy0 + y));
        }
    }
    sink.qp.copy_from_slice(&qp_cells[..sink.qp.len()]);
    sink.no_filter
        .copy_from_slice(&no_filter[..sink.no_filter.len()]);
    let by0 = y0 / 4;
    let by1 = by0 + rows.div_ceil(4);
    field.export_rows(
        by0,
        by1,
        sink.motion_flags,
        if with_motion { sink.motion } else { None },
    );
    sink.sao.copy_from_slice(&sao);
    sink.slice_addr.copy_from_slice(&slice_addr);
    sink.filter_across.copy_from_slice(&filter_across);
    edges
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .merge_band(&band_edges);
}

/// Parse and reconstruct CTB row `r` into a band.
fn decode_row(
    plan: &WavefrontPlan,
    inputs: &WavefrontInputs<'_>,
    params: &ReconParams,
    r: usize,
    halos: &[OnceLock<CtbHalo>],
    progress: &Progress,
) -> Result<BandOutput, SequenceError> {
    let geom = inputs.geom;
    let (width, height) = (geom.width as usize, geom.height as usize);
    let w = geom.pic_w_ctbs as usize;
    let ctb = 1usize << geom.ctb_log2;
    let y0 = r * ctb;
    let rows = ctb.min(height - y0);
    let halo = if r == 0 {
        0
    } else {
        PictureReconstructor::BAND_HALO
    };
    let pps = inputs.pps;
    let sps = inputs.sps;
    let segs = inputs.segs;

    let mut recon = PictureReconstructor::new_band(
        width,
        height,
        params,
        inputs.slice_ctx,
        &geom.tiles,
        inputs.refs,
        inputs.col_field,
        y0,
        rows,
    )?;
    // The parse state's geometry comes from any slice of the picture.
    let first_part = plan.rows[r]
        .first()
        .ok_or(SequenceError::Malformed("CTB row without slice data"))?;
    let first_header = &segs[plan.indep_of[first_part.seg]].header;
    let first_type = first_header
        .slice_type
        .ok_or(SequenceError::Malformed("independent slice without type"))?;
    let mut parse_state = PictureParseState::new_band(
        &build_slice_data_params(first_header, sps, pps, geom, first_type),
        y0 - halo,
        halo + rows,
    );
    // The worker's view of SliceAddrRs per CTB (own row + imported).
    let mut slice_addr_of: Vec<Option<u32>> = vec![None; w * geom.pic_h_ctbs as usize];
    let mut imported = 0usize; // columns of the row above imported so far
    let mut import_above = |upto: usize,
                            recon: &mut PictureReconstructor<'_>,
                            parse_state: &mut PictureParseState,
                            slice_addr_of: &mut Vec<Option<u32>>|
     -> Result<(), SequenceError> {
        if r == 0 {
            return Ok(());
        }
        let upto = upto.min(w);
        if upto > imported {
            if !progress.wait(r - 1, upto) {
                return Err(SequenceError::Malformed("wavefront aborted"));
            }
            for c in imported..upto {
                let rs = (r - 1) * w + c;
                let h = halos[rs].get().expect("published halo");
                recon.import_halo(h);
                parse_state.note_ctu(rs as u32, h.slice_addr_rs, 0);
                let by = y0 / 4 - 1;
                parse_state.import_row_cells(
                    by,
                    c * (ctb / 4),
                    &h.parse_cells.0,
                    &h.parse_cells.1,
                    &h.parse_cells.2,
                );
                slice_addr_of[rs] = Some(h.slice_addr_rs);
            }
            imported = upto;
        }
        Ok(())
    };

    let num_comps = if geom.chroma_array_type == 0 { 1 } else { 3 };
    let base_palette_predictor = pps
        .pps_scc_extension
        .as_ref()
        .filter(|e| e.pps_palette_predictor_initializers_present_flag)
        .map(|e| {
            crate::palette::PalettePredictor::from_initializers(
                &e.pps_palette_predictor_initializer,
                num_comps,
            )
        })
        .or_else(|| {
            sps.sps_scc_extension
                .as_ref()
                .filter(|e| e.sps_palette_predictor_initializers_present_flag)
                .map(|e| {
                    crate::palette::PalettePredictor::from_initializers(
                        &e.sps_palette_predictor_initializer,
                        num_comps,
                    )
                })
        })
        .unwrap_or_default();

    let mut ds_stored: Option<SliceContexts> = None;
    let parts = plan.rows[r].clone();
    for part in parts {
        let seg = &segs[part.seg];
        let header = &segs[plan.indep_of[part.seg]].header;
        let slice_type = header
            .slice_type
            .ok_or(SequenceError::Malformed("independent slice without type"))?;
        let sd_params = build_slice_data_params(header, sps, pps, geom, slice_type);
        let slice_qp_y = header
            .slice_qp_y(pps)
            .ok_or(SequenceError::Malformed("slice header without slice_qp"))?;
        let raw_slice_type = match slice_type {
            SliceType::B => 0,
            SliceType::P => 1,
            SliceType::I => 2,
        };
        let it = init_type(raw_slice_type, header.cabac_init_flag.unwrap_or(false));
        let fresh_contexts = || {
            let mut c = SliceContexts::init(it, slice_qp_y);
            c.palette_predictor = base_palette_predictor.clone();
            c
        };
        let slice_addr_rs = header.slice_segment_address;
        let filter_across_slices = inputs
            .across_of_slice
            .get(&slice_addr_rs)
            .copied()
            .unwrap_or(pps.pps_loop_filter_across_slices_enabled_flag);
        let &(a, b) = plan.substreams[part.seg]
            .get(part.sub)
            .ok_or(SequenceError::Malformed("more CTB rows than substreams"))?;
        let bytes = seg
            .rbsp
            .get(a..b)
            .ok_or(SequenceError::Malformed("substream range out of RBSP"))?;
        let mut engine = CabacEngine::new(BitReader::new(bytes))
            .map_err(|_| SequenceError::Malformed("substream too short for CABAC init"))?;

        // §9.3.2.1 / §9.3.2.5 — the part's initial context state.
        let seg_start = part.start == plan.seg_range[part.seg].0;
        let rx0 = part.start as usize % w;
        // The first CTB of the row needs columns 0 and 1 of the row
        // above (its above and above-right neighbours, and the §9.3.2.4
        // storage made after the second CTB).
        import_above(rx0 + 2, &mut recon, &mut parse_state, &mut slice_addr_of)?;
        let mut ctx = if seg_start && !seg.header.dependent_slice_segment_flag {
            fresh_contexts()
        } else if rx0 == 0 {
            // §9.3.2.5 WPP synchronization from the storage of the
            // row above's second CTB, gated on that CTB (the spatial
            // neighbour T) being available: same slice, in picture.
            let t_avail = r > 0 && w > 1 && slice_addr_of[(r - 1) * w + 1] == Some(slice_addr_rs);
            match (
                t_avail,
                (r > 0 && w > 1)
                    .then(|| halos[(r - 1) * w + 1].get())
                    .flatten()
                    .and_then(|h| h.wpp_contexts.as_deref()),
            ) {
                (true, Some(stored)) => stored.clone(),
                _ => fresh_contexts(),
            }
        } else {
            // A dependent segment starting mid-row continues from
            // TableStateIdxDs of the previous segment (§9.3.2.5).
            ds_stored.take().ok_or(SequenceError::Malformed(
                "dependent segment without Ds state",
            ))?
        };

        for rs in part.start..part.end {
            let rs_u = rs as usize;
            let rx = rs_u % w;
            let x_ctb = (rx * ctb) as u32;
            let y_ctb = y0 as u32;
            import_above(rx + 2, &mut recon, &mut parse_state, &mut slice_addr_of)?;
            slice_addr_of[rs_u] = Some(slice_addr_rs);
            // §7.3.8.3 SAO merge-candidate availability.
            let merge_left = rx > 0 && slice_addr_of[rs_u - 1] == Some(slice_addr_rs);
            let merge_up = r > 0 && slice_addr_of[rs_u - w] == Some(slice_addr_rs);
            let ctu = decode_coding_tree_unit_in_picture(
                &mut engine,
                &mut ctx,
                &sd_params,
                &mut parse_state,
                x_ctb,
                y_ctb,
                slice_addr_rs,
                0,
                merge_left,
                merge_up,
            )?;
            let placed = PlacedInterCtu {
                x_ctb,
                y_ctb,
                slice_addr_rs,
                filter_across_slices,
                ctu: &ctu,
            };
            recon.push_ctu(&placed)?;
            drop(ctu);
            // Publish this CTB to the row below.
            let wpp_contexts = (rx == 1).then(|| Box::new(ctx.clone()));
            let y_last = (y0 + rows).min(height) - 1;
            let parse_cells = parse_state.row_cells(
                y_last / 4,
                rx * (ctb / 4),
                ((rx + 1) * ctb).min(width).div_ceil(4),
            );
            let halo = recon.export_halo(rs, parse_cells, wpp_contexts);
            let _ = halos[rs_u].set(halo);
            progress.publish(r);

            let eos = end_of_slice_segment_flag(&mut engine)
                .map_err(|_| SequenceError::Malformed("CABAC underrun at end_of_slice_segment"))?;
            let last_of_part = rs + 1 == part.end;
            let last_of_seg = rs + 1 == plan.seg_range[part.seg].1;
            if eos {
                if !last_of_seg {
                    return Err(SequenceError::Malformed(
                        "slice segment ended before its successor's address",
                    ));
                }
                if pps.dependent_slice_segments_enabled_flag {
                    ds_stored = Some(ctx.clone());
                }
            } else if last_of_seg {
                if !inputs.tolerant {
                    return Err(SequenceError::Malformed(
                        "end_of_slice_segment_flag not set on the last CTB",
                    ));
                }
            } else if last_of_part {
                // §7.3.8.1 — the segment continues on the next row:
                // end_of_subset_one_bit + byte_alignment( ).
                let one = end_of_slice_segment_flag(&mut engine).map_err(|_| {
                    SequenceError::Malformed("CABAC underrun at end_of_subset_one_bit")
                })?;
                if !one && !inputs.tolerant {
                    return Err(SequenceError::Malformed("end_of_subset_one_bit not set"));
                }
            }
        }
    }
    Ok(recon.finish_band())
}

/// §8.7 in-loop filters row-parallel on `workers` threads, bytes equal
/// to the serial [`crate::inter_recon::filter_picture`]:
///
/// 1. every vertical edge, one CTB row of edges per task — a task
///    touches only its own rows;
/// 2. every horizontal edge, one CTB row of edges per task — a task
///    touches the three rows above its first edge too, so the row
///    chunks of this pass start four rows early;
/// 3. SAO, one CTB row per task from a band copy of its rows whose
///    halo lines (the row above's last line, the row below's first)
///    were saved after deblocking and before any SAO write.
pub(crate) fn filter_picture_parallel(
    mut pic: Picture,
    inputs: &crate::inter_recon::FilterInputs<'_>,
    sao_boundaries: &crate::sao::SaoBoundaries,
    no_filter_map: Option<&crate::deblock::NoFilterMap<'_>>,
    workers: usize,
) -> Picture {
    use crate::deblock::{deblock_rows_planes, EdgeType, QpMap};
    let slice = inputs.slice;
    let params = inputs.params;
    let ctb = 1usize << slice.ctb_log2_size_y;
    let (pw, ph) = pic.plane_dims(Plane::Luma);
    let (cw, ch) = pic.plane_dims(Plane::Cb);
    let chroma = params.chroma_array_type != 0;
    let (sub_w, sub_h) = if chroma {
        crate::picture::sub_wh_c(params.chroma_array_type)
    } else {
        (1, 1)
    };
    let h_ctbs = ph.div_ceil(ctb);
    let w4 = pw.div_ceil(4);
    let qp_map = QpMap {
        cells: inputs.qp_cells,
        w_cells: w4,
    };

    // A row-chunk partition of the three planes at luma rows
    // `bounds[r] .. bounds[r + 1]` (chroma at the matching rows), run as
    // tasks over the workers.
    let run_bands = |pic: &mut Picture, bounds: &[usize], f: &BandTask<'_>| {
        let (ly, cbp, crp) = pic.planes_mut();
        let mut tasks: Vec<Option<RowChunks<'_>>> = Vec::new();
        let (mut lr, mut cbr, mut crr) = (ly, cbp, crp);
        for r in 0..bounds.len() - 1 {
            let (y0, y1) = (bounds[r], bounds[r + 1]);
            let (cy0, cy1) = if chroma {
                (y0 / sub_h, y1.div_ceil(sub_h).min(ch))
            } else {
                (0, 0)
            };
            let (a, b) = std::mem::take(&mut lr).split_at_mut((y1 - y0) * pw);
            lr = b;
            let (c1, c1r) = std::mem::take(&mut cbr).split_at_mut((cy1 - cy0) * cw);
            cbr = c1r;
            let (c2, c2r) = std::mem::take(&mut crr).split_at_mut((cy1 - cy0) * cw);
            crr = c2r;
            tasks.push(Some((r, a, c1, c2)));
        }
        let tasks = Mutex::new(tasks);
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..workers.min(bounds.len() - 1) {
                scope.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::AcqRel);
                    if i >= bounds.len() - 1 {
                        break;
                    }
                    let (r, l, c1, c2) = tasks.lock().unwrap_or_else(|e| e.into_inner())[i]
                        .take()
                        .expect("task taken once");
                    let (y0, _) = (bounds[r], bounds[r + 1]);
                    let mut luma = SamplePlane {
                        samples: l,
                        width: pw,
                        stride: pw,
                        y_origin: y0,
                    };
                    let mut cb = SamplePlane {
                        samples: c1,
                        width: cw,
                        stride: cw,
                        y_origin: y0 / sub_h,
                    };
                    let mut cr = SamplePlane {
                        samples: c2,
                        width: cw,
                        stride: cw,
                        y_origin: y0 / sub_h,
                    };
                    f(r, &mut luma, chroma.then_some((&mut cb, &mut cr)));
                });
            }
        });
    };

    if slice.deblock_enabled {
        if let Some(dparams) = inputs.edges.params() {
            let edges = inputs.edges;
            // Pass 1: vertical edges, natural CTB-row chunks.
            let bounds: Vec<usize> = (0..=h_ctbs).map(|r| (r * ctb).min(ph)).collect();
            run_bands(&mut pic, &bounds, &|r, luma, chroma| {
                deblock_rows_planes(
                    luma,
                    chroma,
                    (pw, ph, cw, ch),
                    edges,
                    &dparams,
                    qp_map,
                    no_filter_map,
                    EdgeType::Vertical,
                    bounds_at(r, ctb, ph),
                    bounds_at(r + 1, ctb, ph),
                );
            });
            // Pass 2: horizontal edges; chunks start four luma rows
            // (the p side of the first edge) early.
            let bounds: Vec<usize> = (0..=h_ctbs)
                .map(|r| if r == 0 { 0 } else { (r * ctb - 4).min(ph) })
                .collect();
            run_bands(&mut pic, &bounds, &|r, luma, chroma| {
                deblock_rows_planes(
                    luma,
                    chroma,
                    (pw, ph, cw, ch),
                    edges,
                    &dparams,
                    qp_map,
                    no_filter_map,
                    EdgeType::Horizontal,
                    bounds_at(r, ctb, ph),
                    bounds_at(r + 1, ctb, ph),
                );
            });
        }
    }

    // Pass 3: SAO. Save every CTB-row boundary's two lines (deblocked,
    // pre-SAO) so each row task classifies from the untouched samples.
    let sao_on = slice.slice_sao_luma_flag || (chroma && slice.slice_sao_chroma_flag);
    if !sao_on {
        return pic;
    }
    let planes: Vec<(Plane, usize, usize, usize)> = {
        let mut v = vec![(Plane::Luma, 0usize, ctb, ctb)];
        if chroma && slice.slice_sao_chroma_flag {
            v.push((Plane::Cb, 1, ctb / sub_w, ctb / sub_h));
            v.push((Plane::Cr, 2, ctb / sub_w, ctb / sub_h));
        }
        if !slice.slice_sao_luma_flag {
            v.remove(0);
        }
        v
    };
    // halo[plane][r] = (line above row r, line below row r), each
    // `None` at the picture edge.
    let mut halo: Vec<Vec<HaloLines>> = Vec::new();
    for &(plane, _, _, n_h) in &planes {
        let (_, h) = pic.plane_dims(plane);
        let mut rows = Vec::with_capacity(h_ctbs);
        for r in 0..h_ctbs {
            let y0 = r * n_h;
            let y1 = (y0 + n_h).min(h);
            let above = (y0 > 0).then(|| pic.row(plane, y0 - 1).to_vec());
            let below = (y1 < h).then(|| pic.row(plane, y1).to_vec());
            rows.push((above, below));
        }
        halo.push(rows);
    }
    let sao_grid = inputs.sao_grid;
    let pic_w_ctbs = inputs.tiling.pic_width_in_ctbs_y() as usize;
    let geoms: Vec<crate::sao::SaoPlaneGeom> = planes
        .iter()
        .map(|&(plane, _, _, _)| crate::sao::SaoPlaneGeom::of(&pic, plane))
        .collect();
    let bounds: Vec<usize> = (0..=h_ctbs).map(|r| (r * ctb).min(ph)).collect();
    run_bands(&mut pic, &bounds, &|r, luma, chroma_planes| {
        let (cb, cr) = match chroma_planes {
            Some((cb, cr)) => (Some(cb), Some(cr)),
            None => (None, None),
        };
        let mut chroma_slots = [cb, cr];
        for (pi, &(plane, cidx, n_w, n_h)) in planes.iter().enumerate() {
            let geom = &geoms[pi];
            let dst: &mut SamplePlane<'_> = match plane {
                Plane::Luma => &mut *luma,
                Plane::Cb => match chroma_slots[0].as_deref_mut() {
                    Some(p) => p,
                    None => continue,
                },
                Plane::Cr => match chroma_slots[1].as_deref_mut() {
                    Some(p) => p,
                    None => continue,
                },
            };
            let y0 = r * n_h;
            if y0 >= geom.ph {
                continue;
            }
            let h = n_h.min(geom.ph - y0);
            let (above, below) = &halo[pi][r];
            // Band: [above line] + own rows + [below line].
            let y_origin = if above.is_some() { y0 - 1 } else { y0 };
            let mut band: Vec<u16> = Vec::with_capacity((h + 2) * geom.pw);
            if let Some(a) = above.as_deref() {
                band.extend_from_slice(a);
            }
            for y in y0..y0 + h {
                let dr = y - dst.y_origin;
                band.extend_from_slice(&dst.samples[dr * dst.stride..dr * dst.stride + geom.pw]);
            }
            if let Some(b) = below.as_deref() {
                band.extend_from_slice(b);
            }
            let src = crate::sao::SaoSource {
                buf: &band,
                stride: geom.pw,
                y_origin,
            };
            let dst_origin = dst.y_origin;
            crate::sao::sao_ctb_row_core(
                src,
                dst.samples,
                dst.stride,
                dst_origin,
                geom,
                sao_grid,
                pic_w_ctbs,
                r,
                cidx,
                (n_w, n_h),
                Some(sao_boundaries),
                no_filter_map,
            );
        }
    });
    pic
}

/// A row-parallel filter task: `(CTB row, luma chunk, chroma chunks)`.
type BandTask<'p> = dyn for<'q> Fn(usize, &mut SamplePlane<'q>, Option<(&mut SamplePlane<'q>, &mut SamplePlane<'q>)>)
    + Sync
    + 'p;
/// One task's row chunks: `(CTB row, luma, Cb, Cr)`.
type RowChunks<'a> = (usize, &'a mut [u16], &'a mut [u16], &'a mut [u16]);
/// The saved `(line above, line below)` of one CTB row of one plane.
type HaloLines = (Option<Vec<u16>>, Option<Vec<u16>>);

/// Luma row `r * ctb` clamped to the picture.
fn bounds_at(r: usize, ctb: usize, ph: usize) -> usize {
    (r * ctb).min(ph)
}
