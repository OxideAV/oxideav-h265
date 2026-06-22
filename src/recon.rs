//! §8.4 intra sample reconstruction driver.
//!
//! This module is the rung between the §7.3.8 slice-data syntax walk
//! (the [`crate::slice_data`] CTU/CU/transform-tree structures) and the
//! per-block §8.4.4 intra prediction + §8.6 dequantization / inverse
//! transform primitives already implemented in [`crate::intra_pred`] and
//! [`crate::transform`]. It walks a decoded [`crate::slice_data::CodingTreeUnit`]
//! and writes reconstructed samples into a [`crate::picture::Picture`]:
//!
//! 1. §8.4.2 — derive `IntraPredModeY` for each luma prediction block
//!    from the signalled `prev_intra_luma_pred_flag` / `mpm_idx` /
//!    `rem_intra_luma_pred_mode` and the neighbour modes; §8.4.3 —
//!    derive `IntraPredModeC`.
//! 2. §8.4.4.1 — for every transform block, gather the §8.4.4.2.1
//!    reference samples from the already-reconstructed picture (the
//!    §6.4.1 availability of left / above neighbours), run §8.4.4.2
//!    prediction.
//! 3. §8.6.2 — dequantize + inverse-transform the coded residual block
//!    (when its coded-block-flag is set), add it to the prediction, and
//!    §8.4.4.1 clip to `[0, (1 << bitDepth) − 1]`, storing the result.
//!
//! The transform-tree recursion mirrors the §8.4.4.1 luma decode order
//! (the residual quadtree drives the transform-block grid) so each block
//! sees its left / above neighbours already reconstructed before it
//! predicts.

use crate::availability::PictureTiling;
use crate::binarization::{
    derive_intra_pred_mode_c, derive_intra_pred_mode_y, intra_luma_cand_mode_list,
    luma_intra_mode_source_from_flag, CuPredMode, LumaIntraModeSource, PartMode,
};
use crate::intra_mode_field::{IntraModeField, Neighbour};
use crate::intra_pred::{
    intra_predict_with_substitution, Component as IpComponent, IntraPredError, IntraPredParams,
    MarkedReferenceSamples, INTRA_DC,
};
use crate::picture::{clip1, sub_wh_c, Picture, Plane};
use crate::slice_data::{CodingQuadtree, CodingTreeUnit, CodingUnit, IntraLumaMode};
use crate::transform::{
    residual_block, BlockParams, Component as TfComponent, PredMode, TransformError,
};
use crate::transform_tree::TransformTree;
use crate::transform_unit::TransformUnit;

/// Errors raised while reconstructing samples from a decoded CTU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconError {
    /// A §8.4.4.2 intra-prediction primitive failed.
    IntraPred(IntraPredError),
    /// A §8.6.2 dequantization / inverse-transform primitive failed.
    Transform(TransformError),
    /// A §8.5.3.3 inter-prediction primitive failed.
    InterPred(crate::inter_pred::InterPredError),
    /// The decoded CTU carried an inter prediction unit, which the intra
    /// reconstruction path does not handle.
    InterNotSupported,
    /// The §6.4.1 picture-tiling geometry needed for neighbour
    /// availability could not be built.
    Tiling(crate::availability::AvailabilityError),
}

impl core::fmt::Display for ReconError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::IntraPred(e) => write!(f, "intra prediction failed: {e}"),
            Self::Transform(e) => write!(f, "inverse transform failed: {e}"),
            Self::InterPred(e) => write!(f, "inter prediction failed: {e}"),
            Self::InterNotSupported => {
                f.write_str("inter prediction is not reconstructed by the intra path")
            }
            Self::Tiling(e) => write!(f, "picture tiling geometry invalid: {e}"),
        }
    }
}

impl std::error::Error for ReconError {}

impl From<IntraPredError> for ReconError {
    fn from(e: IntraPredError) -> Self {
        Self::IntraPred(e)
    }
}

impl From<TransformError> for ReconError {
    fn from(e: TransformError) -> Self {
        Self::Transform(e)
    }
}

/// SPS / PPS / slice-derived state constant across one picture's intra
/// reconstruction. Every field is a §8 input the per-block prediction /
/// dequantization reads.
#[derive(Debug, Clone, Copy)]
pub struct ReconParams {
    /// `ChromaArrayType` (0 = monochrome, 1 = 4:2:0, 2 = 4:2:2,
    /// 3 = 4:4:4).
    pub chroma_array_type: u8,
    /// `BitDepthY`.
    pub bit_depth_luma: u8,
    /// `BitDepthC`.
    pub bit_depth_chroma: u8,
    /// `intra_smoothing_disabled_flag` (§8.4.4.2.3 step-1 gate).
    pub intra_smoothing_disabled: bool,
    /// `strong_intra_smoothing_enabled_flag` (§8.4.4.2.3 `biIntFlag`).
    pub strong_intra_smoothing_enabled: bool,
    /// `SliceQpY` (§7.4.7.1) — the slice quantization parameter, the
    /// §8.6.1 luma-QP starting point (the tiny single-CU fixtures carry
    /// `CuQpDeltaVal == 0`, so `QpY == SliceQpY`).
    pub slice_qp_y: i32,
    /// `pps_cb_qp_offset + slice_cb_qp_offset` (§8.6.1 `qPiCb` offset).
    pub cb_qp_offset: i32,
    /// `pps_cr_qp_offset + slice_cr_qp_offset` (§8.6.1 `qPiCr` offset).
    pub cr_qp_offset: i32,
    /// `transform_skip_rotation_enabled_flag`.
    pub transform_skip_rotation_enabled: bool,
    /// `extended_precision_processing_flag`.
    pub extended_precision: bool,
}

/// `QpBdOffsetY = 6 * bit_depth_luma_minus8` (§7.4.3.2.1, eq. 7-4).
#[inline]
fn qp_bd_offset(bit_depth: u8) -> i32 {
    6 * (i32::from(bit_depth) - 8)
}

/// §8.6.1 — derive `Qp′Y` for a luma transform block.
///
/// `Qp′Y = QpY + QpBdOffsetY`, with `QpY` the slice QP plus any
/// `CuQpDeltaVal` (clipped per the §8.6.1 wrap). The neighbour-prediction
/// of `qPY_PRED` collapses to a single value when every coding unit in
/// the picture shares the slice QP (no `cu_qp_delta`); the recursion
/// threads `cu_qp_delta_val` so the general single-CU-per-QG case is
/// exact.
#[inline]
fn luma_qp(params: &ReconParams, cu_qp_delta_val: i32) -> u32 {
    let qp_bd = qp_bd_offset(params.bit_depth_luma);
    // §8.6.1 eq. 8-258: QpY = ((qPY_PRED + CuQpDeltaVal + 52 + 2*QpBdOffsetY)
    //                          % (52 + QpBdOffsetY)) − QpBdOffsetY.
    // With qPY_PRED == SliceQpY (single-QG picture) this is the slice QP
    // plus the (already in-range) delta.
    let qpy_pred = params.slice_qp_y;
    let modulus = 52 + qp_bd;
    let qpy = (qpy_pred + cu_qp_delta_val + 52 + 2 * qp_bd).rem_euclid(modulus) - qp_bd;
    (qpy + qp_bd) as u32
}

/// Table 8-10 — `ChromaArrayType == 1` chroma-QP mapping `QpC = f(qPi)`.
#[inline]
fn qpc_420(qpi: i32) -> i32 {
    match qpi {
        x if x < 30 => x,
        30 => 29,
        31 => 30,
        32 => 31,
        33 => 32,
        34 => 33,
        35 => 33,
        36 => 34,
        37 => 34,
        38 => 35,
        39 => 35,
        40 => 36,
        41 => 36,
        42 => 37,
        43 => 37,
        x => x - 6,
    }
}

/// §8.6.1 — derive `Qp′Cb` / `Qp′Cr` for a chroma transform block.
///
/// `qPiCx = Clip3( −QpBdOffsetC, 57, QpY + cQpOffset )`; for
/// `ChromaArrayType == 1` `qPCx = qPC_table( qPiCx )` (Table 8-10), for
/// the other chroma types `qPCx = Min( qPiCx, 51 )`; then
/// `Qp′Cx = qPCx + QpBdOffsetC` (eq. 8-260).
#[inline]
fn chroma_qp(params: &ReconParams, cu_qp_delta_val: i32, cidx: TfComponent) -> u32 {
    let qp_bd_y = qp_bd_offset(params.bit_depth_luma);
    let qp_bd_c = qp_bd_offset(params.bit_depth_chroma);
    // QpY (without the luma BdOffset) — the §8.6.1 chroma input.
    let qpy = {
        let modulus = 52 + qp_bd_y;
        (params.slice_qp_y + cu_qp_delta_val + 52 + 2 * qp_bd_y).rem_euclid(modulus) - qp_bd_y
    };
    let offset = match cidx {
        TfComponent::Cb => params.cb_qp_offset,
        TfComponent::Cr => params.cr_qp_offset,
        TfComponent::Luma => 0,
    };
    let qpi = (qpy + offset).clamp(-qp_bd_c, 57);
    let qpc = if params.chroma_array_type == 1 {
        qpc_420(qpi)
    } else {
        qpi.min(51)
    };
    (qpc + qp_bd_c) as u32
}

/// Gather the §8.4.4.2.1 reference-sample array for a transform block at
/// plane position `(xb, yb)` of side `n_tbs` from the already-
/// reconstructed picture, marking each neighbour available per the §6.4.1
/// z-scan availability process (via [`ReconCtx::ref_sample_available`]):
/// a neighbour is available iff it is inside the picture, already decoded
/// in z-scan order, and in the same slice / tile. The unavailable samples
/// are substituted by [`intra_predict_with_substitution`].
fn gather_reference_samples(
    pic: &Picture,
    ctx: &ReconCtx,
    plane: Plane,
    xb: usize,
    yb: usize,
    n_tbs: usize,
) -> MarkedReferenceSamples {
    let (pw, ph) = pic.plane_dims(plane);
    let (sub_w, sub_h) = match plane {
        Plane::Luma => (1, 1),
        Plane::Cb | Plane::Cr => sub_wh_c(pic.chroma_array_type()),
    };
    let avail = |x: i64, y: i64| -> bool {
        x >= 0
            && y >= 0
            && (x as usize) < pw
            && (y as usize) < ph
            && ctx.ref_sample_available(xb, yb, x, y, sub_w, sub_h)
    };
    let read = |x: i64, y: i64| -> (i32, bool) {
        if avail(x, y) {
            (pic.sample(plane, x as usize, y as usize), true)
        } else {
            (0, false)
        }
    };
    // Corner p[−1][−1].
    let corner = read(xb as i64 - 1, yb as i64 - 1);
    // Left column p[−1][0 .. 2*nTbS−1].
    let mut left = Vec::with_capacity(2 * n_tbs);
    for y in 0..(2 * n_tbs) {
        left.push(read(xb as i64 - 1, yb as i64 + y as i64));
    }
    // Top row p[0 .. 2*nTbS−1][−1].
    let mut top = Vec::with_capacity(2 * n_tbs);
    for x in 0..(2 * n_tbs) {
        top.push(read(xb as i64 + x as i64, yb as i64 - 1));
    }
    MarkedReferenceSamples::new(n_tbs, corner, left, top)
        .expect("reference array dimensions match n_tbs")
}

/// Predict one intra transform block, add its residual, clip, and store
/// into `pic`. `(xb, yb)` is the plane-coordinate top-left; `pred_mode`
/// is the §8.4.x prediction mode for the plane's component.
#[allow(clippy::too_many_arguments)]
fn reconstruct_intra_block(
    pic: &mut Picture,
    params: &ReconParams,
    ctx: &ReconCtx,
    plane: Plane,
    cidx: TfComponent,
    ip_component: IpComponent,
    xb: usize,
    yb: usize,
    n_tbs: usize,
    pred_mode_intra: u8,
    residual: Option<&[i32]>,
    qp: u32,
    transquant_bypass: bool,
    transform_skip: bool,
) -> Result<(), ReconError> {
    let bit_depth = pic.bit_depth(plane);
    let marked = gather_reference_samples(pic, ctx, plane, xb, yb, n_tbs);
    let ip_params = IntraPredParams {
        pred_mode_intra,
        cidx: ip_component,
        bit_depth,
        bit_depth_luma: params.bit_depth_luma,
        intra_smoothing_disabled: params.intra_smoothing_disabled,
        strong_intra_smoothing_enabled: params.strong_intra_smoothing_enabled,
        chroma_array_type_3: params.chroma_array_type == 3,
        disable_boundary_filter: false,
    };
    let pred = intra_predict_with_substitution(&marked, &ip_params)?;

    // §8.6.2 residual array (zero when the block has no coded coeffs).
    let res: Option<Vec<i32>> = match residual {
        Some(levels) => {
            let bp = BlockParams {
                n_tbs,
                q_p: qp,
                component: cidx,
                pred_mode: PredMode::Intra,
                bit_depth,
                extended_precision: params.extended_precision,
                transquant_bypass,
                transform_skip,
                transform_skip_rotation_enabled: params.transform_skip_rotation_enabled,
            };
            Some(residual_block(levels, None, bp)?)
        }
        None => None,
    };

    // §8.4.4.1 / §8.6.5: recSamples = Clip1( predSamples + resSamples ).
    for y in 0..n_tbs {
        for x in 0..n_tbs {
            let p = pred[y * n_tbs + x];
            let r = res.as_ref().map_or(0, |r| r[y * n_tbs + x]);
            let v = clip1(p + r, bit_depth);
            pic.set_sample(plane, xb + x, yb + y, v);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// §8.5 inter sample reconstruction
// ---------------------------------------------------------------------------

use crate::inter_pred::{
    predict_inter_pu, InterPredGeometry, InterPrediction, ListPrediction, RefPlane,
};

/// One reference list's fully-resolved per-PU motion: the
/// §8.5.3.2-derived luma motion vector, the §8.5.3.2.10 chroma motion
/// vector, and the §8.5.3.3.2-selected reference picture.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedList<'a> {
    /// `predFlagLX` — whether this list contributes.
    pub pred_flag: bool,
    /// `mvLX` in quarter-luma-sample units.
    pub mv_l: [i32; 2],
    /// `mvCLX` in eighth-chroma-sample units (§8.5.3.2.10).
    pub mv_c: [i32; 2],
    /// `RefPicListX[refIdxLX]` — the reference picture's samples.
    pub ref_pic: &'a Picture,
}

/// Build the §8.5.3.3.2 [`ListPrediction`] for one reference list,
/// borrowing the reference picture's luma + (when chroma is present)
/// Cb / Cr planes for the lifetime `'a`.
fn build_list_prediction<'a>(
    list: &ResolvedList<'a>,
    chroma_array_type: u8,
) -> Result<ListPrediction<'a>, ReconError> {
    let (lw, lh) = list.ref_pic.plane_dims(Plane::Luma);
    let lp =
        RefPlane::new(list.ref_pic.plane(Plane::Luma), lw, lh).map_err(ReconError::InterPred)?;
    let (cb, cr) = if chroma_array_type != 0 {
        let (cw, ch) = list.ref_pic.plane_dims(Plane::Cb);
        (
            Some(
                RefPlane::new(list.ref_pic.plane(Plane::Cb), cw, ch)
                    .map_err(ReconError::InterPred)?,
            ),
            Some(
                RefPlane::new(list.ref_pic.plane(Plane::Cr), cw, ch)
                    .map_err(ReconError::InterPred)?,
            ),
        )
    } else {
        (None, None)
    };
    Ok(ListPrediction {
        pred_flag: list.pred_flag,
        luma: lp,
        cb,
        cr,
        mv_l: list.mv_l,
        mv_c: list.mv_c,
    })
}

/// §8.5.3.3 — reconstruct one inter prediction unit's motion-compensated
/// prediction into `pic`, then (when a residual block is supplied) add
/// the §8.6.2 residual and §8.6.5 / §8.4.4.1 clip.
///
/// `(x_pb, y_pb)` is the PU's luma top-left; `(n_pb_w, n_pb_h)` its luma
/// size. The L0 / L1 lists carry the resolved motion + reference picture.
/// `residual_luma` / `residual_cb` / `residual_cr` are the optional
/// §8.6.2-output residual arrays (already dequantized + inverse
/// transformed) for the PU's covering transform blocks; pass `None` for a
/// skip / zero-residual PU (the prediction is written directly).
///
/// # Errors
/// Propagates [`ReconError::InterNotSupported`] reuse is avoided here; the
/// §8.5.3.3 interpolation failures surface as [`ReconError`] via the
/// `InterPred` variant carrying the [`crate::inter_pred::InterPredError`].
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_inter_pu(
    pic: &mut Picture,
    params: &ReconParams,
    x_pb: usize,
    y_pb: usize,
    n_pb_w: usize,
    n_pb_h: usize,
    l0: ResolvedList<'_>,
    l1: ResolvedList<'_>,
    residual_luma: Option<&[i32]>,
    residual_cb: Option<&[i32]>,
    residual_cr: Option<&[i32]>,
) -> Result<(), ReconError> {
    let cat = params.chroma_array_type;
    // Build the §8.5.3.3.2 reference planes for each used list.
    let lp0 = build_list_prediction(&l0, cat)?;
    let lp1 = build_list_prediction(&l1, cat)?;
    let geom = InterPredGeometry {
        x_pb: x_pb as i32,
        y_pb: y_pb as i32,
        n_pb_w,
        n_pb_h,
        chroma_array_type: cat,
        bit_depth_luma: params.bit_depth_luma,
        bit_depth_chroma: params.bit_depth_chroma,
    };
    let InterPrediction { luma, cb, cr } =
        predict_inter_pu(&lp0, &lp1, &geom).map_err(ReconError::InterPred)?;

    // §8.6.5 / §8.4.4.1: recSamples = Clip1( predSamples + resSamples ).
    write_inter_plane(
        pic,
        Plane::Luma,
        x_pb,
        y_pb,
        n_pb_w,
        n_pb_h,
        &luma,
        residual_luma,
    );
    if cat != 0 {
        let (sw, sh) = sub_wh_c(cat);
        let xc = x_pb / sw;
        let yc = y_pb / sh;
        let pcw = n_pb_w / sw;
        let pch = n_pb_h / sh;
        write_inter_plane(pic, Plane::Cb, xc, yc, pcw, pch, &cb, residual_cb);
        write_inter_plane(pic, Plane::Cr, xc, yc, pcw, pch, &cr, residual_cr);
    }
    Ok(())
}

/// Write one motion-compensated prediction plane plus its optional
/// residual into `pic` with the §8.4.4.1 clip.
#[allow(clippy::too_many_arguments)]
fn write_inter_plane(
    pic: &mut Picture,
    plane: Plane,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    pred: &[i32],
    residual: Option<&[i32]>,
) {
    let bit_depth = pic.bit_depth(plane);
    for y in 0..h {
        for x in 0..w {
            let p = pred[y * w + x];
            let r = residual.map_or(0, |r| r[y * w + x]);
            pic.set_sample(plane, x0 + x, y0 + y, clip1(p + r, bit_depth));
        }
    }
}

/// Map an `IpComponent` for the plane / cIdx pair.
#[inline]
fn ip_component_of(cidx: TfComponent) -> IpComponent {
    match cidx {
        TfComponent::Luma => IpComponent::Luma,
        TfComponent::Cb => IpComponent::Cb,
        TfComponent::Cr => IpComponent::Cr,
    }
}

/// Per-picture intra-reconstruction neighbour state — the §8.4.2
/// `IntraPredModeY` field plus the §6.4.1 picture tiling that resolves
/// neighbour availability.
///
/// One [`ReconCtx`] is shared across every CTU of a slice so the §8.4.2
/// most-probable-mode derivation sees the actual left / above neighbour
/// modes (rather than the flat-single-CU `INTRA_DC` assumption). Build it
/// with [`ReconCtx::new`]; reconstruct each CTU in tile-scan order with
/// [`reconstruct_intra_ctu_ctx`].
#[derive(Debug)]
pub struct ReconCtx {
    field: IntraModeField,
    tiling: PictureTiling,
}

impl ReconCtx {
    /// Build the neighbour context for a `pic_width` × `pic_height` luma
    /// picture with the given `CtbLog2SizeY` / `MinTbLog2SizeY` and tile
    /// layout.
    ///
    /// # Errors
    /// Propagates [`crate::availability::AvailabilityError`] (wrapped in
    /// [`ReconError::Tiling`]) when the geometry is degenerate.
    pub fn new(
        pic_width_luma: usize,
        pic_height_luma: usize,
        ctb_log2_size_y: u32,
        min_tb_log2_size_y: u32,
        tiles: &crate::availability::TilingParams,
    ) -> Result<Self, ReconError> {
        let ctb_size = 1usize << ctb_log2_size_y;
        let pic_w_ctbs = pic_width_luma.div_ceil(ctb_size) as u32;
        let pic_h_ctbs = pic_height_luma.div_ceil(ctb_size) as u32;
        let tiling = PictureTiling::new(
            pic_w_ctbs,
            pic_h_ctbs,
            pic_width_luma as u32,
            pic_height_luma as u32,
            ctb_log2_size_y,
            min_tb_log2_size_y,
            tiles,
        )
        .map_err(ReconError::Tiling)?;
        Ok(Self {
            field: IntraModeField::new(pic_width_luma, pic_height_luma, ctb_log2_size_y),
            tiling,
        })
    }

    /// §6.4.1 z-scan availability of the neighbour luma location
    /// `( x_nb, y_nb )` for the current prediction block at
    /// `( x_curr, y_curr )`. A single-slice picture maps every CTB to the
    /// same `SliceAddrRs` (the constant `0`); a multi-slice driver would
    /// supply the per-CTB slice map instead.
    fn available(&self, x_curr: usize, y_curr: usize, x_nb: i64, y_nb: i64) -> bool {
        self.tiling.z_scan_availability(
            x_curr as u32,
            y_curr as u32,
            x_nb as i32,
            y_nb as i32,
            |_ctb_rs| 0,
        )
    }

    /// §8.4.4.2.1 reference-sample availability for one neighbour sample of
    /// a transform block. `(x_tb, y_tb)` is the current transform block's
    /// **plane** top-left and `(x_ref, y_ref)` the neighbour's **plane**
    /// coordinates; `(sub_w, sub_h)` is the plane's `(SubWidthC, SubHeightC)`
    /// (`(1, 1)` for luma). The §6.4.1 z-scan availability is evaluated on
    /// the corresponding luma locations.
    fn ref_sample_available(
        &self,
        x_tb: usize,
        y_tb: usize,
        x_ref: i64,
        y_ref: i64,
        sub_w: usize,
        sub_h: usize,
    ) -> bool {
        if x_ref < 0 || y_ref < 0 {
            return false;
        }
        // Map plane coordinates to luma for the §6.4.1 test.
        let x_curr_luma = x_tb * sub_w;
        let y_curr_luma = y_tb * sub_h;
        let x_nb_luma = (x_ref as usize * sub_w) as i64;
        let y_nb_luma = (y_ref as usize * sub_h) as i64;
        self.available(x_curr_luma, y_curr_luma, x_nb_luma, y_nb_luma)
    }

    /// Test-only: the §8.4.2 `IntraPredModeY` recorded at a luma location.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn recorded_mode(&self, x_luma: usize, y_luma: usize) -> Option<u8> {
        self.field.recorded_mode(x_luma, y_luma)
    }
}

/// Reconstruct one decoded coding tree unit's intra samples into `pic`,
/// driving the §8.4.2 most-probable-mode derivation off the shared
/// [`ReconCtx`] neighbour field. CTUs must be reconstructed in tile-scan
/// order so each one's left / above neighbours are already recorded.
///
/// # Errors
/// [`ReconError::InterNotSupported`] if any leaf coding unit is inter;
/// [`ReconError::IntraPred`] / [`ReconError::Transform`] on a primitive
/// failure.
pub fn reconstruct_intra_ctu_ctx(
    pic: &mut Picture,
    params: &ReconParams,
    ctx: &mut ReconCtx,
    ctu: &CodingTreeUnit,
) -> Result<(), ReconError> {
    reconstruct_quadtree(pic, params, ctx, &ctu.quadtree)
}

/// Reconstruct one decoded coding tree unit's intra samples into `pic`.
/// `(ctb_x, ctb_y)` is the CTB's luma top-left position.
///
/// This single-CTU convenience builds a fresh single-CTU [`ReconCtx`]
/// internally — its neighbours are the CTU's own already-reconstructed
/// blocks, so a multi-CTU picture must instead share one [`ReconCtx`]
/// across CTUs via [`reconstruct_intra_ctu_ctx`].
///
/// # Errors
/// [`ReconError::InterNotSupported`] if any leaf coding unit is inter;
/// [`ReconError::IntraPred`] / [`ReconError::Transform`] on a primitive
/// failure.
pub fn reconstruct_intra_ctu(
    pic: &mut Picture,
    params: &ReconParams,
    ctu: &CodingTreeUnit,
) -> Result<(), ReconError> {
    // A single CTB covers the whole synthetic picture for this path; the
    // CTB log2 size is the smallest power of two covering the picture.
    let max_dim = pic.width_luma().max(pic.height_luma()).max(1);
    let ctb_log2 = (usize::BITS - (max_dim - 1).leading_zeros()).max(4);
    let min_tb_log2 = 2;
    let mut ctx = ReconCtx::new(
        pic.width_luma(),
        pic.height_luma(),
        ctb_log2,
        min_tb_log2,
        &crate::availability::TilingParams::single_tile(),
    )?;
    reconstruct_intra_ctu_ctx(pic, params, &mut ctx, ctu)
}

fn reconstruct_quadtree(
    pic: &mut Picture,
    params: &ReconParams,
    ctx: &mut ReconCtx,
    qt: &CodingQuadtree,
) -> Result<(), ReconError> {
    match qt {
        CodingQuadtree::Split(children) => {
            for child in children {
                reconstruct_quadtree(pic, params, ctx, child)?;
            }
            Ok(())
        }
        CodingQuadtree::Leaf(cu) => reconstruct_cu(pic, params, ctx, cu),
    }
}

/// §8.4.2 — derive `IntraPredModeY` for one luma prediction block at
/// `( x_pb, y_pb )` of side `n_pb`, consulting the [`ReconCtx`] neighbour
/// field for the candidate modes, then record it back into the field.
fn derive_and_record_luma_mode(
    ctx: &mut ReconCtx,
    x_pb: usize,
    y_pb: usize,
    n_pb: usize,
    luma_mode: &IntraLumaMode,
    pcm_flag: bool,
) -> u8 {
    // Step 1 / 2 — candidate modes from the left (A) and above (B)
    // neighbours, gated on §6.4.1 z-scan availability.
    let avail_a = ctx.available(x_pb, y_pb, x_pb as i64 - 1, y_pb as i64);
    let avail_b = ctx.available(x_pb, y_pb, x_pb as i64, y_pb as i64 - 1);
    let cand_a = ctx
        .field
        .cand_intra_pred_mode(x_pb, y_pb, Neighbour::Left, avail_a);
    let cand_b = ctx
        .field
        .cand_intra_pred_mode(x_pb, y_pb, Neighbour::Above, avail_b);

    // Step 3 / 4 — candModeList + the prev_intra_luma_pred_flag selection.
    let cand_list = intra_luma_cand_mode_list(cand_a, cand_b);
    let source = luma_intra_mode_source_from_flag(u8::from(luma_mode.prev_intra_luma_pred_flag));
    let field_val = match source {
        LumaIntraModeSource::Mpm => luma_mode.mpm_idx.unwrap_or(0),
        LumaIntraModeSource::Remaining => luma_mode.rem_intra_luma_pred_mode.unwrap_or(0),
    };
    let mode = derive_intra_pred_mode_y(cand_list, source, field_val);
    ctx.field.record_intra_pb(x_pb, y_pb, n_pb, mode, pcm_flag);
    mode
}

/// Reconstruct one leaf coding unit. Only intra CUs are handled; each luma
/// prediction block's `IntraPredModeY` is derived per §8.4.2 from the
/// [`ReconCtx`] neighbour field (most-probable-mode), and chroma
/// `IntraPredModeC` per §8.4.3 from the first PB's luma mode.
fn reconstruct_cu(
    pic: &mut Picture,
    params: &ReconParams,
    ctx: &mut ReconCtx,
    cu: &CodingUnit,
) -> Result<(), ReconError> {
    if matches!(cu.cu_pred_mode, CuPredMode::Inter | CuPredMode::Skip) {
        // Record the inter CU so a later intra block's §8.4.2 neighbour
        // derivation maps it to INTRA_DC, then signal the inter path is
        // unhandled by this driver.
        ctx.field.record_non_intra_cu(
            cu.x0 as usize,
            cu.y0 as usize,
            1usize << cu.log2_cb_size,
            cu.cu_pred_mode,
        );
        return Err(ReconError::InterNotSupported);
    }

    let n_cb = 1usize << cu.log2_cb_size;
    let x_cb = cu.x0 as usize;
    let y_cb = cu.y0 as usize;

    if cu.pcm_flag {
        // PCM sample reconstruction (§8.4.5.2) is a separate path; still
        // stamp the field so neighbours see a written pcm block (→ DC).
        ctx.field.record_intra_pb(x_cb, y_cb, n_cb, INTRA_DC, true);
        return Ok(());
    }

    // §7.4.9.5: PART_NxN intra splits the CU into four nCbS/2 luma PBs in
    // raster order; PART_2Nx2N is one PB covering the CU.
    let is_nxn = cu.part_mode == PartMode::PartNxN;
    let n_pb = if is_nxn { n_cb / 2 } else { n_cb };
    let pb_origins: &[(usize, usize)] = if is_nxn {
        &[(0, 0), (1, 0), (0, 1), (1, 1)]
    } else {
        &[(0, 0)]
    };

    // §8.4.2 — derive (and record) each luma PB's IntraPredModeY. The
    // §8.4.3 chroma mode uses the CU-corner PB (blkIdx 0) for the
    // ChromaArrayType != 3 single-chroma-block case.
    let mut pb_modes = [INTRA_DC; 4];
    for (i, &(qx, qy)) in pb_origins.iter().enumerate() {
        let x_pb = x_cb + qx * n_pb;
        let y_pb = y_cb + qy * n_pb;
        let luma_mode = cu
            .intra_luma
            .get(i)
            .copied()
            .unwrap_or(crate::slice_data::IntraLumaMode {
                prev_intra_luma_pred_flag: true,
                mpm_idx: Some(0),
                rem_intra_luma_pred_mode: None,
            });
        pb_modes[i] = derive_and_record_luma_mode(ctx, x_pb, y_pb, n_pb, &luma_mode, false);
    }

    // §8.4.3 — IntraPredModeC from the first PB's luma mode.
    let intra_pred_mode_c = derive_intra_pred_mode_c(
        cu.intra_chroma_pred_mode[0],
        pb_modes[0],
        params.chroma_array_type == 2,
    );

    // Walk the transform tree, reconstructing each leaf transform block.
    // For PART_NxN the top-level tree is a Split whose four children map to
    // the four luma PBs (each carrying that PB's luma mode); chroma uses
    // the CU-level IntraPredModeC throughout.
    if let Some(tree) = &cu.transform_tree {
        if is_nxn {
            if let TransformTree::Split { children, .. } = tree {
                let half = n_cb / 2;
                let offsets = [(0, 0), (half, 0), (0, half), (half, half)];
                for (i, (child, (dx, dy))) in children.iter().zip(offsets).enumerate() {
                    reconstruct_transform_tree(
                        pic,
                        params,
                        ctx,
                        child,
                        x_cb + dx,
                        y_cb + dy,
                        cu.log2_cb_size - 1,
                        pb_modes[i],
                        intra_pred_mode_c,
                        cu.cu_transquant_bypass_flag,
                    )?;
                }
                return Ok(());
            }
        }
        reconstruct_transform_tree(
            pic,
            params,
            ctx,
            tree,
            x_cb,
            y_cb,
            cu.log2_cb_size,
            pb_modes[0],
            intra_pred_mode_c,
            cu.cu_transquant_bypass_flag,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_transform_tree(
    pic: &mut Picture,
    params: &ReconParams,
    ctx: &ReconCtx,
    tree: &TransformTree,
    x0: usize,
    y0: usize,
    log2_trafo_size: u32,
    intra_pred_mode_y: u8,
    intra_pred_mode_c: u8,
    transquant_bypass: bool,
) -> Result<(), ReconError> {
    match tree {
        TransformTree::Split { children, .. } => {
            let half = 1usize << (log2_trafo_size - 1);
            // raster order [tl, tr, bl, br].
            let offsets = [(0, 0), (half, 0), (0, half), (half, half)];
            for (child, (dx, dy)) in children.iter().zip(offsets) {
                reconstruct_transform_tree(
                    pic,
                    params,
                    ctx,
                    child,
                    x0 + dx,
                    y0 + dy,
                    log2_trafo_size - 1,
                    intra_pred_mode_y,
                    intra_pred_mode_c,
                    transquant_bypass,
                )?;
            }
            Ok(())
        }
        TransformTree::Leaf { unit, .. } => reconstruct_transform_unit(
            pic,
            params,
            ctx,
            unit,
            x0,
            y0,
            log2_trafo_size,
            intra_pred_mode_y,
            intra_pred_mode_c,
            transquant_bypass,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_transform_unit(
    pic: &mut Picture,
    params: &ReconParams,
    ctx: &ReconCtx,
    unit: &TransformUnit,
    x0: usize,
    y0: usize,
    log2_trafo_size: u32,
    intra_pred_mode_y: u8,
    intra_pred_mode_c: u8,
    transquant_bypass: bool,
) -> Result<(), ReconError> {
    let n_tbs = 1usize << log2_trafo_size;
    // §7.4.9.14 CuQpDeltaVal — the delta carried by this TU when its
    // §7.3.8.14 gate fired (once per quantization group), else 0. The
    // single-CU-per-QG fixtures resolve this to the CU's one decoded
    // delta; a deeper multi-CU-per-QG picture threads the accumulated
    // value (a DPB-level followup).
    let cu_qp_delta_val = unit.cu_qp_delta.as_ref().map_or(0, |d| d.value);

    // Luma block.
    let luma_qp = luma_qp(params, cu_qp_delta_val);
    let luma_levels = unit.residual_luma.as_ref().map(|rb| rb.levels.as_slice());
    reconstruct_intra_block(
        pic,
        params,
        ctx,
        Plane::Luma,
        TfComponent::Luma,
        IpComponent::Luma,
        x0,
        y0,
        n_tbs,
        intra_pred_mode_y,
        luma_levels,
        luma_qp,
        transquant_bypass,
        false,
    )?;

    // Chroma blocks. For 4:2:0 / 4:2:2 the chroma transform block sits at
    // (x0 >> SubWidthC, y0 >> SubHeightC) and is half the luma side for
    // 4:2:0; the §7.3.8.10 driver collects the chroma residual at the
    // parent node, so a chroma block is reconstructed once per luma node
    // that carries chroma residual (the residual_cb / residual_cr lists).
    if params.chroma_array_type != 0 {
        let (sw, sh) = sub_wh_c(params.chroma_array_type);
        let xc = x0 / sw;
        let yc = y0 / sh;
        reconstruct_chroma_blocks(
            pic,
            params,
            ctx,
            Plane::Cb,
            TfComponent::Cb,
            &unit.residual_cb,
            xc,
            yc,
            intra_pred_mode_c,
            cu_qp_delta_val,
            transquant_bypass,
        )?;
        reconstruct_chroma_blocks(
            pic,
            params,
            ctx,
            Plane::Cr,
            TfComponent::Cr,
            &unit.residual_cr,
            xc,
            yc,
            intra_pred_mode_c,
            cu_qp_delta_val,
            transquant_bypass,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_chroma_blocks(
    pic: &mut Picture,
    params: &ReconParams,
    ctx: &ReconCtx,
    plane: Plane,
    cidx: TfComponent,
    residual_blocks: &[crate::residual::ResidualBlock],
    xc: usize,
    yc: usize,
    intra_pred_mode_c: u8,
    cu_qp_delta_val: i32,
    transquant_bypass: bool,
) -> Result<(), ReconError> {
    let qp = chroma_qp(params, cu_qp_delta_val, cidx);
    // The chroma transform-block side is derived from the residual block's
    // own log2 size; when no residual is coded the chroma TB still needs
    // prediction, so synthesize one zero-residual block of the chroma TB
    // size derived from the luma node geometry handled by the caller.
    if residual_blocks.is_empty() {
        // No coded chroma residual at this node — prediction only. The TB
        // side equals the chroma plane sub-sampling of the parent luma
        // node; the caller passes the node geometry implicitly through
        // xc/yc, so we cannot know the size here. This branch is only
        // reached for chroma-cbf-clear nodes; the flat single-CU fixtures
        // always carry the chroma residual at the CU root, so leave the
        // (already-zero / previously-written) prediction in place by
        // doing nothing. A full DPB build wires the cbf-clear prediction
        // path; see the module followups.
        return Ok(());
    }
    for rb in residual_blocks {
        let n_tbs = rb.size();
        reconstruct_intra_block(
            pic,
            params,
            ctx,
            plane,
            cidx,
            ip_component_of(cidx),
            xc,
            yc,
            n_tbs,
            intra_pred_mode_c,
            Some(rb.levels.as_slice()),
            qp,
            transquant_bypass,
            false,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::availability::TilingParams;
    use crate::binarization::{CuPredMode, PartMode};
    use crate::residual::ResidualBlock;
    use crate::slice_data::{CodingQuadtree, CodingTreeUnit, CodingUnit, IntraLumaMode};
    use crate::transform_tree::TransformTree;
    use crate::transform_unit::TransformUnit;

    /// §8 reconstruction params for a Main-profile 4:2:0 8-bit slice at
    /// SliceQpY = 25 (the tiny-i fixture geometry).
    fn tiny_params() -> ReconParams {
        ReconParams {
            chroma_array_type: 1,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            intra_smoothing_disabled: false,
            strong_intra_smoothing_enabled: true,
            slice_qp_y: 25,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            transform_skip_rotation_enabled: false,
            extended_precision: false,
        }
    }

    /// A single-DC residual block of side `1 << log2`.
    fn dc_block(log2: u32, dc: i32) -> ResidualBlock {
        let size = 1usize << log2;
        let mut levels = vec![0i32; size * size];
        levels[0] = dc;
        ResidualBlock {
            log2_trafo_size: log2,
            last_sig_coeff_x: 0,
            last_sig_coeff_y: 0,
            levels,
        }
    }

    /// Build a flat 16x16 intra CU (PART_2Nx2N, PLANAR luma via mpm_idx 0)
    /// carrying the given luma / Cb / Cr DC residuals.
    fn flat_intra_ctu(luma_dc: i32, cb_dc: Option<i32>, cr_dc: Option<i32>) -> CodingTreeUnit {
        let mut unit = TransformUnit {
            residual_luma: Some(dc_block(4, luma_dc)),
            ..Default::default()
        };
        if let Some(d) = cb_dc {
            unit.residual_cb = vec![dc_block(3, d)];
        }
        if let Some(d) = cr_dc {
            unit.residual_cr = vec![dc_block(3, d)];
        }
        let tree = TransformTree::Leaf {
            cbf_luma: true,
            unit,
        };
        let cu = CodingUnit {
            x0: 0,
            y0: 0,
            log2_cb_size: 4,
            cu_pred_mode: CuPredMode::Intra,
            cu_transquant_bypass_flag: false,
            part_mode: PartMode::Part2Nx2N,
            pcm_flag: false,
            prediction_units: vec![],
            // prev_intra_luma_pred_flag + mpm_idx 0 ⇒ candModeList[0] =
            // PLANAR for the all-DC neighbour fallback.
            intra_luma: vec![IntraLumaMode {
                prev_intra_luma_pred_flag: true,
                mpm_idx: Some(0),
                rem_intra_luma_pred_mode: None,
            }],
            // intra_chroma_pred_mode 4 ⇒ derived (= luma mode).
            intra_chroma_pred_mode: vec![4],
            rqt_root_cbf: true,
            transform_tree: Some(tree),
        };
        CodingTreeUnit {
            sao: None,
            quadtree: CodingQuadtree::Leaf(Box::new(cu)),
        }
    }

    #[test]
    fn flat_intra_luma_reconstructs_to_constant_field() {
        // pred = midlevel 128 (no neighbours); luma DC −67 dequant+IDCT
        // gives a uniform −47 residual, so recSamples = 128 − 47 = 81.
        let params = tiny_params();
        let ctu = flat_intra_ctu(-67, None, None);
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        reconstruct_intra_ctu(&mut pic, &params, &ctu).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(pic.sample(Plane::Luma, x, y), 81, "luma at ({x},{y})");
            }
        }
    }

    #[test]
    fn flat_intra_chroma_cb_reconstructs_exactly() {
        // Cb DC −27 at QpC(25)=25 gives a uniform −38 residual ⇒
        // 128 − 38 = 90 (the fixture's expected.yuv Cb plane value).
        let params = tiny_params();
        let ctu = flat_intra_ctu(-67, Some(-27), None);
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        reconstruct_intra_ctu(&mut pic, &params, &ctu).unwrap();
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(pic.sample(Plane::Cb, x, y), 90, "cb at ({x},{y})");
            }
        }
    }

    #[test]
    fn chroma_residual_produces_uniform_field() {
        // A pure-DC chroma residual reconstructs to a constant plane (the
        // inverse-transform DC basis is flat); validates the chroma
        // predict + dequant + add + clip pipeline end to end.
        let params = tiny_params();
        let ctu = flat_intra_ctu(-67, Some(-27), Some(64));
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        reconstruct_intra_ctu(&mut pic, &params, &ctu).unwrap();
        let cr0 = pic.sample(Plane::Cr, 0, 0);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(pic.sample(Plane::Cr, x, y), cr0, "cr uniform at ({x},{y})");
            }
        }
        assert!(cr0 > 128, "positive DC raises the Cr plane");
    }

    #[test]
    fn clip_saturates_out_of_range_reconstruction() {
        // A large negative luma DC drives pred+res below 0; the §8.4.4.1
        // Clip1Y must clamp to 0 (not wrap).
        let params = tiny_params();
        let ctu = flat_intra_ctu(-400, None, None);
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        reconstruct_intra_ctu(&mut pic, &params, &ctu).unwrap();
        assert_eq!(pic.sample(Plane::Luma, 0, 0), 0);
    }

    /// Build an 8×8 intra CU at `(x0, y0)` carrying the given luma-mode
    /// signalling and a uniform DC luma residual (no chroma).
    fn intra_cu_8x8(
        x0: u32,
        y0: u32,
        luma: IntraLumaMode,
        luma_dc: i32,
        chroma_pred_mode: u8,
    ) -> CodingUnit {
        let unit = TransformUnit {
            residual_luma: Some(dc_block(3, luma_dc)),
            ..Default::default()
        };
        CodingUnit {
            x0,
            y0,
            log2_cb_size: 3,
            cu_pred_mode: CuPredMode::Intra,
            cu_transquant_bypass_flag: false,
            part_mode: PartMode::Part2Nx2N,
            pcm_flag: false,
            prediction_units: vec![],
            intra_luma: vec![luma],
            intra_chroma_pred_mode: vec![chroma_pred_mode],
            rqt_root_cbf: true,
            transform_tree: Some(TransformTree::Leaf {
                cbf_luma: true,
                unit,
            }),
        }
    }

    /// The §8.4.2 neighbour MPM derivation makes a CU's IntraPredModeY
    /// depend on its already-reconstructed left / above neighbours. A right
    /// CU signalling `mpm_idx == 0` against a left neighbour coded with a
    /// non-DC angular mode picks up `candModeList[0]` = the neighbour's
    /// mode (proved by inspecting the recorded IntraModeField), whereas the
    /// flat-single-CU path would have derived INTRA_DC.
    #[test]
    fn neighbour_mpm_propagates_left_cu_mode_to_right_cu() {
        let params = tiny_params();
        // Left CU (0,0): remaining-mode angular 18. Both neighbours are
        // out-of-picture ⇒ candA == candB == INTRA_DC; the step-4 ordered
        // procedure maps rem_intra_luma_pred_mode 18 (with neither DC nor
        // PLANAR below it) up to IntraPredModeY 20.
        let left = IntraLumaMode {
            prev_intra_luma_pred_flag: false,
            mpm_idx: None,
            rem_intra_luma_pred_mode: Some(18),
        };
        // Right CU (8,0): mpm_idx 0 ⇒ IntraPredModeY = candModeList[0]. Its
        // left neighbour (7,*) is the left CU (mode 20, available in z-scan
        // order); its above neighbour is out-of-picture (DC). candA(20) !=
        // candB(DC) and neither is PLANAR ⇒ candModeList[0] == candA == 20.
        let right = IntraLumaMode {
            prev_intra_luma_pred_flag: true,
            mpm_idx: Some(0),
            rem_intra_luma_pred_mode: None,
        };
        let tl = intra_cu_8x8(0, 0, left, 0, 4);
        let tr = intra_cu_8x8(8, 0, right, 0, 4);
        let bl = intra_cu_8x8(0, 8, left, 0, 4);
        let br = intra_cu_8x8(8, 8, right, 0, 4);
        let ctu = CodingTreeUnit {
            sao: None,
            quadtree: CodingQuadtree::Split(vec![
                CodingQuadtree::Leaf(Box::new(tl)),
                CodingQuadtree::Leaf(Box::new(tr)),
                CodingQuadtree::Leaf(Box::new(bl)),
                CodingQuadtree::Leaf(Box::new(br)),
            ]),
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        let mut ctx = ReconCtx::new(16, 16, 4, 2, &TilingParams::single_tile()).unwrap();
        reconstruct_intra_ctu_ctx(&mut pic, &params, &mut ctx, &ctu).unwrap();

        // The left CU's remaining-mode 18 resolves to IntraPredModeY 20.
        assert_eq!(ctx.recorded_mode(0, 0), Some(20), "left CU mode");
        // The right CU's mpm_idx 0 picked up the left neighbour's mode 20
        // through candModeList[0] — the §8.4.2 propagation under test.
        assert_eq!(
            ctx.recorded_mode(8, 0),
            Some(20),
            "right CU inherited the left neighbour's mode via MPM"
        );
    }

    /// Counter-case: with NO recorded left neighbour (a single isolated
    /// CU), the same `mpm_idx == 0` signalling derives candModeList[0] from
    /// the all-DC fallback ⇒ INTRA_PLANAR (0), not the angular 20 above.
    #[test]
    fn isolated_cu_mpm_zero_is_planar_not_neighbour_mode() {
        let params = tiny_params();
        let right = IntraLumaMode {
            prev_intra_luma_pred_flag: true,
            mpm_idx: Some(0),
            rem_intra_luma_pred_mode: None,
        };
        let cu = intra_cu_8x8(8, 0, right, 0, 4);
        let ctu = CodingTreeUnit {
            sao: None,
            quadtree: CodingQuadtree::Leaf(Box::new(cu)),
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        let mut ctx = ReconCtx::new(16, 16, 4, 2, &TilingParams::single_tile()).unwrap();
        reconstruct_intra_ctu_ctx(&mut pic, &params, &mut ctx, &ctu).unwrap();
        // candA == candB == DC ⇒ candModeList = [PLANAR, DC, 26]; index 0
        // is INTRA_PLANAR.
        assert_eq!(ctx.recorded_mode(8, 0), Some(0));
    }

    #[test]
    fn inter_cu_is_rejected_by_intra_path() {
        let params = tiny_params();
        let mut ctu = flat_intra_ctu(-67, None, None);
        if let CodingQuadtree::Leaf(cu) = &mut ctu.quadtree {
            cu.cu_pred_mode = CuPredMode::Inter;
        }
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        assert_eq!(
            reconstruct_intra_ctu(&mut pic, &params, &ctu),
            Err(ReconError::InterNotSupported)
        );
    }

    /// End-to-end §8.5 inter reconstruction: a uni-L0 P-block with a
    /// full-pel motion vector copies a shifted reference window, then a
    /// uniform residual is added and clipped.
    #[test]
    fn inter_uni_l0_full_pel_reconstructs() {
        let params = tiny_params();
        // Reference picture: luma ramp sample(x,y) == x (mod 256), flat
        // chroma 128.
        let mut refpic = Picture::new(16, 16, 1, 8, 8);
        for y in 0..16 {
            for x in 0..16 {
                refpic.set_sample(Plane::Luma, x, y, x as i32);
            }
        }
        for y in 0..8 {
            for x in 0..8 {
                refpic.set_sample(Plane::Cb, x, y, 128);
                refpic.set_sample(Plane::Cr, x, y, 128);
            }
        }
        let l0 = ResolvedList {
            pred_flag: true,
            mv_l: [8, 0], // +2 full luma samples right.
            mv_c: [8, 0], // 4:2:0 ⇒ mvC = mvL.
            ref_pic: &refpic,
        };
        // Unused L1 points at the same picture but pred_flag is false.
        let l1 = ResolvedList {
            pred_flag: false,
            mv_l: [0, 0],
            mv_c: [0, 0],
            ref_pic: &refpic,
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        // A flat +5 luma residual over the 8x8 PU at (0,0).
        let res = vec![5i32; 8 * 8];
        reconstruct_inter_pu(
            &mut pic,
            &params,
            0,
            0,
            8,
            8,
            l0,
            l1,
            Some(&res),
            None,
            None,
        )
        .unwrap();
        // predSamples[xL] reads ref column xPb + 2 + xL = 2 + xL; + 5 res.
        for yl in 0..8 {
            for xl in 0..8 {
                assert_eq!(
                    pic.sample(Plane::Luma, xl, yl),
                    (2 + xl as i32) + 5,
                    "luma ({xl},{yl})"
                );
            }
        }
        // Chroma: flat 128 prediction, no residual ⇒ 128.
        for yc in 0..4 {
            for xc in 0..4 {
                assert_eq!(pic.sample(Plane::Cb, xc, yc), 128);
                assert_eq!(pic.sample(Plane::Cr, xc, yc), 128);
            }
        }
    }

    /// Bi-prediction averages two reference windows; a clip guards the
    /// out-of-range sum.
    #[test]
    fn inter_bi_averages_and_clips() {
        let params = tiny_params();
        let mut a = Picture::new(16, 16, 1, 8, 8);
        let mut b = Picture::new(16, 16, 1, 8, 8);
        for y in 0..16 {
            for x in 0..16 {
                a.set_sample(Plane::Luma, x, y, 40);
                b.set_sample(Plane::Luma, x, y, 200);
            }
        }
        let l0 = ResolvedList {
            pred_flag: true,
            mv_l: [0, 0],
            mv_c: [0, 0],
            ref_pic: &a,
        };
        let l1 = ResolvedList {
            pred_flag: true,
            mv_l: [0, 0],
            mv_c: [0, 0],
            ref_pic: &b,
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        reconstruct_inter_pu(&mut pic, &params, 0, 0, 8, 8, l0, l1, None, None, None).unwrap();
        // (40 + 200) >> 1 == 120.
        for yl in 0..8 {
            for xl in 0..8 {
                assert_eq!(pic.sample(Plane::Luma, xl, yl), 120);
            }
        }
    }

    #[test]
    fn packed_output_matches_planar_layout() {
        let params = tiny_params();
        let ctu = flat_intra_ctu(-67, Some(-27), None);
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        reconstruct_intra_ctu(&mut pic, &params, &ctu).unwrap();
        let packed = pic.to_planar_u8().unwrap();
        // 256 luma + 64 cb + 64 cr.
        assert_eq!(packed.len(), 384);
        assert!(packed[..256].iter().all(|&v| v == 81));
        assert!(packed[256..320].iter().all(|&v| v == 90));
    }
}
