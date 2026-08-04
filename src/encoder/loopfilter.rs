//! Encoder-side §8.7 in-loop filters — deblocking + SAO on the
//! encoder's reconstruction path.
//!
//! A conforming decoder's reference pictures are the *filtered*
//! reconstructions, so an encoder that signals the loop filters must
//! run the very same §8.7.2 deblocking and §8.7.3 SAO over its own
//! reconstruction before using it as a reference (and before claiming
//! it as "what the decoder outputs"). This module drives the crate's
//! DECODE-side filter implementations ([`crate::deblock`] /
//! [`crate::sao`]) over an encoder frame:
//!
//! * **Deblocking** (§8.7.2) — applied through
//!   [`crate::deblock::deblock_picture_full`] with per-CTB
//!   [`DeblockCuDesc`]s built from the encoder's coding decisions
//!   (partition mode + transform-split topology) and the same
//!   [`MotionField`] the mode decisions maintained. The per-frame
//!   election is distortion-driven over off + a 3x3
//!   `slice_beta_offset_div2` / `slice_tc_offset_div2` sweep
//!   ({−2, 0, 2}²): the slice signals
//!   `deblocking_filter_override_flag == 1` /
//!   `slice_deblocking_filter_disabled_flag == 0` with the winning
//!   offsets only when some filtered picture is closer to the source.
//! * **SAO** (§8.7.3) — per-CTB statistics-driven offset estimation
//!   (the derivation of the offsets is encoder freedom; the *applied*
//!   modification is the decoder's own
//!   [`crate::sao::apply_sao_ctb_full`], so every candidate's
//!   distortion is measured with the exact decode-side arithmetic).
//!   Candidates per component: off, band offset (best
//!   `sao_band_position` by per-band gain), and the four edge-offset
//!   classes; chroma obeys the §7.4.9.3 shared-type rule (`SaoTypeIdx
//!   [2]` / `SaoEoClass[2]` inherit cIdx 1). Whole-CTB
//!   `sao_merge_left_flag` / `sao_merge_up_flag` candidates are priced
//!   too. The elected parameters are applied with
//!   [`crate::sao::apply_sao_picture_full`] — the same picture-level
//!   driver the decoder runs.
//! * **Syntax** — [`encode_sao_ctb`] is the bin-exact §7.3.8.3
//!   `sao( rx, ry )` dual of [`crate::slice_data::decode_sao`].
//!
//! Geometry contract (shared with the intra / low-delay encoders):
//! `CtbSizeY == 16`, one coding unit per CTB, 4:2:0 8-bit, a single
//! slice and tile per picture, constant QP (no `cu_qp_delta`), no PCM
//! / transquant-bypass CUs (so the §8.7 `NoFilterMap` is empty).

use crate::binarization::PartMode;
use crate::ctx_init::SliceContexts;
use crate::deblock::{deblock_picture_full, DeblockCu, DeblockCuDesc, DeblockCuParams};
use crate::encoder::bitwriter::BitWriter;
use crate::encoder::cabac::CabacEncoder;
use crate::motion::MotionField;
use crate::picture::{Picture, Plane};
use crate::sao::{apply_sao_ctb_full, apply_sao_picture_full, ResolvedSao, ResolvedSaoComponent};
use crate::slice_data::{SaoComponent, SaoCtbParams};

/// The fixed CTB log2 size of the bootstrap encoders (16x16).
const CTB_LOG2: u32 = 4;
/// The fixed CTB size.
const CTB: usize = 1 << CTB_LOG2;
/// §7.3.8.3 `sao_offset_abs` cMax at 8-bit:
/// `(1 << (Min(bitDepth, 10) − 5)) − 1`.
const SAO_OFFSET_MAX: i32 = 7;

/// Which in-loop filters the encoder signals and applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LoopFilterCfg {
    /// Enable the §8.7.2 deblocking filter (PPS
    /// `deblocking_filter_override_enabled_flag == 1`; each slice
    /// elects on/off against distortion).
    pub deblocking: bool,
    /// Enable luma SAO (SPS `sample_adaptive_offset_enabled_flag == 1`
    /// + per-slice `slice_sao_luma_flag` election).
    pub sao_luma: bool,
    /// Enable chroma SAO (`slice_sao_chroma_flag` election).
    pub sao_chroma: bool,
}

impl LoopFilterCfg {
    /// No in-loop filters (the pre-round-429 encoder behaviour).
    #[must_use]
    pub fn off() -> Self {
        Self::default()
    }

    /// Deblocking + luma and chroma SAO.
    #[must_use]
    pub fn all() -> Self {
        Self {
            deblocking: true,
            sao_luma: true,
            sao_chroma: true,
        }
    }

    /// `true` when SAO is enabled for any component (drives the SPS
    /// `sample_adaptive_offset_enabled_flag`).
    #[must_use]
    pub fn sao(&self) -> bool {
        self.sao_luma || self.sao_chroma
    }

    /// `true` when any filter is enabled.
    #[must_use]
    pub fn any(&self) -> bool {
        self.deblocking || self.sao()
    }
}

/// One CTB-sized coding unit's deblocking shape (the encoder-side
/// counterpart of what the decoder reads off the parsed CU).
#[derive(Debug, Clone, Copy)]
pub struct CtbShape {
    /// The CU's prediction partition mode.
    pub part_mode: PartMode,
    /// `true` when the CU's transform tree is the single-level split
    /// (four depth-1 leaves: the §7.4.9.8 `IntraSplitFlag` /
    /// `interSplitFlag` forced split); `false` for a whole-CB leaf
    /// (depth-0 TU, skip, or `rqt_root_cbf == 0`).
    pub split_depth1: bool,
}

/// Build the per-CTB [`DeblockCuDesc`] list for a fixed-geometry
/// encoder picture (CTB == CU == 16, raster order, single slice /
/// tile, constant QP) at the given slice β/tC offsets.
#[must_use]
pub(crate) fn ctb_deblock_descs(
    shapes: &[CtbShape],
    width: usize,
    height: usize,
    qp: i32,
    beta_offset_div2: i32,
    tc_offset_div2: i32,
) -> Vec<DeblockCuDesc> {
    let ctbs_x = width / CTB;
    let ctbs_y = height / CTB;
    debug_assert_eq!(shapes.len(), ctbs_x * ctbs_y);
    let params = DeblockCuParams {
        qp_y: qp,
        beta_offset_div2,
        tc_offset_div2,
        cb_qp_offset: 0,
        cr_qp_offset: 0,
        bit_depth_luma: 8,
        bit_depth_chroma: 8,
        chroma_array_type: 1,
    };
    shapes
        .iter()
        .enumerate()
        .map(|(ctb, shape)| {
            let x0 = (ctb % ctbs_x) * CTB;
            let y0 = (ctb / ctbs_x) * CTB;
            DeblockCuDesc {
                cu: DeblockCu {
                    x_cb: x0,
                    y_cb: y0,
                    log2_cb_size: CTB_LOG2,
                    params,
                    // Constant QP: the p-side scalars equal the CU QP.
                    qp_y_p_left: qp,
                    qp_y_p_top: qp,
                },
                transform_split: if shape.split_depth1 {
                    crate::deblock::TransformSplit::split_once()
                } else {
                    crate::deblock::TransformSplit::leaf()
                },
                part_mode: shape.part_mode,
                // §8.7.2.1: picture-boundary edges are never filtered;
                // single slice / tile, so no other exclusions.
                filter_left: x0 > 0,
                filter_top: y0 > 0,
            }
        })
        .collect()
}

/// The borrowed inputs of one frame's filter pass.
pub(crate) struct FilterInput<'a> {
    /// Picture width in luma samples (multiple of 16).
    pub width: usize,
    /// Picture height in luma samples.
    pub height: usize,
    /// `SliceQpY`.
    pub qp: i32,
    /// The SSD-per-bin λ of the slice's mode decisions.
    pub lambda: u64,
    /// Pre-filter reconstruction planes `[y, cb, cr]`.
    pub recon: [&'a [u8]; 3],
    /// Source planes `[y, cb, cr]`.
    pub src: [&'a [u8]; 3],
    /// The picture's motion / mode field (§8.7.2.4 bS input).
    pub field: &'a MotionField,
    /// Per-CTB coding shapes (raster order).
    pub shapes: &'a [CtbShape],
}

/// One frame's elected filter signalling + the filtered reconstruction.
pub(crate) struct FilteredFrame {
    /// The slice's deblocking election
    /// (`slice_deblocking_filter_disabled_flag == !deblock_on`).
    pub deblock_on: bool,
    /// Elected `slice_beta_offset_div2` (meaningful iff
    /// [`Self::deblock_on`]).
    pub beta_offset_div2: i32,
    /// Elected `slice_tc_offset_div2`.
    pub tc_offset_div2: i32,
    /// Elected `slice_sao_luma_flag`.
    pub slice_sao_luma: bool,
    /// Elected `slice_sao_chroma_flag`.
    pub slice_sao_chroma: bool,
    /// Per-CTB §7.3.8.3 syntax values (raster order; empty when both
    /// slice SAO flags are 0).
    pub sao_ctbs: Vec<SaoCtbParams>,
    /// The filtered reconstruction (what a conforming decoder outputs
    /// and stores as a reference).
    pub y: Vec<u8>,
    /// Filtered Cb plane.
    pub cb: Vec<u8>,
    /// Filtered Cr plane.
    pub cr: Vec<u8>,
}

/// Pack three u8 planes into a [`Picture`].
fn planes_to_picture(y: &[u8], cb: &[u8], cr: &[u8], width: usize, height: usize) -> Picture {
    let mut pic = Picture::new(width, height, 1, 8, 8);
    for (plane, data) in [(Plane::Luma, y), (Plane::Cb, cb), (Plane::Cr, cr)] {
        let (buf, _stride) = pic.plane_mut(plane);
        debug_assert_eq!(buf.len(), data.len());
        for (dst, &src) in buf.iter_mut().zip(data.iter()) {
            *dst = i32::from(src);
        }
    }
    pic
}

/// SSD of one whole plane against a u8 source plane.
fn plane_ssd(pic: &Picture, plane: Plane, src: &[u8]) -> u64 {
    pic.plane(plane)
        .iter()
        .zip(src.iter())
        .map(|(&a, &b)| {
            let d = a - i32::from(b);
            (d * d) as u64
        })
        .sum()
}

/// SSD of a `w`x`h` region of a picture plane against the matching
/// region of a u8 source plane (`plane_w` wide).
#[allow(clippy::too_many_arguments)]
fn region_ssd(
    pic: &Picture,
    plane: Plane,
    src: &[u8],
    plane_w: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> u64 {
    let mut acc = 0u64;
    for j in 0..h {
        for i in 0..w {
            let d = pic.sample(plane, x0 + i, y0 + j) - i32::from(src[(y0 + j) * plane_w + x0 + i]);
            acc += (d * d) as u64;
        }
    }
    acc
}

/// Copy a `w`x`h` region of `from` into `to` (same plane geometry).
fn restore_region(
    to: &mut Picture,
    from: &Picture,
    plane: Plane,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) {
    for j in 0..h {
        for i in 0..w {
            to.set_sample(plane, x0 + i, y0 + j, from.sample(plane, x0 + i, y0 + j));
        }
    }
}

/// Round-half-away-from-zero integer division for the offset means.
fn round_div(sum: i64, cnt: i64) -> i32 {
    if cnt == 0 {
        0
    } else {
        let bias = if sum >= 0 { cnt } else { -cnt };
        ((2 * sum + bias) / (2 * cnt)) as i32
    }
}

/// §9.3.3 TR (cMax 7, bypass) bin count of one `sao_offset_abs` value.
fn tr7_bins(v: u32) -> u64 {
    if v as i32 >= SAO_OFFSET_MAX {
        SAO_OFFSET_MAX as u64
    } else {
        u64::from(v) + 1
    }
}

/// Rate proxy (bins) of one component's §7.3.8.3 fields. `with_shared`
/// counts the `sao_type_idx_*` (and, for edge offset, `sao_eo_class_*`)
/// bins — true for cIdx 0 / 1, false for cIdx 2 (which inherits both
/// per §7.4.9.3).
fn component_rate(comp: &SaoComponent, with_shared: bool) -> u64 {
    match comp.sao_type_idx {
        0 => u64::from(with_shared),
        1 => {
            let offs: u64 = comp.offset_abs.iter().map(|&a| tr7_bins(a)).sum();
            let signs = comp.offset_abs.iter().filter(|&&a| a != 0).count() as u64;
            u64::from(with_shared) * 2 + offs + signs + 5
        }
        _ => {
            let offs: u64 = comp.offset_abs.iter().map(|&a| tr7_bins(a)).sum();
            u64::from(with_shared) * 2 + offs + u64::from(with_shared) * 2
        }
    }
}

/// Exact distortion of applying `comp` (already §7.4.9.3-resolved) to
/// one CTB region of one plane: runs the decoder's own
/// [`apply_sao_ctb_full`] into `scratch` (which must equal `base`
/// outside the region on entry) and restores the region afterwards.
#[allow(clippy::too_many_arguments)]
fn eval_component(
    base: &Picture,
    scratch: &mut Picture,
    plane: Plane,
    comp: &ResolvedSaoComponent,
    src: &[u8],
    plane_w: usize,
    x0: usize,
    y0: usize,
    n: usize,
) -> u64 {
    if comp.sao_type_idx == 0 {
        return region_ssd(base, plane, src, plane_w, x0, y0, n, n);
    }
    apply_sao_ctb_full(base, scratch, plane, comp, x0, y0, n, n, None, None);
    let d = region_ssd(scratch, plane, src, plane_w, x0, y0, n, n);
    restore_region(scratch, base, plane, x0, y0, n, n);
    d
}

/// Statistics-driven §8.7.3.2 edge-offset candidate for one component
/// region: classify every sample against the pre-SAO picture (the
/// §8.7.3.2 picture-boundary guard included), then derive the four
/// per-category offsets from the src−rec means (categories 0/1
/// non-negative, 2/3 non-positive per the §7.4.9.3 inferred signs).
#[allow(clippy::too_many_arguments)]
fn edge_candidate(
    base: &Picture,
    plane: Plane,
    src: &[u8],
    plane_w: usize,
    x0: usize,
    y0: usize,
    n: usize,
    eo_class: u8,
) -> SaoComponent {
    let (h0, v0, h1, v1) = crate::sao::eo_pos(eo_class);
    let (pw, ph) = base.plane_dims(plane);
    let mut cnt = [0i64; 5];
    let mut sum = [0i64; 5];
    for j in 0..n {
        for i in 0..n {
            let (x, y) = ((x0 + i) as i32, (y0 + j) as i32);
            let in_pic =
                |xx: i32, yy: i32| xx >= 0 && yy >= 0 && (xx as usize) < pw && (yy as usize) < ph;
            if !in_pic(x + h0, y + v0) || !in_pic(x + h1, y + v1) {
                continue;
            }
            let cur = base.sample(plane, x as usize, y as usize);
            let s0 = base.sample(plane, (x + h0) as usize, (y + v0) as usize);
            let s1 = base.sample(plane, (x + h1) as usize, (y + v1) as usize);
            let sign = |v: i32| -> i32 {
                match v.cmp(&0) {
                    core::cmp::Ordering::Greater => 1,
                    core::cmp::Ordering::Equal => 0,
                    core::cmp::Ordering::Less => -1,
                }
            };
            let mut edge_idx = 2 + sign(cur - s0) + sign(cur - s1);
            if edge_idx <= 2 {
                edge_idx = if edge_idx == 2 { 0 } else { edge_idx + 1 };
            }
            if edge_idx == 0 {
                continue;
            }
            cnt[edge_idx as usize] += 1;
            sum[edge_idx as usize] +=
                i64::from(src[(y as usize) * plane_w + x as usize]) - i64::from(cur);
        }
    }
    let mut offset_abs = [0u32; 4];
    for i in 0..4 {
        let raw = round_div(sum[i + 1], cnt[i + 1]);
        // §7.4.9.3 inferred signs: categories 1/2 add, 3/4 subtract.
        let clamped = if i < 2 {
            raw.clamp(0, SAO_OFFSET_MAX)
        } else {
            raw.clamp(-SAO_OFFSET_MAX, 0)
        };
        offset_abs[i] = clamped.unsigned_abs();
    }
    SaoComponent {
        sao_type_idx: 2,
        offset_abs,
        offset_sign: [0; 4],
        band_position: 0,
        eo_class,
    }
}

/// Statistics-driven §8.7.3.2 band-offset candidate: per-band src−rec
/// means, then the `sao_band_position` whose four consecutive bands
/// carry the largest estimated SSD gain.
#[allow(clippy::too_many_arguments)]
fn band_candidate(
    base: &Picture,
    plane: Plane,
    src: &[u8],
    plane_w: usize,
    x0: usize,
    y0: usize,
    n: usize,
) -> SaoComponent {
    let mut cnt = [0i64; 32];
    let mut sum = [0i64; 32];
    for j in 0..n {
        for i in 0..n {
            let cur = base.sample(plane, x0 + i, y0 + j);
            let band = (cur >> 3) as usize & 31;
            cnt[band] += 1;
            sum[band] += i64::from(src[(y0 + j) * plane_w + x0 + i]) - i64::from(cur);
        }
    }
    let mut offset = [0i32; 32];
    let mut gain = [0i64; 32];
    for b in 0..32 {
        let o = round_div(sum[b], cnt[b]).clamp(-SAO_OFFSET_MAX, SAO_OFFSET_MAX);
        // Estimated SSD reduction: 2·o·Σdiff − o²·N (clipping ignored —
        // the exact election re-measures with the decode-side apply).
        let g = 2 * i64::from(o) * sum[b] - i64::from(o) * i64::from(o) * cnt[b];
        if g > 0 {
            offset[b] = o;
            gain[b] = g;
        }
    }
    let best_pos = (0..32u8)
        .max_by_key(|&p| (0..4).map(|k| gain[(p as usize + k) & 31]).sum::<i64>())
        .unwrap_or(0);
    let mut offset_abs = [0u32; 4];
    let mut offset_sign = [0u8; 4];
    for k in 0..4 {
        let o = offset[(best_pos as usize + k) & 31];
        offset_abs[k] = o.unsigned_abs();
        offset_sign[k] = u8::from(o < 0);
    }
    SaoComponent {
        sao_type_idx: 1,
        offset_abs,
        offset_sign,
        band_position: best_pos,
        eo_class: 0,
    }
}

/// Resolve a syntax component into the applied form (8-bit: no
/// range-extension offset scale).
fn resolved(comp: &SaoComponent) -> ResolvedSaoComponent {
    ResolvedSaoComponent::from_decoded(comp, 0)
}

/// Elect one CTB's luma SAO parameters: off vs band vs the four edge
/// classes, each measured with the decode-side apply. Returns the
/// syntax component and its `dist + λ·rate` cost (rate including the
/// component's own §7.3.8.3 bins).
#[allow(clippy::too_many_arguments)]
fn choose_luma(
    base: &Picture,
    scratch: &mut Picture,
    src: &[u8],
    width: usize,
    x0: usize,
    y0: usize,
    lambda: u64,
) -> (SaoComponent, u64) {
    let off = SaoComponent::default();
    let mut best = (
        off,
        region_ssd(base, Plane::Luma, src, width, x0, y0, CTB, CTB) + lambda,
    );
    let mut consider = |comp: SaoComponent, scratch: &mut Picture| {
        let d = eval_component(
            base,
            scratch,
            Plane::Luma,
            &resolved(&comp),
            src,
            width,
            x0,
            y0,
            CTB,
        );
        let cost = d + lambda * component_rate(&comp, true);
        if cost < best.1 {
            best = (comp, cost);
        }
    };
    consider(
        band_candidate(base, Plane::Luma, src, width, x0, y0, CTB),
        scratch,
    );
    for eo_class in 0..4u8 {
        consider(
            edge_candidate(base, Plane::Luma, src, width, x0, y0, CTB, eo_class),
            scratch,
        );
    }
    best
}

/// Elect one CTB's chroma SAO parameters under the §7.4.9.3 shared
/// type / eo-class rule: one `SaoTypeIdx` (and edge class) for both
/// Cb and Cr, per-component offsets / band positions.
#[allow(clippy::too_many_arguments)]
fn choose_chroma(
    base: &Picture,
    scratch: &mut Picture,
    src_cb: &[u8],
    src_cr: &[u8],
    cw: usize,
    cx0: usize,
    cy0: usize,
    lambda: u64,
) -> ([SaoComponent; 2], u64) {
    let n = CTB / 2;
    let dist_pair = |cb: &SaoComponent, cr: &SaoComponent, scratch: &mut Picture| -> u64 {
        eval_component(
            base,
            scratch,
            Plane::Cb,
            &resolved(cb),
            src_cb,
            cw,
            cx0,
            cy0,
            n,
        ) + eval_component(
            base,
            scratch,
            Plane::Cr,
            &resolved(cr),
            src_cr,
            cw,
            cx0,
            cy0,
            n,
        )
    };
    let off = SaoComponent::default();
    let mut best = ([off, off], dist_pair(&off, &off, scratch) + lambda);
    let mut consider = |cb: SaoComponent, cr: SaoComponent, scratch: &mut Picture| {
        let d = dist_pair(&cb, &cr, scratch);
        let rate = component_rate(&cb, true) + component_rate(&cr, false);
        let cost = d + lambda * rate;
        if cost < best.1 {
            best = ([cb, cr], cost);
        }
    };
    consider(
        band_candidate(base, Plane::Cb, src_cb, cw, cx0, cy0, n),
        band_candidate(base, Plane::Cr, src_cr, cw, cx0, cy0, n),
        scratch,
    );
    for eo_class in 0..4u8 {
        consider(
            edge_candidate(base, Plane::Cb, src_cb, cw, cx0, cy0, n, eo_class),
            edge_candidate(base, Plane::Cr, src_cr, cw, cx0, cy0, n, eo_class),
            scratch,
        );
    }
    best
}

/// Distortion of applying a fully-resolved CTB parameter set (all
/// three components) — the merge-candidate measurement.
#[allow(clippy::too_many_arguments)]
fn resolved_ctb_dist(
    base: &Picture,
    scratch: &mut Picture,
    r: &ResolvedSao,
    input: &FilterInput<'_>,
    x0: usize,
    y0: usize,
    sao_luma: bool,
    sao_chroma: bool,
) -> u64 {
    let (cw, cx0, cy0, n) = (input.width / 2, x0 / 2, y0 / 2, CTB / 2);
    let mut d = 0u64;
    let luma = if sao_luma {
        &r.components[0]
    } else {
        &ResolvedSaoComponent::off()
    };
    d += eval_component(
        base,
        scratch,
        Plane::Luma,
        luma,
        input.src[0],
        input.width,
        x0,
        y0,
        CTB,
    );
    let (cbc, crc) = if sao_chroma {
        (&r.components[1], &r.components[2])
    } else {
        (&ResolvedSaoComponent::off(), &ResolvedSaoComponent::off())
    };
    d += eval_component(base, scratch, Plane::Cb, cbc, input.src[1], cw, cx0, cy0, n);
    d += eval_component(base, scratch, Plane::Cr, crc, input.src[2], cw, cx0, cy0, n);
    d
}

/// Run the elected in-loop filters over one encoder frame.
///
/// Returns the filter signalling (per-slice deblocking election, the
/// slice SAO flags, per-CTB SAO syntax) and the filtered
/// reconstruction — exactly what a conforming decoder reconstructs
/// from that signalling.
pub(crate) fn filter_frame(input: &FilterInput<'_>, cfg: &LoopFilterCfg) -> FilteredFrame {
    let (width, height) = (input.width, input.height);
    let pre = planes_to_picture(
        input.recon[0],
        input.recon[1],
        input.recon[2],
        width,
        height,
    );

    // ---- §8.7.2 deblocking: per-slice on/off + β/tC election ----
    let mut deblock_on = false;
    let mut beta_offset_div2 = 0i32;
    let mut tc_offset_div2 = 0i32;
    let base = if cfg.deblocking {
        let ssd = |p: &Picture| {
            plane_ssd(p, Plane::Luma, input.src[0])
                + plane_ssd(p, Plane::Cb, input.src[1])
                + plane_ssd(p, Plane::Cr, input.src[2])
        };
        // se(v) bin-length proxy of one slice offset field.
        let se_bits = |v: i32| -> u64 {
            match v.unsigned_abs() {
                0 => 1,
                n => 2 * u64::from(32 - n.leading_zeros()) + 1,
            }
        };
        // Off (override_flag 0: one bit) vs each (β, tC) offset pair
        // (override group: ~3 bits + the two se(v) fields).
        let mut best_pic = pre.clone();
        let mut best_cost = ssd(&pre) + input.lambda;
        for beta in [-2i32, 0, 2] {
            for tc in [-2i32, 0, 2] {
                let mut filtered = pre.clone();
                let descs = ctb_deblock_descs(input.shapes, width, height, input.qp, beta, tc);
                deblock_picture_full(&mut filtered, input.field, &descs, None, None);
                let cost = ssd(&filtered) + input.lambda * (3 + se_bits(beta) + se_bits(tc));
                if cost < best_cost {
                    best_cost = cost;
                    best_pic = filtered;
                    deblock_on = true;
                    beta_offset_div2 = beta;
                    tc_offset_div2 = tc;
                }
            }
        }
        best_pic
    } else {
        pre
    };

    // ---- §8.7.3 SAO estimation (per CTB, on the deblocked picture) ----
    let ctbs_x = width / CTB;
    let ctbs_y = height / CTB;
    let mut sao_ctbs: Vec<SaoCtbParams> = Vec::new();
    let mut grid: Vec<ResolvedSao> = Vec::new();
    if cfg.sao() {
        let mut scratch = base.clone();
        sao_ctbs.reserve(ctbs_x * ctbs_y);
        grid.reserve(ctbs_x * ctbs_y);
        for ctb in 0..ctbs_x * ctbs_y {
            let (rx, ry) = (ctb % ctbs_x, ctb / ctbs_x);
            let (x0, y0) = (rx * CTB, ry * CTB);
            let (cx0, cy0) = (x0 / 2, y0 / 2);

            // Explicit candidate: per-component election.
            let mut comps = [SaoComponent::default(); 3];
            let mut explicit_cost = 0u64;
            if cfg.sao_luma {
                let (c, cost) = choose_luma(
                    &base,
                    &mut scratch,
                    input.src[0],
                    width,
                    x0,
                    y0,
                    input.lambda,
                );
                comps[0] = c;
                explicit_cost += cost;
            } else {
                explicit_cost +=
                    region_ssd(&base, Plane::Luma, input.src[0], width, x0, y0, CTB, CTB);
            }
            if cfg.sao_chroma {
                let ([cb, cr], cost) = choose_chroma(
                    &base,
                    &mut scratch,
                    input.src[1],
                    input.src[2],
                    width / 2,
                    cx0,
                    cy0,
                    input.lambda,
                );
                comps[1] = cb;
                comps[2] = cr;
                explicit_cost += cost;
            } else {
                let (cw, n) = (width / 2, CTB / 2);
                explicit_cost += region_ssd(&base, Plane::Cb, input.src[1], cw, cx0, cy0, n, n)
                    + region_ssd(&base, Plane::Cr, input.src[2], cw, cx0, cy0, n, n);
            }
            // Declining the available merges costs one bin each.
            let decline_bins = u64::from(rx > 0) + u64::from(ry > 0);
            explicit_cost += input.lambda * decline_bins;

            let explicit_params = SaoCtbParams {
                merge_left: false,
                merge_up: false,
                components: comps,
            };
            let explicit_resolved = ResolvedSao::resolve(&explicit_params, None, None, 0, 0);

            // Whole-CTB merge candidates (inherit ALL components).
            let mut best = (explicit_params, explicit_resolved, explicit_cost);
            if rx > 0 {
                let left = grid[ctb - 1];
                let d = resolved_ctb_dist(
                    &base,
                    &mut scratch,
                    &left,
                    input,
                    x0,
                    y0,
                    cfg.sao_luma,
                    cfg.sao_chroma,
                );
                let cost = d + input.lambda;
                if cost < best.2 {
                    best = (
                        SaoCtbParams {
                            merge_left: true,
                            merge_up: false,
                            components: [SaoComponent::default(); 3],
                        },
                        left,
                        cost,
                    );
                }
            }
            if ry > 0 {
                let above = grid[ctb - ctbs_x];
                let d = resolved_ctb_dist(
                    &base,
                    &mut scratch,
                    &above,
                    input,
                    x0,
                    y0,
                    cfg.sao_luma,
                    cfg.sao_chroma,
                );
                let cost = d + input.lambda * (1 + u64::from(rx > 0));
                if cost < best.2 {
                    best = (
                        SaoCtbParams {
                            merge_left: false,
                            merge_up: true,
                            components: [SaoComponent::default(); 3],
                        },
                        above,
                        cost,
                    );
                }
            }

            sao_ctbs.push(best.0);
            grid.push(best.1);
        }
    }

    // ---- slice-flag election + the decode-side picture apply ----
    let slice_sao_luma = cfg.sao_luma && grid.iter().any(|r| r.components[0].sao_type_idx != 0);
    let slice_sao_chroma = cfg.sao_chroma && grid.iter().any(|r| r.components[1].sao_type_idx != 0);
    let out = if slice_sao_luma || slice_sao_chroma {
        apply_sao_picture_full(
            &base,
            &grid,
            CTB_LOG2,
            1,
            slice_sao_luma,
            slice_sao_chroma,
            None,
            None,
        )
    } else {
        sao_ctbs.clear();
        base
    };

    let planar = out.to_planar_u8().expect("8-bit planes");
    let (cw, ch) = (width / 2, height / 2);
    let y = planar[..width * height].to_vec();
    let cb = planar[width * height..width * height + cw * ch].to_vec();
    let cr = planar[width * height + cw * ch..].to_vec();
    FilteredFrame {
        deblock_on,
        beta_offset_div2,
        tc_offset_div2,
        slice_sao_luma,
        slice_sao_chroma,
        sao_ctbs,
        y,
        cb,
        cr,
    }
}

/// §7.3.8.3 `sao( rx, ry )` — the bin-exact dual of
/// [`crate::slice_data::decode_sao`] for a single-slice, single-tile
/// picture (merge-left available iff `rx > 0`, merge-up iff `ry > 0`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_sao_ctb(
    w: &mut BitWriter,
    cabac: &mut CabacEncoder,
    ctxs: &mut SliceContexts,
    params: &SaoCtbParams,
    rx: usize,
    ry: usize,
    slice_sao_luma: bool,
    slice_sao_chroma: bool,
) {
    if rx > 0 {
        cabac.encode_decision(w, &mut ctxs.sao_merge_flag[0], u8::from(params.merge_left));
        if params.merge_left {
            return;
        }
    }
    if ry > 0 {
        cabac.encode_decision(w, &mut ctxs.sao_merge_flag[0], u8::from(params.merge_up));
        if params.merge_up {
            return;
        }
    }
    for c_idx in 0..3usize {
        let read = (slice_sao_luma && c_idx == 0) || (slice_sao_chroma && c_idx > 0);
        if !read {
            continue;
        }
        let comp = &params.components[c_idx];
        if c_idx < 2 {
            // sao_type_idx_luma / _chroma: TR cMax 2 — bin 0
            // context-coded, bin 1 bypass.
            cabac.encode_decision(
                w,
                &mut ctxs.sao_type_idx[0],
                u8::from(comp.sao_type_idx > 0),
            );
            if comp.sao_type_idx > 0 {
                cabac.encode_bypass(w, u8::from(comp.sao_type_idx == 2));
            }
        } else {
            debug_assert_eq!(
                comp.sao_type_idx, params.components[1].sao_type_idx,
                "SaoTypeIdx[2] inherits cIdx 1 (§7.4.9.3)"
            );
        }
        if comp.sao_type_idx != 0 {
            for &abs in &comp.offset_abs {
                // sao_offset_abs: TR cMax 7 (8-bit), all bypass.
                for _ in 0..abs.min(SAO_OFFSET_MAX as u32) {
                    cabac.encode_bypass(w, 1);
                }
                if (abs as i32) < SAO_OFFSET_MAX {
                    cabac.encode_bypass(w, 0);
                }
            }
            if comp.sao_type_idx == 1 {
                for i in 0..4 {
                    if comp.offset_abs[i] != 0 {
                        cabac.encode_bypass(w, comp.offset_sign[i]);
                    }
                }
                cabac.encode_bypass_bits(w, u32::from(comp.band_position), 5);
            } else if c_idx < 2 {
                cabac.encode_bypass_bits(w, u32::from(comp.eo_class), 2);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;
    use crate::cabac::CabacEngine;
    use crate::slice_data::{decode_sao, SliceDataParams};

    fn sao_params(sao_luma: bool, sao_chroma: bool) -> SliceDataParams {
        SliceDataParams {
            ctb_log2_size_y: 4,
            min_cb_log2_size_y: 4,
            max_tb_log2_size_y: 4,
            min_tb_log2_size_y: 2,
            pic_width_in_luma_samples: 64,
            pic_height_in_luma_samples: 64,
            chroma_array_type: 1,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            slice_type_is_i: true,
            slice_type_is_b: false,
            slice_sao_luma_flag: sao_luma,
            slice_sao_chroma_flag: sao_chroma,
            transquant_bypass_enabled_flag: false,
            cu_qp_delta_enabled_flag: false,
            log2_min_cu_qp_delta_size: 4,
            cu_chroma_qp_offset_enabled_flag: false,
            log2_min_cu_chroma_qp_offset_size: 4,
            chroma_qp_offset_list_len_minus1: 0,
            amp_enabled_flag: false,
            pcm_enabled_flag: false,
            log2_min_ipcm_cb_size_y: 3,
            log2_max_ipcm_cb_size_y: 5,
            pcm_bit_depth_luma: 8,
            pcm_bit_depth_chroma: 8,
            max_transform_hierarchy_depth_intra: 0,
            max_transform_hierarchy_depth_inter: 0,
            max_num_merge_cand: 5,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            mvd_l1_zero_flag: false,
            sign_data_hiding_enabled_flag: false,
            cross_component_prediction_enabled_flag: false,
            residual_adaptive_colour_transform_enabled_flag: false,
            transform_skip_enabled_flag: false,
            log2_max_transform_skip_size: 2,
            implicit_rdpcm_enabled_flag: false,
            explicit_rdpcm_enabled_flag: false,
            transform_skip_context_enabled_flag: false,
            persistent_rice_adaptation_enabled_flag: false,
            cabac_bypass_alignment_enabled_flag: false,
            extended_precision_processing_flag: false,
            palette_mode_enabled_flag: false,
            palette_max_size: 0,
            palette_max_predictor_size: 0,
        }
    }

    fn band(pos: u8, abs: [u32; 4], sign: [u8; 4]) -> SaoComponent {
        SaoComponent {
            sao_type_idx: 1,
            offset_abs: abs,
            offset_sign: sign,
            band_position: pos,
            eo_class: 0,
        }
    }

    fn edge(class: u8, abs: [u32; 4]) -> SaoComponent {
        SaoComponent {
            sao_type_idx: 2,
            offset_abs: abs,
            offset_sign: [0; 4],
            band_position: 0,
            eo_class: class,
        }
    }

    /// Round-trip one CTB's syntax through the write-side dual and the
    /// decode-side `decode_sao`, comparing the decoded fields.
    fn assert_sao_roundtrip(
        params: &SaoCtbParams,
        rx: usize,
        ry: usize,
        sao_luma: bool,
        sao_chroma: bool,
    ) {
        let mut w = BitWriter::new();
        let mut enc = CabacEncoder::new();
        let mut ectx = SliceContexts::init(0, 26);
        encode_sao_ctb(
            &mut w, &mut enc, &mut ectx, params, rx, ry, sao_luma, sao_chroma,
        );
        enc.encode_terminate(&mut w, 1);
        let bytes = w.finish();

        let mut engine = CabacEngine::new(BitReader::new(&bytes)).expect("engine");
        let mut dctx = SliceContexts::init(0, 26);
        let got = decode_sao(
            &mut engine,
            &mut dctx,
            &sao_params(sao_luma, sao_chroma),
            rx > 0,
            ry > 0,
        )
        .expect("decode_sao");

        assert_eq!(got.merge_left, params.merge_left, "merge_left");
        assert_eq!(got.merge_up, params.merge_up, "merge_up");
        if params.merge_left || params.merge_up {
            return;
        }
        for c in 0..3 {
            let read = (sao_luma && c == 0) || (sao_chroma && c > 0);
            if !read {
                continue;
            }
            assert_eq!(got.components[c], params.components[c], "component {c}");
        }
    }

    /// The §7.3.8.3 writer is the exact dual of `decode_sao` across
    /// off / band / edge / merge shapes, all flag combinations, and
    /// the offset extremes (incl. the TR cMax 7 no-terminator case).
    #[test]
    fn sao_syntax_roundtrips_through_decode_sao() {
        let shapes: Vec<[SaoComponent; 3]> = vec![
            [SaoComponent::default(); 3],
            [
                band(8, [1, 2, 3, 4], [0, 1, 0, 1]),
                SaoComponent::default(),
                SaoComponent::default(),
            ],
            [
                edge(1, [7, 0, 2, 7]),
                edge(3, [1, 1, 1, 1]),
                edge(3, [0, 7, 3, 0]),
            ],
            [
                band(31, [7, 7, 7, 7], [1, 1, 1, 1]),
                band(0, [0, 0, 0, 0], [0, 0, 0, 0]),
                band(29, [4, 0, 1, 7], [0, 0, 1, 1]),
            ],
            [
                SaoComponent::default(),
                edge(0, [2, 1, 0, 3]),
                edge(0, [5, 4, 3, 2]),
            ],
        ];
        for comps in &shapes {
            for (sao_luma, sao_chroma) in [(true, true), (true, false), (false, true)] {
                for (rx, ry) in [(0usize, 0usize), (1, 0), (0, 1), (2, 3)] {
                    let params = SaoCtbParams {
                        merge_left: false,
                        merge_up: false,
                        components: *comps,
                    };
                    assert_sao_roundtrip(&params, rx, ry, sao_luma, sao_chroma);
                }
            }
        }
        // Merge shapes.
        for (ml, mu, rx, ry) in [
            (true, false, 1, 1),
            (false, true, 1, 1),
            (false, true, 0, 1),
        ] {
            let params = SaoCtbParams {
                merge_left: ml,
                merge_up: mu,
                components: [SaoComponent::default(); 3],
            };
            assert_sao_roundtrip(&params, rx, ry, true, true);
        }
    }

    /// Band estimation on a constant-shift region recovers the shift
    /// (a flat +4 error in one band ⇒ offset +4 in that band).
    #[test]
    fn band_candidate_recovers_flat_shift() {
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        for y in 0..16 {
            for x in 0..16 {
                pic.set_sample(Plane::Luma, x, y, 100); // band 12
            }
        }
        let src = vec![104u8; 256];
        let cand = band_candidate(&pic, Plane::Luma, &src, 16, 0, 0, 16);
        assert_eq!(cand.sao_type_idx, 1);
        // Band 12 must be one of the four covered bands, with offset +4.
        let covered: Vec<usize> = (0..4)
            .map(|k| (cand.band_position as usize + k) & 31)
            .collect();
        let pos = covered
            .iter()
            .position(|&b| b == 12)
            .expect("band 12 covered");
        assert_eq!(cand.offset_abs[pos], 4);
        assert_eq!(cand.offset_sign[pos], 0);
    }

    /// Edge estimation classifies against the pre-SAO neighbours and
    /// clamps to the §7.4.9.3 inferred signs (categories 3/4 never
    /// positive).
    #[test]
    fn edge_candidate_derives_clamped_offsets() {
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        // A horizontal comb: odd columns are local minima (category 1
        // for eo_class 0), even columns local maxima (category 4).
        for y in 0..16 {
            for x in 0..16 {
                pic.set_sample(Plane::Luma, x, y, if x % 2 == 0 { 110 } else { 90 });
            }
        }
        // Source: flatten toward 100 ⇒ minima want +10 (clamped 7),
        // maxima want −10 (clamped −7 ⇒ abs 7).
        let src = vec![100u8; 256];
        let cand = edge_candidate(&pic, Plane::Luma, &src, 16, 0, 0, 16, 0);
        assert_eq!(cand.sao_type_idx, 2);
        assert_eq!(cand.eo_class, 0);
        assert_eq!(cand.offset_abs[0], 7, "category 1 (local min) clamped +7");
        assert_eq!(cand.offset_abs[3], 7, "category 4 (local max) clamped −7");
    }

    /// The whole-frame SAO election never worsens the reconstruction:
    /// on content with a systematic band error the filtered output is
    /// strictly closer to the source, and re-applying the elected
    /// parameters through the decode-side picture driver reproduces
    /// the returned planes exactly.
    #[test]
    fn filter_frame_sao_improves_and_matches_decoder_apply() {
        let (w, h) = (32usize, 32usize);
        // Recon: gradient. Source: recon + 3 (a flat band-offset-able
        // error), chroma constant.
        let recon_y: Vec<u8> = (0..w * h).map(|i| ((i % w) * 4) as u8).collect();
        let src_y: Vec<u8> = recon_y.iter().map(|&v| v.saturating_add(3)).collect();
        let recon_c = vec![100u8; w * h / 4];
        let src_c: Vec<u8> = vec![102u8; w * h / 4];
        let field = MotionField::new(w, h);
        let shapes = vec![
            CtbShape {
                part_mode: PartMode::Part2Nx2N,
                split_depth1: false,
            };
            (w / 16) * (h / 16)
        ];
        let input = FilterInput {
            width: w,
            height: h,
            qp: 30,
            lambda: 4,
            recon: [&recon_y, &recon_c, &recon_c],
            src: [&src_y, &src_c, &src_c],
            field: &field,
            shapes: &shapes,
        };
        let out = filter_frame(&input, &LoopFilterCfg::all());
        assert!(out.slice_sao_luma || out.slice_sao_chroma, "SAO elected");

        let ssd = |a: &[u8], b: &[u8]| -> u64 {
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| {
                    let d = i64::from(x) - i64::from(y);
                    (d * d) as u64
                })
                .sum()
        };
        let before = ssd(&recon_y, &src_y) + 2 * ssd(&recon_c, &src_c);
        let after = ssd(&out.y, &src_y) + ssd(&out.cb, &src_c) + ssd(&out.cr, &src_c);
        assert!(after < before, "SAO improves SSD ({after} < {before})");

        // Decoder-side re-application: resolve the emitted syntax with
        // merge inheritance and run the picture driver — byte-exact
        // against the returned planes.
        let ctbs_x = w / 16;
        let mut grid = vec![ResolvedSao::off(); out.sao_ctbs.len()];
        for (i, p) in out.sao_ctbs.iter().enumerate() {
            let left = (i % ctbs_x > 0).then(|| grid[i - 1]);
            let above = (i / ctbs_x > 0).then(|| grid[i - ctbs_x]);
            grid[i] = ResolvedSao::resolve(p, left.as_ref(), above.as_ref(), 0, 0);
        }
        let pre = planes_to_picture(&recon_y, &recon_c, &recon_c, w, h);
        let applied = apply_sao_picture_full(
            &pre,
            &grid,
            4,
            1,
            out.slice_sao_luma,
            out.slice_sao_chroma,
            None,
            None,
        );
        let planar = applied.to_planar_u8().expect("8-bit");
        let mut expect = out.y.clone();
        expect.extend_from_slice(&out.cb);
        expect.extend_from_slice(&out.cr);
        assert_eq!(planar, expect, "decode-side apply reproduces filter_frame");
    }

    /// Deblocking election: on a blocky reconstruction of smooth
    /// source content the filter is elected (and the output differs);
    /// on a perfect reconstruction it is declined.
    #[test]
    fn deblock_election_is_distortion_driven() {
        let (w, h) = (32usize, 32usize);
        // Source: smooth horizontal ramp.
        let src_y: Vec<u8> = (0..w * h).map(|i| ((i % w) * 3 + 40) as u8).collect();
        // Recon: the ramp quantized to per-16 blocks (strong blocking).
        let recon_y: Vec<u8> = (0..w * h)
            .map(|i| {
                let x = (i % w) / 16 * 16 + 8;
                (x * 3 + 40) as u8
            })
            .collect();
        let cpl = vec![128u8; w * h / 4];
        let field = MotionField::new(w, h); // all intra ⇒ bS 2 at CB edges
        let shapes = vec![
            CtbShape {
                part_mode: PartMode::Part2Nx2N,
                split_depth1: false,
            };
            (w / 16) * (h / 16)
        ];
        let input = FilterInput {
            width: w,
            height: h,
            qp: 37,
            lambda: 1,
            recon: [&recon_y, &cpl, &cpl],
            src: [&src_y, &cpl, &cpl],
            field: &field,
            shapes: &shapes,
        };
        let cfg = LoopFilterCfg {
            deblocking: true,
            sao_luma: false,
            sao_chroma: false,
        };
        let out = filter_frame(&input, &cfg);
        assert!(out.deblock_on, "blocky recon elects deblocking");
        assert_ne!(out.y, recon_y, "deblocking modified the luma plane");
        assert!(
            (-2..=2).contains(&out.beta_offset_div2) && (-2..=2).contains(&out.tc_offset_div2),
            "elected offsets stay in the swept range"
        );

        // The election is optimal over its candidate set: brute-force
        // every (β, tC) pair through the decode-side driver and check
        // the elected pair's distortion is the minimum.
        let ssd = |a: &[u8], b: &[u8]| -> u64 {
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| {
                    let d = i64::from(x) - i64::from(y);
                    (d * d) as u64
                })
                .sum()
        };
        let dist_at = |beta: i32, tc: i32| -> u64 {
            let mut pic = planes_to_picture(&recon_y, &cpl, &cpl, w, h);
            let descs = ctb_deblock_descs(&shapes, w, h, 37, beta, tc);
            deblock_picture_full(&mut pic, &field, &descs, None, None);
            let planar = pic.to_planar_u8().expect("8-bit");
            ssd(&planar[..w * h], &src_y)
        };
        let elected = dist_at(out.beta_offset_div2, out.tc_offset_div2);
        for beta in [-2i32, 0, 2] {
            for tc in [-2i32, 0, 2] {
                // λ == 1 makes the rate term negligible against the
                // luma SSD scale of this content.
                assert!(
                    elected <= dist_at(beta, tc) + 16,
                    "({beta},{tc}) beats the elected pair"
                );
            }
        }

        // Perfect reconstruction: filtering can only hurt ⇒ declined.
        let input2 = FilterInput {
            recon: [&src_y, &cpl, &cpl],
            src: [&src_y, &cpl, &cpl],
            ..input
        };
        let out2 = filter_frame(&input2, &cfg);
        assert!(!out2.deblock_on, "perfect recon declines deblocking");
        assert_eq!(out2.y, src_y);
    }
}
