//! Integration test exercising the HEIF / HEIC scaffold against the
//! 14-fixture corpus shipped at `tests/fixtures/heif/`.
//!
//! Compiled only when the `heif` feature is enabled — which is the
//! crate's default since `09a9203`, so this file actually executes in
//! CI on every PR.
//!
//! # Tier system
//!
//! Each fixture is tagged with one of:
//!
//! * [`Tier::BitExact`] — we expect to decode the primary item end-to-
//!   end and the resulting `VideoFrame` to match `expected.png` byte-
//!   for-byte after a YUV→RGB conversion of our HEVC output. The
//!   conversion follows the BT.601 limited-range matrix by default —
//!   the same convention oxideav-webp settled on in `155c954` (RFC
//!   9649 §2.5; HEIC fixtures without an `nclx` `colr` follow the
//!   ISO/IEC 23008-12 §6.5.5 default of BT.709 primaries with BT.601
//!   matrix coefficients). When the primary item carries an `nclx`
//!   `colr` we honour its `matrix_coefficients` (1 → BT.709, 6/5 →
//!   BT.601) and `full_range_flag`.
//!
//! * [`Tier::ReportOnly`] — the test runs end-to-end but does not
//!   fail on a divergence. Per-fixture statistics (decode succeeded /
//!   failed-with-reason; expected vs. actual dimensions; alpha /
//!   grid features detected) are surfaced via `eprintln!` to make
//!   future tightening to `BitExact` straightforward.
//!
//! * [`Tier::Ignored`] — feature genuinely out of scope for this
//!   round (image sequences, raw metadata items). Reported but never
//!   tested.
//!
//! # What this round verifies
//!
//! For every fixture:
//!
//! 1. [`heif::probe`] accepts the bytes (the `ftyp` claims an
//!    HEIC/HEIF brand).
//! 2. [`heif::parse_header`] walks `ftyp` + `meta` cleanly.
//! 3. The primary item (`pitm`) is locatable.
//! 4. We dispatch on primary item type:
//!    * `hvc1` / `hev1` — try [`heif::decode_primary`].
//!    * `grid` — try [`heif::decode_primary`] (per-tile decode + paste).
//!    * `iovl` — try [`heif::decode_primary`] (round-3 phase C
//!      composition).
//!    * other (image sequence / metadata-only) — skipped per Tier.
//! 5. Optional features are sniffed (alpha auxiliary item, embedded
//!    ICC profile, dimg / cdsc / thmb iref edges) and reported.
//! 6. For `BitExact` fixtures: HEVC YUV420 output is converted to
//!    packed RGB24 via the per-fixture matrix and compared byte-for-
//!    byte with the `expected.png` oracle.

#![cfg(feature = "heif")]

use oxideav_h265::heif::{self, Colr, ImageGrid, ImageOverlay, Property};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tier {
    /// Expect a successful decode whose output matches `expected.png`
    /// byte-for-byte after YUV→RGB conversion. Promote a `ReportOnly`
    /// fixture to this once the underlying HEVC decoder pipeline is
    /// complete enough to satisfy it.
    BitExact,
    /// Run end-to-end and report stats but never fail on divergence.
    ReportOnly,
    /// Out of scope this round (image sequences, pure-metadata items).
    Ignored,
}

/// Per-fixture reason a `ReportOnly` tier hasn't been promoted to
/// `BitExact`. Surfaced in the round-end summary so the next round's
/// task scope is a one-line read.
fn report_only_reason(name: &str) -> &'static str {
    match name {
        "single-image-512x512-q60" => "decoder DPB residual mismatch (no exact ground truth)",
        "single-image-with-thumbnail" => {
            "thumbnail item iref present; primary decode parity not yet verified"
        }
        "still-image-with-alpha" => "alpha auxiliary uses HEVC monochrome (chroma_format_idc=0)",
        "still-image-with-icc" => {
            "ICC profile retained as Property::Other; no ICC-aware compare yet"
        }
        "still-image-with-exif" => {
            "Exif metadata item only; primary decode parity not yet verified"
        }
        "still-image-with-xmp" => {
            "XMP metadata item only; primary decode parity not yet verified"
        }
        "still-image-grid-2x2" => {
            "grid composition lands; bit-exact tile-boundary parity not yet verified"
        }
        "still-image-overlay" => {
            "iovl canvas fill uses BT.709; corpus expected.png is BT.601 default → 1 LSB drift"
        }
        "multi-image-burst-3" => "multiple still items; primary decode parity not yet verified",
        "still-monochrome" => "HEVC monochrome (chroma_format_idc=0) not pixel-emitting",
        "still-10bit-main10" => "HEVC Main 10 not pixel-emitting end-to-end",
        "still-yuv444" => "HEVC 4:4:4 (chroma_format_idc=3) not pixel-emitting",
        _ => "tier left as ReportOnly pending bit-exact verification",
    }
}

struct Fixture {
    /// Folder name under `tests/fixtures/heif/`.
    name: &'static str,
    tier: Tier,
    /// Raw HEIF bytes (`input.heic`), embedded at compile time.
    heic: &'static [u8],
    /// Raw expected-output bytes (`expected.png`), embedded at compile
    /// time. Decoded via `oxideav_png::decode_png_to_frame` for
    /// dimensional + (eventually) byte-level comparison.
    expected_png: &'static [u8],
}

// `include_bytes!` is resolved relative to this source file, so each
// macro invocation walks into `tests/fixtures/heif/<name>/`.
macro_rules! fixture {
    ($name:literal, $tier:ident) => {
        Fixture {
            name: $name,
            tier: Tier::$tier,
            heic: include_bytes!(concat!("fixtures/heif/", $name, "/input.heic")),
            expected_png: include_bytes!(concat!("fixtures/heif/", $name, "/expected.png")),
        }
    };
}

fn fixtures() -> Vec<Fixture> {
    vec![
        // Round-4 promoted: 64x64 HEVC → clap → 1x1 YUV → 1x1 RGB.
        fixture!("single-image-1x1", BitExact),
        fixture!("single-image-512x512-q60", ReportOnly),
        fixture!("single-image-with-thumbnail", ReportOnly),
        fixture!("still-image-with-alpha", ReportOnly),
        fixture!("still-image-with-icc", ReportOnly),
        fixture!("still-image-with-exif", ReportOnly),
        fixture!("still-image-with-xmp", ReportOnly),
        fixture!("still-image-grid-2x2", ReportOnly),
        // Round-4 left at ReportOnly: iovl composition lands on a YUV
        // canvas, but `compose_overlay_frames` computes its canvas fill
        // via the BT.709 limited-range matrix (see
        // `bt709_limited_rgb_to_yuv` in src/heif/mod.rs) while the
        // expected.png was generated without a `colr` — so the test's
        // YUV→RGB compare defaults to BT.601 and round-trips the grey
        // fill 1 LSB high. Promoting requires either matching the iovl
        // fill matrix to the corpus convention or accepting a known-
        // bounded ±1 LSB tolerance on canvas-fill pixels; both are out
        // of scope for round 4.
        fixture!("still-image-overlay", ReportOnly),
        fixture!("multi-image-burst-3", ReportOnly),
        fixture!("still-monochrome", ReportOnly),
        fixture!("still-10bit-main10", ReportOnly),
        fixture!("still-yuv444", ReportOnly),
        fixture!("image-sequence-3frame", Ignored),
    ]
}

#[derive(Default)]
struct Stats {
    total: usize,
    probed_ok: usize,
    parsed_header_ok: usize,
    decoded_ok: usize,
    decode_failed: usize,
    bit_exact: usize,
    bit_exact_failed: usize,
    ignored: usize,
}

/// Round-end summary classification driving the report-card output.
/// Filled in once per fixture by [`run_one`].
#[derive(Clone, Debug)]
enum Outcome {
    /// Tier::BitExact + comparison succeeded.
    BitExactPass,
    /// Tier::BitExact + comparison failed (build-fail).
    BitExactFail(String),
    /// Tier::ReportOnly with a documented reason.
    ReportOnly(&'static str),
    /// Tier::Ignored — out of scope this round.
    Ignored,
    /// Probe / header / pitm failure: structural problem (build-fail).
    Skipped(String),
}

#[test]
fn corpus_walk_and_report() {
    let mut stats = Stats::default();
    let mut all_messages: Vec<(String, Vec<String>, Outcome)> = Vec::new();
    for fx in fixtures() {
        stats.total += 1;
        let (msgs, outcome) = run_one(&fx, &mut stats);
        all_messages.push((fx.name.to_string(), msgs, outcome));
    }

    eprintln!();
    eprintln!("=== HEIF corpus per-fixture log ===");
    for (name, msgs, _) in &all_messages {
        eprintln!("[{name}]");
        for m in msgs {
            eprintln!("  {m}");
        }
    }

    // ---- Report card ------------------------------------------------
    // Three sections so the next round's task scope is a one-line read.
    eprintln!();
    eprintln!("=== HEIF corpus report card ===");

    eprintln!();
    eprintln!("PROMOTED TO BIT-EXACT ({}):", stats.bit_exact);
    let mut any_pass = false;
    for (name, _, outcome) in &all_messages {
        if matches!(outcome, Outcome::BitExactPass) {
            eprintln!("  PASS  {name}");
            any_pass = true;
        }
    }
    if !any_pass {
        eprintln!("  (none)");
    }

    eprintln!();
    eprintln!("BIT-EXACT FAILED ({}):", stats.bit_exact_failed);
    let mut any_fail = false;
    for (name, _, outcome) in &all_messages {
        if let Outcome::BitExactFail(why) = outcome {
            eprintln!("  FAIL  {name}: {why}");
            any_fail = true;
        }
    }
    if !any_fail {
        eprintln!("  (none)");
    }

    eprintln!();
    eprintln!("STILL REPORT-ONLY (with reason):");
    for (name, _, outcome) in &all_messages {
        if let Outcome::ReportOnly(reason) = outcome {
            eprintln!("  - {name}: {reason}");
        }
    }

    eprintln!();
    eprintln!("IGNORED (out of scope this round):");
    for (name, _, outcome) in &all_messages {
        if matches!(outcome, Outcome::Ignored) {
            eprintln!("  - {name}");
        }
    }

    let any_skipped = all_messages
        .iter()
        .any(|(_, _, o)| matches!(o, Outcome::Skipped(_)));
    if any_skipped {
        eprintln!();
        eprintln!("STRUCTURAL FAILURE (probe / header / pitm):");
        for (name, _, outcome) in &all_messages {
            if let Outcome::Skipped(why) = outcome {
                eprintln!("  ! {name}: {why}");
            }
        }
    }

    eprintln!();
    eprintln!(
        "Totals: probed_ok={}/{} parsed_header_ok={} decoded_ok={} decode_failed={} bit_exact={}/{} ignored={}",
        stats.probed_ok,
        stats.total,
        stats.parsed_header_ok,
        stats.decoded_ok,
        stats.decode_failed,
        stats.bit_exact,
        stats.bit_exact + stats.bit_exact_failed,
        stats.ignored,
    );

    // Every fixture must at least pass probe + header walk — those
    // come from the box parser side which is mature. The Ignored tier
    // short-circuits before parse_header (probe + skip), so it's
    // excluded from the parse_header total.
    assert_eq!(
        stats.probed_ok, stats.total,
        "probe should accept every corpus fixture"
    );
    assert_eq!(
        stats.parsed_header_ok,
        stats.total - stats.ignored,
        "parse_header should walk every non-Ignored corpus fixture"
    );
    // BitExact tier promotes are strict — once a fixture is tagged
    // BitExact, a divergence MUST fail the build.
    assert_eq!(
        stats.bit_exact_failed, 0,
        "BitExact-tier fixtures must match expected.png byte-for-byte"
    );
}

fn run_one(fx: &Fixture, stats: &mut Stats) -> (Vec<String>, Outcome) {
    let mut msgs = Vec::new();

    // 1. probe
    let p = heif::probe(fx.heic);
    msgs.push(format!("tier={:?} heic_bytes={}", fx.tier, fx.heic.len()));
    if !p {
        msgs.push("probe: REJECTED".to_string());
        return (
            msgs,
            Outcome::Skipped("probe rejected ftyp brands".to_string()),
        );
    }
    stats.probed_ok += 1;
    msgs.push("probe: ok".to_string());

    if matches!(fx.tier, Tier::Ignored) {
        stats.ignored += 1;
        msgs.push("ignored: out of scope this round (e.g. image sequence track)".to_string());
        return (msgs, Outcome::Ignored);
    }

    // 2. parse_header
    let hdr = match heif::parse_header(fx.heic) {
        Ok(h) => h,
        Err(e) => {
            msgs.push(format!("parse_header: ERR {e}"));
            return (msgs, Outcome::Skipped(format!("parse_header: {e}")));
        }
    };
    stats.parsed_header_ok += 1;
    msgs.push(format!(
        "parse_header: ok (items={} props={} irefs={})",
        hdr.meta.items.len(),
        hdr.meta.properties.len(),
        hdr.meta.irefs.len()
    ));

    // 3. primary
    let primary_id = match hdr.meta.primary_item_id {
        Some(p) => p,
        None => {
            msgs.push("pitm: ABSENT".to_string());
            return (msgs, Outcome::Skipped("pitm absent".to_string()));
        }
    };
    let info = hdr
        .meta
        .item_by_id(primary_id)
        .expect("pitm references a known item");
    msgs.push(format!(
        "primary: id={} type='{}'",
        primary_id,
        std::str::from_utf8(&info.item_type).unwrap_or("?")
    ));

    // 4. Detected feature signals (informational).
    if let Some(alpha_id) = heif::find_alpha_item_id(&hdr.meta, primary_id) {
        msgs.push(format!("alpha: aux_item={alpha_id} (urn=auxid:1)"));
        // Try to decode the alpha plane on its own. Most monochrome
        // HEVC bitstreams won't decode through `oxideav-h265` yet
        // (chroma_format_idc=0 isn't pixel-emitting), so this is
        // expected to ERR with an Unsupported message — the test
        // surfaces it for round-3 visibility.
        match heif::decode_alpha_for_primary(fx.heic) {
            Ok(Some(vf)) => msgs.push(format!(
                "alpha decode: ok (planes={} stride[0]={})",
                vf.planes.len(),
                vf.planes[0].stride
            )),
            Ok(None) => msgs.push("alpha decode: no aux item (race?)".to_string()),
            Err(e) => msgs.push(format!("alpha decode: ERR {e}")),
        }
    }
    if let Some(Property::Ispe(e)) = hdr.meta.property_for(primary_id, b"ispe") {
        msgs.push(format!("ispe: {}x{}", e.width, e.height));
    }
    let primary_colr = match hdr.meta.property_for(primary_id, b"colr") {
        Some(Property::Colr(c)) => Some(c.clone()),
        _ => None,
    };
    if let Some(ref c) = primary_colr {
        msgs.push(format!("colr: {c:?}"));
    }
    if let Some(Property::Clap(c)) = hdr.meta.property_for(primary_id, b"clap") {
        msgs.push(format!(
            "clap: width={}/{} height={}/{} h_off={}/{} v_off={}/{}",
            c.width_n,
            c.width_d,
            c.height_n,
            c.height_d,
            c.horiz_off_n,
            c.horiz_off_d,
            c.vert_off_n,
            c.vert_off_d,
        ));
    }
    if let Some(Property::Irot(r)) = hdr.meta.property_for(primary_id, b"irot") {
        msgs.push(format!(
            "irot: angle={} ({}°)",
            r.angle,
            (r.angle as u32) * 90,
        ));
    }
    if let Some(Property::Imir(m)) = hdr.meta.property_for(primary_id, b"imir") {
        msgs.push(format!(
            "imir: axis={} ({})",
            m.axis,
            if m.axis == 0 {
                "vertical/flip-cols"
            } else {
                "horizontal/flip-rows"
            },
        ));
    }
    let dimg_targets = hdr.meta.iref_targets(b"dimg", primary_id);
    if !dimg_targets.is_empty() {
        msgs.push(format!("dimg: {dimg_targets:?}"));
    }
    let thmb_targets = hdr.meta.iref_targets(b"thmb", primary_id);
    if !thmb_targets.is_empty() {
        msgs.push(format!("thmb (from primary): {thmb_targets:?}"));
    }
    // Inverse: who points at the primary via thmb?
    if let Some(t) = hdr.meta.iref_source_of(b"thmb", primary_id) {
        msgs.push(format!("thumbnail item -> primary: id={t}"));
    }
    let cdsc_targets = hdr.meta.iref_targets(b"cdsc", primary_id);
    if !cdsc_targets.is_empty() {
        msgs.push(format!("cdsc (from primary): {cdsc_targets:?}"));
    }
    // Overlay descriptor inspection.
    if info.item_type == *b"iovl" {
        let n_refs = dimg_targets.len();
        match heif::item_bytes_in(fx.heic, &hdr.meta, primary_id) {
            Ok(iovl_bytes) => match ImageOverlay::parse(iovl_bytes, n_refs) {
                Ok(o) => msgs.push(format!(
                    "iovl: canvas={}x{} fill_rgba={:?} offsets={:?}",
                    o.output_width, o.output_height, o.canvas_fill_rgba, o.offsets,
                )),
                Err(e) => msgs.push(format!("iovl: parse ERR {e}")),
            },
            Err(e) => msgs.push(format!("iovl: item_bytes ERR {e}")),
        }
    }
    // Grid descriptor inspection.
    if info.item_type == *b"grid" {
        match heif::item_bytes_in(fx.heic, &hdr.meta, primary_id) {
            Ok(grid_bytes) => match ImageGrid::parse(grid_bytes) {
                Ok(g) => msgs.push(format!(
                    "grid: {}x{} -> {}x{} (tiles={})",
                    g.rows,
                    g.columns,
                    g.output_width,
                    g.output_height,
                    g.expected_tile_count(),
                )),
                Err(e) => msgs.push(format!("grid: parse ERR {e}")),
            },
            Err(e) => msgs.push(format!("grid: item_bytes ERR {e}")),
        }
    }

    // 5. expected.png oracle (dimensions + format, used for compare).
    let oracle = match oxideav_png::decode_png_to_frame(fx.expected_png, None) {
        Ok(f) => Some(f),
        Err(e) => {
            msgs.push(format!("expected.png: PNG decode ERR {e}"));
            None
        }
    };
    if let Some(ref o) = oracle {
        msgs.push(format!(
            "expected: {} planes, plane[0] stride={} rows={}",
            o.planes.len(),
            o.planes[0].stride,
            o.planes[0].data.len() / o.planes[0].stride.max(1),
        ));
    }

    // 6. End-to-end decode attempt via the high-level shim.
    let mut outcome = match fx.tier {
        Tier::BitExact => Outcome::BitExactFail("decode never returned a frame".to_string()),
        Tier::ReportOnly => Outcome::ReportOnly(report_only_reason(fx.name)),
        Tier::Ignored => Outcome::Ignored, // unreachable: handled earlier
    };
    match heif::decode_primary(fx.heic) {
        Ok(vf) => {
            stats.decoded_ok += 1;
            let plane0_h = vf.planes[0].data.len() / vf.planes[0].stride.max(1);
            msgs.push(format!(
                "decode_primary: ok ({} planes, plane[0] stride={} rows={})",
                vf.planes.len(),
                vf.planes[0].stride,
                plane0_h
            ));

            // BitExact comparison — only attempted for tiers explicitly
            // promoted to BitExact.
            if matches!(fx.tier, Tier::BitExact) {
                let matrix = matrix_from_colr(primary_colr.as_ref());
                msgs.push(format!("bit-exact: matrix={matrix:?}"));
                if let Some(ref o) = oracle {
                    match compare_bit_exact(o, &vf, matrix) {
                        Ok(()) => {
                            stats.bit_exact += 1;
                            msgs.push("bit-exact: MATCH".to_string());
                            outcome = Outcome::BitExactPass;
                        }
                        Err(why) => {
                            stats.bit_exact_failed += 1;
                            msgs.push(format!("bit-exact: DIFF ({why})"));
                            outcome = Outcome::BitExactFail(why);
                        }
                    }
                } else {
                    let why = "oracle PNG decode failed — can't compare".to_string();
                    msgs.push(format!("bit-exact: SKIP ({why})"));
                    stats.bit_exact_failed += 1;
                    outcome = Outcome::BitExactFail(why);
                }
            }
        }
        Err(e) => {
            stats.decode_failed += 1;
            msgs.push(format!("decode_primary: ERR {e}"));
            if matches!(fx.tier, Tier::BitExact) {
                outcome = Outcome::BitExactFail(format!("decode_primary: {e}"));
            }
        }
    }

    (msgs, outcome)
}

/// Choice of YUV→RGB conversion matrix for the bit-exact compare.
/// Anchored on the primary item's `colr nclx` `matrix_coefficients` /
/// `full_range` flag when present, defaulting to BT.601 limited (the
/// convention oxideav-webp settled on in `155c954` and the ISO/IEC
/// 23008-12 §6.5.5 default for HEIC images without an explicit
/// colour profile).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Matrix {
    Bt601Limited,
    Bt601Full,
    Bt709Limited,
    Bt709Full,
}

fn matrix_from_colr(colr: Option<&Colr>) -> Matrix {
    match colr {
        Some(Colr::Nclx {
            matrix_coefficients,
            full_range,
            ..
        }) => match (*matrix_coefficients, *full_range) {
            (1, true) => Matrix::Bt709Full,
            (1, false) => Matrix::Bt709Limited,
            // matrix_coefficients = 5 (BT.470BG) / 6 (BT.601) /
            // 7 (SMPTE-170M) all share the BT.601 luma weights.
            (_, true) => Matrix::Bt601Full,
            (_, false) => Matrix::Bt601Limited,
        },
        // No nclx (or icc-only): default to BT.601 limited per webp #253
        // convention. ISO/IEC 23008-12 §6.5.5 says the absence of
        // `colr` makes colour interpretation implementation-defined; we
        // pick the convention shared with libwebp / libheif's defaults.
        _ => Matrix::Bt601Limited,
    }
}

/// Bit-exact compare of an HEVC YUV planar frame against an
/// `expected.png`-decoded packed-RGB frame. The oracle is always a
/// single packed plane (oxideav-png returns 1-plane Rgb24/Rgba/Gray8);
/// our HEVC output is always 3-plane planar YUV (4:2:0 or 4:2:2). We
/// convert the actual YUV to packed RGB24 using `matrix` and compare
/// byte-for-byte.
///
/// Returns `Ok(())` on byte-equal match; `Err(reason)` otherwise.
fn compare_bit_exact(
    oracle: &oxideav_core::VideoFrame,
    actual: &oxideav_core::VideoFrame,
    matrix: Matrix,
) -> Result<(), String> {
    if oracle.planes.len() != 1 {
        return Err(format!(
            "oracle is not single-plane packed (got {} planes)",
            oracle.planes.len()
        ));
    }
    if actual.planes.len() != 3 {
        return Err(format!(
            "actual frame is not 3-plane planar YUV (got {} planes)",
            actual.planes.len()
        ));
    }
    let oracle_stride = oracle.planes[0].stride.max(1);
    let oracle_h = oracle.planes[0].data.len() / oracle_stride;
    // Three packed channels per pixel → expected stride = w * 3.
    if oracle_stride % 3 != 0 {
        return Err(format!(
            "oracle stride {oracle_stride} is not a multiple of 3 (only Rgb24 oracles supported)"
        ));
    }
    let oracle_w = oracle_stride / 3;

    let y_stride = actual.planes[0].stride.max(1);
    let y_rows = actual.planes[0].data.len() / y_stride;
    let cb_stride = actual.planes[1].stride.max(1);
    let cb_rows = actual.planes[1].data.len() / cb_stride;
    let cr_stride = actual.planes[2].stride.max(1);
    let cr_rows = actual.planes[2].data.len() / cr_stride;

    if y_stride != oracle_w || y_rows != oracle_h {
        return Err(format!(
            "luma dims {y_stride}x{y_rows} != oracle dims {oracle_w}x{oracle_h}"
        ));
    }
    if cb_stride != cr_stride || cb_rows != cr_rows {
        return Err(format!(
            "chroma planes disagree: Cb={cb_stride}x{cb_rows} Cr={cr_stride}x{cr_rows}"
        ));
    }

    // Infer chroma sub-sampling from the per-plane dims.
    let sub_x = if cb_stride == y_stride {
        1
    } else if cb_stride * 2 == y_stride {
        2
    } else {
        return Err(format!(
            "unsupported chroma subsampling sx (luma stride {y_stride}, chroma stride {cb_stride})"
        ));
    };
    let sub_y = if cb_rows == y_rows {
        1
    } else if cb_rows * 2 == y_rows {
        2
    } else {
        return Err(format!(
            "unsupported chroma subsampling sy (luma rows {y_rows}, chroma rows {cb_rows})"
        ));
    };

    let yp = &actual.planes[0].data;
    let up = &actual.planes[1].data;
    let vp = &actual.planes[2].data;

    let mut rgb = vec![0u8; oracle_w * oracle_h * 3];
    for y in 0..oracle_h {
        let cy = y / sub_y;
        for x in 0..oracle_w {
            let cx = x / sub_x;
            let yv = yp[y * y_stride + x];
            let uv = up[cy * cb_stride + cx];
            let vv = vp[cy * cr_stride + cx];
            let (r, g, b) = yuv_to_rgb(yv, uv, vv, matrix);
            let off = (y * oracle_w + x) * 3;
            rgb[off] = r;
            rgb[off + 1] = g;
            rgb[off + 2] = b;
        }
    }

    // Compare byte-for-byte against the oracle.
    if rgb == oracle.planes[0].data {
        return Ok(());
    }
    let mut diff = 0usize;
    let mut max_abs = 0i32;
    for (a, e) in rgb.iter().zip(oracle.planes[0].data.iter()) {
        if a != e {
            diff += 1;
            let d = (*a as i32 - *e as i32).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
    }
    Err(format!(
        "{diff} of {} bytes differ (max |Δ|={max_abs})",
        rgb.len()
    ))
}

/// Per-pixel YUV→RGB. Scalar implementation matching the textbook
/// BT.601 / BT.709 formulas. Limited-range path uses Y' ∈ [16,235],
/// C ∈ [16,240]; full-range maps directly. Output channels are clamped
/// to [0, 255]. Hand-rolled in the test (rather than reaching for
/// `oxideav-pixfmt::yuv::yuv_to_rgb`) so we don't pull a workspace dep
/// in just for the `heif` integration test — task scope.
fn yuv_to_rgb(y: u8, u: u8, v: u8, matrix: Matrix) -> (u8, u8, u8) {
    // Coefficients per Rec. ITU-R BT.601-7 §2.5.1 / BT.709-6 §3.
    let (kr, kb, limited) = match matrix {
        Matrix::Bt601Limited => (0.299_f32, 0.114_f32, true),
        Matrix::Bt601Full => (0.299_f32, 0.114_f32, false),
        Matrix::Bt709Limited => (0.2126_f32, 0.0722_f32, true),
        Matrix::Bt709Full => (0.2126_f32, 0.0722_f32, false),
    };
    let kg = 1.0 - kr - kb;
    let (yv, cb, cr) = if limited {
        // Y' linear-extends [16,235] → [0,255]; chroma centres at 128
        // and scales by 2*(1-k) * 255/224 to recover (R-Y') etc.
        let yv = (y as f32 - 16.0) * (255.0 / 219.0);
        let cb = (u as f32 - 128.0) * (255.0 / 224.0);
        let cr = (v as f32 - 128.0) * (255.0 / 224.0);
        (yv, cb, cr)
    } else {
        let yv = y as f32;
        let cb = u as f32 - 128.0;
        let cr = v as f32 - 128.0;
        (yv, cb, cr)
    };
    let r = yv + 2.0 * (1.0 - kr) * cr;
    let b = yv + 2.0 * (1.0 - kb) * cb;
    let g = yv - (2.0 * kr * (1.0 - kr) / kg) * cr - (2.0 * kb * (1.0 - kb) / kg) * cb;
    (clamp_u8(r), clamp_u8(g), clamp_u8(b))
}

fn clamp_u8(v: f32) -> u8 {
    let r = v.round();
    if r < 0.0 {
        0
    } else if r > 255.0 {
        255
    } else {
        r as u8
    }
}
