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
#[allow(dead_code)] // Ignored is part of the corpus tier vocabulary
                    // even when no current fixture lands there.
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
        "still-image-with-icc" => {
            "ICC profile retained as Property::Other; no ICC-aware compare yet"
        }
        "still-image-with-exif" => {
            "Exif metadata item only; primary decode parity not yet verified"
        }
        "still-image-with-xmp" => "XMP metadata item only; primary decode parity not yet verified",
        "still-image-grid-2x2" => {
            "grid composition lands; bit-exact tile-boundary parity not yet verified"
        }
        "still-image-overlay" => {
            "iovl canvas-fill matrix matches the corpus convention (round 5 \
             phase A) but the underlying HEVC layer pixels still diverge from \
             the oracle: 97.3% of bytes differ with max |Δ|=160 after planar→\
             packed RGB conversion through compare_bit_exact. Bisect the per-\
             layer decode (background item 1 vs overlay item 2) before re-\
             promoting"
        }
        "multi-image-burst-3" => "multiple still items; primary decode parity not yet verified",
        "still-yuv444" => {
            "HEVC 4:4:4 I-slice decode is enabled (round 30 lift); decoder \
             pipeline produces a 3-plane planar YUV at full chroma resolution \
             (sub_x=sub_y=1). The compare_bit_exact path infers sub_x / sub_y \
             from cb_stride vs y_stride and would compare cleanly, but bit-\
             exact parity against the oracle PNG isn't yet validated for an \
             arbitrary HEIF-Main-4:4:4 stream — the encoder unit tests \
             exercise 4:4:4 round-trip at PSNR ≥ 30 dB rather than byte-\
             exact, so a real-encoder fixture may diverge"
        }
        "image-sequence-3frame" => {
            "moov walker lifts sample table + hvcC; per-sample HEVC decode parity not yet verified"
        }
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
        // Round 6 + task #320 promoted: chroma_format_idc=0 lift
        // unblocks the alpha-aux decode (every HEIF alpha aux is a
        // monochrome HEVC stream per ISO/IEC 23008-12 §6.6.2.1); the
        // comparator's compare_rgba branch assembles RGBA from the
        // primary 4:2:0 colour decode + the 1-plane Gray8 alpha aux.
        fixture!("still-image-with-alpha", BitExact),
        fixture!("still-image-with-icc", ReportOnly),
        fixture!("still-image-with-exif", ReportOnly),
        fixture!("still-image-with-xmp", ReportOnly),
        fixture!("still-image-grid-2x2", ReportOnly),
        // Round-5 phase A made the iovl canvas-fill matrix match the
        // corpus convention via `FillMatrix` in src/heif/mod.rs (see
        // commit 57e21e1). That promotion was reverted in 293ac2a
        // because the underlying HEVC layer pixels still diverge from
        // the oracle once converted to RGB through `compare_bit_exact`
        // (97.3% of bytes differ, max |Δ|=160 — far beyond a per-pixel
        // round-trip precision drift). The comparator already runs the
        // planar-YUV → packed-RGB step uniformly for every BitExact
        // fixture (see `compare_bit_exact` below — wired in round 4 by
        // f78b612), so the gap is in the layer pixel decode itself,
        // not the comparator wiring. Stays ReportOnly until the per-
        // layer divergence is bisected.
        fixture!("still-image-overlay", ReportOnly),
        fixture!("multi-image-burst-3", ReportOnly),
        // Round 6 + task #320 promoted: chroma_format_idc=0 lift in
        // emit_monochrome_frame produces a 1-plane luma VideoFrame
        // (Gray8); the comparator's compare_rgb24 1-plane branch
        // splats Y to (Y,Y,Y) packed RGB and matches the oracle.
        fixture!("still-monochrome", BitExact),
        // Round 7 + task #320 promoted: bit_depth=10 lift in
        // emit_frame produces a 3-plane 16-bit-LE-packed YUV
        // VideoFrame; the comparator's compare_rgb48le branch reads
        // u16 samples, runs the YUV→RGB matrix at the source bit
        // depth (10-bit, mask 0x3FF), and writes LSB-aligned 16-bit
        // RGB matching the Rgb48Le oracle PNG.
        fixture!("still-10bit-main10", BitExact),
        // Task #321 promotes: HEVC 4:4:4 I-slice decode (round 30 lift)
        // produces a 3-plane planar YUV at full chroma resolution
        // (sub_x=sub_y=1). The compare_bit_exact path infers sub_x /
        // sub_y from cb_stride vs y_stride and runs the BT.601-limited
        // YUV→RGB matrix uniformly. The comparator extension in 3e7c6f5
        // (PNG-IHDR-driven dispatch + compare_rgb24 / compare_rgba /
        // compare_rgb48le specialised paths) lands the missing 4:4:4
        // chroma-stride accounting needed for this fixture; promotion
        // exercises the wired path against the oracle.
        fixture!("still-yuv444", BitExact),
        // Round-5 promoted: moov/trak/mdia/minf/stbl walker now lifts
        // a sample table + decoder hvcC out of the image-sequence
        // file. The HEVC decode of each sample isn't yet validated
        // bit-exact (no oracle PNG per sample) — this stays at
        // ReportOnly until that pipeline lands.
        fixture!("image-sequence-3frame", ReportOnly),
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

    // 5b. Image-sequence walker — informational. Surfaces sample
    // count + per-sample sizes when the file carries a `moov` (HEIF
    // image-sequence brands `msf1` / `hevc` / `heis`). Round-5
    // landed [`heif::parse_moov`]; per-sample HEVC decode parity is
    // round-6 work.
    match heif::parse_moov(fx.heic) {
        Ok(Some(summary)) => {
            msgs.push(format!(
                "moov: timescale={} sample_entry='{}' samples={} display={:?}",
                summary.timescale,
                std::str::from_utf8(&summary.sample_entry).unwrap_or("?"),
                summary.samples.len(),
                summary.display_dims,
            ));
            for (i, s) in summary.samples.iter().enumerate() {
                msgs.push(format!(
                    "moov sample[{i}]: offset={} size={} duration={} sync={}",
                    s.offset, s.size, s.duration, s.is_sync,
                ));
            }
        }
        Ok(None) => {}
        Err(e) => msgs.push(format!("moov: ERR {e}")),
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
                // Pre-decode alpha aux if present (Rgba-oracle compare
                // path needs it). Surface decode failure as a soft
                // signal — the comparator will only require it when
                // the oracle actually carries an alpha channel.
                let alpha = match heif::decode_alpha_for_primary(fx.heic) {
                    Ok(opt) => opt,
                    Err(e) => {
                        msgs.push(format!(
                            "bit-exact: alpha aux decode failed (will skip alpha-bearing compare): {e}"
                        ));
                        None
                    }
                };
                if let Some(ref o) = oracle {
                    match compare_bit_exact(o, fx.expected_png, &vf, matrix, alpha.as_ref()) {
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

/// Oracle PNG layout extracted from the IHDR bytes (width, height,
/// bit_depth, colour_type). The 4-tuple is enough to disambiguate
/// Rgb24 (bd=8 ct=2) / Rgba (bd=8 ct=6) / Rgb48Le (bd=16 ct=2) /
/// Gray8 (bd=8 ct=0) — collisions in stride-based heuristics
/// (e.g. width=128 8-bit Rgb24 vs. 16-bit Rgb48Le both end up with
/// luma stride=256) make a real header parse the only sound choice.
#[derive(Clone, Copy, Debug)]
struct OracleHeader {
    width: u32,
    height: u32,
    bit_depth: u8,
    colour_type: u8,
}

fn parse_png_ihdr(png: &[u8]) -> Result<OracleHeader, String> {
    // 8-byte signature + 8-byte chunk header (length + type) + 13-byte
    // IHDR payload at the very start of every PNG.
    if png.len() < 8 + 8 + 13 {
        return Err("PNG too short for IHDR".to_string());
    }
    if &png[0..8] != b"\x89PNG\r\n\x1a\n" {
        return Err("not a PNG signature".to_string());
    }
    if &png[12..16] != b"IHDR" {
        return Err("first chunk is not IHDR".to_string());
    }
    let width = u32::from_be_bytes([png[16], png[17], png[18], png[19]]);
    let height = u32::from_be_bytes([png[20], png[21], png[22], png[23]]);
    let bit_depth = png[24];
    let colour_type = png[25];
    Ok(OracleHeader {
        width,
        height,
        bit_depth,
        colour_type,
    })
}

/// Bit-exact compare of an HEVC-decoded `VideoFrame` against an
/// `expected.png`-decoded packed-pixel oracle. The oracle is always a
/// single packed plane; our HEVC output may be 3-plane planar YUV
/// (4:2:0 / 4:2:2 / 4:4:4 at 8-bit or 10/12-bit LE-packed) or 1-plane
/// luma (monochrome `chroma_format_idc==0`, round-6 lift).
///
/// Dispatch is driven by the oracle PNG header (parsed from
/// `expected_png` bytes — we can't infer reliably from the decoded
/// `VideoFrame` strides because at width=128 the 8-bit-Rgb24 layout
/// and the 16-bit-Rgb48Le layout produce colliding y_stride values).
/// Strategy table:
///
/// | oracle (IHDR)         | actual                       | strategy                                  |
/// |-----------------------|------------------------------|-------------------------------------------|
/// | bd=8  ct=2 (Rgb24)    | 3-plane 8-bit YUV            | YUV→RGB24 then byte-compare               |
/// | bd=8  ct=2 (Rgb24)    | 1-plane 8-bit luma (Y8)      | expand Y to (Y,Y,Y) then byte-compare     |
/// | bd=8  ct=6 (Rgba)     | 3-plane 8-bit YUV + alpha aux| YUV→RGB then pack {R,G,B,A} (alpha plane) |
/// | bd=16 ct=2 (Rgb48Le)  | 3-plane 16-bit YUV (LE-pkd)  | YUV16→RGB16 then byte-compare             |
///
/// `actual_alpha` is the optional 1-plane Gray8 / Gray16Le frame
/// produced by [`heif::decode_alpha_for_primary`]; required only for
/// the Rgba-oracle path.
///
/// Returns `Ok(())` on byte-equal match; `Err(reason)` otherwise.
fn compare_bit_exact(
    oracle: &oxideav_core::VideoFrame,
    oracle_png: &[u8],
    actual: &oxideav_core::VideoFrame,
    matrix: Matrix,
    actual_alpha: Option<&oxideav_core::VideoFrame>,
) -> Result<(), String> {
    if oracle.planes.len() != 1 {
        return Err(format!(
            "oracle is not single-plane packed (got {} planes)",
            oracle.planes.len()
        ));
    }
    let oracle_stride = oracle.planes[0].stride.max(1);
    let oracle_h_observed = oracle.planes[0].data.len() / oracle_stride;
    let hdr = parse_png_ihdr(oracle_png)?;
    let oracle_w = hdr.width as usize;
    let oracle_h = hdr.height as usize;
    if oracle_h != oracle_h_observed {
        return Err(format!(
            "oracle PNG height {oracle_h} != observed plane rows {oracle_h_observed}"
        ));
    }

    match (hdr.colour_type, hdr.bit_depth) {
        // Rgb24 oracle (bd=8 ct=2). Both 1-plane luma and 3-plane
        // 8-bit YUV actual frames are accepted.
        (2, 8) => compare_rgb24(oracle, actual, matrix, oracle_w, oracle_h),
        // Rgba oracle (bd=8 ct=6). Requires an alpha aux to assemble
        // the alpha channel.
        (6, 8) => {
            let alpha = actual_alpha.ok_or_else(|| {
                "Rgba oracle requires an alpha auxiliary frame; \
                 decode_alpha_for_primary returned None or Err — check \
                 the prior `alpha decode:` log line"
                    .to_string()
            })?;
            compare_rgba(oracle, actual, alpha, matrix, oracle_w, oracle_h)
        }
        // Rgb48Le oracle (bd=16 ct=2).
        (2, 16) => compare_rgb48le(oracle, actual, matrix, oracle_w, oracle_h),
        // Other oracle formats (Gray8, Gray16Le, Pal8, RGBA64Le)
        // aren't yet present in the corpus — surface a clear error
        // so a future fixture addition can extend the dispatch.
        (ct, bd) => Err(format!(
            "unsupported oracle PNG format: colour_type={ct} bit_depth={bd}"
        )),
    }
}

/// Rgb24 oracle compare: handles both 1-plane luma actual (monochrome,
/// expand Y→(Y,Y,Y)) and the round-4 3-plane 8-bit YUV path.
fn compare_rgb24(
    oracle: &oxideav_core::VideoFrame,
    actual: &oxideav_core::VideoFrame,
    matrix: Matrix,
    oracle_w: usize,
    oracle_h: usize,
) -> Result<(), String> {
    let y_stride = actual.planes[0].stride.max(1);
    let y_rows = actual.planes[0].data.len() / y_stride;
    if y_stride != oracle_w || y_rows != oracle_h {
        return Err(format!(
            "luma dims {y_stride}x{y_rows} != oracle dims {oracle_w}x{oracle_h}"
        ));
    }
    let yp = &actual.planes[0].data;

    // Single-plane luma: expand Y to packed (Y,Y,Y).
    if actual.planes.len() == 1 {
        let mut rgb = vec![0u8; oracle_w * oracle_h * 3];
        for y in 0..oracle_h {
            for x in 0..oracle_w {
                let yv = yp[y * y_stride + x];
                let off = (y * oracle_w + x) * 3;
                rgb[off] = yv;
                rgb[off + 1] = yv;
                rgb[off + 2] = yv;
            }
        }
        return diff_or_ok(&rgb, &oracle.planes[0].data);
    }

    if actual.planes.len() != 3 {
        return Err(format!(
            "actual frame is not 1-plane luma or 3-plane planar YUV (got {} planes)",
            actual.planes.len()
        ));
    }

    let cb_stride = actual.planes[1].stride.max(1);
    let cb_rows = actual.planes[1].data.len() / cb_stride;
    let cr_stride = actual.planes[2].stride.max(1);
    let cr_rows = actual.planes[2].data.len() / cr_stride;
    if cb_stride != cr_stride || cb_rows != cr_rows {
        return Err(format!(
            "chroma planes disagree: Cb={cb_stride}x{cb_rows} Cr={cr_stride}x{cr_rows}"
        ));
    }

    let (sub_x, sub_y) = infer_subsampling(y_stride, y_rows, cb_stride, cb_rows)?;

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
    diff_or_ok(&rgb, &oracle.planes[0].data)
}

/// Rgba oracle compare: assemble 8-bit RGBA from a 3-plane YUV primary
/// + a 1-plane Gray8 alpha auxiliary (decoded via
/// [`heif::decode_alpha_for_primary`]).
fn compare_rgba(
    oracle: &oxideav_core::VideoFrame,
    actual: &oxideav_core::VideoFrame,
    actual_alpha: &oxideav_core::VideoFrame,
    matrix: Matrix,
    oracle_w: usize,
    oracle_h: usize,
) -> Result<(), String> {
    if actual.planes.len() != 3 {
        return Err(format!(
            "Rgba oracle requires 3-plane planar YUV primary (got {} planes)",
            actual.planes.len()
        ));
    }
    let y_stride = actual.planes[0].stride.max(1);
    let y_rows = actual.planes[0].data.len() / y_stride;
    if y_stride != oracle_w || y_rows != oracle_h {
        return Err(format!(
            "luma dims {y_stride}x{y_rows} != oracle dims {oracle_w}x{oracle_h}"
        ));
    }
    let cb_stride = actual.planes[1].stride.max(1);
    let cb_rows = actual.planes[1].data.len() / cb_stride;
    let cr_stride = actual.planes[2].stride.max(1);
    let cr_rows = actual.planes[2].data.len() / cr_stride;
    if cb_stride != cr_stride || cb_rows != cr_rows {
        return Err(format!(
            "chroma planes disagree: Cb={cb_stride}x{cb_rows} Cr={cr_stride}x{cr_rows}"
        ));
    }
    if actual_alpha.planes.len() != 1 {
        return Err(format!(
            "alpha aux must be 1-plane luma (got {} planes)",
            actual_alpha.planes.len()
        ));
    }
    let a_stride = actual_alpha.planes[0].stride.max(1);
    let a_rows = actual_alpha.planes[0].data.len() / a_stride;
    if a_stride != oracle_w || a_rows != oracle_h {
        return Err(format!(
            "alpha dims {a_stride}x{a_rows} != oracle dims {oracle_w}x{oracle_h}"
        ));
    }

    let (sub_x, sub_y) = infer_subsampling(y_stride, y_rows, cb_stride, cb_rows)?;

    let yp = &actual.planes[0].data;
    let up = &actual.planes[1].data;
    let vp = &actual.planes[2].data;
    let ap = &actual_alpha.planes[0].data;

    let mut rgba = vec![0u8; oracle_w * oracle_h * 4];
    for y in 0..oracle_h {
        let cy = y / sub_y;
        for x in 0..oracle_w {
            let cx = x / sub_x;
            let yv = yp[y * y_stride + x];
            let uv = up[cy * cb_stride + cx];
            let vv = vp[cy * cr_stride + cx];
            let (r, g, b) = yuv_to_rgb(yv, uv, vv, matrix);
            let av = ap[y * a_stride + x];
            let off = (y * oracle_w + x) * 4;
            rgba[off] = r;
            rgba[off + 1] = g;
            rgba[off + 2] = b;
            rgba[off + 3] = av;
        }
    }
    diff_or_ok(&rgba, &oracle.planes[0].data)
}

/// Rgb48Le oracle compare: 3-plane 16-bit-LE-packed YUV → 16-bit RGB.
/// Matches what oxideav-png produces for colour_type=2 / bit_depth=16
/// (`Rgb48Le`, stride = width * 6, two LE bytes per channel).
///
/// Channels are converted at full source precision (e.g. 10-bit input
/// → 10-bit RGB output, LSB-aligned in the 16-bit container with the
/// upper bits zero), matching the convention HEIC encoders use when
/// round-tripping a Main 10 fixture out to a PNG oracle.
fn compare_rgb48le(
    oracle: &oxideav_core::VideoFrame,
    actual: &oxideav_core::VideoFrame,
    matrix: Matrix,
    oracle_w: usize,
    oracle_h: usize,
) -> Result<(), String> {
    if actual.planes.len() != 3 {
        return Err(format!(
            "Rgb48Le oracle requires 3-plane planar YUV (got {} planes)",
            actual.planes.len()
        ));
    }
    let y_stride = actual.planes[0].stride.max(1);
    if y_stride != oracle_w * 2 {
        return Err(format!(
            "luma stride {y_stride} != oracle_w*2 ({})  (high-bit-depth path requires 2 bytes/sample)",
            oracle_w * 2
        ));
    }
    let y_rows = actual.planes[0].data.len() / y_stride;
    if y_rows != oracle_h {
        return Err(format!("luma rows {y_rows} != oracle rows {oracle_h}"));
    }
    let cb_stride = actual.planes[1].stride.max(1);
    let cb_rows = actual.planes[1].data.len() / cb_stride;
    let cr_stride = actual.planes[2].stride.max(1);
    let cr_rows = actual.planes[2].data.len() / cr_stride;
    if cb_stride != cr_stride || cb_rows != cr_rows {
        return Err(format!(
            "chroma planes disagree: Cb={cb_stride}x{cb_rows} Cr={cr_stride}x{cr_rows}"
        ));
    }
    // For 16-bit data the chroma stride is also in bytes (2 per
    // sample). Subsampling check uses sample counts, not bytes.
    let (sub_x, sub_y) = infer_subsampling_bytes(y_stride, y_rows, cb_stride, cb_rows, 2usize)?;

    // Infer the source bit depth from the oracle's first non-zero
    // pixel range. We don't carry PixelFormat through VideoFrame, so
    // peek at the maximum sample value across luma to determine the
    // mask. For Main 10 the mask is 0x3FF; for Main 12 it would be
    // 0xFFF. We default to the conservative full-range 16-bit and
    // trust the actual decoder to clamp.
    let yp = &actual.planes[0].data;
    let up = &actual.planes[1].data;
    let vp = &actual.planes[2].data;
    let mut max_y = 0u16;
    for chunk in yp.chunks_exact(2) {
        let v = u16::from_le_bytes([chunk[0], chunk[1]]);
        if v > max_y {
            max_y = v;
        }
    }
    let bit_depth = if max_y < (1 << 8) {
        8u32
    } else if max_y < (1 << 10) {
        10u32
    } else if max_y < (1 << 12) {
        12u32
    } else if max_y < (1 << 14) {
        14u32
    } else {
        16u32
    };

    let mut rgb = vec![0u8; oracle_w * oracle_h * 6];
    for y in 0..oracle_h {
        let cy = y / sub_y;
        for x in 0..oracle_w {
            let cx = x / sub_x;
            let yv = read_u16_le(yp, y * y_stride + x * 2);
            let uv = read_u16_le(up, cy * cb_stride + cx * 2);
            let vv = read_u16_le(vp, cy * cr_stride + cx * 2);
            let (r, g, b) = yuv16_to_rgb16(yv, uv, vv, matrix, bit_depth);
            let off = (y * oracle_w + x) * 6;
            // LSB-aligned 16-bit per channel, little-endian.
            rgb[off] = (r & 0xFF) as u8;
            rgb[off + 1] = (r >> 8) as u8;
            rgb[off + 2] = (g & 0xFF) as u8;
            rgb[off + 3] = (g >> 8) as u8;
            rgb[off + 4] = (b & 0xFF) as u8;
            rgb[off + 5] = (b >> 8) as u8;
        }
    }
    diff_or_ok(&rgb, &oracle.planes[0].data)
}

fn read_u16_le(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([buf[off], buf[off + 1]])
}

fn infer_subsampling(
    y_stride: usize,
    y_rows: usize,
    cb_stride: usize,
    cb_rows: usize,
) -> Result<(usize, usize), String> {
    infer_subsampling_bytes(y_stride, y_rows, cb_stride, cb_rows, 1)
}

fn infer_subsampling_bytes(
    y_stride: usize,
    y_rows: usize,
    cb_stride: usize,
    cb_rows: usize,
    bps: usize,
) -> Result<(usize, usize), String> {
    let y_samples = y_stride / bps;
    let cb_samples = cb_stride / bps;
    let sub_x = if cb_samples == y_samples {
        1
    } else if cb_samples * 2 == y_samples {
        2
    } else {
        return Err(format!(
            "unsupported chroma subsampling sx (luma samples {y_samples}, chroma samples {cb_samples})"
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
    Ok((sub_x, sub_y))
}

/// Byte-level diff helper. Returns Ok if vectors equal; otherwise an
/// Err with the count of differing bytes and the maximum |Δ|.
fn diff_or_ok(actual: &[u8], oracle: &[u8]) -> Result<(), String> {
    if actual == oracle {
        return Ok(());
    }
    let mut diff = 0usize;
    let mut max_abs = 0i32;
    for (a, e) in actual.iter().zip(oracle.iter()) {
        if a != e {
            diff += 1;
            let d = (*a as i32 - *e as i32).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
    }
    if actual.len() != oracle.len() {
        return Err(format!(
            "{diff} of {} bytes differ (max |Δ|={max_abs}); LENGTH MISMATCH actual={} oracle={}",
            actual.len().min(oracle.len()),
            actual.len(),
            oracle.len()
        ));
    }
    Err(format!(
        "{diff} of {} bytes differ (max |Δ|={max_abs})",
        actual.len()
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

/// Per-pixel YUV→RGB at the source bit depth (10 / 12 / 14 / 16-bit).
/// Returns three LSB-aligned `u16` channels. Limited-range path scales
/// from `[16<<(bd-8), 235<<(bd-8)]` for luma and centres chroma at
/// `128<<(bd-8)`; full-range maps directly. Output channels are
/// clamped to `[0, (1<<bit_depth)-1]` to keep them representable in
/// the source domain (so the oracle PNG, which round-trips the
/// encoder's source-domain RGB out via Rgb48Le, can match byte-for-
/// byte).
fn yuv16_to_rgb16(y: u16, u: u16, v: u16, matrix: Matrix, bit_depth: u32) -> (u16, u16, u16) {
    let (kr, kb, limited) = match matrix {
        Matrix::Bt601Limited => (0.299_f64, 0.114_f64, true),
        Matrix::Bt601Full => (0.299_f64, 0.114_f64, false),
        Matrix::Bt709Limited => (0.2126_f64, 0.0722_f64, true),
        Matrix::Bt709Full => (0.2126_f64, 0.0722_f64, false),
    };
    let kg = 1.0 - kr - kb;
    let max_val = ((1u32 << bit_depth) - 1) as f64;
    let (yv, cb, cr) = if limited {
        // Per ITU-R BT.2100 / BT.709 §B.1.6: black/white code values
        // are 16<<(bd-8) and 235<<(bd-8); chroma midpoint is
        // 128<<(bd-8); chroma extreme is at 240<<(bd-8). The same
        // 219 / 224 scale factors apply, just with values shifted up
        // by `bd - 8` bits.
        let shift = bit_depth - 8;
        let black = 16.0 * (1u32 << shift) as f64;
        let chroma_mid = 128.0 * (1u32 << shift) as f64;
        let luma_range = 219.0 * (1u32 << shift) as f64;
        let chroma_range = 224.0 * (1u32 << shift) as f64;
        let yv = (y as f64 - black) * (max_val / luma_range);
        let cb = (u as f64 - chroma_mid) * (max_val / chroma_range);
        let cr = (v as f64 - chroma_mid) * (max_val / chroma_range);
        (yv, cb, cr)
    } else {
        let mid = (1u32 << (bit_depth - 1)) as f64;
        let yv = y as f64;
        let cb = u as f64 - mid;
        let cr = v as f64 - mid;
        (yv, cb, cr)
    };
    let r = yv + 2.0 * (1.0 - kr) * cr;
    let b = yv + 2.0 * (1.0 - kb) * cb;
    let g = yv - (2.0 * kr * (1.0 - kr) / kg) * cr - (2.0 * kb * (1.0 - kb) / kg) * cb;
    (
        clamp_u16(r, max_val),
        clamp_u16(g, max_val),
        clamp_u16(b, max_val),
    )
}

fn clamp_u16(v: f64, max_val: f64) -> u16 {
    let r = v.round();
    if r < 0.0 {
        0
    } else if r > max_val {
        max_val as u16
    } else {
        r as u16
    }
}
