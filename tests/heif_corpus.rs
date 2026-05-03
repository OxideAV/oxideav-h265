//! Integration test exercising the HEIF / HEIC scaffold against the
//! 14-fixture corpus shipped at `tests/fixtures/heif/`.
//!
//! Compiled only when the `heif` feature is enabled — on a default
//! build this file is empty and contributes no tests.
//!
//! # Tier system
//!
//! Each fixture is tagged with one of:
//!
//! * [`Tier::BitExact`] — we expect to decode the primary item end-to-
//!   end and the resulting `VideoFrame` to match the planar projection
//!   of the per-fixture `expected.png` byte-for-byte (after YUV→RGBA
//!   conversion). This is the eventual goal; in round 2 nothing is
//!   tagged BitExact yet because:
//!   - We do not perform YUV → RGBA colorspace conversion in this
//!     crate (that lives in `oxideav-pixfmt`).
//!   - Several fixtures use HEVC profile/chroma combinations that the
//!     `oxideav-h265` decoder does not yet emit pixels for
//!     (Main 10, 4:4:4, monochrome).
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
//!    * `grid`           — try [`heif::decode_primary`] (composes via
//!                          per-tile decode + canvas paste).
//!    * `iovl`           — recognised; not yet composited.
//!    * other (image sequence / metadata-only) — skipped per Tier.
//! 5. Optional features are sniffed (alpha auxiliary item, embedded
//!    ICC profile, dimg / cdsc / thmb iref edges) and reported.
//!
//! The expected PNG oracle is decoded for dimensional comparison; a
//! true byte-level RGBA diff is deferred to round 3+ once the
//! HEVC-decoder side covers Main10/4:4:4/monochrome and a YUV→RGBA
//! converter is wired in.

#![cfg(feature = "heif")]

use oxideav_h265::heif::{self, ImageGrid, ImageOverlay, Property};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // BitExact is the round-3+ promotion target — no fixture is
                    // tagged BitExact in round 2 yet, but the variant is wired
                    // through so the future tightening is a one-line tag flip.
enum Tier {
    /// Expect a successful decode whose output matches `expected.png`
    /// byte-for-byte. Promote a `ReportOnly` fixture to this once the
    /// underlying HEVC decoder + YUV→RGBA conversion pipeline are
    /// complete enough to satisfy it.
    BitExact,
    /// Run end-to-end and report stats but never fail on divergence.
    /// Default for round 2 — every fixture starts here.
    ReportOnly,
    /// Out of scope this round (image sequences, pure-metadata items).
    Ignored,
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
        fixture!("single-image-1x1", ReportOnly),
        fixture!("single-image-512x512-q60", ReportOnly),
        fixture!("single-image-with-thumbnail", ReportOnly),
        fixture!("still-image-with-alpha", ReportOnly),
        fixture!("still-image-with-icc", ReportOnly),
        fixture!("still-image-with-exif", ReportOnly),
        fixture!("still-image-with-xmp", ReportOnly),
        fixture!("still-image-grid-2x2", ReportOnly),
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

#[test]
fn corpus_walk_and_report() {
    let mut stats = Stats::default();
    let mut all_messages = Vec::new();
    for fx in fixtures() {
        stats.total += 1;
        let msgs = run_one(&fx, &mut stats);
        all_messages.push((fx.name.to_string(), msgs));
    }

    eprintln!();
    eprintln!("=== HEIF corpus report ===");
    for (name, msgs) in &all_messages {
        eprintln!("[{name}]");
        for m in msgs {
            eprintln!("  {m}");
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
    // come from the box parser side which is mature.
    assert_eq!(
        stats.probed_ok, stats.total,
        "probe should accept every corpus fixture"
    );
    assert_eq!(
        stats.parsed_header_ok, stats.total,
        "parse_header should walk every corpus fixture"
    );
    // BitExact tier promotes are strict — once a fixture is tagged
    // BitExact, a divergence MUST fail the build.
    assert_eq!(
        stats.bit_exact_failed, 0,
        "BitExact-tier fixtures must match expected.png byte-for-byte"
    );
}

fn run_one(fx: &Fixture, stats: &mut Stats) -> Vec<String> {
    let mut msgs = Vec::new();

    // 1. probe
    let p = heif::probe(fx.heic);
    msgs.push(format!("tier={:?} heic_bytes={}", fx.tier, fx.heic.len()));
    if !p {
        msgs.push("probe: REJECTED".to_string());
        return msgs;
    }
    stats.probed_ok += 1;
    msgs.push("probe: ok".to_string());

    if matches!(fx.tier, Tier::Ignored) {
        stats.ignored += 1;
        msgs.push("ignored: out of scope this round (e.g. image sequence track)".to_string());
        return msgs;
    }

    // 2. parse_header
    let hdr = match heif::parse_header(fx.heic) {
        Ok(h) => h,
        Err(e) => {
            msgs.push(format!("parse_header: ERR {e}"));
            return msgs;
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
            return msgs;
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
    if let Some(Property::Colr(c)) = hdr.meta.property_for(primary_id, b"colr") {
        msgs.push(format!("colr: {c:?}"));
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
            // promoted to BitExact. Round 2 has none promoted.
            if matches!(fx.tier, Tier::BitExact) {
                if let Some(ref o) = oracle {
                    let ok = compare_bit_exact(o, &vf);
                    if ok {
                        stats.bit_exact += 1;
                        msgs.push("bit-exact: MATCH".to_string());
                    } else {
                        stats.bit_exact_failed += 1;
                        msgs.push("bit-exact: DIFF (see plane stats)".to_string());
                    }
                } else {
                    msgs.push(
                        "bit-exact: SKIP (oracle PNG decode failed — can't compare)".to_string(),
                    );
                }
            }
        }
        Err(e) => {
            stats.decode_failed += 1;
            msgs.push(format!("decode_primary: ERR {e}"));
        }
    }

    msgs
}

/// Plane-level byte equality. Compares only the planes both frames
/// have in common; mismatched plane counts always fail.
#[allow(dead_code)] // Wired in for the round-3+ BitExact promotion; round 2
                    // ships every fixture as ReportOnly so this helper isn't
                    // hit yet.
fn compare_bit_exact(
    oracle: &oxideav_core::VideoFrame,
    actual: &oxideav_core::VideoFrame,
) -> bool {
    if oracle.planes.len() != actual.planes.len() {
        eprintln!(
            "  bit-exact: plane count {} != {}",
            actual.planes.len(),
            oracle.planes.len()
        );
        return false;
    }
    for (i, (a, e)) in actual.planes.iter().zip(oracle.planes.iter()).enumerate() {
        if a.stride != e.stride {
            eprintln!(
                "  bit-exact: plane {i} stride {} != {}",
                a.stride, e.stride
            );
            return false;
        }
        if a.data != e.data {
            let mut diff = 0usize;
            for (av, ev) in a.data.iter().zip(e.data.iter()) {
                if av != ev {
                    diff += 1;
                }
            }
            eprintln!(
                "  bit-exact: plane {i} differs in {diff} of {} bytes",
                a.data.len()
            );
            return false;
        }
    }
    true
}
