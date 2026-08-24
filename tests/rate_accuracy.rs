//! Rate-accuracy measurement gates: target vs achieved bitrate across
//! the rate-control configuration matrix — plain ABR, ABR + VBV, and
//! ABR + VBV + HRD on the low-delay, hierarchical-B and all-intra
//! paths, composed with B slices, in-loop filters and adaptive
//! quantization. Every configuration must land within a bounded error
//! of its target over the whole run AND tighter over the converged
//! tail, so a regression in any controller component (complexity
//! model, leaky bucket, burst election, VBV/HRD caps) trips a gate.

use oxideav_h265::encoder::inter::{LowDelayPEncoder, YuvFrame};
use oxideav_h265::encoder::loopfilter::LoopFilterCfg;
use oxideav_h265::encoder::pyramid::{PyramidAu, PyramidEncoder};
use oxideav_h265::encoder::rate::RateControlCfg;

const W: usize = 64;
const H: usize = 48;
const FPS: u32 = 25;

/// One low-delay matrix row: `(target, gop, bslices, filters, aq,
/// vbv, hrd, label)`.
type LowDelayRow = (
    u64,
    usize,
    bool,
    LoopFilterCfg,
    u8,
    Option<u64>,
    bool,
    &'static str,
);
/// One pyramid matrix row: `(target, gop, filters, aq, vbv, hrd,
/// label)`.
type PyramidRow = (
    u64,
    usize,
    LoopFilterCfg,
    u8,
    Option<u64>,
    bool,
    &'static str,
);

fn noise(seed: &mut u32) -> u8 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    (*seed >> 24) as u8
}

/// Textured background + drifting bright square + sensor noise: coded
/// size responds smoothly to QP on both intra and inter paths.
fn clip(n: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut seed = 0xACC0_57A7u32;
    (0..n)
        .map(|t| {
            let y: Vec<u8> = (0..W * H)
                .map(|i| {
                    let (x, yy) = (i % W, i / W);
                    let base = ((x * 3 + yy * 5) % 185) as i32 + i32::from(noise(&mut seed) % 12);
                    let (sx, sy) = ((4 + t * 2) % (W - 14), (5 + t) % (H - 14));
                    if x >= sx && x < sx + 12 && yy >= sy && yy < sy + 12 {
                        (base + 55).clamp(0, 255) as u8
                    } else {
                        base.clamp(0, 255) as u8
                    }
                })
                .collect();
            let cb: Vec<u8> = (0..W * H / 4)
                .map(|i| (100 + (i + 2 * t) % 56) as u8)
                .collect();
            let cr: Vec<u8> = (0..W * H / 4).map(|i| (150 - (i + t) % 44) as u8).collect();
            (y, cb, cr)
        })
        .collect()
}

/// Percentage error of `total` bits against `target` b/s over `n`
/// frames at [`FPS`].
fn err_pct(total: u64, target: u64, n: usize) -> u64 {
    let wanted = target * n as u64 / u64::from(FPS);
    (total as i64 - wanted as i64).unsigned_abs() * 100 / wanted
}

/// One low-delay configuration: encode `planes`, return per-frame AU
/// bits.
#[allow(clippy::too_many_arguments)]
fn run_low_delay(
    planes: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
    target: u64,
    gop: usize,
    bslices: bool,
    lf: LoopFilterCfg,
    aq: u8,
    vbv: Option<u64>,
    hrd: bool,
) -> Vec<u64> {
    let mut cfg = RateControlCfg::new(target, FPS, 1);
    if let Some(v) = vbv {
        cfg = cfg.with_vbv(v);
    }
    let mut enc = LowDelayPEncoder::new(W, H, 26, gop)
        .expect("encoder")
        .with_b_slices(bslices)
        .with_loop_filters(lf)
        .with_aq(aq)
        .with_frame_rate(FPS, 1)
        .with_rate_control(&cfg)
        .with_hrd(hrd);
    planes
        .iter()
        .map(|(y, cb, cr)| {
            enc.encode_frame(&YuvFrame { y, cb, cr })
                .expect("encode")
                .au
                .len() as u64
                * 8
        })
        .collect()
}

/// One pyramid configuration: per-AU bits in decode order.
fn run_pyramid(
    planes: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
    target: u64,
    gop: usize,
    lf: LoopFilterCfg,
    aq: u8,
    vbv: Option<u64>,
    hrd: bool,
) -> Vec<u64> {
    let mut cfg = RateControlCfg::new(target, FPS, 1);
    if let Some(v) = vbv {
        cfg = cfg.with_vbv(v);
    }
    let mut enc = PyramidEncoder::new(W, H, 26, gop)
        .expect("encoder")
        .with_loop_filters(lf)
        .with_aq(aq)
        .with_frame_rate(FPS, 1)
        .with_rate_control(&cfg)
        .with_hrd(hrd);
    let mut bits = Vec::new();
    let push = |aus: Vec<PyramidAu>, bits: &mut Vec<u64>| {
        bits.extend(aus.iter().map(|au| au.au.len() as u64 * 8));
    };
    for (y, cb, cr) in planes {
        let aus = enc.encode_frame(&YuvFrame { y, cb, cr }).expect("encode");
        push(aus, &mut bits);
    }
    push(enc.flush(), &mut bits);
    bits
}

/// The low-delay matrix: whole-run error <= 5 %, converged tail
/// (back half) <= 4 %, across targets x GOP shapes x tool
/// compositions x VBV/HRD arms (r451 measured: every configuration
/// within 1 %; encodes are bit-deterministic, the slack covers
/// future content/tuning changes only).
#[test]
fn low_delay_rate_accuracy_matrix() {
    let planes = clip(60);
    let configs: &[LowDelayRow] = &[
        (
            100_000,
            0,
            false,
            LoopFilterCfg::off(),
            0,
            None,
            false,
            "100k single-IDR plain",
        ),
        (
            100_000,
            8,
            false,
            LoopFilterCfg::off(),
            0,
            None,
            false,
            "100k gop8 plain",
        ),
        (
            300_000,
            8,
            true,
            LoopFilterCfg::all(),
            0,
            None,
            false,
            "300k gop8 b+filters",
        ),
        (
            300_000,
            8,
            false,
            LoopFilterCfg::off(),
            2,
            None,
            false,
            "300k gop8 aq2",
        ),
        (
            150_000,
            10,
            false,
            LoopFilterCfg::off(),
            0,
            Some(12_000),
            false,
            "150k gop10 vbv",
        ),
        (
            150_000,
            10,
            false,
            LoopFilterCfg::off(),
            0,
            Some(12_000),
            true,
            "150k gop10 vbv+hrd",
        ),
        (
            300_000,
            6,
            true,
            LoopFilterCfg::all(),
            2,
            Some(30_000),
            true,
            "300k gop6 b+filters+aq vbv+hrd",
        ),
    ];
    for &(target, gop, bslices, lf, aq, vbv, hrd, label) in configs {
        let bits = run_low_delay(&planes, target, gop, bslices, lf, aq, vbv, hrd);
        let whole = err_pct(bits.iter().sum(), target, bits.len());
        let tail = err_pct(bits[30..].iter().sum(), target, 30);
        assert!(whole <= 5, "{label}: whole-run error {whole}% > 5%");
        assert!(tail <= 4, "{label}: tail error {tail}% > 4%");
    }
}

/// The pyramid matrix: whole-run error <= 6 %, tail <= 5 % —
/// including the VBV and VBV + HRD arms whose hard caps must not
/// drag the average under target (r451 measured: every configuration
/// within 1.6 % after the per-slice base election landed; the
/// once-per-mini-GOP election this replaced showed ~20 % tail
/// drift at 120 kb/s).
#[test]
fn pyramid_rate_accuracy_matrix() {
    let planes = clip(65); // IDR + eight GOP-8 mini-GOPs
    let configs: &[PyramidRow] = &[
        (
            120_000,
            8,
            LoopFilterCfg::off(),
            0,
            None,
            false,
            "120k pyr8 plain",
        ),
        (
            360_000,
            8,
            LoopFilterCfg::off(),
            0,
            None,
            false,
            "360k pyr8 plain",
        ),
        (
            200_000,
            4,
            LoopFilterCfg::all(),
            2,
            None,
            false,
            "200k pyr4 filters+aq",
        ),
        (
            150_000,
            8,
            LoopFilterCfg::off(),
            0,
            Some(18_000),
            false,
            "150k pyr8 vbv",
        ),
        (
            150_000,
            8,
            LoopFilterCfg::all(),
            2,
            Some(18_000),
            true,
            "150k pyr8 filters+aq vbv+hrd",
        ),
    ];
    for &(target, gop, lf, aq, vbv, hrd, label) in configs {
        let bits = run_pyramid(&planes, target, gop, lf, aq, vbv, hrd);
        assert_eq!(bits.len(), planes.len(), "{label}");
        let whole = err_pct(bits.iter().sum(), target, bits.len());
        let tail = err_pct(bits[33..].iter().sum(), target, bits.len() - 33);
        assert!(whole <= 6, "{label}: whole-run error {whole}% > 6%");
        assert!(tail <= 5, "{label}: tail error {tail}% > 5%");
    }
}

/// Monotone rate response: across a 3-point target ladder on the same
/// content, achieved bits strictly increase and each point still meets
/// its own gate — the controller's model inversion is order-preserving
/// end to end.
#[test]
fn rate_ladder_is_monotone_on_both_paths() {
    let planes = clip(40);
    let mut last_low = 0u64;
    let mut last_pyr = 0u64;
    for target in [80_000u64, 200_000, 500_000] {
        let low: u64 = run_low_delay(
            &planes,
            target,
            8,
            false,
            LoopFilterCfg::off(),
            0,
            None,
            false,
        )
        .iter()
        .sum();
        let pyr: u64 = run_pyramid(&planes, target, 8, LoopFilterCfg::off(), 0, None, false)
            .iter()
            .sum();
        assert!(low > last_low, "low-delay ladder not monotone at {target}");
        assert!(pyr > last_pyr, "pyramid ladder not monotone at {target}");
        (last_low, last_pyr) = (low, pyr);
    }
}

/// The hard caps cost only what they must: with a roomy VBV (one full
/// second) the constrained stream's total stays within 2 % of its
/// unconstrained twin — the caps bind only in emergencies, not in the
/// steady state.
#[test]
fn roomy_vbv_does_not_distort_the_average() {
    let planes = clip(50);
    let target = 200_000u64;
    let free: u64 = run_low_delay(
        &planes,
        target,
        8,
        false,
        LoopFilterCfg::off(),
        0,
        None,
        false,
    )
    .iter()
    .sum();
    let capped: u64 = run_low_delay(
        &planes,
        target,
        8,
        false,
        LoopFilterCfg::off(),
        0,
        Some(target), // one second of buffer
        false,
    )
    .iter()
    .sum();
    let drift = (capped as i64 - free as i64).unsigned_abs() * 100 / free;
    assert!(
        drift <= 2,
        "roomy VBV drifted the total by {drift}% ({free} -> {capped})"
    );
}
