# oxideav-h265

Pure-Rust **H.265 / HEVC** (ITU-T H.265 | ISO/IEC 23008-2) decoder for
oxideav. Zero C dependencies, no FFI, no `*-sys` crates — the whole
pipeline from NAL framing through intra prediction, inverse transform,
and reconstruction is written in safe Rust.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-h265 = "0.0"
```

## Parse support

* **NAL framing** — Annex B byte-stream (`0x000001` / `0x00000001`
  start codes) and length-prefixed (HVCC / MP4 `hvc1` / `hev1`),
  including emulation-prevention byte removal.
* **Parameter sets** — VPS, SPS, PPS fully parsed. Out of range field
  values (bit depths, RPS counts, tile dimensions, picture size) are
  rejected with `Error::InvalidData`.
* **Slice segment header** — parsed through `byte_alignment()` for
  IDR I-slices under the current decode scope.
* **HEVCDecoderConfigurationRecord (`hvcC`)** — used to populate
  VPS/SPS/PPS from MP4 / ISOBMFF `extradata` and to select the
  length-prefix size.

## Decode support

**I, P, and B slices, 8-bit 4:2:0.** Keyframe (IDR), forward-ref
P-slice, and bi-predicted B-slice packets all decode end-to-end;
`receive_frame` returns a `VideoFrame` in `PixelFormat::Yuv420P`
with the conformance-window crop already applied. Reference pictures
are looked up in a small DPB keyed by picture order count. B slices
combine two motion-compensated predictors from RPL0 and RPL1
(§8.5.3.3.3) and honour the `pred_weight_table()` weights when
signalled (§8.5.3.3.4).

### Pipeline (§9.3 / §8.4 / §8.6)

* **CABAC** — full arithmetic engine (§9.3.4.3: `ivlCurrRange` /
  `ivlOffset`, regular + bypass + terminate decode paths), per-context
  initialisation (§9.3.4.2.1), and the `rangeTabLps` / transition
  tables from §9.3.4.2.
* **Coding-tree walk** — `coding_quadtree` → `coding_unit` →
  `transform_tree` recursion driven by `split_cu_flag` and
  `split_transform_flag`; implicit splits at picture boundaries.
* **Intra mode decode** — 3-element MPM list, `prev_intra_luma_pred_flag`
  / `mpm_idx` / `rem_intra_luma_pred_mode` (§8.4.2); chroma intra mode
  via `intra_chroma_pred_mode` with luma-coincident fallback to
  mode 34.
* **Intra prediction** — all 35 modes (§8.4.4.2):
  * **PLANAR** (0), **DC** (1) — with the DC edge filter for sizes
    < 32 in luma.
  * **33 angular** (2..=34) — including the `invAngle`-projected
    reference extension for negative angles and the horizontal /
    vertical post-prediction edge filter at modes 10 / 26.
  * **Reference-sample substitution** (§8.4.4.2.3) — "all unavailable"
    neutral fill (128 at 8-bit) plus the single-direction propagation
    for partially-available neighbours.
  * **MDIS filter** — `[1 2 1]/4` 3-tap smoother and the bilinear
    strong-intra-smoothing path for 32×32 blocks when enabled at SPS
    level.
* **Inverse transforms** (§8.6.4.2):
  * **4×4 DST-VII** for intra luma 4×4 blocks.
  * **4×4 / 8×8 / 16×16 / 32×32 DCT-II** — integer basis matrices from
    Tables 8-7 / 8-8, separable 2-D pass (shift 7 → shift 20−bitDepth).
* **Dequantisation** — §8.6.3 with the `levelScale[]` rem6 table. When the
  active SPS / PPS signals `scaling_list_enabled_flag = 1` (§7.4.5),
  the `dequantize_with_matrix` path applies the explicit
  `scaling_list_data()` matrices (or the spec's default Tables 7-5 / 7-6
  when `scaling_list_pred_matrix_id_delta == 0`); otherwise the
  flat-quant `m[x][y] = 16` shortcut is used.
* **Reconstruction** — `clip1Y(pred + residual)` into the 8-bit picture
  buffer.

### Inter decode (§8.5)

* **Reference picture lists** — RPL0 and RPL1 are built from the
  current slice's short-term RPS (resolved against the SPS RPS list
  or an inline one) by looking up each delta-POC entry in the DPB.
  RPL0 lists negative deltas first (past refs), RPL1 positive
  deltas first (future refs).
* **Prediction unit syntax** — `merge_flag`, `merge_idx`, per-list
  `ref_idx_lx` (truncated rice), `mvd_coding()` (two greater-flag
  bins + EG1 remainder), `mvp_lx_flag`, and `inter_pred_idc` for
  B slices (L0 / L1 / BI). `mvd_l1_zero_flag` suppresses MVD on L1
  when signalled. R21: `inter_pred_idc` bin 0 now uses
  `CtDepth[x0][y0]` (the CB's quadtree depth, ∈ {0,1,2,3}) for
  ctxInc when `nPbW + nPbH != 12` and ctx 4 otherwise — pre-r21
  we forced ctx 0 for every non-small PU, biasing the bin's
  arithmetic-coding context independently of CB size.
* **Partition modes** — 2Nx2N, 2NxN, Nx2N, NxN at minimum CB size, plus
  the four asymmetric motion partition (AMP) shapes
  (`PART_2NxnU` / `PART_2NxnD` / `PART_nLx2N` / `PART_nRx2N`) when the SPS
  sets `amp_enabled_flag`. AMP partition geometry (`(x0, y0, q, cb-q)`
  strip placement per §7.4.9.5 Table 7-10), partIdx-1 neighbour
  suppression (§8.5.3.2.3), and merge / AMVP candidate derivation share
  the symmetric-rect code path. The smoke test
  `hevc_amp_smoke_decodes_without_panic` exercises end-to-end AMP decode
  and asserts bit-exact post-P IDR re-alignment.
* **Merge list** (§8.5.3.2.2) — spatial candidates A1 / B1 / B0 / A0 /
  B2 with HEVC pruning, temporal candidate from the collocated
  reference when `slice_temporal_mvp_enabled_flag` is set
  (§8.5.3.2.8 / §8.5.3.2.9), combined bi-predictive fillers for B
  slices (§8.5.3.2.4), and zero-MV / zero-bi-MV filler
  (§8.5.3.2.5). The round-21 audit pinned three remaining
  divergences: (a) §8.5.3.2.4 now follows the spec's Table 8-7
  combIdx ordering and adds `combCandk` whenever
  `DiffPicOrderCnt(L0, L1) != 0 || mvL0 != mvL1` (no global dedup
  against existing entries); (b) §8.5.3.2.5 zero-MV pads now ramp
  `refIdxLX = (zeroIdx < numRefIdx) ? zeroIdx : 0` per spec rather
  than collapsing every pad slot onto `refIdx = 0`; (c) §8.5.3.2.2
  step 10 forces `predFlagL1 = 0, refIdxL1 = -1` after candidate
  selection on 4×8 / 8×4 PUs (`nOrigPbW + nOrigPbH == 12`).
  §8.5.3.2.3 redundancy now compares only `(predFlag, refIdx, mv)`
  on each list — the shadow `ref_poc` / `ref_lt` metadata is no
  longer part of the equality, matching the spec's "same motion
  vectors and same reference indices" wording.
* **AMVP** (§8.5.3.1.6) — per-list spatial MV predictor pair with
  deduplication, plus a TMVP candidate sourced from the collocated
  reference (no POC-distance scaling — the short GOPs we target
  keep the scale factor ≈ 1).
* **Motion compensation** — 8-tap luma and 4-tap chroma sub-pel
  interpolation (§8.5.3.2.2 / §8.5.3.2.3) with the 16-phase filter
  tables; full 2-D separable filter with 8-bit sample clipping.
* **Bi-prediction** (§8.5.3.3.3) — per-list MC samples are kept at
  16-bit pre-shift precision and combined as `(a + b + 64) >> 7`
  into the final 8-bit predictor.
* **Weighted bi-pred** (§8.5.3.3.4) — when `pred_weight_table()` is
  emitted the decoder applies per-reference luma and chroma weights
  and offsets during the bi-pred combine.
* **Residual-on-MC** — inter TUs share the CABAC residual decoder
  used for intra and are added to the MC prediction (clipped to
  8-bit). 4×4 inter luma uses DCT-II (not DST-VII).

### Loop filters

* **Deblocking filter** (§8.7.2) — implemented. Boundary-strength
  derivation is a best-effort approximation (see comments in
  `deblock.rs`); the integration test
  `hevc_deblock_smptebars_64_psnr` keeps the reconstructed Y plane
  within a 40 dB floor of the ffmpeg reference (currently >50 dB on
  `smptebars-64`).
* **SAO** (§8.7.3) — implemented. Per-CTB params are parsed during the
  CABAC walk and the EO / BO filter passes are applied after deblocking.

### Bit depths and chroma

* **Main (8-bit)** and **Main 10** are supported. Higher bit depths
  (Main 12) and `bit_depth_luma_minus8 != bit_depth_chroma_minus8`
  surface as `Error::Unsupported`.
* **4:2:0 (`chroma_format_idc=1`) and 4:2:2 (`chroma_format_idc=2`)** are
  supported for I, P, and B slices. The 4:2:2 inter path applies
  §8.5.3.2.10 chroma MV derivation
  (`mvCLX[1] = mvLX[1] * 2 / SubHeightC`) so the same `chroma_mc` /
  `chroma_mc_hp` covers both layouts.
* **4:4:4 (`chroma_format_idc=3`)**, monochrome, and
  `separate_colour_plane_flag = 1` all surface as
  `Error::Unsupported`.

### Tooling already in scope

* **Tiles** (§6.5.1) — multi-tile streams decode through a per-tile
  CABAC re-init at each entry-point byte (`hevc_tiles_fixture_decodes`).
* **Wavefront parallel processing** (§6.3.2) — single-thread WPP
  re-init from the row-above context snapshot
  (`hevc_wpp_fixture_decodes`).
* **Scaling lists** (§7.4.5) — explicit `scaling_list_data()` from the
  SPS or PPS, plus the spec defaults (Table 7-5 / 7-6) when
  `scaling_list_pred_matrix_id_delta == 0`. The
  `dequantize_with_matrix` path is byte-exact against ffmpeg on the
  intra fixture (`hevc_scaling_lists_intra_64_matches_ffmpeg`).

### Not yet implemented

* **List modification** — `ref_pic_list_modification()` is rejected.
* **Long-term reference pictures** — rejected.
* **Bit depths > 10** — Main 12 streams are rejected with
  `Error::Unsupported("h265 pixel decode limited to bit_depth <= 10")`.
* **PCM coding units** — `pcm_enabled_flag = 1` streams are rejected.
* **Transform skip** — rejected.
* **Slice segment header extension** bytes past the v1 I-slice path.
* **Scalable / multiview / 3D extensions** (SHVC, MV-HEVC, 3D-HEVC).

## Encode support

The encoder lives in `src/encoder/` and implements the
`oxideav_core::Encoder` trait. Output is Annex B byte-stream HEVC,
4:2:0 (Main / Main 10 — round 25), 16×16 CTU + 16×16 CU layout, fixed
QP 26, no SAO, no deblocking. The current envelope is small but
spec-correct: every bitstream we emit is decodable by ffmpeg's
libavcodec hevc and by our own decoder.

* **I slices (round 1+)** — IDR `IdrNLp` keyframes with VPS / SPS /
  PPS prefixed. Per-CU intra mode decision over a 7-mode subset
  (planar, DC, four cardinals, two diagonals). One 16×16 luma TB +
  two 8×8 chroma TBs per CU.
* **P slices (round 13+)** — `TrailR` slices referencing the
  immediately preceding picture. Per-CU integer-pel ±8 SAD search,
  one AMVP candidate (`mvp_l0_flag = 0`), explicit MVD coding.
* **B slices (round 22)** — `TrailR` slices referencing one earlier
  + one later anchor (`I-P-B-P-B` decode order, `I-B-P-B-P` display
  order). Opt-in via
  `HevcEncoder::from_params_with_mini_gop(params, 2)`. Per-CU
  rate-distortion choice between L0-only / L1-only / Bi (each a
  separate ±8 luma SAD search; bipred is a round-half-up
  `(P0 + P1 + 1) >> 1` average matching §8.5.3.3.3.1 default
  weighting). Both AMVP candidates per list collapse to a single
  predictor; explicit MVD on each list (`mvd_l1_zero_flag = 0`).
  `inter_pred_idc` ctxInc derivation matches the §9.3.4.2.2 Table
  9-32 CtDepth path. The display-order frames the caller sends are
  reordered internally so the held-back B always emits **after** its
  future anchor.
* **Merge mode + B_Skip (round 23)** — B-slice CUs now choose between
  three modes per CU based on luma SAD:
  - **B_Skip** (`cu_skip_flag = 1` + `merge_idx`): emitted when the
    best merge candidate's prediction SAD is at or below
    `SKIP_RESIDUAL_THRESHOLD` (≈ 1024 luma pels for a 16×16 CU,
    ≈ 4 pels per pixel — one Qstep at QP 26). Static / near-static
    CUs collapse from a full residual to a 2-bit skip CU, dropping
    a 5-frame static-content B-slice from hundreds of bytes down to
    24-26 bytes total. No residual, no AMVP, no `merge_flag` —
    just `cu_skip_flag = 1` then the `merge_idx` truncated-rice bins.
  - **Merge** (`merge_flag = 1` + `merge_idx`): emitted when the best
    merge candidate beats the explicit AMVP candidate on luma SAD but
    still has non-trivial residual. The candidate is materialised
    against the same merge derivation the decoder runs
    (`build_merge_list_full` — spatial A0/A1/B0/B1/B2, combined
    bi-pred, zero-MV pad), so encoder + decoder agree on the picked
    candidate's MV byte-for-byte.
  - **Explicit AMVP** (round 22 path): the fall-through when neither
    skip nor merge wins.
  Slice-header `five_minus_max_num_merge_cand` is now `0` (was `4`)
  so the merge list grows from 1 → 5 entries — the SAD search picks
  whichever spatial / combined / pad candidate matches the source
  best. `cu_skip_flag` ctxInc derivation uses the spec's `condTermFlagX
  = neighbour.is_skip` rule (decoder's r19 fix), with the encoder
  maintaining a per-4×4 `is_skip` grid via `InterState` so the
  context derivation stays in lock-step with the decoder.
* **Quality at QP 26 / 5-frame I-P-B-P-B**, 64×64 gradient with 1-pel
  per-frame motion: I 44.99 dB, P 32.51 dB, B 38.56 dB, P 27.17 dB,
  B 30.82 dB (self-roundtrip). On static (zero-motion) input the
  skip path dominates: ffmpeg cross-decode produces 44.99 / 28.77 /
  35.18 / 22.86 / 25.97 dB at POC 0..4. The ffmpeg cross-decode test
  in `tests/encoder_b_slice.rs` confirms each frame is bitstream-valid.
* **Main 10 encode (round 25)** — `Yuv420P10Le` source frames are now
  accepted by `HevcEncoder::from_params`. The Main 10 path emits a
  **profile_idc = 2** SPS with `bit_depth_luma_minus8 =
  bit_depth_chroma_minus8 = 2` and the matching VPS PTL (the Main +
  Main 10 compatibility bits in `general_profile_compatibility_flag`
  are both set per §A.3.5). The pipeline keeps the round-1+ I-slice
  CTU layout (16×16 luma TB + two 8×8 chroma TBs, intra mode subset)
  but switches every sample container to `u16` and threads
  `bit_depth = 10` through `intra_pred::predict` /
  `transform::{forward,inverse,quantize,dequantize}_*`. The forward
  quantiser uses **Qp'Y = SliceQpY + QpBdOffsetY = 26 + 12 = 38**
  to mirror the decoder's `get_qp` (eq. 8-284); the same QP'C = 38
  drives the chroma TBs. Self-roundtrip and ffmpeg cross-decode of a
  64×64 / 128×128 10-bit gradient both clear ~45 dB Y on the
  10-bit (peak = 1023) scale. Scope: I-slice only — the encoder
  emits an IDR for every input frame at 10-bit (the 8-bit P/B path
  is unchanged and still requires `Yuv420P`); `mini_gop_size > 1` at
  10-bit is rejected at construction time.
* **Pending:** AMP / rectangular partitions (Nx2N / 2NxN), weighted
  bi-pred, mini-GOP > 2 (B-pyramid), `mvd_l1_zero_flag` optimisation,
  10-bit P/B-slice encode (currently the 10-bit emit path is
  IDR-only), 12-bit / 4:4:4 encode.

## HEIF / HEIC still images

**Scaffold — feature-gated, off by default.** HEIC is an HEVC-coded
still image wrapped in the HEIF (ISO/IEC 23008-12) container, which
itself is a specialisation of ISO/IEC 14496-12. Because every layer
above the HEVC bitstream is container-level, the HEIF parser can
live inside this crate and delegate the pixel decode to the existing
`HevcDecoder` — no separate codec crate is needed.

Enable with:

```toml
[dependencies]
oxideav-h265 = { version = "0.0", features = ["heif"] }
```

```rust
#[cfg(feature = "heif")]
{
    use oxideav_h265::heif;
    let bytes = std::fs::read("photo.heic")?;
    if heif::probe(&bytes) {
        let frame = heif::decode_primary(&bytes)?;
        // frame.format == PixelFormat::Yuv420P (or Yuv420P10Le for Main 10).
    }
}
# Ok::<(), oxideav_core::Error>(())
```

### What the scaffold walks

`ftyp` (brand sniff for `heic`, `heix`, `heim`, `heis`, `hevc`,
`hevx`, `mif1`, `msf1`) → `meta` → nested `hdlr`, `pitm`, `iinf` /
`infe` (v2 / v3), `iloc` (v0 / v1 / v2, `construction_method == 0`),
`iref` (v0 / v1), `iprp` / `ipco` / `ipma`. Typed item properties
extracted: `hvcC`, `ispe`, `colr`. Unknown properties are retained
as raw bytes so `ipma` association indices stay consistent.

### Decode path

`probe` → `parse` → locate the primary item via `pitm` + `iloc` →
pre-pend VPS / SPS / PPS from the primary item's `hvcC` property
into the decoder's extradata → send the primary item's
length-prefixed HEVC payload as a single packet → pull the first
frame.

### Scaffold limits

* **HEIC-only.** Primary item types other than `hvc1` / `hev1` are
  rejected with `Error::Unsupported` — this crate does not decode AVIF
  (`av01`, see `oxideav-avif`), JPEG (`jpeg`), or image sequences
  requiring `moov` / sample-table walks.
* **Single-extent items only.** Multi-extent `iloc` entries and
  `construction_method != 0` (idat-embedded item data) surface as
  `Error::Unsupported`.
* **Grid / derivation items.** `iref`-driven grid composition
  (`dimg`), alpha auxiliaries (`auxl` + `auxC`), and image-transform
  properties (`irot` / `imir` / `clap`) are not applied — the
  scaffold decodes the HEVC payload of the primary item exactly as
  it appears in `mdat`.
* **No HEIF spec in docs/.** ISO/IEC 23008-12 was not available in
  `docs/`, so property-box decode follows the AVIF reference and
  ISO/IEC 14496-12 §8.11 (item box structure shared with AVIF).
* **Error messages name the missing box** (e.g. `"heif: missing
  'meta' box"`) to keep diagnostics actionable at scaffold stage.

## Usage

Registering the codec wires the decoder into `oxideav`'s codec
registry:

```rust
use oxideav_codec::CodecRegistry;
let mut codecs = CodecRegistry::new();
oxideav_h265::register(&mut codecs);
```

Parsing parameter sets directly without going through the registry:

```rust
use oxideav_h265::nal::{iter_annex_b, extract_rbsp, NalUnitType};
use oxideav_h265::sps::parse_sps;

let bytes: &[u8] = /* Annex B stream */;
for nal in iter_annex_b(bytes) {
    if nal.header.nal_unit_type == NalUnitType::Sps {
        let rbsp = extract_rbsp(nal.payload());
        let sps = parse_sps(&rbsp)?;
        println!(
            "{}x{} {}-bit",
            sps.cropped_width(),
            sps.cropped_height(),
            sps.bit_depth_y(),
        );
    }
}
# Ok::<(), oxideav_core::Error>(())
```

## Test fixtures

`tests/fixtures/hevc-intra.h265` is a 256×144 single IDR Annex B
stream generated with:

```sh
ffmpeg -f lavfi -i "testsrc=size=256x144:rate=24:duration=0.04" \
    -c:v libx265 \
    -x265-params "keyint=1:no-open-gop=1:wpp=0:pmode=0:pme=0:frame-threads=1:no-sao=1" \
    -pix_fmt yuv420p -f hevc tests/fixtures/hevc-intra.h265
```

`tests/fixtures/hevc-p.h265` is the matching P-slice fixture — a
256×144 two-frame (1 I + 1 P) clip:

```sh
ffmpeg -f lavfi -i "testsrc=size=256x144:rate=24:duration=0.083" \
    -pix_fmt yuv420p -c:v libx265 \
    -x265-params "keyint=2:bframes=0:no-sao=1:no-scenecut=1:no-open-gop=1:wpp=0:pmode=0:pme=0:frame-threads=1:no-tmvp=1:no-amp=1:no-rect=1:no-weightp=1:max-merge=1" \
    -f hevc tests/fixtures/hevc-p.h265
```

The `no-amp` / `no-rect` / `max-merge=1` flags keep the encoder
within the partition shapes and motion-vector fan-out that this
decoder currently supports; the wavefront / multithreading flags
avoid tiles and WPP.

`tests/fixtures/hevc-b.h265` is a 3-frame (I + B + P) 256×144 clip
that exercises the B-slice decode path:

```sh
ffmpeg -f lavfi -i "testsrc=size=256x144:rate=24:duration=0.125" \
    -pix_fmt yuv420p -c:v libx265 \
    -x265-params "keyint=3:bframes=1:no-sao=1:no-scenecut=1:no-open-gop=1:wpp=0:pmode=0:pme=0:frame-threads=1:no-amp=1:no-rect=1:no-weightp=1:no-weightb=1:max-merge=1:ref=1" \
    -f hevc tests/fixtures/hevc-b.h265
```

The integration tests
[`hevc_intra_fixture_decodes_to_plausible_picture`](tests/reference_clip.rs),
[`hevc_p_slice_fixture_decodes`](tests/reference_clip.rs), and
[`hevc_b_slice_fixture_decodes`](tests/reference_clip.rs) assert
that the I frame is plausibly coloured, the P frame decodes to
pixels distinct from the I frame, and the B frame decodes to
pixels distinct from both the I and P frames.

## License

MIT — see [LICENSE](LICENSE).
