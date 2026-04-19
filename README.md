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

**I-slice, 8-bit 4:2:0 only.** For packets that contain an IDR I
slice within the supported shape, `receive_frame` returns a
`VideoFrame` in `PixelFormat::Yuv420P` with the conformance-window
crop already applied.

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
* **Dequantisation** — flat scaling per §8.6.3 using the `levelScale[]`
  rem6 table.
* **Reconstruction** — `clip1Y(pred + residual)` into the 8-bit picture
  buffer.

### Not yet implemented

* **P / B slice inter prediction** — `receive_frame` returns
  `Error::Unsupported("h265 inter slice pending")` as soon as a
  non-I slice is seen. No DPB, no motion compensation, no weighted
  prediction, no merge / AMVP candidate lists.
* **Deblocking filter** (§8.7.2) — not yet applied. The reconstructed
  picture therefore carries visible block-edge artefacts; downstream
  consumers that need a post-filtered frame need to apply an external
  deblocker for now.
* **SAO** (§8.7.3) — the per-CTU SAO parameters are parsed out of the
  bitstream so the CABAC position stays correct, but the filter itself
  is not applied.
* **Bit depths other than 8** — 10-bit / 12-bit streams are rejected
  with `Error::Unsupported("h265 only 8-bit pixel decode supported")`.
* **Chroma formats other than 4:2:0** — 4:2:2, 4:4:4, monochrome, and
  `separate_colour_plane_flag = 1` are all rejected with
  `Error::Unsupported`.
* **PCM coding units** — `pcm_enabled_flag = 1` streams are rejected.
* **Scaling lists** — `scaling_list_enabled_flag = 1` streams are
  rejected (flat dequantisation only).
* **Transform skip** — rejected.
* **Tiles / wavefront parallel processing** — single-tile, WPP-off
  streams only.
* **Slice segment header extension** bytes past the v1 I-slice path.
* **Encoder** — this crate only decodes.
* **Scalable / multiview / 3D extensions** (SHVC, MV-HEVC, 3D-HEVC).

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

## Test fixture

`tests/fixtures/hevc-intra.h265` is a 256×144 single IDR Annex B
stream generated with:

```sh
ffmpeg -f lavfi -i "testsrc=size=256x144:rate=24:duration=0.04" \
    -c:v libx265 \
    -x265-params "keyint=1:no-open-gop=1:wpp=0:pmode=0:pme=0:frame-threads=1:no-sao=1" \
    -pix_fmt yuv420p -f hevc tests/fixtures/hevc-intra.h265
```

The wavefront / multithreading flags are disabled so the resulting
stream stays within the v1 single-tile decode scope. The integration
test
[`hevc_intra_fixture_decodes_to_plausible_picture`](tests/reference_clip.rs)
asserts the reconstructed luma plane has a mean in `32..=224` and
more than 20 distinct sample values.

## License

MIT — see [LICENSE](LICENSE).
