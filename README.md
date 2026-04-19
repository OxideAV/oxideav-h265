# oxideav-h265

Pure-Rust **H.265 / HEVC** (ITU-T H.265 | ISO/IEC 23008-2) bitstream
parser for oxideav. The pixel-reconstruction pipeline is **not yet
implemented** — this crate currently decodes parameter sets and slice
headers but refuses to produce frames.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone. Zero C dependencies, no FFI, no
`*-sys` crates.

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
  first-segment IDR I-slices under:
  * 4:2:0 / 4:2:2 / 4:4:4 chroma (no `separate_colour_plane`),
  * no multi-tile and no wavefront entry points,
  * no slice-header extension payload.

  For those slices the parsed header exposes `slice_sao_luma_flag`,
  `slice_sao_chroma_flag`, `slice_qp_delta`, derived `SliceQpY`, and a
  byte-aligned `slice_data_bit_offset` that marks where the CTU
  entropy payload begins. Other slice shapes (P/B, non-IDR, tiles,
  wavefronts) parse the leading portion of the header (type, POC, PPS
  id, RPS index) and then stop — `is_full_i_slice` reports `false`
  and CTU decode refuses as below.
* **HEVCDecoderConfigurationRecord (`hvcC`)** — used to populate
  VPS/SPS/PPS from MP4 / ISOBMFF `extradata` and to select the
  length-prefix size.
* **CABAC init kernel** (§9.3.4.2.1) — the
  `(pStateIdx, valMps)` derivation from an 8-bit `initValue` byte
  plus `SliceQpY`, with a unit-tested exemplar table
  (`split_cu_flag`). The rest of the context tables and the
  arithmetic decode engine itself are not yet written.

## Decode support

**No pixel decode yet.** Calling `Decoder::receive_frame` on an
`HevcDecoder` always returns
`Error::Unsupported("hevc CTU decode not yet implemented")`.

The decoder does run every step described in *Parse support* above on
incoming packets, so tests can verify that a real ffmpeg-generated
IDR I-slice reaches the CTU-decode boundary with correctly derived
`SliceQpY` and byte-aligned `slice_data()` offset before refusing.

### Missing to produce frames

* CABAC arithmetic decode engine (ivlCurrRange / ivlOffset / renorm,
  §9.3.4.3) and the full per-syntax context tables (~50 tables from
  §9.3.4.2.X).
* Coding-quadtree traversal (CTU → CU recursion, §7.3.8.4 /
  §7.3.8.5).
* Intra prediction — planar, DC, and the 33 angular modes
  (§8.4.4.2).
* Inverse transform pipeline — 4×4 DST-VII plus 4/8/16/32 DCT-II
  inverse (§8.6.4), dequantisation (§8.6.3).
* In-loop filters — deblocking (§8.7.2) and Sample Adaptive Offset
  (§8.7.3).
* Decoded Picture Buffer / reference picture set management for
  inter-coded slices.

No HEVC encoder. No scalable / multiview / 3D extensions (SHVC,
MV-HEVC, 3D-HEVC).

## Usage

Registering the codec surfaces the parser + the graceful
`Unsupported` error so the codec registry behaves correctly when an
HEVC stream is encountered:

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

## License

MIT — see [LICENSE](LICENSE).
