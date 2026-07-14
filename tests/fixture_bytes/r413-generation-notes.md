# Round-413 conformance pins — generation notes

Whole-bitstream decode pins (`r413.rs`), each an HEVC Annex B stream
produced by a black-box encoder binary plus the expected YUV captured
from a black-box reference decode. Every stream targets a coding-tool
axis that exposed a decoder bug during the round-413 conformance
sweep.

## Tooling (black-box binary invocations only)

* Encoder: `x265` CLI (Homebrew build, 8bit+10bit+12bit).
* Source synthesis + expected YUV: `ffmpeg` 8.1 CLI
  (`ffmpeg version 8.1`, Apple clang build).
* All encodes use `--frame-threads 1 --pools 1` for determinism and
  `--fps 25`.

## Source YUV

```
ffmpeg -f lavfi -i "testsrc2=size=96x64:rate=25" -frames:v 6 -pix_fmt yuv444p src444.yuv
```

## Expected YUV capture

```
ffmpeg -threads 1 -i <name>.hevc -f rawvideo -pix_fmt yuv444p <name>.exp.yuv
```

## Streams

| Pin | x265 arguments (after `--input <src> --input-res 96x64 --input-csp i444`) | Axis |
| --- | --- | --- |
| `LL444` (src444, 6 frames) | `--profile main444-8 --lossless --no-wpp --bframes 0 --frames 6` | 4:4:4 PART_NxN per-chroma-PB `IntraPredModeC` derivation (§7.3.8.5 four `intra_chroma_pred_mode` elements, §8.4.3 per-PB DM mapping), exposed losslessly (transquant bypass) |

## SHA-256

| Pin | input.hevc (bytes) | expected.yuv (bytes) |
| --- | --- | --- |
| `LL444` | `77f2b66aa36df14a02a1960a00da3db43a0bd697a321b5c678bb498d6a10776d` (25643) | `8c9bdf4f43f96986743eb5d84872a3e0cafcc417396c2862e2a99ba02ac607fd` (110592) |

## Self-built RDPCM pins (`r413-rdpcm-implicit.hevc` / `r413-rdpcm-explicit.hevc`)

No black-box encoder binary exposes the range-extension RDPCM tools
(the CLI encoder used above has no RDPCM switch and never emits an
`sps_range_extension()`), so these two streams are SELF-BUILT by the
deterministic generator in `src/encoder/rdpcm_streams.rs` (this
crate's own header writers + CABAC encoder; 64x48, transquant-bypass
lossless, procedural source). The unit tests in that module pin the
builder output to these exact bytes and the decode to the procedural
source planes.

Black-box reference validation (`ffmpeg -threads 1 -i <s>.hevc -f
rawvideo -pix_fmt yuv420p`):

* `r413-rdpcm-implicit.hevc` (1 IDR frame, implicit RDPCM: luma mode
  26 down the left CTB column / mode 10 along the top row, chroma
  mode 26/10 everywhere via `intra_chroma_pred_mode` 1/2, DC/PLANAR
  luma controls): reference decode is **byte-exact** against the
  procedural source — both §8.6.5 accumulation directions are
  black-box-confirmed, luma and chroma.
* `r413-rdpcm-explicit.hevc` (IDR + P, per-component
  `explicit_rdpcm_flag` with both directions, flag-0 controls, one
  skip CU): the reference decoder parses the stream cleanly, matches
  frame 0 byte-exact and every flag-0 / horizontal-direction P block,
  but diverges on the vertical-direction (`explicit_rdpcm_dir_flag
  == 1`) blocks (and neighbouring chroma blocks decoded after them).
  T-REC-H.265 is literal here — §8.5.4.2 step 3 / §8.5.4.3 step 4 set
  mDir equal to `explicit_rdpcm_dir_flag`, §8.6.5 maps mDir 0/1 to
  horizontal/vertical, and the §9.3.4.2 ctxIdx table gives the dir
  flag its own per-component context pair — and the implicit-RDPCM
  stream already black-box-confirms both accumulation directions, so
  this crate follows the spec text (documented as a known reference
  deviation in the README).

| Pin | bytes | SHA-256 |
| --- | --- | --- |
| `r413-rdpcm-implicit.hevc` | 4073 | `e1259b18ee4fe063f7a1be770920d5a834aad9e718c01fe1ecad84e741b2da03` |
| implicit expected YUV (= procedural source, 4608 B) | | `9ccc624f9c8ec4edabfaabd1f4386ec5b269e5fb680465ec6ea30a36ace4ccfe` |
| `r413-rdpcm-explicit.hevc` | 6882 | `22c20ae71bea20437c69c5ed50c497316f3ce700bf566ce7a5f0a121f51fcca5` |
| explicit expected YUV (= procedural source, 9216 B) | | `f526bdbe6aaab35a0c7941922190ae96ab0162152e4c0b4a9175ae721b50a44c` |

## Self-built SCC palette pin (`r413-palette.hevc`)

No black-box encoder OR decoder binary available to this workspace
supports the Screen Content Coding palette mode (the CLI encoder has
no SCC profile; the reference decoder binary, asked to decode this
stream, emits a mid-grey concealment frame rather than palette
reconstruction). The stream is SELF-BUILT by the deterministic
generator in `src/encoder/palette_streams.rs` (this crate's own
header writers + CABAC encoder): a 64x48 all-palette IDR picture
whose twelve 16x16 palette CUs exercise new-entry signalling,
`palette_predictor_run` reuse, explicit-index and copy-above runs,
run-to-end inference, a `MaxPaletteIndex == 0` block,
`palette_transpose_flag`, and escape samples in both forms —
transquant-bypass (FL) and quantized (EG3 at `SliceQpY == 4`, where
eq. 8-77 dequantizes exactly, keeping the stream lossless).

Validation therefore rests on (a) the spec-transcribed §7.3.8.13 /
§8.4.4.2.7 / §9.3.3.6 / §9.3.3.14 unit tests with hand-computed
vectors, and (b) the lossless whole-stream roundtrip through the real
CABAC engine and slice machinery, pinned byte-for-byte:

| Pin | bytes | SHA-256 |
| --- | --- | --- |
| `r413-palette.hevc` | 402 | `faf242663abfb8f9ce8d6e874be1ec1339f3369ac609f60def8d702510603214` |
| expected YUV (= planned palette content, 4608 B) | | `d54199726077c032f78546c5fe2ee7653a6a60f072006c33a0ec66e3013f3afd` |

## Round-413 sweep record (beyond the pinned streams)

After the per-PB chroma-mode fix, the following also byte-exact-decode
(all 96x64, 6 frames, `--no-wpp --frame-threads 1 --pools 1`):
`main444-8 --lossless --bframes 2`, `main444-8 --lossless --keyint 1`,
`main444-8 --qp 12 --keyint 1`, `main444-8 --qp 20 --bframes 2`,
`main444-10 --lossless --bframes 2` (10-bit), `main444-12 --lossless`
(12-bit), `--lossless` 4:2:0, `main422-10 --lossless --bframes 2`
(10-bit 4:2:2), `main444-8 --cu-lossless --qp 8 --bframes 2`, and
`main444-8 --tskip --qp 14 --bframes 2`.
