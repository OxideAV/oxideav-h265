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

## Round-413 sweep record (beyond the pinned streams)

After the per-PB chroma-mode fix, the following also byte-exact-decode
(all 96x64, 6 frames, `--no-wpp --frame-threads 1 --pools 1`):
`main444-8 --lossless --bframes 2`, `main444-8 --lossless --keyint 1`,
`main444-8 --qp 12 --keyint 1`, `main444-8 --qp 20 --bframes 2`,
`main444-10 --lossless --bframes 2` (10-bit), `main444-12 --lossless`
(12-bit), `--lossless` 4:2:0, `main422-10 --lossless --bframes 2`
(10-bit 4:2:2), `main444-8 --cu-lossless --qp 8 --bframes 2`, and
`main444-8 --tskip --qp 14 --bframes 2`.
