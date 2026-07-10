# Round-410 tool-axis conformance pins — generation notes

Nine whole-bitstream decode pins (`r410.rs`), each an HEVC Annex B
stream produced by a black-box encoder binary plus the expected YUV
captured from a black-box reference decode. Every stream targets a
coding-tool axis that exposed a decoder bug during the round-410
conformance sweep; all nine decode byte-exact.

## Tooling (black-box binary invocations only)

* Encoder: `x265` CLI 3.6 (Homebrew build, 8bit+10bit+12bit).
* Source synthesis + expected YUV: `ffmpeg` 8.1 CLI
  (`ffmpeg version 8.1`, Apple clang build).
* All encodes use `--frame-threads 1 --pools 1` for determinism and
  `--fps 25 --input-csp i420`.

## Source YUV

```
ffmpeg -f lavfi -i "testsrc2=size=96x64:rate=25"   -frames:v 12 -pix_fmt yuv420p src96x64.yuv
ffmpeg -f lavfi -i "gradients=size=96x64:rate=25:speed=0.2" -frames:v 8 -pix_fmt yuv420p grad96x64.yuv
tail -c 9216 grad96x64.yuv > grad_f5.yuv          # last gradients frame
ffmpeg -f lavfi -i "testsrc2=size=192x192:rate=25" -frames:v 2 -pix_fmt yuv420p src192.yuv
```

## Expected YUV capture

For each stream:

```
ffmpeg -threads 1 -i <name>.hevc -f rawvideo -pix_fmt yuv420p <name>.exp.yuv
```

## Streams

| Pin | x265 arguments (after `--input <src> --input-res <WxH>`) | Axis |
| --- | --- | --- |
| `BPYR` (src96x64, 6 frames) | `--no-wpp --bframes 4 --b-adapt 0 --b-pyramid --ref 3 --frames 6` | B pyramid: bi-predicted reference B as collocated picture (§8.5.3.2.9 listCol) |
| `SCALING` (src96x64, 4 frames) | `--no-wpp --scaling-list default --bframes 2 --b-adapt 0 --frames 4` | §7.4.5 default scaling lists applied in intra + inter dequant |
| `STRONG` (grad_f5, 1 frame) | `--no-wpp --keyint 1 --frames 1` | §8.4.4.2.3 strong intra smoothing (biIntFlag) on smooth 32×32 gradients |
| `RECTAMP` (src96x64, 3 frames) | `--no-wpp --rect --amp --bframes 0 --tu-inter-depth 3 --frames 3` | Rectangular/AMP partitions + deep inter RQT (§7.3.8.10 deferred chroma) |
| `CI` (src96x64, 4 frames) | `--no-wpp --constrained-intra --bframes 2 --b-adapt 0 --frames 4` | §8.4.4.2.1 constrained_intra_pred_flag |
| `TSKIP` (grad96x64, 4 frames) | `--no-wpp --tskip --bframes 2 --b-adapt 0 --frames 4` | §7.3.8.11 transform_skip_flag + §8.6.2 transform-skip path |
| `WPPSLICES` (src192 192x192, 1 frame) | `--wpp --slices 2 --keyint 1 --frames 1` | WPP + multiple slices: per-slice init, in-slice §9.3.2.5 row sync, entry point in slice 2 |
| `OPENGOP` (src96x64, 8 frames) | `--no-wpp --open-gop --keyint 4 --min-keyint 4 --bframes 2 --b-adapt 0 --frames 8` | CRA + leading pictures mid-stream (2 I / 2 P / 4 B) |
| `M422` (src422 = testsrc2 96x64 yuv422p, 4 frames) | `--no-wpp --profile main422-10 -D 10 --bframes 2 --b-adapt 0 --frames 4` (expected via `-pix_fmt yuv422p10le`) | 4:2:2 lower-half cbf inheritance + stacked chroma-half residual pairing |

## SHA-256

| Pin | input.hevc (bytes) | expected.yuv (bytes) |
| --- | --- | --- |
| `BPYR` | `eb3eca0312dfce89349481dc4935eb15c53b05f21bcdcd3d64d236d1291f70f8` (4161) | `29077ce2146b63b1c5dd67aeffe8232a07685771d5b603b3e57fa94cd6d4e599` (55296) |
| `SCALING` | `000648c8ba9597e424d9323df86d56a8d988742a9e41232a4c9ff5c4bfd14526` (3739) | `500203f48ae72112374b857820392b1c2819ef837728cc9176b8543155621198` (36864) |
| `STRONG` | `003e2ed57ba993c82bd36130f590ace6454d1eeb2af7f75cf721cf16571e11fb` (2485) | `f88d1aaaa9aadeae411baf99a854046ca3ad377613d274a0eeeac4f3ee846658` (9216) |
| `RECTAMP` | `63ee0881f4eddc51c6084d070c848894de8552855aea7d1c159e1da08fc427d4` (3666) | `870dba91c449ac99be996f8bbcdb8e722fd52dfedbf021d68ce11120a40c7eb9` (27648) |
| `CI` | `cd90d6bbbc480f208800e4ff503454ef2f7d7d949f5222b099678b31d18dfeb4` (3809) | `412004642be05dc6db42b1187e5a8b6383267d4217e4d7b4071ecc83742c21ff` (36864) |
| `TSKIP` | `7ca4a85f01b598d389268f6e07b1c8a61087c103ec06fd6f42fadba874d1665d` (2570) | `07e2744ee7af212342b3bf4db49a9a42fb35284ba9f2d33304b3dad0aedd0006` (36864) |
| `WPPSLICES` | `4f57be9f0744ab5e015218f07935e84d516b61b35673e3f5df87fadc9f5579a4` (5885) | `b2046d338f763ea70a776406c06091d5ae541dca13c99e5b7405a1efe5dcafc4` (55296) |
| `OPENGOP` | `42d7968f0231949d0d5aca3cfcccedd54a35f33c4ffcae593c39fd0d38073b36` (5377) | `bb3c63bf79ce0c45016c25827962a52bf44b80be65e5a1b5c1d0b0938f723e1c` (73728) |
| `M422` | `b64bafb8030f56225752774d86b4ef938f499c003e0e94f0f69dbcbb3aeb8e78` (3927) | `7e772138ba07c664cc6965e9eab496ced646ae7f4cca77a091953d8a91904f7a` (98304) |

## Round-410 sweep record (beyond the pinned streams)

The full sweep also byte-exact-decoded: encoder defaults at multiple
geometries, `--no-strong-intra-smoothing`, `--scaling-list default`
at 10-bit (`-D 10`), `--aq-mode 2 --qg-size 16` (cu_qp_delta),
`--cu-lossless`, `--lossless` (whole-frame transquant bypass),
`--deblock=-6:-6`, `--cbqpoffs 5 --crqpoffs -5`, `--ctu 16`,
`--max-merge 1/5`, `--intra-refresh`, `--rd 6 --rdoq-level 2`,
`--hrd --aud --repeat-headers` under VBV, `--temporal-layers`,
`--radl 2`, `--weightp --weightb`, `--tu-intra-depth 4`,
plain `--wpp`, and `--wpp --slices 2/3` at 320x256.

Additional 4:2:2 / 4:4:4 byte-exact sweep records: `main422-10` at
8- and 10-bit (all-intra keyint 1, P-only, B GOPs, `--rect
--tu-inter-depth 3`, `--tskip`, `--scaling-list default`) plus
`main444-8` GOPs (`4:2:2` source: `testsrc2` yuv422p; `4:4:4`:
`gradients` yuv444p).

Note: `--wpp --slices 2` at 128x96 makes x265 disable WPP internally
("Too few rows/columns") and emit a truncated second-slice NAL that
the black-box reference decoder also rejects ("Skipping invalid
undecodable NALU"); that emission is an encoder artifact, not a
decode-conformance case.
