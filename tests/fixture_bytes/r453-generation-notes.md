# Round-453 coding-quadtree encoder pins — generation notes

Quadtree-coder (`encoder::ctu`) golden streams
(`../tree_encoder_interop.rs`): each stream is this crate's own
deterministic encode routed through the recursive coding-quadtree
coder (`with_tree` / the registry `ctb` option), and each was
validated OUT OF BAND against a black-box reference HEVC decoder,
which reproduces the encoder's reconstruction byte for byte.

## Tooling (black-box binary invocations only)

* Reference decode: `ffmpeg` 8.1.2 CLI (Apple clang build), invoked as

  ```
  ffmpeg -threads 1 -i <name>.hevc -f rawvideo -pix_fmt yuv420p <name>.yuv
  ```

  and compared byte-exactly (`cmp`) against the planar concatenation
  of the per-frame `recon` planes the encoder returned (display
  order).

## Streams

| file | content | shape |
| --- | --- | --- |
| `r453-tree-intra-ctb64-qp30.hevc` | textured square scene, 96x80, 1 frame | intra IDR, CTB 64, QP 30, deblock + SAO |
| `r453-tree-pgop-ctb32-qp30.hevc` | moving-square scene, 96x64, 5 frames | `IDR + 4x P`, CTB 32, QP 30, deblock + SAO, AQ 2 |
| `r453-tree-bpyr-ctb64-qp31.hevc` | same scene generator, 96x64, 5 frames | hierarchical-B GOP 4 + low-delay tail, CTB 64 + AMP, QP 31 |

The picture sizes are deliberately NOT CTB multiples (96x80 / 96x64
at CTB 64), so the streams exercise the §7.3.8.4 boundary-inferred
`split_cu_flag == 1` partial-CTB path; the RD election produces
mixed 32/16/8 CUs, multi-depth RQTs, 4x4 DST-VII intra luma TUs and
(in the pyramid stream) AMP shapes above MinCb.

## Temporal-MVP / multi-reference streams (`../tmvp_multiref.rs`)

Same tooling and comparison. The clip is a noisy diagonal pan with a
block that flickers on even frames only (so a two-frames-back
reference wins there); every P / B slice signals
`slice_temporal_mvp_enabled_flag == 1`.

| file | content | shape |
| --- | --- | --- |
| `r453-lowdelay-b-refs4-tmvp-qp29.hevc` | 64x48, 7 frames | low-delay B, 4 active references per list, TMVP (collocated `RefPicList0[0]`), deblock + SAO, QP 29 |
| `r453-pyramid-refs2-tmvp-qp29.hevc` | 64x48, 9 frames | hierarchical-B GOP 8, §8.3.4 lists of 2 (past+future / future+past), TMVP (collocated `RefPicList1[0]` on B), QP 29 |
| `r453-tree-pyramid-refs3-tmvp-qp29.hevc` | 64x48, 10 frames | GOP-8 pyramid + tail, 3 references, TMVP, CTB 32 quadtree, deblock + SAO, AQ 1, QP 29 |

SHA-256:

```
cccdd13185551f03d5aaf0ba43637831302e2b01776cdf4f47a6266330a2a5e4  r453-tree-intra-ctb64-qp30.hevc
8d1e52db5929b28fdc464408664ab5258305bb8e1dc83bb6531d0f3b01457b70  r453-tree-pgop-ctb32-qp30.hevc
c4d6d4f307cb046f320c7f17d37ddfe2746ae3317cc678214f273fbb1fbc522c  r453-tree-bpyr-ctb64-qp31.hevc
7a039bf3111dd09cfdf1f8499e92dadbbf688484111fd876f428070057fc4adc  r453-lowdelay-b-refs4-tmvp-qp29.hevc
8a459d762bcd0c0ca816bc6a135c94253d5ccd5e3e68db66deacd36ce1d12e8a  r453-pyramid-refs2-tmvp-qp29.hevc
019a2793af8f682d8db39669e6fab0f6904bf9bcd680bcc426cdd660d224e779  r453-tree-pyramid-refs3-tmvp-qp29.hevc
```

## Motion-search re-pin

Every inter stream above was regenerated after the integer motion
search gained its two-start form (subsampled ±24 grid scan +
coarse-to-fine square search beside the seed-refined result; motion
λ = 3·isqrt(mode λ)/2) — the change that closed the pyramid's
periodic-texture collapse — and re-validated byte-exact through the
same black-box invocation; the SHA-256 lines are the final values.

## Non-uniform tile grid (`../nonuniform_tiles.rs`)

Same tooling; the expected output is the SOURCE picture (PCM is
lossless). `r453-pcm-tiles-explicit-96x64.hevc`: a 96x64 PCM IDR
coded as one slice over an explicit 3x2 tile grid
(`uniform_spacing_flag == 0`, `column_width_minus1 = {0, 2}`,
`row_height_minus1 = {2}`) with per-tile subsets + entry points.

```
76d2746c83b86390a50b0cb63ea02067ac277e16a83c1fdc2c9af4ed0a0a49a4  r453-pcm-tiles-explicit-96x64.hevc
```

`r453-lowdelay-b-refs4-tmvp-qp29.hevc` was regenerated once more when
the low-delay SPS started signalling `sps_max_dec_pic_buffering_minus1
= refs` (4 here; it had kept the two-reference value) — re-validated
the same way, SHA-256 updated.

## CTU-level rate feedback (`../ctu_rate_control.rs`)

Same tooling and comparison. `r453-tree-lowdelay-cturc-120k.hevc`:
64x64, 12 frames (flat left half / busy right half, slow pan),
low-delay P GOP 20, CTB-32 quadtree, deblock + SAO, ABR 120 kb/s at
25 fps with CTU-level feedback (per-CTB `cu_qp_delta` against the
pro-rata frame budget).

```
81e050632665bd442ab7426803c905b7448daec23c5c7211533c3c57651e908f  r453-tree-lowdelay-cturc-120k.hevc
```

## Round-456 ladder re-pin

`r453-tree-pgop-ctb32-qp30.hevc`, `r453-tree-pyramid-refs3-tmvp-qp29.hevc`
and `r453-tree-lowdelay-cturc-120k.hevc` were regenerated when the
quadtree ladder gained the 8x4 / 4x8 `PART_2NxN` / `PART_Nx2N` inter
PUs at `MinCbSizeY` (uni-predicted per §8.5.3.2.2 step 10 /
Table 9-46) — every other pin was unaffected — and re-validated
byte-exact through the same black-box invocation. SHA-256 (final):

```
75bf091cdfec737a9897785e16fa633a502e3d002ae2c54c3b8d3455f8136d0b  r453-tree-lowdelay-cturc-120k.hevc
3ec83d6d21d838325a0ac9196abcb1a2f31123f3f8aa86775ee2099c89537f9a  r453-tree-pgop-ctb32-qp30.hevc
c9711b1ff92609f78f2cd106b6fc75b4fad26cbcbf84dc360c703ed58c917f56  r453-tree-pyramid-refs3-tmvp-qp29.hevc
```
