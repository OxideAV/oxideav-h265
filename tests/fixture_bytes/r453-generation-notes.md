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
3fe7878512ff6bed4f238e0ecbc6d4330ae1421d45fbe56949429756ad681bfd  r453-tree-pgop-ctb32-qp30.hevc
f0899d3047119903d9ac9da37b2fa72f1eb619e801bd18a0ed29dd85ba29cc90  r453-tree-bpyr-ctb64-qp31.hevc
177054a7cfcb7e5338d228b6a4258fa3e0ca6df149d308ee6a3bff11953d365c  r453-lowdelay-b-refs4-tmvp-qp29.hevc
b1ed063e88cd76567d67c36405006670f9c984ebaec633b23fa415b8265f3c3c  r453-pyramid-refs2-tmvp-qp29.hevc
dfc093ee1401ab7cfc52cf0b1599d15360a31a8f8b7f89cfa28091745ac82e72  r453-tree-pyramid-refs3-tmvp-qp29.hevc
```
