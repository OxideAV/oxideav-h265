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

SHA-256:

```
cccdd13185551f03d5aaf0ba43637831302e2b01776cdf4f47a6266330a2a5e4  r453-tree-intra-ctb64-qp30.hevc
95242a259829366c6fd20f2eac14ce9c8c5fde67956a3b7ef680b1b12fb8f1e3  r453-tree-pgop-ctb32-qp30.hevc
9ff30afe1fa64dbbc935638cfbdc402fcbddd2a7d58cfb17529cc8b49252e25b  r453-tree-bpyr-ctb64-qp31.hevc
```
