# Round-456 encoder-tool pins — generation notes

Composed quadtree-coder streams (`../r456_tools_interop.rs`) from the
round-456 tool set — RDOQ, sign data hiding, depth-2 residual
quadtrees, weighted prediction, WPP, tiles, scaling lists — each this
crate's own deterministic encode, validated OUT OF BAND against a
black-box reference HEVC decoder, which reproduces the encoder's
reconstruction byte for byte.

## Tooling (black-box binary invocations only)

* Reference decode: `ffmpeg` 8.1.2 CLI, invoked as

  ```
  ffmpeg -i <name>.hevc -f rawvideo -pix_fmt yuv420p <name>.yuv
  ```

  and compared byte-exactly (`cmp`) against the planar concatenation
  of the per-frame `recon` planes the encoder returned (display
  order).

## Streams

The clip is a textured world panning (2, 1) px/frame under a
100 % -> 60 % luminance fade with a brighter square drifting the
other way, 96x64 (NOT a CTB multiple at CTB 64: the boundary-inferred
split path is exercised).

| file | content | shape |
| --- | --- | --- |
| `r456-pyramid-rdoq-sdh-tu2-wp-wpp-qp30.hevc` | 6 frames | GOP-4 pyramid + tail, CTB 64, RDOQ + sign hiding + `max_transform_hierarchy_depth_* == 2` + weighted prediction (fade-fitted `pred_weight_table( )` on every P / B slice) + WPP (one substream per CTB row, entry points), deblock + SAO, QP 30 |
| `r456-lowdelay-tiles-sl-rdoq-aq-qp29.hevc` | 4 frames | low-delay B, CTB 32, explicit 2x2 tile grid (`uniform_spacing_flag == 0`, one CTB column / row then the rest), RDOQ, default scaling lists (`scaling_list_enabled_flag == 1`, no `scaling_list_data( )`), AQ 1 (per-tile `qPY_PREV` resets), deblock + SAO, QP 29 |

SHA-256:

```
5cf1252559413240849f11a1469d3cbe0dee925dce46ade29cfa4d3992819141  r456-pyramid-rdoq-sdh-tu2-wp-wpp-qp30.hevc
7acaa72fc2e78cb32f6aa365b0bfceab3087befb047636497270b584bd453b4d  r456-lowdelay-tiles-sl-rdoq-aq-qp29.hevc
```
