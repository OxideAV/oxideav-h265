# Round-429 loop-filter encoder pins — generation notes

Encoder-side in-loop-filter pins (`../loopfilter_encoder_interop.rs`):
each stream is this crate's own deterministic encode with the §8.7
loop filters enabled (`LoopFilterCfg::all()` — deblocking + luma +
chroma SAO), and each was validated OUT OF BAND against a black-box
reference HEVC decoder, which reproduces the encoder's FILTERED
reconstruction byte for byte.

## Tooling (black-box binary invocations only)

* Reference decode: `ffmpeg` 8.1 CLI (Apple clang build), invoked as

  ```
  ffmpeg -threads 1 -i <name>.hevc -f rawvideo -pix_fmt yuv420p <name>.yuv
  ```

  and compared byte-exactly (`cmp`) against the planar concatenation
  of the per-frame `recon` planes the encoder returned.

## Streams

| file | content | shape |
| --- | --- | --- |
| `r429-lf-pgop-qp27.hevc` | the `p_gop_encoder_interop` moving-square clip, 64x64, 5 frames | `IDR + 4x P`, QP 27, deblock + SAO |
| `r429-lf-bgop-qp33.hevc` | same clip generator, 48x48, 5 frames | low-delay B, GOP 3 (mid-stream IDR refresh), QP 33, deblock + SAO |
| `r429-lf-intra-qp32.hevc` | the `intra_encoder_interop` gradient frame, 64x64 | one intra IDR AU, QP 32, deblock + SAO |

The deblocking election sweeps `slice_beta_offset_div2` /
`slice_tc_offset_div2` over {−2, 0, 2}² (plus off), so pinned streams
may carry non-zero slice offsets.

SHA-256:

```
a04fae07c49f603cf2da0eae10bed0c8efef15abd10e3bb0afdf4f98695f1582  r429-lf-pgop-qp27.hevc
567130b696815bff5b45e6e1bfc9039312682b43d7a2443da9e52db057cd0409  r429-lf-bgop-qp33.hevc
519f84b77cd6fe7f7a93fe367a639c2f8033934bb85776833ad12fd0c78c367d  r429-lf-intra-qp32.hevc
```

## Out-of-band sweep

Beyond the three pinned streams, a 72-configuration filtered sweep
(geometries 64x64 / 48x32 / 32x80 × QP 17 / 27 / 38 × P / B slices ×
{deblock-only, SAO-luma+chroma, SAO-luma-only, deblock+SAO}, 4 frames
each) was generated the same way and every stream's black-box
reference decode matched the encoder reconstruction byte-exactly
(72/72; re-run after the β/tC-offset election landed — 75/75 with
the three pins included). The in-tree unit tests hold the same
streams bit-exact through this crate's own decoder, closing the
three-way contract.

## Round-453 re-pin

The two GOP streams were regenerated in round 453 after the
motion-search λ moved to the SAD domain (integer square root of the
SSD-domain mode λ; the previous SSD-domain value collapsed the search
to the zero vector above QP 35). Both were re-validated byte-exact
against the same black-box reference decoder (ffmpeg 8.1.2 CLI, same
invocation) and the SHA-256 lines above were updated; the intra
stream is unaffected.

The same two streams were regenerated once more in round 453 when the
integer motion search gained its two-start form (subsampled ±24 grid
scan + coarse-to-fine square search beside the seed-refined result)
and the motion λ its 1.5× weight; re-validated byte-exact the same
way, SHA-256 lines updated.
