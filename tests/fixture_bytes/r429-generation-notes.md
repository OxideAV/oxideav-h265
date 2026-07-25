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

SHA-256:

```
835597f0abc2b79c616050f698d1eb8f7f1fb23c0d6023029d0bf5a6375d400a  r429-lf-pgop-qp27.hevc
4aa2284a9dee54e9dd2f448197f5095e3238f656fe1539b3affbaa3f1059cf71  r429-lf-bgop-qp33.hevc
4ec3649c8456ca549538879a764a9370fb7785963196b81910f6b426ebff5561  r429-lf-intra-qp32.hevc
```

## Out-of-band sweep

Beyond the three pinned streams, a 72-configuration filtered sweep
(geometries 64x64 / 48x32 / 32x80 × QP 17 / 27 / 38 × P / B slices ×
{deblock-only, SAO-luma+chroma, SAO-luma-only, deblock+SAO}, 4 frames
each) was generated the same way and every stream's black-box
reference decode matched the encoder reconstruction byte-exactly
(72/72). The in-tree unit tests hold the same streams bit-exact
through this crate's own decoder, closing the three-way contract.
