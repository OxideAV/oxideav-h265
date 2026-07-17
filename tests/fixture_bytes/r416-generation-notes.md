# Round-416 conformance pins — generation notes

## Self-built CCP pin (`r416-ccp.hevc`)

No black-box encoder binary exposes the range-extension
cross-component prediction tool (§7.3.8.12 / §8.6.6), so this stream
is SELF-BUILT by the deterministic generator in
`src/encoder/ccp_streams.rs` (this crate's own header writers + CABAC
encoder): 64x48, 4:4:4 8-bit, transquant-bypass lossless, all-intra
IDR, one 16x16 PART_2Nx2N CU per CTB with DM chroma. The per-CTB
`ResScaleVal` plan sweeps every legal magnitude (1/2/4/8) and both
signs on Cb and Cr independently, includes zero-scale controls, and
CTB 5 codes NO chroma residual while still signalling non-zero scales
(the cbf-clear + CCP block).

The unit tests in `src/encoder/ccp_streams.rs` pin the builder output
to these exact bytes and the decode to the procedural source planes.

### Black-box validation

The checked-in bytes were decoded with a black-box reference decoder
CLI and compared byte-exactly against the builder's expected planes:

```
ffmpeg -y -threads 1 -i r416-ccp.hevc -f rawvideo -pix_fmt yuv444p out.yuv
cmp out.yuv r416-ccp.exp.yuv   # byte-exact (ffmpeg version 8.1)
```

### SHA-256

| File | SHA-256 | bytes |
| --- | --- | --- |
| `r416-ccp.hevc` | `396206c63fdd9562be51a9f5495c149641299e0a1f4c9a35d9ed48e03cbc6995` | 10895 |
| expected planes (builder output, not checked in) | `0b87534fa23b19d2bf92e531a5020182b6510e6606a584c33a16c9e40f90148d` | 9216 |

## Self-built SCC pins (`r416-act.hevc` / `r416-ibc.hevc`)

Both streams are SELF-BUILT by the deterministic generators in
`src/encoder/scc_streams.rs` (same methodology as the CCP pin above).

* `r416-act.hevc` — 64x48 4:4:4 8-bit transquant-bypass all-intra
  IDR, SCC profile, `residual_adaptive_colour_transform_enabled_flag
  == 1`. CUs alternate `tu_residual_act_flag` 1 / 0; act-1 CUs carry
  forward-lifted residual triples that the §8.6.8.2 inverse restores
  exactly.
* `r416-ibc.hevc` — 64x48 4:2:0 8-bit transquant-bypass IDR whose
  single slice is `slice_type == P` with
  `sps/pps_curr_pic_ref_enabled_flag == 1` (`RefPicList0 ==
  [ currPic ]`). CTB 0 is an intra seed CU; every other CTB is an
  AMVP coding unit whose integer motion vector copies the
  already-decoded CTB to the left (or above, first column), with a
  bypass residual correcting the copy.

### Black-box validation status

The surveyed black-box reference decoder CLI (`ffmpeg` 8.1) REJECTS
both streams at the slice NAL with "Not yet implemented" — its HEVC
decoder does not support the screen-content-coding extensions. These
two pins are therefore decoder-pins only (the round-413 palette-pin
precedent): the builders and this crate's decoder are independent
codepaths meeting at the bitstream, and the §8.6.8.2 / eq. 8-98
arithmetic is additionally pinned by direct unit tests against the
spec equations.

### SHA-256

| File | SHA-256 | bytes |
| --- | --- | --- |
| `r416-act.hevc` | `6308810f32d9da478fe16352ebcaf09da72d0f5c1288830aac1d9ab50993e217` | 10937 |
| act expected planes | `01e6c0bb5e5ba6b8493e038f12fa3e466efd289f774e421a38366cfb8558a0c8` | 9216 |
| `r416-ibc.hevc` | `ec901b93aa843d22dd014d92bc87409eced0e14c04928cf235cc190ab24131d9` | 2830 |
| ibc expected planes | `b66825e55b627e956bd13450a2495676d83f447d8ed3226a746c47d44369190a` | 4608 |
