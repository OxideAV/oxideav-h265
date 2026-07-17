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
