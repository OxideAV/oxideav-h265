# Round-479 HEIF conformance still pins — generation notes

The `r479/*.hevc` streams (`../heif_conformance_stills.rs`) are the
HEVC payloads of two HEIF conformance-suite items (ISO/IEC
23008-12 conformance, `docs/image/heif/conformance/heic/`):

| stream | source file | item | layout |
|---|---|---|---|
| `iff_hevc_single_item.hevc` | `iff_hevc_single_item.heic` | primary `hvc1` | 1280x720 4:2:0 8-bit, Main |
| `iff_hevc_single_item_main10.hevc` | `iff_hevc_single_item_main10.heic` | primary `hvc1` | 1920x1080 4:2:0 10-bit, Main 10 |

Both SPSs carry a `bitstream_restriction` block whose
`max_bytes_per_pic_denom` (26 / 66) breaks the §E.3.1 bound of 16 —
the producer's declaration about its own bitstream, not an input to
the decoding process (§E.1). The crate refused them before round 479;
the block is now carried as coded and reported through
`BitstreamRestriction::range_violation`.

* Extraction: `ffmpeg -i x.heic -c:v copy -bsf:v hevc_mp4toannexb
  -f hevc x.hevc` (black-box; Annex B elementary stream of the
  primary item, the `hvcC` parameter sets in band).
* Reference digest: `ffmpeg -i x.hevc -f rawvideo -` → MD5 of the
  planar output (8-bit samples as bytes, 10-bit as 16-bit
  little-endian), which this crate reproduces byte for byte.

SHA-256:

```
7f5e47e79cd60feeb1c8a2b20271a3e3d40d99da1da83d707e24c7c2277021ec  iff_hevc_single_item.hevc
28bb370eca6aff0a7b7d0e7a421e59034197974c9181a093a23fc498c6775b0d  iff_hevc_single_item_main10.hevc
```
