# Round-460 HEIF/HEIC still-picture pins — generation notes

The `r460/*.hevc` streams (`../heic_stills.rs`) are HEVC still items
extracted black-box from HEIC files written by three real-world
producers on macOS. Nothing here was produced by this crate; the
streams are opaque validator output and the pins hold the digest of a
black-box reference decode (which this crate reproduces byte for
byte, conformance-cropped).

## Tooling (black-box binary invocations only)

* Sources: a photograph resized to 133x97 (`p133.png`, 8-bit RGB),
  120x90 16-bit RGB with Gaussian noise (`p120_16.png`), 160x120
  8-bit and 16-bit grayscale (`g160.png`, `g160_16.png`), 17x13
  (`p17.png`) and 2x2 (`p2x2.png`) crops.
* Producers:
  * `sips -s format heic [-s formatOptions 100] in.png --out out.heic`
    (the OS image-conversion tool; `formatOptions 100` yields 4:4:4,
    16-bit sources yield Main 10 / 4:4:4 10);
  * `heif-enc [-q N | -L] [-b 10|12] [-p chroma=444|422]
    [-p x265:<param>=<value> ...] -p x265:hash=1 in.png -o out.heic`
    (HEIF library CLI 1.23.4 over a third-party HEVC encoder; every
    such stream carries a §D.2 decoded-picture-hash SEI — an in-band
    MD5 of the encoder's own reconstruction);
  * `magick in.png [-quality 90 -define heic:chroma=444]
    [-depth 12 -define heic:depth=12] out.heic` (general-purpose
    image converter 7.x).
* Extraction: `ffmpeg -i x.heic -map 0:v:0 -c:v copy -bsf:v
  hevc_mp4toannexb x.hevc` (Annex B elementary stream of the primary
  item).
* Reference decode: `ffmpeg -i x.hevc -f rawvideo -pix_fmt <fmt>
  x.yuv` with the stream's own pixel format (`yuv420p`, `yuv422p`,
  `yuv444p`, `gray`, `gbrp`, `yuv420p10le`, `yuv444p10le`,
  `yuv422p12le`, `gbrp10le`, `yuv420p12le`, `gray10le`); the pinned
  MD5 is of that file (planes Y / Cb / Cr; `gbrp` is G / B / R which
  is the coded 4:4:4 plane order).

## Streams

| file | producer | coded / output | format | features |
| --- | --- | --- | --- | --- |
| `henc-133` | heif-enc `-q 50` | 134x98 | 4:2:0 8-bit, Main Still Picture | defaults (CTB 64, tu-intra-depth 2, SAO, sign hiding, strong intra smoothing) |
| `henc-133-L` | heif-enc `-L` | 134x98 | 4:4:4 8-bit RGB (`gbrp`), Main 4:4:4 | lossless (`cu_transquant_bypass`) |
| `henc-133-444` | heif-enc `-q 50 -p chroma=444` | 134x98 | 4:4:4 8-bit | RExt |
| `henc-133-422` | heif-enc `-q 50 -p chroma=422` | 134x98 | 4:2:2 8-bit | RExt |
| `henc-133-ctu16-slices-wpp` | `-p x265:ctu=16 -p x265:slices=2 -p x265:wpp=1` | 134x98 | 4:2:0 8-bit | CTB 16, two slices, WPP entry points |
| `henc-133-tskip-culossless-rdoq` | `-q 40 -p x265:tskip=1 -p x265:cu-lossless=1 -p x265:rdoq-level=2 -p x265:tu-intra-depth=4` | 134x98 | 4:2:0 8-bit | transform skip, per-CU transquant bypass, RDOQ, deep RQTs |
| `henc-133-nosao-nosis-cip-sl` | `-p x265:no-sao=1 -p x265:no-strong-intra-smoothing=1 -p x265:constrained-intra=1 -p x265:scaling-list=default -p x265:no-signhide=1` | 134x98 | 4:2:0 8-bit | SAO off, strong smoothing off, constrained intra, default scaling lists, sign hiding off |
| `henc-133-aq8-cqp-deblock` | `-p x265:aq-mode=3 -p x265:qg-size=8 -p x265:cbqpoffs=-3 -p x265:crqpoffs=4 -p x265:deblock=-2,2` | 134x98 | 4:2:0 8-bit | 8x8 quantization groups, chroma QP offsets, deblocking offsets |
| `henc-133-vui-hrd-sei` | `-p x265:vui-timing-info=1 -p x265:vui-hrd-info=1 -p x265:hrd=1 -p x265:vbv-bufsize=2000 -p x265:vbv-maxrate=2000 -p x265:display-window=2,2,2,2 -p x265:sar=4:3 -p x265:range=full -p x265:aud=1` | 134x98 | 4:2:0 8-bit | VUI timing + HRD, default display window, AUD, buffering-period / pic-timing SEI |
| `henc-17` | heif-enc `-q 50` of 17x13 | 64x64 | 4:2:0 8-bit | producer pads the tiny source to one CTB |
| `henc-120-b10` | `-q 50 -b 10` | 120x90 | 4:2:0 10-bit, Main 10 | |
| `henc-120-b10-444-tskip` | `-q 50 -b 10 -p chroma=444 -p x265:tskip=1` | 120x90 | 4:4:4 10-bit | transform skip |
| `henc-120-b12-422` | `-q 50 -b 12 -p chroma=422` | 120x90 | 4:2:2 12-bit | |
| `henc-120-L-b10` | `-L -b 10` | 120x90 | 4:4:4 10-bit RGB (`gbrp10le`) | lossless |
| `henc-g160` | `-q 50` of 8-bit grayscale | 160x120 | monochrome 8-bit | `chroma_format_idc == 0` |
| `henc-g160-b10` | `-q 50 -b 10` of 16-bit grayscale | 160x120 | monochrome 10-bit | |
| `sips-133` | sips default | 134x98 | 4:2:0 8-bit, Main Still Picture | conformance window (133x97 source) |
| `sips-133-q100` | sips `formatOptions 100` | 134x98 | 4:4:4 8-bit | |
| `sips-120-16` | sips of 16-bit source | 120x90 | 4:2:0 10-bit, Main 10 | `general_one_picture_only_constraint_flag` |
| `sips-120-16-q100` | sips `formatOptions 100`, 16-bit source | 120x90 | 4:4:4 10-bit | |
| `sips-17` | sips of 17x13 | 18x14 | 4:2:0 8-bit | odd source padded to even, conformance window from 32x16 coded size |
| `sips-2x2` | sips of 2x2 | 2x2 | 4:2:0 10-bit | smallest picture (`pic_width_in_luma_samples == 2` under conformance cropping) |
| `sips-g160` | sips of grayscale | 160x120 | 4:2:0 8-bit | grayscale source coded with flat chroma |
| `magick-133-444` | `magick -quality 90 -define heic:chroma=444` | 134x98 | 4:4:4 8-bit | |
| `magick-120-d12` | `magick -depth 12 -define heic:depth=12` | 120x90 | 4:2:0 12-bit | Main 12 |

(`magick` with default options produces byte-identical streams to
`heif-enc -q 50` — same library, same encoder — so only its
non-default outputs are vendored.)

## Reference-decoder corner (not vendored)

On two CTB-16 stills of a 1024x768 photograph from the third-party
encoder (`-p x265:ctu=16 -p x265:max-tu-size=8`, and `-p x265:ctu=16
-p x265:max-tu-size=4 -p x265:min-cu-size=8`), the black-box reference
decoder's output differs from this crate's by ONE chroma sample per
plane (±1, Cb and Cr, at the bottom-right corner sample of a chroma
CTB adjacent to a CTB whose first 8x8 CU carries no residual — so its
`QpY` is `qPY_PRED` while the rest of the quantization group carries
the coded `cu_qp_delta`). The encoder's own in-band §D.2
decoded-picture-hash SEI (`-p x265:hash=1`, deterministic re-encode)
matches THIS crate's reconstruction on all three planes, so the
divergence is on the reference decoder's side; the remaining 132
streams of the 134-stream local matrix (three producers x sizes 2x2
.. 8000x2000 and 4032x3024 grid tiles x 8/10/12-bit x 4:2:0 / 4:2:2 /
4:4:4 / monochrome x lossless / QP ladders / tool axes) are
byte-exact against it.

## SHA-256

```
8c6c80bd70ee51a2792ce1f31aeed271797e0d31643d7f3ff972f357d24812c7  henc-120-L-b10.hevc
15ce193d7519ff3c452a309cdb81587714a8c970cab3ab1a494d347f77d8203c  henc-120-b10-444-tskip.hevc
1bcb7e4f5a394d3aab8e7b7e8853574f4a0dc1f602594edf1bd88026554b06cd  henc-120-b10.hevc
208d1303d115f20613539cc202077e8cb4015077b60ed31abf657b86217513ca  henc-120-b12-422.hevc
827e2c3afb1b54abd0abf0f25ac1c9a0dbf9d89ec5f4c3d333e45c21f5aedc4b  henc-133-422.hevc
374f2f6983fccfc1b0dd3baef6247969c6659d9049afadbb41442fc8864a1ab9  henc-133-444.hevc
5c0e18ccd7820d2c8281e6e3281f82d7c577cc378b0a8ee852647c115051c2e0  henc-133-L.hevc
e3f7f94a0491d2262ecbb8befa6e13d8863f79c8239397f762c4672bf4761417  henc-133-aq8-cqp-deblock.hevc
4d4157439013dfae4478c279869df137d7c69558d48be8cd09b287de3cbbf6b6  henc-133-ctu16-slices-wpp.hevc
0bdb99508f4f4d3f2d646fac889bd4456fe4708fd6b9d805b729416688a2a3ba  henc-133-nosao-nosis-cip-sl.hevc
f54f88064df1d8a8903fc496e8b0019b2e39d1a65243fa3502fa8d98f7982da8  henc-133-tskip-culossless-rdoq.hevc
5ecf24b65512ae8342ba02acc5a6c6bfbbf41ba27770215d90af6903d9af9aa6  henc-133-vui-hrd-sei.hevc
20d060af39bd51ea8b61a994b826edd7ec270288e6fa829012453ac2a0057cbc  henc-133.hevc
c76a1152dc199be8ce3198f26a110bb92767ad9808c18b454104a2bc2aaa7a12  henc-17.hevc
a2cbbf0de9068e95d409b33f8f8fe4c887a2b752ad19a3df6f5a6c304dfea9bb  henc-g160-b10.hevc
8388f0ef492e2afb006bf76326fca322b9a02c1b280b4fd006856c1bbf9d15c5  henc-g160.hevc
0dcec0f4971ad3b76ce1a3016427d7506c1231bb8eb07accf980ee72fbf0a7b6  magick-120-d12.hevc
ca6c9696b5ecdaeeb4e450c6f55246769eb2758559eaf391f3145c61acc7da2c  magick-133-444.hevc
e9f2cd93a39e3c263b2725e2689e6c0b5901d2d078920c94090adacfd0e73329  sips-120-16-q100.hevc
211da91dc089b6dced50dcfffd3663ebbc728cdea3700fbab0ec5e80aa879f8e  sips-120-16.hevc
eaa99e686a16d374cf55ab83275a46f055f843eef708578b7001295af08ad712  sips-133-q100.hevc
7dd23cf6ed2ff3badaf69fab9524a5d59de077645cf2130b7f7b226f612c8b99  sips-133.hevc
06e3bd2b4fd844cde2c1bacdb5f00ad6ef3659516f14ec6df137602b9dd2ef9a  sips-17.hevc
3649056ae7c05d32cb719ecc162818cacdca0154bf62ed3f033b818aae5b182c  sips-2x2.hevc
e453551d4fdf21f0d75fd887d535127ac4339d3bc5fc52fe10e5823fb67eae6e  sips-g160.hevc
```
