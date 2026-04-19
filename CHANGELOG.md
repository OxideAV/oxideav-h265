# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.4](https://github.com/OxideAV/oxideav-h265/compare/v0.0.3...v0.0.4) - 2026-04-19

### Other

- README + lib docs reflect P-slice decode support
- wire P-slice inter decode into CTU walker + decoder
- add inter module (DPB, MV, merge/AMVP, 8-tap/4-tap MC)
- extend slice header parser with P-slice extension
- capture short-term RPS deltas + counts in SPS parser
- rewrite README + lib.rs docs to reflect I-slice decode support
- land I-slice pixel decode (intra pred + transforms + CTU walker)
- port CABAC arithmetic engine + I-slice context tables
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
