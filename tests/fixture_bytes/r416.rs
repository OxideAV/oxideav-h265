//! Round-416 self-built conformance pins: the Rext/SCC application
//! tail (cross-component prediction, adaptive colour transform, intra
//! block copy). The bitstreams were produced by this crate's own
//! deterministic generators (`src/encoder/ccp_streams.rs` /
//! `src/encoder/scc_streams.rs`); generation details, black-box
//! validation status and SHA-256 sums are in
//! `r416-generation-notes.md`.

/// §8.6.6 cross-component prediction, 4:4:4 lossless all-intra
/// (black-box reference decode: byte-exact).
pub const CCP_HEVC: &[u8] = include_bytes!("r416-ccp.hevc");
/// Expected planes for [`CCP_HEVC`] (one 64x48 yuv444p frame).
pub const CCP_YUV: &[u8] = include_bytes!("r416-ccp.exp.yuv");

/// §8.6.8 adaptive colour transform, 4:4:4 lossless all-intra.
pub const ACT_HEVC: &[u8] = include_bytes!("r416-act.hevc");
/// Expected planes for [`ACT_HEVC`] (one 64x48 yuv444p frame).
pub const ACT_YUV: &[u8] = include_bytes!("r416-act.exp.yuv");

/// Current-picture referencing (intra block copy), 4:2:0 lossless IDR
/// with a P slice listing only the current picture.
pub const IBC_HEVC: &[u8] = include_bytes!("r416-ibc.hevc");
/// Expected planes for [`IBC_HEVC`] (one 64x48 yuv420p frame).
pub const IBC_YUV: &[u8] = include_bytes!("r416-ibc.exp.yuv");
