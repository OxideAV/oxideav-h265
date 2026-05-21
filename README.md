# oxideav-h265

A pure-Rust H.265 / HEVC video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Clean-room rebuild — round 2 (2026-05-22).** The prior implementation was
retired under the workspace
[clean-room policy](https://github.com/OxideAV/oxideav/blob/master/docs/IMPLEMENTOR_ROUND.md):
a CTU-level source comment cited a specific named variable and line
number in an external library's HEVC decoder — clean-room provenance
for the surrounding code path could not be defended. Master history
was fully erased per the Hat-3 cold-enforcement procedure.

The rebuild is in progress against the published H.265 specification
(ITU-T Recommendation H.265 | ISO/IEC 23008-2). Round 2 adds the
VPS structural parse (§7.3.2.1) and the profile-tier-level walk
(§7.3.3) on top of round 1's Annex B / NAL-header foundation;
SPS/PPS semantic parse, slice decode, and CABAC remain unimplemented.

## Scope so far

* Annex B byte-stream splitting — 3-byte (`00 00 01`) and 4-byte
  (`00 00 00 01`) start codes; multiple NAL units per buffer.
* §7.3.1.2 NAL header parse — `forbidden_zero_bit`,
  `nal_unit_type`, `nuh_layer_id`, and `TemporalId` (derived from
  `nuh_temporal_id_plus1`).
* §7.4.1.1 emulation-prevention strip — every `0x00 0x00 0x03` in
  the on-wire payload is decoded back to `0x00 0x00`.
* MSB-first bit reader with `u(n)` and 0-th-order
  unsigned-Exp-Golomb `ue(v)` (§9.2) descriptors.
* §7.3.2.1 [`HevcVps`] — `vps_video_parameter_set_id`,
  base-layer / max-layers / sub-layers / temporal-nesting flags,
  `vps_reserved_0xffff_16bits` validation, the §7.3.3
  profile-tier-level walk (general profile + level + per-sub-layer
  present-flag gates + `sub_layer_level_idc`), and the per-sub-layer
  DPB / reorder / latency `ue(v)` triple loop with
  ordering-info-present-flag propagation.

Top-level entry points: [`NalIter`], [`collect_nal_units`],
[`NalHeader::parse`], [`strip_emulation_prevention`],
[`BitReader`], [`HevcVps::parse`], [`ProfileTierLevel::parse`].

## Not yet implemented

* SPS/PPS semantic parse (§7.3.2.2, §7.3.2.3)
* VPS tail: `vps_max_layer_id`, layer-set inclusion matrix,
  `vps_timing_info_present_flag`, HRD parameters,
  `vps_extension_data_flag`.
* Slice header parse (§7.3.6) and slice data (§7.3.8)
* CABAC entropy coding (§9.3) — blocked on the docs `cu_qp_delta`
  + `last_sig_coeff` multi-QG / multi-CTU 4:2:2 trace gap.
* Intra / inter prediction, transform, in-loop filters (deblock /
  SAO), DPB management.
* Encoder.

The runtime registration hook (`register`) is still a no-op.

## License

MIT — see [LICENSE](./LICENSE).
