# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — clean-room rebuild round 2 (2026-05-22)

- MSB-first `BitReader` with `u(n)` and 0-th-order
  unsigned-Exp-Golomb `ue(v)` (§9.2) descriptors; `skip(n)` to
  bit-walk over not-yet-materialised fields without parsing them.
- §7.3.2.1 `HevcVps` structural parse — `vps_video_parameter_set_id`
  (u4), `vps_base_layer_internal_flag` / `vps_base_layer_available_flag`,
  `vps_max_layers_minus1` (u6), `vps_max_sub_layers_minus1` (u3 with
  the §7.4.3.1 0..=6 range check), `vps_temporal_id_nesting_flag`,
  `vps_reserved_0xffff_16bits` validation (rejects any value other
  than `0xFFFF`), the §7.3.3 `profile_tier_level()` walk, and the
  per-sub-layer DPB / reorder / latency `ue(v)` triple loop with
  ordering-info-present-flag propagation.
- §7.3.3 `ProfileTierLevel` — materialises
  `general_profile_space` / `_tier_flag` / `_profile_idc` /
  `_level_idc` and the per-sub-layer
  `sub_layer_profile_present_flag` / `_level_present_flag` gates
  plus `sub_layer_level_idc[i]`. The constraint-flag / reserved-zero
  blocks are walked structurally (their bit width is fixed at 43
  bits regardless of the inner conditional branch per the
  `/* not affected by this condition */` note in the syntax table)
  so subsequent VPS fields land on the right bit boundary.
- 17 new unit tests: bit-reader `u(n)` MSB-first packing /
  cross-byte read / `u(0)` zero-consume / `u(32)` full word /
  end-of-buffer / too-many-bits / skip / `ue(v)` codewords 0..=6
  per Table 9-2 / single-bit zero / leading-zero-overrun;
  VPS fixture parse (against
  `docs/video/h265/fixtures/tiny-i-only-16x16-main/input.hevc` — the
  on-wire bytes are inlined into the test, no I/O); end-to-end VPS
  parse via the Annex B walker; reserved-field mismatch rejection;
  truncated-RBSP rejection; emulation-prevention round-trip equality;
  hand-assembled two-sub-layer ordering-info-present-flag=1 parse;
  hand-assembled ordering-info-present-flag=0 propagation parse.

### Added — clean-room rebuild round 1 (2026-05-20)

- Annex B byte-stream walker: `NalIter`, `collect_nal_units`,
  supporting both 3-byte (`00 00 01`) and 4-byte (`00 00 00 01`)
  start codes.
- §7.3.1.2 NAL header parse: `NalHeader` exposes
  `nal_unit_type`, `nuh_layer_id`, and `TemporalId` (derived from
  `nuh_temporal_id_plus1`); `forbidden_zero_bit` set and
  zero-`nuh_temporal_id_plus1` are surfaced as `NalError`.
- §7.4.1.1 emulation-prevention strip
  (`strip_emulation_prevention`) — `0x00 0x00 0x03` decodes to
  `0x00 0x00`.
- 7 unit tests covering: 3-byte start code single NAL, 4-byte
  start code with two NAL units, emulation-prevention round-trip,
  forbidden-bit rejection, zero-temporal-id rejection, no
  start-code-at-all rejection, and header field-packing round
  trip (incl. non-zero `nuh_layer_id`).

### Erased

- Prior master history was force-erased on **2026-05-18** under
  Hat-3 cold enforcement of the workspace clean-room policy
  (`docs/IMPLEMENTOR_ROUND.md`).

### Reset

- Crate reduced to a minimal `oxideav_core::register!` stub. Every
  public API returns `Error::NotImplemented`. The crates.io version
  (`0.0.8`) is preserved on the new master to avoid breaking
  downstream version pins; the published versions on crates.io will
  be yanked by the maintainer.
- HEIF/HEIC `heif` cargo feature is dropped from the scaffold
  (re-introduced in a future rebuild round once the decoder core is
  back).

### Next

- SPS/PPS semantic parse (§7.3.2.2, §7.3.2.3). RBSP-stop-bit
  handling.
- VPS tail: `vps_max_layer_id`, `vps_num_layer_sets_minus1`,
  `layer_id_included_flag` matrix, `vps_timing_info_present_flag`,
  HRD parameters, `vps_extension_data_flag`.
- Slice header parse.
- CABAC remains blocked on docs #444 (`cu_qp_delta` +
  `last_sig_coeff` multi-QG / multi-CTU 4:2:2 trace gap).
