//! Video Parameter Set (VPS) parser per ITU-T Rec. H.265 §7.3.2.1.
//!
//! Round-2 scope: parse the *structural* prefix of a VPS RBSP up to
//! and including the per-sub-layer ordering-info loop. The trailing
//! `vps_max_layer_id` / `vps_num_layer_sets_minus1` / HRD / extension
//! tail is parsed only to the extent needed to find the start of the
//! ordering info — the layer-set inclusion matrix, VUI timing block,
//! HRD parameters, and `vps_extension_data_flag` tail are **not** yet
//! materialised on [`HevcVps`].
//!
//! The profile-tier-level subroutine of §7.3.3 is also parsed
//! structurally: the bit positions are walked but only the leading
//! `general_profile_space` / `general_tier_flag` / `general_profile_idc`
//! / `general_level_idc` fields and the per-sub-layer
//! `sub_layer_profile_present_flag` / `sub_layer_level_present_flag`
//! gates are materialised. The remaining (mostly-reserved-zero or
//! constraint-flag) fields are skipped, but the bit-walk advances the
//! reader correctly so subsequent VPS fields land on the right bit
//! boundary.
//!
//! ## Layout summary
//!
//! ```text
//! vps_video_parameter_set_id            u(4)
//! vps_base_layer_internal_flag          u(1)
//! vps_base_layer_available_flag         u(1)
//! vps_max_layers_minus1                 u(6)
//! vps_max_sub_layers_minus1             u(3)
//! vps_temporal_id_nesting_flag          u(1)
//! vps_reserved_0xffff_16bits            u(16)   /* must be 0xFFFF */
//! profile_tier_level( 1, vps_max_sub_layers_minus1 )
//! vps_sub_layer_ordering_info_present_flag  u(1)
//! for( i = (...) ; i <= vps_max_sub_layers_minus1; i++ ) {
//!   vps_max_dec_pic_buffering_minus1[i] ue(v)
//!   vps_max_num_reorder_pics[i]         ue(v)
//!   vps_max_latency_increase_plus1[i]   ue(v)
//! }
//! ...                                          /* tail not parsed yet */
//! ```

use crate::bitreader::{BitReader, BitReaderError};

/// Maximum number of sub-layers an HEVC stream may declare.
/// `vps_max_sub_layers_minus1` is u(3), so the count is bounded at 7
/// in the bitstream; this constant is reused by [`HevcVps`] as the
/// fixed-size capacity for per-sub-layer arrays.
pub const HEVC_MAX_SUB_LAYERS: usize = 7;

/// Errors that can arise while parsing a VPS RBSP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VpsError {
    /// The RBSP ran out of bits before the VPS was fully parsed.
    Truncated,
    /// `vps_reserved_0xffff_16bits` was not `0xFFFF`. §7.4.3.1
    /// mandates the literal value.
    ReservedFieldMismatch {
        /// The (incorrect) value that was actually read.
        got: u16,
    },
    /// An Exp-Golomb code's `codeNum` exceeded what the corresponding
    /// syntax element can legally hold.
    ValueOutOfRange {
        /// Name of the offending syntax element.
        field: &'static str,
        /// The (illegal) value.
        got: u32,
    },
    /// An unexpected bitstream-level error surfaced from the reader.
    Bitstream(BitReaderError),
}

impl core::fmt::Display for VpsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => f.write_str("VPS RBSP truncated"),
            Self::ReservedFieldMismatch { got } => write!(
                f,
                "vps_reserved_0xffff_16bits was 0x{got:04X}, expected 0xFFFF"
            ),
            Self::ValueOutOfRange { field, got } => {
                write!(f, "syntax element {field} out of range: {got}")
            }
            Self::Bitstream(e) => write!(f, "bitstream error during VPS parse: {e}"),
        }
    }
}

impl std::error::Error for VpsError {}

impl From<BitReaderError> for VpsError {
    fn from(e: BitReaderError) -> Self {
        match e {
            BitReaderError::EndOfBuffer => Self::Truncated,
            other => Self::Bitstream(other),
        }
    }
}

/// Parsed profile-tier-level structure (§7.3.3).
///
/// Only the leading "general" fields and the per-sub-layer
/// present-flag gates are materialised at round-2 scope. The
/// constraint flags / reserved-zero blocks are walked over to keep
/// bit alignment but their values are intentionally discarded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileTierLevel {
    /// `general_profile_space` (`u(2)`).
    pub general_profile_space: u8,
    /// `general_tier_flag` (`u(1)`).
    pub general_tier_flag: bool,
    /// `general_profile_idc` (`u(5)`). Profile mnemonics are listed in
    /// Annex A; for the Main / Main 10 / Main Still / Main 4:2:2 family
    /// values 1..=4 are most common.
    pub general_profile_idc: u8,
    /// `general_level_idc` (`u(8)`). Per A.4 the on-wire value is
    /// `30 × level_number`, e.g. 30 == level 1.0, 90 == level 3.0,
    /// 120 == level 4.0.
    pub general_level_idc: u8,
    /// For each present sub-layer, whether its profile entry was
    /// signalled (`sub_layer_profile_present_flag[i]`).
    pub sub_layer_profile_present: [bool; HEVC_MAX_SUB_LAYERS],
    /// For each present sub-layer, whether its level_idc was signalled
    /// (`sub_layer_level_present_flag[i]`).
    pub sub_layer_level_present: [bool; HEVC_MAX_SUB_LAYERS],
    /// Per-sub-layer `sub_layer_level_idc[i]`; only valid for indices
    /// `i` where `sub_layer_level_present[i]` is true.
    pub sub_layer_level_idc: [u8; HEVC_MAX_SUB_LAYERS],
}

impl ProfileTierLevel {
    /// Parse a `profile_tier_level(profilePresentFlag, maxNumSubLayersMinus1)`
    /// invocation per §7.3.3. `profile_present_flag` is supplied by
    /// the calling context — for the VPS / SPS path it is always 1.
    pub fn parse(
        br: &mut BitReader<'_>,
        profile_present_flag: bool,
        max_num_sub_layers_minus1: u8,
    ) -> Result<Self, VpsError> {
        let mut ptl = Self {
            general_profile_space: 0,
            general_tier_flag: false,
            general_profile_idc: 0,
            general_level_idc: 0,
            sub_layer_profile_present: [false; HEVC_MAX_SUB_LAYERS],
            sub_layer_level_present: [false; HEVC_MAX_SUB_LAYERS],
            sub_layer_level_idc: [0; HEVC_MAX_SUB_LAYERS],
        };

        if profile_present_flag {
            ptl.general_profile_space = br.u(2)? as u8;
            ptl.general_tier_flag = br.u1()? != 0;
            ptl.general_profile_idc = br.u(5)? as u8;
            // 32 compatibility flags — skipped wholesale; the calling
            // application can re-parse them from the bit position if
            // needed later.
            br.skip(32)?;
            // progressive / interlaced / non_packed / frame_only
            br.skip(4)?;
            // The conditional block beneath these flags always consumes
            // exactly 43 bits regardless of profile_idc (per the
            // `/* not affected by this condition */` comment in §7.3.3
            // — the chroma-constraint, range-extension, and reserved
            // alternatives all sum to 43 bits).
            br.skip(43)?;
            // general_inbld_flag OR general_reserved_zero_bit — always 1 bit.
            br.skip(1)?;
        }
        // general_level_idc is always present (no `profilePresentFlag`
        // guard in the §7.3.3 syntax).
        ptl.general_level_idc = br.u(8)? as u8;

        // Per-sub-layer present-flag gates: 2 bits per sublayer up to
        // (but excluding) maxNumSubLayersMinus1.
        let max = max_num_sub_layers_minus1 as usize;
        for i in 0..max {
            let prof = br.u1()? != 0;
            let lvl = br.u1()? != 0;
            ptl.sub_layer_profile_present[i] = prof;
            ptl.sub_layer_level_present[i] = lvl;
        }

        // §7.3.3: if maxNumSubLayersMinus1 > 0, then for i in
        // max..8: reserved_zero_2bits — exactly 2 bits each — to keep
        // the per-sub-layer body byte-aligned regardless of how many
        // sublayers were actually signalled.
        if max_num_sub_layers_minus1 > 0 {
            for _ in max..8 {
                br.skip(2)?;
            }
        }

        // Per-sub-layer profile/level body for each i in 0..max.
        for i in 0..max {
            if ptl.sub_layer_profile_present[i] {
                // 2 + 1 + 5 + 32 + 4 + 43 + 1 = 88 bits, identical
                // layout to the general profile block above.
                br.skip(88)?;
            }
            if ptl.sub_layer_level_present[i] {
                ptl.sub_layer_level_idc[i] = br.u(8)? as u8;
            }
        }

        Ok(ptl)
    }
}

/// One per-sub-layer ordering-info triple from §7.3.2.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SubLayerOrderingInfo {
    /// `vps_max_dec_pic_buffering_minus1[i]` (`ue(v)`).
    /// Implies a DPB size of `value + 1`.
    pub max_dec_pic_buffering_minus1: u32,
    /// `vps_max_num_reorder_pics[i]` (`ue(v)`).
    pub max_num_reorder_pics: u32,
    /// `vps_max_latency_increase_plus1[i]` (`ue(v)`).
    /// 0 disables the constraint.
    pub max_latency_increase_plus1: u32,
}

/// Parsed Video Parameter Set per §7.3.2.1 (structural prefix only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HevcVps {
    /// `vps_video_parameter_set_id` (4 bits, range 0..=15).
    pub vps_id: u8,
    /// `vps_base_layer_internal_flag`.
    pub base_layer_internal_flag: bool,
    /// `vps_base_layer_available_flag`.
    pub base_layer_available_flag: bool,
    /// `vps_max_layers_minus1` (u(6)). The maximum number of layers is
    /// `value + 1`; for single-layer (non-SHVC) streams this is 0.
    pub max_layers_minus1: u8,
    /// `vps_max_sub_layers_minus1` (u(3), range 0..=6). The number of
    /// temporal sub-layers is `value + 1`.
    pub max_sub_layers_minus1: u8,
    /// `vps_temporal_id_nesting_flag`. When the stream has only one
    /// sub-layer the spec requires this to be 1.
    pub temporal_id_nesting_flag: bool,
    /// Parsed `profile_tier_level()` subroutine.
    pub ptl: ProfileTierLevel,
    /// `vps_sub_layer_ordering_info_present_flag`. When 0, only entry
    /// `[max_sub_layers_minus1]` is signalled and the others inherit
    /// its value.
    pub sub_layer_ordering_info_present_flag: bool,
    /// Per-sub-layer DPB / reorder / latency triples. Indices outside
    /// `0..=max_sub_layers_minus1` are zero-initialised.
    pub sub_layer_ordering_info: [SubLayerOrderingInfo; HEVC_MAX_SUB_LAYERS],
}

impl HevcVps {
    /// Parse `video_parameter_set_rbsp()` starting from the first bit
    /// of the (already-unescaped) RBSP body — that is, *after* the
    /// two-byte NAL header has been removed (see
    /// [`crate::nal::NalUnit`]).
    pub fn parse(rbsp: &[u8]) -> Result<Self, VpsError> {
        let mut br = BitReader::new(rbsp);
        Self::parse_inner(&mut br)
    }

    fn parse_inner(br: &mut BitReader<'_>) -> Result<Self, VpsError> {
        let vps_id = br.u(4)? as u8;
        let base_layer_internal_flag = br.u1()? != 0;
        let base_layer_available_flag = br.u1()? != 0;
        let max_layers_minus1 = br.u(6)? as u8;
        let max_sub_layers_minus1_raw = br.u(3)? as u8;
        if max_sub_layers_minus1_raw > 6 {
            // §7.4.3.1: range is 0..=6 (max 7 sub-layers); a 7 here is
            // illegal even though u(3) can encode it.
            return Err(VpsError::ValueOutOfRange {
                field: "vps_max_sub_layers_minus1",
                got: max_sub_layers_minus1_raw as u32,
            });
        }
        let temporal_id_nesting_flag = br.u1()? != 0;
        let reserved = br.u(16)? as u16;
        if reserved != 0xFFFF {
            return Err(VpsError::ReservedFieldMismatch { got: reserved });
        }

        let ptl = ProfileTierLevel::parse(br, true, max_sub_layers_minus1_raw)?;

        let sub_layer_ordering_info_present_flag = br.u1()? != 0;
        let start = if sub_layer_ordering_info_present_flag {
            0usize
        } else {
            max_sub_layers_minus1_raw as usize
        };
        let mut sub_layer_ordering_info = [SubLayerOrderingInfo::default(); HEVC_MAX_SUB_LAYERS];
        let last = max_sub_layers_minus1_raw as usize;
        for entry in sub_layer_ordering_info
            .iter_mut()
            .take(last + 1)
            .skip(start)
        {
            let max_dpb = br.ue()?;
            let max_reorder = br.ue()?;
            let max_lat = br.ue()?;
            *entry = SubLayerOrderingInfo {
                max_dec_pic_buffering_minus1: max_dpb,
                max_num_reorder_pics: max_reorder,
                max_latency_increase_plus1: max_lat,
            };
        }
        // When the present flag was 0, propagate the
        // [max_sub_layers_minus1] entry across the lower-indexed
        // sub-layers — §7.4.3.1 dictates that they take the same value.
        if !sub_layer_ordering_info_present_flag {
            let copy = sub_layer_ordering_info[last];
            for entry in sub_layer_ordering_info.iter_mut().take(last) {
                *entry = copy;
            }
        }

        Ok(Self {
            vps_id,
            base_layer_internal_flag,
            base_layer_available_flag,
            max_layers_minus1,
            max_sub_layers_minus1: max_sub_layers_minus1_raw,
            temporal_id_nesting_flag,
            ptl,
            sub_layer_ordering_info_present_flag,
            sub_layer_ordering_info,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nal::{collect_nal_units, strip_emulation_prevention};

    /// VPS RBSP body extracted from the workspace fixture
    /// `docs/video/h265/fixtures/tiny-i-only-16x16-main/input.hevc`,
    /// after the Annex B start code and the two-byte NAL header have
    /// been removed and emulation-prevention bytes stripped. Captured
    /// inline so the test runs without docs/ on the include path.
    ///
    /// Source byte sequence (Annex B): `00 00 00 01 40 01 0C 01 FF FF
    /// 04 08 00 00 03 00 9F A8 00 00 03 00 00 1E BA 02 40` — after
    /// stripping the start code and the `40 01` NAL header, and after
    /// the §7.4.1.1 strip dropping both `03` emulation bytes:
    /// `0C 01 FF FF 04 08 00 00 00 9F A8 00 00 00 00 1E BA 02 40`.
    const TINY_VPS_RBSP: &[u8] = &[
        0x0C, 0x01, 0xFF, 0xFF, 0x04, 0x08, 0x00, 0x00, 0x00, 0x9F, 0xA8, 0x00, 0x00, 0x00, 0x00,
        0x1E, 0xBA, 0x02, 0x40,
    ];

    #[test]
    fn parses_tiny_fixture_vps() {
        let vps = HevcVps::parse(TINY_VPS_RBSP).expect("VPS parse");
        assert_eq!(vps.vps_id, 0);
        assert!(vps.base_layer_internal_flag);
        assert!(vps.base_layer_available_flag);
        assert_eq!(vps.max_layers_minus1, 0); // 1 layer
        assert_eq!(vps.max_sub_layers_minus1, 0); // 1 sub-layer
        assert!(vps.temporal_id_nesting_flag);
        assert_eq!(vps.ptl.general_profile_space, 0);
        assert!(!vps.ptl.general_tier_flag);
        assert_eq!(vps.ptl.general_profile_idc, 4);
        // §A.4 maps `30 == level 1.0 × 30` ⇒ level 1.0. (Not "level
        // 3.0" — that would be value 90.) The fixture's `notes.md`
        // groups it as "level 3.0" but the on-wire `level_idc` is the
        // raw encoded value, which is 30 ⇒ level 1.0.
        assert_eq!(vps.ptl.general_level_idc, 30);
        // With max_sub_layers_minus1 == 0 the per-sub-layer ordering
        // loop still runs once for i == 0. The fixture encoder
        // populates DPB size 3 / no reorder / latency 1 for a
        // single-frame stream, so the on-wire `ue(v)` triple decodes
        // to (2, 0, 1).
        assert!(vps.sub_layer_ordering_info_present_flag);
        assert_eq!(
            vps.sub_layer_ordering_info[0].max_dec_pic_buffering_minus1,
            2
        );
        assert_eq!(vps.sub_layer_ordering_info[0].max_num_reorder_pics, 0);
        assert_eq!(vps.sub_layer_ordering_info[0].max_latency_increase_plus1, 1);
    }

    #[test]
    fn parses_tiny_fixture_vps_via_nal_walker() {
        // End-to-end check: feed the raw Annex B stream through the
        // NAL walker, then parse the first NAL's RBSP as a VPS.
        let raw = &[
            0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0C, 0x01, 0xFF, 0xFF, 0x04, 0x08, 0x00, 0x00,
            0x03, 0x00, 0x9F, 0xA8, 0x00, 0x00, 0x03, 0x00, 0x00, 0x1E, 0xBA, 0x02, 0x40,
        ];
        let units = collect_nal_units(raw).expect("walker");
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].header.nal_unit_type, 32); // VPS_NUT
        let vps = HevcVps::parse(&units[0].rbsp).expect("VPS parse");
        assert_eq!(vps.vps_id, 0);
        assert_eq!(vps.ptl.general_profile_idc, 4);
        assert_eq!(vps.ptl.general_level_idc, 30);
    }

    #[test]
    fn rejects_wrong_reserved_field() {
        // Same prefix as the fixture but flip the last bit of the
        // 16-bit reserved field so it reads 0xFFFE instead of 0xFFFF.
        let mut bad = TINY_VPS_RBSP.to_vec();
        bad[3] = 0xFE; // was 0xFF
        let err = HevcVps::parse(&bad).unwrap_err();
        assert_eq!(err, VpsError::ReservedFieldMismatch { got: 0xFFFE });
    }

    #[test]
    fn rejects_truncated_rbsp() {
        // Truncate to before the reserved field finishes.
        let err = HevcVps::parse(&TINY_VPS_RBSP[..3]).unwrap_err();
        assert_eq!(err, VpsError::Truncated);
    }

    #[test]
    fn strip_emulation_prevention_then_parse_matches_inline_decode() {
        // Build the wire-format RBSP (with 03 bytes), strip them, and
        // confirm the decoded VPS matches the inline-stripped fixture.
        let wire = &[
            0x0C, 0x01, 0xFF, 0xFF, 0x04, 0x08, 0x00, 0x00, 0x03, 0x00, 0x9F, 0xA8, 0x00, 0x00,
            0x03, 0x00, 0x00, 0x1E, 0xBA, 0x02, 0x40,
        ];
        let unescaped = strip_emulation_prevention(wire);
        assert_eq!(unescaped, TINY_VPS_RBSP);
        let vps = HevcVps::parse(&unescaped).expect("VPS parse");
        let direct = HevcVps::parse(TINY_VPS_RBSP).expect("VPS parse");
        assert_eq!(vps, direct);
    }

    /// Hand-assembled minimal VPS to exercise the loop expansion path
    /// when `sub_layer_ordering_info_present_flag == 1`. This test
    /// stresses the multi-sub-layer code path without any fixture.
    #[test]
    fn parses_two_sub_layer_ordering_info_present() {
        // Build a VPS with max_sub_layers_minus1 = 1 (two sublayers)
        // and sub_layer_ordering_info_present_flag = 1, so two
        // ordering-info triples are read.
        //
        // Field layout bit-by-bit:
        //   vps_id u4                          : 0000
        //   base_layer_internal_flag u1        : 1
        //   base_layer_available_flag u1       : 1
        //   max_layers_minus1 u6               : 000000
        //   max_sub_layers_minus1 u3           : 001
        //   temporal_id_nesting_flag u1        : 1
        //   reserved u16                       : 1111_1111_1111_1111
        //   profile_tier_level:
        //     profile_space u2                 : 00
        //     tier_flag u1                     : 0
        //     profile_idc u5                   : 00001  (Main)
        //     32 compat flags                  : all 0
        //     4 source flags (prog/i/np/fo)    : 1000
        //     43-bit block                     : all 0
        //     1-bit reserved                   : 0
        //     level_idc u8                     : 00011110 (= 30)
        //     2 bits per inner sublayer for i in 0..1:
        //       sub_layer_profile_present[0] u1: 0
        //       sub_layer_level_present[0] u1  : 1
        //     8-N inner reserved 2-bit pads (N=1 → 7 entries × 2 bits = 14 bits): all 0
        //     since sub_layer_profile_present[0] == 0 → no extra body
        //     sub_layer_level_present[0] == 1 → sub_layer_level_idc[0] u8
        //                                       : 00011110 (= 30)
        //   sub_layer_ordering_info_present_flag u1 : 1
        //   loop i = 0..1 (two iterations) each with:
        //     ue(v) = '1' (codeNum 0)
        //     ue(v) = '1' (codeNum 0)
        //     ue(v) = '1' (codeNum 0)
        //
        // We assemble the bit string and then chunk to bytes MSB-first.
        let mut bits = Vec::<u8>::new();
        let mut push = |s: &str| {
            for c in s.chars() {
                if c == '0' || c == '1' {
                    bits.push((c as u8) - b'0');
                }
            }
        };
        push("0000"); // vps_id
        push("1"); // base_layer_internal_flag
        push("1"); // base_layer_available_flag
        push("000000"); // max_layers_minus1
        push("001"); // max_sub_layers_minus1 = 1
        push("1"); // temporal_id_nesting_flag
        push("1111111111111111"); // reserved
        push("00"); // profile_space
        push("0"); // tier_flag
        push("00001"); // profile_idc = 1
        push(&"0".repeat(32)); // compat flags
        push("1000"); // prog/interlaced/non_packed/frame_only
        push(&"0".repeat(43)); // 43-bit block
        push("0"); // 1-bit reserved
        push("00011110"); // general_level_idc = 30
        push("01"); // sub_layer_profile_present[0]=0, sub_layer_level_present[0]=1
        push(&"0".repeat(2 * 7)); // 14-bit padding for i=1..8
        push("00011110"); // sub_layer_level_idc[0] = 30
        push("1"); // sub_layer_ordering_info_present_flag
        push("1"); // ue=0 (max_dec_pic_buffering_minus1[0])
        push("1"); // ue=0 (max_num_reorder_pics[0])
        push("1"); // ue=0 (max_latency_increase_plus1[0])
        push("1"); // ue=0 (max_dec_pic_buffering_minus1[1])
        push("1"); // ue=0 (max_num_reorder_pics[1])
        push("1"); // ue=0 (max_latency_increase_plus1[1])

        // Pad with zeros to next byte boundary and pack.
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut bytes = Vec::with_capacity(bits.len() / 8);
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for &bit in chunk {
                b = (b << 1) | bit;
            }
            bytes.push(b);
        }

        let vps = HevcVps::parse(&bytes).expect("VPS parse");
        assert_eq!(vps.vps_id, 0);
        assert_eq!(vps.max_sub_layers_minus1, 1);
        assert!(vps.temporal_id_nesting_flag);
        assert_eq!(vps.ptl.general_profile_idc, 1);
        assert_eq!(vps.ptl.general_level_idc, 30);
        assert!(!vps.ptl.sub_layer_profile_present[0]);
        assert!(vps.ptl.sub_layer_level_present[0]);
        assert_eq!(vps.ptl.sub_layer_level_idc[0], 30);
        assert!(vps.sub_layer_ordering_info_present_flag);
        for i in 0..=1 {
            assert_eq!(
                vps.sub_layer_ordering_info[i].max_dec_pic_buffering_minus1,
                0
            );
            assert_eq!(vps.sub_layer_ordering_info[i].max_num_reorder_pics, 0);
            assert_eq!(vps.sub_layer_ordering_info[i].max_latency_increase_plus1, 0);
        }
    }

    #[test]
    fn parses_ordering_info_present_flag_zero_propagates() {
        // max_sub_layers_minus1 = 1, but
        // sub_layer_ordering_info_present_flag = 0 → only the [1]
        // entry is signalled; the [0] entry inherits it per §7.4.3.1.
        let mut bits = Vec::<u8>::new();
        let mut push = |s: &str| {
            for c in s.chars() {
                if c == '0' || c == '1' {
                    bits.push((c as u8) - b'0');
                }
            }
        };
        push("0000"); // vps_id
        push("1");
        push("1");
        push("000000");
        push("001"); // max_sub_layers_minus1 = 1
        push("1");
        push("1111111111111111");
        push("00");
        push("0");
        push("00001"); // profile_idc=1
        push(&"0".repeat(32));
        push("1000");
        push(&"0".repeat(43));
        push("0");
        push("00011110"); // level_idc=30
        push("00"); // sub_layer present flags both 0
        push(&"0".repeat(14));
        push("0"); // sub_layer_ordering_info_present_flag = 0
                   // Only the i=1 triple is read; we want the [1] DPB = 2 (codeNum 2 = '011').
        push("011"); // max_dec_pic_buffering_minus1[1] = 2
        push("1"); // max_num_reorder_pics[1] = 0
        push("1"); // max_latency_increase_plus1[1] = 0

        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut bytes = Vec::with_capacity(bits.len() / 8);
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for &bit in chunk {
                b = (b << 1) | bit;
            }
            bytes.push(b);
        }
        let vps = HevcVps::parse(&bytes).expect("VPS parse");
        assert!(!vps.sub_layer_ordering_info_present_flag);
        // [1] was signalled; [0] inherits.
        assert_eq!(
            vps.sub_layer_ordering_info[1].max_dec_pic_buffering_minus1,
            2
        );
        assert_eq!(
            vps.sub_layer_ordering_info[0].max_dec_pic_buffering_minus1,
            2
        );
    }
}
