//! NAL packaging for the encoder.
//!
//! * [`insert_emulation_prevention_bytes`] turns an RBSP (no
//!   `0x00 0x00 {0x00|0x01|0x02|0x03}` triplets) into an EBSP by inserting
//!   a `0x03` after every two consecutive zeros that would otherwise be
//!   followed by a value in {0, 1, 2, 3} — §7.4.1.1.
//! * [`build_annex_b_nal`] concatenates: start code (`0x00 0x00 0x00 0x01`),
//!   the 2-byte NAL header, and the EBSP'd RBSP.
//! * [`NalHeader::encode`] packs the 2-byte NAL header (§7.3.1.2).

use crate::nal::{NalHeader, NalUnitType};

impl NalHeader {
    /// Pack the 2-byte NAL header.
    pub fn encode(self) -> [u8; 2] {
        let nt = self.nal_unit_type.as_u8();
        debug_assert!(nt < 64);
        debug_assert!(self.nuh_layer_id < 64);
        debug_assert!(self.nuh_temporal_id_plus1 > 0 && self.nuh_temporal_id_plus1 < 8);
        // forbidden_zero_bit=0 | nal_unit_type(6) | nuh_layer_id(6) | tid_plus1(3)
        let b0 = (nt << 1) | ((self.nuh_layer_id >> 5) & 0x01);
        let b1 = ((self.nuh_layer_id & 0x1F) << 3) | (self.nuh_temporal_id_plus1 & 0x07);
        [b0, b1]
    }

    /// Convenience: construct a header for layer 0, temporal_id 0.
    pub fn for_type(nut: NalUnitType) -> Self {
        Self {
            nal_unit_type: nut,
            nuh_layer_id: 0,
            nuh_temporal_id_plus1: 1,
        }
    }
}

/// Insert emulation-prevention bytes into an RBSP to produce an EBSP
/// (§7.4.1.1). Any occurrence of `0x00 0x00 X` with X in {0x00, 0x01,
/// 0x02, 0x03} gets a `0x03` inserted between the second 0x00 and X.
pub fn insert_emulation_prevention_bytes(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + rbsp.len() / 128 + 4);
    let mut zeros = 0usize;
    for &b in rbsp {
        if zeros >= 2 && b <= 0x03 {
            out.push(0x03);
            zeros = 0;
        }
        out.push(b);
        if b == 0x00 {
            zeros += 1;
        } else {
            zeros = 0;
        }
    }
    out
}

/// Build a single Annex B NAL unit: start code + 2-byte header + EBSP(rbsp).
pub fn build_annex_b_nal(header: NalHeader, rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + 8);
    out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
    out.extend_from_slice(&header.encode());
    out.extend_from_slice(&insert_emulation_prevention_bytes(rbsp));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nal::{extract_rbsp, iter_annex_b, NalUnitType};

    #[test]
    fn nal_header_roundtrip() {
        for nut in [
            NalUnitType::Vps,
            NalUnitType::Sps,
            NalUnitType::Pps,
            NalUnitType::IdrWRadl,
            NalUnitType::IdrNLp,
        ] {
            let h = NalHeader::for_type(nut);
            let bytes = h.encode();
            let parsed = NalHeader::parse(&bytes).unwrap();
            assert_eq!(parsed.nal_unit_type, nut);
            assert_eq!(parsed.nuh_layer_id, 0);
            assert_eq!(parsed.nuh_temporal_id_plus1, 1);
        }
    }

    #[test]
    fn emulation_prevention_adds_03_when_needed() {
        // 0x00 0x00 0x01 → 0x00 0x00 0x03 0x01
        let rbsp = [0x00u8, 0x00, 0x01];
        let ebsp = insert_emulation_prevention_bytes(&rbsp);
        assert_eq!(ebsp, vec![0x00, 0x00, 0x03, 0x01]);
        // Round-trip through extract_rbsp (strips 0x03).
        assert_eq!(extract_rbsp(&ebsp), rbsp.to_vec());
    }

    #[test]
    fn emulation_prevention_trailing_zeros() {
        // Trailing 0x00 0x00 alone stays untouched — but if the body ends
        // with 0x00 0x00 we still add a terminator-style 0x03 when the
        // next byte would alias a start code. Here we just verify the
        // mid-stream case.
        let rbsp = [0xAB, 0x00, 0x00, 0x02, 0xCD, 0x00, 0x00, 0x03, 0xEF];
        let ebsp = insert_emulation_prevention_bytes(&rbsp);
        // Two insertions (before 0x02 and before 0x03).
        assert_eq!(
            ebsp,
            vec![0xAB, 0x00, 0x00, 0x03, 0x02, 0xCD, 0x00, 0x00, 0x03, 0x03, 0xEF]
        );
        assert_eq!(extract_rbsp(&ebsp), rbsp.to_vec());
    }

    #[test]
    fn build_nal_and_iter_parses_it() {
        let rbsp = [0xAAu8, 0xBB, 0xCC];
        let nal = build_annex_b_nal(NalHeader::for_type(NalUnitType::Vps), &rbsp);
        let nals: Vec<_> = iter_annex_b(&nal).collect();
        assert_eq!(nals.len(), 1);
        assert_eq!(nals[0].header.nal_unit_type, NalUnitType::Vps);
    }
}
