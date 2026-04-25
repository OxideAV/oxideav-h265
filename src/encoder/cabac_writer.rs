//! HEVC CABAC arithmetic encoder (§9.3.4.4).
//!
//! Mirror of [`crate::cabac::CabacEngine`] in the opposite direction. The
//! spec's encoder algorithm §9.3.4.4.x maintains:
//!
//! * `ivlLow` (10 bits) and `ivlCurrRange` (9 bits) — the current interval.
//! * `bitsOutstanding` — count of deferred bits pending for bit-propagation.
//! * `firstBitFlag` — set by the init, cleared after the first bit is
//!   emitted.
//!
//! This encoder writes bits directly into a supplied [`BitWriter`]. After
//! the last bin is emitted, the caller invokes [`CabacWriter::encode_flush`]
//! which applies §9.3.4.4.1 to spill `ivlLow` and the outstanding bits
//! into the bitstream, terminating the arithmetic run.

use crate::cabac::{CtxState, RANGE_TAB_LPS, TRANS_IDX_LPS, TRANS_IDX_MPS};
use crate::encoder::bit_writer::BitWriter;

/// Arithmetic encoder engine.
pub struct CabacWriter<'a> {
    out: &'a mut BitWriter,
    ivl_low: u32,
    ivl_curr_range: u32,
    bits_outstanding: u32,
    first_bit_flag: bool,
}

impl<'a> CabacWriter<'a> {
    /// Initialise the encoder (§9.3.4.4.1 `initialise_encoder`).
    pub fn new(out: &'a mut BitWriter) -> Self {
        Self {
            out,
            ivl_low: 0,
            ivl_curr_range: 510,
            bits_outstanding: 0,
            first_bit_flag: true,
        }
    }

    /// Encode one regular (context-modelled) bin (§9.3.4.4.2
    /// `encode_bin`).
    pub fn encode_bin(&mut self, ctx: &mut CtxState, bin_val: u32) {
        let rang_lps_idx = ((self.ivl_curr_range >> 6) & 3) as usize;
        let ivl_lps_range = RANGE_TAB_LPS[ctx.p_state_idx as usize][rang_lps_idx] as u32;
        self.ivl_curr_range -= ivl_lps_range;
        if bin_val != ctx.val_mps as u32 {
            // LPS path.
            self.ivl_low += self.ivl_curr_range;
            self.ivl_curr_range = ivl_lps_range;
            if ctx.p_state_idx == 0 {
                ctx.val_mps ^= 1;
            }
            ctx.p_state_idx = TRANS_IDX_LPS[ctx.p_state_idx as usize];
        } else {
            ctx.p_state_idx = TRANS_IDX_MPS[ctx.p_state_idx as usize];
        }
        self.renormalise();
    }

    /// Encode one bypass (equiprobable) bin (§9.3.4.4.4
    /// `encode_bin_EP`).
    pub fn encode_bypass(&mut self, bin_val: u32) {
        self.ivl_low <<= 1;
        if bin_val != 0 {
            self.ivl_low += self.ivl_curr_range;
        }
        // Renormalise with a single output step per §9.3.4.4.4.
        if self.ivl_low >= 1024 {
            self.put_bit(1);
            self.ivl_low -= 1024;
        } else if self.ivl_low < 512 {
            self.put_bit(0);
        } else {
            self.ivl_low -= 512;
            self.bits_outstanding += 1;
        }
    }

    /// Encode the terminating bin (§9.3.4.4.5 `encode_bin_TRM`). Pass
    /// `bin_val = 1` to emit the sticky end-of-slice (or PCM-eligible)
    /// terminator; `bin_val = 0` for a regular continue bin.
    pub fn encode_terminate(&mut self, bin_val: u32) {
        self.ivl_curr_range -= 2;
        if bin_val == 1 {
            self.ivl_low += self.ivl_curr_range;
            // Flush is done externally after a `1` terminator.
        } else {
            self.renormalise();
        }
    }

    /// §9.3.4.4.6 `encode_flush`: after a `1` terminator, spill the
    /// remaining state. Writes `stop_one_bit` and byte-aligns the
    /// arithmetic output so the CABAC run ends on a byte boundary.
    ///
    /// The spec defines this as:
    ///   ivlCurrRange = 2;
    ///   renormalise;  ← emits bits
    ///   PutBit((ivlLow >> 9) & 1);
    ///   WriteBits(((ivlLow >> 7) & 3) | 1, 2);  ← the "1" is the stop bit
    ///   byte-align if necessary (but §9.3.4.4.6 ends naturally aligned).
    pub fn encode_flush(&mut self) {
        self.ivl_curr_range = 2;
        self.renormalise();
        self.put_bit((self.ivl_low >> 9) & 1);
        // Write 2 bits: top bit of the remaining 8 is in bit 8, then bit 7;
        // spec combines as `((ivl_low >> 7) & 3) | 1`.
        let bits = ((self.ivl_low >> 7) & 3) | 1;
        self.out.write_bits(bits, 2);
        // After this 2-bit write the CABAC run is not yet byte-aligned in
        // the general case — the caller must either byte-align manually
        // or trust that the slice_data trailing bits path will do so.
    }

    /// §9.3.4.4.1.2 `renormalise_enc`.
    fn renormalise(&mut self) {
        while self.ivl_curr_range < 256 {
            if self.ivl_low < 256 {
                self.put_bit(0);
            } else if self.ivl_low >= 512 {
                self.ivl_low -= 512;
                self.put_bit(1);
            } else {
                self.ivl_low -= 256;
                self.bits_outstanding += 1;
            }
            self.ivl_curr_range <<= 1;
            self.ivl_low <<= 1;
        }
    }

    /// §9.3.4.4.1.3 `put_bit`. Handles the first-bit special case and
    /// emits any outstanding bits.
    fn put_bit(&mut self, b: u32) {
        if self.first_bit_flag {
            self.first_bit_flag = false;
        } else {
            self.out.write_u1(b);
        }
        while self.bits_outstanding > 0 {
            self.out.write_u1(b ^ 1);
            self.bits_outstanding -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::{init_context, CabacEngine};

    #[test]
    fn roundtrip_bypass_bits() {
        let bits = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0];
        let mut bw = BitWriter::new();
        let mut cw = CabacWriter::new(&mut bw);
        for &b in &bits {
            cw.encode_bypass(b);
        }
        cw.encode_terminate(1);
        cw.encode_flush();
        let bytes = bw.finish();

        let mut eng = CabacEngine::new(&bytes, 0);
        for &b in &bits {
            let d = eng.decode_bypass();
            assert_eq!(d, b, "bypass bit mismatch");
        }
        assert_eq!(eng.decode_terminate(), 1);
    }

    #[test]
    fn roundtrip_regular_bin() {
        // One context seeded to a neutral probability, encode then decode.
        let init_value = 154u8;
        let slice_qp_y = 26;
        let bits = [0u32, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1];
        let mut ctx_enc = init_context(init_value, slice_qp_y);
        let mut ctx_dec = init_context(init_value, slice_qp_y);

        let mut bw = BitWriter::new();
        let mut cw = CabacWriter::new(&mut bw);
        for &b in &bits {
            cw.encode_bin(&mut ctx_enc, b);
        }
        cw.encode_terminate(1);
        cw.encode_flush();
        let bytes = bw.finish();

        let mut eng = CabacEngine::new(&bytes, 0);
        for &b in &bits {
            let d = eng.decode_bin(&mut ctx_dec);
            assert_eq!(d, b, "regular bin mismatch");
        }
        assert_eq!(eng.decode_terminate(), 1);
    }

    #[test]
    fn roundtrip_mixed() {
        // Mix regular, bypass, and terminator-0 bins before the final
        // terminator-1.
        let mut ctx_e = init_context(139, 26);
        let mut ctx_d = init_context(139, 26);
        let seq: [(u8, u32); 12] = [
            (0, 1),
            (1, 0),
            (0, 1),
            (0, 0),
            (1, 1),
            (1, 1),
            (0, 0),
            (2, 0), // terminator continue
            (1, 1),
            (0, 1),
            (1, 0),
            (2, 0),
        ];
        let mut bw = BitWriter::new();
        let mut cw = CabacWriter::new(&mut bw);
        for &(kind, val) in &seq {
            match kind {
                0 => cw.encode_bin(&mut ctx_e, val),
                1 => cw.encode_bypass(val),
                2 => cw.encode_terminate(val),
                _ => unreachable!(),
            }
        }
        cw.encode_terminate(1);
        cw.encode_flush();
        let bytes = bw.finish();

        let mut eng = CabacEngine::new(&bytes, 0);
        for &(kind, val) in &seq {
            let d = match kind {
                0 => eng.decode_bin(&mut ctx_d),
                1 => eng.decode_bypass(),
                2 => eng.decode_terminate(),
                _ => unreachable!(),
            };
            assert_eq!(d, val, "mixed bin mismatch");
        }
        assert_eq!(eng.decode_terminate(), 1);
    }
}
