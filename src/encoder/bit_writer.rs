//! MSB-first bit writer + Exp-Golomb emitter for HEVC RBSPs.
//!
//! Mirror of [`crate::bitreader::BitReader`] in the opposite direction.
//! The output is a byte buffer with bits packed MSB-first; partial-byte
//! writes are held in an accumulator and flushed when full. Callers
//! finish with [`BitWriter::finish`] which pads any trailing partial
//! byte with zero bits and returns the byte buffer.
//!
//! Exp-Golomb encoding follows ITU-T H.265 §9.2:
//! * `ue(v)`: for codeNum k, write (leading_zeros+1)-bit group = `1` prefix
//!   followed by a `ceil(log2(k+2))-1`-bit suffix, where leading_zeros =
//!   `floor(log2(k+1))`.
//! * `se(v)`: map v → k with v=0→0, v=1→1, v=-1→2, v=2→3, v=-2→4, …
//!   (k = v>0 ? 2v-1 : -2v).

/// MSB-first bit writer.
#[derive(Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    acc: u64,
    bits_in_acc: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(cap),
            acc: 0,
            bits_in_acc: 0,
        }
    }

    pub fn bit_position(&self) -> u64 {
        self.bytes.len() as u64 * 8 + self.bits_in_acc as u64
    }

    pub fn is_byte_aligned(&self) -> bool {
        self.bits_in_acc % 8 == 0
    }

    /// Write `n` bits (0..=32) of `value`, MSB-first.
    pub fn write_bits(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        if n == 0 {
            return;
        }
        // Mask off high garbage.
        let v = if n == 32 { value } else { value & ((1u32 << n) - 1) };
        self.acc = (self.acc << n) | v as u64;
        self.bits_in_acc += n;
        while self.bits_in_acc >= 8 {
            let shift = self.bits_in_acc - 8;
            self.bytes.push(((self.acc >> shift) & 0xFF) as u8);
            // Drop the emitted bits.
            self.bits_in_acc -= 8;
            self.acc &= (1u64 << self.bits_in_acc).wrapping_sub(1);
        }
    }

    /// Write a single bit (flag).
    pub fn write_u1(&mut self, bit: u32) {
        self.write_bits(bit & 1, 1);
    }

    /// Write `n` bits (0..=64).
    pub fn write_bits_long(&mut self, value: u64, n: u32) {
        debug_assert!(n <= 64);
        if n <= 32 {
            self.write_bits(value as u32, n);
            return;
        }
        self.write_bits((value >> 32) as u32, n - 32);
        self.write_bits(value as u32, 32);
    }

    /// Write `n` zero bits.
    pub fn write_zero_bits(&mut self, n: u32) {
        let mut rem = n;
        while rem > 32 {
            self.write_bits(0, 32);
            rem -= 32;
        }
        self.write_bits(0, rem);
    }

    /// Write an unsigned Exp-Golomb code (§9.2). `value` is the code number.
    pub fn write_ue(&mut self, value: u32) {
        // codeNum → (k = floor(log2(value+1)); prefix = k zeros + 1;
        //           suffix = (value+1) - 2^k in k bits).
        let v = value as u64 + 1;
        let k = (63u32 - v.leading_zeros()) as u32; // floor(log2(v))
        // Write k zero bits, then the k+1 low bits of v (which starts with a 1).
        self.write_zero_bits(k);
        // v has exactly k+1 significant bits.
        self.write_bits_long(v, k + 1);
    }

    /// Write a signed Exp-Golomb code (§9.2.2).
    pub fn write_se(&mut self, value: i32) {
        let k: u32 = if value <= 0 {
            (-(value as i64) as u64 * 2) as u32
        } else {
            (value as u32) * 2 - 1
        };
        self.write_ue(k);
    }

    /// Pad with zero bits so the next write starts on a byte boundary.
    pub fn align_to_byte_zero(&mut self) {
        let pad = (8 - self.bits_in_acc % 8) % 8;
        if pad != 0 {
            self.write_bits(0, pad);
        }
    }

    /// Write raw bytes (requires byte alignment).
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        assert!(self.is_byte_aligned(), "write_bytes requires byte alignment");
        self.bytes.extend_from_slice(bytes);
    }

    /// Finish the stream, padding the last partial byte with zeros, and
    /// return the raw byte buffer.
    pub fn finish(mut self) -> Vec<u8> {
        self.align_to_byte_zero();
        self.bytes
    }

    /// Return the bytes written so far (flushing any complete bytes held
    /// in the accumulator is automatic — partial bits are not returned).
    pub fn bytes_so_far(&self) -> &[u8] {
        &self.bytes
    }
}

/// Emit the HEVC `rbsp_trailing_bits()` (§7.3.2.11): a `1` bit followed by
/// zero bits until the next byte boundary.
pub fn write_rbsp_trailing_bits(bw: &mut BitWriter) {
    bw.write_u1(1);
    bw.align_to_byte_zero();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;

    #[test]
    fn write_and_read_back_bits() {
        let mut bw = BitWriter::new();
        bw.write_bits(0b1, 1);
        bw.write_bits(0b01, 2);
        bw.write_bits(0b10001, 5);
        bw.write_bits(0b0101_0101, 8);
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.u(1).unwrap(), 1);
        assert_eq!(br.u(2).unwrap(), 0b01);
        assert_eq!(br.u(5).unwrap(), 0b10001);
        assert_eq!(br.u(8).unwrap(), 0b0101_0101);
    }

    #[test]
    fn exp_golomb_roundtrip_unsigned() {
        for k in [0u32, 1, 2, 3, 4, 7, 15, 16, 31, 255, 1023, 65535] {
            let mut bw = BitWriter::new();
            bw.write_ue(k);
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            assert_eq!(br.ue().unwrap(), k, "ue roundtrip failed for {k}");
        }
    }

    #[test]
    fn exp_golomb_roundtrip_signed() {
        for v in [0i32, 1, -1, 2, -2, 3, -3, 255, -255, 1023, -1023] {
            let mut bw = BitWriter::new();
            bw.write_se(v);
            let bytes = bw.finish();
            let mut br = BitReader::new(&bytes);
            assert_eq!(br.se().unwrap(), v, "se roundtrip failed for {v}");
        }
    }

    #[test]
    fn exp_golomb_known_codes() {
        // ue: 0→'1', 1→'010', 2→'011', 3→'00100', 4→'00101'
        let mut bw = BitWriter::new();
        bw.write_ue(0);
        bw.write_ue(1);
        bw.write_ue(2);
        bw.write_ue(3);
        // Expected bit sequence: 1 010 011 00100 = 1010_0110_0100 = 0xA6 0x40 (pad)
        let bytes = bw.finish();
        assert_eq!(bytes, vec![0b1010_0110, 0b0100_0000]);
    }

    #[test]
    fn rbsp_trailing_bits_aligns() {
        let mut bw = BitWriter::new();
        bw.write_bits(0b101, 3); // 3 bits in acc
        write_rbsp_trailing_bits(&mut bw);
        let bytes = bw.finish();
        // 101 | 1 | 0000 → 10110000 = 0xB0
        assert_eq!(bytes, vec![0xB0]);
    }
}
