//! Generic ISOBMFF box walker for the HEIF scaffold.
//!
//! Spec: ISO/IEC 14496-12 §4.2 (box structure), §4.2.2 (FullBox).
//!
//! Mirrors the AVIF sibling's parser — we could depend on `oxideav-avif`
//! but that would introduce an image-format crate to what is a codec
//! crate. A small self-contained walker keeps the dependency graph
//! flat. All error messages say "heif" so `Error::invalid(...)` calls
//! remain self-identifying inside this crate.

use oxideav_core::{Error, Result};

/// 4-character box type.
pub type BoxType = [u8; 4];

pub const fn b(s: &[u8; 4]) -> BoxType {
    *s
}

pub fn type_str(t: &BoxType) -> String {
    String::from_utf8_lossy(t).into_owned()
}

#[derive(Clone, Debug)]
pub struct BoxHeader {
    pub box_type: BoxType,
    pub payload_start: usize,
    pub payload_len: usize,
    pub total_len: usize,
}

impl BoxHeader {
    pub fn end(&self) -> usize {
        self.payload_start + self.payload_len
    }
}

pub fn iter_boxes(buf: &[u8]) -> BoxIter<'_> {
    BoxIter { buf, cursor: 0 }
}

pub struct BoxIter<'a> {
    buf: &'a [u8],
    cursor: usize,
}

impl<'a> Iterator for BoxIter<'a> {
    type Item = Result<BoxHeader>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.buf.len() {
            return None;
        }
        match parse_box_header(self.buf, self.cursor) {
            Ok(h) => {
                self.cursor = h
                    .payload_start
                    .checked_add(h.payload_len)
                    .unwrap_or(self.buf.len());
                Some(Ok(h))
            }
            Err(e) => {
                self.cursor = self.buf.len();
                Some(Err(e))
            }
        }
    }
}

pub fn parse_box_header(buf: &[u8], start: usize) -> Result<BoxHeader> {
    if start + 8 > buf.len() {
        return Err(Error::invalid("heif: truncated box header"));
    }
    let size = read_u32(buf, start)?;
    let mut box_type = [0u8; 4];
    box_type.copy_from_slice(&buf[start + 4..start + 8]);
    let (payload_start, total_len) = if size == 1 {
        if start + 16 > buf.len() {
            return Err(Error::invalid("heif: truncated 'largesize' box"));
        }
        let ls = read_u64(buf, start + 8)?;
        if ls < 16 || (ls as usize) > buf.len() - start {
            return Err(Error::invalid(format!(
                "heif: box '{}' largesize {ls} out of range",
                type_str(&box_type)
            )));
        }
        (start + 16, ls as usize)
    } else if size == 0 {
        (start + 8, buf.len() - start)
    } else {
        let s = size as usize;
        if s < 8 || s > buf.len() - start {
            return Err(Error::invalid(format!(
                "heif: box '{}' size {s} out of range",
                type_str(&box_type)
            )));
        }
        (start + 8, s)
    };
    let payload_len = total_len
        .checked_sub(payload_start - start)
        .ok_or_else(|| {
            Error::invalid(format!(
                "heif: box '{}' header longer than total",
                type_str(&box_type)
            ))
        })?;
    Ok(BoxHeader {
        box_type,
        payload_start,
        payload_len,
        total_len,
    })
}

pub fn parse_full_box(payload: &[u8]) -> Result<(u8, u32, &[u8])> {
    if payload.len() < 4 {
        return Err(Error::invalid("heif: truncated FullBox header"));
    }
    let version = payload[0];
    let flags = ((payload[1] as u32) << 16) | ((payload[2] as u32) << 8) | (payload[3] as u32);
    Ok((version, flags, &payload[4..]))
}

pub fn find_box<'a>(buf: &'a [u8], target: &BoxType) -> Result<Option<(&'a [u8], BoxHeader)>> {
    for h in iter_boxes(buf) {
        let h = h?;
        if &h.box_type == target {
            let payload = &buf[h.payload_start..h.end()];
            return Ok(Some((payload, h)));
        }
    }
    Ok(None)
}

pub fn read_u16(buf: &[u8], at: usize) -> Result<u16> {
    if at + 2 > buf.len() {
        return Err(Error::invalid("heif: truncated u16 read"));
    }
    Ok(u16::from_be_bytes([buf[at], buf[at + 1]]))
}

pub fn read_u32(buf: &[u8], at: usize) -> Result<u32> {
    if at + 4 > buf.len() {
        return Err(Error::invalid("heif: truncated u32 read"));
    }
    Ok(u32::from_be_bytes([
        buf[at],
        buf[at + 1],
        buf[at + 2],
        buf[at + 3],
    ]))
}

pub fn read_u64(buf: &[u8], at: usize) -> Result<u64> {
    if at + 8 > buf.len() {
        return Err(Error::invalid("heif: truncated u64 read"));
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&buf[at..at + 8]);
    Ok(u64::from_be_bytes(b))
}

/// Variable-width big-endian unsigned integer of `width_bytes` bytes.
/// `width_bytes` may be 0, 4, or 8 per ISO/IEC 14496-12 §8.11.3 (iloc).
pub fn read_var_uint(buf: &[u8], at: usize, width_bytes: usize) -> Result<u64> {
    match width_bytes {
        0 => Ok(0),
        4 => read_u32(buf, at).map(|v| v as u64),
        8 => read_u64(buf, at),
        _ => Err(Error::invalid(format!(
            "heif: unsupported iloc field width {width_bytes}"
        ))),
    }
}

pub fn read_cstr(buf: &[u8], at: usize) -> Result<(String, usize)> {
    let mut i = at;
    while i < buf.len() && buf[i] != 0 {
        i += 1;
    }
    if i >= buf.len() {
        return Err(Error::invalid("heif: unterminated C string"));
    }
    let s = String::from_utf8_lossy(&buf[at..i]).into_owned();
    Ok((s, i + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walks_ftyp_then_meta() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x20u32.to_be_bytes());
        buf.extend_from_slice(b"ftyp");
        buf.extend_from_slice(&[0u8; 0x18]);
        buf.extend_from_slice(&0x08u32.to_be_bytes());
        buf.extend_from_slice(b"meta");
        let headers: Vec<_> = iter_boxes(&buf).collect::<Result<_>>().unwrap();
        assert_eq!(headers.len(), 2);
        assert_eq!(&headers[0].box_type, b"ftyp");
        assert_eq!(&headers[1].box_type, b"meta");
    }

    #[test]
    fn parse_error_names_the_box() {
        let buf = [0, 0, 0, 0x20, b'f', b't', b'y', b'p', 0, 0, 0]; // size=32 advertised, 11 present
        let err = parse_box_header(&buf, 0).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("ftyp"), "error should name the box: {msg}");
    }
}
