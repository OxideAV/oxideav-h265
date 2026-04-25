//! HEIF / HEIC (ISO/IEC 23008-12) container scaffold on top of the
//! crate's H.265 decoder.
//!
//! HEIC = HEIF container + HEVC-coded image items. The container layout
//! (`ftyp` / `meta` / `mdat` and the item-property hierarchy under
//! `meta`) is shared verbatim with AVIF — only the codec-specific
//! property box changes (`hvcC` for HEIC, `av1C` for AVIF) and the
//! item-type four-character code (`hvc1` for HEIC, `av01` for AVIF).
//! The meta-box syntax is specified in ISO/IEC 14496-12
//! (`docs/container/isobmff/ISO_IEC_14496-12_2015a.pdf`); the item
//! property machinery reused here follows the AVIF sibling
//! (`oxideav-avif`) and the published AV1-AVIF spec
//! (`docs/image/avif/av1-avif.html`), which is itself a specialisation
//! of the HEIF framework.
//!
//! # Scope (scaffold)
//!
//! * [`probe`] — ftyp brand sniff for HEIC (`heic`, `heix`, `heim`,
//!   `heis`, `hevc`, `hevx`) and the neutral HEIF image-collection
//!   brands (`mif1`, `msf1`).
//! * [`parse`] — walks `ftyp`, `meta` (and inside it `hdlr`, `pitm`,
//!   `iinf`, `iloc`, `iref`, `iprp` / `ipco` / `ipma`). Extracts the
//!   primary item's HEVC NAL payload byte range + its associated
//!   `hvcC` property.
//! * [`decode_primary`] — probe → walk → pre-pend VPS/SPS/PPS from
//!   `hvcC` → hand the concatenated NAL stream to [`crate::decoder::HevcDecoder`]
//!   via `send_packet` / `receive_frame`.
//!
//! Boxes the walker **handles** (descends into / decodes):
//!
//! * `ftyp` (compatibility brands)
//! * `meta` — FullBox header + nested:
//!   * `hdlr` — handler type surfaced (expected `pict`).
//!   * `pitm` — primary item id (v0 16-bit, v1 32-bit).
//!   * `iinf` / `infe` — per-item type strings (`hvc1`, `Exif`, …).
//!   * `iloc` — per-item file offset / length extents (v0/v1/v2,
//!     offset / length / base_offset widths taken from the iloc
//!     prefix; construction_method == 0 only).
//!   * `iref` — typed references (v0 16-bit IDs, v1 32-bit).
//!   * `iprp` → `ipco` + `ipma` — item property store + per-item
//!     associations, specifically typed extraction of `hvcC`, `ispe`,
//!     `colr`; anything else is retained as [`Property::Other`] so an
//!     association index stays valid.
//! * `mdat` — located only for byte-offset sanity; iloc offsets are
//!   file-absolute, so the walker does not descend into it.
//!
//! Boxes the walker **skips** (recognised but not decoded, by design):
//!
//! * `idat` — iloc `construction_method == 1` (item data embedded in
//!   `idat`) is surfaced as [`Error::Unsupported`] at decode time.
//! * `moov` / sample-table boxes — HEIF image sequences (`msf1` brand)
//!   reuse the movie-box track hierarchy. Out of scope for this
//!   scaffold; the `iinf`/`iloc` path only covers the still-image
//!   collection form.
//! * Item-property boxes beyond `hvcC` / `ispe` / `colr` (e.g. `pixi`,
//!   `pasp`, `irot`, `imir`, `clap`, `auxC`). They are stored as
//!   [`Property::Other`] but not typed — enough to keep `ipma`
//!   indices consistent.
//! * Everything at the top level that isn't `ftyp` / `meta` / `mdat`
//!   (e.g. `free`, `skip`).

use oxideav_core::{Error, Frame, Packet, Result, TimeBase, VideoFrame};

use crate::decoder::HevcDecoder;
use crate::hvcc::{parse_hvcc, HvcConfig};
use oxideav_core::CodecId;
use oxideav_core::Decoder;

mod box_parser;
mod meta;

pub use box_parser::{BoxHeader, BoxType};
pub use meta::{Colr, Ispe, ItemInfo, ItemLocation, Meta, Property};

use box_parser::{b, iter_boxes, read_u32, type_str};

const FTYP: BoxType = b(b"ftyp");
const META: BoxType = b(b"meta");
const MDAT: BoxType = b(b"mdat");

/// HEIC item type — HEVC-coded still image item, VPS/SPS/PPS carried in
/// the item's `hvcC` property rather than inline.
pub const ITEM_TYPE_HVC1: BoxType = b(b"hvc1");
/// Alternative HEIC item type — HEVC-coded image item, VPS/SPS/PPS may
/// appear inline in the NAL stream. Treated identically by this
/// scaffold (parameter sets always pre-pended from `hvcC`).
pub const ITEM_TYPE_HEV1: BoxType = b(b"hev1");

/// FourCC brands that imply HEIC / HEIF compatibility. We accept any
/// brand in this set either as the `major_brand` or in the `compatible_brands`
/// list of `ftyp`.
const HEIC_BRANDS: &[BoxType] = &[
    b(b"heic"), // ISO/IEC 23008-12: HEVC Main, single image
    b(b"heix"), // HEVC Main 10 / extended profiles, single image
    b(b"heim"), // L-HEVC image
    b(b"heis"), // L-HEVC image sequence
    b(b"hevc"), // HEVC image sequence
    b(b"hevx"), // HEVC extended image sequence
    b(b"mif1"), // neutral HEIF image (codec-agnostic)
    b(b"msf1"), // neutral HEIF image sequence
];

/// Probe `buf` for an HEIC / HEIF ftyp. Returns `true` when the first
/// box is an `ftyp` and its `major_brand` or any compatible brand
/// matches an HEIC-family or neutral HEIF brand. Unlike [`parse`],
/// this never reads past `ftyp` — it is safe to call on arbitrary
/// potentially-non-HEIC input.
pub fn probe(buf: &[u8]) -> bool {
    let Ok(Some((payload, _hdr))) = first_box(buf, &FTYP) else {
        return false;
    };
    let Ok((major, _minor, compat)) = parse_ftyp(payload) else {
        return false;
    };
    is_heic_brand(&major) || compat.iter().any(is_heic_brand)
}

fn is_heic_brand(b: &BoxType) -> bool {
    HEIC_BRANDS.contains(b)
}

/// Parse result: the whole HEIC file, ready for payload extraction.
#[derive(Clone, Debug)]
pub struct HeifImage<'a> {
    pub major_brand: BoxType,
    pub minor_version: u32,
    pub compatible_brands: Vec<BoxType>,
    pub meta: Meta,
    pub primary_item_id: u32,
    pub primary_item: ItemInfo,
    /// HEVC elementary stream bytes for the primary item, as stored in
    /// the file. For HEIC this is a concatenation of length-prefixed
    /// NAL units (ISO/IEC 14496-15 §8.3.3). `length_size` is carried
    /// alongside in `hvcc` (`length_size_minus_one + 1`).
    pub primary_item_data: &'a [u8],
    /// Raw body of the `hvcC` property associated with the primary
    /// item. The parsed form is available as `hvcc`.
    pub hvcc_raw: Vec<u8>,
    pub hvcc: HvcConfig,
    pub ispe: Option<Ispe>,
    pub colr: Option<Colr>,
}

/// Header-only parse — stops after `ftyp` + `meta`. The file slice is
/// retained so [`item_bytes`] can be used to resolve additional items
/// (e.g. thumbnail items referenced via `thmb`).
#[derive(Debug)]
pub struct HeifHeader<'a> {
    pub file: &'a [u8],
    pub major_brand: BoxType,
    pub minor_version: u32,
    pub compatible_brands: Vec<BoxType>,
    pub meta: Meta,
}

/// Walk `ftyp` + `meta`. Fails with a descriptive [`Error::invalid`]
/// naming the missing or malformed box when the file is not a valid
/// HEIC / HEIF still image.
pub fn parse_header(file: &[u8]) -> Result<HeifHeader<'_>> {
    let mut ftyp_payload: Option<&[u8]> = None;
    let mut meta_payload: Option<&[u8]> = None;
    for hdr in iter_boxes(file) {
        let hdr = hdr?;
        let payload = &file[hdr.payload_start..hdr.end()];
        match &hdr.box_type {
            x if x == &FTYP => ftyp_payload = Some(payload),
            x if x == &META => meta_payload = Some(payload),
            x if x == &MDAT => {}
            _ => {}
        }
    }
    let ftyp = ftyp_payload.ok_or_else(|| Error::invalid("heif: missing 'ftyp' box"))?;
    let (major_brand, minor_version, compatible_brands) = parse_ftyp(ftyp)?;
    if !(is_heic_brand(&major_brand) || compatible_brands.iter().any(is_heic_brand)) {
        return Err(Error::invalid(format!(
            "heif: ftyp major='{}' compatible_brands=[{}] doesn't claim any HEIC/HEIF brand",
            type_str(&major_brand),
            compatible_brands
                .iter()
                .map(type_str)
                .collect::<Vec<_>>()
                .join(","),
        )));
    }
    let meta_p =
        meta_payload.ok_or_else(|| Error::invalid("heif: missing top-level 'meta' box"))?;
    let meta = Meta::parse(meta_p)?;
    Ok(HeifHeader {
        file,
        major_brand,
        minor_version,
        compatible_brands,
        meta,
    })
}

/// Full parse that also resolves the primary item's HEVC payload + its
/// `hvcC` property, ready for decode.
pub fn parse(file: &[u8]) -> Result<HeifImage<'_>> {
    let hdr = parse_header(file)?;
    let HeifHeader {
        file: _,
        major_brand,
        minor_version,
        compatible_brands,
        meta,
    } = hdr;

    let primary_id = meta
        .primary_item_id
        .ok_or_else(|| Error::invalid("heif: missing 'pitm' box (no primary item)"))?;
    let primary_info = meta
        .item_by_id(primary_id)
        .ok_or_else(|| {
            Error::invalid(format!(
                "heif: 'pitm' references unknown item id {primary_id}"
            ))
        })?
        .clone();
    if primary_info.item_type != ITEM_TYPE_HVC1 && primary_info.item_type != ITEM_TYPE_HEV1 {
        return Err(Error::unsupported(format!(
            "heif: primary item type '{}' is not 'hvc1' / 'hev1' — only HEVC-coded image items are handled by this crate",
            type_str(&primary_info.item_type)
        )));
    }

    let loc = meta.location_by_id(primary_id).ok_or_else(|| {
        Error::invalid(format!(
            "heif: primary item {primary_id} missing from 'iloc'"
        ))
    })?;
    let primary_data = item_bytes(file, loc)?;

    let hvcc_raw = match meta.property_for(primary_id, b"hvcC") {
        Some(Property::HvcC(bytes)) => bytes.clone(),
        _ => {
            return Err(Error::invalid(
                "heif: primary item missing 'hvcC' property (no VPS/SPS/PPS)",
            ))
        }
    };
    let hvcc = parse_hvcc(&hvcc_raw)?;

    let ispe = match meta.property_for(primary_id, b"ispe") {
        Some(Property::Ispe(v)) => Some(*v),
        _ => None,
    };
    let colr = match meta.property_for(primary_id, b"colr") {
        Some(Property::Colr(v)) => Some(v.clone()),
        _ => None,
    };

    Ok(HeifImage {
        major_brand,
        minor_version,
        compatible_brands,
        meta,
        primary_item_id: primary_id,
        primary_item: primary_info,
        primary_item_data: primary_data,
        hvcc_raw,
        hvcc,
        ispe,
        colr,
    })
}

/// Resolve an arbitrary item's extent bytes via its `iloc` entry.
/// `construction_method == 0` (file offset) only.
pub fn item_bytes<'a>(file: &'a [u8], loc: &ItemLocation) -> Result<&'a [u8]> {
    if loc.construction_method != 0 {
        return Err(Error::unsupported(format!(
            "heif: iloc construction_method {} not supported (only file-offset is handled)",
            loc.construction_method
        )));
    }
    match loc.extents.len() {
        0 => Err(Error::invalid("heif: 'iloc' entry has no extents")),
        1 => {
            let e = &loc.extents[0];
            let start = loc
                .base_offset
                .checked_add(e.offset)
                .ok_or_else(|| Error::invalid("heif: 'iloc' offset overflow"))?;
            let end = start
                .checked_add(e.length)
                .ok_or_else(|| Error::invalid("heif: 'iloc' length overflow"))?;
            let (start, end) = (start as usize, end as usize);
            if end > file.len() {
                return Err(Error::invalid(format!(
                    "heif: 'iloc' extent {start}..{end} exceeds file length {}",
                    file.len()
                )));
            }
            Ok(&file[start..end])
        }
        _ => {
            // Concatenate into a new buffer would change lifetime — not
            // exposed in the scaffold. Surface as Unsupported so the
            // caller can fall back to a byte-level walk if needed.
            Err(Error::unsupported(
                "heif: multi-extent items not yet handled by the HEIC scaffold",
            ))
        }
    }
}

/// Decode the primary HEVC-coded image item end-to-end:
/// probe → parse → pre-pend VPS/SPS/PPS from `hvcC` → feed through
/// [`HevcDecoder::send_packet`] / `receive_frame`.
///
/// The HEVC payload inside HEIC is length-prefixed (ISO/IEC 14496-15
/// §8.3.3), mirroring how HEVC travels inside MP4 `hvc1` sample
/// entries. We build a single synthetic packet containing:
///
/// 1. Each NAL unit from every `nal_arrays[]` entry of `hvcC` (VPS,
///    SPS, PPS, SEI, …) as a length-prefixed NAL (2-byte big-endian
///    length chosen so all NAL sizes fit; the decoder is told the
///    `length_size` via [`HevcDecoder::consume_extradata`]).
/// 2. The primary item's `mdat` bytes verbatim.
///
/// Since `consume_extradata` already feeds the `hvcC` parameter-set
/// NALs into the decoder's VPS/SPS/PPS tables, the primary item's
/// length-prefixed NAL stream only needs to carry the VCL NALs. We
/// still forward the parameter sets via extradata and then send the
/// item payload untouched as the packet body — this is what MP4
/// demuxers do in practice.
pub fn decode_primary(bytes: &[u8]) -> Result<VideoFrame> {
    if !probe(bytes) {
        return Err(Error::invalid(
            "heif: probe failed (ftyp does not claim HEIC/HEIF compatibility)",
        ));
    }
    let img = parse(bytes)?;

    let mut dec = HevcDecoder::new(CodecId::new(crate::CODEC_ID_STR));
    dec.consume_extradata(&img.hvcc_raw)?;

    let packet =
        Packet::new(0, TimeBase::new(1, 1), img.primary_item_data.to_vec()).with_keyframe(true);
    dec.send_packet(&packet)?;
    dec.flush()?;
    match dec.receive_frame() {
        Ok(Frame::Video(vf)) => Ok(vf),
        Ok(_) => Err(Error::invalid(
            "heif: HEVC decoder produced a non-video frame for an image item",
        )),
        Err(e) => Err(e),
    }
}

fn first_box<'a>(buf: &'a [u8], target: &BoxType) -> Result<Option<(&'a [u8], BoxHeader)>> {
    for h in iter_boxes(buf) {
        let h = h?;
        if &h.box_type == target {
            let payload = &buf[h.payload_start..h.end()];
            return Ok(Some((payload, h)));
        }
    }
    Ok(None)
}

fn parse_ftyp(payload: &[u8]) -> Result<(BoxType, u32, Vec<BoxType>)> {
    if payload.len() < 8 {
        return Err(Error::invalid("heif: 'ftyp' too short (< 8 bytes)"));
    }
    let mut major = [0u8; 4];
    major.copy_from_slice(&payload[..4]);
    let minor = read_u32(payload, 4)?;
    let mut brands = Vec::new();
    let mut cursor = 8;
    while cursor + 4 <= payload.len() {
        let mut b4 = [0u8; 4];
        b4.copy_from_slice(&payload[cursor..cursor + 4]);
        brands.push(b4);
        cursor += 4;
    }
    Ok((major, minor, brands))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_rejects_non_heic() {
        // ftyp claiming 'mp42' brand — not HEIC.
        let mut buf = Vec::new();
        let ftyp = {
            let mut p = Vec::new();
            p.extend_from_slice(b"mp42"); // major
            p.extend_from_slice(&0u32.to_be_bytes()); // minor
            p.extend_from_slice(b"mp42");
            p.extend_from_slice(b"isom");
            p
        };
        buf.extend_from_slice(&((ftyp.len() + 8) as u32).to_be_bytes());
        buf.extend_from_slice(b"ftyp");
        buf.extend_from_slice(&ftyp);
        assert!(!probe(&buf));
    }

    #[test]
    fn probe_accepts_heic_major() {
        let mut buf = Vec::new();
        let ftyp = {
            let mut p = Vec::new();
            p.extend_from_slice(b"heic");
            p.extend_from_slice(&0u32.to_be_bytes());
            p.extend_from_slice(b"mif1");
            p.extend_from_slice(b"heic");
            p
        };
        buf.extend_from_slice(&((ftyp.len() + 8) as u32).to_be_bytes());
        buf.extend_from_slice(b"ftyp");
        buf.extend_from_slice(&ftyp);
        assert!(probe(&buf));
    }

    #[test]
    fn probe_accepts_mif1_compat() {
        // Major 'heix', compat includes mif1.
        let mut buf = Vec::new();
        let ftyp = {
            let mut p = Vec::new();
            p.extend_from_slice(b"heix");
            p.extend_from_slice(&0u32.to_be_bytes());
            p.extend_from_slice(b"mif1");
            p
        };
        buf.extend_from_slice(&((ftyp.len() + 8) as u32).to_be_bytes());
        buf.extend_from_slice(b"ftyp");
        buf.extend_from_slice(&ftyp);
        assert!(probe(&buf));
    }

    #[test]
    fn parse_header_errors_name_missing_box() {
        // Bytes that parse as an `ftyp` but with no meta box.
        let mut buf = Vec::new();
        let ftyp_payload = {
            let mut p = Vec::new();
            p.extend_from_slice(b"heic");
            p.extend_from_slice(&0u32.to_be_bytes());
            p.extend_from_slice(b"mif1");
            p
        };
        buf.extend_from_slice(&((ftyp_payload.len() + 8) as u32).to_be_bytes());
        buf.extend_from_slice(b"ftyp");
        buf.extend_from_slice(&ftyp_payload);
        let err = parse_header(&buf).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("'meta'"),
            "missing-meta error must name the box: {msg}"
        );
    }
}
