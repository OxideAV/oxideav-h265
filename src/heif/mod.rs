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
//!   `idat`) is now resolved by [`item_bytes`]; the meta walker stores
//!   the body on [`Meta::idat`].
//! * `moov` / sample-table boxes — HEIF image sequences (`msf1` brand)
//!   reuse the movie-box track hierarchy. Out of scope for this
//!   scaffold; the `iinf`/`iloc` path only covers the still-image
//!   collection form.
//! * Item-property boxes beyond `hvcC` / `ispe` / `colr` / `auxC` /
//!   `clap` / `irot` / `imir` (e.g. `pixi`, `pasp`). They are stored
//!   as [`Property::Other`] but not typed — enough to keep `ipma`
//!   indices consistent. The transformative property triplet
//!   (`clap` / `irot` / `imir`) is applied in `ipma` order to every
//!   frame produced by [`decode_primary`] / [`decode_item`] —
//!   ISO/IEC 14496-12 §12.1.4 + ISO/IEC 23008-12 §6.5.10 / §6.5.12.
//! * Everything at the top level that isn't `ftyp` / `meta` / `mdat`
//!   (e.g. `free`, `skip`).

use oxideav_core::{Error, Frame, Packet, Result, TimeBase, VideoFrame, VideoPlane};

use crate::decoder::HevcDecoder;
use crate::hvcc::{parse_hvcc, HvcConfig};
use oxideav_core::CodecId;
use oxideav_core::Decoder;

mod box_parser;
mod meta;
mod moov;

pub use box_parser::{BoxHeader, BoxType};
pub use meta::{
    AuxC, Clap, Colr, Imir, IrefEntry, Irot, Ispe, ItemInfo, ItemLocation, Meta, Property,
};
pub use moov::{
    decode_image_sequence, parse_moov, sample_bytes as moov_sample_bytes, MoovSummary,
    Sample as MoovSample,
};

/// Auxiliary-image type URN that identifies the alpha plane of an HEVC
/// item per ISO/IEC 23008-12 §6.6.2.1.1. The matching item is
/// recognised by an `auxC` property whose `aux_type` equals this URN
/// and an `auxl` iref pointing at the colour primary item.
pub const ALPHA_URN_HEVC: &str = "urn:mpeg:hevc:2015:auxid:1";

/// `grid` derived item type — ISO/IEC 23008-12 §6.6.2.3.2. The item
/// payload is an `ImageGrid` struct (parsed via [`ImageGrid::parse`])
/// and `dimg` references list the constituent HEVC tile items in
/// row-major order.
pub const ITEM_TYPE_GRID: BoxType = b(b"grid");

/// `iovl` derived item type — ISO/IEC 23008-12 §6.6.2.3.3. Layered
/// composition of several constituent items, each with its own
/// `(offset_x, offset_y)` placement, on a canvas filled with a 16-bit
/// RGBA fill colour. Composition is implemented end-to-end by
/// [`decode_primary`] (round 3 phase C); per-layer transformative
/// properties (`clap` / `irot` / `imir`) are honoured because each
/// layer is decoded via [`decode_item`].
pub const ITEM_TYPE_IOVL: BoxType = b(b"iovl");

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

    // `parse()` returns slices borrowed from `file`, so we can only
    // resolve cm=0 (file-relative) extents here — a cm=1 extent would
    // be borrowed from `meta.idat` which goes out of scope when this
    // function returns. cm=1 callers should use `decode_primary` (which
    // keeps the meta alive) or `item_bytes_in` directly.
    let loc = meta.location_by_id(primary_id).ok_or_else(|| {
        Error::invalid(format!(
            "heif: primary item {primary_id} missing from 'iloc'"
        ))
    })?;
    if loc.construction_method != 0 {
        return Err(Error::unsupported(format!(
            "heif: parse() requires file-relative iloc; primary item uses construction_method={} (call decode_primary or item_bytes_in instead)",
            loc.construction_method
        )));
    }
    let primary_data = resolve_extent(file, loc, 0)?;

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
/// Supports `construction_method == 0` (file offset, ISO/IEC 14496-12
/// §8.11.3) and `construction_method == 1` (item-data box embedded in
/// `meta`, ibid. §8.11.4). Multi-extent items remain unsupported.
///
/// Returns a borrowed slice — for cm=0 into `file`, for cm=1 into the
/// meta's `idat` body. Most callers should use [`item_bytes_in`] which
/// looks the [`ItemLocation`] up by item id in one step.
pub fn item_bytes<'a>(file: &'a [u8], meta: &'a Meta, loc: &ItemLocation) -> Result<&'a [u8]> {
    match loc.construction_method {
        0 => resolve_extent(file, loc, 0),
        1 => {
            // §8.11.4 — offsets are relative to the start of the idat
            // payload, NOT the file. base_offset still applies (for
            // multi-source files) but is normally 0.
            if meta.idat.is_empty() {
                return Err(Error::invalid(
                    "heif: iloc construction_method=1 references absent 'idat' box",
                ));
            }
            resolve_extent(&meta.idat, loc, 0)
        }
        2 => Err(Error::unsupported(
            "heif: iloc construction_method=2 (item-relative) not yet supported",
        )),
        v => Err(Error::invalid(format!(
            "heif: iloc construction_method={v} (not in spec)"
        ))),
    }
}

/// Locate item `item_id` in `meta` and return its extent bytes via the
/// shared cm-aware [`item_bytes`].
pub fn item_bytes_in<'a>(file: &'a [u8], meta: &'a Meta, item_id: u32) -> Result<&'a [u8]> {
    let loc = meta
        .location_by_id(item_id)
        .ok_or_else(|| Error::invalid(format!("heif: item {item_id} missing from 'iloc'")))?;
    item_bytes(file, meta, loc)
}

/// Single-extent resolver shared by cm=0 (off `file`) and cm=1 (off
/// `meta.idat`). Both paths feed into here once the source slice has
/// been picked.
fn resolve_extent<'a>(src: &'a [u8], loc: &ItemLocation, base: u64) -> Result<&'a [u8]> {
    match loc.extents.len() {
        0 => Err(Error::invalid("heif: 'iloc' entry has no extents")),
        1 => {
            let e = &loc.extents[0];
            let start = base
                .checked_add(loc.base_offset)
                .and_then(|b| b.checked_add(e.offset))
                .ok_or_else(|| Error::invalid("heif: 'iloc' offset overflow"))?;
            let end = start
                .checked_add(e.length)
                .ok_or_else(|| Error::invalid("heif: 'iloc' length overflow"))?;
            let (start, end) = (start as usize, end as usize);
            if end > src.len() {
                return Err(Error::invalid(format!(
                    "heif: 'iloc' extent {start}..{end} exceeds source length {}",
                    src.len()
                )));
            }
            Ok(&src[start..end])
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

/// `ImageGrid` derived-item payload — ISO/IEC 23008-12 §6.6.2.3.2. The
/// item bytes for an `item_type == 'grid'` consist of:
///
/// ```text
/// u8 version                (must be 0)
/// u8 flags                  (bit 0 = 1: 32-bit dims, else 16-bit)
/// u8 rows_minus_one
/// u8 columns_minus_one
/// uN output_width           (16- or 32-bit per flags bit 0)
/// uN output_height          (...)
/// ```
///
/// `rows`/`columns` here carry the post-`+1` value; `output_width` and
/// `output_height` are the final composited canvas size in luma pixels.
#[derive(Clone, Copy, Debug)]
pub struct ImageGrid {
    pub version: u8,
    pub flags: u8,
    pub rows: u16,
    pub columns: u16,
    pub output_width: u32,
    pub output_height: u32,
}

impl ImageGrid {
    /// Parse a `grid`-item payload (the bytes returned by
    /// [`item_bytes_in`] for an item whose type is [`ITEM_TYPE_GRID`]).
    pub fn parse(payload: &[u8]) -> Result<Self> {
        if payload.len() < 8 {
            return Err(Error::invalid(format!(
                "heif: grid payload {} bytes < 8",
                payload.len()
            )));
        }
        let version = payload[0];
        if version != 0 {
            return Err(Error::invalid(format!("heif: grid version {version} != 0")));
        }
        let flags = payload[1];
        let wide = (flags & 1) != 0;
        let rows = (payload[2] as u16) + 1;
        let columns = (payload[3] as u16) + 1;
        let mut pos = 4;
        let (output_width, output_height) = if wide {
            if payload.len() < pos + 8 {
                return Err(Error::invalid("heif: grid 32-bit dims truncated"));
            }
            let w = u32::from_be_bytes([
                payload[pos],
                payload[pos + 1],
                payload[pos + 2],
                payload[pos + 3],
            ]);
            pos += 4;
            let h = u32::from_be_bytes([
                payload[pos],
                payload[pos + 1],
                payload[pos + 2],
                payload[pos + 3],
            ]);
            (w, h)
        } else {
            if payload.len() < pos + 4 {
                return Err(Error::invalid("heif: grid 16-bit dims truncated"));
            }
            let w = u16::from_be_bytes([payload[pos], payload[pos + 1]]) as u32;
            let h = u16::from_be_bytes([payload[pos + 2], payload[pos + 3]]) as u32;
            (w, h)
        };
        Ok(Self {
            version,
            flags,
            rows,
            columns,
            output_width,
            output_height,
        })
    }

    pub fn expected_tile_count(&self) -> usize {
        (self.rows as usize) * (self.columns as usize)
    }
}

/// `ImageOverlay` derived-item payload — ISO/IEC 23008-12 §6.6.2.3.3.
/// The item bytes for an `item_type == 'iovl'` have the layout:
///
/// ```text
/// u8 version                (must be 0)
/// u8 flags                  (bit 0 = 1: 32-bit dims+offsets, else 16-bit)
/// u16 canvas_fill_R         (always 16-bit per spec, regardless of flags)
/// u16 canvas_fill_G
/// u16 canvas_fill_B
/// u16 canvas_fill_A
/// uN output_width
/// uN output_height
/// per dimg-referenced source image (in iref order):
///   sN horizontal_offset
///   sN vertical_offset
/// ```
///
/// `dimg` reference order on the item gives the layer order (back-to-
/// front); the per-source `(h_offset, v_offset)` is signed and may be
/// negative so a layer can spill past the canvas edge. The number of
/// `(offset_x, offset_y)` pairs equals the `dimg` reference count;
/// pass that count via `n_refs` so the parser knows when to stop.
#[derive(Clone, Debug)]
pub struct ImageOverlay {
    pub version: u8,
    pub flags: u8,
    pub canvas_fill_rgba: [u16; 4],
    pub output_width: u32,
    pub output_height: u32,
    /// Per constituent `(h_offset, v_offset)` in **luma** pixels.
    pub offsets: Vec<(i32, i32)>,
}

impl ImageOverlay {
    /// Parse an `iovl`-item payload. `n_refs` is the count of `dimg`
    /// references on the item — the parser reads exactly that many
    /// `(h_offset, v_offset)` pairs after the canvas declaration.
    pub fn parse(payload: &[u8], n_refs: usize) -> Result<Self> {
        if payload.len() < 12 {
            return Err(Error::invalid(format!(
                "heif: iovl payload {} bytes < 12 (version + flags + RGBA fill)",
                payload.len()
            )));
        }
        let version = payload[0];
        if version != 0 {
            return Err(Error::invalid(format!("heif: iovl version {version} != 0")));
        }
        let flags = payload[1];
        let wide = (flags & 1) != 0;
        let mut canvas_fill_rgba = [0u16; 4];
        for (i, w) in canvas_fill_rgba.iter_mut().enumerate() {
            *w = u16::from_be_bytes([payload[2 + i * 2], payload[3 + i * 2]]);
        }
        let mut pos = 10;
        let dim_bytes = if wide { 4 } else { 2 };
        if payload.len() < pos + dim_bytes * 2 {
            return Err(Error::invalid(
                "heif: iovl truncated at output_width/height",
            ));
        }
        let output_width = read_var_be(payload, pos, wide)?;
        pos += dim_bytes;
        let output_height = read_var_be(payload, pos, wide)?;
        pos += dim_bytes;
        let off_bytes = dim_bytes;
        if payload.len() < pos + 2 * off_bytes * n_refs {
            return Err(Error::invalid(format!(
                "heif: iovl truncated at per-ref offsets (need {} bytes for {} refs)",
                2 * off_bytes * n_refs,
                n_refs
            )));
        }
        let mut offsets = Vec::with_capacity(n_refs);
        for _ in 0..n_refs {
            let h = read_var_be_signed(payload, pos, wide)?;
            pos += off_bytes;
            let v = read_var_be_signed(payload, pos, wide)?;
            pos += off_bytes;
            offsets.push((h, v));
        }
        Ok(Self {
            version,
            flags,
            canvas_fill_rgba,
            output_width,
            output_height,
            offsets,
        })
    }
}

fn read_var_be(buf: &[u8], at: usize, wide: bool) -> Result<u32> {
    if wide {
        if at + 4 > buf.len() {
            return Err(Error::invalid("heif: var_be truncated u32"));
        }
        Ok(u32::from_be_bytes([
            buf[at],
            buf[at + 1],
            buf[at + 2],
            buf[at + 3],
        ]))
    } else {
        if at + 2 > buf.len() {
            return Err(Error::invalid("heif: var_be truncated u16"));
        }
        Ok(u16::from_be_bytes([buf[at], buf[at + 1]]) as u32)
    }
}

fn read_var_be_signed(buf: &[u8], at: usize, wide: bool) -> Result<i32> {
    if wide {
        if at + 4 > buf.len() {
            return Err(Error::invalid("heif: var_be_signed truncated i32"));
        }
        Ok(i32::from_be_bytes([
            buf[at],
            buf[at + 1],
            buf[at + 2],
            buf[at + 3],
        ]))
    } else {
        if at + 2 > buf.len() {
            return Err(Error::invalid("heif: var_be_signed truncated i16"));
        }
        Ok(i16::from_be_bytes([buf[at], buf[at + 1]]) as i32)
    }
}

/// Decode any single HEVC-coded item by id (`hvc1` / `hev1`). Useful
/// for callers that want to walk auxiliary / thumbnail / burst items
/// alongside the primary; the public [`decode_primary`] is the right
/// entry point for the typical "open this HEIC and give me the
/// picture" path.
pub fn decode_item(file: &[u8], item_id: u32) -> Result<VideoFrame> {
    let hdr = parse_header(file)?;
    let info = hdr
        .meta
        .item_by_id(item_id)
        .ok_or_else(|| Error::invalid(format!("heif: decode_item: id {item_id} not in 'iinf'")))?;
    if info.item_type != ITEM_TYPE_HVC1 && info.item_type != ITEM_TYPE_HEV1 {
        return Err(Error::unsupported(format!(
            "heif: decode_item: id {item_id} item_type '{}' is not 'hvc1' / 'hev1'",
            type_str(&info.item_type)
        )));
    }
    decode_hvc_item(&hdr, item_id)
}

/// Decode the alpha auxiliary item attached to the primary, if any.
/// Returns `Ok(None)` when the primary has no alpha aux; an
/// `Err(Error::Unsupported)` when the alpha plane is monochrome HEVC
/// and the underlying decoder doesn't yet emit `Gray8` plane output.
///
/// # Phase-E status (2026-05)
///
/// HEIF's alpha aux convention always encodes the alpha plane as a
/// monochrome (`chroma_format_idc == 0`) HEVC bitstream — every
/// fixture in the corpus that has an alpha auxiliary (
/// `still-image-with-alpha`, `still-image-overlay`'s stamp side)
/// triggers this path. The underlying [`HevcDecoder`] does not yet
/// emit single-plane `Gray8` output, so this function surfaces the
/// decoder's `Error::Unsupported` verbatim.
///
/// Adding HEVC monochrome decoding is a separate, larger task — out
/// of scope for HEIF round 3 (which lands clap / irot / imir / iovl
/// composition on top of an existing colour-HEVC decode pipeline).
/// Once the decoder grows monochrome support, this function will
/// just start working — no HEIF-side change is needed.
pub fn decode_alpha_for_primary(file: &[u8]) -> Result<Option<VideoFrame>> {
    let hdr = parse_header(file)?;
    let primary_id = hdr
        .meta
        .primary_item_id
        .ok_or_else(|| Error::invalid("heif: missing 'pitm'"))?;
    let Some(alpha_id) = find_alpha_item_id(&hdr.meta, primary_id) else {
        return Ok(None);
    };
    decode_hvc_item(&hdr, alpha_id).map(Some)
}

/// Locate the alpha auxiliary item for `primary_id`. An item qualifies
/// when:
///
/// 1. An `auxl` iref entry has `to_ids` containing `primary_id` and
///    `from_id` set to the alpha candidate.
/// 2. The candidate carries an `auxC` property whose `aux_type` equals
///    [`ALPHA_URN_HEVC`].
///
/// Spec: ISO/IEC 23008-12 §6.6.2.1 (auxiliary images) + §6.5.5
/// (`auxl`).
pub fn find_alpha_item_id(meta: &Meta, primary_id: u32) -> Option<u32> {
    let cand = meta.iref_source_of(b"auxl", primary_id)?;
    if let Some(Property::AuxC(aux)) = meta.property_for(cand, b"auxC") {
        if aux.aux_type.starts_with(ALPHA_URN_HEVC) {
            return Some(cand);
        }
    }
    None
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
    let hdr = parse_header(bytes)?;
    let primary_id = hdr
        .meta
        .primary_item_id
        .ok_or_else(|| Error::invalid("heif: missing 'pitm' box (no primary item)"))?;
    let primary_info = hdr
        .meta
        .item_by_id(primary_id)
        .ok_or_else(|| {
            Error::invalid(format!(
                "heif: 'pitm' references unknown item id {primary_id}"
            ))
        })?
        .clone();
    if primary_info.item_type == ITEM_TYPE_GRID {
        decode_grid_primary(&hdr, primary_id)
    } else if primary_info.item_type == ITEM_TYPE_HVC1 || primary_info.item_type == ITEM_TYPE_HEV1 {
        decode_hvc_item(&hdr, primary_id)
    } else if primary_info.item_type == ITEM_TYPE_IOVL {
        decode_iovl_primary(&hdr, primary_id)
    } else {
        Err(Error::unsupported(format!(
            "heif: primary item type '{}' not supported",
            type_str(&primary_info.item_type)
        )))
    }
}

/// Decode a single HEVC-coded item (`hvc1` / `hev1`) end-to-end. The
/// item's `hvcC` is fed into the decoder as extradata; the item bytes
/// are submitted as one length-prefixed-NAL packet. After HEVC decode,
/// any transformative item properties associated with the item
/// (`clap`, `irot`, `imir`) are applied in `ipma` order — see
/// [`apply_transforms`].
fn decode_hvc_item(hdr: &HeifHeader<'_>, item_id: u32) -> Result<VideoFrame> {
    let item_data = item_bytes_in(hdr.file, &hdr.meta, item_id)?;
    let hvcc_raw = match hdr.meta.property_for(item_id, b"hvcC") {
        Some(Property::HvcC(bytes)) => bytes.clone(),
        _ => {
            return Err(Error::invalid(format!(
                "heif: item {item_id} missing 'hvcC' property"
            )))
        }
    };
    let mut dec = HevcDecoder::new(CodecId::new(crate::CODEC_ID_STR));
    dec.consume_extradata(&hvcc_raw)?;
    let packet = Packet::new(0, TimeBase::new(1, 1), item_data.to_vec()).with_keyframe(true);
    dec.send_packet(&packet)?;
    dec.flush()?;
    let raw = match dec.receive_frame() {
        Ok(Frame::Video(vf)) => vf,
        Ok(_) => {
            return Err(Error::invalid(
                "heif: HEVC decoder produced a non-video frame for an image item",
            ))
        }
        Err(e) => return Err(e),
    };
    apply_transforms(&hdr.meta, item_id, raw)
}

/// Apply the chain of transformative item properties associated with
/// `item_id` to a freshly decoded HEVC frame. Per ISO/IEC 23008-12
/// §6.5, `clap` / `irot` / `imir` are *transformative* properties and
/// readers shall apply them in the order they appear in `ipma`.
///
/// Each transform is implemented for planar YUV frames produced by the
/// HEVC decoder. `clap` extracts the centred rectangle declared by
/// CleanApertureBox; `irot` rotates by 0/90/180/270° anti-clockwise;
/// `imir` flips about a vertical (axis=0) or horizontal (axis=1)
/// axis. A frame whose plane 0 stride doesn't equal its declared row
/// count after `data.len() / stride` is treated as already cropped at
/// stride width — no implicit row crop is performed.
fn apply_transforms(meta: &Meta, item_id: u32, mut frame: VideoFrame) -> Result<VideoFrame> {
    for (prop, _essential) in meta.properties_for(item_id) {
        match prop {
            Property::Clap(c) => frame = apply_clap(*c, frame)?,
            Property::Irot(r) => frame = apply_irot(*r, frame)?,
            Property::Imir(m) => frame = apply_imir(*m, frame)?,
            _ => {}
        }
    }
    Ok(frame)
}

/// Resolve a `clap` to an integer pixel rectangle within an encoded
/// `width × height` picture. Returns `(left, top, w, h)` in luma
/// samples. The clean-aperture rectangle is centred about
/// `(horizOff + (width - 1)/2, vertOff + (height - 1)/2)` per
/// 14496-12 §12.1.4.1; we round the resolved fractions to the nearest
/// integer (libheif behaviour). Returns `Err` for degenerate inputs:
/// zero denominators, zero or negative aperture extent, or a rectangle
/// that lies outside the encoded frame.
pub fn clap_rect(clap: &Clap, width: u32, height: u32) -> Result<(u32, u32, u32, u32)> {
    if clap.width_d == 0 || clap.height_d == 0 || clap.horiz_off_d == 0 || clap.vert_off_d == 0 {
        return Err(Error::invalid("heif: 'clap' has zero denominator"));
    }
    if clap.width_n == 0 || clap.height_n == 0 {
        return Err(Error::invalid(
            "heif: 'clap' aperture has zero numerator (degenerate)",
        ));
    }
    // All math in i64 to keep negative offsets + multiplications by
    // u32 numerators clear of overflow on 32-bit boundary cases.
    let w_n = clap.width_n as i64;
    let w_d = clap.width_d as i64;
    let h_n = clap.height_n as i64;
    let h_d = clap.height_d as i64;
    let aperture_w = (w_n + w_d / 2) / w_d; // round-to-nearest
    let aperture_h = (h_n + h_d / 2) / h_d;
    if aperture_w <= 0 || aperture_h <= 0 {
        return Err(Error::invalid(format!(
            "heif: 'clap' aperture {}x{} after rounding is non-positive",
            aperture_w, aperture_h
        )));
    }
    // Compute pcX * 2 in a unit that keeps integer precision.
    // pcX = horizOff_n / horizOff_d + (width - 1) / 2
    // 2*pcX = 2*horizOff_n / horizOff_d + (width - 1)
    // For typical cases horizOff_d divides 2 evenly (it's normally 1
    // or 2 in HEIF). We take the floor of 2*horizOff_n / horizOff_d
    // and then halve the resulting rectangle origin.
    let two_pc_x_off = if clap.horiz_off_d == 1 {
        2 * (clap.horiz_off_n as i64)
    } else {
        // round-to-nearest division
        let num = 2 * (clap.horiz_off_n as i64);
        let d = clap.horiz_off_d as i64;
        if num >= 0 {
            (num + d / 2) / d
        } else {
            -(((-num) + d / 2) / d)
        }
    };
    let two_pc_y_off = if clap.vert_off_d == 1 {
        2 * (clap.vert_off_n as i64)
    } else {
        let num = 2 * (clap.vert_off_n as i64);
        let d = clap.vert_off_d as i64;
        if num >= 0 {
            (num + d / 2) / d
        } else {
            -(((-num) + d / 2) / d)
        }
    };
    let two_pc_x = two_pc_x_off + (width as i64 - 1);
    let two_pc_y = two_pc_y_off + (height as i64 - 1);
    // Leftmost pixel at pcX - (aperture_w - 1)/2:
    // 2*left = 2*pcX - (aperture_w - 1)
    let two_left = two_pc_x - (aperture_w - 1);
    let two_top = two_pc_y - (aperture_h - 1);
    if two_left < 0 || two_top < 0 || (two_left & 1) != 0 || (two_top & 1) != 0 {
        // Fractional-pixel offsets aren't representable on a YUV grid
        // (sub-pixel cropping needs resampling). Reject.
        return Err(Error::invalid(format!(
            "heif: 'clap' resolves to fractional / negative origin (2*left={two_left}, 2*top={two_top})"
        )));
    }
    let left = (two_left / 2) as u32;
    let top = (two_top / 2) as u32;
    let aw = aperture_w as u32;
    let ah = aperture_h as u32;
    if left.saturating_add(aw) > width || top.saturating_add(ah) > height {
        return Err(Error::invalid(format!(
            "heif: 'clap' rect ({left},{top}, {aw}x{ah}) exceeds encoded picture {width}x{height}"
        )));
    }
    Ok((left, top, aw, ah))
}

/// Apply a `clap` (clean-aperture) crop to a planar YUV `VideoFrame`.
/// Plane 0 is treated as luma at full resolution; planes ≥ 1 inherit
/// horizontal / vertical sub-sampling factors derived from the ratio
/// of plane 0's stride / row-count to the chroma plane's. The cropped
/// rectangle origin and extent must be expressible at integer chroma-
/// sample positions; if the resolved (left, top) lands on an odd
/// chroma boundary, we floor toward the nearest valid origin so the
/// resulting chroma samples remain aligned.
fn apply_clap(clap: Clap, frame: VideoFrame) -> Result<VideoFrame> {
    if frame.planes.is_empty() {
        return Err(Error::invalid("heif: 'clap' on a zero-plane frame"));
    }
    let luma_stride = frame.planes[0].stride.max(1);
    let luma_h = frame.planes[0].data.len() / luma_stride;
    if luma_h == 0 {
        return Err(Error::invalid("heif: 'clap' on an empty luma plane"));
    }
    let (left, top, aw, ah) = clap_rect(&clap, luma_stride as u32, luma_h as u32)?;
    crop_planar(frame, left, top, aw, ah)
}

/// Generic planar crop helper. `(left, top, w, h)` is in luma samples;
/// per-plane sub-sampling is inferred from each plane's stride / row
/// count vs. plane 0's. Used by [`apply_clap`] and (potentially) by
/// future overlay clipping.
fn crop_planar(frame: VideoFrame, left: u32, top: u32, w: u32, h: u32) -> Result<VideoFrame> {
    let luma_stride = frame.planes[0].stride.max(1);
    let luma_h = frame.planes[0].data.len() / luma_stride;
    let mut out_planes: Vec<VideoPlane> = Vec::with_capacity(frame.planes.len());
    for (p, plane) in frame.planes.iter().enumerate() {
        let s = plane.stride.max(1);
        let ph = plane.data.len() / s;
        let (sx, sy) = if p == 0 {
            (0u32, 0u32)
        } else {
            let sx = if luma_stride == s {
                0
            } else if luma_stride == s * 2 {
                1
            } else {
                return Err(Error::invalid(format!(
                    "heif: crop_planar: plane {p} stride {s} not 1x or 2x luma stride {luma_stride}"
                )));
            };
            let sy = if luma_h == ph {
                0
            } else if luma_h == ph * 2 {
                1
            } else {
                return Err(Error::invalid(format!(
                    "heif: crop_planar: plane {p} rows {ph} not 1x or 2x luma rows {luma_h}"
                )));
            };
            (sx, sy)
        };
        // Chroma origin = floor(luma origin / 2^shift) when sub-sampled.
        let pl_left = (left >> sx) as usize;
        let pl_top = (top >> sy) as usize;
        let pl_w = ceil_shift(w as usize, sx);
        let pl_h = ceil_shift(h as usize, sy);
        if pl_left + pl_w > s || pl_top + pl_h > ph {
            return Err(Error::invalid(format!(
                "heif: crop_planar: plane {p} crop ({pl_left},{pl_top}, {pl_w}x{pl_h}) exceeds plane {s}x{ph}"
            )));
        }
        let mut out = vec![0u8; pl_w * pl_h];
        for r in 0..pl_h {
            let src_off = (pl_top + r) * s + pl_left;
            let dst_off = r * pl_w;
            out[dst_off..dst_off + pl_w].copy_from_slice(&plane.data[src_off..src_off + pl_w]);
        }
        out_planes.push(VideoPlane {
            stride: pl_w,
            data: out,
        });
    }
    Ok(VideoFrame {
        pts: frame.pts,
        planes: out_planes,
    })
}

/// Apply an `irot` (image-rotation) transform — rotates the picture by
/// `angle * 90°` anti-clockwise. `angle == 0` is the identity. For
/// `angle == 1` (90° CCW) and `angle == 3` (270° CCW) the output's
/// width and height are swapped, which means chroma sub-sampling
/// factors that are horizontal-only (4:2:2) become ill-defined post-
/// rotation. We refuse such rotations and require either 4:2:0
/// (symmetric) or 4:4:4 (no sub-sampling).
fn apply_irot(irot: Irot, frame: VideoFrame) -> Result<VideoFrame> {
    let angle = irot.angle & 0x03;
    if angle == 0 {
        return Ok(frame);
    }
    if frame.planes.is_empty() {
        return Err(Error::invalid("heif: 'irot' on a zero-plane frame"));
    }
    let luma_stride = frame.planes[0].stride.max(1);
    let luma_h = frame.planes[0].data.len() / luma_stride;
    // Validate per-plane sub-sampling is symmetric for 90° / 270°.
    if angle == 1 || angle == 3 {
        for (p, plane) in frame.planes.iter().enumerate().skip(1) {
            let s = plane.stride.max(1);
            let ph = plane.data.len() / s;
            let sx = if luma_stride == s {
                0
            } else if luma_stride == s * 2 {
                1
            } else {
                return Err(Error::invalid(format!(
                    "heif: 'irot' plane {p} sub-sampling not 1x/2x"
                )));
            };
            let sy = if luma_h == ph {
                0
            } else if luma_h == ph * 2 {
                1
            } else {
                return Err(Error::invalid(format!(
                    "heif: 'irot' plane {p} sub-sampling not 1x/2x"
                )));
            };
            if sx != sy {
                return Err(Error::unsupported(
                    "heif: 'irot' 90/270° on non-symmetric chroma (e.g. 4:2:2) is not supported",
                ));
            }
        }
    }
    let mut out_planes: Vec<VideoPlane> = Vec::with_capacity(frame.planes.len());
    for plane in &frame.planes {
        let s = plane.stride.max(1);
        let h = plane.data.len() / s;
        let rotated = match angle {
            1 => rotate_ccw_90(plane, s, h),
            2 => rotate_180(plane, s, h),
            3 => rotate_ccw_270(plane, s, h),
            _ => unreachable!(),
        };
        out_planes.push(rotated);
    }
    Ok(VideoFrame {
        pts: frame.pts,
        planes: out_planes,
    })
}

fn rotate_ccw_90(plane: &VideoPlane, w: usize, h: usize) -> VideoPlane {
    // Anti-clockwise 90°: the right column becomes the top row.
    // Output dims: w_out = h, h_out = w.
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let src = plane.data[y * w + x];
            let dx = y;
            let dy = w - 1 - x;
            out[dy * h + dx] = src;
        }
    }
    VideoPlane {
        stride: h,
        data: out,
    }
}

fn rotate_180(plane: &VideoPlane, w: usize, h: usize) -> VideoPlane {
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let src = plane.data[y * w + x];
            out[(h - 1 - y) * w + (w - 1 - x)] = src;
        }
    }
    VideoPlane {
        stride: w,
        data: out,
    }
}

fn rotate_ccw_270(plane: &VideoPlane, w: usize, h: usize) -> VideoPlane {
    // Anti-clockwise 270° == clockwise 90°: the top row becomes the
    // right column.
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let src = plane.data[y * w + x];
            let dx = h - 1 - y;
            let dy = x;
            out[dy * h + dx] = src;
        }
    }
    VideoPlane {
        stride: h,
        data: out,
    }
}

/// Apply an `imir` (image-mirror) transform — `axis == 0` mirrors
/// about the vertical axis (flips columns left↔right), `axis == 1`
/// mirrors about the horizontal axis (flips rows top↔bottom). Per
/// ISO/IEC 23008-12 §6.5.12.3.
fn apply_imir(imir: Imir, frame: VideoFrame) -> Result<VideoFrame> {
    if frame.planes.is_empty() {
        return Err(Error::invalid("heif: 'imir' on a zero-plane frame"));
    }
    let mut out_planes: Vec<VideoPlane> = Vec::with_capacity(frame.planes.len());
    for plane in &frame.planes {
        let s = plane.stride.max(1);
        let h = plane.data.len() / s;
        let mut out = vec![0u8; s * h];
        if imir.axis == 0 {
            // Flip horizontally (about the vertical axis): every row
            // is reversed in place.
            for y in 0..h {
                for x in 0..s {
                    out[y * s + (s - 1 - x)] = plane.data[y * s + x];
                }
            }
        } else {
            // Flip vertically (about the horizontal axis): row order
            // is reversed; each row is copied verbatim.
            for y in 0..h {
                let src_row = &plane.data[y * s..y * s + s];
                let dst = &mut out[(h - 1 - y) * s..(h - 1 - y) * s + s];
                dst.copy_from_slice(src_row);
            }
        }
        out_planes.push(VideoPlane {
            stride: s,
            data: out,
        });
    }
    Ok(VideoFrame {
        pts: frame.pts,
        planes: out_planes,
    })
}

/// Decode a `grid`-derived primary item: parse the `ImageGrid` payload,
/// resolve `dimg` references in row-major order, decode each tile via
/// [`decode_hvc_item`], and tile them into a single planar `VideoFrame`
/// of the declared output dimensions.
///
/// All tiles must agree on plane count + per-plane stride (the HEIF
/// spec mandates uniform tile sizing per §6.6.2.3.2). Plane data is
/// pasted with stride = `output_width >> sub_x_for_plane(p)`; the
/// caller (or downstream pixfmt path) is responsible for treating the
/// composited buffer as the same chroma format as the tiles.
fn decode_grid_primary(hdr: &HeifHeader<'_>, grid_id: u32) -> Result<VideoFrame> {
    let grid_bytes = item_bytes_in(hdr.file, &hdr.meta, grid_id)?;
    let grid = ImageGrid::parse(grid_bytes)?;
    let tile_ids = hdr.meta.iref_targets(b"dimg", grid_id);
    if tile_ids.is_empty() {
        return Err(Error::invalid(format!(
            "heif: grid item {grid_id} has no 'dimg' references"
        )));
    }
    if tile_ids.len() != grid.expected_tile_count() {
        return Err(Error::invalid(format!(
            "heif: grid declares {}×{} = {} tiles but 'dimg' lists {}",
            grid.rows,
            grid.columns,
            grid.expected_tile_count(),
            tile_ids.len()
        )));
    }
    let mut tiles = Vec::with_capacity(tile_ids.len());
    for (i, tid) in tile_ids.iter().enumerate() {
        let info = hdr
            .meta
            .item_by_id(*tid)
            .ok_or_else(|| Error::invalid(format!("heif: grid tile {i} id {tid} unknown")))?;
        if info.item_type != ITEM_TYPE_HVC1 && info.item_type != ITEM_TYPE_HEV1 {
            return Err(Error::unsupported(format!(
                "heif: grid tile {i} item_type '{}' is not 'hvc1' / 'hev1'",
                type_str(&info.item_type)
            )));
        }
        let tile = decode_hvc_item(hdr, *tid)?;
        tiles.push(tile);
    }
    composite_grid_frames(&grid, &tiles)
}

/// Tile-paste a row-major sequence of HEVC-decoded tile frames into a
/// single output frame of `(grid.output_width, grid.output_height)`.
/// Tiles must share the same plane count + per-plane stride; the
/// per-plane horizontal/vertical sub-sampling is inferred from the
/// ratio of `tiles[0].planes[p].stride` to `tiles[0].planes[0].stride`
/// — i.e. plane 0 is luma, planes 1.. share whatever sub-sampling the
/// source HEVC sequence used (4:2:0 → stride/2, 4:2:2 → stride/2,
/// 4:4:4 → stride). Trailing rows / columns that spill past the
/// declared output rectangle are clipped at the canvas edge per §6.6.2.3.2.
fn composite_grid_frames(grid: &ImageGrid, tiles: &[VideoFrame]) -> Result<VideoFrame> {
    if tiles.is_empty() {
        return Err(Error::invalid("heif: grid composite called with no tiles"));
    }
    let out_w = grid.output_width as usize;
    let out_h = grid.output_height as usize;
    if out_w == 0 || out_h == 0 {
        return Err(Error::invalid(format!(
            "heif: grid output dims {}x{} contain a zero",
            out_w, out_h
        )));
    }
    let n_planes = tiles[0].planes.len();
    if n_planes == 0 {
        return Err(Error::invalid("heif: grid tile 0 has no planes"));
    }
    // Per-plane (sub_x_shift, sub_y_shift) derived from tile 0. Plane 0
    // is always luma; for planes ≥ 1 we look at stride and at row count
    // (data.len() / stride) versus plane 0's, taking ceiling shifts.
    let luma = &tiles[0].planes[0];
    let luma_stride = luma.stride.max(1);
    let luma_h = luma.data.len() / luma_stride;
    if luma_h == 0 {
        return Err(Error::invalid("heif: grid tile 0 luma plane is empty"));
    }
    let mut shifts: Vec<(u32, u32)> = Vec::with_capacity(n_planes);
    for p in 0..n_planes {
        if p == 0 {
            shifts.push((0, 0));
            continue;
        }
        let pl = &tiles[0].planes[p];
        let s = pl.stride.max(1);
        let h = pl.data.len() / s;
        // Common HEVC chroma sub-sampling shifts (1 -> half-resolution).
        let sx = match (luma_stride, s) {
            (a, b) if a == b => 0,
            (a, b) if a == b * 2 => 1,
            _ => 0,
        };
        let sy = match (luma_h, h) {
            (a, b) if a == b => 0,
            (a, b) if a == b * 2 => 1,
            _ => 0,
        };
        shifts.push((sx, sy));
    }
    // Validate uniform tile shape across the corpus.
    for (i, tile) in tiles.iter().enumerate().skip(1) {
        if tile.planes.len() != n_planes {
            return Err(Error::invalid(format!(
                "heif: grid tile {i} has {} planes != tile 0 has {}",
                tile.planes.len(),
                n_planes
            )));
        }
        for (p, plane) in tile.planes.iter().enumerate() {
            let want = &tiles[0].planes[p];
            if plane.stride != want.stride {
                return Err(Error::invalid(format!(
                    "heif: grid tile {i} plane {p} stride {} != tile 0 stride {}",
                    plane.stride, want.stride
                )));
            }
        }
    }
    // Build per-plane output buffers at the canvas size.
    let mut out_planes: Vec<VideoPlane> = Vec::with_capacity(n_planes);
    for &(sx, sy) in shifts.iter().take(n_planes) {
        let pw = ceil_shift(out_w, sx);
        let ph = ceil_shift(out_h, sy);
        out_planes.push(VideoPlane {
            stride: pw,
            data: vec![0u8; pw * ph],
        });
    }
    let cols = grid.columns as usize;
    let tile_lw = luma_stride; // assume luma plane is tightly strided (HEVC decoder emits stride==width on cropped output)
    let tile_lh = luma_h;
    for (i, tile) in tiles.iter().enumerate() {
        let row = i / cols;
        let col = i % cols;
        let dst_x = col * tile_lw;
        let dst_y = row * tile_lh;
        if dst_x >= out_w || dst_y >= out_h {
            continue;
        }
        let copy_w = (out_w - dst_x).min(tile_lw);
        let copy_h = (out_h - dst_y).min(tile_lh);
        for (p, plane) in tile.planes.iter().enumerate() {
            let (sx, sy) = shifts[p];
            let plane_dst_x = dst_x >> sx;
            let plane_dst_y = dst_y >> sy;
            let plane_copy_w = ceil_shift(copy_w, sx);
            let plane_copy_h = ceil_shift(copy_h, sy);
            let src_stride = plane.stride.max(1);
            let src_h = plane.data.len() / src_stride;
            let plane_copy_w = plane_copy_w.min(src_stride);
            let plane_copy_h = plane_copy_h.min(src_h);
            let dst = &mut out_planes[p];
            let dst_stride = dst.stride;
            let dst_h = dst.data.len() / dst_stride.max(1);
            let plane_copy_w = plane_copy_w.min(dst_stride.saturating_sub(plane_dst_x));
            let plane_copy_h = plane_copy_h.min(dst_h.saturating_sub(plane_dst_y));
            for r in 0..plane_copy_h {
                let dst_off = (plane_dst_y + r) * dst_stride + plane_dst_x;
                let src_off = r * src_stride;
                dst.data[dst_off..dst_off + plane_copy_w]
                    .copy_from_slice(&plane.data[src_off..src_off + plane_copy_w]);
            }
        }
    }
    Ok(VideoFrame {
        pts: tiles[0].pts,
        planes: out_planes,
    })
}

/// Decode an `iovl`-derived primary item — ISO/IEC 23008-12 §6.6.2.3.3.
///
/// The overlay descriptor lives in the item payload (parsed as
/// [`ImageOverlay`]); the constituent images appear as `dimg`
/// references *from* the overlay item, in back-to-front layer order.
/// Each constituent is decoded via [`decode_hvc_item`] (so it inherits
/// any of its own transformative-property chain — clap / irot / imir),
/// then pasted onto a canvas of the declared `(output_width,
/// output_height)` at its `(h_offset, v_offset)`. The canvas is filled
/// with [`ImageOverlay::canvas_fill_rgba`] converted to YUV via the
/// matrix selected from the iovl item's `colr nclx` property (or
/// BT.601 limited-range when no `colr` is associated — matching the
/// corpus convention used by the bit-exact compare in
/// `tests/heif_corpus.rs`, which itself follows the ISO/IEC 23008-12
/// §6.5.5 default for HEIC images without an explicit colour profile).
fn decode_iovl_primary(hdr: &HeifHeader<'_>, iovl_id: u32) -> Result<VideoFrame> {
    let layer_ids = hdr.meta.iref_targets(b"dimg", iovl_id);
    if layer_ids.is_empty() {
        return Err(Error::invalid(format!(
            "heif: iovl item {iovl_id} has no 'dimg' references"
        )));
    }
    let iovl_bytes = item_bytes_in(hdr.file, &hdr.meta, iovl_id)?;
    let overlay = ImageOverlay::parse(iovl_bytes, layer_ids.len())?;
    let mut layers: Vec<VideoFrame> = Vec::with_capacity(layer_ids.len());
    // Task #346 — alongside each colour layer, decode its alpha
    // auxiliary item (per-layer auxl iref) when present. The iovl
    // descriptor doesn't carry alpha; per-layer alpha is the only
    // way to express RGBA constituents in HEIF (ISO/IEC 23008-12
    // §6.6.2.1 / §6.6.2.3.3). Without per-layer alpha blending the
    // composer just paints the layer's full bounding rectangle
    // opaque — which produces yellow squares where the encoder
    // intended yellow circles, etc.
    let mut layer_alphas: Vec<Option<VideoFrame>> = Vec::with_capacity(layer_ids.len());
    for (i, lid) in layer_ids.iter().enumerate() {
        let info = hdr
            .meta
            .item_by_id(*lid)
            .ok_or_else(|| Error::invalid(format!("heif: iovl layer {i} id {lid} unknown")))?;
        if info.item_type != ITEM_TYPE_HVC1 && info.item_type != ITEM_TYPE_HEV1 {
            return Err(Error::unsupported(format!(
                "heif: iovl layer {i} item_type '{}' is not 'hvc1' / 'hev1'",
                type_str(&info.item_type)
            )));
        }
        let layer = decode_hvc_item(hdr, *lid)?;
        layers.push(layer);
        let alpha = match find_alpha_item_id(&hdr.meta, *lid) {
            Some(aid) => match decode_hvc_item(hdr, aid) {
                Ok(vf) => Some(vf),
                Err(e) => {
                    return Err(Error::invalid(format!(
                        "heif: iovl layer {i} (id {lid}) alpha aux {aid} decode failed: {e}"
                    )));
                }
            },
            None => None,
        };
        layer_alphas.push(alpha);
    }
    // Task #346 — iovl colour-matrix selection. The canvas-fill
    // conversion (descriptor RGBA → per-plane YUV fill bytes) MUST
    // match the colour-space the constituent layers were encoded in,
    // otherwise the fill regions and the layer regions of the canvas
    // disagree on what `(Y,U,V)` means (limited-range BT.601 vs
    // full-range BT.601 differ by ~16/255 ≈ 6% in luma, enough to
    // shift `Y=1,U=128,V=128` from `(1,1,1)` to `(0,0,0)`).
    //
    // ISO/IEC 23008-12 §6.5.5 makes a HEIF image's colour
    // interpretation pull from its `colr` property; for derived
    // items (`iovl`, `grid`) the spec is ambiguous about whether
    // missing colr should inherit from constituents. Real-world
    // encoders (libheif / ImageMagick) tag layers with colr but
    // leave the iovl bare. We follow the constituents — fall back
    // to the first layer's `colr nclx` when the iovl item has no
    // colr of its own. This matches the corpus convention (the
    // oracle PNG is rendered from the layer YUV samples).
    let layer0_colr =
        layer_ids
            .first()
            .and_then(|lid| match hdr.meta.property_for(*lid, b"colr") {
                Some(Property::Colr(c)) => Some(c.clone()),
                _ => None,
            });
    let iovl_colr = match hdr.meta.property_for(iovl_id, b"colr") {
        Some(Property::Colr(c)) => Some(c.clone()),
        _ => layer0_colr,
    };
    let fill_matrix = FillMatrix::from_colr(iovl_colr.as_ref());
    compose_overlay_frames_with_matrix_and_alphas(&overlay, &layers, &layer_alphas, fill_matrix)
}

/// Backwards-compatible shim for [`compose_overlay_frames_with_matrix`]
/// that hard-codes BT.709 limited-range fill conversion. Retained so
/// the older unit tests keep building unchanged; the iovl pipeline now
/// flows through `compose_overlay_frames_with_matrix` so the canvas-
/// fill matrix can track the iovl item's `colr nclx` property.
#[cfg(test)]
fn compose_overlay_frames(overlay: &ImageOverlay, layers: &[VideoFrame]) -> Result<VideoFrame> {
    compose_overlay_frames_with_matrix(overlay, layers, FillMatrix::Bt709Limited)
}

/// Composite a back-to-front sequence of HEVC-decoded layers onto a
/// canvas of `(overlay.output_width, overlay.output_height)` at the
/// per-layer `(h_offset, v_offset)` declared in the overlay
/// descriptor. Layers must agree on plane count and per-plane chroma
/// sub-sampling factors (those are inferred from layer 0). Pixels
/// outside the canvas after offsetting are clipped; areas not covered
/// by any layer remain at the converted fill colour.
///
/// The fill is RGB-to-YUV converted via `matrix` from the descriptor's
/// 16-bit RGBA (alpha channel ignored — overlay composition here is
/// straight paste, not alpha-blend; alpha auxiliary handling is the
/// auxC / auxl path covered separately). Pick the matrix that matches
/// what the consumer of the resulting YUV frame will assume — see
/// [`FillMatrix::from_colr`] for the iovl convention used by
/// [`decode_iovl_primary`].
fn compose_overlay_frames_with_matrix(
    overlay: &ImageOverlay,
    layers: &[VideoFrame],
    matrix: FillMatrix,
) -> Result<VideoFrame> {
    if layers.is_empty() {
        return Err(Error::invalid("heif: iovl composite called with no layers"));
    }
    let out_w = overlay.output_width as usize;
    let out_h = overlay.output_height as usize;
    if out_w == 0 || out_h == 0 {
        return Err(Error::invalid(format!(
            "heif: iovl output dims {}x{} contain a zero",
            out_w, out_h
        )));
    }
    let n_planes = layers[0].planes.len();
    if n_planes == 0 {
        return Err(Error::invalid("heif: iovl layer 0 has no planes"));
    }
    // Per-plane (sub_x_shift, sub_y_shift) — same derivation as the
    // grid composer; plane 0 is luma.
    let luma_stride = layers[0].planes[0].stride.max(1);
    let luma_h = layers[0].planes[0].data.len() / luma_stride;
    if luma_h == 0 {
        return Err(Error::invalid("heif: iovl layer 0 luma plane is empty"));
    }
    let mut shifts: Vec<(u32, u32)> = Vec::with_capacity(n_planes);
    for p in 0..n_planes {
        if p == 0 {
            shifts.push((0, 0));
            continue;
        }
        let pl = &layers[0].planes[p];
        let s = pl.stride.max(1);
        let h = pl.data.len() / s;
        let sx = if luma_stride == s {
            0
        } else if luma_stride == s * 2 {
            1
        } else {
            return Err(Error::invalid(format!(
                "heif: iovl plane {p} sub_x not 1x/2x luma"
            )));
        };
        let sy = if luma_h == h {
            0
        } else if luma_h == h * 2 {
            1
        } else {
            return Err(Error::invalid(format!(
                "heif: iovl plane {p} sub_y not 1x/2x luma"
            )));
        };
        shifts.push((sx, sy));
    }
    // Validate uniform plane shape across all layers.
    for (i, layer) in layers.iter().enumerate().skip(1) {
        if layer.planes.len() != n_planes {
            return Err(Error::invalid(format!(
                "heif: iovl layer {i} has {} planes != layer 0 has {}",
                layer.planes.len(),
                n_planes
            )));
        }
    }
    // Convert the canvas RGBA fill to per-plane fill bytes. The
    // descriptor's RGBA components are 16-bit; we down-shift to 8-bit
    // for the YUV packing path below (the HEVC decoder also emits
    // 8-bit planes for the fixtures in this round). Higher bit depths
    // are deferred — they require widening every plane to u16.
    let r8 = (overlay.canvas_fill_rgba[0] >> 8) as u8;
    let g8 = (overlay.canvas_fill_rgba[1] >> 8) as u8;
    let b8 = (overlay.canvas_fill_rgba[2] >> 8) as u8;
    let (y_fill, u_fill, v_fill) = matrix.rgb_to_yuv(r8, g8, b8);
    // Build per-plane output buffers at the canvas size, pre-filled
    // with the fill bytes. Plane 0 = Y, plane 1 = U/Cb, plane 2 = V/Cr
    // (the only layouts produced by the HEVC decoder this round). For
    // single-plane layouts (monochrome) only Y_fill is used.
    let mut out_planes: Vec<VideoPlane> = Vec::with_capacity(n_planes);
    for (p, &(sx, sy)) in shifts.iter().enumerate().take(n_planes) {
        let pw = ceil_shift(out_w, sx);
        let ph = ceil_shift(out_h, sy);
        let fill = match p {
            0 => y_fill,
            1 => u_fill,
            2 => v_fill,
            _ => 0,
        };
        out_planes.push(VideoPlane {
            stride: pw,
            data: vec![fill; pw * ph],
        });
    }
    // Paste each layer in declared order (back-to-front).
    for (i, layer) in layers.iter().enumerate() {
        let (h_off, v_off) = overlay.offsets[i];
        for (p, plane) in layer.planes.iter().enumerate() {
            let (sx, sy) = shifts[p];
            let src_stride = plane.stride.max(1);
            let src_h = plane.data.len() / src_stride;
            // Clip the source rectangle to the part that lands inside
            // the canvas after offsetting. All math is in luma samples
            // first, then divided by the per-plane sub-sampling.
            let layer_w_luma = src_stride << sx; // luma-equivalent width
            let layer_h_luma = src_h << sy;
            let dst_left_luma = h_off; // can be negative
            let dst_top_luma = v_off;
            // Compute the visible rectangle in luma coords.
            let visible_left = dst_left_luma.max(0);
            let visible_top = dst_top_luma.max(0);
            let visible_right = (dst_left_luma + layer_w_luma as i32).min(out_w as i32);
            let visible_bottom = (dst_top_luma + layer_h_luma as i32).min(out_h as i32);
            if visible_right <= visible_left || visible_bottom <= visible_top {
                continue; // layer is entirely outside the canvas
            }
            // Convert the visible rectangle into per-plane coords.
            // For sub-sampled planes we round the destination down to
            // the nearest sub-sample boundary so chroma samples line
            // up with their luma neighbourhood.
            let pl_dst_left = (visible_left as usize) >> sx;
            let pl_dst_top = (visible_top as usize) >> sy;
            let pl_dst_w = ((visible_right as usize) >> sx).saturating_sub(pl_dst_left);
            let pl_dst_h = ((visible_bottom as usize) >> sy).saturating_sub(pl_dst_top);
            if pl_dst_w == 0 || pl_dst_h == 0 {
                continue;
            }
            // Source-side offset = how far into the layer the visible
            // rectangle starts. If the layer's top-left is left/above
            // the canvas (negative offset), the source skips that
            // many pixels.
            let pl_src_left = (visible_left - dst_left_luma) as usize >> sx;
            let pl_src_top = (visible_top - dst_top_luma) as usize >> sy;
            let dst = &mut out_planes[p];
            let dst_stride = dst.stride;
            for r in 0..pl_dst_h {
                let src_off = (pl_src_top + r) * src_stride + pl_src_left;
                let dst_off = (pl_dst_top + r) * dst_stride + pl_dst_left;
                dst.data[dst_off..dst_off + pl_dst_w]
                    .copy_from_slice(&plane.data[src_off..src_off + pl_dst_w]);
            }
        }
    }
    Ok(VideoFrame {
        pts: layers[0].pts,
        planes: out_planes,
    })
}

/// Task #346 alpha-aware iovl composer. Falls back to
/// [`compose_overlay_frames_with_matrix`] (straight paste) when no
/// layer carries a per-layer alpha auxiliary; otherwise composites
/// each layer using the per-pixel alpha from the layer's auxl
/// (Gray8 / Gray16Le, 0 = transparent ↔ background, 255 = opaque ↔
/// layer) **in RGB space**, not YUV space.
///
/// YUV-domain alpha blending is wrong: chroma channels (U, V) carry
/// signed colour-difference values centred at 128, so a linear
/// `α·src + (1-α)·dst` of two YUV samples doesn't match the
/// `α·src_rgb + (1-α)·dst_rgb` you'd get if you converted the
/// samples to RGB first, blended, and converted back. The error
/// shows up as a colour shift at every anti-aliased alpha edge
/// (the binary 0/255 plateau is fine — only the smooth-edge pixels
/// are bitten).
///
/// Output bit depth follows layer 0's bit depth (we only support
/// 8-bit colour planes here — 10/12-bit iovl is deferred to a later
/// task once a higher-bit-depth iovl fixture lands).
fn compose_overlay_frames_with_matrix_and_alphas(
    overlay: &ImageOverlay,
    layers: &[VideoFrame],
    layer_alphas: &[Option<VideoFrame>],
    matrix: FillMatrix,
) -> Result<VideoFrame> {
    if layer_alphas.iter().all(|a| a.is_none()) {
        return compose_overlay_frames_with_matrix(overlay, layers, matrix);
    }
    if layers.is_empty() {
        return Err(Error::invalid("heif: iovl composite called with no layers"));
    }
    let out_w = overlay.output_width as usize;
    let out_h = overlay.output_height as usize;
    if out_w == 0 || out_h == 0 {
        return Err(Error::invalid(format!(
            "heif: iovl output dims {}x{} contain a zero",
            out_w, out_h
        )));
    }
    let n_planes = layers[0].planes.len();
    if n_planes == 0 {
        return Err(Error::invalid("heif: iovl layer 0 has no planes"));
    }
    let luma_stride = layers[0].planes[0].stride.max(1);
    let luma_h = layers[0].planes[0].data.len() / luma_stride;
    if luma_h == 0 {
        return Err(Error::invalid("heif: iovl layer 0 luma plane is empty"));
    }
    // Per-plane sub-sampling derived from layer 0 (same convention as
    // [`compose_overlay_frames_with_matrix`]).
    let mut shifts: Vec<(u32, u32)> = Vec::with_capacity(n_planes);
    for p in 0..n_planes {
        if p == 0 {
            shifts.push((0, 0));
            continue;
        }
        let pl = &layers[0].planes[p];
        let s = pl.stride.max(1);
        let h = pl.data.len() / s;
        let sx = if luma_stride == s {
            0
        } else if luma_stride == s * 2 {
            1
        } else {
            return Err(Error::invalid(format!(
                "heif: iovl plane {p} sub_x not 1x/2x luma"
            )));
        };
        let sy = if luma_h == h {
            0
        } else if luma_h == h * 2 {
            1
        } else {
            return Err(Error::invalid(format!(
                "heif: iovl plane {p} sub_y not 1x/2x luma"
            )));
        };
        shifts.push((sx, sy));
    }
    for (i, layer) in layers.iter().enumerate().skip(1) {
        if layer.planes.len() != n_planes {
            return Err(Error::invalid(format!(
                "heif: iovl layer {i} has {} planes != layer 0 has {}",
                layer.planes.len(),
                n_planes
            )));
        }
    }
    // RGB-domain compositing path: maintain a packed RGB canvas
    // sized at the iovl output dimensions, paint the canvas-fill
    // colour into it, blend each layer onto it (per-pixel α from
    // the layer's alpha aux), then convert the final RGB back to
    // YUV at the layer's chroma sub-sampling.
    let r_fill = (overlay.canvas_fill_rgba[0] >> 8) as u8;
    let g_fill = (overlay.canvas_fill_rgba[1] >> 8) as u8;
    let b_fill = (overlay.canvas_fill_rgba[2] >> 8) as u8;
    let mut canvas_rgb = vec![0u8; out_w * out_h * 3];
    for px in 0..(out_w * out_h) {
        canvas_rgb[px * 3] = r_fill;
        canvas_rgb[px * 3 + 1] = g_fill;
        canvas_rgb[px * 3 + 2] = b_fill;
    }
    for (i, layer) in layers.iter().enumerate() {
        let (h_off, v_off) = overlay.offsets[i];
        let layer_luma_stride = layer.planes[0].stride.max(1);
        let layer_luma_h = layer.planes[0].data.len() / layer_luma_stride;
        let layer_w = layer_luma_stride;
        let layer_h = layer_luma_h;
        // Resolve alpha — must be at luma resolution if present.
        let alpha_plane = if let Some(ap) = layer_alphas[i].as_ref() {
            if ap.planes.len() != 1 {
                return Err(Error::invalid(format!(
                    "heif: iovl layer {i} alpha aux must be 1-plane (got {})",
                    ap.planes.len()
                )));
            }
            let a_stride = ap.planes[0].stride.max(1);
            let a_rows = ap.planes[0].data.len() / a_stride;
            if a_stride != layer_w || a_rows != layer_h {
                return Err(Error::invalid(format!(
                    "heif: iovl layer {i} alpha aux dims {a_stride}x{a_rows} != \
                     luma dims {layer_w}x{layer_h}"
                )));
            }
            Some(&ap.planes[0])
        } else {
            None
        };
        // Decide sub-sampling for the colour planes (only support
        // 1-plane luma OR 3-plane planar YUV).
        let sub_x = if n_planes == 1 {
            1
        } else {
            (1usize << shifts[1].0).max(1)
        };
        let sub_y = if n_planes == 1 {
            1
        } else {
            (1usize << shifts[1].1).max(1)
        };
        let cb_stride = if n_planes >= 2 {
            layer.planes[1].stride.max(1)
        } else {
            1
        };
        // Visible window in canvas luma coords.
        let v_left = h_off.max(0);
        let v_top = v_off.max(0);
        let v_right = (h_off + layer_w as i32).min(out_w as i32);
        let v_bottom = (v_off + layer_h as i32).min(out_h as i32);
        if v_right <= v_left || v_bottom <= v_top {
            continue;
        }
        for y in v_top..v_bottom {
            let layer_y = (y - v_off) as usize;
            for x in v_left..v_right {
                let layer_x = (x - h_off) as usize;
                // Resolve layer (Y, U, V) → RGB at luma resolution.
                let yv = layer.planes[0].data[layer_y * layer_luma_stride + layer_x];
                let (r_src, g_src, b_src) = if n_planes == 1 {
                    // Monochrome layer — splat luma.
                    (yv, yv, yv)
                } else {
                    let cy = layer_y / sub_y;
                    let cx = layer_x / sub_x;
                    let uv = layer.planes[1].data[cy * cb_stride + cx];
                    let vrv = layer.planes[2].data[cy * cb_stride + cx];
                    matrix.yuv_to_rgb(yv, uv, vrv)
                };
                let alpha = match alpha_plane {
                    Some(ap) => ap.data[layer_y * layer_luma_stride + layer_x] as u32,
                    None => 255u32,
                };
                let canvas_off = ((y as usize) * out_w + (x as usize)) * 3;
                let r_dst = canvas_rgb[canvas_off] as u32;
                let g_dst = canvas_rgb[canvas_off + 1] as u32;
                let b_dst = canvas_rgb[canvas_off + 2] as u32;
                let r_src = r_src as u32;
                let g_src = g_src as u32;
                let b_src = b_src as u32;
                let blend = |s: u32, d: u32| ((s * alpha + d * (255 - alpha) + 127) / 255) as u8;
                canvas_rgb[canvas_off] = blend(r_src, r_dst);
                canvas_rgb[canvas_off + 1] = blend(g_src, g_dst);
                canvas_rgb[canvas_off + 2] = blend(b_src, b_dst);
            }
        }
    }
    // Convert canvas RGB → YUV at the output sub-sampling. For the
    // alpha-blended path we deliberately emit **4:4:4** chroma even
    // when the layers are 4:2:0 — sub-sampling the canvas RGB to
    // 4:2:0 would lose chroma precision at the alpha edges (where
    // adjacent pixels alpha-blend to wildly different colours, but
    // 2×2 RGB averaging then YUV conversion smears them together).
    // The downstream comparator infers sub_x/sub_y from the chroma
    // stride and handles 4:4:4 transparently. Monochrome (n_planes
    // == 1) keeps a single luma plane.
    let n_out_planes = if n_planes == 1 { 1 } else { 3 };
    let mut out_planes: Vec<VideoPlane> = Vec::with_capacity(n_out_planes);
    for _ in 0..n_out_planes {
        out_planes.push(VideoPlane {
            stride: out_w,
            data: vec![0u8; out_w * out_h],
        });
    }
    if n_out_planes == 1 {
        for y in 0..out_h {
            for x in 0..out_w {
                let off = (y * out_w + x) * 3;
                let (yv, _, _) =
                    matrix.rgb_to_yuv(canvas_rgb[off], canvas_rgb[off + 1], canvas_rgb[off + 2]);
                out_planes[0].data[y * out_w + x] = yv;
            }
        }
    } else {
        for y in 0..out_h {
            for x in 0..out_w {
                let off = (y * out_w + x) * 3;
                let (yv, u, v) =
                    matrix.rgb_to_yuv(canvas_rgb[off], canvas_rgb[off + 1], canvas_rgb[off + 2]);
                out_planes[0].data[y * out_w + x] = yv;
                out_planes[1].data[y * out_w + x] = u;
                out_planes[2].data[y * out_w + x] = v;
            }
        }
    }
    Ok(VideoFrame {
        pts: layers[0].pts,
        planes: out_planes,
    })
}

/// Selector for the RGB-to-YUV matrix used to convert the iovl canvas
/// fill into per-plane fill bytes. Mirrored from
/// `tests/heif_corpus.rs::Matrix` — kept private to this module since
/// only the iovl pipeline currently needs it. Promote to a public type
/// once additional callers (e.g. monochrome / Main 10) need a
/// matrix-aware fill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FillMatrix {
    /// BT.601 limited-range — the iovl default when no `colr nclx` is
    /// associated with the iovl item, matching the corpus convention
    /// used by the bit-exact compare (which itself follows the
    /// ISO/IEC 23008-12 §6.5.5 unspecified-`colr` default and the
    /// libwebp / libheif convention shared with oxideav-webp #253).
    Bt601Limited,
    Bt601Full,
    Bt709Limited,
    Bt709Full,
}

impl FillMatrix {
    /// Resolve the canvas-fill matrix from the iovl item's `colr`.
    /// Honours `nclx` `matrix_coefficients` (1 → BT.709, anything else
    /// → BT.601) and `full_range_flag`. Falls back to BT.601 **full**
    /// when `colr` is `None` or carries an ICC profile only — this
    /// matches the convention modern HEIC encoders (libheif /
    /// ImageMagick) use when they don't write a colr (full-range YUV
    /// is the dominant profile for new content; ISO/IEC 23008-12
    /// §6.5.5 leaves the missing-colr case implementation-defined,
    /// so any consistent choice works as long as it matches what the
    /// content was encoded against).
    fn from_colr(colr: Option<&Colr>) -> Self {
        match colr {
            Some(Colr::Nclx {
                matrix_coefficients,
                full_range,
                ..
            }) => match (*matrix_coefficients, *full_range) {
                (1, true) => FillMatrix::Bt709Full,
                (1, false) => FillMatrix::Bt709Limited,
                (_, true) => FillMatrix::Bt601Full,
                (_, false) => FillMatrix::Bt601Limited,
            },
            _ => FillMatrix::Bt601Full,
        }
    }

    /// Convert a single 8-bit-per-channel RGB sample to (Y, U/Cb, V/Cr)
    /// per the selected matrix. Limited-range outputs land in the
    /// studio quantisation Y' ∈ [16,235], C ∈ [16,240]; full-range
    /// outputs use the [0,255] mapping.
    #[inline]
    fn rgb_to_yuv(self, r: u8, g: u8, b: u8) -> (u8, u8, u8) {
        match self {
            FillMatrix::Bt709Limited => bt709_limited_rgb_to_yuv(r, g, b),
            FillMatrix::Bt709Full => rgb_to_yuv_with(r, g, b, 0.2126, 0.0722, false),
            FillMatrix::Bt601Limited => rgb_to_yuv_with(r, g, b, 0.299, 0.114, true),
            FillMatrix::Bt601Full => rgb_to_yuv_with(r, g, b, 0.299, 0.114, false),
        }
    }

    /// Inverse of [`Self::rgb_to_yuv`] — convert (Y, U/Cb, V/Cr) back
    /// to 8-bit RGB. Used by the alpha-aware iovl composer (task #346)
    /// to lift each layer's YUV samples into RGB so alpha-blending
    /// happens in a colour space where linear interpolation is
    /// chromatically correct.
    #[inline]
    fn yuv_to_rgb(self, y: u8, u: u8, v: u8) -> (u8, u8, u8) {
        let (kr, kb, limited) = match self {
            FillMatrix::Bt601Limited => (0.299_f32, 0.114_f32, true),
            FillMatrix::Bt601Full => (0.299_f32, 0.114_f32, false),
            FillMatrix::Bt709Limited => (0.2126_f32, 0.0722_f32, true),
            FillMatrix::Bt709Full => (0.2126_f32, 0.0722_f32, false),
        };
        let kg = 1.0 - kr - kb;
        let (yv, cb, cr) = if limited {
            let yv = (y as f32 - 16.0) * (255.0 / 219.0);
            let cb = (u as f32 - 128.0) * (255.0 / 224.0);
            let cr = (v as f32 - 128.0) * (255.0 / 224.0);
            (yv, cb, cr)
        } else {
            let yv = y as f32;
            let cb = u as f32 - 128.0;
            let cr = v as f32 - 128.0;
            (yv, cb, cr)
        };
        let r = yv + 2.0 * (1.0 - kr) * cr;
        let b = yv + 2.0 * (1.0 - kb) * cb;
        let g = yv - (2.0 * kr * (1.0 - kr) / kg) * cr - (2.0 * kb * (1.0 - kb) / kg) * cb;
        (clamp_u8(r), clamp_u8(g), clamp_u8(b))
    }
}

/// Generic RGB→YUV with explicit `kr` / `kb` luma weights and a
/// `limited` flag selecting between studio quantisation and full-range
/// quantisation. Mirrors the per-pixel formula in
/// `tests/heif_corpus.rs::yuv_to_rgb` (inverted), so the iovl canvas-
/// fill path round-trips through the bit-exact compare without LSB
/// drift on grey / coloured fill regions.
#[inline]
fn rgb_to_yuv_with(r: u8, g: u8, b: u8, kr: f32, kb: f32, limited: bool) -> (u8, u8, u8) {
    let kg = 1.0 - kr - kb;
    let r = r as f32;
    let g = g as f32;
    let b = b as f32;
    let y_lin = kr * r + kg * g + kb * b;
    let cb_lin = (b - y_lin) / (2.0 * (1.0 - kb));
    let cr_lin = (r - y_lin) / (2.0 * (1.0 - kr));
    let (y, cb, cr) = if limited {
        // [0,255] full → [16,235] luma, [16,240] chroma.
        let y = 16.0 + y_lin * (219.0 / 255.0);
        let cb = 128.0 + cb_lin * (224.0 / 255.0);
        let cr = 128.0 + cr_lin * (224.0 / 255.0);
        (y, cb, cr)
    } else {
        (y_lin, 128.0 + cb_lin, 128.0 + cr_lin)
    };
    (clamp_u8(y), clamp_u8(cb), clamp_u8(cr))
}

/// BT.709 limited-range RGB-to-YUV (8-bit input, 8-bit Y/U/V output).
/// Used by [`compose_overlay_frames`] to render the iovl canvas-fill
/// RGBA into Y/Cb/Cr fill bytes that match what the HEVC layers expect.
/// Coefficients per Rec. ITU-R BT.709 §3.5 (with the studio-range
/// quantisation Y' ∈ [16,235], C ∈ [16,240]).
#[inline]
fn bt709_limited_rgb_to_yuv(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as f32;
    let g = g as f32;
    let b = b as f32;
    let y = 16.0 + (0.18259 * r) + (0.61423 * g) + (0.06201 * b);
    let cb = 128.0 + (-0.10064 * r) + (-0.33857 * g) + (0.43922 * b);
    let cr = 128.0 + (0.43922 * r) + (-0.39895 * g) + (-0.04027 * b);
    (clamp_u8(y), clamp_u8(cb), clamp_u8(cr))
}

#[inline]
fn clamp_u8(v: f32) -> u8 {
    let r = v.round();
    if r < 0.0 {
        0
    } else if r > 255.0 {
        255
    } else {
        r as u8
    }
}

#[inline]
fn ceil_shift(v: usize, s: u32) -> usize {
    if s == 0 {
        v
    } else {
        let unit = 1usize << s;
        (v + unit - 1) >> s
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
    fn image_grid_parses_16bit() {
        // version=0 flags=0 rows-1=1 cols-1=1 → 2x2 grid; output 256x128.
        let buf = [0u8, 0, 1, 1, 0x01, 0x00, 0x00, 0x80];
        let g = ImageGrid::parse(&buf).unwrap();
        assert_eq!(g.rows, 2);
        assert_eq!(g.columns, 2);
        assert_eq!(g.output_width, 256);
        assert_eq!(g.output_height, 128);
        assert_eq!(g.expected_tile_count(), 4);
    }

    #[test]
    fn image_grid_parses_32bit() {
        // flags bit 0 = 1 → 32-bit dims.
        let mut buf = vec![0u8, 1, 0, 0]; // version, flags=1, rows=1 cols=1
        buf.extend_from_slice(&65536u32.to_be_bytes());
        buf.extend_from_slice(&32768u32.to_be_bytes());
        let g = ImageGrid::parse(&buf).unwrap();
        assert_eq!(g.rows, 1);
        assert_eq!(g.columns, 1);
        assert_eq!(g.output_width, 65536);
        assert_eq!(g.output_height, 32768);
    }

    #[test]
    fn image_grid_rejects_short_payload() {
        let buf = [0u8, 0, 0, 0]; // 4 bytes < 8
        assert!(ImageGrid::parse(&buf).is_err());
    }

    #[test]
    fn image_overlay_parses_16bit_two_layers() {
        // version 0, flags 0, RGBA fill = (0x4000, 0x4000, 0x4000, 0xFFFF),
        // canvas 256x256, two layers at (96, 96) and (-1, 0).
        let mut buf = Vec::new();
        buf.push(0); // version
        buf.push(0); // flags
        buf.extend_from_slice(&0x4000u16.to_be_bytes());
        buf.extend_from_slice(&0x4000u16.to_be_bytes());
        buf.extend_from_slice(&0x4000u16.to_be_bytes());
        buf.extend_from_slice(&0xFFFFu16.to_be_bytes());
        buf.extend_from_slice(&256u16.to_be_bytes());
        buf.extend_from_slice(&256u16.to_be_bytes());
        // Layer 1: (96, 96)
        buf.extend_from_slice(&96i16.to_be_bytes());
        buf.extend_from_slice(&96i16.to_be_bytes());
        // Layer 2: (-1, 0) — exercises signed offset
        buf.extend_from_slice(&(-1i16).to_be_bytes());
        buf.extend_from_slice(&0i16.to_be_bytes());
        let o = ImageOverlay::parse(&buf, 2).unwrap();
        assert_eq!(o.canvas_fill_rgba, [0x4000, 0x4000, 0x4000, 0xFFFF]);
        assert_eq!(o.output_width, 256);
        assert_eq!(o.output_height, 256);
        assert_eq!(o.offsets, vec![(96, 96), (-1, 0)]);
    }

    #[test]
    fn alpha_urn_constant_matches_spec() {
        // §6.6.2.1.1 — the URN for the HEVC alpha auxiliary type.
        assert_eq!(ALPHA_URN_HEVC, "urn:mpeg:hevc:2015:auxid:1");
    }

    #[test]
    fn clap_rect_centred_crop_64_to_1() {
        // The single-image-1x1 fixture: 64x64 encoded picture, clap
        // numerator 1/1 and offsets (-63/2, -63/2) → 1x1 at origin (0, 0).
        let clap = Clap {
            width_n: 1,
            width_d: 1,
            height_n: 1,
            height_d: 1,
            horiz_off_n: -63,
            horiz_off_d: 2,
            vert_off_n: -63,
            vert_off_d: 2,
        };
        let (left, top, w, h) = clap_rect(&clap, 64, 64).unwrap();
        assert_eq!((left, top, w, h), (0, 0, 1, 1));
    }

    #[test]
    fn clap_rect_no_offset_centred() {
        // Crop a 100x100 to 80x80 centred — horizOff/vertOff = 0/1.
        let clap = Clap {
            width_n: 80,
            width_d: 1,
            height_n: 80,
            height_d: 1,
            horiz_off_n: 0,
            horiz_off_d: 1,
            vert_off_n: 0,
            vert_off_d: 1,
        };
        // pcX = 0 + 99/2 = 49.5; aperture left = 49.5 - 79/2 = 10.0
        // 2*pcX = 99, 2*left = 99 - 79 = 20 -> left = 10.
        let (left, top, w, h) = clap_rect(&clap, 100, 100).unwrap();
        assert_eq!((left, top, w, h), (10, 10, 80, 80));
    }

    #[test]
    fn clap_rect_rejects_zero_denominator() {
        let clap = Clap {
            width_n: 1,
            width_d: 0,
            height_n: 1,
            height_d: 1,
            horiz_off_n: 0,
            horiz_off_d: 1,
            vert_off_n: 0,
            vert_off_d: 1,
        };
        assert!(clap_rect(&clap, 64, 64).is_err());
    }

    #[test]
    fn clap_rect_rejects_overflow() {
        // Aperture wider than the encoded picture.
        let clap = Clap {
            width_n: 200,
            width_d: 1,
            height_n: 200,
            height_d: 1,
            horiz_off_n: 0,
            horiz_off_d: 1,
            vert_off_n: 0,
            vert_off_d: 1,
        };
        assert!(clap_rect(&clap, 100, 100).is_err());
    }

    #[test]
    fn rotate_ccw_90_inverts_columns_to_rows() {
        // 2x3 plane:    rotated 90° CCW becomes 3x2:
        //  A B          B D F
        //  C D    -->   A C E
        //  E F
        let plane = VideoPlane {
            stride: 2,
            data: vec![b'A', b'B', b'C', b'D', b'E', b'F'],
        };
        let r = rotate_ccw_90(&plane, 2, 3);
        assert_eq!(r.stride, 3);
        assert_eq!(r.data, vec![b'B', b'D', b'F', b'A', b'C', b'E']);
    }

    #[test]
    fn rotate_180_flips_both_axes() {
        let plane = VideoPlane {
            stride: 2,
            data: vec![b'A', b'B', b'C', b'D'],
        };
        let r = rotate_180(&plane, 2, 2);
        assert_eq!(r.stride, 2);
        assert_eq!(r.data, vec![b'D', b'C', b'B', b'A']);
    }

    #[test]
    fn rotate_ccw_270_inverts_rows_to_columns() {
        // 2x3 plane → 3x2 rotated 270° CCW (= 90° CW):
        //  A B          E C A
        //  C D    -->   F D B
        //  E F
        let plane = VideoPlane {
            stride: 2,
            data: vec![b'A', b'B', b'C', b'D', b'E', b'F'],
        };
        let r = rotate_ccw_270(&plane, 2, 3);
        assert_eq!(r.stride, 3);
        assert_eq!(r.data, vec![b'E', b'C', b'A', b'F', b'D', b'B']);
    }

    #[test]
    fn imir_axis_0_flips_horizontally() {
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 3,
                data: vec![1, 2, 3, 4, 5, 6],
            }],
        };
        let r = apply_imir(Imir { axis: 0 }, frame).unwrap();
        assert_eq!(r.planes[0].data, vec![3, 2, 1, 6, 5, 4]);
    }

    #[test]
    fn imir_axis_1_flips_vertically() {
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 3,
                data: vec![1, 2, 3, 4, 5, 6],
            }],
        };
        let r = apply_imir(Imir { axis: 1 }, frame).unwrap();
        assert_eq!(r.planes[0].data, vec![4, 5, 6, 1, 2, 3]);
    }

    #[test]
    fn apply_irot_identity_returns_input_unchanged() {
        let frame = VideoFrame {
            pts: Some(7),
            planes: vec![VideoPlane {
                stride: 4,
                data: vec![1, 2, 3, 4, 5, 6, 7, 8],
            }],
        };
        let out = apply_irot(Irot { angle: 0 }, frame.clone()).unwrap();
        assert_eq!(out.planes[0].data, frame.planes[0].data);
        assert_eq!(out.planes[0].stride, 4);
    }

    #[test]
    fn apply_irot_90_swaps_dimensions_for_yuv420() {
        // 4x4 luma, 2x2 chroma — symmetric 4:2:0 so 90° is allowed.
        let luma = VideoPlane {
            stride: 4,
            data: (0..16u8).collect(),
        };
        let cb = VideoPlane {
            stride: 2,
            data: vec![100, 101, 102, 103],
        };
        let cr = VideoPlane {
            stride: 2,
            data: vec![200, 201, 202, 203],
        };
        let frame = VideoFrame {
            pts: None,
            planes: vec![luma, cb, cr],
        };
        let r = apply_irot(Irot { angle: 1 }, frame).unwrap();
        // Luma is still 4x4 because input was square. Verify a corner
        // moved correctly: in[0][0] (=0) should become out[3][0] (=12
        // in row-major after 90° CCW rotation of a 4x4 grid).
        assert_eq!(r.planes[0].stride, 4);
        // After CCW 90: out[w-1-x][y] = in[y][x]. For y=0,x=0: out[3][0] = in[0][0]=0.
        assert_eq!(r.planes[0].data[3 * 4], 0);
    }

    #[test]
    fn crop_planar_extracts_yuv420_subrect() {
        // 8x4 luma, 4x2 chroma. Crop (2, 0, 4, 2) -> 4x2 luma + 2x1 chroma.
        let luma = VideoPlane {
            stride: 8,
            data: (0..32u8).collect(),
        };
        let cb = VideoPlane {
            stride: 4,
            data: vec![100, 101, 102, 103, 104, 105, 106, 107],
        };
        let cr = VideoPlane {
            stride: 4,
            data: vec![200, 201, 202, 203, 204, 205, 206, 207],
        };
        let frame = VideoFrame {
            pts: None,
            planes: vec![luma, cb, cr],
        };
        let r = crop_planar(frame, 2, 0, 4, 2).unwrap();
        assert_eq!(r.planes[0].stride, 4);
        assert_eq!(r.planes[0].data, vec![2, 3, 4, 5, 10, 11, 12, 13]);
        assert_eq!(r.planes[1].stride, 2);
        assert_eq!(r.planes[1].data, vec![101, 102]);
        assert_eq!(r.planes[2].stride, 2);
        assert_eq!(r.planes[2].data, vec![201, 202]);
    }

    #[test]
    fn bt709_limited_grey_round_trips_to_centre_chroma() {
        // RGB (0x40, 0x40, 0x40) is mid-grey → Cb/Cr should land near
        // 128 (limited-range neutral) and Y near studio-mid (~80).
        let (y, cb, cr) = bt709_limited_rgb_to_yuv(0x40, 0x40, 0x40);
        assert!((cb as i32 - 128).abs() <= 1);
        assert!((cr as i32 - 128).abs() <= 1);
        // Y for mid-grey ≈ 16 + (0.18259 + 0.61423 + 0.06201) * 64 = ~71.
        assert!(y > 60 && y < 90, "y = {y}");
    }

    #[test]
    fn bt709_limited_pure_red_lands_on_red_chroma() {
        // Pure red 0xFF, 0, 0: Cr should swing well above 128, Cb below.
        let (_y, cb, cr) = bt709_limited_rgb_to_yuv(0xFF, 0, 0);
        assert!(cr > 200, "cr = {cr}");
        assert!(cb < 110, "cb = {cb}");
    }

    #[test]
    fn compose_overlay_paints_fill_when_no_layer_covers_pixel() {
        // 4x4 canvas, single 2x2 layer at (2, 2) — top-left 2x2
        // remains the fill colour.
        let layer = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 2,
                data: vec![10, 10, 10, 10],
            }],
        };
        let overlay = ImageOverlay {
            version: 0,
            flags: 0,
            canvas_fill_rgba: [0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF],
            output_width: 4,
            output_height: 4,
            offsets: vec![(2, 2)],
        };
        let r = compose_overlay_frames(&overlay, &[layer]).unwrap();
        // Pure-white fill → BT.709 Y near 235.
        let (y_fill, _, _) = bt709_limited_rgb_to_yuv(0xFF, 0xFF, 0xFF);
        assert_eq!(r.planes[0].data[0], y_fill); // top-left covered by fill
        assert_eq!(r.planes[0].data[10], 10); // bottom-right covered by layer (row 2 col 2)
    }

    #[test]
    fn compose_overlay_clips_layer_at_canvas_edge() {
        // 4x4 canvas, single 4x4 layer at (-2, -2) — only the bottom-
        // right 2x2 of the layer makes it onto the canvas.
        let layer = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 4,
                data: (0..16u8).collect(),
            }],
        };
        let overlay = ImageOverlay {
            version: 0,
            flags: 0,
            canvas_fill_rgba: [0, 0, 0, 0xFFFF],
            output_width: 4,
            output_height: 4,
            offsets: vec![(-2, -2)],
        };
        let r = compose_overlay_frames(&overlay, &[layer]).unwrap();
        // Visible source rectangle is (left=2, top=2, w=2, h=2) — the
        // bottom-right 2x2 of the layer (10, 11, 14, 15) is pasted at
        // canvas (0, 0).
        assert_eq!(r.planes[0].data[0], 10);
        assert_eq!(r.planes[0].data[1], 11);
        assert_eq!(r.planes[0].data[4], 14);
        assert_eq!(r.planes[0].data[5], 15);
    }

    #[test]
    fn fill_matrix_defaults_to_bt601_full_when_no_colr() {
        // Task #346 — the iovl canvas-fill matrix defaults to
        // **BT.601 full** when no `colr` is present, matching the
        // convention modern HEIC encoders (libheif / ImageMagick)
        // use when they don't write colr. The previous BT.601
        // limited default produced a luma-floor mismatch with
        // full-range fixtures (Y=1 mapped to R=G=B=0 instead of the
        // expected (1,1,1)).
        assert_eq!(FillMatrix::from_colr(None), FillMatrix::Bt601Full);
    }

    #[test]
    fn fill_matrix_honours_nclx_matrix_coefficients() {
        let nclx_bt709 = Colr::Nclx {
            colour_primaries: 1,
            transfer_characteristics: 1,
            matrix_coefficients: 1,
            full_range: false,
        };
        assert_eq!(
            FillMatrix::from_colr(Some(&nclx_bt709)),
            FillMatrix::Bt709Limited
        );
        let nclx_bt601 = Colr::Nclx {
            colour_primaries: 5,
            transfer_characteristics: 6,
            matrix_coefficients: 6,
            full_range: false,
        };
        assert_eq!(
            FillMatrix::from_colr(Some(&nclx_bt601)),
            FillMatrix::Bt601Limited
        );
        let nclx_bt709_full = Colr::Nclx {
            colour_primaries: 1,
            transfer_characteristics: 1,
            matrix_coefficients: 1,
            full_range: true,
        };
        assert_eq!(
            FillMatrix::from_colr(Some(&nclx_bt709_full)),
            FillMatrix::Bt709Full
        );
    }

    #[test]
    fn fill_matrix_rgb_to_yuv_white_lands_on_high_luma() {
        // Pure white round-trips to Y = 235 limited / 255 full and
        // chroma = 128 (neutral) on every matrix.
        for m in [
            FillMatrix::Bt601Limited,
            FillMatrix::Bt601Full,
            FillMatrix::Bt709Limited,
            FillMatrix::Bt709Full,
        ] {
            let (y, cb, cr) = m.rgb_to_yuv(0xFF, 0xFF, 0xFF);
            assert_eq!(cb, 128, "matrix {m:?} cb");
            assert_eq!(cr, 128, "matrix {m:?} cr");
            let expected = match m {
                FillMatrix::Bt601Limited | FillMatrix::Bt709Limited => 235,
                FillMatrix::Bt601Full | FillMatrix::Bt709Full => 255,
            };
            assert_eq!(y, expected, "matrix {m:?} y");
        }
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
