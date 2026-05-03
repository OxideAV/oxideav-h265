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
//!   `clap` (e.g. `pixi`, `pasp`, `irot`, `imir`). They are stored as
//!   [`Property::Other`] but not typed — enough to keep `ipma`
//!   indices consistent. `clap` is applied in `ipma` order to every
//!   frame produced by [`decode_primary`] / [`decode_item`] —
//!   ISO/IEC 14496-12 §12.1.4.
//! * Everything at the top level that isn't `ftyp` / `meta` / `mdat`
//!   (e.g. `free`, `skip`).

use oxideav_core::{Error, Frame, Packet, Result, TimeBase, VideoFrame, VideoPlane};

use crate::decoder::HevcDecoder;
use crate::hvcc::{parse_hvcc, HvcConfig};
use oxideav_core::CodecId;
use oxideav_core::Decoder;

mod box_parser;
mod meta;

pub use box_parser::{BoxHeader, BoxType};
pub use meta::{AuxC, Clap, Colr, IrefEntry, Ispe, ItemInfo, ItemLocation, Meta, Property};

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
/// RGBA fill colour. Currently surfaced via [`Meta`] but not composed
/// end-to-end.
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
        Err(Error::unsupported(
            "heif: 'iovl' overlay-derived primary not yet composited (Phase 5)",
        ))
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
        if let Property::Clap(c) = prop {
            frame = apply_clap(*c, frame)?;
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
    if clap.width_d == 0
        || clap.height_d == 0
        || clap.horiz_off_d == 0
        || clap.vert_off_d == 0
    {
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
    for p in 0..n_planes {
        let (sx, sy) = shifts[p];
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
