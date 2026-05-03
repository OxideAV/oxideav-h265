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
//! * Item-property boxes beyond `hvcC` / `ispe` / `colr` / `auxC`
//!   (e.g. `pixi`, `pasp`, `irot`, `imir`, `clap`). They are stored as
//!   [`Property::Other`] but not typed — enough to keep `ipma`
//!   indices consistent.
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
pub use meta::{AuxC, Colr, IrefEntry, Ispe, ItemInfo, ItemLocation, Meta, Property};

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
    let loc = meta.location_by_id(item_id).ok_or_else(|| {
        Error::invalid(format!("heif: item {item_id} missing from 'iloc'"))
    })?;
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
    } else if primary_info.item_type == ITEM_TYPE_HVC1
        || primary_info.item_type == ITEM_TYPE_HEV1
    {
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
/// are submitted as one length-prefixed-NAL packet.
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
    let packet =
        Packet::new(0, TimeBase::new(1, 1), item_data.to_vec()).with_keyframe(true);
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
