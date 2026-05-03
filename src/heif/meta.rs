//! `meta` hierarchy walker for the HEIF scaffold.
//!
//! Subset covered (ISO/IEC 14496-12 §8.11 + HEIF §6 / §9):
//!
//! * `hdlr` — handler type (should be `pict` for an HEIF image
//!   collection; we surface the raw value without rejecting).
//! * `pitm` — primary item id (v0 / v1).
//! * `iinf` / `infe` — per-item type + name. Only `infe` v2 / v3 are
//!   accepted (v0 / v1 predate `item_type` and are not legal for image
//!   items).
//! * `iloc` — per-item byte extents. v0 / v1 / v2. Offset, length,
//!   base_offset widths honour the prefix; `construction_method` is
//!   stored but only method 0 is resolvable by [`super::item_bytes`].
//! * `iref` — typed references between items (v0 16-bit IDs, v1 32-bit).
//! * `iprp` → `ipco` + `ipma` — item property store + associations.
//!   The scaffold types `hvcC`, `ispe`, and `colr`; every other property
//!   is kept as a raw [`Property::Other`] so `ipma` indices stay consistent.

use oxideav_core::{Error, Result};

use super::box_parser::{
    b, find_box, iter_boxes, parse_box_header, parse_full_box, read_cstr, read_u16, read_u32,
    read_var_uint, type_str, BoxType,
};

const HDLR: BoxType = b(b"hdlr");
const PITM: BoxType = b(b"pitm");
const IINF: BoxType = b(b"iinf");
const INFE: BoxType = b(b"infe");
const ILOC: BoxType = b(b"iloc");
const IPRP: BoxType = b(b"iprp");
const IPCO: BoxType = b(b"ipco");
const IPMA: BoxType = b(b"ipma");
const IREF: BoxType = b(b"iref");

const IDAT: BoxType = b(b"idat");

const HVCC: BoxType = b(b"hvcC");
const ISPE: BoxType = b(b"ispe");
const COLR: BoxType = b(b"colr");
const AUXC: BoxType = b(b"auxC");
const CLAP: BoxType = b(b"clap");
const IROT: BoxType = b(b"irot");
const IMIR: BoxType = b(b"imir");

#[derive(Clone, Debug)]
pub struct ItemInfo {
    pub id: u32,
    pub item_type: BoxType,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct IlocExtent {
    pub offset: u64,
    pub length: u64,
}

#[derive(Clone, Debug)]
pub struct ItemLocation {
    pub id: u32,
    pub construction_method: u8,
    pub data_reference_index: u16,
    pub base_offset: u64,
    pub extents: Vec<IlocExtent>,
}

#[derive(Clone, Debug)]
pub struct ItemPropertyAssociation {
    pub item_id: u32,
    pub entries: Vec<PropertyAssociation>,
}

#[derive(Clone, Copy, Debug)]
pub struct PropertyAssociation {
    pub index: u16,
    pub essential: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Ispe {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug)]
pub enum Colr {
    Nclx {
        colour_primaries: u16,
        transfer_characteristics: u16,
        matrix_coefficients: u16,
        full_range: bool,
    },
    Icc(Vec<u8>),
    Unknown(BoxType),
}

#[derive(Clone, Debug)]
pub struct IrefEntry {
    pub reference_type: BoxType,
    pub from_id: u32,
    pub to_ids: Vec<u32>,
}

/// Parsed `auxC` (auxiliary configuration, ISO/IEC 23008-12 §6.6.2 /
/// 14496-12 §8.11.5). Carries the per-aux-type URN that classifies an
/// auxiliary item (alpha, depth, …). The HEIF/AVIF alpha-plane URN for
/// HEVC is `urn:mpeg:hevc:2015:auxid:1` (§6.6.2.1.1).
#[derive(Clone, Debug)]
pub struct AuxC {
    /// Null-terminated UTF-8 URN identifying the aux type. Matched
    /// against [`crate::heif::ALPHA_URN_HEVC`] etc. by callers.
    pub aux_type: String,
    /// Trailing aux_subtype bytes (variable-length). Empty for HEVC
    /// alpha; reserved for future aux types.
    pub aux_subtype: Vec<u8>,
}

/// Parsed `clap` (CleanApertureBox, ISO/IEC 14496-12 §12.1.4 /
/// 23008-12 §6.5.9). All four fields are signed-numerator /
/// unsigned-non-zero-denominator fractions (per spec, only horizOff
/// and vertOff have signed N — but the wire layout is `unsigned int(32)`
/// for every N; signedness applies on interpretation). We store the
/// raw on-wire u32 / i32 forms; resolving to a pixel rectangle is the
/// caller's job (see [`crate::heif::clap_rect`]).
///
/// Width/height numerators are the *exact* clean-aperture extent in
/// counted pixels (after pasp correction in the general case; for HEIF
/// items pasp is rarely present and these reduce to integers).
/// horizOff/vertOff are the offset of the clean-aperture centre minus
/// `(width-1)/2` (resp. `(height-1)/2`) — typically zero so the clean
/// aperture is centred on the encoded picture.
#[derive(Clone, Copy, Debug)]
pub struct Clap {
    pub width_n: u32,
    pub width_d: u32,
    pub height_n: u32,
    pub height_d: u32,
    pub horiz_off_n: i32,
    pub horiz_off_d: u32,
    pub vert_off_n: i32,
    pub vert_off_d: u32,
}

/// Parsed `irot` (ImageRotation, ISO/IEC 23008-12 §6.5.10). Single
/// byte: `(reserved << 2) | angle`. `angle * 90` is the rotation in
/// degrees, anti-clockwise.
#[derive(Clone, Copy, Debug)]
pub struct Irot {
    /// 0..=3 — the value stored in the low 2 bits of the box body.
    pub angle: u8,
}

/// Parsed `imir` (ImageMirror, ISO/IEC 23008-12 §6.5.12). Single
/// byte: `(reserved << 1) | axis`. `axis == 0` mirrors about the
/// vertical axis (i.e. flips columns left↔right); `axis == 1` mirrors
/// about the horizontal axis (i.e. flips rows top↔bottom).
#[derive(Clone, Copy, Debug)]
pub struct Imir {
    /// 0 or 1 — the value stored in the low bit of the box body.
    pub axis: u8,
}

#[derive(Clone, Debug)]
pub enum Property {
    /// Raw `hvcC` body (HEVCDecoderConfigurationRecord). Parse with
    /// [`crate::hvcc::parse_hvcc`].
    HvcC(Vec<u8>),
    Ispe(Ispe),
    Colr(Colr),
    AuxC(AuxC),
    /// `clap` clean-aperture transform (ISO/IEC 14496-12 §12.1.4).
    /// Applied after HEVC decode to crop the displayed picture out of
    /// the (potentially padded) coded picture.
    Clap(Clap),
    /// `irot` image-rotation transform (ISO/IEC 23008-12 §6.5.10).
    /// Applied after `clap` (per `ipma` order) to rotate the displayed
    /// picture in 90° increments anti-clockwise.
    Irot(Irot),
    /// `imir` image-mirror transform (ISO/IEC 23008-12 §6.5.12).
    /// Applied per `ipma` order to flip the displayed picture about a
    /// vertical (axis=0) or horizontal (axis=1) axis.
    Imir(Imir),
    Other(BoxType, Vec<u8>),
}

impl Property {
    pub fn kind(&self) -> BoxType {
        match self {
            Property::HvcC(_) => HVCC,
            Property::Ispe(_) => ISPE,
            Property::Colr(_) => COLR,
            Property::AuxC(_) => AUXC,
            Property::Clap(_) => CLAP,
            Property::Irot(_) => IROT,
            Property::Imir(_) => IMIR,
            Property::Other(t, _) => *t,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Meta {
    pub handler: Option<BoxType>,
    pub primary_item_id: Option<u32>,
    pub items: Vec<ItemInfo>,
    pub locations: Vec<ItemLocation>,
    pub properties: Vec<Property>,
    pub associations: Vec<ItemPropertyAssociation>,
    pub irefs: Vec<IrefEntry>,
    /// Body of the (optional) `idat` box living inside `meta`. ISO/IEC
    /// 14496-12 §8.11.4 — payloads of items whose iloc entry uses
    /// `construction_method == 1` (item data embedded in `idat`) live
    /// here. Empty when the file has no `idat` box.
    pub idat: Vec<u8>,
}

impl Meta {
    pub fn parse(meta_payload: &[u8]) -> Result<Self> {
        let (_version, _flags, body) = parse_full_box(meta_payload)?;
        let mut me = Meta::default();
        for hdr in iter_boxes(body) {
            let hdr = hdr?;
            let payload = &body[hdr.payload_start..hdr.end()];
            match &hdr.box_type {
                x if x == &HDLR => {
                    me.handler = Some(parse_hdlr(payload)?);
                }
                x if x == &PITM => {
                    me.primary_item_id = Some(parse_pitm(payload)?);
                }
                x if x == &IINF => {
                    me.items = parse_iinf(payload)?;
                }
                x if x == &ILOC => {
                    me.locations = parse_iloc(payload)?;
                }
                x if x == &IPRP => {
                    let (props, assocs) = parse_iprp(payload)?;
                    me.properties = props;
                    me.associations = assocs;
                }
                x if x == &IREF => {
                    me.irefs = parse_iref(payload)?;
                }
                x if x == &IDAT => {
                    me.idat = payload.to_vec();
                }
                _ => {}
            }
        }
        Ok(me)
    }

    /// Return the list of target item IDs referenced from `from_id` via
    /// an iref entry of the given `reference_type` (e.g. `b"dimg"` for
    /// grid tiles, `b"auxl"` for alpha auxiliaries).
    pub fn iref_targets(&self, reference_type: &BoxType, from_id: u32) -> Vec<u32> {
        for e in &self.irefs {
            if &e.reference_type == reference_type && e.from_id == from_id {
                return e.to_ids.clone();
            }
        }
        Vec::new()
    }

    /// Return the source of the first iref of `reference_type` whose
    /// `to_ids` contains `to_id`. Useful for finding the alpha auxiliary
    /// that points at the primary item via `auxl`.
    pub fn iref_source_of(&self, reference_type: &BoxType, to_id: u32) -> Option<u32> {
        for e in &self.irefs {
            if &e.reference_type == reference_type && e.to_ids.contains(&to_id) {
                return Some(e.from_id);
            }
        }
        None
    }

    pub fn item_by_id(&self, id: u32) -> Option<&ItemInfo> {
        self.items.iter().find(|i| i.id == id)
    }

    pub fn location_by_id(&self, id: u32) -> Option<&ItemLocation> {
        self.locations.iter().find(|l| l.id == id)
    }

    pub fn assoc_by_id(&self, id: u32) -> Option<&ItemPropertyAssociation> {
        self.associations.iter().find(|a| a.item_id == id)
    }

    pub fn property_for<'a>(&'a self, item_id: u32, kind: &BoxType) -> Option<&'a Property> {
        let assoc = self.assoc_by_id(item_id)?;
        for pa in &assoc.entries {
            let prop = self.properties.get(pa.index as usize)?;
            if &prop.kind() == kind {
                return Some(prop);
            }
        }
        None
    }

    /// Iterate `(property, essential)` pairs for `item_id` in the order
    /// they appear in `ipma`. Order matters for transformative
    /// properties (clap → irot → imir is the expected chain when all
    /// three are associated, per ISO/IEC 23008-12 §6.5 — readers must
    /// apply them in `ipma` order).
    pub fn properties_for<'a>(
        &'a self,
        item_id: u32,
    ) -> impl Iterator<Item = (&'a Property, bool)> + 'a {
        let assoc = self.assoc_by_id(item_id);
        assoc
            .into_iter()
            .flat_map(move |a| a.entries.iter())
            .filter_map(move |pa| {
                self.properties
                    .get(pa.index as usize)
                    .map(|p| (p, pa.essential))
            })
    }
}

fn parse_hdlr(payload: &[u8]) -> Result<BoxType> {
    let (_v, _f, body) = parse_full_box(payload)?;
    if body.len() < 8 {
        return Err(Error::invalid("heif: 'hdlr' too short"));
    }
    let mut t = [0u8; 4];
    t.copy_from_slice(&body[4..8]);
    Ok(t)
}

fn parse_pitm(payload: &[u8]) -> Result<u32> {
    let (version, _flags, body) = parse_full_box(payload)?;
    if version == 0 {
        if body.len() < 2 {
            return Err(Error::invalid("heif: 'pitm' too short"));
        }
        Ok(read_u16(body, 0)? as u32)
    } else {
        if body.len() < 4 {
            return Err(Error::invalid("heif: 'pitm' v1 too short"));
        }
        read_u32(body, 0)
    }
}

fn parse_iinf(payload: &[u8]) -> Result<Vec<ItemInfo>> {
    let (version, _flags, body) = parse_full_box(payload)?;
    let (count, mut cursor) = if version == 0 {
        (read_u16(body, 0)? as u32, 2)
    } else {
        (read_u32(body, 0)?, 4)
    };
    let mut out = Vec::with_capacity(count as usize);
    while out.len() < count as usize {
        if cursor >= body.len() {
            return Err(Error::invalid("heif: 'iinf' ran off end"));
        }
        let hdr = parse_box_header(body, cursor)?;
        if hdr.box_type != INFE {
            return Err(Error::invalid(format!(
                "heif: 'iinf' child '{}' != 'infe'",
                type_str(&hdr.box_type)
            )));
        }
        let infe_payload = &body[hdr.payload_start..hdr.end()];
        out.push(parse_infe(infe_payload)?);
        cursor = hdr.end();
    }
    Ok(out)
}

fn parse_infe(payload: &[u8]) -> Result<ItemInfo> {
    let (version, _flags, body) = parse_full_box(payload)?;
    let (id, item_type, mut cursor) = match version {
        2 => {
            if body.len() < 8 {
                return Err(Error::invalid("heif: 'infe' v2 too short"));
            }
            let id = read_u16(body, 0)? as u32;
            let mut t = [0u8; 4];
            t.copy_from_slice(&body[4..8]);
            (id, t, 8usize)
        }
        3 => {
            if body.len() < 10 {
                return Err(Error::invalid("heif: 'infe' v3 too short"));
            }
            let id = read_u32(body, 0)?;
            let mut t = [0u8; 4];
            t.copy_from_slice(&body[6..10]);
            (id, t, 10usize)
        }
        v => {
            return Err(Error::invalid(format!(
                "heif: unsupported 'infe' version {v}"
            )))
        }
    };
    let (name, next) = read_cstr(body, cursor)?;
    cursor = next;
    let _ = cursor;
    Ok(ItemInfo {
        id,
        item_type,
        name,
    })
}

fn parse_iloc(payload: &[u8]) -> Result<Vec<ItemLocation>> {
    let (version, _flags, body) = parse_full_box(payload)?;
    if body.len() < 2 {
        return Err(Error::invalid("heif: 'iloc' too short"));
    }
    let b0 = body[0];
    let b1 = body[1];
    let offset_size = (b0 >> 4) as usize;
    let length_size = (b0 & 0x0f) as usize;
    let base_offset_size = (b1 >> 4) as usize;
    let index_size = if version == 1 || version == 2 {
        (b1 & 0x0f) as usize
    } else {
        0
    };
    let mut cursor = 2usize;
    let item_count = match version {
        0 | 1 => {
            let v = read_u16(body, cursor)? as u32;
            cursor += 2;
            v
        }
        2 => {
            let v = read_u32(body, cursor)?;
            cursor += 4;
            v
        }
        v => return Err(Error::invalid(format!("heif: 'iloc' version {v}"))),
    };
    let mut out = Vec::with_capacity(item_count as usize);
    for _ in 0..item_count {
        let item_id = match version {
            0 | 1 => {
                let v = read_u16(body, cursor)? as u32;
                cursor += 2;
                v
            }
            2 => {
                let v = read_u32(body, cursor)?;
                cursor += 4;
                v
            }
            _ => unreachable!(),
        };
        let construction_method = if version == 1 || version == 2 {
            let w = read_u16(body, cursor)?;
            cursor += 2;
            (w & 0x0f) as u8
        } else {
            0
        };
        let data_reference_index = read_u16(body, cursor)?;
        cursor += 2;
        let base_offset = read_var_uint(body, cursor, base_offset_size)?;
        cursor += base_offset_size;
        let extent_count = read_u16(body, cursor)?;
        cursor += 2;
        let mut extents = Vec::with_capacity(extent_count as usize);
        for _ in 0..extent_count {
            if (version == 1 || version == 2) && index_size > 0 {
                cursor += index_size;
            }
            let offset = read_var_uint(body, cursor, offset_size)?;
            cursor += offset_size;
            let length = read_var_uint(body, cursor, length_size)?;
            cursor += length_size;
            extents.push(IlocExtent { offset, length });
        }
        out.push(ItemLocation {
            id: item_id,
            construction_method,
            data_reference_index,
            base_offset,
            extents,
        });
    }
    Ok(out)
}

fn parse_iprp(payload: &[u8]) -> Result<(Vec<Property>, Vec<ItemPropertyAssociation>)> {
    let (ipco_payload, _) =
        find_box(payload, &IPCO)?.ok_or_else(|| Error::invalid("heif: 'iprp' missing 'ipco'"))?;
    let properties = parse_ipco(ipco_payload)?;
    let mut assocs = Vec::new();
    for hdr in iter_boxes(payload) {
        let hdr = hdr?;
        if hdr.box_type == IPMA {
            let p = &payload[hdr.payload_start..hdr.end()];
            assocs.extend(parse_ipma(p)?);
        }
    }
    Ok((properties, assocs))
}

fn parse_ipco(payload: &[u8]) -> Result<Vec<Property>> {
    let mut out = Vec::new();
    for hdr in iter_boxes(payload) {
        let hdr = hdr?;
        let body = &payload[hdr.payload_start..hdr.end()];
        let prop = match &hdr.box_type {
            x if x == &HVCC => Property::HvcC(body.to_vec()),
            x if x == &ISPE => Property::Ispe(parse_ispe(body)?),
            x if x == &COLR => Property::Colr(parse_colr(body)?),
            x if x == &AUXC => Property::AuxC(parse_auxc(body)?),
            x if x == &CLAP => Property::Clap(parse_clap(body)?),
            x if x == &IROT => Property::Irot(parse_irot(body)?),
            x if x == &IMIR => Property::Imir(parse_imir(body)?),
            other => Property::Other(*other, body.to_vec()),
        };
        out.push(prop);
    }
    Ok(out)
}

/// Parse an `auxC` (AuxiliaryTypeProperty) body — ISO/IEC 23008-12
/// §6.6.2 / 14496-12 §8.11.5. Layout after the FullBox header:
///
/// ```text
/// utf8 aux_type      (NUL-terminated)
/// u8[] aux_subtype   (rest of the box; may be empty)
/// ```
fn parse_auxc(body: &[u8]) -> Result<AuxC> {
    let (_v, _f, rest) = parse_full_box(body)?;
    let (aux_type, next) = read_cstr(rest, 0)?;
    let aux_subtype = rest[next..].to_vec();
    Ok(AuxC {
        aux_type,
        aux_subtype,
    })
}

/// Parse a `clap` (CleanApertureBox) body — ISO/IEC 14496-12 §12.1.4.
/// Layout (no FullBox header, eight `unsigned int(32)` fields):
///
/// ```text
/// u32 cleanApertureWidthN
/// u32 cleanApertureWidthD
/// u32 cleanApertureHeightN
/// u32 cleanApertureHeightD
/// u32 horizOffN          // interpreted as i32
/// u32 horizOffD
/// u32 vertOffN           // interpreted as i32
/// u32 vertOffD
/// ```
///
/// Per spec, width/height N and D must be positive; horizOff/vertOff D
/// must be positive but their N may be negative. We keep the on-wire
/// values verbatim; downstream resolvers ([`crate::heif::clap_rect`])
/// reject the degenerate D == 0 case.
fn parse_clap(body: &[u8]) -> Result<Clap> {
    if body.len() < 32 {
        return Err(Error::invalid(format!(
            "heif: 'clap' body {} bytes < 32",
            body.len()
        )));
    }
    let width_n = read_u32(body, 0)?;
    let width_d = read_u32(body, 4)?;
    let height_n = read_u32(body, 8)?;
    let height_d = read_u32(body, 12)?;
    let horiz_off_n = read_u32(body, 16)? as i32;
    let horiz_off_d = read_u32(body, 20)?;
    let vert_off_n = read_u32(body, 24)? as i32;
    let vert_off_d = read_u32(body, 28)?;
    Ok(Clap {
        width_n,
        width_d,
        height_n,
        height_d,
        horiz_off_n,
        horiz_off_d,
        vert_off_n,
        vert_off_d,
    })
}

/// Parse an `irot` (ImageRotation) body — ISO/IEC 23008-12 §6.5.10.
/// Single byte: top 6 bits reserved (must be 0); low 2 bits carry the
/// rotation count in units of 90° anti-clockwise.
fn parse_irot(body: &[u8]) -> Result<Irot> {
    if body.is_empty() {
        return Err(Error::invalid("heif: 'irot' body is empty"));
    }
    Ok(Irot {
        angle: body[0] & 0x03,
    })
}

/// Parse an `imir` (ImageMirror) body — ISO/IEC 23008-12 §6.5.12.
/// Single byte: top 7 bits reserved (must be 0); low bit selects the
/// mirror axis (0 = vertical axis, 1 = horizontal axis).
fn parse_imir(body: &[u8]) -> Result<Imir> {
    if body.is_empty() {
        return Err(Error::invalid("heif: 'imir' body is empty"));
    }
    Ok(Imir {
        axis: body[0] & 0x01,
    })
}

fn parse_ispe(body: &[u8]) -> Result<Ispe> {
    let (_v, _f, rest) = parse_full_box(body)?;
    if rest.len() < 8 {
        return Err(Error::invalid("heif: 'ispe' too short"));
    }
    Ok(Ispe {
        width: read_u32(rest, 0)?,
        height: read_u32(rest, 4)?,
    })
}

fn parse_colr(body: &[u8]) -> Result<Colr> {
    if body.len() < 4 {
        return Err(Error::invalid("heif: 'colr' too short"));
    }
    let mut tag = [0u8; 4];
    tag.copy_from_slice(&body[..4]);
    match &tag {
        b"nclx" => {
            if body.len() < 4 + 7 {
                return Err(Error::invalid("heif: 'colr' nclx too short"));
            }
            let colour_primaries = read_u16(body, 4)?;
            let transfer_characteristics = read_u16(body, 6)?;
            let matrix_coefficients = read_u16(body, 8)?;
            let full_range = (body[10] & 0x80) != 0;
            Ok(Colr::Nclx {
                colour_primaries,
                transfer_characteristics,
                matrix_coefficients,
                full_range,
            })
        }
        b"rICC" | b"prof" => Ok(Colr::Icc(body[4..].to_vec())),
        other => Ok(Colr::Unknown(*other)),
    }
}

fn parse_iref(payload: &[u8]) -> Result<Vec<IrefEntry>> {
    let (version, _flags, body) = parse_full_box(payload)?;
    if version != 0 && version != 1 {
        return Err(Error::invalid(format!("heif: 'iref' version {version}")));
    }
    let mut out = Vec::new();
    for hdr in iter_boxes(body) {
        let hdr = hdr?;
        let child = &body[hdr.payload_start..hdr.end()];
        let mut cursor = 0usize;
        let from_id = if version == 0 {
            let v = read_u16(child, cursor)? as u32;
            cursor += 2;
            v
        } else {
            let v = read_u32(child, cursor)?;
            cursor += 4;
            v
        };
        let ref_count = read_u16(child, cursor)? as usize;
        cursor += 2;
        let mut to_ids = Vec::with_capacity(ref_count);
        for _ in 0..ref_count {
            let v = if version == 0 {
                let x = read_u16(child, cursor)? as u32;
                cursor += 2;
                x
            } else {
                let x = read_u32(child, cursor)?;
                cursor += 4;
                x
            };
            to_ids.push(v);
        }
        out.push(IrefEntry {
            reference_type: hdr.box_type,
            from_id,
            to_ids,
        });
    }
    Ok(out)
}

fn parse_ipma(payload: &[u8]) -> Result<Vec<ItemPropertyAssociation>> {
    let (version, flags, body) = parse_full_box(payload)?;
    if body.len() < 4 {
        return Err(Error::invalid("heif: 'ipma' too short"));
    }
    let entry_count = read_u32(body, 0)?;
    let mut cursor = 4usize;
    let mut out = Vec::with_capacity(entry_count as usize);
    let index_is_large = (flags & 1) != 0;
    for _ in 0..entry_count {
        let item_id = if version < 1 {
            let v = read_u16(body, cursor)? as u32;
            cursor += 2;
            v
        } else {
            let v = read_u32(body, cursor)?;
            cursor += 4;
            v
        };
        if cursor >= body.len() {
            return Err(Error::invalid("heif: 'ipma' truncated at assoc count"));
        }
        let n = body[cursor] as usize;
        cursor += 1;
        let mut entries = Vec::with_capacity(n);
        for _ in 0..n {
            let (index, essential) = if index_is_large {
                let w = read_u16(body, cursor)?;
                cursor += 2;
                let essential = (w & 0x8000) != 0;
                let raw = (w & 0x7fff) as i32 - 1;
                if raw < 0 {
                    return Err(Error::invalid("heif: 'ipma' index 0"));
                }
                (raw as u16, essential)
            } else {
                if cursor >= body.len() {
                    return Err(Error::invalid("heif: 'ipma' truncated at entry"));
                }
                let w = body[cursor];
                cursor += 1;
                let essential = (w & 0x80) != 0;
                let raw = (w & 0x7f) as i32 - 1;
                if raw < 0 {
                    return Err(Error::invalid("heif: 'ipma' index 0"));
                }
                (raw as u16, essential)
            };
            entries.push(PropertyAssociation { index, essential });
        }
        out.push(ItemPropertyAssociation { item_id, entries });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ispe_round_trip() {
        let mut buf = vec![0u8; 4]; // fullbox header
        buf.extend_from_slice(&100u32.to_be_bytes());
        buf.extend_from_slice(&200u32.to_be_bytes());
        let ispe = parse_ispe(&buf).unwrap();
        assert_eq!(ispe.width, 100);
        assert_eq!(ispe.height, 200);
    }

    #[test]
    fn auxc_round_trip() {
        // FullBox header (4 zero bytes) + NUL-terminated URN.
        let mut body = vec![0u8; 4];
        body.extend_from_slice(b"urn:mpeg:hevc:2015:auxid:1\0");
        let auxc = parse_auxc(&body).unwrap();
        assert_eq!(auxc.aux_type, "urn:mpeg:hevc:2015:auxid:1");
        assert!(auxc.aux_subtype.is_empty());
    }

    #[test]
    fn idat_box_captured_in_meta() {
        // Build a minimal meta containing only an `idat` box (and the
        // FullBox prefix for `meta` itself).
        let idat_body: &[u8] = &[0x01, 0x02, 0x03, 0x04];
        let mut idat = Vec::new();
        let total = (8 + idat_body.len()) as u32;
        idat.extend_from_slice(&total.to_be_bytes());
        idat.extend_from_slice(b"idat");
        idat.extend_from_slice(idat_body);
        let mut meta_payload = vec![0u8; 4]; // FullBox v=0 flags=0
        meta_payload.extend_from_slice(&idat);
        let meta = Meta::parse(&meta_payload).unwrap();
        assert_eq!(meta.idat, idat_body);
    }

    #[test]
    fn clap_parses_eight_u32() {
        let mut body = Vec::new();
        body.extend_from_slice(&1u32.to_be_bytes()); // widthN
        body.extend_from_slice(&1u32.to_be_bytes()); // widthD
        body.extend_from_slice(&1u32.to_be_bytes()); // heightN
        body.extend_from_slice(&1u32.to_be_bytes()); // heightD
        body.extend_from_slice(&(-63i32).to_be_bytes()); // horizOffN
        body.extend_from_slice(&2u32.to_be_bytes()); // horizOffD
        body.extend_from_slice(&(-63i32).to_be_bytes()); // vertOffN
        body.extend_from_slice(&2u32.to_be_bytes()); // vertOffD
        let c = parse_clap(&body).unwrap();
        assert_eq!(c.width_n, 1);
        assert_eq!(c.width_d, 1);
        assert_eq!(c.height_n, 1);
        assert_eq!(c.height_d, 1);
        assert_eq!(c.horiz_off_n, -63);
        assert_eq!(c.horiz_off_d, 2);
        assert_eq!(c.vert_off_n, -63);
        assert_eq!(c.vert_off_d, 2);
    }

    #[test]
    fn clap_rejects_short_body() {
        let body = vec![0u8; 31];
        assert!(parse_clap(&body).is_err());
    }

    #[test]
    fn irot_masks_to_low_two_bits() {
        // Spec: top 6 bits reserved (must be zero), low 2 bits carry
        // the angle. We mask defensively rather than reject — surfaces
        // angle even on out-of-spec writers.
        assert_eq!(parse_irot(&[0x00]).unwrap().angle, 0);
        assert_eq!(parse_irot(&[0x01]).unwrap().angle, 1);
        assert_eq!(parse_irot(&[0x02]).unwrap().angle, 2);
        assert_eq!(parse_irot(&[0x03]).unwrap().angle, 3);
        assert_eq!(parse_irot(&[0xFF]).unwrap().angle, 3);
    }

    #[test]
    fn imir_masks_to_low_bit() {
        assert_eq!(parse_imir(&[0x00]).unwrap().axis, 0);
        assert_eq!(parse_imir(&[0x01]).unwrap().axis, 1);
        assert_eq!(parse_imir(&[0xFE]).unwrap().axis, 0);
        assert_eq!(parse_imir(&[0xFF]).unwrap().axis, 1);
    }

    #[test]
    fn hvcc_property_preserves_bytes() {
        // Build a tiny ipco containing a single 'hvcC' with payload 0xAB 0xCD.
        let mut ipco = Vec::new();
        let child_size = (8 + 2) as u32;
        ipco.extend_from_slice(&child_size.to_be_bytes());
        ipco.extend_from_slice(b"hvcC");
        ipco.extend_from_slice(&[0xAB, 0xCD]);
        let props = parse_ipco(&ipco).unwrap();
        assert_eq!(props.len(), 1);
        match &props[0] {
            Property::HvcC(v) => assert_eq!(v, &vec![0xABu8, 0xCD]),
            other => panic!("expected HvcC, got {other:?}"),
        }
    }
}
