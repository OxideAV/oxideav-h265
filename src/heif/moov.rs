//! HEIF image-sequence (`msf1` / `hevc` / `heis` brand) sample-table
//! walk.
//!
//! HEIF reuses the ISO/IEC 14496-12 movie-box hierarchy to express
//! image sequences (timelines of HEVC-coded image samples) on top of
//! the still-image meta-box container. This module recovers the per-
//! sample byte ranges by walking
//! `moov` → `trak` → `mdia` → `minf` → `stbl` → `stsd` / `stts` /
//! `stsc` / `stsz` / `stco` (or `co64`) / optional `stss`.
//!
//! It also extracts the `hvcC` decoder configuration record from the
//! first `hvc1` / `hev1` sample-entry inside `stsd`, so callers can
//! feed each sample plus its decoder config into [`HevcDecoder`] one
//! frame at a time.
//!
//! Patterned after the AVIS walker in `oxideav-avif::avis` — same
//! sample-table layout, just tailored for the HEVC sample-entry shape
//! and reusing the HEIF box parser. No external libraries: the layout
//! is from ISO/IEC 14496-12 §8.6 + 14496-15 §8.3 directly, validated
//! against the in-tree `image-sequence-3frame` fixture.
//!
//! # Scope (round 5)
//!
//! * Single video track per file (the only shape produced by HEIF
//!   image-sequence encoders in practice).
//! * `stsd` walked far enough to grab the inner `hvcC` from the first
//!   sample entry; multi-entry `stsd` is folded by always picking the
//!   first entry.
//! * `stss` is honoured when present; absent `stss` → every sample is
//!   a sync sample (mirrors AVIS).
//! * No `tref` / multi-track plumbing.
//!
//! # Out of scope (deferred to a later round)
//!
//! * Edit lists (`elst`) — sample order on the timeline still uses the
//!   `stts` walking order even when `elst` is present.
//! * Composition offsets (`ctts`) — durations still come from `stts`.
//! * Fragmented `moof` / `mfra` — image-sequence files in the corpus
//!   never use fragmentation.

use oxideav_core::{Error, Frame, Packet, Result, TimeBase, VideoFrame};

use super::box_parser::{
    b, find_box, iter_boxes, parse_full_box, read_u32, read_u64, type_str, BoxType,
};
use super::{ITEM_TYPE_HEV1, ITEM_TYPE_HVC1};
use crate::decoder::HevcDecoder;
use crate::hvcc::{parse_hvcc, HvcConfig};
use oxideav_core::CodecId;
use oxideav_core::Decoder;

const MOOV: BoxType = b(b"moov");
const MVHD: BoxType = b(b"mvhd");
const TRAK: BoxType = b(b"trak");
const TKHD: BoxType = b(b"tkhd");
const MDIA: BoxType = b(b"mdia");
const HDLR: BoxType = b(b"hdlr");
const MINF: BoxType = b(b"minf");
const STBL: BoxType = b(b"stbl");
const STSD: BoxType = b(b"stsd");
const STTS: BoxType = b(b"stts");
const STSC: BoxType = b(b"stsc");
const STSZ: BoxType = b(b"stsz");
const STCO: BoxType = b(b"stco");
const CO64: BoxType = b(b"co64");
const STSS: BoxType = b(b"stss");
const HVCC: BoxType = b(b"hvcC");

/// One sample (frame) inside the image-sequence track. `offset` is
/// absolute inside the source file; `size` is the sample's byte length;
/// `duration` is in `timescale` units (see [`MoovSummary::timescale`]).
/// `is_sync` flags sync samples — keyframes that can be decoded
/// standalone. When `stss` is absent every sample is a sync sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sample {
    pub offset: u64,
    pub size: u32,
    pub duration: u32,
    pub is_sync: bool,
}

/// Container-side summary of a HEIF image sequence — what
/// [`parse_moov`] returns when the file carries a `moov` with at least
/// one HEVC-coded image-sequence track.
#[derive(Clone, Debug)]
pub struct MoovSummary {
    /// Movie timescale from `mvhd`. A duration of `timescale` units
    /// is one second of presentation time.
    pub timescale: u32,
    /// Display width / height from `tkhd` (32.16 fixed-point trimmed
    /// to integer pixels). `None` when `tkhd` is missing or malformed.
    pub display_dims: Option<(u32, u32)>,
    /// Sample-entry FourCC from `stsd` — `b"hvc1"` for separated
    /// parameter sets (in `hvcC` only) or `b"hev1"` for inline ones
    /// (in the NAL stream as well). The decoder pre-pends `hvcC`
    /// parameter-set NALs in either case so this is informational.
    pub sample_entry: BoxType,
    /// Raw `hvcC` body lifted out of the sample entry — feed verbatim
    /// to [`HevcDecoder::consume_extradata`].
    pub hvcc_raw: Vec<u8>,
    /// Parsed view of `hvcc_raw`. Parameter-set NALs live inside
    /// `nal_arrays[]`; `length_size_minus_one + 1` is the NAL length
    /// prefix used by every sample.
    pub hvcc: HvcConfig,
    /// Ordered sample table, one entry per frame in decode order.
    pub samples: Vec<Sample>,
}

/// Walk the `moov` box and build a sample table. Returns `Ok(None)`
/// when the file has no `moov` (it's a still-image-only HEIF). Returns
/// `Err(Error::Invalid)` when a `moov` is present but malformed (e.g.
/// missing `stsd` / `stsz` / `stco`).
pub fn parse_moov(file: &[u8]) -> Result<Option<MoovSummary>> {
    let Some((moov_payload, _)) = find_box(file, &MOOV)? else {
        return Ok(None);
    };
    let timescale = find_mvhd_timescale(moov_payload).unwrap_or(1000);
    let display_dims = find_tkhd_display_size(moov_payload);
    let trak = find_first_video_trak(moov_payload).ok_or_else(|| {
        Error::invalid("heif: moov has no video / picture track (no trak with hdlr=vide/pict)")
    })?;
    let mdia = find_box(trak, &MDIA)?
        .ok_or_else(|| Error::invalid("heif: trak missing 'mdia'"))?
        .0;
    let minf = find_box(mdia, &MINF)?
        .ok_or_else(|| Error::invalid("heif: mdia missing 'minf'"))?
        .0;
    let stbl = find_box(minf, &STBL)?
        .ok_or_else(|| Error::invalid("heif: minf missing 'stbl'"))?
        .0;
    let (sample_entry, hvcc_raw) = first_hvcc_in_stsd(stbl)?;
    let hvcc = parse_hvcc(&hvcc_raw)?;
    let samples = sample_table(stbl)?;
    Ok(Some(MoovSummary {
        timescale,
        display_dims,
        sample_entry,
        hvcc_raw,
        hvcc,
        samples,
    }))
}

/// Convenience: fetch the byte slice of `sample` inside `file`. Errors
/// when the declared `(offset, size)` doesn't fit the file length.
pub fn sample_bytes<'a>(file: &'a [u8], sample: &Sample) -> Result<&'a [u8]> {
    let start = sample.offset as usize;
    let end = sample
        .offset
        .checked_add(sample.size as u64)
        .ok_or_else(|| Error::invalid("heif: moov sample offset/size overflow"))?
        as usize;
    if end > file.len() {
        return Err(Error::invalid(format!(
            "heif: moov sample {start}..{end} exceeds file length {}",
            file.len()
        )));
    }
    Ok(&file[start..end])
}

/// Decode every sample in `summary.samples` end-to-end and return the
/// resulting frames in presentation order. The decoder is constructed
/// once, primed with the track's `hvcC` extradata, then fed one
/// `Packet` per sample with its sample-table-derived duration. After
/// the last sample we issue a `flush()` and drain the remaining
/// frames out of the decoder.
///
/// Errors propagate verbatim from [`HevcDecoder`] — the caller can
/// distinguish a structural sample-table problem (which would have
/// surfaced from [`parse_moov`]) from a decoder-side failure.
pub fn decode_image_sequence(file: &[u8], summary: &MoovSummary) -> Result<Vec<VideoFrame>> {
    let mut dec = HevcDecoder::new(CodecId::new(crate::CODEC_ID_STR));
    dec.consume_extradata(&summary.hvcc_raw)?;
    let mut out = Vec::with_capacity(summary.samples.len());
    let mut pts: u64 = 0;
    for s in &summary.samples {
        let bytes = sample_bytes(file, s)?;
        let pkt = Packet::new(
            pts as i64,
            TimeBase::new(1, summary.timescale.max(1)),
            bytes.to_vec(),
        )
        .with_keyframe(s.is_sync);
        dec.send_packet(&pkt)?;
        // Drain any frames the decoder has ready before sending the
        // next packet — keeps the picture buffer small for long
        // sequences and surfaces decoder errors at the right sample.
        while let Ok(Frame::Video(vf)) = dec.receive_frame() {
            out.push(vf);
        }
        pts = pts.saturating_add(s.duration as u64);
    }
    dec.flush()?;
    while let Ok(Frame::Video(vf)) = dec.receive_frame() {
        out.push(vf);
    }
    Ok(out)
}

/// Extract `mvhd`'s timescale. Payload starts with a FullBox header;
/// the timescale lives at body offset 8 (v0) or 16 (v1).
fn find_mvhd_timescale(moov_payload: &[u8]) -> Option<u32> {
    let (p, _) = find_box(moov_payload, &MVHD).ok()??;
    if p.is_empty() {
        return None;
    }
    let (version, _flags, body) = parse_full_box(p).ok()?;
    match version {
        0 => {
            // creation(4) + modification(4) + timescale(4) + duration(4)
            if body.len() < 16 {
                return None;
            }
            Some(u32::from_be_bytes([body[8], body[9], body[10], body[11]]))
        }
        1 => {
            // creation(8) + modification(8) + timescale(4) + duration(8)
            if body.len() < 28 {
                return None;
            }
            Some(u32::from_be_bytes([body[16], body[17], body[18], body[19]]))
        }
        _ => None,
    }
}

/// Display width × height from the first `trak`'s `tkhd`. Stored as
/// 32.16 fixed-point at offsets 76 (v0) or 88 (v1) from the *start of
/// the box payload* — i.e. after the FullBox header but as a flat
/// offset (not from the body).
fn find_tkhd_display_size(moov_payload: &[u8]) -> Option<(u32, u32)> {
    for hdr in iter_boxes(moov_payload) {
        let hdr = match hdr {
            Ok(h) => h,
            Err(_) => continue,
        };
        if hdr.box_type != TRAK {
            continue;
        }
        let trak_payload = &moov_payload[hdr.payload_start..hdr.end()];
        let Some((p, _)) = find_box(trak_payload, &TKHD).ok().flatten() else {
            continue;
        };
        if p.is_empty() {
            continue;
        }
        let version = p[0];
        let off = match version {
            0 => 76,
            1 => 88,
            _ => continue,
        };
        if p.len() < off + 8 {
            continue;
        }
        let w = u32::from_be_bytes([p[off], p[off + 1], p[off + 2], p[off + 3]]) >> 16;
        let h = u32::from_be_bytes([p[off + 4], p[off + 5], p[off + 6], p[off + 7]]) >> 16;
        return Some((w, h));
    }
    None
}

/// Walk every `trak` and return the payload of the first one whose
/// `mdia/hdlr` reports a video / picture handler (FourCC `vide`,
/// `pict`, or `auxv`). Falls back to "first trak we found" when no
/// handler matches — keeps this resilient against minor `hdlr` quirks
/// in the corpus while still preferring the explicitly-tagged track.
fn find_first_video_trak(moov_payload: &[u8]) -> Option<&[u8]> {
    let mut fallback: Option<&[u8]> = None;
    for hdr in iter_boxes(moov_payload) {
        let hdr = match hdr {
            Ok(h) => h,
            Err(_) => continue,
        };
        if hdr.box_type != TRAK {
            continue;
        }
        let trak_payload = &moov_payload[hdr.payload_start..hdr.end()];
        if fallback.is_none() {
            fallback = Some(trak_payload);
        }
        let Some((mdia, _)) = find_box(trak_payload, &MDIA).ok().flatten() else {
            continue;
        };
        let Some((hdlr, _)) = find_box(mdia, &HDLR).ok().flatten() else {
            continue;
        };
        // hdlr body layout (FullBox + pre_defined u32 + handler_type
        // 4cc + reserved[3] u32 + name string).
        if hdlr.len() < 16 {
            continue;
        }
        let handler_type = [hdlr[8], hdlr[9], hdlr[10], hdlr[11]];
        if &handler_type == b"vide" || &handler_type == b"pict" || &handler_type == b"auxv" {
            return Some(trak_payload);
        }
    }
    fallback
}

/// Walk `stsd` and return the first sample entry whose FourCC is
/// `hvc1` or `hev1`, paired with the body of its inner `hvcC` box.
/// Sample-entry layout (ISO/IEC 14496-12 §8.5.2 + 14496-15 §8.3.3):
///
/// ```text
/// SampleEntry (size + 4cc + 6 reserved bytes + data_reference_index u16)
/// VisualSampleEntry (16 reserved bytes + width u16 + height u16 +
///   horizresolution u32 + vertresolution u32 + 4 reserved bytes +
///   frame_count u16 + compressorname[32] + depth u16 + pre_defined i16)
/// HEVCSampleEntry adds: HEVCConfigurationBox (`hvcC`)
/// ```
///
/// `stsd` itself is a FullBox header + `entry_count` u32 + entries
/// back-to-back as full boxes.
fn first_hvcc_in_stsd(stbl: &[u8]) -> Result<(BoxType, Vec<u8>)> {
    let stsd = find_box(stbl, &STSD)?
        .ok_or_else(|| Error::invalid("heif: stbl missing 'stsd'"))?
        .0;
    let (_v, _f, body) = parse_full_box(stsd)?;
    if body.len() < 4 {
        return Err(Error::invalid("heif: stsd entry_count truncated"));
    }
    // entry_count is the first u32 of the FullBox body; the rest is a
    // sequence of sample-entry boxes.
    let entries_start = 4;
    if body.len() <= entries_start {
        return Err(Error::invalid("heif: stsd has no entries"));
    }
    for hdr in iter_boxes(&body[entries_start..]) {
        let hdr = hdr?;
        let entry_payload = &body[entries_start + hdr.payload_start..entries_start + hdr.end()];
        if hdr.box_type != ITEM_TYPE_HVC1 && hdr.box_type != ITEM_TYPE_HEV1 {
            continue;
        }
        // VisualSampleEntry preamble = 8 bytes (SampleEntry) + 70 bytes
        // (VisualSampleEntry-specific). After that the HEVCConfiguration
        // box follows as a regular sub-box.
        const VISUAL_SAMPLE_ENTRY_FIXED: usize = 8 + 70;
        if entry_payload.len() < VISUAL_SAMPLE_ENTRY_FIXED {
            return Err(Error::invalid(format!(
                "heif: '{}' VisualSampleEntry truncated ({} bytes)",
                type_str(&hdr.box_type),
                entry_payload.len()
            )));
        }
        let inner = &entry_payload[VISUAL_SAMPLE_ENTRY_FIXED..];
        let hvcc = find_box(inner, &HVCC)?.ok_or_else(|| {
            Error::invalid(format!(
                "heif: '{}' sample entry missing inner 'hvcC'",
                type_str(&hdr.box_type)
            ))
        })?;
        return Ok((hdr.box_type, hvcc.0.to_vec()));
    }
    Err(Error::invalid(
        "heif: stsd has no 'hvc1' / 'hev1' sample entry",
    ))
}

/// Build a flat list of samples from an `stbl` payload by expanding
/// stts / stsc / stsz / stco / co64 the same way ISO/IEC 14496-12
/// §8.6.1 prescribes. Mirrors `oxideav_avif::avis::sample_table`.
pub fn sample_table(stbl: &[u8]) -> Result<Vec<Sample>> {
    let mut stts_payload: Option<&[u8]> = None;
    let mut stsc_payload: Option<&[u8]> = None;
    let mut stsz_payload: Option<&[u8]> = None;
    let mut stco_payload: Option<&[u8]> = None;
    let mut co64_payload: Option<&[u8]> = None;
    let mut stss_payload: Option<&[u8]> = None;
    for hdr in iter_boxes(stbl) {
        let hdr = hdr?;
        let p = &stbl[hdr.payload_start..hdr.end()];
        match &hdr.box_type {
            x if x == &STTS => stts_payload = Some(p),
            x if x == &STSC => stsc_payload = Some(p),
            x if x == &STSZ => stsz_payload = Some(p),
            x if x == &STCO => stco_payload = Some(p),
            x if x == &CO64 => co64_payload = Some(p),
            x if x == &STSS => stss_payload = Some(p),
            _ => {}
        }
    }
    let stts_p = stts_payload.ok_or_else(|| Error::invalid("heif: moov stbl missing 'stts'"))?;
    let stsc_p = stsc_payload.ok_or_else(|| Error::invalid("heif: moov stbl missing 'stsc'"))?;
    let stsz_p = stsz_payload.ok_or_else(|| Error::invalid("heif: moov stbl missing 'stsz'"))?;
    let (sample_size, sizes) = parse_stsz(stsz_p)?;
    let stsc_entries = parse_stsc(stsc_p)?;
    let sample_deltas = parse_stts(stts_p)?;
    let stss_set = match stss_payload {
        Some(p) => Some(parse_stss(p)?),
        None => None,
    };
    let chunk_offsets: Vec<u64> = if let Some(p) = stco_payload {
        parse_stco(p)?
    } else if let Some(p) = co64_payload {
        parse_co64(p)?
    } else {
        return Err(Error::invalid("heif: moov stbl missing 'stco' / 'co64'"));
    };
    let chunk_count = chunk_offsets.len();
    // Expand stsc to a per-chunk samples_per_chunk slice.
    let mut per_chunk = vec![0u32; chunk_count];
    for (i, e) in stsc_entries.iter().enumerate() {
        let start = (e.first_chunk.saturating_sub(1)) as usize;
        let end = if i + 1 < stsc_entries.len() {
            (stsc_entries[i + 1].first_chunk.saturating_sub(1)) as usize
        } else {
            chunk_count
        };
        if start > end || end > chunk_count {
            return Err(Error::invalid(format!(
                "heif: moov stsc entry {i} out of range (start={start} end={end} chunks={chunk_count})"
            )));
        }
        for c in &mut per_chunk[start..end] {
            *c = e.samples_per_chunk;
        }
    }
    let mut out = Vec::new();
    let mut sample_idx: u32 = 0;
    for c in 0..chunk_count {
        let mut off = chunk_offsets[c];
        for _ in 0..per_chunk[c] {
            let size = if sample_size != 0 {
                sample_size
            } else {
                let idx = sample_idx as usize;
                if idx >= sizes.len() {
                    return Err(Error::invalid(format!(
                        "heif: moov stsz has {} sizes but sample index {idx}",
                        sizes.len()
                    )));
                }
                sizes[idx]
            };
            let duration = sample_deltas.get(sample_idx as usize).copied().unwrap_or(0);
            let is_sync = match &stss_set {
                Some(s) => s.binary_search(&(sample_idx + 1)).is_ok(),
                None => true,
            };
            out.push(Sample {
                offset: off,
                size,
                duration,
                is_sync,
            });
            off = off.saturating_add(size as u64);
            sample_idx = sample_idx.saturating_add(1);
        }
    }
    Ok(out)
}

fn parse_stts(payload: &[u8]) -> Result<Vec<u32>> {
    let (_v, _f, body) = parse_full_box(payload)?;
    if body.len() < 4 {
        return Err(Error::invalid("heif: moov stts truncated"));
    }
    let n = read_u32(body, 0)? as usize;
    let mut cursor = 4usize;
    let mut out = Vec::new();
    for _ in 0..n {
        if cursor + 8 > body.len() {
            return Err(Error::invalid("heif: moov stts entries truncated"));
        }
        let count = read_u32(body, cursor)?;
        cursor += 4;
        let delta = read_u32(body, cursor)?;
        cursor += 4;
        for _ in 0..count {
            out.push(delta);
        }
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug)]
struct StscEntry {
    first_chunk: u32,
    samples_per_chunk: u32,
    _description_idx: u32,
}

fn parse_stsc(payload: &[u8]) -> Result<Vec<StscEntry>> {
    let (_v, _f, body) = parse_full_box(payload)?;
    if body.len() < 4 {
        return Err(Error::invalid("heif: moov stsc truncated"));
    }
    let n = read_u32(body, 0)? as usize;
    let mut cursor = 4usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if cursor + 12 > body.len() {
            return Err(Error::invalid("heif: moov stsc entries truncated"));
        }
        out.push(StscEntry {
            first_chunk: read_u32(body, cursor)?,
            samples_per_chunk: read_u32(body, cursor + 4)?,
            _description_idx: read_u32(body, cursor + 8)?,
        });
        cursor += 12;
    }
    Ok(out)
}

/// Returns `(sample_size, per_sample_sizes)`. When `sample_size != 0`
/// every sample shares that size and the per-sample vector is empty.
fn parse_stsz(payload: &[u8]) -> Result<(u32, Vec<u32>)> {
    let (_v, _f, body) = parse_full_box(payload)?;
    if body.len() < 8 {
        return Err(Error::invalid("heif: moov stsz truncated"));
    }
    let sample_size = read_u32(body, 0)?;
    let sample_count = read_u32(body, 4)? as usize;
    let mut sizes = Vec::new();
    if sample_size == 0 {
        let mut cursor = 8usize;
        for _ in 0..sample_count {
            if cursor + 4 > body.len() {
                return Err(Error::invalid("heif: moov stsz sizes truncated"));
            }
            sizes.push(read_u32(body, cursor)?);
            cursor += 4;
        }
    }
    Ok((sample_size, sizes))
}

fn parse_stco(payload: &[u8]) -> Result<Vec<u64>> {
    let (_v, _f, body) = parse_full_box(payload)?;
    if body.len() < 4 {
        return Err(Error::invalid("heif: moov stco truncated"));
    }
    let n = read_u32(body, 0)? as usize;
    let mut cursor = 4usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if cursor + 4 > body.len() {
            return Err(Error::invalid("heif: moov stco entries truncated"));
        }
        out.push(read_u32(body, cursor)? as u64);
        cursor += 4;
    }
    Ok(out)
}

fn parse_co64(payload: &[u8]) -> Result<Vec<u64>> {
    let (_v, _f, body) = parse_full_box(payload)?;
    if body.len() < 4 {
        return Err(Error::invalid("heif: moov co64 truncated"));
    }
    let n = read_u32(body, 0)? as usize;
    let mut cursor = 4usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if cursor + 8 > body.len() {
            return Err(Error::invalid("heif: moov co64 entries truncated"));
        }
        out.push(read_u64(body, cursor)?);
        cursor += 8;
    }
    Ok(out)
}

/// Parse stss and return a sorted vec of 1-based sample indices.
fn parse_stss(payload: &[u8]) -> Result<Vec<u32>> {
    let (_v, _f, body) = parse_full_box(payload)?;
    if body.len() < 4 {
        return Err(Error::invalid("heif: moov stss truncated"));
    }
    let n = read_u32(body, 0)? as usize;
    let mut cursor = 4usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if cursor + 4 > body.len() {
            return Err(Error::invalid("heif: moov stss entries truncated"));
        }
        out.push(read_u32(body, cursor)?);
        cursor += 4;
    }
    // Spec says sorted ascending — enforce it so binary_search works.
    out.sort_unstable();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal stbl payload containing stts/stsc/stsz/stco/stss
    /// for a single-chunk, 3-sample layout with sizes [10,20,30] at
    /// chunk offset 100. Used to exercise `sample_table` end-to-end
    /// without needing a real HEIC sequence.
    fn minimal_stbl() -> Vec<u8> {
        fn full_box(v: u8, flags: u32, body: &[u8]) -> Vec<u8> {
            let mut out = vec![v, (flags >> 16) as u8, (flags >> 8) as u8, flags as u8];
            out.extend_from_slice(body);
            out
        }
        fn wrap(btype: &[u8; 4], payload: &[u8]) -> Vec<u8> {
            let size = (8 + payload.len()) as u32;
            let mut out = size.to_be_bytes().to_vec();
            out.extend_from_slice(btype);
            out.extend_from_slice(payload);
            out
        }
        let stts_body = {
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&3u32.to_be_bytes());
            b.extend_from_slice(&100u32.to_be_bytes());
            b
        };
        let stsc_body = {
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&1u32.to_be_bytes());
            b.extend_from_slice(&3u32.to_be_bytes());
            b.extend_from_slice(&1u32.to_be_bytes());
            b
        };
        let stsz_body = {
            let mut b = 0u32.to_be_bytes().to_vec();
            b.extend_from_slice(&3u32.to_be_bytes());
            b.extend_from_slice(&10u32.to_be_bytes());
            b.extend_from_slice(&20u32.to_be_bytes());
            b.extend_from_slice(&30u32.to_be_bytes());
            b
        };
        let stco_body = {
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&100u32.to_be_bytes());
            b
        };
        let stss_body = {
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&1u32.to_be_bytes());
            b
        };
        let mut out = Vec::new();
        out.extend_from_slice(&wrap(b"stts", &full_box(0, 0, &stts_body)));
        out.extend_from_slice(&wrap(b"stsc", &full_box(0, 0, &stsc_body)));
        out.extend_from_slice(&wrap(b"stsz", &full_box(0, 0, &stsz_body)));
        out.extend_from_slice(&wrap(b"stco", &full_box(0, 0, &stco_body)));
        out.extend_from_slice(&wrap(b"stss", &full_box(0, 0, &stss_body)));
        out
    }

    #[test]
    fn sample_table_three_samples() {
        let stbl = minimal_stbl();
        let samples = sample_table(&stbl).unwrap();
        assert_eq!(samples.len(), 3);
        assert_eq!(
            samples[0],
            Sample {
                offset: 100,
                size: 10,
                duration: 100,
                is_sync: true,
            }
        );
        assert_eq!(samples[1].offset, 110);
        assert_eq!(samples[1].size, 20);
        assert!(!samples[1].is_sync);
        assert_eq!(samples[2].offset, 130);
        assert_eq!(samples[2].size, 30);
    }

    #[test]
    fn sample_table_missing_stts_errors() {
        let mut stbl = Vec::new();
        let wrap = |t: &[u8; 4], p: &[u8]| {
            let size = (8 + p.len()) as u32;
            let mut out = size.to_be_bytes().to_vec();
            out.extend_from_slice(t);
            out.extend_from_slice(p);
            out
        };
        stbl.extend_from_slice(&wrap(b"stsc", &[0u8; 4]));
        stbl.extend_from_slice(&wrap(b"stsz", &[0u8; 8]));
        stbl.extend_from_slice(&wrap(b"stco", &[0u8; 4]));
        let err = sample_table(&stbl).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("stts"), "error must name the box: {msg}");
    }

    #[test]
    fn sample_table_absent_stss_marks_all_sync() {
        let full = minimal_stbl();
        // stss is the last box — strip it by walking backwards from
        // the "stss" tag.
        let idx = full
            .windows(4)
            .position(|w| w == b"stss")
            .expect("stss present in fixture");
        let stss_size_start = idx - 4;
        let stss_size = u32::from_be_bytes([
            full[stss_size_start],
            full[stss_size_start + 1],
            full[stss_size_start + 2],
            full[stss_size_start + 3],
        ]) as usize;
        let stss_end = stss_size_start + stss_size;
        let stbl_no_stss: Vec<u8> = full
            .iter()
            .take(stss_size_start)
            .chain(full.iter().skip(stss_end))
            .copied()
            .collect();
        let samples = sample_table(&stbl_no_stss).unwrap();
        assert!(samples.iter().all(|s| s.is_sync));
    }

    #[test]
    fn sample_bytes_rejects_overrun() {
        let file = vec![0u8; 16];
        let s = Sample {
            offset: 12,
            size: 8, // 12 + 8 = 20 > 16
            duration: 0,
            is_sync: true,
        };
        assert!(sample_bytes(&file, &s).is_err());
    }

    #[test]
    fn parse_moov_returns_none_on_meta_only_heif() {
        // Synthesise a buffer with only an `ftyp` + `meta` shell so
        // `find_box(file, &MOOV)` returns Ok(None) and parse_moov
        // surfaces None to its caller.
        let mut buf = Vec::new();
        // ftyp(8 + 4 major + 4 minor) — payload 8 bytes -> total 16.
        buf.extend_from_slice(&16u32.to_be_bytes());
        buf.extend_from_slice(b"ftyp");
        buf.extend_from_slice(b"mif1");
        buf.extend_from_slice(&0u32.to_be_bytes());
        // meta(8 + 4 FullBox header) — total 12.
        buf.extend_from_slice(&12u32.to_be_bytes());
        buf.extend_from_slice(b"meta");
        buf.extend_from_slice(&[0u8; 4]);
        let r = parse_moov(&buf).unwrap();
        assert!(r.is_none(), "moov-less HEIF should return None");
    }
}
