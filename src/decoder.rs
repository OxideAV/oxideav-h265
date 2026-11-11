//! Registry decoder — the [`oxideav_core::Decoder`] contract over the
//! whole-bitstream [`crate::sequence`] driver.
//!
//! [`make_decoder`] is the direct factory endpoint (the crate's
//! historical direct-API convention); [`crate::register`] wires the
//! same factory into the [`oxideav_core`] codec registry under the
//! `"h265"` / `"hevc"` ids and the common container tags.
//!
//! Packets carry either Annex B byte-stream chunks (start-code
//! delimited NAL units) or, when `CodecParameters::extradata` is an
//! `hvcC` / `HEVCDecoderConfigurationRecord` (ISO/IEC 14496-15
//! §8.3.3.1), length-prefixed NAL runs as ISO-BMFF samples carry them.
//! Extradata in either form is fed ahead of the first packet so
//! out-of-band parameter sets activate. Output frames come in output
//! (PicOrderCntVal) order, with packet PTS values re-attached in
//! ascending order; an empty packet flushes the reorder queue.

use std::collections::BinaryHeap;
use std::collections::VecDeque;

use oxideav_core::{
    CodecId, CodecOptions, CodecOptionsStruct, CodecParameters, Decoder, Error, Frame,
    LayerIdentity, LayerInfo, OptionField, OptionKind, OptionValue, Packet, Result, VideoFrame,
    VideoPlane, parse_options,
};

use crate::hvcc::{
    extradata_is_hvcc, extradata_is_lhvc, parse_hvcc_with_len, parse_lhvc, split_length_prefixed,
};
use crate::out_shape::{OutShape, pixel_format};
use crate::picture::{Picture, Plane};
use crate::sequence::{DecodedFrame, LayerTarget, SequenceDecoder};

/// The default reorder depth when no SPS has been activated yet (the
/// §7.4.3.2.1 `sps_max_num_reorder_pics` bound once one has).
const DEFAULT_REORDER: usize = 8;

/// H.265 / HEVC Annex B streaming decoder.
pub struct H265Decoder {
    codec_id: CodecId,
    seq: SequenceDecoder,
    /// Decoded pictures not yet emitted, sorted on demand by
    /// `(cvs_index, poc)`.
    reorder: Vec<DecodedFrame>,
    /// Frames ready to hand out, in output order.
    ready: VecDeque<Ready>,
    /// Min-heap of packet PTS values, re-attached in output order.
    pts_queue: BinaryHeap<std::cmp::Reverse<i64>>,
    /// `Some(n)` when the extradata was an `hvcC` record: packets are
    /// length-prefixed NAL runs with `n`-byte big-endian sizes
    /// (ISO/IEC 14496-15 §8.3.3.1.3 `lengthSizeMinusOne + 1`).
    /// `None` for Annex B packets.
    nal_length_size: Option<usize>,
    flushed: bool,
    /// The description `output_params` answers: that of the last
    /// shape-bearing frame `receive_frame` returned (or announced with
    /// a `Frame::StreamParams` event).
    emitted: Option<OutDesc>,
    /// `Decoder::output_params_generation` — bumped, together with a
    /// `Frame::StreamParams` event, when the next shape-bearing frame's
    /// description differs from `emitted`.
    params_generation: u64,
}

/// What `output_params` states about the frames: the picture shape
/// plus, for a multi-layer stream, the layer structure.
#[derive(Clone, Debug, PartialEq, Eq)]
struct OutDesc {
    shape: OutShape,
    layers: Vec<LayerInfo>,
}

/// One frame waiting in output order.
struct Ready {
    frame: Frame,
    /// `Some` for a *shape-bearing* frame — every frame of a
    /// single-layer stream; of a multi-layer stream, the frames of the
    /// highest output layer (the layer `output_params` describes).
    desc: Option<OutDesc>,
}

impl std::fmt::Debug for H265Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("H265Decoder")
            .field("codec_id", &self.codec_id)
            .field("reorder", &self.reorder.len())
            .field("ready", &self.ready.len())
            .finish()
    }
}

/// The decoder's options: the typed form of the `CodecParameters::
/// options` keys [`make_decoder`] reads, and the argument of
/// [`make_decoder_with_options`]. All of them concern multi-layer
/// (Annex F / G / H) streams; a single-layer stream ignores them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct H265DecoderOptions {
    /// `layer`: decode the layer with this `nuh_layer_id` (0..=63) plus
    /// its reference layers, and output it alone.
    pub layer: Option<u32>,
    /// `view`: as `layer`, selecting the layer that carries this
    /// Annex G `ViewId` (0..=65535). Ignored when `layer` is set.
    pub view: Option<u32>,
    /// `ols`: decode and output this output layer set. Ignored when
    /// `layer` or `view` is set.
    pub ols: Option<u32>,
    /// `layerset`: decode and output every layer of this layer set —
    /// a primary picture together with its auxiliary (alpha / depth)
    /// layers, whatever the output layer sets select. Ignored when
    /// `layer`, `view` or `ols` is set.
    pub layer_set: Option<u32>,
    /// `output_uninitialized_layers`: also output the pictures of an
    /// enhancement layer whose `LayerInitializedFlag` is still 0 — a
    /// layer entered at a non-IRAP picture, which F.8.1.6 decodes
    /// without output until the layer's next IRAP picture. Off by
    /// default. See
    /// [`SequenceDecoder::set_output_uninitialized_layers`].
    pub output_uninitialized_layers: bool,
}

impl H265DecoderOptions {
    /// The defaults: the highest output layer set, conforming output.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set [`Self::layer`].
    #[must_use]
    pub fn with_layer(mut self, nuh_layer_id: u32) -> Self {
        self.layer = Some(nuh_layer_id);
        self
    }

    /// Set [`Self::view`].
    #[must_use]
    pub fn with_view(mut self, view_id: u32) -> Self {
        self.view = Some(view_id);
        self
    }

    /// Set [`Self::ols`].
    #[must_use]
    pub fn with_ols(mut self, ols_idx: u32) -> Self {
        self.ols = Some(ols_idx);
        self
    }

    /// Set [`Self::layer_set`].
    #[must_use]
    pub fn with_layer_set(mut self, ls_idx: u32) -> Self {
        self.layer_set = Some(ls_idx);
        self
    }

    /// Set [`Self::output_uninitialized_layers`].
    #[must_use]
    pub fn with_output_uninitialized_layers(mut self, enabled: bool) -> Self {
        self.output_uninitialized_layers = enabled;
        self
    }

    /// The options a `CodecParameters::options` bag states; keys this
    /// decoder does not know are left alone.
    ///
    /// # Errors
    /// [`Error::InvalidData`] when a known key holds a malformed value.
    pub fn from_codec_options(options: &CodecOptions) -> Result<Self> {
        let mut known = CodecOptions::new();
        for (k, v) in options.iter() {
            if Self::SCHEMA.iter().any(|f| f.name == k) {
                known.insert(k, v);
            }
        }
        parse_options::<Self>(&known).map_err(|e| Error::InvalidData(format!("h265 decode: {e}")))
    }

    fn layer_target(&self) -> Result<LayerTarget> {
        Ok(match (self.layer, self.view, self.ols) {
            (Some(l), _, _) if l < 64 => LayerTarget::Layer(l as u8),
            (Some(l), _, _) => {
                return Err(Error::InvalidData(format!(
                    "h265 decode: layer must be 0..=63, got {l}"
                )));
            }
            (None, Some(v), _) if v <= u32::from(u16::MAX) => LayerTarget::View(v as u16),
            (None, Some(v), _) => {
                return Err(Error::InvalidData(format!(
                    "h265 decode: view must be 0..=65535, got {v}"
                )));
            }
            (None, None, Some(o)) => LayerTarget::Ols(o as usize),
            (None, None, None) => match self.layer_set {
                Some(ls) => LayerTarget::LayerSet(ls as usize),
                None => LayerTarget::HighestOls,
            },
        })
    }
}

impl CodecOptionsStruct for H265DecoderOptions {
    const SCHEMA: &'static [OptionField] = &[
        OptionField {
            name: "layer",
            kind: OptionKind::U32,
            default: OptionValue::U32(0),
            help: "multi-layer: decode this nuh_layer_id (0..=63) and its reference layers, output it alone (unset: the highest output layer set)",
        },
        OptionField {
            name: "view",
            kind: OptionKind::U32,
            default: OptionValue::U32(0),
            help: "multi-layer: as layer, for the layer carrying this ViewId (unset: the highest output layer set)",
        },
        OptionField {
            name: "ols",
            kind: OptionKind::U32,
            default: OptionValue::U32(0),
            help: "multi-layer: decode and output this output layer set (unset: the highest one)",
        },
        OptionField {
            name: "layerset",
            kind: OptionKind::U32,
            default: OptionValue::U32(0),
            help: "multi-layer: decode and output every layer of this layer set, auxiliary layers included (unset: the highest output layer set)",
        },
        OptionField {
            name: "output_uninitialized_layers",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(false),
            help: "multi-layer: also output the pictures of an enhancement layer not initialized yet (entered at a non-IRAP picture)",
        },
    ];

    fn apply(&mut self, key: &str, value: &OptionValue) -> Result<()> {
        match key {
            "layer" => self.layer = Some(value.as_u32()?),
            "view" => self.view = Some(value.as_u32()?),
            "ols" => self.ols = Some(value.as_u32()?),
            "layerset" => self.layer_set = Some(value.as_u32()?),
            "output_uninitialized_layers" => {
                self.output_uninitialized_layers = value.as_bool()?;
            }
            other => {
                return Err(Error::InvalidData(format!("unknown option '{other}'")));
            }
        }
        Ok(())
    }
}

/// Direct factory endpoint: construct the software H.265 decoder.
///
/// The `extradata` form selects the packet framing: an `hvcC`
/// (`HEVCDecoderConfigurationRecord`) extradata activates its carried
/// parameter sets and switches packets to length-prefixed NAL runs;
/// Annex B extradata (or none) keeps start-code framing. An `hvcC`
/// record may be followed by an `lhvC` (`LHEVCDecoderConfigurationRecord`)
/// carrying the non-base layers' parameter sets — the extradata a
/// layered HEIF item (`lhv1`, e.g. a stereo / spatial photo) resolves
/// to; the packets then carry every layer's NAL units and the decoder
/// runs the Annex F/G/H multi-layer processes.
///
/// Codec options (multi-layer streams only; single-layer streams
/// ignore them): `layer=<nuh_layer_id>` decodes that layer (plus its
/// reference layers) and outputs it alone, `view=<ViewId>` selects the
/// layer carrying that Annex G view, `ols=<idx>` selects an output
/// layer set, `layerset=<idx>` decodes and outputs every layer of a
/// layer set (a primary picture with its auxiliary layers). Without any, the highest output layer set is decoded and
/// every one of its output layers is emitted — the frames of one access
/// unit come out consecutively in increasing `nuh_layer_id` order (the
/// base view first, then the second view).
/// `output_uninitialized_layers` (a boolean; default off) also emits the pictures of an
/// enhancement layer that is not initialized yet — a layer entered at
/// a non-IRAP picture, which F.8.1.6 otherwise decodes without output
/// ([`SequenceDecoder::set_output_uninitialized_layers`]).
/// [`H265DecoderOptions`] is the typed form of these keys (its schema
/// is declared on the registry entry); keys it does not list are
/// ignored.
///
/// Output description: `output_params` answers from the parameter sets
/// before the first frame, then describes the frames returned. A new
/// active SPS (or VPS layer structure) that changes it is announced by
/// one [`Frame::StreamParams`] event immediately before the first
/// frame of the new shape — frames of the old shape still queued for
/// reordering come out first — and `output_params_generation` moves by
/// one at that event. For a multi-layer stream the description is that
/// of the **highest output layer** (the highest `nuh_layer_id` the
/// target output layer set outputs); frames of lower output layers keep
/// their own plane geometry and never move the generation, since
/// `CodecParameters` states one picture shape per stream.
///
/// # Errors
/// [`Error::InvalidData`] when the `extradata` or an option fails to
/// parse.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let options = H265DecoderOptions::from_codec_options(&params.options)?;
    make_decoder_with_options(params, &options)
}

/// [`make_decoder`] with the options given in typed form: `options`
/// replaces whatever `params.options` holds.
///
/// # Errors
/// [`Error::InvalidData`] when the `extradata` fails to parse or an
/// option is out of range.
pub fn make_decoder_with_options(
    params: &CodecParameters,
    options: &H265DecoderOptions,
) -> Result<Box<dyn Decoder>> {
    let mut seq = SequenceDecoder::new();
    let mut nal_length_size = None;
    seq.set_layer_target(options.layer_target()?);
    seq.set_output_uninitialized_layers(options.output_uninitialized_layers);
    if !params.extradata.is_empty() {
        if extradata_is_hvcc(&params.extradata) {
            // hvcC record: out-of-band VPS/SPS/PPS (+ SEI) arrays,
            // optionally followed by an lhvC record with the non-base
            // layers' parameter sets.
            let (rec, end) = parse_hvcc_with_len(&params.extradata)
                .map_err(|e| Error::InvalidData(format!("h265 hvcC extradata: {e}")))?;
            for unit in rec.nal_units {
                seq.push_nal_unit(unit)
                    .map_err(|e| Error::InvalidData(format!("h265 hvcC extradata: {e}")))?;
            }
            nal_length_size = Some(rec.length_size);
            let rest = &params.extradata[end..];
            if !rest.is_empty() {
                if !extradata_is_lhvc(rest) {
                    return Err(Error::InvalidData(
                        "h265 hvcC extradata: trailing bytes are not an lhvC record".into(),
                    ));
                }
                let (lrec, _) = parse_lhvc(rest)
                    .map_err(|e| Error::InvalidData(format!("h265 lhvC extradata: {e}")))?;
                for unit in lrec.nal_units {
                    seq.push_nal_unit(unit)
                        .map_err(|e| Error::InvalidData(format!("h265 lhvC extradata: {e}")))?;
                }
            }
        } else {
            // Out-of-band parameter sets in Annex B form.
            seq.push_annexb(&params.extradata)
                .map_err(|e| Error::InvalidData(format!("h265 extradata: {e}")))?;
        }
    }
    Ok(Box::new(H265Decoder {
        codec_id: params.codec_id.clone(),
        seq,
        reorder: Vec::new(),
        ready: VecDeque::new(),
        pts_queue: BinaryHeap::new(),
        nal_length_size,
        flushed: false,
        emitted: None,
        params_generation: 0,
    }))
}

impl H265Decoder {
    /// Move decoded pictures into the reorder buffer and emit every
    /// frame that is guaranteed next in output order.
    fn drain(&mut self, flush: bool) {
        // A skipped RASL picture never comes out; it precedes its IRAP
        // picture (and every later one) in output order, so its PTS is
        // the smallest still queued.
        for _ in 0..self.seq.take_skipped_pictures() {
            self.pts_queue.pop();
        }
        self.reorder.extend(self.seq.take_decoded());
        // Output order: POC within a CVS, the layers of one access unit
        // consecutively by increasing nuh_layer_id.
        self.reorder
            .sort_by_key(|f| (f.cvs_index, f.poc, f.layer_id));
        // The reorder bound counts access units; every output layer
        // of a multi-layer stream adds one frame per access unit.
        let layers_out = self.seq.layer_plan().1.len().max(1);
        let depth = if flush {
            0
        } else {
            self.seq
                .max_num_reorder_pics()
                .map_or(DEFAULT_REORDER, |n| n as usize)
                * layers_out
        };
        let layers = self.seq.layer_structure();
        // The layer `output_params` describes: the highest output layer
        // of the decoded layer plan (layer 0 for a single-layer stream).
        let described_layer = self.seq.layer_plan().1.into_iter().max().unwrap_or(0);
        let layer_infos = layers.as_ref().map(|l| layer_infos(l)).unwrap_or_default();
        while self.reorder.len() > depth {
            let f = self.reorder.remove(0);
            if !f.output {
                continue;
            }
            let pts = self.pts_queue.pop().map(|r| r.0);
            // §7.4.3.2.1 — output the conformance-cropped picture. The
            // frame owns its planes when the DPB no longer shares them
            // (a still, or a picture already unmarked), so the packing
            // consumes them plane by plane instead of holding a second
            // whole-picture copy.
            let picture = if f.crop.is_whole(&f.picture) {
                f.picture
            } else {
                f.output_picture()
            };
            let shape = OutShape {
                width: picture.width_luma() as u32,
                height: picture.height_luma() as u32,
                pixel_format: pixel_format(
                    picture.chroma_array_type(),
                    picture.bit_depth_luma(),
                    picture.bit_depth_chroma(),
                ),
                traits: self.seq.frame_traits(f.cvs_index, f.layer_id),
            };
            let mut frame = video_frame_owned(picture, pts);
            if let Some(structure) = &layers {
                // A multi-layer stream: every frame names its layer, view
                // and access unit (the per-frame identity pairing with
                // the `CodecParameters::layers` description).
                let mut id = LayerIdentity::new(u16::from(f.layer_id)).with_access_unit(f.au_index);
                if let Some(v) = structure
                    .iter()
                    .find(|l| l.layer_id == f.layer_id)
                    .and_then(|l| l.view_id)
                {
                    id = id.with_view_id(v);
                }
                frame.set_layer(id);
            }
            let desc = (f.layer_id == described_layer).then(|| OutDesc {
                shape,
                layers: layer_infos.clone(),
            });
            self.ready.push_back(Ready {
                frame: Frame::Video(frame),
                desc,
            });
        }
        // Coded video sequences no queued picture belongs to any more
        // keep no output traits.
        if let Some(oldest) = self.reorder.iter().map(|f| f.cvs_index).min() {
            self.seq.forget_traits_before(oldest);
        }
    }

    /// The description of the output before the first shape-bearing
    /// frame: the next one released, else the parameter sets.
    fn announced(&self) -> Option<OutDesc> {
        if let Some(d) = self.ready.iter().find_map(|r| r.desc.as_ref()) {
            return Some(d.clone());
        }
        let shape = self.seq.announced_shape()?;
        let layers = self
            .seq
            .layer_structure()
            .map(|l| layer_infos(&l))
            .unwrap_or_default();
        Some(OutDesc { shape, layers })
    }

    /// The next item of `receive_frame`: the next ready frame, or — when
    /// it is the first shape-bearing frame of a new description — the
    /// `Frame::StreamParams` event announcing it (the frame itself stays
    /// queued and comes out on the next call). The description and the
    /// generation move together, exactly when the event is returned;
    /// the first description resolves silently.
    fn next_ready(&mut self) -> Option<Frame> {
        let front = self.ready.front()?;
        if let Some(desc) = &front.desc
            && self.emitted.as_ref() != Some(desc)
        {
            let first = self.emitted.is_none();
            self.emitted = Some(desc.clone());
            if !first {
                self.params_generation = self.params_generation.wrapping_add(1);
                let params = self.output_params().expect("a description was just set");
                return Some(Frame::StreamParams(Box::new(params)));
            }
        }
        self.ready.pop_front().map(|r| r.frame)
    }
}

/// The `CodecParameters::layers` description of a VPS layer structure.
fn layer_infos(structure: &[crate::sequence::LayerDescription]) -> Vec<LayerInfo> {
    structure
        .iter()
        .map(|l| {
            let info = LayerInfo::new(u16::from(l.layer_id)).with_depends_on(
                l.direct_ref_layers
                    .iter()
                    .map(|&r| u16::from(r))
                    .collect::<Vec<_>>(),
            );
            match l.view_id {
                Some(v) => info.with_view_id(v),
                None => info,
            }
        })
        .collect()
}

impl Decoder for H265Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if packet.data.is_empty() {
            // An empty packet is treated as a flush signal too.
            return self.flush();
        }
        if let Some(pts) = packet.pts {
            self.pts_queue.push(std::cmp::Reverse(pts));
        }
        if let Some(length_size) = self.nal_length_size {
            let units = split_length_prefixed(&packet.data, length_size)
                .map_err(|e| Error::InvalidData(format!("h265 decode: {e}")))?;
            for unit in units {
                self.seq
                    .push_nal_unit(unit)
                    .map_err(|e| Error::InvalidData(format!("h265 decode: {e}")))?;
            }
        } else {
            self.seq
                .push_annexb(&packet.data)
                .map_err(|e| Error::InvalidData(format!("h265 decode: {e}")))?;
        }
        self.drain(false);
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.next_ready() {
            return Ok(f);
        }
        if self.flushed {
            if !self.reorder.is_empty() {
                self.drain(true);
                if let Some(f) = self.next_ready() {
                    return Ok(f);
                }
            }
            return Err(Error::Eof);
        }
        Err(Error::NeedMore)
    }

    fn output_params(&self) -> Option<CodecParameters> {
        let desc = match &self.emitted {
            Some(d) => d.clone(),
            None => self.announced()?,
        };
        let mut params = desc.shape.to_params(&self.codec_id);
        // A multi-layer stream: its layer structure from the VPS.
        params.layers = desc.layers;
        Some(params)
    }

    fn output_pixel_format(&self) -> Option<oxideav_core::PixelFormat> {
        self.output_params().and_then(|p| p.pixel_format)
    }

    fn output_params_generation(&self) -> u64 {
        self.params_generation
    }

    fn set_execution_context(&mut self, ctx: &oxideav_core::ExecutionContext) {
        // Bounded through the core contract: at most `threads` workers
        // for the per-picture wavefront / row-parallel filters.
        self.seq.set_threads(ctx.threads);
    }

    fn reset(&mut self) -> Result<()> {
        // A seek: nothing decoded before it is output. The description
        // of the last returned frame stays until the first frame of the
        // new position is about to be returned (the generation moves,
        // with a `Frame::StreamParams` event, only then and only if the
        // description differs).
        self.seq.reset_stream();
        self.reorder.clear();
        self.ready.clear();
        self.pts_queue.clear();
        self.flushed = false;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        // Decode any pending picture and release the reorder queue.
        // A pending picture that fails to decode (a corrupt last slice)
        // is reported, but the decoder still reaches the flushed state:
        // the pictures decoded before it drain, and `receive_frame`
        // then answers `Eof` instead of asking for more input forever.
        let flushed = self.seq.flush();
        // End of stream: the reference pictures go, so the frames
        // drained below own their planes outright.
        self.seq.release_references();
        self.flushed = true;
        self.drain(true);
        flushed.map_err(|e| Error::InvalidData(format!("h265 flush: {e}")))
    }
}

/// Pack a reconstructed [`Picture`] into a [`VideoFrame`] (8-bit planes
/// as one byte per sample; higher bit depths as little-endian 16-bit,
/// the planar `p010le`-family layout), consuming the picture so each
/// source plane is freed as soon as it is packed.
fn video_frame_owned(pic: Picture, pts: Option<i64>) -> VideoFrame {
    let chroma = pic.chroma_array_type() != 0;
    let wide = pic.bit_depth(Plane::Luma) > 8 || (chroma && pic.bit_depth(Plane::Cb) > 8);
    let dims = [
        pic.plane_dims(Plane::Luma),
        pic.plane_dims(Plane::Cb),
        pic.plane_dims(Plane::Cr),
    ];
    let (y, cb, cr) = pic.into_planes();
    let sources = if chroma {
        vec![(y, dims[0]), (cb, dims[1]), (cr, dims[2])]
    } else {
        vec![(y, dims[0])]
    };
    let mut out = Vec::with_capacity(sources.len());
    for (buf, (w, h)) in sources {
        let n = w * h;
        let (stride, data) = if wide {
            let mut data = Vec::with_capacity(n * 2);
            for &v in buf.iter().take(n) {
                data.extend_from_slice(&v.to_le_bytes());
            }
            (w * 2, data)
        } else {
            (w, buf.iter().take(n).map(|&v| v as u8).collect())
        };
        drop(buf);
        out.push(VideoPlane { stride, data });
    }
    VideoFrame { pts, planes: out }
}
