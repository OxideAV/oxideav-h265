//! Registry decoder — the [`oxideav_core::Decoder`] contract over the
//! whole-bitstream [`crate::sequence`] driver.
//!
//! [`make_decoder`] is the direct factory endpoint (the crate's
//! historical direct-API convention); [`crate::register`] wires the
//! same factory into the [`oxideav_core`] codec registry under the
//! `"h265"` / `"hevc"` ids and the common container tags.
//!
//! Packets carry Annex B byte-stream chunks (start-code delimited NAL
//! units); `CodecParameters::extradata`, when present in Annex B form,
//! is fed ahead of the first packet so out-of-band parameter sets
//! activate. Output frames come in output (PicOrderCntVal) order, with
//! packet PTS values re-attached in ascending order; an empty packet
//! flushes the reorder queue.

use std::collections::BinaryHeap;
use std::collections::VecDeque;

use oxideav_core::{
    CodecId, CodecParameters, Decoder, Error, Frame, Packet, Result, VideoFrame, VideoPlane,
};

use crate::picture::{Picture, Plane};
use crate::sequence::{DecodedFrame, SequenceDecoder};

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
    /// Frames ready to hand out.
    ready: VecDeque<Frame>,
    /// Min-heap of packet PTS values, re-attached in output order.
    pts_queue: BinaryHeap<std::cmp::Reverse<i64>>,
    flushed: bool,
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

/// Direct factory endpoint: construct the software H.265 decoder.
///
/// # Errors
/// [`Error::DecodeError`] when the Annex B `extradata` fails to parse.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let mut seq = SequenceDecoder::new();
    if !params.extradata.is_empty() {
        // Out-of-band parameter sets in Annex B form. (The hvcC /
        // HEVCDecoderConfigurationRecord form is the container
        // sibling's concern; it re-frames into Annex B.)
        seq.push_annexb(&params.extradata)
            .map_err(|e| Error::InvalidData(format!("h265 extradata: {e}")))?;
    }
    Ok(Box::new(H265Decoder {
        codec_id: params.codec_id.clone(),
        seq,
        reorder: Vec::new(),
        ready: VecDeque::new(),
        pts_queue: BinaryHeap::new(),
        flushed: false,
    }))
}

impl H265Decoder {
    /// Move decoded pictures into the reorder buffer and emit every
    /// frame that is guaranteed next in output order.
    fn drain(&mut self, flush: bool) {
        self.reorder.extend(self.seq.take_decoded());
        self.reorder.sort_by_key(|f| (f.cvs_index, f.poc));
        let depth = if flush {
            0
        } else {
            self.seq
                .max_num_reorder_pics()
                .map_or(DEFAULT_REORDER, |n| n as usize)
        };
        while self.reorder.len() > depth {
            let f = self.reorder.remove(0);
            if !f.output {
                continue;
            }
            let pts = self.pts_queue.pop().map(|r| r.0);
            self.ready
                .push_back(Frame::Video(video_frame(&f.picture, pts)));
        }
    }
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
        self.seq
            .push_annexb(&packet.data)
            .map_err(|e| Error::InvalidData(format!("h265 decode: {e}")))?;
        self.drain(false);
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.ready.pop_front() {
            return Ok(f);
        }
        if self.flushed {
            if !self.reorder.is_empty() {
                self.drain(true);
                if let Some(f) = self.ready.pop_front() {
                    return Ok(f);
                }
            }
            return Err(Error::Eof);
        }
        Err(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        // Decode any pending picture and release the reorder queue.
        self.seq
            .flush()
            .map_err(|e| Error::InvalidData(format!("h265 flush: {e}")))?;
        self.flushed = true;
        self.drain(true);
        Ok(())
    }
}

/// Pack a reconstructed [`Picture`] into a [`VideoFrame`] (8-bit planes
/// as one byte per sample; higher bit depths as little-endian 16-bit,
/// the planar `p010le`-family layout).
fn video_frame(pic: &Picture, pts: Option<i64>) -> VideoFrame {
    let planes = if pic.chroma_array_type() == 0 {
        vec![Plane::Luma]
    } else {
        vec![Plane::Luma, Plane::Cb, Plane::Cr]
    };
    let wide = pic.bit_depth(Plane::Luma) > 8
        || (pic.chroma_array_type() != 0 && pic.bit_depth(Plane::Cb) > 8);
    let mut out = Vec::with_capacity(planes.len());
    for plane in planes {
        let (w, h) = pic.plane_dims(plane);
        let buf = pic.plane(plane);
        let mut data;
        let stride;
        if wide {
            stride = w * 2;
            data = Vec::with_capacity(w * h * 2);
            for &v in buf.iter().take(w * h) {
                data.extend_from_slice(&(v as u16).to_le_bytes());
            }
        } else {
            stride = w;
            data = Vec::with_capacity(w * h);
            data.extend(buf.iter().take(w * h).map(|&v| v as u8));
        }
        out.push(VideoPlane { stride, data });
    }
    VideoFrame { pts, planes: out }
}
