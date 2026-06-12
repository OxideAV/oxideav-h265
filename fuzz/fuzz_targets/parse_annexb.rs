//! Annex B walk + parameter-set + slice-header fuzz entry.
//!
//! Treats the raw fuzz input as an Annex B byte stream (§B.1): walks
//! every NAL unit, parses the §7.3.1.2 two-byte header, and dispatches
//! the unescaped RBSP to the matching parser — VPS (§7.3.2.1,
//! `nal_unit_type` 32), SPS (§7.3.2.2, 33), PPS (§7.3.2.3.1, 34).
//! Once an SPS + PPS pair has parsed successfully, every VCL NAL unit
//! (`nal_unit_type` < 32) is additionally run through the §7.3.6.1
//! `slice_segment_header()` parse against that activated pair.
//!
//! Every path must return `Ok`/`Err` — no panics, no aborts, no
//! unbounded allocation — regardless of input.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_h265::nal::NalIter;
use oxideav_h265::{HevcVps, PicParameterSet, SeqParameterSet, SliceSegmentHeader};

fuzz_target!(|data: &[u8]| {
    let mut sps: Option<SeqParameterSet> = None;
    let mut pps: Option<PicParameterSet> = None;
    for unit in NalIter::new(data) {
        let Ok(unit) = unit else { break };
        match unit.header.nal_unit_type {
            32 => {
                let _ = HevcVps::parse(&unit.rbsp);
            }
            33 => {
                if let Ok(parsed) = SeqParameterSet::parse(&unit.rbsp) {
                    sps = Some(parsed);
                }
            }
            34 => {
                if let Ok(parsed) = PicParameterSet::parse(&unit.rbsp) {
                    pps = Some(parsed);
                }
            }
            t if t < 32 => {
                if let (Some(s), Some(p)) = (sps.as_ref(), pps.as_ref()) {
                    let _ = SliceSegmentHeader::parse(&unit.rbsp, t, s, p);
                }
            }
            _ => {}
        }
    }
});
