//! Annex B walk + parameter-set + slice-header fuzz entry.
//!
//! Treats the raw fuzz input as an Annex B byte stream (§B.1): walks
//! every NAL unit, parses the §7.3.1.2 two-byte header, and dispatches
//! the unescaped RBSP to the matching parser — VPS (§7.3.2.1,
//! `nal_unit_type` 32), SPS (§7.3.2.2, 33), PPS (§7.3.2.3.1, 34).
//! Once an SPS + PPS pair has parsed successfully, every VCL NAL unit
//! (`nal_unit_type` < 32) is additionally run through the §7.3.6.1
//! `slice_segment_header()` parse against that activated pair. SEI
//! NAL units (39 / 40) run through the §7.3.5 message walk, and —
//! when the active SPS carries VUI `hrd_parameters( )` — every
//! type-0 / type-1 payload body additionally through the
//! context-dependent §D.2.2 `buffering_period( )` / §D.2.3
//! `pic_timing( )` parsers against that HRD context.
//!
//! Every path must return `Ok`/`Err` — no panics, no aborts, no
//! unbounded allocation — regardless of input.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_h265::nal::NalIter;
use oxideav_h265::sei::{parse_sei_rbsp, BufferingPeriodSei, PicTimingSei, SeiNalType, SeiPayload};
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
            t @ (39 | 40) => {
                let nal_type = if t == 39 {
                    SeiNalType::Prefix
                } else {
                    SeiNalType::Suffix
                };
                let Ok(messages) = parse_sei_rbsp(&unit.rbsp, nal_type) else {
                    continue;
                };
                // The context-dependent §D.2.2 / §D.2.3 bodies parse
                // against the active SPS's HRD common info.
                let hrd = sps
                    .as_ref()
                    .and_then(|s| s.vui_parameters.as_ref())
                    .and_then(|v| v.timing_info.as_ref())
                    .and_then(|t| t.hrd_parameters.as_ref());
                let Some(hrd) = hrd else { continue };
                let Some(common) = hrd.common.as_ref() else {
                    continue;
                };
                let cpb_cnt = hrd.sub_layers.first().map_or(1, |sl| sl.cpb_cnt_minus1 + 1);
                for msg in &messages {
                    if let SeiPayload::Reserved { payload_type, data } = &msg.payload {
                        match payload_type {
                            0 => {
                                let _ = BufferingPeriodSei::parse(data, common, cpb_cnt);
                            }
                            1 => {
                                let _ = PicTimingSei::parse(data, common, true);
                                let _ = PicTimingSei::parse(data, common, false);
                            }
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }
});
