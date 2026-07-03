//! Debug: decode NAL-by-NAL, reporting per-picture success/failure.
use oxideav_h265::sequence::SequenceDecoder;
use oxideav_h265::NalIter;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let data = std::fs::read(path).unwrap();
    let mut dec = SequenceDecoder::new();
    let mut n = 0;
    for unit in NalIter::new(&data) {
        let unit = unit.unwrap();
        let t = unit.header.nal_unit_type;
        let first = t < 32 && unit.rbsp.first().is_some_and(|b| b & 0x80 != 0);
        if first {
            n += 1;
        }
        if let Err(e) = dec.push_nal_unit(unit) {
            eprintln!("error while pushing NAL type {t} (picture #{n}): {e}");
            return;
        }
    }
    match dec.finish() {
        Ok(frames) => {
            for f in &frames {
                eprintln!("cvs={} poc={} out={}", f.cvs_index, f.poc, f.output);
            }
        }
        Err(e) => eprintln!("finish error: {e}"),
    }
}
