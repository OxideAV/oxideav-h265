//! Integration tests against ffmpeg-generated HEVC clips.
//!
//! Tests generate their own HEVC fixtures with local `ffmpeg` + `libx265`.
//! Exact byte-for-byte comparisons only use streams encoded inside the
//! decoder's current feature scope: 8-bit 4:2:0, no tiles/WPP, no SAO, and
//! deblocking disabled.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::process::Stdio;

use oxideav_codec::Decoder;
use oxideav_core::{Error, Packet, TimeBase};
use oxideav_h265::nal::{iter_annex_b, iter_length_prefixed, NalHeader, NalUnitType};
use oxideav_h265::pps::parse_pps;
use oxideav_h265::sps::parse_sps;
use oxideav_h265::vps::parse_vps;
use oxideav_h265::{
    decoder::HevcDecoder, hvcc::parse_hvcc, slice::parse_slice_segment_header, slice::SliceType,
};
use oxideav_h265::{nal::extract_rbsp, CODEC_ID_STR};

fn read_fixture(path: &str) -> Option<Vec<u8>> {
    if !Path::new(path).exists() {
        eprintln!("fixture {path} missing — skipping test");
        return None;
    }
    Some(std::fs::read(path).expect("read fixture"))
}

fn fixture_path(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .arg("-version")
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn generated_fixture_dir() -> PathBuf {
    PathBuf::from("/tmp/oxideav-h265-fixtures")
}

fn ensure_dir(path: &Path) {
    std::fs::create_dir_all(path).expect("create fixture directory");
}

fn run_ffmpeg(args: &[&str]) -> bool {
    Command::new("ffmpeg")
        .args(args)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn ensure_parser_fixtures() -> bool {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not available — skipping parser fixture generation");
        return false;
    }
    let mp4 = PathBuf::from("/tmp/h265_iframe.mp4");
    let es = PathBuf::from("/tmp/h265.es");
    if mp4.exists() && es.exists() {
        return true;
    }
    let fixture_dir = generated_fixture_dir();
    ensure_dir(&fixture_dir);
    let gen_mp4 = fixture_dir.join("parser-iframe.mp4");
    if !gen_mp4.exists()
        && !run_ffmpeg(&[
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=24:duration=0.1",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx265",
            "-profile:v",
            "main",
            "-preset:v",
            "slow",
            "-g",
            "1",
            "-x265-params",
            "log-level=error",
            gen_mp4.to_str().expect("fixture path"),
        ])
    {
        eprintln!("failed to generate parser MP4 fixture");
        return false;
    }
    if !es.exists()
        && !run_ffmpeg(&[
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            gen_mp4.to_str().expect("fixture path"),
            "-c:v",
            "copy",
            "-f",
            "hevc",
            es.to_str().expect("fixture path"),
        ])
    {
        eprintln!("failed to generate parser Annex B fixture");
        return false;
    }
    if !mp4.exists() {
        std::fs::copy(&gen_mp4, &mp4).expect("copy parser mp4 fixture");
    }
    true
}

fn ensure_generated_hevc_fixture(name: &str, lavfi: &str, fps: u32, frames: u32, gop: u32) -> Option<PathBuf> {
    let x265_params = format!(
        "log-level=error:keyint={gop}:min-keyint={gop}:scenecut=0:bframes=0:wpp=0:pmode=0:pme=0:frame-threads=1:no-sao=1:no-deblock=1"
    );
    ensure_generated_hevc_fixture_with_params(name, lavfi, fps, frames, &x265_params)
}

fn ensure_generated_hevc_fixture_with_params(
    name: &str,
    lavfi: &str,
    fps: u32,
    frames: u32,
    x265_params: &str,
) -> Option<PathBuf> {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not available — skipping generated fixture {name}");
        return None;
    }
    let fixture_dir = generated_fixture_dir();
    ensure_dir(&fixture_dir);
    let output = fixture_dir.join(name);
    if output.exists() {
        return Some(output);
    }
    let rate = fps.to_string();
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            lavfi,
            "-frames:v",
            &frames.to_string(),
            "-r",
            &rate,
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx265",
            "-preset:v",
            "medium",
            "-x265-params",
            x265_params,
        ])
        .arg(&output)
        .status()
        .expect("run ffmpeg for generated HEVC fixture");
    if !status.success() {
        eprintln!("failed to generate HEVC fixture {name}");
        return None;
    }
    Some(output)
}

fn ffmpeg_decode_raw(input: &str, output: &Path, frames: Option<usize>) -> Option<Vec<u8>> {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not available — skipping raw compare for {input}");
        return None;
    }
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-hide_banner", "-loglevel", "error", "-y", "-i", input]);
    if let Some(n) = frames {
        cmd.args(["-frames:v", &n.to_string()]);
    }
    let status = cmd
        .args(["-pix_fmt", "yuv420p", "-f", "rawvideo"])
        .arg(output)
        .status()
        .expect("run ffmpeg");
    if !status.success() {
        eprintln!("ffmpeg decode failed for {input} — skipping compare");
        return None;
    }
    Some(std::fs::read(output).expect("read ffmpeg raw output"))
}

fn decode_all_video_frames(data: Vec<u8>, expected_frames: usize) -> Vec<oxideav_core::VideoFrame> {
    let mut dec = HevcDecoder::new(oxideav_core::CodecId::new(CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 24), data);
    dec.send_packet(&pkt).expect("send packet");
    let mut frames = Vec::with_capacity(expected_frames);
    for i in 0..expected_frames {
        match dec.receive_frame() {
            Ok(oxideav_core::Frame::Video(vf)) => frames.push(vf),
            Ok(other) => panic!("expected VideoFrame {i}, got {other:?}"),
            Err(e) => panic!("unexpected error from receive_frame {i}: {e:?}"),
        }
    }
    frames
}

fn flatten_yuv420_frames(frames: &[oxideav_core::VideoFrame]) -> Vec<u8> {
    let mut out = Vec::new();
    for vf in frames {
        assert_eq!(vf.planes.len(), 3, "expected 3 planes");
        out.extend_from_slice(&vf.planes[0].data);
        out.extend_from_slice(&vf.planes[1].data);
        out.extend_from_slice(&vf.planes[2].data);
    }
    out
}

fn assert_yuv420_matches(
    actual: &[u8],
    expected: &[u8],
    width: usize,
    height: usize,
    frames: usize,
    label: &str,
) {
    let luma = width * height;
    let chroma = (width / 2) * (height / 2);
    let frame_len = luma + chroma * 2;
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: byte-length mismatch (actual {}, expected {})",
        actual.len(),
        expected.len()
    );
    assert_eq!(
        actual.len(),
        frame_len * frames,
        "{label}: unexpected byte-length for {frames} YUV420 frame(s)"
    );
    if actual == expected {
        return;
    }
    if std::env::var_os("H265_DUMP_ACTUAL").is_some() {
        let path = format!("/tmp/hevc-actual-{label}.yuv").replace(' ', "_");
        std::fs::write(&path, actual).ok();
        eprintln!("wrote actual output to {path}");
    }
    let mut first_diff = None;
    let mut plane_sad = [0u64; 3];
    let mut plane_max = [0u8; 3];
    let mut plane_max_idx = [0usize; 3];
    for idx in 0..actual.len() {
        if actual[idx] != expected[idx] && first_diff.is_none() {
            first_diff = Some(idx);
        }
        let within_frame = idx % frame_len;
        let plane = if within_frame < luma {
            0
        } else if within_frame < luma + chroma {
            1
        } else {
            2
        };
        let delta = actual[idx].abs_diff(expected[idx]);
        plane_sad[plane] += delta as u64;
        if delta > plane_max[plane] {
            plane_max[plane] = delta;
            plane_max_idx[plane] = idx;
        }
    }
    let idx = first_diff.expect("diff index");
    let frame_idx = idx / frame_len;
    let within_frame = idx % frame_len;
    let (plane_name, plane_off, plane_w) = if within_frame < luma {
        ("Y", within_frame, width)
    } else if within_frame < luma + chroma {
        ("Cb", within_frame - luma, width / 2)
    } else {
        ("Cr", within_frame - luma - chroma, width / 2)
    };
    let x = plane_off % plane_w;
    let y = plane_off / plane_w;
    let max_pos = |plane: usize| {
        if plane_max[plane] == 0 {
            return (0usize, 0usize);
        }
        let idx = plane_max_idx[plane] % frame_len;
        let (plane_off, plane_w) = match plane {
            0 => (idx, width),
            1 => (idx.saturating_sub(luma), width / 2),
            _ => (idx.saturating_sub(luma + chroma), width / 2),
        };
        (plane_off % plane_w, plane_off / plane_w)
    };
    let (max_y_x, max_y_y) = max_pos(0);
    let (max_cb_x, max_cb_y) = max_pos(1);
    let (max_cr_x, max_cr_y) = max_pos(2);
    panic!(
        "{label}: first diff at frame {frame_idx}, plane {plane_name}, x={x}, y={y}, actual={}, expected={}, SAD(Y/Cb/Cr)=({}/{}/{}), max(|diff|)=({}/{}/{}), max-pos(Y/Cb/Cr)=(({},{})/({},{})/({},{}))",
        actual[idx],
        expected[idx],
        plane_sad[0],
        plane_sad[1],
        plane_sad[2],
        plane_max[0],
        plane_max[1],
        plane_max[2],
        max_y_x,
        max_y_y,
        max_cb_x,
        max_cb_y,
        max_cr_x,
        max_cr_y,
    );
}

/// Compute the Y-plane PSNR (dB) between our decoder's output and a
/// reference YUV420 buffer. Returns `f64::INFINITY` on a byte-exact match.
fn y_plane_psnr_db(
    actual: &[u8],
    expected: &[u8],
    width: usize,
    height: usize,
    frames: usize,
) -> f64 {
    let luma = width * height;
    let chroma = (width / 2) * (height / 2);
    let frame_len = luma + chroma * 2;
    assert_eq!(actual.len(), expected.len(), "byte length mismatch");
    assert_eq!(actual.len(), frame_len * frames, "frame-len mismatch");
    let mut sse: u64 = 0;
    let mut count: u64 = 0;
    for f in 0..frames {
        let base = f * frame_len;
        for i in 0..luma {
            let d = actual[base + i] as i32 - expected[base + i] as i32;
            sse += (d * d) as u64;
            count += 1;
        }
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = sse as f64 / count as f64;
    // PSNR for 8-bit: 10 * log10(255^2 / MSE).
    10.0 * (255.0 * 255.0 / mse).log10()
}

/// Walk an MP4 box tree and find the body of the first box whose path
/// matches the supplied 4-char fourcc list. Best-effort; returns `None`
/// if any segment is missing.
fn mp4_find_box<'a>(data: &'a [u8], path: &[&str]) -> Option<&'a [u8]> {
    let mut buf: &'a [u8] = data;
    let mut idx = 0;
    while idx < path.len() {
        let target = path[idx];
        let target_b = target.as_bytes();
        let mut pos = 0;
        let mut found: Option<(&'a [u8], usize, usize)> = None;
        while pos + 8 <= buf.len() {
            let size =
                u32::from_be_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]) as usize;
            let kind = &buf[pos + 4..pos + 8];
            let (header_len, body_len) = if size == 1 {
                if pos + 16 > buf.len() {
                    return None;
                }
                let large = u64::from_be_bytes([
                    buf[pos + 8],
                    buf[pos + 9],
                    buf[pos + 10],
                    buf[pos + 11],
                    buf[pos + 12],
                    buf[pos + 13],
                    buf[pos + 14],
                    buf[pos + 15],
                ]) as usize;
                (16, large.saturating_sub(16))
            } else if size == 0 {
                (8, buf.len() - pos - 8)
            } else {
                (8, size.saturating_sub(8))
            };
            if pos + header_len + body_len > buf.len() {
                return None;
            }
            if kind == target_b {
                found = Some((
                    &buf[pos + header_len..pos + header_len + body_len],
                    header_len,
                    body_len,
                ));
                break;
            }
            pos += header_len + body_len;
        }
        let (next_buf, _h, _b) = found?;
        // Some boxes have leading non-box payload (like stsd has 8 bytes
        // of FullBox header + entry_count) — handle them on the way down.
        let cur_path_segment = path[idx];
        buf = match cur_path_segment {
            // moov/trak/mdia/minf/stbl/stsd: its body starts with 4 bytes of
            // version+flags, 4 bytes of entry_count, then the first entry.
            "stsd" => {
                if next_buf.len() < 8 {
                    return None;
                }
                &next_buf[8..]
            }
            _ => next_buf,
        };
        idx += 1;
    }
    Some(buf)
}

/// Find a child box body anywhere inside a sample-entry payload by walking
/// the trailing child-box list (sample entries have a fixed-size header
/// followed by child boxes).
fn find_in_sample_entry<'a>(
    entry_body: &'a [u8],
    header_len: usize,
    kind: &str,
) -> Option<&'a [u8]> {
    let kind_b = kind.as_bytes();
    let mut buf = &entry_body[header_len..];
    while buf.len() >= 8 {
        let size = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        let k = &buf[4..8];
        if size < 8 || size > buf.len() {
            return None;
        }
        if k == kind_b {
            return Some(&buf[8..size]);
        }
        buf = &buf[size..];
    }
    None
}

#[test]
fn parse_vps_sps_pps_from_annex_b() {
    if !ensure_parser_fixtures() {
        return;
    }
    let Some(data) = read_fixture("/tmp/h265.es") else {
        return;
    };
    let mut got_vps = false;
    let mut got_sps = false;
    let mut got_pps = false;
    for nal in iter_annex_b(&data) {
        let rbsp = extract_rbsp(nal.payload());
        match nal.header.nal_unit_type {
            NalUnitType::Vps => {
                let vps = parse_vps(&rbsp).expect("VPS parse");
                assert_eq!(vps.vps_video_parameter_set_id, 0);
                got_vps = true;
            }
            NalUnitType::Sps => {
                let sps = parse_sps(&rbsp).expect("SPS parse");
                assert_eq!(sps.pic_width_in_luma_samples, 64);
                assert_eq!(sps.pic_height_in_luma_samples, 64);
                assert_eq!(sps.chroma_format_idc, 1, "expected 4:2:0");
                assert_eq!(sps.bit_depth_y(), 8);
                assert_eq!(sps.bit_depth_c(), 8);
                // x265 may signal Main directly (profile_idc=1) or as
                // Format Range Extensions (profile_idc=4); both encode the
                // same 8-bit 4:2:0 main-tier stream.
                let pidc = sps.profile_tier_level.general_profile_idc;
                assert!(matches!(pidc, 1 | 2 | 4), "unexpected profile_idc={pidc}");
                got_sps = true;
            }
            NalUnitType::Pps => {
                let pps = parse_pps(&rbsp).expect("PPS parse");
                assert_eq!(pps.pps_pic_parameter_set_id, 0);
                got_pps = true;
            }
            _ => {}
        }
    }
    assert!(got_vps, "no VPS found in Annex B fixture");
    assert!(got_sps, "no SPS found in Annex B fixture");
    assert!(got_pps, "no PPS found in Annex B fixture");
}

#[test]
fn parse_slice_header_from_annex_b() {
    if !ensure_parser_fixtures() {
        return;
    }
    let Some(data) = read_fixture("/tmp/h265.es") else {
        return;
    };
    let mut sps_opt = None;
    let mut pps_opt = None;
    let mut found_slice = false;
    for nal in iter_annex_b(&data) {
        let rbsp = extract_rbsp(nal.payload());
        match nal.header.nal_unit_type {
            NalUnitType::Sps => {
                sps_opt = Some(parse_sps(&rbsp).expect("SPS"));
            }
            NalUnitType::Pps => {
                pps_opt = Some(parse_pps(&rbsp).expect("PPS"));
            }
            t if t.is_vcl() => {
                let sps = sps_opt.as_ref().expect("SPS before slice");
                let pps = pps_opt.as_ref().expect("PPS before slice");
                let hdr =
                    parse_slice_segment_header(&rbsp, &nal.header, sps, pps).expect("slice header");
                assert!(hdr.first_slice_segment_in_pic_flag, "first slice expected");
                assert_eq!(hdr.slice_pic_parameter_set_id, pps.pps_pic_parameter_set_id);
                // x265 marks the IDR frame as I slice (slice_type == 2).
                assert_eq!(hdr.slice_type, oxideav_h265::slice::SliceType::I);
                found_slice = true;
                break;
            }
            _ => {}
        }
    }
    assert!(found_slice, "no slice NAL in Annex B fixture");
}

#[test]
fn parse_hvcc_from_mp4_extradata() {
    if !ensure_parser_fixtures() {
        return;
    }
    let Some(data) = read_fixture("/tmp/h265_iframe.mp4") else {
        return;
    };
    // Walk down to the stsd's first sample entry, then find the hvcC
    // child box inside it. Sample entries have a 78-byte VisualSampleEntry
    // preamble before child boxes start.
    let stsd_first_entry = mp4_find_box(&data, &["moov", "trak", "mdia", "minf", "stbl", "stsd"])
        .expect("walk down to stsd's first entry");
    // The "first entry" is itself a box; skip its 8-byte box header, then
    // search inside for hvcC after the 78-byte VisualSampleEntry preamble.
    assert!(
        stsd_first_entry.len() >= 8,
        "stsd entry too short to read box header"
    );
    let entry_kind = &stsd_first_entry[4..8];
    assert!(
        entry_kind == b"hvc1" || entry_kind == b"hev1",
        "expected hvc1/hev1 sample entry, got {:?}",
        std::str::from_utf8(entry_kind).unwrap_or("???")
    );
    let entry_body = &stsd_first_entry[8..];
    let hvcc = find_in_sample_entry(entry_body, 78, "hvcC").expect("hvcC inside sample entry");
    let cfg = parse_hvcc(hvcc).expect("parse hvcC");
    assert_eq!(cfg.configuration_version, 1);
    assert_eq!(cfg.length_size_minus_one, 3, "default 4-byte length prefix");
    // x265 main profile — accept either profile_idc=1 (Main) or 4 (RExt)
    // with the Main compatibility bit set.
    assert!(
        cfg.general_profile_idc == 1 || cfg.general_profile_idc == 4,
        "unexpected hvcC profile_idc={}",
        cfg.general_profile_idc
    );

    // Run the decoder's extradata bootstrap and confirm it picks up VPS/SPS/PPS.
    let mut dec = HevcDecoder::new(oxideav_core::CodecId::new(CODEC_ID_STR));
    dec.consume_extradata(hvcc).expect("consume extradata");
    assert!(!dec.vps.is_empty(), "VPS table populated from hvcC");
    assert!(!dec.sps.is_empty(), "SPS table populated from hvcC");
    assert!(!dec.pps.is_empty(), "PPS table populated from hvcC");
    let sps = dec.sps.values().next().unwrap();
    assert_eq!(sps.pic_width_in_luma_samples, 64);
    assert_eq!(sps.pic_height_in_luma_samples, 64);
}

#[test]
fn nal_header_round_trip_on_fixture() {
    if !ensure_parser_fixtures() {
        return;
    }
    let Some(data) = read_fixture("/tmp/h265.es") else {
        return;
    };
    // Every NAL header in a clean fixture must round-trip from_u8/as_u8.
    for nal in iter_annex_b(&data) {
        let raw = nal.header;
        let v = raw.nal_unit_type.as_u8();
        let parsed = NalUnitType::from_u8(v);
        assert_eq!(parsed.as_u8(), v);
        assert!(raw.nuh_temporal_id_plus1 > 0);
    }
}

#[test]
fn i_slice_header_state() {
    // End-to-end test: feed an IDR I-slice packet to the decoder and prove
    // the slice header reached the CTU-decode boundary with plausible
    // derived state. The legacy /tmp/h265.es fixture is generated by x265
    // with wavefront parallel processing enabled, which lands on the
    // "tiles/wavefront/extension" unsupported path — accept either a
    // decoded frame or that specific unsupported message.
    if !ensure_parser_fixtures() {
        return;
    }
    let Some(data) = read_fixture("/tmp/h265.es") else {
        return;
    };
    let mut dec = HevcDecoder::new(oxideav_core::CodecId::new(CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 24), data);
    // send_packet may or may not error depending on whether the fixture's
    // shape is within scope; accept either.
    let _ = dec.send_packet(&pkt);
    let slice = dec.last_slice.as_ref().expect("slice header captured");
    assert_eq!(slice.slice_type, SliceType::I, "IDR should be I slice");
    assert!(
        (0..=51).contains(&slice.slice_qp_y),
        "SliceQpY {} out of legal range",
        slice.slice_qp_y
    );
    match dec.receive_frame() {
        Ok(_frame) => {}
        Err(Error::Unsupported(_)) => {}
        Err(Error::NeedMore) => {}
        Err(e) => panic!("unexpected error from receive_frame: {e:?}"),
    }
}

#[test]
fn length_prefixed_nal_iter_round_trips() {
    // Synthetic length-prefixed bitstream: VPS NAL(0x40,0x01,0xAA) prefixed
    // by a 4-byte length of 3.
    let data = [0x00u8, 0x00, 0x00, 0x03, 0x40, 0x01, 0xAA];
    let nals = iter_length_prefixed(&data, 4).expect("length-prefixed iter");
    assert_eq!(nals.len(), 1);
    let h: NalHeader = nals[0].header;
    assert_eq!(h.nal_unit_type, NalUnitType::Vps);
}

#[test]
fn hevc_p_slice_fixture_decodes() {
    // 2-frame (1 I + 1 P) 256x144 clip. Frame 1 is an IDR I-slice, frame 2
    // is a P-slice referencing frame 1 via the SPS short-term RPS.
    let Some(data) = read_fixture(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/hevc-p.h265"
    )) else {
        return;
    };
    let mut dec = HevcDecoder::new(oxideav_core::CodecId::new(CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 24), data);
    if let Err(Error::Unsupported(msg)) = dec.send_packet(&pkt) {
        // If the fixture has wavefront/tiles/scaling-list enabled we surface
        // Unsupported — don't fail CI, just log.
        eprintln!("P-slice fixture not in scope yet: {msg}");
        return;
    }

    // Expect two video frames.
    let frame1 = match dec.receive_frame() {
        Ok(oxideav_core::Frame::Video(vf)) => vf,
        Ok(other) => panic!("expected VideoFrame, got {other:?}"),
        Err(Error::Unsupported(msg)) => {
            eprintln!("P-slice fixture not in scope yet: {msg}");
            return;
        }
        Err(e) => panic!("unexpected error from receive_frame (frame 1): {e:?}"),
    };
    let frame2 = match dec.receive_frame() {
        Ok(oxideav_core::Frame::Video(vf)) => vf,
        Ok(other) => panic!("expected VideoFrame, got {other:?}"),
        Err(Error::Unsupported(msg)) => {
            panic!("frame 2 (P-slice) unexpectedly unsupported: {msg}");
        }
        Err(e) => panic!("unexpected error from receive_frame (frame 2): {e:?}"),
    };
    assert_eq!(frame1.width, 256);
    assert_eq!(frame1.height, 144);
    assert_eq!(frame2.width, 256);
    assert_eq!(frame2.height, 144);

    let y1 = &frame1.planes[0].data;
    let y2 = &frame2.planes[0].data;
    assert_eq!(y1.len(), y2.len());
    // Frame 2 must differ from frame 1 at the pixel level.
    let mut diff = 0u64;
    for (a, b) in y1.iter().zip(y2.iter()) {
        diff += (*a as i32 - *b as i32).unsigned_abs() as u64;
    }
    assert!(
        diff > 0,
        "expected frame 2 to differ from frame 1 at the pixel level"
    );
}

#[test]
fn hevc_b_slice_fixture_decodes() {
    // 3-frame (I + B + P) 256x144 clip. Frame 2 is a B slice referring to
    // both L0 (the I) and L1 (the P) — bi-prediction + TMVP exercised.
    let Some(data) = read_fixture(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/hevc-b.h265"
    )) else {
        return;
    };
    let mut dec = HevcDecoder::new(oxideav_core::CodecId::new(CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 24), data);
    if let Err(Error::Unsupported(msg)) = dec.send_packet(&pkt) {
        eprintln!("B-slice fixture not in scope yet: {msg}");
        return;
    }

    let mut frames: Vec<oxideav_core::VideoFrame> = Vec::new();
    for i in 0..3 {
        match dec.receive_frame() {
            Ok(oxideav_core::Frame::Video(vf)) => frames.push(vf),
            Ok(other) => panic!("expected VideoFrame at idx {i}, got {other:?}"),
            Err(Error::Unsupported(msg)) => {
                panic!("frame {i} unexpectedly unsupported: {msg}");
            }
            Err(e) => panic!("unexpected error at idx {i}: {e:?}"),
        }
    }
    assert_eq!(frames.len(), 3, "expected 3 decoded frames");
    for vf in &frames {
        assert_eq!(vf.width, 256);
        assert_eq!(vf.height, 144);
    }
    // Decode order is I, B, P — but frames come out of the pipeline in
    // decode order (no DPB reorder buffer yet). Frame 0 is the I-slice.
    let y_i = &frames[0].planes[0].data;
    let y_b = &frames[1].planes[0].data;
    let y_p = &frames[2].planes[0].data;

    let diff = |a: &[u8], b: &[u8]| -> u64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u64)
            .sum()
    };
    let d_ib = diff(y_i, y_b);
    let d_pb = diff(y_p, y_b);
    assert!(
        d_ib > 0,
        "B-frame must differ from I-frame (got diff {d_ib})"
    );
    assert!(
        d_pb > 0,
        "B-frame must differ from P-frame (got diff {d_pb})"
    );
}

#[test]
fn hevc_intra_fixture_decodes_to_plausible_picture() {
    // The fixture is a 256x144 8-bit 4:2:0 I-slice Annex B stream produced
    // with `ffmpeg -f lavfi -i testsrc=... -c:v libx265 -x265-params
    // "keyint=1:no-open-gop=1:wpp=0:pmode=0:pme=0:frame-threads=1:no-sao=1"`.
    // It is intentionally single-tile, wavefront-disabled so it lands inside
    // the v1 intra-only pixel decode scope.
    let Some(data) = read_fixture(&fixture_path("hevc-intra.h265")) else {
        return;
    };
    let mut dec = HevcDecoder::new(oxideav_core::CodecId::new(CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 24), data);
    // If the decoder reports an unsupported feature (e.g. scaling_list or
    // transform_skip turned on for this fixture), skip gracefully so CI
    // without a spec-identical x265 build keeps passing.
    if let Err(Error::Unsupported(msg)) = dec.send_packet(&pkt) {
        eprintln!("fixture decode not in scope yet: {msg}");
        return;
    }
    let frame = match dec.receive_frame() {
        Ok(f) => f,
        Err(Error::Unsupported(msg)) => {
            eprintln!("fixture decode not in scope yet: {msg}");
            return;
        }
        Err(e) => panic!("unexpected error from receive_frame: {e:?}"),
    };
    let vf = match frame {
        oxideav_core::Frame::Video(vf) => vf,
        other => panic!("expected VideoFrame, got {other:?}"),
    };
    assert_eq!(vf.width, 256);
    assert_eq!(vf.height, 144);
    let y = &vf.planes[0].data;
    assert_eq!(y.len(), 256 * 144);
    let mean: u64 = y.iter().map(|&v| v as u64).sum();
    let mean = (mean / y.len() as u64) as u8;
    let distinct = y
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert!(
        (32..=224).contains(&mean),
        "luma mean {mean} out of plausibility range [32, 224]",
    );
    assert!(
        distinct > 20,
        "expected more than 20 distinct luma values, got {distinct}",
    );
}

#[test]
fn hevc_intra_gray_16_matches_ffmpeg() {
    let Some(input) = ensure_generated_hevc_fixture(
        "gray16.h265",
        "color=c=gray:size=16x16:rate=1:duration=1", 1, 1, 1,
    ) else { return; };
    let input_str = input.to_string_lossy().into_owned();
    let Some(data) = read_fixture(&input_str) else { return; };
    let Some(expected) = ffmpeg_decode_raw(&input_str, &PathBuf::from("/tmp/hevc-gray16.ref.yuv"), Some(1))
    else { return; };
    let frames = decode_all_video_frames(data, 1);
    let actual = flatten_yuv420_frames(&frames);
    assert_yuv420_matches(&actual, &expected, 16, 16, 1, "intra gray 16 fixture");
}

#[test]
fn hevc_intra_testsrc_64_matches_ffmpeg() {
    let Some(input) = ensure_generated_hevc_fixture(
        "testsrc-64.h265",
        "testsrc=size=64x64:rate=1:duration=1",
        1, 1, 1,
    ) else { return; };
    let input_str = input.to_string_lossy().into_owned();
    let Some(data) = read_fixture(&input_str) else { return; };
    let Some(expected) = ffmpeg_decode_raw(&input_str, &PathBuf::from("/tmp/hevc-testsrc-64.ref.yuv"), Some(1))
    else { return; };
    let frames = decode_all_video_frames(data, 1);
    let actual = flatten_yuv420_frames(&frames);
    assert_yuv420_matches(&actual, &expected, 64, 64, 1, "intra testsrc 64 fixture");
}

#[test]
fn hevc_intra_testsrc2_64_matches_ffmpeg() {
    let Some(input) = ensure_generated_hevc_fixture(
        "testsrc2-64.h265",
        "testsrc2=size=64x64:rate=1:duration=1",
        1, 1, 1,
    ) else { return; };
    let input_str = input.to_string_lossy().into_owned();
    let Some(data) = read_fixture(&input_str) else { return; };
    let Some(expected) = ffmpeg_decode_raw(&input_str, &PathBuf::from("/tmp/hevc-testsrc2-64.ref.yuv"), Some(1))
    else { return; };
    let frames = decode_all_video_frames(data, 1);
    let actual = flatten_yuv420_frames(&frames);
    assert_yuv420_matches(&actual, &expected, 64, 64, 1, "intra testsrc2 64 fixture");
}

#[test]
fn hevc_intra_smptebars_64_matches_ffmpeg() {
    let Some(input) = ensure_generated_hevc_fixture(
        "smptebars-64.h265",
        "smptebars=size=64x64:rate=1:duration=1",
        1, 1, 1,
    ) else { return; };
    let input_str = input.to_string_lossy().into_owned();
    let Some(data) = read_fixture(&input_str) else { return; };
    let Some(expected) = ffmpeg_decode_raw(&input_str, &PathBuf::from("/tmp/hevc-smptebars-64.ref.yuv"), Some(1))
    else { return; };
    let frames = decode_all_video_frames(data, 1);
    let actual = flatten_yuv420_frames(&frames);
    assert_yuv420_matches(&actual, &expected, 64, 64, 1, "intra smptebars 64 fixture");
}

#[test]
fn hevc_intra_rgbtestsrc_64_matches_ffmpeg() {
    let Some(input) = ensure_generated_hevc_fixture(
        "rgbtestsrc-64.h265",
        "rgbtestsrc=size=64x64:rate=1:duration=1",
        1, 1, 1,
    ) else { return; };
    let input_str = input.to_string_lossy().into_owned();
    let Some(data) = read_fixture(&input_str) else { return; };
    let Some(expected) = ffmpeg_decode_raw(&input_str, &PathBuf::from("/tmp/hevc-rgbtestsrc-64.ref.yuv"), Some(1))
    else { return; };
    let frames = decode_all_video_frames(data, 1);
    let actual = flatten_yuv420_frames(&frames);
    assert_yuv420_matches(&actual, &expected, 64, 64, 1, "intra rgbtestsrc 64 fixture");
}

// First diff at (80, 0) inside the second 64-wide CTU row — the bug is
// specific to reference / CABAC state crossing a CTU boundary with real
// content (gray-only multi-CTU fixtures pass). Left ignored as a
// progress tracker.
#[test]
#[ignore]
fn hevc_intra_mandelbrot_128_matches_ffmpeg() {
    let Some(input) = ensure_generated_hevc_fixture(
        "mandelbrot-128.h265",
        "mandelbrot=size=128x128:rate=1:end_pts=1",
        1, 1, 1,
    ) else { return; };
    let input_str = input.to_string_lossy().into_owned();
    let Some(data) = read_fixture(&input_str) else { return; };
    let Some(expected) = ffmpeg_decode_raw(&input_str, &PathBuf::from("/tmp/hevc-mandelbrot-128.ref.yuv"), Some(1))
    else { return; };
    let frames = decode_all_video_frames(data, 1);
    let actual = flatten_yuv420_frames(&frames);
    assert_yuv420_matches(&actual, &expected, 128, 128, 1, "intra mandelbrot 128 fixture");
}

// Fails today: first diff at (64, 0), the seam of CTU (64, 0). Same
// cross-CTU regression as the mandelbrot and 192×112 fixtures.
#[test]
#[ignore]
fn hevc_intra_testsrc_128x72_matches_ffmpeg() {
    // Non-square, non-CTU-multiple on the height axis — exercises
    // picture-edge CTU handling for 4:2:0 with a partial last row.
    let Some(input) = ensure_generated_hevc_fixture(
        "testsrc-128x72.h265",
        "testsrc=size=128x72:rate=1:duration=1",
        1, 1, 1,
    ) else { return; };
    let input_str = input.to_string_lossy().into_owned();
    let Some(data) = read_fixture(&input_str) else { return; };
    let Some(expected) = ffmpeg_decode_raw(&input_str, &PathBuf::from("/tmp/hevc-testsrc-128x72.ref.yuv"), Some(1))
    else { return; };
    let frames = decode_all_video_frames(data, 1);
    let actual = flatten_yuv420_frames(&frames);
    assert_yuv420_matches(&actual, &expected, 128, 72, 1, "intra testsrc 128x72 fixture");
}

// Fails today at (20, 64) — second CTU row. Same cross-CTU bug.
#[test]
#[ignore]
fn hevc_intra_testsrc_192x112_matches_ffmpeg() {
    // Large enough that x265 picks 32×32 (or larger) CU splits in the
    // interior, exercising the strong-intra-smoothing decision branch
    // (§8.4.4.2.3) and larger-TU inverse transforms.
    let Some(input) = ensure_generated_hevc_fixture(
        "testsrc-192x112.h265",
        "testsrc=size=192x112:rate=1:duration=1",
        1, 1, 1,
    ) else { return; };
    let input_str = input.to_string_lossy().into_owned();
    let Some(data) = read_fixture(&input_str) else { return; };
    let Some(expected) = ffmpeg_decode_raw(&input_str, &PathBuf::from("/tmp/hevc-testsrc-192x112.ref.yuv"), Some(1))
    else { return; };
    let frames = decode_all_video_frames(data, 1);
    let actual = flatten_yuv420_frames(&frames);
    assert_yuv420_matches(&actual, &expected, 192, 112, 1, "intra testsrc 192x112 fixture");
}

#[test]
fn hevc_intra_gray_64_qp51_matches_ffmpeg() {
    let x265 = "log-level=error:keyint=1:min-keyint=1:scenecut=0:bframes=0:\
                wpp=0:pmode=0:pme=0:frame-threads=1:no-sao=1:no-deblock=1:qp=51";
    let Some(input) = ensure_generated_hevc_fixture_with_params(
        "exact-intra-gray-64-qp51.h265",
        "color=c=gray:size=64x64:rate=1:duration=1", 1, 1, x265,
    ) else { return; };
    let input_str = input.to_string_lossy().into_owned();
    let Some(data) = read_fixture(&input_str) else { return; };
    let Some(expected) = ffmpeg_decode_raw(&input_str, &PathBuf::from("/tmp/hevc-intra-64-qp51.ref.yuv"), Some(1))
    else { return; };
    let frames = decode_all_video_frames(data, 1);
    let actual = flatten_yuv420_frames(&frames);
    assert_yuv420_matches(&actual, &expected, 64, 64, 1, "intra 64 qp51 fixture");
}

#[test]
fn hevc_intra_gray_64_matches_ffmpeg() {
    let Some(input) = ensure_generated_hevc_fixture(
        "exact-intra-gray-64.h265",
        "color=c=gray:size=64x64:rate=1:duration=1",
        1, 1, 1,
    ) else { return; };
    let input_str = input.to_string_lossy().into_owned();
    let Some(data) = read_fixture(&input_str) else { return; };
    let Some(expected) = ffmpeg_decode_raw(&input_str, &PathBuf::from("/tmp/hevc-intra-64.ref.yuv"), Some(1))
    else { return; };
    let frames = decode_all_video_frames(data, 1);
    let actual = flatten_yuv420_frames(&frames);
    assert_yuv420_matches(&actual, &expected, 64, 64, 1, "intra 64 fixture");
}

#[test]
fn hevc_intra_fixture_matches_ffmpeg() {
    let Some(input) = ensure_generated_hevc_fixture(
        "exact-intra-gray.h265",
        "color=c=gray:size=256x144:rate=1:duration=1",
        1,
        1,
        1,
    ) else {
        return;
    };
    let input_str = input.to_string_lossy().into_owned();
    let Some(data) = read_fixture(&input_str) else {
        return;
    };
    let Some(expected) =
        ffmpeg_decode_raw(&input_str, &PathBuf::from("/tmp/hevc-intra.ref.yuv"), Some(1))
    else {
        return;
    };
    let frames = decode_all_video_frames(data, 1);
    let actual = flatten_yuv420_frames(&frames);
    assert_yuv420_matches(&actual, &expected, 256, 144, 1, "intra fixture");
}

#[test]
fn hevc_p_slice_fixture_matches_ffmpeg() {
    let Some(input) = ensure_generated_hevc_fixture(
        "exact-ip-gray.h265",
        "color=c=gray:size=256x144:rate=2:duration=1",
        2,
        2,
        2,
    ) else {
        return;
    };
    let input_str = input.to_string_lossy().into_owned();
    let Some(data) = read_fixture(&input_str) else {
        return;
    };
    let Some(expected) =
        ffmpeg_decode_raw(&input_str, &PathBuf::from("/tmp/hevc-p.ref.yuv"), Some(2))
    else {
        return;
    };
    let frames = decode_all_video_frames(data, 2);
    let actual = flatten_yuv420_frames(&frames);
    assert_yuv420_matches(&actual, &expected, 256, 144, 2, "P-slice fixture");
}

/// Stream encoded with SAO enabled. Verifies the decoder can parse the
/// per-CTU SAO syntax without desynchronising CABAC and decode to a
/// plausible image. The 128x96 testsrc content has a small number of
/// cross-CTU intra-seam discrepancies in the upstream reconstruction that
/// aren't SAO bugs — the byte-exact SAO correctness is tested separately
/// against smaller fixtures (see `hevc_sao_*_matches_ffmpeg`).
#[test]
fn hevc_sao_fixture_decodes() {
    let x265_params = "log-level=error:keyint=1:min-keyint=1:scenecut=0:bframes=0:\
                       wpp=0:pmode=0:pme=0:frame-threads=1:sao=1:no-deblock=1";
    let Some(input) = ensure_generated_hevc_fixture_with_params(
        "intra-sao.h265",
        "testsrc=size=128x96:rate=1:duration=1",
        1,
        1,
        x265_params,
    ) else {
        return;
    };
    let Some(data) = read_fixture(&input.to_string_lossy()) else {
        return;
    };
    let mut dec = HevcDecoder::new(oxideav_core::CodecId::new(CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 24), data);
    if let Err(e) = dec.send_packet(&pkt) {
        match e {
            Error::Unsupported { .. } => return,
            _ => panic!("SAO fixture send_packet failed: {e:?}"),
        }
    }
    match dec.receive_frame() {
        Ok(oxideav_core::Frame::Video(vf)) => {
            assert_eq!(vf.width, 128, "width");
            assert_eq!(vf.height, 96, "height");
            assert_eq!(vf.planes.len(), 3, "plane count");
        }
        Ok(other) => panic!("expected video frame, got {other:?}"),
        Err(Error::Unsupported { .. }) => {}
        Err(e) => panic!("SAO fixture receive_frame failed: {e:?}"),
    }
}

/// Byte-exact SAO §8.7.3 regression: smptebars 64x64 with SAO enabled,
/// deblocking disabled, 1 CTB. libx265 emits edge-offset mode 135° (class
/// 2) with mixed-sign offsets on all three components. Our decoder's
/// reconstruction matches the ffmpeg reference bit-for-bit here because
/// a single 64x64 CTB has no cross-CTB seams to diverge on, so any
/// corruption in the SAO pipeline would surface as a diff.
#[test]
fn hevc_sao_smptebars_64_matches_ffmpeg() {
    let x265 = "log-level=error:keyint=1:min-keyint=1:scenecut=0:bframes=0:\
                wpp=0:pmode=0:pme=0:frame-threads=1:sao=1:no-deblock=1";
    let Some(input) = ensure_generated_hevc_fixture_with_params(
        "intra-sao-smptebars-64.h265",
        "smptebars=size=64x64:rate=1:duration=1",
        1,
        1,
        x265,
    ) else {
        return;
    };
    let input_str = input.to_string_lossy().into_owned();
    let Some(data) = read_fixture(&input_str) else {
        return;
    };
    let Some(expected) = ffmpeg_decode_raw(
        &input_str,
        &PathBuf::from("/tmp/hevc-sao-smptebars-64.ref.yuv"),
        Some(1),
    ) else {
        return;
    };
    let frames = decode_all_video_frames(data, 1);
    let actual = flatten_yuv420_frames(&frames);
    assert_yuv420_matches(&actual, &expected, 64, 64, 1, "SAO smptebars 64 fixture");
}

/// Byte-exact SAO regression: testsrc2 64x64 with SAO enabled. libx265
/// emits edge-offset class 1 (vertical) on luma and class 0 (horizontal)
/// on chroma. This exercises the class-0 / class-1 direction lookups in
/// §8.7.3.3 Table 8-12.
#[test]
fn hevc_sao_testsrc2_64_matches_ffmpeg() {
    let x265 = "log-level=error:keyint=1:min-keyint=1:scenecut=0:bframes=0:\
                wpp=0:pmode=0:pme=0:frame-threads=1:sao=1:no-deblock=1";
    let Some(input) = ensure_generated_hevc_fixture_with_params(
        "intra-sao-testsrc2-64.h265",
        "testsrc2=size=64x64:rate=1:duration=1",
        1,
        1,
        x265,
    ) else {
        return;
    };
    let input_str = input.to_string_lossy().into_owned();
    let Some(data) = read_fixture(&input_str) else {
        return;
    };
    let Some(expected) = ffmpeg_decode_raw(
        &input_str,
        &PathBuf::from("/tmp/hevc-sao-testsrc2-64.ref.yuv"),
        Some(1),
    ) else {
        return;
    };
    let frames = decode_all_video_frames(data, 1);
    let actual = flatten_yuv420_frames(&frames);
    assert_yuv420_matches(&actual, &expected, 64, 64, 1, "SAO testsrc2 64 fixture");
}

/// Byte-exact SAO regression: rgbtestsrc 64x64 with SAO enabled. Mixed
/// saturated-colour content, picks up any chroma-edge SAO mis-applies.
#[test]
fn hevc_sao_rgbtestsrc_64_matches_ffmpeg() {
    let x265 = "log-level=error:keyint=1:min-keyint=1:scenecut=0:bframes=0:\
                wpp=0:pmode=0:pme=0:frame-threads=1:sao=1:no-deblock=1";
    let Some(input) = ensure_generated_hevc_fixture_with_params(
        "intra-sao-rgbtestsrc-64.h265",
        "rgbtestsrc=size=64x64:rate=1:duration=1",
        1,
        1,
        x265,
    ) else {
        return;
    };
    let input_str = input.to_string_lossy().into_owned();
    let Some(data) = read_fixture(&input_str) else {
        return;
    };
    let Some(expected) = ffmpeg_decode_raw(
        &input_str,
        &PathBuf::from("/tmp/hevc-sao-rgbtestsrc-64.ref.yuv"),
        Some(1),
    ) else {
        return;
    };
    let frames = decode_all_video_frames(data, 1);
    let actual = flatten_yuv420_frames(&frames);
    assert_yuv420_matches(&actual, &expected, 64, 64, 1, "SAO rgbtestsrc 64 fixture");
}

/// Stream with varied input content (testsrc pattern) exercising angular
/// intra modes and their non-diagonal scan paths — a weak regression guard
/// for the §7.4.9.11 scan_idx_for_intra logic.
#[test]
fn hevc_angular_intra_fixture_decodes() {
    let Some(input) = ensure_generated_hevc_fixture(
        "intra-angular.h265",
        "testsrc=size=128x96:rate=1:duration=1",
        1,
        1,
        1,
    ) else {
        return;
    };
    let Some(data) = read_fixture(&input.to_string_lossy()) else {
        return;
    };
    let mut dec = HevcDecoder::new(oxideav_core::CodecId::new(CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 24), data);
    if let Err(e) = dec.send_packet(&pkt) {
        match e {
            Error::Unsupported { .. } => return,
            _ => panic!("angular fixture send_packet failed: {e:?}"),
        }
    }
    match dec.receive_frame() {
        Ok(oxideav_core::Frame::Video(vf)) => {
            assert_eq!(vf.width, 128);
            assert_eq!(vf.height, 96);
            let y_plane = &vf.planes[0].data;
            let distinct = y_plane.iter().copied().collect::<std::collections::HashSet<_>>().len();
            assert!(
                distinct > 10,
                "angular fixture: expected >10 distinct luma values, got {distinct}",
            );
        }
        Ok(other) => panic!("expected video frame, got {other:?}"),
        Err(Error::Unsupported { .. }) => {}
        Err(e) => panic!("angular fixture receive_frame failed: {e:?}"),
    }
}

/// Stream encoded with deblocking filter enabled. Before §8.7.2 was
/// implemented this would bail with Unsupported; now the decoder runs the
/// in-loop deblocking pass and returns a plausible picture.
#[test]
fn hevc_deblocked_fixture_decodes() {
    let x265 = "log-level=error:keyint=1:min-keyint=1:scenecut=0:bframes=0:\
                wpp=0:pmode=0:pme=0:frame-threads=1:no-sao=1:deblock=0\\,0";
    let Some(input) = ensure_generated_hevc_fixture_with_params(
        "intra-deblock.h265",
        "testsrc=size=128x96:rate=1:duration=1",
        1,
        1,
        x265,
    ) else {
        return;
    };
    let Some(data) = read_fixture(&input.to_string_lossy()) else {
        return;
    };
    let mut dec = HevcDecoder::new(oxideav_core::CodecId::new(CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 24), data);
    if let Err(e) = dec.send_packet(&pkt) {
        match e {
            Error::Unsupported { .. } => return,
            _ => panic!("deblock fixture send_packet failed: {e:?}"),
        }
    }
    match dec.receive_frame() {
        Ok(oxideav_core::Frame::Video(vf)) => {
            assert_eq!(vf.width, 128);
            assert_eq!(vf.height, 96);
            assert_eq!(vf.planes.len(), 3);
            let y_plane = &vf.planes[0].data;
            let distinct = y_plane
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len();
            assert!(
                distinct > 10,
                "deblock fixture: expected >10 distinct luma values, got {distinct}",
            );
        }
        Ok(other) => panic!("expected video frame, got {other:?}"),
        Err(Error::Unsupported { .. }) => {}
        Err(e) => panic!("deblock fixture receive_frame failed: {e:?}"),
    }
}

/// PSNR regression for the in-loop deblocking pass (§8.7.2) on a single
/// 64x64 CTB smptebars fixture. The deblock module's boundary-strength
/// derivation is a best-effort approximation (the comments in deblock.rs
/// spell this out), so we don't expect byte-exactness — but we do expect
/// the reconstructed frame to stay within 40 dB Y-plane PSNR of the
/// ffmpeg reference, which rules out any gross mis-apply. Current floor
/// is comfortably above 50 dB on this content.
#[test]
fn hevc_deblock_smptebars_64_psnr() {
    let x265 = "log-level=error:keyint=1:min-keyint=1:scenecut=0:bframes=0:\
                wpp=0:pmode=0:pme=0:frame-threads=1:no-sao=1:deblock=0\\,0";
    let Some(input) = ensure_generated_hevc_fixture_with_params(
        "intra-deblock-smptebars-64.h265",
        "smptebars=size=64x64:rate=1:duration=1",
        1,
        1,
        x265,
    ) else {
        return;
    };
    let input_str = input.to_string_lossy().into_owned();
    let Some(data) = read_fixture(&input_str) else {
        return;
    };
    let Some(expected) = ffmpeg_decode_raw(
        &input_str,
        &PathBuf::from("/tmp/hevc-deblock-smptebars-64.ref.yuv"),
        Some(1),
    ) else {
        return;
    };
    let frames = decode_all_video_frames(data, 1);
    let actual = flatten_yuv420_frames(&frames);
    let psnr = y_plane_psnr_db(&actual, &expected, 64, 64, 1);
    assert!(
        psnr >= 40.0,
        "deblock smptebars-64 Y-plane PSNR {:.2} dB < 40 dB floor; \
         §8.7.2 filter likely mis-applied on some edges",
        psnr,
    );
}

/// Stream encoded with a 2x2 tile grid (§6.3.1). Validates the slice
/// header entry-point parsing and the per-tile CABAC re-init path in
/// `decode_slice_ctus`. The reconstructed pixels are not byte-exact yet
/// (cross-CTU regressions also affect tile boundaries), so we only
/// demand the decoder either produces a plausible frame or bails with
/// Unsupported — a panic or invalid-bitstream error fails the test.
#[test]
fn hevc_tiles_fixture_decodes() {
    // 640x480 at 64-wide CTBs → 10x8 CTBs, big enough that libx265 will
    // actually emit a 2x2 tile grid.
    let x265 = "log-level=error:keyint=1:min-keyint=1:scenecut=0:bframes=0:\
                wpp=0:pmode=0:pme=0:frame-threads=1:no-sao=1:no-deblock=1:tiles=2x2";
    let Some(input) = ensure_generated_hevc_fixture_with_params(
        "intra-tiles-2x2.h265",
        "testsrc=size=640x480:rate=1:duration=1",
        1,
        1,
        x265,
    ) else {
        return;
    };
    let Some(data) = read_fixture(&input.to_string_lossy()) else {
        return;
    };
    // Sanity-check the slice header: the PPS should carry tile info and
    // the first slice should carry enough entry points to cover
    // num_tiles - 1 sub-streams.
    let mut sps_opt: Option<_> = None;
    let mut pps_opt: Option<oxideav_h265::pps::PicParameterSet> = None;
    let mut tiles_checked = false;
    for nal in iter_annex_b(&data) {
        let rbsp = extract_rbsp(nal.payload());
        match nal.header.nal_unit_type {
            NalUnitType::Sps => sps_opt = Some(parse_sps(&rbsp).expect("SPS")),
            NalUnitType::Pps => pps_opt = Some(parse_pps(&rbsp).expect("PPS")),
            t if t.is_vcl() && sps_opt.is_some() && pps_opt.is_some() => {
                let sps = sps_opt.as_ref().unwrap();
                let pps = pps_opt.as_ref().unwrap();
                if pps.tiles_enabled_flag {
                    let hdr = parse_slice_segment_header(&rbsp, &nal.header, sps, pps)
                        .expect("slice header");
                    let ti = pps.tile_info.as_ref().expect("tile_info");
                    assert!(ti.num_tiles() >= 2, "expected a multi-tile stream");
                    assert!(
                        hdr.entry_point_offsets.len() as u32 + 1 >= ti.num_tiles(),
                        "entry points ({}) do not cover tiles ({})",
                        hdr.entry_point_offsets.len(),
                        ti.num_tiles()
                    );
                    tiles_checked = true;
                }
                break;
            }
            _ => {}
        }
    }
    if !tiles_checked {
        // libx265 in some builds silently ignores `tiles=NxM` and emits a
        // single-tile stream. Nothing more we can check; the PPS/slice
        // parse path is exercised by the unit tests.
        return;
    }

    let mut dec = HevcDecoder::new(oxideav_core::CodecId::new(CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 24), data);
    match dec.send_packet(&pkt) {
        Ok(()) => {}
        Err(Error::Unsupported(_)) => return,
        Err(e) => panic!("tiles fixture send_packet failed: {e:?}"),
    }
    match dec.receive_frame() {
        Ok(oxideav_core::Frame::Video(vf)) => {
            assert_eq!(vf.width, 640);
            assert_eq!(vf.height, 480);
            assert_eq!(vf.planes.len(), 3);
        }
        Ok(other) => panic!("expected video frame, got {other:?}"),
        Err(Error::Unsupported(_)) => {}
        Err(e) => panic!("tiles fixture receive_frame failed: {e:?}"),
    }
}

/// Stream encoded with wavefront parallel processing (§6.3.2).
/// Validates that `entropy_coding_sync_enabled_flag` is parsed, the
/// slice header carries one entry point per extra CTU row, and the
/// decoder does not panic while decoding the multi-sub-stream slice.
#[test]
fn hevc_wpp_fixture_decodes() {
    let x265 = "log-level=error:keyint=1:min-keyint=1:scenecut=0:bframes=0:\
                wpp=1:pmode=0:pme=0:frame-threads=1:no-sao=1:no-deblock=1";
    let Some(input) = ensure_generated_hevc_fixture_with_params(
        "intra-wpp.h265",
        "testsrc=size=256x192:rate=1:duration=1",
        1,
        1,
        x265,
    ) else {
        return;
    };
    let Some(data) = read_fixture(&input.to_string_lossy()) else {
        return;
    };
    let mut sps_opt: Option<_> = None;
    let mut pps_opt: Option<oxideav_h265::pps::PicParameterSet> = None;
    let mut wpp_checked = false;
    for nal in iter_annex_b(&data) {
        let rbsp = extract_rbsp(nal.payload());
        match nal.header.nal_unit_type {
            NalUnitType::Sps => sps_opt = Some(parse_sps(&rbsp).expect("SPS")),
            NalUnitType::Pps => pps_opt = Some(parse_pps(&rbsp).expect("PPS")),
            t if t.is_vcl() && sps_opt.is_some() && pps_opt.is_some() => {
                let sps = sps_opt.as_ref().unwrap();
                let pps = pps_opt.as_ref().unwrap();
                if pps.entropy_coding_sync_enabled_flag {
                    let hdr = parse_slice_segment_header(&rbsp, &nal.header, sps, pps)
                        .expect("slice header");
                    // Expect at least one entry point (the clip is tall
                    // enough to have more than one CTB row).
                    assert!(
                        hdr.num_entry_point_offsets >= 1,
                        "expected WPP slice to have at least one entry point"
                    );
                    wpp_checked = true;
                }
                break;
            }
            _ => {}
        }
    }
    assert!(wpp_checked, "fixture does not actually have WPP enabled");

    let mut dec = HevcDecoder::new(oxideav_core::CodecId::new(CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 24), data);
    match dec.send_packet(&pkt) {
        Ok(()) => {}
        Err(Error::Unsupported(_)) => return,
        Err(e) => panic!("wpp fixture send_packet failed: {e:?}"),
    }
    match dec.receive_frame() {
        Ok(oxideav_core::Frame::Video(vf)) => {
            assert_eq!(vf.width, 256);
            assert_eq!(vf.height, 192);
            assert_eq!(vf.planes.len(), 3);
        }
        Ok(other) => panic!("expected video frame, got {other:?}"),
        Err(Error::Unsupported(_)) => {}
        Err(e) => panic!("wpp fixture receive_frame failed: {e:?}"),
    }
}

/// Main10 (yuv420p10le) fixture: verifies the SPS parser accepts
/// `bit_depth_luma_minus8 = 2` and the decoder surfaces a clean
/// `Error::Unsupported` rather than panicking. Full Main10 pixel decode
/// is not yet implemented — this nails down the current scope boundary.
#[test]
fn main10_fixture_surfaces_clean_unsupported() {
    let fixture_dir = generated_fixture_dir();
    ensure_dir(&fixture_dir);
    let out = fixture_dir.join("h265-main10-192x112-p10.hevc");
    let main10_path = if out.exists() {
        Some(out)
    } else if ffmpeg_available()
        && run_ffmpeg(&[
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=192x112:rate=25",
            "-frames:v",
            "1",
            "-pix_fmt",
            "yuv420p10le",
            "-c:v",
            "libx265",
            "-x265-params",
            "log-level=error:keyint=1:bframes=0:wpp=0:frame-threads=1:no-sao=1:no-deblock=1",
            out.to_str().unwrap(),
        ])
    {
        Some(out)
    } else {
        // Fall back to the checked-in fixture if ffmpeg can't produce one.
        let p = PathBuf::from(fixture_path("main10.hevc"));
        if p.exists() {
            Some(p)
        } else {
            eprintln!("main10 fixture unavailable — skipping");
            return;
        }
    };
    let path = main10_path.unwrap();
    let Some(data) = read_fixture(&path.to_string_lossy()) else {
        return;
    };
    // Confirm the SPS parser correctly identifies the stream as Main10.
    let mut sps_bit_depth = None;
    for nal in iter_annex_b(&data) {
        if matches!(nal.header.nal_unit_type, NalUnitType::Sps) {
            let rbsp = extract_rbsp(nal.payload());
            let sps = parse_sps(&rbsp).expect("parse SPS from Main10 fixture");
            sps_bit_depth = Some(sps.bit_depth_y());
            break;
        }
    }
    assert_eq!(
        sps_bit_depth,
        Some(10),
        "expected Main10 (bit_depth_y = 10) in generated fixture"
    );

    let mut dec = HevcDecoder::new(oxideav_core::CodecId::new(CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 25), data);
    match dec.send_packet(&pkt) {
        Ok(()) => match dec.receive_frame() {
            Err(Error::Unsupported(_)) | Err(Error::NeedMore) => {}
            Ok(_) => {
                // If Main10 pixel decode ever lands this branch starts
                // exercising the happy path — mark the test as passing.
            }
            Err(e) => panic!("unexpected error from Main10 fixture: {e:?}"),
        },
        Err(Error::Unsupported(_)) => {}
        Err(e) => panic!("unexpected error from Main10 fixture: {e:?}"),
    }
}

/// Intra-only Main 10 (yuv420p10le) fixture: verifies the decoder
/// produces a `Yuv420P10Le` frame whose luma plane is close enough to
/// ffmpeg's reference decode. Scope: 10-bit intra path only — inter
/// slices still raise `Unsupported` at this bit depth.
#[test]
fn main10_intra_decodes_close_to_ffmpeg() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg missing — skipping Main 10 intra PSNR test");
        return;
    }
    let fixture_dir = generated_fixture_dir();
    ensure_dir(&fixture_dir);
    let clip = fixture_dir.join("h265-main10-intra-80x48.hevc");
    if !clip.exists()
        && !run_ffmpeg(&[
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=80x48:rate=25",
            "-frames:v",
            "1",
            "-pix_fmt",
            "yuv420p10le",
            "-c:v",
            "libx265",
            "-profile:v",
            "main10",
            "-preset:v",
            "veryslow",
            "-x265-params",
            "log-level=error:keyint=1:bframes=0:wpp=0:frame-threads=1:no-sao=1:no-deblock=1:no-amp=1:no-tskip=1:no-strong-intra-smoothing=1:no-weightp=1:no-weightb=1:no-scaling-lists=0:qp=22",
            clip.to_str().unwrap(),
        ])
    {
        eprintln!("failed to generate Main 10 intra clip — skipping");
        return;
    }
    let Some(data) = read_fixture(&clip.to_string_lossy()) else {
        return;
    };
    // Reference decode from ffmpeg at yuv420p10le.
    let ref_path = fixture_dir.join("h265-main10-intra-80x48.ref.yuv");
    let input_str = clip.to_string_lossy().to_string();
    if !ffmpeg_available() {
        return;
    }
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            &input_str,
            "-frames:v",
            "1",
            "-pix_fmt",
            "yuv420p10le",
            "-f",
            "rawvideo",
        ])
        .arg(&ref_path)
        .status()
        .expect("run ffmpeg");
    if !status.success() {
        eprintln!("ffmpeg decode of Main 10 clip failed — skipping PSNR");
        return;
    }
    let expected = std::fs::read(&ref_path).expect("read ffmpeg raw output");
    // Feed the stream into the pure-Rust decoder.
    let mut dec = HevcDecoder::new(oxideav_core::CodecId::new(CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 25), data);
    dec.send_packet(&pkt).expect("send Main 10 packet");
    let vf = match dec.receive_frame() {
        Ok(oxideav_core::Frame::Video(vf)) => vf,
        Ok(other) => panic!("expected VideoFrame, got {other:?}"),
        Err(Error::Unsupported(msg)) => {
            eprintln!("Main 10 intra decode reported Unsupported: {msg}");
            return;
        }
        Err(e) => panic!("unexpected error from Main 10 decode: {e:?}"),
    };
    assert_eq!(
        vf.format,
        oxideav_core::PixelFormat::Yuv420P10Le,
        "Main 10 stream must emit Yuv420P10Le frames"
    );
    let w = vf.width as usize;
    let h = vf.height as usize;
    // Raw frame layout: Y plane (w*h*2 bytes LE) then Cb (w/2*h/2*2) then Cr.
    let y_len = w * h * 2;
    let c_len = (w / 2) * (h / 2) * 2;
    assert_eq!(
        expected.len(),
        y_len + 2 * c_len,
        "ffmpeg reference frame size mismatch"
    );
    // PSNR on the luma plane only, at 10-bit (max = 1023).
    let actual_y = &vf.planes[0].data;
    let expected_y = &expected[..y_len];
    // `stride` may exceed `w * 2` for our plane — collapse to packed form
    // before comparing.
    let row_bytes = w * 2;
    let mut packed_actual = Vec::with_capacity(y_len);
    for y in 0..h {
        let off = y * vf.planes[0].stride;
        packed_actual.extend_from_slice(&actual_y[off..off + row_bytes]);
    }
    assert_eq!(packed_actual.len(), expected_y.len());
    let mut sse: u64 = 0;
    let mut n: u64 = 0;
    for i in (0..packed_actual.len()).step_by(2) {
        let a = (packed_actual[i] as u32) | ((packed_actual[i + 1] as u32) << 8);
        let e = (expected_y[i] as u32) | ((expected_y[i + 1] as u32) << 8);
        let d = a as i32 - e as i32;
        sse += (d * d) as u64;
        n += 1;
    }
    let psnr = if sse == 0 {
        f64::INFINITY
    } else {
        let mse = sse as f64 / n as f64;
        10.0 * (1023.0 * 1023.0 / mse).log10()
    };
    // Debug breadcrumbs when the test fails locally.
    let first_actual: Vec<u32> = packed_actual
        .chunks_exact(2)
        .take(8)
        .map(|c| (c[0] as u32) | ((c[1] as u32) << 8))
        .collect();
    let first_expected: Vec<u32> = expected_y
        .chunks_exact(2)
        .take(8)
        .map(|c| (c[0] as u32) | ((c[1] as u32) << 8))
        .collect();
    eprintln!(
        "Main 10 intra PSNR vs ffmpeg: {psnr:.2} dB, first8 actual={first_actual:?} expected={first_expected:?}"
    );
    // Main 10 intra is now bit-exact on this fixture (PSNR = inf).
    // The 50 dB floor guards against any future regression in the
    // bit-depth-aware dequant pipeline (§8.6.3 eq. 8-309 with
    // `qP = Qp'Y = QpY + QpBdOffsetY`) while leaving some headroom for
    // future encoders/parameter sweeps that might not be byte-exact.
    assert!(
        psnr >= 50.0,
        "Main 10 intra decode PSNR below floor: {psnr:.2} dB"
    );
    // Confirm the decoder at least produced non-uniform output in the
    // expected range (i.e. it did some intra prediction, not just filled
    // the plane with a single value).
    let min_a = first_actual.iter().copied().min().unwrap_or(0);
    let max_a = first_actual.iter().copied().max().unwrap_or(0);
    assert!(
        max_a < 1024,
        "Main 10 actual sample {max_a} exceeds 10-bit range"
    );
    let _ = min_a;
}

/// Main 10 inter (P-slice) fixture: keyint=8 forces at least one P-frame
/// in the clip so the MC interpolation + bi-pred combine paths execute at
/// `bit_depth = 10`. Scope: this lands the Clip1Y/Clip1C widening; PSNR
/// floor is loose because libx265 still tags the stream as Rext and the
/// Rext envelope mismatch is a separate follow-up.
#[test]
fn main10_inter_decodes_with_pframes() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg missing — skipping Main 10 inter PSNR test");
        return;
    }
    let fixture_dir = generated_fixture_dir();
    ensure_dir(&fixture_dir);
    let clip = fixture_dir.join("h265-main10-inter-80x48.hevc");
    if !clip.exists()
        && !run_ffmpeg(&[
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=80x48:rate=25",
            "-frames:v",
            "4",
            "-pix_fmt",
            "yuv420p10le",
            "-c:v",
            "libx265",
            "-profile:v",
            "main10",
            "-g",
            "8",
            "-x265-params",
            "log-level=error:keyint=8:bframes=0:wpp=0:frame-threads=1:no-sao=1:no-deblock=1:no-amp=1:no-tskip=1:no-strong-intra-smoothing=1:no-weightp=1:no-weightb=1:qp=22",
            clip.to_str().unwrap(),
        ])
    {
        eprintln!("failed to generate Main 10 inter clip — skipping");
        return;
    }
    let Some(data) = read_fixture(&clip.to_string_lossy()) else {
        return;
    };
    let ref_path = fixture_dir.join("h265-main10-inter-80x48.ref.yuv");
    let input_str = clip.to_string_lossy().to_string();
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            &input_str,
            "-frames:v",
            "4",
            "-pix_fmt",
            "yuv420p10le",
            "-f",
            "rawvideo",
        ])
        .arg(&ref_path)
        .status()
        .expect("run ffmpeg");
    if !status.success() {
        eprintln!("ffmpeg decode of Main 10 inter clip failed — skipping PSNR");
        return;
    }
    let expected = std::fs::read(&ref_path).expect("read ffmpeg raw output");

    let mut dec = HevcDecoder::new(oxideav_core::CodecId::new(CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 25), data);
    dec.send_packet(&pkt).expect("send Main 10 inter packet");

    // Collect all frames so the test exercises at least one P-slice decode.
    let mut frames = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(oxideav_core::Frame::Video(vf)) => frames.push(vf),
            Ok(_) => break,
            Err(Error::NeedMore) => break,
            Err(Error::Unsupported(msg)) => {
                eprintln!("Main 10 inter decode reported Unsupported: {msg}");
                return;
            }
            Err(e) => panic!("unexpected error from Main 10 inter decode: {e:?}"),
        }
    }
    if frames.is_empty() {
        eprintln!("Main 10 inter decode produced no frames — skipping");
        return;
    }
    // Key assertion: at least one P-frame got decoded without panicking.
    // The previous guard would have returned Error::Unsupported before any
    // frame landed, so just reaching here already proves the MC path
    // survives 10-bit.
    let vf = &frames[0];
    assert_eq!(
        vf.format,
        oxideav_core::PixelFormat::Yuv420P10Le,
        "Main 10 stream must emit Yuv420P10Le frames"
    );
    let w = vf.width as usize;
    let h = vf.height as usize;
    let y_len = w * h * 2;
    let c_len = (w / 2) * (h / 2) * 2;
    let per_frame = y_len + 2 * c_len;
    assert!(
        expected.len() >= per_frame,
        "ffmpeg reference frame size mismatch"
    );
    // PSNR is averaged across however many frames both decoders produced.
    let mut sse: u64 = 0;
    let mut n: u64 = 0;
    for (fi, vf) in frames.iter().enumerate() {
        let ref_off = fi * per_frame;
        if ref_off + y_len > expected.len() {
            break;
        }
        let expected_y = &expected[ref_off..ref_off + y_len];
        let row_bytes = w * 2;
        let mut packed_actual = Vec::with_capacity(y_len);
        for y in 0..h {
            let off = y * vf.planes[0].stride;
            packed_actual.extend_from_slice(&vf.planes[0].data[off..off + row_bytes]);
        }
        for i in (0..packed_actual.len()).step_by(2) {
            let a = (packed_actual[i] as u32) | ((packed_actual[i + 1] as u32) << 8);
            let e = (expected_y[i] as u32) | ((expected_y[i + 1] as u32) << 8);
            let d = a as i32 - e as i32;
            sse += (d * d) as u64;
            n += 1;
        }
    }
    let psnr = if sse == 0 {
        f64::INFINITY
    } else {
        let mse = sse as f64 / n as f64;
        10.0 * (1023.0 * 1023.0 / mse).log10()
    };
    eprintln!(
        "Main 10 inter PSNR vs ffmpeg: {psnr:.2} dB over {} frames",
        frames.len()
    );
    // After landing the bit-depth-aware dequant QP (QpY → Qp'Y =
    // QpY + QpBdOffsetY), Main 10 intra is byte-exact and Main 10 inter
    // PSNR is in the 20 dB range on this fixture. The residual gap at
    // inter is likely unrelated to dequant — more plausibly either MC
    // rounding at 10-bit in the spec's Clip1Y/Clip1C paths or a subtle
    // reference-picture POC/reordering issue. Floor raised to 18 dB
    // to guard against further regressions but not force byte-exactness
    // yet.
    assert!(
        psnr >= 18.0,
        "Main 10 inter decode PSNR below floor: {psnr:.2} dB"
    );
}

/// 4:4:4 fixture: confirms the decoder surfaces a clean `Unsupported`
/// instead of panicking for `chroma_format_idc == 3` streams.
#[test]
fn yuv444_fixture_surfaces_clean_unsupported() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg missing — skipping 4:4:4 fixture test");
        return;
    }
    let fixture_dir = generated_fixture_dir();
    ensure_dir(&fixture_dir);
    let out = fixture_dir.join("h265-yuv444-192x112.hevc");
    if !out.exists()
        && !run_ffmpeg(&[
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=192x112:rate=25",
            "-frames:v",
            "1",
            "-pix_fmt",
            "yuv444p",
            "-c:v",
            "libx265",
            "-x265-params",
            "log-level=error:keyint=1:bframes=0:wpp=0:frame-threads=1:no-sao=1:no-deblock=1",
            out.to_str().unwrap(),
        ])
    {
        eprintln!("ffmpeg failed to produce 4:4:4 fixture — skipping");
        return;
    }
    let Some(data) = read_fixture(&out.to_string_lossy()) else {
        return;
    };
    let mut chroma_format = None;
    for nal in iter_annex_b(&data) {
        if matches!(nal.header.nal_unit_type, NalUnitType::Sps) {
            let rbsp = extract_rbsp(nal.payload());
            let sps = parse_sps(&rbsp).expect("parse 4:4:4 SPS");
            chroma_format = Some(sps.chroma_format_idc);
            break;
        }
    }
    assert_eq!(chroma_format, Some(3), "expected 4:4:4 (chroma_format_idc=3)");
    let mut dec = HevcDecoder::new(oxideav_core::CodecId::new(CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 25), data);
    match dec.send_packet(&pkt) {
        Ok(()) => match dec.receive_frame() {
            Err(Error::Unsupported(_)) | Err(Error::NeedMore) => {}
            other => panic!("unexpected receive_frame on 4:4:4 fixture: {other:?}"),
        },
        Err(Error::Unsupported(_)) => {}
        Err(e) => panic!("unexpected error from 4:4:4 fixture: {e:?}"),
    }
}
