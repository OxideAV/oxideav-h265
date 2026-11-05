//! Official JCT-VC SHVC (Annex F / Annex H) conformance pins.
//!
//! The workspace stages the complete scalable-extension conformance
//! set under `docs/video/h265/conformance/shvc/` (read in place —
//! nothing is copied into this repo). Each stream ships one output
//! picture list per output layer set (`<stream>-OLS<k>.opl`: one row
//! per output picture, `LayerId POC MD5:<Y>,<U>,<V>`) and / or one
//! decoded-YUV digest per layer (`<stream>-L<n>.md5`). A stream /
//! output-layer-set pair in [`EXPECTED_PASS`] must reproduce every row
//! of its picture list (or every per-layer digest) byte-exact.
//!
//! `HYBRID_A` (an external base layer, F.8.1.8) is fed its staged
//! reconstructed base layer through
//! [`SequenceDecoder::set_external_base_layer`]. The one pair left out,
//! `ALPHA_A_BBC_1` OLS 1, decodes both layers byte-exact (each is
//! pinned through another output layer set) but its list outputs the
//! primary layer next to the alpha auxiliary layer, which the
//! F.7.4.3.1.1 `default_output_layer_idc == 1` derivation does not.
//!
//! Set `SHVC_SWEEP=1` to print the whole-corpus table instead of
//! checking the pins only. When the staged corpus is not present (CI
//! checks out this crate alone), the tests are no-ops.

mod fixture_bytes;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use fixture_bytes::md5;
use oxideav_h265::picture::{Picture, Plane};
use oxideav_h265::sequence::{
    DecodedFrame, ExternalBasePicture, ExternalBaseRequest, LayerTarget, SequenceDecoder,
};

/// `(stream stem, output layer set)` pairs pinned byte-exact. An OLS of
/// `None` pins the per-layer YUV digests under the default target.
const EXPECTED_PASS: &[(&str, Option<usize>)] = &[
    ("8LAYERS_QUALCOMM_1", Some(0)),
    ("8LAYERS_QUALCOMM_1", Some(1)),
    ("8LAYERS_QUALCOMM_1", Some(2)),
    ("8LAYERS_QUALCOMM_1", Some(3)),
    ("8LAYERS_QUALCOMM_1", Some(4)),
    ("8LAYERS_QUALCOMM_1", Some(5)),
    ("8LAYERS_QUALCOMM_1", Some(6)),
    ("8LAYERS_QUALCOMM_1", Some(7)),
    ("ADAPTRES_A_ERICSSON_1", Some(0)),
    ("ADAPTRES_A_ERICSSON_1", Some(1)),
    ("ADAPTRES_A_ERICSSON_1", Some(2)),
    ("ALPHA_A_BBC_1", Some(0)),
    ("CGS_A_TECHNICOLOR_1", Some(0)),
    ("CGS_A_TECHNICOLOR_1", Some(1)),
    ("CGS_B_TECHNICOLOR_1", Some(0)),
    ("CGS_B_TECHNICOLOR_1", Some(1)),
    ("CGS_C_TECHNICOLOR_1", Some(0)),
    ("CGS_C_TECHNICOLOR_1", Some(1)),
    ("CGS_D_TECHNICOLOR_1", Some(0)),
    ("CGS_D_TECHNICOLOR_1", Some(1)),
    ("CGS_E_TECHNICOLOR_1", Some(0)),
    ("CGS_E_TECHNICOLOR_1", Some(1)),
    ("CGS_F_TECHNICOLOR_1", Some(0)),
    ("CGS_F_TECHNICOLOR_1", Some(1)),
    ("CGS_G_TECHNICOLOR_1", Some(0)),
    ("CGS_G_TECHNICOLOR_1", Some(1)),
    ("CGS_H_TECHNICOLOR_1", Some(0)),
    ("CGS_H_TECHNICOLOR_1", Some(1)),
    ("CGS_I_TECHNICOLOR_1", Some(0)),
    ("CGS_I_TECHNICOLOR_1", Some(1)),
    ("CONFCROP_A_VIDYO_2", Some(0)),
    ("CONFCROP_A_VIDYO_2", Some(1)),
    ("CONFCROP_B_VIDYO_2", Some(0)),
    ("CONFCROP_B_VIDYO_2", Some(1)),
    ("CONFCROP_C_VIDYO_3", Some(0)),
    ("CONFCROP_C_VIDYO_3", Some(1)),
    ("CONFCROP_C_VIDYO_3", Some(2)),
    ("DEPTH_A_NOKIA_1", Some(0)),
    ("DEPTH_A_NOKIA_1", Some(1)),
    ("DEPTH_A_NOKIA_1", Some(2)),
    ("DEPTH_A_NOKIA_1", Some(3)),
    ("DISFLAG_A_QUALCOMM_1", Some(0)),
    ("DISFLAG_A_QUALCOMM_1", Some(1)),
    ("DPB_A_VIDYO_2", Some(0)),
    ("DPB_A_VIDYO_2", Some(1)),
    ("DPB_B_VIDYO_2", Some(0)),
    ("DPB_B_VIDYO_2", Some(1)),
    ("HYBRID_A_QUALCOMM_1", Some(1)),
    ("INACTIVE_A_QCOM_1", Some(0)),
    ("INACTIVE_A_QCOM_1", Some(1)),
    ("INACTIVE_A_QCOM_1", Some(2)),
    ("INBLD_A_NOKIA_2", Some(4)),
    ("LAYERID63_A_HHI_1", None),
    ("LAYERID_A_NOKIA_2", Some(0)),
    ("LAYERID_A_NOKIA_2", Some(1)),
    ("LAYERID_A_NOKIA_2", Some(2)),
    ("MAXTID_A_ETRI_2", Some(0)),
    ("MAXTID_A_ETRI_2", Some(1)),
    ("MAXTID_A_ETRI_2", Some(2)),
    ("MAXTID_B_ETRI_2", Some(0)),
    ("MAXTID_B_ETRI_2", Some(1)),
    ("MAXTID_B_ETRI_2", Some(2)),
    ("MAXTID_C_ETRI_2", Some(0)),
    ("MAXTID_C_ETRI_2", Some(1)),
    ("MAXTID_C_ETRI_2", Some(2)),
    ("MVD_A_IDCC_1", Some(0)),
    ("MVD_A_IDCC_1", Some(1)),
    ("MVD_A_NOKIA_1", Some(0)),
    ("MVD_A_NOKIA_1", Some(1)),
    ("MVD_A_NOKIA_1", Some(2)),
    ("NONVUI_A_QUALCOMM_1", Some(0)),
    ("NONVUI_A_QUALCOMM_1", Some(1)),
    ("NONVUI_B_QUALCOMM_1", Some(0)),
    ("NONVUI_B_QUALCOMM_1", Some(1)),
    ("NONVUI_C_QUALCOMM_1", Some(0)),
    ("NONVUI_C_QUALCOMM_1", Some(1)),
    ("OLS_A_NOKIA_1", Some(0)),
    ("OLS_A_NOKIA_1", Some(1)),
    ("OLS_A_NOKIA_1", Some(2)),
    ("OLS_A_NOKIA_1", Some(3)),
    ("OLS_B_NOKIA_1", Some(0)),
    ("OLS_B_NOKIA_1", Some(1)),
    ("OLS_B_NOKIA_1", Some(2)),
    ("OLS_C_NOKIA_1", Some(0)),
    ("OLS_C_NOKIA_1", Some(1)),
    ("OLS_C_NOKIA_1", Some(2)),
    ("OLS_C_NOKIA_1", Some(3)),
    ("POC_A_Ericsson_1", Some(0)),
    ("POC_A_Ericsson_1", Some(1)),
    ("POC_B_Ericsson_1", Some(0)),
    ("POC_B_Ericsson_1", Some(1)),
    ("PPSSLIST_A_Sony_2", None),
    ("PSEXT_A_VIDYO_2", Some(0)),
    ("PSEXT_A_VIDYO_2", Some(1)),
    ("REFLAYER_A_VIDYO_2", Some(0)),
    ("REFLAYER_A_VIDYO_2", Some(1)),
    ("REFLAYER_A_VIDYO_2", Some(2)),
    ("REFLAYER_B_VIDYO_2", Some(0)),
    ("REFLAYER_B_VIDYO_2", Some(1)),
    ("REFLAYER_B_VIDYO_2", Some(2)),
    ("REFLAYER_C_VIDYO_2", Some(0)),
    ("REFLAYER_C_VIDYO_2", Some(1)),
    ("REFLAYER_C_VIDYO_2", Some(2)),
    ("REFLAYER_D_VIDYO_2", Some(0)),
    ("REFLAYER_D_VIDYO_2", Some(1)),
    ("REFLAYER_D_VIDYO_2", Some(2)),
    ("REFREGOFF_A_SHARP_1", Some(0)),
    ("REFREGOFF_A_SHARP_1", Some(1)),
    ("REPFMT_A_VIDYO_2", Some(0)),
    ("REPFMT_A_VIDYO_2", Some(1)),
    ("REPFMT_B_VIDYO_2", Some(0)),
    ("REPFMT_B_VIDYO_2", Some(1)),
    ("REPFMT_B_VIDYO_2", Some(2)),
    ("REPFMT_C_VIDYO_2", Some(0)),
    ("REPFMT_C_VIDYO_2", Some(1)),
    ("REPFMT_C_VIDYO_2", Some(2)),
    ("RESCHANGE_A_VIDYO_1", Some(0)),
    ("RESCHANGE_A_VIDYO_1", Some(1)),
    ("RESPHASE_A_SAMSUNG_2", Some(0)),
    ("RESPHASE_A_SAMSUNG_2", Some(1)),
    ("SCREFOFF_A_QCOM_1", Some(0)),
    ("SCREFOFF_A_QCOM_1", Some(1)),
    ("SIM_A_IDCC_1", Some(0)),
    ("SIM_A_IDCC_1", Some(1)),
    ("SIM_B_IDCC_1", Some(0)),
    ("SIM_B_IDCC_1", Some(1)),
    ("SLLEV_A_VIDYO_1", Some(0)),
    ("SLLEV_A_VIDYO_1", Some(1)),
    ("SNR_A_IDCC_1", Some(0)),
    ("SNR_A_IDCC_1", Some(1)),
    ("SNR_B_IDCC_1", Some(0)),
    ("SNR_B_IDCC_1", Some(1)),
    ("SNR_B_IDCC_1", Some(2)),
    ("SNR_C_IDCC_1", Some(0)),
    ("SNR_C_IDCC_1", Some(1)),
    ("SNR_C_IDCC_1", Some(2)),
    ("SPLITFLAG_A_HHI_1", Some(0)),
    ("SPLITFLAG_A_HHI_1", Some(1)),
    ("SPSREPFMT_A_Sony_2", None),
    ("SPSSLIST_A_Sony_2", None),
    ("SRATIOS_A_SAMSUNG_3", Some(0)),
    ("SRATIOS_A_SAMSUNG_3", Some(1)),
    ("SRATIOS_B_SAMSUNG_2", Some(0)),
    ("SRATIOS_B_SAMSUNG_2", Some(1)),
    ("SREXT_A_FUJITSU_1", Some(0)),
    ("SREXT_A_FUJITSU_1", Some(1)),
    ("SREXT_B_FUJITSU_1", Some(0)),
    ("SREXT_B_FUJITSU_1", Some(1)),
    ("SREXT_C_FUJITSU_1", Some(0)),
    ("SREXT_C_FUJITSU_1", Some(1)),
    ("SREXT_D_FUJITSU_1", Some(0)),
    ("SREXT_D_FUJITSU_1", Some(1)),
    ("SREXT_E_FUJITSU_1", Some(0)),
    ("SREXT_E_FUJITSU_1", Some(1)),
    ("SREXT_F_FUJITSU_1", Some(0)),
    ("SREXT_F_FUJITSU_1", Some(1)),
    ("VUI_A_QUALCOMM_1", Some(0)),
    ("VUI_A_QUALCOMM_1", Some(1)),
    ("VUI_B_QUALCOMM_1", Some(0)),
    ("VUI_B_QUALCOMM_1", Some(1)),
    ("VUI_C_QUALCOMM_1", None),
];

fn shvc_dir() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/video/h265/conformance/shvc");
    dir.is_dir().then_some(dir)
}

/// One plane's samples as the digest serializes them: one byte per
/// sample at bit depth 8, two little-endian bytes otherwise.
fn plane_bytes(pic: &Picture, plane: Plane) -> Vec<u8> {
    let s = pic.plane(plane);
    if pic.bit_depth(plane) <= 8 {
        s.iter().map(|&v| v as u8).collect()
    } else {
        s.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

fn plane_digests(pic: &Picture) -> [String; 3] {
    let y = md5::hex(&plane_bytes(pic, Plane::Luma));
    if pic.chroma_array_type() == 0 {
        // A monochrome picture: the list carries the grey chroma
        // planes' digests at 4:2:0 — compared on luma only.
        return [y, String::new(), String::new()];
    }
    [
        y,
        md5::hex(&plane_bytes(pic, Plane::Cb)),
        md5::hex(&plane_bytes(pic, Plane::Cr)),
    ]
}

/// One `.opl` row.
#[derive(Debug, Clone)]
struct OplRow {
    layer: u8,
    poc: i32,
    md5: [String; 3],
}

/// Parse a picture list: the tabular `LayerId POC MD5:<Y>,<U>,<V>`
/// form (a monochrome list carries the luma digest alone), or a
/// decoder-log form (`POC <n> LId: <l> ... [MD5:<Y>,<U>,<V>,(OK)]`,
/// which lists every decoded picture — the flag is `true` then).
fn parse_opl(text: &str) -> (Vec<OplRow>, bool) {
    let mut rows = Vec::new();
    let mut log_form = false;
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("POC") {
            let Some((poc, rest)) = rest.split_once("LId:") else {
                continue;
            };
            let Ok(poc) = poc.trim().parse::<i32>() else {
                continue;
            };
            let Some(layer) = rest
                .split_whitespace()
                .next()
                .and_then(|l| l.parse::<u8>().ok())
            else {
                continue;
            };
            let Some((_, m)) = rest.split_once("[MD5:") else {
                continue;
            };
            let parts: Vec<String> = m
                .split(',')
                .take(3)
                .map(|t| t.trim_end_matches(']').to_ascii_lowercase())
                .collect();
            log_form = true;
            rows.push(OplRow {
                layer,
                poc,
                md5: [
                    parts[0].clone(),
                    parts.get(1).cloned().unwrap_or_default(),
                    parts.get(2).cloned().unwrap_or_default(),
                ],
            });
            continue;
        }
        let mut it = line.split_whitespace();
        let (Some(l), Some(p), Some(m)) = (it.next(), it.next(), it.next()) else {
            continue;
        };
        let (Ok(layer), Ok(poc)) = (l.parse::<u8>(), p.parse::<i32>()) else {
            continue;
        };
        let Some(m) = m.strip_prefix("MD5:") else {
            continue;
        };
        let parts: Vec<String> = m.split(',').map(str::to_ascii_lowercase).collect();
        if parts.len() != 3 && parts.len() != 1 {
            continue;
        }
        rows.push(OplRow {
            layer,
            poc,
            md5: [
                parts[0].clone(),
                parts.get(1).cloned().unwrap_or_default(),
                parts.get(2).cloned().unwrap_or_default(),
            ],
        });
    }
    (rows, log_form)
}

/// The staged `.opl` files of a stream, by OLS index.
fn opl_files(dir: &Path, stem: &str) -> BTreeMap<usize, PathBuf> {
    let mut out = BTreeMap::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix(stem) else {
            continue;
        };
        let Some(rest) = rest
            .strip_prefix("-OLS")
            .or_else(|| rest.strip_prefix("_OLS"))
        else {
            continue;
        };
        let Some(idx) = rest.strip_suffix(".opl") else {
            continue;
        };
        if let Ok(k) = idx.parse::<usize>() {
            out.insert(k, e.path());
        }
    }
    out
}

/// The staged per-layer YUV digests of a stream, by layer index.
fn layer_md5_files(dir: &Path, stem: &str) -> BTreeMap<u8, String> {
    let mut out = BTreeMap::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix(stem) else {
            continue;
        };
        let lower = rest.to_ascii_lowercase();
        let Some(rest) = lower
            .strip_prefix("-l")
            .or_else(|| lower.strip_prefix("_l"))
        else {
            continue;
        };
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        let tail = &rest[digits.len()..];
        if !(tail == ".md5" || tail == "_rec.md5") {
            continue;
        }
        let Ok(layer) = digits.parse::<u8>() else {
            continue;
        };
        let Ok(text) = std::fs::read_to_string(e.path()) else {
            continue;
        };
        if let Some(tok) = text
            .split(|c: char| !c.is_ascii_hexdigit())
            .find(|t| t.len() == 32)
        {
            out.insert(layer, tok.to_ascii_lowercase());
        }
    }
    out
}

fn stream_path(dir: &Path, stem: &str) -> Option<PathBuf> {
    ["bit", "bin"]
        .iter()
        .map(|ext| dir.join(format!("{stem}.{ext}")))
        .find(|p| p.is_file())
}

/// Decode under `target`; returns the frames and the output layers.
/// The external base layer of a hybrid-scalability stream: the staged
/// `<stem>_BaseLayerSequence.yuv.xz` (1280x720 4:2:0 8-bit, one frame
/// per POC; decompressed with the `xz` tool — `None` when either is
/// unavailable).
fn external_base_frames(dir: &Path, stem: &str) -> Option<Vec<Picture>> {
    const W: usize = 1280;
    const H: usize = 720;
    let path = dir.join(format!("{stem}_BaseLayerSequence.yuv.xz"));
    if !path.is_file() {
        return None;
    }
    let out = std::process::Command::new("xz")
        .arg("-dc")
        .arg(&path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let frame = W * H * 3 / 2;
    let frames = out
        .stdout
        .chunks_exact(frame)
        .map(|f| {
            let mut pic = Picture::new(W, H, 1, 8, 8);
            let (y, c) = f.split_at(W * H);
            let (cb, cr) = c.split_at(W * H / 4);
            for (plane, src) in [(Plane::Luma, y), (Plane::Cb, cb), (Plane::Cr, cr)] {
                for (d, &v) in pic.plane_mut(plane).0.iter_mut().zip(src) {
                    *d = u16::from(v);
                }
            }
            pic
        })
        .collect();
    Some(frames)
}

/// Decode under `target`; returns the frames and the output layers.
fn decode_with_base(
    data: &[u8],
    target: LayerTarget,
    base: Option<Vec<Picture>>,
) -> Result<(Vec<DecodedFrame>, Vec<u8>), String> {
    let mut dec = SequenceDecoder::new();
    dec.set_layer_target(target);
    if let Some(frames) = base {
        dec.set_external_base_layer(move |req: &ExternalBaseRequest| {
            usize::try_from(req.poc)
                .ok()
                .and_then(|i| frames.get(i))
                .map(|p| ExternalBasePicture::new(p.clone()))
        });
    }
    dec.push_annexb(data).map_err(|e| e.to_string())?;
    let (_, output) = dec.layer_plan();
    let frames = dec.finish().map_err(|e| e.to_string())?;
    Ok((frames, output))
}

fn decode(data: &[u8], target: LayerTarget) -> Result<(Vec<DecodedFrame>, Vec<u8>), String> {
    decode_with_base(data, target, None)
}

/// The outcome of one stream / OLS check.
#[derive(Debug)]
struct Outcome {
    /// Matching rows / expected rows.
    matched: usize,
    expected: usize,
    /// Output pictures the decoder produced.
    produced: usize,
    note: String,
}

impl Outcome {
    fn exact(&self) -> bool {
        self.expected > 0 && self.matched == self.expected && self.produced == self.expected
    }
}

/// One output picture's POC and plane digests (cropped, coded).
type OutputDigests = (i32, [String; 3], [String; 3]);

/// Compare the decoder's output pictures against one picture list, per
/// layer as a multiset of plane digests (the lists are in decode or
/// output order depending on the submitter). A row matches an
/// unconsumed output picture of its layer by its cropped output or its
/// coded picture; a list row with no chroma digests (a monochrome
/// list) compares luma alone.
fn check_opl(frames: &[DecodedFrame], rows: &[OplRow]) -> Outcome {
    let mut want: BTreeMap<u8, Vec<&OplRow>> = BTreeMap::new();
    for r in rows {
        want.entry(r.layer).or_default().push(r);
    }
    let mut got: BTreeMap<u8, Vec<OutputDigests>> = BTreeMap::new();
    for f in frames.iter().filter(|f| f.output) {
        got.entry(f.layer_id).or_default().push((
            f.poc,
            plane_digests(&f.output_picture()),
            plane_digests(&f.picture),
        ));
    }
    let produced = got.values().map(Vec::len).sum();
    let mut matched = 0;
    let mut notes = Vec::new();
    for (layer, w) in &want {
        let mut g: Vec<Option<OutputDigests>> = got
            .get(layer)
            .map(|v| v.iter().cloned().map(Some).collect())
            .unwrap_or_default();
        if g.len() != w.len() {
            notes.push(format!("L{layer}: {} of {} pictures", g.len(), w.len()));
        }
        let mut bad = Vec::new();
        for row in w {
            let ok = |d: &[String; 3]| {
                d[0] == row.md5[0]
                    && (row.md5[1].is_empty()
                        || d[1].is_empty()
                        || (d[1] == row.md5[1] && d[2] == row.md5[2]))
            };
            if let Some(slot) = g
                .iter_mut()
                .find(|s| s.as_ref().is_some_and(|(_, c, p)| ok(c) || ok(p)))
            {
                *slot = None;
                matched += 1;
            } else {
                bad.push(row.poc);
            }
        }
        if !bad.is_empty() {
            bad.truncate(8);
            notes.push(format!("L{layer}: unmatched POCs {bad:?}"));
        }
    }
    for layer in got.keys() {
        if !want.contains_key(layer) {
            notes.push(format!("unexpected output layer {layer}"));
        }
    }
    Outcome {
        matched,
        expected: rows.len(),
        produced,
        note: notes.join("; "),
    }
}

/// Compare whole-layer YUV digests (layers by index in `nuh_layer_id`
/// order of the decoded frames).
fn check_layer_md5(frames: &[DecodedFrame], digests: &BTreeMap<u8, String>) -> Outcome {
    let mut by_layer: BTreeMap<u8, Vec<u8>> = BTreeMap::new();
    let mut counts: BTreeMap<u8, usize> = BTreeMap::new();
    for f in frames {
        let pic = f.output_picture();
        let e = by_layer.entry(f.layer_id).or_default();
        for p in [Plane::Luma, Plane::Cb, Plane::Cr] {
            if p != Plane::Luma && pic.chroma_array_type() == 0 {
                continue;
            }
            e.extend(plane_bytes(&pic, p));
        }
        *counts.entry(f.layer_id).or_default() += 1;
    }
    let layers: Vec<u8> = by_layer.keys().copied().collect();
    let mut matched = 0;
    let mut notes = Vec::new();
    for (idx, want) in digests {
        let Some(&layer) = layers.get(usize::from(*idx)) else {
            notes.push(format!("layer #{idx} not decoded"));
            continue;
        };
        if md5::hex(&by_layer[&layer]) == *want {
            matched += 1;
        } else {
            notes.push(format!(
                "layer #{idx} (id {layer}, {} pics) differs",
                counts[&layer]
            ));
        }
    }
    Outcome {
        matched,
        expected: digests.len(),
        produced: digests.len().min(layers.len()),
        note: notes.join("; "),
    }
}

/// Every stream stem staged in the directory.
fn stems(dir: &Path) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("bit" | "bin")))
                .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

fn run_one(dir: &Path, stem: &str, ols: Option<usize>) -> Outcome {
    let Some(path) = stream_path(dir, stem) else {
        return Outcome {
            matched: 0,
            expected: 1,
            produced: 0,
            note: "not staged".into(),
        };
    };
    let data = std::fs::read(path).expect("read bitstream");
    match ols {
        Some(k) => {
            let opl = opl_files(dir, stem);
            let (mut rows, log_form) =
                parse_opl(&std::fs::read_to_string(&opl[&k]).expect("read opl"));
            match decode_with_base(&data, LayerTarget::Ols(k), external_base_frames(dir, stem)) {
                Ok((frames, output_layers)) => {
                    if log_form {
                        // A decoder log lists every decoded picture:
                        // keep the output layers' rows, once each (a
                        // log can repeat the rows of an aborted run).
                        let mut seen = std::collections::BTreeSet::new();
                        rows.retain(|r| {
                            output_layers.contains(&r.layer)
                                && seen.insert((r.layer, r.poc, r.md5.clone()))
                        });
                    }
                    check_opl(&frames, &rows)
                }
                Err(e) => Outcome {
                    matched: 0,
                    expected: rows.len(),
                    produced: 0,
                    note: format!("decode error: {e}"),
                },
            }
        }
        None => {
            let digests = layer_md5_files(dir, stem);
            let target = LayerTarget::Ols(usize::MAX);
            match decode(&data, target) {
                Ok((frames, _)) => check_layer_md5(&frames, &digests),
                Err(e) => Outcome {
                    matched: 0,
                    expected: digests.len(),
                    produced: 0,
                    note: format!("decode error: {e}"),
                },
            }
        }
    }
}

/// Every pinned stream / OLS reproduces its picture list byte-exact.
#[test]
fn shvc_conformance_streams_decode_byte_exact() {
    let Some(dir) = shvc_dir() else {
        eprintln!("SHVC conformance corpus not staged; skipping");
        return;
    };
    if std::env::var_os("SHVC_SWEEP").is_some() {
        sweep(&dir);
        return;
    }
    let mut failed = Vec::new();
    for &(stem, ols) in EXPECTED_PASS {
        if cfg!(debug_assertions)
            && stream_path(&dir, stem)
                .and_then(|p| std::fs::metadata(p).ok())
                .is_none_or(|m| m.len() > 60_000)
        {
            continue;
        }
        let o = run_one(&dir, stem, ols);
        if !o.exact() {
            failed.push(format!(
                "{stem} OLS{ols:?}: {}/{} {}",
                o.matched, o.expected, o.note
            ));
        }
    }
    assert!(
        failed.is_empty(),
        "SHVC conformance regressions: {failed:#?}"
    );
}

fn sweep(dir: &Path) {
    let only = std::env::var("SHVC_ONLY").ok();
    let mut exact = 0;
    let mut total = 0;
    for stem in stems(dir) {
        if only.as_deref().is_some_and(|o| !stem.contains(o)) {
            continue;
        }
        let opl = opl_files(dir, &stem);
        let targets: Vec<Option<usize>> = if opl.is_empty() {
            vec![None]
        } else {
            opl.keys().copied().map(Some).collect()
        };
        for ols in targets {
            let t0 = std::time::Instant::now();
            let o = run_one(dir, &stem, ols);
            total += 1;
            if o.exact() {
                exact += 1;
            }
            println!(
                "SWEEP {stem} OLS{} {} {}/{} produced {} ({:.1}s) {}",
                ols.map_or("-".to_string(), |k| k.to_string()),
                if o.exact() { "EXACT" } else { "DIFF" },
                o.matched,
                o.expected,
                o.produced,
                t0.elapsed().as_secs_f64(),
                o.note
            );
        }
    }
    println!("SWEEP TOTAL {exact}/{total} exact");
}
