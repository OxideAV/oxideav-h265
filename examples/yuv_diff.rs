//! Debug: decode an Annex B stream and compare every output picture
//! against a raw planar reference decode, reporting where they differ.
//!
//! ```text
//! cargo run --release --example yuv_diff -- stream.bit reference.yuv [block]
//! ```
//!
//! For each output picture and plane: the number of differing samples,
//! their bounding box, the first differing sample in raster order, and
//! (with `block`, default 64) the list of the first few `block x block`
//! luma-grid cells holding a difference — in raster order, so the first
//! one is the first coding tree block that went wrong.

use oxideav_h265::picture::Plane;

fn main() {
    let mut args = std::env::args().skip(1);
    let stream = args
        .next()
        .expect("usage: yuv_diff <stream> <ref.yuv> [block]");
    let reference = args
        .next()
        .expect("usage: yuv_diff <stream> <ref.yuv> [block]");
    let block: usize = args.next().map_or(64, |b| b.parse().expect("block size"));
    let data = std::fs::read(&stream).expect("read stream");
    let want = std::fs::read(&reference).expect("read reference");
    let frames = match oxideav_h265::decode_annexb_sequence(&data) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("decode error: {e}");
            std::process::exit(1);
        }
    };
    let mut offset = 0usize;
    let mut bad_pictures = 0usize;
    for (n, frame) in frames.iter().filter(|f| f.output).enumerate() {
        let pic = frame.output_picture();
        let mut picture_bad = false;
        let planes = [Plane::Luma, Plane::Cb, Plane::Cr];
        let n_planes = if pic.chroma_array_type() == 0 { 1 } else { 3 };
        for &plane in planes.iter().take(n_planes) {
            let (w, h) = pic.plane_dims(plane);
            let bytes = if pic.bit_depth(plane) > 8 { 2 } else { 1 };
            let scale = pic.width_luma() / w.max(1);
            let scale_y = pic.height_luma() / h.max(1);
            let (mut count, mut first) = (0usize, None);
            let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
            let mut cells: Vec<(usize, usize)> = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    let at = offset + (y * w + x) * bytes;
                    let Some(chunk) = want.get(at..at + bytes) else {
                        eprintln!("reference ends inside picture {n}");
                        std::process::exit(1);
                    };
                    let r = if bytes == 2 {
                        i32::from(u16::from_le_bytes([chunk[0], chunk[1]]))
                    } else {
                        i32::from(chunk[0])
                    };
                    let got = pic.sample(plane, x, y);
                    if got != r {
                        count += 1;
                        first.get_or_insert((x, y, got, r));
                        x0 = x0.min(x);
                        y0 = y0.min(y);
                        x1 = x1.max(x);
                        y1 = y1.max(y);
                        let cell = (y * scale_y / block, x * scale / block);
                        if !cells.contains(&cell) && cells.len() < 4096 {
                            cells.push(cell);
                        }
                    }
                }
            }
            offset += w * h * bytes;
            if count > 0 {
                picture_bad = true;
                cells.sort_unstable();
                let (fx, fy, got, r) = first.unwrap();
                println!(
                    "pic {n} poc {} {plane:?}: {count} differ, box ({x0},{y0})..=({x1},{y1}), first ({fx},{fy}) got {got} want {r}, {} cells of {block}; first cells (row,col): {:?}",
                    frame.poc,
                    cells.len(),
                    &cells[..cells.len().min(8)]
                );
            }
        }
        if picture_bad {
            bad_pictures += 1;
        } else {
            println!("pic {n} poc {}: exact", frame.poc);
        }
    }
    println!(
        "{bad_pictures} picture(s) differ; reference bytes consumed {offset} of {}",
        want.len()
    );
}
