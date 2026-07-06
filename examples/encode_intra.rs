//! Encode one raw 4:2:0 frame as a real CABAC intra IDR stream.
//!
//! ```text
//! cargo run --example encode_intra -- in.yuv WxH QP out.hevc [recon.yuv]
//! ```
//!
//! `in.yuv` is one planar 8-bit 4:2:0 frame; dimensions must be
//! multiples of 16. The optional `recon.yuv` receives the encoder's
//! own reconstruction (what a conforming decoder must output).

use oxideav_h265::encoder::intra::encode_idr_intra_au;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: encode_intra <in.yuv> <WxH> <QP> <out.hevc> [recon.yuv]";
    if args.len() < 4 {
        eprintln!("{usage}");
        std::process::exit(2);
    }
    let (w, h) = args[1]
        .split_once('x')
        .and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?)))
        .expect("WxH");
    let qp: i32 = args[2].parse().expect("QP");
    let data = std::fs::read(&args[0]).expect("read input");
    assert_eq!(data.len(), w * h * 3 / 2, "one planar 4:2:0 frame");
    let (y, c) = data.split_at(w * h);
    let (cb, cr) = c.split_at(w * h / 4);
    let enc = encode_idr_intra_au(y, cb, cr, w, h, qp).expect("encode");
    std::fs::write(&args[3], &enc.au).expect("write output");
    if let Some(recon_path) = args.get(4) {
        let mut recon = enc.recon_y.clone();
        recon.extend_from_slice(&enc.recon_cb);
        recon.extend_from_slice(&enc.recon_cr);
        std::fs::write(recon_path, &recon).expect("write recon");
    }
    eprintln!("{w}x{h} qp{qp} -> {} bytes", enc.au.len());
}
