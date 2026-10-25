//! Row-progressive publication of a picture still being decoded — the
//! frame-threaded decoder's reference store.
//!
//! With a thread budget the [`crate::sequence::SequenceDecoder`] runs
//! the slice data of several pictures at once, in decode order. A
//! picture whose reconstruction is still under way can already be a
//! reference of a later one: its worker publishes every CTB row the
//! moment it is final — reconstructed, deblocked (§8.7.2) and SAO-
//! filtered (§8.7.3), which is one CTB row behind the reconstruction
//! because the next row's horizontal edges and the SAO window reach
//! into it — together with that row's motion cells (the §8.5.3.2.8
//! collocated motion). A reader blocks until the rows its motion
//! vector (plus the interpolation filter taps, after the `Clip3` edge
//! extension) or its collocated position reaches are published; every
//! sample it reads is therefore the final sample the serial decoder
//! would read.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use crate::motion::{MotionCell, MotionField};
use crate::picture::{Picture, Plane};

/// Luma lines per published band.
const BAND: usize = 4;

/// One in-flight picture: its geometry, the published sample lines and
/// the published motion rows.
#[derive(Debug)]
pub(crate) struct InFlightPicture {
    /// `CtbLog2SizeY`.
    ctb_log2: u32,
    /// Number of CTB rows (`PicHeightInCtbsY`).
    rows: usize,
    /// Plane widths (luma, Cb, Cr) and heights.
    dims: [(usize, usize); 3],
    /// `SubHeightC` of each plane (1 for luma).
    sub_h: [usize; 3],
    /// Final samples per plane per band of [`BAND`] luma lines.
    bands: [Vec<OnceLock<Box<[u16]>>>; 3],
    /// Motion cells per CTB row.
    motion: Vec<OnceLock<MotionField>>,
    /// Luma lines published (a prefix; chroma follows at `SubHeightC`).
    lines: AtomicUsize,
    /// CTB rows of motion published (a prefix).
    motion_rows: AtomicUsize,
    lock: Mutex<()>,
    cv: Condvar,
}

impl InFlightPicture {
    /// The store for a picture of `geometry`'s format (only its
    /// dimensions are read) with `1 << ctb_log2` luma rows per CTB row.
    pub(crate) fn new(geometry: &Picture, ctb_log2: u32) -> Self {
        let ctb = 1usize << ctb_log2;
        let (pw, ph) = geometry.plane_dims(Plane::Luma);
        let rows = ph.div_ceil(ctb);
        let chroma = geometry.chroma_array_type() != 0;
        let (cw, ch) = if chroma {
            geometry.plane_dims(Plane::Cb)
        } else {
            (0, 0)
        };
        let sh = if chroma && ch > 0 { ph / ch } else { 1 };
        let n_bands = ph.div_ceil(BAND);
        let bands = || (0..n_bands).map(|_| OnceLock::new()).collect::<Vec<_>>();
        Self {
            ctb_log2,
            rows,
            dims: [(pw, ph), (cw, ch), (cw, ch)],
            sub_h: [1, sh, sh],
            bands: [bands(), bands(), bands()],
            motion: (0..rows).map(|_| OnceLock::new()).collect(),
            lines: AtomicUsize::new(0),
            motion_rows: AtomicUsize::new(0),
            lock: Mutex::new(()),
            cv: Condvar::new(),
        }
    }

    /// Luma lines published so far.
    pub(crate) fn published_lines(&self) -> usize {
        self.lines.load(Ordering::Acquire)
    }

    /// The picture height in luma lines.
    pub(crate) fn height(&self) -> usize {
        self.dims[0].1
    }

    /// Block until `counter` reaches `n`.
    fn wait_for(&self, counter: &AtomicUsize, n: usize) {
        if counter.load(Ordering::Acquire) >= n {
            return;
        }
        let mut guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        while counter.load(Ordering::Acquire) < n {
            guard = self.cv.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Block until plane row `y` (clamped into the plane) and every row
    /// above it are published.
    pub(crate) fn wait_plane_row(&self, plane: usize, y: i32) {
        let h = self.dims[plane].1 as i32;
        let y = y.clamp(0, (h - 1).max(0)) as usize;
        let need = ((y + 1) * self.sub_h[plane]).min(self.dims[0].1);
        self.wait_for(&self.lines, need);
    }

    /// Plane row `y` (published).
    pub(crate) fn row(&self, plane: usize, y: usize) -> &[u16] {
        let bh = BAND / self.sub_h[plane];
        let w = self.dims[plane].0;
        let band = self.bands[plane][y / bh].get().expect("a published band");
        let o = (y % bh) * w;
        &band[o..o + w]
    }

    /// The collocated motion cell covering luma `(x, y)`, once its CTB
    /// row is published.
    pub(crate) fn cell_at(&self, x: usize, y: usize) -> MotionCell {
        let r = (y >> self.ctb_log2).min(self.rows.saturating_sub(1));
        self.wait_for(&self.motion_rows, r + 1);
        self.motion[r]
            .get()
            .expect("a published CTB row")
            .cell_at(x, y)
    }

    fn notify(&self, counter: &AtomicUsize, value: usize) {
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        counter.store(value, Ordering::Release);
        self.cv.notify_all();
    }

    /// Publish the final luma lines `published .. upto` (a multiple of
    /// the band height, or the picture height) and the matching chroma
    /// lines from the worker's whole-picture `pic`.
    pub(crate) fn publish_lines(&self, upto: usize, pic: &Picture) {
        let (_, ph) = self.dims[0];
        let upto = upto.min(ph);
        let from = self.lines.load(Ordering::Acquire);
        if upto <= from {
            return;
        }
        for b in from / BAND..upto.div_ceil(BAND) {
            for (pi, plane) in [Plane::Luma, Plane::Cb, Plane::Cr].into_iter().enumerate() {
                let (w, h) = self.dims[pi];
                if w == 0 {
                    let _ = self.bands[pi][b].set(Box::new([]));
                    continue;
                }
                let bh = BAND / self.sub_h[pi];
                let (y0, y1) = (b * bh, ((b + 1) * bh).min(h));
                let (buf, _) = pic.plane_slice(plane);
                let _ = self.bands[pi][b].set(buf[y0 * w..y1 * w].into());
            }
        }
        self.notify(&self.lines, upto);
    }

    /// Publish the motion cells of CTB rows `published .. upto`.
    pub(crate) fn publish_motion(&self, upto: usize, field: &MotionField) {
        let upto = upto.min(self.rows);
        let from = self.motion_rows.load(Ordering::Acquire);
        if upto <= from {
            return;
        }
        let ctb = 1usize << self.ctb_log2;
        let (_, ph) = self.dims[0];
        for r in from..upto {
            let y0 = r * ctb;
            let _ = self.motion[r].set(field.band_copy(y0, ctb.min(ph - y0)));
        }
        self.notify(&self.motion_rows, upto);
    }

    /// Publish everything not published yet with whatever the worker
    /// holds (zero samples, intra cells — a failed or abandoned
    /// picture), so no reader blocks.
    pub(crate) fn abandon(&self) {
        for (pi, bands) in self.bands.iter().enumerate() {
            let (w, h) = self.dims[pi];
            let bh = BAND / self.sub_h[pi];
            for (b, slot) in bands.iter().enumerate() {
                let n = if w == 0 {
                    0
                } else {
                    (((b + 1) * bh).min(h).saturating_sub(b * bh)) * w
                };
                let _ = slot.set(vec![0u16; n].into());
            }
        }
        let ctb = 1usize << self.ctb_log2;
        let (pw, ph) = self.dims[0];
        for (r, slot) in self.motion.iter().enumerate() {
            let y0 = r * ctb;
            let _ = slot.set(MotionField::new_band(pw, ph, y0, ctb.min(ph - y0)));
        }
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        self.lines.store(ph, Ordering::Release);
        self.motion_rows.store(self.rows, Ordering::Release);
        self.cv.notify_all();
    }
}

/// A shared handle on an [`InFlightPicture`]: equal only to itself.
#[derive(Debug, Clone)]
pub(crate) struct Pending(pub(crate) Arc<InFlightPicture>);

impl PartialEq for Pending {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Pending {}
