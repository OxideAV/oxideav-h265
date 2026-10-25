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

/// One in-flight picture: its geometry and the published CTB rows.
#[derive(Debug)]
pub(crate) struct InFlightPicture {
    /// `CtbLog2SizeY`.
    ctb_log2: u32,
    /// Number of CTB rows (`PicHeightInCtbsY`).
    rows: usize,
    /// Plane widths (luma, Cb, Cr) and heights.
    dims: [(usize, usize); 3],
    /// Plane rows per CTB row (luma, Cb, Cr).
    strip_h: [usize; 3],
    /// Final samples per plane per CTB row.
    strips: [Vec<OnceLock<Box<[u16]>>>; 3],
    /// Motion cells per CTB row.
    motion: Vec<OnceLock<MotionField>>,
    /// Number of CTB rows published (a prefix).
    published: AtomicUsize,
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
        let strips = || (0..rows).map(|_| OnceLock::new()).collect::<Vec<_>>();
        Self {
            ctb_log2,
            rows,
            dims: [(pw, ph), (cw, ch), (cw, ch)],
            strip_h: [ctb, ctb / sh, ctb / sh],
            strips: [strips(), strips(), strips()],
            motion: (0..rows).map(|_| OnceLock::new()).collect(),
            published: AtomicUsize::new(0),
            lock: Mutex::new(()),
            cv: Condvar::new(),
        }
    }

    /// Number of CTB rows.
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// Number of CTB rows published so far.
    pub(crate) fn published(&self) -> usize {
        self.published.load(Ordering::Acquire)
    }

    /// Block until at least `n` CTB rows (clamped to the picture) are
    /// published.
    pub(crate) fn wait_rows(&self, n: usize) {
        let n = n.min(self.rows);
        if self.published.load(Ordering::Acquire) >= n {
            return;
        }
        let mut guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        while self.published.load(Ordering::Acquire) < n {
            guard = self.cv.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Block until the CTB row holding plane row `y` (clamped into the
    /// plane) and every row above it are published.
    pub(crate) fn wait_plane_row(&self, plane: usize, y: i32) {
        let h = self.dims[plane].1 as i32;
        let y = y.clamp(0, (h - 1).max(0)) as usize;
        self.wait_rows(y / self.strip_h[plane].max(1) + 1);
    }

    /// Plane row `y` (published).
    pub(crate) fn row(&self, plane: usize, y: usize) -> &[u16] {
        let sh = self.strip_h[plane];
        let w = self.dims[plane].0;
        let strip = self.strips[plane][y / sh]
            .get()
            .expect("a published CTB row");
        let o = (y % sh) * w;
        &strip[o..o + w]
    }

    /// The collocated motion cell covering luma `(x, y)`, once its CTB
    /// row is published.
    pub(crate) fn cell_at(&self, x: usize, y: usize) -> MotionCell {
        let r = (y >> self.ctb_log2).min(self.rows.saturating_sub(1));
        self.wait_rows(r + 1);
        self.motion[r]
            .get()
            .expect("a published CTB row")
            .cell_at(x, y)
    }

    /// Publish CTB rows `published .. upto` from the worker's picture
    /// and motion field (both whole-picture, the rows final).
    pub(crate) fn publish(&self, upto: usize, pic: &Picture, field: &MotionField) {
        let upto = upto.min(self.rows);
        let from = self.published.load(Ordering::Acquire);
        if upto <= from {
            return;
        }
        let ctb = 1usize << self.ctb_log2;
        for r in from..upto {
            for (pi, plane) in [Plane::Luma, Plane::Cb, Plane::Cr].into_iter().enumerate() {
                let (w, h) = self.dims[pi];
                if w == 0 {
                    let _ = self.strips[pi][r].set(Box::new([]));
                    continue;
                }
                let sh = self.strip_h[pi];
                let (y0, y1) = (r * sh, ((r + 1) * sh).min(h));
                let (buf, _) = pic.plane_slice(plane);
                let _ = self.strips[pi][r].set(buf[y0 * w..y1 * w].into());
            }
            let (_, ph) = self.dims[0];
            let y0 = r * ctb;
            let rows = ctb.min(ph - y0);
            let _ = self.motion[r].set(field.band_copy(y0, rows));
        }
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        self.published.store(upto, Ordering::Release);
        self.cv.notify_all();
    }

    /// Publish every row not published yet with whatever the worker
    /// holds (a failed or abandoned picture), so no reader blocks.
    pub(crate) fn abandon(&self) {
        let ctb = 1usize << self.ctb_log2;
        for r in self.published.load(Ordering::Acquire)..self.rows {
            for pi in 0..3 {
                let (w, h) = self.dims[pi];
                let sh = self.strip_h[pi];
                let n = if w == 0 {
                    0
                } else {
                    (((r + 1) * sh).min(h) - r * sh) * w
                };
                let _ = self.strips[pi][r].set(vec![0u16; n].into());
            }
            let (pw, ph) = self.dims[0];
            let y0 = r * ctb;
            let _ = self.motion[r].set(MotionField::new_band(pw, ph, y0, ctb.min(ph - y0)));
        }
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        self.published.store(self.rows, Ordering::Release);
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
