//! HEVC coefficient scan order tables (§6.5).
//!
//! Three scans exist for HEVC residual coding: up-right diagonal,
//! horizontal, and vertical. They operate on 4×4 sub-blocks of the
//! coefficient array — each sub-block has its own scan order derived
//! from the `scan_idx` selected by the intra prediction mode (Table 8-2).

/// Up-right diagonal scan of a 4×4 sub-block (§6.5.3). 16 entries,
/// each `(x, y)` pair.
pub const DIAG_SCAN_4X4: [(u8, u8); 16] = [
    (0, 0),
    (0, 1),
    (1, 0),
    (0, 2),
    (1, 1),
    (2, 0),
    (0, 3),
    (1, 2),
    (2, 1),
    (3, 0),
    (1, 3),
    (2, 2),
    (3, 1),
    (2, 3),
    (3, 2),
    (3, 3),
];

/// Horizontal scan of a 4×4 sub-block — left-to-right, top-to-bottom.
pub const HORZ_SCAN_4X4: [(u8, u8); 16] = [
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (0, 1),
    (1, 1),
    (2, 1),
    (3, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (3, 2),
    (0, 3),
    (1, 3),
    (2, 3),
    (3, 3),
];

/// Vertical scan of a 4×4 sub-block — top-to-bottom, left-to-right.
pub const VERT_SCAN_4X4: [(u8, u8); 16] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
];

/// Return the appropriate 4×4 scan table for a given scan_idx.
pub fn scan_4x4(scan_idx: u32) -> &'static [(u8, u8); 16] {
    match scan_idx {
        1 => &HORZ_SCAN_4X4,
        2 => &VERT_SCAN_4X4,
        _ => &DIAG_SCAN_4X4,
    }
}

/// Determine scan_idx (§8.5.2) for an intra prediction mode.
/// Returns 0 (diagonal), 1 (horizontal), or 2 (vertical).
pub fn scan_idx_for_intra(log2_tb: u32, pred_mode: u32, is_luma: bool) -> u32 {
    if log2_tb != 2 && log2_tb != 3 {
        return 0; // Only 4x4 and 8x8 use horizontal/vertical scan.
    }
    if is_luma {
        // Luma: horizontal for modes 22..=30 (diagonal near vertical),
        //       vertical for modes 6..=14.
        if (6..=14).contains(&pred_mode) {
            2
        } else if (22..=30).contains(&pred_mode) {
            1
        } else {
            0
        }
    } else {
        // Chroma: same rule applied to intraPredModeC. Since we run the
        // caller-resolved chroma prediction mode here, treat identically.
        if (6..=14).contains(&pred_mode) {
            2
        } else if (22..=30).contains(&pred_mode) {
            1
        } else {
            0
        }
    }
}
