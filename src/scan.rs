//! §6.5.3 up-right diagonal scan order array initialization process.
//!
//! ITU-T Rec. H.265 §6.5.3 defines the `diagScan[ sPos ][ sComp ]`
//! array that maps a scan position `sPos` (0..`blkSize * blkSize − 1`)
//! to a `(horizontal, vertical)` coordinate pair walking the block
//! along up-right diagonals (equation 6-11). It is the `scanIdx == 0`
//! entry of the §7.4.2 `ScanOrder[ log2BlockSize ][ scanIdx ]` table.
//!
//! This module supplies the up-right diagonal scan only — the form the
//! §7.4.5 `ScalingFactor` derivation (equations 7-44..7-51) reads, via
//! `ScanOrder[ 2 ][ 0 ]` (a 4x4 block) and `ScanOrder[ 3 ][ 0 ]` (an
//! 8x8 block). The horizontal / vertical / traverse scans (§6.5.4 /
//! §6.5.5 / §6.5.6) belong to the residual-coding path and are added
//! when that path lands.

/// One scan-position entry: the `(x, y)` coordinate the position maps
/// to. `x` is the horizontal component (`sComp == 0`) and `y` is the
/// vertical component (`sComp == 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanPos {
    /// Horizontal component — `diagScan[ sPos ][ 0 ]`.
    pub x: u8,
    /// Vertical component — `diagScan[ sPos ][ 1 ]`.
    pub y: u8,
}

/// Build the §6.5.3 up-right diagonal scan order for a `blkSize`x`blkSize`
/// block, returning the `diagScan` array as a `Vec<ScanPos>` of length
/// `blkSize * blkSize`, indexed by scan position `sPos`.
///
/// This is a direct transcription of equation 6-11: starting at the
/// top-left, each inner `while( y >= 0 )` loop walks one up-right
/// diagonal (decrementing `y`, incrementing `x`), emitting only the
/// in-bounds `(x, y)` cells; the outer loop seeds the next diagonal at
/// `y = x, x = 0` and stops once every cell has been visited.
pub fn up_right_diagonal(blk_size: usize) -> Vec<ScanPos> {
    let n = blk_size * blk_size;
    let mut diag = Vec::with_capacity(n);

    // The 6-11 pseudocode uses a signed `y` that the inner loop drives
    // below 0 as its termination test, so the coordinates are tracked
    // as `i32` and only the in-bounds cells are recorded.
    let blk = blk_size as i32;
    let mut x: i32 = 0;
    let mut y: i32 = 0;
    let mut stop = false;
    while !stop {
        while y >= 0 {
            if x < blk && y < blk {
                diag.push(ScanPos {
                    x: x as u8,
                    y: y as u8,
                });
            }
            y -= 1;
            x += 1;
        }
        y = x;
        x = 0;
        if diag.len() >= n {
            stop = true;
        }
    }

    diag
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 4x4 up-right diagonal scan (§6.5.3, `blkSize == 4`) visits
    /// the 16 cells in the canonical diagonal order. Hand-derived from
    /// equation 6-11.
    #[test]
    fn diag_scan_4x4() {
        let scan = up_right_diagonal(4);
        let coords: Vec<(u8, u8)> = scan.iter().map(|p| (p.x, p.y)).collect();
        assert_eq!(
            coords,
            vec![
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
            ]
        );
    }

    /// The 2x2 scan is the smallest non-trivial case and exercises the
    /// outer-loop re-seed (`y = x, x = 0`) twice.
    #[test]
    fn diag_scan_2x2() {
        let scan = up_right_diagonal(2);
        let coords: Vec<(u8, u8)> = scan.iter().map(|p| (p.x, p.y)).collect();
        assert_eq!(coords, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }

    /// The scan is a permutation of every `(x, y)` cell in the block:
    /// each coordinate appears exactly once, and the length is
    /// `blkSize * blkSize`. Checked for the 4x4 and 8x8 blocks the
    /// §7.4.5 derivation reads.
    #[test]
    fn diag_scan_is_a_permutation() {
        for &blk in &[4usize, 8] {
            let scan = up_right_diagonal(blk);
            assert_eq!(scan.len(), blk * blk);
            let mut seen = vec![false; blk * blk];
            for p in &scan {
                let (x, y) = (p.x as usize, p.y as usize);
                assert!(x < blk && y < blk);
                let flat = y * blk + x;
                assert!(!seen[flat], "cell ({x},{y}) visited twice");
                seen[flat] = true;
            }
            assert!(seen.iter().all(|&v| v), "not every cell visited");
        }
    }

    /// Equation 6-11 walks each diagonal from its bottom-left toward its
    /// top-right: the sum `x + y` (the diagonal index) is
    /// non-decreasing across the scan, and within a diagonal `y`
    /// strictly decreases. Verified on the 8x8 block.
    #[test]
    fn diag_scan_8x8_diagonal_ordering() {
        let scan = up_right_diagonal(8);
        let mut prev_diag = 0usize;
        let mut prev_y = i32::MAX;
        for p in &scan {
            let diag = p.x as usize + p.y as usize;
            assert!(diag >= prev_diag, "diagonal index decreased");
            if diag != prev_diag {
                prev_diag = diag;
                prev_y = i32::MAX;
            }
            assert!(
                (p.y as i32) < prev_y,
                "y not strictly decreasing in diagonal"
            );
            prev_y = p.y as i32;
        }
    }
}
