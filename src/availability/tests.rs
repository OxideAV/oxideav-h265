//! Tests for §6.4 availability + §6.5.1 / §6.5.2 scan conversions.
//!
//! Expected values are hand-derived from the ITU-T H.265 equations, not
//! taken from any implementation.

use super::*;

/// Build the simplest single-tile geometry: `w`×`h` CTBs, CtbLog2 = 6
/// (64-sample CTBs), MinTbLog2 = 2 (4-sample minimum blocks).
fn single_tile(w: u32, h: u32) -> PictureTiling {
    PictureTiling::new(w, h, w * 64, h * 64, 6, 2, &TilingParams::single_tile()).unwrap()
}

#[test]
fn single_tile_rs_to_ts_is_identity() {
    let t = single_tile(3, 2);
    for rs in 0..6 {
        assert_eq!(t.ctb_addr_rs_to_ts(rs), rs, "rs->ts identity at {rs}");
        assert_eq!(t.ctb_addr_ts_to_rs(rs), rs, "ts->rs identity at {rs}");
        assert_eq!(t.tile_id(rs), 0, "single tile id at {rs}");
    }
}

#[test]
fn min_tb_addr_zs_within_one_ctb_is_morton() {
    // One 64-sample CTB = 16×16 4-sample min-blocks. shift = 4.
    // Within a single CTB (tile-scan addr 0), MinTbAddrZs is the pure
    // Morton interleave of the block (x, y) bits (eq. 6-10 with the
    // CtbAddrRsToTs term zero).
    let t = single_tile(1, 1);
    // (0,0) -> 0
    assert_eq!(t.min_tb_addr_zs(0, 0), 0);
    // (1,0): x bit0 set -> p += 1*1 = 1
    assert_eq!(t.min_tb_addr_zs(1, 0), 1);
    // (0,1): y bit0 set -> p += 2*1*1 = 2
    assert_eq!(t.min_tb_addr_zs(0, 1), 2);
    // (1,1): 1 + 2 = 3
    assert_eq!(t.min_tb_addr_zs(1, 1), 3);
    // (2,0): x bit1 (m=2) set -> p += 2*2 = 4
    assert_eq!(t.min_tb_addr_zs(2, 0), 4);
    // (0,2): y bit1 set -> p += 2*2*2 = 8
    assert_eq!(t.min_tb_addr_zs(0, 2), 8);
    // (3,3): x bits 0,1 -> 1 + 4 = 5; y bits 0,1 -> 2 + 8 = 10; sum 15
    assert_eq!(t.min_tb_addr_zs(3, 3), 15);
    // (15,15): all four x bits -> 1+4+16+64 = 85; y doubles -> 170; 255
    assert_eq!(t.min_tb_addr_zs(15, 15), 255);
}

#[test]
fn min_tb_addr_zs_second_ctb_offsets_by_256() {
    // Two CTBs wide, single tile/slice. The min-block grid is 32 wide.
    // A block in the second CTB (x >= 16) sits at tile-scan CTB addr 1,
    // so its z-address starts at 1 << (shift*2) = 256.
    let t = single_tile(2, 1);
    assert_eq!(t.min_tb_addr_zs(16, 0), 256);
    assert_eq!(t.min_tb_addr_zs(17, 0), 257);
    // First CTB still occupies 0..255.
    assert_eq!(t.min_tb_addr_zs(15, 15), 255);
}

#[test]
fn uniform_two_column_tiles_split_evenly() {
    // 4 CTBs wide, 1 tall, 2 uniform columns -> colWidth = [2, 2].
    // colBd = [0, 2, 4]. CtbAddrRsToTs: tile 0 holds CTBs (0,1), tile 1
    // holds (2,3); tile scan visits tile 0 fully then tile 1.
    let tiles = TilingParams {
        num_tile_columns_minus1: 1,
        num_tile_rows_minus1: 0,
        uniform_spacing_flag: true,
        column_width_minus1: vec![],
        row_height_minus1: vec![],
    };
    let t = PictureTiling::new(4, 1, 256, 64, 6, 2, &tiles).unwrap();
    // rs 0,1 (tile 0) -> ts 0,1; rs 2,3 (tile 1) -> ts 2,3 (here the
    // single row keeps tile-scan == raster for this layout).
    assert_eq!(t.ctb_addr_rs_to_ts(0), 0);
    assert_eq!(t.ctb_addr_rs_to_ts(1), 1);
    assert_eq!(t.ctb_addr_rs_to_ts(2), 2);
    assert_eq!(t.ctb_addr_rs_to_ts(3), 3);
    assert_eq!(t.tile_id(t.ctb_addr_rs_to_ts(0)), 0);
    assert_eq!(t.tile_id(t.ctb_addr_rs_to_ts(2)), 1);
}

#[test]
fn uniform_two_by_two_tiles_reorder_tile_scan() {
    // 4×4 CTBs, 2×2 uniform tiles. colWidth = rowHeight = [2, 2].
    // Tile scan visits tile (0,0)'s 4 CTBs, then (1,0), then (0,1),
    // then (1,1). So raster CTB 2 (top-right tile) maps to tile-scan 4.
    let tiles = TilingParams {
        num_tile_columns_minus1: 1,
        num_tile_rows_minus1: 1,
        uniform_spacing_flag: true,
        column_width_minus1: vec![],
        row_height_minus1: vec![],
    };
    let t = PictureTiling::new(4, 4, 256, 256, 6, 2, &tiles).unwrap();
    // Raster addr of CTB at (col=0,row=0) = 0 -> ts 0, tile 0.
    assert_eq!(t.ctb_addr_rs_to_ts(0), 0);
    assert_eq!(t.tile_id(0), 0);
    // CTB at (col=2,row=0) raster = 2; it is tile (1,0) = tile idx 1,
    // first CTB after tile 0's 4 -> ts 4.
    assert_eq!(t.ctb_addr_rs_to_ts(2), 4);
    assert_eq!(t.tile_id(4), 1);
    // CTB at (col=0,row=2) raster = 8; tile (0,1) = tile idx 2 -> ts 8.
    assert_eq!(t.ctb_addr_rs_to_ts(8), 8);
    assert_eq!(t.tile_id(8), 2);
    // CTB at (col=2,row=2) raster = 10; tile (1,1) = tile idx 3 -> ts 12.
    assert_eq!(t.ctb_addr_rs_to_ts(10), 12);
    assert_eq!(t.tile_id(12), 3);
    // ts->rs is the inverse permutation.
    for rs in 0..16 {
        assert_eq!(t.ctb_addr_ts_to_rs(t.ctb_addr_rs_to_ts(rs)), rs);
    }
}

#[test]
fn explicit_tile_widths() {
    // 5 CTBs wide, 1 tall, 2 columns, column_width_minus1[0] = 0
    // -> colWidth = [1, 4].
    let tiles = TilingParams {
        num_tile_columns_minus1: 1,
        num_tile_rows_minus1: 0,
        uniform_spacing_flag: false,
        column_width_minus1: vec![0],
        row_height_minus1: vec![],
    };
    let t = PictureTiling::new(5, 1, 320, 64, 6, 2, &tiles).unwrap();
    assert_eq!(t.tile_id(t.ctb_addr_rs_to_ts(0)), 0); // first CTB, tile 0
    assert_eq!(t.tile_id(t.ctb_addr_rs_to_ts(1)), 1); // rest, tile 1
    assert_eq!(t.tile_id(t.ctb_addr_rs_to_ts(4)), 1);
}

#[test]
fn explicit_tile_overflow_rejected() {
    // column_width_minus1[0] = 5 implies first tile is 6 CTBs wide in a
    // 5-CTB-wide picture -> overflow.
    let tiles = TilingParams {
        num_tile_columns_minus1: 1,
        num_tile_rows_minus1: 0,
        uniform_spacing_flag: false,
        column_width_minus1: vec![5],
        row_height_minus1: vec![],
    };
    assert!(matches!(
        PictureTiling::new(5, 1, 320, 64, 6, 2, &tiles),
        Err(AvailabilityError::TileOverflow {
            dimension: "column"
        })
    ));
}

#[test]
fn explicit_array_length_mismatch_rejected() {
    let tiles = TilingParams {
        num_tile_columns_minus1: 2, // expects 2 entries
        num_tile_rows_minus1: 0,
        uniform_spacing_flag: false,
        column_width_minus1: vec![0], // only 1 supplied
        row_height_minus1: vec![],
    };
    assert!(matches!(
        PictureTiling::new(6, 1, 384, 64, 6, 2, &tiles),
        Err(AvailabilityError::TileArrayLength { .. })
    ));
}

#[test]
fn invalid_log2_sizes_rejected() {
    // CtbLog2 < MinTbLog2.
    assert!(matches!(
        PictureTiling::new(1, 1, 64, 64, 2, 4, &TilingParams::single_tile()),
        Err(AvailabilityError::InvalidLog2Sizes { .. })
    ));
}

#[test]
fn zero_geometry_rejected() {
    assert!(matches!(
        PictureTiling::new(0, 1, 0, 64, 6, 2, &TilingParams::single_tile()),
        Err(AvailabilityError::ZeroGeometry {
            field: "PicWidthInCtbsY"
        })
    ));
}

// ---- §6.4.1 z-scan availability ----

#[test]
fn neighbour_outside_picture_is_unavailable() {
    let t = single_tile(2, 2);
    let same_slice = |_rs: u32| 0u32;
    // Current block at (64, 64). Left neighbour at x = -1 -> out.
    assert!(!t.z_scan_availability(64, 64, -1, 64, same_slice));
    // Top neighbour at y = -1 -> out.
    assert!(!t.z_scan_availability(64, 64, 64, -1, same_slice));
    // Right of picture (picture is 128 wide).
    assert!(!t.z_scan_availability(64, 64, 128, 64, same_slice));
    // Below picture (128 tall).
    assert!(!t.z_scan_availability(64, 64, 64, 128, same_slice));
}

#[test]
fn already_decoded_left_and_above_are_available() {
    // 2×2 CTBs, single slice/tile. Current block top-left at (64, 64)
    // (the bottom-right CTB, decoded last in raster order). Its left
    // (63, 64) and above (64, 63) neighbours decode earlier -> available.
    let t = single_tile(2, 2);
    let same_slice = |_rs: u32| 0u32;
    assert!(t.z_scan_availability(64, 64, 63, 64, same_slice)); // left
    assert!(t.z_scan_availability(64, 64, 64, 63, same_slice)); // above
    assert!(t.z_scan_availability(64, 64, 63, 63, same_slice)); // above-left
}

#[test]
fn not_yet_decoded_neighbour_is_unavailable() {
    // Current at the top-left CTB (0,0). The below-right neighbour at
    // (64, 64) decodes later -> minBlockAddrN > minBlockAddrCurr ->
    // unavailable even though it is inside the picture.
    let t = single_tile(2, 2);
    let same_slice = |_rs: u32| 0u32;
    assert!(!t.z_scan_availability(0, 0, 64, 64, same_slice));
    // The above-right of the bottom-left CTB: current (0,64),
    // neighbour (64, 63) is in the top-right CTB which decodes before
    // the bottom-left CTB in raster order -> available.
    assert!(t.z_scan_availability(0, 64, 64, 63, same_slice));
}

#[test]
fn different_slice_is_unavailable() {
    // 2×2 CTBs. Put the second slice starting at raster CTB 3 (the
    // bottom-right). Current block in CTB 3, left neighbour in CTB 2.
    let t = single_tile(2, 2);
    let slice_of = |rs: u32| if rs >= 3 { 1u32 } else { 0u32 };
    // current (64,64) is CTB 3 (slice 1); left (63,64) is CTB 2 (slice 0)
    assert!(!t.z_scan_availability(64, 64, 63, 64, slice_of));
    // within-slice neighbour: above (64,63) is CTB 1 (slice 0) too, but
    // it differs from current's slice 1 -> still unavailable.
    assert!(!t.z_scan_availability(64, 64, 64, 63, slice_of));
}

#[test]
fn different_tile_is_unavailable() {
    // 2 columns × 1 row of tiles over 2×1 CTBs: each CTB is its own
    // tile. Current block in the right CTB, left neighbour in the left
    // CTB -> different tile -> unavailable even within one slice.
    let tiles = TilingParams {
        num_tile_columns_minus1: 1,
        num_tile_rows_minus1: 0,
        uniform_spacing_flag: true,
        column_width_minus1: vec![],
        row_height_minus1: vec![],
    };
    let t = PictureTiling::new(2, 1, 128, 64, 6, 2, &tiles).unwrap();
    let same_slice = |_rs: u32| 0u32;
    // current (64,0) right tile; left (63,0) left tile.
    assert!(!t.z_scan_availability(64, 0, 63, 0, same_slice));
    // within the same (right) tile a left neighbour is available.
    assert!(t.z_scan_availability(127, 0, 126, 0, same_slice));
}

// ---- §6.4.2 prediction block availability ----

#[test]
fn prediction_block_intra_neighbour_masked_out() {
    // 2×2 CTBs single slice/tile. Current PB top-left (64,64). Left
    // neighbour (63,64) would be z-scan available, but its CuPredMode
    // is MODE_INTRA -> masked to unavailable for the inter query.
    let t = single_tile(2, 2);
    let same_slice = |_rs: u32| 0u32;
    let all_intra = |_x: u32, _y: u32| MODE_INTRA;
    let all_inter = |_x: u32, _y: u32| 0u8; // MODE_INTER
                                            // 64×64 CU == one 64×64 PB (2Nx2N), partIdx 0.
    assert!(!t.prediction_block_availability(
        64, 64, 64, 64, 64, 64, 64, 0, 63, 64, same_slice, all_intra,
    ));
    assert!(t.prediction_block_availability(
        64, 64, 64, 64, 64, 64, 64, 0, 63, 64, same_slice, all_inter,
    ));
}

#[test]
fn prediction_block_second_partition_cannot_see_first() {
    // The §6.4.2 same-CB forced-FALSE branch fires only for the NxN
    // partitioning ((nPbW << 1) == nCbS AND (nPbH << 1) == nCbS): a
    // 64×64 CU split into four 32×32 PBs. The top-right PB (partIdx 1,
    // top-left (32, 0)) must not reference the bottom-left PB region
    // (yNbY >= yCb + nPbH = 32, xNbY < xCb + nPbW = 32), which the
    // §7.4.9 partitioning order has not decoded yet.
    let t = single_tile(2, 2);
    let same_slice = |_rs: u32| 0u32;
    let all_inter = |_x: u32, _y: u32| 0u8;
    // CU (0,0) size 64; PB1 (32,0) 32×32, partIdx 1. Neighbour (31, 32)
    // is the bottom-left PB region — same CB, the forced-FALSE branch.
    assert!(!t
        .prediction_block_availability(0, 0, 64, 32, 0, 32, 32, 1, 31, 32, same_slice, all_inter,));
    // The bottom-right region (xNbY >= 32) is NOT forced false by this
    // branch (it sits at xNbY = 32, failing xCb+nPbW > xNbY); it falls
    // through to availableN = TRUE since it is the same CB.
    assert!(
        t.prediction_block_availability(0, 0, 64, 32, 0, 32, 32, 1, 32, 32, same_slice, all_inter,)
    );
}

#[test]
fn prediction_block_normal_left_neighbour_available() {
    // Same-CB short-cut not triggered (neighbour outside the CU): a
    // normal left neighbour of an inter 2Nx2N PB resolves through
    // §6.4.1 and is available.
    let t = single_tile(2, 2);
    let same_slice = |_rs: u32| 0u32;
    let all_inter = |_x: u32, _y: u32| 0u8;
    assert!(t.prediction_block_availability(
        64, 64, 64, 64, 64, 64, 64, 0, 63, 64, same_slice, all_inter,
    ));
}
