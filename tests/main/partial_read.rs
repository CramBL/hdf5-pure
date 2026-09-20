//! Tests for the windowed (leading-dimension row-range) read API:
//! `Dataset::read_raw_rows` and the typed `read_*_rows` helpers (issue: partial
//! reads).
//!
//! The invariant under test everywhere: a row window must be byte-for-byte the
//! same as the whole-dataset read sliced to that row range — for every layout
//! (contiguous, chunked, inner-chunked grids), on both the in-memory and the
//! streaming backend, and across edge windows (empty, past-the-end, straddling
//! chunk boundaries).

use hdf5_pure::{CharacterSet, Datatype, File, FileBuilder, StringPadding};

/// Open the bytes produced by `build` on both backends: buffered (in-memory) and
/// streaming (from a temp file). The windowed read dispatches per backend, so
/// every case is checked on both. The `TempDir` is returned so it outlives the
/// streaming file handle.
fn on_both_backends(build: impl FnOnce(&mut FileBuilder)) -> (File, File, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");
    let mut builder = FileBuilder::new();
    build(&mut builder);
    builder.write(&path).unwrap();

    let buffered = File::open(&path).unwrap();
    let streaming = File::open_streaming(&path).unwrap();
    (buffered, streaming, dir)
}

/// Row windows worth checking for a dataset whose leading dimension is `n0`:
/// whole, first row, a middle band, the last row, empty, past-the-end (clamps to
/// empty), and an over-long window (clamps to the tail).
fn windows(n0: u64) -> Vec<(u64, u64)> {
    let mut w = vec![
        (0, n0),
        (0, 1.min(n0)),
        (n0 / 2, n0 - n0 / 2),
        (n0.saturating_sub(1), 1.min(n0)),
        (n0 / 3, n0 / 3),
        (0, 0),
        (n0, 5),                     // start past the end -> empty
        (n0.saturating_sub(2), 100), // over-long -> clamps to the tail
    ];
    w.dedup();
    w
}

fn check_f64(file: &File, name: &str, row_elems: usize, extra: &[(u64, u64)]) {
    let ds = file.dataset(name).unwrap();
    let full = ds.read_f64().unwrap();
    let n0 = (full.len() / row_elems.max(1)) as u64;

    for (start, count) in windows(n0).into_iter().chain(extra.iter().copied()) {
        let clamped_start = start.min(n0);
        let clamped_count = count.min(n0 - clamped_start);
        let lo = clamped_start as usize * row_elems;
        let hi = lo + clamped_count as usize * row_elems;

        let window = ds.read_f64_rows(start, count).unwrap();
        assert_eq!(
            window,
            &full[lo..hi],
            "{name}: read_f64_rows({start}, {count}) != full[{lo}..{hi}]"
        );
    }
}

fn check_u8(file: &File, name: &str, row_elems: usize, extra: &[(u64, u64)]) {
    let ds = file.dataset(name).unwrap();
    let full = ds.read_u8().unwrap();
    let n0 = (full.len() / row_elems.max(1)) as u64;

    for (start, count) in windows(n0).into_iter().chain(extra.iter().copied()) {
        let clamped_start = start.min(n0);
        let clamped_count = count.min(n0 - clamped_start);
        let lo = clamped_start as usize * row_elems;
        let hi = lo + clamped_count as usize * row_elems;

        let window = ds.read_u8_rows(start, count).unwrap();
        assert_eq!(
            window,
            &full[lo..hi],
            "{name}: read_u8_rows({start}, {count}) != full[{lo}..{hi}]"
        );
    }
}

#[test]
fn contiguous_1d_f64() {
    let data: Vec<f64> = (0..100).map(|i| i as f64 * 1.5).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("x")
            .with_f64_data(&data)
            .with_shape(&[100]);
    });
    check_f64(&buffered, "x", 1, &[]);
    check_f64(&streaming, "x", 1, &[]);
}

#[test]
fn contiguous_2d_f64_preserves_inner_shape() {
    // shape [20, 3]: each row is 3 elements; the window must keep that stride.
    let data: Vec<f64> = (0..60).map(f64::from).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("m")
            .with_f64_data(&data)
            .with_shape(&[20, 3]);
    });
    check_f64(&buffered, "m", 3, &[(5, 7), (0, 20)]);
    check_f64(&streaming, "m", 3, &[(5, 7), (0, 20)]);
}

#[test]
fn chunked_1d_deflate_multi_chunk() {
    // 1000 rows in 64-row chunks -> many chunks; windows straddle boundaries.
    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.25).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("c")
            .with_f64_data(&data)
            .with_shape(&[1000])
            .with_chunks(&[64])
            .with_deflate(4);
    });
    // Windows that start/end inside a chunk, cross exactly one boundary, and span
    // several chunks.
    let extra = &[(60, 10), (64, 64), (100, 300), (63, 2), (999, 1)];
    check_f64(&buffered, "c", 1, extra);
    check_f64(&streaming, "c", 1, extra);
}

#[test]
fn chunked_frame_per_chunk_4d_u8() {
    // The re_hdf5 image case: [10, 2, 2, 3] uint8, one frame per chunk, deflated.
    let data: Vec<u8> = (0..10 * 2 * 2 * 3).map(|i| (i % 251) as u8).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("frames")
            .with_u8_data(&data)
            .with_shape(&[10, 2, 2, 3])
            .with_chunks(&[1, 2, 2, 3])
            .with_deflate(4);
    });
    let extra = &[(3, 4), (0, 1), (9, 1)];
    check_u8(&buffered, "frames", 2 * 2 * 3, extra);
    check_u8(&streaming, "frames", 2 * 2 * 3, extra);
}

#[test]
fn chunked_multi_row_chunk_partial_edges() {
    // shape [50, 4], chunks [8, 4]: a chunk spans 8 rows, so a window straddling a
    // chunk exercises the top/bottom row-band trim inside a single chunk.
    let data: Vec<f64> = (0..200).map(f64::from).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("d")
            .with_f64_data(&data)
            .with_shape(&[50, 4])
            .with_chunks(&[8, 4])
            .with_deflate(2);
    });
    let extra = &[(3, 2), (6, 5), (7, 1), (8, 8), (0, 9), (49, 1)];
    check_f64(&buffered, "d", 4, extra);
    check_f64(&streaming, "d", 4, extra);
}

#[test]
fn inner_chunked_2d_grid_windows() {
    // shape [20, 6], chunks [4, 2]: the inner dim is chunked, so each window
    // scatters the overlapping chunks of a 2-D chunk grid into a window-shaped
    // output (no whole-read fallback).
    let data: Vec<f64> = (0..120).map(f64::from).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("t")
            .with_f64_data(&data)
            .with_shape(&[20, 6])
            .with_chunks(&[4, 2]);
    });
    let extra = &[(3, 5), (0, 20), (10, 1)];
    check_f64(&buffered, "t", 6, extra);
    check_f64(&streaming, "t", 6, extra);
}

#[test]
fn inner_chunked_edge_overhang_deflate() {
    // shape [23, 10], chunks [4, 3], deflated: neither dimension is a multiple of
    // its chunk extent, so the grid's last chunk row and column overhang the
    // dataset — the scatter must clip them in both the leading and inner dims,
    // and windows straddling chunk boundaries must trim inside a chunk.
    let data: Vec<f64> = (0..230).map(|i| f64::from(i) * 0.5 - 3.0).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("e")
            .with_f64_data(&data)
            .with_shape(&[23, 10])
            .with_chunks(&[4, 3])
            .with_deflate(4);
    });
    let extra = &[(3, 2), (4, 4), (7, 9), (19, 4), (22, 1), (0, 23)];
    check_f64(&buffered, "e", 10, extra);
    check_f64(&streaming, "e", 10, extra);
}

#[test]
fn inner_chunked_4d_image_grid() {
    // The image-tile case: shape [13, 5, 7, 3] u8 with chunks [4, 2, 3, 2] — an
    // inner-split grid in every dimension, with overhanging edge chunks in every
    // dimension. Windows must assemble each row from the full inner chunk grid.
    let data: Vec<u8> = (0..13 * 5 * 7 * 3).map(|i| (i % 251) as u8).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("img")
            .with_u8_data(&data)
            .with_shape(&[13, 5, 7, 3])
            .with_chunks(&[4, 2, 3, 2])
            .with_deflate(2);
    });
    let extra = &[(3, 2), (4, 4), (11, 2), (12, 1), (0, 13)];
    check_u8(&buffered, "img", 5 * 7 * 3, extra);
    check_u8(&streaming, "img", 5 * 7 * 3, extra);
}

#[test]
fn scalar_0d_is_a_single_row() {
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("s").with_i64_data(&[42]).with_shape(&[]);
    });
    for file in [&buffered, &streaming] {
        let ds = file.dataset("s").unwrap();
        assert_eq!(ds.read_i64_rows(0, 1).unwrap(), vec![42]);
        // Past the single row -> empty.
        assert_eq!(ds.read_i64_rows(1, 1).unwrap(), Vec::<i64>::new());
    }
}

#[test]
fn fixed_length_strings_window() {
    // Fixed-length strings decode straight from the raw window (the bounded path).
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        let dtype = Datatype::String {
            size: 4,
            padding: StringPadding::NullPad,
            charset: CharacterSet::Utf8,
        };
        let mut bytes = Vec::new();
        for value in ["ab", "cde", "f", "ghij"] {
            let mut element = value.as_bytes().to_vec();
            element.resize(4, 0);
            bytes.extend_from_slice(&element);
        }
        b.create_dataset("fixed").with_raw_data(dtype, bytes, 4);
    });
    for file in [&buffered, &streaming] {
        let ds = file.dataset("fixed").unwrap();
        let full = ds.read_string().unwrap();
        assert_eq!(ds.read_string_rows(0, 4).unwrap(), full);
        assert_eq!(ds.read_string_rows(1, 2).unwrap(), &full[1..3]);
        assert_eq!(ds.read_string_rows(3, 5).unwrap(), &full[3..4]);
    }
}

#[test]
fn vlen_strings_window() {
    // Variable-length strings are heap-backed; `read_string_rows` resolves only
    // the window's heap references, and the window must match the whole read
    // sliced to the same rows.
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("labels")
            .with_vlen_strings(&["idle", "reach", "grasp", "lift", "place"]);
    });
    for file in [&buffered, &streaming] {
        let ds = file.dataset("labels").unwrap();
        let full = ds.read_string().unwrap();
        assert_eq!(ds.read_string_rows(0, 5).unwrap(), full);
        assert_eq!(ds.read_string_rows(1, 2).unwrap(), &full[1..3]);
        assert_eq!(ds.read_string_rows(4, 10).unwrap(), &full[4..5]);
        assert_eq!(ds.read_string_rows(5, 3).unwrap(), Vec::<String>::new());
    }
}

#[test]
fn vlen_strings_multidim_window_slices_by_row() {
    // On a multi-dimensional VL string dataset a row spans its inner dimensions,
    // so `read_string_rows` must slice by row — the same rows `read_raw_rows`
    // returns — not treat the flat per-element array as one string per row.
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("grid")
            .with_vlen_strings(&["a", "bb", "ccc", "dddd", "e", "ff"])
            .with_shape(&[3, 2]); // 3 rows x 2 columns -> each row is 2 elements
    });
    for file in [&buffered, &streaming] {
        let ds = file.dataset("grid").unwrap();
        let all = ds.read_string().unwrap();
        assert_eq!(all.len(), 6);

        // Each row is the pair of inner-dimension elements, not a single string.
        assert_eq!(ds.read_string_rows(0, 1).unwrap(), vec!["a", "bb"]);
        assert_eq!(ds.read_string_rows(1, 1).unwrap(), vec!["ccc", "dddd"]);
        assert_eq!(ds.read_string_rows(2, 1).unwrap(), vec!["e", "ff"]);
        assert_eq!(
            ds.read_string_rows(1, 2).unwrap(),
            vec!["ccc", "dddd", "e", "ff"]
        );

        // Full-range and over-long windows return every element.
        assert_eq!(ds.read_string_rows(0, 3).unwrap(), all);
        assert_eq!(ds.read_string_rows(0, 99).unwrap(), all);
        // A window starting past the last row is empty.
        assert!(ds.read_string_rows(3, 1).unwrap().is_empty());

        // Lockstep with the raw path: the string count of a window equals the
        // number of elements `read_raw_rows` reads for the same window.
        let elem = ds.datatype().unwrap().type_size() as usize;
        for (start, count) in [(0u64, 1u64), (1, 1), (2, 1), (0, 2), (1, 2), (0, 3)] {
            let n_strings = ds.read_string_rows(start, count).unwrap().len();
            let n_raw_elems = ds.read_raw_rows(start, count).unwrap().len() / elem;
            assert_eq!(n_strings, n_raw_elems, "window ({start}, {count})");
        }
    }
}

#[test]
fn typed_readers_match_whole_dataset() {
    // Spot-check several dtypes go through the same window path and decode right.
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("i32").with_i32_data(&[10, 20, 30, 40, 50]);
        b.create_dataset("u64")
            .with_u64_data(&[1, 2, 3, 4])
            .with_chunks(&[2])
            .with_shape(&[4]);
        b.create_dataset("f32").with_f32_data(&[1.5, 2.5, 3.5]);
    });
    for file in [&buffered, &streaming] {
        let i32s = file.dataset("i32").unwrap();
        assert_eq!(
            i32s.read_i32_rows(1, 3).unwrap(),
            &i32s.read_i32().unwrap()[1..4]
        );

        let u64s = file.dataset("u64").unwrap();
        assert_eq!(
            u64s.read_u64_rows(1, 2).unwrap(),
            &u64s.read_u64().unwrap()[1..3]
        );

        let f32s = file.dataset("f32").unwrap();
        assert_eq!(
            f32s.read_f32_rows(0, 2).unwrap(),
            &f32s.read_f32().unwrap()[0..2]
        );
    }
}

#[test]
fn read_raw_rows_matches_read_raw() {
    // The raw-byte contract, independent of dtype decoding, on a compressed chunked
    // dataset large enough to span chunks.
    let data: Vec<f64> = (0..500).map(|i| i as f64 * -0.5).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("r")
            .with_f64_data(&data)
            .with_shape(&[500])
            .with_chunks(&[100])
            .with_deflate(6);
    });
    for file in [&buffered, &streaming] {
        let ds = file.dataset("r").unwrap();
        let full = ds.read_raw().unwrap();
        let elem = 8; // f64
        for (start, count) in [(0u64, 500u64), (150, 200), (99, 3), (450, 100)] {
            let s = start as usize * elem;
            let e = (start + count).min(500) as usize * elem;
            assert_eq!(ds.read_raw_rows(start, count).unwrap(), &full[s..e]);
        }
    }
}

#[test]
fn full_range_window_delegates_to_whole_read() {
    // A window covering every row (including one clamped down from over-long)
    // must cost and return exactly the whole read: `read_raw_rows` delegates to
    // it, and a vlen-string window resolves the same full set of heap
    // references. Checked on an inner-chunked grid and on variable-length
    // strings; the observable contract is equality with the whole read.
    let data: Vec<f64> = (0..120).map(f64::from).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.create_dataset("t")
            .with_f64_data(&data)
            .with_shape(&[20, 6])
            .with_chunks(&[4, 2]); // inner dim chunked
        b.create_dataset("labels")
            .with_vlen_strings(&["idle", "reach", "grasp", "lift", "place"]);
    });
    for file in [&buffered, &streaming] {
        let ds = file.dataset("t").unwrap();
        let full = ds.read_raw().unwrap();
        assert_eq!(ds.read_raw_rows(0, 20).unwrap(), full); // exact full range
        assert_eq!(ds.read_raw_rows(0, 21).unwrap(), full); // over-long clamps to full range
        assert_eq!(ds.read_f64_rows(0, 20).unwrap(), data);

        let ds = file.dataset("labels").unwrap();
        let full = ds.read_string().unwrap();
        assert_eq!(ds.read_string_rows(0, 5).unwrap(), full);
        assert_eq!(ds.read_string_rows(0, 99).unwrap(), full);
    }
}

#[test]
fn userblock_windows_go_through_base_framing() {
    // A userblock makes the superblock base non-zero, so reads route through the
    // base-framed source. Exercise that path for both contiguous and chunked
    // windowed reads, on both backends.
    let chunked: Vec<f64> = (0..300).map(|i| i as f64 * 0.5).collect();
    let contig: Vec<f64> = (0..40).map(f64::from).collect();
    let (buffered, streaming, _dir) = on_both_backends(|b| {
        b.with_userblock(512);
        b.create_dataset("chunked")
            .with_f64_data(&chunked)
            .with_shape(&[300])
            .with_chunks(&[64])
            .with_deflate(3);
        b.create_dataset("contig")
            .with_f64_data(&contig)
            .with_shape(&[40]);
    });
    check_f64(&buffered, "chunked", 1, &[(70, 100), (63, 5)]);
    check_f64(&streaming, "chunked", 1, &[(70, 100), (63, 5)]);
    check_f64(&buffered, "contig", 1, &[(10, 20)]);
    check_f64(&streaming, "contig", 1, &[(10, 20)]);
}
