//! Dataset region and row-window reads across layouts and reader backends.

use std::io::Cursor;
use std::sync::Arc;
use std::sync::Mutex;

use hdf5_pure::CharacterSet;
use hdf5_pure::Datatype;
use hdf5_pure::DeflateLevel;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FormatError;
use hdf5_pure::ReadSeekSource;
use hdf5_pure::Source;
use hdf5_pure::StringPadding;
use hdf5_pure::UserblockSize;
use rstest::rstest;
use test_util::btree_v1;
use test_util::data_layout;
use test_util::dataspace;
use test_util::datatype;
use test_util::image::Image;
use test_util::link_message::HardLink;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v2::Header;
use test_util::superblock::v2::Superblock;
use test_util::widths::Widths;

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
            .with_deflate(DeflateLevel::Four);
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
            .with_deflate(DeflateLevel::Four);
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
            .with_deflate(DeflateLevel::Two);
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
            .with_deflate(DeflateLevel::Four);
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
            .with_deflate(DeflateLevel::Two);
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
            .with_deflate(DeflateLevel::Six);
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
        b.with_userblock(UserblockSize::MIN);
        b.create_dataset("chunked")
            .with_f64_data(&chunked)
            .with_shape(&[300])
            .with_chunks(&[64])
            .with_deflate(DeflateLevel::Three);
        b.create_dataset("contig")
            .with_f64_data(&contig)
            .with_shape(&[40]);
    });
    check_f64(&buffered, "chunked", 1, &[(70, 100), (63, 5)]);
    check_f64(&streaming, "chunked", 1, &[(70, 100), (63, 5)]);
    check_f64(&buffered, "contig", 1, &[(10, 20)]);
    check_f64(&streaming, "contig", 1, &[(10, 20)]);
}

#[rstest]
#[case::full_rows(1, 1)]
#[case::inner_split(6, 2)]
fn high_row_windows_read_allocated_chunks_on_both_backends(
    #[case] columns: u32,
    #[case] chunk_columns: u32,
    #[values(false, true)] streaming: bool,
) {
    let bytes = sparse_high_row_file(columns, chunk_columns);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("high_rows.h5");
    std::fs::write(&path, &bytes).unwrap();
    let file = if streaming {
        File::open_streaming(&path)
    } else {
        File::open(&path)
    }
    .unwrap();
    let dataset = file.dataset("d").unwrap();
    assert_eq!(
        dataset.shape().unwrap(),
        vec![4_294_967_424, u64::from(columns)]
    );
    assert_eq!(
        dataset.read_f64_rows(HIGH_ROW, 1).unwrap(),
        high_row_values(columns, 0, 1)
    );
    assert_eq!(
        dataset.read_f64_rows(HIGH_ROW + 1, 2).unwrap(),
        high_row_values(columns, 1, 2)
    );
    assert_eq!(
        dataset.read_f64_rows(HIGH_ROW + 3, 3).unwrap(),
        high_row_values(columns, 3, 3)
    );
    assert_eq!(
        dataset.read_f64_rows(HIGH_ROW + 3, 3).unwrap(),
        high_row_values(columns, 3, 3)
    );
    assert_eq!(
        dataset.read_f64_rows(1, 2).unwrap(),
        (columns + 1..3 * columns + 1)
            .map(f64::from)
            .collect::<Vec<_>>()
    );
}

#[rstest]
#[case::full_rows(1, 1)]
#[case::inner_split(6, 2)]
fn high_row_index_selection_reads_only_selected_payloads(
    #[case] columns: u32,
    #[case] chunk_columns: u32,
    #[values(false, true)] warm_index: bool,
) {
    let reads = Arc::new(Mutex::new(Vec::new()));
    let source = RecordingRowSource {
        inner: ReadSeekSource::new(Cursor::new(sparse_high_row_file(columns, chunk_columns)))
            .unwrap(),
        reads: Arc::clone(&reads),
    };
    let file = File::from_source(source).unwrap();
    let dataset = file.dataset("d").unwrap();
    if warm_index {
        assert_eq!(
            dataset.read_f64_rows(HIGH_ROW, 1).unwrap(),
            high_row_values(columns, 0, 1)
        );
    }
    reads.lock().unwrap().clear();
    assert_eq!(
        dataset.read_f64_rows(0, 1).unwrap(),
        (1..=columns).map(f64::from).collect::<Vec<_>>()
    );
    let payload_reads: Vec<_> = reads
        .lock()
        .unwrap()
        .iter()
        .copied()
        .filter(|&(offset, _)| {
            (PAYLOAD_START..PAYLOAD_START + u64::from(3 * 4 * columns * 8)).contains(&offset)
        })
        .collect();
    assert_eq!(
        payload_reads,
        vec![(PAYLOAD_START, usize::try_from(4 * columns * 8).unwrap())]
    );
    assert_eq!(
        dataset.read_f64_rows(HIGH_ROW, 1).unwrap(),
        high_row_values(columns, 0, 1)
    );
    reads.lock().unwrap().clear();
    assert_eq!(
        dataset.read_f64_rows(HIGH_ROW, 1).unwrap(),
        high_row_values(columns, 0, 1)
    );
    assert!(reads.lock().unwrap().is_empty());
}

fn high_row_values(columns: u32, skip: u32, rows: u32) -> Vec<f64> {
    (101 + skip * columns..101 + (skip + rows) * columns)
        .map(f64::from)
        .collect()
}

fn sparse_high_row_file(columns: u32, chunk_columns: u32) -> Vec<u8> {
    let mut image = Image::new();
    let mut keys = Vec::new();
    let mut addresses = Vec::new();
    for row in [0, HIGH_ROW, HIGH_ROW + 4] {
        for column in (0..columns).step_by(usize::try_from(chunk_columns).unwrap()) {
            let values: Vec<_> = (0..4)
                .flat_map(|local_row| {
                    (column..column + chunk_columns).flat_map(move |column| {
                        let base = if row == 0 {
                            1
                        } else {
                            101 + u32::try_from(row - HIGH_ROW).unwrap() * columns
                        };
                        f64::from(base + local_row * columns + column).to_le_bytes()
                    })
                })
                .collect();
            let address = PAYLOAD_START + addresses.len() as u64 * u64::from(4 * chunk_columns * 8);
            image.place(usize::try_from(address).unwrap(), &values);
            keys.push(btree_v1::chunk_key(
                4 * chunk_columns * 8,
                0,
                &[row, u64::from(column), 0],
            ));
            addresses.push(address);
        }
    }
    keys.push(btree_v1::chunk_key(0, 0, &[u64::MAX; 3]));
    let index = image.append(&btree_v1::node(
        btree_v1::NodeType::CHUNK,
        0,
        &keys,
        &addresses,
        Widths::EIGHT,
    ));
    let shape = dataspace::v2(
        2,
        dataspace::Flags::NONE,
        dataspace::Kind::SIMPLE,
        &[4_294_967_424, u64::from(columns)],
        None,
    );
    let dataset = image.append(
        &Header::new()
            .messages([
                Message::new(MessageType::DATASPACE, &shape),
                Message::new(MessageType::DATATYPE, &datatype::f64_le()),
                Message::new(
                    MessageType::DATA_LAYOUT,
                    &data_layout::chunked(index as u64, &[4, chunk_columns], 8),
                ),
            ])
            .build(),
    );
    let root = image.append(
        &Header::new()
            .message(Message::new(
                MessageType::LINK,
                &HardLink::new("d", dataset as u64).build(Widths::EIGHT),
            ))
            .build(),
    );
    image.place(
        0,
        &Superblock::new(Widths::EIGHT)
            .root_header_address(root as u64)
            .eof_address(image.as_bytes().len() as u64)
            .build(),
    );
    image.build()
}

#[rstest]
#[case::huge_then_zero(&[2, 4_294_967_296, 0], 1, 1)]
#[case::zero_then_huge(&[2, 0, 4_294_967_296], 1, 1)]
#[case::zero_rows_huge_then_zero(&[2, 4_294_967_296, 0], 1, 0)]
#[case::zero_rows_zero_then_huge(&[2, 0, 4_294_967_296], 1, 0)]
#[case::high_row(&[4_294_967_424, 0], 4_294_967_360, 1)]
#[case::many_rows(&[4_294_967_424, 0], 1, 4_294_967_296)]
fn empty_row_windows_accept_large_extents(
    #[case] shape: &[u64],
    #[case] start: u64,
    #[case] count: u64,
    #[values("compact", "contiguous", "unallocated", "chunked")] layout: &str,
    #[values(false, true)] streaming: bool,
) {
    let bytes = empty_row_file(shape, layout);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty_rows.h5");
    std::fs::write(&path, bytes).unwrap();
    let file = if streaming {
        File::open_streaming(&path)
    } else {
        File::open(&path)
    }
    .unwrap();
    let dataset = file.dataset("d").unwrap();
    assert_eq!(dataset.shape().unwrap(), shape);
    assert_eq!(
        dataset.read_raw_rows(start, count).unwrap(),
        Vec::<u8>::new()
    );
    if shape.contains(&0) {
        assert_eq!(dataset.read_raw().unwrap(), Vec::<u8>::new());
    }
}

#[rstest]
fn zero_row_windows_return_empty_vectors_for_large_rows(#[values(false, true)] streaming: bool) {
    let shape = [2, 4_294_967_296];
    let bytes = empty_row_file(&shape, "unallocated");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zero_rows.h5");
    std::fs::write(&path, bytes).unwrap();
    let file = if streaming {
        File::open_streaming(&path)
    } else {
        File::open(&path)
    }
    .unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_raw_rows(1, 0).unwrap(),
        Vec::<u8>::new(),
    );
}

#[cfg(target_pointer_width = "32")]
#[rstest]
fn nonempty_row_windows_report_the_materialized_byte_length(
    #[values(false, true)] streaming: bool,
) {
    assert_eq!(usize::BITS, 32);
    let bytes = empty_row_file(&[2, 4_294_967_296], "unallocated");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large_rows.h5");
    std::fs::write(&path, bytes).unwrap();
    let file = if streaming {
        File::open_streaming(&path)
    } else {
        File::open(&path)
    }
    .unwrap();
    let err = file.dataset("d").unwrap().read_raw_rows(1, 1).unwrap_err();
    let hdf5_pure::Error::Format(err) = err else {
        panic!("expected format error, got {err:?}");
    };
    assert_eq!(
        err,
        FormatError::ValueTooLargeForPlatform {
            value: 34_359_738_368,
            target: "usize",
        }
    );
}

fn empty_row_file(shape: &[u64], layout: &str) -> Vec<u8> {
    let mut image = Image::starting_with(&[0; 64]);
    let layout = match layout {
        "compact" => data_layout::compact(&[]),
        "contiguous" => data_layout::contiguous(64, 0),
        "unallocated" => data_layout::contiguous(u64::MAX, 0),
        "chunked" => data_layout::chunked(u64::MAX, &vec![1; shape.len()], 8),
        _ => panic!("unknown test layout {layout:?}"),
    };
    let shape = dataspace::v2(
        u8::try_from(shape.len()).unwrap(),
        dataspace::Flags::NONE,
        dataspace::Kind::SIMPLE,
        shape,
        None,
    );
    let dataset = image.append(
        &Header::new()
            .messages([
                Message::new(MessageType::DATASPACE, &shape),
                Message::new(MessageType::DATATYPE, &datatype::f64_le()),
                Message::new(MessageType::DATA_LAYOUT, &layout),
            ])
            .build(),
    );
    let root = image.append(
        &Header::new()
            .message(Message::new(
                MessageType::LINK,
                &HardLink::new("d", dataset as u64).build(Widths::EIGHT),
            ))
            .build(),
    );
    image.place(
        0,
        &Superblock::new(Widths::EIGHT)
            .root_header_address(root as u64)
            .eof_address(image.as_bytes().len() as u64)
            .build(),
    );
    image.build()
}

struct RecordingRowSource {
    inner: ReadSeekSource<Cursor<Vec<u8>>>,
    reads: Arc<Mutex<Vec<(u64, usize)>>>,
}

impl Source for RecordingRowSource {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        let Self { inner, reads } = self;
        reads.lock().unwrap().push((offset, bytes.len()));
        inner.read_at(offset, bytes)
    }
}

#[rstest]
fn numeric_region_methods_read_selected_columns(
    #[values(false, true)] chunked: bool,
    #[values(false, true)] streaming: bool,
) {
    let values: Vec<i32> = (0..30).collect();
    let (buffered, streamed, _directory) = on_both_backends(|builder| {
        let dataset = builder.create_dataset("matrix");
        dataset.with_i32_data(&values).with_shape(&[5, 6]);
        if chunked {
            dataset
                .with_chunks(&[2, 4])
                .with_deflate(DeflateLevel::Four);
        }
    });
    let file = if streaming { streamed } else { buffered };
    let dataset = file.dataset("matrix").unwrap();
    let start = &[1, 1];
    let count = &[2, 3];
    assert_eq!(
        dataset.read_f64_region(start, count).unwrap(),
        [7.0, 8.0, 9.0, 13.0, 14.0, 15.0]
    );
    assert_eq!(
        dataset.read_f32_region(start, count).unwrap(),
        [7.0, 8.0, 9.0, 13.0, 14.0, 15.0]
    );
    assert_eq!(
        dataset.read_i8_region(start, count).unwrap(),
        [7, 8, 9, 13, 14, 15]
    );
    assert_eq!(
        dataset.read_i16_region(start, count).unwrap(),
        [7, 8, 9, 13, 14, 15]
    );
    assert_eq!(
        dataset.read_i32_region(start, count).unwrap(),
        [7, 8, 9, 13, 14, 15]
    );
    assert_eq!(
        dataset.read_i64_region(start, count).unwrap(),
        [7, 8, 9, 13, 14, 15]
    );
    assert_eq!(
        dataset.read_u8_region(start, count).unwrap(),
        [7, 8, 9, 13, 14, 15]
    );
    assert_eq!(
        dataset.read_u16_region(start, count).unwrap(),
        [7, 8, 9, 13, 14, 15]
    );
    assert_eq!(
        dataset.read_u32_region(start, count).unwrap(),
        [7, 8, 9, 13, 14, 15]
    );
    assert_eq!(
        dataset.read_u64_region(start, count).unwrap(),
        [7, 8, 9, 13, 14, 15]
    );
    assert_eq!(
        dataset.read_raw_region(start, count).unwrap(),
        [7i32, 8, 9, 13, 14, 15]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>()
    );
}

#[rstest]
#[case::fixed(false)]
#[case::variable(true)]
fn string_regions_read_selected_columns(
    #[case] variable: bool,
    #[values(false, true)] streaming: bool,
    #[values(false, true)] chunked: bool,
) {
    let (buffered, streamed, _directory) = on_both_backends(|builder| {
        let dataset = builder.create_dataset("labels");
        if variable {
            dataset.with_vlen_strings(&["zero", "one", "two", "three", "four", "five"]);
        } else {
            dataset
                .with_strings(&["zero", "one", "two", "three", "four", "five"])
                .unwrap();
        }
        dataset.with_shape(&[2, 3]);
        if chunked {
            dataset
                .with_chunks(&[1, 2])
                .with_deflate(DeflateLevel::Four);
        }
    });
    let file = if streaming { streamed } else { buffered };
    assert_eq!(
        file.dataset("labels")
            .unwrap()
            .read_string_region(&[0, 1], &[2, 2])
            .unwrap(),
        ["one", "two", "four", "five"]
    );
}

#[rstest]
#[case::start_rank(&[0], &[1, 1], FormatError::RegionRankMismatch { rank: 2, start: 1, count: 2 })]
#[case::count_rank(&[0, 0], &[1], FormatError::RegionRankMismatch { rank: 2, start: 2, count: 1 })]
#[case::start_bound(&[6, 0], &[0, 0], FormatError::RegionOutOfBounds { dimension: 0, start: 6, count: 0, extent: 5 })]
#[case::count_bound(&[1, 4], &[1, 3], FormatError::RegionOutOfBounds { dimension: 1, start: 4, count: 3, extent: 6 })]
#[case::count_overflow(&[1, 1], &[1, u64::MAX], FormatError::RegionOutOfBounds { dimension: 1, start: 1, count: u64::MAX, extent: 6 })]
fn region_requests_return_exact_rank_and_bounds_errors(
    #[case] start: &[u64],
    #[case] count: &[u64],
    #[case] expected: FormatError,
) {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("matrix")
        .with_u8_data(&[0; 30])
        .with_shape(&[5, 6]);
    let dataset = File::from_bytes(builder.finish().unwrap())
        .unwrap()
        .dataset("matrix")
        .unwrap();
    let error = dataset.read_raw_region(start, count).unwrap_err();
    let hdf5_pure::Error::Format(error) = error else {
        panic!("expected format error, got {error:?}")
    };
    assert_eq!(error, expected);
}

#[rstest]
#[case::empty_rows(&[1, 1], &[0, 2])]
#[case::empty_columns(&[1, 1], &[2, 0])]
#[case::empty_at_end(&[5, 6], &[0, 0])]
fn empty_regions_read_no_payload(
    #[case] start: &[u64],
    #[case] count: &[u64],
    #[values(false, true)] chunked: bool,
) {
    let mut builder = FileBuilder::new();
    let dataset = builder.create_dataset("matrix");
    dataset.with_u8_data(&[7; 30]).with_shape(&[5, 6]);
    if chunked {
        dataset.with_chunks(&[2, 3]);
    }
    let reads = Arc::new(Mutex::new(Vec::new()));
    let file = File::from_source(RecordingRowSource {
        inner: ReadSeekSource::new(Cursor::new(builder.finish().unwrap())).unwrap(),
        reads: Arc::clone(&reads),
    })
    .unwrap();
    let dataset = file.dataset("matrix").unwrap();
    reads.lock().unwrap().clear();
    assert_eq!(
        dataset.read_raw_region(start, count).unwrap(),
        Vec::<u8>::new()
    );
    assert!(reads.lock().unwrap().is_empty());
}

#[rstest]
#[case::contiguous(false)]
#[case::filtered_chunks(true)]
fn region_reads_fetch_only_intersecting_payload(#[case] chunked: bool) {
    let values: Vec<u32> = (0..64 * 128).collect();
    let mut builder = FileBuilder::new();
    let dataset = builder.create_dataset("matrix");
    dataset.with_u32_data(&values).with_shape(&[64, 128]);
    if chunked {
        dataset
            .with_chunks(&[4, 8])
            .with_deflate(DeflateLevel::Four);
    }
    let bytes = builder.finish().unwrap();
    let reads = Arc::new(Mutex::new(Vec::new()));
    let file = File::from_source(RecordingRowSource {
        inner: ReadSeekSource::new(Cursor::new(bytes)).unwrap(),
        reads: Arc::clone(&reads),
    })
    .unwrap();
    let dataset = file.dataset("matrix").unwrap();
    let (expected_reads, payload) = if chunked {
        let chunks = dataset.chunks().unwrap();
        let selected = chunks
            .iter()
            .find(|chunk| chunk.offset == [16, 40])
            .unwrap();
        (
            vec![(
                selected.address,
                usize::try_from(selected.storage_size).unwrap(),
            )],
            chunks
                .into_iter()
                .map(|chunk| chunk.address..chunk.address + chunk.storage_size)
                .collect::<Vec<_>>(),
        )
    } else {
        let hdf5_pure::Layout::Contiguous {
            address: Some(address),
            size,
        } = dataset.layout().unwrap()
        else {
            panic!("expected allocated contiguous layout")
        };
        (
            vec![(address + 2218 * 4, 12), (address + 2346 * 4, 12)],
            std::iter::once(address..address + size).collect(),
        )
    };
    reads.lock().unwrap().clear();
    assert_eq!(
        dataset.read_u32_region(&[17, 42], &[2, 3]).unwrap(),
        [2218, 2219, 2220, 2346, 2347, 2348]
    );
    let payload_reads: Vec<_> = reads
        .lock()
        .unwrap()
        .iter()
        .copied()
        .filter(|&(offset, length)| {
            payload
                .iter()
                .any(|range| offset < range.end && offset + length as u64 > range.start)
        })
        .collect();
    assert_eq!(payload_reads, expected_reads);
}

#[rstest]
#[case::scalar(&[], &[42.0])]
#[case::empty(&[0], &[])]
fn region_reads_preserve_scalar_and_empty_dataspaces(
    #[case] shape: &[u64],
    #[case] expected: &[f64],
) {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(expected)
        .with_shape(shape);
    let dataset = File::from_bytes(builder.finish().unwrap())
        .unwrap()
        .dataset("data")
        .unwrap();
    let start = vec![0; shape.len()];
    assert_eq!(dataset.read_f64_region(&start, shape).unwrap(), expected);
    assert_eq!(
        dataset.read_f64_rows(u64::MAX, u64::MAX).unwrap(),
        Vec::<f64>::new()
    );
}

#[rstest]
#[case::wide_leading_coordinate(&[4_294_967_424, 0], &[4_294_967_360, 0], &[1, 0])]
#[case::wide_trailing_coordinate(&[2, 0, 4_294_967_296], &[1, 0, 4_294_967_295], &[1, 0, 1])]
fn empty_regions_preserve_file_width_coordinates(
    #[case] shape: &[u64],
    #[case] start: &[u64],
    #[case] count: &[u64],
    #[values("compact", "contiguous", "unallocated", "chunked")] layout: &str,
) {
    let file = File::from_bytes(empty_row_file(shape, layout)).unwrap();
    assert_eq!(
        file.dataset("d")
            .unwrap()
            .read_raw_region(start, count)
            .unwrap(),
        Vec::<u8>::new()
    );
}

#[rstest]
#[case::full_columns(4, 4)]
#[case::split_columns(4, 2)]
fn regions_read_high_chunk_coordinates(#[case] columns: u32, #[case] chunk_columns: u32) {
    let file = File::from_bytes(sparse_high_row_file(columns, chunk_columns)).unwrap();
    assert_eq!(
        file.dataset("d")
            .unwrap()
            .read_f64_region(&[HIGH_ROW + 1, 1], &[2, 2])
            .unwrap(),
        [106.0, 107.0, 110.0, 111.0]
    );
}

#[test]
fn zero_count_regions_preserve_large_nonempty_extents() {
    let file = File::from_bytes(empty_row_file(&[2, 4_294_967_296], "unallocated")).unwrap();
    assert_eq!(
        file.dataset("d")
            .unwrap()
            .read_raw_region(&[1, 4_294_967_295], &[0, 1])
            .unwrap(),
        Vec::<u8>::new()
    );
}

#[rstest]
#[case::partial(2)]
#[case::whole(3)]
fn oversized_region_buffers_return_a_platform_error(#[case] rows: u64) {
    let limit = isize::MAX as u64 + 1;
    let columns = limit / 16;
    let file = File::from_bytes(empty_row_file(&[3, columns], "unallocated")).unwrap();
    let error = file
        .dataset("d")
        .unwrap()
        .read_raw_region(&[0, 0], &[rows, columns])
        .unwrap_err();
    let hdf5_pure::Error::Format(error) = error else {
        panic!("expected format error, got {error:?}")
    };
    assert_eq!(
        error,
        FormatError::ValueTooLargeForPlatform {
            value: rows * columns * 8,
            target: "isize"
        }
    );
}

const HIGH_ROW: u64 = 4_294_967_360;
const PAYLOAD_START: u64 = 4096;
