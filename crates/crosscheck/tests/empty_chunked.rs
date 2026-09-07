// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! An empty (zero-element) chunked dataset, across both writers and both readers
//! (issue #284). This is the shape an incremental writer declares its schema at —
//! one resizable dataset per column, grown as batches arrive — so it has to be
//! writable by the in-place edit engine, readable by the C library, and growable
//! afterwards by either.
//!
//! The two libraries reach the same dataset from opposite directions, and the
//! difference is what these tests pin. This crate allocates the chunk index
//! eagerly, so its empty dataset names an index over zero chunks. The C library
//! allocates lazily and leaves the layout message's address undefined until the
//! first chunk is written, so *its* empty dataset names no index at all. Both are
//! valid, and reading either has to yield an empty buffer rather than an error.

use hdf5::Extent;
use hdf5::file::LibraryVersion;
use hdf5_pure::{File, FileBuilder};
use tempfile::tempdir;

/// Write a starter file and add one empty chunked dataset to it through the
/// in-place edit engine, the path issue #284 refused.
fn edit_in_an_empty_chunked_dataset(path: &std::path::Path, unlimited: bool) {
    let mut b = FileBuilder::new();
    b.create_dataset("existing").with_f64_data(&[1.0, 2.0]);
    b.write(path).unwrap();

    let session = File::open_rw(path).unwrap();
    session
        .root()
        .create_dataset("col", |b| {
            b.with_i64_data(&[]).with_shape(&[0]).with_chunks(&[512]);
            if unlimited {
                b.with_maxshape(&[u64::MAX]);
            }
        })
        .unwrap();
    session.commit().unwrap();
}

/// The issue's round trip: the edit engine declares the column empty, then the C
/// library reads it, extends it, and writes into it — and this crate reads back
/// what the C library wrote. An index built over zero chunks that the C library
/// could read but not grow would pass a read-only check and fail here.
#[test]
fn c_grows_an_empty_chunked_dataset_added_in_place() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("edited.h5");
    edit_in_an_empty_chunked_dataset(&path, true);

    {
        let file = hdf5::File::open_rw(&path).unwrap();
        let ds = file.dataset("col").unwrap();
        assert_eq!(ds.shape(), vec![0]);
        assert!(ds.read_raw::<i64>().unwrap().is_empty());
        ds.resize((4,)).unwrap();
        ds.write(&[10i64, 20, 30, 40]).unwrap();
        drop(ds);
        file.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("col").unwrap().read_i64().unwrap(),
        vec![10, 20, 30, 40]
    );
    // The pre-existing dataset the edit appended past is untouched.
    assert_eq!(
        file.dataset("existing").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0]
    );
}

/// The same dataset without `maxshape`: a fixed-shape empty chunked dataset,
/// whose index is a fixed array over zero chunks rather than an extensible one.
/// The C library has to accept that too, or the edit engine is emitting a
/// structure only its own reader understands.
#[test]
fn c_reads_a_fixed_shape_empty_chunked_dataset_added_in_place() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("edited_fixed.h5");
    edit_in_an_empty_chunked_dataset(&path, false);

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("col").unwrap();
    assert_eq!(ds.shape(), vec![0]);
    assert!(ds.read_raw::<i64>().unwrap().is_empty());
    // Shape and an empty read are equally true of a *contiguous* dataset, so
    // they cannot tell whether the edit engine emitted chunked storage at all.
    // These reach the C library's own view of the chunk index, which is the
    // thing under test.
    assert!(ds.is_chunked(), "the C library must see chunked storage");
    assert_eq!(ds.chunk(), Some(vec![512]));
    // `num_chunks` opens the index. A Fixed Array declaring zero entries — what
    // this crate wrote before it adopted the C library's convention of leaving a
    // chunk-less fixed-shape dataset unallocated — makes this call *fail*, while
    // reads and `shape` keep working. It is the only assertion here that can see
    // the difference.
    assert_eq!(
        ds.num_chunks(),
        Some(0),
        "the C library must be able to open the chunk index"
    );
}

/// The same property for the whole-file writer, at every rank and both ways of
/// arriving at zero elements. `num_chunks` is the assertion that opens the index.
#[test]
fn c_opens_the_chunk_index_of_every_empty_chunked_dataset_we_write() {
    let cases: &[(&str, &[u64], &[u64], bool)] = &[
        ("1-D fixed", &[0], &[64], false),
        ("1-D unlimited", &[0], &[64], true),
        ("2-D fixed, leading zero", &[0, 3], &[4, 3], false),
        ("2-D unlimited, leading zero", &[0, 3], &[4, 3], true),
        ("2-D fixed, inner zero", &[4, 0], &[2, 8], false),
        ("3-D fixed", &[0, 2, 3], &[4, 2, 3], false),
    ];
    for &(label, shape, chunks, unlimited) in cases {
        let dir = tempdir().unwrap();
        let path = dir.path().join("w.h5");
        let mut b = FileBuilder::new();
        {
            let ds = b
                .create_dataset("col")
                .with_i64_data(&[])
                .with_shape(shape)
                .with_chunks(chunks);
            if unlimited {
                // Only the leading dimension: an Extensible Array indexes
                // exactly one unlimited dimension, and the C library refuses to
                // open a dataset that declares two.
                let mut ms = shape.to_vec();
                ms[0] = u64::MAX;
                ds.with_maxshape(&ms);
            }
        }
        b.write(&path).unwrap();

        let file = hdf5::File::open(&path).unwrap();
        let ds = file.dataset("col").unwrap();
        assert!(ds.is_chunked(), "[{label}] chunked storage");
        let expected_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        assert_eq!(ds.shape(), expected_shape, "[{label}] shape");
        assert_eq!(
            ds.num_chunks(),
            Some(0),
            "[{label}] the C library must be able to open the chunk index"
        );
        assert!(ds.read_raw::<i64>().unwrap().is_empty(), "[{label}] read");
    }
}

/// A filtered dataset created empty has no chunk to size its index element from,
/// and the width it declares has to be the one its *first* chunk will need — the
/// reference C library encodes to whatever the header says.
///
/// Sized deliberately past 64 KiB of incompressible data per chunk, which is
/// exactly what the old 2-byte field could not hold: the C library wrote chunks
/// the field could not address, and the result read back as a truncated deflate
/// stream. The chunk geometry is the same in both files, so the widths must
/// match; asserted as equality rather than as a literal, since the rule is that
/// the width tracks the geometry, not that it takes a particular value.
#[test]
fn an_empty_filtered_dataset_declares_the_width_its_first_chunk_needs() {
    /// The `EAHD` element-size byte: signature (4), version (1), client id (1).
    fn ea_element_size(path: &std::path::Path) -> u8 {
        let b = std::fs::read(path).unwrap();
        let h = (0..b.len() - 4)
            .find(|&i| &b[i..i + 4] == b"EAHD")
            .expect("an extensible array header");
        b[h + 6]
    }

    // xorshift, so deflate cannot shrink a chunk below the old field's ceiling.
    let mut state = 0x1234_5678u32;
    let chunk_elems = 32_768usize; // 128 KiB per chunk
    let data: Vec<i32> = (0..chunk_elems)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as i32
        })
        .collect();

    let dir = tempdir().unwrap();
    let empty = dir.path().join("empty.h5");
    let seeded = dir.path().join("seeded.h5");
    let write = |path: &std::path::Path, values: &[i32]| {
        let mut b = FileBuilder::new();
        b.create_dataset("col")
            .with_i32_data(values)
            .with_shape(&[values.len() as u64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[chunk_elems as u64])
            .with_deflate(6);
        b.write(path).unwrap();
    };
    write(&empty, &[]);
    write(&seeded, &data);
    assert_eq!(
        ea_element_size(&empty),
        ea_element_size(&seeded),
        "an empty filtered dataset must declare the same element width as the \
         identical one that already holds a chunk"
    );

    // The consequence, end to end: the C library fills the dataset the edit
    // engine declared, and this crate reads back what it wrote.
    {
        let file = hdf5::File::open_rw(&empty).unwrap();
        let ds = file.dataset("col").unwrap();
        ds.resize((chunk_elems,)).unwrap();
        ds.write(&data).unwrap();
        drop(ds);
        file.close().unwrap();
    }
    let file = File::open(&empty).unwrap();
    assert_eq!(file.dataset("col").unwrap().read_i32().unwrap(), data);
}

/// And the same dataset grown by *this* crate rather than the C library. The
/// in-place append writes into the index the empty dataset declared, so a width
/// that does not fit the chunk is refused outright — a dataset that should have
/// been appendable, refused because of how it was created.
#[test]
fn an_empty_filtered_dataset_accepts_an_append_that_fills_its_chunk() {
    let mut state = 0x9E37_79B9u32;
    let chunk_elems = 32_768usize;
    let data: Vec<i32> = (0..chunk_elems)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as i32
        })
        .collect();

    let dir = tempdir().unwrap();
    let path = dir.path().join("append.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("col")
        .with_i32_data(&[])
        .with_shape(&[0])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[chunk_elems as u64])
        .with_deflate(6);
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session.dataset("col").unwrap().append(&data).unwrap();
    }
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("col").unwrap().read_i32().unwrap(), data);
    drop(file);
    let file = hdf5::File::open(&path).unwrap();
    assert_eq!(
        file.dataset("col").unwrap().read_raw::<i32>().unwrap(),
        data
    );
}

/// The whole-file writer's empty chunked datasets, which the edit engine's are
/// meant to be indistinguishable from. (Only the 1.10+ output is exercised:
/// chunked storage needs the version 4 layout message, so the 1.8 format refuses
/// it outright.)
#[test]
fn c_reads_the_whole_file_writers_empty_chunked_datasets() {
    for unlimited in [false, true] {
        let dir = tempdir().unwrap();
        let path = dir.path().join("whole.h5");
        let mut b = FileBuilder::new();
        {
            let ds = b
                .create_dataset("col")
                .with_i64_data(&[])
                .with_shape(&[0])
                .with_chunks(&[512]);
            if unlimited {
                ds.with_maxshape(&[u64::MAX]);
            }
        }
        b.write(&path).unwrap();

        let file = hdf5::File::open(&path).unwrap();
        let ds = file.dataset("col").unwrap();
        assert_eq!(ds.shape(), vec![0], "unlimited={unlimited}");
        assert!(
            ds.read_raw::<i64>().unwrap().is_empty(),
            "unlimited={unlimited}"
        );
    }
}

/// The other direction, and the one this crate used to fail: the C library
/// allocates a chunked dataset's index lazily, so an empty one carries the
/// undefined address. Reading it is the empty buffer, and reading an unallocated
/// dataset that *owns* elements is that many fill values — both fall out of the
/// same expression, which is why a zero-element one needs no special case.
#[test]
fn pure_reads_unallocated_empty_chunked_datasets_the_c_library_wrote() {
    for unlimited in [false, true] {
        for latest in [false, true] {
            let dir = tempdir().unwrap();
            let path = dir.path().join("c.h5");
            {
                let mut opts = hdf5::File::with_options();
                if latest {
                    opts.with_fapl(|p| {
                        p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest())
                    });
                }
                let file = opts.create(&path).unwrap();
                let builder = file.new_dataset::<i64>().chunk((512,));
                let ds = if unlimited {
                    builder.shape((Extent::resizable(0),)).create("col")
                } else {
                    builder.shape((0,)).create("col")
                }
                .unwrap();
                drop(ds);
                file.close().unwrap();
            }

            let label = format!("unlimited={unlimited} latest={latest}");
            // Both readers: the buffered one and the streaming one reach
            // different functions, and only one of them was covered when this
            // was written.
            let file = File::open(&path).unwrap();
            let ds = file.dataset("col").unwrap();
            assert_eq!(ds.shape().unwrap(), vec![0], "[{label}] buffered");
            assert_eq!(
                ds.read_i64().unwrap(),
                Vec::<i64>::new(),
                "[{label}] buffered"
            );
            drop(file);

            let file = File::open_streaming(&path).unwrap();
            let ds = file.dataset("col").unwrap();
            assert_eq!(ds.shape().unwrap(), vec![0], "[{label}] streaming");
            assert_eq!(
                ds.read_i64().unwrap(),
                Vec::<i64>::new(),
                "[{label}] streaming"
            );
        }
    }
}

/// A chunked dataset that owns elements and has no allocated index is the other
/// half of the same convention: the C library materializes its fill value for
/// every element, and so does this crate. It used to be an error, which made a
/// perfectly ordinary C-written file — one created and not yet written —
/// unreadable here.
///
/// Both fill flavors, and both readers, checked against what the C library
/// itself returns for the very same file rather than against a hard-coded
/// expectation.
#[test]
fn pure_materializes_an_unallocated_dataset_that_owns_elements() {
    for fill in [None, Some(7i32)] {
        for chunked in [false, true] {
            let label = format!("fill={fill:?} chunked={chunked}");
            let dir = tempdir().unwrap();
            let path = dir.path().join("unwritten.h5");
            {
                let file = hdf5::File::create(&path).unwrap();
                let mut builder = file.new_dataset::<i32>().shape((10,));
                if chunked {
                    builder = builder.chunk((4,));
                }
                if let Some(v) = fill {
                    builder = builder.fill_value(v);
                }
                let ds = builder.create("col").unwrap();
                drop(ds);
                file.close().unwrap();
            }

            // Ground truth: whatever the C library reads from this file.
            let expected = {
                let file = hdf5::File::open(&path).unwrap();
                file.dataset("col").unwrap().read_raw::<i32>().unwrap()
            };
            assert_eq!(
                expected,
                vec![fill.unwrap_or(0); 10],
                "[{label}] C ground truth"
            );

            for streaming in [false, true] {
                let file = if streaming {
                    File::open_streaming(&path).unwrap()
                } else {
                    File::open(&path).unwrap()
                };
                let ds = file.dataset("col").unwrap();
                assert_eq!(
                    ds.read_i32().unwrap(),
                    expected,
                    "[{label} streaming={streaming}] whole read"
                );
                // A window over unallocated storage answers the same way.
                assert_eq!(
                    ds.read_i32_rows(2, 5).unwrap(),
                    expected[2..7].to_vec(),
                    "[{label} streaming={streaming}] window"
                );
            }
        }
    }
}

/// The partially-allocated case, which is where the old behavior was *silently
/// wrong* rather than merely refusing: some chunks written, the rest never
/// allocated. Those never-allocated chunks read as zeros regardless of the
/// dataset's fill value, so a dataset filled with 7 came back with 0s in the
/// gaps and no error.
#[test]
fn pure_fills_the_gaps_of_a_partially_written_chunked_dataset() {
    for fill in [None, Some(7i32)] {
        let label = format!("fill={fill:?}");
        let dir = tempdir().unwrap();
        let path = dir.path().join("partial.h5");
        {
            let file = hdf5::File::create(&path).unwrap();
            let mut builder = file.new_dataset::<i32>().shape((10,)).chunk((5,));
            if let Some(v) = fill {
                builder = builder.fill_value(v);
            }
            let ds = builder.create("col").unwrap();
            // Only the first chunk; the second is never allocated.
            ds.write_slice(&[1i32, 2, 3, 4, 5], 0..5).unwrap();
            drop(ds);
            file.close().unwrap();
        }

        let expected = {
            let file = hdf5::File::open(&path).unwrap();
            file.dataset("col").unwrap().read_raw::<i32>().unwrap()
        };
        let mut want = vec![1, 2, 3, 4, 5];
        want.extend(std::iter::repeat_n(fill.unwrap_or(0), 5));
        assert_eq!(expected, want, "[{label}] C ground truth");

        for streaming in [false, true] {
            let file = if streaming {
                File::open_streaming(&path).unwrap()
            } else {
                File::open(&path).unwrap()
            };
            let ds = file.dataset("col").unwrap();
            assert_eq!(
                ds.read_i32().unwrap(),
                expected,
                "[{label} streaming={streaming}] whole read"
            );
            // A window landing entirely inside the unallocated chunk.
            assert_eq!(
                ds.read_i32_rows(5, 5).unwrap(),
                expected[5..].to_vec(),
                "[{label} streaming={streaming}] window over the gap"
            );
            // And one straddling the written chunk and the gap.
            assert_eq!(
                ds.read_i32_rows(3, 4).unwrap(),
                expected[3..7].to_vec(),
                "[{label} streaming={streaming}] straddling window"
            );
        }
    }
}

/// `H5D_FILL_TIME_NEVER` says the library never writes the fill value into
/// storage, so a dataset carrying it has no defined contents where nothing was
/// written. This crate answers that with deterministic zeros rather than with
/// the declared value, which would assert data nothing ever put there.
///
/// **The C library is deliberately not the oracle here**, unlike every other
/// test in this file. It has nothing to read for such a dataset, so `H5Dread`
/// leaves the caller's buffer as it found it: the identical file yields zeros on
/// macOS and the fill bytes on Linux, purely from what the last allocation left
/// behind. An earlier version of this test asserted the C library's answer and
/// passed locally while failing on two of three CI platforms. What is pinned
/// instead is this crate's own determinism, across both readers and both
/// storage layouts.
#[test]
fn a_fill_value_the_file_says_is_never_written_reads_as_zeros() {
    for (label, chunked, partial) in [
        ("contiguous, unwritten", false, false),
        ("chunked, unwritten", true, false),
        ("chunked, partly written", true, true),
    ] {
        let dir = tempdir().unwrap();
        let path = dir.path().join("never.h5");
        {
            let file = hdf5::File::create(&path).unwrap();
            let mut builder = file
                .new_dataset::<i32>()
                .shape((8,))
                .fill_value(7i32)
                .fill_time(hdf5::dataset::FillTime::Never);
            if chunked {
                builder = builder.chunk((4,));
            }
            let ds = builder.create("col").unwrap();
            if partial {
                ds.write_slice(&[1i32, 2, 3, 4], 0..4).unwrap();
            }
            drop(ds);
            file.close().unwrap();
        }

        // What was actually written is the only defined part; the rest is the
        // region under test.
        let mut expected = vec![0i32; 8];
        if partial {
            expected[..4].copy_from_slice(&[1, 2, 3, 4]);
        }

        for streaming in [false, true] {
            let file = if streaming {
                File::open_streaming(&path).unwrap()
            } else {
                File::open(&path).unwrap()
            };
            let ds = file.dataset("col").unwrap();
            assert_eq!(
                ds.read_i32().unwrap(),
                expected,
                "[{label} streaming={streaming}] whole read"
            );
            assert_eq!(
                ds.read_i32_rows(4, 4).unwrap(),
                expected[4..].to_vec(),
                "[{label} streaming={streaming}] window over the unwritten tail"
            );
            // The value is still *declared*, and the accessor still reports it:
            // what the write time changes is only what unallocated storage
            // reads as.
            assert_eq!(ds.fill_value::<i32>().unwrap(), Some(7));
        }
    }
}

/// `repack` of a dataset whose storage was never allocated **preserves** it: the
/// destination holds no chunks either, where it used to hold one written chunk
/// per grid slot, each full of the fill value (issue #293).
///
/// The values a read answers with cannot distinguish the two, which is what made
/// the materializing version look correct: a dataset storing a grid of fill
/// values reads exactly like one storing nothing. So the destination is checked
/// for what it *stores* — no chunks, and a file too small to hold the elements —
/// and the C library is asked to read it.
///
/// That last one is a smoke test rather than a check of the encoding: measured,
/// libhdf5 reads a Fixed Array over ten *empty* slots exactly as happily, and
/// reports zero chunks for it too. What pins the encoding is the unit test in
/// `chunked_write.rs`, which is what fails if the undefined address is traded
/// for an index over an empty grid.
#[test]
fn repack_preserves_a_never_written_dataset() {
    const N: usize = 1000;
    let dir = tempdir().unwrap();
    let src = dir.path().join("src.h5");
    let dst = dir.path().join("dst.h5");
    {
        let file = hdf5::File::create(&src).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape((N,))
            .chunk((100,))
            .fill_value(7i32)
            .create("col")
            .unwrap();
        drop(ds);
        file.close().unwrap();
    }
    // The source holds no chunks at all.
    let before = File::open(&src).unwrap();
    assert_eq!(before.dataset("col").unwrap().chunks().unwrap().len(), 0);
    drop(before);

    hdf5_pure::repack(&src, &dst, &hdf5_pure::RepackOptions::default()).unwrap();

    let after = File::open(&dst).unwrap();
    let ds = after.dataset("col").unwrap();
    assert_eq!(ds.read_i32().unwrap(), vec![7i32; N], "values round-trip");
    assert_eq!(
        ds.fill_value::<i32>().unwrap(),
        Some(7),
        "fill carried across"
    );
    assert_eq!(ds.shape().unwrap(), vec![N as u64], "shape carried across");
    assert_eq!(
        ds.chunks().unwrap().len(),
        0,
        "nothing was stored, so nothing is written out"
    );
    drop(after);

    // The elements are absent rather than merely compressed: the whole file is
    // smaller than the run of bytes they would occupy. Stated as a rule against
    // the dataset's own size so it holds whatever the surrounding metadata costs.
    let materialized = (N * core::mem::size_of::<i32>()) as u64;
    let dst_len = std::fs::metadata(&dst).unwrap().len();
    assert!(
        dst_len < materialized,
        "the destination ({dst_len} B) still carries the {materialized} B of \
         elements the source never stored"
    );

    // The reference library reads its own convention back.
    let c = hdf5::File::open(&dst).unwrap();
    assert_eq!(
        c.dataset("col").unwrap().read_raw::<i32>().unwrap(),
        vec![7i32; N],
        "the C library reads the repacked dataset as its fill value"
    );
}

/// A typed whole-dataset read sweeps row windows (issue #289), and a dataset the
/// C library allocated lazily is where that sweep meets storage that is not
/// there: no chunk exists at all, or some chunk grid slots hold data and others
/// never will.
///
/// Only the C library can write this file. Both cases have to answer the fill
/// value for every element no chunk covers, in *every* window and not just the
/// first — a sweep that carried the fill pattern into its first window and zeros
/// into the rest would pass a whole-read test and fail here.
///
/// The dataset is deliberately larger than the read's window budget: at 1 MiB of
/// stored bytes to a window rounded down to whole 1,000-row chunk bands, 400,000
/// `i32` is two windows — 262,000 rows and then 138,000.
#[test]
fn a_swept_typed_read_of_lazily_allocated_storage_reads_holes_as_fill() {
    const N: usize = 400_000;
    const CHUNK: usize = 1000;
    const FILL: i32 = -12_345;

    let dir = tempdir().unwrap();
    let never = dir.path().join("never.h5");
    let partly = dir.path().join("partly.h5");

    for (path, write_some) in [(&never, false), (&partly, true)] {
        let file = hdf5::File::create(path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape((N,))
            .chunk((CHUNK,))
            .fill_value(FILL)
            .create("col")
            .unwrap();
        if write_some {
            // Every fourth chunk, so written and unwritten slots fall in every
            // window of the sweep.
            let block: Vec<i32> = (0..CHUNK).map(|i| i as i32).collect();
            let mut start = 0;
            while start + CHUNK <= N {
                ds.write_slice(&block, start..start + CHUNK).unwrap();
                start += 4 * CHUNK;
            }
        }
        drop(ds);
        file.close().unwrap();
    }

    let expected_never = vec![FILL; N];
    let mut expected_partly = vec![FILL; N];
    let mut start = 0;
    while start + CHUNK <= N {
        for (i, slot) in expected_partly[start..start + CHUNK].iter_mut().enumerate() {
            *slot = i as i32;
        }
        start += 4 * CHUNK;
    }

    for (path, expected) in [(&never, &expected_never), (&partly, &expected_partly)] {
        let file = File::open_streaming(path).unwrap();
        let ds = file.dataset("col").unwrap();
        assert_eq!(
            &ds.read_i32().unwrap(),
            expected,
            "a swept read of {} did not match what the C library stored",
            path.display()
        );
        // The same values the whole-read path answers: a full-range window
        // delegates to it, which makes this the before-and-after comparison.
        assert_eq!(
            ds.read_i32().unwrap(),
            ds.read_i32_rows(0, N as u64).unwrap(),
            "the sweep and the whole read disagree over {}",
            path.display()
        );
    }
}
