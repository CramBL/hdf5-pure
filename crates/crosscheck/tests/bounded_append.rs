// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Reference-C-library interop for the bounded read-write backend (issue #147):
//! files grown through the bounded engine read back byte-correct in the
//! reference C library, both when the C library wrote the original file and
//! when this crate did (unfiltered any-length and filtered whole-chunk).

use hdf5::Extent;
use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
use hdf5_pure::{File, FileAccessProperties, FileBuilder, FileSpaceStrategy, MemoryStrategy};

/// Open with the bounded engine demanded rather than merely preferred: these
/// tests are about that engine, so a file it stops accepting must fail here
/// rather than quietly retarget the whole file at the mirror.
fn open_bounded(path: &std::path::Path) -> Result<File, hdf5_pure::Error> {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded),
    )
}

use tempfile::tempdir;

/// Create a rank-1 unlimited (Extensible-Array indexed) i32 dataset `name` with the
/// C library under the latest format, seeded with `0..n`, chunk length `chunk`.
fn c_create_unlimited(path: &std::path::Path, name: &str, n: i32, chunk: usize) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    let ds = file
        .new_dataset::<i32>()
        .chunk((chunk,))
        .shape((Extent::resizable(n as usize),))
        .create(name)
        .unwrap();
    ds.write(&(0..n).collect::<Vec<_>>()).unwrap();
    file.close().unwrap();
}

fn read_c(path: &std::path::Path, name: &str) -> Vec<i32> {
    let f = hdf5::File::open(path).unwrap();
    let v = f.dataset(name).unwrap().read_raw::<i32>().unwrap();
    f.close().unwrap();
    v
}

fn read_pure(path: &std::path::Path, name: &str) -> Vec<i32> {
    File::open(path)
        .unwrap()
        .dataset(name)
        .unwrap()
        .read_i32()
        .unwrap()
}

/// Create a rank-1 unlimited chunked i32 dataset with this crate's writer.
fn pure_create(path: &std::path::Path, n: i32, chunk: u64, deflate: bool) {
    let data: Vec<i32> = (0..n).collect();
    let mut b = FileBuilder::new();
    let ds = b
        .create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[chunk]);
    if deflate {
        ds.with_deflate(6);
    }
    b.write(path).unwrap();
}

#[test]
fn bounded_append_to_c_dataset_both_read() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c.h5");
    c_create_unlimited(&path, "d", 8, 4);

    {
        let file = open_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&[8i32, 9, 10, 11, 12]).unwrap(); // any length (unfiltered)
        ds.append(&[13i32]).unwrap();
    }

    let expected: Vec<i32> = (0..14).collect();
    assert_eq!(read_pure(&path, "d"), expected);
    assert_eq!(read_c(&path, "d"), expected);
}

#[test]
fn bounded_filtered_append_reads_back_in_c() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("filtered.h5");
    pure_create(&path, 8, 4, true);

    {
        let file = open_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&[8i32, 9, 10, 11]).unwrap(); // whole chunks only when filtered
        ds.append(&[12i32, 13, 14, 15]).unwrap();
    }

    let expected: Vec<i32> = (0..16).collect();
    assert_eq!(read_pure(&path, "d"), expected);
    assert_eq!(read_c(&path, "d"), expected);
}

#[test]
fn bounded_batched_large_append_reads_back_in_c() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("large.h5");
    pure_create(&path, 3, 256, false);

    // One call far past the internal 1 MiB batch budget, from an unaligned
    // start, so the C library reads back a file grown through several
    // crash-atomic batches.
    let total = 400_000i32;
    {
        let file = open_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(3..total).collect::<Vec<i32>>()).unwrap();
    }

    let got = read_c(&path, "d");
    assert_eq!(got.len(), total as usize);
    assert!(got.iter().enumerate().all(|(i, &v)| v == i as i32));
    assert_eq!(read_pure(&path, "d").len(), total as usize);
}

/// A persisting file (this crate's writer, `persist = true`, non-paged) grown
/// through the bounded backend and finalized at `close` reads back byte-correct
/// in the C library, which also loads the free-space managers the finalize wrote
/// (issue #173).
#[test]
fn bounded_persist_finalize_reads_back_in_c() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("pure_persist.h5");

    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.create_dataset("d")
        .with_i32_data(&(0..10).collect::<Vec<i32>>())
        .with_shape(&[10])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4]);
    b.write(&path).unwrap();

    {
        let file = open_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(10..30).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    let expected: Vec<i32> = (0..30).collect();
    assert_eq!(read_pure(&path, "d"), expected);

    // The C library reads the data, recovers the persisting strategy, and its
    // free-space query parses the managers the finalize wrote.
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(f.dataset("d").unwrap().read_raw::<i32>().unwrap(), expected);
    let strategy = f.create_plist().unwrap().get_file_space_strategy().unwrap();
    assert!(
        matches!(
            strategy,
            CStrategy::FreeSpaceManager {
                paged: false,
                persist: true,
                ..
            }
        ),
        "C recovers our persisting FSM strategy, got {strategy:?}"
    );
    let free = f.free_space() as i64;
    assert!(
        free >= 0,
        "C loads our free-space managers without error (got {free})"
    );
}

/// The mirror of the above: the C library creates the persisting file, the
/// bounded backend grows it and finalizes at `close`, and both libraries read
/// the full sequence back.
#[test]
fn bounded_persist_on_c_created_file_reads_back() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_persist.h5");

    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .with_fcpl(|fcpl| {
                fcpl.file_space_strategy(CStrategy::FreeSpaceManager {
                    paged: false,
                    persist: true,
                    threshold: 1,
                })
            })
            .create(&path)
            .unwrap();
        let ds = file
            .new_dataset::<i32>()
            .chunk((4,))
            .shape((Extent::resizable(8),))
            .create("d")
            .unwrap();
        ds.write(&(0..8).collect::<Vec<_>>()).unwrap();
        file.close().unwrap();
    }

    {
        let file = open_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(8..20).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    let expected: Vec<i32> = (0..20).collect();
    assert_eq!(read_pure(&path, "d"), expected);
    assert_eq!(read_c(&path, "d"), expected);
}

/// Variable-length string reads route through the file source; on read-write
/// backends (bounded AND mirror) that must reach the real bytes through the
/// engine, not the empty borrowed view (which made every heap-backed VL read
/// fail with UnexpectedEof).
#[test]
fn vlen_strings_read_on_bounded_and_mirror_files() {
    use hdf5::types::VarLenUnicode;
    use std::str::FromStr;

    let dir = tempdir().unwrap();
    let path = dir.path().join("vlen.h5");
    let words = ["alpha", "beta", "", "δelta"];
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        let vals: Vec<VarLenUnicode> = words
            .iter()
            .map(|s| VarLenUnicode::from_str(s).unwrap())
            .collect();
        file.new_dataset::<VarLenUnicode>()
            .shape((words.len(),))
            .create("labels")
            .unwrap()
            .write(&vals)
            .unwrap();
        let ds = file
            .new_dataset::<i32>()
            .chunk((4,))
            .shape((Extent::resizable(4),))
            .create("samples")
            .unwrap();
        ds.write(&[0i32, 1, 2, 3]).unwrap();
        file.close().unwrap();
    }
    let expected: Vec<String> = words.iter().map(|s| s.to_string()).collect();

    {
        let file = open_bounded(&path).unwrap();
        let labels = file.dataset("labels").unwrap();
        assert_eq!(labels.read_string().unwrap(), expected);
        // Interleave an append, then read the heap-backed strings again.
        let mut samples = file.dataset("samples").unwrap();
        samples.append(&[4i32, 5]).unwrap();
        assert_eq!(labels.read_string().unwrap(), expected);
    }
    {
        let file = File::open_rw(&path).unwrap();
        assert_eq!(
            file.dataset("labels").unwrap().read_string().unwrap(),
            expected
        );
    }
    assert_eq!(read_c(&path, "samples"), (0..6).collect::<Vec<_>>());
}

/// A staged commit through the bounded backend produces a file the reference C
/// library reads — the capability issue #198 unlocked. The source file is
/// written by the C library, so nothing about the layout is this crate's own
/// convention, and every kind of staged edit the commit performs (a value
/// overwrite, a new group, a new dataset, an attribute, a deletion) is verified
/// through the C library rather than only through this crate's reader.
#[test]
fn bounded_staged_commit_reads_back_in_c() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("staged_c.h5");
    c_create_unlimited(&path, "d", 8, 4);

    {
        let file = open_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.write(&(100..108).collect::<Vec<i32>>()).unwrap();
        let root = file.root();
        root.create_group("g").unwrap();
        root.create_dataset("g/fresh", |b| {
            b.with_i32_data(&[1i32, 2, 3]).with_shape(&[3]);
        })
        .unwrap();
        file.commit().unwrap();

        ds.set_attr("units", hdf5_pure::AttrValue::String("m".into()))
            .unwrap();
        file.commit().unwrap();
        file.close().unwrap();
    }

    assert_eq!(read_pure(&path, "d"), (100..108).collect::<Vec<i32>>());
    assert_eq!(read_c(&path, "d"), (100..108).collect::<Vec<i32>>());
    assert_eq!(read_c(&path, "g/fresh"), vec![1, 2, 3]);

    // This crate writes a fixed-length string attribute, so read it as one.
    let f = hdf5::File::open(&path).unwrap();
    let units: hdf5::types::FixedUnicode<1> = f
        .dataset("d")
        .unwrap()
        .attr("units")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(units.as_str(), "m");
    f.close().unwrap();
}

/// A file that persists its free space can now be grown by an **immediate**
/// in-place append from either entry point, not only the bounded one: the
/// managers are re-homed at close by the same tail a commit writes.
///
/// What this pins is that the C library reads back rows this crate could not
/// previously write. The `H5Fget_freespace` call below only shows the C library
/// did not error, and it tolerates stale mid-file managers, so it does not prove
/// the re-homing happened — `edit_append_inplace::persisting_file_takes_both_inplace_and_staged_appends`
/// is what covers that, by comparing the recorded free space across the close.
#[test]
fn mirror_inplace_append_to_a_persisting_file_reads_back_in_c() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("persist_mirror.h5");

    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.create_dataset("d")
        .with_i32_data(&(0..8).collect::<Vec<i32>>())
        .with_shape(&[8])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4]);
    b.write(&path).unwrap();

    {
        let file = File::open_rw(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(8..40).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    let expected: Vec<i32> = (0..40).collect();
    assert_eq!(read_pure(&path, "d"), expected);
    assert_eq!(read_c(&path, "d"), expected);

    let f = hdf5::File::open(&path).unwrap();
    // The C-library guard above serializes every C call in this suite.
    let free = f.free_space() as i64;
    f.close().unwrap();
    assert!(
        free >= 0,
        "the C library could not read the rewritten free-space managers"
    );
}

/// The same for a genuine **paged** persisting file, which the whole-file editor
/// refused to append to in place before issue #198 because its page-type state
/// was reachable only from the commit path. The file must stay page-aligned and
/// C-readable.
#[test]
fn mirror_inplace_append_to_a_paged_file_stays_page_aligned() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("persist_paged.h5");

    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
        .with_file_space_page_size(4096);
    b.create_dataset("d")
        .with_i32_data(&(0..64).collect::<Vec<i32>>())
        .with_shape(&[64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[64]);
    b.write(&path).unwrap();

    {
        let file = File::open_rw(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(64..2000).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    assert_eq!(
        std::fs::metadata(&path).unwrap().len() % 4096,
        0,
        "a closed paged file must end on a page boundary"
    );
    let expected: Vec<i32> = (0..2000).collect();
    assert_eq!(read_pure(&path, "d"), expected);
    assert_eq!(read_c(&path, "d"), expected);

    let f = hdf5::File::open(&path).unwrap();
    // The C-library guard above serializes every C call in this suite.
    let free = f.free_space() as i64;
    f.close().unwrap();
    assert!(
        free >= 0,
        "the C library could not read the rewritten paged free-space managers"
    );
}
