#![cfg(feature = "__hdf5-1.10")]
//! Reference-C-library interop for chunked, filtered, and resizable
//! variable-length string datasets (issue #109).
//!
//! These datasets are the ones the writer used to refuse. Their element
//! references live inside chunks that are split (and compressed) before the file
//! layout would normally fix the global-heap addresses, so their collections are
//! placed *first* — immediately after the superblock — and patched in before any
//! chunk is encoded.
//!
//! That is a layout this crate had never emitted. The pure reader catches an
//! address that lands on no collection at all (it checks the `GCOL` signature),
//! but not one that lands on the *wrong* collection, and it accepts any file
//! whose placement and emission are consistently wrong in the same way. The C
//! library independently validates that the addresses name real collections at
//! offsets it computes itself, in a file it did not write.

use core::str::FromStr;

use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
use hdf5::types::VarLenUnicode;
use hdf5_pure::DeflateLevel;
use hdf5_pure::{AttrValue, File, FileBuilder, FileSpaceStrategy, MaxExtent};
use rstest::rstest;

use test_util_hdf5::{dataset, file, lock};

/// Assert both libraries agree with `expected`, and that the C library also
/// reports the layout we intended (chunked, not silently contiguous).
fn assert_both_read(path: &std::path::Path, name: &str, expected: &[String]) {
    assert_eq!(
        dataset::read_pure_strings(path, name),
        expected,
        "pure reader disagreed"
    );
    assert_eq!(
        dataset::read_libhdf5_strings(path, name),
        expected,
        "C library disagreed"
    );
}

fn words(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("value-{i}")).collect()
}

/// Plain chunked: no filter, so the references are merely split across chunks.
#[test]
fn c_library_reads_chunked_vlen_strings() {
    let _c = lock::libhdf5_guard();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chunked.h5");
    let data = words(10);

    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&data.iter().map(String::as_str).collect::<Vec<_>>())
        .with_chunks(&[3]);
    b.write(&path).unwrap();

    assert_both_read(&path, "labels", &data);

    // The layout really is chunked — otherwise this test would pass trivially by
    // falling back to the long-supported contiguous path.
    let f = hdf5::File::open(&path).unwrap();
    assert!(
        f.dataset("labels").unwrap().is_chunked(),
        "expected a chunked layout"
    );
}

/// Filtered: deflate compresses the reference bytes, so a placeholder address
/// left in them could not be patched afterwards. This is the case the early
/// placement exists for.
#[test]
fn c_library_reads_filtered_chunked_vlen_strings() {
    let _c = lock::libhdf5_guard();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("filtered.h5");
    let data = words(64);

    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&data.iter().map(String::as_str).collect::<Vec<_>>())
        .with_chunks(&[8])
        .with_deflate(DeflateLevel::Six);
    b.write(&path).unwrap();

    assert_both_read(&path, "labels", &data);

    let f = hdf5::File::open(&path).unwrap();
    assert!(
        !f.dataset("labels").unwrap().filters().is_empty(),
        "expected a filter pipeline on the dataset"
    );
}

/// Resizable: `maxshape` alone makes the dataset chunked, and the C library must
/// see a genuine unlimited dimension.
#[test]
fn c_library_reads_resizable_vlen_strings() {
    let _c = lock::libhdf5_guard();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resizable.h5");
    let data = words(7);

    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&data.iter().map(String::as_str).collect::<Vec<_>>())
        .with_shape(&[data.len() as u64])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[4]);
    b.write(&path).unwrap();

    assert_both_read(&path, "labels", &data);

    let f = hdf5::File::open(&path).unwrap();
    let ds = f.dataset("labels").unwrap();
    assert!(ds.is_resizable(), "expected an unlimited dimension");
}

/// A multi-dimensional chunked VL dataset: the chunk splitter walks rows, so the
/// per-element reference offsets differ from the flat 1-D case.
#[test]
fn c_library_reads_2d_chunked_vlen_strings() {
    let _c = lock::libhdf5_guard();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("grid.h5");
    let data = words(12); // 4 x 3

    let mut b = FileBuilder::new();
    b.create_dataset("grid")
        .with_vlen_strings(&data.iter().map(String::as_str).collect::<Vec<_>>())
        .with_shape(&[4, 3])
        .with_chunks(&[2, 3]);
    b.write(&path).unwrap();

    assert_both_read(&path, "grid", &data);
}

/// More heap objects than one collection holds (65,535 — see issue #189), so the
/// early placement must assign an address per collection and the references must
/// select the right one. A single wrong collection address here reads back as
/// garbage in the C library.
#[test]
fn c_library_reads_chunked_vlen_strings_across_heap_collections() {
    let _c = lock::libhdf5_guard();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("many.h5");
    let data = words(70_000);

    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&data.iter().map(String::as_str).collect::<Vec<_>>())
        .with_chunks(&[4096]);
    b.write(&path).unwrap();

    // Spot-check the boundaries rather than comparing 70k strings twice.
    let pure = dataset::read_pure_strings(&path, "labels");
    let c = dataset::read_libhdf5_strings(&path, "labels");
    assert_eq!(pure.len(), data.len());
    assert_eq!(c.len(), data.len());
    for i in [0usize, 1, 65_534, 65_535, 65_536, 69_999] {
        assert_eq!(pure[i], data[i], "pure reader differs at {i}");
        assert_eq!(c[i], data[i], "C library differs at {i}");
    }
}

/// The paged layout is a separate code path in the writer, with the collections
/// living in the metadata region. Cover it too, since the early placement moved
/// where that region starts.
#[test]
fn c_library_reads_paged_chunked_vlen_strings() {
    let _c = lock::libhdf5_guard();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paged.h5");
    let data = words(20);

    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&data.iter().map(String::as_str).collect::<Vec<_>>())
        .with_chunks(&[5]);
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1);
    b.write(&path).unwrap();

    assert_both_read(&path, "labels", &data);
}

/// A file mixing a chunked VL dataset (placed early) with a contiguous one
/// (placed late) exercises both placements in one layout, which is where an
/// address/emission-order mismatch would show up.
#[test]
fn c_library_reads_mixed_chunked_and_contiguous_vlen_strings() {
    let _c = lock::libhdf5_guard();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.h5");
    let chunked = words(9);
    let contiguous: Vec<String> = ["alpha", "beta", "gamma"]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();

    let mut b = FileBuilder::new();
    b.create_dataset("chunked")
        .with_vlen_strings(&chunked.iter().map(String::as_str).collect::<Vec<_>>())
        .with_chunks(&[4]);
    b.create_dataset("plain")
        .with_vlen_strings(&contiguous.iter().map(String::as_str).collect::<Vec<_>>());
    b.write(&path).unwrap();

    assert_both_read(&path, "chunked", &chunked);
    assert_both_read(&path, "plain", &contiguous);
}

/// Two chunked VL datasets whose collections are both placed early, with
/// **distinct** content.
///
/// This is what pins the placement/emission ordering invariant: `early_gcol` is
/// walked once to assign addresses and again to write the bytes, and the two
/// walks must agree. With one early dataset no ordering is expressible, and with
/// two identical ones a swap is a no-op — so a reversed emission order would be
/// invisible. Here it makes each dataset read the other's strings, and the C
/// library rejects the file outright.
#[test]
fn c_library_reads_two_distinct_chunked_vlen_string_datasets() {
    let _c = lock::libhdf5_guard();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two_early.h5");

    // Different lengths as well as different text, so a swap cannot survive by
    // the two collections happening to be the same size.
    let first: Vec<String> = (0..12).map(|i| format!("first-{i}")).collect();
    let second: Vec<String> = (0..30)
        .map(|i| format!("second-{i}-with-a-longer-body"))
        .collect();

    let mut b = FileBuilder::new();
    b.create_dataset("a")
        .with_vlen_strings(&first.iter().map(String::as_str).collect::<Vec<_>>())
        .with_chunks(&[4]);
    b.create_dataset("b")
        .with_vlen_strings(&second.iter().map(String::as_str).collect::<Vec<_>>())
        .with_chunks(&[7])
        .with_deflate(DeflateLevel::Six);
    b.write(&path).unwrap();

    assert_both_read(&path, "a", &first);
    assert_both_read(&path, "b", &second);
}

/// Five early-placed datasets, mixed with a contiguous VL dataset that keeps the
/// late placement. Exercises the two cursors in one file and makes an ordering
/// slip anywhere in the middle of `early_gcol` detectable, not just a full
/// reversal of two.
#[test]
fn c_library_reads_many_chunked_vlen_string_datasets() {
    let _c = lock::libhdf5_guard();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("many_early.h5");

    let sets: Vec<Vec<String>> = (0..5)
        .map(|d| {
            (0..(6 + d * 5))
                .map(|i| format!("ds{d}-item{i}{}", "x".repeat(d)))
                .collect()
        })
        .collect();
    let contiguous: Vec<String> = (0..4).map(|i| format!("late-{i}")).collect();

    let mut b = FileBuilder::new();
    for (d, data) in sets.iter().enumerate() {
        b.create_dataset(&format!("chunked{d}"))
            .with_vlen_strings(&data.iter().map(String::as_str).collect::<Vec<_>>())
            .with_chunks(&[3]);
    }
    b.create_dataset("contiguous")
        .with_vlen_strings(&contiguous.iter().map(String::as_str).collect::<Vec<_>>());
    b.write(&path).unwrap();

    for (d, data) in sets.iter().enumerate() {
        assert_both_read(&path, &format!("chunked{d}"), data);
    }
    assert_both_read(&path, "contiguous", &contiguous);
}

/// Overwriting a variable-length-string dataset in place (issue #321): the
/// staged element references are patched against a heap collection the *edit*
/// engine places, in a file that already has a layout around it.
///
/// The contiguous case, which is the one whose data block keeps its length and
/// so would otherwise take the commit's in-place fast path. That path never
/// rewrites the superblock, and placing a collection moves end-of-file — so
/// this is also the test that a file whose recorded end-of-file went stale
/// would fail, since the C library validates it where the pure reader, indexing
/// by offset, does not.
#[test]
fn c_library_reads_an_overwritten_contiguous_vlen_string_dataset() {
    let _c = lock::libhdf5_guard();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("overwrite_contiguous.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&["a", "bb", "ccc"]);
    b.write(&path).unwrap();

    // Deliberately longer than the originals: the element references keep their
    // width, so the data block does not move and only the heap collection grows.
    let after = vec![
        "replacement-one".to_string(),
        "replacement-two".to_string(),
        "replacement-three".to_string(),
    ];
    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("labels")
            .unwrap()
            .write_staged(|b| {
                b.with_vlen_strings(&after.iter().map(String::as_str).collect::<Vec<_>>());
            })
            .unwrap();
        session.commit().unwrap();
    }

    assert_both_read(&path, "labels", &after);
}

/// The filtered chunked counterpart. A deflated chunk's compressed length
/// depends on the heap addresses patched into it, so the overwrite cannot split
/// and encode before it resolves — which is what `MovingWrite::ChunkedStaged`
/// exists for. Nothing about that ordering is visible to a reader that only
/// checks the strings come back, so the C library reading the file it did not
/// write is the assertion that matters.
#[test]
fn c_library_reads_an_overwritten_filtered_chunked_vlen_string_dataset() {
    let _c = lock::libhdf5_guard();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("overwrite_filtered.h5");
    let before = words(64);

    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&before.iter().map(String::as_str).collect::<Vec<_>>())
        .with_chunks(&[8])
        .with_deflate(DeflateLevel::Six);
    b.write(&path).unwrap();

    let after: Vec<String> = (0..64).map(|i| format!("overwritten-value-{i}")).collect();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("labels")
            .unwrap()
            .write_staged(|b| {
                b.with_vlen_strings(&after.iter().map(String::as_str).collect::<Vec<_>>());
            })
            .unwrap();
        session.commit().unwrap();
    }

    assert_both_read(&path, "labels", &after);

    // The overwrite must not have quietly dropped the storage properties it was
    // supposed to preserve: this is a value overwrite, not a re-layout.
    let f = hdf5::File::open(&path).unwrap();
    let ds = f.dataset("labels").unwrap();
    assert!(
        ds.is_chunked(),
        "the overwrite must keep the chunked layout"
    );
    assert!(
        !ds.filters().is_empty(),
        "the overwrite must keep the filter pipeline"
    );
}

/// A file the **C library wrote**, overwritten by this crate. The layout, the
/// heap collections, and the object header are all the reference library's, so
/// this is the case where an assumption about how *this* crate lays a
/// variable-length dataset out would show up.
#[test]
#[cfg(target_endian = "little")]
fn a_c_written_vlen_string_dataset_can_be_overwritten() {
    let _c = lock::libhdf5_guard();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_written.h5");

    {
        let f = file::libhdf5_create_v18(&path);
        let data: Vec<VarLenUnicode> = ["alpha", "beta", "gamma"]
            .iter()
            .map(|s| s.parse::<VarLenUnicode>().unwrap())
            .collect();
        f.new_dataset::<VarLenUnicode>()
            .shape([3])
            .create("labels")
            .unwrap()
            .write(&data)
            .unwrap();
    }

    let after = vec![
        "delta".to_string(),
        "epsilon".to_string(),
        "zeta".to_string(),
    ];
    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("labels")
            .unwrap()
            .write_staged(|b| {
                b.with_vlen_strings(&after.iter().map(String::as_str).collect::<Vec<_>>());
            })
            .unwrap();
        session.commit().unwrap();
    }

    assert_both_read(&path, "labels", &after);
}

/// A **compact** variable-length-string dataset, overwritten in place
/// (issue #321).
///
/// The compact arm of `prepare_write` relocates the dataset and rebuilds its
/// header with the element bytes inline, so the resolved references travel in
/// the header rather than in a data block — the one plan where they do. Nothing
/// this crate writes is compact, so the source has to come from the reference
/// library, which is also what makes this a test of the arm rather than of the
/// writer's own habits.
#[test]
#[cfg(target_endian = "little")]
fn c_library_reads_an_overwritten_compact_vlen_string_dataset() {
    let _c = lock::libhdf5_guard();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("overwrite_compact.h5");

    {
        let f = file::libhdf5_create_v18(&path);
        let data: Vec<VarLenUnicode> = ["one", "two", "three"]
            .iter()
            .map(|s| s.parse::<VarLenUnicode>().unwrap())
            .collect();
        f.new_dataset::<VarLenUnicode>()
            .layout(hdf5::dataset::Layout::Compact)
            .shape([3])
            .create("labels")
            .unwrap()
            .write(&data)
            .unwrap();
    }
    // The source really is compact, or this test proves nothing about that arm.
    {
        let f = hdf5::File::open(&path).unwrap();
        assert!(
            matches!(
                f.dataset("labels").unwrap().layout(),
                hdf5::dataset::Layout::Compact
            ),
            "expected a compact source layout"
        );
    }

    let after = vec![
        "replacement-one".to_string(),
        "replacement-two".to_string(),
        "replacement-three".to_string(),
    ];
    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("labels")
            .unwrap()
            .write_staged(|b| {
                b.with_vlen_strings(&after.iter().map(String::as_str).collect::<Vec<_>>());
            })
            .unwrap();
        session.commit().unwrap();
    }

    assert_both_read(&path, "labels", &after);
}

#[rstest]
#[case::large_collections(1_024, 0, true)]
#[case::whole_page_collections(4_096, 4_096, true)]
#[case::small_collections(8_192, 8_192, true)]
#[case::nonpersistent_collections(8_192, 0, false)]
fn paged_global_heap_references_survive_creation_edit_and_reopen(
    #[case] page_size: u64,
    #[case] userblock: u64,
    #[case] persist: bool,
) {
    let _c = lock::libhdf5_guard();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paged_global_heaps.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_strategy(FileSpaceStrategy::Page, persist, 1);
    builder.with_file_space_page_size(page_size);
    builder.with_userblock(userblock);
    builder.set_attr("root_label", AttrValue::VarLenString("root-value".into()));
    let mut group = builder.create_group("group");
    group.set_attr("group_label", AttrValue::VarLenString("group-value".into()));
    builder.add_group(group.finish());
    builder
        .create_dataset("contiguous")
        .with_vlen_strings(&["first", "second"])
        .set_attr(
            "dataset_label",
            AttrValue::VarLenString("dataset-value".into()),
        );
    builder
        .create_dataset("filtered")
        .with_vlen_strings(&["compressed-first", "compressed-second"])
        .with_chunks(&[1])
        .with_deflate(DeflateLevel::Six);
    builder.write(&path).unwrap();
    assert_eq!(
        dataset::read_libhdf5_strings(&path, "contiguous"),
        ["first", "second"]
    );
    assert_eq!(
        dataset::read_libhdf5_strings(&path, "filtered"),
        ["compressed-first", "compressed-second"]
    );
    {
        let reference = hdf5::File::open(&path).unwrap();
        assert_eq!(
            reference
                .attr("root_label")
                .unwrap()
                .read_scalar::<VarLenUnicode>()
                .unwrap()
                .as_str(),
            "root-value"
        );
        assert_eq!(
            reference
                .group("group")
                .unwrap()
                .attr("group_label")
                .unwrap()
                .read_scalar::<VarLenUnicode>()
                .unwrap()
                .as_str(),
            "group-value"
        );
        assert_eq!(
            reference
                .dataset("contiguous")
                .unwrap()
                .attr("dataset_label")
                .unwrap()
                .read_scalar::<VarLenUnicode>()
                .unwrap()
                .as_str(),
            "dataset-value"
        );
    }
    {
        let editable = File::open_rw(&path).unwrap();
        editable
            .dataset("contiguous")
            .unwrap()
            .write_staged(|dataset| {
                dataset.with_vlen_strings(&["replacement-first", "replacement-second"]);
            })
            .unwrap();
        editable.commit().unwrap();
        editable
            .dataset("contiguous")
            .unwrap()
            .write_staged(|dataset| {
                dataset.with_vlen_strings(&["final-first", "final-second"]);
            })
            .unwrap();
        editable.commit().unwrap();
    }
    let expected = ["final-first", "final-second"];
    assert_eq!(dataset::read_libhdf5_strings(&path, "contiguous"), expected);
    assert_eq!(dataset::read_pure_strings(&path, "contiguous"), expected);
    assert_eq!(
        dataset::read_libhdf5_strings(&path, "filtered"),
        ["compressed-first", "compressed-second"]
    );
    {
        let reference = hdf5::File::open_rw(&path).unwrap();
        reference
            .new_dataset_builder()
            .with_data(&[12i32, 34])
            .create("c_added")
            .unwrap();
    }
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("c_added")
            .unwrap()
            .read_i32()
            .unwrap(),
        [12, 34]
    );
}

#[rstest]
#[case::whole_page(4_096, 0)]
#[case::small_collection(8_192, 8_192)]
fn libhdf5_reads_its_paged_vlen_dataset_after_a_raw_collection_replacement(
    #[case] page_size: u64,
    #[case] userblock: u64,
) {
    let _c = lock::libhdf5_guard();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reference_paged_vlen.h5");
    {
        let reference = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| {
                fcpl.userblock(userblock)
                    .file_space_strategy(CStrategy::FreeSpaceManager {
                        persist: true,
                        threshold: 1,
                        paged: true,
                    })
                    .file_space_page_size(page_size)
            })
            .create(&path)
            .unwrap();
        let words = [
            VarLenUnicode::from_str("reference-first").unwrap(),
            VarLenUnicode::from_str("reference-second").unwrap(),
        ];
        reference
            .new_dataset_builder()
            .with_data(&words)
            .create("labels")
            .unwrap();
    }
    assert_eq!(
        dataset::read_pure_strings(&path, "labels"),
        ["reference-first", "reference-second"]
    );
    {
        let editable = File::open_rw(&path).unwrap();
        editable
            .dataset("labels")
            .unwrap()
            .write_staged(|dataset| {
                dataset.with_vlen_strings(&["pure-first", "pure-second"]);
            })
            .unwrap();
        editable.commit().unwrap();
    }
    assert_eq!(
        dataset::read_libhdf5_strings(&path, "labels"),
        ["pure-first", "pure-second"]
    );
    assert_eq!(
        dataset::read_pure_strings(&path, "labels"),
        ["pure-first", "pure-second"]
    );
}
