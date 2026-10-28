#![cfg(feature = "__hdf5-1.10")]
//! Interoperability checks for staged and immediate appends to Extensible Array datasets.
//!
//! Independent libhdf5 observations cover values, chunk records, and persisted free-space totals.
//! Reclamation checks include sparse tails across supported record locations.

use hdf5::Extent;
use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::{FileSpaceStrategy as NativeStrategy, Sizeof, SizeofInfo};
use hdf5_pure::FileBuilder;
use hdf5_pure::UserblockSize;
use hdf5_pure::{Chunk, Error, File, FileAccessProperties, MemoryStrategy, SyncPolicy};
use hdf5_pure_format::__private::ExtensibleArrayHeader;
use rstest::rstest;

use test_util_hdf5::dataset::{self, AppendInteropFixture, Unlimited};

use crate::test_util::array_index;

#[test]
fn pure_creates_pure_appends_c_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("d.h5");
    AppendInteropFixture::filtered_pure(12, 4).write(&path);
    // Aligned append (12 -> 20) then unaligned (20 -> 27).
    dataset::pure_append_staged(&path, "d", &(12..20).collect::<Vec<_>>());
    dataset::pure_append_staged(&path, "d", &(20..27).collect::<Vec<_>>());
    let expected: Vec<i32> = (0..27).collect();
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}

#[test]
#[cfg(target_endian = "little")]
fn c_creates_pure_appends_both_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("d.h5");
    AppendInteropFixture::filtered_libhdf5(20, 5).write(&path);
    dataset::pure_append_staged(&path, "d", &(20..33).collect::<Vec<_>>()); // unaligned 20 -> 33
    let expected: Vec<i32> = (0..33).collect();
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}

#[test]
#[cfg(target_endian = "little")]
fn c_partial_tail_replacement_preserves_values() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("partial-tail.h5");
    AppendInteropFixture::filtered_libhdf5(7, 5).write(&path);
    dataset::pure_append_staged(&path, "d", &[7, 8]);
    dataset::pure_append_staged(&path, "d", &[9, 10, 11, 12]);
    let expected: Vec<i32> = (0..13).collect();
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
}

#[test]
#[cfg(target_endian = "little")]
fn c_incompressible_kept_chunks_filter_mask_preserved() {
    // The load-bearing interop case: the C library stores incompressible chunks
    // uncompressed and records a nonzero per-chunk filter mask. The append must
    // carry each kept chunk's original mask (not force 0) into the rebuilt index,
    // or the C library would try to inflate stored-uncompressed bytes and read
    // garbage.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("d.h5");
    // 8 chunks of 5, all incompressible.
    let base = AppendInteropFixture::incompressible_libhdf5(0xABCD_1234, 40, 5).write(&path);

    let extra = dataset::incompressible(0x5555_AAAA, 17); // unaligned append (40 -> 57)
    dataset::pure_append_staged(&path, "d", &extra);

    let mut expected = base.clone();
    expected.extend_from_slice(&extra);
    assert_eq!(
        dataset::read_libhdf5::<i32>(&path, "d"),
        expected,
        "C must read the kept incompressible chunks intact"
    );
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
}

#[test]
#[cfg(target_endian = "little")]
fn append_to_c_empty_extensible() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("d.h5");
    // C creates an empty (0-length) resizable filtered dataset.
    Unlimited::<i32>::new("d", &[], 4)
        .filters(dataset::SHUFFLE_DEFLATE_6)
        .libhdf5_create(&path);
    dataset::pure_append_staged(&path, "d", &(0..10).collect::<Vec<_>>());
    let expected: Vec<i32> = (0..10).collect();
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}

#[test]
fn pure_unfiltered_append_c_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("d.h5");
    AppendInteropFixture::unfiltered_pure(10, 4).write(&path);
    dataset::pure_append_staged(&path, "d", &(10..23).collect::<Vec<_>>());
    let expected: Vec<i32> = (0..23).collect();
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}

#[test]
fn userblock_append_c_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ub.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UserblockSize::MIN);
    Unlimited::new("d", &(0..10).collect::<Vec<i32>>(), 4)
        .filters(dataset::SHUFFLE_DEFLATE_6)
        .add_to(&mut b);
    b.write(&path).unwrap();
    dataset::pure_append_staged(&path, "d", &(10..21).collect::<Vec<_>>()); // unaligned
    let expected: Vec<i32> = (0..21).collect();
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}

#[test]
fn large_paged_ea_append_c_reads() {
    // Append enough chunks (chunk length 1) that the rebuilt Extensible Array uses
    // super blocks / paged data blocks, then confirm the C library reads it back.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("d.h5");
    AppendInteropFixture::filtered_pure(64, 1).write(&path);
    dataset::pure_append_staged(&path, "d", &(64..4096).collect::<Vec<_>>());
    let expected: Vec<i32> = (0..4096).collect();
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}

#[rstest]
#[case::inline(1, false)]
#[case::direct(4, false)]
#[case::secondary(244, false)]
#[case::paged(132_090, false)]
#[case::page_paged(132_090, true)]
#[case::page_inline(1, true)]
#[case::page_secondary(244, true)]
fn immediate_replacement_reclaims_native_tail_at_each_record_location(
    #[case] slot: usize,
    #[case] paged: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native-retirement.h5");
    let len = slot * 4 + 1;
    let file = hdf5::File::with_options()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::V110, LibraryVersion::latest())
        })
        .with_fcpl(|properties| {
            properties
                .sizes(SizeofInfo {
                    sizeof_addr: Sizeof::Bytes8,
                    sizeof_size: Sizeof::Bytes8,
                })
                .file_space_strategy(NativeStrategy::FreeSpaceManager {
                    paged,
                    persist: true,
                    threshold: 1,
                })
        })
        .create(&path)
        .unwrap();
    let dataset = file
        .new_dataset::<i32>()
        .chunk((4,))
        .shape((Extent::resizable(len),))
        .deflate(6)
        .create("d")
        .unwrap();
    dataset.write_slice(&[7_i32], len - 1..len).unwrap();
    drop(dataset);
    file.close().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let header_at = test_util::bytes::sole_signature(&bytes, EA_HEADER_SIGNATURE);
    let header = ExtensibleArrayHeader::parse(&bytes, header_at, 8, 8).unwrap();
    assert_eq!(
        (
            header.idx_blk_elmts,
            header.min_dblk_nelmts,
            header.super_blk_min_data_ptrs
        ),
        (4, 16, 4)
    );
    assert_eq!(header.bit_geometry().page_nelmts().get(), 1024);
    let original = array_index::libhdf5_chunks(&path, 0);
    assert_eq!(original.len(), 1);
    assert_eq!(original[0].offset(), &[u64::try_from(slot * 4).unwrap()]);
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(MemoryStrategy::Auto)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    assert_eq!(file.dataset("d").unwrap().chunks().unwrap().len(), 1);
    let mut dataset = file.dataset("d").unwrap();
    let predecessor = dataset.chunks().unwrap().remove(0);
    dataset.append(&[8_i32]).unwrap();
    assert_retired_chunk_is_reusable(&file, &predecessor);
    let predecessor = dataset.chunks().unwrap().remove(0);
    dataset.append(&[9_i32]).unwrap();
    assert_retired_chunk_is_reusable(&file, &predecessor);
    let current = dataset.chunks().unwrap();
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].offset, original[0].offset());
    assert_ne!(current[0].address, original[0].address());
    let mut expected = vec![0; len - 1];
    expected.extend([7, 8, 9]);
    assert_eq!(dataset.read_i32().unwrap(), expected);
    drop(dataset);
    file.commit().unwrap();
    file.close().unwrap();
    let native = hdf5::File::open(&path).unwrap();
    assert_eq!(
        native.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        expected
    );
    let native_free = native.free_space();
    native.close().unwrap();
    let file = File::open(&path).unwrap();
    let chunks = array_index::libhdf5_chunks(&path, 0);
    assert_eq!(array_index::pure_chunks(&file), chunks);
    let persisted = file.persisted_free_sections().unwrap().unwrap();
    assert_eq!(persisted.free_sections().free_bytes(), native_free);
    for section in persisted.free_sections().sections() {
        for chunk in &chunks {
            let start = section.address();
            assert!(
                start + section.length().get() <= chunk.address()
                    || chunk.address() + chunk.size() <= start
            );
        }
    }
}

#[rstest]
#[case::two(Sizeof::Bytes2)]
#[case::four(Sizeof::Bytes4)]
fn narrow_native_tail_editing_is_rejected_without_changing_ownership(#[case] width: Sizeof) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("narrow-tail.h5");
    let native = hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_v110())
        .with_fcpl(|properties| {
            properties.sizes(SizeofInfo {
                sizeof_addr: width,
                sizeof_size: Sizeof::Bytes8,
            })
        })
        .create(&path)
        .unwrap();
    let dataset = native
        .new_dataset::<i32>()
        .chunk((4,))
        .shape((Extent::resizable(3),))
        .deflate(6)
        .create("d")
        .unwrap();
    dataset.write(&[1, 2, 3]).unwrap();
    drop(dataset);
    native.close().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [1, 2, 3]);
    drop(file);
    let error = File::open_rw(&path).unwrap_err();
    let Error::EditUnsupported(reason) = error else {
        panic!("expected unsupported editing width, got {error:?}")
    };
    assert_eq!(
        reason,
        "only 8-byte offsets and lengths are supported for in-place editing"
    );
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), [1, 2, 3]);
}

fn assert_retired_chunk_is_reusable(file: &File, predecessor: &Chunk) {
    assert!(
        file.reusable_free_sections()
            .unwrap()
            .sections()
            .iter()
            .any(|section| section.address() <= predecessor.address
                && predecessor.address + predecessor.storage_size <= section.end_address()),
        "predecessor {}..{} is unavailable",
        predecessor.address,
        predecessor.address + predecessor.storage_size
    );
}

/// The signature of a version 0 Extensible Array header, defined in "The Extensible Array Index"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
const EA_HEADER_SIGNATURE: &[u8; 4] = b"EAHD";
