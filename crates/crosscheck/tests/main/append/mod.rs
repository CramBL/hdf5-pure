#![cfg(feature = "__hdf5-1.10")]
//! Interoperability checks for staged and immediate appends to Extensible Array datasets.
//!
//! Independent libhdf5 observations cover values, chunk records, and persisted free-space totals.
//! Reclamation checks include sparse tails across supported record locations.

use hdf5::Extent;
use hdf5::datatype::ByteOrder as NativeByteOrder;
use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::{FileSpaceStrategy as NativeStrategy, Sizeof, SizeofInfo};
use hdf5_pure::FileBuilder;
use hdf5_pure::UserblockSize;
use hdf5_pure::{Chunk, Error, File, FileAccessProperties, MemoryStrategy, SyncPolicy};
use hdf5_pure_format::__private::ExtensibleArrayHeader;
use rstest::rstest;

use test_util_hdf5::__private::ownership_append;
use test_util_hdf5::dataset::{self, AppendInteropFixture, Unlimited};
use test_util_hdf5::numeric::ByteOrder;

use crate::test_util::array_index;

mod unallocated;

#[rstest]
#[case::flat_plain(false, false, false)]
#[case::flat_filtered(false, true, false)]
#[case::page_plain(true, false, false)]
#[case::page_filtered(true, true, false)]
#[case::page_plain_incompressible(true, false, true)]
#[case::page_filtered_incompressible(true, true, true)]
fn immediate_ownership_publication_matches_native_free_sections(
    #[case] paged: bool,
    #[case] filtered: bool,
    #[case] incompressible: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native-ownership.h5");
    let fixture = ownership_append::Fixture::write(&path, paged, filtered, incompressible);
    for cycle in 0..3 {
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new()
                .with_memory_strategy(MemoryStrategy::Bounded)
                .with_sync_policy(SyncPolicy::OnClose),
        )
        .unwrap();
        let mut target = file.dataset("d").unwrap();
        target
            .append(&ownership_append::payload(0x3cb6_7710 + cycle))
            .unwrap();
        drop(target);
        file.close().unwrap();
        fixture.assert_contents(&path, cycle);
    }
}

#[rstest]
#[case::flat_borrowed(false, false, false, false)]
#[case::flat_owned_filtered(false, true, true, false)]
#[case::page_borrowed(true, false, false, false)]
#[case::page_owned_filtered(true, true, true, false)]
#[case::flat_borrowed_metadata(false, false, false, true)]
#[case::flat_owned_filtered_metadata(false, true, true, true)]
#[cfg_attr(
    not(feature = "__hdf5-1.12"),
    ignore = "HDF5 1.10 PAGE metadata creation produces an invalid free-section checksum"
)]
#[case::page_borrowed_metadata(true, false, false, true)]
#[cfg_attr(
    not(feature = "__hdf5-1.12"),
    ignore = "HDF5 1.10 PAGE metadata creation produces an invalid free-section checksum"
)]
#[case::page_owned_filtered_metadata(true, true, true, true)]
fn appender_values_persist_across_staged_commits_and_native_writes(
    #[case] paged: bool,
    #[case] owned: bool,
    #[case] filtered: bool,
    #[case] native_metadata: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] policy: SyncPolicy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("appender-lifecycle.h5");
    let native = hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_v110())
        .with_fcpl(|properties| {
            properties.file_space_strategy(NativeStrategy::FreeSpaceManager {
                paged,
                persist: true,
                threshold: 1,
            })
        })
        .create(&path)
        .unwrap();
    let builder = native
        .new_dataset_builder()
        .empty_as(ByteOrder::LittleEndian.u8_datatype())
        .shape([Extent::resizable(4)])
        .chunk([4]);
    let builder = if filtered {
        builder.deflate(6)
    } else {
        builder
    };
    let dataset = builder.create("d").unwrap();
    drop(builder);
    assert_eq!(
        dataset.dtype().unwrap().byte_order(),
        NativeByteOrder::LittleEndian
    );
    dataset.write_raw(&[0_u8, 1, 2, 3]).unwrap();
    drop(dataset);
    let scratch = native
        .new_dataset_builder()
        .with_data(&vec![7_u8; 65_536])
        .create("scratch")
        .unwrap();
    let hole = (scratch.offset().unwrap(), scratch.storage_size());
    drop(scratch);
    let ceiling = native
        .new_dataset_builder()
        .with_data(&vec![9_u8; 8192])
        .create("ceiling")
        .unwrap();
    assert!(ceiling.offset().unwrap() >= hole.0 + hole.1);
    drop(ceiling);
    native.unlink("scratch").unwrap();
    native.close().unwrap();
    test_util_hdf5::file_space_oracle::assert_persisted_sections(&path);
    let mut access = FileAccessProperties::new()
        .with_memory_strategy(backing)
        .with_sync_policy(policy);
    if policy == SyncPolicy::OnClose {
        access = access.with_page_buffer_size(1 << 20);
    }
    let file = File::open_rw_with_options(&path, access).unwrap();
    let mut target = file.dataset("d").unwrap();
    if owned {
        let mut appender = file.dataset("d").unwrap().into_buffered_appender().unwrap();
        appender.append(&[4_u8, 5]).unwrap();
        drop(file.root().create_group("unrelated").unwrap());
        file.commit().unwrap();
        appender.flush().unwrap();
        assert_eq!(target.read_u8().unwrap(), (0..6).collect::<Vec<_>>());
        appender.append(&(6..14).collect::<Vec<u8>>()).unwrap();
        assert_eq!(appender.buffered_elements(), 2);
        drop(file.root().create_group("later").unwrap());
        file.commit().unwrap();
        appender.finish().unwrap();
    } else {
        let mut appender = target.buffered_appender().unwrap();
        appender.append(&[4_u8, 5]).unwrap();
        drop(file.root().create_group("unrelated").unwrap());
        file.commit().unwrap();
        appender.flush().unwrap();
        assert_eq!(
            file.dataset("d").unwrap().read_u8().unwrap(),
            (0..6).collect::<Vec<_>>()
        );
        appender.append(&(6..14).collect::<Vec<u8>>()).unwrap();
        assert_eq!(appender.buffered_elements(), 2);
        drop(file.root().create_group("later").unwrap());
        file.commit().unwrap();
        appender.finish().unwrap();
    }
    assert_eq!(target.read_u8().unwrap(), (0..14).collect::<Vec<_>>());
    drop(target);
    file.close().unwrap();
    let completed = std::fs::read(&path).unwrap();
    test_util_hdf5::file_space_oracle::assert_persisted_sections(&path);
    let native = hdf5::File::open(&path).unwrap();
    let dataset = native.dataset("d").unwrap();
    assert_eq!(
        dataset.read_raw::<u8>().unwrap(),
        (0..14).collect::<Vec<_>>()
    );
    assert_eq!(dataset.num_chunks(), Some(4));
    assert!(native.link_exists("unrelated"));
    assert!(native.link_exists("later"));
    drop(dataset);
    native.close().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), completed);
    let native = hdf5::File::open_rw(&path).unwrap();
    let ceiling = native.dataset("ceiling").unwrap();
    ceiling.write_raw(&[73_u8; 8192]).unwrap();
    drop(ceiling);
    if native_metadata {
        let dataset = native
            .new_dataset_builder()
            .with_data(&[73_u8; 8192])
            .create("native")
            .unwrap();
        assert_eq!(dataset.read_raw::<u8>().unwrap(), [73; 8192]);
        drop(dataset);
    }
    native.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_u8().unwrap(),
        (0..14).collect::<Vec<_>>()
    );
    assert_eq!(
        file.dataset("ceiling").unwrap().read_u8().unwrap(),
        [73; 8192]
    );
    if native_metadata {
        assert_eq!(
            file.dataset("native").unwrap().read_u8().unwrap(),
            [73; 8192]
        );
    }
    drop(file);
    test_util_hdf5::file_space_oracle::assert_persisted_sections(&path);
}

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
    let path = test_util::temp::temp_path("partial-tail.h5");
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
    let path = test_util::temp::temp_path("native-retirement.h5");
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
        .new_dataset_builder()
        .empty_as(ByteOrder::LittleEndian.i32_datatype())
        .chunk((4,))
        .shape((Extent::resizable(len),))
        .deflate(6)
        .create("d")
        .unwrap();
    assert_eq!(
        dataset.dtype().unwrap().byte_order(),
        NativeByteOrder::LittleEndian
    );
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
    assert_eq!(dataset.datatype().unwrap(), hdf5_pure::make_i32_type());
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
    let path = test_util::temp::temp_path("narrow-tail.h5");
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

#[test]
fn immediate_append_rejects_big_endian_storage_without_changing_the_file() {
    let path = test_util::temp::temp_path("big-endian.h5");
    let native = hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_v110())
        .create(&path)
        .unwrap();
    let dataset = native
        .new_dataset_builder()
        .empty_as(ByteOrder::BigEndian.i32_datatype())
        .shape([Extent::resizable(3)])
        .chunk([4])
        .deflate(6)
        .create("d")
        .unwrap();
    assert_eq!(
        dataset.dtype().unwrap().byte_order(),
        NativeByteOrder::BigEndian
    );
    dataset.write_raw(&[1_i32, 2, 3]).unwrap();
    drop(dataset);
    native.close().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let mut dataset = file.dataset("d").unwrap();
    assert_eq!(dataset.read_i32().unwrap(), [1, 2, 3]);
    let original = dataset.chunks().unwrap();
    let error = dataset.append(&[4_i32]).unwrap_err();
    let Error::AppendInPlaceUnsupported(reason) = error else {
        panic!("expected AppendInPlaceUnsupported, got {error:?}");
    };
    assert_eq!(
        reason,
        "append datatype does not match the on-disk dataset (wrong element type or byte order)"
    );
    assert_eq!(dataset.chunks().unwrap(), original);
    assert_eq!(dataset.shape().unwrap(), [3]);
    assert_eq!(dataset.read_i32().unwrap(), [1, 2, 3]);
    drop(dataset);
    file.close().unwrap();
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
