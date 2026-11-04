use core::num::NonZeroUsize;
use std::fs;
use std::ops::Range;
use std::path::Path;

use ::test_util::extensible_array;
use hdf5_pure_format::__private::ExtensibleArrayCreationParameters;
use rstest::rstest;
use test_util_hdf5::__private::unallocated_append;
use test_util_hdf5::__private::unallocated_append::ReadOutcome;
use test_util_hdf5::dataset;
use test_util_hdf5::file_space_oracle;

use crate::File;
use crate::FileAccessProperties;
use crate::FileSpacePageSize;
use crate::MemoryStrategy;
use crate::SyncPolicy;
use crate::edit::test_util;

#[rstest]
#[case::direct(Growth::Direct)]
#[case::secondary(Growth::Secondary)]
#[case::paged_parent(Growth::PagedParent)]
fn short_array_pages_preserve_values_and_record_native_buffered_reads(
    #[case] growth: Growth,
    #[values(false, true)] filtered: bool,
    #[values(512, 4_096)] page: u64,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] sync: SyncPolicy,
) {
    let directory = tempfile::tempdir().unwrap();
    let native = directory.path().join("native.h5");
    let pure = directory.path().join("pure.h5");
    let parameters = ExtensibleArrayCreationParameters::parse(&[16, 4, 2, 16, 5]).unwrap();
    let size = FileSpacePageSize::try_from(page).unwrap();
    let native_header = unallocated_append::paged_fixture(&native, filtered, size, parameters);
    let pure_header = unallocated_append::paged_fixture(&pure, filtered, size, parameters);
    let expected: Vec<i32> = (0..growth.chunks() * 4)
        .map(|index| index * 7_919 + 17)
        .collect();
    unallocated_append::write_values(&native, &expected);
    let file = File::open_rw_with_options(
        &pure,
        FileAccessProperties::new()
            .with_memory_strategy(backing)
            .with_sync_policy(sync),
    )
    .unwrap();
    let mut target = file.dataset("d").unwrap();
    target.append(&expected).unwrap();
    assert_eq!(target.read_i32().unwrap(), expected);
    drop(target);
    file.close().unwrap();
    let native_bytes = fs::read(&native).unwrap();
    let pure_bytes = fs::read(&pure).unwrap();
    assert_geometry(&native, native_header, parameters, growth, filtered);
    assert_geometry(&pure, pure_header, parameters, growth, filtered);
    let native_file = File::open(&native).unwrap();
    let native_target = native_file.dataset("d").unwrap();
    assert_eq!(native_target.read_i32().unwrap(), expected);
    let tail = expected.len() - 64;
    assert_eq!(
        native_target
            .read_i32_rows(u64::try_from(tail).unwrap(), 64)
            .unwrap(),
        expected[tail..]
    );
    drop(native_target);
    native_file.close().unwrap();
    let pure_file = File::open(&pure).unwrap();
    let pure_target = pure_file.dataset("d").unwrap();
    assert_eq!(pure_target.read_i32().unwrap(), expected);
    assert_eq!(
        pure_target
            .read_i32_rows(u64::try_from(tail).unwrap(), 64)
            .unwrap(),
        expected[tail..]
    );
    drop(pure_target);
    pure_file.close().unwrap();
    let buffer = NonZeroUsize::new(usize::try_from(2 * page).unwrap()).unwrap();
    compare_reads(&native, &pure, None, None, &expected, ExpectedReads::Values);
    compare_reads(
        &native,
        &pure,
        None,
        Some(tail..expected.len()),
        &expected[tail..],
        ExpectedReads::Values,
    );
    let outcomes = growth.buffered_whole_reads(filtered, page);
    compare_reads(&native, &pure, Some(buffer), None, &expected, outcomes);
    compare_reads(
        &native,
        &pure,
        Some(buffer),
        Some(tail..expected.len()),
        &expected[tail..],
        if matches!(growth, Growth::PagedParent) && page == 512 {
            ExpectedReads::Checksums
        } else {
            ExpectedReads::Values
        },
    );
    compare_reads(
        &native,
        &pure,
        Some(buffer),
        Some(0..4),
        &expected[..4],
        ExpectedReads::Values,
    );
    assert_eq!(fs::read(&native).unwrap(), native_bytes);
    assert_eq!(fs::read(&pure).unwrap(), pure_bytes);
    dataset::pure_append(&pure, "d", &[31_i32, 37, 41]);
    let mut expected = expected;
    expected.extend([31, 37, 41]);
    assert_eq!(dataset::read_libhdf5::<i32>(&pure, "d"), expected);
    assert_eq!(dataset::read_pure::<i32>(&pure, "d"), expected);
    file_space_oracle::assert_persisted_sections(&pure);
    file_space_oracle::assert_manager_metadata_bytes(&pure);
}

fn assert_geometry(
    path: &Path,
    header: usize,
    parameters: ExtensibleArrayCreationParameters,
    growth: Growth,
    filtered: bool,
) {
    unallocated_append::inspect_initialized_index(path, header, parameters.encode());
    let engine = test_util::open_engine(path);
    let header = u64::try_from(header).unwrap();
    engine.assert_array_metadata_pages(header);
    let mut blocks = 0;
    let mut secondary = 0;
    let mut largest = 0;
    for range in engine.chunked_index_spans(header).unwrap() {
        let signature = engine
            .image()
            .read_exact_at(range.start().get(), extensible_array::SIGNATURE.len())
            .unwrap();
        if signature == super::DATA_BLOCK_SIGNATURE {
            blocks += 1;
            largest = largest.max(range.length());
        } else if signature == super::SUPER_BLOCK_SIGNATURE {
            secondary += 1;
        } else {
            assert!(
                signature == extensible_array::SIGNATURE
                    || signature == extensible_array::INDEX_SIGNATURE
            );
        }
    }
    assert_eq!((blocks, secondary), growth.blocks());
    if matches!(growth, Growth::PagedParent) {
        let element = if filtered { 14 } else { 8 };
        let checksum = u64::try_from(extensible_array::CHECKSUM_BYTES).unwrap();
        assert_eq!(largest, 20 + 2 * (32 * element + checksum));
    }
    drop(engine);
}

fn compare_reads(
    native: &Path,
    pure: &Path,
    capacity: Option<NonZeroUsize>,
    range: Option<Range<usize>>,
    expected: &[i32],
    outcomes: ExpectedReads,
) {
    assert_eq!(
        observe(native, capacity, range.clone()),
        if matches!(
            outcomes,
            ExpectedReads::Checksums | ExpectedReads::NativeChecksum
        ) {
            ReadOutcome::MetadataChecksum
        } else {
            ReadOutcome::Values(expected.to_vec())
        }
    );
    assert_eq!(
        observe(pure, capacity, range),
        if matches!(outcomes, ExpectedReads::Checksums) {
            ReadOutcome::MetadataChecksum
        } else {
            ReadOutcome::Values(expected.to_vec())
        }
    );
}

fn observe(
    path: &Path,
    capacity: Option<NonZeroUsize>,
    range: Option<Range<usize>>,
) -> ReadOutcome {
    let directory = tempfile::tempdir().unwrap();
    let copy = directory.path().join("read.h5");
    let before = fs::read(path).unwrap();
    fs::write(&copy, &before).unwrap();
    assert_eq!(fs::read(&copy).unwrap(), before);
    let result = unallocated_append::read_values(&copy, capacity, range);
    assert_eq!(fs::read(&copy).unwrap(), before);
    result
}

#[derive(Clone, Copy)]
enum Growth {
    Direct,
    PagedParent,
    Secondary,
}

impl Growth {
    fn chunks(self) -> i32 {
        match self {
            Self::Direct => 24,
            Self::PagedParent => 164,
            Self::Secondary => 80,
        }
    }

    // Whole reads of pure unfiltered PAGE512 parents succeed with HDF5 1.14.6 and 2.2.0.
    fn buffered_whole_reads(self, filtered: bool, page: u64) -> ExpectedReads {
        if matches!(self, Self::PagedParent) && page == 512 {
            if filtered || unallocated_append::library_version() < (1, 14, 0) {
                ExpectedReads::Checksums
            } else {
                ExpectedReads::NativeChecksum
            }
        } else {
            ExpectedReads::Values
        }
    }

    fn blocks(self) -> (usize, usize) {
        match self {
            Self::Direct => (2, 0),
            Self::PagedParent => (5, 2),
            Self::Secondary => (3, 1),
        }
    }
}

#[derive(Clone, Copy)]
enum ExpectedReads {
    /// Metadata checksum failures for both producers
    Checksums,
    /// A metadata checksum failure for the native producer and values for the pure producer
    NativeChecksum,
    /// Values for both producers
    Values,
}
