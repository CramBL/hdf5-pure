use core::num::NonZeroUsize;
use std::fs;
use std::path::Path;

use rstest::rstest;
use test_util_hdf5::__private::array_index;
use test_util_hdf5::__private::ownership_append;
use test_util_hdf5::__private::unallocated_append;
use test_util_hdf5::dataset;
use test_util_hdf5::file;
use test_util_hdf5::file_space_oracle;
use test_util_hdf5::numeric::ByteOrder;
use test_util_hdf5::page_object_headers;
use test_util_hdf5::page_object_headers::SourceLayout;

use crate::File;
use crate::FileAccessProperties;
use crate::FileSpacePageSize;
use crate::MemoryStrategy;
use crate::SyncPolicy;
use crate::UserblockSize;
use crate::edit::WriteEngine;
use crate::edit::dataset::tests::Creation;
use crate::edit::dataset::tests::CreationAccess;
use crate::edit::test_util;

#[rstest]
#[case::plain(false, false)]
#[case::filtered(true, false)]
#[case::incompressible(true, true)]
fn staged_appends_keep_native_array_blocks_in_metadata_pages(
    #[case] filtered: bool,
    #[case] incompressible: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("staged-array-blocks.h5");
    let fixture = ownership_append::Fixture::write(&path, true, filtered, incompressible);
    for cycle in 0..5 {
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new()
                .with_memory_strategy(backing)
                .with_sync_policy(SyncPolicy::OnClose),
        )
        .unwrap();
        file.dataset("d")
            .unwrap()
            .append_staged(|append| {
                append.append_u8(&ownership_append::payload(0x3cb6_7710 + cycle));
            })
            .unwrap();
        file.commit().unwrap();
        let dataset = file.dataset("d").unwrap().header_address().unwrap();
        file.close().unwrap();
        let engine = test_util::open_engine(&path);
        engine.assert_array_metadata_pages(dataset);
        drop(engine);
        fixture.assert_contents(&path, cycle);
    }
}

#[rstest]
fn staged_first_writes_preserve_native_deferred_array_values(
    #[values(false, true)] filtered: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values([32, 4, 4, 16, 10], [24, 8, 2, 8, 8])] parameters: [u8; 5],
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("staged-deferred-array.h5");
    unallocated_append::fixture(&path, filtered, true, parameters);
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(backing)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    let expected: Vec<i32> = (0..40).collect();
    file.dataset("d")
        .unwrap()
        .append_staged(|append| {
            append.append_i32(&expected);
        })
        .unwrap();
    file.commit().unwrap();
    let header = file.dataset("d").unwrap().header_address().unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), expected);
    file.close().unwrap();
    let engine = test_util::open_engine(&path);
    engine.assert_array_metadata_pages(header);
    drop(engine);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}

#[rstest]
#[case::fixed(Creation::Fixed(40))]
#[case::fixed_sparse(Creation::FixedSparse)]
#[case::fixed_paged(Creation::Fixed(1_025))]
#[case::empty_extensible(Creation::Extensible(0))]
#[case::extensible(Creation::Extensible(40))]
#[case::rotated(Creation::Rotated(8))]
#[case::paged_parent(Creation::Rotated(20_000))]
#[case::pending_append(Creation::Pending)]
#[case::unallocated(Creation::Unallocated)]
fn staged_creations_keep_native_array_blocks_in_metadata_pages(
    #[case] creation: Creation,
    #[values(false, true)] filtered: bool,
    #[values(
        CreationAccess::Mirrored,
        CreationAccess::Bounded,
        CreationAccess::Userblock
    )]
    access: CreationAccess,
) {
    let base = access.base();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("created-array.h5");
    page_object_headers::write_native(&path, 512, base, SourceLayout::Contiguous, true, 48);
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(access.strategy())
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    let mut dataset = file
        .root()
        .create_dataset("d", |builder| creation.configure(builder, filtered))
        .unwrap();
    if matches!(creation, Creation::Pending) {
        dataset
            .append_staged(|builder| {
                builder.append_i32(&[5, 6, 7, 8]);
            })
            .unwrap();
    }
    file.commit().unwrap();
    let header = dataset.header_address().unwrap();
    assert_eq!(dataset.read_i32().unwrap(), creation.expected());
    drop(dataset);
    file.close().unwrap();
    let engine = test_util::open_engine(&path);
    engine.assert_array_metadata_pages(header);
    if matches!(creation, Creation::Fixed(1_025) | Creation::Rotated(20_000)) {
        assert!(
            engine
                .chunked_index_spans(header)
                .unwrap()
                .iter()
                .any(|range| range.length() > 512)
        );
    }
    drop(engine);
    let native = file::libhdf5_open_with_page_buffer(&path, NonZeroUsize::new(2_048).unwrap());
    assert_eq!(
        native.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        creation.expected()
    );
    assert_eq!(
        native
            .dataset("parent/nested/data")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [1, 2, 3, 4]
    );
    native.close().unwrap();
    let pure = File::open(&path).unwrap();
    let records = pure.dataset("d").unwrap().chunks().unwrap();
    let mut coordinates: Vec<_> = records.iter().map(|chunk| chunk.offset.clone()).collect();
    coordinates.sort();
    assert_eq!(coordinates, creation.coordinates());
    let mut chunks: Vec<_> = records
        .iter()
        .map(|chunk| {
            (
                chunk.address,
                chunk.storage_size,
                chunk.filter_mask().raw_bits(),
            )
        })
        .collect();
    chunks.sort_unstable();
    pure.close().unwrap();
    let mut native_chunks: Vec<_> = array_index::libhdf5_chunks(&path, base)
        .into_iter()
        .map(|chunk| (chunk.address(), chunk.size(), chunk.filter_mask()))
        .collect();
    native_chunks.sort_unstable();
    assert_eq!(native_chunks, chunks);
    file_space_oracle::assert_persisted_sections(&path);
    file_space_oracle::assert_manager_metadata_bytes(&path);
}

#[rstest]
#[case::rotated(8)]
#[case::paged(20_000)]
fn native_written_rotated_arrays_preserve_values_and_physical_records(#[case] maximum: usize) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native-rotated.h5");
    let native = file::libhdf5_create_v110(&path);
    let dataset = native
        .new_dataset::<i32>()
        .shape((2..=maximum, 2..))
        .chunk([1, 1])
        .create("d")
        .unwrap();
    dataset.write_raw(&[11, 12, 13, 14]).unwrap();
    assert_eq!(dataset.read_raw::<i32>().unwrap(), [11, 12, 13, 14]);
    drop(dataset);
    native.close().unwrap();
    let pure = File::open(&path).unwrap();
    assert_eq!(
        pure.dataset("d").unwrap().read_i32().unwrap(),
        [11, 12, 13, 14]
    );
    let chunks = pure.dataset("d").unwrap().chunks().unwrap();
    let mut logical: Vec<_> = chunks.iter().map(|chunk| chunk.offset.clone()).collect();
    logical.sort();
    assert_eq!(logical, [vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]);
    let mut expected: Vec<_> = chunks
        .iter()
        .map(|chunk| {
            (
                chunk.address,
                chunk.storage_size,
                chunk.filter_mask().raw_bits(),
            )
        })
        .collect();
    expected.sort_unstable();
    pure.close().unwrap();
    let chunks = array_index::libhdf5_chunks(&path, 0);
    let coordinates: Vec<_> = chunks.iter().map(|chunk| chunk.offset().to_vec()).collect();
    eprintln!("NATIVE ROTATED maximum={maximum}: logical={logical:?} native_query={coordinates:?}");
    let mut actual: Vec<_> = chunks
        .iter()
        .map(|chunk| (chunk.address(), chunk.size(), chunk.filter_mask()))
        .collect();
    actual.sort_unstable();
    assert_eq!(actual, expected);
}

#[rstest]
#[case::fixed(ProviderGeometry::Fixed)]
#[case::fixed_sparse(ProviderGeometry::FixedSparse)]
#[case::fixed_paged(ProviderGeometry::FixedPaged)]
#[case::extensible(ProviderGeometry::Extensible)]
#[case::rotated(ProviderGeometry::Rotated)]
fn copied_and_replaced_native_arrays_use_complete_metadata_blocks(
    #[case] geometry: ProviderGeometry,
    #[values(
        ProviderRun::Copy(CreationAccess::Mirrored),
        ProviderRun::Copy(CreationAccess::Bounded),
        ProviderRun::Copy(CreationAccess::Userblock),
        ProviderRun::CopyFrom(CreationAccess::Mirrored),
        ProviderRun::CopyFrom(CreationAccess::Bounded),
        ProviderRun::CopyFrom(CreationAccess::Userblock),
        ProviderRun::OverwriteMirrored,
        ProviderRun::OverwriteUserblock
    )]
    run: ProviderRun,
    #[values(false, true)] filtered: bool,
) {
    let access = run.access();
    let overwrite = matches!(
        run,
        ProviderRun::OverwriteMirrored | ProviderRun::OverwriteUserblock
    );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("provider-array.h5");
    let source_path = directory.path().join("source.h5");
    let name = if overwrite { "d" } else { "original" };
    write_provider_array(&path, geometry, filtered, access.base(), !overwrite, name);
    if matches!(run, ProviderRun::CopyFrom(_)) {
        write_provider_array(&source_path, geometry, filtered, 0, true, "original");
    }
    let input_path = if matches!(run, ProviderRun::CopyFrom(_)) {
        &source_path
    } else {
        &path
    };
    let input = encoded_provider_chunks(input_path, name);
    let source_bytes = fs::read(input_path).unwrap();
    let reader = File::open(input_path).unwrap();
    let input_records = reader.dataset(name).unwrap().chunks().unwrap();
    assert_eq!(
        reader.dataset(name).unwrap().read_i32().unwrap(),
        geometry.initial_values()
    );
    reader.close().unwrap();
    let old_records = if matches!(run, ProviderRun::CopyFrom(_)) {
        let destination = File::open(&path).unwrap();
        let records = destination.dataset(name).unwrap().chunks().unwrap();
        destination.close().unwrap();
        records
    } else {
        input_records
    };

    let expected = if overwrite {
        (0..geometry.elements())
            .map(|index| i32::try_from(index).unwrap() * 7_919 + 17)
            .collect()
    } else {
        geometry.initial_values()
    };
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(access.strategy())
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    match run {
        ProviderRun::Copy(_) => file.copy("original", "d").unwrap(),
        ProviderRun::CopyFrom(_) => {
            let source = File::open(&source_path).unwrap();
            file.copy_from(&source, "original", "d").unwrap();
            source.close().unwrap();
        }
        ProviderRun::OverwriteMirrored | ProviderRun::OverwriteUserblock => file
            .dataset("d")
            .unwrap()
            .write_staged(|builder| {
                builder
                    .with_i32_data(&expected)
                    .with_shape(&geometry.shape());
            })
            .unwrap(),
    }
    file.commit().unwrap();
    let dataset = file.dataset("d").unwrap();
    let header = dataset.header_address().unwrap();
    assert_eq!(dataset.read_i32().unwrap(), expected);
    let records = dataset.chunks().unwrap();
    for record in &records {
        assert!(
            old_records
                .iter()
                .all(|old| record.address + record.storage_size <= old.address
                    || old.address + old.storage_size <= record.address)
        );
    }
    drop(dataset);
    file.close().unwrap();
    let engine = test_util::open_engine(&path);
    engine.assert_array_metadata_pages(header);
    if matches!(
        geometry,
        ProviderGeometry::FixedPaged | ProviderGeometry::Rotated
    ) {
        assert!(
            engine
                .chunked_index_spans(header)
                .unwrap()
                .iter()
                .any(|range| range.length() > 512)
        );
    }
    drop(engine);
    if !overwrite {
        assert_eq!(encoded_provider_chunks(&path, "d"), input);
    }
    if matches!(run, ProviderRun::CopyFrom(_)) {
        assert_eq!(fs::read(&source_path).unwrap(), source_bytes);
    }
    let native = file::libhdf5_open_with_page_buffer(&path, NonZeroUsize::new(2_048).unwrap());
    assert_eq!(
        native.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        expected
    );
    assert_eq!(
        native.dataset("kept").unwrap().read_raw::<i32>().unwrap(),
        [31; 16]
    );
    if !overwrite {
        assert_eq!(
            native
                .dataset("original")
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            geometry.initial_values()
        );
    }
    native.close().unwrap();
    let mut expected_records: Vec<_> = records
        .iter()
        .map(|record| {
            (
                record.address,
                record.storage_size,
                record.filter_mask().raw_bits(),
            )
        })
        .collect();
    expected_records.sort_unstable();
    let mut actual: Vec<_> = array_index::libhdf5_chunks(&path, access.base())
        .iter()
        .map(|record| (record.address(), record.size(), record.filter_mask()))
        .collect();
    actual.sort_unstable();
    assert_eq!(actual, expected_records);
    file_space_oracle::assert_persisted_sections(&path);
    file_space_oracle::assert_manager_metadata_bytes(&path);
}

fn write_provider_array(
    path: &Path,
    geometry: ProviderGeometry,
    filtered: bool,
    base: u64,
    persist: bool,
    name: &str,
) {
    let native = file::libhdf5_create_page(
        path,
        FileSpacePageSize::MIN,
        UserblockSize::try_from(base).unwrap(),
        persist,
    );
    let builder = native
        .new_dataset_builder()
        .empty_as(ByteOrder::LittleEndian.i32_datatype());
    let builder = match geometry {
        ProviderGeometry::Fixed => builder.shape([160]),
        ProviderGeometry::FixedPaged => builder.shape([4_100]),
        ProviderGeometry::FixedSparse => builder.shape((2..=2, 2..=8)),
        ProviderGeometry::Extensible => builder.shape((160..,)),
        ProviderGeometry::Rotated => builder.shape((2..=20_000, 2..)),
    };
    let builder = if matches!(
        geometry,
        ProviderGeometry::FixedSparse | ProviderGeometry::Rotated
    ) {
        builder.chunk([1, 1])
    } else {
        builder.chunk([4])
    };
    let builder = if filtered {
        builder.deflate(6)
    } else {
        builder
    };
    let dataset = builder.create(name).unwrap();
    drop(builder);
    dataset.write_raw(&geometry.initial_values()).unwrap();
    drop(dataset);
    native
        .new_dataset_builder()
        .with_data(&[31; 16])
        .create("kept")
        .unwrap();
    native.close().unwrap();
}

fn encoded_provider_chunks(path: &Path, name: &str) -> Vec<(Vec<u64>, u32, Vec<u8>)> {
    let file = File::open(path).unwrap();
    let records = file.dataset(name).unwrap().chunks().unwrap();
    file.close().unwrap();
    let bytes = fs::read(path).unwrap();
    let mut chunks: Vec<_> = records
        .into_iter()
        .map(|record| {
            let start = usize::try_from(record.address).unwrap();
            let end = usize::try_from(record.address + record.storage_size).unwrap();
            let mask = record.filter_mask().raw_bits();
            (record.offset, mask, bytes[start..end].to_vec())
        })
        .collect();
    chunks.sort();
    chunks
}

#[derive(Clone, Copy)]
enum ProviderGeometry {
    Extensible,
    Fixed,
    FixedPaged,
    FixedSparse,
    Rotated,
}

impl ProviderGeometry {
    fn elements(self) -> usize {
        match self {
            Self::Fixed | Self::Extensible => 160,
            Self::FixedPaged => 4_100,
            Self::FixedSparse | Self::Rotated => 4,
        }
    }
    fn initial_values(self) -> Vec<i32> {
        (0..self.elements())
            .map(|index| i32::try_from(index % 7 + 1).unwrap())
            .collect()
    }

    fn shape(self) -> Vec<u64> {
        if matches!(self, Self::FixedSparse | Self::Rotated) {
            vec![2, 2]
        } else {
            vec![u64::try_from(self.elements()).unwrap()]
        }
    }
}

#[derive(Clone, Copy)]
enum ProviderRun {
    Copy(CreationAccess),
    CopyFrom(CreationAccess),
    OverwriteMirrored,
    OverwriteUserblock,
}

impl ProviderRun {
    fn access(self) -> CreationAccess {
        match self {
            Self::Copy(access) | Self::CopyFrom(access) => access,
            Self::OverwriteMirrored => CreationAccess::Mirrored,
            Self::OverwriteUserblock => CreationAccess::Userblock,
        }
    }
}

impl WriteEngine {
    fn assert_array_metadata_pages(&self, dataset: u64) {
        let index = self.chunked_index_spans(dataset).unwrap();
        let spans = self.chunked_storage_spans(dataset).unwrap();
        let page = self.space.page_size().unwrap().get();
        let addresses = self.context.addresses();
        assert!(index.len() >= 2);
        for allocation in &index {
            let start = addresses.relative(allocation.start()).unwrap().get();
            let last = start + allocation.length() - 1;
            if allocation.length() < page {
                assert_eq!(start / page, last / page);
            } else {
                assert_eq!(start % page, 0);
            }
            for &(raw, _) in spans.iter().filter(|(range, _)| !index.contains(range)) {
                let raw_start = addresses.relative(raw.start()).unwrap().get();
                let raw_last = raw_start + raw.length() - 1;
                assert!(last / page < raw_start / page || raw_last / page < start / page);
            }
        }
        let ownership = self.space_ownership().unwrap();
        assert!(ownership.analysis_complete);
        for allocation in index {
            assert!(
                ownership
                    .live_metadata_space
                    .iter()
                    .any(|&(start, length)| {
                        start <= allocation.start().get()
                            && allocation.end().get() <= start + length
                    })
            );
        }
    }
}
