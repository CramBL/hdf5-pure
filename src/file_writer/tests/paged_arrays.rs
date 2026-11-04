use std::path::Path;

use rstest::rstest;

use crate::File;
use crate::FileAccessProperties;
use crate::FileBuilder;
use crate::FileLocking;
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::FreeSpaceManagerRole;
use crate::MaxExtent;
use crate::PersistedFreeSpacePolicy;
use crate::RepackOptions;
use crate::UserblockSize;
use crate::chunked_read;
use crate::edit::WriteEngine;
use crate::repack;
use crate::source::PhysicalSource;
use crate::type_builders::DatasetBuilder;

#[rstest]
#[case::single(ArrayCase::Fixed(1))]
#[case::fixed_small(ArrayCase::Fixed(4))]
#[case::fixed_large(ArrayCase::Fixed(65))]
#[case::fixed_paged(ArrayCase::Fixed(1_025))]
#[case::fixed_empty(ArrayCase::FixedEmpty)]
#[case::extensible(ArrayCase::Extensible(128))]
#[case::extensible_paged(ArrayCase::Extensible(20_480))]
#[case::extensible_empty(ArrayCase::Extensible(0))]
#[case::unallocated(ArrayCase::Unallocated)]
#[case::unallocated_empty(ArrayCase::UnallocatedEmpty)]
fn page_arrays_preserve_complete_allocations_through_reopen_and_edits(
    #[case] array: ArrayCase,
    #[values(512, 4_096)] page: u64,
    #[values(false, true)] source_backed: bool,
    #[values(false, true)] filtered: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let initial = directory.path().join("initial.h5");
    let output = directory.path().join("output.h5");
    let settings = FileSpaceSettings::new(FileSpaceStrategy::Page)
        .with_persistence(true)
        .with_threshold(1)
        .with_page_size(FileSpacePageSize::try_from(page).unwrap());
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(settings);
    builder.with_userblock(UserblockSize::try_from(page).unwrap());
    builder.create_dataset("kept").with_i32_data(&[37; 16]);
    array.configure(builder.create_dataset("values"), filtered);
    builder.write(&initial).unwrap();
    let path = if source_backed {
        repack::repack(&initial, &output, &RepackOptions::new()).unwrap();
        &output
    } else {
        &initial
    };
    let file = File::open(path).unwrap();
    assert_eq!(file.file_space_settings().unwrap(), Some(settings));
    let dataset = file.dataset("values").unwrap();
    assert_eq!(dataset.shape().unwrap(), [array.count() as u64]);
    assert_eq!(dataset.chunk_shape().unwrap(), Some(vec![1]));
    assert_eq!(dataset.read_i32().unwrap(), array.values());
    let source = file.source();
    let stored = PhysicalSource::for_test(&source, file.addresses()).stored();
    let spans = chunked_read::collect_chunked_storage_spans(
        &stored,
        &dataset.data_layout().unwrap(),
        &dataset.dataspace().unwrap(),
        file.widths(),
    )
    .unwrap();
    if array.has_array(source_backed) {
        assert!(spans.index.len() >= 2);
        if matches!(array, ArrayCase::Fixed(_) | ArrayCase::FixedEmpty) {
            assert_eq!(spans.index.len(), 2);
        }
        let first_payload = spans.data.iter().map(|range| range.start().get()).min();
        for span in &spans.index {
            let start = span.start().get();
            let length = span.length();
            if length < page {
                assert_eq!(start / page, (span.end().get() - 1) / page);
            } else {
                assert_eq!(start % page, 0);
            }
            if let Some(payload) = first_payload {
                assert!(span.end().get() <= payload);
            }
        }
        if matches!(
            array,
            ArrayCase::Fixed(1_025) | ArrayCase::Extensible(20_480)
        ) {
            assert!(spans.index.iter().any(|range| range.length() > page));
        }
    } else {
        assert!(spans.index.is_empty());
    }
    drop(dataset);
    file.close().unwrap();
    assert_recovered_ownership(path, page);
    let file = File::open_rw_with_options(
        path,
        FileAccessProperties::new()
            .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict),
    )
    .unwrap();
    let group = file.root().create_group("added").unwrap();
    group
        .create_dataset("more", |builder| {
            builder.with_i32_data(&[11; 8]);
        })
        .unwrap();
    file.commit().unwrap();
    drop(group);
    file.close().unwrap();
    let file = File::open(path).unwrap();
    assert_eq!(
        file.dataset("values").unwrap().read_i32().unwrap(),
        array.values()
    );
    assert_eq!(file.dataset("kept").unwrap().read_i32().unwrap(), [37; 16]);
    assert_eq!(
        file.dataset("added/more").unwrap().read_i32().unwrap(),
        [11; 8]
    );
    file.close().unwrap();
    assert_recovered_ownership(path, page);
}

fn assert_recovered_ownership(path: &Path, page: u64) {
    let file = File::open(path).unwrap();
    let base = file.addresses().effective_base().get();
    let report = file.persisted_free_sections().unwrap().unwrap();
    let mut unknown = Vec::new();
    for manager in report
        .managers()
        .iter()
        .filter(|manager| manager.role() == FreeSpaceManagerRole::PageLargeGeneric)
    {
        for section in manager.free_sections().sections() {
            let start = section.address().get();
            let end = section.end_address().get();
            let aligned_start = start.next_multiple_of(page);
            let aligned_end = end / page * page;
            if aligned_start >= aligned_end {
                unknown.push((base + start, end - start));
            } else {
                if start < aligned_start {
                    unknown.push((base + start, aligned_start - start));
                }
                if aligned_end < end {
                    unknown.push((base + aligned_end, end - aligned_end));
                }
            }
        }
    }
    unknown.sort_unstable();
    let mut expected: Vec<(u64, u64)> = Vec::new();
    for (start, length) in unknown {
        if let Some((_, size)) = expected
            .last_mut()
            .filter(|(previous, size)| *previous + *size == start)
        {
            *size += length;
        } else {
            expected.push((start, length));
        }
    }
    let free_bytes = report.free_sections().free_bytes();
    file.close().unwrap();
    let engine = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
    let ownership = engine.space_ownership().unwrap();
    ownership.assert_classified(&expected);
    assert_eq!(
        ownership
            .reusable_free_space
            .iter()
            .map(|(_, size)| size)
            .sum::<u64>()
            + expected.iter().map(|(_, size)| size).sum::<u64>(),
        free_bytes
    );
}

#[derive(Clone, Copy)]
enum ArrayCase {
    Extensible(usize),
    Fixed(usize),
    FixedEmpty,
    Unallocated,
    UnallocatedEmpty,
}

impl ArrayCase {
    fn configure(self, builder: &mut DatasetBuilder, filtered: bool) {
        match self {
            Self::Unallocated | Self::UnallocatedEmpty => {
                builder
                    .with_unallocated_storage(crate::make_i32_type(), &[self.count() as u64])
                    .with_maxshape(&[MaxExtent::Unlimited]);
            }
            _ => {
                builder
                    .with_i32_data(&self.values())
                    .with_shape(&[self.count() as u64]);
            }
        }
        builder.with_chunks(&[1]);
        match self {
            Self::Extensible(_) => {
                builder.with_maxshape(&[MaxExtent::Unlimited]);
            }
            Self::FixedEmpty => {
                builder.with_maxshape(&[MaxExtent::Fixed(1_025)]);
            }
            Self::Fixed(_) | Self::Unallocated | Self::UnallocatedEmpty => {}
        }
        if filtered {
            builder.with_fletcher32();
        }
    }

    fn values(self) -> Vec<i32> {
        if matches!(self, Self::Unallocated | Self::UnallocatedEmpty) {
            vec![0; self.count()]
        } else {
            (0..self.count())
                .map(|index| i32::try_from(index).unwrap())
                .collect()
        }
    }

    fn count(self) -> usize {
        match self {
            Self::Extensible(count) | Self::Fixed(count) => count,
            Self::FixedEmpty | Self::UnallocatedEmpty => 0,
            Self::Unallocated => 65,
        }
    }

    fn has_array(self, source_backed: bool) -> bool {
        match self {
            Self::Fixed(0 | 1) | Self::FixedEmpty | Self::UnallocatedEmpty => false,
            Self::Extensible(0) => !source_backed,
            _ => true,
        }
    }
}
