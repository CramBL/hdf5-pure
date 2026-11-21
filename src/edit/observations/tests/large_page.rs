use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_core::__private::NonZeroByteLength;
use hdf5_pure_space::__private::{AllocationClass, PageType, ReservationRequest, SessionSpace};
use rstest::rstest;

use crate::edit::WriteEngine;
use crate::edit::block::{BlockRequest, EncodingDependency};
use crate::image::WriteBuffering;
use crate::source::{MetadataCacheConfig, Source};
use crate::{
    FileBuilder, FileLocking, FileSpacePageSize, FileSpaceSettings, FileSpaceStrategy,
    MemoryStrategy, PersistedFreeSpacePolicy,
};

#[rstest]
#[case::quarter_megabyte(262_144)]
#[case::four_megabytes(4_194_304)]
fn large_page_reservations_preserve_padding_and_buffering_limits(
    #[case] page: u64,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] backing: MemoryStrategy,
    #[values(Buffering::Unbuffered, Buffering::Operation, Buffering::Session)] buffering: Buffering,
    #[values(false, true)] ample: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cost.h5");
    let page_size = FileSpacePageSize::try_from(page).unwrap();
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_page_size(page_size),
    );
    builder.write(&path).unwrap();
    let mut engine = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        backing,
        PersistedFreeSpacePolicy::Strict,
    )
    .unwrap();
    engine.aggregation = None;
    engine.space =
        SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), ByteLength::new(1), page_size);
    let budget = if ample { 4 * page as usize } else { 131_072 };
    engine
        .set_write_buffering(buffering.mode(page, budget))
        .unwrap();
    let start = engine.image.len();
    assert_eq!(start % page, 0);
    let before = engine.image.buffering_work();
    let requests = [
        request(64, 16, PageType::Raw),
        request(48, 48, PageType::Meta),
        request(80, 32, PageType::Raw),
    ];
    let mut plan = engine
        .plan_blocks(&requests)
        .unwrap_or_else(|error| panic!("planning failed: {error:?}"));
    let end = plan.end().get();
    let placements = plan
        .reservations()
        .iter()
        .map(|entry| {
            (
                entry.range().start().get(),
                entry.range().length(),
                entry.request().page_type(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        placements,
        [
            (start, 64, PageType::Raw),
            (start + page, 48, PageType::Meta),
            (start + 2 * page, 80, PageType::Raw)
        ]
    );
    assert_eq!(plan.write_next(&[7; 16]).unwrap().get(), start);
    assert_eq!(plan.write_next(&[8; 48]).unwrap().get(), start + page);
    assert_eq!(plan.write_next(&[9; 32]).unwrap().get(), start + 2 * page);
    plan.finish().unwrap();
    assert_eq!(engine.image.len(), end);
    assert_eq!(end, start + 2 * page + 80);
    assert_eq!(engine.space.paged_tail_type(), Some(Some(PageType::Raw)));
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), []);
    let pending = engine.image.buffering_work();
    assert!(pending.pending_capacity >= pending.pending_bytes);
    assert!(pending.pending_bytes <= budget as u64);
    if ample && !matches!(buffering, Buffering::Unbuffered) {
        assert_eq!(pending.pending_runs, 1);
        assert_eq!(pending.pending_bytes, end - start);
        assert_eq!(pending.copied_bytes - before.copied_bytes, end - start);
        assert_eq!(pending.write_calls, before.write_calls);
    }
    engine.image.flush().unwrap();
    let after = engine.image.buffering_work();
    assert_eq!(after.pending_bytes, 0);
    assert_eq!(after.pending_capacity, 0);
    assert_eq!(after.write_bytes - before.write_bytes, end - start);
    assert_eq!(after.clean_gap_bytes - before.clean_gap_bytes, 0);
    assert_eq!(after.clean_gap_calls - before.clean_gap_calls, 0);
    let copied = after.copied_bytes - before.copied_bytes;
    match buffering {
        Buffering::Unbuffered => {
            assert_eq!(copied, 0);
            assert_eq!(after.flush_peak_bytes, 0);
            assert_eq!(
                after.write_calls - before.write_calls,
                5 + 2 * page.div_ceil(65_536)
            );
        }
        Buffering::Operation | Buffering::Session if ample => {
            assert_eq!(copied, end - start);
            assert_eq!(after.write_calls - before.write_calls, 1);
            assert_eq!(after.flush_peak_bytes, end - start);
            assert!(after.flush_peak_capacity >= after.flush_peak_bytes);
        }
        Buffering::Operation | Buffering::Session => {
            assert!(copied >= 192, "{copied}");
            assert!(copied <= 384, "{copied}");
            assert_eq!(
                after.write_calls - before.write_calls,
                3 + 2 * page.div_ceil(65_536)
            );
            assert!(after.flush_peak_bytes <= budget as u64);
        }
    }
    match backing {
        MemoryStrategy::Mirrored => {
            assert_eq!(after.mirror_bytes, end);
            assert!(after.mirror_capacity >= end);
        }
        MemoryStrategy::Bounded => {
            assert_eq!(after.mirror_bytes, 0);
            assert_eq!(after.mirror_capacity, 0);
        }
        MemoryStrategy::Auto => panic!("the case must select an explicit backing"),
    }
    let mut expected = vec![0; usize::try_from(end - start).unwrap()];
    expected[..16].fill(7);
    expected[page as usize..page as usize + 48].fill(8);
    expected[2 * page as usize..2 * page as usize + 32].fill(9);
    assert_eq!(
        engine.image().read_exact_at(start, expected.len()).unwrap(),
        expected
    );
    drop(engine);
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes.len() as u64, end);
    assert_eq!(&bytes[start as usize..], expected);
}

#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn large_page_commit_bounds_padding_allocations() {
    use test_util::allocation;

    use crate::{File, FileAccessProperties};

    const PAGE: usize = 4 * 1024 * 1024;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("padding.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_page_size(FileSpacePageSize::try_from(PAGE as u64).unwrap()),
    );
    builder.write(&path).unwrap();
    let original_len = std::fs::metadata(&path).unwrap().len() as usize;
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded),
    )
    .unwrap();
    let expected = [17, 23, 41, 59];
    file.root()
        .create_dataset("numbers", |builder| {
            builder.with_i32_data(&expected);
        })
        .unwrap();
    let ((), commit) = allocation::measure("large_page_commit_padding", || {
        file.commit().unwrap();
    });
    assert!(!file.has_staged_edits());
    let committed_len = usize::try_from(file.file_size()).unwrap();
    file.close().unwrap();

    let stored = std::fs::read(&path).unwrap();
    let gap = stored[original_len..committed_len]
        .split(|&byte| byte != 0)
        .max_by_key(|run| run.len())
        .unwrap();
    assert!(gap.len() > PAGE / 2);
    let (control, copy) = allocation::measure("large_page_commit_padding_control", || gap.to_vec());
    assert_eq!(control, gap);
    assert!(copy.peak_bytes >= gap.len() as u64);
    assert!(commit.peak_bytes < 128 * 1024, "{commit:?}");
    let reopened = File::open(&path).unwrap();
    assert_eq!(
        reopened.dataset("numbers").unwrap().read_i32().unwrap(),
        expected
    );
    assert_eq!(reopened.superblock().eof_address, stored.len() as u64);
    reopened.close().unwrap();
}

#[derive(Clone, Copy)]
enum Buffering {
    Operation,
    Session,
    Unbuffered,
}

impl Buffering {
    fn mode(self, page_size: u64, max_bytes: usize) -> WriteBuffering {
        match self {
            Self::Unbuffered => WriteBuffering::Unbuffered,
            Self::Operation => WriteBuffering::Operation {
                page_size,
                max_bytes,
            },
            Self::Session => WriteBuffering::Session {
                page_size,
                max_bytes,
            },
        }
    }
}

fn request(allocation: u64, encoding: u64, page_type: PageType) -> BlockRequest {
    let class = match page_type {
        PageType::Meta => AllocationClass::Metadata,
        PageType::Raw => AllocationClass::Raw,
    };
    BlockRequest::new(
        ReservationRequest::with_page_type(
            NonZeroByteLength::new(allocation).unwrap(),
            class,
            page_type,
        ),
        ByteLength::new(encoding),
        EncodingDependency::Independent,
    )
    .unwrap()
}
