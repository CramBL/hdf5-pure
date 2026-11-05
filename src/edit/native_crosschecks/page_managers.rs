use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::FreeList;
use rstest::rstest;
use test_util_hdf5::file_space_oracle;
use test_util_hdf5::page_managers;

use crate::AttrValue;
use crate::File;
use crate::FileLocking;
use crate::MemoryStrategy;
use crate::PersistedFreeSpacePolicy;
use crate::edit::WriteEngine;
use crate::edit::object_header;
use crate::free_space_manager;
use crate::object_path::ObjectPathBuf;
use crate::source::MetadataCacheConfig;
use crate::source::Source;

#[rstest]
#[case::small(page_managers::Geometry::Small, MemoryStrategy::Bounded)]
#[case::non_power_of_two(page_managers::Geometry::NonPowerOfTwo, MemoryStrategy::Bounded)]
#[case::userblock(page_managers::Geometry::Userblock, MemoryStrategy::Mirrored)]
#[case::userblock_aligned_page(
    page_managers::Geometry::UserblockAlignedPage,
    MemoryStrategy::Mirrored
)]
#[case::large(page_managers::Geometry::Large, MemoryStrategy::Mirrored)]
#[case::mirrored(page_managers::Geometry::Small, MemoryStrategy::Mirrored)]
fn native_page_buffer_reads_repeated_metadata_publications(
    #[case] geometry: page_managers::Geometry,
    #[case] memory: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("managers.h5");
    let page = geometry.page_size();
    page_managers::write_native(&path, geometry);
    let open = || {
        WriteEngine::open_rw_with_strategy(
            &path,
            MetadataCacheConfig::disabled(),
            FileLocking::Enabled,
            memory,
            PersistedFreeSpacePolicy::Strict,
        )
        .unwrap()
    };
    let mut engine = open();
    engine.delete(&ObjectPathBuf::parse("d2")).unwrap();
    engine.commit().unwrap();
    let first = inspect(&engine, page);
    engine
        .set_group_attr(&ObjectPathBuf::root(), "generation", AttrValue::U32(2))
        .unwrap();
    engine.commit().unwrap();
    let second = inspect(&engine, page);
    assert!(first.0 > 0);
    assert!(second.0 > 0);
    engine.force_sync().unwrap();
    engine.release_status_flags().unwrap();
    drop(engine);
    let mut engine = open();
    assert_eq!(inspect(&engine, page), second);
    engine
        .set_group_attr(&ObjectPathBuf::root(), "generation", AttrValue::U32(3))
        .unwrap();
    engine.commit().unwrap();
    inspect(&engine, page);
    engine.force_sync().unwrap();
    engine.release_status_flags().unwrap();
    drop(engine);
    page_managers::assert_native(&path, geometry);
    file_space_oracle::assert_persisted_sections(&path);
}

pub(super) fn inspect(engine: &WriteEngine, page: u64) -> (u64, Vec<(u64, u64)>) {
    let bytes = engine
        .image()
        .read_exact_at(0, usize::try_from(engine.image.len()).unwrap())
        .unwrap();
    let file = File::from_bytes(bytes).unwrap();
    let info = file.file_space_info().unwrap().unwrap();
    let stored = engine
        .context
        .superblock()
        .superblock_extension_address
        .unwrap();
    let chunks = object_header::read_oh_chunks(
        engine.image(),
        engine.context.addresses().resolve(stored).unwrap(),
        engine.context.addresses(),
        engine.context.widths(),
    )
    .unwrap();
    let (sections, mut allocations) = free_space_manager::read_persisted_sections_source(
        &engine.image(),
        engine.context.widths(),
        engine.context.addresses().effective_base(),
        &info.manager_addrs,
    )
    .unwrap();
    allocations.extend(chunks.into_iter().map(|chunk| chunk.span));
    let mut owned = allocations
        .into_iter()
        .map(|range| {
            Extent::new(
                engine.context.addresses().relative(range.start()).unwrap(),
                range.length(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    owned.sort_unstable();
    for allocation in &owned {
        if allocation.len() < page {
            assert_eq!(
                allocation.start().get() / page,
                (allocation.end().get() - 1) / page,
                "{allocation:?}"
            );
        } else {
            assert_eq!(allocation.start().get() % page, 0, "{allocation:?}");
        }
        assert!(sections.iter().all(|free| free.addr.get() + free.size
            <= allocation.start().get()
            || free.addr >= allocation.end()));
    }
    let actual = FreeList::from_sorted(owned).unwrap();
    let mut recorded = engine.persist.as_ref().unwrap().old_blocks.clone();
    recorded.sort_unstable();
    assert_eq!(
        FreeList::from_sorted(recorded).unwrap().extents(),
        actual.extents()
    );
    let published = FreeList::from_sorted({
        let mut sections = sections
            .into_iter()
            .map(|section| Extent::new(section.addr, section.size).unwrap())
            .collect::<Vec<_>>();
        sections.sort_unstable();
        sections
    })
    .unwrap();
    let mut adopted = engine
        .space
        .admitted_snapshot()
        .sections()
        .map(|(range, _)| range)
        .collect::<Vec<_>>();
    adopted.sort_unstable();
    assert_eq!(
        published.extents(),
        FreeList::from_sorted(adopted).unwrap().extents()
    );
    (
        published.total_bytes(),
        actual
            .extents()
            .iter()
            .map(|allocation| (allocation.start().get(), allocation.len()))
            .collect(),
    )
}
