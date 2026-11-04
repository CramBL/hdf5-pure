use std::collections::BTreeSet;

use ::test_util::range;
use rstest::rstest;
use test_util_hdf5::__private::array_index;
use test_util_hdf5::__private::creation_holes;
use test_util_hdf5::__private::creation_holes::HoleLayout;
use test_util_hdf5::file;
use test_util_hdf5::file_space::persisted::PersistedStrategy;
use test_util_hdf5::file_space_oracle;

use crate::DatasetCreateProperties;
use crate::DeflateLevel;
use crate::File;
use crate::FileAccessProperties;
use crate::FileSpacePageSize;
use crate::FilterDisposition;
use crate::MemoryStrategy;
use crate::PartialEdgeFiltering;
use crate::SyncPolicy;
use crate::UserblockSize;
use crate::dataspace::MaxExtent;
use crate::edit::dataset::tests::lifecycle;
use crate::edit::test_util;

#[rstest]
#[case::flat_fixed(false, false)]
#[case::flat_extensible(false, true)]
#[case::page_fixed(true, false)]
#[case::page_extensible(true, true)]
fn creation_reuses_native_holes_and_preserves_survivors_and_chunk_records(
    #[case] paged: bool,
    #[case] extensible: bool,
    #[values(false, true)] filtered: bool,
    #[values(HoleLayout::Contiguous, HoleLayout::Fragmented, HoleLayout::WholePages)]
    layout: HoleLayout,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native-holes.h5");
    let holes = creation_holes::create(&path, paged, layout);
    let contiguous = matches!(layout, HoleLayout::Contiguous);
    let count = if matches!(layout, HoleLayout::WholePages) {
        4_096
    } else {
        2_048
    };
    let expected: Vec<i32> = (0..count)
        .scan(0x34a9_12b7_u32, |state, _| {
            *state ^= *state << 13;
            *state ^= *state >> 17;
            *state ^= *state << 5;
            Some(i32::from_le_bytes(state.to_le_bytes()))
        })
        .collect();
    let pure = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(backing)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    let available = pure
        .reusable_free_sections()
        .unwrap()
        .sections()
        .iter()
        .map(|section| section.address().get()..section.end_address().get())
        .collect::<Vec<_>>();
    if !contiguous {
        assert!(
            pure.reusable_free_sections()
                .unwrap()
                .sections()
                .iter()
                .all(|section| section.length().get() < u64::try_from(expected.len() * 4).unwrap())
        );
    }
    let dataset = pure
        .root()
        .create_dataset("d", |builder| {
            builder.with_i32_data(&expected).with_chunks(&[128]);
            if extensible {
                builder.with_maxshape(&[MaxExtent::Unlimited]);
            }
            if filtered {
                builder.with_deflate(DeflateLevel::Six);
            }
        })
        .unwrap();
    pure.commit().unwrap();
    let header = dataset.header_address().unwrap();
    let chunks = dataset.chunks().unwrap();
    assert_eq!(chunks.len(), expected.len() / 128);
    let published = holes
        .iter()
        .map(|section| (section.start, section.end - section.start))
        .collect::<Vec<_>>();
    for chunk in &chunks {
        let extent = chunk.address..chunk.address + chunk.storage_size;
        assert!(
            available
                .iter()
                .any(|section| section.start <= extent.start && extent.end <= section.end),
            "{chunk:?}, {available:?}"
        );
        assert_eq!(
            range::covered_len(&published, &extent),
            chunk.storage_size,
            "{chunk:?}, {holes:?}"
        );
    }
    if contiguous {
        assert!(
            chunks
                .windows(2)
                .all(|pair| pair[0].address + pair[0].storage_size == pair[1].address)
        );
    }
    assert_eq!(dataset.read_i32().unwrap(), expected);
    drop(dataset);
    pure.close().unwrap();
    let engine = test_util::open_engine(&path);
    if paged {
        engine.assert_array_metadata_pages(header);
    }
    assert!(engine.space_ownership().unwrap().analysis_complete);
    drop(engine);
    let native = file::libhdf5_open(&path);
    assert_eq!(
        native.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        expected
    );
    for index in 0..if contiguous { 1 } else { 16 } {
        assert_eq!(
            native
                .dataset(&format!("spacer{index}"))
                .unwrap()
                .read_raw::<u8>()
                .unwrap(),
            vec![31; layout.spacer_size()]
        );
    }
    native.close().unwrap();
    let native_records = array_index::libhdf5_chunks(&path, 0);
    let pure = File::open(&path).unwrap();
    assert_eq!(pure.dataset("d").unwrap().read_i32().unwrap(), expected);
    let pure_records = pure
        .dataset("d")
        .unwrap()
        .chunks()
        .unwrap()
        .into_iter()
        .map(|chunk| {
            let mask = chunk.filter_mask().raw_bits();
            (chunk.offset, chunk.address, chunk.storage_size, mask)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        pure_records,
        native_records
            .iter()
            .map(|chunk| (
                chunk.offset().to_vec(),
                chunk.address(),
                chunk.size(),
                chunk.filter_mask()
            ))
            .collect::<Vec<_>>()
    );
    pure.close().unwrap();
}

#[rstest]
fn native_aliases_preserve_fragmented_allocations_until_the_last_removal(
    #[values(PersistedStrategy::FsmAggr, PersistedStrategy::Page)] strategy: PersistedStrategy,
    #[values(false, true)] extensible: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("aliased-fragments.h5");
    let expected = lifecycle::created_fixture(&path, strategy, extensible);
    creation_holes::add_hard_link(&path, "replacement", "alias");
    let live = lifecycle::owned_ranges(&path, "alias");
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    let mut names = file.root().datasets().unwrap();
    names.retain(|name| name != "replacement");
    file.root().delete("replacement").unwrap();
    file.commit().unwrap();
    assert_eq!(file.root().datasets().unwrap(), names);
    assert_eq!(file.dataset("alias").unwrap().read_i32().unwrap(), expected);
    let free = file
        .reusable_free_sections()
        .unwrap()
        .sections()
        .iter()
        .map(|section| (section.address().get(), section.length().get()))
        .collect::<Vec<_>>();
    for allocation in live {
        assert_eq!(
            range::covered_len(&free, &(allocation.start().get()..allocation.end().get())),
            0
        );
    }
    file.close().unwrap();
    let native = file::libhdf5_open(&path);
    assert_eq!(
        native.dataset("alias").unwrap().read_raw::<i32>().unwrap(),
        expected
    );
    native.close().unwrap();
    file_space_oracle::assert_persisted_sections(&path);
    let successor = lifecycle::retire_and_reuse(&path, "alias", &expected);
    let native = file::libhdf5_open(&path);
    assert_eq!(
        native
            .dataset("successor")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        successor
    );
    assert_eq!(
        native.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        (0..64).collect::<Vec<i32>>()
    );
    native.close().unwrap();
    file_space_oracle::assert_persisted_sections(&path);
    file_space_oracle::assert_manager_metadata_bytes(&path);
}

#[rstest]
fn creation_preserves_sparse_slots_variable_filters_and_reuse_policy(
    #[values(Input::Flat, Input::Page, Input::TransientPage)] input: Input,
    #[values(false, true)] rotated: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("variable-fragments.h5");
    let transient = matches!(input, Input::TransientPage);
    if transient {
        let native = file::libhdf5_create_page(
            &path,
            FileSpacePageSize::DEFAULT,
            UserblockSize::NONE,
            false,
        );
        native.close().unwrap();
    } else {
        creation_holes::create(&path, matches!(input, Input::Page), HoleLayout::Fragmented);
    }
    let pure = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    if transient {
        for index in 0..16 {
            for role in ["scratch", "spacer"] {
                pure.root()
                    .create_dataset(&format!("{role}{index}"), |builder| {
                        builder.with_u8_data(&[31; 1_024]);
                    })
                    .unwrap();
            }
        }
        pure.commit().unwrap();
        for index in 0..16 {
            pure.root().delete(&format!("scratch{index}")).unwrap();
        }
        pure.commit().unwrap();
    }
    let available = pure
        .reusable_free_sections()
        .unwrap()
        .sections()
        .iter()
        .map(|section| section.address().get()..section.end_address().get())
        .collect::<Vec<_>>();
    let shape = if rotated { vec![2, 2_047] } else { vec![4_095] };
    let mut state = 0x34a9_12b7_u32;
    let expected = (0..shape.iter().product::<u64>())
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            if (index % 2_047) / 128 % 2 == 0 {
                0
            } else {
                i32::from_le_bytes(state.to_le_bytes())
            }
        })
        .collect::<Vec<_>>();
    let initial_size = pure.file_size();
    crate::edit::observations::reset_creation();
    let created = pure
        .root()
        .create_dataset("d", |builder| {
            builder
                .with_create_properties(
                    DatasetCreateProperties::new()
                        .with_partial_edge_filtering(PartialEdgeFiltering::Skip),
                )
                .with_i32_data(&expected)
                .with_shape(&shape);
            if rotated {
                builder
                    .with_chunks(&[1, 128])
                    .with_maxshape(&[MaxExtent::Fixed(4), MaxExtent::Unlimited]);
            } else {
                builder
                    .with_chunks(&[128])
                    .with_maxshape(&[MaxExtent::Fixed(8_192)]);
            }
            builder.with_shuffle().with_deflate(DeflateLevel::Six);
        })
        .unwrap();
    pure.commit().unwrap();
    assert_eq!(created.read_i32().unwrap(), expected);
    let header = created.header_address().unwrap();
    let chunks = created.chunks().unwrap();
    assert_eq!(chunks.len(), 32);
    let payload = chunks.iter().map(|chunk| chunk.storage_size).sum::<u64>();
    let work = crate::edit::observations::creation_work();
    assert_eq!(work.chunks, 32);
    assert_eq!(work.filtered_chunks, if rotated { 30 } else { 31 });
    assert_eq!(work.prepared_bytes, payload);
    assert_eq!(work.index_encodings, 1);
    assert_eq!(work.preferred_probes, u64::from(!transient));
    if transient {
        assert!(work.batch_builds <= 1);
    } else {
        assert_eq!(
            work.batch_builds,
            if matches!(input, Input::Page) { 2 } else { 1 }
        );
    }
    assert!(
        available
            .iter()
            .all(|range| range.end - range.start < payload)
    );
    if transient {
        assert!(chunks.iter().all(|chunk| chunk.address >= initial_size));
        assert!(pure.file_size() >= initial_size + payload);
    } else {
        assert!(chunks.iter().all(|chunk| available.iter().any(|range| {
            range.start <= chunk.address && chunk.address + chunk.storage_size <= range.end
        })));
    }
    let sizes = chunks
        .iter()
        .map(|chunk| chunk.storage_size)
        .collect::<BTreeSet<_>>();
    assert!(sizes.len() >= 3, "{sizes:?}");
    let offsets = if rotated {
        (0..16)
            .flat_map(|column| (0..2).map(move |row| vec![row, column * 128]))
            .collect::<Vec<_>>()
    } else {
        (0..32).map(|index| vec![index * 128]).collect::<Vec<_>>()
    };
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.offset.clone())
            .collect::<Vec<_>>(),
        offsets
    );
    for chunk in &chunks {
        let partial = chunk.offset.last().copied().unwrap() == if rotated { 1_920 } else { 3_968 };
        assert_eq!(chunk.filter_mask().raw_bits(), 0);
        assert_eq!(
            chunk
                .filters()
                .dispositions()
                .map(|(_, disposition)| disposition)
                .collect::<Vec<_>>(),
            vec![
                if partial {
                    FilterDisposition::BypassedRaw
                } else {
                    FilterDisposition::Applied
                };
                2
            ],
        );
        if partial {
            assert_eq!(chunk.storage_size, 512);
        }
    }
    let mut records = chunks
        .into_iter()
        .map(|chunk| {
            let mask = chunk.filter_mask().raw_bits();
            (chunk.address, chunk.storage_size, mask)
        })
        .collect::<Vec<_>>();
    records.sort();
    drop(created);
    pure.close().unwrap();
    let native = file::libhdf5_open(&path);
    assert_eq!(
        native.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        expected
    );
    for index in 0..16 {
        assert_eq!(
            native
                .dataset(&format!("spacer{index}"))
                .unwrap()
                .read_raw::<u8>()
                .unwrap(),
            [31; 1_024]
        );
    }
    native.close().unwrap();
    let mut native_records = array_index::libhdf5_chunks(&path, 0)
        .into_iter()
        .map(|chunk| (chunk.address(), chunk.size(), chunk.filter_mask()))
        .collect::<Vec<_>>();
    native_records.sort();
    assert_eq!(records, native_records);
    let engine = test_util::open_engine(&path);
    if !matches!(input, Input::Flat) {
        engine.assert_array_metadata_pages(header);
    }
    assert!(engine.space_ownership().unwrap().analysis_complete);
    drop(engine);
    let reopened = File::open(&path).unwrap();
    assert_eq!(reopened.dataset("d").unwrap().read_i32().unwrap(), expected);
    reopened.close().unwrap();
}

#[derive(Clone, Copy)]
enum Input {
    Flat,
    Page,
    TransientPage,
}
