use core::num::NonZeroUsize;
use std::io::ErrorKind;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use hdf5_pure_format::__private::DataLayout;
use hdf5_pure_space::__private::AllocationClass;
use rstest::rstest;
use test_util::disk_log;
use test_util::disk_log::IoEvent;
use test_util::disk_log::IoOutcome;

use super::super::FlatDataset;
use super::Creation;
use crate::File;
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::SyncPolicy;
use crate::UserblockSize;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::address::StoredAddress;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::address::StoredAddressExt;
use crate::chunked_write;
use crate::chunked_write::ChunkedDataResult;
use crate::chunked_write::ChunkedHeader;
use crate::chunked_write::CompressedChunkSet;
use crate::dataset_request::PreparedFill;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::edit::WriteEngine;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::file_writer;
use crate::fill_value::FillPattern;
use crate::filters::ChunkContext;
use crate::image::WriteBuffering;
use crate::object_path::LinkNameBuf;
use crate::object_path::ObjectPathBuf;
use crate::type_builders::DatasetBuilder;
use crate::writer::FileBuilder;

#[rstest]
#[case::single(Creation::Fixed(1), FileSpaceStrategy::FsmAggr)]
#[case::fixed(Creation::Fixed(40), FileSpaceStrategy::FsmAggr)]
#[case::fixed_paged(Creation::Fixed(1_025), FileSpaceStrategy::FsmAggr)]
#[case::sparse(Creation::FixedSparse, FileSpaceStrategy::FsmAggr)]
#[case::extensible(Creation::Extensible(40), FileSpaceStrategy::FsmAggr)]
#[case::rotated(Creation::Rotated(8), FileSpaceStrategy::FsmAggr)]
#[case::extensible_paged(Creation::Rotated(20_000), FileSpaceStrategy::FsmAggr)]
#[case::allocated_empty(Creation::Extensible(0), FileSpaceStrategy::FsmAggr)]
#[case::undefined(Creation::Unallocated, FileSpaceStrategy::FsmAggr)]
#[case::aggregate(Creation::Fixed(40), FileSpaceStrategy::Aggr)]
#[case::append_only(Creation::Extensible(40), FileSpaceStrategy::None)]
#[case::page_single(Creation::Fixed(1), FileSpaceStrategy::Page)]
fn creation_preserves_consecutive_bytes_and_complete_placement(
    #[case] creation: Creation,
    #[case] strategy: FileSpaceStrategy,
    #[values(false, true)] filtered: bool,
    #[values(0, 512)] base: u64,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("direct.h5");
    let reference = directory.path().join("assembled.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(base).unwrap());
    builder.with_file_space_settings(
        FileSpaceSettings::new(strategy)
            .with_persistence(true)
            .with_page_size(FileSpacePageSize::try_from(512).unwrap()),
    );
    builder.create_dataset("kept").with_i32_data(&[31; 32]);
    builder.write(&path).unwrap();
    std::fs::copy(&path, &reference).unwrap();
    let set = prepared(creation, filtered);
    let mut direct = edit_test_util::open_engine(&path);
    let mut assembled = edit_test_util::open_engine(&reference);
    direct
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    assembled
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let ChunkedHeader {
        layout_message,
        pipeline_message,
    } = direct.write_contiguous_chunk_set(&set).unwrap();
    let (_, expected) = assembled
        .place_relocatable(
            chunked_write::chunked_data_len(&set).unwrap(),
            AllocationClass::Raw,
            |address| {
                let ChunkedDataResult {
                    data_bytes,
                    layout_message,
                    pipeline_message,
                } = chunked_write::assemble_chunked_at(&set, address)?;
                Ok((data_bytes, (layout_message, pipeline_message)))
            },
        )
        .unwrap();
    assert_eq!(layout_message, expected.0);
    assert_eq!(pipeline_message, expected.1);
    assert_eq!(
        direct
            .image()
            .read_exact_at(0, usize::try_from(direct.image.len()).unwrap())
            .unwrap(),
        assembled
            .image()
            .read_exact_at(0, usize::try_from(assembled.image.len()).unwrap())
            .unwrap()
    );
    assert_eq!(
        direct.end_of_allocation().unwrap(),
        assembled.end_of_allocation().unwrap()
    );
    assert_eq!(
        direct
            .space
            .admitted_snapshot()
            .sections()
            .collect::<Vec<_>>(),
        assembled
            .space
            .admitted_snapshot()
            .sections()
            .collect::<Vec<_>>()
    );
}

#[rstest]
#[case::payload(WriteBuffering::Unbuffered, 2, 16)]
#[case::index(WriteBuffering::Unbuffered, 40, 366)]
#[case::gathered_payload(WriteBuffering::Operation { page_size: 1, max_bytes: 128 }, 2, 144)]
#[case::gathered_index(WriteBuffering::Operation { page_size: 1, max_bytes: 128 }, 5, 366)]
fn creation_write_failures_restore_reuse_without_publishing(
    #[case] buffering: WriteBuffering,
    #[case] completed: usize,
    #[case] requested: usize,
    #[values(0, 3)] prefix: usize,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failure.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(1),
    );
    builder
        .create_dataset("hole")
        .with_u8_data(&vec![13; 8_192]);
    builder.create_dataset("kept").with_i32_data(&[31; 1_024]);
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine.sync_policy = SyncPolicy::OnClose;
    engine.delete(&ObjectPathBuf::parse("hole")).unwrap();
    engine.commit().unwrap();
    engine.set_write_buffering(buffering).unwrap();
    let before: Vec<_> = engine.space.admitted_snapshot().sections().collect();
    assert!(
        engine
            .space
            .reusable_sections()
            .any(|extent| extent.len() >= 1_006)
    );
    let mut builder = DatasetBuilder::new("d");
    Creation::Fixed(40).configure(&mut builder, false);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("d"), builder)
        .unwrap();
    disk_log::start();
    disk_log::fail_write_after(completed, prefix);
    let error = engine.commit().unwrap_err();
    let recording = disk_log::finish();
    let Error::Io(error) = error else {
        panic!("expected I/O failure, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    let failed: Vec<_> = recording
        .events
        .iter()
        .filter_map(|event| match event {
            IoEvent::Write {
                requested,
                written,
                outcome: IoOutcome::Failed(ErrorKind::Other),
                ..
            } => Some((*requested, *written)),
            _ => None,
        })
        .collect();
    assert_eq!(failed, [(requested, prefix)]);
    assert_eq!(
        engine
            .space
            .admitted_snapshot()
            .sections()
            .collect::<Vec<_>>(),
        before
    );
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), ["kept"]);
    assert_eq!(
        file.dataset("kept").unwrap().read_i32().unwrap(),
        [31; 1_024]
    );
    file.close().unwrap();
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::small_unbuffered(256, 0)]
#[case::large_unbuffered(2_048, 0)]
#[case::small_bounded(256, 8_192)]
#[case::large_bounded(2_048, 8_192)]
#[case::small_default(256, crate::edit::WRITE_GATHER_BYTES)]
#[case::large_default(2_048, crate::edit::WRITE_GATHER_BYTES)]
#[case::larger_default(16_384, crate::edit::WRITE_GATHER_BYTES)]
fn creation_bounds_temporary_storage_and_gathers_adjacent_writes(
    #[case] count: usize,
    #[case] budget: usize,
) {
    use test_util::allocation;

    use crate::FileLocking;
    use crate::FillTime;
    use crate::MemoryStrategy;
    use crate::PersistedFreeSpacePolicy;
    use crate::chunked_write::ChunkOptions;
    use crate::chunked_write::StorageAllocation;
    use crate::source::MetadataCacheConfig;

    let gathered = budget != 0;
    let buffering = if gathered {
        WriteBuffering::Operation {
            page_size: FileSpacePageSize::DEFAULT.get(),
            max_bytes: budget,
        }
    } else {
        WriteBuffering::Unbuffered
    };
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("memory.h5");
    FileBuilder::new().write(&path).unwrap();
    let reference = directory.path().join("reference.h5");
    std::fs::copy(&path, &reference).unwrap();
    let mut engine = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Bounded,
        PersistedFreeSpacePolicy::Tolerant,
    )
    .unwrap();
    engine.set_write_buffering(buffering).unwrap();
    let raw = vec![17; count * 1_024];
    let options = ChunkOptions {
        chunk_dims: Some(vec![256]),
        ..ChunkOptions::default()
    };
    let set = chunked_write::compress_chunks(
        engine.context.widths(),
        &raw,
        &[u64::try_from(count * 256).unwrap()],
        ChunkContext::basic(&[256], 4),
        &options,
        None,
        FillPattern::ZERO,
        FillTime::IfSet,
        StorageAllocation::Allocated,
    )
    .unwrap();
    let before_writes = engine.image.issued_write_order().len();
    let before_work = engine.image.buffering_work();
    let ((header, held), measured) =
        allocation::measure(&format!("contiguous_creation_{count}_{budget}"), || {
            let header = engine.write_contiguous_chunk_set(&set).unwrap();
            let held = engine.image.buffering_work();
            engine.image.ordering_barrier().unwrap();
            (header, held)
        });
    let ChunkedHeader {
        layout_message,
        pipeline_message,
    } = header;
    assert!(measured.blocks > 0, "{measured}");
    let DataLayout::Chunked { index, .. } = DataLayout::parse(&layout_message, 8, 8).unwrap()
    else {
        panic!("expected chunked layout");
    };
    let index = index.address().unwrap().get();
    let start = index - u64::try_from(raw.len()).unwrap();
    let physical = engine
        .context
        .addresses()
        .resolve(StoredAddress::new(start))
        .unwrap()
        .get();
    assert_eq!(
        engine.image().read_exact_at(physical, raw.len()).unwrap(),
        raw
    );
    let issued = engine.image.issued_write_order();
    let writes: Vec<_> = issued[before_writes..]
        .iter()
        .map(|&(offset, length)| (offset, usize::try_from(length).unwrap()))
        .collect();
    if gathered {
        assert!(writes.len() <= count.div_ceil(budget / 1_024) + 1);
        if raw.len() > budget {
            assert!(writes.len() > 1);
        }
    } else {
        assert_eq!(writes.len(), count + 1);
        assert_eq!(
            &writes[..count],
            (0..count)
                .map(|chunk| { (physical + u64::try_from(chunk * 1_024).unwrap(), 1_024) })
                .collect::<Vec<_>>()
        );
    }
    let (control, copied) = allocation::measure(
        &format!("contiguous_creation_assembly_{count}_{budget}"),
        || chunked_write::assemble_chunked_at(&set, StoredAddress::new(start)).unwrap(),
    );
    assert!(control.data_bytes.starts_with(&raw));
    assert!(
        copied.live_bytes >= u64::try_from(raw.len()).unwrap(),
        "{copied}"
    );
    if budget <= raw.len() / 16 {
        assert!(
            copied.peak_bytes > measured.peak_bytes + u64::try_from(raw.len() / 2).unwrap(),
            "{copied}; {measured}"
        );
    }
    assert_eq!(
        engine
            .image()
            .read_exact_at(physical, control.data_bytes.len())
            .unwrap(),
        control.data_bytes
    );
    let index_length = control.data_bytes.len() - raw.len();
    assert!(
        measured.peak_bytes
            <= u64::try_from(count * 64 + index_length * 2 + budget * 2 + 65_536).unwrap(),
        "{measured}"
    );
    let work = engine.image.buffering_work();
    assert!(
        work.copied_bytes - before_work.copied_bytes
            <= 2 * u64::try_from(control.data_bytes.len()).unwrap()
    );
    let mut legacy = WriteEngine::open_rw_with_strategy(
        &reference,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Bounded,
        PersistedFreeSpacePolicy::Tolerant,
    )
    .unwrap();
    legacy.set_write_buffering(buffering).unwrap();
    let ((old_header, old_held), old) = allocation::measure(
        &format!("contiguous_creation_legacy_{count}_{budget}"),
        || {
            let header = legacy
                .place_relocatable(
                    chunked_write::chunked_data_len(&set).unwrap(),
                    AllocationClass::Raw,
                    |address| {
                        let ChunkedDataResult {
                            data_bytes,
                            layout_message,
                            pipeline_message,
                        } = chunked_write::assemble_chunked_at(&set, address)?;
                        Ok((data_bytes, (layout_message, pipeline_message)))
                    },
                )
                .unwrap()
                .1;
            let held = legacy.image.buffering_work();
            legacy.image.ordering_barrier().unwrap();
            (header, held)
        },
    );
    assert_eq!(old_header, (layout_message, pipeline_message));
    assert!(old.peak_bytes >= u64::try_from(raw.len()).unwrap(), "{old}");
    assert_eq!(legacy.image.issued_write_order().len(), 1);
    assert_eq!(
        legacy
            .image()
            .read_exact_at(physical, control.data_bytes.len())
            .unwrap(),
        control.data_bytes
    );
    eprintln!(
        "legacy {old}; before_drain={held:?}; legacy_before_drain={old_held:?}; buffering={work:?}"
    );

    let mut cursor = physical;
    for &(offset, length) in &writes {
        assert_eq!(offset, cursor);
        if gathered {
            assert!(length <= budget + 1_024.max(index_length));
        }
        cursor += u64::try_from(length).unwrap();
    }
    assert_eq!(
        cursor,
        physical + u64::try_from(control.data_bytes.len()).unwrap()
    );
    eprintln!(
        "count={count} budget={budget} writes={} prepared={} index={index_length} {measured}",
        writes.len(),
        raw.len()
    );
}

pub(super) fn prepared(creation: Creation, filtered: bool) -> CompressedChunkSet {
    let mut builder = DatasetBuilder::new("d");
    creation.configure(&mut builder, filtered);
    let FlatDataset {
        name: _,
        dt,
        ds,
        storage,
        extent: _,
        attrs: _,
        chunk_options,
        maxshape,
        vl_attrs: _,
        dense_attrs: _,
        fill,
        fill_time,
        #[cfg(feature = "provenance")]
            provenance: _,
    } = super::super::flatten_dataset(file_writer::WIDTHS, builder, LinkNameBuf::new("d").unwrap())
        .unwrap();
    let dimensions = chunk_options.resolve_chunk_dims(&ds.dimensions);
    let context = ChunkContext::from_datatype(&dimensions, &dt).unwrap();
    let element = NonZeroUsize::new(usize::try_from(context.element_size.get()).unwrap()).unwrap();
    chunked_write::compress_chunks(
        file_writer::WIDTHS,
        storage.raw(),
        &ds.dimensions,
        context,
        &chunk_options,
        maxshape.as_deref(),
        FillPattern::from_value(fill.as_ref().map(PreparedFill::as_bytes), element),
        fill_time,
        storage.allocation(),
    )
    .unwrap()
}
