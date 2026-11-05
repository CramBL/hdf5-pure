use std::io::ErrorKind;

use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::ChunkIndexLayout;
use hdf5_pure_format::__private::DataLayout;
use hdf5_pure_format::__private::ObjectHeader;
use hdf5_pure_space::__private::PageTransition;
use hdf5_pure_space::__private::PageType;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::disk_log;
use test_util::disk_log::IoEvent;
use test_util::disk_log::IoOutcome;

use crate::AttrValue;
use crate::DeflateLevel;
use crate::File;
use crate::FileAccessProperties;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::FileLocking;
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::FillTime;
use crate::MemoryStrategy;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::PersistedFreeSpacePolicy;
use crate::PhysicalAddress;
use crate::SyncPolicy;
use crate::UserblockSize;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::address::StoredAddress;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::address::StoredAddressExt;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::chunked_write;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::chunked_write::ChunkOptions;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::chunked_write::StorageAllocation;
use crate::dataset_request::{PreparedDataset, StagedFill};
use crate::dataspace::MaxExtent;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::edit::WriteEngine;
use crate::edit::test_util as edit_test_util;
use crate::error::{Error, FormatError};
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::fill_value::FillPattern;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::filters::ChunkContext;
use crate::image::WriteBuffering;
use crate::message_type::MessageType;
use crate::object_graph::ObjectId;
use crate::object_path::LinkNameBuf;
use crate::object_path::ObjectPathBuf;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::source::MetadataCacheConfig;
use crate::type_builders;
use crate::type_builders::DatasetBuilder;
use crate::vl_data::VlElement;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};
use crate::writer::FileBuilder;

mod contiguous;
mod fragmented;
pub(in crate::edit) mod lifecycle;
mod providers;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod resources;

#[rstest]
#[case::deferred_fixed(false, false)]
#[case::deferred_extensible(true, false)]
#[case::empty_fixed(false, true)]
#[case::allocated_extensible(true, true)]
fn empty_page_creation_reserves_only_defined_index_storage(
    #[case] unlimited: bool,
    #[case] allocated: bool,
    #[values(false, true)] filtered: bool,
    #[values(false, true)] persist: bool,
    #[values(512, 4_096)] page: u64,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("empty-created.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(persist)
            .with_page_size(FileSpacePageSize::try_from(page).unwrap()),
    );
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine.begin_page(PageType::Meta).unwrap();
    engine.append(&[17; 17]).unwrap();
    let before_eoa = engine.end_of_allocation().unwrap();
    assert_ne!(before_eoa.get() % page, 0);
    assert_eq!(
        engine
            .space
            .plan_transition(before_eoa, PageType::Meta)
            .unwrap(),
        Some(PageTransition::Unchanged)
    );
    let before_sections: Vec<_> = engine.space.admitted_snapshot().sections().collect();
    let before_writes = engine.image.issued_write_order().len();
    let mut dataset = DatasetBuilder::new("d");
    if allocated {
        dataset.with_i32_data(&[]).with_shape(&[0]);
    } else {
        dataset.with_unallocated_storage_for::<i32>(&[0]);
    }
    dataset.with_chunks(&[4]);
    if unlimited {
        dataset.with_maxshape(&[MaxExtent::Unlimited]);
    }
    if filtered {
        dataset.with_shuffle();
    }
    let mut flat = super::flatten_dataset(
        engine.context.widths(),
        dataset,
        LinkNameBuf::new("d").unwrap(),
    )
    .unwrap();
    let bytes = engine.build_chunked_dataset(&mut flat).unwrap();
    let header = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
    assert_eq!(
        header
            .messages
            .iter()
            .filter(|message| message.msg_type == MessageType::FILTER_PIPELINE)
            .count(),
        usize::from(filtered)
    );
    let layout = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .unwrap();
    let DataLayout::Chunked { index, .. } = DataLayout::parse(&layout.data, 8, 8).unwrap() else {
        panic!("expected chunked storage");
    };
    let address = match index {
        ChunkIndexLayout::FixedArray { address, .. } if !unlimited => address,
        ChunkIndexLayout::ExtensibleArray { address, .. } if unlimited => address,
        index => panic!("unexpected chunk index: {index:?}"),
    };
    let has_index = unlimited && allocated;
    assert_eq!(address.is_some(), has_index);
    if has_index {
        assert!(engine.image.issued_write_order().len() > before_writes);
    } else {
        assert_eq!(engine.end_of_allocation().unwrap(), before_eoa);
        assert_eq!(engine.image.issued_write_order().len(), before_writes);
        assert_eq!(
            engine
                .space
                .plan_transition(before_eoa, PageType::Meta)
                .unwrap(),
            Some(PageTransition::Unchanged)
        );
        assert_eq!(
            engine
                .space
                .admitted_snapshot()
                .sections()
                .collect::<Vec<_>>(),
            before_sections
        );
    }
}

#[rstest]
#[case::empty_fixed(Creation::Fixed(0))]
#[case::single(Creation::Fixed(1))]
#[case::fixed(Creation::Fixed(40))]
#[case::fixed_sparse(Creation::FixedSparse)]
#[case::fixed_paged(Creation::Fixed(1_025))]
#[case::empty_extensible(Creation::Extensible(0))]
#[case::extensible(Creation::Extensible(40))]
#[case::rotated(Creation::Rotated(8))]
#[case::paged_parent(Creation::Rotated(20_000))]
#[case::pending_append(Creation::Pending)]
#[case::unallocated(Creation::Unallocated)]
fn staged_page_creation_preserves_values_and_complete_index_allocations(
    #[case] creation: Creation,
    #[values(512, 4_096)] page: u64,
    #[values(
        CreationAccess::Mirrored,
        CreationAccess::Bounded,
        CreationAccess::Userblock
    )]
    access: CreationAccess,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("created.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(access.base()).unwrap());
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(1)
            .with_page_size(FileSpacePageSize::try_from(page).unwrap()),
    );
    builder.create_dataset("kept").with_i32_data(&[31; 32]);
    builder.write(&path).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(access.strategy())
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    let mut dataset = file
        .root()
        .create_dataset("d", |builder| creation.configure(builder, false))
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
    let mut coordinates: Vec<_> = dataset
        .chunks()
        .unwrap()
        .into_iter()
        .map(|chunk| chunk.offset)
        .collect();
    coordinates.sort();
    assert_eq!(coordinates, creation.coordinates());
    drop(dataset);
    file.close().unwrap();
    let engine = edit_test_util::open_engine(&path);
    let index = engine
        .chunked_index_spans(ObjectId::new(PhysicalAddress::new(header)))
        .unwrap();
    match creation {
        Creation::Fixed(0 | 1) => assert!(index.is_empty()),
        Creation::Fixed(_) | Creation::FixedSparse => assert_eq!(index.len(), 2),
        _ => assert!(index.len() >= 2),
    }
    if matches!(creation, Creation::Fixed(1_025) | Creation::Rotated(20_000)) {
        assert!(index.iter().any(|range| range.length() > page));
    }
    assert!(engine.space_ownership().unwrap().analysis_complete);
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        creation.expected()
    );
    assert_eq!(file.dataset("kept").unwrap().read_i32().unwrap(), [31; 32]);
    file.close().unwrap();
}

#[rstest]
#[case::first_payload(0, 0, 16)]
#[case::partial_payload(1, 3, 16)]
#[case::array_header(40, 0, 28)]
#[case::array_data(41, 3, 338)]
fn failed_page_creation_preserves_reusable_state_and_existing_values(
    #[case] writes: usize,
    #[case] prefix: usize,
    #[case] requested: usize,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("creation-failure.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
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
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine.delete(&ObjectPathBuf::parse("hole")).unwrap();
    engine.commit().unwrap();
    let before: Vec<_> = engine.space.admitted_snapshot().sections().collect();
    assert!(
        engine
            .space
            .reusable_sections()
            .any(|extent| extent.len() >= 640)
    );
    let mut builder = DatasetBuilder::new("d");
    Creation::Fixed(40).configure(&mut builder, false);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("d"), builder)
        .unwrap();
    disk_log::start();
    disk_log::fail_write_after(writes, prefix);
    let error = engine.commit().unwrap_err();
    let recording = disk_log::finish();
    let failed: Vec<_> = recording
        .events
        .iter()
        .filter_map(|event| match event {
            IoEvent::Write {
                requested,
                outcome: IoOutcome::Failed(ErrorKind::Other),
                ..
            } => Some(*requested),
            _ => None,
        })
        .collect();
    assert_eq!(failed, [requested]);
    let Error::Io(error) = error else {
        panic!("expected an I/O error, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
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
#[case::small(256)]
#[case::large(2_048)]
fn page_creation_emits_retained_chunks_without_an_aggregate_payload_buffer(#[case] count: usize) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("creation-memory.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
    );
    builder.write(&path).unwrap();
    let mut engine = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Bounded,
        PersistedFreeSpacePolicy::Tolerant,
    )
    .unwrap();
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let raw = vec![17; count * 1_024];
    let options = ChunkOptions {
        chunk_dims: Some(vec![256]),
        ..ChunkOptions::default()
    };
    let set = chunked_write::compress_chunks(
        crate::file_writer::WIDTHS,
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
    let geometry = set.array_geometry().unwrap();
    let writes = engine.image.issued_write_order().len();
    let (header, measured) =
        allocation::measure(&format!("page_creation_encoding_{count}"), || {
            engine.write_array_chunk_set(&set, geometry).unwrap()
        });
    assert!(
        measured.peak_bytes < u64::try_from(raw.len() / 2).unwrap(),
        "{measured:?}, payload={}",
        raw.len()
    );
    assert!(!header.layout_message.is_empty());
    let issued = engine.image.issued_write_order();
    let payload = &issued[writes..writes + count];
    let start = payload[0].0;
    assert_eq!(
        payload,
        (0..count)
            .map(|index| (start + u64::try_from(index * 1_024).unwrap(), 1_024))
            .collect::<Vec<_>>()
    );
    assert_eq!(engine.image().read_exact_at(start, raw.len()).unwrap(), raw);
    let (control, copied) = allocation::measure(&format!("page_creation_assembly_{count}"), || {
        chunked_write::assemble_chunked_at(&set, StoredAddress::new(0)).unwrap()
    });
    assert!(control.data_bytes.starts_with(&raw));
    assert!(
        copied.live_bytes >= u64::try_from(raw.len()).unwrap(),
        "{copied:?}"
    );
}

#[derive(Clone, Copy)]
pub(in crate::edit) enum Creation {
    Extensible(usize),
    Fixed(usize),
    FixedSparse,
    Pending,
    Rotated(u64),
    Unallocated,
}

impl Creation {
    pub(in crate::edit) fn configure(self, builder: &mut DatasetBuilder, filtered: bool) {
        if matches!(self, Self::Unallocated) {
            builder
                .with_unallocated_storage(type_builders::make_i32_type(), &[8])
                .with_fill_value(-7_i32);
        } else {
            builder.with_i32_data(&self.initial());
        }
        match self {
            Self::Extensible(_) | Self::Pending | Self::Unallocated => {
                builder
                    .with_chunks(&[4])
                    .with_maxshape(&[MaxExtent::Unlimited]);
            }
            Self::Fixed(_) => {
                builder.with_chunks(&[4]);
            }
            Self::FixedSparse => {
                builder
                    .with_chunks(&[4])
                    .with_maxshape(&[MaxExtent::Fixed(320)]);
            }
            Self::Rotated(maximum) => {
                builder
                    .with_shape(&[2, 2])
                    .with_chunks(&[1, 1])
                    .with_maxshape(&[MaxExtent::Fixed(maximum), MaxExtent::Unlimited]);
            }
        }
        if filtered {
            builder.with_deflate(DeflateLevel::Six);
        }
    }

    pub(in crate::edit) fn expected(self) -> Vec<i32> {
        if matches!(self, Self::Pending) {
            return (1..=8).collect();
        }
        self.initial()
    }

    pub(in crate::edit) fn coordinates(self) -> Vec<Vec<u64>> {
        match self {
            Self::Extensible(count) | Self::Fixed(count) => (0..count)
                .map(|index| vec![u64::try_from(index * 4).unwrap()])
                .collect(),
            Self::FixedSparse => (0..40).map(|index| vec![index * 4]).collect(),
            Self::Pending => vec![vec![0], vec![4]],
            Self::Unallocated => Vec::new(),
            Self::Rotated(_) => vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]],
        }
    }

    fn initial(self) -> Vec<i32> {
        match self {
            Self::Extensible(count) | Self::Fixed(count) => (0..count * 4)
                .map(|index| i32::try_from(index / 16 % 7).unwrap())
                .collect(),
            Self::FixedSparse => Self::Fixed(40).initial(),
            Self::Pending => (1..=4).collect(),
            Self::Unallocated => vec![-7; 8],
            Self::Rotated(_) => vec![11, 12, 13, 14],
        }
    }
}

#[derive(Clone, Copy)]
pub(in crate::edit) enum CreationAccess {
    Bounded,
    Mirrored,
    Userblock,
}

impl CreationAccess {
    pub(in crate::edit) fn base(self) -> u64 {
        match self {
            Self::Bounded | Self::Mirrored => 0,
            Self::Userblock => 8_192,
        }
    }

    pub(in crate::edit) fn strategy(self) -> MemoryStrategy {
        match self {
            Self::Bounded => MemoryStrategy::Bounded,
            Self::Mirrored | Self::Userblock => MemoryStrategy::Mirrored,
        }
    }
}

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two)]
#[case::two_four(OffsetWidth::Two, LengthWidth::Four)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::eight_four(OffsetWidth::Eight, LengthWidth::Four)]
#[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight)]
fn staging_preserves_payload_allocation_with_stored_variable_length_sizes(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
) {
    let widths = FormatWidths { offsets, lengths };
    let mut builder = DatasetBuilder::new("values");
    builder.with_vlen_strings(&["alpha", "", "gamma"]);
    let pointer = builder.data.as_ref().unwrap().as_ref().unwrap().as_ptr();
    let capacity = builder.data.as_ref().unwrap().as_ref().unwrap().capacity();
    let datatype = builder.datatype.as_ref().unwrap().clone();
    builder.fill = Some(StagedFill::source(datatype, vec![0; 16]));
    builder.set_attr(
        "labels",
        AttrValue::VarLenStringArray(vec!["one".into(), "two".into()]),
    );
    let mut prepared =
        super::flatten_dataset(widths, builder, LinkNameBuf::new("values").unwrap()).unwrap();
    let stride = VlElement::serialized_size(offsets);
    assert_eq!(prepared.extent, u64::try_from(3 * stride).unwrap());
    assert_eq!(prepared.storage.raw().len(), 3 * stride);
    assert_eq!(prepared.storage.raw().as_ptr(), pointer);
    assert_eq!(prepared.storage.raw_mut().unwrap().capacity(), capacity);
    assert_eq!(prepared.fill.as_ref().unwrap().as_bytes(), vec![0; stride]);
    assert_eq!(prepared.attrs[0].raw_data.len(), 2 * stride);
    let staging = prepared.storage.take_vl_string_staging().unwrap();
    assert_eq!(staging.patch_offsets, [0, stride, 2 * stride]);
    assert_eq!(staging.collections.len(), 1);
}

#[rstest]
#[case::two(OffsetWidth::Two, 10)]
#[case::four(OffsetWidth::Four, 12)]
#[case::eight(OffsetWidth::Eight, 16)]
fn overwrites_prepare_variable_length_bytes_once_before_application(
    #[case] offsets: OffsetWidth,
    #[case] stride: usize,
) {
    let mut builder = DatasetBuilder::new("values");
    builder.with_vlen_strings(&["first", "second"]);
    let pointer = builder.data.as_ref().unwrap().as_ref().unwrap().as_ptr();
    let mut prepared = PreparedDataset::parse(builder)
        .unwrap()
        .into_overwrite()
        .unwrap()
        .into_widths(FormatWidths {
            offsets,
            lengths: LengthWidth::Two,
        })
        .unwrap();
    assert_eq!(prepared.raw().len(), 2 * stride);
    let (bytes, staging) = prepared.take_payload();
    assert_eq!(bytes.as_ptr(), pointer);
    assert_eq!(staging.unwrap().patch_offsets, [0, stride]);
}

#[rstest]
#[case::dimension(65_536, None, 65_536)]
#[case::maximum(1, Some(MaxExtent::Fixed(65_536)), 65_536)]
#[case::unlimited(1, Some(MaxExtent::Unlimited), u64::MAX)]
fn staged_declarations_reject_unrepresentable_length_fields(
    #[case] dimension: u64,
    #[case] maximum: Option<MaxExtent>,
    #[case] length: u64,
) {
    let mut builder = DatasetBuilder::new("values");
    builder.with_unallocated_storage(type_builders::make_u8_type(), &[dimension]);
    if let Some(maximum) = maximum {
        builder.with_maxshape(&[maximum]);
    }
    let result = super::flatten_dataset(
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Two,
        },
        builder,
        LinkNameBuf::new("values").unwrap(),
    );
    let Err(error) = result else {
        panic!("the unrepresentable declaration must fail");
    };
    assert!(
        matches!(error, Error::Format(FormatError::LengthTooLarge { length: actual, length_size: 2 }) if actual == length),
        "{error:?}"
    );
}

#[rstest]
#[case::creation(false)]
#[case::overwrite(true)]
fn dataset_preparation_rejects_unrepresentable_heap_collections(#[case] overwrite: bool) {
    let mut builder = DatasetBuilder::new("values");
    builder.with_vlen_strings(&[&"x".repeat(70_000)]);
    let length =
        u64::try_from(builder.vl_string_staging.as_ref().unwrap().collections[0].len()).unwrap();
    assert!(length > u64::from(u16::MAX));
    let error = preparation_error(
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Two,
        },
        builder,
        overwrite,
    );
    assert!(
        matches!(error, Error::Format(FormatError::LengthTooLarge { length: actual, length_size: 2 }) if actual == length),
        "{error:?}"
    );
}

#[rstest]
#[case::two(OffsetWidth::Two)]
#[case::four(OffsetWidth::Four)]
fn dataset_preparation_rejects_narrow_legacy_references(
    #[case] offsets: OffsetWidth,
    #[values(false, true)] overwrite: bool,
) {
    let mut builder = DatasetBuilder::new("values");
    builder.with_raw_data(type_builders::make_object_reference_type(), vec![0; 8], 1);
    let error = preparation_error(
        FormatWidths {
            offsets,
            lengths: LengthWidth::Eight,
        },
        builder,
        overwrite,
    );
    assert!(
        matches!(error, Error::Format(FormatError::ReferenceWidthUnsupported { offset_size }) if offset_size == offsets.get()),
        "{error:?}"
    );
}

fn preparation_error(widths: FormatWidths, builder: DatasetBuilder, overwrite: bool) -> Error {
    let result = if overwrite {
        PreparedDataset::parse(builder)
            .unwrap()
            .into_overwrite()
            .unwrap()
            .into_widths(widths)
            .map(|_| ())
            .map_err(Error::Format)
    } else {
        super::flatten_dataset(widths, builder, LinkNameBuf::new("values").unwrap()).map(|_| ())
    };
    let Err(error) = result else {
        panic!("the unrepresentable preparation must fail");
    };
    error
}
