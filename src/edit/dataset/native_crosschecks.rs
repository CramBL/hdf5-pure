use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::ChunkRecord;
use rstest::rstest;
use test_util_hdf5::numeric::ByteOrder;
use test_util_hdf5::{attr, dataset, file};

use crate::address::AddressContext;
use crate::data_layout::{ChunkIndexLayout, DataLayout};
use crate::dataset_request::PreparedFill;
use crate::dataspace::Dataspace;
use crate::edit::AppendBuilder;
use crate::edit::WriteEngine;
use crate::edit::context;
use crate::edit::object_header;
use crate::error::Error;
use crate::file_writer;
use crate::link_message::{LinkMessage, LinkTarget};
use crate::message_type::MessageType;
use crate::object_path::LinkNameBuf;
use crate::reader::OpenedSource;
use crate::shared_message::DatatypeLocation;
use crate::source::{BytesSource, PhysicalSource};
use crate::superblock;
use crate::type_builders;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};
use crate::{
    AttrValue, DatasetBuilder, DeflateLevel, File, FileAccessProperties, FileBuilder,
    FileSpaceSettings, FileSpaceStrategy, LibVer, MaxExtent, MemoryStrategy, StoredAddress,
};

use super::MovingWrite;

#[derive(Clone, Copy, Debug)]
enum IndexCase {
    Empty,
    Fixed,
    Single,
    Unlimited,
}

impl IndexCase {
    fn initial(self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Fixed | Self::Unlimited => 5,
            Self::Single => 3,
        }
    }

    fn maximum(self) -> Option<usize> {
        match self {
            Self::Empty | Self::Fixed => Some(12),
            Self::Single => Some(4),
            Self::Unlimited => None,
        }
    }

    fn grown(self) -> usize {
        match self {
            Self::Empty | Self::Single => 4,
            Self::Fixed | Self::Unlimited => 10,
        }
    }

    fn matches(self, index: ChunkIndexLayout) -> bool {
        match self {
            Self::Empty | Self::Fixed => matches!(index, ChunkIndexLayout::FixedArray { .. }),
            Self::Single => matches!(index, ChunkIndexLayout::SingleChunk { .. }),
            Self::Unlimited => matches!(index, ChunkIndexLayout::ExtensibleArray { .. }),
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
fn finite_append_plans_preserve_native_chunks_at_independent_widths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(IndexCase::Empty, IndexCase::Fixed, IndexCase::Single)] index: IndexCase,
    #[values(false, true)] filtered: bool,
) {
    append_native_dataset(FormatWidths { offsets, lengths }, index, filtered);
}

#[rstest]
#[case::two(OffsetWidth::Two)]
#[case::four(OffsetWidth::Four)]
#[case::eight(OffsetWidth::Eight)]
fn unlimited_append_plans_preserve_native_chunks_at_selected_offset_width(
    #[case] offsets: OffsetWidth,
    #[values(false, true)] filtered: bool,
) {
    append_native_dataset(
        FormatWidths {
            offsets,
            lengths: LengthWidth::Eight,
        },
        IndexCase::Unlimited,
        filtered,
    );
}

fn append_native_dataset(widths: FormatWidths, index_case: IndexCase, filtered: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("append.h5");
    create_native_dataset(&path, widths, 512, index_case, filtered);
    let mut bytes = std::fs::read(&path).unwrap();
    let located = superblock::read(&BytesSource::new(bytes.as_slice())).unwrap();
    assert_eq!(located.widths, widths);
    let addresses = AddressContext::from_effective_base(located.superblock.base_address);
    let root = WriteEngine::gather_oh_messages(
        &BytesSource::new(bytes.as_slice()),
        addresses
            .resolve(located.superblock.root_group_address)
            .unwrap()
            .get(),
        addresses,
        widths,
    )
    .unwrap();
    let mut link = root_link(&root);
    let LinkTarget::Hard {
        object_header_address,
    } = link.link_target
    else {
        panic!("the native dataset must have a hard link");
    };
    let mut builder = AppendBuilder::new();
    builder.append(
        &(u32::try_from(index_case.initial() + 1).unwrap()
            ..=u32::try_from(index_case.grown()).unwrap())
            .collect::<Vec<_>>(),
    );
    let plan = {
        let source = BytesSource::new(bytes.as_slice());
        let opened = OpenedSource::for_test(&source, addresses);
        let dataset_addr = addresses.resolve(object_header_address).unwrap().get();
        let region =
            WriteEngine::gather_oh_messages(&source, dataset_addr, addresses, widths).unwrap();
        let (dataspace, layout) = dataset_layout(&region);
        let DataLayout::Chunked { index, .. } = layout else {
            panic!("the native dataset must use chunks");
        };
        assert!(index_case.matches(index), "{index:?}");
        let chunks = crate::chunked_read::enumerate_chunks_from_source(
            &PhysicalSource::new(&opened).stored(),
            &layout,
            &dataspace,
            widths,
        )
        .unwrap();
        let plan = WriteEngine::prepare_append(&opened, dataset_addr, &builder, widths).unwrap();
        let MovingWrite::AppendedChunks { kept_chunks, .. } = &plan else {
            panic!("append must prepare chunk storage");
        };
        assert_eq!(kept_chunks.len(), index_case.initial() / 4);
        assert_eq!(
            kept_chunks
                .iter()
                .map(|chunk| chunk.address)
                .collect::<Vec<_>>(),
            chunks
                .iter()
                .take(index_case.initial() / 4)
                .map(|chunk| chunk.address)
                .collect::<Vec<_>>()
        );
        plan
    };
    let MovingWrite::AppendedChunks {
        region,
        new_dataspace_body,
        flags,
        chunk_dims_u32,
        element_size,
        raw_size,
        has_filters,
        index,
        mut kept_chunks,
        new_chunks,
        ..
    } = plan
    else {
        panic!("append must prepare chunk storage");
    };
    for chunk in new_chunks {
        let address = append_checked(&mut bytes, widths, addresses, chunk.bytes());
        kept_chunks.push(ChunkRecord {
            address,
            stored_size: chunk.stored_size(),
            filter_mask: chunk.filter_mask(),
        });
    }
    let index_address = addresses
        .relative(PhysicalAddress::new(u64::try_from(bytes.len()).unwrap()))
        .unwrap();
    let encoded = index
        .build_at(widths, &kept_chunks, raw_size, has_filters, index_address)
        .unwrap();
    assert_eq!(
        u64::try_from(encoded.len()).unwrap(),
        index
            .encoded_len(widths, &kept_chunks, raw_size, has_filters)
            .unwrap()
    );
    append_checked(&mut bytes, widths, addresses, &encoded);
    let layout = index
        .kind
        .layout_message(
            widths,
            flags,
            &chunk_dims_u32,
            element_size,
            &kept_chunks,
            index_address,
            has_filters,
        )
        .unwrap();
    let region = super::replace_dataspace_message(&region, &new_dataspace_body).unwrap();
    let region = super::replace_layout_message(&region, &layout).unwrap();
    let (dataspace, layout) = dataset_layout(&region);
    assert_eq!(
        dataspace.dimensions,
        [u64::try_from(index_case.grown()).unwrap()]
    );
    let DataLayout::Chunked { index, .. } = layout else {
        panic!("append must retain chunked storage");
    };
    assert!(index_case.matches(index), "{index:?}");
    let header = object_header::build_v2_object_header(&region).unwrap();
    link.link_target = LinkTarget::Hard {
        object_header_address: append_checked(&mut bytes, widths, addresses, &header),
    };
    let root = root.with_bytes(
        root.splice_first(MessageType::LINK)
            .unwrap()
            .unwrap()
            .replace_body(&link.serialize(widths.offsets).unwrap())
            .unwrap(),
    );
    let header = object_header::build_v2_object_header(&root).unwrap();
    let mut block = located.superblock;
    block.root_group_address = append_checked(&mut bytes, widths, addresses, &header);
    block.eof_address = u64::try_from(bytes.len()).unwrap();
    let encoded = hdf5_pure_format::__private::serialize_superblock(&block).unwrap();
    bytes[512..512 + encoded.len()].copy_from_slice(&encoded);
    std::fs::write(&path, &bytes).unwrap();
    {
        let native = file::libhdf5_open(&path);
        let dataset = native.dataset("values").unwrap();
        assert_eq!(dataset.shape(), [index_case.grown()]);
        assert_eq!(
            dataset.read_raw::<u32>().unwrap(),
            (1..=u32::try_from(index_case.grown()).unwrap()).collect::<Vec<_>>()
        );
    }
    let reopened = crate::File::from_bytes(bytes).unwrap();
    assert_eq!(
        reopened.dataset("values").unwrap().read_u32().unwrap(),
        (1..=u32::try_from(index_case.grown()).unwrap()).collect::<Vec<_>>()
    );
    reopened.close().unwrap();
    directory.close().unwrap();
}

fn create_native_dataset(
    path: &Path,
    widths: FormatWidths,
    userblock: u64,
    index: IndexCase,
    filtered: bool,
) {
    let native =
        file::libhdf5_create_v110_with_widths(path, widths.offsets, widths.lengths, userblock);
    let mut builder = native
        .new_dataset_builder()
        .empty_as(ByteOrder::LittleEndian.u32_datatype());
    builder = builder.chunk((4,));
    if filtered {
        builder = builder.deflate(1);
    }
    let dataset = match index.maximum() {
        Some(maximum) => builder
            .shape((index.initial().max(4)..=maximum,))
            .create("values")
            .unwrap(),
        None => builder
            .shape((index.initial()..,))
            .create("values")
            .unwrap(),
    };
    dataset.resize((index.initial(),)).unwrap();
    dataset
        .write_raw(&(1..=u32::try_from(index.initial()).unwrap()).collect::<Vec<_>>())
        .unwrap();
}

fn root_link(region: &super::OhRegion) -> LinkMessage {
    let mut position = 0;
    while let Some((kind, start, end)) = region.next_message(position).unwrap() {
        if kind == MessageType::LINK {
            let link =
                LinkMessage::parse_with_offset_width(&region[start..end], region.widths().offsets)
                    .unwrap();
            assert_eq!(link.name, "values");
            return link;
        }
        position = end;
    }
    panic!("the native root must contain the dataset link");
}

fn dataset_layout(region: &super::OhRegion) -> (Dataspace, DataLayout) {
    let mut dataspace = None;
    let mut layout = None;
    let mut position = 0;
    while let Some((kind, start, end)) = region.next_message(position).unwrap() {
        if kind == MessageType::DATASPACE {
            dataspace = Some(
                Dataspace::parse_with_length_width(&region[start..end], region.widths().lengths)
                    .unwrap(),
            );
        } else if kind == MessageType::DATA_LAYOUT {
            layout =
                Some(DataLayout::parse_with_widths(&region[start..end], region.widths()).unwrap());
        }
        position = end;
    }
    (dataspace.unwrap(), layout.unwrap())
}

fn append_checked(
    bytes: &mut Vec<u8>,
    widths: FormatWidths,
    addresses: AddressContext,
    contents: &[u8],
) -> StoredAddress {
    let start = PhysicalAddress::new(u64::try_from(bytes.len()).unwrap());
    context::require_range(
        widths.offsets,
        addresses,
        FileRange::new(start, u64::try_from(contents.len()).unwrap()).unwrap(),
    )
    .unwrap();
    bytes.extend_from_slice(contents);
    addresses.relative(start).unwrap()
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
fn finite_append_rejects_growth_beyond_native_maximum(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("maximum.h5");
    let widths = FormatWidths { offsets, lengths };
    create_native_dataset(&path, widths, 512, IndexCase::Single, false);
    let bytes = std::fs::read(&path).unwrap();
    let located = superblock::read(&BytesSource::new(bytes.as_slice())).unwrap();
    let addresses = AddressContext::from_effective_base(located.superblock.base_address);
    let source = BytesSource::new(bytes.as_slice());
    let root = WriteEngine::gather_oh_messages(
        &source,
        addresses
            .resolve(located.superblock.root_group_address)
            .unwrap()
            .get(),
        addresses,
        widths,
    )
    .unwrap();
    let LinkTarget::Hard {
        object_header_address,
    } = root_link(&root).link_target
    else {
        panic!("a dataset must have a hard link");
    };
    let mut builder = AppendBuilder::new();
    builder.append(&[4_u32, 5]);
    let error = WriteEngine::prepare_append(
        &OpenedSource::for_test(&source, addresses),
        addresses.resolve(object_header_address).unwrap().get(),
        &builder,
        widths,
    )
    .err()
    .unwrap();
    assert!(
        matches!(
            error,
            Error::AppendUnsupported("append exceeds the dataset's maximum first dimension")
        ),
        "{error:?}"
    );
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    {
        let native = file::libhdf5_open(&path);
        assert_eq!(
            native.dataset("values").unwrap().read_raw::<u32>().unwrap(),
            [1, 2, 3]
        );
    }
    directory.close().unwrap();
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
fn staged_variable_length_storage_is_native_readable_at_independent_widths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("strings.h5");
    drop(file::libhdf5_create_with_widths(
        &path, offsets, lengths, 512,
    ));
    let mut bytes = std::fs::read(&path).unwrap();
    let located = superblock::read(&BytesSource::new(bytes.as_slice())).unwrap();
    let widths = located.widths;
    let addresses = AddressContext::from_effective_base(located.superblock.base_address);
    let mut root = WriteEngine::gather_oh_messages(
        &BytesSource::new(bytes.as_slice()),
        addresses
            .resolve(located.superblock.root_group_address)
            .unwrap()
            .get(),
        addresses,
        widths,
    )
    .unwrap();
    let mut builder = DatasetBuilder::new("values");
    builder.with_vlen_strings(&["alpha", "", "ζ"]);
    builder.set_attr("label", AttrValue::VarLenString("payload".into()));
    let mut dataset =
        super::flatten_dataset(widths, builder, LinkNameBuf::new("values").unwrap()).unwrap();
    let staging = dataset.storage.take_vl_string_staging().unwrap();
    let heaps = staging
        .collections
        .iter()
        .map(|collection| append_checked(&mut bytes, widths, addresses, collection).get())
        .collect::<Vec<_>>();
    type_builders::patch_vl_refs_masked_with_width(
        offsets,
        dataset.storage.raw_mut().unwrap(),
        &staging.patch_offsets,
        &heaps,
    )
    .unwrap();
    for (index, collections) in dataset.vl_attrs {
        let heaps = collections
            .iter()
            .map(|collection| append_checked(&mut bytes, widths, addresses, collection).get())
            .collect::<Vec<_>>();
        type_builders::patch_vl_refs_with_width(
            offsets,
            &mut dataset.attrs[index].raw_data,
            &heaps,
        )
        .unwrap();
    }
    let data = append_checked(&mut bytes, widths, addresses, dataset.storage.raw());
    let header = file_writer::build_dataset_oh(
        &dataset.dt,
        &DatatypeLocation::Inline,
        &dataset.ds,
        data,
        dataset.extent,
        &dataset.attrs,
        None,
        dataset.fill_time,
        dataset.fill.as_ref().map(PreparedFill::as_bytes),
        LibVer::V110,
        file_writer::HeaderOptions::PLAIN.with_widths(widths),
    )
    .unwrap();
    let dataset_addr = append_checked(&mut bytes, widths, addresses, &header);
    root.push(
        MessageType::LINK,
        &file_writer::make_link("values", dataset_addr)
            .serialize(offsets)
            .unwrap(),
    );
    let root_header = object_header::build_v2_object_header(&root).unwrap();
    let mut block = located.superblock;
    block.root_group_address = append_checked(&mut bytes, widths, addresses, &root_header);
    block.eof_address = u64::try_from(bytes.len()).unwrap();
    let encoded = hdf5_pure_format::__private::serialize_superblock(&block).unwrap();
    bytes[512..512 + encoded.len()].copy_from_slice(&encoded);
    std::fs::write(&path, &bytes).unwrap();
    assert_eq!(
        dataset::read_libhdf5_strings(&path, "values"),
        ["alpha", "", "ζ"]
    );
    {
        let native = file::libhdf5_open(&path);
        let dataset = native.dataset("values").unwrap();
        assert_eq!(attr::unicode_array(&dataset, "label"), ["payload"]);
    }
    let reopened = crate::File::from_bytes(bytes).unwrap();
    assert_eq!(
        reopened.dataset("values").unwrap().read_string().unwrap(),
        ["alpha", "", "ζ"]
    );
    reopened.close().unwrap();
    directory.close().unwrap();
}

#[rstest]
#[case::empty(IndexCase::Empty)]
#[case::fixed(IndexCase::Fixed)]
#[case::single(IndexCase::Single)]
fn finite_staged_commits_preserve_native_values(
    #[case] index: IndexCase,
    #[values(false, true)] filtered: bool,
    #[values(false, true)] paged: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("commit.h5");
    let widths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    };
    if paged {
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
        );
        let dataset = builder.create_dataset("values");
        dataset
            .with_u32_data(&(1..=u32::try_from(index.initial()).unwrap()).collect::<Vec<_>>())
            .with_chunks(&[4])
            .with_maxshape(&[MaxExtent::Fixed(
                u64::try_from(index.maximum().unwrap()).unwrap(),
            )]);
        if filtered {
            dataset.with_deflate(DeflateLevel::One);
        }
        builder.write(&path).unwrap();
    } else {
        create_native_dataset(&path, widths, 0, index, filtered);
    }
    assert!(index.matches(stored_index(&path)));
    {
        let native = file::libhdf5_open(&path);
        assert_eq!(
            native.dataset("values").unwrap().read_raw::<u32>().unwrap(),
            (1..=u32::try_from(index.initial()).unwrap()).collect::<Vec<_>>()
        );
    }
    let edited = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(backing),
    )
    .unwrap();
    let mut target = edited.dataset("values").unwrap();
    target
        .append_staged(|append| {
            append.append(
                &(u32::try_from(index.initial() + 1).unwrap()
                    ..=u32::try_from(index.grown()).unwrap())
                    .collect::<Vec<_>>(),
            );
        })
        .unwrap();
    drop(target);
    edited.commit().unwrap();
    assert_eq!(
        edited.dataset("values").unwrap().read_u32().unwrap(),
        (1..=u32::try_from(index.grown()).unwrap()).collect::<Vec<_>>()
    );
    edited.close().unwrap();
    assert!(index.matches(stored_index(&path)));
    let located = superblock::read(&BytesSource::new(std::fs::read(&path).unwrap())).unwrap();
    assert_eq!(located.widths, widths);
    {
        let native = file::libhdf5_open(&path);
        let dataset = native.dataset("values").unwrap();
        assert_eq!(dataset.shape(), [index.grown()]);
        assert_eq!(
            dataset.read_raw::<u32>().unwrap(),
            (1..=u32::try_from(index.grown()).unwrap()).collect::<Vec<_>>()
        );
    }
    directory.close().unwrap();
}

fn stored_index(path: &Path) -> ChunkIndexLayout {
    let bytes = std::fs::read(path).unwrap();
    let source = BytesSource::new(bytes.as_slice());
    let located = superblock::read(&source).unwrap();
    let addresses = AddressContext::from_effective_base(located.superblock.base_address);
    let root = WriteEngine::gather_oh_messages(
        &source,
        addresses
            .resolve(located.superblock.root_group_address)
            .unwrap()
            .get(),
        addresses,
        located.widths,
    )
    .unwrap();
    let LinkTarget::Hard {
        object_header_address,
    } = root_link(&root).link_target
    else {
        panic!("the dataset must have a hard link");
    };
    let region = WriteEngine::gather_oh_messages(
        &source,
        addresses.resolve(object_header_address).unwrap().get(),
        addresses,
        located.widths,
    )
    .unwrap();
    let (_, layout) = dataset_layout(&region);
    let DataLayout::Chunked { index, .. } = layout else {
        panic!("the dataset must retain chunks");
    };
    index
}
