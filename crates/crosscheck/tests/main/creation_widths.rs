#![cfg(feature = "__hdf5-1.10")]

use hdf5::ObjectReference;
use hdf5::ObjectReference1;
use hdf5::ReferencedObject;
use hdf5::plist::file_create::Sizeof;
use hdf5::plist::file_create::SizeofInfo;
use hdf5::types::VarLenUnicode;
use hdf5_pure::AttrValue;
use hdf5_pure::ChunkIndex;
use hdf5_pure::DatasetBuilder;
use hdf5_pure::DeflateLevel;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileCreateProperties;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FormatError;
use hdf5_pure::Layout as DiskLayout;
use hdf5_pure::LengthWidth;
use hdf5_pure::MaxExtent;
use hdf5_pure::OffsetWidth;
use hdf5_pure::RepackOptions;
use hdf5_pure::UserblockSize;
use hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT;
use hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT_DIRECT;
use hdf5_pure_format::__private::BTreeV2Header;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::FractalHeapHeader;
use rstest::rstest;
use test_util_hdf5::object_copy;

#[derive(Clone, Copy, Debug)]
enum SupportedStorage {
    Bounded {
        lengths: LengthWidth,
        layout: BoundedLayout,
    },
    Extensible(ExtensibleLayout),
}

impl SupportedStorage {
    fn lengths(self) -> LengthWidth {
        match self {
            Self::Bounded { lengths, .. } => lengths,
            Self::Extensible(_) => LengthWidth::Eight,
        }
    }

    fn layout(self) -> Layout {
        match self {
            Self::Bounded { layout, .. } => layout.into(),
            Self::Extensible(layout) => layout.into(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Layout {
    Contiguous,
    Extensible,
    FilteredExtensible,
    FilteredFixed,
    FilteredSingle,
    Fixed,
    Single,
}

impl From<BoundedLayout> for Layout {
    fn from(layout: BoundedLayout) -> Self {
        match layout {
            BoundedLayout::Contiguous => Self::Contiguous,
            BoundedLayout::FilteredFixed => Self::FilteredFixed,
            BoundedLayout::FilteredSingle => Self::FilteredSingle,
            BoundedLayout::Fixed => Self::Fixed,
            BoundedLayout::Single => Self::Single,
        }
    }
}

impl From<ExtensibleLayout> for Layout {
    fn from(layout: ExtensibleLayout) -> Self {
        match layout {
            ExtensibleLayout::Extensible => Self::Extensible,
            ExtensibleLayout::FilteredExtensible => Self::FilteredExtensible,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum BoundedLayout {
    Contiguous,
    FilteredFixed,
    FilteredSingle,
    Fixed,
    Single,
}

#[derive(Clone, Copy, Debug)]
enum ExtensibleLayout {
    Extensible,
    FilteredExtensible,
}

#[derive(Clone, Copy, Debug)]
enum Space {
    Default,
    FsmAggr,
    Page,
}

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two, Sizeof::Bytes2, Sizeof::Bytes2)]
#[case::two_four(OffsetWidth::Two, LengthWidth::Four, Sizeof::Bytes2, Sizeof::Bytes4)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight, Sizeof::Bytes2, Sizeof::Bytes8)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two, Sizeof::Bytes4, Sizeof::Bytes2)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four, Sizeof::Bytes4, Sizeof::Bytes4)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight, Sizeof::Bytes4, Sizeof::Bytes8)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two, Sizeof::Bytes8, Sizeof::Bytes2)]
#[case::eight_four(OffsetWidth::Eight, LengthWidth::Four, Sizeof::Bytes8, Sizeof::Bytes4)]
#[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight, Sizeof::Bytes8, Sizeof::Bytes8)]
fn creates_files_with_independent_widths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[case] sizeof_addr: Sizeof,
    #[case] sizeof_size: Sizeof,
    #[values(
        BoundedLayout::Contiguous,
        BoundedLayout::FilteredFixed,
        BoundedLayout::FilteredSingle,
        BoundedLayout::Fixed,
        BoundedLayout::Single
    )]
    layout: BoundedLayout,
    #[values(Space::Default, Space::FsmAggr, Space::Page)] space: Space,
    #[values(false, true)] dense: bool,
    #[values(false, true)] streamed: bool,
) {
    assert_creates_file(
        offsets,
        SupportedStorage::Bounded { lengths, layout },
        space,
        dense,
        streamed,
        SizeofInfo {
            sizeof_addr,
            sizeof_size,
        },
    );
}

#[rstest]
#[case::two_eight(OffsetWidth::Two, Sizeof::Bytes2)]
#[case::four_eight(OffsetWidth::Four, Sizeof::Bytes4)]
#[case::eight_eight(OffsetWidth::Eight, Sizeof::Bytes8)]
fn creates_extensible_datasets_with_eight_byte_lengths(
    #[case] offsets: OffsetWidth,
    #[case] sizeof_addr: Sizeof,
    #[values(ExtensibleLayout::Extensible, ExtensibleLayout::FilteredExtensible)]
    layout: ExtensibleLayout,
    #[values(Space::Default, Space::FsmAggr, Space::Page)] space: Space,
    #[values(false, true)] dense: bool,
    #[values(false, true)] streamed: bool,
) {
    assert_creates_file(
        offsets,
        SupportedStorage::Extensible(layout),
        space,
        dense,
        streamed,
        SizeofInfo {
            sizeof_addr,
            sizeof_size: Sizeof::Bytes8,
        },
    );
}

#[derive(Clone, Copy, Debug)]
enum Creation {
    Edit,
    Finish,
    FinishTo,
}

#[rstest]
#[case::two_two_one(OffsetWidth::Two, LengthWidth::Two, &[3], &[1024])]
#[case::four_two_one(OffsetWidth::Four, LengthWidth::Two, &[3], &[1024])]
#[case::eight_two_one(OffsetWidth::Eight, LengthWidth::Two, &[3], &[1024])]
#[case::four_four_one(OffsetWidth::Four, LengthWidth::Four, &[3], &[1 << 20])]
#[case::eight_four_one(OffsetWidth::Eight, LengthWidth::Four, &[3], &[1 << 20])]
#[case::four_two_two(OffsetWidth::Four, LengthWidth::Two, &[3, 4], &[512, 512])]
#[case::eight_two_two(OffsetWidth::Eight, LengthWidth::Two, &[3, 4], &[512, 512])]
fn unlimited_maxima_at_narrow_lengths_reopen_with_the_length_width_maximum(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[case] shape: &[u64],
    #[case] chunks: &[u64],
    #[values(Creation::Edit, Creation::Finish, Creation::FinishTo)] creation: Creation,
) {
    let maximum = u64::MAX >> (64 - 8 * lengths.get());
    let stored = vec![MaxExtent::Fixed(maximum); shape.len()];
    let declared = vec![MaxExtent::Unlimited; shape.len()];
    let values: Vec<i32> = (0..i32::try_from(shape.iter().product::<u64>()).unwrap()).collect();
    let configure = |dataset: &mut DatasetBuilder| {
        dataset
            .with_i32_data(&values)
            .with_shape(shape)
            .with_chunks(chunks)
            .with_maxshape(&declared);
    };
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("unlimited.h5");
    match creation {
        Creation::Edit => {
            let file = File::create_with_options(
                &path,
                FileCreateProperties::new()
                    .with_offset_width(offsets)
                    .with_length_width(lengths),
                FileAccessProperties::new(),
            )
            .unwrap();
            file.root().create_dataset("d", configure).unwrap();
            assert_eq!(
                file.dataset("d").unwrap().maxshape().unwrap(),
                Some(stored.clone())
            );
            file.commit().unwrap();
            file.close().unwrap();
        }
        Creation::Finish | Creation::FinishTo => {
            let mut builder = FileBuilder::new();
            builder
                .with_offset_width(offsets)
                .with_length_width(lengths);
            configure(builder.create_dataset("d"));
            let bytes = if matches!(creation, Creation::FinishTo) {
                let mut bytes = Vec::new();
                builder.finish_to(&mut bytes).unwrap();
                bytes
            } else {
                builder.finish().unwrap()
            };
            std::fs::write(&path, bytes).unwrap();
        }
    }
    let file = File::open(&path).unwrap();
    let dataset = file.dataset("d").unwrap();
    assert_eq!(dataset.maxshape().unwrap(), Some(stored));
    assert!(matches!(
        dataset.layout().unwrap(),
        DiskLayout::Chunked {
            index: ChunkIndex::FixedArray,
            ..
        }
    ));
    assert_eq!(dataset.read_i32().unwrap(), values);
    drop(dataset);
    file.close().unwrap();
    let native = hdf5::File::open(&path).unwrap();
    let dataset = native.dataset("d").unwrap();
    assert_eq!(
        dataset.space().unwrap().maxdims(),
        vec![Some(usize::try_from(maximum).unwrap()); shape.len()]
    );
    assert_eq!(
        dataset.chunk(),
        Some(
            chunks
                .iter()
                .map(|&chunk| usize::try_from(chunk).unwrap())
                .collect()
        )
    );
    assert_eq!(dataset.read_raw::<i32>().unwrap(), values);
    drop(dataset);
    native.close().unwrap();
}

fn assert_creates_file(
    offsets: OffsetWidth,
    storage: SupportedStorage,
    space: Space,
    dense: bool,
    streamed: bool,
    native_sizes: SizeofInfo,
) {
    let lengths = storage.lengths();
    let layout = storage.layout();
    let builder = file_builder(offsets, lengths, space, layout, dense);
    let bytes = if streamed {
        let mut bytes = Vec::new();
        builder.finish_to(&mut bytes).unwrap();
        bytes
    } else {
        builder.finish().unwrap()
    };
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("created.h5");
    std::fs::write(&path, &bytes).unwrap();
    assert_eq!(&bytes[9..11], &[offsets.get(), lengths.get()]);
    let file = File::open(&path).unwrap();
    assert_eq!(file.superblock().offset_size, offsets);
    assert_eq!(file.superblock().length_size, lengths);
    assert_eq!(file.superblock().eof_address, bytes.len() as u64);
    let superblock_size = 16 + 4 * usize::from(offsets.get());
    if matches!(space, Space::Page) {
        assert!(bytes[superblock_size..4096].iter().all(|byte| *byte == 0));
        assert_eq!(&bytes[4096..4100], b"GCOL");
    } else if matches!(layout, Layout::Contiguous) {
        assert_eq!(
            file.superblock().root_group_address.get(),
            superblock_size as u64
        );
        assert_eq!(&bytes[superblock_size..superblock_size + 4], b"OHDR");
    } else {
        assert_eq!(&bytes[superblock_size..superblock_size + 4], b"GCOL");
    }
    let actual_layout = file.dataset("numbers").unwrap().layout().unwrap();
    match (layout, actual_layout) {
        (Layout::Contiguous, DiskLayout::Contiguous { size: 12, .. }) => {}
        (
            Layout::Single | Layout::FilteredSingle,
            DiskLayout::Chunked {
                index: ChunkIndex::SingleChunk,
                ..
            },
        ) => {}
        (
            Layout::Fixed | Layout::FilteredFixed,
            DiskLayout::Chunked {
                index: ChunkIndex::FixedArray,
                ..
            },
        ) => {}
        (
            Layout::Extensible | Layout::FilteredExtensible,
            DiskLayout::Chunked {
                index: ChunkIndex::ExtensibleArray,
                ..
            },
        ) => {}
        (expected, actual) => panic!("expected {expected:?}, got {actual:?}"),
    }
    assert_eq!(
        file.dataset("numbers").unwrap().read_i32().unwrap(),
        [3, 5, 7]
    );
    assert_eq!(
        file.dataset("strings").unwrap().read_string().unwrap(),
        ["first", "second"]
    );
    let free = file
        .persisted_free_sections()
        .unwrap()
        .as_ref()
        .map_or(0, |report| report.free_sections().free_bytes());
    assert_eq!(file.dataset("scalar").unwrap().read_i32().unwrap(), [11]);
    assert_eq!(
        file.dataset("empty").unwrap().read_i32().unwrap(),
        Vec::<i32>::new()
    );
    file.close().unwrap();
    let native = hdf5::File::open(&path).unwrap();
    assert_eq!(
        native.create_plist().unwrap().get_sizes().unwrap(),
        native_sizes
    );
    assert_eq!(
        native
            .dataset("numbers")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [3, 5, 7]
    );
    let strings = native
        .dataset("strings")
        .unwrap()
        .read_raw::<VarLenUnicode>()
        .unwrap();
    assert_eq!(
        strings
            .iter()
            .map(VarLenUnicode::as_str)
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
    assert_eq!(
        native
            .dataset("scalar")
            .unwrap()
            .read_scalar::<i32>()
            .unwrap(),
        11
    );
    assert_eq!(
        native.dataset("empty").unwrap().read_raw::<i32>().unwrap(),
        Vec::<i32>::new()
    );
    assert_eq!(
        native
            .attr("title")
            .unwrap()
            .read_raw::<VarLenUnicode>()
            .unwrap()[0]
            .as_str(),
        "label"
    );
    assert_eq!(
        native.info().unwrap().storage_type,
        if dense {
            hdf5::GroupStorageType::Dense
        } else {
            hdf5::GroupStorageType::Compact
        }
    );
    assert!(
        native
            .dataset("numbers")
            .unwrap()
            .dtype()
            .unwrap()
            .is_committed()
    );
    assert_eq!(native.free_space(), free);
    native.close().unwrap();
}

fn file_builder(
    offsets: OffsetWidth,
    lengths: LengthWidth,
    space: Space,
    layout: Layout,
    dense: bool,
) -> FileBuilder {
    let mut builder = FileBuilder::new();
    builder.with_create_properties(
        FileCreateProperties::new()
            .with_offset_width(offsets)
            .with_length_width(lengths),
    );
    match space {
        Space::Default => {}
        Space::FsmAggr => {
            builder.with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
            );
        }
        Space::Page => {
            builder.with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
            );
        }
    }
    builder.commit_datatype("integer_type", hdf5_pure::make_i32_type());
    let numbers = builder.create_dataset("numbers");
    numbers
        .with_i32_data(&[3, 5, 7])
        .with_committed_datatype("integer_type");
    configure_dataset(numbers, layout, 3);
    if dense {
        for index in 0..9 {
            numbers.set_attr(&format!("field_{index}"), AttrValue::I64(index));
        }
    }
    let strings = builder.create_dataset("strings");
    strings.with_vlen_strings(&["first", "second"]);
    configure_dataset(strings, layout, 2);
    builder
        .create_dataset("scalar")
        .with_i32_data(&[11])
        .with_shape(&[]);
    builder
        .create_dataset("empty")
        .with_i32_data(&[])
        .with_shape(&[0]);
    if dense {
        for index in 0..9 {
            builder.set_attr(&format!("field_{index}"), AttrValue::I64(index));
            let group = builder.create_group(&format!("group_{index}"));
            builder.add_group(group.finish());
        }
    }
    builder.set_attr("title", AttrValue::VarLenStringArray(vec!["label".into()]));
    builder
}

fn configure_dataset(builder: &mut DatasetBuilder, layout: Layout, elements: u64) {
    match layout {
        Layout::Contiguous => {}
        Layout::Single | Layout::FilteredSingle => {
            builder.with_chunks(&[elements]);
        }
        Layout::Fixed | Layout::FilteredFixed => {
            builder.with_chunks(&[1]);
        }
        Layout::Extensible | Layout::FilteredExtensible => {
            builder
                .with_chunks(&[1])
                .with_maxshape(&[MaxExtent::Unlimited]);
        }
    }
    if matches!(
        layout,
        Layout::FilteredExtensible | Layout::FilteredFixed | Layout::FilteredSingle
    ) {
        builder.with_deflate(DeflateLevel::Six);
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
fn userblocks_unallocated_storage_and_large_attributes_preserve_widths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(false, true)] paged: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.h5");
    let mut builder = FileBuilder::new();
    let base = if paged { 4096 } else { 512 };
    builder
        .with_offset_width(offsets)
        .with_length_width(lengths)
        .with_userblock(UserblockSize::try_from(base).unwrap())
        .with_userblock_content(b"wrapper");
    if paged {
        builder.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
        );
    }
    builder
        .create_dataset("values")
        .with_i32_data(&[7, 11])
        .with_chunks(&[1])
        .with_maxshape(&[MaxExtent::Fixed(4)]);
    builder
        .create_dataset("unallocated")
        .with_unallocated_storage(hdf5_pure::make_i32_type(), &[2]);
    builder.set_attr("large", AttrValue::U8Array(vec![42; 33_000]));
    builder.write(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..7], b"wrapper");
    assert!(bytes[7..base as usize].iter().all(|&byte| byte == 0));
    assert_eq!(
        &bytes[base as usize..base as usize + 8],
        b"\x89HDF\r\n\x1a\n"
    );
    let pure = File::open(&path).unwrap();
    assert_eq!(pure.superblock().base_address.get(), base);
    assert_eq!(pure.superblock().eof_address, bytes.len() as u64);
    assert_eq!(pure.dataset("values").unwrap().read_i32().unwrap(), [7, 11]);
    assert_eq!(
        pure.dataset("unallocated").unwrap().read_i32().unwrap(),
        [0, 0]
    );
    pure.close().unwrap();
    let native = hdf5::File::open(&path).unwrap();
    assert_eq!(
        native.attr("large").unwrap().read_raw::<u8>().unwrap(),
        vec![42; 33_000]
    );
    assert_eq!(
        native.dataset("values").unwrap().read_raw::<i32>().unwrap(),
        [7, 11]
    );
    assert_eq!(
        native
            .dataset("unallocated")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [0, 0]
    );
    assert_eq!(native.dataset("unallocated").unwrap().storage_size(), 0);
    native.close().unwrap();
    let output = directory.path().join("repacked.h5");
    hdf5_pure::repack(&path, &output, &RepackOptions::new()).unwrap();
    let repacked = File::open(&output).unwrap();
    assert_eq!(repacked.superblock().offset_size, OffsetWidth::Eight);
    assert_eq!(repacked.superblock().length_size, LengthWidth::Eight);
    assert_eq!(
        repacked.dataset("values").unwrap().read_i32().unwrap(),
        [7, 11]
    );
    repacked.close().unwrap();
    let native = hdf5::File::open(&output).unwrap();
    assert_eq!(
        native.attr("large").unwrap().read_raw::<u8>().unwrap(),
        vec![42; 33_000]
    );
    native.close().unwrap();
}

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two, Sizeof::Bytes2, Sizeof::Bytes2)]
#[case::two_four(OffsetWidth::Two, LengthWidth::Four, Sizeof::Bytes2, Sizeof::Bytes4)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight, Sizeof::Bytes2, Sizeof::Bytes8)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two, Sizeof::Bytes4, Sizeof::Bytes2)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four, Sizeof::Bytes4, Sizeof::Bytes4)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight, Sizeof::Bytes4, Sizeof::Bytes8)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two, Sizeof::Bytes8, Sizeof::Bytes2)]
#[case::eight_four(OffsetWidth::Eight, LengthWidth::Four, Sizeof::Bytes8, Sizeof::Bytes4)]
#[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight, Sizeof::Bytes8, Sizeof::Bytes8)]
fn legacy_reference_creation_and_repack_check_the_offset_width(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[case] sizeof_addr: Sizeof,
    #[case] sizeof_size: Sizeof,
) {
    let directory = tempfile::tempdir().unwrap();
    let created = directory.path().join("created.h5");
    let mut builder = FileBuilder::new();
    builder
        .with_offset_width(offsets)
        .with_length_width(lengths);
    builder.create_dataset("target").with_i32_data(&[7]);
    builder
        .create_dataset("references")
        .with_path_references(&["target"]);
    if offsets == OffsetWidth::Eight {
        builder.write(&created).unwrap();
        let file = hdf5::File::open(&created).unwrap();
        let references = file
            .dataset("references")
            .unwrap()
            .read_raw::<ObjectReference1>()
            .unwrap();
        let ReferencedObject::Dataset(target) = references[0].dereference(&file).unwrap() else {
            panic!("expected a dataset reference");
        };
        assert_eq!(target.read_raw::<i32>().unwrap(), [7]);
        drop(target);
        file.close().unwrap();
    } else {
        let error = builder.finish().unwrap_err();
        let Error::Format(error) = error else {
            panic!("expected Format, got {error:?}");
        };
        assert_eq!(
            error,
            FormatError::ReferenceWidthUnsupported {
                offset_size: offsets.get()
            }
        );
    }
    let source = directory.path().join("native.h5");
    let file = hdf5::File::with_options()
        .with_create_plist(|properties| {
            properties.sizes(SizeofInfo {
                sizeof_addr,
                sizeof_size,
            })
        })
        .create(&source)
        .unwrap();
    let target = file
        .new_dataset::<i32>()
        .shape([1])
        .create("target")
        .unwrap();
    target.write_raw(&[11]).unwrap();
    let reference = ObjectReference1::create(&file, "target").unwrap();
    let references = file
        .new_dataset::<ObjectReference1>()
        .shape([1])
        .create("references")
        .unwrap();
    references.write_raw(&[reference]).unwrap();
    let stored_reference_size = object_copy::legacy_reference_storage_size(offsets);
    assert_eq!(references.storage_size(), stored_reference_size);
    assert_eq!(
        file.create_plist().unwrap().get_sizes().unwrap(),
        SizeofInfo {
            sizeof_addr,
            sizeof_size
        }
    );
    drop(references);
    drop(target);
    file.close().unwrap();
    let output = directory.path().join("repacked.h5");
    let result = hdf5_pure::repack(&source, &output, &RepackOptions::new());
    if stored_reference_size != 8 {
        let error = result.unwrap_err();
        let Error::RepackUnsupported(reason) = error else {
            panic!("expected RepackUnsupported, got {error:?}");
        };
        assert_eq!(
            reason,
            "dataset references: non-8-byte object reference datatype cannot be repacked faithfully yet"
        );
        assert!(!output.exists());
        return;
    }
    result.unwrap();
    let file = hdf5::File::open(&output).unwrap();
    assert_eq!(
        file.create_plist().unwrap().get_sizes().unwrap(),
        SizeofInfo {
            sizeof_addr: Sizeof::Bytes8,
            sizeof_size: Sizeof::Bytes8
        }
    );
    let references = file
        .dataset("references")
        .unwrap()
        .read_raw::<ObjectReference1>()
        .unwrap();
    let ReferencedObject::Dataset(target) = references[0].dereference(&file).unwrap() else {
        panic!("expected a dataset reference");
    };
    assert_eq!(target.read_raw::<i32>().unwrap(), [11]);
    drop(target);
    file.close().unwrap();
}

#[rstest]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four, Sizeof::Bytes4, Sizeof::Bytes4)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight, Sizeof::Bytes4, Sizeof::Bytes8)]
#[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight, Sizeof::Bytes8, Sizeof::Bytes8)]
fn dense_variable_length_attributes_resolve_every_collection(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[case] sizeof_addr: Sizeof,
    #[case] sizeof_size: Sizeof,
    #[values(false, true)] populated_tail: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collections.h5");
    let mut values = vec![String::new(); 65_536];
    if populated_tail {
        values.last_mut().unwrap().push_str("tail");
    }
    let mut builder = FileBuilder::new();
    builder
        .with_offset_width(offsets)
        .with_length_width(lengths);
    builder.set_attr("labels", AttrValue::VarLenStringArray(values));
    let bytes = builder.finish().unwrap();
    assert_eq!(test_util::global_heap::collection_starts(&bytes).len(), 2);
    std::fs::write(&path, bytes).unwrap();
    let file = hdf5::File::open(&path).unwrap();
    assert_eq!(
        file.create_plist().unwrap().get_sizes().unwrap(),
        SizeofInfo {
            sizeof_addr,
            sizeof_size
        }
    );
    let labels = file.attr("labels").unwrap();
    assert_eq!(labels.shape(), [65_536]);
    let values = labels.read_raw::<VarLenUnicode>().unwrap();
    assert_eq!(values.len(), 65_536);
    assert!(
        values[..65_535]
            .iter()
            .all(|value| value.as_str().is_empty())
    );
    assert_eq!(
        values.last().unwrap().as_str(),
        if populated_tail { "tail" } else { "" }
    );
    drop(labels);
    file.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum HugeStorage {
    Direct,
    Indexed,
    Unrepresentable,
}

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two, 33_000, 1, HugeStorage::Direct)]
#[case::two_four(
    OffsetWidth::Two,
    LengthWidth::Four,
    70_000,
    1,
    HugeStorage::Unrepresentable
)]
#[case::two_eight(
    OffsetWidth::Two,
    LengthWidth::Eight,
    70_000,
    1,
    HugeStorage::Unrepresentable
)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two, 33_000, 1, HugeStorage::Direct)]
#[case::four_two_multiple(OffsetWidth::Four, LengthWidth::Two, 32_723, 2, HugeStorage::Direct)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four, 70_000, 1, HugeStorage::Indexed)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight, 70_000, 1, HugeStorage::Indexed)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two, 33_000, 1, HugeStorage::Indexed)]
#[case::eight_four(OffsetWidth::Eight, LengthWidth::Four, 70_000, 1, HugeStorage::Indexed)]
#[case::eight_eight(
    OffsetWidth::Eight,
    LengthWidth::Eight,
    70_000,
    1,
    HugeStorage::Indexed
)]
fn dense_huge_attributes_select_the_heap_id_and_tracker_geometry(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[case] payload_size: usize,
    #[case] huge_count: usize,
    #[case] storage: HugeStorage,
    #[values(false, true)] streamed: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("huge.h5");
    let mut builder = FileBuilder::new();
    builder
        .with_offset_width(offsets)
        .with_length_width(lengths);
    for index in 0..8 {
        builder.set_attr(&format!("field_{index}"), AttrValue::I64(index));
    }
    for index in 0..huge_count {
        builder.set_attr(
            &format!("large{index}"),
            AttrValue::U8Array(vec![42; payload_size]),
        );
    }
    let mut bytes = Vec::new();
    let result = if streamed {
        builder.finish_to(&mut bytes)
    } else {
        builder.finish().map(|output| bytes = output)
    };
    if matches!(storage, HugeStorage::Unrepresentable) {
        let error = result.unwrap_err();
        let Error::Format(FormatError::AddressTooLarge {
            address,
            offset_size,
        }) = error
        else {
            panic!("expected AddressTooLarge, got {error:?}");
        };
        assert_eq!(offset_size, 2);
        assert!(address > u64::from(u16::MAX), "{address}");
        assert!(bytes.is_empty());
        return;
    }
    result.unwrap();
    let heaps = bytes
        .windows(test_util::fractal_heap::SIGNATURE.len())
        .enumerate()
        .filter_map(|(position, signature)| {
            (signature == test_util::fractal_heap::SIGNATURE).then_some(position)
        })
        .collect::<Vec<_>>();
    assert_eq!(heaps.len(), 1);
    let heap = FractalHeapHeader::parse_bytes(&bytes[heaps[0]..], offsets, lengths).unwrap();
    assert_eq!(heap.huge_objects_count, huge_count as u64);
    assert!(heap.huge_objects_size > u64::from(heap.max_managed_object_size));
    let direct = matches!(storage, HugeStorage::Direct);
    assert_eq!(
        heap.huge_ids_direct_with_widths(FormatWidths { offsets, lengths }),
        direct
    );
    let tracker = BTreeV2Header::parse(
        &bytes,
        usize::try_from(heap.btree_huge_objects_address.get()).unwrap(),
        offsets.get(),
        lengths.get(),
    )
    .unwrap();
    assert_eq!(
        tracker.tree_type,
        if direct {
            BTREE_V2_HUGE_OBJECT_DIRECT
        } else {
            BTREE_V2_HUGE_OBJECT
        }
    );
    assert_eq!(
        tracker.record_size,
        u16::from(offsets.get()) + u16::from(lengths.get()) * if direct { 1 } else { 2 }
    );
    assert_eq!(tracker.total_records, huge_count as u64);
    std::fs::write(&path, bytes).unwrap();
    let file = hdf5::File::open(&path).unwrap();
    for index in 0..huge_count {
        assert_eq!(
            file.attr(&format!("large{index}"))
                .unwrap()
                .read_raw::<u8>()
                .unwrap(),
            vec![42; payload_size]
        );
    }
    assert_eq!(
        file.attr("field_7").unwrap().read_scalar::<i64>().unwrap(),
        7
    );
    file.close().unwrap();
    if direct && !streamed {
        let output = directory.path().join("repacked.h5");
        hdf5_pure::repack(&path, &output, &RepackOptions::new()).unwrap();
        let file = hdf5::File::open(&output).unwrap();
        assert_eq!(
            file.create_plist().unwrap().get_sizes().unwrap(),
            SizeofInfo {
                sizeof_addr: Sizeof::Bytes8,
                sizeof_size: Sizeof::Bytes8
            }
        );
        for index in 0..huge_count {
            assert_eq!(
                file.attr(&format!("large{index}"))
                    .unwrap()
                    .read_raw::<u8>()
                    .unwrap(),
                vec![42; payload_size]
            );
        }
        file.close().unwrap();
    }
}
