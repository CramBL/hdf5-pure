use hdf5_pure::Datatype;
use hdf5_pure::EditBacking;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileCreateProperties;
use hdf5_pure::FileSpacePageSize;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FormatError;
use hdf5_pure::LengthWidth;
use hdf5_pure::MaxExtent;
use hdf5_pure::MemoryStrategy;
use hdf5_pure::OffsetWidth;
use hdf5_pure::ReferenceType;
use hdf5_pure::UserblockSize;
use rstest::rstest;

#[derive(Clone, Copy, Debug)]
enum Selection {
    FreshProperties,
    LengthAfterProperties,
    OffsetAfterProperties,
    ReversedSetters,
    Setters,
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
fn creation_width_properties_compose_and_replace_independently(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(
        Selection::FreshProperties,
        Selection::LengthAfterProperties,
        Selection::OffsetAfterProperties,
        Selection::ReversedSetters,
        Selection::Setters
    )]
    selection: Selection,
) {
    let properties = FileCreateProperties::new()
        .with_offset_width(offsets)
        .with_length_width(lengths);
    assert_eq!(properties.offset_width(), offsets);
    assert_eq!(properties.length_width(), lengths);
    assert_eq!(
        properties,
        FileCreateProperties::new()
            .with_length_width(lengths)
            .with_offset_width(offsets)
    );
    let mut builder = FileBuilder::new();
    let expected = match selection {
        Selection::FreshProperties => {
            builder
                .with_create_properties(properties)
                .with_create_properties(FileCreateProperties::new());
            (8, 8)
        }
        Selection::LengthAfterProperties => {
            builder
                .with_create_properties(properties)
                .with_length_width(LengthWidth::Eight);
            (offsets.get(), 8)
        }
        Selection::OffsetAfterProperties => {
            builder
                .with_create_properties(properties)
                .with_offset_width(OffsetWidth::Eight);
            (8, lengths.get())
        }
        Selection::ReversedSetters => {
            builder
                .with_length_width(lengths)
                .with_offset_width(offsets);
            (offsets.get(), lengths.get())
        }
        Selection::Setters => {
            builder
                .with_offset_width(offsets)
                .with_length_width(lengths);
            (offsets.get(), lengths.get())
        }
    };
    let bytes = builder.finish().unwrap();
    assert_eq!(&bytes[9..11], &[expected.0, expected.1]);
    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(file.superblock().offset_size.get(), expected.0);
    assert_eq!(file.superblock().length_size.get(), expected.1);
}

#[test]
fn creation_widths_default_to_eight_bytes() {
    let properties = FileCreateProperties::new();
    assert_eq!(properties.offset_width(), OffsetWidth::Eight);
    assert_eq!(properties.length_width(), LengthWidth::Eight);
    assert_eq!(&FileBuilder::new().finish().unwrap()[9..11], &[8, 8]);
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
fn creation_returns_an_editable_handle_at_independent_widths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(
        MemoryStrategy::Auto,
        MemoryStrategy::Mirrored,
        MemoryStrategy::Bounded
    )]
    memory: MemoryStrategy,
    #[values(
        FileSpaceStrategy::None,
        FileSpaceStrategy::Aggr,
        FileSpaceStrategy::FsmAggr,
        FileSpaceStrategy::Page
    )]
    strategy: FileSpaceStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("editable.h5");
    let file = File::create_with_options(
        &path,
        FileCreateProperties::new()
            .with_offset_width(offsets)
            .with_length_width(lengths)
            .with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(matches!(
                strategy,
                FileSpaceStrategy::FsmAggr | FileSpaceStrategy::Page
            ))),
        FileAccessProperties::new().with_memory_strategy(memory),
    )
    .unwrap();
    file.root()
        .create_dataset("d", |dataset| {
            dataset.with_i32_data(&[11, 13, 17]);
        })
        .unwrap();
    file.root()
        .create_dataset("empty", |dataset| {
            dataset.with_i32_data(&[]).with_shape(&[0]);
        })
        .unwrap();
    file.commit().unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [11, 13, 17]);
    assert_eq!(
        file.dataset("empty").unwrap().read_i32().unwrap(),
        Vec::<i32>::new()
    );
    file.close().unwrap();
    assert_eq!(
        &std::fs::read(&path).unwrap()[9..11],
        &[offsets.get(), lengths.get()]
    );
    let reopened = File::open(&path).unwrap();
    assert_eq!(
        reopened.dataset("d").unwrap().read_i32().unwrap(),
        [11, 13, 17]
    );
    reopened.close().unwrap();
}

#[rstest]
#[case::offset_two(OffsetWidth::Two, LengthWidth::Eight)]
#[case::length_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::both_four(OffsetWidth::Four, LengthWidth::Four)]
fn creation_preflight_preserves_a_destination_for_unsupported_access(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(AccessConflict::NonpersistentPage, AccessConflict::Userblock)]
    conflict: AccessConflict,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("preserved.h5");
    let original = b"existing destination";
    std::fs::write(&path, original).unwrap();
    let properties = FileCreateProperties::new()
        .with_offset_width(offsets)
        .with_length_width(lengths);
    let (properties, expected) = match conflict {
        AccessConflict::NonpersistentPage => (
            properties.with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(false),
            ),
            "bounded editing of PAGE files without persisted free space is unsupported; use MemoryStrategy::Auto or MemoryStrategy::Mirrored for staged edits",
        ),
        AccessConflict::Userblock => (
            properties.with_userblock(UserblockSize::try_from(512).unwrap()),
            "a userblock cannot be combined with MemoryStrategy::Bounded: the bounded engine cannot edit a file with a non-zero base address, so creating one this way would write the file and then refuse to open it; drop the userblock, or leave MemoryStrategy unset to mirror this file",
        ),
    };
    let error = File::create_with_options(
        &path,
        properties,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded),
    )
    .unwrap_err();
    let Error::EditUnsupported(reason) = error else {
        panic!("expected EditUnsupported, got {error:?}")
    };
    assert_eq!(reason, expected);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    let file = File::create_with_options(
        &path,
        properties,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Auto),
    )
    .unwrap();
    assert_eq!(file.edit_backing(), Some(EditBacking::Mirrored));
    file.root()
        .create_dataset("d", |dataset| {
            dataset.with_i32_data(&[11, 13]);
        })
        .unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    let reopened = File::open(&path).unwrap();
    assert_eq!(reopened.superblock().offset_size, offsets);
    assert_eq!(reopened.superblock().length_size, lengths);
    assert_eq!(reopened.dataset("d").unwrap().read_i32().unwrap(), [11, 13]);
    reopened.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum AccessConflict {
    NonpersistentPage,
    Userblock,
}

#[rstest]
#[case::dimension(LengthFailure::Dimension)]
#[case::maximum(LengthFailure::Maximum)]
#[case::unlimited(LengthFailure::Unlimited)]
fn staged_length_failures_preserve_published_content(
    #[case] failure: LengthFailure,
    #[values(LengthWidth::Two, LengthWidth::Four)] lengths: LengthWidth,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] memory: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("length-boundary.h5");
    let beyond = (u64::MAX >> (64 - 8 * lengths.get())) + 1;
    let (dimension, maximum, length) = match failure {
        LengthFailure::Dimension => (beyond, None, beyond),
        LengthFailure::Maximum => (1, Some(MaxExtent::Fixed(beyond)), beyond),
        LengthFailure::Unlimited => (1, Some(MaxExtent::Unlimited), u64::MAX),
    };
    let file = File::create_with_options(
        &path,
        FileCreateProperties::new()
            .with_offset_width(offsets)
            .with_length_width(lengths)
            .with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::None)),
        FileAccessProperties::new().with_memory_strategy(memory),
    )
    .unwrap();
    file.root()
        .create_dataset("kept", |dataset| {
            dataset.with_i32_data(&[11, 13]);
        })
        .unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    let before = std::fs::read(&path).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(memory),
    )
    .unwrap();
    let error = file
        .root()
        .create_dataset("oversized", |dataset| {
            dataset.with_unallocated_storage(hdf5_pure::make_u8_type(), &[dimension]);
            if let Some(maximum) = maximum {
                dataset.with_maxshape(&[maximum]);
            }
        })
        .unwrap_err();
    assert!(
        matches!(error, Error::Format(FormatError::LengthTooLarge { length: actual, length_size }) if actual == length && length_size == lengths.get()),
        "{error:?}"
    );
    assert!(!file.has_staged_edits());
    assert_eq!(file.dataset("kept").unwrap().read_i32().unwrap(), [11, 13]);
    file.close().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let reopened = File::open(&path).unwrap();
    assert_eq!(reopened.root().datasets().unwrap(), ["kept"]);
    assert_eq!(
        reopened.dataset("kept").unwrap().read_i32().unwrap(),
        [11, 13]
    );
    reopened.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum LengthFailure {
    Dimension,
    Maximum,
    Unlimited,
}

#[rstest]
#[case::two(LengthWidth::Two)]
#[case::eight(LengthWidth::Eight)]
fn oversized_commit_reservations_preserve_the_published_root(
    #[case] lengths: LengthWidth,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] memory: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("address-boundary.h5");
    let mut builder = FileBuilder::new();
    builder
        .with_offset_width(OffsetWidth::Two)
        .with_length_width(lengths)
        .with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::None));
    builder.create_dataset("kept").with_u8_data(&[11, 13]);
    builder.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(memory),
    )
    .unwrap();
    let root = file.superblock().root_group_address;
    file.root()
        .create_dataset("oversized", |dataset| {
            dataset.with_u8_data(&vec![17; 65_535]);
        })
        .unwrap();
    let error = file.commit().unwrap_err();
    assert!(
        matches!(error, Error::Format(FormatError::AddressTooLarge { address, offset_size: 2 }) if address == u64::try_from(before.len()).unwrap() + 65_535),
        "{error:?}"
    );
    assert_eq!(file.superblock().root_group_address, root);
    assert_eq!(file.dataset("kept").unwrap().read_u8().unwrap(), [11, 13]);
    drop(file);
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let reopened = File::open(&path).unwrap();
    assert_eq!(reopened.root().datasets().unwrap(), ["kept"]);
    assert_eq!(
        reopened.dataset("kept").unwrap().read_u8().unwrap(),
        [11, 13]
    );
    reopened.close().unwrap();
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
fn mismatched_cross_file_copies_preserve_the_destination(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(WidthAxis::Offsets, WidthAxis::Lengths)] axis: WidthAxis,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] memory: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("destination.h5");
    let destination = File::create_with_options(
        &path,
        FileCreateProperties::new()
            .with_offset_width(offsets)
            .with_length_width(lengths)
            .with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::None)),
        FileAccessProperties::new().with_memory_strategy(memory),
    )
    .unwrap();
    destination
        .root()
        .create_dataset("kept", |dataset| {
            dataset.with_u8_data(&[11, 13]);
        })
        .unwrap();
    destination.commit().unwrap();
    destination.close().unwrap();
    let before = std::fs::read(&path).unwrap();
    let destination = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(memory),
    )
    .unwrap();
    let mut source = FileBuilder::new();
    source.with_offset_width(offsets).with_length_width(lengths);
    match axis {
        WidthAxis::Offsets => {
            source.with_offset_width(if offsets == OffsetWidth::Eight {
                OffsetWidth::Two
            } else {
                OffsetWidth::Eight
            });
        }
        WidthAxis::Lengths => {
            source.with_length_width(if lengths == LengthWidth::Eight {
                LengthWidth::Two
            } else {
                LengthWidth::Eight
            });
        }
    }
    source.create_dataset("source").with_u8_data(&[17, 19]);
    let source_bytes = source.finish().unwrap();
    let source = File::from_bytes(source_bytes.clone()).unwrap();
    let error = destination
        .copy_from(&source, "source", "copied")
        .unwrap_err();
    assert!(
        matches!(
            error,
            Error::EditUnsupported("cross-file copy requires matching address and length widths")
        ),
        "{error:?}"
    );
    assert!(!destination.has_staged_edits());
    assert_eq!(
        destination.dataset("kept").unwrap().read_u8().unwrap(),
        [11, 13]
    );
    assert_eq!(source.as_bytes(), source_bytes);
    source.close().unwrap();
    destination.close().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[derive(Clone, Copy, Debug)]
enum WidthAxis {
    Lengths,
    Offsets,
}

#[rstest]
#[case::userblock(true, FormatError::AddressTooLarge { address: 65_560, offset_size: 2 })]
#[case::page(false, FormatError::LengthTooLarge { length: 65_536, length_size: 2 })]
fn unrepresentable_creation_settings_are_rejected_before_sink_output(
    #[case] userblock: bool,
    #[case] expected: FormatError,
) {
    let mut builder = FileBuilder::new();
    if userblock {
        builder
            .with_offset_width(OffsetWidth::Two)
            .with_userblock(UserblockSize::try_from(65_536).unwrap());
    } else {
        builder
            .with_length_width(LengthWidth::Two)
            .with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::Page)
                    .with_page_size(FileSpacePageSize::try_from(65_536).unwrap()),
            );
    }
    let mut output = Vec::new();
    let error = builder.finish_to(&mut output).unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected Format, got {error:?}");
    };
    assert_eq!(error, expected);
    assert!(output.is_empty());
}

#[derive(Clone, Copy, Debug)]
enum ReferenceStorage {
    Committed,
    Nested,
    Paths,
    Raw,
    Region,
}

#[rstest]
#[case::two(OffsetWidth::Two)]
#[case::four(OffsetWidth::Four)]
fn narrow_reference_creation_is_rejected_before_sink_output(
    #[case] offsets: OffsetWidth,
    #[values(
        ReferenceStorage::Committed,
        ReferenceStorage::Nested,
        ReferenceStorage::Paths,
        ReferenceStorage::Raw,
        ReferenceStorage::Region
    )]
    storage: ReferenceStorage,
) {
    let mut builder = FileBuilder::new();
    builder.with_offset_width(offsets);
    let datatype = Datatype::Reference {
        size: 8,
        ref_type: ReferenceType::Object,
    };
    match storage {
        ReferenceStorage::Committed => {
            builder.commit_datatype("reference", datatype);
        }
        ReferenceStorage::Nested => {
            builder.create_dataset("references").with_raw_data(
                Datatype::Array {
                    base_type: Box::new(datatype),
                    dimensions: vec![1],
                },
                vec![0; 8],
                1,
            );
        }
        ReferenceStorage::Paths => {
            builder.create_dataset("target").with_i32_data(&[7]);
            builder
                .create_dataset("references")
                .with_path_references(&["target"]);
        }
        ReferenceStorage::Raw => {
            builder
                .create_dataset("references")
                .with_reference_data(&[0]);
        }
        ReferenceStorage::Region => {
            builder.create_dataset("references").with_raw_data(
                Datatype::Reference {
                    size: 12,
                    ref_type: ReferenceType::DatasetRegion,
                },
                vec![0; 12],
                1,
            );
        }
    }
    let mut output = Vec::new();
    let error = builder.finish_to(&mut output).unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected Format, got {error:?}");
    };
    assert_eq!(
        error,
        FormatError::ReferenceWidthUnsupported {
            offset_size: offsets.get()
        }
    );
    assert!(output.is_empty());
}
