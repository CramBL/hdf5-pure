use hdf5_pure::Datatype;
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
#[case::offset_two(OffsetWidth::Two, LengthWidth::Eight)]
#[case::length_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::both_four(OffsetWidth::Four, LengthWidth::Four)]
fn creation_preflight_preserves_a_destination_for_unsupported_reopen_widths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(
        MemoryStrategy::Auto,
        MemoryStrategy::Mirrored,
        MemoryStrategy::Bounded
    )]
    memory: MemoryStrategy,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("preserved.h5");
    let original = b"existing destination";
    std::fs::write(&path, original).unwrap();
    let error = File::create_with_options(
        &path,
        FileCreateProperties::new()
            .with_offset_width(offsets)
            .with_length_width(lengths),
        FileAccessProperties::new().with_memory_strategy(memory),
    )
    .unwrap_err();
    let Error::EditUnsupported(reason) = error else {
        panic!("expected EditUnsupported, got {error:?}");
    };
    assert_eq!(reason, "editing requires 8-byte address and length widths");
    assert_eq!(std::fs::read(&path).unwrap(), original);
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
