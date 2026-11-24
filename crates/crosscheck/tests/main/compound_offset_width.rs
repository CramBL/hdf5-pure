#![cfg(feature = "libhdf5")]

use hdf5_pure::CharacterSet;
use hdf5_pure::CompoundTypeBuilder;
use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FixedPointType;
use hdf5_pure::LibVer;
use hdf5_pure::OffsetWidth;
use hdf5_pure::StringPadding;
use hdf5_pure::VariableLengthType;
use libhdf5::plist::file_access::LibraryVersion;
use libhdf5::types::CompoundField;
use libhdf5::types::CompoundType;
use libhdf5::types::IntSize;
use libhdf5::types::TypeDescriptor;
use rstest::rstest;
use test_util_hdf5::lock;

#[rstest]
#[case::two_bytes(OffsetWidth::Two, 10)]
#[case::four_bytes(OffsetWidth::Four, 12)]
fn libhdf5_reads_committed_compounds_with_narrow_file_offsets(
    #[case] width: OffsetWidth,
    #[case] tag_offset: u64,
) {
    let mut datatype = CompoundTypeBuilder::with_size(17)
        .field(
            "v",
            0,
            Datatype::VariableLength(VariableLengthType::string(
                StringPadding::NullTerminate,
                CharacterSet::Ascii,
            )),
        )
        .u8_field("tag", 16)
        .build()
        .unwrap();
    let Datatype::Compound { members, .. } = &mut datatype else {
        panic!("expected a compound datatype, got {datatype:?}");
    };
    members[1].byte_offset = tag_offset;
    let mut builder = FileBuilder::new();
    builder.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    builder.with_offset_width(width);
    builder.commit_datatype("record", datatype.clone());
    builder
        .create_dataset("value")
        .with_unallocated_storage(datatype, &[3])
        .with_committed_datatype("record");
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("committed.h5");
    builder.write(&path).unwrap();

    let _guard = lock::libhdf5_guard();
    let file = libhdf5::File::open(&path).unwrap();
    {
        let dataset = file.dataset("value").unwrap();
        let datatype = dataset.dtype().unwrap();
        assert!(datatype.is_committed());
        assert_eq!(dataset.shape(), [3]);
        assert_eq!(dataset.storage_size(), 0);
        let string_size = size_of::<*const core::ffi::c_char>();
        assert_eq!(
            datatype.to_descriptor().unwrap(),
            TypeDescriptor::Compound(CompoundType {
                size: 17 - usize::try_from(tag_offset).unwrap() + string_size,
                fields: vec![
                    CompoundField::new("v", TypeDescriptor::VarLenAscii, 0, 0),
                    CompoundField::new(
                        "tag",
                        TypeDescriptor::Unsigned(IntSize::U1),
                        string_size,
                        1
                    ),
                ],
            })
        );
    }
    file.close().unwrap();
}

#[rstest]
#[case::below_two_byte_boundary(255)]
#[case::two_byte_boundary(256)]
#[case::below_three_byte_boundary(65_535)]
#[case::three_byte_boundary(65_536)]
#[case::interior_three_byte_size(70_000)]
#[case::below_four_byte_boundary(16_777_215)]
#[case::four_byte_boundary(16_777_216)]
fn c_library_reads_compound_offset_widths(#[case] size: u32) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pure_compound.h5");
    let mut builder = FileBuilder::new();
    builder.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    builder
        .create_dataset("compound")
        .with_raw_data(
            compound_type(size, DatatypeByteOrder::LittleEndian),
            Vec::new(),
            0,
        )
        .with_shape(&[0]);
    builder.write(&path).unwrap();

    let file = libhdf5::File::open(&path).unwrap();
    let dataset = file.dataset("compound").unwrap();
    assert_eq!(dataset.shape(), vec![0]);
    assert_eq!(
        dataset.dtype().unwrap().to_descriptor().unwrap(),
        compound_descriptor(usize::try_from(size).unwrap()),
    );
}

#[rstest]
#[case::below_two_byte_boundary(255)]
#[case::two_byte_boundary(256)]
#[case::below_three_byte_boundary(65_535)]
#[case::three_byte_boundary(65_536)]
#[case::interior_three_byte_size(70_000)]
#[case::below_four_byte_boundary(16_777_215)]
#[case::four_byte_boundary(16_777_216)]
fn pure_reader_reads_c_compound_offset_widths(#[case] size: u32) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_compound.h5");
    {
        let file = libhdf5::File::with_options()
            .with_fapl(|properties| {
                properties.libver_bounds(LibraryVersion::V18, LibraryVersion::V18)
            })
            .create(&path)
            .unwrap();
        let datatype = libhdf5::Datatype::from_descriptor(&compound_descriptor(
            usize::try_from(size).unwrap(),
        ))
        .unwrap();
        file.new_dataset_builder()
            .empty_as(&datatype)
            .shape([0])
            .create("compound")
            .unwrap();
    }

    let file = File::open(&path).unwrap();
    let dataset = file.dataset("compound").unwrap();
    assert_eq!(dataset.shape().unwrap(), vec![0]);
    assert_eq!(
        dataset.datatype().unwrap(),
        compound_type(
            size,
            if cfg!(target_endian = "big") {
                DatatypeByteOrder::BigEndian
            } else {
                DatatypeByteOrder::LittleEndian
            },
        ),
    );
}

fn compound_type(size: u32, byte_order: DatatypeByteOrder) -> Datatype {
    let member = Datatype::FixedPoint(FixedPointType::new(1, byte_order, false, 0, 8).unwrap());
    CompoundTypeBuilder::with_size(size)
        .field("first", 0, member.clone())
        .field("last", u64::from(size - 1), member)
        .build()
        .unwrap()
}

fn compound_descriptor(size: usize) -> TypeDescriptor {
    TypeDescriptor::Compound(CompoundType {
        fields: vec![
            CompoundField::new("first", TypeDescriptor::Unsigned(IntSize::U1), 0, 0),
            CompoundField::new("last", TypeDescriptor::Unsigned(IntSize::U1), size - 1, 1),
        ],
        size,
    })
}
