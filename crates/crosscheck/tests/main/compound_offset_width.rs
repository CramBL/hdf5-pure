#![cfg(feature = "hdf5")]

use hdf5::plist::file_access::LibraryVersion;
use hdf5::types::CompoundField;
use hdf5::types::CompoundType;
use hdf5::types::IntSize;
use hdf5::types::TypeDescriptor;
use hdf5_pure::CompoundTypeBuilder;
use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FixedPointLayout;
use hdf5_pure::LibVer;
use rstest::rstest;

#[rstest]
#[case(255)]
#[case(256)]
#[case(65_535)]
#[case(65_536)]
#[case(70_000)]
#[case(16_777_215)]
#[case(16_777_216)]
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

    let file = hdf5::File::open(&path).unwrap();
    let dataset = file.dataset("compound").unwrap();
    assert_eq!(dataset.shape(), vec![0]);
    assert_eq!(
        dataset.dtype().unwrap().to_descriptor().unwrap(),
        compound_descriptor(usize::try_from(size).unwrap()),
    );
}

#[rstest]
#[case(255)]
#[case(256)]
#[case(65_535)]
#[case(65_536)]
#[case(70_000)]
#[case(16_777_215)]
#[case(16_777_216)]
fn pure_reader_reads_c_compound_offset_widths(#[case] size: u32) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_compound.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|properties| {
                properties.libver_bounds(LibraryVersion::V18, LibraryVersion::V18)
            })
            .create(&path)
            .unwrap();
        let datatype =
            hdf5::Datatype::from_descriptor(&compound_descriptor(usize::try_from(size).unwrap()))
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
    let member = Datatype::FixedPoint {
        size: 1,
        byte_order,
        layout: FixedPointLayout {
            signed: false,
            bit_offset: 0,
            bit_precision: 8,
        },
    };
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
