#![cfg(feature = "libhdf5")]

use std::fmt::Debug;

use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::File;
use hdf5_pure_core::FixedPointType;
use hdf5_pure_format::__private::{
    CompoundV1Member, CompoundV1Shape, DatatypeMessage, DatatypeMessageMember, DatatypeProperties,
    DatatypeVersion, OffsetWidth,
};
use libhdf5::H5Type;
use test_util::temp;
use test_util_hdf5::legacy_compound;

#[test]
fn libhdf5_reads_compound_v1_array_members() {
    assert_native_record(
        test_util::legacy_compound::file(),
        Record {
            arr: [11, 22, 33, 44],
        },
    );
}

#[test]
fn libhdf5_reads_encoded_version_one_compound_inline_array_members() {
    let bytes = version_one_compound()
        .serialize(OffsetWidth::Eight)
        .unwrap();
    assert_native_record(
        test_util::legacy_compound::file_with_datatype(bytes),
        Record {
            arr: [11, 22, 33, 44],
        },
    );
}

#[test]
fn libhdf5_reads_encoded_version_two_compound_members() {
    let description = DatatypeMessage::new(
        DatatypeVersion::Two,
        DatatypeProperties::Compound {
            size: 16,
            members: vec![DatatypeMessageMember::new(
                c"record".into(),
                0,
                version_one_compound(),
            )],
        },
    )
    .unwrap();
    let bytes = description.serialize(OffsetWidth::Eight).unwrap();
    assert_native_record(
        test_util::legacy_compound::file_with_datatype(bytes),
        NestedRecord {
            record: Record {
                arr: [11, 22, 33, 44],
            },
        },
    );
}

#[test]
fn pure_reader_reads_libhdf5_compound_array_members() {
    let path = temp::temp_path("native_written_compound_array.h5");
    {
        let native = libhdf5::File::create(&path).unwrap();
        native
            .new_dataset_builder()
            .with_data(&[Record {
                arr: [11, 22, 33, 44],
            }])
            .create("data")
            .unwrap();
    }
    let file = File::open(&path).unwrap();
    legacy_compound::assert_contents(
        &file,
        if cfg!(target_endian = "big") {
            DatatypeByteOrder::BigEndian
        } else {
            DatatypeByteOrder::LittleEndian
        },
    );
    drop(file);
}

fn assert_native_record<T: H5Type + Debug + PartialEq>(image: Vec<u8>, expected: T) {
    let path = temp::temp_path("native_compound.h5");
    std::fs::write(&path, image).unwrap();
    let native = libhdf5::File::open(&path).unwrap();
    let dataset = native.dataset("data").unwrap();
    assert_eq!(
        dataset.dtype().unwrap().to_descriptor().unwrap(),
        T::type_descriptor(),
    );
    assert_eq!(dataset.read_raw::<T>().unwrap(), [expected]);
    drop(dataset);
    drop(native);
}

fn version_one_compound() -> DatatypeMessage {
    let integer = DatatypeMessage::new(
        DatatypeVersion::One,
        DatatypeProperties::FixedPoint(FixedPointType::i32()),
    )
    .unwrap();
    DatatypeMessage::new(
        DatatypeVersion::One,
        DatatypeProperties::CompoundV1 {
            size: 16,
            members: vec![CompoundV1Member::new(
                DatatypeMessageMember::new(c"arr".into(), 0, integer),
                CompoundV1Shape::new(1, [4, 0, 0, 0]).unwrap(),
            )],
        },
    )
    .unwrap()
}

#[derive(libhdf5::H5Type, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct Record {
    arr: [i32; 4],
}

#[derive(libhdf5::H5Type, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct NestedRecord {
    record: Record,
}
