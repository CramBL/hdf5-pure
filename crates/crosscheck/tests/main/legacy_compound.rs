#![cfg(feature = "hdf5")]

use hdf5::types::CompoundField;
use hdf5::types::CompoundType;
use hdf5::types::IntSize;
use hdf5::types::TypeDescriptor;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::File;
use test_util::temp;
use test_util_hdf5::legacy_compound;

#[test]
fn libhdf5_reads_compound_v1_array_members() {
    let path = temp::temp_path("native_compound_v1_array.h5");
    std::fs::write(&path, test_util::legacy_compound::file()).unwrap();
    let native = hdf5::File::open(&path).unwrap();
    let dataset = native.dataset("data").unwrap();
    assert_eq!(
        dataset.dtype().unwrap().to_descriptor().unwrap(),
        descriptor()
    );
    assert_eq!(
        dataset.read_raw::<Record>().unwrap(),
        [Record {
            arr: [11, 22, 33, 44]
        }]
    );
    drop(dataset);
    drop(native);
}

#[test]
fn pure_reader_reads_libhdf5_compound_array_members() {
    let path = temp::temp_path("native_written_compound_array.h5");
    {
        let native = hdf5::File::create(&path).unwrap();
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

fn descriptor() -> TypeDescriptor {
    TypeDescriptor::Compound(CompoundType {
        fields: vec![CompoundField::new(
            "arr",
            TypeDescriptor::FixedArray(Box::new(TypeDescriptor::Integer(IntSize::U4)), 4),
            0,
            0,
        )],
        size: 16,
    })
}

#[derive(hdf5::H5Type, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct Record {
    arr: [i32; 4],
}
