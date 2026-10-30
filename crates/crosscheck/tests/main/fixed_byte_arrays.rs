#![cfg(feature = "hdf5")]

use hdf5::types::{IntSize, TypeDescriptor};
use hdf5_pure::{Datatype, File, FileBuilder};
use rstest::rstest;

#[rstest]
#[case::width_one(vec![[0], [255], [128]])]
#[case::width_three(vec![[0, 255, 128], [1, 2, 3], [4, 5, 6], [7, 8, 9]])]
#[case::empty(Vec::<[u8; 7]>::new())]
fn libhdf5_reads_typed_byte_arrays<const N: usize>(#[case] values: Vec<[u8; N]>) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("values.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("values")
        .with_data(&values)
        .with_chunks(&[3]);
    builder.write(&path).unwrap();
    let file = hdf5::File::open(&path).unwrap();
    let dataset = file.dataset("values").unwrap();
    assert_eq!(dataset.shape(), [values.len()]);
    assert_eq!(
        dataset.dtype().unwrap().to_descriptor().unwrap(),
        TypeDescriptor::FixedArray(Box::new(TypeDescriptor::Unsigned(IntSize::U1)), N)
    );
    assert_eq!(dataset.read_raw::<[u8; N]>().unwrap(), values);
}

#[rstest]
#[case::width_one(vec![[0], [255], [128]])]
#[case::width_three(vec![[0, 255, 128], [1, 2, 3], [4, 5, 6], [7, 8, 9]])]
#[case::empty(Vec::<[u8; 7]>::new())]
fn typed_byte_arrays_read_libhdf5_files<const N: usize>(#[case] values: Vec<[u8; N]>) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("values.h5");
    {
        let file = hdf5::File::create(&path).unwrap();
        file.new_dataset::<[u8; N]>()
            .shape([values.len()])
            .create("values")
            .unwrap()
            .write_raw(&values)
            .unwrap();
    }
    let file = File::open(&path).unwrap();
    let dataset = file.dataset("values").unwrap();
    let Datatype::Array {
        base_type,
        dimensions,
    } = dataset.datatype().unwrap()
    else {
        panic!("expected an array datatype");
    };
    assert_eq!(dimensions, [N as u32]);
    let Datatype::FixedPoint(base) = *base_type else {
        panic!("expected a fixed-point base datatype");
    };
    assert_eq!(base.size().get(), 1);
    assert!(!base.layout().signed());
    assert_eq!(base.layout().bit_offset(), 0);
    assert_eq!(base.layout().bit_precision(), 8);
    assert_eq!(dataset.read::<[u8; N]>().unwrap(), values);
}
