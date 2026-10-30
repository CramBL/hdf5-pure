#![cfg(feature = "hdf5")]

use hdf5::types::{FixedAscii as NativeAscii, TypeDescriptor};
use hdf5_pure::{CharacterSet, Datatype, File, FileBuilder, FixedAscii, LibVer, StringPadding};
use rstest::rstest;

#[rstest]
#[case::width_one_v18(ascii::<1>(&["", "A", "\x7f"]), LibVer::V18)]
#[case::width_four_v18(ascii::<4>(&["ab", "", "wxyz", "a\0b"]), LibVer::V18)]
#[case::empty_v18(Vec::<FixedAscii<7>>::new(), LibVer::V18)]
#[cfg_attr(feature = "__hdf5-1.10", case::width_one_v110(ascii::<1>(&["", "A", "\x7f"]), LibVer::V110))]
#[cfg_attr(feature = "__hdf5-1.10", case::width_four_v110(ascii::<4>(&["ab", "", "wxyz", "a\0b"]), LibVer::V110))]
#[cfg_attr(feature = "__hdf5-1.10", case::empty_v110(Vec::<FixedAscii<7>>::new(), LibVer::V110))]
fn libhdf5_reads_typed_ascii<const N: usize>(
    #[case] values: Vec<FixedAscii<N>>,
    #[case] libver: LibVer,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("values.h5");
    let mut builder = FileBuilder::new();
    builder.with_libver_bounds(LibVer::Earliest, libver);
    let dataset = builder.create_dataset("values").with_data(&values);
    if libver >= LibVer::V110 {
        dataset.with_chunks(&[3]);
    }
    builder.write(&path).unwrap();
    assert_eq!(File::open(&path).unwrap().libver_bound(), libver);
    let file = hdf5::File::open(&path).unwrap();
    let dataset = file.dataset("values").unwrap();
    assert_eq!(dataset.shape(), [values.len()]);
    assert_eq!(dataset.chunk(), (libver >= LibVer::V110).then(|| vec![3]));
    assert_eq!(
        dataset.dtype().unwrap().to_descriptor().unwrap(),
        TypeDescriptor::FixedAscii(N)
    );
    assert_eq!(
        dataset.read_raw::<NativeAscii<N>>().unwrap(),
        values
            .iter()
            .map(|value| NativeAscii::from_ascii(value.as_bytes()).unwrap())
            .collect::<Vec<_>>()
    );
}

#[rstest]
#[case::width_one(ascii::<1>(&["", "A", "\x7f"]))]
#[case::width_four(ascii::<4>(&["ab", "", "wxyz", "a\0b"]))]
#[case::empty(Vec::<FixedAscii<7>>::new())]
fn typed_ascii_reads_libhdf5_files<const N: usize>(#[case] values: Vec<FixedAscii<N>>) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("values.h5");
    {
        let file = hdf5::File::create(&path).unwrap();
        let native: Vec<_> = values
            .iter()
            .map(|value| NativeAscii::<N>::from_ascii(value.as_bytes()).unwrap())
            .collect();
        file.new_dataset::<NativeAscii<N>>()
            .shape([values.len()])
            .create("values")
            .unwrap()
            .write_raw(&native)
            .unwrap();
    }
    let file = File::open(&path).unwrap();
    let dataset = file.dataset("values").unwrap();
    assert_eq!(
        dataset.datatype().unwrap(),
        Datatype::String {
            size: N as u32,
            padding: StringPadding::NullPad,
            charset: CharacterSet::Ascii
        }
    );
    assert_eq!(dataset.read::<FixedAscii<N>>().unwrap(), values);
}

fn ascii<const N: usize>(values: &[&str]) -> Vec<FixedAscii<N>> {
    values
        .iter()
        .map(|value| FixedAscii::new(value).unwrap())
        .collect()
}
