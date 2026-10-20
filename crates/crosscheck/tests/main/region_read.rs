#![cfg(feature = "hdf5")]

use std::path::Path;

use hdf5::dataset::Layout;
use ndarray::IxDyn;
use rstest::rstest;
use test_util_hdf5::file;
use test_util_hdf5::reader::ReadBackend;
use test_util_hdf5::region::{self, StringKind};

#[rstest]
#[case::matrix(&[4, 5], &[1, 1], &[2, 3], &[6, 7, 8, 11, 12, 13])]
#[case::volume(&[4, 5, 6], &[1, 1, 2], &[2, 3, 3], &[38, 39, 40, 44, 45, 46, 50, 51, 52, 68, 69, 70, 74, 75, 76, 80, 81, 82])]
fn numeric_regions_match_reference_selections(
    #[case] shape: &[usize],
    #[case] start: &[usize],
    #[case] count: &[usize],
    #[case] expected: &[u8],
    #[values(region::write_numeric::<i8>, region::write_numeric::<i16>, region::write_numeric::<i32>, region::write_numeric::<i64>, region::write_numeric::<u8>, region::write_numeric::<u16>, region::write_numeric::<u32>, region::write_numeric::<u64>, region::write_numeric::<f32>, region::write_numeric::<f64>)]
    create: fn(&Path, Layout, &[usize]),
    #[values(Layout::Compact, Layout::Contiguous, Layout::Chunked)] layout: Layout,
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("regions.h5");
    create(&path, layout, shape);
    let reference_dataset = hdf5::File::open(&path).unwrap().dataset("data").unwrap();
    let expected_bytes = region::numeric_bytes(&reference_dataset.dtype().unwrap(), expected);
    let reference = reference_dataset
        .read_slice::<f64, _, IxDyn>(region::selection(start, count))
        .unwrap()
        .into_raw_vec_and_offset()
        .0;
    let expected: Vec<_> = expected.iter().map(|&value| f64::from(value)).collect();
    assert_eq!(reference, expected);
    let dataset = backend.open(&path).unwrap().dataset("data").unwrap();
    let start: Vec<_> = start.iter().map(|&coordinate| coordinate as u64).collect();
    let count: Vec<_> = count.iter().map(|&length| length as u64).collect();
    assert_eq!(
        dataset.read_raw_region(&start, &count).unwrap(),
        expected_bytes
    );
    assert_eq!(dataset.read_f64_region(&start, &count).unwrap(), expected);
    assert_eq!(dataset.read_f64_region(&start, &count).unwrap(), expected);
}

#[rstest]
#[case::matrix(&[4, 5], &[1, 1], &[2, 3], &[6, 7, 8, 11, 12, 13])]
#[case::volume(&[4, 5, 6], &[1, 1, 2], &[2, 3, 3], &[38, 39, 40, 44, 45, 46, 50, 51, 52, 68, 69, 70, 74, 75, 76, 80, 81, 82])]
fn string_regions_match_reference_selections(
    #[case] shape: &[usize],
    #[case] start: &[usize],
    #[case] count: &[usize],
    #[case] expected: &[u8],
    #[values(
        StringKind::FixedAscii,
        StringKind::FixedUnicode,
        StringKind::VariableAscii,
        StringKind::VariableUnicode
    )]
    kind: StringKind,
    #[values(Layout::Compact, Layout::Contiguous, Layout::Chunked)] layout: Layout,
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("strings.h5");
    let reference = kind.generate(&path, layout, shape, start, count);
    let expected: Vec<_> = expected.iter().map(|&index| kind.label(index)).collect();
    assert_eq!(reference, expected);
    let start: Vec<_> = start.iter().map(|&coordinate| coordinate as u64).collect();
    let count: Vec<_> = count.iter().map(|&length| length as u64).collect();
    let dataset = backend.open(&path).unwrap().dataset("data").unwrap();
    assert_eq!(
        dataset.read_string_region(&start, &count).unwrap(),
        expected
    );
    assert_eq!(
        dataset.read_string_region(&start, &count).unwrap(),
        expected
    );
}

#[rstest]
fn regions_read_fill_and_allocated_intersections(
    #[values(Layout::Contiguous, Layout::Chunked)] layout: Layout,
    #[values(false, true)] allocated: bool,
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fill.h5");
    {
        let reference = file::libhdf5_create_v18(&path);
        let builder = reference.new_dataset::<i32>().fill_value(17);
        let builder = if layout == Layout::Chunked {
            builder.chunk([2, 3, 4]).deflate(4)
        } else {
            builder.layout(layout)
        };
        let dataset = builder.shape([4, 5, 6]).create("data").unwrap();
        assert_eq!(dataset.storage_size(), 0);
        if allocated {
            dataset
                .write_slice(
                    &ndarray::Array3::from_elem((2, 3, 4), 23),
                    region::selection(&[0, 0, 0], &[2, 3, 4]),
                )
                .unwrap();
        }
    }
    let expected = if allocated {
        vec![
            23, 23, 17, 23, 23, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
        ]
    } else {
        vec![17; 18]
    };
    let reference = hdf5::File::open(&path)
        .unwrap()
        .dataset("data")
        .unwrap()
        .read_slice::<i32, _, IxDyn>(region::selection(&[1, 1, 2], &[2, 3, 3]))
        .unwrap()
        .into_raw_vec_and_offset()
        .0;
    assert_eq!(reference, expected);
    assert_eq!(
        backend
            .open(&path)
            .unwrap()
            .dataset("data")
            .unwrap()
            .read_i32_region(&[1, 1, 2], &[2, 3, 3])
            .unwrap(),
        expected
    );
}
