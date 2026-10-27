#![cfg(feature = "hdf5")]

use hdf5::Datatype;
use hdf5::dataset::Layout;
use hdf5_pure::Error;
use hdf5_pure::FormatError;
use hdf5_sys::h5t;
use rstest::rstest;
use test_util::temp;
use test_util_hdf5::file;
use test_util_hdf5::numeric;
use test_util_hdf5::reader::ReadBackend;
use test_util_hdf5::region;

macro_rules! exact_reads {
    ($name:ident, $ty:ty, $whole:ident, $rows:ident, $region:ident) => {
        #[rstest]
        fn $name(
            #[values(Layout::Compact, Layout::Contiguous, Layout::Chunked)] layout: Layout,
            #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
        ) {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("primitives.h5");
            region::write_numeric::<$ty>(&path, layout, &[4, 5]);
            let expected: Vec<$ty> = (0u8..20).map(|value| value as $ty).collect();
            let bytes = |values: &[$ty]| {
                values
                    .iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect::<Vec<_>>()
            };
            let file = backend.open(&path).unwrap();
            let dataset = file.dataset("data").unwrap();
            assert_eq!(bytes(&dataset.$whole().unwrap()), bytes(&expected));
            assert_eq!(
                bytes(&dataset.$rows(1, 2).unwrap()),
                bytes(&expected[5..15])
            );
            let selected = [
                expected[6],
                expected[7],
                expected[8],
                expected[11],
                expected[12],
                expected[13],
            ];
            assert_eq!(
                bytes(&dataset.$region(&[1, 1], &[2, 3]).unwrap()),
                bytes(&selected)
            );
            assert_eq!(
                bytes(&dataset.$region(&[1, 1], &[2, 3]).unwrap()),
                bytes(&selected)
            );
        }
    };
}

exact_reads!(reads_i8, i8, read_i8, read_i8_rows, read_i8_region);
exact_reads!(reads_i16, i16, read_i16, read_i16_rows, read_i16_region);
exact_reads!(reads_i32, i32, read_i32, read_i32_rows, read_i32_region);
exact_reads!(reads_i64, i64, read_i64, read_i64_rows, read_i64_region);
exact_reads!(reads_u8, u8, read_u8, read_u8_rows, read_u8_region);
exact_reads!(reads_u16, u16, read_u16, read_u16_rows, read_u16_region);
exact_reads!(reads_u32, u32, read_u32, read_u32_rows, read_u32_region);
exact_reads!(reads_u64, u64, read_u64, read_u64_rows, read_u64_region);
exact_reads!(reads_f32, f32, read_f32, read_f32_rows, read_f32_region);
exact_reads!(reads_f64, f64, read_f64, read_f64_rows, read_f64_region);

macro_rules! windowed_reads {
    ($name:ident, $ty:ty, $whole:ident, $rows:ident, $region:ident) => {
        #[rstest]
        fn $name(
            #[values(false, true)] filtered: bool,
            #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
        ) {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("windows.h5");
            let values: Vec<$ty> = (0..8_195 * 33).map(|value| (value % 113) as $ty).collect();
            {
                let reference = file::libhdf5_create_v18(&path);
                let builder = reference.new_dataset::<$ty>();
                let builder = if filtered {
                    builder.chunk([257, 33]).shuffle().deflate(4).fletcher32()
                } else {
                    builder
                };
                builder
                    .shape([8_195, 33])
                    .create("data")
                    .unwrap()
                    .write_raw(&values)
                    .unwrap();
            }
            let bytes = |values: &[$ty]| {
                values
                    .iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect::<Vec<_>>()
            };
            let file = backend.open(&path).unwrap();
            let dataset = file.dataset("data").unwrap();
            assert_eq!(bytes(&dataset.$whole().unwrap()), bytes(&values));
            assert_eq!(
                bytes(&dataset.$rows(7_931, 264).unwrap()),
                bytes(&values[7_931 * 33..])
            );
            assert_eq!(
                bytes(&dataset.$region(&[7_931, 0], &[264, 33]).unwrap()),
                bytes(&values[7_931 * 33..])
            );
        }
    };
}

windowed_reads!(
    reads_i32_windows,
    i32,
    read_i32,
    read_i32_rows,
    read_i32_region
);
windowed_reads!(
    reads_f64_windows,
    f64,
    read_f64,
    read_f64_rows,
    read_f64_region
);

#[rstest]
fn reads_i32_enumeration(
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("enumeration.h5");
    let values = [i32::MIN, 0, i32::MAX];
    let datatype = i32_enumeration();
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    numeric::write_stored(&path, &datatype, &bytes, values.len());
    let file = backend.open(&path).unwrap();
    let dataset = file.dataset("values").unwrap();
    assert_eq!(dataset.read_i32().unwrap(), values);
    assert_eq!(dataset.read_i32_rows(1, 2).unwrap(), values[1..]);
    assert_eq!(dataset.read_i32_region(&[1], &[2]).unwrap(), values[1..]);
}

#[rstest]
#[case::fixed_point(Datatype::from_type::<i16>().unwrap(), false)]
#[case::offset(Datatype::from_type::<i16>().unwrap(), true)]
#[case::float32(Datatype::from_type::<f32>().unwrap(), false)]
#[case::float64(Datatype::from_type::<f64>().unwrap(), false)]
fn converts_fill_across_numeric_readers(
    #[case] datatype: Datatype,
    #[case] offset: bool,
    #[values(false, true)] big_endian: bool,
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let path = temp::temp_path("fill_windows.h5");
    let count = 600_003;
    let status = hdf5::sync::sync(|| {
        // SAFETY: The library lock and live transient numeric datatype cover the calls.
        // The optional twelve-bit range fits within the sixteen-bit integer representation.
        unsafe {
            if offset {
                assert_eq!(h5t::H5Tset_precision(datatype.id(), 12), 0);
                assert_eq!(h5t::H5Tset_offset(datatype.id(), 4), 0);
            }
            h5t::H5Tset_order(datatype.id(), numeric::byte_order(big_endian))
        }
    });
    assert_eq!(status, 0);
    {
        let reference = file::libhdf5_create_v18(&path);
        let dataset = reference
            .new_dataset_builder()
            .empty_as(&datatype)
            .fill_value(42u16)
            .shape([count])
            .create("values")
            .unwrap();
        assert_eq!(dataset.read_raw::<u16>().unwrap(), vec![42; count]);
    }
    let file = backend.open(&path).unwrap();
    let dataset = file.dataset("values").unwrap();
    macro_rules! assert_conversion {
        ($ty:ty, $whole:ident, $rows:ident, $region:ident, $expected:expr) => {
            let expected: $ty = $expected;
            assert_eq!(dataset.fill_value::<$ty>().unwrap(), Some(expected));
            assert_eq!(dataset.$whole().unwrap(), vec![expected; count]);
            assert_eq!(dataset.$rows(599_999, 4).unwrap(), [expected; 4]);
            assert_eq!(dataset.$region(&[599_999], &[4]).unwrap(), [expected; 4]);
        };
    }
    assert_conversion!(i8, read_i8, read_i8_rows, read_i8_region, 42);
    assert_conversion!(i16, read_i16, read_i16_rows, read_i16_region, 42);
    assert_conversion!(i32, read_i32, read_i32_rows, read_i32_region, 42);
    assert_conversion!(i64, read_i64, read_i64_rows, read_i64_region, 42);
    assert_conversion!(u8, read_u8, read_u8_rows, read_u8_region, 42);
    assert_conversion!(u16, read_u16, read_u16_rows, read_u16_region, 42);
    assert_conversion!(u32, read_u32, read_u32_rows, read_u32_region, 42);
    assert_conversion!(u64, read_u64, read_u64_rows, read_u64_region, 42);
    assert_conversion!(f32, read_f32, read_f32_rows, read_f32_region, 42.0);
    assert_conversion!(f64, read_f64, read_f64_rows, read_f64_region, 42.0);
}

#[rstest]
fn rejects_empty_nonnumeric_reads(
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
    #[values(
        NumericSelection::Whole,
        NumericSelection::Rows,
        NumericSelection::Region
    )]
    selection: NumericSelection,
) {
    let path = temp::temp_path("empty.h5");
    {
        let reference = file::libhdf5_create_v18(&path);
        reference
            .new_dataset::<hdf5::types::FixedAscii<4>>()
            .shape([0])
            .create("values")
            .unwrap();
    }
    let file = backend.open(&path).unwrap();
    let dataset = file.dataset("values").unwrap();
    macro_rules! assert_type_mismatch {
        ($whole:ident, $rows:ident, $region:ident) => {
            let error = match selection {
                NumericSelection::Whole => dataset.$whole().unwrap_err(),
                NumericSelection::Rows => dataset.$rows(0, 0).unwrap_err(),
                NumericSelection::Region => dataset.$region(&[0], &[0]).unwrap_err(),
            };
            let Error::Format(FormatError::TypeMismatch { expected, actual }) = error else {
                panic!("expected TypeMismatch, got {error:?}");
            };
            assert_eq!(expected, "numeric");
            assert_eq!(actual, "String");
        };
    }
    assert_type_mismatch!(read_i8, read_i8_rows, read_i8_region);
    assert_type_mismatch!(read_u8, read_u8_rows, read_u8_region);
    assert_type_mismatch!(read_f64, read_f64_rows, read_f64_region);
}

#[derive(Clone, Copy)]
enum NumericSelection {
    Region,
    Rows,
    Whole,
}

fn i32_enumeration() -> Datatype {
    let base = Datatype::from_type::<i32>().unwrap();
    hdf5::sync::sync(|| {
        // SAFETY: The library lock serializes access and `base` owns a live integer datatype.
        let id = unsafe { h5t::H5Tenum_create(base.id()) };
        assert!(id >= 0);
        // SAFETY: The returned handle owns the newly created enumeration datatype.
        let datatype = unsafe { hdf5::from_id::<Datatype>(id) }.unwrap();
        for (name, value) in [(c"min", i32::MIN), (c"zero", 0), (c"max", i32::MAX)] {
            // SAFETY: The lock and live handle cover the call, and `value` has the enumeration's
            // native `i32` representation. Each name is a distinct, null-terminated string.
            let status = unsafe {
                h5t::H5Tenum_insert(
                    datatype.id(),
                    name.as_ptr(),
                    core::ptr::from_ref(&value).cast(),
                )
            };
            assert_eq!(status, 0);
        }
        datatype
    })
}
