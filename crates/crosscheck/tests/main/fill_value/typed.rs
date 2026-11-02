use std::fmt::Debug;
use std::path::Path;

use hdf5::Datatype as NativeDatatype;
use hdf5::Extent as NativeExtent;
use hdf5::H5Type;
use hdf5::ObjectReference;
use hdf5::ObjectReference1;
use hdf5::plist::dataset_create::{
    DatasetCreate, FillTime as NativeFillTime, FillValue as NativeFillValue,
};
use hdf5::types::{FixedAscii as NativeAscii, IntSize, TypeDescriptor, VarLenUnicode};
use hdf5_pure::{
    CharacterSet, DatasetBuilder, DatasetCreateProperties, Datatype, Error, File, FileBuilder,
    FillTime, FixedAscii, H5Element, MaxExtent, RepackOptions, StringPadding,
};
use hdf5_sys::h5d;
use hdf5_sys::h5p;
use hdf5_sys::h5p::H5P_DEFAULT;
use hdf5_sys::h5s::H5S_ALL;
use hdf5_sys::h5t::{self, H5T_cset_t, H5T_str_t};
use rstest::rstest;
use test_util_hdf5::dataset::AppendMode;
use test_util_hdf5::file;

#[rstest]
#[case::two_bytes([255_u8, 0], [255_u8, 0], vec![[1_u8, 2], [3, 4], [5, 6]], vec![[1_u8, 2], [3, 4], [5, 6]])]
#[case::four_bytes([255_u8, 0, 7, 128], [255_u8, 0, 7, 128], vec![[1_u8, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], vec![[1_u8, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])]
#[case::ascii(
    FixedAscii::<4>::new(b"f\0x").unwrap(), NativeAscii::<4>::from_ascii(b"f\0x\0").unwrap(),
    [b"a\0b\0", b"cd\0\0", b"wxyz"].map(|value| FixedAscii::<4>::new(value).unwrap()).to_vec(),
    [b"a\0b\0", b"cd\0\0", b"wxyz"].map(|value| NativeAscii::<4>::from_ascii(value).unwrap()).to_vec()
)]
fn complete_typed_fills_preserve_metadata_and_unwritten_elements<T, N>(
    #[case] fill: T,
    #[case] native_fill: N,
    #[case] values: Vec<T>,
    #[case] native_values: Vec<N>,
    #[values(
        Schema::Scalar,
        Schema::Contiguous,
        Schema::Chunked,
        Schema::Filtered,
        Schema::Deferred,
        Schema::Allocated,
        Schema::AllocatedFiltered
    )]
    schema: Schema,
    #[values(Creation::Buffered, Creation::Edit, Creation::Streaming)] creation: Creation,
    #[values(false, true)] properties: bool,
    #[values(FillTime::Alloc, FillTime::IfSet, FillTime::Never)] fill_time: FillTime,
) where
    T: H5Element + Debug + PartialEq,
    N: H5Type + Copy,
{
    let directory = tempfile::tempdir().unwrap();
    let pure_path = directory.path().join("pure.h5");
    let native_path = directory.path().join("native.h5");
    let fill_bytes = T::fill_bytes(fill);
    let native_time = match fill_time {
        FillTime::Alloc => NativeFillTime::Alloc,
        FillTime::IfSet => NativeFillTime::IfSet,
        FillTime::Never => NativeFillTime::Never,
    };
    let native = file::libhdf5_create_v110(&native_path);
    let builder = native.new_dataset::<N>();
    let mut builder = if matches!(schema, Schema::Scalar) {
        builder.shape(())
    } else if schema.chunked() {
        builder.shape(NativeExtent::resizable(schema.elements()))
    } else {
        builder.shape(schema.elements())
    };
    builder = builder.fill_value(native_fill).fill_time(native_time);
    if schema.chunked() {
        builder = builder.chunk([4]);
    }
    if schema.filtered() {
        builder = builder.fletcher32();
    }
    let dataset = builder.create("values").unwrap();
    if schema.allocated() {
        dataset.write_raw(&native_values).unwrap();
    }
    drop(dataset);
    drop(builder);
    native.close().unwrap();
    creation.write(&pure_path, |dataset| {
        if schema.allocated() {
            dataset.with_data(&values);
        } else {
            dataset.with_unallocated_storage_for::<T>(schema.dims());
        }
        if schema.chunked() {
            dataset
                .with_chunks(&[4])
                .with_maxshape(&[MaxExtent::Unlimited]);
        }
        if schema.filtered() {
            dataset.with_fletcher32();
        }
        if properties {
            let mut settings = DatasetCreateProperties::new()
                .with_fill_value(fill)
                .with_fill_time(fill_time);
            if schema.chunked() {
                settings = settings.with_chunks(&[4]);
            }
            if schema.filtered() {
                settings = settings.with_fletcher32();
            }
            dataset.with_create_properties(settings);
        } else {
            dataset.with_fill_value(fill).with_fill_time(fill_time);
        }
    });
    let initial_bytes = if schema.allocated() {
        values
            .iter()
            .flat_map(|value| T::fill_bytes(*value))
            .collect::<Vec<_>>()
    } else {
        fill_bytes.repeat(schema.elements())
    };
    assert_native(
        &pure_path,
        &N::type_descriptor(),
        &fill_bytes,
        native_time,
        (fill_time != FillTime::Never).then_some(initial_bytes.as_slice()),
        schema.allocated(),
    );
    assert_native(
        &native_path,
        &N::type_descriptor(),
        &fill_bytes,
        native_time,
        (fill_time != FillTime::Never).then_some(initial_bytes.as_slice()),
        schema.allocated(),
    );
    let pure = File::open(&pure_path).unwrap();
    let dataset = pure.dataset("values").unwrap();
    assert_eq!(dataset.datatype().unwrap(), T::datatype().unwrap());
    assert_eq!(dataset.fill_value::<T>().unwrap(), Some(fill));
    assert_eq!(dataset.fill_time().unwrap(), fill_time);
    if fill_time != FillTime::Never {
        assert_eq!(dataset.read_raw().unwrap(), initial_bytes);
    }
    if !schema.allocated() {
        if schema.chunked() {
            assert!(dataset.chunks().unwrap().is_empty());
        }
    }
    drop(dataset);
    pure.close().unwrap();
    if matches!(schema, Schema::Deferred) {
        crate::unallocated_datasets::assert_deferred_layout(&pure_path, "values");
    }
    if !schema.chunked() || fill_time == FillTime::Never {
        return;
    }
    let before = std::fs::read(&pure_path).unwrap();
    let pure = File::open_rw(&pure_path).unwrap();
    let mut dataset = pure.dataset("values").unwrap();
    AppendMode::Staged
        .append::<T>(&pure, &mut dataset, &[])
        .unwrap();
    drop(dataset);
    pure.close().unwrap();
    assert_eq!(std::fs::read(&pure_path).unwrap(), before);
    if !schema.allocated() && !matches!(schema, Schema::Deferred) {
        extend_native(&pure_path, 8);
        let expected = fill_bytes.repeat(8);
        assert_native(
            &pure_path,
            &N::type_descriptor(),
            &fill_bytes,
            native_time,
            Some(&expected),
            false,
        );
        let pure = File::open(&pure_path).unwrap();
        let dataset = pure.dataset("values").unwrap();
        assert_eq!(dataset.read_raw().unwrap(), expected);
        drop(dataset);
        pure.close().unwrap();
        return;
    }
    let mut expected = initial_bytes;
    let mut initial_elements = schema.elements();
    if schema.allocated() {
        initial_elements = 4;
        extend_native(&pure_path, initial_elements);
        expected.extend(&fill_bytes);
        assert_native(
            &pure_path,
            &N::type_descriptor(),
            &fill_bytes,
            native_time,
            Some(&expected),
            true,
        );
    }
    let pure = File::open_rw(&pure_path).unwrap();
    let mut dataset = pure.dataset("values").unwrap();
    AppendMode::Staged
        .append(&pure, &mut dataset, &values[..2])
        .unwrap();
    drop(dataset);
    pure.close().unwrap();
    expected.extend(values[..2].iter().flat_map(|value| T::fill_bytes(*value)));
    let live_elements = initial_elements + 2;
    let extended_elements = live_elements.next_multiple_of(4);
    expected.extend(fill_bytes.repeat(extended_elements - live_elements));
    extend_native(&pure_path, extended_elements);
    assert_native(
        &pure_path,
        &N::type_descriptor(),
        &fill_bytes,
        native_time,
        Some(&expected),
        true,
    );
    let pure = File::open(&pure_path).unwrap();
    let dataset = pure.dataset("values").unwrap();
    assert_eq!(dataset.read_raw().unwrap(), expected);
    assert_eq!(dataset.fill_value::<T>().unwrap(), Some(fill));
    drop(dataset);
    pure.close().unwrap();
}

#[rstest]
#[case::space_ascii(StringPadding::SpacePad, CharacterSet::Ascii, H5T_str_t::H5T_STR_SPACEPAD, H5T_cset_t::H5T_CSET_ASCII, *b"x   ")]
#[case::terminated_ascii(StringPadding::NullTerminate, CharacterSet::Ascii, H5T_str_t::H5T_STR_NULLTERM, H5T_cset_t::H5T_CSET_ASCII, *b"xy\0\0")]
#[case::nullpad_utf8(StringPadding::NullPad, CharacterSet::Utf8, H5T_str_t::H5T_STR_NULLPAD, H5T_cset_t::H5T_CSET_UTF8, [0xc3, 0xa9, 0, b'x'])]
#[case::space_utf8(StringPadding::SpacePad, CharacterSet::Utf8, H5T_str_t::H5T_STR_SPACEPAD, H5T_cset_t::H5T_CSET_UTF8, [0xc3, 0xa9, b' ', b' '])]
#[case::terminated_utf8(StringPadding::NullTerminate, CharacterSet::Utf8, H5T_str_t::H5T_STR_NULLTERM, H5T_cset_t::H5T_CSET_UTF8, [0xc3, 0xa9, 0, 0])]
fn repack_preserves_source_string_fill_bytes(
    #[case] padding: StringPadding,
    #[case] charset: CharacterSet,
    #[case] native_padding: H5T_str_t,
    #[case] native_charset: H5T_cset_t,
    #[case] fill: [u8; 4],
    #[values(false, true)] chunked: bool,
    #[values(false, true)] allocated: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.h5");
    let destination = directory.path().join("repacked.h5");
    let datatype = NativeDatatype::from_type::<NativeAscii<4>>().unwrap();
    let properties = DatasetCreate::try_new().unwrap();
    hdf5::sync::sync(|| {
        // SAFETY: The library lock and live transient handles cover each call. The datatype is
        // a fixed four-byte string, and `fill` contains one complete element with no pointers.
        unsafe {
            assert_eq!(h5t::H5Tset_strpad(datatype.id(), native_padding), 0);
            assert_eq!(h5t::H5Tset_cset(datatype.id(), native_charset), 0);
            assert_eq!(
                h5p::H5Pset_fill_value(properties.id(), datatype.id(), fill.as_ptr().cast()),
                0
            );
        }
    });
    let native = file::libhdf5_create_v110(&source);
    let mut builder = native
        .new_dataset_builder()
        .set_dcpl(&properties)
        .empty_as(&datatype)
        .shape(3);
    if chunked {
        builder = builder.chunk([2]).fletcher32();
    }
    let dataset = builder.create("values").unwrap();
    let stored = match padding {
        StringPadding::SpacePad => *b"v   ",
        StringPadding::NullPad | StringPadding::NullTerminate => *b"v\0\0\0",
    };
    let expected = if allocated {
        stored.repeat(3)
    } else {
        fill.repeat(3)
    };
    if allocated {
        hdf5::sync::sync(|| {
            // SAFETY: The lock and live handles cover the call. The fixed string datatype has
            // no pointers, and `expected` holds three complete elements matching the extent.
            unsafe {
                assert_eq!(
                    h5d::H5Dwrite(
                        dataset.id(),
                        datatype.id(),
                        H5S_ALL,
                        H5S_ALL,
                        H5P_DEFAULT,
                        expected.as_ptr().cast()
                    ),
                    0
                );
            }
        });
    }
    drop(dataset);
    drop(builder);
    native.close().unwrap();
    hdf5_pure::repack(&source, &destination, &RepackOptions::default()).unwrap();
    let expected_datatype = Datatype::String {
        size: 4,
        padding,
        charset,
    };
    assert_source_string(
        &source,
        &expected_datatype,
        native_padding,
        native_charset,
        &fill,
        &expected,
        allocated,
    );
    assert_source_string(
        &destination,
        &expected_datatype,
        native_padding,
        native_charset,
        &fill,
        &expected,
        allocated,
    );
}

#[rstest]
#[case::object_reference(create_reference_fill)]
#[case::variable_length(create_vlen_fill)]
fn repack_rejects_source_fills_with_file_references(#[case] create: fn(&hdf5::File)) {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.h5");
    let destination = directory.path().join("destination.h5");
    let native = file::libhdf5_create_v110(&source);
    create(&native);
    assert_eq!(
        native
            .dataset("values")
            .unwrap()
            .dcpl()
            .unwrap()
            .get_fill_value_defined()
            .unwrap(),
        NativeFillValue::UserDefined
    );
    native.close().unwrap();
    let error = hdf5_pure::repack(&source, &destination, &RepackOptions::default()).unwrap_err();
    let Error::RepackUnsupported(reason) = error else {
        panic!("expected an unsupported source fill, got {error:?}");
    };
    assert_eq!(
        reason,
        "dataset values: a fill value on a variable-length or object-reference dataset cannot be repacked faithfully"
    );
    assert!(!destination.exists());
}

fn create_reference_fill(native: &hdf5::File) {
    native.create_group("target").unwrap();
    let reference = ObjectReference1::create(native, "target").unwrap();
    let datatype = NativeDatatype::from_type::<ObjectReference1>().unwrap();
    let properties = DatasetCreate::try_new().unwrap();
    hdf5::sync::sync(|| {
        // SAFETY: The library lock and live handles cover the call. The reference's pointer
        // addresses one complete legacy object reference of the matching memory datatype.
        unsafe {
            assert_eq!(
                h5p::H5Pset_fill_value(properties.id(), datatype.id(), reference.ptr()),
                0
            );
        }
    });
    native
        .new_dataset_builder()
        .set_dcpl(&properties)
        .empty_as(&datatype)
        .shape(3)
        .create("values")
        .unwrap();
}

fn create_vlen_fill(native: &hdf5::File) {
    native
        .new_dataset::<VarLenUnicode>()
        .shape(3)
        .fill_value("fill".parse::<VarLenUnicode>().unwrap())
        .create("values")
        .unwrap();
}

#[derive(Clone, Copy)]
enum Creation {
    Buffered,
    Edit,
    Streaming,
}

impl Creation {
    fn write(self, path: &Path, configure: impl FnOnce(&mut DatasetBuilder)) {
        match self {
            Self::Buffered | Self::Streaming => {
                let mut builder = FileBuilder::new();
                configure(builder.create_dataset("values"));
                if matches!(self, Self::Buffered) {
                    std::fs::write(path, builder.finish().unwrap()).unwrap();
                } else {
                    builder.write(path).unwrap();
                }
            }
            Self::Edit => {
                FileBuilder::new().write(path).unwrap();
                let file = File::open_rw(path).unwrap();
                file.root().create_dataset("values", configure).unwrap();
                file.commit().unwrap();
                file.close().unwrap();
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Schema {
    Allocated,
    AllocatedFiltered,
    Chunked,
    Contiguous,
    Deferred,
    Filtered,
    Scalar,
}

impl Schema {
    fn allocated(self) -> bool {
        matches!(self, Self::Allocated | Self::AllocatedFiltered)
    }
    fn filtered(self) -> bool {
        matches!(self, Self::Filtered | Self::AllocatedFiltered)
    }
    fn chunked(self) -> bool {
        !matches!(self, Self::Contiguous | Self::Scalar)
    }
    fn dims(self) -> &'static [u64] {
        match self {
            Self::Scalar => &[],
            Self::Deferred => &[0],
            _ => &[3],
        }
    }
    fn elements(self) -> usize {
        self.dims()
            .iter()
            .map(|value| usize::try_from(*value).unwrap())
            .product()
    }
}

fn assert_source_string(
    path: &Path,
    expected_datatype: &Datatype,
    native_padding: H5T_str_t,
    native_charset: H5T_cset_t,
    fill: &[u8],
    expected: &[u8],
    allocated: bool,
) {
    let native = hdf5::File::open(path).unwrap();
    let dataset = native.dataset("values").unwrap();
    let dtype = dataset.dtype().unwrap();
    hdf5::sync::sync(|| {
        // SAFETY: The library lock serializes access, and `dtype` owns the live string type.
        unsafe {
            assert_eq!(h5t::H5Tget_strpad(dtype.id()), native_padding);
            assert_eq!(h5t::H5Tget_cset(dtype.id()), native_charset);
        }
    });
    assert_eq!(native_fill_bytes(&dataset), fill);
    assert_eq!(native_element_bytes(&dataset), expected);
    assert_eq!(dataset.storage_size() == 0, !allocated);
    drop(dtype);
    drop(dataset);
    native.close().unwrap();
    let pure = File::open(path).unwrap();
    let dataset = pure.dataset("values").unwrap();
    assert_eq!(&dataset.datatype().unwrap(), expected_datatype);
    assert_eq!(dataset.read_raw().unwrap(), expected);
    drop(dataset);
    pure.close().unwrap();
}

fn assert_native(
    path: &Path,
    datatype: &TypeDescriptor,
    fill: &[u8],
    fill_time: NativeFillTime,
    values: Option<&[u8]>,
    allocated: bool,
) {
    let native = hdf5::File::open(path).unwrap();
    let dataset = native.dataset("values").unwrap();
    assert_eq!(dataset.dtype().unwrap().to_descriptor().unwrap(), *datatype);
    assert_eq!(dataset.dcpl().unwrap().get_fill_time().unwrap(), fill_time);
    assert_eq!(native_fill_bytes(&dataset), fill);
    if let Some(values) = values {
        assert_eq!(native_element_bytes(&dataset), values);
    }
    assert_eq!(dataset.storage_size() > 0, allocated);
    if !allocated {
        assert_eq!(dataset.offset(), None);
        if dataset.chunk().is_some() {
            assert_eq!(dataset.num_chunks(), Some(0));
        }
    }
    drop(dataset);
    native.close().unwrap();
}

fn extend_native(path: &Path, elements: usize) {
    let native = hdf5::File::open_rw(path).unwrap();
    native
        .dataset("values")
        .unwrap()
        .resize([elements])
        .unwrap();
    native.close().unwrap();
}

fn native_fill_bytes(dataset: &hdf5::Dataset) -> Vec<u8> {
    let datatype = fixed_datatype(dataset);
    let properties = dataset.dcpl().unwrap();
    let mut bytes = vec![0; datatype.size()];
    hdf5::sync::sync(|| {
        // SAFETY: The lock and live handles cover the call. `bytes` has exactly one complete
        // element of the checked fixed datatype, which has no pointers or variable-length data.
        unsafe {
            assert_eq!(
                h5p::H5Pget_fill_value(properties.id(), datatype.id(), bytes.as_mut_ptr().cast()),
                0
            );
        }
    });
    bytes
}

fn native_element_bytes(dataset: &hdf5::Dataset) -> Vec<u8> {
    let datatype = fixed_datatype(dataset);
    let mut bytes = vec![0; datatype.size().checked_mul(dataset.size()).unwrap()];
    hdf5::sync::sync(|| {
        // SAFETY: The lock and live handles cover the call. The initialized buffer holds the
        // complete dataset in its checked fixed datatype, with no pointers or variable lengths.
        unsafe {
            assert_eq!(
                h5d::H5Dread(
                    dataset.id(),
                    datatype.id(),
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    bytes.as_mut_ptr().cast()
                ),
                0
            );
        }
    });
    bytes
}

fn fixed_datatype(dataset: &hdf5::Dataset) -> NativeDatatype {
    let datatype = dataset.dtype().unwrap();
    match datatype.to_descriptor().unwrap() {
        TypeDescriptor::FixedAscii(_) | TypeDescriptor::FixedUnicode(_) => (),
        TypeDescriptor::FixedArray(base, _) if *base == TypeDescriptor::Unsigned(IntSize::U1) => (),
        actual => panic!("expected a fixed string or byte array, got {actual:?}"),
    }
    datatype
}
