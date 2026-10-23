#![cfg(feature = "hdf5")]

use hdf5::Datatype as CDatatype;
use hdf5_pure::BitFieldType;
use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FixedPointType;
use hdf5_pure::FloatingPointType;
use hdf5_pure::LibVer;
use hdf5_sys::h5p::H5P_DEFAULT;
use hdf5_sys::h5t;
use rstest::rstest;
use test_util_hdf5::numeric;
use test_util_hdf5::numeric::NumericClass;

#[rstest]
#[case(NumericClass::FixedPoint, false)]
#[case(NumericClass::FloatingPoint, false)]
#[case(NumericClass::FloatingPoint, true)]
#[case(NumericClass::BitField, false)]
fn preserves_c_padding_through_nested_types(
    #[case] class: NumericClass,
    #[case] internal: bool,
    #[values(false, true)] low: bool,
    #[values(false, true)] high: bool,
    #[values(false, true)] big_endian: bool,
    #[values(0, 1, 2)] depth: usize,
) {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.h5");
    let destination = dir.path().join("destination.h5");
    let datatype = (0..depth).fold(class.datatype(), |element_type, _| {
        hdf5::sync::sync(|| {
            // SAFETY: The lock serializes access and `element_type` owns a live datatype.
            let id = unsafe { h5t::H5Tvlen_create(element_type.id()) };
            assert!(id >= 0);
            // SAFETY: The returned handle owns the newly created datatype ID.
            unsafe { hdf5::from_id::<CDatatype>(id) }.unwrap()
        })
    });
    hdf5::sync::sync(|| {
        // SAFETY: The lock and live transient datatype cover the setters. The padding values
        // and byte order are defined for the numeric parent reached through its VL wrappers.
        unsafe {
            assert_eq!(
                h5t::H5Tset_pad(
                    datatype.id(),
                    numeric::bit_padding(low),
                    numeric::bit_padding(high)
                ),
                0
            );
            assert_eq!(
                h5t::H5Tset_order(datatype.id(), numeric::byte_order(big_endian)),
                0
            );
            if matches!(class, NumericClass::FloatingPoint) {
                assert_eq!(
                    h5t::H5Tset_inpad(datatype.id(), numeric::bit_padding(internal)),
                    0
                );
            }
        }
    });
    {
        let file = hdf5::File::create(&source).unwrap();
        file.new_dataset_builder()
            .empty_as(&datatype)
            .shape(0)
            .create("values")
            .unwrap();
    }
    let actual = File::open(&source)
        .unwrap()
        .dataset("values")
        .unwrap()
        .datatype()
        .unwrap();
    let expected = (0..depth).fold(
        padded_datatype(
            class,
            if big_endian {
                DatatypeByteOrder::BigEndian
            } else {
                DatatypeByteOrder::LittleEndian
            },
            low,
            high,
            internal,
        ),
        |base, _| Datatype::VariableLength {
            is_string: false,
            padding: None,
            charset: None,
            base_type: Box::new(base),
        },
    );
    assert_eq!(actual, expected);
    let mut builder = FileBuilder::new();
    builder.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    builder.commit_datatype("padded", actual);
    builder.write(&destination).unwrap();
    let file = hdf5::File::open(&destination).unwrap();
    hdf5::sync::sync(|| {
        // SAFETY: The file and lock cover the call, and the datatype name is NUL-terminated.
        let id = unsafe { h5t::H5Topen2(file.id(), c"padded".as_ptr(), H5P_DEFAULT) };
        assert!(id >= 0);
        // SAFETY: The returned handle owns the newly opened datatype ID.
        let written = unsafe { hdf5::from_id::<CDatatype>(id) }.unwrap();
        // SAFETY: Both handles remain live and the lock serializes access.
        assert_eq!(unsafe { h5t::H5Tequal(datatype.id(), written.id()) }, 1);
    });
}

fn padded_datatype(
    class: NumericClass,
    byte_order: DatatypeByteOrder,
    low: bool,
    high: bool,
    internal: bool,
) -> Datatype {
    match class {
        NumericClass::FixedPoint => Datatype::FixedPoint(
            FixedPointType::new(2, byte_order, false, 0, 16)
                .unwrap()
                .with_padding(low, high),
        ),
        NumericClass::FloatingPoint => Datatype::FloatingPoint(
            FloatingPointType::ieee754_binary32(byte_order).with_padding(low, high, internal),
        ),
        NumericClass::BitField => Datatype::BitField(
            BitFieldType::new(2, byte_order, 0, 16)
                .unwrap()
                .with_padding(low, high),
        ),
    }
}
