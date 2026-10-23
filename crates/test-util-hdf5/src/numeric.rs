//! Numeric dataset fixtures written through the reference library.

use std::ffi::CStr;
use std::ffi::CString;
use std::path::Path;

use hdf5::Datatype;
use hdf5::globals;
use hdf5::types::TypeDescriptor;
use hdf5_sys::h5d;
use hdf5_sys::h5p::H5P_DEFAULT;
use hdf5_sys::h5s;
use hdf5_sys::h5s::H5S_ALL;
use hdf5_sys::h5t;
use hdf5_sys::h5t::H5T_class_t;
use hdf5_sys::h5t::H5T_order_t;
use hdf5_sys::h5t::H5T_pad_t;

/// Writes `count` elements to a dataset named `values`, using the stored datatype in memory.
///
/// # Panics
///
/// Panics if the byte count differs from the datatype size times `count`, the datatype is not
/// numeric, or the library cannot create the file or write the dataset.
pub fn write_stored(path: &Path, datatype: &Datatype, bytes: &[u8], count: usize) {
    assert_eq!(bytes.len(), datatype.size() * count);
    let file = hdf5::File::create(path).unwrap();
    let dataset = file
        .new_dataset_builder()
        .empty_as(datatype)
        .shape(count)
        .create("values")
        .unwrap();
    let status = hdf5::sync::sync(|| {
        // SAFETY: The lock and live handles cover the calls, and the checked numeric type has
        // no pointer fields. `bytes` holds `count` complete elements, matching the dataset extent.
        unsafe {
            assert!(
                matches!(
                    h5t::H5Tget_class(datatype.id()),
                    H5T_class_t::H5T_INTEGER | H5T_class_t::H5T_FLOAT | H5T_class_t::H5T_ENUM
                ),
                "fixture datatype must be fixed-point, floating-point, or enumeration"
            );
            h5d::H5Dwrite(
                dataset.id(),
                datatype.id(),
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                bytes.as_ptr().cast(),
            )
        }
    });
    assert_eq!(status, 0);
}

/// The byte order of a numeric fixture's stored datatype.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ByteOrder {
    /// The most significant byte is stored first.
    BigEndian,
    /// The least significant byte is stored first.
    LittleEndian,
}

impl ByteOrder {
    /// Returns a signed 32-bit integer datatype with this byte order.
    ///
    /// # Panics
    ///
    /// Panics if the reference library cannot create the datatype or set its byte order.
    pub fn i32_datatype(self) -> Datatype {
        let datatype = Datatype::from_type::<i32>().unwrap();
        let order = match self {
            Self::BigEndian => H5T_order_t::H5T_ORDER_BE,
            Self::LittleEndian => H5T_order_t::H5T_ORDER_LE,
        };
        let status = hdf5::sync::sync(|| {
            // SAFETY: The library lock serializes access and `datatype` owns a live transient
            // integer type. Both orders are valid for its four-byte representation.
            unsafe { h5t::H5Tset_order(datatype.id(), order) }
        });
        assert_eq!(status, 0);
        datatype
    }
}

/// Returns big-endian byte order if `big_endian` is set, and little-endian byte order otherwise.
pub fn byte_order(big_endian: bool) -> H5T_order_t {
    if big_endian {
        H5T_order_t::H5T_ORDER_BE
    } else {
        H5T_order_t::H5T_ORDER_LE
    }
}

/// The numeric datatype classes used for padding fixtures.
#[derive(Clone, Copy)]
pub enum NumericClass {
    /// A 16-bit bitfield.
    BitField,
    /// An unsigned 16-bit integer.
    FixedPoint,
    /// An IEEE 754 binary32 value.
    FloatingPoint,
}

impl NumericClass {
    /// Creates a transient datatype for this fixture class through the reference library.
    ///
    /// # Panics
    ///
    /// Panics if the reference library cannot create the datatype.
    pub fn datatype(self) -> Datatype {
        match self {
            Self::FixedPoint => Datatype::from_type::<u16>().unwrap(),
            Self::FloatingPoint => Datatype::from_type::<f32>().unwrap(),
            Self::BitField => hdf5::sync::sync(|| {
                // SAFETY: The lock serializes access to the predefined bit-field datatype.
                let id = unsafe { h5t::H5Tcopy(*globals::H5T_STD_B16LE) };
                assert!(id >= 0);
                // SAFETY: The returned handle owns the copied datatype ID.
                unsafe { hdf5::from_id::<Datatype>(id) }.unwrap()
            }),
        }
    }
}

/// Returns the reference library's padding code for the selected bit value.
pub fn bit_padding(one: bool) -> H5T_pad_t {
    if one {
        H5T_pad_t::H5T_PAD_ONE
    } else {
        H5T_pad_t::H5T_PAD_ZERO
    }
}

/// Creates a compound datatype with the supplied padding on each numeric member.
///
/// The entries of `padding` select low, high, and internal padding, in that order, with `true`
/// representing one. Internal padding applies to floating-point members.
///
/// # Examples
///
/// ```
/// use hdf5::Datatype;
/// use hdf5::types::CompoundField;
/// use hdf5::types::CompoundType;
/// use hdf5::types::TypeDescriptor;
/// use test_util_hdf5::numeric;
///
/// let datatype = Datatype::from_descriptor(&TypeDescriptor::Compound(CompoundType {
///     fields: vec![
///         CompoundField::typed::<u16>("count", 0, 0),
///         CompoundField::typed::<f32>("value", 4, 1),
///     ],
///     size: 8,
/// }))?;
/// let padded = numeric::padded_compound(&datatype, [true, false, true]);
/// assert_eq!(padded.size(), 8);
/// # Ok::<(), hdf5::Error>(())
/// ```
///
/// # Panics
///
/// Panics if `datatype` is not a compound with numeric members, a member name contains a null
/// byte, or the reference library cannot describe or create a datatype or set its padding.
pub fn padded_compound(datatype: &Datatype, padding: [bool; 3]) -> Datatype {
    let TypeDescriptor::Compound(compound) = datatype.to_descriptor().unwrap() else {
        panic!("expected compound datatype");
    };
    hdf5::sync::sync(|| {
        // SAFETY: The lock serializes access and the descriptor supplies the compound size.
        let id = unsafe { h5t::H5Tcreate(H5T_class_t::H5T_COMPOUND, compound.size) };
        assert!(id >= 0);
        // SAFETY: The returned handle owns the newly created datatype ID.
        let datatype = unsafe { hdf5::from_id::<Datatype>(id) }.unwrap();
        for field in compound.fields {
            let member = Datatype::from_descriptor(&field.ty).unwrap();
            let name = CString::new(field.name).unwrap();
            // SAFETY: The lock and live handles cover the calls. Numeric fields admit zero
            // and one padding, and the descriptor supplies each member's offset and name.
            unsafe {
                assert_eq!(
                    h5t::H5Tset_pad(
                        member.id(),
                        bit_padding(padding[0]),
                        bit_padding(padding[1])
                    ),
                    0
                );
                if h5t::H5Tget_class(member.id()) == H5T_class_t::H5T_FLOAT {
                    assert_eq!(h5t::H5Tset_inpad(member.id(), bit_padding(padding[2])), 0);
                }
                assert_eq!(
                    h5t::H5Tinsert(datatype.id(), name.as_ptr(), field.offset, member.id()),
                    0
                );
            }
        }
        datatype
    })
}

/// Creates a variable-length string datatype with `base` as its parent.
///
/// # Panics
///
/// Panics if the reference library cannot create, encode, or decode the datatype.
///
/// # Examples
///
/// ```
/// use hdf5::Datatype;
/// use test_util_hdf5::numeric;
///
/// let base = Datatype::from_type::<f32>()?;
/// let string = numeric::variable_length_string(&base);
/// assert_eq!(string.size(), core::mem::size_of::<usize>());
/// # Ok::<(), hdf5::Error>(())
/// ```
pub fn variable_length_string(base: &Datatype) -> Datatype {
    hdf5::sync::sync(|| {
        // SAFETY: The lock serializes access and `base` owns a live datatype.
        let id = unsafe { h5t::H5Tvlen_create(base.id()) };
        assert!(id >= 0);
        // SAFETY: The returned handle owns the newly created datatype ID.
        let sequence = unsafe { hdf5::from_id::<Datatype>(id) }.unwrap();
        let mut size = 0;
        assert_eq!(
            // SAFETY: The lock and live handle cover the call. A null buffer requests its encoded size.
            unsafe { h5t::H5Tencode(sequence.id(), core::ptr::null_mut(), &mut size) },
            0
        );
        let mut encoded = vec![0; size];
        assert_eq!(
            // SAFETY: `encoded` has `size` writable bytes. The datatype remains live.
            unsafe { h5t::H5Tencode(sequence.id(), encoded.as_mut_ptr().cast(), &mut size) },
            0
        );
        encoded[ENCODED_DATATYPE_FLAGS_OFFSET] |= VARIABLE_LENGTH_STRING_BIT;
        // SAFETY: The buffer is a complete C-encoded VL datatype. Only its defined subtype bit differs.
        let id = unsafe { h5t::H5Tdecode1(encoded.as_ptr().cast()) };
        assert!(id >= 0);
        // SAFETY: The returned handle owns the newly decoded datatype ID.
        unsafe { hdf5::from_id::<Datatype>(id) }.unwrap()
    })
}

/// Writes null-terminated strings using the dataset's variable-length datatype.
///
/// # Panics
///
/// Panics if the dataset is not a variable-length string dataset, its element count differs
/// from `values.len()`, or the reference library cannot inspect or write it.
///
/// # Examples
///
/// ```
/// # use std::error::Error;
/// use hdf5::Datatype;
/// use hdf5::types::TypeDescriptor;
/// use test_util_hdf5::numeric;
/// # let dir = tempfile::tempdir()?;
/// # let file = hdf5::File::create(dir.path().join("strings.h5"))?;
///
/// let datatype = Datatype::from_descriptor(&TypeDescriptor::VarLenAscii)?;
/// let dataset = file.new_dataset_builder().empty_as(&datatype).shape(1).create("strings")?;
/// numeric::write_variable_length_strings(&dataset, &[c"alpha"]);
/// # Ok::<(), Box<dyn Error>>(())
/// ```
pub fn write_variable_length_strings(dataset: &hdf5::Dataset, values: &[&CStr]) {
    let datatype = dataset.dtype().unwrap();
    let space = dataset.space().unwrap();
    let pointers: Vec<_> = values.iter().map(|value| value.as_ptr()).collect();
    hdf5::sync::sync(|| {
        // SAFETY: The lock serializes access and the handles remain live throughout the calls.
        let count = unsafe {
            assert_eq!(h5t::H5Tis_variable_str(datatype.id()), 1);
            assert_eq!(
                h5t::H5Tget_size(datatype.id()),
                core::mem::size_of::<*const core::ffi::c_char>()
            );
            h5s::H5Sget_simple_extent_npoints(space.id())
        };
        assert_eq!(usize::try_from(count).unwrap(), values.len());
        assert_eq!(
            // SAFETY: The checked VL-string datatype has a pointer-sized memory representation.
            // `pointers` contains one live NUL-terminated string per selected dataset element.
            unsafe {
                h5d::H5Dwrite(
                    dataset.id(),
                    datatype.id(),
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    pointers.as_ptr().cast(),
                )
            },
            0
        );
    });
}

// H5T_encode in H5T.c, libhdf5 1.14.6, prefixes a datatype message with two bytes.
// The first class-specific byte follows the class/version byte in section
// `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`, version 4.0.
const ENCODED_DATATYPE_FLAGS_OFFSET: usize = 3;

// Section `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`, version 4.0, assigns VL subtype 1 to
// strings in datatype messages 1 through 5.
const VARIABLE_LENGTH_STRING_BIT: u8 = 0x01;
