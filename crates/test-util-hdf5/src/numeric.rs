//! Numeric dataset fixtures written through the reference library.

use std::ffi::CString;
use std::path::Path;

use hdf5::Datatype;
use hdf5::globals;
use hdf5::types::TypeDescriptor;
use hdf5_sys::h5d;
use hdf5_sys::h5p::H5P_DEFAULT;
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
