//! Numeric dataset fixtures written through the reference library.

use std::path::Path;

use hdf5::Datatype;
use hdf5_sys::h5d;
use hdf5_sys::h5p::H5P_DEFAULT;
use hdf5_sys::h5s::H5S_ALL;
use hdf5_sys::h5t::{self, H5T_class_t, H5T_order_t};

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
