//! Native observations of verbatim attributes on the root group and a dataset.

use std::path::Path;

use hdf5::Datatype;
use hdf5::H5Type;
use hdf5::Location;
use hdf5::dataset::Layout;
use hdf5::datatype::ByteOrder;
use hdf5::types::FixedAscii;
use hdf5::types::FixedUnicode;
use hdf5::types::TypeDescriptor;
use hdf5_sys::h5t;
use hdf5_sys::h5t::H5T_str_t as NativePadding;

use crate::numeric::ByteOrder as NumericOrder;

/// Checks attribute values, datatypes, dataspaces, and string padding through libhdf5.
///
/// Both the root group and `data` must have these attributes:
///
/// | Name | Type and dataspace | Value |
/// |---|---|---|
/// | `units` | 16-byte ASCII string, null-terminated, scalar | `m/s` |
/// | `spaced` | 8-byte UTF-8 string, space-padded, scalar | `ab` |
/// | `nothing` | Little-endian signed 32-bit integer, null dataspace | Empty |
/// | `big_endian` | Big-endian signed 32-bit integer, scalar | -7 |
/// | `grid` | Little-endian signed 32-bit integer, shape `[2, 3]` | 1 through 6 |
///
/// The contiguous, unfiltered dataset `data` must contain one little-endian [`f64`]
/// value equal to 1.0. Closes the file before returning.
///
/// # Panics
///
/// Panics if opening, reading, or close fails, or any observation differs from the fixture.
pub fn assert_libhdf5_contents(path: &Path) {
    let file = crate::file::libhdf5_open(path);
    assert_eq!(file.member_names().unwrap(), ["data"]);
    assert_attributes(&file);
    {
        let dataset = file.dataset("data").unwrap();
        assert_eq!(dataset.shape(), [1]);
        assert_eq!(
            dataset.dtype().unwrap().to_descriptor().unwrap(),
            f64::type_descriptor()
        );
        assert_eq!(
            dataset.dtype().unwrap().byte_order(),
            ByteOrder::LittleEndian
        );
        assert_eq!(dataset.read_raw::<f64>().unwrap(), [1.0]);
        assert_eq!(dataset.layout(), Layout::Contiguous);
        assert_eq!(dataset.chunk(), None);
        assert!(dataset.filters().is_empty());
        assert_attributes(&dataset);
    }
    file.close().unwrap();
}

fn assert_attributes(owner: &Location) {
    let mut names = owner.attr_names().unwrap();
    names.sort();
    assert_eq!(names, ["big_endian", "grid", "nothing", "spaced", "units"]);
    let units = owner.attr("units").unwrap();
    assert!(units.space().unwrap().is_scalar());
    assert_eq!(
        units.dtype().unwrap().to_descriptor().unwrap(),
        TypeDescriptor::FixedAscii(16)
    );
    assert_eq!(
        string_padding(&units.dtype().unwrap()),
        NativePadding::H5T_STR_NULLTERM
    );
    assert_eq!(
        units.read_scalar::<FixedAscii<16>>().unwrap().as_str(),
        "m/s"
    );
    let spaced = owner.attr("spaced").unwrap();
    assert!(spaced.space().unwrap().is_scalar());
    assert_eq!(
        spaced.dtype().unwrap().to_descriptor().unwrap(),
        TypeDescriptor::FixedUnicode(8)
    );
    assert_eq!(
        string_padding(&spaced.dtype().unwrap()),
        NativePadding::H5T_STR_SPACEPAD
    );
    assert_eq!(
        spaced.read_scalar::<FixedUnicode<8>>().unwrap().as_str(),
        "ab"
    );
    let nothing = owner.attr("nothing").unwrap();
    assert_eq!(
        nothing.dtype().unwrap(),
        NumericOrder::LittleEndian.i32_datatype()
    );
    assert!(nothing.space().unwrap().is_null());
    assert_eq!(nothing.space().unwrap().size(), 0);
    assert_eq!(nothing.storage_size(), 0);
    assert_eq!(
        nothing.dtype().unwrap().to_descriptor().unwrap(),
        i32::type_descriptor()
    );
    assert_eq!(
        nothing.dtype().unwrap().byte_order(),
        ByteOrder::LittleEndian
    );
    assert_eq!(nothing.read_raw::<i32>().unwrap(), Vec::<i32>::new());
    let big_endian = owner.attr("big_endian").unwrap();
    assert_eq!(
        big_endian.dtype().unwrap(),
        NumericOrder::BigEndian.i32_datatype()
    );
    assert!(big_endian.space().unwrap().is_scalar());
    assert_eq!(
        big_endian.dtype().unwrap().to_descriptor().unwrap(),
        i32::type_descriptor()
    );
    assert_eq!(
        big_endian.dtype().unwrap().byte_order(),
        ByteOrder::BigEndian
    );
    assert_eq!(big_endian.read_scalar::<i32>().unwrap(), -7);
    let grid = owner.attr("grid").unwrap();
    assert_eq!(
        grid.dtype().unwrap(),
        NumericOrder::LittleEndian.i32_datatype()
    );
    assert_eq!(grid.shape(), [2, 3]);
    assert!(grid.space().unwrap().is_simple());
    assert_eq!(
        grid.dtype().unwrap().to_descriptor().unwrap(),
        i32::type_descriptor()
    );
    assert_eq!(grid.dtype().unwrap().byte_order(), ByteOrder::LittleEndian);
    assert_eq!(grid.read_raw::<i32>().unwrap(), [1, 2, 3, 4, 5, 6]);
}

fn string_padding(datatype: &Datatype) -> NativePadding {
    hdf5::sync::sync(|| {
        // SAFETY: The datatype owns a live HDF5 handle, and the library lock serializes access.
        unsafe { h5t::H5Tget_strpad(datatype.id()) }
    })
}
