//! Fixed-width string fixtures and native observations.

use hdf5_pure::{FileBuilder, MaxExtent};

/// Reads a scalar UTF-8 attribute after checking its fixed width in bytes.
///
/// # Panics
///
/// Panics if the attribute cannot be opened or read, its datatype cannot be
/// inspected, or it is not a scalar fixed-width UTF-8 string of `N` bytes.
///
/// # Examples
///
/// ```
/// use hdf5_pure::{AttrValue, FileBuilder};
/// use test_util_hdf5::{file, fixed_string};
///
/// # let path = test_util::temp::temp_path("attribute.h5");
/// let mut builder = FileBuilder::new();
/// builder.set_attr("units", AttrValue::String("meters".to_owned()));
/// builder.write(&path)?;
///
/// let native = file::libhdf5_open(&path);
/// assert_eq!(fixed_string::read_libhdf5_attribute::<6>(&native, "units"), "meters");
/// native.close()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "hdf5")]
#[track_caller]
pub fn read_libhdf5_attribute<const N: usize>(location: &hdf5::Location, name: &str) -> String {
    let attribute = location
        .attr(name)
        .unwrap_or_else(|error| panic!("open attribute {name:?}: {error}"));
    assert!(attribute.is_scalar());
    assert_eq!(
        attribute.dtype().unwrap().to_descriptor().unwrap(),
        hdf5::types::TypeDescriptor::FixedUnicode(N)
    );
    attribute
        .read_scalar::<hdf5::types::FixedUnicode<N>>()
        .unwrap_or_else(|error| panic!("read attribute {name:?}: {error}"))
        .as_str()
        .to_owned()
}

/// Adds derived and explicit-width ASCII and UTF-8 datasets.
pub fn matrix(builder: &mut FileBuilder) {
    builder
        .create_dataset("derived_ascii")
        .with_ascii_strings(&MATRIX_VALUES)
        .unwrap();
    builder
        .create_dataset("sized_ascii")
        .with_ascii_strings_sized(&MATRIX_VALUES, 16)
        .unwrap();
    builder
        .create_dataset("derived_utf8")
        .with_strings(&MATRIX_VALUES)
        .unwrap();
    builder
        .create_dataset("sized_utf8")
        .with_strings_sized(&MATRIX_VALUES, 16)
        .unwrap();
}

/// Values used by the fixed-width matrix, including an empty interior element.
pub const MATRIX_VALUES: [&str; 4] = ["north", "s", "", "east"];

/// Adds a UTF-8 dataset whose longest value is wider in bytes than in characters.
pub fn utf8_units(builder: &mut FileBuilder) {
    builder
        .create_dataset("units")
        .with_strings(&UTF8_VALUES)
        .unwrap();
}

/// Values whose longest UTF-8 spelling occupies six bytes.
pub const UTF8_VALUES: [&str; 3] = ["mètre", "K", "°C"];

/// Adds an ASCII dataset whose values are all empty strings.
pub fn empty_ascii(builder: &mut FileBuilder) {
    builder
        .create_dataset("blank")
        .with_ascii_strings(&["", "", ""])
        .unwrap();
}

/// Adds an extensible ASCII dataset with an explicit 16-byte element width.
pub fn extensible_ascii(builder: &mut FileBuilder) {
    builder
        .create_dataset("station")
        .with_ascii_strings_sized(&["north", "s"], 16)
        .unwrap()
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[4]);
}

/// Values appended to the extensible fixed-width fixture.
pub const EXTENDED_VALUES: [&str; 2] = ["north-northeast", "e"];
