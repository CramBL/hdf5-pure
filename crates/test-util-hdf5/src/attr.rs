//! Attribute-value helpers shared by integration tests.

use std::collections::HashMap;

#[cfg(feature = "hdf5")]
use hdf5::Location;
#[cfg(feature = "hdf5")]
use hdf5::types::VarLenUnicode;
use hdf5_pure::AttrValue;

/// Returns the strings in an array-of-strings attribute.
///
/// Accepts either variable-length ASCII character arrays or fixed-width string arrays, since the
/// reader may represent equivalent HDF5 string-array datatypes with either variant.
#[track_caller]
pub fn string_array(attrs: &HashMap<String, AttrValue>, name: &str) -> Vec<String> {
    match attrs.get(name) {
        Some(AttrValue::VarLenAsciiCharArray(values) | AttrValue::StringArray(values)) => {
            values.clone()
        }
        other => panic!("expected an array-of-strings attribute {name:?}, got {other:?}"),
    }
}

/// Writes a one-dimensional native variable-length Unicode attribute.
///
/// # Panics
///
/// Panics if a value contains a NUL byte or libhdf5 cannot create or write the attribute.
#[cfg(feature = "hdf5")]
pub fn write_unicode_array(location: &Location, name: &str, values: &[&str]) {
    let strings = values
        .iter()
        .map(|value| value.parse::<VarLenUnicode>().unwrap())
        .collect::<Vec<_>>();
    location
        .new_attr::<VarLenUnicode>()
        .shape(values.len())
        .create(name)
        .unwrap()
        .write_raw(&strings)
        .unwrap();
}

/// Reads a native variable-length Unicode attribute into owned strings.
///
/// # Panics
///
/// Panics if libhdf5 cannot open or read the attribute as variable-length Unicode strings.
#[cfg(feature = "hdf5")]
pub fn unicode_array(location: &Location, name: &str) -> Vec<String> {
    location
        .attr(name)
        .unwrap()
        .read_raw::<VarLenUnicode>()
        .unwrap()
        .iter()
        .map(|value| value.as_str().to_owned())
        .collect()
}
