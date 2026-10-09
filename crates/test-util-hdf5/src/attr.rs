//! Attribute-value helpers shared by integration tests.

use std::collections::HashMap;

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
