//! Provides fixed-width string fixtures shared by pure and interoperability tests.

use hdf5_pure::{FileBuilder, MaxExtent};

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
