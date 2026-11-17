use std::path::Path;

use hdf5_pure::FileSpaceSettings;
use hdf5_pure::LengthWidth;
use hdf5_pure::OffsetWidth;

use crate::file;
use crate::global_heap::attributes;
use crate::numeric::ByteOrder;

/// Writes a native source with eight-byte widths, root attributes, and little-endian integers.
///
/// # Panics
///
/// Panics if the file, dataset, or attributes cannot be created or closed.
pub fn write_source(path: &Path, settings: FileSpaceSettings) {
    let native = file::libhdf5_create_with_widths_and_space(
        path,
        OffsetWidth::Eight,
        LengthWidth::Eight,
        0,
        settings,
    );
    native
        .new_dataset_builder()
        .with_data_as(&[17_i32], ByteOrder::LittleEndian.i32_datatype())
        .create("numbers")
        .unwrap();
    attributes::write_ascii(&native, "first", "removed");
    attributes::write_ascii(&native, "second", "discard");
    attributes::write_ascii(&native, "third", "retained");
    native.close().unwrap();
}
