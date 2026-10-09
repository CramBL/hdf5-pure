//! Fixtures for dense attribute storage boundary tests.

use hdf5_pure::{AttrValue, FileBuilder};
use test_util::fractal_heap;

/// Builds a file with nine attributes, enough to select dense storage.
///
/// The first attribute contains `payload` bytes of text. The remaining eight are
/// small integer attributes.
pub fn nine_attrs(payload: usize) -> FileBuilder {
    let mut builder = FileBuilder::new();
    builder.set_attr("big", AttrValue::AsciiString("y".repeat(payload)));
    for i in 0..8 {
        builder.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder
}

/// Finds the largest text payload still stored as a managed heap object.
///
/// The threshold applies to the serialized attribute, whose overhead is internal,
/// so the fixture probes the boundary rather than assuming it.
pub fn largest_managed_payload() -> usize {
    for payload in (1..=70_000).rev() {
        let bytes = nine_attrs(payload)
            .finish()
            .expect("every size is writable");
        if fractal_heap::huge_object_count(&bytes) == 0 {
            return payload;
        }
    }
    panic!("no payload was stored as a managed object");
}
