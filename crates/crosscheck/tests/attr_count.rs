// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Reference-C-library interop for the object header's *attribute count*.
//!
//! On a version 2 object header the C library does not count attribute messages
//! to answer `H5Oget_info().num_attrs`. It reads the count out of the Attribute
//! Info message (0x0015) — `H5O__attr_count_real` in `H5Oattribute.c`, which
//! reports zero when no such message exists, and otherwise defers to
//! `H5A__get_ainfo` in `H5Aint.c`, where an *undefined* fractal-heap address
//! means "use the header's own tally of attribute messages".
//!
//! That makes the count a separate claim from the attributes themselves.
//! `H5Aiterate` and `H5Aopen_by_name` walk the messages directly and so agree
//! with this crate's own reader no matter what the count says, which is why
//! releases through 0.33.0 shipped every compactly-stored attribute set
//! declaring zero attributes without any test noticing. A consumer that *sizes*
//! its work by the count instead sees an object with no attributes: `h5repack`
//! is one, and a round trip through it silently produced a `.mat` file with no
//! `MATLAB_class` on anything.
//!
//! So what each test asserts is the count against the iteration, which is the
//! disagreement itself. Note that the C library's own `H5Ocopy` is *not* a guard
//! here — it enumerates rather than counting, so it copied the attributes
//! correctly out of files whose headers declared none. The count-driven loop is
//! in `h5repack`'s tool code, not in the library.

use hdf5_pure::{AttrValue, File, FileBuilder};
use tempfile::tempdir;

/// The C library's declared attribute count for `obj`, and the names it finds by
/// iterating. A file this crate writes correctly makes these agree; the defect
/// this module guards is exactly their disagreement.
fn declared_and_iterated(obj: &hdf5::Location) -> (usize, Vec<String>) {
    let declared = obj
        .loc_info()
        .expect("C library reads object info")
        .num_attrs;
    let mut iterated = obj.attr_names().expect("C library iterates attributes");
    iterated.sort();
    (declared, iterated)
}

#[track_caller]
fn assert_count_agrees(obj: &hdf5::Location, expected: &[&str]) {
    let (declared, iterated) = declared_and_iterated(obj);
    let mut expected: Vec<String> = expected.iter().map(|s| (*s).to_string()).collect();
    expected.sort();
    assert_eq!(iterated, expected, "the C library found other attributes");
    assert_eq!(
        declared,
        expected.len(),
        "the object header declares {declared} attributes but carries {iterated:?}"
    );
}

/// A file with one compactly-stored attribute set on each kind of object: the
/// root group, a subgroup, and a dataset.
fn write_compact_attrs(path: &std::path::Path, count: usize) {
    let names: Vec<String> = (0..count).map(|i| format!("a{i}")).collect();
    let mut b = FileBuilder::new();
    for name in &names {
        b.set_attr(name, AttrValue::I64(1));
    }
    {
        let ds = b.create_dataset("values").with_f64_data(&[1.0, 2.0, 3.0]);
        for name in &names {
            ds.set_attr(name, AttrValue::I64(2));
        }
    }
    let mut g = b.create_group("grp");
    for name in &names {
        g.set_attr(name, AttrValue::I64(3));
    }
    g.create_dataset("inner").with_f64_data(&[1.0]);
    let g = g.finish();
    b.add_group(g);
    b.write(path).unwrap();
}

/// The names `write_compact_attrs` puts on every object.
fn attr_names(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("a{i}")).collect()
}

#[test]
fn c_counts_a_single_compact_attribute_on_every_kind_of_object() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("one.h5");
    write_compact_attrs(&path, 1);

    let f = hdf5::File::open(&path).unwrap();
    assert_count_agrees(&f, &["a0"]);
    assert_count_agrees(&f.dataset("values").unwrap(), &["a0"]);
    assert_count_agrees(&f.group("grp").unwrap(), &["a0"]);
}

/// The whole compact range, up to the last count stored in the object header.
/// One attribute and eight are different message layouts in the header, and the
/// count is declared once for the set, so a guard that only ever checks one of
/// them would not notice a count that tracked the wrong quantity.
#[test]
fn c_counts_every_size_of_compact_attribute_set() {
    let dir = tempdir().unwrap();
    for count in 1..=8 {
        let path = dir.path().join(format!("n{count}.h5"));
        write_compact_attrs(&path, count);
        let names = attr_names(count);
        let expect: Vec<&str> = names.iter().map(String::as_str).collect();

        let f = hdf5::File::open(&path).unwrap();
        assert_count_agrees(&f, &expect);
        assert_count_agrees(&f.dataset("values").unwrap(), &expect);
        assert_count_agrees(&f.group("grp").unwrap(), &expect);
    }
}

/// Dense storage was already correct — its Attribute Info message names a
/// fractal heap, and the C library counts the heap's name B-tree records — so
/// this holds the side of the boundary the fix did not touch.
#[test]
fn c_counts_a_densely_stored_attribute_set() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("dense.h5");
    let count = 12;
    write_compact_attrs(&path, count);

    let names = attr_names(count);
    let expect: Vec<&str> = names.iter().map(String::as_str).collect();
    let f = hdf5::File::open(&path).unwrap();
    assert_count_agrees(&f, &expect);
    assert_count_agrees(&f.dataset("values").unwrap(), &expect);
    assert_count_agrees(&f.group("grp").unwrap(), &expect);
}

#[test]
fn c_counts_the_attributes_an_edit_session_added() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("edited.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3, 4]);
    let g = b.create_group("g").finish();
    b.add_group(g);
    b.write(&path).unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr("MATLAB_class", AttrValue::AsciiString("double".into()))
            .unwrap();
        s.group("g")
            .unwrap()
            .set_attr("tag", AttrValue::I64(7))
            .unwrap();
        s.root().set_attr("root_tag", AttrValue::I64(9)).unwrap();
        s.commit().unwrap();
    }

    let f = hdf5::File::open(&path).unwrap();
    assert_count_agrees(&f, &["root_tag"]);
    assert_count_agrees(&f.dataset("d").unwrap(), &["MATLAB_class"]);
    assert_count_agrees(&f.group("g").unwrap(), &["tag"]);
}
