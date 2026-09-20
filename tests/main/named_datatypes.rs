//! Committed (`H5Tcommit`) datatypes in files the reference C library wrote,
//! from committed bytes.

use hdf5_pure::File;

/// A version 1 object header keeps its reference count in the header prefix,
/// where a version 2 header stores an Object Reference Count message. The C
/// library wrote this type with one link and two dataset uses.
#[test]
fn the_reference_count_of_a_version_1_header_is_read_from_its_prefix() {
    let file = File::open("tests/data/c/committed_datatype_v1.h5").unwrap();
    assert_eq!(file.root().named_datatype_references("mytype").unwrap(), 3);
}
