//! The C library and this crate read the same object header, and reject it under write access:
//! one with a message of an unknown type, flagged as a message a decoder with write access must
//! understand.
//!
//! Bit 3 of the "Header Message #n Flags" field of "Version 1 Data Object Header Prefix" in the
//! [format specification, version 4.0][spec] requires a decoder that does not understand the
//! message's type to reject the object while the file is open for write access. `H5Fopen` reads
//! the edited file and `H5Fopen` under `H5F_ACC_RDWR` fails on it, and this crate reads the
//! object through [`File::open`] and reports [`FormatError::UnsupportedMessage`] through
//! [`File::open_rw`]. `hdf5_pure::repack` reads its source through a read-only open, as
//! `h5repack` does, and writes a destination both libraries open for writing. The message
//! sits in the fixture's root group header, which `h5repack` rebuilds as well. Under
//! `RepackOptions::reject_unknown_messages_only_a_writer_must_understand`, which this crate
//! offers beyond `h5repack`, the same repack reports the error.
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one

use hdf5::{MajorErrorCode, MinorErrorCode};
use hdf5_pure::{Error, File, FormatError, RepackOptions};
use tempfile::tempdir;
use test_util::object_header::{MessageFlags, MessageType};

use super::fixture;
use super::fixture::{ATTRIBUTE_NAME, DATA, DATASET_NAME};

/// The name a copy of the fixture's dataset is written under.
const COPY_NAME: &str = "copied";

#[test]
fn an_unknown_message_only_a_writer_must_understand_is_read_as_the_c_library_reads_it() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("fail_if_unknown_for_write.h5");
    write_fixture_with_the_flagged_message(&path);

    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset(DATASET_NAME).unwrap().read_raw::<i32>().unwrap(),
        DATA
    );
    assert_eq!(
        c.attr_names().unwrap(),
        Vec::<String>::new(),
        "the retyped message is no longer the {ATTRIBUTE_NAME} attribute"
    );
    c.close().unwrap();

    let pure = File::open(&path).unwrap();
    assert_eq!(
        pure.dataset(DATASET_NAME).unwrap().read_i32().unwrap(),
        DATA
    );
    assert_eq!(
        pure.root().attrs().unwrap().into_keys().collect::<Vec<_>>(),
        Vec::<String>::new(),
        "the retyped message is no longer the {ATTRIBUTE_NAME} attribute"
    );
    drop(pure);

    let c_err = hdf5::File::open_rw(&path).unwrap_err();
    assert!(
        c_err.contains_major(MajorErrorCode::ObjectHeader),
        "{c_err}"
    );
    assert!(c_err.contains_minor(MinorErrorCode::BadMessage), "{c_err}");

    let err = File::open_rw(&path)
        .unwrap()
        .dataset(DATASET_NAME)
        .unwrap_err();
    let Error::Format(FormatError::UnsupportedMessage(id)) = &err else {
        panic!("expected UnsupportedMessage, got {err:?}");
    };
    assert_eq!(*id, MessageType::UNKNOWN.0);
}

#[test]
fn a_repack_reads_past_the_message_and_writes_a_destination_both_libraries_open_for_writing() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("source.h5");
    let dst = dir.path().join("repacked.h5");
    write_fixture_with_the_flagged_message(&src);

    hdf5_pure::repack(&src, &dst, &RepackOptions::new()).unwrap();

    let pure = File::open(&dst).unwrap();
    let pure_data = pure.dataset(DATASET_NAME).unwrap().read_i32().unwrap();
    let pure_attributes: Vec<String> = pure.root().attrs().unwrap().into_keys().collect();
    drop(pure);
    assert_eq!(pure_data, DATA);
    assert_eq!(
        pure_attributes,
        Vec::<String>::new(),
        "the retyped message is no longer the {ATTRIBUTE_NAME} attribute, and a repack \
         reproduces neither"
    );

    let c = hdf5::File::open(&dst).unwrap();
    assert_eq!(
        c.dataset(DATASET_NAME).unwrap().read_raw::<i32>().unwrap(),
        pure_data
    );
    assert_eq!(c.attr_names().unwrap(), pure_attributes);
    c.close().unwrap();

    // A write open is what rejects a message of an unknown type flagged for one, so it is what
    // shows the repack carried none into `dst`. This crate's open takes the file's exclusive
    // lock, so it goes first and is dropped before the C library's.
    File::open_rw(&dst).unwrap().dataset(DATASET_NAME).unwrap();
    hdf5::File::open_rw(&dst).unwrap().close().unwrap();
}

#[test]
fn a_repack_asked_to_reject_unknown_messages_only_a_writer_must_understand_names_the_type_and_leaves_the_destination_absent()
 {
    let dir = tempdir().unwrap();
    let src = dir.path().join("source.h5");
    let dst = dir.path().join("repacked.h5");
    write_fixture_with_the_flagged_message(&src);

    let err = hdf5_pure::repack(
        &src,
        &dst,
        &RepackOptions::new().reject_unknown_messages_only_a_writer_must_understand(),
    )
    .unwrap_err();
    let Error::Format(FormatError::UnsupportedMessage(id)) = &err else {
        panic!("expected UnsupportedMessage, got {err:?}");
    };
    assert_eq!(*id, MessageType::UNKNOWN.0);
    assert!(
        !dst.exists(),
        "a rejected repack leaves the destination absent"
    );
}

#[test]
fn a_cross_file_copy_reads_past_the_message_and_stops_only_at_the_version_1_header() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("source.h5");
    let dst = dir.path().join("destination.h5");
    write_fixture_with_the_flagged_message(&src);
    hdf5::File::create(&dst).unwrap().close().unwrap();

    let source = File::open(&src).unwrap();
    let destination = File::open_rw(&dst).unwrap();
    let err = destination
        .copy_from(&source, DATASET_NAME, COPY_NAME)
        .unwrap_err();
    let Error::EditUnsupported(reason) = &err else {
        panic!("expected EditUnsupported, got {err:?}");
    };
    assert_eq!(*reason, "an object does not use a version 2 object header");
    drop(destination);
    drop(source);

    let c_src = hdf5::File::open(&src).unwrap();
    let c_dst = hdf5::File::create(dir.path().join("c_destination.h5")).unwrap();
    c_src
        .dataset(DATASET_NAME)
        .unwrap()
        .copy_to(&c_dst, COPY_NAME)
        .unwrap();
    assert_eq!(
        c_dst.dataset(COPY_NAME).unwrap().read_raw::<i32>().unwrap(),
        DATA
    );
    c_dst.close().unwrap();
    c_src.close().unwrap();
}

#[test]
fn a_same_file_copy_rejects_a_file_whose_root_group_header_holds_a_message_only_a_writer_must_understand()
 {
    let dir = tempdir().unwrap();
    let path = dir.path().join("source.h5");
    write_fixture_with_the_flagged_message(&path);

    let file = File::open_rw(&path).unwrap();
    file.copy(DATASET_NAME, COPY_NAME).unwrap();
    let err = file.commit().unwrap_err();
    let Error::Format(FormatError::UnsupportedMessage(id)) = &err else {
        panic!("expected UnsupportedMessage, got {err:?}");
    };
    assert_eq!(*id, MessageType::UNKNOWN.0);
}

/// Writes the fixture with its root group attribute retyped to a message of an unknown type,
/// flagged as one a decoder with write access must understand.
fn write_fixture_with_the_flagged_message(path: &std::path::Path) {
    fixture::write_earliest_format_fixture(path);
    let mut bytes = std::fs::read(path).unwrap();
    fixture::retype_the_root_group_attribute_message_as_unknown(
        &mut bytes,
        MessageFlags::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE,
    );
    std::fs::write(path, &bytes).unwrap();
}
