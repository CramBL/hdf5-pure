//! The C library and this crate reject the same object header: one with a message of an unknown
//! type, flagged as a message every decoder must understand.
//!
//! The test gives the [`fixture`]'s retyped message bit 7 of the "Header Message #n Flags" field
//! of "Version 1 Data Object Header Prefix" in the [format specification, version 4.0][spec],
//! which requires a decoder that does not understand the message's type to reject the object
//! whether the file is open for reading or for writing. `H5Fopen` then fails on the file, and
//! this crate reports [`FormatError::UnsupportedMessage`] for the first read that reaches the
//! root group.
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one

use hdf5::{MajorErrorCode, MinorErrorCode};
use hdf5_pure::{Error, File, FormatError};
use tempfile::tempdir;
use test_util::object_header::{MessageFlags, MessageType};

use super::fixture;
use super::fixture::{ATTRIBUTE_NAME, ATTRIBUTE_VALUE, DATA, DATASET_NAME};

#[test]
fn an_unknown_message_that_must_always_be_understood_is_rejected_as_the_c_library_rejects_it() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("fail_if_unknown_always.h5");
    fixture::write_earliest_format_fixture(&path);

    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset(DATASET_NAME).unwrap().read_raw::<i32>().unwrap(),
        DATA
    );
    assert_eq!(
        c.attr(ATTRIBUTE_NAME)
            .unwrap()
            .read_scalar::<i32>()
            .unwrap(),
        ATTRIBUTE_VALUE
    );
    c.close().unwrap();
    let pure = File::open(&path).unwrap();
    assert_eq!(
        pure.dataset(DATASET_NAME).unwrap().read_i32().unwrap(),
        DATA
    );
    drop(pure);

    let mut bytes = std::fs::read(&path).unwrap();
    fixture::retype_the_root_group_attribute_message_as_unknown(
        &mut bytes,
        MessageFlags::FAIL_IF_UNKNOWN_ALWAYS,
    );
    std::fs::write(&path, &bytes).unwrap();

    let c_err = hdf5::File::open(&path).unwrap_err();
    assert!(
        c_err.contains_major(MajorErrorCode::ObjectHeader),
        "{c_err}"
    );
    assert!(c_err.contains_minor(MinorErrorCode::BadMessage), "{c_err}");

    let err = File::open(&path)
        .unwrap()
        .dataset(DATASET_NAME)
        .unwrap_err();
    let Error::Format(FormatError::UnsupportedMessage(id)) = &err else {
        panic!("expected UnsupportedMessage, got {err:?}");
    };
    assert_eq!(*id, MessageType::UNKNOWN.0);
}
