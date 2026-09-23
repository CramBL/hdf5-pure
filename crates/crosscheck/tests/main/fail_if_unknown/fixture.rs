//! The file the unknown-message tests share, and the edit that gives it its unknown message.
//!
//! No writer emits a message of a type neither library defines, so a test writes a file through
//! the C library and edits the bytes: [`write_earliest_format_fixture`] writes a dataset and a
//! root group attribute under a low library bound of `H5F_LIBVER_EARLIEST`, which gives the file
//! a version 0 superblock and version 1 object headers that no checksum covers, and
//! [`retype_the_root_group_attribute_message_as_unknown`] retypes the attribute message and sets
//! the flags byte its caller chooses.

use test_util::object_header::{MessageFlags, MessageType, v1};
use test_util_hdf5::file;

/// Writes a file with one dataset and one attribute on the root group, under a low library bound
/// of `H5F_LIBVER_EARLIEST`.
pub fn write_earliest_format_fixture(path: &std::path::Path) {
    let file = file::libhdf5_create_earliest(path);
    file.new_dataset::<i32>()
        .shape([DATA.len()])
        .create(DATASET_NAME)
        .unwrap()
        .write(&DATA)
        .unwrap();
    file.new_attr::<i32>()
        .create(ATTRIBUTE_NAME)
        .unwrap()
        .write_scalar(&ATTRIBUTE_VALUE)
        .unwrap();
    file.close().unwrap();
}

/// Gives the root group's Attribute message record the type [`MessageType::UNKNOWN`] and the
/// flags byte `flags`.
///
/// Walks the object header at the address the superblock's root group symbol table entry stores,
/// following each continuation message, and edits the first Attribute record it finds.
///
/// # Panics
///
/// Panics if `file` is not shaped as [`write_earliest_format_fixture`] writes it: a version 0
/// superblock with no userblock, and a version 1 root group header with an Attribute message.
pub fn retype_the_root_group_attribute_message_as_unknown(file: &mut [u8], flags: MessageFlags) {
    let chunks = v1::root_group_chunks(file);
    let records = chunks.iter().flat_map(|chunk| &chunk.records);
    let Some(attribute) = records
        .clone()
        .find(|record| record.msg_type == MessageType::ATTRIBUTE)
    else {
        let seen: Vec<_> = records.map(|record| record.msg_type).collect();
        panic!("the root group's object header holds no attribute message, only {seen:02X?}");
    };
    attribute.set_type(file, MessageType::UNKNOWN);
    attribute.set_flags(file, flags);
}

pub const DATASET_NAME: &str = "d";

pub const DATA: [i32; 4] = [1, 2, 3, 4];

pub const ATTRIBUTE_NAME: &str = "note";

pub const ATTRIBUTE_VALUE: i32 = 7;
