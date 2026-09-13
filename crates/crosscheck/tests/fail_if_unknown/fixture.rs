//! The file the unknown-message tests share, and the edit that gives it its unknown message.
//!
//! No writer emits a message of a type neither library defines, so a test writes a file through
//! the C library and edits the bytes: [`write_earliest_format_fixture`] writes a dataset and a
//! root group attribute under a low library bound of `H5F_LIBVER_EARLIEST`, which gives the file
//! a version 0 superblock and version 1 object headers that no checksum covers, and
//! [`retype_the_root_group_attribute_message_as_unknown`] retypes the attribute message and sets
//! the flags byte its caller chooses.

use hdf5::file::LibraryVersion;

/// Writes a file with one dataset and one attribute on the root group, under a low library bound
/// of `H5F_LIBVER_EARLIEST`.
pub fn write_earliest_format_fixture(path: &std::path::Path) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::Earliest, LibraryVersion::latest()))
        .create(path)
        .unwrap();
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

/// Gives the root group's Attribute message record the type [`UNKNOWN_MESSAGE_TYPE`] and the
/// flags byte `flags`.
///
/// Walks the object header at the address the superblock's root group symbol table entry stores,
/// following each continuation message, and edits the first Attribute record it finds.
///
/// # Panics
///
/// Panics if `file` is not shaped as [`write_earliest_format_fixture`] writes it: a version 0
/// superblock with eight-byte offsets and no userblock, and a version 1 root group header with an
/// Attribute message.
pub fn retype_the_root_group_attribute_message_as_unknown(file: &mut [u8], flags: u8) {
    assert_eq!(file[8], 0, "version 0 superblock");
    assert_eq!(file[13], 8, "eight-byte offsets");
    assert_eq!(
        u64::from_le_bytes(file[24..32].try_into().unwrap()),
        0,
        "no userblock, so every address is a file offset"
    );

    // A version 0 superblock ends with the root group's symbol table entry: a link name offset,
    // then the address of the group's object header. "Format Signature and Superblock", version
    // 4.0.
    let header = u64::from_le_bytes(file[64..72].try_into().unwrap()) as usize;
    assert_eq!(file[header], 1, "version 1 object header");
    let header_data_size =
        u32::from_le_bytes(file[header + 8..header + 12].try_into().unwrap()) as usize;

    // The prefix is padded to sixteen bytes, and each message record that follows it is a type, a
    // size, a flags byte, three reserved bytes and a body the size counts. A continuation message
    // refers to a further chunk of records, which a version 1 header stores without a signature.
    let mut chunks = vec![(header + 16, header + 16 + header_data_size)];
    let mut seen = Vec::new();
    while let Some((mut pos, end)) = chunks.pop() {
        while pos + 8 <= end {
            let msg_type = u16::from_le_bytes(file[pos..pos + 2].try_into().unwrap());
            let msg_size = u16::from_le_bytes(file[pos + 2..pos + 4].try_into().unwrap()) as usize;
            let body = pos + 8;
            if msg_type == ATTRIBUTE_MESSAGE_TYPE {
                file[pos..pos + 2].copy_from_slice(&UNKNOWN_MESSAGE_TYPE.to_le_bytes());
                file[pos + 4] = flags;
                return;
            }
            if msg_type == OBJECT_HEADER_CONTINUATION_MESSAGE_TYPE {
                let offset = u64::from_le_bytes(file[body..body + 8].try_into().unwrap()) as usize;
                let length =
                    u64::from_le_bytes(file[body + 8..body + 16].try_into().unwrap()) as usize;
                chunks.push((offset, offset + length));
            }
            seen.push(msg_type);
            pos = body + msg_size;
        }
    }
    panic!("the root group's object header holds no attribute message, only {seen:02X?}");
}

pub const DATASET_NAME: &str = "d";

pub const DATA: [i32; 4] = [1, 2, 3, 4];

pub const ATTRIBUTE_NAME: &str = "note";

pub const ATTRIBUTE_VALUE: i32 = 7;

// A type the format specification assigns to no message: the highest it assigns is 0x0017,
// "Data Object Header Messages", version 4.0.
pub const UNKNOWN_MESSAGE_TYPE: u16 = 0x00FF;

// Bit 3 of the "Header Message #n Flags" field, "Version 1 Data Object Header Prefix",
// version 4.0.
pub const FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE: u8 = 0x08;

// Bit 7 of the "Header Message #n Flags" field, "Version 1 Data Object Header Prefix",
// version 4.0.
pub const FAIL_IF_UNKNOWN_ALWAYS: u8 = 0x80;

// The type of the message that holds a compact attribute, "The Attribute Message",
// version 4.0.
const ATTRIBUTE_MESSAGE_TYPE: u16 = 0x000C;

// The type of the message that identifies another chunk of the same object header, "The Object
// Header Continuation Message", version 4.0.
const OBJECT_HEADER_CONTINUATION_MESSAGE_TYPE: u16 = 0x0010;
