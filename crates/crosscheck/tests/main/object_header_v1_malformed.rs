#![cfg(feature = "hdf5")]
//! Cross-checks malformed version 1 object-header message records.
//!
//! The reference C library creates earliest-format files whose root groups use
//! version 1 object headers. Tests apply narrowly scoped mutations while
//! preserving the surrounding file structure, then compare `libhdf5` with both
//! hdf5-pure parser backends.
//!
//! The tests pin release-dependent `libhdf5` behavior. When behavior differs
//! across releases, modern HDF5 releases are also checked under explicit
//! `libver` bounds where applicable.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

#[cfg(any(feature = "__hdf5-1.14", feature = "__hdf5-2"))]
use hdf5::file::LibraryVersion;
use tempfile::tempdir;

const HDF5_SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', b'\r', b'\n', 0x1a, b'\n'];

const V1_HEADER_PREFIX_LEN: usize = 16;
const V1_MESSAGE_PREFIX_LEN: usize = 8;

const NIL_MESSAGE: u16 = 0x0000;
const ATTRIBUTE_MESSAGE: u16 = 0x000c;
const CONTINUATION_MESSAGE: u16 = 0x0010;
const SYMBOL_TABLE_MESSAGE: u16 = 0x0011;

#[derive(Clone, Copy, Debug)]
struct FileLayout {
    offset_size: usize,
    length_size: usize,
    eof_address_field: usize,
    root_chunk_start: usize,
    root_chunk_end: usize,
}

#[derive(Clone, Copy, Debug)]
struct Record {
    offset: usize,
    msg_type: u16,
    body_size: u16,
    body_start: usize,
    body_end: usize,
}

#[derive(Clone, Copy, Debug)]
struct AttributeCandidate {
    record: Record,
    chunk_start: usize,
    chunk_end: usize,
    chunk_length_field: usize,
    chunk_length_width: usize,
}

#[derive(Clone, Copy, Debug)]
struct AlignmentTarget {
    size_field_offset: usize,
    chunk_start: usize,
    chunk_end: usize,
    chunk_length_field: usize,
    chunk_length_width: usize,
}

/// Creates a simple earliest-format file with the reference C library.
fn write_earliest_file(path: &Path) {
    let file = hdf5::File::with_options()
        .with_fapl(|fapl| fapl.libver_earliest())
        .create(path)
        .expect("create earliest-format file with libhdf5");

    let dataset = file
        .new_dataset::<i32>()
        .shape([1])
        .create("data")
        .expect("create control dataset");
    dataset
        .write_raw(&[42])
        .expect("write control dataset value");

    // Creates one optional root-header message after the group has otherwise
    // been populated. Depending on the HDF5 release, the message may reside in
    // chunk 0 or in an object-header continuation chunk.
    let root = file.group("/").expect("open root group");
    let marker = root
        .new_attr::<u8>()
        .shape(())
        .create("crosscheck_marker")
        .expect("create root attribute");
    marker.write_scalar(&7).expect("write root attribute value");

    file.close().expect("close control file");
}

/// Creates an earliest-format file whose root object header uses a continuation.
fn write_earliest_file_with_continuation(path: &Path) {
    let file = hdf5::File::with_options()
        .with_fapl(|fapl| fapl.libver_earliest())
        .create(path)
        .expect("create earliest-format file with libhdf5");

    let dataset = file
        .new_dataset::<i32>()
        .shape([1])
        .create("data")
        .expect("create control dataset");
    dataset
        .write_raw(&[42])
        .expect("write control dataset value");

    // A large attribute added after creation forces the version 1 root object
    // header to allocate additional message storage.
    let root = file.group("/").expect("open root group");
    let marker = root
        .new_attr::<u8>()
        .shape([1024])
        .create("continuation_payload")
        .expect("create large root attribute");
    marker
        .write_raw(&vec![0x5a; 1024])
        .expect("write large root attribute");

    file.close().expect("close control file");
}

/// Writes an unsigned little-endian integer of at most eight bytes.
fn write_uint(bytes: &mut [u8], offset: usize, width: usize, value: u64) {
    assert!(
        (1..=8).contains(&width),
        "unsupported integer width {width}"
    );

    if width < 8 {
        assert!(
            value < (1u64 << (width * 8)),
            "value {value:#x} does not fit in {width} bytes"
        );
    }

    let end = offset
        .checked_add(width)
        .expect("integer field offset overflow");
    let field = bytes
        .get_mut(offset..end)
        .unwrap_or_else(|| panic!("integer at {offset:#x} lies outside the file"));

    for (index, byte) in field.iter_mut().enumerate() {
        *byte = ((value >> (index * 8)) & 0xff) as u8;
    }
}

/// Copies the first root continuation to EOF and adds one trailing byte.
///
/// The continuation message is repointed to the copy and its declared length is
/// increased by one. The copied records themselves remain unchanged, leaving a
/// single byte after the last complete version 1 message prefix and body.
fn add_trailing_byte_to_root_continuation(bytes: &mut Vec<u8>) {
    let layout = file_layout(bytes);
    let root_records = chunk_records(bytes, layout.root_chunk_start, layout.root_chunk_end);

    let continuation = root_records
        .into_iter()
        .find(|record| record.msg_type == CONTINUATION_MESSAGE)
        .expect("large root attribute did not create a version 1 continuation");

    let continuation_fields_len = layout
        .offset_size
        .checked_add(layout.length_size)
        .expect("continuation field width overflow");

    assert!(
        usize::from(continuation.body_size) >= continuation_fields_len,
        "continuation message at {:#x} is too short for its address and length",
        continuation.offset
    );

    let original_address = read_uint(bytes, continuation.body_start, layout.offset_size);
    let original_length = read_uint(
        bytes,
        continuation.body_start + layout.offset_size,
        layout.length_size,
    );

    let original_start =
        usize::try_from(original_address).expect("continuation address exceeds usize");
    let original_length =
        usize::try_from(original_length).expect("continuation length exceeds usize");
    let original_end = original_start
        .checked_add(original_length)
        .expect("continuation end overflow");

    assert!(
        original_end <= bytes.len(),
        "valid continuation {original_start:#x}..{original_end:#x} \
         extends beyond EOF {:#x}",
        bytes.len()
    );

    // Verifies that the original continuation consists entirely of complete
    // version 1 records before introducing the malformed trailing byte.
    let records = chunk_records(bytes, original_start, original_end);
    assert!(
        !records.is_empty(),
        "generated continuation contains no object-header messages"
    );

    let original_chunk = bytes[original_start..original_end].to_vec();

    // Places the replacement continuation at an aligned address. Any alignment
    // padding is unreachable file space, not part of the continuation.
    let alignment_padding = (8 - bytes.len() % 8) % 8;
    bytes.resize(bytes.len() + alignment_padding, 0);

    let replacement_start = bytes.len();
    bytes.extend_from_slice(&original_chunk);
    bytes.push(0);

    let replacement_length = original_length
        .checked_add(1)
        .expect("replacement continuation length overflow");

    assert_eq!(
        &bytes[replacement_start..replacement_start + original_length],
        original_chunk.as_slice(),
        "replacement continuation differs from the valid source chunk"
    );
    assert_eq!(
        bytes[replacement_start + original_length],
        0,
        "replacement continuation does not end with the expected trailing byte"
    );

    write_uint(
        bytes,
        continuation.body_start,
        layout.offset_size,
        replacement_start as u64,
    );
    write_uint(
        bytes,
        continuation.body_start + layout.offset_size,
        layout.length_size,
        replacement_length as u64,
    );

    let declared_eof = bytes.len() as u64;
    write_uint(
        bytes,
        layout.eof_address_field,
        layout.offset_size,
        declared_eof,
    );

    assert_eq!(
        read_uint(bytes, continuation.body_start, layout.offset_size),
        replacement_start as u64,
        "continuation address was not updated"
    );
    assert_eq!(
        read_uint(
            bytes,
            continuation.body_start + layout.offset_size,
            layout.length_size,
        ),
        replacement_length as u64,
        "continuation length was not updated"
    );

    // The original records occupy exactly the original length. The replacement
    // therefore contains one byte that cannot begin an eight-byte v1 prefix.
    assert_eq!(
        replacement_length - original_length,
        1,
        "malformed continuation must contain exactly one trailing byte"
    );
}

/// Reads a little-endian `u16` at `offset`.
fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    let end = offset.checked_add(2).expect("u16 offset overflow");
    let field = bytes
        .get(offset..end)
        .unwrap_or_else(|| panic!("u16 at {offset:#x} lies outside the file"));

    u16::from_le_bytes(field.try_into().unwrap())
}

/// Reads a little-endian `u32` at `offset`.
fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    let end = offset.checked_add(4).expect("u32 offset overflow");
    let field = bytes
        .get(offset..end)
        .unwrap_or_else(|| panic!("u32 at {offset:#x} lies outside the file"));

    u32::from_le_bytes(field.try_into().unwrap())
}

/// Reads an unsigned little-endian integer of at most eight bytes.
fn read_uint(bytes: &[u8], offset: usize, width: usize) -> u64 {
    assert!(
        (1..=8).contains(&width),
        "unsupported integer width {width}"
    );

    let end = offset
        .checked_add(width)
        .expect("integer field offset overflow");
    let field = bytes
        .get(offset..end)
        .unwrap_or_else(|| panic!("integer at {offset:#x} lies outside the file"));

    field.iter().enumerate().fold(0u64, |value, (index, byte)| {
        value | (u64::from(*byte) << (index * 8))
    })
}

/// Locates the root group's version 1 object header.
///
/// Versions 0 and 1 of the superblock contain a root-group symbol-table entry.
/// Its second address field identifies the root object header. The generated
/// file has no userblock, so its base address is expected to be zero.
fn file_layout(bytes: &[u8]) -> FileLayout {
    assert!(
        bytes.len() >= 24,
        "libhdf5 produced an unexpectedly short file"
    );
    assert_eq!(
        &bytes[..HDF5_SIGNATURE.len()],
        &HDF5_SIGNATURE,
        "libhdf5 did not produce an HDF5 signature at offset zero"
    );

    let superblock_version = bytes[8];
    assert!(
        matches!(superblock_version, 0 | 1),
        "earliest-format libhdf5 output used superblock version \
         {superblock_version}, expected version 0 or 1"
    );

    let offset_size = usize::from(bytes[13]);
    let length_size = usize::from(bytes[14]);

    assert!(
        (1..=8).contains(&offset_size),
        "unexpected HDF5 offset width {offset_size}"
    );
    assert!(
        (1..=8).contains(&length_size),
        "unexpected HDF5 length width {length_size}"
    );

    // Superblock version 1 adds the indexed-storage K value and its reserved
    // field before the address fields.
    let address_fields: usize = match superblock_version {
        0 => 24,
        1 => 32,
        _ => panic!("unreachable"),
    };
    let eof_address_field = address_fields
        .checked_add(2 * offset_size)
        .expect("end-of-file address field offset overflow");

    let base_address = read_uint(bytes, address_fields, offset_size);
    assert_eq!(
        base_address, 0,
        "the generated control file unexpectedly has a nonzero base address"
    );

    // Four superblock addresses precede the root-group symbol-table entry.
    // The entry starts with the link-name offset followed by the object-header
    // address.
    let root_entry = address_fields
        .checked_add(4 * offset_size)
        .expect("root symbol-table entry offset overflow");
    let root_header_address_field = root_entry
        .checked_add(offset_size)
        .expect("root object-header address field overflow");

    let root_header_address = read_uint(bytes, root_header_address_field, offset_size);
    let root_header_offset =
        usize::try_from(root_header_address).expect("root object-header address exceeds usize");

    let version = *bytes.get(root_header_offset).unwrap_or_else(|| {
        panic!("root object header at {root_header_offset:#x} is outside the file")
    });
    assert_eq!(
        version, 1,
        "earliest-format root object header has version {version}, expected version 1"
    );

    let header_data_size = usize::try_from(read_u32(bytes, root_header_offset + 8)).unwrap();

    let root_chunk_start = root_header_offset
        .checked_add(V1_HEADER_PREFIX_LEN)
        .expect("object-header chunk offset overflow");
    let root_chunk_end = root_chunk_start
        .checked_add(header_data_size)
        .expect("object-header chunk end overflow");

    assert!(
        root_chunk_end <= bytes.len(),
        "valid root object-header chunk {root_chunk_start:#x}..{root_chunk_end:#x} \
         extends beyond EOF {:#x}",
        bytes.len()
    );

    FileLayout {
        offset_size,
        length_size,
        eof_address_field,
        root_chunk_start,
        root_chunk_end,
    }
}

/// Parses the records physically stored in one version 1 object-header chunk.
fn chunk_records(bytes: &[u8], chunk_start: usize, chunk_end: usize) -> Vec<Record> {
    assert!(
        chunk_start <= chunk_end && chunk_end <= bytes.len(),
        "invalid object-header chunk {chunk_start:#x}..{chunk_end:#x}"
    );

    let mut records = Vec::new();
    let mut pos = chunk_start;

    while pos < chunk_end {
        let remaining = chunk_end - pos;
        assert!(
            remaining >= V1_MESSAGE_PREFIX_LEN,
            "valid control chunk ends with only {remaining} bytes at {pos:#x}"
        );

        let msg_type = read_u16(bytes, pos);
        let body_size = read_u16(bytes, pos + 2);

        assert_eq!(
            body_size % 8,
            0,
            "libhdf5 wrote unaligned v1 message size {body_size} at {pos:#x}"
        );

        let body_start = pos
            .checked_add(V1_MESSAGE_PREFIX_LEN)
            .expect("message body offset overflow");
        let body_end = body_start
            .checked_add(usize::from(body_size))
            .expect("message body end overflow");

        assert!(
            body_end <= chunk_end,
            "valid control message at {pos:#x} overruns its chunk: \
             body_end={body_end:#x}, chunk_end={chunk_end:#x}"
        );

        records.push(Record {
            offset: pos,
            msg_type,
            body_size,
            body_start,
            body_end,
        });

        pos = body_end;
    }

    assert_eq!(
        pos, chunk_end,
        "control records do not end at the declared chunk boundary"
    );

    records
}

/// Walks version 1 object-header chunks and locates the root attribute.
///
/// A candidate Attribute message may be followed only by Nil messages in its
/// physical chunk. This keeps the mutation from covering any later meaningful
/// message when the Attribute body is extended to cross the chunk boundary.
fn find_root_attribute(bytes: &[u8], layout: FileLayout) -> AttributeCandidate {
    fn walk(
        bytes: &[u8],
        layout: FileLayout,
        chunk_start: usize,
        chunk_end: usize,
        chunk_length_field: usize,
        chunk_length_width: usize,
        visited: &mut BTreeSet<(usize, usize)>,
        attribute_count: &mut usize,
        saw_symbol_table: &mut bool,
        candidate: &mut Option<AttributeCandidate>,
    ) {
        assert!(
            visited.insert((chunk_start, chunk_end)),
            "object-header continuation cycle or duplicate chunk \
             {chunk_start:#x}..{chunk_end:#x}"
        );

        let records = chunk_records(bytes, chunk_start, chunk_end);

        for (index, record) in records.iter().copied().enumerate() {
            if record.msg_type == SYMBOL_TABLE_MESSAGE {
                *saw_symbol_table = true;
            }

            if record.msg_type == ATTRIBUTE_MESSAGE {
                *attribute_count += 1;

                let trailing_records_are_nil = records[index + 1..]
                    .iter()
                    .all(|record| record.msg_type == NIL_MESSAGE);

                if trailing_records_are_nil {
                    assert!(
                        candidate.is_none(),
                        "multiple root Attribute messages are suitable mutation targets"
                    );
                    *candidate = Some(AttributeCandidate {
                        record,
                        chunk_start,
                        chunk_end,
                        chunk_length_field,
                        chunk_length_width,
                    });
                }
            }
        }

        for record in records {
            if record.msg_type != CONTINUATION_MESSAGE {
                continue;
            }

            let fields_len = layout
                .offset_size
                .checked_add(layout.length_size)
                .expect("continuation field width overflow");

            assert!(
                usize::from(record.body_size) >= fields_len,
                "continuation message at {:#x} is too short for its address and length",
                record.offset
            );

            let continuation_address = read_uint(bytes, record.body_start, layout.offset_size);
            let continuation_length = read_uint(
                bytes,
                record.body_start + layout.offset_size,
                layout.length_size,
            );

            let continuation_start =
                usize::try_from(continuation_address).expect("continuation address exceeds usize");
            let continuation_length =
                usize::try_from(continuation_length).expect("continuation length exceeds usize");
            let continuation_end = continuation_start
                .checked_add(continuation_length)
                .expect("continuation end overflow");

            assert!(
                continuation_end <= bytes.len(),
                "valid continuation {continuation_start:#x}..{continuation_end:#x} \
                 extends beyond EOF {:#x}",
                bytes.len()
            );

            walk(
                bytes,
                layout,
                continuation_start,
                continuation_end,
                record.body_start + layout.offset_size,
                layout.length_size,
                visited,
                attribute_count,
                saw_symbol_table,
                candidate,
            );
        }
    }

    let mut visited = BTreeSet::new();
    let mut attribute_count = 0;
    let mut saw_symbol_table = false;
    let mut candidate = None;

    let root_header_offset = layout
        .root_chunk_start
        .checked_sub(V1_HEADER_PREFIX_LEN)
        .expect("root object-header prefix offset underflow");
    let root_chunk_length_field = root_header_offset
        .checked_add(8)
        .expect("object-header data-size field offset overflow");

    walk(
        bytes,
        layout,
        layout.root_chunk_start,
        layout.root_chunk_end,
        root_chunk_length_field,
        4,
        &mut visited,
        &mut attribute_count,
        &mut saw_symbol_table,
        &mut candidate,
    );

    assert!(
        saw_symbol_table,
        "generated earliest-format root header has no Symbol Table message"
    );
    assert_eq!(
        attribute_count, 1,
        "generated root header contains {attribute_count} Attribute messages, expected one"
    );

    candidate.expect("the root Attribute message is followed by a meaningful message in its chunk")
}

/// Extends the root Attribute message eight bytes beyond its containing chunk.
///
/// Returns the offset of the two-byte size field so the caller can verify that
/// no unrelated file bytes changed.
fn corrupt_root_attribute_message(bytes: &mut [u8]) -> usize {
    let layout = file_layout(bytes);
    let candidate = find_root_attribute(bytes, layout);
    let record = candidate.record;

    assert_eq!(
        record.msg_type, ATTRIBUTE_MESSAGE,
        "mutation target is not an Attribute message"
    );
    assert!(
        record.body_end <= candidate.chunk_end,
        "control Attribute message already crosses its chunk boundary"
    );

    let malformed_body_size = candidate
        .chunk_end
        .checked_sub(record.body_start)
        .and_then(|size| size.checked_add(8))
        .expect("malformed Attribute size overflow");

    let malformed_body_size =
        u16::try_from(malformed_body_size).expect("malformed Attribute size exceeds u16");

    assert!(
        malformed_body_size > record.body_size,
        "malformed Attribute size must exceed its original size"
    );
    assert_eq!(
        malformed_body_size % 8,
        0,
        "malformed Attribute size must remain eight-byte aligned"
    );

    let malformed_body_end = record
        .body_start
        .checked_add(usize::from(malformed_body_size))
        .expect("malformed Attribute body end overflow");

    assert_eq!(
        malformed_body_end,
        candidate.chunk_end + 8,
        "mutation must extend exactly eight bytes beyond its containing chunk"
    );

    let size_field_offset = record.offset + 2;
    bytes[size_field_offset..size_field_offset + 2]
        .copy_from_slice(&malformed_body_size.to_le_bytes());

    size_field_offset
}

/// Replaces the final root Attribute message with an aligned Nil message.
///
/// The message's prefix location, declared size, and body bytes remain unchanged.
/// This produces a controlled record whose body has no message-specific semantics.
fn replace_final_root_attribute_with_nil(bytes: &mut [u8]) -> AlignmentTarget {
    let layout = file_layout(bytes);
    let candidate = find_root_attribute(bytes, layout);
    let record = candidate.record;

    assert_eq!(
        record.msg_type, ATTRIBUTE_MESSAGE,
        "mutation target is not an Attribute message"
    );
    assert_eq!(
        record.body_end, candidate.chunk_end,
        "control Attribute message is not the final physical record in its chunk"
    );
    assert_eq!(
        record.body_size % 8,
        0,
        "control Attribute message size is not eight-byte aligned"
    );

    bytes[record.offset..record.offset + 2].copy_from_slice(&NIL_MESSAGE.to_le_bytes());

    AlignmentTarget {
        size_field_offset: record.offset + 2,
        chunk_start: candidate.chunk_start,
        chunk_end: candidate.chunk_end,
        chunk_length_field: candidate.chunk_length_field,
        chunk_length_width: candidate.chunk_length_width,
    }
}

/// Makes the final controlled Nil message one byte shorter and unaligned.
///
/// The containing chunk is shortened by the same byte, so the malformed Nil
/// record still ends exactly at the declared chunk boundary.
fn corrupt_final_root_nil_message_alignment(bytes: &mut [u8], target: AlignmentTarget) {
    let records = chunk_records(bytes, target.chunk_start, target.chunk_end);
    let record = records
        .last()
        .copied()
        .expect("target object-header chunk contains no messages");

    assert_eq!(
        record.msg_type, NIL_MESSAGE,
        "alignment target is not the final Nil message"
    );
    assert_eq!(
        record.offset + 2,
        target.size_field_offset,
        "final Nil message moved from its controlled location"
    );
    assert_eq!(
        record.body_end, target.chunk_end,
        "final Nil message does not end at the chunk boundary"
    );
    assert_eq!(
        record.body_size % 8,
        0,
        "control Nil message size is not eight-byte aligned"
    );

    let declared_chunk_length =
        read_uint(bytes, target.chunk_length_field, target.chunk_length_width);
    let physical_chunk_length = target
        .chunk_end
        .checked_sub(target.chunk_start)
        .expect("target chunk length underflow");

    assert_eq!(
        declared_chunk_length, physical_chunk_length as u64,
        "target chunk's declared length does not match its physical extent"
    );

    let malformed_body_size = record
        .body_size
        .checked_sub(1)
        .expect("malformed Nil message size underflow");

    assert_ne!(
        malformed_body_size % 8,
        0,
        "malformed Nil message size must not be eight-byte aligned"
    );

    bytes[target.size_field_offset..target.size_field_offset + 2]
        .copy_from_slice(&malformed_body_size.to_le_bytes());

    let malformed_chunk_length = physical_chunk_length
        .checked_sub(1)
        .expect("malformed chunk length underflow");

    write_uint(
        bytes,
        target.chunk_length_field,
        target.chunk_length_width,
        malformed_chunk_length as u64,
    );

    let malformed_body_end = record
        .body_start
        .checked_add(usize::from(malformed_body_size))
        .expect("malformed Nil message body end overflow");
    let malformed_chunk_end = target
        .chunk_start
        .checked_add(malformed_chunk_length)
        .expect("malformed chunk end overflow");

    assert_eq!(
        malformed_body_end, malformed_chunk_end,
        "malformed Nil message does not end at the shortened chunk boundary"
    );
    assert_eq!(
        read_u16(bytes, record.offset),
        NIL_MESSAGE,
        "malformed alignment target is not a Nil message"
    );
    assert_eq!(
        read_u16(bytes, target.size_field_offset),
        malformed_body_size,
        "malformed Nil message has the wrong declared size"
    );
    assert_eq!(
        read_uint(bytes, target.chunk_length_field, target.chunk_length_width,),
        malformed_chunk_length as u64,
        "containing chunk has the wrong malformed length"
    );
}

/// Reads the control dataset through the reference C library.
fn read_with_c(path: &Path) -> Result<Vec<i32>, hdf5::Error> {
    let file = hdf5::File::open(path)?;
    let dataset = file.dataset("data")?;
    dataset.read_raw::<i32>()
}

/// Reads the control dataset through the reference C library under explicit
/// file-format version bounds.
#[cfg(any(feature = "__hdf5-1.14", feature = "__hdf5-2"))]
fn read_with_c_libver(
    path: &Path,
    low: LibraryVersion,
    high: LibraryVersion,
) -> Result<Vec<i32>, hdf5::Error> {
    let file = hdf5::File::with_options()
        .with_fapl(|fapl| fapl.libver_bounds(low, high))
        .open(path)?;
    let dataset = file.dataset("data")?;
    dataset.read_raw::<i32>()
}

/// Reads the control dataset through the buffered pure-Rust backend.
fn read_with_pure_buffered(path: &Path) -> Result<Vec<i32>, hdf5_pure::Error> {
    let file = hdf5_pure::File::open(path)?;
    file.dataset("data")?.read_i32()
}

/// Reads the control dataset through the streaming pure-Rust backend.
fn read_with_pure_streaming(path: &Path) -> Result<Vec<i32>, hdf5_pure::Error> {
    let file = hdf5_pure::File::open_streaming(path)?;
    file.dataset("data")?.read_i32()
}

#[test]
fn a_v1_message_body_overrun_is_rejected_by_all_readers() {
    hdf5::silence_errors(true);

    let dir = tempdir().unwrap();
    let valid_path = dir.path().join("valid.h5");
    let malformed_path = dir.path().join("malformed.h5");

    write_earliest_file(&valid_path);

    // Establishes the control before mutating anything
    assert_eq!(
        read_with_c(&valid_path).unwrap(),
        vec![42],
        "libhdf5 cannot read its own valid control file"
    );
    assert_eq!(
        read_with_pure_buffered(&valid_path).unwrap(),
        vec![42],
        "buffered hdf5-pure cannot read the valid libhdf5 control file"
    );
    assert_eq!(
        read_with_pure_streaming(&valid_path).unwrap(),
        vec![42],
        "streaming hdf5-pure cannot read the valid libhdf5 control file"
    );

    let valid = fs::read(&valid_path).unwrap();
    let mut malformed = valid.clone();
    let size_field_offset = corrupt_root_attribute_message(&mut malformed);

    assert_eq!(
        malformed.len(),
        valid.len(),
        "object-header corruption must not change the file length"
    );
    assert_ne!(
        malformed, valid,
        "object-header corruption did not alter the file"
    );

    // Only the two-byte message-size field may change. One of the two bytes may
    // happen to retain its previous value.
    for (offset, (before, after)) in valid.iter().zip(&malformed).enumerate() {
        if before != after {
            assert!(
                (size_field_offset..size_field_offset + 2).contains(&offset),
                "mutation changed byte {offset:#x} outside the Attribute \
                 message-size field at {size_field_offset:#x}"
            );
        }
    }

    fs::write(&malformed_path, &malformed).unwrap();

    let c_result = read_with_c(&malformed_path);
    assert!(
        c_result.is_err(),
        "libhdf5 {:?} accepted a v1 Attribute message whose body overruns \
         its declared object-header chunk: {c_result:?}",
        hdf5::library_version()
    );

    let buffered_result = read_with_pure_buffered(&malformed_path);
    assert!(
        buffered_result.is_err(),
        "buffered hdf5-pure accepted a v1 Attribute message whose body overruns \
         its declared object-header chunk: {buffered_result:?}"
    );

    let streaming_result = read_with_pure_streaming(&malformed_path);
    assert!(
        streaming_result.is_err(),
        "streaming hdf5-pure accepted a v1 Attribute message whose body overruns \
         its declared object-header chunk: {streaming_result:?}"
    );
}

#[test]
fn a_v1_continuation_with_a_trailing_partial_prefix_has_versioned_libhdf5_behavior() {
    hdf5::silence_errors(true);

    let dir = tempdir().unwrap();
    let valid_path = dir.path().join("valid.h5");
    let malformed_path = dir.path().join("malformed.h5");

    write_earliest_file_with_continuation(&valid_path);

    // Establishes that the generated continuation is valid for all readers
    // before changing its declared extent.
    assert_eq!(
        read_with_c(&valid_path).unwrap(),
        vec![42],
        "libhdf5 cannot read its own valid continuation fixture"
    );
    assert_eq!(
        read_with_pure_buffered(&valid_path).unwrap(),
        vec![42],
        "buffered hdf5-pure cannot read the valid continuation fixture"
    );
    assert_eq!(
        read_with_pure_streaming(&valid_path).unwrap(),
        vec![42],
        "streaming hdf5-pure cannot read the valid continuation fixture"
    );

    let mut malformed = fs::read(&valid_path).unwrap();
    add_trailing_byte_to_root_continuation(&mut malformed);
    fs::write(&malformed_path, malformed).unwrap();

    let version = hdf5::library_version();
    let c_result = read_with_c(&malformed_path);

    if version < (1, 14, 0) {
        assert_eq!(
            c_result.unwrap(),
            vec![42],
            "libhdf5 {version:?} rejected the legacy-accepted v1 continuation \
             ending with a partial message prefix"
        );
    } else {
        assert!(
            c_result.is_err(),
            "libhdf5 {version:?} accepted a v1 continuation ending with a partial \
             message prefix: {c_result:?}"
        );
    }

    // hdf5-pure deliberately follows the stricter behavior of libhdf5 1.14
    // and later for continuation boundaries.
    let buffered_result = read_with_pure_buffered(&malformed_path);
    assert!(
        buffered_result.is_err(),
        "buffered hdf5-pure accepted a v1 continuation ending with a partial \
         message prefix: {buffered_result:?}"
    );

    let streaming_result = read_with_pure_streaming(&malformed_path);
    assert!(
        streaming_result.is_err(),
        "streaming hdf5-pure accepted a v1 continuation ending with a partial \
         message prefix: {streaming_result:?}"
    );
}

#[cfg(any(feature = "__hdf5-1.14", feature = "__hdf5-2"))]
#[test]
fn a_v1_continuation_partial_prefix_rejection_is_independent_of_libver_bounds() {
    hdf5::silence_errors(true);

    let dir = tempdir().unwrap();
    let valid_path = dir.path().join("valid.h5");
    let malformed_path = dir.path().join("malformed.h5");

    write_earliest_file_with_continuation(&valid_path);

    let mut malformed = fs::read(&valid_path).unwrap();
    add_trailing_byte_to_root_continuation(&mut malformed);
    fs::write(&malformed_path, malformed).unwrap();

    let version = hdf5::library_version();

    // The default file-access properties establish the modern reference
    // behavior before explicit bounds are varied.
    assert_eq!(
        read_with_c(&valid_path).unwrap(),
        vec![42],
        "libhdf5 {version:?} cannot read the valid continuation fixture \
         with default libver bounds"
    );
    let default_result = read_with_c(&malformed_path);
    assert!(
        default_result.is_err(),
        "libhdf5 {version:?} accepted the malformed continuation with \
         default libver bounds: {default_result:?}"
    );

    // An upper bound of V18 requests HDF5 1.8-era compatibility.
    assert_eq!(
        read_with_c_libver(&valid_path, LibraryVersion::Earliest, LibraryVersion::V18).unwrap(),
        vec![42],
        "libhdf5 {version:?} cannot read the valid continuation fixture \
         with Earliest..V18 bounds"
    );
    let v18_result = read_with_c_libver(
        &malformed_path,
        LibraryVersion::Earliest,
        LibraryVersion::V18,
    );
    assert!(
        v18_result.is_err(),
        "libhdf5 {version:?} accepted the malformed continuation with \
         Earliest..V18 bounds: {v18_result:?}"
    );

    // An upper bound of V110 requests HDF5 1.10-era compatibility.
    assert_eq!(
        read_with_c_libver(&valid_path, LibraryVersion::Earliest, LibraryVersion::V110,).unwrap(),
        vec![42],
        "libhdf5 {version:?} cannot read the valid continuation fixture \
         with Earliest..V110 bounds"
    );
    let v110_result = read_with_c_libver(
        &malformed_path,
        LibraryVersion::Earliest,
        LibraryVersion::V110,
    );
    assert!(
        v110_result.is_err(),
        "libhdf5 {version:?} accepted the malformed continuation with \
         Earliest..V110 bounds: {v110_result:?}"
    );

    // An upper bound of V112 requests HDF5 1.12-era compatibility.
    assert_eq!(
        read_with_c_libver(&valid_path, LibraryVersion::Earliest, LibraryVersion::V112).unwrap(),
        vec![42],
        "libhdf5 {version:?} cannot read the valid continuation fixture \
         with Earliest..V112 bounds"
    );
    let v112_result = read_with_c_libver(
        &malformed_path,
        LibraryVersion::Earliest,
        LibraryVersion::V112,
    );
    assert!(
        v112_result.is_err(),
        "libhdf5 {version:?} accepted the malformed continuation with \
         Earliest..V112 bounds: {v112_result:?}"
    );

    // An upper bound of V114 requests HDF5 1.14-era compatibility.
    assert_eq!(
        read_with_c_libver(&valid_path, LibraryVersion::Earliest, LibraryVersion::V114).unwrap(),
        vec![42],
        "libhdf5 {version:?} cannot read the valid continuation fixture \
         with Earliest..V114 bounds"
    );
    let v114_result = read_with_c_libver(
        &malformed_path,
        LibraryVersion::Earliest,
        LibraryVersion::V114,
    );
    assert!(
        v114_result.is_err(),
        "libhdf5 {version:?} accepted the malformed continuation with \
         Earliest..V114 bounds: {v114_result:?}"
    );

    // Finally, pin the strictest bounds the linked release exposes.
    let latest = LibraryVersion::latest();
    assert_eq!(
        read_with_c_libver(&valid_path, latest, latest).unwrap(),
        vec![42],
        "libhdf5 {version:?} cannot read the valid continuation fixture \
         with {latest:?}..{latest:?} bounds"
    );
    let latest_result = read_with_c_libver(&malformed_path, latest, latest);
    assert!(
        latest_result.is_err(),
        "libhdf5 {version:?} accepted the malformed continuation with \
         {latest:?}..{latest:?} bounds: {latest_result:?}"
    );
}

#[test]
fn a_v1_message_size_not_aligned_to_eight_has_versioned_libhdf5_behavior() {
    hdf5::silence_errors(true);

    let dir = tempdir().unwrap();
    let source_path = dir.path().join("source.h5");
    let valid_path = dir.path().join("valid.h5");
    let malformed_path = dir.path().join("malformed.h5");

    write_earliest_file(&source_path);

    let mut valid = fs::read(&source_path).unwrap();
    let target = replace_final_root_attribute_with_nil(&mut valid);
    fs::write(&valid_path, &valid).unwrap();

    assert_eq!(
        read_with_c(&valid_path).unwrap(),
        vec![42],
        "libhdf5 cannot read the controlled Nil-message fixture"
    );
    assert_eq!(
        read_with_pure_buffered(&valid_path).unwrap(),
        vec![42],
        "buffered hdf5-pure cannot read the controlled Nil-message fixture"
    );
    assert_eq!(
        read_with_pure_streaming(&valid_path).unwrap(),
        vec![42],
        "streaming hdf5-pure cannot read the controlled Nil-message fixture"
    );

    let mut malformed = valid.clone();
    corrupt_final_root_nil_message_alignment(&mut malformed, target);

    assert_eq!(
        malformed.len(),
        valid.len(),
        "alignment corruption must not change the physical file length"
    );

    for (offset, (before, after)) in valid.iter().zip(&malformed).enumerate() {
        if before == after {
            continue;
        }

        let size_field = target.size_field_offset..target.size_field_offset + 2;
        let chunk_length_field =
            target.chunk_length_field..target.chunk_length_field + target.chunk_length_width;

        assert!(
            size_field.contains(&offset) || chunk_length_field.contains(&offset),
            "mutation changed byte {offset:#x} outside the Nil message size \
             or containing chunk-length field"
        );
    }

    fs::write(&malformed_path, &malformed).unwrap();

    let version = hdf5::library_version();
    let c_result = read_with_c(&malformed_path);

    if version < (1, 10, 0) {
        assert_eq!(
            c_result.unwrap(),
            vec![42],
            "libhdf5 {version:?} rejected the legacy-accepted unaligned \
             version 1 Nil message"
        );
    } else {
        assert!(
            c_result.is_err(),
            "libhdf5 {version:?} accepted an unaligned version 1 Nil \
             message: {c_result:?}"
        );
    }

    // hdf5-pure follows the stricter behavior of libhdf5 1.10+
    let buffered_result = read_with_pure_buffered(&malformed_path);
    assert!(
        matches!(
            buffered_result,
            Err(hdf5_pure::Error::Format(
                hdf5_pure::FormatError::InvalidObjectHeaderMessageSize(47)
            ))
        ),
        "buffered hdf5-pure reported the wrong result for an unaligned \
         version 1 Nil message: {buffered_result:?}"
    );

    let streaming_result = read_with_pure_streaming(&malformed_path);
    assert!(
        matches!(
            streaming_result,
            Err(hdf5_pure::Error::Format(
                hdf5_pure::FormatError::InvalidObjectHeaderMessageSize(47)
            ))
        ),
        "streaming hdf5-pure reported the wrong result for an unaligned \
         version 1 Nil message: {streaming_result:?}"
    );
}

#[cfg(any(feature = "__hdf5-1.14", feature = "__hdf5-2"))]
#[test]
fn a_v1_message_size_alignment_validation_is_independent_of_libver_bounds() {
    hdf5::silence_errors(true);

    let dir = tempdir().unwrap();
    let source_path = dir.path().join("source.h5");
    let valid_path = dir.path().join("valid.h5");
    let malformed_path = dir.path().join("malformed.h5");

    write_earliest_file(&source_path);

    let mut valid = fs::read(&source_path).unwrap();
    let target = replace_final_root_attribute_with_nil(&mut valid);
    fs::write(&valid_path, &valid).unwrap();

    let mut malformed = valid;
    corrupt_final_root_nil_message_alignment(&mut malformed, target);
    fs::write(&malformed_path, malformed).unwrap();

    let version = hdf5::library_version();

    assert_eq!(
        read_with_c(&valid_path).unwrap(),
        vec![42],
        "libhdf5 {version:?} cannot read the valid Nil-message fixture \
         with default libver bounds"
    );
    let default_result = read_with_c(&malformed_path);
    assert!(
        default_result.is_err(),
        "libhdf5 {version:?} accepted the unaligned version 1 Nil message \
         with default libver bounds: {default_result:?}"
    );

    assert_eq!(
        read_with_c_libver(&valid_path, LibraryVersion::Earliest, LibraryVersion::V18,).unwrap(),
        vec![42],
        "libhdf5 {version:?} cannot read the valid Nil-message fixture \
         with Earliest..V18 bounds"
    );
    let v18_result = read_with_c_libver(
        &malformed_path,
        LibraryVersion::Earliest,
        LibraryVersion::V18,
    );
    assert!(
        v18_result.is_err(),
        "libhdf5 {version:?} accepted the unaligned version 1 Nil message \
         with Earliest..V18 bounds: {v18_result:?}"
    );

    assert_eq!(
        read_with_c_libver(&valid_path, LibraryVersion::Earliest, LibraryVersion::V110,).unwrap(),
        vec![42],
        "libhdf5 {version:?} cannot read the valid Nil-message fixture \
         with Earliest..V110 bounds"
    );
    let v110_result = read_with_c_libver(
        &malformed_path,
        LibraryVersion::Earliest,
        LibraryVersion::V110,
    );
    assert!(
        v110_result.is_err(),
        "libhdf5 {version:?} accepted the unaligned version 1 Nil message \
         with Earliest..V110 bounds: {v110_result:?}"
    );

    assert_eq!(
        read_with_c_libver(&valid_path, LibraryVersion::Earliest, LibraryVersion::V112,).unwrap(),
        vec![42],
        "libhdf5 {version:?} cannot read the valid Nil-message fixture \
         with Earliest..V112 bounds"
    );
    let v112_result = read_with_c_libver(
        &malformed_path,
        LibraryVersion::Earliest,
        LibraryVersion::V112,
    );
    assert!(
        v112_result.is_err(),
        "libhdf5 {version:?} accepted the unaligned version 1 Nil message \
         with Earliest..V112 bounds: {v112_result:?}"
    );

    assert_eq!(
        read_with_c_libver(&valid_path, LibraryVersion::Earliest, LibraryVersion::V114,).unwrap(),
        vec![42],
        "libhdf5 {version:?} cannot read the valid Nil-message fixture \
         with Earliest..V114 bounds"
    );
    let v114_result = read_with_c_libver(
        &malformed_path,
        LibraryVersion::Earliest,
        LibraryVersion::V114,
    );
    assert!(
        v114_result.is_err(),
        "libhdf5 {version:?} accepted the unaligned version 1 Nil message \
         with Earliest..V114 bounds: {v114_result:?}"
    );

    let latest = LibraryVersion::latest();

    assert_eq!(
        read_with_c_libver(&valid_path, latest, latest).unwrap(),
        vec![42],
        "libhdf5 {version:?} cannot read the valid Nil-message fixture \
         with {latest:?}..{latest:?} bounds"
    );
    let latest_result = read_with_c_libver(&malformed_path, latest, latest);
    assert!(
        latest_result.is_err(),
        "libhdf5 {version:?} accepted the unaligned version 1 Nil message \
         with {latest:?}..{latest:?} bounds: {latest_result:?}"
    );
}
