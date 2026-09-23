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

use std::fs;
use std::path::Path;

#[cfg(any(feature = "__hdf5-1.14", feature = "__hdf5-2"))]
use hdf5::file::LibraryVersion;
use tempfile::tempdir;
use test_util::bytes;
use test_util::object_header::{MessageType, v1};
use test_util::superblock::v0;
use test_util_hdf5::file;

/// A root Attribute message the tests edit, and the chunk that stores it.
#[derive(Clone, Debug)]
struct AlignmentTarget {
    chunk: v1::Chunk,
    record: v1::Record,
}

/// Creates a simple earliest-format file with the reference C library.
fn write_earliest_file(path: &Path) {
    let file = file::libhdf5_create_earliest(path);

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
    let file = file::libhdf5_create_earliest(path);

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

/// Copies the first root continuation to EOF and adds one trailing byte.
///
/// The continuation message is repointed to the copy and its declared length is
/// increased by one. The copied records themselves remain unchanged, leaving a
/// single byte after the last complete version 1 message prefix and body.
fn add_trailing_byte_to_root_continuation(bytes: &mut Vec<u8>) {
    let superblock = v0::Fields::read(bytes, 0);
    let offset_width = superblock.widths.offset;
    let chunks = v1::root_group_chunks(bytes);

    let continuation = chunks[0]
        .records
        .iter()
        .find(|record| record.msg_type == MessageType::OBJECT_HEADER_CONTINUATION)
        .expect("large root attribute did not create a version 1 continuation");
    let original = chunks
        .iter()
        .find(|chunk| chunk.length_field.at == continuation.body.start + offset_width)
        .expect("the walk reaches the chunk the continuation names");

    // Verifies that the original continuation consists entirely of complete
    // version 1 records before introducing the malformed trailing byte.
    assert!(
        !original.records.is_empty(),
        "generated continuation contains no object-header messages"
    );
    let original_chunk = bytes[original.range.clone()].to_vec();

    // Places the replacement continuation at an aligned address. Any alignment
    // padding is unreachable file space, not part of the continuation.
    bytes.resize(bytes.len().next_multiple_of(8), 0);
    let replacement_start = bytes.len();
    bytes.extend_from_slice(&original_chunk);
    bytes.push(0);

    bytes::set_uint_at(
        bytes,
        continuation.body.start,
        offset_width,
        replacement_start as u64,
    );
    original
        .length_field
        .set(bytes, original_chunk.len() as u64 + 1);
    let declared_eof = bytes.len() as u64;
    bytes::set_uint_at(bytes, superblock.eof_address_at, offset_width, declared_eof);
}

/// Walks version 1 object-header chunks and locates the root attribute.
///
/// A candidate Attribute message may be followed only by Nil messages in its
/// physical chunk. This keeps the mutation from covering any later meaningful
/// message when the Attribute body is extended to cross the chunk boundary.
fn find_root_attribute(bytes: &[u8]) -> AlignmentTarget {
    let chunks = v1::root_group_chunks(bytes);
    let records = || chunks.iter().flat_map(|chunk| &chunk.records);

    for record in records() {
        assert_eq!(
            record.body.len() % 8,
            0,
            "libhdf5 wrote unaligned v1 message size {} at {:#x}",
            record.body.len(),
            record.at
        );
    }
    assert!(
        records().any(|record| record.msg_type == MessageType::SYMBOL_TABLE),
        "generated earliest-format root header has no Symbol Table message"
    );
    let attribute_count = records()
        .filter(|record| record.msg_type == MessageType::ATTRIBUTE)
        .count();
    assert_eq!(
        attribute_count, 1,
        "generated root header contains {attribute_count} Attribute messages, expected one"
    );

    chunks
        .iter()
        .find_map(|chunk| {
            let index = chunk
                .records
                .iter()
                .position(|record| record.msg_type == MessageType::ATTRIBUTE)?;
            chunk.records[index + 1..]
                .iter()
                .all(|record| record.msg_type == MessageType::NIL)
                .then(|| AlignmentTarget {
                    chunk: chunk.clone(),
                    record: chunk.records[index].clone(),
                })
        })
        .expect("the root Attribute message is followed by a meaningful message in its chunk")
}

/// Extends the root Attribute message eight bytes beyond its containing chunk.
///
/// Returns the offset of the two-byte size field so the caller can verify that
/// no unrelated file bytes changed.
fn corrupt_root_attribute_message(bytes: &mut [u8]) -> usize {
    let AlignmentTarget { chunk, record } = find_root_attribute(bytes);

    let malformed_body_size = u16::try_from(chunk.range.end - record.body.start + 8)
        .expect("malformed Attribute size exceeds u16");
    assert!(
        usize::from(malformed_body_size) > record.body.len(),
        "malformed Attribute size must exceed its original size"
    );

    record.set_body_size(bytes, malformed_body_size);
    record.size_at()
}

/// Replaces the final root Attribute message with an aligned Nil message.
///
/// The message's prefix location, declared size, and body bytes remain unchanged.
/// This produces a controlled record whose body has no message-specific semantics.
fn replace_final_root_attribute_with_nil(bytes: &mut [u8]) -> AlignmentTarget {
    let target = find_root_attribute(bytes);
    assert_eq!(
        target.record.body.end, target.chunk.range.end,
        "control Attribute message is not the final physical record in its chunk"
    );

    target.record.set_type(bytes, MessageType::NIL);
    target
}

/// Makes the final controlled Nil message one byte shorter and unaligned.
///
/// The containing chunk is shortened by the same byte, so the malformed Nil
/// record still ends exactly at the declared chunk boundary.
fn corrupt_final_root_nil_message_alignment(bytes: &mut [u8], target: &AlignmentTarget) {
    let chunks = v1::root_group_chunks(bytes);
    let chunk = chunks
        .iter()
        .find(|chunk| chunk.range == target.chunk.range)
        .expect("the target chunk is still part of the root header");
    let record = chunk
        .records
        .last()
        .expect("target object-header chunk contains no messages");
    assert_eq!(
        (record.at, record.msg_type),
        (target.record.at, MessageType::NIL),
        "alignment target is not the final Nil message"
    );

    let malformed_body_size = u16::try_from(record.body.len() - 1).expect("a v1 message size");
    record.set_body_size(bytes, malformed_body_size);
    chunk
        .length_field
        .set(bytes, (chunk.range.len() - 1) as u64);
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
    corrupt_final_root_nil_message_alignment(&mut malformed, &target);

    assert_eq!(
        malformed.len(),
        valid.len(),
        "alignment corruption must not change the physical file length"
    );

    for (offset, (before, after)) in valid.iter().zip(&malformed).enumerate() {
        if before == after {
            continue;
        }

        let size_field = target.record.size_at()..target.record.size_at() + 2;
        let length_field = target.chunk.length_field;
        let chunk_length_field = length_field.at..length_field.at + length_field.width;

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
    corrupt_final_root_nil_message_alignment(&mut malformed, &target);
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
