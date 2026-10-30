#![cfg(feature = "hdf5")]
//! Cross-checks malformed version 1 object-header message records.
//!
//! libhdf5 creates earliest-format files whose root groups use
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
use hdf5::{File as NativeFile, MajorErrorCode, MinorErrorCode};
use hdf5_pure::{Error, FormatError};
use rstest::rstest;

use test_util::bytes;
use test_util::object_header::{MessageType, v1};
use test_util::superblock::v0;
use test_util_hdf5::file;
use test_util_hdf5::reader::ReadBackend;

/// Deliberately malformed version 1 object-header records.
#[derive(Clone, Copy, Debug)]
enum Malformation {
    /// A declared message body extending beyond its containing chunk.
    BodyOverrun {
        /// The bytes needed for the prefix and declared body.
        expected: usize,
        /// The bytes from the record prefix to the chunk's end.
        available: usize,
    },
    /// A continuation ending with one byte of a message prefix.
    PartialPrefix,
    /// A final Nil message with a 47-byte body.
    UnalignedBody,
}

impl Malformation {
    // H5O_chunk_deserialize in H5Ocache.c, HDF5 1.8.23, and H5O__chunk_deserialize in
    // H5Ocache.c, HDF5 1.10.11, 1.12.3, 1.14.6 and 2.2.0, report these errors while
    // decoding object header messages.
    /// Asserts a file-open failure with the expected innermost libhdf5 format error.
    ///
    /// # Panics
    ///
    /// Panics if `failure` has another stage, lacks a native error frame, or has an
    /// innermost frame that differs from the expected category, function, or diagnostic.
    fn assert_native_rejection(self, failure: NativeReadFailure) {
        let NativeReadFailure { stage, error } = failure;
        assert_eq!(stage, NativeReadStage::FileOpen, "{error:?}");
        let version = hdf5::library_version();
        let (minor, description) = match self {
            Self::BodyOverrun { .. } if version >= (2, 2, 0) => {
                (MinorErrorCode::BadValue, "message size exceeds buffer end")
            }
            Self::BodyOverrun { .. } => (MinorErrorCode::CantInit, "corrupt object header"),
            Self::PartialPrefix => (
                MinorErrorCode::BadMessage,
                "gap found in early version of file format",
            ),
            Self::UnalignedBody => (MinorErrorCode::CantLoad, "message not aligned"),
        };
        let function = if version < (1, 10, 0) {
            "H5O_chunk_deserialize"
        } else {
            "H5O__chunk_deserialize"
        };
        let Some(stack) = error.stack() else {
            panic!("expected a native HDF5 error stack, got {error:?}");
        };
        let frame = stack.last().expect("native rejection has no error frames");
        assert_eq!(
            (
                frame.major_code(),
                frame.minor_code(),
                frame.func(),
                frame.desc()
            ),
            (MajorErrorCode::ObjectHeader, minor, function, description),
            "libhdf5 {version:?}: {error:?}"
        );
    }

    /// Asserts the pure parser's complete format error for this malformation.
    ///
    /// # Panics
    ///
    /// Panics if `error` has another variant or different fields.
    fn assert_pure_rejection(self, error: Error) {
        let Error::Format(actual) = error else {
            panic!("expected a format error, got {error:?}");
        };
        assert_eq!(
            actual,
            match self {
                Self::BodyOverrun {
                    expected,
                    available,
                } => FormatError::UnexpectedEof {
                    expected,
                    available,
                },
                Self::PartialPrefix => FormatError::UnexpectedEof {
                    expected: v1::RECORD_PREFIX,
                    available: 1,
                },
                Self::UnalignedBody => FormatError::InvalidObjectHeaderMessageSize(47),
            }
        );
    }
}

/// A native read failure with its operation stage.
#[derive(Debug)]
struct NativeReadFailure {
    stage: NativeReadStage,
    error: hdf5::Error,
}

/// The operation that failed while reading the control dataset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeReadStage {
    DatasetOpen,
    DatasetRead,
    FileClose,
    FileOpen,
}

/// The format-version bounds selected for a native read.
#[cfg(any(feature = "__hdf5-1.14", feature = "__hdf5-2"))]
#[derive(Clone, Copy, Debug)]
enum Bounds {
    Default,
    Latest,
    V110,
    V112,
    V114,
    V18,
}

#[cfg(any(feature = "__hdf5-1.14", feature = "__hdf5-2"))]
impl Bounds {
    /// Reads the control dataset with the selected format-version bounds.
    ///
    /// # Errors
    ///
    /// Returns the failed operation's native error with its stage.
    fn read(self, path: &Path) -> Result<Vec<i32>, NativeReadFailure> {
        let high = match self {
            Self::Default => return read_with_c(path),
            Self::Latest => LibraryVersion::latest(),
            Self::V18 => LibraryVersion::V18,
            Self::V110 => LibraryVersion::V110,
            Self::V112 => LibraryVersion::V112,
            Self::V114 => LibraryVersion::V114,
        };
        let low = if matches!(self, Self::Latest) {
            high
        } else {
            LibraryVersion::Earliest
        };
        read_with_c_libver(path, low, high)
    }
}

/// A root Attribute message the tests edit, and the chunk that stores it.
#[derive(Clone, Debug)]
struct AlignmentTarget {
    chunk: v1::Chunk,
    record: v1::Record,
}

/// Creates a simple earliest-format file with libhdf5.
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

    drop(marker);
    drop(root);
    drop(dataset);
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

    drop(marker);
    drop(root);
    drop(dataset);
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
    assert_eq!(
        original
            .records
            .last()
            .expect("generated continuation contains no object-header messages")
            .body
            .end,
        original.range.end,
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
/// Returns the two-byte size field's offset and the expected EOF error's byte counts.
///
/// # Panics
///
/// Panics if the root attribute cannot be located or its extended body size cannot
/// be represented by the two-byte field.
fn corrupt_root_attribute_message(bytes: &mut [u8]) -> (usize, Malformation) {
    let AlignmentTarget { chunk, record } = find_root_attribute(bytes);

    let malformed_body_size = u16::try_from(chunk.range.end - record.body.start + 8)
        .expect("malformed Attribute size exceeds u16");
    assert!(
        usize::from(malformed_body_size) > record.body.len(),
        "malformed Attribute size must exceed its original size"
    );

    record.set_body_size(bytes, malformed_body_size);
    (
        record.size_at(),
        Malformation::BodyOverrun {
            expected: v1::RECORD_PREFIX + usize::from(malformed_body_size),
            available: chunk.range.end - record.at,
        },
    )
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
    assert_eq!(malformed_body_size, 47);
    record.set_body_size(bytes, malformed_body_size);
    chunk
        .length_field
        .set(bytes, (chunk.range.len() - 1) as u64);
}

/// Reads the control dataset through libhdf5.
///
/// # Errors
///
/// Returns the failed operation's native error with its stage.
fn read_with_c(path: &Path) -> Result<Vec<i32>, NativeReadFailure> {
    let file = NativeFile::open(path).map_err(|error| NativeReadFailure {
        stage: NativeReadStage::FileOpen,
        error,
    })?;
    read_native_dataset(file)
}

/// Reads the control dataset through libhdf5 under explicit file-format version bounds.
///
/// # Errors
///
/// Returns the failed operation's native error with its stage.
#[cfg(any(feature = "__hdf5-1.14", feature = "__hdf5-2"))]
fn read_with_c_libver(
    path: &Path,
    low: LibraryVersion,
    high: LibraryVersion,
) -> Result<Vec<i32>, NativeReadFailure> {
    let file = NativeFile::with_options()
        .with_fapl(|fapl| fapl.libver_bounds(low, high))
        .open(path)
        .map_err(|error| NativeReadFailure {
            stage: NativeReadStage::FileOpen,
            error,
        })?;
    read_native_dataset(file)
}

/// Reads `data` and checks file closure after a successful read.
///
/// # Errors
///
/// Returns a failure with its stage if opening or reading the dataset, or closing the file, fails.
fn read_native_dataset(file: NativeFile) -> Result<Vec<i32>, NativeReadFailure> {
    let dataset = file.dataset("data").map_err(|error| NativeReadFailure {
        stage: NativeReadStage::DatasetOpen,
        error,
    })?;
    let values = dataset
        .read_raw::<i32>()
        .map_err(|error| NativeReadFailure {
            stage: NativeReadStage::DatasetRead,
            error,
        })?;
    drop(dataset);
    file.close().map_err(|error| NativeReadFailure {
        stage: NativeReadStage::FileClose,
        error,
    })?;
    Ok(values)
}

/// Reads the control dataset through the selected pure backend.
///
/// # Errors
///
/// Returns an error if the file or dataset cannot be opened, or its values cannot be read.
fn read_with_pure(path: &Path, backend: ReadBackend) -> Result<Vec<i32>, Error> {
    let file = backend.open(path)?;
    file.dataset("data")?.read_i32()
}

#[rstest]
#[case::buffered(ReadBackend::Buffered)]
#[case::streaming(ReadBackend::Streaming)]
fn a_v1_message_body_overrun_is_rejected_by_all_readers(#[case] backend: ReadBackend) {
    hdf5::silence_errors(true);

    let dir = tempfile::tempdir().unwrap();
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
        read_with_pure(&valid_path, backend).unwrap(),
        vec![42],
        "hdf5-pure {backend:?} cannot read the valid libhdf5 control file"
    );

    let valid = fs::read(&valid_path).unwrap();
    let mut malformed = valid.clone();
    let (size_field_offset, malformation) = corrupt_root_attribute_message(&mut malformed);

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

    malformation.assert_native_rejection(read_with_c(&malformed_path).unwrap_err());
    malformation.assert_pure_rejection(read_with_pure(&malformed_path, backend).unwrap_err());
}

#[rstest]
#[case::buffered(ReadBackend::Buffered)]
#[case::streaming(ReadBackend::Streaming)]
fn a_v1_continuation_with_a_trailing_partial_prefix_has_versioned_libhdf5_behavior(
    #[case] backend: ReadBackend,
) {
    hdf5::silence_errors(true);

    let dir = tempfile::tempdir().unwrap();
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
        read_with_pure(&valid_path, backend).unwrap(),
        vec![42],
        "hdf5-pure {backend:?} cannot read the valid continuation fixture"
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
        Malformation::PartialPrefix.assert_native_rejection(c_result.unwrap_err());
    }

    // hdf5-pure deliberately follows the stricter behavior of libhdf5 1.14
    // and later for continuation boundaries.
    Malformation::PartialPrefix
        .assert_pure_rejection(read_with_pure(&malformed_path, backend).unwrap_err());
}

#[cfg(any(feature = "__hdf5-1.14", feature = "__hdf5-2"))]
#[rstest]
#[case::default(Bounds::Default)]
#[case::v18(Bounds::V18)]
#[case::v110(Bounds::V110)]
#[case::v112(Bounds::V112)]
#[case::v114(Bounds::V114)]
#[case::latest(Bounds::Latest)]
fn a_v1_continuation_partial_prefix_rejection_is_independent_of_libver_bounds(
    #[case] bounds: Bounds,
) {
    hdf5::silence_errors(true);

    let dir = tempfile::tempdir().unwrap();
    let valid_path = dir.path().join("valid.h5");
    let malformed_path = dir.path().join("malformed.h5");

    write_earliest_file_with_continuation(&valid_path);

    let mut malformed = fs::read(&valid_path).unwrap();
    add_trailing_byte_to_root_continuation(&mut malformed);
    fs::write(&malformed_path, malformed).unwrap();

    assert_eq!(bounds.read(&valid_path).unwrap(), [42]);
    Malformation::PartialPrefix.assert_native_rejection(bounds.read(&malformed_path).unwrap_err());
}

#[rstest]
#[case::buffered(ReadBackend::Buffered)]
#[case::streaming(ReadBackend::Streaming)]
fn a_v1_message_size_not_aligned_to_eight_has_versioned_libhdf5_behavior(
    #[case] backend: ReadBackend,
) {
    hdf5::silence_errors(true);

    let dir = tempfile::tempdir().unwrap();
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
        read_with_pure(&valid_path, backend).unwrap(),
        vec![42],
        "hdf5-pure {backend:?} cannot read the controlled Nil-message fixture"
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
        Malformation::UnalignedBody.assert_native_rejection(c_result.unwrap_err());
    }

    // hdf5-pure follows the stricter behavior of libhdf5 1.10+
    Malformation::UnalignedBody
        .assert_pure_rejection(read_with_pure(&malformed_path, backend).unwrap_err());
}

#[cfg(any(feature = "__hdf5-1.14", feature = "__hdf5-2"))]
#[rstest]
#[case::default(Bounds::Default)]
#[case::v18(Bounds::V18)]
#[case::v110(Bounds::V110)]
#[case::v112(Bounds::V112)]
#[case::v114(Bounds::V114)]
#[case::latest(Bounds::Latest)]
fn a_v1_message_size_alignment_validation_is_independent_of_libver_bounds(#[case] bounds: Bounds) {
    hdf5::silence_errors(true);

    let dir = tempfile::tempdir().unwrap();
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

    assert_eq!(bounds.read(&valid_path).unwrap(), [42]);
    Malformation::UnalignedBody.assert_native_rejection(bounds.read(&malformed_path).unwrap_err());
}
