#[cfg(feature = "std")]
use std::io::Cursor;

use rstest::rstest;
use test_util::bytes;
use test_util::global_heap;
use test_util::widths::Widths;

use super::*;
#[cfg(feature = "std")]
use crate::source::ReadSeekSource;
#[cfg(feature = "std")]
use crate::source::ValidatedSource;
use crate::width::LengthWidth;

mod embedded_slots;

const FORMAT_WIDTHS: FormatWidths = FormatWidths {
    offsets: OffsetWidth::Eight,
    lengths: LengthWidth::Eight,
};

fn format_widths(lengths: usize) -> FormatWidths {
    let lengths = match lengths {
        2 => LengthWidth::Two,
        4 => LengthWidth::Four,
        8 => LengthWidth::Eight,
        other => panic!("unsupported test length width {other}"),
    };
    FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths,
    }
}

/// Writes a global heap collection of `(index, data)` objects at `offset`,
/// growing `file_data` to hold it.
fn build_gcol_at(file_data: &mut Vec<u8>, offset: usize, objects: &[(u16, &[u8])]) {
    let objects: Vec<_> = objects
        .iter()
        .map(|&(index, data)| global_heap::Object::new(index, data))
        .collect();
    let collection = global_heap::collection(&objects, WIDTHS);
    file_data.resize(file_data.len().max(offset + collection.len()), 0);
    file_data[offset..offset + collection.len()].copy_from_slice(&collection);
}

/// Builds one variable-length reference per string.
///
/// Each reference stores `collection_address` and a consecutive object index
/// starting at `start_index`.
fn build_vl_refs(
    strings: &[&str],
    collection_address: u64,
    start_index: u16,
    widths: Widths,
) -> Vec<u8> {
    strings
        .iter()
        .enumerate()
        .flat_map(|(index, string)| {
            global_heap::reference(
                string.len() as u32,
                collection_address,
                u32::from(start_index) + index as u32,
                widths,
            )
        })
        .collect()
}

#[test]
fn parse_vl_references_two_elements() {
    let raw = build_vl_refs(&["hello", "world"], 0x1000, 1, WIDTHS);
    let refs = parse_vl_references(&raw, 2, OffsetWidth::Eight).unwrap();
    assert_eq!(refs.len(), 2);
    assert_eq!(refs[0].length, 5);
    assert_eq!(refs[0].collection_address, StoredAddress::new(0x1000));
    assert_eq!(refs[0].object_index, 1);
    assert_eq!(refs[1].length, 5);
    assert_eq!(refs[1].object_index, 2);
}

#[test]
fn read_vl_strings_from_heap() {
    let gcol_offset = 256usize;
    let mut file_data = vec![0u8; 512];
    build_gcol_at(&mut file_data, gcol_offset, &[(1, b"Alice"), (2, b"Bob")]);

    let raw = build_vl_refs(&["Alice", "Bob"], gcol_offset as u64, 1, WIDTHS);
    let strings = read_vl_strings(&file_data, &raw, 2, FORMAT_WIDTHS).unwrap();
    assert_eq!(strings, vec!["Alice", "Bob"]);
}

#[cfg(feature = "std")]
#[test]
fn read_vl_strings_from_seekable_source() {
    let gcol_offset = 256usize;
    let mut file_data = vec![0u8; 512];
    build_gcol_at(&mut file_data, gcol_offset, &[(1, b"Alice"), (2, b"Bob")]);
    let raw = build_vl_refs(&["Alice", "Bob"], gcol_offset as u64, 1, WIDTHS);
    let source = ReadSeekSource::new(Cursor::new(file_data)).unwrap();

    let strings = read_vl_strings_from_source(
        &source,
        &raw,
        2,
        FORMAT_WIDTHS,
        AddressContext::from_effective_base(BaseAddress::ZERO),
        VlenStringReadOptions::default(),
    )
    .unwrap();
    assert_eq!(strings, vec!["Alice", "Bob"]);
}

#[test]
fn null_vl_element_empty_string() {
    // length=0, address=undefined
    let mut raw = Vec::new();
    raw.extend_from_slice(&0u32.to_le_bytes()); // length=0
    raw.extend_from_slice(&u64::MAX.to_le_bytes()); // undefined address
    raw.extend_from_slice(&0u32.to_le_bytes()); // index

    let file_data = vec![0u8; 16];
    let strings = read_vl_strings(&file_data, &raw, 1, FORMAT_WIDTHS).unwrap();
    assert_eq!(strings, vec![""]);
}

#[test]
fn null_vl_element_zero_address() {
    let mut raw = Vec::new();
    raw.extend_from_slice(&0u32.to_le_bytes());
    raw.extend_from_slice(&0u64.to_le_bytes());
    raw.extend_from_slice(&0u32.to_le_bytes());

    let file_data = vec![0u8; 16];
    let strings = read_vl_strings(&file_data, &raw, 1, FORMAT_WIDTHS).unwrap();
    assert_eq!(strings, vec![""]);
}

#[test]
fn parse_vl_references_truncated_error() {
    let raw = vec![0u8; 10]; // too short for 1 element with offset_size=8
    let err = parse_vl_references(&raw, 1, OffsetWidth::Eight).unwrap_err();
    assert!(matches!(err, FormatError::UnexpectedEof { .. }));
}

#[rstest]
#[case::one_object(4096, &[(1, b"alpha".as_slice())], &[(1, 5)], &[b"alpha".as_slice()])]
#[case::an_empty_object(4096, &[(1, b"".as_slice())], &[(1, 0)], &[b"".as_slice()])]
#[case::a_prefix_of_an_object(4096, &[(1, b"alpha".as_slice())], &[(1, 3)], &[b"alp".as_slice()])]
#[case::free_space_ending_at_the_collection_end(
    4096,
    &[(1, b"alpha".as_slice()), (2, b"beta".as_slice())],
    &[(2, 4), (1, 5)],
    &[b"beta".as_slice(), b"alpha".as_slice()]
)]
#[case::one_alignment_past_the_minimum_size(4104, &[(1, b"alpha".as_slice())], &[(1, 5)], &[b"alpha".as_slice()])]
#[case::a_tail_shorter_than_an_object_header(
    4096,
    &[(1, [9u8; 4056].as_slice())],
    &[(1, 4056)],
    &[[9u8; 4056].as_slice()]
)]
fn a_read_returns_the_requested_objects(
    #[values(2, 4, 8)] lengths: usize,
    #[case] size: usize,
    #[case] objects: &[(u16, &[u8])],
    #[case] requests: &[(u16, u32)],
    #[case] expected: &[&[u8]],
) {
    assert_every_reader_returns(
        &collection_file(lengths, Some(size), objects, Patch::None),
        &references(requests),
        lengths,
        BaseAddress::ZERO,
        Ok(expected.iter().copied().map(payload).collect()),
    );
}

#[rstest]
#[case::every_object(&(1..=2000).collect::<Vec<u16>>())]
#[case::a_sparse_lookup(&[1500, 7, 2000])]
fn a_read_finds_objects_among_many(#[values(2, 4, 8)] lengths: usize, #[case] requested: &[u16]) {
    let payloads: Vec<[u8; 2]> = (1..=2000u16).map(u16::to_le_bytes).collect();
    let objects: Vec<(u16, &[u8])> = (1..)
        .zip(payloads.iter().map(<[u8; 2]>::as_slice))
        .collect();
    let requests: Vec<(u16, u32)> = requested.iter().map(|&index| (index, 2)).collect();
    assert_every_reader_returns(
        &collection_file(lengths, None, &objects, Patch::None),
        &references(&requests),
        lengths,
        BaseAddress::ZERO,
        Ok(requested
            .iter()
            .map(|index| payload(&index.to_le_bytes()))
            .collect()),
    );
}

#[rstest]
fn a_read_returns_an_object_larger_than_a_two_byte_length(#[values(4, 8)] lengths: usize) {
    let large = vec![7; 70_000];
    assert_every_reader_returns(
        &collection_file(lengths, None, &[(1, &large)], Patch::None),
        &references(&[(1, 70_000)]),
        lengths,
        BaseAddress::ZERO,
        Ok(vec![payload(&large)]),
    );
}

// Byte 8 holds the collection size, and byte 48 the size field of the second object header.
#[rstest]
#[case::a_collection_size_below_the_minimum(
    &[(1, b"alpha".as_slice())],
    Patch::Length { at: 8, value: 4095 },
    1,
    FormatError::GlobalHeapCollectionTooSmall { collection_address: 512, size: 4095 }
)]
#[case::an_unaligned_collection_size(
    &[(1, b"alpha".as_slice())],
    Patch::Length { at: 8, value: 4100 },
    1,
    FormatError::UnalignedGlobalHeapCollection { collection_address: 512, size: 4100 }
)]
#[case::a_duplicate_index_after_the_requested_one(
    &[(1, b"first".as_slice()), (2, b"second".as_slice()), (1, b"again".as_slice())],
    Patch::None,
    1,
    FormatError::DuplicateGlobalHeapObjectIndex { collection_address: 512, index: 1 }
)]
#[case::bytes_after_the_free_space(
    &[(1, b"alpha".as_slice())],
    Patch::Length { at: 48, value: 4048 },
    1,
    FormatError::MisplacedGlobalHeapFreeSpace {
        address: 552,
        size: 4048,
        collection_end: 4608,
    }
)]
#[case::a_nonzero_index_in_a_short_tail(
    &[(1, [9u8; 4056].as_slice())],
    Patch::Index { at: 4088, value: 7 },
    1,
    FormatError::GlobalHeapObjectOutsideCollection {
        address: 4600,
        index: 7,
        collection_end: 4608,
    }
)]
#[case::a_final_object_past_the_collection_end(
    &[(1, b"first".as_slice()), (2, b"second".as_slice())],
    Patch::Length { at: 48, value: 4041 },
    1,
    FormatError::GlobalHeapObjectOutsideCollection {
        address: 552,
        index: 2,
        collection_end: 4608,
    }
)]
#[case::an_absent_index(
    &[(1, b"first".as_slice())],
    Patch::None,
    2,
    FormatError::GlobalHeapObjectNotFound { collection_address: 512, index: 2 }
)]
#[case::index_zero(
    &[(1, b"first".as_slice())],
    Patch::None,
    0,
    FormatError::GlobalHeapObjectNotFound { collection_address: 512, index: 0 }
)]
fn a_read_rejects_a_malformed_collection_or_an_absent_object(
    #[values(2, 4, 8)] lengths: usize,
    #[case] objects: &[(u16, &[u8])],
    #[case] patch: Patch,
    #[case] index: u16,
    #[case] expected: FormatError,
) {
    assert_every_reader_returns(
        &collection_file(lengths, Some(4096), objects, patch),
        &references(&[(index, 1)]),
        lengths,
        BaseAddress::ZERO,
        Err(expected),
    );
}

#[test]
fn a_collection_past_the_end_of_the_source_is_an_unexpected_eof() {
    let mut file = collection_file(8, None, &[(1, b"alpha")], Patch::None);
    file.truncate(4600);
    assert_every_reader_returns(
        &file,
        &references(&[(1, 5)]),
        8,
        BaseAddress::ZERO,
        Err(FormatError::UnexpectedSourceEof {
            expected: 4608,
            available: 4600,
        }),
    );
}

#[cfg(feature = "std")]
#[test]
fn a_short_source_read_is_a_source_error() {
    /// A source whose metadata reads return one byte fewer than the caller requests.
    struct ShortMetadata(Vec<u8>);

    impl Source for ShortMetadata {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            BytesSource::new(&self.0).read_at(offset, buf)
        }

        fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
            BytesSource::new(&self.0).read_exact_at(offset, len - 1)
        }
    }

    let file = collection_file(8, None, &[(1, b"alpha")], Patch::None);
    let refs = references(&[(1, 5)]);
    assert_eq!(
        read_vl_byte_objects_from_source(
            &ValidatedSource::new(ShortMetadata(file)),
            &refs,
            1,
            FORMAT_WIDTHS,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            1,
            VlenStringReadOptions::default(),
        )
        .unwrap_err(),
        FormatError::Source("the source returned 15 bytes for a 16-byte read at offset 512".into())
    );
}

// The rejection reports the physical address of the collection, 512 + 1024.
#[rstest]
#[case::read(4096, Ok(vec![payload(b"alpha")]))]
#[case::rejected(
    4095,
    Err(FormatError::GlobalHeapCollectionTooSmall { collection_address: 1536, size: 4095 })
)]
fn a_userblock_shifts_the_collection_by_the_base(
    #[case] size: u64,
    #[case] expected: Result<Vec<VlByteObject>, FormatError>,
) {
    let mut collection = global_heap::collection(&[global_heap::Object::new(1, b"alpha")], WIDTHS);
    bytes::set_uint_at(&mut collection, 8, 8, size);
    let mut file = vec![0; USERBLOCK + 1024];
    file.extend_from_slice(&collection);
    let refs = global_heap::reference(5, 1024, 1, WIDTHS);
    assert_every_reader_returns(
        &file,
        &refs,
        8,
        BaseAddress::new(USERBLOCK as u64),
        expected,
    );
}

#[test]
fn a_collection_above_four_gib_is_read_at_its_u64_position() {
    /// A source that holds `bytes` from the position `start` on.
    struct Sparse {
        start: u64,
        bytes: Vec<u8>,
    }

    impl Source for Sparse {
        fn len(&self) -> u64 {
            self.start + self.bytes.len() as u64
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            let at = offset.checked_sub(self.start).unwrap();
            BytesSource::new(&self.bytes).read_at(at, buf)
        }
    }

    let address = (1 << 32) + 4096;
    let source = Sparse {
        start: address,
        bytes: global_heap::collection(
            &[
                global_heap::Object::new(1, b"alpha"),
                global_heap::Object::new(2, b"beta"),
            ],
            WIDTHS,
        ),
    };
    let refs: Vec<u8> = [(5, 1), (4, 2)]
        .into_iter()
        .flat_map(|(len, index)| global_heap::reference(len, address, index, WIDTHS))
        .collect();
    assert_eq!(
        read_vl_byte_objects_from_source(
            &source,
            &refs,
            2,
            FORMAT_WIDTHS,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            1,
            VlenStringReadOptions::default(),
        )
        .unwrap(),
        vec![payload(b"alpha"), payload(b"beta")]
    );
}

/// Returns a file that holds a collection of `objects` at byte [`COLLECTION`], with length
/// fields `lengths` bytes wide and `patch` applied.
///
/// The collection is `size` bytes, or the size [`global_heap::collection`] gives it where
/// `size` is `None`.
fn collection_file(
    lengths: usize,
    size: Option<usize>,
    objects: &[(u16, &[u8])],
    patch: Patch,
) -> Vec<u8> {
    let objects: Vec<_> = objects
        .iter()
        .map(|&(index, data)| global_heap::Object::new(index, data))
        .collect();
    let widths = Widths::new(8, lengths);
    let mut collection = match size {
        Some(size) => global_heap::collection_of_size(&objects, widths, size),
        None => global_heap::collection(&objects, widths),
    };
    match patch {
        Patch::None => {}
        Patch::Index { at, value } => bytes::set_u16_at(&mut collection, at, value),
        Patch::Length { at, value } => {
            bytes::set_uint_at(&mut collection, at, lengths, value);
        }
    }
    let mut file = vec![0; COLLECTION];
    file.extend_from_slice(&collection);
    file
}

/// Returns the references to the `(index, length)` pairs of `requests`, in the collection at
/// [`COLLECTION`].
fn references(requests: &[(u16, u32)]) -> Vec<u8> {
    requests
        .iter()
        .flat_map(|&(index, len)| {
            global_heap::reference(len, COLLECTION as u64, u32::from(index), WIDTHS)
        })
        .collect()
}

/// Asserts that [`read_vl_byte_objects_from_source`] returns `expected` for `refs` in `file`,
/// and [`read_vl_strings_from_source`] `expected` decoded as lossy UTF-8.
///
/// The assertions read `file` through a [`BytesSource`] and, with `std`, a [`ReadSeekSource`].
fn assert_every_reader_returns(
    file: &[u8],
    refs: &[u8],
    lengths: usize,
    base: BaseAddress,
    expected: Result<Vec<VlByteObject>, FormatError>,
) {
    let count = (refs.len() / VL_REF_BYTES) as u64;
    let widths = format_widths(lengths);
    let addresses = AddressContext::from_effective_base(base);
    let options = VlenStringReadOptions::default();
    let objects = |source: &dyn Source| {
        read_vl_byte_objects_from_source(source, refs, count, widths, addresses, 1, options)
    };
    let strings = |source: &dyn Source| {
        read_vl_strings_from_source(source, refs, count, widths, addresses, options)
    };
    let decoded = expected.clone().map(|objects| {
        objects
            .iter()
            .map(|object| match object {
                VlByteObject::Null => String::new(),
                VlByteObject::Bytes(bytes) => String::from_utf8_lossy(bytes).into_owned(),
            })
            .collect::<Vec<_>>()
    });
    let buffered = BytesSource::new(file);
    assert_eq!(objects(&buffered), expected);
    assert_eq!(strings(&buffered), decoded);
    #[cfg(feature = "std")]
    {
        let streamed = ReadSeekSource::new(Cursor::new(file.to_vec())).unwrap();
        assert_eq!(objects(&streamed), expected);
        assert_eq!(strings(&streamed), decoded);
    }
}

fn payload(data: &[u8]) -> VlByteObject {
    VlByteObject::Bytes(data.to_vec())
}

/// A change to the bytes of a collection: the object index or the length field at the offset
/// `at` from its start, set to `value`.
enum Patch {
    Index { at: usize, value: u16 },
    Length { at: usize, value: u64 },
    None,
}

/// The address of the collection in a file [`collection_file`] returns.
const COLLECTION: usize = 512;

/// The size of the userblock in the file of [`a_userblock_shifts_the_collection_by_the_base`].
const USERBLOCK: usize = 512;

const WIDTHS: Widths = Widths::EIGHT;
