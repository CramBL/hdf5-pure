#![cfg(feature = "hdf5")]
use std::ops::Range;
use std::path::Path;

use hdf5::plist::file_create::Sizeof;
use hdf5::types::VarLenUnicode;
use hdf5_pure_format::__private::LengthWidth;
use rstest::rstest;

use super::global_heap_framing;

#[rstest]
#[case::two_byte_lengths(Sizeof::Bytes2, LengthWidth::Two)]
#[case::four_byte_lengths(Sizeof::Bytes4, LengthWidth::Four)]
fn a_narrow_length_collection_matches_the_one_libhdf5_writes(
    #[case] sizeof_size: Sizeof,
    #[case] width: LengthWidth,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vlen.h5");
    let bytes = libhdf5_file(&path, sizeof_size);
    let start = global_heap_framing::collection_starts(&bytes)
        .first()
        .copied()
        .map(|start| usize::try_from(start).unwrap())
        .expect("libhdf5 writes the string into a global heap collection");
    let written = &bytes[start..start + COLLECTION_SIZE];
    let encoded =
        hdf5_pure_format::__private::encode_global_heap_collection(width, &[PAYLOAD.as_bytes()])
            .unwrap();

    let fields = written_fields(width);
    assert_eq!(encoded.len(), COLLECTION_SIZE);
    assert_eq!(
        fields.clone().map(|range| &written[range]),
        fields.map(|range| &encoded[range])
    );
}

#[rstest]
#[case::two_byte_lengths(Sizeof::Bytes2, LengthWidth::Two)]
#[case::four_byte_lengths(Sizeof::Bytes4, LengthWidth::Four)]
#[case::eight_byte_lengths(Sizeof::Bytes8, LengthWidth::Eight)]
fn hdf5_pure_reads_the_string_from_an_encoded_collection_in_a_libhdf5_file(
    #[case] sizeof_size: Sizeof,
    #[case] width: LengthWidth,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vlen.h5");
    let mut bytes = libhdf5_file(&path, sizeof_size);
    let start = global_heap_framing::collection_starts(&bytes)
        .first()
        .copied()
        .map(|start| usize::try_from(start).unwrap())
        .expect("libhdf5 writes the string into a global heap collection");
    let encoded =
        hdf5_pure_format::__private::encode_global_heap_collection(width, &[PAYLOAD.as_bytes()])
            .unwrap();
    bytes[start..start + COLLECTION_SIZE].copy_from_slice(&encoded);

    assert_eq!(
        hdf5_pure::File::from_bytes(bytes)
            .unwrap()
            .dataset("string")
            .unwrap()
            .read_string()
            .unwrap(),
        [PAYLOAD]
    );
}

/// Returns a file in which libhdf5 writes [`PAYLOAD`] to a scalar variable-length string dataset,
/// with lengths `sizeof_size` wide.
fn libhdf5_file(path: &Path, sizeof_size: Sizeof) -> Vec<u8> {
    let file = global_heap_framing::create_libhdf5_file(path, sizeof_size, 0);
    file.new_dataset::<VarLenUnicode>()
        .shape(())
        .create("string")
        .unwrap()
        .write_scalar(&PAYLOAD.parse::<VarLenUnicode>().unwrap())
        .unwrap();
    file.close().unwrap();
    std::fs::read(path).unwrap()
}

fn written_fields(width: LengthWidth) -> [Range<usize>; 5] {
    let header = FIXED_HEADER_LEN + usize::from(width.get());
    let data = OBJECT_1 + HEADER_SIZE;
    let free_space = data + PAYLOAD.len().next_multiple_of(ALIGNMENT);
    [
        0..header,
        // Object 1's reference count is skipped: libhdf5 writes 0 and the encoder 1.
        OBJECT_1..OBJECT_1 + REFERENCE_COUNT_BYTES.start,
        OBJECT_1 + REFERENCE_COUNT_BYTES.end..OBJECT_1 + header,
        data..data + PAYLOAD.len(),
        free_space..free_space + header,
    ]
}

const ALIGNMENT: usize = 8;
const COLLECTION_SIZE: usize = 4096;
const FIXED_HEADER_LEN: usize = 8;
const HEADER_SIZE: usize = 16;
const OBJECT_1: usize = HEADER_SIZE;
const PAYLOAD: &str = "alpha";
const REFERENCE_COUNT_BYTES: Range<usize> = 2..4;
