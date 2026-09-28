#![cfg(feature = "hdf5")]
use std::ops::Range;

use hdf5::plist::file_create::Sizeof;
use hdf5::plist::file_create::SizeofInfo;
use hdf5::types::VarLenUnicode;
use hdf5_pure_format::LengthWidth;
use rstest::rstest;

#[rstest]
#[case::two_byte_lengths(Sizeof::Bytes2, LengthWidth::Two)]
#[case::four_byte_lengths(Sizeof::Bytes4, LengthWidth::Four)]
fn a_narrow_length_collection_matches_the_one_libhdf5_writes(
    #[case] sizeof_size: Sizeof,
    #[case] width: LengthWidth,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vlen.h5");
    let file = hdf5::File::with_options()
        .with_create_plist(|p| {
            p.sizes(SizeofInfo {
                sizeof_addr: Sizeof::Bytes8,
                sizeof_size,
            })
        })
        .create(&path)
        .unwrap();
    file.new_dataset::<VarLenUnicode>()
        .shape(())
        .create("string")
        .unwrap()
        .write_scalar(&PAYLOAD.parse::<VarLenUnicode>().unwrap())
        .unwrap();
    file.close().unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let start = bytes
        .windows(4)
        .position(|window| window == b"GCOL")
        .expect("libhdf5 writes the string into a global heap collection");
    let written = &bytes[start..start + COLLECTION_SIZE];
    let encoded =
        hdf5_pure_format::encode_global_heap_collection(width, &[PAYLOAD.as_bytes()]).unwrap();

    let fields = written_fields(width);
    assert_eq!(encoded.len(), COLLECTION_SIZE);
    assert_eq!(
        fields.clone().map(|range| &written[range]),
        fields.map(|range| &encoded[range])
    );
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
