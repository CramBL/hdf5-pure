use rstest::rstest;

use super::fixture;
use crate::bytes;
use crate::global_heap::collection::{
    GCOL_SIGNATURE, GCOL_VERSION, GLOBAL_HEAP_MIN_COLLECTION_SIZE,
};
use crate::global_heap::error::GlobalHeapCollectionError;
use crate::global_heap::index::GLOBAL_HEAP_MAX_OBJECTS;
use crate::global_heap::write;
use crate::width::LengthWidth;

#[rstest]
#[case::padded_to_the_minimum(&[b"alpha".as_slice(), b"".as_slice()], GLOBAL_HEAP_MIN_COLLECTION_SIZE)]
#[case::past_the_minimum(&[[7u8; 5000].as_slice()], 5048)]
fn an_encoded_collection_parses_back_to_its_objects(
    #[case] objects: &[&[u8]],
    #[case] size: usize,
) {
    let encoded = write::encode_global_heap_collection(LengthWidth::Eight, objects).unwrap();

    assert_eq!(encoded.len(), size);
    assert_eq!(bytes::read_length(&encoded, 8, 8), Ok(size as u64));
    let expected: Vec<(u16, &[u8])> = (1..).zip(objects.iter().copied()).collect();
    assert_eq!(
        fixture::walk_objects(fixture::region(0, &encoded), LengthWidth::Eight).unwrap(),
        expected
    );

    let free_space = objects
        .iter()
        .map(|object| 16 + object.len().next_multiple_of(8))
        .sum::<usize>()
        + 16;
    assert_eq!(encoded[free_space..free_space + 8], [0; 8]);
    assert_eq!(
        bytes::read_length(&encoded, free_space + 8, 8),
        Ok((size - free_space) as u64)
    );
}

#[rstest]
#[case::two_byte_lengths(LengthWidth::Two)]
#[case::four_byte_lengths(LengthWidth::Four)]
fn the_encoder_pads_narrow_length_headers_to_sixteen_bytes(#[case] width: LengthWidth) {
    let bytes = write::encode_global_heap_collection(width, &[b"abc"]).unwrap();
    let length = usize::from(width.get());

    let mut expected = GCOL_SIGNATURE.to_vec();
    expected.extend_from_slice(&[GCOL_VERSION, 0, 0, 0]);
    expected.extend_from_slice(&4096u64.to_le_bytes()[..length]);
    expected.resize(16, 0);
    expected.extend_from_slice(&[1, 0, 1, 0, 0, 0, 0, 0]);
    expected.extend_from_slice(&3u64.to_le_bytes()[..length]);
    expected.resize(32, 0);
    expected.extend_from_slice(b"abc\0\0\0\0\0");
    // The free space object at byte 40 spans the rest of the 4096 bytes.
    expected.extend_from_slice(&[0; 8]);
    expected.extend_from_slice(&(4096u64 - 40).to_le_bytes()[..length]);
    expected.resize(4096, 0);
    assert_eq!(bytes, expected);
}

// 16 + 16 + 65,480 + 16 bytes fit the 65,535 of a 2-byte length, and 65,481 bytes of data pad
// to 65,488 and a collection of 65,536.
#[rstest]
#[case::fits(65_480, Ok(65_528))]
#[case::overflows(
    65_481,
    Err(GlobalHeapCollectionError::TooLarge {
        size: 65_536,
        length_width: LengthWidth::Two,
    })
)]
fn a_two_byte_length_collection_holds_at_most_its_field_maximum(
    #[case] object_len: usize,
    #[case] expected: Result<usize, GlobalHeapCollectionError>,
) {
    let object = vec![0u8; object_len];

    assert_eq!(
        write::encode_global_heap_collection(LengthWidth::Two, &[&object]).map(|bytes| bytes.len()),
        expected
    );
}

#[test]
fn the_encoder_rejects_more_than_global_heap_max_objects() {
    let objects = vec![b"".as_slice(); GLOBAL_HEAP_MAX_OBJECTS + 1];

    assert_eq!(
        write::encode_global_heap_collection(LengthWidth::Eight, &objects),
        Err(GlobalHeapCollectionError::TooManyObjects { count: 65_536 })
    );
}
