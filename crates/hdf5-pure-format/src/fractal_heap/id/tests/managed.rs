use rstest::rstest;
use test_util::fractal_heap;
use test_util::widths::Widths;

use crate::error::FormatError;
use crate::fractal_heap::{
    FractalHeapHeader, FractalHeapIdKind, FractalHeapIdLayout, test_support,
};
use crate::width::{LengthWidth, OffsetWidth};

#[test]
fn a_managed_id_decodes_to_its_offset_and_length() {
    let file_data = fractal_heap::heap_with_one_object(b"Hello, World!", Widths::EIGHT);
    let hdr = FractalHeapHeader::parse(&file_data, 0, 8, 8).unwrap();

    // Byte 0 holds version 0 in bits 6-7 and type 0 in bits 4-5. Bytes 1-6 hold the offset in
    // `max_heap_size` (16) bits, then the length: offset 0 and length 13 pack to 0x000D_0000.
    let offset: u64 = 0;
    let length: u64 = 13;
    let payload = offset | (length << hdr.max_heap_size);
    let mut id = vec![0u8; 7];
    id[0] = 0x00; // type=0
    for i in 0..6 {
        id[1 + i] = ((payload >> (i * 8)) & 0xFF) as u8;
    }

    assert_eq!(
        FractalHeapIdLayout::new(&hdr, OffsetWidth::Eight, LengthWidth::Eight)
            .parse(&id)
            .unwrap()
            .kind(),
        FractalHeapIdKind::Managed {
            heap_offset: 0,
            object_length: 13
        },
    );
}

#[rstest]
#[case::zero_bits(0, 9)]
#[case::offset_above_integer(65, 11)]
#[case::missing_length(64, 9)]
fn invalid_managed_id_geometry_is_rejected(#[case] bits: u16, #[case] id_length: u16) {
    let header = FractalHeapHeader {
        heap_id_length: id_length,
        max_heap_size: bits,
        ..test_support::attribute_heap_header()
    };
    assert_eq!(
        FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
            .parse(&vec![0; usize::from(id_length)]),
        Err(FormatError::InvalidFractalHeapGeometry)
    );
    assert_eq!(
        header.encode_managed_id(0, 1),
        Err(FormatError::InvalidFractalHeapGeometry)
    );
}

#[rstest]
#[case::group(32, 7, 0x1234_5678, 0x9ABC, vec![0, 0x78, 0x56, 0x34, 0x12, 0xBC, 0x9A])]
#[case::unaligned(33, 8, 0x1_2345_6789, 0xABCD, vec![0, 0x89, 0x67, 0x45, 0x23, 1, 0xCD, 0xAB])]
#[case::wide_offset(64, 11, u64::MAX, 0xAB, vec![0, 255, 255, 255, 255, 255, 255, 255, 255, 0xAB, 0])]
#[case::padded(8, 12, 0xAB, 0xCDEF, vec![0, 0xAB, 0xEF, 0xCD, 0, 0, 0, 0, 0, 0, 0, 0])]
fn managed_id_fields_are_byte_aligned_and_independent(
    #[case] bits: u16,
    #[case] id_length: u16,
    #[case] heap_offset: u64,
    #[case] object_length: u64,
    #[case] expected: Vec<u8>,
) {
    let header = FractalHeapHeader {
        heap_id_length: id_length,
        max_heap_size: bits,
        ..test_support::attribute_heap_header()
    };
    assert_eq!(
        header.encode_managed_id(heap_offset, object_length),
        Ok(expected.clone())
    );
    assert_eq!(
        FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
            .parse(&expected)
            .unwrap()
            .kind(),
        FractalHeapIdKind::Managed {
            heap_offset,
            object_length
        }
    );
}

#[test]
fn managed_id_padding_is_not_part_of_the_length() {
    let header = FractalHeapHeader {
        heap_id_length: 10,
        ..test_support::attribute_heap_header()
    };
    let id = [0, 42, 0, 0, 0, 0, 13, 0, 0xAA, 0xBB];
    assert_eq!(
        FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
            .parse(&id)
            .unwrap()
            .kind(),
        FractalHeapIdKind::Managed {
            heap_offset: 42,
            object_length: 13
        }
    );
}

#[rstest]
#[case::unused_offset_bits(33, vec![0, 0, 0, 0, 0, 2, 1, 0])]

fn managed_ids_reject_nonzero_unused_offset_bits(#[case] bits: u16, #[case] bytes: Vec<u8>) {
    let header = FractalHeapHeader {
        heap_id_length: u16::try_from(bytes.len()).unwrap(),
        max_heap_size: bits,
        ..test_support::attribute_heap_header()
    };
    assert_eq!(
        FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight).parse(&bytes),
        Err(FormatError::InvalidFractalHeapGeometry)
    );
}

#[rstest]
#[case::the_first_object(0, 13, [0, 0, 0, 0, 0, 0, 13, 0])]
#[case::an_offset_and_a_length(100, 42, [0, 100, 0, 0, 0, 0, 42, 0])]
#[case::the_widest_offset_and_length(
    (1 << 40) - 1,
    u64::from(u16::MAX),
    [0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
)]
fn an_encoded_managed_id_holds_its_offset_then_its_length(
    #[case] heap_offset: u64,
    #[case] length: u64,
    #[case] expected: [u8; 8],
) {
    let header = test_support::attribute_heap_header();
    let id = header.encode_managed_id(heap_offset, length).unwrap();
    assert_eq!(id, expected);
    assert_eq!(
        FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
            .parse(&id)
            .unwrap()
            .kind(),
        FractalHeapIdKind::Managed {
            heap_offset,
            object_length: length
        },
    );
}

#[rstest]
#[case::an_offset_past_the_offset_bits(
    1 << 40,
    1,
    "heap offset 1099511627776 does not fit the heap's 40-bit offsets"
)]
#[case::a_length_past_the_id(0, 1 << 16, "a managed object length does not fit a 8-byte heap ID")]
fn a_managed_id_that_does_not_fit_is_an_error(
    #[case] heap_offset: u64,
    #[case] length: u64,
    #[case] message: &str,
) {
    assert_eq!(
        test_support::attribute_heap_header().encode_managed_id(heap_offset, length),
        Err(FormatError::Internal(message.into()))
    );
}
