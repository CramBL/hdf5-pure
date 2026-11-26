use rstest::rstest;
use test_util::fractal_heap;
use test_util::widths::Widths;

use super::*;
use crate::address::{StoredAddress, StoredAddressExt};
use crate::fractal_heap::FractalHeapHeader;
use crate::width::{LengthWidth, OffsetWidth};

#[rstest]
#[case::short_link(7, 6)]
#[case::long_link(7, 8)]
#[case::short_attribute(8, 7)]
#[case::long_attribute(8, 9)]
fn heap_id_preparation_requires_the_declared_id_width(
    #[case] heap_id_length: u16,
    #[case] actual: usize,
) {
    let header = heap_id_header(heap_id_length);
    let layout = FractalHeapIdLayout::new(&header, OFFSET_WIDTH, LENGTH_WIDTH);
    let bytes = vec![0; actual];
    let error = PreparedHeapId::new(&layout, &bytes).err().unwrap();
    assert_eq!(
        error,
        FormatError::InvalidFractalHeapIdLength {
            expected: usize::from(heap_id_length),
            actual
        }
    );
}

#[rstest]
#[case::managed(false)]
#[case::huge(true)]
fn heap_id_preparation_borrows_complete_generated_ids(
    #[case] huge: bool,
    #[values(7, 8)] heap_id_length: u16,
) {
    let header = heap_id_header(heap_id_length);
    let layout = FractalHeapIdLayout::new(&header, OFFSET_WIDTH, LENGTH_WIDTH);
    let bytes = if huge {
        header
            .encode_huge_id(3, OFFSET_WIDTH, LENGTH_WIDTH)
            .unwrap()
    } else {
        header.encode_managed_id(256, 17).unwrap()
    };
    assert_eq!(bytes.len(), usize::from(heap_id_length));
    let id = PreparedHeapId::new(&layout, &bytes).unwrap();
    assert_eq!(id.bytes(), bytes);
    assert_eq!(id.bytes().as_ptr(), bytes.as_ptr());
}

fn heap_id_header(heap_id_length: u16) -> FractalHeapHeader {
    let mut header = FractalHeapHeader::parse_bytes(
        &fractal_heap::Header::new(256).build(Widths::EIGHT),
        OFFSET_WIDTH,
        LENGTH_WIDTH,
    )
    .unwrap();
    header.heap_id_length = heap_id_length;
    header.btree_huge_objects_address = StoredAddress::new(1024);
    header
}

const OFFSET_WIDTH: OffsetWidth = OffsetWidth::Eight;
const LENGTH_WIDTH: LengthWidth = LengthWidth::Eight;
