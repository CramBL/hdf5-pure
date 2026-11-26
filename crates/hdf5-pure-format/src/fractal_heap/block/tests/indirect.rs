use alloc::vec::Vec;

use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::fractal_heap;
use test_util::widths::Widths;

use super::super::FILTER_MASK_LEN;
use super::support;
use crate::checksum;
use crate::error::FormatError;
use crate::fractal_heap::block::{CHECKSUM_LEN, FractalHeapBlock};
use crate::fractal_heap::{FractalHeapHeader, FractalHeapLayout};
use crate::width::{LengthWidth, OffsetWidth};

#[rstest]
#[case::two(OffsetWidth::Two, false)]
#[case::four(OffsetWidth::Four, true)]
#[case::eight(OffsetWidth::Eight, false)]
fn indirect_views_derive_direct_and_indirect_children(
    #[case] offsets: OffsetWidth,
    #[case] filtered: bool,
) {
    let heap = support::heap(offsets, 8, false, filtered);
    let bytes =
        support::indirect_bytes(&heap, &[(0, 0, 512), (1, 2, 640), (5, 1, 768), (7, 3, 900)]);
    let reference = support::indirect_root(&heap)
        .at(PhysicalAddress::new(256))
        .unwrap();
    let view = reference.parse(support::region(256, &bytes)).unwrap();
    let children = view.children().collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(children.len(), 4);
    let FractalHeapBlock::Direct(first) = children[0] else {
        panic!("expected direct child");
    };
    assert_eq!(first.address().get(), 512);
    assert_eq!(first.heap_offset(), 0);
    assert_eq!(first.decoded_size(), 128);
    assert_eq!(first.filter_mask(), filtered.then_some(1));
    assert_eq!(
        first
            .at(PhysicalAddress::new(1 << 40))
            .unwrap()
            .range()
            .length(),
        if filtered { 47 } else { 128 }
    );
    let FractalHeapBlock::Direct(second) = children[1] else {
        panic!("expected direct child");
    };
    assert_eq!(second.heap_offset(), 768);
    let FractalHeapBlock::Indirect(third) = children[2] else {
        panic!("expected indirect child");
    };
    assert_eq!(third.address().get(), 768);
    assert_eq!(third.rows(), 3);
    assert_eq!(third.heap_offset(), heap.layout.slot_offset(5, 1).unwrap());
    let FractalHeapBlock::Indirect(fourth) = children[3] else {
        panic!("expected indirect child");
    };
    assert_eq!(fourth.rows(), 5);
    assert_eq!(fourth.heap_offset(), heap.layout.slot_offset(7, 3).unwrap());
    assert_eq!(view.child(0, 1).unwrap(), None);
    assert_eq!(
        view.child(8, 0).unwrap_err(),
        FormatError::InvalidFractalHeapGeometry
    );
    assert_eq!(
        view.child(0, 4).unwrap_err(),
        FormatError::InvalidFractalHeapGeometry
    );
    assert_eq!(view.bytes().as_ptr(), bytes.as_ptr());
}
#[rstest]
#[case::size(8, 0, false)]
#[case::mask(16, 2, false)]
#[case::undefined_address(0, 0xFF, true)]
fn filtered_children_reject_invalid_encoded_geometry(
    #[case] at: usize,
    #[case] value: u8,
    #[case] undefined: bool,
) {
    let heap = support::heap(OffsetWidth::Eight, 1, false, true);
    let mut bytes = support::indirect_bytes(&heap, &[(0, 0, 512)]);
    if undefined {
        bytes[heap.prefix_len()..heap.prefix_len() + usize::from(heap.layout.offset_width().get())]
            .fill(value);
    } else {
        bytes[heap.prefix_len() + at] = value;
    }
    let checksum_at = bytes.len() - 4;
    let computed = checksum::jenkins_lookup3(&bytes[..checksum_at]);
    bytes[checksum_at..].copy_from_slice(&computed.to_le_bytes());
    let view = support::indirect_root(&heap)
        .at(PhysicalAddress::new(256))
        .unwrap()
        .parse(support::region(256, &bytes))
        .unwrap();
    assert_eq!(
        view.child(0, 0).unwrap_err(),
        FormatError::InvalidFractalHeapGeometry
    );
    assert_eq!(
        view.children().collect::<Result<Vec<_>, _>>().unwrap_err(),
        FormatError::InvalidFractalHeapGeometry
    );
}

#[test]
fn absent_filtered_children_accept_unused_mask_bits() {
    let heap = support::heap(OffsetWidth::Eight, 1, false, true);
    let mut bytes = support::indirect_bytes(&heap, &[(0, 1, 512)]);
    let mask_at = heap.prefix_len()
        + usize::from(heap.layout.offset_width().get())
        + usize::from(heap.layout.length_width().get());
    bytes[mask_at..mask_at + FILTER_MASK_LEN].copy_from_slice(&u32::MAX.to_le_bytes());
    let checksum_at = bytes.len() - CHECKSUM_LEN;
    let computed = checksum::jenkins_lookup3(&bytes[..checksum_at]);
    bytes[checksum_at..].copy_from_slice(&computed.to_le_bytes());
    let view = support::indirect_root(&heap)
        .at(PhysicalAddress::new(256))
        .unwrap()
        .parse(support::region(256, &bytes))
        .unwrap();
    assert_eq!(view.child(0, 0).unwrap(), None);
    assert_eq!(
        view.children()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .len(),
        1
    );
}
#[rstest]
#[case::two(LengthWidth::Two)]
#[case::four(LengthWidth::Four)]
#[case::eight(LengthWidth::Eight)]
fn filtered_entry_lengths_use_the_superblock_width(#[case] lengths: LengthWidth) {
    let mut heap = support::heap(OffsetWidth::Four, 2, false, true);
    let encoded = fractal_heap::Header::new(256).build(Widths::EIGHT);
    let mut header =
        FractalHeapHeader::parse_bytes(&encoded, OffsetWidth::Eight, LengthWidth::Eight).unwrap();
    header.filtering = heap.filtering.clone();
    heap.layout = FractalHeapLayout::new(&header, OffsetWidth::Four, lengths).unwrap();
    let bytes = support::indirect_bytes(&heap, &[(1, 2, 700)]);
    let view = support::indirect_root(&heap)
        .at(PhysicalAddress::new(768))
        .unwrap()
        .parse(support::region(768, &bytes))
        .unwrap();
    let Some(FractalHeapBlock::Direct(child)) = view.child(1, 2).unwrap() else {
        panic!("expected filtered direct child");
    };
    assert_eq!(child.address().get(), 700);
    assert_eq!(
        child
            .at(PhysicalAddress::new(1212))
            .unwrap()
            .range()
            .length(),
        47
    );
    assert_eq!(child.filter_mask(), Some(1));
}
