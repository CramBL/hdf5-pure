use rstest::rstest;

use super::*;
use crate::fractal_heap::{FractalHeapFiltering, test_support};

#[rstest]
#[case::too_short(9, vec![0; 8])]
#[case::too_long(7, vec![0; 8])]
#[case::empty(8, vec![])]
#[case::short_with_unsupported_version(2, vec![0x60])]
#[case::long_with_reserved_flags(7, vec![0x01, 0, 0, 0, 0, 0, 0, 0])]
fn an_id_must_have_the_declared_length(#[case] declared: u16, #[case] bytes: Vec<u8>) {
    let header = FractalHeapHeader {
        heap_id_length: declared,
        ..test_support::attribute_heap_header()
    };
    let layout = FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight);
    assert_eq!(layout.encoded_len(), usize::from(declared));
    assert_eq!(
        layout.parse(&bytes),
        Err(FormatError::InvalidFractalHeapIdLength {
            expected: usize::from(declared),
            actual: bytes.len(),
        })
    );
}

#[rstest]
#[case::reserved_type(vec![0x30, 0, 0, 0, 0, 0, 0, 0], FormatError::InvalidHeapIdType(3))]
#[case::tiny_past_id(vec![0x2F, 0, 0, 0, 0, 0, 0, 0], FormatError::UnexpectedEof { expected: 17, available: 8 })]
fn a_complete_id_with_invalid_fields_is_an_error(
    #[case] bytes: Vec<u8>,
    #[case] error: FormatError,
) {
    assert_eq!(
        FractalHeapIdLayout::new(
            &test_support::attribute_heap_header(),
            OffsetWidth::Eight,
            LengthWidth::Eight
        )
        .parse(&bytes),
        Err(error)
    );
}

#[rstest]
#[case::managed(MANAGED_ID, false)]
#[case::tiny_normal(&[0x20, 0xAB], false)]
#[case::tiny_extended(b"\x20\x01ab...............", false)]
#[case::huge_indirect(INDIRECT_HUGE_ID, false)]
#[case::huge_direct(DIRECT_HUGE_ID, false)]
#[case::filtered_huge_indirect(INDIRECT_HUGE_ID, true)]
#[case::filtered_huge_direct(FILTERED_DIRECT_HUGE_ID, true)]
#[case::reserved_type(&[0x3F], false)]
fn heap_id_parser_checks_the_version_before_the_type(
    #[case] id: &[u8],
    #[case] filtered: bool,
    #[values((0x40, 1), (0x80, 2), (0xC0, 3))] version: (u8, u8),
) {
    let layout = heap_id_fixture_layout(id.len(), filtered);
    let (version_bits, version) = version;
    let mut id = id.to_vec();
    id[0] |= version_bits;
    assert_eq!(
        layout.parse(&id),
        Err(FormatError::UnsupportedFractalHeapIdVersion(version))
    );
}

#[rstest]
#[case::managed(MANAGED_ID, false)]
#[case::huge_indirect(INDIRECT_HUGE_ID, false)]
#[case::huge_direct(DIRECT_HUGE_ID, false)]
#[case::filtered_huge_indirect(INDIRECT_HUGE_ID, true)]
#[case::filtered_huge_direct(FILTERED_DIRECT_HUGE_ID, true)]
fn heap_id_parser_rejects_reserved_bits_in_managed_and_huge_ids(
    #[case] id: &[u8],
    #[case] filtered: bool,
    #[values(0x01, 0x0F)] flags: u8,
) {
    let layout = heap_id_fixture_layout(id.len(), filtered);
    let mut id = id.to_vec();
    id[0] |= flags;
    assert_eq!(
        layout.parse(&id),
        Err(FormatError::InvalidFractalHeapIdFlags(flags))
    );
}

fn heap_id_fixture_layout(length: usize, filtered: bool) -> FractalHeapIdLayout {
    let header = FractalHeapHeader {
        heap_id_length: u16::try_from(length).unwrap(),
        filtering: if filtered {
            test_support::filtered_heap()
        } else {
            FractalHeapFiltering::Unfiltered
        },
        ..test_support::attribute_heap_header()
    };
    FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
}

const MANAGED_ID: &[u8] = &[0x00, 42, 0, 0, 0, 0, 13, 0];
const INDIRECT_HUGE_ID: &[u8] = &[0x10, 5, 0, 0, 0, 0, 0, 0];
const DIRECT_HUGE_ID: &[u8] = &[0x10, 0x34, 0x12, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0];
const FILTERED_DIRECT_HUGE_ID: &[u8] = &[
    0x10, 0x34, 0x12, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
];
