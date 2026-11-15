use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::FormatError;
use hdf5_pure_core::OffsetWidth;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_format::__private::SymbolTableMessage;
use rstest::rstest;
use test_util::bytes;

#[rstest]
#[case::two_defined(OffsetWidth::Two, 0x1234, 0x5678)]
#[case::four_defined(OffsetWidth::Four, 0x1234_5678, 0x90ab_cdef)]
#[case::eight_defined(OffsetWidth::Eight, 0x1234_5678_9abc_def0, 0xfedc_ba98_7654_3210)]
#[case::two_zero(OffsetWidth::Two, 0, 0)]
#[case::four_zero(OffsetWidth::Four, 0, 0)]
#[case::eight_zero(OffsetWidth::Eight, 0, 0)]
#[case::two_undefined(OffsetWidth::Two, 0xffff, 0xffff)]
#[case::four_undefined(OffsetWidth::Four, 0xffff_ffff, 0xffff_ffff)]
#[case::eight_undefined(OffsetWidth::Eight, u64::MAX, u64::MAX)]
fn symbol_table_messages_preserve_fields_and_unaligned_buffer_suffixes(
    #[case] width: OffsetWidth,
    #[case] btree: u64,
    #[case] heap: u64,
) {
    let message = SymbolTableMessage {
        btree_address: StoredAddress::new(btree),
        local_heap_address: StoredAddress::new(heap),
    };
    let mut expected = Vec::new();
    bytes::push_uint(&mut expected, btree, usize::from(width.get()));
    bytes::push_uint(&mut expected, heap, usize::from(width.get()));
    let mut input = Aligned([0xcd; 32]);
    input.0[1..1 + expected.len()].copy_from_slice(&expected);
    assert_eq!(input.0[1..].as_ptr().addr() % 8, 1);
    assert_eq!(
        SymbolTableMessage::parse_with_offset_width(&input.0[1..], width).unwrap(),
        message,
    );
    let mut output = Aligned([0xab; 32]);
    assert_eq!(output.0[1..].as_ptr().addr() % 8, 1);
    assert_eq!(
        message.write_into(&mut output.0[1..], width).unwrap(),
        expected.len(),
    );
    assert_eq!(&output.0[1..1 + expected.len()], expected);
    assert_eq!(output.0[0], 0xab);
    assert_eq!(
        &output.0[1 + expected.len()..],
        &vec![0xab; 31 - expected.len()],
    );
}

#[rstest]
#[case::two(OffsetWidth::Two)]
#[case::four(OffsetWidth::Four)]
#[case::eight(OffsetWidth::Eight)]
fn symbol_table_messages_require_complete_input_and_output(#[case] width: OffsetWidth) {
    let message = SymbolTableMessage {
        btree_address: StoredAddress::new(0x1234),
        local_heap_address: StoredAddress::new(0x5678),
    };
    let expected = 2 * usize::from(width.get());
    for available in 0..expected {
        let error = FormatError::UnexpectedEof {
            expected,
            available,
        };
        assert_eq!(
            SymbolTableMessage::parse_with_offset_width(&[0; 16][..available], width).unwrap_err(),
            error,
        );
        let mut output = [0xab; 18];
        assert_eq!(
            message
                .write_into(&mut output[1..1 + available], width)
                .unwrap_err(),
            error,
        );
        assert_eq!(output, [0xab; 18]);
    }
}

#[test]
fn raw_symbol_table_width_uses_the_typed_parser() {
    let body = [0x34, 0x12, 0x78, 0x56, 0xaa];
    assert_eq!(
        SymbolTableMessage::parse(&body, 2).unwrap(),
        SymbolTableMessage {
            btree_address: StoredAddress::new(0x1234),
            local_heap_address: StoredAddress::new(0x5678),
        }
    );
}

#[rstest]
#[case::two_btree(OffsetWidth::Two, 0x1_0000, 0, 0x1_0000)]
#[case::two_heap(OffsetWidth::Two, 0, 0x1_0000, 0x1_0000)]
#[case::two_both(OffsetWidth::Two, 0x1_0000, 0x1_0001, 0x1_0000)]
#[case::four_btree(OffsetWidth::Four, 0x1_0000_0000, 0, 0x1_0000_0000)]
#[case::four_heap(OffsetWidth::Four, 0, 0x1_0000_0000, 0x1_0000_0000)]
#[case::four_both(OffsetWidth::Four, 0x1_0000_0000, 0x1_0000_0001, 0x1_0000_0000)]
#[case::two_wide_undefined_btree(OffsetWidth::Two, u64::MAX, 0, u64::MAX)]
#[case::two_wide_undefined_heap(OffsetWidth::Two, 0, u64::MAX, u64::MAX)]
#[case::four_wide_undefined_btree(OffsetWidth::Four, u64::MAX, 0, u64::MAX)]
#[case::four_wide_undefined_heap(OffsetWidth::Four, 0, u64::MAX, u64::MAX)]
fn symbol_table_address_errors_precede_capacity_errors_and_preserve_output(
    #[case] width: OffsetWidth,
    #[case] btree: u64,
    #[case] heap: u64,
    #[case] address: u64,
    #[values(0, 16)] available: usize,
) {
    let message = SymbolTableMessage {
        btree_address: StoredAddress::new(btree),
        local_heap_address: StoredAddress::new(heap),
    };
    let mut output = [0xab; 18];
    assert_eq!(
        message
            .write_into(&mut output[1..1 + available], width)
            .unwrap_err(),
        FormatError::AddressTooLarge {
            address,
            offset_size: width.get(),
        },
    );
    assert_eq!(output, [0xab; 18]);
}

#[rstest]
#[case::zero(0)]
#[case::one(1)]
#[case::three(3)]
#[case::sixteen(16)]
#[case::maximum(u8::MAX)]
fn invalid_symbol_table_address_widths_precede_input_errors(#[case] width: u8) {
    assert_eq!(
        SymbolTableMessage::parse(&[], width).unwrap_err(),
        FormatError::InvalidOffsetSize(width),
    );
}

#[repr(align(8))]
struct Aligned([u8; 32]);
