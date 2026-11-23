use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_core::BaseAddress;
use hdf5_pure_format::__private::{
    AccessMode, FormatWidths, ObjectHeader, ObjectHeaderContinuation,
};
use hdf5_pure_format::FormatError;
use rstest::rstest;
use test_util::object_header::Message;
use test_util::object_header::v2::Header;
use test_util::widths::Widths;

#[rstest]
#[case::empty(0)]
#[case::undefined_address_without_length(2)]
#[case::partial_length(3)]
fn parser_reports_continuation_truncation_before_checking_the_address(#[case] available: usize) {
    let widths = FormatWidths::from_sizes(2, 2).unwrap();
    let mut message = Message::undefined_continuation(0x3456, Widths::new(2, 2));
    message.data.truncate(available);
    let expected = FormatError::UnexpectedEof {
        expected: 4,
        available,
    };

    assert_eq!(
        ObjectHeaderContinuation::parse_with_widths(&message.data, widths),
        Err(expected.clone())
    );
    assert_header_error(message, widths, expected);
}

#[test]
fn a_complete_continuation_with_an_undefined_address_is_rejected() {
    let widths = FormatWidths::from_sizes(2, 2).unwrap();
    let message = Message::undefined_continuation(0x3456, Widths::new(2, 2));
    assert_eq!(
        ObjectHeaderContinuation::parse_with_widths(&message.data, widths),
        Err(FormatError::UndefinedContinuationAddress)
    );
    assert_header_error(message, widths, FormatError::UndefinedContinuationAddress);
}

#[rstest]
#[case::two_byte_address(2, 0x1234)]
#[case::four_byte_address(4, 0x1234_5678)]
#[case::eight_byte_address(8, 0x1234_5678_9abc_def0)]
fn parser_reads_continuation_address_and_length_at_their_selected_widths(
    #[case] offset_size: u8,
    #[case] address: u64,
    #[values((2, 0x2345), (4, 0x2345_6789), (8, 0x2345_6789_abcd_ef01))] length: (u8, u64),
) {
    let (length_size, length) = length;
    let widths = Widths::new(usize::from(offset_size), usize::from(length_size));
    let message = Message::continuation(address, length, widths);
    let continuation =
        ObjectHeaderContinuation::parse(&message.data, offset_size, length_size).unwrap();

    assert_eq!(continuation.address().get(), address);
    assert_eq!(continuation.length(), length);
}

#[test]
fn parser_ignores_trailing_bytes_after_a_continuation() {
    let widths = Widths::new(2, 4);
    let mut message = Message::continuation(0, 0x489a_bcde, widths);
    message.data.extend_from_slice(&[0xa5, 0x5a, 0x37]);
    let continuation = ObjectHeaderContinuation::parse(&message.data, 2, 4).unwrap();

    assert_eq!(continuation.address().get(), 0);
    assert_eq!(continuation.length(), 0x489a_bcde);
}

fn assert_header_error(message: Message, widths: FormatWidths, expected: FormatError) {
    let header = Header::new().message(message).build();
    assert_eq!(
        ObjectHeader::parse_with_widths(&header, AccessMode::ReadOnly, 0, widths).unwrap_err(),
        expected
    );
    assert_eq!(
        ObjectHeader::parse_from_source_with_widths(
            header.as_slice(),
            AccessMode::ReadOnly,
            0,
            widths,
            BaseAddress::ZERO,
        )
        .unwrap_err(),
        expected
    );
}
