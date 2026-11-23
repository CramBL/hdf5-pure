use hdf5_pure_core::FormatError;
use hdf5_pure_format::__private::LinkInfoMessage;
use rstest::rstest;

#[rstest]
#[case::compact(&[0, 0x80, 0xff, 0xff, 0xff, 0xff], 0x80)]
#[case::tracked_without_body(&[0, 0x81], 0x81)]
#[case::indexed_without_body(&[0, 0x06], 0x06)]
fn reserved_link_info_flags_are_rejected(#[case] bytes: &[u8], #[case] flags: u8) {
    assert_eq!(
        LinkInfoMessage::parse(bytes, 2).unwrap_err(),
        FormatError::InvalidLinkInfoFlags(flags),
    );
}
