use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::FormatError;
use hdf5_pure_core::OffsetWidth;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_format::__private::VariableLengthDescriptor;
use rstest::rstest;
use test_util::variable_length_descriptor::Address;

#[rstest]
#[case::zero_count_heap_reference(0, 7)]
#[case::one_element(1, 1)]
#[case::multiple_elements(3, 0x1234_5678)]
#[case::full_wire_count_and_index(u32::MAX, u32::MAX)]
#[case::free_member_index(5, 0)]
fn variable_length_descriptors_preserve_supplied_fields(
    #[case] count: u32,
    #[case] index: u32,
    #[values(
        Address::Two(0x1234),
        Address::Four(0x1234_5678),
        Address::Eight(0x1234_5678_9abc_def0)
    )]
    address: Address,
) {
    assert_descriptor(count, address, index);
}

#[rstest]
fn variable_length_descriptors_preserve_zero_and_undefined_addresses(
    #[values(0, 1, u32::MAX)] count: u32,
    #[values(0, u32::MAX)] index: u32,
    #[values(
        Address::Two(0),
        Address::Four(0),
        Address::Eight(0),
        Address::Two(u16::MAX),
        Address::Four(u32::MAX),
        Address::Eight(u64::MAX)
    )]
    address: Address,
) {
    assert_descriptor(count, address, index);
}

#[rstest]
#[case::two_above_field(OffsetWidth::Two, 0x1_0000)]
#[case::two_wide_undefined(OffsetWidth::Two, u64::MAX)]
#[case::four_above_field(OffsetWidth::Four, 0x1_0000_0000)]
#[case::four_wide_undefined(OffsetWidth::Four, u64::MAX)]
fn variable_length_descriptor_construction_rejects_unrepresentable_addresses(
    #[case] offsets: OffsetWidth,
    #[case] address: u64,
) {
    assert_eq!(
        VariableLengthDescriptor::new(offsets, 0, StoredAddress::new(address), 0).unwrap_err(),
        FormatError::AddressTooLarge {
            address,
            offset_size: offsets.get(),
        }
    );
}

#[rstest]
#[case::empty(Truncation::Empty)]
#[case::partial_count(Truncation::Count)]
#[case::partial_address(Truncation::Address)]
#[case::partial_index(Truncation::Index)]
fn variable_length_descriptor_parsing_requires_the_complete_frame(
    #[case] truncation: Truncation,
    #[values(
        Address::Two(0x1234),
        Address::Four(0x1234_5678),
        Address::Eight(0x1234_5678_9abc_def0)
    )]
    address: Address,
) {
    let expected = address.descriptor(3, u32::MAX);
    let available = truncation.len(address);
    assert_eq!(
        VariableLengthDescriptor::parse(
            OffsetWidth::try_from(address.width()).unwrap(),
            &expected[..available],
        )
        .unwrap_err(),
        FormatError::UnexpectedEof {
            expected: expected.len(),
            available,
        }
    );
}

#[rstest]
#[case::empty(Truncation::Empty)]
#[case::partial_count(Truncation::Count)]
#[case::partial_address(Truncation::Address)]
#[case::partial_index(Truncation::Index)]
fn variable_length_descriptor_encoding_preserves_a_short_output(
    #[case] truncation: Truncation,
    #[values(
        Address::Two(0x1234),
        Address::Four(0x1234_5678),
        Address::Eight(0x1234_5678_9abc_def0)
    )]
    address: Address,
) {
    let offsets = OffsetWidth::try_from(address.width()).unwrap();
    let descriptor =
        VariableLengthDescriptor::new(offsets, 3, StoredAddress::new(address.get()), 7).unwrap();
    let available = truncation.len(address);
    let mut output = vec![0xab; available];
    assert_eq!(
        descriptor.write_into(&mut output).unwrap_err(),
        FormatError::UnexpectedEof {
            expected: address.descriptor(3, 7).len(),
            available,
        }
    );
    assert_eq!(output, vec![0xab; available]);
}

fn assert_descriptor(count: u32, address: Address, index: u32) {
    let offsets = OffsetWidth::try_from(address.width()).unwrap();
    let expected = address.descriptor(count, index);
    let descriptor =
        VariableLengthDescriptor::new(offsets, count, StoredAddress::new(address.get()), index)
            .unwrap();
    assert_eq!(descriptor.element_count(), count);
    assert_eq!(
        descriptor.collection_address(),
        StoredAddress::new(address.get())
    );
    assert_eq!(descriptor.object_index(), index);
    assert_eq!(descriptor.offset_width(), offsets);
    assert_eq!(descriptor.encoded_size(), expected.len());
    assert_eq!(
        VariableLengthDescriptor::serialized_size(offsets),
        expected.len()
    );
    let mut output = vec![0xab; expected.len() + 2];
    assert_eq!(descriptor.write_into(&mut output).unwrap(), expected.len());
    assert_eq!(&output[..expected.len()], expected);
    assert_eq!(&output[expected.len()..], [0xab; 2]);
    let (parsed, consumed) = VariableLengthDescriptor::parse(offsets, &output).unwrap();
    assert_eq!(consumed, expected.len());
    assert_eq!(parsed, descriptor);
    let mut parsed_output = vec![0xcd; expected.len()];
    assert_eq!(
        parsed.write_into(&mut parsed_output).unwrap(),
        expected.len()
    );
    assert_eq!(parsed_output, expected);
}

#[derive(Clone, Copy)]
enum Truncation {
    Address,
    Count,
    Empty,
    Index,
}

impl Truncation {
    fn len(self, address: Address) -> usize {
        let count_size = 0u32.to_le_bytes().len();
        match self {
            Self::Address => count_size + usize::from(address.width()) - 1,
            Self::Count => count_size - 1,
            Self::Empty => 0,
            Self::Index => address.descriptor(0, 0).len() - 1,
        }
    }
}
