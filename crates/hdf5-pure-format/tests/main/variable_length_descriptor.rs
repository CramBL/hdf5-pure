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
#[case::two_0(Address::Two(0x1234), 0)]
#[case::two_1(Address::Two(0x1234), 1)]
#[case::two_2(Address::Two(0x1234), 2)]
#[case::two_3(Address::Two(0x1234), 3)]
#[case::two_4(Address::Two(0x1234), 4)]
#[case::two_5(Address::Two(0x1234), 5)]
#[case::two_6(Address::Two(0x1234), 6)]
#[case::two_7(Address::Two(0x1234), 7)]
#[case::two_8(Address::Two(0x1234), 8)]
#[case::two_9(Address::Two(0x1234), 9)]
#[case::four_0(Address::Four(0x1234_5678), 0)]
#[case::four_1(Address::Four(0x1234_5678), 1)]
#[case::four_2(Address::Four(0x1234_5678), 2)]
#[case::four_3(Address::Four(0x1234_5678), 3)]
#[case::four_4(Address::Four(0x1234_5678), 4)]
#[case::four_5(Address::Four(0x1234_5678), 5)]
#[case::four_6(Address::Four(0x1234_5678), 6)]
#[case::four_7(Address::Four(0x1234_5678), 7)]
#[case::four_8(Address::Four(0x1234_5678), 8)]
#[case::four_9(Address::Four(0x1234_5678), 9)]
#[case::four_10(Address::Four(0x1234_5678), 10)]
#[case::four_11(Address::Four(0x1234_5678), 11)]
#[case::eight_0(Address::Eight(0x1234_5678_9abc_def0), 0)]
#[case::eight_1(Address::Eight(0x1234_5678_9abc_def0), 1)]
#[case::eight_2(Address::Eight(0x1234_5678_9abc_def0), 2)]
#[case::eight_3(Address::Eight(0x1234_5678_9abc_def0), 3)]
#[case::eight_4(Address::Eight(0x1234_5678_9abc_def0), 4)]
#[case::eight_5(Address::Eight(0x1234_5678_9abc_def0), 5)]
#[case::eight_6(Address::Eight(0x1234_5678_9abc_def0), 6)]
#[case::eight_7(Address::Eight(0x1234_5678_9abc_def0), 7)]
#[case::eight_8(Address::Eight(0x1234_5678_9abc_def0), 8)]
#[case::eight_9(Address::Eight(0x1234_5678_9abc_def0), 9)]
#[case::eight_10(Address::Eight(0x1234_5678_9abc_def0), 10)]
#[case::eight_11(Address::Eight(0x1234_5678_9abc_def0), 11)]
#[case::eight_12(Address::Eight(0x1234_5678_9abc_def0), 12)]
#[case::eight_13(Address::Eight(0x1234_5678_9abc_def0), 13)]
#[case::eight_14(Address::Eight(0x1234_5678_9abc_def0), 14)]
#[case::eight_15(Address::Eight(0x1234_5678_9abc_def0), 15)]
fn variable_length_descriptor_requires_complete_input_and_output(
    #[case] address: Address,
    #[case] available: usize,
) {
    let offsets = OffsetWidth::try_from(address.width()).unwrap();
    let expected = address.descriptor(3, u32::MAX);
    let error = FormatError::UnexpectedEof {
        expected: expected.len(),
        available,
    };
    assert_eq!(
        VariableLengthDescriptor::parse(offsets, &expected[..available]).unwrap_err(),
        error
    );
    let descriptor =
        VariableLengthDescriptor::new(offsets, 3, StoredAddress::new(address.get()), u32::MAX)
            .unwrap();
    let mut output = vec![0xab; available];
    assert_eq!(descriptor.write_into(&mut output).unwrap_err(), error);
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

#[rstest]
#[case::two(Address::Two(0x1234))]
#[case::four(Address::Four(0x1234_5678))]
#[case::eight(Address::Eight(0x1234_5678_9abc_def0))]
fn variable_length_descriptors_read_and_write_unaligned_buffers(#[case] address: Address) {
    let offsets = OffsetWidth::try_from(address.width()).unwrap();
    let expected = address.descriptor(0x1234_5678, 0x90ab_cdef);
    let mut input = Aligned([0xab; 32]);
    input.0[1..1 + expected.len()].copy_from_slice(&expected);
    assert_eq!(input.0[1..].as_ptr().addr() % 8, 1);
    let (descriptor, consumed) = VariableLengthDescriptor::parse(offsets, &input.0[1..]).unwrap();
    assert_eq!(consumed, expected.len());
    assert_eq!(descriptor.element_count(), 0x1234_5678);
    assert_eq!(
        descriptor.collection_address(),
        StoredAddress::new(address.get())
    );
    assert_eq!(descriptor.object_index(), 0x90ab_cdef);
    assert_eq!(descriptor.offset_width(), offsets);
    let mut output = Aligned([0xcd; 32]);
    assert_eq!(output.0[1..].as_ptr().addr() % 8, 1);
    assert_eq!(
        descriptor.write_into(&mut output.0[1..]).unwrap(),
        expected.len()
    );
    assert_eq!(&output.0[1..1 + expected.len()], expected);
    assert_eq!(output.0[0], 0xcd);
    assert_eq!(
        &output.0[1 + expected.len()..],
        &vec![0xcd; 31 - expected.len()]
    );
}

#[repr(align(8))]
struct Aligned([u8; 32]);
