use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::FormatError;
use hdf5_pure_core::OffsetWidth;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_format::__private::VariableLengthDescriptor;
use rstest::rstest;
use test_util::variable_length_descriptor::Address;

#[rstest]
#[case::zero_count_heap_reference(0, Address::Two(0x1234), 7)]
#[case::full_fields(u32::MAX, Address::Four(0x1234_5678), u32::MAX)]
#[case::two_zero_address(3, Address::Two(0), 0)]
#[case::four_zero_address(3, Address::Four(0), 0)]
#[case::eight_zero_address(3, Address::Eight(0), 0)]
#[case::two_undefined(1, Address::Two(u16::MAX), 0)]
#[case::four_undefined(1, Address::Four(u32::MAX), 0)]
#[case::eight_undefined(1, Address::Eight(u64::MAX), 0)]
fn variable_length_descriptors_preserve_supplied_fields(
    #[case] count: u32,
    #[case] address: Address,
    #[case] index: u32,
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
#[case::two(Address::Two(0x1234))]
#[case::four(Address::Four(0x1234_5678))]
#[case::eight(Address::Eight(0x1234_5678_9abc_def0))]
fn variable_length_descriptor_requires_complete_input_and_output(#[case] address: Address) {
    let offsets = OffsetWidth::try_from(address.width()).unwrap();
    let expected = address.descriptor(3, u32::MAX);
    let descriptor =
        VariableLengthDescriptor::new(offsets, 3, StoredAddress::new(address.get()), u32::MAX)
            .unwrap();
    for available in 0..expected.len() {
        let error = FormatError::UnexpectedEof {
            expected: expected.len(),
            available,
        };
        assert_eq!(
            VariableLengthDescriptor::parse(offsets, &expected[..available]).unwrap_err(),
            error
        );
        let mut output = vec![0xab; available];
        assert_eq!(descriptor.write_into(&mut output).unwrap_err(), error);
        assert_eq!(output, vec![0xab; available]);
    }
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
