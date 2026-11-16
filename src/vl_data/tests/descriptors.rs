use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::variable_length_descriptor::Address;

use super::*;

#[rstest]
#[case::heap_backed_empty(0, 7)]
#[case::multiple_elements(3, 400)]
#[case::full_wire_fields(u32::MAX, u32::MAX)]
fn descriptor_adapter_preserves_wire_fields_and_count_units(
    #[case] count: u32,
    #[case] index: u32,
    #[values(
        Address::Two(0x1234),
        Address::Four(0x1234_5678),
        Address::Eight(0x1234_5678_9abc_def0)
    )]
    address: Address,
) {
    let offsets = OffsetWidth::try_from(address.width()).unwrap();
    let mut raw = address.descriptor(count, index);
    let size = raw.len();
    raw.extend_from_slice(b"following");
    let element = VlElement::parse(&raw, offsets).unwrap();
    assert_eq!(
        (
            element.length,
            element.collection_address.get(),
            element.object_index
        ),
        (count, address.get(), index)
    );
    assert_eq!(VlElement::serialized_size(offsets), size);
    assert_eq!(element.byte_len(2).unwrap(), u64::from(count) * 2);
}

#[rstest]
fn descriptor_adapter_keeps_null_fields_before_identity_interpretation(
    #[values(
        Address::Two(0),
        Address::Four(0),
        Address::Eight(0),
        Address::Two(u16::MAX),
        Address::Four(u32::MAX),
        Address::Eight(u64::MAX)
    )]
    address: Address,
    #[values(0, 512)] base: u64,
) {
    let offsets = OffsetWidth::try_from(address.width()).unwrap();
    let element = VlElement::parse(&address.descriptor(0, u32::MAX), offsets).unwrap();
    assert_eq!(
        (
            element.length,
            element.collection_address.get(),
            element.object_index
        ),
        (0, address.get(), u32::MAX)
    );
    assert_eq!(element.heap_id(offsets, context(base)).unwrap(), None);
}

#[rstest]
fn descriptor_adapter_resolves_defined_references_even_with_zero_count(
    #[values(0, 1)] count: u32,
    #[values(
        Address::Two(0x1234),
        Address::Four(u32::from(u16::MAX)),
        Address::Eight(u64::from(u32::MAX))
    )]
    address: Address,
    #[values(0, 512)] base: u64,
) {
    let offsets = OffsetWidth::try_from(address.width()).unwrap();
    let element =
        VlElement::parse(&address.descriptor(count, u32::from(u16::MAX)), offsets).unwrap();
    assert_eq!(
        element.heap_id(offsets, context(base)).unwrap(),
        Some(GlobalHeapId {
            collection: PhysicalAddress::new(base + address.get()),
            index: NonZeroU16::new(u16::MAX).unwrap(),
        })
    );
}

#[rstest]
fn descriptor_adapter_preserves_nonempty_zero_address_interpretation(
    #[values(Address::Two(0), Address::Four(0), Address::Eight(0))] address: Address,
    #[values(0, 512)] base: u64,
) {
    let offsets = OffsetWidth::try_from(address.width()).unwrap();
    let element = VlElement::parse(&address.descriptor(1, 7), offsets).unwrap();
    assert_eq!(
        element.heap_id(offsets, context(base)).unwrap(),
        Some(GlobalHeapId {
            collection: PhysicalAddress::new(base),
            index: NonZeroU16::new(7).unwrap(),
        })
    );
}

#[rstest]
#[case::missing(0, IndexFailure::Missing)]
#[case::above_member_field(u32::from(u16::MAX) + 1, IndexFailure::Wide("global heap object index 65536 does not fit u16"))]
#[case::full_wire_index(
    u32::MAX,
    IndexFailure::Wide("global heap object index 4294967295 does not fit u16")
)]
fn descriptor_adapter_preserves_member_index_failures(
    #[case] index: u32,
    #[case] expected: IndexFailure,
    #[values(0, 1)] count: u32,
    #[values(Address::Two(4096), Address::Four(4096), Address::Eight(4096))] address: Address,
) {
    let offsets = OffsetWidth::try_from(address.width()).unwrap();
    let element = VlElement::parse(&address.descriptor(count, index), offsets).unwrap();
    let expected = match expected {
        IndexFailure::Missing => FormatError::GlobalHeapObjectNotFound {
            collection_address: 4608,
            index: 0,
        },
        IndexFailure::Wide(message) => FormatError::VlDataError(message.into()),
    };
    assert_eq!(
        element.heap_id(offsets, context(512)).unwrap_err(),
        expected
    );
}

#[rstest]
fn descriptor_adapter_reports_undefined_before_resolution_or_index_narrowing(
    #[values(
        Address::Two(u16::MAX),
        Address::Four(u32::MAX),
        Address::Eight(u64::MAX)
    )]
    address: Address,
    #[values(1, u32::MAX)] count: u32,
) {
    let offsets = OffsetWidth::try_from(address.width()).unwrap();
    let element = VlElement::parse(&address.descriptor(count, u32::MAX), offsets).unwrap();
    assert_eq!(
        element.heap_id(offsets, context(512)).unwrap_err(),
        FormatError::VlDataError("non-empty VL element has an undefined heap address".into())
    );
}

#[rstest]
fn descriptor_adapter_reports_resolution_overflow_before_index_narrowing(
    #[values(0, 1)] count: u32,
) {
    let raw = Address::Eight(u64::MAX - 1).descriptor(count, u32::MAX);
    let element = VlElement::parse(&raw, OffsetWidth::Eight).unwrap();
    assert_eq!(
        element
            .heap_id(OffsetWidth::Eight, context(512))
            .unwrap_err(),
        FormatError::OffsetOverflow {
            offset: u64::MAX - 1,
            length: 512,
        }
    );
}

#[rstest]
fn descriptor_adapter_preserves_complete_frame_errors(
    #[values(
        Address::Two(0x1234),
        Address::Four(0x1234_5678),
        Address::Eight(0x1234_5678_9abc_def0)
    )]
    address: Address,
    #[values(false, true)] empty: bool,
) {
    let raw = address.descriptor(3, 7);
    let available = if empty { 0 } else { raw.len() - 1 };
    assert_eq!(
        VlElement::parse(
            &raw[..available],
            OffsetWidth::try_from(address.width()).unwrap()
        )
        .unwrap_err(),
        FormatError::UnexpectedEof {
            expected: raw.len(),
            available,
        }
    );
}

fn context(base: u64) -> AddressContext {
    AddressContext::from_effective_base(BaseAddress::new(base))
}

enum IndexFailure {
    Missing,
    Wide(&'static str),
}
