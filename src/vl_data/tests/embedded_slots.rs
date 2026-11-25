use core::ops::Range;

use rstest::rstest;

use super::super::*;
use crate::{
    DatatypeByteOrder,
    datatype::{CompoundMember, StringPadding},
};

fn vlen_slots_in_element(datatype: &Datatype) -> Option<Vec<(Range<usize>, usize)>> {
    slots_in_element(embedded_vlen_slots(datatype))
}

fn slots_in_element(slots: Option<EmbeddedVlenSlots>) -> Option<Vec<(Range<usize>, usize)>> {
    slots.map(|slots| {
        let raw = vec![0; slots.element_size()];
        slots.slots_in(&raw).collect()
    })
}

fn vlen_string() -> Datatype {
    Datatype::VariableLength(VariableLengthType::string(
        StringPadding::NullTerminate,
        CharacterSet::Utf8,
    ))
}

fn vlen_i32_sequence() -> Datatype {
    Datatype::VariableLength(VariableLengthType::sequence(i32_type()))
}

fn i32_type() -> Datatype {
    Datatype::FixedPoint(
        FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
    )
}

fn member(name: &str, byte_offset: u64, datatype: Datatype) -> CompoundMember {
    crate::datatype::__private::compound_member(name.to_string(), byte_offset, datatype)
}

#[test]
fn a_plain_datatype_reaches_no_reference() {
    assert_eq!(vlen_slots_in_element(&i32_type()), Some(Vec::new()));
}

#[test]
fn a_top_level_vlen_is_its_own_slot() {
    assert_eq!(
        vlen_slots_in_element(&vlen_string()),
        Some(vec![(0..16, 1)])
    );
}

#[test]
fn compound_members_keep_their_own_element_size() {
    let dt = Datatype::Compound {
        size: 40,
        members: vec![
            member("label", 0, vlen_string()),
            member("id", 16, i32_type()),
            member("samples", 24, vlen_i32_sequence()),
        ],
    };
    assert_eq!(
        vlen_slots_in_element(&dt),
        Some(vec![(0..16, 1), (24..40, 4),])
    );
}

/// A compound nested inside a compound, and an array of compounds, both reach
/// references that a single-level walk would miss.
#[test]
fn nested_compounds_and_arrays_are_walked() {
    let inner = Datatype::Compound {
        size: 20,
        members: vec![member("id", 0, i32_type()), member("s", 4, vlen_string())],
    };
    let nested = Datatype::Compound {
        size: 24,
        members: vec![
            member("n", 0, i32_type()),
            member("inner", 4, inner.clone()),
        ],
    };
    assert_eq!(vlen_slots_in_element(&nested), Some(vec![(8..24, 1)]));

    let array = Datatype::Array {
        base_type: Box::new(inner),
        dimensions: vec![3],
    };
    assert_eq!(
        vlen_slots_in_element(&array),
        Some(vec![(4..20, 1), (24..40, 1), (44..60, 1),])
    );
}

#[test]
fn a_datatype_too_small_for_its_own_references_is_rejected() {
    let dt = Datatype::Compound {
        size: 8, // one VL reference needs 16
        members: vec![member("s", 0, vlen_string())],
    };
    assert_eq!(vlen_slots_in_element(&dt), None);
}

#[test]
fn an_array_declaring_absurd_dimensions_is_rejected_not_walked() {
    let dt = Datatype::Array {
        base_type: Box::new(vlen_string()),
        dimensions: vec![u32::MAX, u32::MAX],
    };
    assert_eq!(vlen_slots_in_element(&dt), None);
}

#[test]
fn nested_array_slots_require_a_complete_element() {
    let mut dt = vlen_string();
    for _ in 0..17 {
        dt = Datatype::Array {
            base_type: Box::new(dt),
            dimensions: vec![2],
        };
    }
    let slots = embedded_vlen_slots(&dt).unwrap();

    assert_eq!(slots.len(), 1 << 17);
    assert_eq!(slots.element_size(), 16 * (1 << 17));
    assert_eq!(slots.slots_in(&[0; 16]).next(), None);
}

// The member offset exceeds `usize` on 32-bit targets and the element size on 64-bit targets.
#[test]
fn a_member_offset_beyond_the_address_space_is_rejected() {
    let dt = Datatype::Compound {
        size: 40,
        members: vec![member("s", u64::MAX - 8, vlen_string())],
    };
    assert_eq!(vlen_slots_in_element(&dt), None);
}

#[rstest]
#[case::strings(vlen_string(), 1)]
#[case::sequences(vlen_i32_sequence(), 4)]
fn stored_array_slots_use_the_file_reference_width(
    #[case] datatype: Datatype,
    #[case] element_size: usize,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
) {
    let datatype = Datatype::Array {
        base_type: Box::new(datatype),
        dimensions: vec![3],
    };
    let stride = 4 + usize::from(offsets.get()) + 4;
    assert_eq!(
        slots_in_element(embedded_vlen_slots_with_offset_width(&datatype, offsets).unwrap()),
        Some(
            (0..3)
                .map(|index| (index * stride..(index + 1) * stride, element_size))
                .collect()
        )
    );
}

#[rstest]
#[case::two(OffsetWidth::Two, 42_949_672_950)]
#[case::four(OffsetWidth::Four, 51_539_607_540)]
#[case::eight(OffsetWidth::Eight, 68_719_476_720)]
fn unrepresentable_array_slots_return_the_field_error(
    #[case] offsets: OffsetWidth,
    #[case] length: u64,
) {
    let datatype = Datatype::Array {
        base_type: Box::new(vlen_string()),
        dimensions: vec![u32::MAX],
    };
    assert_eq!(
        embedded_vlen_slots_with_offset_width(&datatype, offsets).unwrap_err(),
        FormatError::LengthTooLarge {
            length,
            length_size: 4
        }
    );
}
