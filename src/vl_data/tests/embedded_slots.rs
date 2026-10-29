use super::super::*;
use crate::{
    DatatypeByteOrder,
    datatype::{CompoundMember, StringPadding},
};

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
    assert_eq!(embedded_vlen_slots(&i32_type()), Some(Vec::new()));
}

#[test]
fn a_top_level_vlen_is_its_own_slot() {
    assert_eq!(
        embedded_vlen_slots(&vlen_string()),
        Some(vec![EmbeddedVlSlot {
            byte_offset: 0,
            element_size: 1
        }])
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
        embedded_vlen_slots(&dt),
        Some(vec![
            EmbeddedVlSlot {
                byte_offset: 0,
                element_size: 1
            },
            EmbeddedVlSlot {
                byte_offset: 24,
                element_size: 4
            },
        ])
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
    assert_eq!(
        embedded_vlen_slots(&nested),
        Some(vec![EmbeddedVlSlot {
            byte_offset: 8,
            element_size: 1
        }])
    );

    let array = Datatype::Array {
        base_type: Box::new(inner),
        dimensions: vec![3],
    };
    assert_eq!(
        embedded_vlen_slots(&array),
        Some(vec![
            EmbeddedVlSlot {
                byte_offset: 4,
                element_size: 1
            },
            EmbeddedVlSlot {
                byte_offset: 24,
                element_size: 1
            },
            EmbeddedVlSlot {
                byte_offset: 44,
                element_size: 1
            },
        ])
    );
}

#[test]
fn a_datatype_too_small_for_its_own_references_is_rejected() {
    let dt = Datatype::Compound {
        size: 8, // one VL reference needs 16
        members: vec![member("s", 0, vlen_string())],
    };
    assert_eq!(embedded_vlen_slots(&dt), None);
}

#[test]
fn an_array_declaring_absurd_dimensions_is_rejected_not_walked() {
    let dt = Datatype::Array {
        base_type: Box::new(vlen_string()),
        dimensions: vec![u32::MAX, u32::MAX],
    };
    assert_eq!(embedded_vlen_slots(&dt), None);
}

// Seventeen nested binary arrays exercise base-slot reuse for 131,072 references.
#[test]
fn deeply_nested_arrays_cost_time_proportional_to_their_slots() {
    let mut dt = vlen_string();
    for _ in 0..17 {
        dt = Datatype::Array {
            base_type: Box::new(dt),
            dimensions: vec![2],
        };
    }
    let started = std::time::Instant::now();
    let slots = embedded_vlen_slots(&dt).expect("a well-formed nesting must be walkable");
    let elapsed = started.elapsed();

    assert_eq!(slots.len(), 1 << 17, "one slot per leaf entry");
    // Ten seconds permits slow CI execution while bounding repeated subtree work.
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "walking {} slots took {elapsed:?}; the walk is no longer linear in its output",
        slots.len()
    );
}

// The member offset exceeds `usize` on 32-bit targets and the element size on 64-bit targets.
#[test]
fn a_member_offset_beyond_the_address_space_is_rejected() {
    let dt = Datatype::Compound {
        size: 40,
        members: vec![member("s", u64::MAX - 8, vlen_string())],
    };
    assert_eq!(embedded_vlen_slots(&dt), None);
}
