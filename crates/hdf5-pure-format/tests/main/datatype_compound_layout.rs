use hdf5_pure_core::{CharacterSet, FormatError, StringPadding};
use hdf5_pure_format::__private::{
    DatatypeMessage, DatatypeMessageMember, DatatypeProperties, DatatypeVersion, OffsetWidth,
};
use rstest::rstest;
use test_util::datatype;
use test_util::datatype::containers::{self, CompoundField, MemberOffset};

#[rstest]
#[case::same_start(4, 4, 4, 4, "left", "right")]
#[case::inside(4, 12, 8, 4, "left", "right")]
#[case::contains(8, 4, 4, 12, "right", "left")]
#[case::one_byte(0, 8, 7, 8, "left", "right")]
#[case::reverse_partial(8, 8, 4, 8, "right", "left")]
fn compound_layout_rejects_overlapping_members(
    #[case] left_offset: u64,
    #[case] left_size: u32,
    #[case] right_offset: u64,
    #[case] right_size: u32,
    #[case] first: &str,
    #[case] second: &str,
) {
    let left = datatype::fixed_string(left_size);
    let right = datatype::fixed_string(right_size);
    let members = vec![
        DatatypeMessageMember::new("left".into(), left_offset, string(left_size)).unwrap(),
        DatatypeMessageMember::new("right".into(), right_offset, string(right_size)).unwrap(),
    ];
    let expected = FormatError::CompoundFieldOverlap {
        first: first.into(),
        second: second.into(),
    };
    assert_eq!(
        DatatypeMessage::new(
            DatatypeVersion::Three,
            DatatypeProperties::Compound { size: 32, members }
        )
        .unwrap_err(),
        expected
    );
    let bytes = containers::compound(
        3,
        [2, 0, 0],
        32,
        MemberOffset::One,
        &[
            CompoundField {
                name: b"left",
                offset: left_offset,
                datatype: &left,
            },
            CompoundField {
                name: b"right",
                offset: right_offset,
                datatype: &right,
            },
        ],
    );
    assert_eq!(DatatypeMessage::parse(&bytes).unwrap_err(), expected);
}

#[rstest]
#[case::adjacent(&[(0, 8), (8, 8)])]
#[case::reverse_adjacent(&[(8, 8), (0, 8)])]
#[case::disjoint(&[(0, 4), (8, 4)])]
#[case::reverse_disjoint(&[(16, 4), (0, 4)])]
#[case::between_neighbors(&[(0, 4), (8, 4), (4, 4)])]
fn compound_layout_preserves_disjoint_members_and_record_order(#[case] spans: &[(u64, u32)]) {
    let children = spans
        .iter()
        .map(|&(_, size)| datatype::fixed_string(size))
        .collect::<Vec<_>>();
    let names = (0..spans.len())
        .map(|index| format!("field{index}"))
        .collect::<Vec<_>>();
    let fields = spans
        .iter()
        .zip(&names)
        .zip(&children)
        .map(|((&(offset, _), name), datatype)| CompoundField {
            name: name.as_bytes(),
            offset,
            datatype,
        })
        .collect::<Vec<_>>();
    let members = spans
        .iter()
        .zip(&names)
        .map(|(&(offset, size), name)| {
            DatatypeMessageMember::new(name.clone(), offset, string(size)).unwrap()
        })
        .collect();
    let expected = DatatypeMessage::new(
        DatatypeVersion::Three,
        DatatypeProperties::Compound { size: 32, members },
    )
    .unwrap();
    let [low, high] = u16::try_from(spans.len()).unwrap().to_le_bytes();
    let bytes = containers::compound(3, [low, high, 0], 32, MemberOffset::One, &fields);
    assert_eq!(expected.serialize(OffsetWidth::Eight).unwrap(), bytes);
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap(),
        (expected, bytes.len())
    );
}

#[rstest]
#[case::sequence(false, 12, 32)]
#[case::array(true, 24, 40)]
fn compound_layout_rejects_width_dependent_and_nested_overlap(
    #[case] array: bool,
    #[case] fixed_offset: u64,
    #[case] size: u32,
) {
    let parent = datatype::fixed_string(1);
    let low = datatype::variable_length(3, [0; 3], 10, &parent);
    let middle = datatype::variable_length(3, [0; 3], 12, &parent);
    let high = datatype::variable_length(3, [0; 3], 16, &parent);
    let (low, middle, high) = if array {
        (
            datatype::array(3, [0; 3], 20, 1, &[2], &low),
            datatype::array(3, [0; 3], 24, 1, &[2], &middle),
            datatype::array(3, [0; 3], 32, 1, &[2], &high),
        )
    } else {
        (low, middle, high)
    };
    let fixed = datatype::fixed_string(4);
    let low = containers::compound(
        3,
        [2, 0, 0],
        size,
        MemberOffset::One,
        &[
            CompoundField {
                name: b"variable",
                offset: 0,
                datatype: &low,
            },
            CompoundField {
                name: b"fixed",
                offset: fixed_offset,
                datatype: &fixed,
            },
        ],
    );
    let middle = containers::compound(
        3,
        [2, 0, 0],
        size,
        MemberOffset::One,
        &[
            CompoundField {
                name: b"variable",
                offset: 0,
                datatype: &middle,
            },
            CompoundField {
                name: b"fixed",
                offset: fixed_offset,
                datatype: &fixed,
            },
        ],
    );
    let high = containers::compound(
        3,
        [2, 0, 0],
        size,
        MemberOffset::One,
        &[
            CompoundField {
                name: b"variable",
                offset: 0,
                datatype: &high,
            },
            CompoundField {
                name: b"fixed",
                offset: fixed_offset,
                datatype: &fixed,
            },
        ],
    );
    let (message, consumed) = DatatypeMessage::parse(&low).unwrap();
    assert_eq!(consumed, low.len());
    assert_eq!(message.serialize(OffsetWidth::Two).unwrap(), low);
    assert_eq!(message.serialize(OffsetWidth::Four).unwrap(), middle);
    let expected = FormatError::CompoundFieldOverlap {
        first: "variable".into(),
        second: "fixed".into(),
    };
    assert_eq!(DatatypeMessage::parse(&high).unwrap_err(), expected);
    assert_eq!(message.serialize(OffsetWidth::Eight).unwrap_err(), expected);
    let outer = DatatypeMessage::new(
        DatatypeVersion::Three,
        DatatypeProperties::Compound {
            size,
            members: vec![DatatypeMessageMember::new("inner".into(), 0, message).unwrap()],
        },
    )
    .unwrap();
    assert_eq!(outer.serialize(OffsetWidth::Eight).unwrap_err(), expected);
    let outer = containers::compound(
        3,
        [1, 0, 0],
        size,
        MemberOffset::One,
        &[CompoundField {
            name: b"inner",
            offset: 0,
            datatype: &high,
        }],
    );
    assert_eq!(DatatypeMessage::parse(&outer).unwrap_err(), expected);
}

fn string(size: u32) -> DatatypeMessage {
    DatatypeMessage::new(
        DatatypeVersion::One,
        DatatypeProperties::String {
            size,
            padding: StringPadding::NullPad,
            charset: CharacterSet::Ascii,
        },
    )
    .unwrap()
}
