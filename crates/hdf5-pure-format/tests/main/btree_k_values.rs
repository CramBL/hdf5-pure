use core::num::NonZeroU16;

use hdf5_pure_format::__private::{BTreeKValues, BTreeKValuesOutputTooSmall};
use hdf5_pure_format::FormatError;
use rstest::{fixture, rstest};
use test_util::{btree_k_values, bytes};

#[rstest]
#[case::distinct(0x1234, 0x5678, 0x9abc)]
#[case::wire_boundaries(1, u16::MAX, 0x0102)]
fn parses_independent_bodies(
    #[case] indexed_storage: u16,
    #[case] group_internal: u16,
    #[case] group_leaf: u16,
) {
    let fixture = k_values(indexed_storage, group_internal, group_leaf);
    let parsed = BTreeKValues::parse(&fixture.build()).unwrap();
    assert_eq!(parsed, message(fixture));
    assert_eq!(
        parsed.indexed_storage_internal_node_k(),
        fixture.indexed_storage_internal_node_k,
    );
    assert_eq!(
        parsed.group_internal_node_k(),
        fixture.group_internal_node_k
    );
    assert_eq!(parsed.group_leaf_node_k(), fixture.group_leaf_node_k);
}

#[rstest]
#[case::distinct(0x1234, 0x5678, 0x9abc)]
#[case::wire_boundaries(1, u16::MAX, 0x0102)]
fn writes_bodies_matching_independent_bytes(
    #[case] indexed_storage: u16,
    #[case] group_internal: u16,
    #[case] group_leaf: u16,
) {
    let fixture = k_values(indexed_storage, group_internal, group_leaf);
    let message = message(fixture);
    let mut output = [0xa5; 7];
    assert_eq!(message.encoded_size(), output.len());
    assert_eq!(message.write_into(&mut output).unwrap(), output.len());
    assert_eq!(output, fixture.build());
}

#[rstest]
fn ignores_input_after_the_body(distinct_values: btree_k_values::BTreeKValues) {
    let mut input = distinct_values.build().to_vec();
    input.extend_from_slice(&[0xff, 0x5a]);
    assert_eq!(
        BTreeKValues::parse(&input).unwrap(),
        message(distinct_values),
    );
}

#[rstest]
#[case::missing_version(0, 1)]
#[case::indexed_storage_absent(1, 7)]
#[case::indexed_storage_partial(2, 7)]
#[case::group_internal_absent(3, 7)]
#[case::group_internal_partial(4, 7)]
#[case::group_leaf_absent(5, 7)]
#[case::group_leaf_partial(6, 7)]
fn reports_the_required_extent_for_truncation(
    distinct_values: btree_k_values::BTreeKValues,
    #[case] available: usize,
    #[case] expected: usize,
) {
    let input = distinct_values.build();
    assert_eq!(
        BTreeKValues::parse(&input[..available]).unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available,
        },
    );
}

#[rstest]
#[case::complete_body(7)]
#[case::missing_body(1)]
fn rejects_unsupported_versions_before_reading_the_body(
    distinct_values: btree_k_values::BTreeKValues,
    #[case] available: usize,
) {
    let mut input = distinct_values.build();
    input[0] = 0xfe;
    assert_eq!(
        BTreeKValues::parse(&input[..available]).unwrap_err(),
        FormatError::InvalidBTreeKValuesVersion(0xfe),
    );
}

#[rstest]
fn acquires_all_k_values_before_rejecting_zero(distinct_values: btree_k_values::BTreeKValues) {
    let mut input = distinct_values.build();
    bytes::set_u16_at(&mut input, 1, 0);
    assert_eq!(
        BTreeKValues::parse(&input[..6]).unwrap_err(),
        FormatError::UnexpectedEof {
            expected: 7,
            available: 6,
        },
    );
}

#[rstest]
#[case::indexed_storage(1)]
#[case::group_internal(3)]
#[case::group_leaf(5)]
fn rejects_zero_k_values(
    distinct_values: btree_k_values::BTreeKValues,
    #[case] field_offset: usize,
) {
    let mut input = distinct_values.build();
    bytes::set_u16_at(&mut input, field_offset, 0);
    assert_eq!(
        BTreeKValues::parse(&input).unwrap_err(),
        FormatError::ZeroBTreeKValue { field_offset },
    );
}

#[rstest]
#[case::indexed_storage_before_group_internal(1, 3)]
#[case::group_internal_before_group_leaf(3, 5)]
fn reports_the_first_zero_in_wire_order(
    distinct_values: btree_k_values::BTreeKValues,
    #[case] first: usize,
    #[case] second: usize,
) {
    let mut input = distinct_values.build();
    bytes::set_u16_at(&mut input, first, 0);
    bytes::set_u16_at(&mut input, second, 0);
    assert_eq!(
        BTreeKValues::parse(&input).unwrap_err(),
        FormatError::ZeroBTreeKValue {
            field_offset: first,
        },
    );
}

#[rstest]
fn a_short_destination_is_unchanged(
    distinct_values: btree_k_values::BTreeKValues,
    #[values(0, 1, 2, 3, 4, 5, 6)] available: usize,
) {
    let message = message(distinct_values);
    let mut output = [0xa5; 9];
    assert_eq!(
        message
            .write_into(&mut output[1..1 + available])
            .unwrap_err(),
        BTreeKValuesOutputTooSmall {
            required: 7,
            available,
        },
    );
    assert_eq!(output, [0xa5; 9]);
}

#[rstest]
fn writing_preserves_bytes_outside_the_body(distinct_values: btree_k_values::BTreeKValues) {
    let message = message(distinct_values);
    let mut output = [0xa5; 11];
    assert_eq!(message.write_into(&mut output[1..]).unwrap(), 7);
    let mut expected = [0xa5; 11];
    expected[1..8].copy_from_slice(&distinct_values.build());
    assert_eq!(output, expected);
}

#[rstest]
fn round_trips_all_three_k_values(distinct_values: btree_k_values::BTreeKValues) {
    let message = message(distinct_values);
    let mut output = [0; 7];
    let written = message.write_into(&mut output).unwrap();
    assert_eq!(BTreeKValues::parse(&output[..written]).unwrap(), message);
}

#[fixture]
fn distinct_values() -> btree_k_values::BTreeKValues {
    k_values(0x1234, 0x5678, 0x9abc)
}

fn k_values(
    indexed_storage: u16,
    group_internal: u16,
    group_leaf: u16,
) -> btree_k_values::BTreeKValues {
    btree_k_values::BTreeKValues {
        indexed_storage_internal_node_k: NonZeroU16::new(indexed_storage).unwrap(),
        group_internal_node_k: NonZeroU16::new(group_internal).unwrap(),
        group_leaf_node_k: NonZeroU16::new(group_leaf).unwrap(),
    }
}

fn message(fixture: btree_k_values::BTreeKValues) -> BTreeKValues {
    BTreeKValues::new(
        fixture.indexed_storage_internal_node_k,
        fixture.group_internal_node_k,
        fixture.group_leaf_node_k,
    )
}
