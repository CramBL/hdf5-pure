use rstest::rstest;

use super::{DenseAttrCreationOrder, EncodedDenseAttrs};
use crate::dense_attribute_write::{
    plan::{DENSE_ATTR_LENGTH_WIDTH, DENSE_ATTR_OFFSET_WIDTH},
    test_support,
};
use crate::error::FormatError;
use crate::width::FormatWidths;

#[rstest]
#[case::missing_index(vec![1], 9, false,
    FormatError::Internal("a tracked attribute set needs one creation index per attribute".into()))]
#[case::extra_index(vec![1, 2, 3], 9, true,
    FormatError::Internal("a tracked attribute set needs one creation index per attribute".into()))]
#[case::duplicate_index(vec![3, 3], 9, true,
    FormatError::DuplicateAttributeCreationIndex(3))]
#[case::duplicate_unindexed(vec![3, 3], 9, false,
    FormatError::DuplicateAttributeCreationIndex(3))]
#[case::counter_equals_index(vec![1, 3], 3, true,
    FormatError::InvalidAttributeCreationCounter { index: 3, next: 3 })]
#[case::counter_below_index(vec![1, 3], 2, false,
    FormatError::InvalidAttributeCreationCounter { index: 3, next: 2 })]
fn dense_attribute_preparation_rejects_incoherent_creation_metadata(
    #[case] indices: Vec<u16>,
    #[case] max: u16,
    #[case] indexed: bool,
    #[case] expected: FormatError,
) {
    let attrs = [
        test_support::metadata_attribute("a", 1),
        test_support::metadata_attribute("b", 2),
    ];
    assert_eq!(
        EncodedDenseAttrs::new(
            FormatWidths {
                offsets: DENSE_ATTR_OFFSET_WIDTH,
                lengths: DENSE_ATTR_LENGTH_WIDTH
            },
            &attrs,
            DenseAttrCreationOrder::Tracked {
                indices,
                max,
                indexed
            }
        )
        .unwrap_err(),
        expected,
    );
}

#[rstest]
#[case::tracked_and_indexed(true)]
#[case::tracked_without_index(false)]
fn dense_attribute_preparation_preserves_index_gaps_and_counter(#[case] indexed: bool) {
    let attrs = [
        test_support::metadata_attribute("a", 1),
        test_support::metadata_attribute("b", 2),
    ];
    let prepared = EncodedDenseAttrs::new(
        FormatWidths {
            offsets: DENSE_ATTR_OFFSET_WIDTH,
            lengths: DENSE_ATTR_LENGTH_WIDTH,
        },
        &attrs,
        DenseAttrCreationOrder::Tracked {
            indices: vec![7, 2],
            max: u16::MAX,
            indexed,
        },
    )
    .unwrap();
    assert_eq!(prepared.indexes.next_index, Some(u16::MAX));
    assert_eq!(prepared.indexes.indexed, indexed);
    assert_eq!(
        prepared
            .attributes
            .iter()
            .map(|a| a.creation_index())
            .collect::<Vec<_>>(),
        vec![7, 2]
    );
    assert_eq!(
        prepared
            .attributes
            .iter()
            .map(|a| a.bytes().to_vec())
            .collect::<Vec<_>>(),
        attrs
            .iter()
            .map(|a| a
                .serialize_v3(DENSE_ATTR_OFFSET_WIDTH, DENSE_ATTR_LENGTH_WIDTH.get())
                .unwrap())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        prepared.indexes.corder_order,
        if indexed { vec![1, 0] } else { vec![] }
    );
}

#[test]
fn dense_attribute_preparation_preserves_untracked_positions() {
    let attrs = [
        test_support::metadata_attribute("b", 1),
        test_support::metadata_attribute("a", 2),
    ];
    let prepared = EncodedDenseAttrs::new(
        FormatWidths {
            offsets: DENSE_ATTR_OFFSET_WIDTH,
            lengths: DENSE_ATTR_LENGTH_WIDTH,
        },
        &attrs,
        DenseAttrCreationOrder::Untracked,
    )
    .unwrap();
    assert_eq!(prepared.indexes.next_index, None);
    assert!(!prepared.indexes.indexed);
    assert!(prepared.indexes.corder_order.is_empty());
    assert_eq!(
        prepared
            .attributes
            .iter()
            .map(|a| a.creation_index())
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
}

#[test]
fn dense_attribute_preparation_rejects_duplicate_names() {
    let attrs = [
        test_support::metadata_attribute("duplicate", 1),
        test_support::metadata_attribute("duplicate", 2),
    ];
    assert_eq!(
        EncodedDenseAttrs::new(
            FormatWidths {
                offsets: DENSE_ATTR_OFFSET_WIDTH,
                lengths: DENSE_ATTR_LENGTH_WIDTH
            },
            &attrs,
            DenseAttrCreationOrder::Untracked
        )
        .unwrap_err(),
        FormatError::DuplicateAttributeName("duplicate".into())
    );
}

#[rstest]
#[case::untracked(DenseAttrCreationOrder::Untracked, None, false)]
#[case::tracked(DenseAttrCreationOrder::Tracked { indices: vec![], max: 17, indexed: true }, Some(17), true)]
fn dense_attribute_preparation_preserves_empty_set_metadata(
    #[case] creation: DenseAttrCreationOrder,
    #[case] next: Option<u16>,
    #[case] indexed: bool,
) {
    let prepared = EncodedDenseAttrs::new(
        FormatWidths {
            offsets: DENSE_ATTR_OFFSET_WIDTH,
            lengths: DENSE_ATTR_LENGTH_WIDTH,
        },
        &[],
        creation,
    )
    .unwrap();
    assert!(prepared.attributes.is_empty());
    assert!(prepared.indexes.order.is_empty());
    assert!(prepared.indexes.corder_order.is_empty());
    assert_eq!(prepared.indexes.next_index, next);
    assert_eq!(prepared.indexes.indexed, indexed);
}
