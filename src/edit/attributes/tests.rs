use hdf5_pure_format::__private::{AttributePhaseChange, ObjectHeaderPrefix};
use rstest::rstest;

use super::{AttrOp, EditedAttrs};
use crate::attribute::AttributeMessage;
use crate::edit::object_header::OhRegion;
use crate::edit::test_util;
use crate::error::{Error, FormatError};
use crate::message_type::MessageType;
use crate::type_builders::AttrValue;
use crate::writer::FileBuilder;

#[rstest]
#[case::zero_maximum(0, 1)]
#[case::reversed(7, 8)]
fn attribute_edit_rejects_invalid_stored_thresholds(
    #[case] max_compact: u16,
    #[case] min_dense: u16,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("invalid-policy.h5");
    FileBuilder::new().write(&path).unwrap();
    let engine = test_util::open_engine(&path);
    let before = engine.disk_bytes().unwrap();
    let region = OhRegion::empty(ObjectHeaderPrefix {
        attribute_phase_change: Some(AttributePhaseChange {
            max_compact,
            min_dense,
        }),
        ..ObjectHeaderPrefix::PLAIN
    });
    let operation = AttrOp::Set {
        name: "added".to_string(),
        value: AttrValue::I64(7),
    };
    let Err(error) = super::plan_attr_ops(&engine.opened_source(), None, &region, &[&operation])
    else {
        panic!("invalid stored thresholds were accepted");
    };
    assert!(
        matches!(error, Error::Format(FormatError::InvalidAttributePhaseChange {
        max_compact: actual_maximum, min_dense: actual_minimum,
    }) if actual_maximum == max_compact && actual_minimum == min_dense),
        "{error:?}"
    );
    assert_eq!(engine.disk_bytes().unwrap(), before);
    assert!(region.is_empty());
    drop(engine);
    assert_eq!(std::fs::read(path).unwrap(), before);
}

#[test]
fn compact_overwrite_preserves_storage_above_the_stored_maximum() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("overwrite.h5");
    FileBuilder::new().write(&path).unwrap();
    let engine = test_util::open_engine(&path);
    let mut region = OhRegion::empty(ObjectHeaderPrefix {
        attribute_phase_change: Some(AttributePhaseChange {
            max_compact: 1,
            min_dense: 1,
        }),
        ..ObjectHeaderPrefix::PLAIN
    });
    region.push(
        MessageType::ATTRIBUTE,
        &super::encode_attr_body("kept", &AttrValue::I64(2)).unwrap(),
    );
    let kept = region.to_vec();
    region.push(
        MessageType::ATTRIBUTE,
        &super::encode_attr_body("changed", &AttrValue::I64(3)).unwrap(),
    );
    let operation = AttrOp::Set {
        name: "changed".to_string(),
        value: AttrValue::I64(9),
    };
    let result =
        super::plan_attr_ops(&engine.opened_source(), None, &region, &[&operation]).unwrap();
    assert!(matches!(result.attrs, EditedAttrs::Compact(pending) if pending.is_empty()));
    assert!(!result.replaces_dense_storage);
    assert_eq!(super::compact_attr_count(&result.region).unwrap(), 2);
    assert_eq!(&result.region[..kept.len()], kept);
    let (kind, body, end) = result.region.next_message(kept.len()).unwrap().unwrap();
    assert_eq!(kind, MessageType::ATTRIBUTE);
    let changed = AttributeMessage::parse(&result.region[body..end], 8).unwrap();
    assert_eq!(changed.name, "changed");
    assert_eq!(changed.raw_data, 9i64.to_le_bytes());
    assert_eq!(result.region.props(), region.props());
}
