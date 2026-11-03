use hdf5_pure_format::__private::{AttributePhaseChange, MessageRecordLayout, ObjectHeaderPrefix};
use rstest::rstest;
#[cfg(feature = "checksum")]
use test_util::checksum;
use test_util::object_header::v2::HeaderFlags;

use super::{AttrOp, EditedAttrs};
use crate::attribute::AttributeMessage;
#[cfg(feature = "checksum")]
use crate::edit::WriteEngine;
use crate::edit::object_header::OhRegion;
use crate::edit::test_util as edit_test_util;
use crate::error::{Error, FormatError};
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
#[cfg(feature = "checksum")]
use crate::object_path::ObjectPathBuf;
#[cfg(feature = "checksum")]
use crate::source::MetadataCacheConfig;
use crate::type_builders::AttrValue;
use crate::writer::FileBuilder;
#[cfg(feature = "checksum")]
use crate::{File, FileLocking, MemoryStrategy, PersistedFreeSpacePolicy};

mod predecessors;

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
    let engine = edit_test_util::open_engine(&path);
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
    let Err(error) = super::plan_attr_ops(&engine.opened_source(), &region, &[&operation]) else {
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
    let engine = edit_test_util::open_engine(&path);
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
    let result = super::plan_attr_ops(&engine.opened_source(), &region, &[&operation]).unwrap();
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

#[rstest]
#[case::invalid_version(vec![0xff, 0], FormatError::InvalidAttributeInfoVersion(0xff))]
#[case::truncated(vec![], FormatError::UnexpectedEof { expected: 2, available: 0 })]
fn bulk_attribute_edits_preserve_metadata_parse_errors(
    #[case] body: Vec<u8>,
    #[case] expected: FormatError,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("invalid-info.h5");
    FileBuilder::new().write(&path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let mut region = OhRegion::empty(ObjectHeaderPrefix::PLAIN);
    region.push(MessageType::ATTRIBUTE_INFO, &body);
    let operation = AttrOp::Remove {
        name: "absent".into(),
    };
    let Err(error) = super::plan_attr_ops(&engine.opened_source(), &region, &[&operation]) else {
        panic!("malformed Attribute Info was accepted");
    };
    let Error::Format(actual) = error else {
        panic!("expected metadata parsing error, got {error:?}");
    };
    assert_eq!(actual, expected);
}

#[rstest]
#[case::duplicate([1, 1], 2, FormatError::DuplicateAttributeCreationIndex(1))]
#[case::counter([2, 3], 2, FormatError::InvalidAttributeCreationCounter { index: 2, next: 2 })]
fn compact_rebuild_rejects_invalid_creation_metadata(
    #[case] indices: [u16; 2],
    #[case] next: u16,
    #[case] expected: FormatError,
) {
    let region = OhRegion::empty(ObjectHeaderPrefix::with_layout(
        MessageRecordLayout::from_header_flags(
            (HeaderFlags::TRACKS_CREATION_ORDER | HeaderFlags::INDEXES_CREATION_ORDER).0,
        ),
    ));
    let slots = indices
        .into_iter()
        .enumerate()
        .map(|(position, index)| {
            let msg = crate::type_builders::build_attr_message(
                &format!("a{position}"),
                &AttrValue::I64(7),
            )
            .unwrap();
            let body = msg
                .serialize(
                    crate::file_writer::OFFSET_WIDTH,
                    crate::file_writer::LENGTH_SIZE,
                )
                .unwrap();
            (
                msg.name.clone(),
                super::AttributeSlot {
                    msg,
                    body: Some(body),
                    creation_index: Some(index),
                    collections: None,
                },
            )
        })
        .collect();
    let Err(error) = super::prepare_compact_attrs(
        &region,
        slots,
        next,
        true,
        super::AttributePredecessors::default(),
    ) else {
        panic!("invalid creation metadata was accepted");
    };
    let Error::Format(actual) = error else {
        panic!("expected creation metadata error, got {error:?}");
    };
    assert_eq!(actual, expected);
}

#[rstest]
#[case::always(MessageFlags::FAIL_IF_UNKNOWN_ALWAYS)]
#[case::writing(MessageFlags::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE)]
fn bulk_attribute_edits_reject_unknown_required_messages(#[case] flags: MessageFlags) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("unknown-message.h5");
    FileBuilder::new().write(&path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let mut region = OhRegion::empty(ObjectHeaderPrefix::PLAIN);
    let record =
        region
            .layout()
            .record_with_creation_index(MessageType::from_u16(0xff), flags, 0, &[]);
    region.push_bytes(&record);
    let operation = AttrOp::Set {
        name: "large".into(),
        value: AttrValue::I64Array(vec![0; 10_000]),
    };
    let Err(error) = super::plan_attr_ops(&engine.opened_source(), &region, &[&operation]) else {
        panic!("unknown required message was accepted");
    };
    assert!(
        matches!(error, Error::Format(FormatError::UnsupportedMessage(0xff))),
        "{error:?}"
    );
}

#[cfg(feature = "checksum")]
#[rstest]
fn root_attribute_edit_rejects_a_corrupted_header_checksum(
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("root-checksum.h5");
    let mut writer = FileBuilder::new();
    for index in 0..9 {
        writer.set_attr(&format!("a{index}"), AttrValue::I64(index));
    }
    writer.create_dataset("kept").with_i32_data(&[7, 11]);
    writer.write(&path).unwrap();
    let file = File::open(&path).unwrap();
    let root = usize::try_from(
        file.resolve_address(file.superblock().root_group_address)
            .unwrap()
            .get(),
    )
    .unwrap();
    drop(file);
    let mut bytes = std::fs::read(&path).unwrap();
    let prefix = ObjectHeaderPrefix::parse(&bytes[root..]).unwrap();
    let checksum_at = root + prefix.len + usize::try_from(prefix.chunk0_size).unwrap();
    let end = checksum_at + checksum::CHECKSUM;
    let computed = checksum::lookup3(&bytes[root..checksum_at]);
    assert_eq!(&bytes[checksum_at..end], computed.to_le_bytes());
    bytes[checksum_at] ^= 1;
    std::fs::write(&path, &bytes).unwrap();
    let mut engine = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        memory,
        PersistedFreeSpacePolicy::Tolerant,
    )
    .unwrap();
    for index in 0..4 {
        engine
            .remove_group_attr(&ObjectPathBuf::root(), &format!("a{index}"))
            .unwrap();
    }
    let error = engine.commit().unwrap_err();
    assert!(
        matches!(error, Error::Format(FormatError::ChecksumMismatch {
        expected, computed: actual,
    }) if expected == computed ^ 1 && actual == computed),
        "{error:?}"
    );
    let removals: Vec<_> = engine
        .staged
        .group_attrs()
        .iter()
        .map(|(path, operation)| {
            let AttrOp::Remove { name } = operation else {
                panic!("expected a staged removal");
            };
            (path.clone(), name.clone())
        })
        .collect();
    assert_eq!(
        removals,
        (0..4)
            .map(|index| (ObjectPathBuf::root(), format!("a{index}")))
            .collect::<Vec<_>>()
    );
    assert_eq!(engine.disk_bytes().unwrap(), bytes);
    drop(engine);
    assert_eq!(std::fs::read(path).unwrap(), bytes);
}
