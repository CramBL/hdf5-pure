use hdf5_pure_format::__private::AttributeInfoMessage;
use hdf5_pure_format::__private::MessageRecordLayout;
use hdf5_pure_format::__private::ObjectHeaderPrefix;
use rstest::rstest;
use test_util::attribute::{UnpaddedAttribute, UnpaddedVersion};
use test_util::attribute_info::{AttributeInfo, CreationOrderIndex, Width};
use test_util::object_header::v2::{self, HeaderFlags};
use test_util::object_header::{self, Message};
use test_util::{dataspace, datatype};

use super::super::AttrOp;
use super::super::test_util::encode_attr_body;
use crate::attribute::AttributeMessage;
use crate::edit::object_header::OhRegion;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::error::FormatError;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::type_builders::AttrValue;
use crate::width::{FormatWidths, OffsetWidth};

#[rstest]
#[case::two_byte_offsets(Width::Two)]
#[case::four_byte_offsets(Width::Four)]
#[case::eight_byte_offsets(Width::Eight)]
fn compact_edits_preserve_the_attribute_info_record_metadata(#[case] width: Width) {
    let flags = HeaderFlags::TRACKS_CREATION_ORDER | HeaderFlags::INDEXES_CREATION_ORDER;
    let layout = MessageRecordLayout::from_header_flags(flags.0);
    let props = ObjectHeaderPrefix::with_layout(layout);
    let widths = FormatWidths {
        offsets: OffsetWidth::try_from(width.get()).unwrap(),
        ..crate::file_writer::WIDTHS
    };
    let preceding = Message::nil(3).with_creation_order(2);
    let following = Message::nil(5).with_creation_order(4);
    let source_info = AttributeInfo::new(width, Some(4), None, None, CreationOrderIndex::Absent);
    let record = Message::new(
        object_header::MessageType::ATTRIBUTE_INFO,
        &source_info.build(),
    )
    .with_flags(object_header::MessageFlags::FORBID_SHARING)
    .with_creation_order(3);
    let gap = [0x12, 0x34, 0x56];
    let mut source = v2::message_records(
        &[preceding.clone(), record.clone(), following.clone()],
        flags,
    );
    source.extend_from_slice(&gap);
    let region = OhRegion::new(widths, source, props);
    let mut batch = super::super::CompactBatch::parse(&region).unwrap();
    batch
        .apply(
            &AttrOp::Set {
                name: "added".into(),
                value: AttrValue::AsciiString("new".into()),
            },
            0,
        )
        .unwrap();

    let result = batch
        .into_compact(super::super::AttributePredecessors::default())
        .unwrap();

    let expected_info =
        AttributeInfo::new(width, Some(5), None, None, CreationOrderIndex::Undefined).build();
    let expected = v2::message_records(
        &[
            preceding,
            Message {
                data: expected_info.clone(),
                ..record
            },
            following,
        ],
        flags,
    );
    assert_eq!(&*result.region, expected);
    assert_eq!(result.region.props(), props);
    assert_eq!(result.region.widths(), widths);
    let first = layout.next_message(&result.region, 0).unwrap().unwrap();
    let rewritten = layout
        .next_message(&result.region, first.body_range.end)
        .unwrap()
        .unwrap();
    assert_eq!(rewritten.msg_type, MessageType::ATTRIBUTE_INFO);
    assert_eq!(rewritten.flags, MessageFlags::FORBID_SHARING);
    assert_eq!(rewritten.creation_index, Some(3));
    assert_eq!(rewritten.body, expected_info);
    let super::super::EditedAttrs::RebuildCompact(compact) = result.attrs else {
        panic!("expected compact storage");
    };
    let body = UnpaddedAttribute::inline(
        UnpaddedVersion::Two,
        "added",
        &datatype::fixed_string(3),
        &dataspace::VersionTwo::scalar().build(),
        b"new",
    )
    .build();
    let mut records = v2::message_records(
        &[Message::new(object_header::MessageType::ATTRIBUTE, &body).with_creation_order(4)],
        flags,
    );
    records.extend_from_slice(&gap);
    assert_eq!(compact.records, records);
}

#[rstest]
#[case::fixed(false)]
#[case::variable(true)]
fn compact_batches_assign_creation_indices_in_operation_order(#[case] variable: bool) {
    let layout = MessageRecordLayout::from_header_flags(
        (HeaderFlags::TRACKS_CREATION_ORDER | HeaderFlags::INDEXES_CREATION_ORDER).0,
    );
    let mut region = OhRegion::empty(
        crate::file_writer::WIDTHS,
        ObjectHeaderPrefix::with_layout(layout),
    );
    let info = AttributeInfoMessage::compact(crate::file_writer::OFFSET_WIDTH, Some(4), true);
    region.push_fixture_message(MessageType::ATTRIBUTE_INFO, &info.to_bytes());
    let value = |number| {
        if variable {
            AttrValue::VarLenAsciiString(format!("value{number}"))
        } else {
            AttrValue::I64(number)
        }
    };
    let operations = [
        AttrOp::Set {
            name: "first".into(),
            value: value(1),
        },
        AttrOp::Set {
            name: "second".into(),
            value: value(2),
        },
        AttrOp::Set {
            name: "first".into(),
            value: value(3),
        },
        AttrOp::Remove {
            name: "second".into(),
        },
        AttrOp::Set {
            name: "second".into(),
            value: value(4),
        },
    ];
    let mut batch = super::super::CompactBatch::parse(&region).unwrap();
    for (index, operation) in operations.iter().enumerate() {
        batch.apply(operation, index).unwrap();
    }
    let result = batch
        .into_compact(super::super::AttributePredecessors::default())
        .unwrap();
    let super::super::EditedAttrs::RebuildCompact(compact) = result.attrs else {
        panic!("expected compact storage");
    };
    let (_, _, info) = super::super::find_attribute_info(&result.region)
        .unwrap()
        .unwrap();
    assert_eq!(info.max_creation_index(), Some(7));
    let mut actual = Vec::new();
    let mut position = 0;
    while let Some(record) = layout.next_message(&compact.records, position).unwrap() {
        let attribute = AttributeMessage::parse(record.body, 8).unwrap();
        actual.push((attribute.name, record.creation_index, attribute.raw_data));
        position = record.body_range.end;
    }
    assert_eq!(actual.len(), 2);
    assert_eq!((&*actual[0].0, actual[0].1), ("first", Some(4)));
    assert_eq!((&*actual[1].0, actual[1].1), ("second", Some(6)));
    if variable {
        assert_eq!(compact.fixups.len(), 2);
        for (fixup, attribute) in compact.fixups.iter().zip(&actual) {
            assert_eq!(&compact.records[fixup.range.clone()], attribute.2);
            assert_eq!(fixup.collections.len(), 1);
        }
    } else {
        assert!(compact.fixups.is_empty());
        assert_eq!(actual[0].2, 3i64.to_le_bytes());
        assert_eq!(actual[1].2, 4i64.to_le_bytes());
    }
}

#[rstest]
#[case::remove_before_invalid(ErrorSequence::RemoveBeforeInvalid)]
#[case::invalid_before_remove(ErrorSequence::InvalidBeforeRemove)]
#[case::remove_before_set(ErrorSequence::RemoveBeforeSet)]
fn compact_batch_reports_operation_errors_in_source_order(#[case] sequence: ErrorSequence) {
    let region = OhRegion::empty(crate::file_writer::WIDTHS, ObjectHeaderPrefix::PLAIN);
    let invalid = AttrOp::Set {
        name: String::new(),
        value: AttrValue::I64(1),
    };
    let remove = AttrOp::Remove {
        name: "missing".into(),
    };
    let set = AttrOp::Set {
        name: "missing".into(),
        value: AttrValue::I64(1),
    };
    let operations = match sequence {
        ErrorSequence::InvalidBeforeRemove => [&invalid, &remove],
        ErrorSequence::RemoveBeforeInvalid => [&remove, &invalid],
        ErrorSequence::RemoveBeforeSet => [&remove, &set],
    };
    let path = test_util::temp::temp_path("ordered-attribute-errors.h5");
    crate::FileBuilder::new().write(&path).unwrap();
    let engine = crate::edit::test_util::open_engine(&path);
    let Err(error) = super::super::plan_attr_ops(&engine.opened_source(), &region, &operations)
    else {
        panic!("invalid operations were accepted");
    };
    if matches!(sequence, ErrorSequence::InvalidBeforeRemove) {
        assert!(
            matches!(&error, Error::Format(FormatError::InvalidAttributeName { name }) if name.is_empty()),
            "{error:?}"
        );
    } else {
        assert!(
            matches!(
                error,
                Error::EditUnsupported("attribute to remove was not found")
            ),
            "{error:?}"
        );
    }
}

#[rstest]
#[case::overwrite(false)]
#[case::new_attribute(true)]
fn compact_batch_checks_creation_exhaustion_only_for_new_names(#[case] new_name: bool) {
    let layout = MessageRecordLayout::from_header_flags(HeaderFlags::TRACKS_CREATION_ORDER.0);
    let mut region = OhRegion::empty(
        crate::file_writer::WIDTHS,
        ObjectHeaderPrefix::with_layout(layout),
    );
    region.push_bytes(&edit_test_util::fixture_record(
        layout,
        MessageType::ATTRIBUTE,
        MessageFlags::NONE,
        u16::MAX,
        &encode_attr_body("existing", &AttrValue::I64(1)).unwrap(),
    ));
    let operation = AttrOp::Set {
        name: if new_name { "new" } else { "existing" }.into(),
        value: AttrValue::I64(2),
    };
    let mut batch = super::super::CompactBatch::parse(&region).unwrap();
    if new_name {
        let error = batch.apply(&operation, 0).unwrap_err();
        assert!(
            matches!(
                error,
                Error::EditUnsupported(
                    "an object has assigned every attribute creation index its header can record"
                )
            ),
            "{error:?}"
        );
    } else {
        batch.apply(&operation, 0).unwrap();
        let result = batch
            .into_compact(super::super::AttributePredecessors::default())
            .unwrap();
        let super::super::EditedAttrs::RebuildCompact(compact) = result.attrs else {
            panic!("expected compact storage");
        };
        let record = layout.next_message(&compact.records, 0).unwrap().unwrap();
        assert_eq!(record.creation_index, Some(u16::MAX));
        assert_eq!(
            AttributeMessage::parse(record.body, 8).unwrap().raw_data,
            2i64.to_le_bytes()
        );
    }
}

#[rstest]
#[case::small(false)]
#[case::oversized(true)]
fn superseded_oversized_values_preserve_the_full_attribute_parsing_boundary(
    #[case] oversized: bool,
) {
    let path = test_util::temp::temp_path("superseded-attribute-boundary.h5");
    crate::FileBuilder::new().write(&path).unwrap();
    let engine = crate::edit::test_util::open_engine(&path);
    let before = engine.disk_bytes().unwrap();
    let mut region = OhRegion::empty(crate::file_writer::WIDTHS, ObjectHeaderPrefix::PLAIN);
    let datatype = test_util::datatype::f64_le();
    let dataspace = test_util::dataspace::scalar();
    let mut body =
        test_util::attribute::Attribute::new("kept", &datatype, &dataspace, &1f64.to_le_bytes())
            .flags(test_util::attribute::Flags::NONE)
            .build();
    body.truncate(13);
    region.push_fixture_message(MessageType::ATTRIBUTE, &body);
    let set = AttrOp::Set {
        name: "temporary".into(),
        value: if oversized {
            AttrValue::I64Array(vec![0; 10_000])
        } else {
            AttrValue::I64(1)
        },
    };
    let remove = AttrOp::Remove {
        name: "temporary".into(),
    };
    let result = super::super::plan_attr_ops(&engine.opened_source(), &region, &[&set, &remove]);
    if oversized {
        let Err(error) = result else {
            panic!("a truncated datatype was accepted");
        };
        let Error::Format(error) = error else {
            panic!("expected a format error, got {error:?}");
        };
        assert_eq!(
            error,
            FormatError::UnexpectedEof {
                expected: 33,
                available: 13
            }
        );
    } else {
        let result = result.unwrap();
        assert_eq!(&*result.region, &*region);
        let super::super::EditedAttrs::RebuildCompact(compact) = result.attrs else {
            panic!("expected compact storage");
        };
        assert!(compact.records.is_empty());
        assert!(compact.fixups.is_empty());
    }
    assert_eq!(engine.disk_bytes().unwrap(), before);
}

#[derive(Clone, Copy, Debug)]
enum ErrorSequence {
    InvalidBeforeRemove,
    RemoveBeforeInvalid,
    RemoveBeforeSet,
}
