use core::num::NonZeroU64;
use std::collections::BTreeMap;

use hdf5_pure_format::__private::ObjectHeaderPrefix;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::attribute;
use test_util::dataspace;
use test_util::datatype;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags;
use test_util::object_header::MessageType;
use test_util::object_header::v2;
use test_util::object_header::v2::HeaderFlags;
use test_util::sohm;

use crate::AttrValue;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::attribute::AttributeMessage;
use crate::edit::attributes::DenseAttrSet;
use crate::edit::copy::CopyNode;
use crate::edit::copy::CopyTree;
use crate::edit::copy::CopyTreeBuilder;
use crate::edit::object_header::OhRegion;
use crate::error::{Error, FormatError};
use crate::file_writer::{DenseAttrCreationOrder, WIDTHS};
use crate::shared_message::DatatypeLocation;
use crate::type_builders;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

use super::CopiedDatatypeUsers;

#[rstest]
#[case::compact_v2(false)]
#[case::compact_v3(true)]
fn copied_users_combine_dataset_compact_and_normalized_dense_types(#[case] version_three: bool) {
    let reference = datatype::committed_reference(900);
    let space = dataspace::scalar();
    let attr = attribute::Attribute::new("compact", &reference, &space, &[])
        .flags(attribute::Flags::SHARED_DATATYPE);
    let attr = if version_three {
        attr.character_set(ASCII_CHARACTER_SET).build()
    } else {
        attr.build()
    };
    let tree = tree(
        vec![
            shared_datatype(&reference),
            Message::new(MessageType::ATTRIBUTE, &attr),
        ],
        vec![dense_attribute("a", Some(900)), dense_attribute("b", None)],
    );
    let users = tree.datatype_users(WIDTHS).unwrap().unwrap();
    assert!(users.matches(&tree));
    assert_eq!(counts(&users), vec![(900, 3)]);
    let other = CopyTreeBuilder::default().finish(node(Vec::new(), Vec::new()));
    assert!(!users.matches(&other));
}

#[test]
fn copied_users_count_each_emitted_node_and_sort_distinct_targets() {
    let mut nodes = CopyTreeBuilder::default();
    nodes.push(node(
        vec![shared_datatype(&datatype::committed_reference(900))],
        Vec::new(),
    ));
    nodes.push(node(
        vec![shared_datatype(&datatype::committed_reference(500))],
        Vec::new(),
    ));
    let tree = nodes.finish(node(
        vec![shared_datatype(&datatype::committed_reference(900))],
        Vec::new(),
    ));
    assert_eq!(
        counts(&tree.datatype_users(WIDTHS).unwrap().unwrap()),
        vec![(500, 1), (900, 2)]
    );
}

#[rstest]
#[case::empty(Vec::new())]
#[case::inline_lookalike(vec![Message::new(MessageType::DATATYPE, &datatype::committed_reference(900))])]
#[case::anonymous_heap(vec![shared_datatype(&sohm::heap_reference([0; sohm::HEAP_ID_LEN]))])]
#[case::anonymous_shared_space(vec![shared_record(MessageType::DATASPACE, &sohm::heap_reference([0; sohm::HEAP_ID_LEN]))])]
fn copied_users_complete_inline_and_anonymous_references(#[case] records: Vec<Message>) {
    let tree = tree(records, Vec::new());
    assert_eq!(
        counts(&tree.datatype_users(WIDTHS).unwrap().unwrap()),
        Vec::new()
    );
}

#[rstest]
#[case::shared_attribute(shared_record(MessageType::ATTRIBUTE, &sohm::heap_reference([0; sohm::HEAP_ID_LEN])))]
#[case::object_header_dataspace(shared_record(MessageType::DATASPACE, &datatype::committed_reference(900)))]
#[case::unknown_inline(Message::new(MessageType(0xfe), &[]))]
#[case::unknown_shared(shared_record(MessageType(0xfe), &sohm::heap_reference([0; sohm::HEAP_ID_LEN])))]
fn copied_user_completeness_requires_supported_reference_messages(#[case] message: Message) {
    let tree = tree(
        vec![
            shared_datatype(&datatype::committed_reference(900)),
            message,
        ],
        Vec::new(),
    );
    assert!(tree.datatype_users(WIDTHS).unwrap().is_none());
}

#[rstest]
#[case::two_byte_sentinel(OffsetWidth::Two, 65_535, false)]
#[case::four_byte_defined(OffsetWidth::Four, 65_535, true)]
#[case::four_byte_sentinel(OffsetWidth::Four, u64::from(u32::MAX), false)]
#[case::eight_byte_defined(OffsetWidth::Eight, u64::from(u32::MAX), true)]
#[case::eight_byte_sentinel(OffsetWidth::Eight, u64::MAX, false)]
fn copied_users_classify_undefined_fields_at_their_encoded_width(
    #[case] width: OffsetWidth,
    #[case] address: u64,
    #[case] defined: bool,
) {
    let mut reference = datatype::committed_reference(address);
    reference.truncate(
        reference.len() - usize::from(OffsetWidth::Eight.get()) + usize::from(width.get()),
    );
    let tree = tree(vec![shared_datatype(&reference)], Vec::new());
    let users = tree
        .datatype_users(FormatWidths {
            offsets: width,
            lengths: LengthWidth::Eight,
        })
        .unwrap();
    assert_eq!(
        users.as_ref().map(counts),
        defined.then_some(vec![(address, 1)])
    );
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn copied_user_lookup_storage_starts_with_a_committed_target() {
    let empty_tree = tree(Vec::new(), Vec::new());
    let control_tree = tree(
        vec![shared_datatype(&datatype::committed_reference(900))],
        Vec::new(),
    );
    let (empty, measured) = allocation::measure("empty_copied_type_users", || {
        empty_tree.datatype_users(WIDTHS)
    });
    let (control, positive) = allocation::measure("nonempty_copied_type_users", || {
        control_tree.datatype_users(WIDTHS)
    });
    assert_eq!(counts(&empty.unwrap().unwrap()), Vec::new());
    assert_eq!(counts(&control.unwrap().unwrap()), vec![(900, 1)]);
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(measured.bytes, 0, "{measured}");
    assert!(positive.blocks > 0 && positive.bytes > 0, "{positive}");
}

#[rstest]
#[case::two(OffsetWidth::Two, u64::from(u16::MAX))]
#[case::four(OffsetWidth::Four, u64::from(u32::MAX))]
#[case::eight(OffsetWidth::Eight, u64::MAX)]
fn copied_dense_attributes_reject_undefined_references_before_user_analysis(
    #[case] offsets: OffsetWidth,
    #[case] address: u64,
) {
    let Err(error) = DenseAttrSet::new(
        FormatWidths {
            offsets,
            lengths: LengthWidth::Eight,
        },
        vec![dense_attribute("undefined", Some(address))],
        DenseAttrCreationOrder::Untracked,
        [],
    ) else {
        panic!("expected an undefined committed datatype address to be rejected")
    };
    assert_eq!(
        error,
        FormatError::DefinedAddressIsUndefined {
            address,
            offset_size: offsets.get()
        }
    );
}

#[test]
fn copied_users_preserve_attribute_framing_errors() {
    let reference = datatype::committed_reference(900);
    let mut attr = attribute::Attribute::new("a", &reference, &[], &[])
        .flags(attribute::Flags::SHARED_DATATYPE)
        .build();
    attr.pop();
    let tree = tree(
        vec![Message::new(MessageType::ATTRIBUTE, &attr)],
        Vec::new(),
    );
    let Err(error) = tree.datatype_users(WIDTHS) else {
        panic!("expected malformed copy metadata to be rejected")
    };
    let Error::Format(error) = error else {
        panic!("expected an attribute framing error, got {error:?}")
    };
    assert_eq!(
        error,
        FormatError::UnexpectedEof {
            expected: 20,
            available: 19
        }
    );
}

#[test]
fn copied_users_report_record_framing_errors() {
    let mut region = OhRegion::new(
        WIDTHS,
        v2::message_records(&[Message::nil(1)], HeaderFlags::default()),
        ObjectHeaderPrefix::PLAIN,
    );
    region.bytes_mut()[1..3].copy_from_slice(&2u16.to_le_bytes());
    let tree = CopyTreeBuilder::default().finish(CopyNode::DatasetVerbatim {
        region,
        dense_attrs: DenseAttrSet::default(),
    });
    let Err(error) = tree.datatype_users(WIDTHS) else {
        panic!("expected malformed copy metadata to be rejected")
    };
    let Error::EditUnsupported(reason) = error else {
        panic!("expected a record framing error, got {error:?}")
    };
    assert_eq!(reason, "malformed object header message");
}

#[test]
fn copied_users_preserve_reference_framing_errors() {
    let mut reference = datatype::committed_reference(900);
    reference.pop();
    let tree = tree(vec![shared_datatype(&reference)], Vec::new());
    let Err(error) = tree.datatype_users(WIDTHS) else {
        panic!("expected malformed copy metadata to be rejected")
    };
    let Error::Format(error) = error else {
        panic!("expected a shared reference framing error, got {error:?}")
    };
    assert_eq!(
        error,
        FormatError::UnexpectedEof {
            expected: 10,
            available: 9
        }
    );
}

#[test]
fn copied_user_accumulation_retains_counts_on_overflow() {
    let address = StoredAddress::new(900);
    let mut users = BTreeMap::from([(address, NonZeroU64::MAX)]);
    assert_eq!(
        super::add_user(&mut users, address).unwrap_err(),
        FormatError::Internal("copied datatype user count overflows".into())
    );
    assert_eq!(users, BTreeMap::from([(address, NonZeroU64::MAX)]));
}

#[rstest]
#[case::narrow(OffsetWidth::Two)]
#[case::wide(OffsetWidth::Eight)]
fn foreign_copies_qualify_zero_users_at_their_source_width(#[case] offsets: OffsetWidth) {
    let widths = FormatWidths {
        offsets,
        lengths: LengthWidth::Eight,
    };
    let mut builder = CopyTreeBuilder::default();
    builder.push(CopyNode::DatasetVerbatim {
        region: OhRegion::new(widths, Vec::new(), ObjectHeaderPrefix::PLAIN),
        dense_attrs: DenseAttrSet::default(),
    });
    let tree = builder.finish(CopyNode::DatasetVerbatim {
        region: OhRegion::new(
            widths,
            v2::message_records(
                &[Message::new(MessageType::DATATYPE, &datatype::f64_le())],
                HeaderFlags::default(),
            ),
            ObjectHeaderPrefix::PLAIN,
        ),
        dense_attrs: DenseAttrSet::new(
            widths,
            vec![dense_attribute("numeric", None)],
            DenseAttrCreationOrder::Untracked,
            [],
        )
        .unwrap(),
    });
    tree.require_foreign_datatype_users_absent().unwrap();
    assert_eq!(counts(&tree.datatype_users(widths).unwrap().unwrap()), []);
}

#[rstest]
#[case::shared_type(shared_datatype(&datatype::committed_reference(900)), "a shared (committed/SOHM) object-header message cannot be copied to another file yet")]
#[case::unknown_record(Message::new(MessageType(0xfe), &[]), "a foreign copy has an incomplete datatype inventory")]
fn foreign_copy_qualification_rejects_unclassified_or_committed_descendants(
    #[case] message: Message,
    #[case] expected: &str,
) {
    let mut builder = CopyTreeBuilder::default();
    builder.push(node(vec![message], Vec::new()));
    let tree = builder.finish(node(Vec::new(), Vec::new()));
    let error = tree.require_foreign_datatype_users_absent().unwrap_err();
    let Error::EditUnsupported(reason) = error else {
        panic!("unexpected qualification error: {error:?}");
    };
    assert_eq!(reason, expected);
}

#[test]
fn foreign_copy_qualification_rejects_committed_dense_attribute_types() {
    let tree = tree(Vec::new(), vec![dense_attribute("shared", Some(900))]);
    let error = tree.require_foreign_datatype_users_absent().unwrap_err();
    assert!(
        matches!(
            error,
            Error::EditUnsupported("a foreign copy retains a committed datatype")
        ),
        "{error:?}"
    );
}

fn counts(users: &CopiedDatatypeUsers<'_>) -> Vec<(u64, u64)> {
    users
        .users()
        .map(|(address, count)| (address.get(), count.get()))
        .collect()
}

fn tree(records: Vec<Message>, attrs: Vec<AttributeMessage>) -> CopyTree {
    CopyTreeBuilder::default().finish(node(records, attrs))
}

fn node(records: Vec<Message>, attrs: Vec<AttributeMessage>) -> CopyNode {
    CopyNode::DatasetVerbatim {
        region: OhRegion::new(
            WIDTHS,
            v2::message_records(&records, HeaderFlags::default()),
            ObjectHeaderPrefix::PLAIN,
        ),
        dense_attrs: DenseAttrSet::new(WIDTHS, attrs, DenseAttrCreationOrder::Untracked, [])
            .unwrap(),
    }
}

fn shared_datatype(reference: &[u8]) -> Message {
    shared_record(MessageType::DATATYPE, reference)
}

fn shared_record(msg_type: MessageType, reference: &[u8]) -> Message {
    let mut message = Message::new(msg_type, reference);
    message.flags = MessageFlags::SHARED;
    message
}

fn dense_attribute(name: &str, address: Option<u64>) -> AttributeMessage {
    let mut attr = type_builders::build_attr_message(name, &AttrValue::I32(7)).unwrap();
    attr.datatype_location = address.map_or(DatatypeLocation::Inline, |address| {
        DatatypeLocation::Committed(StoredAddress::new(address))
    });
    attr
}

// "The Attribute Message" of the format specification, version 4.0, assigns zero to ASCII.
const ASCII_CHARACTER_SET: u8 = 0;
