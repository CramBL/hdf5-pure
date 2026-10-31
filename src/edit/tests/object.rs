//! Object header rebuilding, attributes, links, and reference screening.

use std::collections::BTreeMap;
use std::thread;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_format::__private::AttributePhaseChange;
use hdf5_pure_format::__private::MessageRecordLayout;
use hdf5_pure_format::__private::ObjectHeaderPrefix;
use hdf5_pure_format::__private::ObjectTimes;
use rstest::rstest;
use test_util::data_layout as layout_bytes;
use test_util::link_message::HardLink;
use test_util::object_header::Message;
use test_util::object_header::v2 as v2_bytes;
use test_util::object_header::v2::HeaderFlags;
use test_util::widths::Widths;

use super::common;
use crate::PhysicalAddress;
use crate::UserblockSize;
use crate::VariableLengthType;
use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::attribute::AttributeMessage;
use crate::attribute_info::AttributeInfoMessage;
use crate::checksum;
use crate::data_layout::COMPACT_DATA_OFFSET;
use crate::datatype;
use crate::datatype::CharacterSet;
use crate::datatype::Datatype;
use crate::datatype::ReferenceType;
use crate::edit;
use crate::edit::InvalidatedAddresses;
use crate::edit::LinkCreationOrder;
use crate::edit::Node;
use crate::edit::OhRegion;
use crate::edit::REFERENCE_INTO_RECLAIMED_SPACE;
use crate::edit::REFERENCE_TO_A_MOVED_OBJECT;
use crate::edit::SHARED_MESSAGE_INDEX_NAMES_A_MOVED_OBJECT;
use crate::edit::UNDEF;
use crate::edit::WriteEngine;
use crate::edit::attributes::DenseAttrSet;
use crate::edit::copy;
use crate::edit::copy::CopyLink;
use crate::edit::copy::CopyNode;
use crate::edit::copy::CopyReader;
use crate::edit::copy::CopyTree;
use crate::edit::copy::CopyTreeBuilder;
use crate::edit::copy::ObjModel;
use crate::edit::group::GROUP_INFO_BODY;
use crate::error::Error;
use crate::error::FormatError;
use crate::error::OBJECT_HEADER_MESSAGE_MAX;
use crate::file_lock::FileLocking;
use crate::file_writer;
use crate::file_writer::DenseAttrCreationOrder;
use crate::file_writer::LENGTH_SIZE;
use crate::file_writer::OFFSET_WIDTH;
use crate::link_message::LinkMessage;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_graph::TraversalBudget;
use crate::object_graph::TraversalLimit;
use crate::object_header::HeaderMessage;
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPath;
use crate::object_path::ObjectPathBuf;
use crate::source::BytesSource;
use crate::superblock::LocatedSuperblock;
use crate::type_builders::AttrValue;
use crate::type_builders::ObjectRefTarget;
use crate::width::OffsetWidth;
use crate::writer::FileBuilder;

#[test]
fn a_continuation_to_the_undefined_address_fails_the_chunk_walk() {
    let header = v2_bytes::Header::new()
        .message(Message::undefined_continuation(64, Widths::EIGHT))
        .build();

    let Err(err) = edit::object_header::read_oh_chunks(
        &BytesSource::new(header.as_slice()),
        0,
        AddressContext::from_effective_base(BaseAddress::ZERO),
    ) else {
        panic!("expected the chunk walk to fail");
    };
    assert!(
        matches!(
            err,
            Error::Format(FormatError::UndefinedContinuationAddress)
        ),
        "{err:?}"
    );
}

#[test]
fn shared_attribute_detection_uses_attribute_message_flags() {
    let body = crate::type_builders::build_attr_message("a", &AttrValue::I64(1))
        .unwrap()
        .serialize(crate::file_writer::OFFSET_WIDTH, LENGTH_SIZE)
        .unwrap();
    let private = message_record(MessageType::ATTRIBUTE, &body);
    assert!(!edit::attributes::region_has_shared_attr(&plain_region(private.clone())).unwrap());

    let mut shared = private.clone();
    shared[3] = MessageFlags::SHARED.get();
    assert!(edit::attributes::region_has_shared_attr(&plain_region(shared)).unwrap());
    let mut other = message_record(MessageType::DATASPACE, &body);
    other[3] = MessageFlags::SHARED.get();
    assert!(!edit::attributes::region_has_shared_attr(&plain_region(other)).unwrap());
}

#[test]
fn a_hard_link_is_retargeted_at_its_address_field() {
    let mut region = hard_link_region(&[]);
    edit::group::patch_link_target(&mut region, "child", StoredAddress::new(0x2000)).unwrap();
    assert_eq!(
        region[region.len() - 8..],
        0x2000u64.to_le_bytes(),
        "the address field holds the new target"
    );
}

#[test]
fn a_hard_link_with_bytes_past_its_address_is_unsupported() {
    let mut region = hard_link_region(&[0]);
    let err = edit::group::patch_link_target(&mut region, "child", StoredAddress::new(0x2000))
        .unwrap_err();
    let Error::EditUnsupported(reason) = err else {
        panic!("expected EditUnsupported, got {err:?}");
    };
    assert_eq!(
        reason,
        "a link message holds bytes past its hard link address"
    );
}

#[rstest]
#[case::start(248, true)]
#[case::last_byte(318, true)]
#[case::end(319, false)]
fn a_copied_compact_reference_dataset_is_screened(#[case] address: u64, #[case] rejected: bool) {
    let empty = BytesSource::new(Vec::new());
    let invalidated = InvalidatedAddresses {
        removed: physical_ranges(&[(248, 71)]),
        moved: Vec::new(),
        base: AddressContext::from_effective_base(BaseAddress::ZERO),
    };
    let tree = verbatim_copy(compact_reference_region(address), DenseAttrSet::default());
    let got = copy::screen_copied_references(
        &tree,
        AddressContext::from_effective_base(BaseAddress::ZERO),
        crate::file_writer::WIDTHS,
        &invalidated,
        &empty,
    );
    if rejected {
        common::assert_unsupported(got.unwrap_err(), REFERENCE_INTO_RECLAIMED_SPACE);
    } else {
        got.unwrap();
    }
}

#[test]
fn a_copied_reference_dataset_without_layout_is_rejected() {
    let empty = BytesSource::new(Vec::new());
    let invalidated = InvalidatedAddresses {
        removed: physical_ranges(&[(248, 71)]),
        moved: Vec::new(),
        base: AddressContext::from_effective_base(BaseAddress::ZERO),
    };
    let no_layout = verbatim_copy(
        plain_region(message_record(
            MessageType::DATATYPE,
            &hdf5_pure_format::__private::serialize_datatype(
                &crate::type_builders::make_object_reference_type(),
            ),
        )),
        DenseAttrSet::default(),
    );
    let err = copy::screen_copied_references(
        &no_layout,
        AddressContext::from_effective_base(BaseAddress::ZERO),
        crate::file_writer::WIDTHS,
        &invalidated,
        &empty,
    )
    .unwrap_err();
    common::assert_unsupported(
        err,
        "a compact object-reference dataset's elements could not be read to screen \
         them against this commit's deletions; use separate commits",
    );
}

#[test]
fn a_copy_whose_committed_datatype_cannot_be_read_reports_the_read_failure() {
    let empty = BytesSource::new(Vec::new());
    let invalidated = InvalidatedAddresses {
        removed: physical_ranges(&[(248, 71)]),
        moved: Vec::new(),
        base: AddressContext::from_effective_base(BaseAddress::ZERO),
    };
    let mut bytes = message_record(
        MessageType::DATATYPE,
        &crate::shared_message::encode_committed_ref(
            StoredAddress::new(248),
            crate::file_writer::OFFSET_WIDTH,
        ),
    );
    bytes[3] = MessageFlags::SHARED.get();
    let tree = verbatim_copy(plain_region(bytes), DenseAttrSet::default());

    let err = copy::screen_copied_references(
        &tree,
        AddressContext::from_effective_base(BaseAddress::ZERO),
        crate::file_writer::WIDTHS,
        &invalidated,
        &empty,
    )
    .unwrap_err();
    let Error::CopyScreenUnreadable { message, source } = &err else {
        panic!("expected CopyScreenUnreadable, got {err:?}");
    };
    assert_eq!(*message, MessageType::DATATYPE);
    assert_eq!(
        *source,
        FormatError::UnexpectedSourceEof {
            expected: 252,
            available: 0
        }
    );
}

#[rstest]
#[case::compound_with_deletions(UnreadableReference::Compound, vec![(248, 71)])]
#[case::compound_without_deletions(UnreadableReference::Compound, Vec::new())]
#[case::variable_length(UnreadableReference::VariableLength, vec![(248, 71)])]
#[case::wide_with_deletions(UnreadableReference::Wide, vec![(248, 71)])]
#[case::wide_without_deletions(UnreadableReference::Wide, Vec::new())]
fn unreadable_reference_slots_are_rejected(
    #[case] reference: UnreadableReference,
    #[case] removed: Vec<(u64, u64)>,
) {
    let (datatype, raw) = reference.into_fixture();
    assert!(
        datatype::embedded_reference_slots(&datatype).is_none(),
        "the reference slots must be unreadable, not an empty slot list"
    );
    let invalidated = InvalidatedAddresses {
        removed: physical_ranges(&removed),
        moved: Vec::new(),
        base: AddressContext::from_effective_base(BaseAddress::ZERO),
    };
    common::assert_unsupported(
        edit::screen_resolved_references(&datatype, &raw, &invalidated).unwrap_err(),
        "a reference this commit writes sits in a datatype whose addresses this screen \
         cannot read — a width other than eight, a dataset-region reference, a \
         variable length of them, or a compound holding one beside a reference it can \
         read — so it cannot be checked against what this commit vacates; supply an \
         8-byte object reference (`with_reference_data`)",
    );
}

#[test]
fn a_variable_length_string_is_screened_without_object_references() {
    let datatype = crate::type_builders::make_vlen_string_type(CharacterSet::Utf8);
    let raw = vec![0u8; 32];
    let invalidated = InvalidatedAddresses {
        removed: physical_ranges(&[(248, 71)]),
        moved: Vec::new(),
        base: AddressContext::from_effective_base(BaseAddress::ZERO),
    };
    edit::screen_resolved_references(&datatype, &raw, &invalidated).unwrap();
}

#[test]
fn a_file_proved_free_of_references_is_walked_once_and_no_more() {
    use crate::reference_patch;

    let dir = tempfile::tempdir().unwrap();

    let commit_three = |path: &std::path::Path| {
        let session = crate::File::open_rw(path).unwrap();
        for i in 0..3 {
            session
                .root()
                .create_dataset(&format!("added{i}"), |b| {
                    b.with_i32_data(&[i]);
                })
                .unwrap();
            session.commit().unwrap();
        }
    };

    let plain = dir.path().join("plain.h5");
    let mut b = crate::writer::FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3]);
    b.write(&plain).unwrap();
    reference_patch::reset_walks();
    commit_three(&plain);
    assert_eq!(
        reference_patch::walks(),
        1,
        "the first commit's walk proves the file reference-free; the rest \
         must take its word for it"
    );

    let referencing = dir.path().join("referencing.h5");
    let mut b = crate::writer::FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3]);
    b.create_dataset("refs").with_path_references(&["d"]);
    b.write(&referencing).unwrap();
    reference_patch::reset_walks();
    commit_three(&referencing);
    assert_eq!(
        reference_patch::walks(),
        3,
        "a file that holds a reference is never proved free of one, so every \
         commit walks it"
    );
}

#[test]
fn a_raw_reference_target_is_screened_like_a_path_one() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("raw_target.h5");
    let mut b = crate::writer::FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();
    let engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();

    let nodes: BTreeMap<ObjectPathBuf, Node> = BTreeMap::new();
    let path_addr: BTreeMap<ObjectPathBuf, u64> = BTreeMap::new();
    let resolve = |address: u64, removed: Vec<(u64, u64)>, moved: Vec<u64>| {
        WriteEngine::resolve_reference_target(
            &ObjectRefTarget::Raw(StoredAddress::new(address)),
            &path_addr,
            &nodes,
            &[],
            &[],
            &[],
            &InvalidatedAddresses {
                removed: physical_ranges(&removed),
                moved: moved.into_iter().map(PhysicalAddress::new).collect(),
                base: AddressContext::from_effective_base(BaseAddress::ZERO),
            },
            &engine.image(),
            &engine.context,
        )
    };

    common::assert_unsupported(
        resolve(300, vec![(248, 71)], Vec::new()).unwrap_err(),
        REFERENCE_INTO_RECLAIMED_SPACE,
    );
    common::assert_unsupported(
        resolve(300, Vec::new(), vec![300]).unwrap_err(),
        REFERENCE_TO_A_MOVED_OBJECT,
    );
    assert_eq!(
        resolve(300, vec![(400, 71)], vec![299, 301]).unwrap(),
        StoredAddress::new(300),
        "an address outside every reclaimed span and every moved header is carried through"
    );
    assert_eq!(
        resolve(300, Vec::new(), Vec::new()).unwrap(),
        StoredAddress::new(300),
        "a commit that reclaims nothing screens nothing"
    );
    assert_eq!(
        resolve(0, vec![(0, 4096)], vec![0]).unwrap(),
        StoredAddress::new(0)
    );
    assert_eq!(
        resolve(UNDEF, vec![(0, u64::MAX)], vec![UNDEF]).unwrap(),
        StoredAddress::new(UNDEF)
    );
}

#[rstest]
#[case::relative("a/b")]
#[case::absolute("/a/b")]
#[case::interior_dot("a/./b")]
#[case::repeated_separators("a//b/")]
fn reference_targets_resolve_equivalent_path_spellings(#[case] spelling: &str) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("path_target.h5");
    let mut b = crate::writer::FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();
    let engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();

    let nodes: BTreeMap<ObjectPathBuf, Node> = BTreeMap::new();
    let resolve = |path_addr: &BTreeMap<ObjectPathBuf, u64>, add_targets: &[ObjectPathBuf]| {
        WriteEngine::resolve_reference_target(
            &ObjectRefTarget::Path(ObjectPathBuf::parse(spelling)),
            path_addr,
            &nodes,
            add_targets,
            &[],
            &[],
            &InvalidatedAddresses {
                removed: Vec::new(),
                moved: Vec::new(),
                base: AddressContext::from_effective_base(BaseAddress::ZERO),
            },
            &engine.image(),
            &engine.context,
        )
    };

    let placed: BTreeMap<ObjectPathBuf, u64> =
        std::iter::once((ObjectPathBuf::parse("a/b"), 4096)).collect();
    assert_eq!(resolve(&placed, &[]).unwrap(), StoredAddress::new(4096));

    let err = resolve(&BTreeMap::new(), &[ObjectPathBuf::parse("a/b")]).unwrap_err();
    let Error::EditUnsupported(reason) = &err else {
        panic!("expected EditUnsupported, got {err:?}");
    };
    assert_eq!(
        *reason,
        "an object-reference dataset targets a path this commit is still writing; \
         use separate commits"
    );
}

#[rstest]
#[case::start(248, true)]
#[case::last_byte(318, true)]
#[case::end(319, false)]
#[case::before(247, false)]
fn a_copied_reference_attribute_is_screened_in_both_storages(
    #[case] address: u64,
    #[case] rejected: bool,
    #[values(false, true)] dense: bool,
) {
    let empty = BytesSource::new(Vec::new());
    let invalidated = InvalidatedAddresses {
        removed: physical_ranges(&[(248, 71)]),
        moved: Vec::new(),
        base: AddressContext::from_effective_base(BaseAddress::ZERO),
    };
    let attr = reference_attr("target", address);
    let tree = if dense {
        verbatim_copy(
            OhRegion::default(),
            DenseAttrSet::new(vec![attr], DenseAttrCreationOrder::Untracked, []).unwrap(),
        )
    } else {
        verbatim_copy(inline_attr_region(&attr), DenseAttrSet::default())
    };
    let result = copy::screen_copied_references(
        &tree,
        AddressContext::from_effective_base(BaseAddress::ZERO),
        crate::file_writer::WIDTHS,
        &invalidated,
        &empty,
    );
    if rejected {
        common::assert_unsupported(result.unwrap_err(), REFERENCE_INTO_RECLAIMED_SPACE);
    } else {
        result.unwrap();
    }
}

#[test]
fn a_copy_nested_past_the_native_stack_is_screened_written_and_dropped() {
    thread::Builder::new()
        .stack_size(SMALL_STACK)
        .spawn(|| {
            let attr = reference_attr("target", REMOVED_ADDRESS);
            let tree = nested_groups(DEEP_NESTING, attr.clone());
            assert!(tree.has_dense_attributes());

            let invalidated = InvalidatedAddresses {
                removed: physical_ranges(&[(REMOVED_ADDRESS, 1)]),
                moved: Vec::new(),
                base: AddressContext::from_effective_base(BaseAddress::ZERO),
            };
            let err = copy::screen_copied_references(
                &tree,
                AddressContext::from_effective_base(BaseAddress::ZERO),
                crate::file_writer::WIDTHS,
                &invalidated,
                &BytesSource::new(Vec::new()),
            )
            .unwrap_err();
            let Error::EditUnsupported(reason) = err else {
                panic!("expected EditUnsupported, got {err:?}");
            };
            assert_eq!(reason, REFERENCE_INTO_RECLAIMED_SPACE);

            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("nested_copy.h5");
            FileBuilder::new().write(&path).unwrap();
            let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
            let mut addr = engine.write_copy_subtree(&tree).unwrap();
            drop(tree);

            let base = engine.addresses();
            for _ in 0..DEEP_NESTING {
                let object =
                    WriteEngine::read_object(engine.image(), AccessMode::ReadWrite, addr, base)
                        .unwrap();
                let ObjModel::Group {
                    children,
                    dense_attrs,
                    ..
                } = object
                else {
                    panic!("expected a group at {addr}");
                };
                assert!(dense_attrs.is_empty());
                let [(link_name, creation_order, child)] = children.as_slice() else {
                    panic!("expected one link at {addr}, got {children:?}");
                };
                assert_eq!((link_name.as_str(), *creation_order), ("g", None));
                addr = base.resolve(*child).unwrap().get();
            }
            let leaf = WriteEngine::read_object(engine.image(), AccessMode::ReadWrite, addr, base)
                .unwrap();
            let ObjModel::Group {
                children,
                dense_attrs,
                ..
            } = leaf
            else {
                panic!("expected a group at {addr}");
            };
            assert!(children.is_empty(), "{children:?}");
            assert_eq!(dense_attrs.attrs(), [attr]);
        })
        .unwrap()
        .join()
        .unwrap();
}

#[rstest]
#[case::objects(TraversalBudget::new(3, 3), TraversalLimit::Objects, 3)]
#[case::links(TraversalBudget::new(4, 2), TraversalLimit::Links, 2)]
fn a_copy_past_its_traversal_budget_is_rejected(
    #[case] budget: TraversalBudget,
    #[case] expected_limit: TraversalLimit,
    #[case] expected_maximum: usize,
) {
    // Each case allows one fewer of one of the two counts.
    let (bytes, addr) = group_of_three_datasets();
    let Err(err) = copy_reader(&BytesSource::new(bytes.as_slice())).read(budget, addr) else {
        panic!("expected the copy to be rejected");
    };
    let Error::TraversalLimitExceeded { limit, maximum } = err else {
        panic!("expected TraversalLimitExceeded, got {err:?}");
    };
    assert_eq!((limit, maximum), (expected_limit, expected_maximum));
}

#[test]
fn a_copy_within_its_traversal_budget_is_read_in_full() {
    let (bytes, addr) = group_of_three_datasets();
    let tree = copy_reader(&BytesSource::new(bytes.as_slice()))
        .read(TraversalBudget::new(4, 3), addr)
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("budget_copy.h5");
    FileBuilder::new().write(&path).unwrap();
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let addr = engine.write_copy_subtree(&tree).unwrap();
    let ObjModel::Group { children, .. } = WriteEngine::read_object(
        engine.image(),
        AccessMode::ReadWrite,
        addr,
        engine.addresses(),
    )
    .unwrap() else {
        panic!("expected a group at {addr}");
    };
    assert_eq!(
        children
            .iter()
            .map(|(link_name, _, _)| link_name.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b", "c"]
    );
}

#[test]
fn fresh_group_region_pairs_link_info_with_group_info() {
    let types = region_types(&edit::group::fresh_group_region());
    assert_eq!(types, vec![MessageType::LINK_INFO, MessageType::GROUP_INFO]);
}

#[test]
fn converting_version_1_link_messages_drops_the_padding_of_each_body() {
    let link_info = padded(&LINK_INFO_UNTRACKED_COMPACT, 24);
    let group_info = padded(&GROUP_INFO_BODY, 8);
    let link = padded(&hard_link_a(), 16);
    let header = version_1_header(vec![
        (MessageType::LINK_INFO, MessageFlags::CONSTANT, link_info),
        (MessageType::GROUP_INFO, MessageFlags::CONSTANT, group_info),
        (MessageType::LINK, MessageFlags::NONE, link),
    ]);
    let region = edit::group::link_message_region(&header, OffsetWidth::Eight).unwrap();
    let mut records = Vec::new();
    let mut pos = 0;
    while let Some(record) = region.layout().next_message(&region, pos).unwrap() {
        records.push((record.msg_type, record.flags, record.body.to_vec()));
        pos = record.body_range.end;
    }
    assert_eq!(
        records,
        [
            (
                MessageType::LINK_INFO,
                MessageFlags::CONSTANT,
                LINK_INFO_UNTRACKED_COMPACT.to_vec()
            ),
            (
                MessageType::GROUP_INFO,
                MessageFlags::CONSTANT,
                GROUP_INFO_BODY.to_vec()
            ),
            (MessageType::LINK, MessageFlags::NONE, hard_link_a()),
        ]
    );
}

#[rstest]
#[case::shared_flag(
    MessageType::LINK,
    MessageFlags::SHARED,
    padded(&hard_link_a(), 16),
    "a group's link metadata has object-header flags this editor cannot reproduce"
)]
#[case::truncated_group_info(
    MessageType::GROUP_INFO,
    MessageFlags::NONE,
    vec![0, 1, 12, 0],
    "a group's group-info message is truncated"
)]
fn converting_version_1_link_messages_rejects_link_storage_it_cannot_reproduce(
    #[case] msg_type: MessageType,
    #[case] flags: MessageFlags,
    #[case] body: Vec<u8>,
    #[case] reason: &str,
) {
    let header = version_1_header(vec![(msg_type, flags, body)]);
    let Err(error) = edit::group::link_message_region(&header, OffsetWidth::Eight) else {
        panic!("expected the region to be rejected");
    };
    assert!(
        matches!(error, Error::EditUnsupported(actual) if actual == reason),
        "{error:?}"
    );
}

#[rstest]
#[case::truncated_link(MessageType::LINK, hard_link_a()[..8].to_vec(), FormatError::UnexpectedEof { expected: hard_link_a().len(), available: 8 })]
#[case::truncated_link_info(MessageType::LINK_INFO, LINK_INFO_UNTRACKED_COMPACT[..10].to_vec(), FormatError::UnexpectedEof { expected: 18, available: 10 })]
fn converting_version_1_link_messages_rejects_a_truncated_body(
    #[case] msg_type: MessageType,
    #[case] body: Vec<u8>,
    #[case] expected: FormatError,
) {
    let header = version_1_header(vec![(msg_type, MessageFlags::NONE, body)]);
    let Err(error) = edit::group::link_message_region(&header, OffsetWidth::Eight) else {
        panic!("expected the region to be rejected");
    };
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, expected);
}

fn version_1_header(messages: Vec<(MessageType, MessageFlags, Vec<u8>)>) -> ObjectHeader {
    ObjectHeader {
        version: 1,
        messages: messages
            .into_iter()
            .map(|(msg_type, flags, data)| HeaderMessage {
                msg_type,
                size: data.len(),
                flags,
                creation_order: None,
                data,
            })
            .collect(),
        reference_count: Some(1),
        flags: 0,
        access_time: None,
        modification_time: None,
        change_time: None,
        birth_time: None,
    }
}

fn padded(body: &[u8], len: usize) -> Vec<u8> {
    let mut padded = body.to_vec();
    padded.resize(len, 0);
    padded
}

const LINK_INFO_UNTRACKED_COMPACT: [u8; 18] = [
    0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF,
];

fn hard_link_a() -> Vec<u8> {
    HardLink::new("a", 0x100).build(Widths::EIGHT)
}

#[test]
fn ensure_group_info_appends_when_missing() {
    let li_body = {
        let mut b = vec![0u8, 0];
        b.extend_from_slice(&u64::MAX.to_le_bytes());
        b.extend_from_slice(&u64::MAX.to_le_bytes());
        b
    };
    let mut region = plain_region(message_record(MessageType::LINK_INFO, &li_body));
    edit::group::ensure_group_info(&mut region).unwrap();
    assert_eq!(
        region_types(&region),
        vec![MessageType::LINK_INFO, MessageType::GROUP_INFO]
    );
    let mut p = 0;
    while let Some((mt, body, end)) = region.next_message(p).unwrap() {
        if mt == MessageType::GROUP_INFO {
            assert_eq!(&region[body..end], &GROUP_INFO_BODY);
        }
        p = end;
    }
}

#[test]
fn links_added_to_a_tracked_group_take_consecutive_creation_indexes() {
    let mut region = link_tracked_region(3, false, &["a", "b", "c"]);
    let mut order = LinkCreationOrder::for_region(&region).unwrap();
    region.push_link("d", StoredAddress::new(0x200), order.take().unwrap());
    region.push_link("e", StoredAddress::new(0x300), order.take().unwrap());
    order.record(&mut region).unwrap();

    assert_eq!(
        region_links(&region),
        vec![
            ("a".to_string(), Some(0)),
            ("b".to_string(), Some(1)),
            ("c".to_string(), Some(2)),
            ("d".to_string(), Some(3)),
            ("e".to_string(), Some(4)),
        ],
    );
    assert_eq!(recorded_link_max(&region), Some(5));
}

#[test]
fn a_tracked_group_resumes_past_a_gap_a_deletion_left() {
    let region = link_tracked_region(3, false, &["a", "b", "c"]);
    let mut region = edit::group::remove_link_from_region(&region, "a").unwrap();
    region = edit::group::remove_link_from_region(&region, "c").unwrap();
    let mut order = LinkCreationOrder::for_region(&region).unwrap();
    region.push_link("d", StoredAddress::new(0x200), order.take().unwrap());
    order.record(&mut region).unwrap();

    assert_eq!(
        region_links(&region),
        vec![("b".to_string(), Some(1)), ("d".to_string(), Some(3))],
    );
    assert_eq!(recorded_link_max(&region), Some(4));
}

#[test]
fn link_creation_indexing_preserves_index_addresses_while_updating_the_counter() {
    let mut region = link_tracked_region(1, true, &["a"]);
    let mut order = LinkCreationOrder::for_region(&region).unwrap();
    region.push_link("b", StoredAddress::new(0x200), order.take().unwrap());
    order.record(&mut region).unwrap();

    let (_, _, info) = edit::group::find_link_info(&region).unwrap().unwrap();
    assert_eq!(info.max_creation_order, Some(2));
    assert_eq!(info.fractal_heap_address, None);
    assert_eq!(info.btree_name_index_address, None);
    assert_eq!(info.btree_creation_order_address, Some(None));
    assert_eq!(region_links(&region).last().unwrap().1, Some(1));
}

#[test]
fn an_untracked_group_numbers_none_of_its_links() {
    let mut region = edit::group::fresh_group_region();
    let before = region.clone();
    let mut order = LinkCreationOrder::for_region(&region).unwrap();
    assert_eq!(order.take().unwrap(), None);
    region.push_link("d", StoredAddress::new(0x200), None);
    order.record(&mut region).unwrap();

    assert_eq!(region_links(&region), vec![("d".to_string(), None)]);
    assert_eq!(recorded_link_max(&region), None);
    assert_eq!(&region[..before.len()], &*before);
}

#[test]
fn a_tracked_group_that_gains_no_link_keeps_its_maximum_byte_for_byte() {
    let region = link_tracked_region(3, false, &["a", "b", "c"]);
    let mut after = edit::group::remove_link_from_region(&region, "b").unwrap();
    let order = LinkCreationOrder::for_region(&after).unwrap();
    let before = after.clone();
    order.record(&mut after).unwrap();
    assert_eq!(*after, *before);
    assert_eq!(recorded_link_max(&after), Some(3));
}

#[test]
fn ensure_group_info_is_idempotent() {
    let mut region = edit::group::fresh_group_region();
    let before = region.clone();
    edit::group::ensure_group_info(&mut region).unwrap();
    assert_eq!(*region, *before);
}

#[test]
fn a_commit_that_strands_a_shared_message_record_is_rejected() {
    let named = crate::sohm::SohmRecord {
        hash: 1,
        location: crate::sohm::SohmLocation::ObjectHeader {
            message_type: MessageType::ATTRIBUTE.to_u16() as u8,
            creation_index: 0,
            address: StoredAddress::new(0x400),
        },
    };
    let moved = InvalidatedAddresses {
        removed: Vec::new(),
        moved: vec![PhysicalAddress::new(0x400)],
        base: AddressContext::from_effective_base(BaseAddress::ZERO),
    };
    let err =
        edit::screen_shared_message_records(std::slice::from_ref(&named), &moved).unwrap_err();
    assert_eq!(
        err.to_string(),
        Error::EditUnsupported(SHARED_MESSAGE_INDEX_NAMES_A_MOVED_OBJECT).to_string()
    );

    let elsewhere = InvalidatedAddresses {
        removed: Vec::new(),
        moved: vec![PhysicalAddress::new(0x800)],
        base: AddressContext::from_effective_base(BaseAddress::ZERO),
    };
    edit::screen_shared_message_records(std::slice::from_ref(&named), &elsewhere).unwrap();
}

#[test]
fn a_heap_stored_shared_message_record_is_not_screened_for_addresses() {
    let heap = crate::sohm::SohmRecord {
        hash: 1,
        location: crate::sohm::SohmLocation::Heap {
            reference_count: 2,
            heap_id: 0x400u64.to_le_bytes(),
        },
    };
    let moved = InvalidatedAddresses {
        removed: Vec::new(),
        moved: vec![PhysicalAddress::new(0x400)],
        base: AddressContext::from_effective_base(BaseAddress::ZERO),
    };
    edit::screen_shared_message_records(&[heap], &moved).unwrap();
}

#[test]
fn a_version_1_reference_is_rewrapped_as_the_modern_committed_form() {
    let mut v1 = vec![1u8, 0];
    v1.extend_from_slice(&[0u8; 6]); // reserved
    v1.extend_from_slice(&0x1111u64.to_le_bytes()); // name offset in the local heap
    v1.extend_from_slice(&0x5678u64.to_le_bytes()); // object header address

    let modern =
        edit::object_header::modernize_shared_reference(&v1, crate::file_writer::WIDTHS).unwrap();
    assert_eq!(modern[0], 2, "the version this crate writes");
    assert_eq!(
        crate::shared_message::parse_shared_ref_with_widths(&modern, crate::file_writer::WIDTHS,)
            .unwrap()
            .location,
        crate::shared_message::SharedLocation::ObjectHeader(StoredAddress::new(0x5678))
    );
}

#[test]
fn a_heap_reference_is_rewrapped_as_a_heap_reference() {
    let mut v3 = vec![3u8, 1];
    v3.extend_from_slice(&[9, 8, 7, 6, 5, 4, 3, 2]);

    let modern =
        edit::object_header::modernize_shared_reference(&v3, crate::file_writer::WIDTHS).unwrap();
    assert_eq!(modern, v3);
}

#[test]
fn a_rewrapped_shared_message_keeps_its_flag() {
    let mut region = OhRegion::empty(ObjectHeaderPrefix::PLAIN);
    region.push_shared(MessageType::ATTRIBUTE, &[3, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(region[3], MessageFlags::SHARED.get());
    assert!(edit::attributes::region_has_shared_attr(&region).unwrap());
}

#[test]
fn reject_foreign_addresses_rejects_any_shared_message() {
    let mut shared = message_record(MessageType::DATASPACE, &[0u8; 8]);
    shared[3] = MessageFlags::SHARED.get(); // set the message's shared flag
    let err = copy::reject_foreign_addresses(&plain_region(shared)).unwrap_err();
    common::assert_unsupported(
        err,
        "a shared (committed/SOHM) object-header message cannot be copied to another file yet",
    );

    let plain = message_record(MessageType::DATASPACE, &[0u8; 8]);
    copy::reject_foreign_addresses(&plain_region(plain)).unwrap();
}

#[test]
fn a_record_walk_steps_by_the_width_the_header_declares() {
    for layout in [MessageRecordLayout::PLAIN, TRACKED] {
        let region = attr_region(layout, &[("alpha", 0), ("beta", 7)], 8);
        assert_eq!(
            walk_attrs(&region),
            vec![
                (
                    "alpha".to_string(),
                    layout.tracks_creation_order().then_some(0)
                ),
                (
                    "beta".to_string(),
                    layout.tracks_creation_order().then_some(7)
                ),
            ],
            "layout {layout:?}",
        );
    }
    let tracked = attr_region(TRACKED, &[("alpha", 0), ("beta", 7)], 8);
    let misread = OhRegion::new(tracked.to_vec(), ObjectHeaderPrefix::PLAIN);
    let mut p = 0;
    let mut agreed = true;
    while let Ok(Some((_, _, body_end))) = misread.next_message(p) {
        p = body_end;
    }
    agreed &= p == tracked.len();
    assert!(
        !agreed,
        "a four-byte walk of a six-byte-record region must not come out even"
    );
}

#[test]
fn a_rebuilt_header_declares_the_record_layout_it_used() {
    for layout in [MessageRecordLayout::PLAIN, TRACKED] {
        let region = attr_region(layout, &[("alpha", 3)], 4);
        let header = edit::object_header::build_v2_object_header(&region).unwrap();
        let (start, end, read_back) =
            edit::object_header::oh_region_at(&header, 0, header.len() as u64)
                .expect("the header parses");
        assert_eq!(
            read_back,
            ObjectHeaderPrefix::with_layout(layout),
            "the flags lost the layout"
        );
        let round_tripped = OhRegion::new(header[start as usize..end as usize].to_vec(), read_back);
        assert_eq!(
            walk_attrs(&round_tripped),
            walk_attrs(&region),
            "layout {layout:?}",
        );
    }
}

#[test]
fn a_headers_optional_prefix_blocks_survive_a_rebuild() {
    for layout in [MessageRecordLayout::PLAIN, TRACKED] {
        for times in [None, Some(FIXTURE_TIMES)] {
            for attribute_phase_change in [None, Some(FIXTURE_PHASE)] {
                let props = ObjectHeaderPrefix {
                    layout,
                    times,
                    attribute_phase_change,
                };
                let region = attr_region_in(props, &[("alpha", 3), ("beta", 5)], 6);
                let header = v2_header_bytes(props, &region);

                let (start, end, read_back) =
                    edit::object_header::oh_region_at(&header, 0, header.len() as u64)
                        .expect("the hand-built header parses");
                assert_eq!(read_back, props, "the prefix lost a block");
                let parsed =
                    OhRegion::new(header[start as usize..end as usize].to_vec(), read_back);
                assert_eq!(
                    walk_attrs(&parsed),
                    walk_attrs(&region),
                    "the message region was located wrongly for {props:?}",
                );

                let rebuilt = edit::object_header::build_v2_object_header(&parsed).unwrap();
                let (_, _, again) =
                    edit::object_header::oh_region_at(&rebuilt, 0, rebuilt.len() as u64)
                        .expect("the rebuilt header parses");
                assert_eq!(
                    again.attribute_phase_change, attribute_phase_change,
                    "the rebuild lost the attribute phase-change thresholds",
                );
                assert_eq!(
                    again.times.is_some(),
                    times.is_some(),
                    "the rebuild changed whether the header stores times",
                );
                assert_eq!(again.layout, layout, "the rebuild lost the record layout");
            }
        }
    }
}

#[cfg(feature = "std")]
#[test]
fn a_rebuild_stamps_the_modification_and_change_times() {
    let props = ObjectHeaderPrefix {
        times: Some(FIXTURE_TIMES),
        ..ObjectHeaderPrefix::PLAIN
    };
    let region = attr_region_in(props, &[("alpha", 0)], 1);

    let before = edit::object_header::unix_time_now().expect("a std test build reads the clock");
    let header = edit::object_header::build_v2_object_header(&region).unwrap();
    let after = edit::object_header::unix_time_now().expect("a std test build reads the clock");

    let (_, _, read_back) =
        edit::object_header::oh_region_at(&header, 0, header.len() as u64).unwrap();
    let times = read_back.times.expect("the header still stores times");
    assert_eq!(
        (times.access, times.birth),
        (FIXTURE_TIMES.access, FIXTURE_TIMES.birth),
        "a rewrite is not an access and not a birth",
    );
    assert!(
        (before..=after).contains(&times.modification),
        "the modification time {} is outside [{before}, {after}]",
        times.modification,
    );
    assert!(
        (before..=after).contains(&times.change),
        "the change time {} is outside [{before}, {after}]",
        times.change,
    );
}

#[rstest]
#[case::length_6(6)]
#[case::length_7(7)]
#[case::length_8(8)]
#[case::length_9(9)]
#[case::length_10(10)]
#[case::length_11(11)]
#[case::length_12(12)]
#[case::length_13(13)]
#[case::length_14(14)]
#[case::length_15(15)]
#[case::length_16(16)]
#[case::length_17(17)]
#[case::length_18(18)]
#[case::length_19(19)]
#[case::length_20(20)]
#[case::length_21(21)]
#[case::length_22(22)]
#[case::length_23(23)]
#[case::length_24(24)]
#[case::length_25(25)]
#[case::length_26(26)]
fn a_truncated_prefix_with_optional_blocks_is_rejected(#[case] cut: usize) {
    let props = ObjectHeaderPrefix {
        times: Some(FIXTURE_TIMES),
        attribute_phase_change: Some(FIXTURE_PHASE),
        ..ObjectHeaderPrefix::PLAIN
    };
    let region = attr_region_in(props, &[("alpha", 0)], 1);
    let header = v2_header_bytes(props, &region);
    common::assert_unsupported(
        edit::object_header::oh_region_at(&header[..cut], 0, header.len() as u64).unwrap_err(),
        "truncated object header",
    );
}

#[test]
fn a_new_compact_attribute_takes_the_next_creation_index() {
    let region = attr_region(TRACKED, &[("alpha", 0), ("beta", 1)], 2);
    let out = edit::attributes::set_attr_in_region(&region, "gamma", &AttrValue::I64(5)).unwrap();
    assert_eq!(
        walk_attrs(&out),
        vec![
            ("alpha".to_string(), Some(0)),
            ("beta".to_string(), Some(1)),
            ("gamma".to_string(), Some(2)),
        ],
    );
    let (record, _, info) = edit::attributes::find_attribute_info(&out)
        .unwrap()
        .expect("an info message");
    assert_eq!(info.max_creation_index, Some(3));
    assert!(
        info.indexes_creation_order,
        "the index flag survives a rewrite"
    );
    assert_eq!(
        out[record + 3],
        MessageFlags::FORBID_SHARING.get(),
        "the rewritten record lost the message flags it carried"
    );
}

#[rstest]
#[case::single_info(false)]
#[case::repeated_info(true)]
fn an_attribute_info_rewrite_preserves_record_metadata_and_unrelated_bytes(
    #[values(
        TRACKED,
        MessageRecordLayout::from_header_flags(HeaderFlags::TRACKS_CREATION_ORDER.0)
    )]
    layout: MessageRecordLayout,
    #[case] repeated: bool,
) {
    let props = ObjectHeaderPrefix {
        layout,
        times: Some(FIXTURE_TIMES),
        attribute_phase_change: Some(FIXTURE_PHASE),
    };
    let records = |max, indexed| {
        let info = AttributeInfoMessage {
            max_creation_index: Some(max),
            indexes_creation_order: indexed,
            fractal_heap_address: None,
            btree_name_index_address: None,
            btree_creation_order_address: None,
        }
        .serialize(OFFSET_WIDTH);
        let first_info = layout.record_with_creation_index(
            MessageType::ATTRIBUTE_INFO,
            MessageFlags::FORBID_SHARING,
            9,
            &info,
        );
        let second_info = if repeated {
            layout.record_with_creation_index(
                MessageType::ATTRIBUTE_INFO,
                MessageFlags::CONSTANT,
                12,
                &info,
            )
        } else {
            Vec::new()
        };
        [
            layout.record_with_creation_index(
                UNKNOWN_MESSAGE_TYPE,
                MessageFlags::CONSTANT,
                4,
                &[0xAB; 3],
            ),
            first_info,
            layout.record_with_creation_index(
                MessageType::ATTRIBUTE,
                MessageFlags::NONE,
                0,
                &attr_body("alpha", 1),
            ),
            second_info,
        ]
        .concat()
    };
    let source = OhRegion::new([records(1, false), vec![0xCD; 2]].concat(), props);

    let out = edit::attributes::set_attr_in_region(&source, "beta", &AttrValue::I64(2)).unwrap();

    assert_eq!(out.props(), props);
    assert_eq!(
        &out[..],
        &[
            records(2, layout.indexes_creation_order()),
            layout.record_with_creation_index(
                MessageType::ATTRIBUTE,
                MessageFlags::NONE,
                1,
                &attr_body("beta", 2),
            ),
            vec![0xCD; 2],
        ]
        .concat(),
    );
}

#[test]
fn an_overwritten_compact_attribute_keeps_its_creation_index() {
    let region = attr_region(TRACKED, &[("alpha", 0), ("beta", 1)], 2);
    assert_eq!(
        edit::attributes::attr_creation_index(&region, "alpha").unwrap(),
        Some(0)
    );
    let out = edit::attributes::set_attr_in_region(&region, "alpha", &AttrValue::I64(9)).unwrap();
    assert_eq!(
        walk_attrs(&out),
        vec![
            ("beta".to_string(), Some(1)),
            ("alpha".to_string(), Some(0))
        ],
    );
    let (_, _, info) = edit::attributes::find_attribute_info(&out)
        .unwrap()
        .expect("an info message");
    assert_eq!(
        info.max_creation_index,
        Some(2),
        "an overwrite must not advance the counter"
    );
}

#[test]
fn a_deletion_leaves_the_creation_index_counter_where_it_was() {
    let region = attr_region(TRACKED, &[("alpha", 0), ("beta", 1), ("gamma", 2)], 3);
    let after = edit::attributes::remove_attr_from_region(&region, "gamma", true).unwrap();
    assert_eq!(
        walk_attrs(&after),
        vec![
            ("alpha".to_string(), Some(0)),
            ("beta".to_string(), Some(1))
        ],
    );
    assert_eq!(
        edit::attributes::next_attr_creation_index(&after).unwrap(),
        3,
        "the highest index still in use is 1, but the counter is 3"
    );

    let out =
        edit::attributes::put_attr_message(&after, "delta", &attr_body("delta", 1), None).unwrap();
    assert_eq!(
        edit::attributes::attr_creation_index(&out, "delta").unwrap(),
        Some(3)
    );
}

#[test]
fn a_tracked_header_without_an_info_message_gains_one() {
    let mut region = OhRegion::empty(ObjectHeaderPrefix::with_layout(TRACKED));
    for (name, index) in [("alpha", 0u16), ("beta", 4)] {
        let record = TRACKED.record_with_creation_index(
            MessageType::ATTRIBUTE,
            MessageFlags::NONE,
            index,
            &attr_body(name, 1),
        );
        region.push_bytes(&record);
    }
    assert!(
        edit::attributes::find_attribute_info(&region)
            .unwrap()
            .is_none()
    );
    edit::attributes::ensure_attribute_info(&mut region).unwrap();
    let (_, _, info) = edit::attributes::find_attribute_info(&region)
        .unwrap()
        .expect("an info message");
    assert_eq!(info.max_creation_index, Some(5));
    assert!(info.indexes_creation_order);
}

#[rstest]
#[case::alone(None)]
#[case::compact(Some(layout_bytes::compact(&[0xEE; 2])))]
#[case::contiguous(Some(layout_bytes::contiguous(0x800, 4)))]
#[case::chunked(Some(layout_bytes::chunked(0x900, &[4], 1)))]
fn rebuild_compact_layout_replaces_only_the_first_layout(
    #[values(MessageRecordLayout::PLAIN, TRACKED)] record_layout: MessageRecordLayout,
    #[case] later_layout: Option<Vec<u8>>,
) {
    let region = |inline: &[u8]| {
        [
            record_layout.record_with_creation_index(
                UNKNOWN_MESSAGE_TYPE,
                MessageFlags::CONSTANT,
                1,
                &[0xAB; 3],
            ),
            record_layout.record(MessageType::DATASPACE, &[0xAB; 8]),
            record_layout.record_with_creation_index(
                MessageType::DATA_LAYOUT,
                MessageFlags::CONSTANT,
                7,
                &layout_bytes::compact(inline),
            ),
            later_layout
                .as_deref()
                .map(|later| record_layout.record(MessageType::DATA_LAYOUT, later))
                .unwrap_or_default(),
            record_layout.record(MessageType::ATTRIBUTE, &[0xCD; 5]),
        ]
        .concat()
    };
    let props = ObjectHeaderPrefix {
        layout: record_layout,
        times: Some(FIXTURE_TIMES),
        attribute_phase_change: Some(FIXTURE_PHASE),
    };

    let out = edit::object_header::rebuild_compact_layout_region(
        &OhRegion::new(region(&[1, 2, 3, 4]), props),
        &[9, 8, 7, 6, 5],
    )
    .unwrap();

    assert_eq!(
        (out.props(), &out[..]),
        (props, &region(&[9, 8, 7, 6, 5])[..])
    );
}

#[rstest]
#[case::contiguous_only(
    [
        message_record(MessageType::DATA_LAYOUT, &layout_bytes::contiguous(0, 0)),
        message_record(MessageType::DATASPACE, &[0; 8]),
    ]
    .concat(),
    vec![1, 2],
    "compact-layout overwrite found a non-compact data layout",
)]
#[case::contiguous_before_compact(
    [
        message_record(MessageType::DATA_LAYOUT, &layout_bytes::contiguous(0, 0)),
        message_record(MessageType::DATA_LAYOUT, &layout_bytes::compact(&[3, 4])),
    ]
    .concat(),
    vec![1, 2],
    "compact-layout overwrite found a non-compact data layout",
)]
#[case::dataspace_only(
    message_record(MessageType::DATASPACE, &[0; 8]),
    vec![1, 2],
    "compact dataset header has no data-layout message",
)]
#[case::too_large(
    message_record(MessageType::DATA_LAYOUT, &layout_bytes::compact(&[3, 4])),
    vec![0; OBJECT_HEADER_MESSAGE_MAX - COMPACT_DATA_OFFSET + 1],
    "compact dataset data is too large to overwrite in place",
)]
fn rebuild_compact_layout_rejects(
    #[case] region: Vec<u8>,
    #[case] raw: Vec<u8>,
    #[case] expected: &str,
) {
    let err = edit::object_header::rebuild_compact_layout_region(&plain_region(region), &raw)
        .unwrap_err();
    let Error::EditUnsupported(reason) = err else {
        panic!("expected EditUnsupported, got {err:?}");
    };
    assert_eq!(reason, expected);
}

#[rstest]
fn replace_layout_message_replaces_only_the_first_layout_record(
    #[values(MessageRecordLayout::PLAIN, TRACKED)] record_layout: MessageRecordLayout,
) {
    let region = rewrite_fixture(
        record_layout,
        records_around(
            record_layout,
            dataspace_record(record_layout),
            compact_layout_record(record_layout, &[1, 2, 3, 4]),
        ),
    );
    let body = layout_bytes::contiguous(0x900, 16);

    let out = edit::dataset::replace_layout_message(&region, &body).unwrap();

    let expected = records_around(
        record_layout,
        dataspace_record(record_layout),
        record_layout.record(MessageType::DATA_LAYOUT, &body),
    );
    assert_eq!((out.props(), &out[..]), (region.props(), &expected[..]));
}

#[rstest]
fn replace_dataspace_message_replaces_only_the_first_dataspace_record(
    #[values(MessageRecordLayout::PLAIN, TRACKED)] record_layout: MessageRecordLayout,
) {
    let region = rewrite_fixture(
        record_layout,
        records_around(
            record_layout,
            dataspace_record(record_layout),
            compact_layout_record(record_layout, &[1, 2, 3, 4]),
        ),
    );
    let body = [0x22; 12];

    let out = edit::dataset::replace_dataspace_message(&region, &body).unwrap();

    let expected = records_around(
        record_layout,
        record_layout.record(MessageType::DATASPACE, &body),
        compact_layout_record(record_layout, &[1, 2, 3, 4]),
    );
    assert_eq!((out.props(), &out[..]), (region.props(), &expected[..]));
}

#[rstest]
#[case::replace_layout_message(edit::dataset::replace_layout_message)]
#[case::replace_dataspace_message(edit::dataset::replace_dataspace_message)]
#[case::rebuild_compact_layout_region(edit::object_header::rebuild_compact_layout_region)]
fn a_region_rewrite_rejects_a_record_past_the_region(
    #[case] rewrite: fn(&OhRegion, &[u8]) -> Result<OhRegion, Error>,
) {
    let mut bytes = message_record(MessageType::DATASPACE, &[0; 8]);
    bytes.pop();

    let err = rewrite(&plain_region(bytes), &[1, 2]).unwrap_err();

    let Error::EditUnsupported(reason) = err else {
        panic!("expected EditUnsupported, got {err:?}");
    };
    assert_eq!(reason, "malformed object header message");
}

#[test]
fn replace_layout_message_rejects_a_header_without_a_layout() {
    let region = plain_region(message_record(MessageType::DATASPACE, &[0; 8]));

    let err = edit::dataset::replace_layout_message(&region, &[1, 2]).unwrap_err();

    let Error::EditUnsupported(reason) = err else {
        panic!("expected EditUnsupported, got {err:?}");
    };
    assert_eq!(
        reason,
        "chunked dataset header has no data-layout message to relocate"
    );
}

#[test]
fn replace_dataspace_message_rejects_a_header_without_a_dataspace() {
    let region = plain_region(message_record(
        MessageType::DATA_LAYOUT,
        &layout_bytes::compact(&[3, 4]),
    ));

    let err = edit::dataset::replace_dataspace_message(&region, &[1, 2]).unwrap_err();

    let Error::AppendUnsupported(reason) = err else {
        panic!("expected AppendUnsupported, got {err:?}");
    };
    assert_eq!(reason, "dataset header has no dataspace message to grow");
}

#[test]
fn edit_session_reports_root_address_overflow_when_resolving_a_path() {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edit_root_overflow.h5");

    const UB: u64 = 512;
    let mut b = FileBuilder::new();
    b.with_userblock(UserblockSize::try_from(UB).unwrap());
    b.create_dataset("d").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();

    let mut data = std::fs::read(&path).unwrap();
    let LocatedSuperblock {
        address,
        superblock: mut sb,
        ..
    } = crate::superblock::read(data.as_slice()).unwrap();
    let off = usize::try_from(address.get()).unwrap();
    assert_eq!(
        sb.base_address,
        BaseAddress::new(UB),
        "userblock file must have base == UB"
    );
    sb.root_group_address = StoredAddress::undefined(OffsetWidth::Eight);
    let bytes = hdf5_pure_format::__private::serialize_superblock(&sb).unwrap();
    data[off..off + bytes.len()].copy_from_slice(&bytes);
    std::fs::write(&path, &data).unwrap();

    let session = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    assert_eq!(
        session.context.superblock().root_group_address,
        StoredAddress::undefined(OffsetWidth::Eight)
    );
    let err = crate::group_v2::resolve_path_any_from_source(
        session.image(),
        AccessMode::ReadWrite,
        session.context.superblock(),
        session.context.widths(),
        session.context.addresses(),
        &ObjectPath::parse("d"),
    )
    .map_err(Error::from)
    .unwrap_err();
    match err {
        Error::Format(FormatError::OffsetOverflow { offset, length }) => {
            assert_eq!(offset, u64::MAX);
            assert_eq!(length, UB);
        }
        other => panic!("expected root-group address overflow, got {other:?}"),
    }
}

#[test]
fn malformed_message_region_returns_a_header_build_error() {
    let mut region = vec![0x0Cu8]; // Attribute
    region.extend_from_slice(&0xFFFFu16.to_le_bytes());
    region.push(0); // flags
    region.extend_from_slice(&[0u8; 4]); // a body far shorter than declared

    let err = edit::object_header::build_v2_object_header(&OhRegion::new(
        region,
        ObjectHeaderPrefix::PLAIN,
    ))
    .unwrap_err();
    common::assert_unsupported(err, "malformed object header message");
}

#[test]
fn rebuilt_inline_attributes_include_attribute_info() {
    let body = [0u8; 8];
    let mut region = vec![0x0Cu8]; // Attribute
    region.extend_from_slice(&(body.len() as u16).to_le_bytes());
    region.push(0); // flags
    region.extend_from_slice(&body);

    let oh = edit::object_header::build_v2_object_header(&OhRegion::new(
        region.clone(),
        ObjectHeaderPrefix::PLAIN,
    ))
    .unwrap();
    assert_eq!(&oh[..4], b"OHDR");
    assert!(
        oh.len() > 8 + region.len() + 4,
        "the wrapped header did not grow by an Attribute Info message"
    );
}

/// Reference layouts whose object addresses cannot be screened.
enum UnreadableReference {
    Compound,
    VariableLength,
    Wide,
}

impl UnreadableReference {
    fn into_fixture(self) -> (Datatype, Vec<u8>) {
        match self {
            Self::Compound => (
                Datatype::Compound {
                    size: 8,
                    members: vec![crate::datatype::__private::compound_member(
                        "r".to_string(),
                        4,
                        Datatype::Reference {
                            size: 8,
                            ref_type: ReferenceType::Object,
                        },
                    )],
                },
                vec![0; 8],
            ),
            Self::VariableLength => (
                Datatype::VariableLength(VariableLengthType::sequence(Datatype::Reference {
                    size: 8,
                    ref_type: ReferenceType::Object,
                })),
                vec![0; 32],
            ),
            Self::Wide => {
                let mut raw = vec![0; 16];
                raw[..8].copy_from_slice(&300u64.to_le_bytes());
                (
                    Datatype::Reference {
                        size: 16,
                        ref_type: ReferenceType::Object,
                    },
                    raw,
                )
            }
        }
    }
}

/// Constructs an object-reference attribute by replacing a `u64` attribute's datatype.
fn reference_attr(name: &str, address: u64) -> crate::attribute::AttributeMessage {
    let mut attr =
        crate::type_builders::build_attr_message(name, &AttrValue::U64(address)).unwrap();
    attr.datatype = Datatype::Reference {
        size: 8,
        ref_type: crate::datatype::ReferenceType::Object,
    };
    assert_eq!(
        attr.raw_data,
        address.to_le_bytes(),
        "the value is the address"
    );
    attr
}

/// Returns a file with a group `g` of three datasets, four objects and three hard links, and the
/// address of the header of `g`.
fn group_of_three_datasets() -> (Vec<u8>, u64) {
    let mut builder = FileBuilder::new();
    let mut group = builder.create_group("g");
    for link_name in ["a", "b", "c"] {
        group.create_dataset(link_name).with_i32_data(&[1]);
    }
    builder.add_group(group.finish());
    let bytes = builder.finish().unwrap();
    let addr = crate::reader::File::from_bytes(bytes.clone())
        .unwrap()
        .group("g")
        .unwrap()
        .header_address()
        .unwrap();
    (bytes, addr)
}

fn copy_reader<'a>(src: &'a BytesSource<&'a [u8]>) -> CopyReader<'a, BytesSource<&'a [u8]>> {
    CopyReader {
        src,
        access_mode: AccessMode::ReadWrite,
        cross_file: true,
        base: AddressContext::from_effective_base(BaseAddress::ZERO),
        widths: crate::file_writer::WIDTHS,
    }
}

fn verbatim_copy(region: OhRegion, dense_attrs: DenseAttrSet) -> CopyTree {
    CopyTreeBuilder::default().finish(CopyNode::DatasetVerbatim {
        region,
        dense_attrs,
    })
}

fn nested_groups(depth: usize, leaf_attr: AttributeMessage) -> CopyTree {
    let mut nodes = CopyTreeBuilder::default();
    let leaf = CopyNode::Group {
        non_link_region: edit::group::fresh_group_region(),
        links: Vec::new(),
        dense_attrs: DenseAttrSet::new(vec![leaf_attr], DenseAttrCreationOrder::Untracked, [])
            .unwrap(),
    };
    let root = (0..depth).fold(leaf, |child, _| CopyNode::Group {
        non_link_region: edit::group::fresh_group_region(),
        links: vec![CopyLink {
            name: "g".into(),
            creation_order: None,
            target: nodes.push(child),
        }],
        dense_attrs: DenseAttrSet::default(),
    });
    nodes.finish(root)
}

fn message_record(msg_type: MessageType, body: &[u8]) -> Vec<u8> {
    MessageRecordLayout::PLAIN.record(msg_type, body)
}

fn plain_region(bytes: Vec<u8>) -> OhRegion {
    OhRegion::new(bytes, ObjectHeaderPrefix::PLAIN)
}

fn hard_link_region(trailing: &[u8]) -> OhRegion {
    let mut body = file_writer::make_link("child", StoredAddress::new(0x1000))
        .serialize(crate::file_writer::OFFSET_WIDTH);
    body.extend_from_slice(trailing);
    plain_region(message_record(MessageType::LINK, &body))
}

fn inline_attr_region(attr: &crate::attribute::AttributeMessage) -> OhRegion {
    plain_region(message_record(
        MessageType::ATTRIBUTE,
        &attr
            .serialize_v3(crate::file_writer::OFFSET_WIDTH, LENGTH_SIZE)
            .unwrap(),
    ))
}

/// Constructs a one-element object-reference dataset region with compact storage.
///
/// The version 3 layout is defined in "The Data Layout Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
fn compact_reference_region(address: u64) -> OhRegion {
    let mut region = message_record(
        MessageType::DATATYPE,
        &hdf5_pure_format::__private::serialize_datatype(
            &crate::type_builders::make_object_reference_type(),
        ),
    );
    let mut layout = vec![3u8, 0];
    layout.extend_from_slice(&8u16.to_le_bytes());
    layout.extend_from_slice(&address.to_le_bytes());
    region.extend_from_slice(&message_record(MessageType::DATA_LAYOUT, &layout));
    plain_region(region)
}

fn region_types(region: &OhRegion) -> Vec<MessageType> {
    let mut out = Vec::new();
    let mut p = 0;
    while let Some((mt, _, end)) = region.next_message(p).unwrap() {
        out.push(mt);
        p = end;
    }
    out
}

/// Constructs compact links with creation indexes starting at zero and a counter of `max`.
///
/// The version 0 Link Info fields are defined in "The Link Info Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_linkinfo
fn link_tracked_region(max: u64, indexed: bool, links: &[&str]) -> OhRegion {
    let mut li = vec![0u8, if indexed { 0x03 } else { 0x01 }];
    li.extend_from_slice(&max.to_le_bytes());
    li.extend_from_slice(&u64::MAX.to_le_bytes()); // fractal heap: compact
    li.extend_from_slice(&u64::MAX.to_le_bytes()); // b-tree name index
    if indexed {
        li.extend_from_slice(&u64::MAX.to_le_bytes()); // b-tree creation order
    }
    let mut region = OhRegion::empty(ObjectHeaderPrefix::PLAIN);
    region.push(MessageType::LINK_INFO, &li);
    region.push(MessageType::GROUP_INFO, &GROUP_INFO_BODY);
    for (i, name) in links.iter().enumerate() {
        region.push_link(name, StoredAddress::new(0x100 + i as u64), Some(i as u64));
    }
    region
}

fn region_links(region: &OhRegion) -> Vec<(String, Option<u64>)> {
    let mut out = Vec::new();
    let mut p = 0;
    while let Some((mt, body, end)) = region.next_message(p).unwrap() {
        if mt == MessageType::LINK {
            let link =
                LinkMessage::parse_with_offset_width(&region[body..end], OFFSET_WIDTH).unwrap();
            out.push((link.name, link.creation_order));
        }
        p = end;
    }
    out
}

fn recorded_link_max(region: &OhRegion) -> Option<u64> {
    edit::group::find_link_info(region)
        .unwrap()
        .and_then(|(_, _, info)| info.max_creation_order)
}

/// The record layout for tracked and indexed attribute creation order.
///
/// Bits 2 and 3 of a version 2 header's flags are defined in "Version 2 Data Object Header
/// Prefix" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
const TRACKED: MessageRecordLayout = MessageRecordLayout::from_header_flags(0x0C);
fn attr_body(name: &str, value: i64) -> Vec<u8> {
    crate::type_builders::build_attr_message(name, &AttrValue::I64(value))
        .unwrap()
        .serialize(crate::file_writer::OFFSET_WIDTH, LENGTH_SIZE)
        .unwrap()
}

fn attr_region(layout: MessageRecordLayout, attrs: &[(&str, u16)], max: u16) -> OhRegion {
    attr_region_in(ObjectHeaderPrefix::with_layout(layout), attrs, max)
}

/// Constructs inline attributes, recording creation indexes and `max` when tracking is enabled.
fn attr_region_in(props: ObjectHeaderPrefix, attrs: &[(&str, u16)], max: u16) -> OhRegion {
    let layout = props.layout;
    let mut region = OhRegion::empty(props);
    let info = AttributeInfoMessage {
        max_creation_index: layout.tracks_creation_order().then_some(max),
        indexes_creation_order: layout.indexes_creation_order(),
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: None,
    };
    let mut record = layout.record(
        MessageType::ATTRIBUTE_INFO,
        &info.serialize(crate::file_writer::OFFSET_WIDTH),
    );
    record[3] = MessageFlags::FORBID_SHARING.get();
    region.push_bytes(&record);
    for (name, index) in attrs {
        let record = layout.record_with_creation_index(
            MessageType::ATTRIBUTE,
            MessageFlags::NONE,
            *index,
            &attr_body(name, 1),
        );
        region.push_bytes(&record);
    }
    region
}

fn walk_attrs(region: &OhRegion) -> Vec<(String, Option<u16>)> {
    let mut out = Vec::new();
    let mut p = 0;
    while let Some((msg_type, body, body_end)) = region.next_message(p).unwrap() {
        if msg_type == MessageType::ATTRIBUTE {
            let name =
                edit::attributes::parse_compact_attr_name(region, p, body, body_end).unwrap();
            out.push((name, region.creation_index(p)));
        }
        p = body_end;
    }
    out
}

/// Distinct timestamps used to detect swapped or omitted fields.
const FIXTURE_TIMES: ObjectTimes = ObjectTimes {
    access: 0x1111_1111,
    modification: 0x2222_2222,
    change: 0x3333_3333,
    birth: 0x4444_4444,
};

/// Attribute phase-change values used to detect omitted prefix fields.
const FIXTURE_PHASE: AttributePhaseChange = AttributePhaseChange {
    max_compact: 32,
    min_dense: 24,
};

/// Constructs a version 2 header independently of the editor's header serializer.
///
/// The prefix uses a one-byte chunk size and optional fields from `props`, as defined in
/// "Version 2 Data Object Header Prefix" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
fn v2_header_bytes(props: ObjectHeaderPrefix, region: &OhRegion) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"OHDR");
    buf.push(2); // version
    let mut flags = HeaderFlags::default();
    if props.layout == TRACKED {
        flags = flags | HeaderFlags::TRACKS_CREATION_ORDER | HeaderFlags::INDEXES_CREATION_ORDER;
    }
    if props.times.is_some() {
        flags = flags | HeaderFlags::STORES_TIMES;
    }
    if props.attribute_phase_change.is_some() {
        flags = flags | HeaderFlags::STORES_ATTRIBUTE_PHASE_CHANGE;
    }
    buf.push(flags.0); // size flags 0: a 1-byte length field
    if let Some(times) = props.times {
        for time in [times.access, times.modification, times.change, times.birth] {
            buf.extend_from_slice(&time.to_le_bytes());
        }
    }
    if let Some(phase) = props.attribute_phase_change {
        buf.extend_from_slice(&phase.max_compact.to_le_bytes());
        buf.extend_from_slice(&phase.min_dense.to_le_bytes());
    }
    buf.push(u8::try_from(region.len()).expect("the fixture region is under 256 bytes"));
    buf.extend_from_slice(region);
    let checksum = checksum::jenkins_lookup3(&buf);
    buf.extend_from_slice(&checksum.to_le_bytes());
    buf
}

fn rewrite_fixture(record_layout: MessageRecordLayout, bytes: Vec<u8>) -> OhRegion {
    OhRegion::new(
        bytes,
        ObjectHeaderPrefix {
            layout: record_layout,
            times: Some(FIXTURE_TIMES),
            attribute_phase_change: Some(FIXTURE_PHASE),
        },
    )
}

fn records_around(
    record_layout: MessageRecordLayout,
    dataspace: Vec<u8>,
    data_layout: Vec<u8>,
) -> Vec<u8> {
    [
        record_layout.record_with_creation_index(
            UNKNOWN_MESSAGE_TYPE,
            MessageFlags::CONSTANT,
            1,
            &[0xAB; 3],
        ),
        dataspace,
        data_layout,
        record_layout.record_with_creation_index(
            MessageType::ATTRIBUTE,
            MessageFlags::NONE,
            2,
            &[0xCD; 5],
        ),
        record_layout.record(MessageType::DATASPACE, &[0xEE; 8]),
        record_layout.record(
            MessageType::DATA_LAYOUT,
            &layout_bytes::contiguous(0x800, 4),
        ),
    ]
    .concat()
}

fn dataspace_record(record_layout: MessageRecordLayout) -> Vec<u8> {
    record_layout.record_with_creation_index(
        MessageType::DATASPACE,
        MessageFlags::CONSTANT,
        3,
        &[0x11; 8],
    )
}

fn compact_layout_record(record_layout: MessageRecordLayout, inline: &[u8]) -> Vec<u8> {
    record_layout.record_with_creation_index(
        MessageType::DATA_LAYOUT,
        MessageFlags::CONSTANT,
        4,
        &layout_bytes::compact(inline),
    )
}

fn physical_ranges(bounds: &[(u64, u64)]) -> Vec<FileRange> {
    bounds
        .iter()
        .map(|&(at, length)| FileRange::new(PhysicalAddress::new(at), length).unwrap())
        .collect()
}

const UNKNOWN_MESSAGE_TYPE: MessageType = MessageType::from_u16(0x00C8);
const DEEP_NESTING: usize = 100_000;
const REMOVED_ADDRESS: u64 = 248;
const SMALL_STACK: usize = 256 * 1024;
