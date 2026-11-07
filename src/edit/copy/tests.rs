use core::num::NonZeroU32;

use hdf5_pure_format::__private::ObjectHeaderPrefix;
use rstest::rstest;
use test_util::attribute::Attribute;
use test_util::datatype;
use test_util::dense_attributes;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags;
use test_util::object_header::MessageType;
use test_util::object_header::v2;
use test_util::widths::Widths;

use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::error::{Error, FormatError};
use crate::object_graph::ObjectId;
use crate::reader::OpenedSource;
use crate::source::BytesSource;
use crate::superblock;
use crate::width::{LengthWidth, OffsetWidth};

use super::super::attributes::DenseAttrSet;
use super::super::object_header::OhRegion;
use super::{CopyLink, CopyNode, CopyNodeId, CopyTree, FreshCopyCount, WriteEngine};

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod allocation;

#[rstest]
#[case::root(vec![], vec![], vec![])]
#[case::child(vec![vec![]], vec![0], vec![1])]
#[case::aliases(vec![vec![]], vec![0, 0], vec![2])]
#[case::nested(vec![vec![], vec![0]], vec![1], vec![1, 1])]
fn fresh_copy_counts_follow_destination_links(
    #[case] descendants: Vec<Vec<usize>>,
    #[case] root: Vec<usize>,
    #[case] expected: Vec<u32>,
) {
    let tree = CopyTree {
        descendants: descendants.into_iter().map(group_node).collect(),
        root: group_node(root),
    };
    assert_eq!(
        tree.fresh_counts()
            .unwrap()
            .into_iter()
            .map(|count| count.0.get())
            .collect::<Vec<_>>(),
        expected
    );
}

#[rstest]
#[case::unlinked(vec![vec![]], vec![], "a copied descendant has no destination link")]
#[case::missing(vec![], vec![0], "a copied link has no destination object")]
fn fresh_copy_counts_reject_incomplete_destination_topology(
    #[case] descendants: Vec<Vec<usize>>,
    #[case] root: Vec<usize>,
    #[case] expected: &str,
) {
    let tree = CopyTree {
        descendants: descendants.into_iter().map(group_node).collect(),
        root: group_node(root),
    };
    let Err(error) = tree.fresh_counts() else {
        panic!("an incomplete topology must be rejected");
    };
    assert!(
        matches!(&error, Error::Format(FormatError::Internal(reason)) if reason == expected),
        "{error:?}"
    );
}

#[rstest]
#[case::implicit(vec![], 1, None)]
#[case::inserted(vec![], 2, None)]
#[case::explicit(vec![Message::reference_count(2)], 1, None)]
#[case::excess(vec![Message::reference_count(u32::MAX)], 1, None)]
#[case::repeated(vec![Message::reference_count(2), Message::reference_count(2)], 1, Some("a copied object has repeated reference-count messages"))]
#[case::shared(vec![Message::reference_count(2).with_flags(MessageFlags::SHARED)], 1, Some("a copied object has an unsupported reference-count message"))]
#[case::version(vec![Message::new(MessageType::OBJECT_REFERENCE_COUNT, &[1, 2, 0, 0, 0])], 1, Some("a copied object has an unsupported reference-count message"))]
#[case::short(vec![Message::new(MessageType::OBJECT_REFERENCE_COUNT, &[0, 2])], 1, Some("a copied object has an unsupported reference-count message"))]
fn fresh_copy_counts_replace_only_supported_destination_counts(
    #[case] messages: Vec<Message>,
    #[case] count: u32,
    #[case] expected: Option<&str>,
) {
    let source = OhRegion::new(
        crate::file_writer::WIDTHS,
        v2::message_records(&messages, v2::HeaderFlags::default()),
        ObjectHeaderPrefix::PLAIN,
    );
    let original = source.clone();
    let mut destination = source.clone();
    let result = FreshCopyCount(NonZeroU32::new(count).unwrap()).apply(&mut destination);
    if let Some(expected) = expected {
        let error = result.unwrap_err();
        assert!(
            matches!(error, Error::EditUnsupported(reason) if reason == expected),
            "{error:?}"
        );
        assert_eq!(&destination[..], &original[..]);
    } else {
        result.unwrap();
        assert_eq!(destination.reference_count_mut().unwrap().count(), count);
    }
    assert_eq!(&source[..], &original[..]);
}

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two)]
#[case::two_four(OffsetWidth::Two, LengthWidth::Four)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight)]
fn dense_copy_inspects_datatypes_before_short_values(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(false, true)] reference: bool,
) {
    let datatype = if reference {
        datatype::object_reference(u32::from(offsets.get()))
    } else {
        datatype::f64_le()
    };
    let message = Attribute::new("value", &datatype, &test_util::dataspace::scalar(), &[]).build();
    let bytes = dense_attributes::one_message(
        &message,
        Widths::new(usize::from(offsets.get()), usize::from(lengths.get())),
    );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("short-attribute.h5");
    std::fs::write(&path, &bytes).unwrap();
    let source = BytesSource::new(bytes.as_slice());
    let located = superblock::read(&source).unwrap();
    let addresses = AddressContext::from_effective_base(located.superblock.base_address);
    let object = WriteEngine::read_object(
        &OpenedSource::for_test(&source, addresses),
        AccessMode::ReadOnly,
        ObjectId::new(
            addresses
                .resolve(located.superblock.root_group_address)
                .unwrap(),
        ),
        located.widths,
    );
    let Err(error) = object else {
        panic!("a short dense attribute must be rejected");
    };
    if reference {
        assert!(
            matches!(error, Error::Format(FormatError::ReferenceWidthUnsupported { offset_size }) if offset_size == offsets.get()),
            "{error:?}"
        );
    } else {
        assert!(
            matches!(error, Error::Format(FormatError::UnexpectedEof { expected, available }) if expected == message.len() + core::mem::size_of::<f64>() && available == message.len()),
            "{error:?}"
        );
    }
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    directory.close().unwrap();
}

fn group_node(targets: Vec<usize>) -> CopyNode {
    CopyNode::Group {
        region: OhRegion::default(),
        links: targets
            .into_iter()
            .map(|target| CopyLink {
                addr_off: 0,
                target: CopyNodeId(target),
            })
            .collect(),
        dense_attrs: DenseAttrSet::default(),
    }
}
