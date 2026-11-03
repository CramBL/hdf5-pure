use hdf5_pure_format::__private as format;
use rstest::rstest;
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture;

use super::super::AttrOp;
use super::super::EditedAttrs;
use crate::AttrValue;
use crate::FileBuilder;
use crate::StoredAddress;
use crate::access_mode::AccessMode;
use crate::address::StoredAddressExt;
use crate::attribute::AttributeMessage;
use crate::dataspace::Dataspace;
use crate::edit::WriteEngine;
use crate::edit::test_util as edit_test_util;
use crate::group::v2 as group_v2;
use crate::object_path::ObjectPath;

#[rstest]
#[case::compact_numeric(Initial::Compact, Edit::Numeric)]
#[case::dense_numeric(Initial::Dense, Edit::Numeric)]
#[case::compact_remove(Initial::Compact, Edit::Remove)]
#[case::dense_remove(Initial::Dense, Edit::Remove)]
#[case::compact_repeated(Initial::Compact, Edit::Repeated)]
#[case::dense_repeated(Initial::Dense, Edit::Repeated)]
#[case::compact_new_attribute(Initial::Compact, Edit::New)]
#[case::dense_new_attribute(Initial::Dense, Edit::New)]
#[case::compact_empty(Initial::Compact, Edit::Empty)]
#[case::dense_empty(Initial::Dense, Edit::Empty)]
#[case::compact_to_dense_changed(Initial::Compact, Edit::PromoteReplacing)]
#[case::compact_to_dense_unchanged(Initial::Compact, Edit::PromoteRetaining)]
#[case::dense_to_compact_changed(Initial::Dense, Edit::RevertReplacing)]
#[case::dense_to_compact_unchanged(Initial::Dense, Edit::RevertRetaining)]
fn attribute_edits_capture_only_original_touched_references(
    #[case] initial: Initial,
    #[case] edit: Edit,
) {
    let path = test_util::temp::temp_path("attribute-predecessors.h5");
    let mut builder = FileBuilder::new();
    builder.set_attr("label", AttrValue::VarLenAsciiString("before".into()));
    for index in 0..initial.extra_attributes() {
        builder.set_attr(&format!("a{index}"), AttrValue::I64(7));
    }
    builder.write(&path).unwrap();
    let collections = global_heap::collection_starts(&std::fs::read(&path).unwrap());
    assert_eq!(collections.len(), 1);
    let engine = edit_test_util::open_engine(&path);
    let root = engine.context.superblock().root_group_address.get();
    let region =
        WriteEngine::gather_oh_messages(engine.image(), root, engine.context.addresses()).unwrap();
    assert_eq!(
        super::super::region_uses_dense_attrs(&region).unwrap(),
        matches!(initial, Initial::Dense)
    );
    let operations = edit.operations(initial.extra_attributes());
    let operations: Vec<_> = operations.iter().collect();
    let result =
        super::super::plan_attr_ops(&engine.opened_source(), &region, &operations).unwrap();
    assert_eq!(
        result.replaces_dense_storage,
        matches!(initial, Initial::Dense)
    );
    let dense = matches!(result.attrs, EditedAttrs::Dense(_));
    assert_eq!(dense, edit.final_dense(initial));
    let predecessors: Vec<_> = result.predecessors.into_iter().collect();
    let expected = AttributeMessage {
        name: "label".into(),
        datatype: global_heap_fixture::vlen_string(),
        dataspace: Dataspace {
            space_type: format::DataspaceType::Scalar,
            rank: 0,
            dimensions: Vec::new(),
            max_dimensions: None,
        },
        raw_data: global_heap::reference(6, collections[0], 1, Widths::EIGHT),
        datatype_location: format::DatatypeLocation::Inline,
    };
    assert_eq!(
        predecessors,
        edit.replaces_label()
            .then_some(expected)
            .into_iter()
            .collect::<Vec<_>>()
    );
    drop(engine);
}

#[rstest]
#[case::compact(Initial::Compact)]
#[case::dense(Initial::Dense)]
fn attribute_predecessors_retain_resolved_committed_datatypes(#[case] initial: Initial) {
    let path = test_util::temp::temp_path("committed-attribute-predecessor.h5");
    let mut builder = FileBuilder::new();
    builder.commit_datatype("strings", global_heap_fixture::vlen_string());
    builder.set_attr_committed(
        "label",
        AttrValue::VarLenAsciiString("before".into()),
        "strings",
    );
    for index in 0..initial.extra_attributes() {
        builder.set_attr(&format!("a{index}"), AttrValue::I64(7));
    }
    builder.write(&path).unwrap();
    let collections = global_heap::collection_starts(&std::fs::read(&path).unwrap());
    assert_eq!(collections.len(), 1);
    let engine = edit_test_util::open_engine(&path);
    let definition = group_v2::resolve_path_any_from_source(
        &engine.opened_source(),
        AccessMode::ReadWrite,
        engine.context.superblock(),
        engine.context.widths(),
        &ObjectPath::parse("strings"),
    )
    .unwrap();
    let root = engine.context.superblock().root_group_address.get();
    let region =
        WriteEngine::gather_oh_messages(engine.image(), root, engine.context.addresses()).unwrap();
    assert_eq!(
        super::super::region_uses_dense_attrs(&region).unwrap(),
        matches!(initial, Initial::Dense)
    );
    let edited = super::super::plan_attr_ops(
        &engine.opened_source(),
        &region,
        &[&AttrOp::Set {
            name: "label".into(),
            value: AttrValue::I64(19),
        }],
    )
    .unwrap();
    let predecessors: Vec<_> = edited.predecessors.into_iter().collect();
    assert_eq!(
        predecessors,
        [AttributeMessage {
            name: "label".into(),
            datatype: global_heap_fixture::vlen_string(),
            dataspace: Dataspace {
                space_type: format::DataspaceType::Scalar,
                rank: 0,
                dimensions: Vec::new(),
                max_dimensions: None,
            },
            raw_data: global_heap::reference(6, collections[0], 1, Widths::EIGHT),
            datatype_location: format::DatatypeLocation::Committed(StoredAddress::new(definition)),
        }]
    );
    drop(engine);
}

#[derive(Clone, Copy, Debug)]
enum Initial {
    Compact,
    Dense,
}

impl Initial {
    fn extra_attributes(self) -> usize {
        match self {
            Self::Compact => 1,
            Self::Dense => 8,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Edit {
    Empty,
    New,
    Numeric,
    PromoteReplacing,
    PromoteRetaining,
    Remove,
    Repeated,
    RevertReplacing,
    RevertRetaining,
}

impl Edit {
    fn operations(self, extras: usize) -> Vec<AttrOp> {
        let set = |name: &str, value| AttrOp::Set {
            name: name.into(),
            value,
        };
        let remove = |name: &str| AttrOp::Remove { name: name.into() };
        match self {
            Self::Numeric => vec![set("label", AttrValue::I64(19))],
            Self::Remove => vec![remove("label")],
            Self::Repeated | Self::New => {
                let name = if matches!(self, Self::New) {
                    "new"
                } else {
                    "label"
                };
                vec![
                    set(name, AttrValue::VarLenAsciiString("staged".into())),
                    remove(name),
                    set(name, AttrValue::I64(19)),
                ]
            }
            Self::Empty => std::iter::once(remove("label"))
                .chain((0..extras).map(|index| remove(&format!("a{index}"))))
                .collect(),
            Self::PromoteReplacing | Self::PromoteRetaining => (0..8)
                .map(|index| set(&format!("new{index}"), AttrValue::I64(19)))
                .chain(
                    matches!(self, Self::PromoteReplacing)
                        .then(|| set("label", AttrValue::I64(19))),
                )
                .collect(),
            Self::RevertReplacing | Self::RevertRetaining => (0..6)
                .map(|index| remove(&format!("a{index}")))
                .chain(
                    matches!(self, Self::RevertReplacing).then(|| set("label", AttrValue::I64(19))),
                )
                .collect(),
        }
    }

    fn final_dense(self, initial: Initial) -> bool {
        match self {
            Self::Empty | Self::RevertReplacing | Self::RevertRetaining => false,
            Self::PromoteReplacing | Self::PromoteRetaining => true,
            _ => matches!(initial, Initial::Dense),
        }
    }

    fn replaces_label(self) -> bool {
        !matches!(
            self,
            Self::New | Self::PromoteRetaining | Self::RevertRetaining
        )
    }
}
