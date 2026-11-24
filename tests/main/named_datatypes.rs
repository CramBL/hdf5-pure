//! Committed datatype inspection and in-place creation restrictions.

use std::collections::HashMap;

use hdf5_pure::{
    AttrValue, CharacterSet, DatasetBuilder, Datatype, DatatypeByteOrder, Error, File, FileBuilder,
    FixedPointType, FormatError, Layout, MessageType, OffsetWidth, StringPadding,
    VariableLengthType,
};
use hdf5_pure_core::__private;
use hdf5_pure_format::__private::{AccessMode, MessageFlags, ObjectHeader};
use rstest::rstest;
use test_util::datatype;
use test_util::datatype::containers::{self, CompoundField, MemberOffset};
use test_util::datatype::leaf::{self, Version};
use test_util::identity::Fixture;
use test_util_hdf5::reader::ReadBackend;
use test_util_hdf5::{identity, storage};

#[rstest]
#[case::two_bytes(OffsetWidth::Two, 10)]
#[case::four_bytes(OffsetWidth::Four, 12)]
#[case::eight_bytes(OffsetWidth::Eight, 16)]
fn file_builder_writes_committed_compounds_at_the_file_offset_width(
    #[case] width: OffsetWidth,
    #[case] descriptor_size: u8,
) {
    let tag_offset = u64::from(descriptor_size);
    let datatype = string_and_tag(tag_offset);
    let expected = string_and_tag_bytes(descriptor_size, tag_offset);
    let mut builder = FileBuilder::new();
    builder.with_offset_width(width);
    builder.commit_datatype("record", datatype.clone());
    builder
        .create_dataset("value")
        .with_unallocated_storage(datatype.clone(), &[3])
        .with_committed_datatype("record");
    let bytes = builder.finish().unwrap();
    let file = File::from_bytes(bytes.clone()).unwrap();
    let superblock = file.superblock();
    assert_eq!(superblock.offset_size, width);
    let root = file.root();
    assert_eq!(root.named_datatype("record").unwrap(), datatype);
    assert_eq!(root.named_datatype_references("record").unwrap(), 2);
    let dataset = file.dataset("value").unwrap();
    assert_eq!(dataset.datatype().unwrap(), datatype);
    assert_eq!(dataset.shape().unwrap(), [3]);
    assert_eq!(dataset.element_size().unwrap(), 17);
    assert_eq!(
        dataset.layout().unwrap(),
        Layout::Contiguous {
            address: None,
            size: 51,
        }
    );

    let datatype_address = storage::root_object_address(&file, &bytes, "record");
    let mut shared = datatype::committed_reference(datatype_address.get());
    let unused_address_bytes = size_of::<u64>() - usize::from(width.get());
    shared.truncate(shared.len() - unused_address_bytes);
    for (name, flags, body) in [
        (
            "record",
            MessageFlags::CONSTANT | MessageFlags::FORBID_SHARING,
            expected,
        ),
        (
            "value",
            MessageFlags::CONSTANT | MessageFlags::SHARED,
            shared,
        ),
    ] {
        let address = storage::root_object_address(&file, &bytes, name);
        let header = ObjectHeader::parse(
            &bytes,
            AccessMode::ReadOnly,
            usize::try_from(address.get()).unwrap(),
            width.get(),
            superblock.length_size.get(),
        )
        .unwrap();
        let messages = header
            .messages
            .iter()
            .filter(|message| message.msg_type == MessageType::DATATYPE)
            .map(|message| (message.flags, message.data.as_slice()))
            .collect::<Vec<_>>();
        assert_eq!(messages, [(flags, body.as_slice())]);
    }
}

#[rstest]
#[case::different_layout(OffsetWidth::Two, 11, FormatError::CommittedDatatypeMismatch {
    path: "record".into(),
    user: "dataset \"value\"".into(),
})]
#[case::invalid_layout(OffsetWidth::Four, 10, FormatError::CompoundFieldOverlap {
    first: "v".into(),
    second: "tag".into(),
})]
fn file_builder_rejects_incompatible_committed_compound_uses_before_output(
    #[case] width: OffsetWidth,
    #[case] tag_offset: u64,
    #[case] expected: FormatError,
) {
    let mut builder = FileBuilder::new();
    builder.with_offset_width(width);
    builder.commit_datatype("record", string_and_tag(12));
    builder
        .create_dataset("value")
        .with_unallocated_storage(string_and_tag(tag_offset), &[3])
        .with_committed_datatype("record");
    let mut destination = b"previous bytes".to_vec();
    let error = builder.finish_to(&mut destination).unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, expected);
    assert_eq!(destination, b"previous bytes");
}

#[test]
fn file_builder_counts_committed_datatype_references_from_all_attribute_owners() {
    let datatype = Datatype::VariableLength(VariableLengthType::string(
        StringPadding::NullTerminate,
        CharacterSet::Ascii,
    ));
    let mut builder = FileBuilder::new();
    builder.with_offset_width(OffsetWidth::Two);
    builder.commit_datatype("string", datatype.clone());
    let value = AttrValue::VarLenAsciiString("shared".into());
    builder.set_attr_committed("label", value.clone(), "string");
    builder
        .create_group_with("group", |group| {
            group.set_attr_committed("label", value.clone(), "/string");
            Ok(())
        })
        .unwrap();
    builder
        .create_dataset("dataset")
        .with_u8_data(&[5])
        .set_attr_committed("label", value.clone(), "string");
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let root = file.root();
    let group = file.group("group").unwrap();
    let dataset = file.dataset("dataset").unwrap();
    assert_eq!(root.named_datatype_references("string").unwrap(), 4);
    let expected_datatypes = HashMap::from([("label".into(), datatype)]);
    let expected_attributes = HashMap::from([("label".into(), value)]);
    for (datatypes, attributes) in [
        (root.attr_datatypes().unwrap(), root.attrs().unwrap()),
        (group.attr_datatypes().unwrap(), group.attrs().unwrap()),
        (dataset.attr_datatypes().unwrap(), dataset.attrs().unwrap()),
    ] {
        assert_eq!(datatypes, expected_datatypes);
        assert_eq!(attributes, expected_attributes);
    }
}

#[rstest]
#[case::dataset(|builder: &mut DatasetBuilder| { builder.with_committed_datatype("integer_type"); })]
#[case::attribute(|builder: &mut DatasetBuilder| { builder.set_attr_committed("scale", AttrValue::I32(7), "integer_type"); })]
fn in_place_creation_rejects_committed_datatypes_without_changing_the_file(
    #[case] configure: fn(&mut DatasetBuilder),
) {
    let path = test_util::temp::temp_path("committed.h5");
    let mut builder = FileBuilder::new();
    builder.commit_datatype("integer_type", hdf5_pure::make_i32_type());
    builder.create_dataset("preserved").with_i32_data(&[11, 13]);
    builder.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        let error = file
            .root()
            .create_dataset("created", |builder| {
                builder.with_i32_data(&[1]);
                configure(builder);
            })
            .unwrap_err();
        let Error::EditUnsupported(reason) = error else {
            panic!("expected an unsupported committed datatype, got {error:?}");
        };
        assert_eq!(
            reason,
            "a dataset or attribute naming a committed (shared) datatype cannot be added in place; write the file with FileBuilder instead"
        );
        assert!(!file.has_staged_edits());
        file.commit().unwrap();
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("preserved")
            .unwrap()
            .read_i32()
            .unwrap(),
        [11, 13]
    );
}

#[rstest]
#[case::committed_datatype_v1(Fixture::CommittedDatatypeV1)]
#[case::hard_link_undo(Fixture::HardLinkUndo)]
fn independent_identity_files_have_complete_contents(
    #[case] fixture: Fixture,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let bytes = fixture.build();
    let file = backend.open_image(bytes.clone()).unwrap();
    identity::assert_contents(&file, fixture, DatatypeByteOrder::LittleEndian);
    identity::assert_storage(&file, &bytes, fixture);
    file.close().unwrap();
}

fn string_and_tag(tag_offset: u64) -> Datatype {
    Datatype::Compound {
        size: 17,
        members: vec![
            __private::compound_member(
                "v".into(),
                0,
                Datatype::VariableLength(VariableLengthType::string(
                    StringPadding::NullTerminate,
                    CharacterSet::Ascii,
                )),
            ),
            __private::compound_member(
                "tag".into(),
                tag_offset,
                Datatype::FixedPoint(FixedPointType::u8()),
            ),
        ],
    }
}

fn string_and_tag_bytes(descriptor_size: u8, tag_offset: u64) -> Vec<u8> {
    let byte = leaf::unsigned_byte(Version::One);
    let string = datatype::variable_length(
        Version::One.get(),
        NULL_TERMINATED_ASCII_FLAGS,
        u32::from(descriptor_size),
        &byte,
    );
    containers::compound(
        Version::Three.get(),
        [2, 0, 0],
        17,
        MemberOffset::One,
        &[
            CompoundField {
                name: b"v",
                offset: 0,
                datatype: &string,
            },
            CompoundField {
                name: b"tag",
                offset: tag_offset,
                datatype: &byte,
            },
        ],
    )
}

const NULL_TERMINATED_ASCII_FLAGS: [u8; 3] = [0x01, 0, 0];
