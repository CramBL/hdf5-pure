use hdf5_pure_format::__private::BTreeV2Header;
use rstest::rstest;

use super::*;
use crate::File;
use crate::access_mode::AccessMode;
use crate::group_v2;
use crate::link_info::LinkInfoMessage;
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPath;
use crate::source;
use crate::superblock::Superblock;
use crate::type_builders;

#[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
mod allocation;

#[rstest]
#[case::empty("", None)]
#[case::nul("prefix\0suffix", None)]
#[case::too_long(&"x".repeat(usize::from(u16::MAX)), Some(usize::from(u16::MAX) + 1))]
fn rejects_attribute_names_before_sink_output(
    #[case] name: &str,
    #[case] encoded_size: Option<usize>,
    #[values(
        AttributeForm::Committed,
        AttributeForm::CommittedVarLen,
        AttributeForm::CommittedVarLenVerbatim,
        AttributeForm::CommittedVerbatim,
        AttributeForm::Value,
        AttributeForm::VarLenValue,
        AttributeForm::VarLenVerbatim,
        AttributeForm::Verbatim
    )]
    form: AttributeForm,
    #[values(
        AttributeTarget::Dataset,
        AttributeTarget::Group,
        AttributeTarget::Root
    )]
    target: AttributeTarget,
    #[values(false, true)] dense: bool,
) {
    let mut writer = attribute_writer(false);
    let mut group = writer.create_group("group");
    {
        let mut owner = match target {
            AttributeTarget::Dataset => {
                AttributeOwner::Dataset(writer.create_dataset("data").with_i32_data(&[1]))
            }
            AttributeTarget::Group => AttributeOwner::Group(&mut group),
            AttributeTarget::Root => AttributeOwner::Root(&mut writer),
        };
        owner.insert("preserved", AttributeForm::Value, 1);
        if dense {
            for index in 0..10 {
                owner.insert(&format!("filler{index}"), AttributeForm::Value, index);
            }
        }
        owner.insert(name, form, 42);
    }
    writer.add_group(group.finish());
    let mut bytes = b"previous bytes".to_vec();
    let error = writer.finish_to_sink(&mut bytes).unwrap_err().into_format();
    let expected = match encoded_size {
        Some(size) => FormatError::AttributeFieldTooLong {
            name: name.into(),
            field: "name",
            size,
            limit: usize::from(u16::MAX),
        },
        None => FormatError::InvalidAttributeName { name: name.into() },
    };
    assert_eq!(error, expected);
    assert_eq!(bytes, b"previous bytes");
}

#[rstest]
#[case::ascii("scale")]
#[case::utf8("échelle")]
fn attribute_names_retain_encoding_across_staging_forms(
    #[case] name: &str,
    #[values(
        AttributeForm::Committed,
        AttributeForm::CommittedVarLen,
        AttributeForm::CommittedVarLenVerbatim,
        AttributeForm::CommittedVerbatim,
        AttributeForm::Value,
        AttributeForm::VarLenValue,
        AttributeForm::VarLenVerbatim,
        AttributeForm::Verbatim
    )]
    form: AttributeForm,
    #[values(false, true)] dense: bool,
    #[values(false, true)] paged: bool,
) {
    let mut writer = attribute_writer(paged);
    stage_encoded_name_attributes(AttributeOwner::Root(&mut writer), name, form, dense);
    stage_encoded_name_attributes(
        AttributeOwner::Dataset(writer.create_dataset("data").with_i32_data(&[1])),
        name,
        form,
        dense,
    );
    let mut group = writer.create_group("group");
    stage_encoded_name_attributes(AttributeOwner::Group(&mut group), name, form, dense);
    writer.add_group(group.finish());
    let bytes = writer.finish().unwrap();
    assert_eq!(test_util::fractal_heap::has_fractal_heap(&bytes), dense);
    let file = File::from_bytes(bytes.clone()).unwrap();
    for messages in [
        file.root().attr_messages().unwrap(),
        file.group("group").unwrap().attr_messages().unwrap(),
        file.dataset("data").unwrap().attr_messages().unwrap(),
    ] {
        assert_eq!(messages.len(), if dense { 11 } else { 1 });
        let message = messages
            .iter()
            .find(|message| message.name == name)
            .unwrap();
        let encoded = if dense {
            message.serialize_v3(OFFSET_WIDTH, LENGTH_SIZE).unwrap()
        } else {
            message.serialize(OFFSET_WIDTH, LENGTH_SIZE).unwrap()
        };
        assert_eq!(
            encoded[0],
            if dense || !name.is_ascii() {
                ATTRIBUTE_MESSAGE_VERSION_THREE
            } else {
                ATTRIBUTE_MESSAGE_VERSION_TWO
            }
        );
        if encoded[0] == ATTRIBUTE_MESSAGE_VERSION_THREE {
            assert_eq!(
                encoded[ATTRIBUTE_NAME_CHARACTER_SET_OFFSET],
                if name.is_ascii() {
                    ASCII_NAME_CHARACTER_SET
                } else {
                    UTF8_NAME_CHARACTER_SET
                }
            );
        }
        assert!(bytes.windows(encoded.len()).any(|window| window == encoded));
    }
    let expected = match form {
        AttributeForm::CommittedVarLen
        | AttributeForm::CommittedVarLenVerbatim
        | AttributeForm::VarLenValue
        | AttributeForm::VarLenVerbatim => AttrValue::VarLenString("value-42".into()),
        _ => AttrValue::I32(42),
    };
    for attributes in [
        file.root().attrs().unwrap(),
        file.group("group").unwrap().attrs().unwrap(),
        file.dataset("data").unwrap().attrs().unwrap(),
    ] {
        assert_eq!(attributes.get(name), Some(&expected));
        if dense {
            assert_eq!(attributes.get("filler0"), Some(&AttrValue::I32(0)));
            assert_eq!(attributes.get("filler9"), Some(&AttrValue::I32(9)));
        }
    }
}

fn stage_encoded_name_attributes(
    mut owner: AttributeOwner<'_>,
    name: &str,
    form: AttributeForm,
    dense: bool,
) {
    owner.insert(name, AttributeForm::Value, 7);
    if dense {
        for index in 0..10 {
            owner.insert(&format!("filler{index}"), AttributeForm::Value, index);
        }
    }
    owner.insert(name, form, 42);
}

#[rstest]
#[case::ascii_at_limit("xx", 0, false)]
#[case::utf8_at_limit("é", 1, false)]
#[case::utf8_above_limit("é", 0, true)]
fn attribute_name_encoding_counts_toward_compact_storage_limit(
    #[case] name: &str,
    #[case] trim: usize,
    #[case] dense: bool,
    #[values(false, true)] paged: bool,
) {
    let probe =
        type_builders::build_attr_message("xx", &AttrValue::AsciiString("y".into())).unwrap();
    let overhead = probe.serialize(OFFSET_WIDTH, LENGTH_SIZE).unwrap().len() - 1;
    let value = AttrValue::AsciiString("y".repeat(OBJECT_HEADER_MESSAGE_MAX - overhead - trim));
    let message = type_builders::build_attr_message(name, &value).unwrap();
    assert_eq!(
        message.serialize(OFFSET_WIDTH, LENGTH_SIZE).unwrap().len(),
        OBJECT_HEADER_MESSAGE_MAX + usize::from(!name.is_ascii()) - trim
    );
    assert_eq!(needs_dense_attrs(&[message]).unwrap(), dense);
    let mut writer = attribute_writer(paged);
    writer.set_root_attr(name, value.clone());
    let bytes = writer.finish().unwrap();
    assert_eq!(test_util::fractal_heap::has_fractal_heap(&bytes), dense);
    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(file.root().attrs().unwrap().get(name), Some(&value));
}

#[derive(Clone, Copy)]
enum AttributeTarget {
    Dataset,
    Group,
    Root,
}

#[test]
fn repeated_root_attribute_has_one_serialized_message() {
    let mut writer = FileWriter::new();
    writer.set_root_attr("answer", AttrValue::I32(7));
    writer.set_root_attr("answer", AttrValue::I32(42));
    let file = File::from_bytes(writer.finish().unwrap()).unwrap();
    let messages = file.root().attr_messages().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].name, "answer");
    assert_eq!(messages[0].raw_data, 42_i32.to_le_bytes());
}

#[rstest]
fn repeated_attributes_replace_complete_specs(
    #[values(
        AttributeForm::Committed,
        AttributeForm::CommittedVarLen,
        AttributeForm::CommittedVarLenVerbatim,
        AttributeForm::CommittedVerbatim,
        AttributeForm::Value,
        AttributeForm::VarLenValue,
        AttributeForm::VarLenVerbatim,
        AttributeForm::Verbatim
    )]
    before: AttributeForm,
    #[values(
        AttributeForm::Committed,
        AttributeForm::CommittedVarLen,
        AttributeForm::CommittedVarLenVerbatim,
        AttributeForm::CommittedVerbatim,
        AttributeForm::Value,
        AttributeForm::VarLenValue,
        AttributeForm::VarLenVerbatim,
        AttributeForm::Verbatim
    )]
    after: AttributeForm,
    #[values(false, true)] dense: bool,
    #[values(false, true)] paged: bool,
) {
    let bytes = replacement_writer(Some(before), after, dense, paged)
        .finish()
        .unwrap();
    assert_eq!(test_util::fractal_heap::has_fractal_heap(&bytes), dense);
    assert_eq!(
        bytes,
        replacement_writer(None, after, dense, paged)
            .finish()
            .unwrap()
    );
    let file = File::from_bytes(bytes).unwrap();
    let expected = if dense { 13 } else { 3 };
    assert_eq!(
        [
            file.root().attr_messages().unwrap().len(),
            file.group("group").unwrap().attr_messages().unwrap().len(),
            file.dataset("data").unwrap().attr_messages().unwrap().len(),
        ],
        [expected; 3]
    );
}

fn replacement_writer(
    before: Option<AttributeForm>,
    after: AttributeForm,
    dense: bool,
    paged: bool,
) -> FileWriter {
    let mut writer = attribute_writer(paged);
    replacement_attrs(AttributeOwner::Root(&mut writer), before, after, dense);
    replacement_attrs(
        AttributeOwner::Dataset(writer.create_dataset("data").with_i32_data(&[1])),
        before,
        after,
        dense,
    );
    let mut group = writer.create_group("group");
    replacement_attrs(AttributeOwner::Group(&mut group), before, after, dense);
    writer.add_group(group.finish());
    writer
}

fn attribute_writer(paged: bool) -> FileWriter {
    let mut writer = FileWriter::new();
    if paged {
        writer.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Page));
    }
    writer.commit_datatype("integer_type", type_builders::make_i32_type());
    writer.commit_datatype(
        "string_type",
        type_builders::build_attr_message("type", &AttrValue::VarLenString("type".into()))
            .unwrap()
            .datatype,
    );
    writer
}

fn replacement_attrs(
    mut owner: AttributeOwner<'_>,
    before: Option<AttributeForm>,
    after: AttributeForm,
    dense: bool,
) {
    owner.insert("first", AttributeForm::Value, 1);
    owner.insert(
        "replace",
        before.unwrap_or(after),
        if before.is_some() { 7 } else { 42 },
    );
    owner.insert("last", AttributeForm::Value, 3);
    if dense {
        for index in 0..10 {
            owner.insert(&format!("filler{index}"), AttributeForm::Value, index);
        }
    }
    if before.is_some() {
        owner.insert("replace", after, 42);
    }
}

enum AttributeOwner<'a> {
    Dataset(&'a mut DatasetBuilder),
    Group(&'a mut GroupBuilder),
    Root(&'a mut FileWriter),
}

impl AttributeOwner<'_> {
    fn insert(&mut self, name: &str, form: AttributeForm, value: i32) {
        let value = match form {
            AttributeForm::CommittedVarLen
            | AttributeForm::CommittedVarLenVerbatim
            | AttributeForm::VarLenValue
            | AttributeForm::VarLenVerbatim => AttrValue::VarLenString(format!("value-{value}")),
            _ => AttrValue::I32(value),
        };
        match form {
            AttributeForm::Value | AttributeForm::VarLenValue => match self {
                Self::Dataset(owner) => {
                    owner.set_attr(name, value);
                }
                Self::Group(owner) => owner.set_attr(name, value),
                Self::Root(owner) => owner.set_root_attr(name, value),
            },
            AttributeForm::Committed | AttributeForm::CommittedVarLen => {
                let path = if matches!(form, AttributeForm::CommittedVarLen) {
                    "string_type"
                } else {
                    "integer_type"
                };
                match self {
                    Self::Dataset(owner) => {
                        owner.set_attr_committed(name, value, path);
                    }
                    Self::Group(owner) => owner.set_attr_committed(name, value, path),
                    Self::Root(owner) => owner.set_root_attr_committed(name, value, path),
                }
            }
            AttributeForm::CommittedVerbatim | AttributeForm::Verbatim => {
                let message = type_builders::build_attr_message(name, &value).unwrap();
                let path = matches!(form, AttributeForm::CommittedVerbatim)
                    .then(|| ObjectPathBuf::parse("integer_type"));
                match self {
                    Self::Dataset(owner) => {
                        owner.set_attr_verbatim(message, path);
                    }
                    Self::Group(owner) => owner.set_attr_verbatim(message, path),
                    Self::Root(owner) => owner.set_root_attr_verbatim(message, path),
                }
            }
            AttributeForm::CommittedVarLenVerbatim | AttributeForm::VarLenVerbatim => {
                let message = type_builders::build_attr_message(name, &value).unwrap();
                let strings = value.var_len_strings().unwrap().to_vec();
                let path = matches!(form, AttributeForm::CommittedVarLenVerbatim)
                    .then(|| ObjectPathBuf::parse("string_type"));
                match self {
                    Self::Dataset(owner) => {
                        owner
                            .set_attr_var_len_verbatim(message, strings, path)
                            .unwrap();
                    }
                    Self::Group(owner) => owner
                        .set_attr_var_len_verbatim(message, strings, path)
                        .unwrap(),
                    Self::Root(owner) => owner
                        .set_root_attr_var_len_verbatim(message, strings, path)
                        .unwrap(),
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum AttributeForm {
    Committed,
    CommittedVarLen,
    CommittedVarLenVerbatim,
    CommittedVerbatim,
    Value,
    VarLenValue,
    VarLenVerbatim,
    Verbatim,
}

#[rstest::rstest]
#[case::before_exact_fill(394, 0, false, Some(35))]
#[case::exact_fill(395, 0, false, Some(36))]
#[case::after_exact_fill(396, 0, false, Some(35))]
#[case::no_tail(512, 0, false, None)]
#[case::other_manager_fills_page(395, 117, false, None)]
#[case::existing_section(377, 0, true, Some(53))]
fn paged_metadata_capacity_keeps_known_sections(
    #[case] end: u64,
    #[case] other: u64,
    #[case] existing: bool,
    #[case] expected: Option<u64>,
) {
    let sections = if existing {
        vec![FreeSection {
            addr: StoredAddress::new(48),
            size: 17,
        }]
    } else {
        Vec::new()
    };
    assert_eq!(
        layout::paged_metadata_section_allocation(
            FileSpacePageSize::try_from(512).unwrap(),
            StoredAddress::new(end),
            [other],
            &sections
        )
        .unwrap(),
        expected
    );
}

#[rstest::rstest]
#[case::other_manager(u64::MAX, 1, u64::MAX, 1)]
#[case::header(u64::MAX - 10, 0, u64::MAX - 10, 82)]
#[case::section_list(u64::MAX - 90, 0, u64::MAX - 8, 35)]
fn paged_metadata_capacity_rejects_overflow(
    #[case] end: u64,
    #[case] other: u64,
    #[case] offset: u64,
    #[case] length: u64,
) {
    assert_eq!(
        layout::paged_metadata_section_allocation(
            FileSpacePageSize::try_from(512).unwrap(),
            StoredAddress::new(end),
            [other],
            &[]
        )
        .unwrap_err(),
        FormatError::OffsetOverflow { offset, length }
    );
}

#[test]
fn small_paged_collections_can_place_metadata_above_four_gibibytes() {
    let mut writer = FileWriter::new();
    writer.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_page_size(FileSpacePageSize::MAX),
    );
    writer.set_root_attr("first", AttrValue::VarLenString("first-value".into()));
    writer.set_root_attr("second", AttrValue::VarLenString("second-value".into()));
    writer.set_root_attr("third", AttrValue::VarLenString("third-value".into()));
    let mut sink = SparseSink {
        position: 0,
        chunks: Vec::new(),
    };
    writer.finish_to_sink(&mut sink).unwrap();
    let superblock = crate::superblock::read(sink.chunks[0].1.as_slice())
        .unwrap()
        .superblock;
    assert_eq!(superblock.root_group_address.get(), 1u64 << 32);
    assert_eq!(superblock.eof_address, sink.position);
    let collections: Vec<u64> = sink
        .chunks
        .iter()
        .filter(|(_, bytes)| {
            bytes.get(..GLOBAL_HEAP_COLLECTION_SIGNATURE.len())
                == Some(GLOBAL_HEAP_COLLECTION_SIGNATURE)
        })
        .map(|(position, _)| *position)
        .collect();
    assert_eq!(collections, [1u64 << 30, 2u64 << 30, 3u64 << 30]);
    let root = sink
        .chunks
        .iter()
        .find(|(position, _)| *position == 1u64 << 32)
        .unwrap();
    assert_eq!(
        root.1.get(..OBJECT_HEADER_V2_SIGNATURE.len()),
        Some(OBJECT_HEADER_V2_SIGNATURE)
    );
    assert!(
        sink.chunks
            .iter()
            .map(|(_, bytes)| bytes.len())
            .sum::<usize>()
            < 20_000
    );
}

#[rstest::rstest]
#[case::large(1_024, 0)]
#[case::whole(4_096, 4_096)]
#[case::small(8_192, 8_192)]
fn paged_global_heaps_use_pages_disjoint_from_object_headers(
    #[case] page_size: u64,
    #[case] userblock: u64,
) {
    let mut writer = FileWriter::new();
    writer.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_page_size(FileSpacePageSize::try_from(page_size).unwrap()),
    );
    writer.with_userblock(UserblockSize::try_from(userblock).unwrap());
    writer.set_root_attr("root_label", AttrValue::VarLenString("root-value".into()));
    writer
        .create_dataset("contiguous")
        .with_vlen_strings(&["first", "second"]);
    writer
        .create_dataset("chunked")
        .with_vlen_strings(&["third", "fourth"])
        .with_chunks(&[1])
        .with_deflate(DeflateLevel::Six);
    let bytes = writer.finish().unwrap();
    let heap_addresses: Vec<u64> = bytes
        .windows(GLOBAL_HEAP_COLLECTION_SIGNATURE.len())
        .enumerate()
        .filter(|(_, signature)| *signature == GLOBAL_HEAP_COLLECTION_SIGNATURE)
        .map(|(offset, _)| offset as u64 - userblock)
        .collect();
    assert_eq!(heap_addresses.len(), 3);
    let metadata_addresses: Vec<u64> = bytes
        .windows(OBJECT_HEADER_V2_SIGNATURE.len())
        .enumerate()
        .filter(|(_, signature)| *signature == OBJECT_HEADER_V2_SIGNATURE)
        .map(|(offset, _)| offset as u64 - userblock)
        .collect();
    assert_eq!(metadata_addresses.len(), 4);
    for address in heap_addresses {
        assert_eq!(address % page_size, 0);
        let index = (userblock + address + 8).to_usize().unwrap();
        let size = u64::from_le_bytes(bytes[index..index + 8].try_into().unwrap());
        let end = address + size;
        for metadata in &metadata_addresses {
            assert!(
                *metadata / page_size < address / page_size
                    || *metadata / page_size > (end - 1) / page_size
            );
        }
    }
}

struct SparseSink {
    position: u64,
    chunks: Vec<(u64, Vec<u8>)>,
}

impl ByteSink for SparseSink {
    type Error = FormatError;

    fn put(&mut self, bytes: &[u8]) -> Result<(), FormatError> {
        self.chunks.push((self.position, bytes.to_vec()));
        self.position += bytes.len() as u64;
        Ok(())
    }

    fn put_zeros(&mut self, len: usize) -> Result<(), FormatError> {
        self.position += len as u64;
        Ok(())
    }

    fn position(&self) -> u64 {
        self.position
    }
}

#[test]
fn a_committed_datatype_header_holds_only_its_type() {
    let bytes = build_committed_datatype_oh(&type_builders::make_i32_type(), 1).unwrap();
    let hdr =
        ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();

    let types: Vec<MessageType> = hdr.messages.iter().map(|m| m.msg_type).collect();
    assert_eq!(
        types,
        vec![MessageType::DATATYPE],
        "a singly referenced committed type carries its datatype and nothing else"
    );
    assert_eq!(
        hdf5_pure_format::__private::parse_datatype(&hdr.messages[0].data)
            .unwrap()
            .0,
        type_builders::make_i32_type()
    );
    assert_eq!(
        hdr.messages[0].flags,
        MessageFlags::CONSTANT | MessageFlags::FORBID_SHARING,
        "the type of a committed object never changes and must not be shared onward"
    );
}

#[test]
fn a_committed_datatype_header_records_a_count_above_one() {
    let bytes = build_committed_datatype_oh(&type_builders::make_i32_type(), 4).unwrap();
    let hdr =
        ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();

    let refcount = hdr
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::OBJECT_REFERENCE_COUNT)
        .expect("a count above one is stored");
    assert_eq!(
        refcount.data,
        vec![0, 4, 0, 0, 0],
        "version 0 followed by the 4-byte count"
    );
}

#[test]
fn the_reference_count_is_the_link_plus_every_user() {
    let mut w = FileWriter::new();
    w.commit_datatype("mytype", type_builders::make_i32_type());
    w.set_root_attr_committed("root_attr", AttrValue::I32(1), "mytype");
    let ds = w.create_dataset("typed");
    ds.with_i32_data(&[1, 2]);
    ds.with_committed_datatype("mytype");
    ds.set_attr_committed("shared_attr", AttrValue::I32(2), "mytype");
    let bytes = w.finish().unwrap();

    // 1 hard link + the dataset + two attributes.
    assert_eq!(committed_reference_count(&bytes, "mytype"), Some(4));
}

#[test]
fn an_unreferenced_committed_type_stores_no_count() {
    let mut w = FileWriter::new();
    w.commit_datatype("mytype", type_builders::make_i32_type());
    let bytes = w.finish().unwrap();

    assert_eq!(committed_reference_count(&bytes, "mytype"), None);
}

/// Reads a committed datatype's reference count, returning `None` for an omitted message.
fn committed_reference_count(bytes: &[u8], path: &str) -> Option<u32> {
    let sb = crate::superblock::read(bytes).unwrap().superblock;
    let addr = group_v2::resolve_path_any(
        bytes,
        AccessMode::ReadOnly,
        &sb,
        WIDTHS,
        AddressContext::from_effective_base(sb.base_address),
        &ObjectPath::parse(path),
    )
    .unwrap();
    let hdr = ObjectHeader::parse_with_widths(
        bytes,
        AccessMode::ReadOnly,
        addr as usize,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    )
    .unwrap();
    let msg = hdr
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::OBJECT_REFERENCE_COUNT)?;
    Some(u32::from_le_bytes(msg.data[1..5].try_into().unwrap()))
}

// Check the shared-message flag as well as the referenced datatype address.
#[test]
fn a_dataset_naming_a_committed_type_stores_a_reference_to_it() {
    let mut w = FileWriter::new();
    w.commit_datatype("mytype", type_builders::make_i32_type());
    let ds = w.create_dataset("typed");
    ds.with_i32_data(&[1, 2]);
    ds.with_committed_datatype("mytype");
    let bytes = w.finish().unwrap();

    let sb = crate::superblock::read(bytes.as_slice())
        .unwrap()
        .superblock;
    let type_addr = group_v2::resolve_path_any(
        &bytes,
        AccessMode::ReadOnly,
        &sb,
        WIDTHS,
        AddressContext::from_effective_base(sb.base_address),
        &ObjectPath::parse("mytype"),
    )
    .unwrap();
    let ds_addr = group_v2::resolve_path_any(
        &bytes,
        AccessMode::ReadOnly,
        &sb,
        WIDTHS,
        AddressContext::from_effective_base(sb.base_address),
        &ObjectPath::parse("typed"),
    )
    .unwrap();
    let hdr = ObjectHeader::parse_with_widths(
        &bytes,
        AccessMode::ReadOnly,
        ds_addr as usize,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    )
    .unwrap();
    let msg = hdr
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::DATATYPE)
        .expect("a dataset always has a datatype message");

    assert!(
        msg.flags.is_shared(),
        "the record must say its body is a reference, or the reference decodes as a type"
    );
    assert_eq!(
        msg.data,
        crate::shared_message::encode_committed_ref(StoredAddress::new(type_addr), OFFSET_WIDTH),
        "the reference must name the object the link resolves to"
    );
}

fn parse_file(bytes: &[u8]) -> (Superblock, ObjectHeader) {
    let sb = crate::superblock::read(bytes).unwrap().superblock;
    let oh = ObjectHeader::parse_with_widths(
        bytes,
        AccessMode::ReadOnly,
        AddressContext::from_effective_base(sb.base_address)
            .resolve(sb.root_group_address)
            .unwrap()
            .get()
            .to_usize()
            .unwrap(),
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    )
    .unwrap();
    (sb, oh)
}

fn read_dataset_f64(bytes: &[u8], path: &str) -> Vec<f64> {
    let sb = crate::superblock::read(bytes).unwrap().superblock;
    let addr = group_v2::resolve_path_any(
        bytes,
        AccessMode::ReadOnly,
        &sb,
        WIDTHS,
        AddressContext::from_effective_base(sb.base_address),
        &ObjectPath::parse(path),
    )
    .unwrap();
    let hdr = ObjectHeader::parse_with_widths(
        bytes,
        AccessMode::ReadOnly,
        addr as usize,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    )
    .unwrap();
    let dt_data = &hdr
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::DATATYPE)
        .unwrap()
        .data;
    let ds_data = &hdr
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::DATASPACE)
        .unwrap()
        .data;
    let dl_data = &hdr
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::DATA_LAYOUT)
        .unwrap()
        .data;
    let (dt, _) = hdf5_pure_format::__private::parse_datatype(dt_data).unwrap();
    let ds = Dataspace::parse_with_length_width(ds_data, sb.length_size).unwrap();
    let dl = crate::data_layout::DataLayout::parse_with_widths(
        dl_data,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    )
    .unwrap();
    let raw = crate::data_read::read_raw_data(bytes, &dl, &ds, &dt).unwrap();
    crate::data_read::read_as_f64(&raw, &dt).unwrap()
}

#[test]
fn empty_file_root_group_only() {
    let fw = FileWriter::new();
    let bytes = fw.finish().unwrap();
    let (sb, oh) = parse_file(&bytes);
    assert_eq!(sb.version(), SuperblockVersion::Three);
    assert_eq!(oh.version, 2);
}

#[test]
fn file_with_dataset_attrs() {
    let mut fw = FileWriter::new();
    fw.create_dataset("data")
        .with_f64_data(&[1.0, 2.0])
        .set_attr("scale", AttrValue::F64(0.5));
    let bytes = fw.finish().unwrap();
    assert_eq!(read_dataset_f64(&bytes, "data"), vec![1.0, 2.0]);
    let sb = crate::superblock::read(bytes.as_slice())
        .unwrap()
        .superblock;
    let addr = group_v2::resolve_path_any(
        &bytes,
        AccessMode::ReadOnly,
        &sb,
        WIDTHS,
        AddressContext::from_effective_base(sb.base_address),
        &ObjectPath::parse("data"),
    )
    .unwrap();
    let hdr = ObjectHeader::parse_with_widths(
        &bytes,
        AccessMode::ReadOnly,
        addr as usize,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    )
    .unwrap();
    let attrs = crate::attribute::extract_attributes(&hdr, LENGTH_WIDTH).unwrap();
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name, "scale");
}

#[test]
fn file_with_group_and_dataset() {
    let mut fw = FileWriter::new();
    let mut gb = fw.create_group("grp");
    gb.create_dataset("vals").with_f64_data(&[10.0, 20.0]);
    fw.add_group(gb.finish());
    let bytes = fw.finish().unwrap();
    assert_eq!(read_dataset_f64(&bytes, "grp/vals"), vec![10.0, 20.0]);
}

#[test]
fn root_group_carries_no_timestamps() {
    let fw = FileWriter::new();
    let bytes = fw.finish().unwrap();
    let (_, oh) = parse_file(&bytes);
    assert_eq!(oh.flags & 0x20, 0, "times-stored flag must be clear");
    assert!(oh.modification_time.is_none());
    assert!(oh.access_time.is_none());
    assert!(oh.change_time.is_none());
    assert!(oh.birth_time.is_none());
}

#[test]
fn sub_group_carries_no_timestamps() {
    let mut fw = FileWriter::new();
    let mut gb = fw.create_group("grp");
    gb.create_dataset("vals").with_f64_data(&[1.0]);
    fw.add_group(gb.finish());
    let bytes = fw.finish().unwrap();
    let sb = crate::superblock::read(bytes.as_slice())
        .unwrap()
        .superblock;
    let addr = group_v2::resolve_path_any(
        &bytes,
        AccessMode::ReadOnly,
        &sb,
        WIDTHS,
        AddressContext::from_effective_base(sb.base_address),
        &ObjectPath::parse("grp"),
    )
    .unwrap();
    let hdr = ObjectHeader::parse_with_widths(
        &bytes,
        AccessMode::ReadOnly,
        addr as usize,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    )
    .unwrap();
    assert_eq!(hdr.flags & 0x20, 0, "times-stored flag must be clear");
    assert!(hdr.modification_time.is_none());
}

#[rstest]
#[case::empty(0)]
#[case::below_threshold(MAX_COMPACT_LINKS - 1)]
#[case::at_threshold(MAX_COMPACT_LINKS)]
#[case::above_threshold(MAX_COMPACT_LINKS + 1)]
#[case::multiple_blocks(100)]
#[case::multiple_levels(2_000)]
fn group_link_representation_matches_child_count(
    #[case] count: usize,
    #[values(false, true)] nested: bool,
    #[values(
        None,
        Some(FileSpaceStrategy::None),
        Some(FileSpaceStrategy::Aggr),
        Some(FileSpaceStrategy::FsmAggr),
        Some(FileSpaceStrategy::Page)
    )]
    strategy: Option<FileSpaceStrategy>,
    #[values(0, 4_096)] userblock: u64,
) {
    let bytes = link_writer(count, nested, strategy, userblock)
        .finish()
        .unwrap();
    let sb = crate::superblock::read(bytes.as_slice())
        .unwrap()
        .superblock;
    assert_eq!(sb.eof_address, bytes.len().to_u64());
    let path = if nested { "parent/group" } else { "/" };
    let address = group_v2::resolve_path_any(
        &bytes,
        AccessMode::ReadOnly,
        &sb,
        WIDTHS,
        AddressContext::from_effective_base(sb.base_address),
        &ObjectPath::parse(path),
    )
    .unwrap();
    let header = ObjectHeader::parse_with_widths(
        &bytes,
        AccessMode::ReadOnly,
        address.to_usize().unwrap(),
        WIDTHS,
    )
    .unwrap();
    let info = LinkInfoMessage::parse_with_offset_width(
        &header
            .messages
            .iter()
            .find(|message| message.msg_type == MessageType::LINK_INFO)
            .unwrap()
            .data,
        OFFSET_WIDTH,
    )
    .unwrap();
    let dense = count > MAX_COMPACT_LINKS;
    assert_eq!(info.fractal_heap_address.is_some(), dense);
    assert_eq!(info.btree_name_index_address.is_some(), dense);
    assert_eq!(info.max_creation_order, None);
    assert_eq!(info.btree_creation_order_address, None);
    assert_eq!(
        header
            .messages
            .iter()
            .filter(|message| message.msg_type == MessageType::LINK)
            .count(),
        if dense { 0 } else { count }
    );
    assert_eq!(
        header
            .messages
            .iter()
            .find(|message| message.msg_type == MessageType::GROUP_INFO)
            .unwrap()
            .data,
        DEFAULT_GROUP_INFO
    );
    if count >= 100 {
        let heap = FractalHeapHeader::parse(
            &bytes,
            (info.fractal_heap_address.unwrap().get() + userblock)
                .to_usize()
                .unwrap(),
            OFFSET_SIZE,
            LENGTH_SIZE,
        )
        .unwrap();
        assert!(heap.current_rows_in_root_indirect_block > 0);
        assert_eq!(heap.managed_objects_count, count.to_u64());
    }
    if count == 2_000 {
        let index = BTreeV2Header::parse_with_widths(
            &bytes,
            (info.btree_name_index_address.unwrap().get() + userblock)
                .to_usize()
                .unwrap(),
            WIDTHS,
        )
        .unwrap();
        assert!(index.depth >= 2, "{index:?}");
    }
    let file = File::from_bytes(bytes).unwrap();
    for index in 0..count {
        let child = format!("{path}/data_{index:04}_λ");
        assert_eq!(
            file.dataset(&child).unwrap().read_i32().unwrap(),
            [i32::try_from(index).unwrap(), 17]
        );
    }
}

#[rstest]
#[case::root(false)]
#[case::nested(true)]
fn dense_link_addresses_exceed_four_gibibytes(#[case] nested: bool) {
    let mut writer = link_writer(9, nested, Some(FileSpaceStrategy::Page), 0);
    writer.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page).with_page_size(FileSpacePageSize::MAX),
    );
    for name in ["first", "second", "third"] {
        writer.set_root_attr(name, AttrValue::VarLenString(name.into()));
    }
    let mut sink = SparseSink {
        position: 0,
        chunks: Vec::new(),
    };
    writer.finish_to_sink(&mut sink).unwrap();
    let superblock = crate::superblock::read(sink.chunks[0].1.as_slice())
        .unwrap()
        .superblock;
    assert_eq!(superblock.root_group_address.get(), 1u64 << 32);
    assert_eq!(superblock.eof_address, sink.position);
    let (_, bytes) = sink
        .chunks
        .iter()
        .find(|(_, bytes)| bytes.starts_with(FRACTAL_HEAP_SIGNATURE))
        .unwrap();
    let heap = FractalHeapHeader::parse_bytes(bytes, OFFSET_WIDTH, LENGTH_WIDTH).unwrap();
    assert!(heap.root_block_address.get() > u64::from(u32::MAX));
    assert_eq!(heap.managed_objects_count, 9);
    assert!(
        sink.chunks
            .iter()
            .filter(|(position, _)| *position >= 1u64 << 32)
            .all(|(position, bytes)| position + bytes.len().to_u64() <= sink.position)
    );
}

fn link_writer(
    count: usize,
    nested: bool,
    strategy: Option<FileSpaceStrategy>,
    userblock: u64,
) -> FileWriter {
    let mut writer = FileWriter::new();
    if let Some(strategy) = strategy {
        writer.with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(
            matches!(
                strategy,
                FileSpaceStrategy::FsmAggr | FileSpaceStrategy::Page
            ),
        ));
    }
    if userblock != 0 {
        writer.with_userblock(UserblockSize::try_from(userblock).unwrap());
    }
    let mut group = writer.create_group("group");
    for index in 0..count {
        let link_name = format!("data_{index:04}_λ");
        let data = [i32::try_from(index).unwrap(), 17];
        if nested {
            group.create_dataset(&link_name).with_i32_data(&data);
        } else {
            writer.create_dataset(&link_name).with_i32_data(&data);
        }
    }
    if nested {
        let mut parent = writer.create_group("parent");
        parent.add_group(group.finish());
        writer.add_group(parent.finish());
    }
    writer
}

#[test]
fn file_with_root_attr() {
    let mut fw = FileWriter::new();
    fw.set_root_attr("version", AttrValue::I64(42));
    let bytes = fw.finish().unwrap();
    let (_sb, oh) = parse_file(&bytes);
    let attrs = crate::attribute::extract_attributes(&oh, LENGTH_WIDTH).unwrap();
    assert_eq!(attrs[0].name, "version");
}

#[test]
fn dense_attrs_self_roundtrip() {
    let mut fw = FileWriter::new();
    let ds = fw.create_dataset("data");
    ds.with_f64_data(&[1.0, 2.0, 3.0]);
    for i in 0..20 {
        ds.set_attr(&format!("attr_{i:03}"), AttrValue::F64(i as f64 * 1.5));
    }
    let bytes = fw.finish().unwrap();
    let sb = crate::superblock::read(bytes.as_slice())
        .unwrap()
        .superblock;
    let addr = group_v2::resolve_path_any(
        &bytes,
        AccessMode::ReadOnly,
        &sb,
        WIDTHS,
        AddressContext::from_effective_base(sb.base_address),
        &ObjectPath::parse("data"),
    )
    .unwrap();
    let hdr = ObjectHeader::parse_with_widths(
        &bytes,
        AccessMode::ReadOnly,
        addr as usize,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    )
    .unwrap();
    let attrs = crate::attribute::extract_attributes_full(
        source::zero_base_bytes(&bytes),
        AccessMode::ReadOnly,
        &hdr,
        WIDTHS,
        None,
    )
    .unwrap();
    assert_eq!(attrs.len(), 20);
    for i in 0..20 {
        let attr = attrs
            .iter()
            .find(|a| a.name == format!("attr_{i:03}"))
            .unwrap();
        let v = crate::data_read::read_as_f64(&attr.raw_data, &attr.datatype).unwrap();
        assert!((v[0] - i as f64 * 1.5).abs() < 1e-10);
    }
    assert_eq!(read_dataset_f64(&bytes, "data"), vec![1.0, 2.0, 3.0]);
}

#[test]
fn dense_attrs_root_group_self_roundtrip() {
    let mut fw = FileWriter::new();
    fw.create_dataset("dummy").with_f64_data(&[0.0]);
    for i in 0..15 {
        fw.set_root_attr(&format!("root_{i:02}"), AttrValue::F64(i as f64 * 2.0));
    }
    let bytes = fw.finish().unwrap();
    let sb = crate::superblock::read(bytes.as_slice())
        .unwrap()
        .superblock;
    let oh = ObjectHeader::parse_with_widths(
        &bytes,
        AccessMode::ReadOnly,
        AddressContext::from_effective_base(sb.base_address)
            .resolve(sb.root_group_address)
            .unwrap()
            .get()
            .to_usize()
            .unwrap(),
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    )
    .unwrap();
    let attrs = crate::attribute::extract_attributes_full(
        source::zero_base_bytes(&bytes),
        AccessMode::ReadOnly,
        &oh,
        WIDTHS,
        None,
    )
    .unwrap();
    assert_eq!(attrs.len(), 15);
}

#[test]
fn inline_attrs_below_threshold() {
    let mut fw = FileWriter::new();
    let ds = fw.create_dataset("data");
    ds.with_f64_data(&[1.0]);
    for i in 0..5 {
        ds.set_attr(&format!("a{i}"), AttrValue::F64(i as f64));
    }
    let bytes = fw.finish().unwrap();
    let sb = crate::superblock::read(bytes.as_slice())
        .unwrap()
        .superblock;
    let addr = group_v2::resolve_path_any(
        &bytes,
        AccessMode::ReadOnly,
        &sb,
        WIDTHS,
        AddressContext::from_effective_base(sb.base_address),
        &ObjectPath::parse("data"),
    )
    .unwrap();
    let hdr = ObjectHeader::parse_with_widths(
        &bytes,
        AccessMode::ReadOnly,
        addr as usize,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    )
    .unwrap();
    // Compact attributes have undefined heap and index addresses.
    let info = hdr
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::ATTRIBUTE_INFO)
        .map(|m| {
            crate::attribute_info::AttributeInfoMessage::parse_with_offset_width(
                &m.data,
                sb.offset_size,
            )
            .unwrap()
        })
        .expect("a compact attribute set still carries an Attribute Info message");
    assert_eq!(info.fractal_heap_address, None);
    assert_eq!(info.btree_name_index_address, None);
    let attrs = crate::attribute::extract_attributes(&hdr, LENGTH_WIDTH).unwrap();
    assert_eq!(attrs.len(), 5);
}

/// Builds an ASCII string attribute with exactly `size` serialized version 3 bytes.
fn dense_attr_of_size(name: &str, size: usize) -> AttributeMessage {
    // An empty fixed string includes padding, so measure overhead with a one-byte value.
    let probe =
        type_builders::build_attr_message(name, &AttrValue::AsciiString("y".to_string())).unwrap();
    let overhead = probe.serialize_v3(OFFSET_WIDTH, LENGTH_SIZE).unwrap().len() - 1;
    let attr = type_builders::build_attr_message(
        name,
        &AttrValue::AsciiString("y".repeat(size - overhead)),
    )
    .unwrap();
    assert_eq!(
        attr.serialize_v3(OFFSET_WIDTH, LENGTH_SIZE).unwrap().len(),
        size
    );
    attr
}

#[test]
fn dense_attribute_preparation_accepts_managed_and_huge_messages() {
    let many: Vec<AttributeMessage> = (0..40)
        .map(|i| dense_attr_of_size(&format!("a{i}"), 60_000))
        .collect();
    let total: usize = many
        .iter()
        .map(|a| a.serialize_v3(OFFSET_WIDTH, LENGTH_SIZE).unwrap().len())
        .sum();
    assert!(total > 2_000_000, "expected a multi-megabyte set");
    PreparedDenseAttrs::new(&many, DenseAttrCreationOrder::Untracked, []).unwrap();

    let at_limit = vec![dense_attr_of_size("edge", DENSE_ATTR_MAX_MANAGED_OBJECT)];
    PreparedDenseAttrs::new(&at_limit, DenseAttrCreationOrder::Untracked, []).unwrap();

    let past = vec![dense_attr_of_size(
        "edge",
        DENSE_ATTR_MAX_MANAGED_OBJECT + 1,
    )];
    PreparedDenseAttrs::new(&past, DenseAttrCreationOrder::Untracked, []).unwrap();
    assert_eq!(
        huge_object_count(
            &dense_attribute_write::build_dense_attrs(
                &past,
                DenseAttrCreationOrder::Untracked,
                StoredAddress::new(0)
            )
            .unwrap()
            .blob
        ),
        1
    );
    assert_eq!(
        huge_object_count(
            &dense_attribute_write::build_dense_attrs(
                &at_limit,
                DenseAttrCreationOrder::Untracked,
                StoredAddress::new(0)
            )
            .unwrap()
            .blob
        ),
        0
    );
}

// Sweep consecutive counts and managed/huge boundaries at multiple heap addresses.
#[test]
fn dense_attr_plan_length_matches_what_it_builds() {
    let mut shapes: Vec<(String, Vec<AttributeMessage>)> = Vec::new();

    for n in 0..=64usize {
        shapes.push((
            format!("{n} x 512B"),
            (0..n)
                .map(|i| dense_attr_of_size(&format!("a{i}"), 512))
                .collect(),
        ));
    }
    for n in [100usize, 200] {
        shapes.push((
            format!("{n} x 512B"),
            (0..n)
                .map(|i| dense_attr_of_size(&format!("a{i}"), 512))
                .collect(),
        ));
    }

    // Objects this size need a block from a row past the last direct one, so
    // the heap reaches them through nested indirect blocks.
    shapes.push((
        "40 x 60,000B".to_string(),
        (0..40)
            .map(|i| dense_attr_of_size(&format!("big{i}"), 60_000))
            .collect(),
    ));

    // The managed/huge boundary, from both sides and mixed. "One past" also
    // leaves the doubling table with no managed objects at all.
    shapes.push((
        "at the managed limit".to_string(),
        vec![dense_attr_of_size("edge", DENSE_ATTR_MAX_MANAGED_OBJECT)],
    ));
    shapes.push((
        "one past the managed limit".to_string(),
        vec![dense_attr_of_size(
            "edge",
            DENSE_ATTR_MAX_MANAGED_OBJECT + 1,
        )],
    ));
    // Managed attributes on both sides of the huge ones, so a managed object
    // that *follows* a huge one takes the heap offset its own position
    // in the managed order implies.
    let mut mixed: Vec<AttributeMessage> = (0..10)
        .map(|i| dense_attr_of_size(&format!("small{i}"), 300))
        .collect();
    mixed.push(dense_attr_of_size(
        "huge0",
        DENSE_ATTR_MAX_MANAGED_OBJECT + 1,
    ));
    mixed.push(dense_attr_of_size(
        "huge1",
        DENSE_ATTR_MAX_MANAGED_OBJECT * 4,
    ));
    mixed.extend((10..20).map(|i| dense_attr_of_size(&format!("small{i}"), 300)));
    shapes.push(("managed and huge mixed".to_string(), mixed));

    for (label, attrs) in shapes {
        let plan =
            dense_attribute_write::dense_attrs_plan(&attrs, DenseAttrCreationOrder::Untracked)
                .unwrap();
        for base in [0u64, 0x1000, 0x8000_0000].map(StoredAddress::new) {
            let built = plan.build(base).unwrap();
            assert_eq!(
                plan.blob_len(),
                built.blob.len() as u64,
                "planned length must match the emitted heap for {label} at base {:#x}",
                base.get()
            );
        }
    }
}

/// Returns attribute names in a single-leaf name index's stored order.
///
/// Uses the records' creation indices to look up names in `attrs`.
fn name_index_order(attrs: &[AttributeMessage]) -> Vec<String> {
    const RECORD: usize = 8 + 1 + 4 + 4;
    let blob = dense_attribute_write::build_dense_attrs(
        attrs,
        DenseAttrCreationOrder::Untracked,
        StoredAddress::new(0),
    )
    .unwrap()
    .blob;
    let header = blob
        .windows(4)
        .position(|w| w == b"BTHD")
        .expect("a name index has a header");
    // Depth follows signature(4), version(1), type(1), node size(4), and record size(2).
    let depth = u16::from_le_bytes(blob[header + 12..header + 14].try_into().expect("2 bytes"));
    assert_eq!(depth, 0, "the fixture outgrew a single leaf");
    let leaf = blob
        .windows(4)
        .position(|w| w == b"BTLF")
        .expect("a name index has a leaf node");
    // signature(4) + version(1) + type(1), then the records.
    (0..attrs.len())
        .map(|i| {
            let at = leaf + 6 + i * RECORD + 9;
            let order = u32::from_le_bytes(blob[at..at + 4].try_into().expect("4 bytes"));
            attrs[order as usize].name.clone()
        })
        .collect()
}

#[test]
fn the_name_index_order_does_not_depend_on_insertion_order() {
    // "k155448" sorts before "k69209" and hashes the same, so an order that is
    // not fully determined by the names shows up as a difference here.
    let names = [
        "k69209", "k155448", "zeta", "alpha", "m", "beta", "gamma", "delta", "epsilon", "eta",
    ];
    let forward: Vec<AttributeMessage> = names
        .iter()
        .map(|n| type_builders::build_attr_message(n, &AttrValue::I64(1)).unwrap())
        .collect();
    let mut reversed = forward.clone();
    reversed.reverse();

    let order = name_index_order(&forward);
    assert_eq!(
        order,
        name_index_order(&reversed),
        "the index order changed with the insertion order"
    );

    // Verify the input hash collision before checking the records' positions.
    assert_eq!(
        crate::checksum::jenkins_lookup3(b"k69209"),
        crate::checksum::jenkins_lookup3(b"k155448"),
        "the fixture names no longer hash alike; pick a new colliding pair"
    );
    let at = |name: &str| {
        order
            .iter()
            .position(|n| n == name)
            .expect("every attribute is indexed")
    };
    assert_eq!(
        at("k155448") + 1,
        at("k69209"),
        "the colliding pair is not indexed in name order"
    );
}

/// Creation indices in input attribute order for [`scrambled_creation_order`].
const SCRAMBLE: [u16; 6] = [3, 1, 4, 0, 2, 5];

/// Builds a dense set with the creation indices specified by [`SCRAMBLE`].
fn scrambled_creation_order() -> (Vec<AttributeMessage>, DenseAttrCreationOrder) {
    let attrs: Vec<AttributeMessage> = (0..SCRAMBLE.len())
        .map(|i| {
            type_builders::build_attr_message(&format!("a{i:02}"), &AttrValue::I64(i as i64))
                .unwrap()
        })
        .collect();
    (
        attrs,
        DenseAttrCreationOrder::Tracked {
            indices: SCRAMBLE.to_vec(),
            max: 40,
            indexed: true,
        },
    )
}

/// Reads a B-tree header as `(offset, record size, root address, record count)`.
fn btree_header(blob: &[u8], tree_type: u8) -> (usize, u16, u64, u16) {
    let at = (0..blob.len().saturating_sub(6))
        .find(|&i| &blob[i..i + 4] == b"BTHD" && blob[i + 4] == 0 && blob[i + 5] == tree_type)
        .unwrap_or_else(|| panic!("no v2 B-tree of type {tree_type} in the blob"));
    let record_size = u16::from_le_bytes(blob[at + 10..at + 12].try_into().unwrap());
    let depth = u16::from_le_bytes(blob[at + 12..at + 14].try_into().unwrap());
    assert_eq!(depth, 0, "the fixture outgrew a single leaf");
    let root = u64::from_le_bytes(blob[at + 16..at + 24].try_into().unwrap());
    let nrec = u16::from_le_bytes(blob[at + 24..at + 26].try_into().unwrap());
    (at, record_size, root, nrec)
}

// Check the advertised creation-order address and correspondence with name-index heap IDs.
#[test]
fn an_indexed_dense_set_emits_a_creation_order_btree() {
    const COUNT: usize = SCRAMBLE.len();
    let (attrs, creation) = scrambled_creation_order();
    let heap_address = StoredAddress::new(0x4000);
    let built = dense_attribute_write::build_dense_attrs(&attrs, creation, heap_address).unwrap();

    let info =
        crate::attribute_info::AttributeInfoMessage::parse(&built.attr_info_message, OFFSET_SIZE)
            .expect("the emitted message parses");
    assert_eq!(info.max_creation_index, Some(40), "the recorded maximum");
    assert!(info.indexes_creation_order);
    let corder_addr = info
        .btree_creation_order_address
        .expect("an indexed set names its creation-order B-tree");

    let (at, record_size, root, nrec) = btree_header(&built.blob, 9);
    assert_eq!(
        heap_address.checked_add(at as u64).unwrap(),
        corder_addr,
        "the Attribute Info message names a different address than the tree sits at"
    );
    assert_eq!(
        record_size,
        8 + 1 + 4,
        "heap ID, message flags, creation order"
    );
    assert_eq!(nrec as usize, COUNT);

    // Creation indices are sorted independently of the input permutation.
    let leaf = (root - heap_address.get()) as usize + 6; // signature(4) + version(1) + type(1)
    let orders: Vec<u32> = (0..COUNT)
        .map(|i| {
            let at = leaf + i * record_size as usize + 9;
            u32::from_le_bytes(built.blob[at..at + 4].try_into().unwrap())
        })
        .collect();
    assert_eq!(orders, (0..COUNT as u32).collect::<Vec<_>>());

    // Identify each attribute through heap IDs shared with the name index.
    let (name_at, name_record, name_root, _) = btree_header(&built.blob, 8);
    assert_ne!(name_at, at, "the two indexes must be separate trees");
    let name_leaf = (name_root - heap_address.get()) as usize + 6;
    for (position, index) in SCRAMBLE.iter().enumerate() {
        let corder_at = leaf + *index as usize * record_size as usize;
        let id = &built.blob[corder_at..corder_at + 8];
        let hash = (0..COUNT)
            .find_map(|j| {
                let at = name_leaf + j * name_record as usize;
                (&built.blob[at..at + 8] == id)
                    .then(|| u32::from_le_bytes(built.blob[at + 13..at + 17].try_into().unwrap()))
            })
            .expect("every creation-order record's heap ID is in the name index");
        assert_eq!(
            hash,
            crate::checksum::jenkins_lookup3(format!("a{position:02}").as_bytes()),
            "creation index {index} names the wrong attribute",
        );
    }
}

#[test]
fn an_untracked_dense_set_emits_no_creation_order_btree() {
    let (attrs, _) = scrambled_creation_order();
    let built = dense_attribute_write::build_dense_attrs(
        &attrs,
        DenseAttrCreationOrder::Untracked,
        StoredAddress::new(0x4000),
    )
    .unwrap();
    let info =
        crate::attribute_info::AttributeInfoMessage::parse(&built.attr_info_message, OFFSET_SIZE)
            .expect("the emitted message parses");
    assert_eq!(info.max_creation_index, None);
    assert!(!info.indexes_creation_order);
    assert_eq!(info.btree_creation_order_address, None);
    assert!(
        !built
            .blob
            .windows(6)
            .any(|w| &w[..4] == b"BTHD" && w[5] == 9),
        "an untracked set must carry no creation-order index"
    );
}

/// Reads the huge-object count from a built fractal-heap header.
fn huge_object_count(blob: &[u8]) -> u64 {
    let ls = LENGTH_SIZE as usize;
    let os = OFFSET_SIZE as usize;
    assert_eq!(&blob[..4], b"FRHP");
    // version(1) + heap ID length(2) + I/O filter length(2) + flags(1) +
    // max managed object size(4) + next huge object ID + huge B-tree address +
    // free space + free-space manager address + managed space + allocated
    // managed space + allocation iterator + managed object count + huge
    // objects size.
    let at = 4 + 1 + 2 + 2 + 1 + 4 + ls + os + ls + os + ls + ls + ls + ls + ls;
    u64::from_le_bytes(blob[at..at + 8].try_into().unwrap())
}

/// Both classes are counted, and the split follows the declared managed-object
/// limit exactly.
#[test]
fn the_managed_object_limit_is_where_storage_changes_class() {
    // Sized under their final names: the name is part of the serialized
    // message, so renaming afterwards would move the attribute back across
    // the very threshold this is testing.
    let mixed = vec![
        dense_attr_of_size("at", DENSE_ATTR_MAX_MANAGED_OBJECT),
        dense_attr_of_size("past", DENSE_ATTR_MAX_MANAGED_OBJECT + 1),
    ];
    let blob = dense_attribute_write::build_dense_attrs(
        &mixed,
        DenseAttrCreationOrder::Untracked,
        StoredAddress::new(0),
    )
    .unwrap()
    .blob;
    assert_eq!(huge_object_count(&blob), 1);
    assert_eq!(managed_object_count(&blob), 1);
}

/// Reads the managed-object count from a built fractal-heap header.
fn managed_object_count(blob: &[u8]) -> u64 {
    let ls = LENGTH_SIZE as usize;
    let os = OFFSET_SIZE as usize;
    assert_eq!(&blob[..4], b"FRHP");
    let at = 4 + 1 + 2 + 2 + 1 + 4 + ls + os + ls + os + ls + ls + ls;
    u64::from_le_bytes(blob[at..at + 8].try_into().unwrap())
}

/// Reads the heap header's root-block address and current root-row count.
fn root_block(blob: &[u8]) -> (usize, u16) {
    let ls = LENGTH_SIZE as usize;
    let os = OFFSET_SIZE as usize;
    assert_eq!(&blob[..4], b"FRHP");
    // The fixed prefix, then next huge object ID, huge B-tree address, free
    // space, free-space manager address, the eight 8-byte statistics, table
    // width, starting block size, maximum direct block size, maximum heap
    // size and starting root rows.
    let at = 4 + 1 + 2 + 2 + 1 + 4 + ls + os + ls + os + 8 * ls + 2 + ls + ls + 2 + 2;
    let address = u64::from_le_bytes(blob[at..at + os].try_into().unwrap());
    let rows = u16::from_le_bytes(blob[at + os..at + os + 2].try_into().unwrap());
    (address as usize, rows)
}

#[test]
fn the_root_grows_into_an_indirect_block_rather_than_a_bigger_direct_one() {
    let fits = vec![dense_attr_of_size("a", 400)];
    let blob = dense_attribute_write::build_dense_attrs(
        &fits,
        DenseAttrCreationOrder::Untracked,
        StoredAddress::new(0),
    )
    .unwrap()
    .blob;
    let (address, rows) = root_block(&blob);
    assert_eq!(rows, 0, "one starting-size block still holds this heap");
    assert_eq!(&blob[address..address + 4], b"FHDB");

    let spills: Vec<AttributeMessage> = (0..8)
        .map(|i| dense_attr_of_size(&format!("a{i}"), 400))
        .collect();
    let blob = dense_attribute_write::build_dense_attrs(
        &spills,
        DenseAttrCreationOrder::Untracked,
        StoredAddress::new(0),
    )
    .unwrap()
    .blob;
    let (address, rows) = root_block(&blob);
    assert!(rows >= 1, "content past one block needs an indirect root");
    assert_eq!(&blob[address..address + 4], b"FHIB");
}

/// An attribute close to the managed-object limit needs the table's largest
/// direct block, and those only recur inside *nested* indirect blocks once
/// the root's own row of them is used up. Reading every attribute back proves
/// the emitter and the reader agree about where the deeper blocks sit.
#[test]
fn attributes_survive_a_heap_with_nested_indirect_blocks() {
    let mut fw = FileWriter::new();
    let ds = fw.create_dataset("data");
    ds.with_f64_data(&[1.0]);
    // Past DENSE_ATTR_THRESHOLD so the set goes to the heap at all, and each
    // one big enough that only the table's largest direct block will take it.
    let count = DENSE_ATTR_THRESHOLD + 2;
    let value = |i: usize| char::from(b'a' + i as u8).to_string().repeat(60_000);
    for i in 0..count {
        ds.set_attr(&format!("big{i}"), AttrValue::AsciiString(value(i)));
    }
    let bytes = fw.finish().unwrap();

    let nested = bytes.windows(4).filter(|w| *w == b"FHIB").count();
    assert!(
        nested >= 2,
        "expected a nested indirect block, got {nested}"
    );

    let sb = crate::superblock::read(bytes.as_slice())
        .unwrap()
        .superblock;
    let addr = group_v2::resolve_path_any(
        &bytes,
        AccessMode::ReadOnly,
        &sb,
        WIDTHS,
        AddressContext::from_effective_base(sb.base_address),
        &ObjectPath::parse("data"),
    )
    .unwrap();
    let hdr = ObjectHeader::parse_with_widths(
        &bytes,
        AccessMode::ReadOnly,
        addr as usize,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    )
    .unwrap();
    let attrs = crate::attribute::extract_attributes_full(
        source::zero_base_bytes(&bytes),
        AccessMode::ReadOnly,
        &hdr,
        WIDTHS,
        None,
    )
    .unwrap();
    assert_eq!(attrs.len(), count);
    for i in 0..count {
        let attr = attrs.iter().find(|a| a.name == format!("big{i}")).unwrap();
        assert_eq!(
            crate::data_read::read_as_strings(&attr.raw_data, &attr.datatype).unwrap(),
            vec![value(i)]
        );
    }
}

/// Returns the largest compact `i64`-array length using the encoder's measured overhead.
fn largest_fitting_i64_attr_elements() -> usize {
    let one =
        type_builders::build_attr_message("boundary", &AttrValue::I64Array(vec![0i64; 1])).unwrap();
    let overhead = one.serialize(OFFSET_WIDTH, LENGTH_SIZE).unwrap().len() - 8;
    (OBJECT_HEADER_MESSAGE_MAX - overhead) / 8
}

#[test]
fn compact_attr_at_the_message_size_limit_is_written() {
    let n = largest_fitting_i64_attr_elements();
    let attr =
        type_builders::build_attr_message("boundary", &AttrValue::I64Array(vec![7i64; n])).unwrap();
    // Pin the probe to the boundary itself: one more element must not fit,
    // or this test would still pass while exercising a tiny attribute.
    let size = attr.serialize(OFFSET_WIDTH, LENGTH_SIZE).unwrap().len();
    assert!(size <= OBJECT_HEADER_MESSAGE_MAX);
    assert!(
        size + 8 > OBJECT_HEADER_MESSAGE_MAX,
        "probe is not at the limit (got {size})"
    );

    let mut fw = FileWriter::new();
    fw.set_root_attr("boundary", AttrValue::I64Array(vec![7i64; n]));
    fw.create_dataset("d").with_f64_data(&[1.0]);
    let bytes = fw.finish().unwrap();

    let file = crate::reader::File::from_bytes(bytes).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.len(), 1);
}

// One attribute crosses the size limit without reaching the count threshold.
#[test]
fn an_attr_past_the_message_size_limit_moves_to_dense_storage() {
    let n = largest_fitting_i64_attr_elements() + 1;
    let attrs = vec![
        type_builders::build_attr_message("boundary", &AttrValue::I64Array(vec![0i64; n])).unwrap(),
    ];
    assert!(
        attrs[0].serialize(OFFSET_WIDTH, LENGTH_SIZE).unwrap().len() > OBJECT_HEADER_MESSAGE_MAX
    );
    assert!(
        needs_dense_attrs(&attrs).unwrap(),
        "one oversized attribute must select dense storage by itself"
    );

    let mut fw = FileWriter::new();
    fw.set_root_attr("boundary", AttrValue::I64Array(vec![7i64; n]));
    fw.create_dataset("d").with_f64_data(&[1.0]);
    let bytes = fw.finish().expect("written, not refused");

    let file = crate::reader::File::from_bytes(bytes).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.len(), 1);
    match attrs.get("boundary") {
        Some(AttrValue::I64Array(v)) => assert_eq!(v.len(), n),
        other => panic!("expected the attribute back, got {other:?}"),
    }
}

#[test]
fn an_attr_at_the_message_size_limit_stays_compact() {
    let n = largest_fitting_i64_attr_elements();
    let attrs = vec![
        type_builders::build_attr_message("boundary", &AttrValue::I64Array(vec![0i64; n])).unwrap(),
    ];
    assert!(!needs_dense_attrs(&attrs).unwrap());
}

/// Reads a dataset's VL-string byte objects from a complete file image.
fn read_vl_bytes(bytes: Vec<u8>, path: &str) -> Vec<crate::vl_data::VlByteObject> {
    let file = crate::reader::File::from_bytes(bytes).unwrap();
    file.dataset(path)
        .unwrap()
        .read_vlen_string_bytes(crate::vl_data::VlenStringReadOptions::default())
        .unwrap()
}

#[test]
fn vlen_string_dataset_roundtrips_values() {
    let mut fw = FileWriter::new();
    fw.create_dataset("labels")
        .with_vlen_strings(&["alpha", "beta", "gamma"]);
    let bytes = fw.finish().unwrap();
    let objs = read_vl_bytes(bytes, "labels");
    let got: Vec<_> = objs
        .iter()
        .map(|o| match o {
            crate::vl_data::VlByteObject::Bytes(b) => String::from_utf8(b.clone()).unwrap(),
            crate::vl_data::VlByteObject::Null => "<null>".to_string(),
        })
        .collect();
    assert_eq!(got, vec!["alpha", "beta", "gamma"]);
}

#[test]
fn vlen_string_dataset_preserves_null_vs_empty() {
    use crate::type_builders::VlStringElement;
    use crate::vl_data::VlByteObject;

    let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Utf8);
    let elements = vec![
        VlStringElement::Bytes(b"hi".to_vec()),
        VlStringElement::Null,
        VlStringElement::Bytes(Vec::new()), // empty string, not null
        VlStringElement::Bytes(b"end".to_vec()),
    ];
    let mut fw = FileWriter::new();
    fw.create_dataset("mixed")
        .with_vlen_string_elements(dt, &elements)
        .unwrap();
    let bytes = fw.finish().unwrap();
    let objs = read_vl_bytes(bytes, "mixed");
    assert_eq!(
        objs,
        vec![
            VlByteObject::Bytes(b"hi".to_vec()),
            VlByteObject::Null,
            VlByteObject::Bytes(Vec::new()),
            VlByteObject::Bytes(b"end".to_vec()),
        ]
    );
}

#[test]
fn vlen_string_dataset_preserves_embedded_nul() {
    use crate::type_builders::VlStringElement;
    use crate::vl_data::VlByteObject;

    let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Ascii);
    let payload = b"a\0b\0c".to_vec();
    let elements = vec![VlStringElement::Bytes(payload.clone())];
    let mut fw = FileWriter::new();
    fw.create_dataset("nul")
        .with_vlen_string_elements(dt, &elements)
        .unwrap();
    let bytes = fw.finish().unwrap();
    let objs = read_vl_bytes(bytes, "nul");
    assert_eq!(objs, vec![VlByteObject::Bytes(payload)]);
}

#[test]
fn vlen_string_dataset_preserves_non_utf8_bytes() {
    // Include invalid UTF-8 and an embedded null byte in the byte-preserving path.
    use crate::type_builders::VlStringElement;
    use crate::vl_data::VlByteObject;

    let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Ascii);
    let payload = vec![0xffu8, 0xfe, 0x80, 0x00, 0x41];
    let elements = vec![VlStringElement::Bytes(payload.clone())];
    let mut fw = FileWriter::new();
    fw.create_dataset("raw")
        .with_vlen_string_elements(dt, &elements)
        .unwrap();
    let bytes = fw.finish().unwrap();
    let objs = read_vl_bytes(bytes, "raw");
    assert_eq!(objs, vec![VlByteObject::Bytes(payload)]);
}

#[test]
fn vlen_string_dataset_2d_shape_roundtrips() {
    let mut fw = FileWriter::new();
    fw.create_dataset("grid")
        .with_vlen_strings(&["a", "bb", "ccc", "dddd"])
        .with_shape(&[2, 2]);
    let bytes = fw.finish().unwrap();
    let file = crate::reader::File::from_bytes(bytes).unwrap();
    let ds = file.dataset("grid").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![2, 2]);
    assert_eq!(
        ds.read_vlen_strings(crate::vl_data::VlenStringReadOptions::default())
            .unwrap(),
        vec!["a", "bb", "ccc", "dddd"]
    );
}

#[test]
fn vlen_string_dataset_all_null_no_heap() {
    use crate::type_builders::VlStringElement;
    use crate::vl_data::VlByteObject;

    let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Utf8);
    let elements = vec![VlStringElement::Null, VlStringElement::Null];
    let mut fw = FileWriter::new();
    fw.create_dataset("nulls")
        .with_vlen_string_elements(dt, &elements)
        .unwrap();
    let bytes = fw.finish().unwrap();
    let objs = read_vl_bytes(bytes, "nulls");
    assert_eq!(objs, vec![VlByteObject::Null, VlByteObject::Null]);
}

#[test]
fn vlen_string_dataset_with_nulls_spans_multiple_heap_collections() {
    // Interleave null elements so heap-object and dataset-element indices diverge.
    use crate::type_builders::VlStringElement;
    use crate::vl_data::VlByteObject;

    let count = 100_000;
    let elements: Vec<VlStringElement> = (0..count)
        .map(|i| {
            if i % 3 == 0 {
                VlStringElement::Null
            } else {
                VlStringElement::Bytes(format!("s{i}").into_bytes())
            }
        })
        .collect();
    let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Utf8);
    let mut fw = FileWriter::new();
    fw.create_dataset("mixed")
        .with_vlen_string_elements(dt, &elements)
        .unwrap();
    let bytes = fw.finish().unwrap();
    let objs = read_vl_bytes(bytes, "mixed");

    assert_eq!(objs.len(), count);
    // The 65,536th heap object belongs to dataset element 98,303.
    for i in [0, 1, 65_535, 98_302, 98_303, 98_304, count - 1] {
        let expected = if i % 3 == 0 {
            VlByteObject::Null
        } else {
            VlByteObject::Bytes(format!("s{i}").into_bytes())
        };
        assert_eq!(objs[i], expected, "element {i} did not round-trip");
    }
}

#[test]
fn chunked_vlen_string_dataset_roundtrips() {
    let mut fw = FileWriter::new();
    fw.create_dataset("chunked")
        .with_vlen_strings(&["a", "bb", "ccc", "dddd"])
        .with_chunks(&[2]);
    let bytes = fw.finish().unwrap();
    let f = crate::reader::File::from_bytes(bytes).unwrap();
    let ds = f.dataset("chunked").unwrap();
    assert_eq!(ds.read_string().unwrap(), ["a", "bb", "ccc", "dddd"]);
}

// Decode filtered chunks to check that their heap references resolve to the expected strings.
#[test]
#[cfg(feature = "deflate")]
fn filtered_chunked_vlen_string_dataset_roundtrips() {
    let mut fw = FileWriter::new();
    fw.create_dataset("filtered")
        .with_vlen_strings(&["alpha", "beta", "gamma", "delta"])
        .with_chunks(&[2])
        .with_deflate(DeflateLevel::Six);
    let bytes = fw.finish().unwrap();
    let f = crate::reader::File::from_bytes(bytes).unwrap();
    let ds = f.dataset("filtered").unwrap();
    assert_eq!(
        ds.read_string().unwrap(),
        ["alpha", "beta", "gamma", "delta"]
    );
}

#[test]
fn resizable_vlen_string_dataset_roundtrips() {
    let mut fw = FileWriter::new();
    fw.create_dataset("growable")
        .with_vlen_strings(&["one", "two", "three"])
        .with_shape(&[3])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[2]);
    let bytes = fw.finish().unwrap();
    let f = crate::reader::File::from_bytes(bytes).unwrap();
    let ds = f.dataset("growable").unwrap();
    assert_eq!(ds.read_string().unwrap(), ["one", "two", "three"]);
}

/// Null elements keep an undefined address through the early patch, exactly as
/// they do on the contiguous path: the mask selects which references move.
#[test]
fn chunked_vlen_string_dataset_preserves_nulls() {
    use crate::type_builders::VlStringElement;
    use crate::vl_data::VlByteObject;

    let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Utf8);
    let elements = vec![
        VlStringElement::Bytes(b"set".to_vec()),
        VlStringElement::Null,
        VlStringElement::Bytes(Vec::new()), // empty string, not null
        VlStringElement::Bytes(b"tail".to_vec()),
    ];
    let mut fw = FileWriter::new();
    fw.create_dataset("mixed")
        .with_vlen_string_elements(dt, &elements)
        .unwrap()
        .with_chunks(&[2]);
    let bytes = fw.finish().unwrap();
    assert_eq!(
        read_vl_bytes(bytes, "mixed"),
        vec![
            VlByteObject::Bytes(b"set".to_vec()),
            VlByteObject::Null,
            VlByteObject::Bytes(Vec::new()),
            VlByteObject::Bytes(b"tail".to_vec()),
        ]
    );
}

#[test]
fn contiguous_vlen_file_places_root_header_after_superblock() {
    let build = || {
        let mut fw = FileWriter::new();
        fw.create_dataset("plain")
            .with_vlen_strings(&["a", "bb", "ccc"]);
        fw.finish().unwrap()
    };
    // The root group object header sits immediately after the superblock when
    // nothing is placed early.
    let bytes = build();
    assert_eq!(
        &bytes[SUPERBLOCK_SIZE..SUPERBLOCK_SIZE + 4],
        b"OHDR",
        "a contiguous-only VL file must still open with the root OH at SUPERBLOCK_SIZE"
    );
}

#[test]
fn vlen_sequence_dataset_roundtrips_i32() {
    // Non-string VL (`H5T_VLEN { i32 }`): the per-element reference stores an
    // element *count*, while the heap object holds count*4 bytes. The
    // writer/reader pair must agree on that, including an empty sequence.
    use crate::type_builders::VlStringElement;
    use crate::vl_data::{VlByteObject, VlenStringReadOptions};

    let dt = Datatype::VariableLength(VariableLengthType::sequence(
        crate::type_builders::make_i32_type(),
    ));
    let seqs: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![], vec![-7, 42]];
    let elements: Vec<VlStringElement> = seqs
        .iter()
        .map(|s| VlStringElement::Bytes(s.iter().flat_map(|v| v.to_le_bytes()).collect()))
        .collect();
    let mut fw = FileWriter::new();
    fw.create_dataset("seq")
        .with_vlen_sequence_elements(dt, &elements)
        .unwrap();
    let bytes = fw.finish().unwrap();

    let file = crate::reader::File::from_bytes(bytes).unwrap();
    let ds = file.dataset("seq").unwrap();
    assert!(
        matches!(
            ds.datatype().unwrap(),
            Datatype::VariableLength(VariableLengthType::Sequence { .. })
        ),
        "datatype must stay a non-string variable-length sequence"
    );
    let (objs, elem_size) = ds
        .read_vlen_sequence_bytes(VlenStringReadOptions::default())
        .unwrap();
    assert_eq!(elem_size, 4);
    let got: Vec<Vec<i32>> = objs
        .iter()
        .map(|o| match o {
            VlByteObject::Null => Vec::new(),
            VlByteObject::Bytes(b) => b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c))
                .collect(),
        })
        .collect();
    assert_eq!(got, seqs);
}

#[test]
fn vlen_sequence_rejects_string_datatype() {
    // Pass a VL-string datatype to the sequence builder.
    use crate::type_builders::VlStringElement;
    let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Utf8);
    assert!(matches!(
        &dt,
        Datatype::VariableLength(VariableLengthType::String(_))
    ));
    let mut fw = FileWriter::new();
    let res = fw
        .create_dataset("x")
        .with_vlen_sequence_elements(dt, &[VlStringElement::Bytes(b"hi".to_vec())]);
    let Err(error) = res else {
        panic!("a variable-length string was accepted as a sequence");
    };
    assert_eq!(
        error,
        FormatError::TypeMismatch {
            expected: "non-string VariableLength",
            actual: "non-VariableLength",
        }
    );
}

// ---- library-version bounds select the on-disk format ----

/// Writes one dataset under `bounds` for format-version comparisons.
///
/// # Errors
///
/// Propagates errors from [`FileWriter::finish`].
fn write_bounded(bounds: Option<(LibVer, LibVer)>) -> Result<Vec<u8>, FormatError> {
    let mut fw = FileWriter::new();
    if let Some((low, high)) = bounds {
        fw.with_libver_bounds(low, high);
    }
    fw.create_dataset("values").with_f64_data(&[1.0, 2.0, 3.0]);
    fw.finish()
}

/// Reads a dataset's layout-version byte directly from its object header.
fn layout_message_version(bytes: &[u8], path: &str) -> u8 {
    let sb = crate::superblock::read(bytes).unwrap().superblock;
    let addr = group_v2::resolve_path_any(
        bytes,
        AccessMode::ReadOnly,
        &sb,
        WIDTHS,
        AddressContext::from_effective_base(sb.base_address),
        &ObjectPath::parse(path),
    )
    .unwrap();
    let oh = ObjectHeader::parse_with_widths(
        bytes,
        AccessMode::ReadOnly,
        addr as usize,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    )
    .unwrap();
    oh.messages
        .iter()
        .find(|m| m.msg_type == MessageType::DATA_LAYOUT)
        .expect("a dataset carries a data-layout message")
        .data[0]
}

/// `high` selects the format, and every version field in the file follows
/// that one decision.
#[test]
fn libver_bounds_select_the_superblock_and_layout_versions() {
    for (bounds, superblock, layout) in [
        (None, 3, 4),
        (Some((LibVer::Earliest, LibVer::LATEST)), 3, 4),
        (Some((LibVer::Earliest, LibVer::V110)), 3, 4),
        (Some((LibVer::V110, LibVer::LATEST)), 3, 4),
        // A lower bound above the newest format this crate writes licenses
        // newer encodings without requiring any, so it writes the 1.10 one.
        (Some((LibVer::LATEST, LibVer::LATEST)), 3, 4),
        (Some((LibVer::V112, LibVer::V112)), 3, 4),
        (Some((LibVer::Earliest, LibVer::V18)), 2, 3),
        (Some((LibVer::V18, LibVer::V18)), 2, 3),
    ] {
        let bytes = write_bounded(bounds).expect("these bounds are satisfiable");
        let sig = usize::try_from(
            crate::superblock::read(bytes.as_slice())
                .unwrap()
                .address
                .get(),
        )
        .unwrap();
        assert_eq!(
            bytes[sig + 8],
            superblock,
            "superblock version under bounds {bounds:?}"
        );
        assert_eq!(
            layout_message_version(&bytes, "values"),
            layout,
            "data-layout message version under bounds {bounds:?}"
        );
    }
}

// Compare complete output, accounting for version bytes and their checksums.
#[test]
fn v18_and_v110_outputs_differ_only_in_versions_and_checksums() {
    let newer = write_bounded(Some((LibVer::Earliest, LibVer::V110))).unwrap();
    let older = write_bounded(Some((LibVer::Earliest, LibVer::V18))).unwrap();
    assert_eq!(newer.len(), older.len(), "the two formats differ in size");

    let differing: Vec<usize> = (0..newer.len()).filter(|&i| newer[i] != older[i]).collect();
    let sig = usize::try_from(
        crate::superblock::read(newer.as_slice())
            .unwrap()
            .address
            .get(),
    )
    .unwrap();

    // Ten bytes, and every one of them accounted for: the superblock's
    // version byte and the four-byte checksum covering it, the layout
    // message's version byte, and the four-byte checksum of the object header
    // holding it. Any eleventh byte means a body diverged.
    assert_eq!(
        differing.len(),
        10,
        "expected two version bytes and the two checksums over them, got {differing:?}"
    );
    assert_eq!(
        &differing[..5],
        &[sig + 8, sig + 44, sig + 45, sig + 46, sig + 47],
        "the superblock version byte and its checksum must be the first to differ"
    );
    assert_eq!(
        newer[differing[5]], 4,
        "the sixth differing byte is the data-layout message version"
    );
    assert_eq!(older[differing[5]], 3);
    let trailing = &differing[6..];
    assert!(
        trailing[0] > differing[5] && trailing.windows(2).all(|w| w[1] == w[0] + 1),
        "the last four differing bytes must be the contiguous checksum trailing \
         the object header that holds the layout message, got {differing:?}"
    );
}

// Include inverted ranges with otherwise supported upper bounds.
#[rstest]
#[case::earliest_only(LibVer::Earliest, LibVer::Earliest, "earliest", "earliest")]
#[case::latest_below_v18(LibVer::LATEST, LibVer::V18, "v1.14", "v1.8")]
#[case::v114_below_v110(LibVer::V114, LibVer::V110, "v1.14", "v1.10")]
#[case::latest_below_v112(LibVer::LATEST, LibVer::V112, "v1.14", "v1.12")]
#[case::v112_below_v110(LibVer::V112, LibVer::V110, "v1.12", "v1.10")]
fn bounds_admitting_no_format_this_crate_writes_are_rejected(
    #[case] low: LibVer,
    #[case] high: LibVer,
    #[case] requested_low: &'static str,
    #[case] requested_high: &'static str,
) {
    assert_eq!(
        write_bounded(Some((low, high))).unwrap_err(),
        FormatError::LibverBoundsUnsatisfiable {
            writes: "v1.10",
            requested_low,
            requested_high,
        }
    );
}

#[rstest]
#[case::chunked(
    OlderFormatContent::Chunked,
    "a chunked, filtered, or resizable dataset"
)]
#[case::paged(OlderFormatContent::Paged, "a file-space setting")]
#[case::fsm(OlderFormatContent::Fsm, "a file-space setting")]
#[case::resizable(
    OlderFormatContent::Resizable,
    "a chunked, filtered, or resizable dataset"
)]
fn content_the_older_format_cannot_carry_is_rejected(
    #[case] fixture: OlderFormatContent,
    #[case] content: &'static str,
) {
    let mut fw = FileWriter::new();
    fw.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    match fixture {
        OlderFormatContent::Chunked => {
            fw.create_dataset("d")
                .with_i32_data(&(0..100).collect::<Vec<_>>())
                .with_chunks(&[10]);
        }
        OlderFormatContent::Paged => {
            fw.with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
            );
            fw.create_dataset("d").with_i32_data(&[1]);
        }
        OlderFormatContent::Fsm => {
            fw.with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
                    .with_page_size(FileSpacePageSize::DEFAULT),
            );
            fw.create_dataset("d").with_i32_data(&[1]);
        }
        OlderFormatContent::Resizable => {
            // An unlimited dimension requires chunked storage under the same format bound.
            fw.create_dataset("d")
                .with_i32_data(&[1, 2, 3])
                .with_shape(&[3])
                .with_maxshape(&[MaxExtent::Unlimited]);
        }
    }
    assert_eq!(
        fw.finish().unwrap_err(),
        FormatError::LibverTooOldForContent {
            content,
            needs: "v1.10",
            writing: "v1.8"
        }
    );
}

enum OlderFormatContent {
    Chunked,
    Fsm,
    Paged,
    Resizable,
}
// "The Attribute Message" in format specification, version 4.0, defines message
// versions 2 and 3 and the version 3 Name Character Set Encoding field.
const ATTRIBUTE_MESSAGE_VERSION_TWO: u8 = 2;
const ATTRIBUTE_MESSAGE_VERSION_THREE: u8 = 3;
const ATTRIBUTE_NAME_CHARACTER_SET_OFFSET: usize = 8;
const ASCII_NAME_CHARACTER_SET: u8 = 0;
const UTF8_NAME_CHARACTER_SET: u8 = 1;
// "Fractal Heap" in format specification, version 4.0, defines the heap header signature.
const FRACTAL_HEAP_SIGNATURE: &[u8] = b"FRHP";
// "The Group Info Message" in format specification, version 4.0, defines version zero
// and flags zero for Group Info messages with default phase-change and entry settings.
const DEFAULT_GROUP_INFO: [u8; 2] = [0, 0];
const GLOBAL_HEAP_COLLECTION_SIGNATURE: &[u8] = b"GCOL";
const OBJECT_HEADER_V2_SIGNATURE: &[u8] = b"OHDR";
