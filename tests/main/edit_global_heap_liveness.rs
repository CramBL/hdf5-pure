use std::collections::BTreeSet;

use hdf5_pure::AttrValue;
use hdf5_pure::Datatype;
use hdf5_pure::DeflateLevel;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::GlobalHeapLiveness;
use hdf5_pure::MaxExtent;
use hdf5_pure::StoredAddress;
use hdf5_pure::UserblockSize;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_format::__private as format;
use rstest::rstest;
use test_util::attribute;
use test_util::attribute::Attribute;
use test_util::btree_v1;
use test_util::data_layout;
use test_util::dataspace;
use test_util::global_heap;
use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags;
use test_util::object_header::MessageType;
use test_util::temp;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

#[rstest]
#[case::compact_attribute_of_strings(Form::CompactAttribute, global_heap_fixture::vlen_string())]
#[case::compact_attribute_of_sequences(
    Form::CompactAttribute,
    global_heap_fixture::vlen_sequence()
)]
#[case::contiguous_dataset_of_strings(Form::ContiguousDataset, global_heap_fixture::vlen_string())]
#[case::contiguous_dataset_of_sequences(
    Form::ContiguousDataset,
    global_heap_fixture::vlen_sequence()
)]
#[case::compact_dataset_of_strings(Form::CompactDataset, global_heap_fixture::vlen_string())]
#[case::compact_dataset_of_sequences(Form::CompactDataset, global_heap_fixture::vlen_sequence())]
#[case::fill_value_of_strings(Form::FillValue, global_heap_fixture::vlen_string())]
fn each_supported_form_refers_to_its_collection(#[case] form: Form, #[case] datatype: Datatype) {
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"alpha"), Object::new(2, b"beta")]);
    let elements = [
        global_heap_fixture::reference(5, collection, 1),
        global_heap_fixture::reference(4, collection, 2),
    ]
    .concat();
    let messages = form.messages(&mut fixture, &datatype, &elements);
    fixture.object("x", messages);

    assert_eq!(fixture.liveness(), complete([collection]));
}

#[test]
fn objects_that_share_a_collection_refer_to_it_once() {
    let mut fixture = Fixture::new();
    let shared = fixture.collection(&[Object::new(1, b"alpha"), Object::new(2, b"beta")]);
    let other = fixture.collection(&[Object::new(1, b"gamma")]);
    for (name, elements) in [
        ("a", global_heap_fixture::reference(5, shared, 1)),
        ("b", global_heap_fixture::reference(4, shared, 2)),
        (
            "c",
            [
                global_heap_fixture::reference(5, shared, 1),
                global_heap_fixture::reference(5, other, 1),
            ]
            .concat(),
        ),
    ] {
        let messages = Form::ContiguousDataset.messages(
            &mut fixture,
            &global_heap_fixture::vlen_string(),
            &elements,
        );
        fixture.object(name, messages);
    }

    assert_eq!(fixture.liveness(), complete([shared, other]));
}

#[test]
fn an_object_behind_two_links_and_a_cycle_refers_to_its_collection() {
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"alpha")]);
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, collection, 1),
    );
    let dataset = fixture.append_header(messages);
    let group = fixture.append_header(vec![
        global_heap_fixture::hard_link("d", dataset),
        global_heap_fixture::hard_link("root", Fixture::ROOT),
    ]);
    fixture.link("g", group);
    fixture.link("d", dataset);

    assert_eq!(fixture.liveness(), complete([collection]));
}

#[rstest]
#[case::null_elements(&[Element::NullAtUndefined, Element::NullAtZero], false)]
#[case::an_empty_element_beside_null_ones(
    &[Element::NullAtUndefined, Element::NullAtZero, Element::Empty],
    true
)]
fn only_an_element_with_an_object_refers_to_its_collection(
    #[case] elements: &[Element],
    #[case] referenced: bool,
) {
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"")]);
    let elements: Vec<u8> = elements
        .iter()
        .flat_map(|element| element.reference(collection))
        .collect();
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &elements,
    );
    fixture.object("d", messages);

    assert_eq!(
        fixture.liveness(),
        complete(referenced.then_some(collection))
    );
}

// `slots` holds the byte offset of each variable-length reference in an element.
#[rstest]
#[case::compound_with_a_string_member(global_heap_fixture::with_integer(global_heap_fixture::vlen_string()), &[4])]
#[case::compound_with_a_sequence_member(global_heap_fixture::with_integer(global_heap_fixture::vlen_sequence()), &[4])]
#[case::array_of_strings(global_heap_fixture::array_of(global_heap_fixture::vlen_string()), &[0, 16])]
#[case::array_of_compounds_with_a_sequence_member(global_heap_fixture::array_of(global_heap_fixture::with_integer(global_heap_fixture::vlen_sequence())), &[4, 24])]
fn embedded_members_refer_to_their_collections(
    #[values(
        Form::CompactAttribute,
        Form::CompactDataset,
        Form::ContiguousDataset,
        Form::FillValue
    )]
    form: Form,
    #[case] datatype: Datatype,
    #[case] slots: &[usize],
) {
    let mut fixture = Fixture::new();
    let collections: Vec<u64> = slots
        .iter()
        .map(|_| fixture.collection(&[Object::new(1, b"alpha"), Object::new(2, b"beta")]))
        .collect();
    let mut element = vec![0; datatype.type_size() as usize];
    let elements: Vec<u8> = [1, 2]
        .into_iter()
        .flat_map(|index| {
            for (&slot, &collection) in slots.iter().zip(&collections) {
                element[slot..slot + global_heap_fixture::REFERENCE_SIZE]
                    .copy_from_slice(&global_heap_fixture::reference(5, collection, index));
            }
            element.clone()
        })
        .collect();
    let messages = form.messages(&mut fixture, &datatype, &elements);
    fixture.object("x", messages);

    assert_eq!(fixture.liveness(), complete(collections));
}

#[rstest]
#[case::sequence_of_sequences(Datatype::VariableLength {
    is_string: false,
    padding: None,
    charset: None,
    base_type: Box::new(global_heap_fixture::vlen_sequence()),
})]
#[case::object_reference(global_heap_fixture::object_reference())]
#[case::compound_with_an_object_reference(global_heap_fixture::with_integer(
    global_heap_fixture::object_reference()
))]
fn a_datatype_with_unlocated_addresses_is_incomplete(#[case] datatype: Datatype) {
    let mut fixture = Fixture::new();
    let mut messages = global_heap_fixture::dataset_messages(&datatype, &dataspace::scalar());
    messages.push(Message::new(
        MessageType::DATA_LAYOUT,
        &data_layout::contiguous(global_heap_fixture::UNDEFINED_ADDRESS, 0),
    ));
    fixture.object("d", messages);

    assert_eq!(fixture.liveness(), incomplete());
}

#[rstest]
#[case::unknown_message(Message::new(MessageType(0x7f), &[]))]
#[case::shared_attribute(
    Message::new(MessageType::ATTRIBUTE, &shared_message()).with_flags(MessageFlags::SHARED)
)]
#[case::attribute_with_a_shared_datatype(Message::new(
    MessageType::ATTRIBUTE,
    &Attribute::new("a", &shared_message(), &dataspace::scalar(), &[])
        .flags(attribute::Flags::SHARED_DATATYPE)
        .build(),
))]
#[case::virtual_layout(Message::new(MessageType::DATA_LAYOUT, &data_layout::virtual_storage()))]
fn a_message_the_analysis_cannot_read_is_incomplete(#[case] message: Message) {
    let mut fixture = Fixture::new();
    let mut messages =
        global_heap_fixture::dataset_messages(&hdf5_pure::make_i32_type(), &dataspace::scalar());
    messages.push(message);
    fixture.object("d", messages);

    assert_eq!(fixture.liveness(), incomplete());
}

#[rstest]
#[case::shared_datatype(vec![
    Message::new(MessageType::DATATYPE, &shared_message()).with_flags(MessageFlags::SHARED),
    Message::new(MessageType::DATASPACE, &dataspace::scalar()),
])]
#[case::shared_filter_pipeline(vec![
    Message::new(MessageType::DATATYPE, &format::serialize_datatype(&global_heap_fixture::vlen_string())),
    Message::new(MessageType::DATASPACE, &dataspace::scalar()),
    Message::new(MessageType::FILTER_PIPELINE, &shared_message()).with_flags(MessageFlags::SHARED),
])]
fn a_dataset_with_a_shared_message_is_incomplete(#[case] mut messages: Vec<Message>) {
    let mut fixture = Fixture::new();
    messages.push(Message::new(
        MessageType::DATA_LAYOUT,
        &data_layout::contiguous(global_heap_fixture::UNDEFINED_ADDRESS, 0),
    ));
    fixture.object("d", messages);

    assert_eq!(fixture.liveness(), incomplete());
}

#[test]
fn every_batch_of_a_contiguous_dataset_is_read() {
    let mut fixture = Fixture::new();
    let first = fixture.collection(&[Object::new(1, b"alpha")]);
    let last = fixture.collection(&[Object::new(1, b"omega")]);
    let mut elements =
        global_heap_fixture::reference(5, first, 1).repeat(ELEMENTS_PAST_A_BATCH - 1);
    elements.extend(global_heap_fixture::reference(5, last, 1));
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &elements,
    );
    fixture.object("d", messages);

    assert_eq!(fixture.liveness(), complete([first, last]));
}

#[rstest]
#[case::corrupt_deflate_stream(DEFLATE_FILTER, 32)]
#[case::filter_without_a_decoder(UNKNOWN_FILTER, 32)]
// The filtered size runs past the end of the file.
#[case::unreadable_chunk(DEFLATE_FILTER, 1 << 20)]
fn a_chunk_that_cannot_be_read_or_decoded_is_incomplete(
    #[case] filter_id: u16,
    #[case] filtered_size: u64,
) {
    let mut fixture = Fixture::new();
    let stored = [0xff; 32];
    let chunk = fixture.image.append(&stored) as u64;
    let pipeline = format::FilterPipeline {
        version: FILTER_PIPELINE_VERSION,
        filters: vec![format::FilterDescription {
            filter_id,
            name: None,
            flags: 0,
            client_data: vec![6],
        }],
    };
    let layout = format::DataLayout::encode_chunked(
        format::ChunkedLayoutFlags::NONE,
        &[2],
        global_heap_fixture::REFERENCE_SIZE as u32,
        format::ChunkIndexInfo::SingleChunk {
            filtered: Some(format::FilteredSingleChunk {
                filtered_size,
                filter_mask: 0,
            }),
        },
        StoredAddress::new(chunk),
        format::OffsetWidth::Eight,
        format::LengthWidth::Eight,
    )
    .unwrap();
    let space = dataspace::v2(
        1,
        dataspace::Flags::NONE,
        dataspace::Kind::SIMPLE,
        &[2],
        None,
    );
    let mut messages =
        global_heap_fixture::dataset_messages(&global_heap_fixture::vlen_string(), &space);
    messages.push(Message::new(
        MessageType::FILTER_PIPELINE,
        &pipeline.serialize().unwrap(),
    ));
    messages.push(Message::new(MessageType::DATA_LAYOUT, &layout));
    fixture.object("d", messages);

    assert_eq!(fixture.liveness(), incomplete());
}

#[test]
fn an_unallocated_chunked_dataset_refers_to_no_collection() {
    let mut fixture = Fixture::new();
    let space = dataspace::v2(
        1,
        dataspace::Flags::NONE,
        dataspace::Kind::SIMPLE,
        &[2],
        None,
    );
    let mut messages =
        global_heap_fixture::dataset_messages(&global_heap_fixture::vlen_string(), &space);
    messages.push(Message::new(
        MessageType::DATA_LAYOUT,
        &data_layout::chunked(
            global_heap_fixture::UNDEFINED_ADDRESS,
            &[2],
            global_heap_fixture::REFERENCE_SIZE as u32,
        ),
    ));
    fixture.object("d", messages);

    assert_eq!(fixture.liveness(), complete([]));
}

#[test]
fn an_unfiltered_chunk_of_the_wrong_length_is_incomplete() {
    let mut fixture = Fixture::new();
    // One element, where a chunk of two holds 32 bytes.
    let stored = [0; global_heap_fixture::REFERENCE_SIZE];
    let chunk = fixture.image.append(&stored) as u64;
    let index = fixture.image.append(&btree_v1::node(
        btree_v1::NodeType::CHUNK,
        0,
        &[
            btree_v1::chunk_key(stored.len() as u32, 0, &[0, 0]),
            btree_v1::chunk_key(0, 0, &[2, 0]),
        ],
        &[chunk],
        Widths::EIGHT,
    )) as u64;
    let space = dataspace::v2(
        1,
        dataspace::Flags::NONE,
        dataspace::Kind::SIMPLE,
        &[2],
        None,
    );
    let mut messages =
        global_heap_fixture::dataset_messages(&global_heap_fixture::vlen_string(), &space);
    messages.push(Message::new(
        MessageType::DATA_LAYOUT,
        &data_layout::chunked(index, &[2], global_heap_fixture::REFERENCE_SIZE as u32),
    ));
    fixture.object("d", messages);

    assert_eq!(fixture.liveness(), incomplete());
}

#[rstest]
#[case::contiguous_dataset(Written::Dataset)]
#[case::compact_attribute(Written::Attributes(0))]
// The writer stores the attributes of an object with more than 8 in dense storage.
#[case::dense_attribute(Written::Attributes(12))]
#[case::chunked_dataset(Written::Chunked { filtered: false, extendable: false })]
#[case::filtered_chunked_dataset(Written::Chunked { filtered: true, extendable: false })]
#[case::extendable_chunked_dataset(Written::Chunked { filtered: false, extendable: true })]
#[case::filtered_extendable_chunked_dataset(Written::Chunked { filtered: true, extendable: true })]
#[case::no_variable_length_data(Written::Integers)]
fn a_written_file_refers_to_every_collection_it_holds(
    #[case] written: Written,
    #[values(0, USERBLOCK_SIZE)] userblock: u64,
) {
    let path = temp::temp_path("liveness_written.h5");
    written.builder(userblock).write(&path).unwrap();
    let expected = complete(global_heap::collection_starts(
        &std::fs::read(&path).unwrap(),
    ));
    let file = File::open_rw(&path).unwrap();

    assert_eq!(file.__global_heap_liveness().unwrap(), expected);
    file.close().unwrap();
}

/// A variable-length element of length 0.
#[derive(Clone, Copy, Debug)]
enum Element {
    /// Refers to object 1 of a collection.
    Empty,
    /// Has the undefined collection address and object index 0.
    NullAtUndefined,
    /// Has collection address 0 and object index 0.
    NullAtZero,
}

impl Element {
    fn reference(self, collection: u64) -> Vec<u8> {
        match self {
            Self::Empty => global_heap_fixture::reference(0, collection, 1),
            Self::NullAtUndefined => {
                global_heap_fixture::reference(0, global_heap_fixture::UNDEFINED_ADDRESS, 0)
            }
            Self::NullAtZero => global_heap_fixture::reference(0, 0, 0),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Written {
    /// A variable-length attribute and this many integer attributes beside it.
    Attributes(u8),
    Chunked {
        filtered: bool,
        extendable: bool,
    },
    Dataset,
    Integers,
}

impl Written {
    fn builder(self, userblock: u64) -> FileBuilder {
        let mut builder = FileBuilder::new();
        builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
        match self {
            Self::Attributes(others) => {
                builder.set_attr(
                    "labels",
                    AttrValue::VarLenAsciiCharArray(vec!["alpha".into(), "beta".into()]),
                );
                for index in 0..others {
                    builder.set_attr(&format!("a{index}"), AttrValue::I64(1));
                }
            }
            Self::Chunked {
                filtered,
                extendable,
            } => {
                let dataset = builder
                    .create_dataset("d")
                    .with_vlen_strings(&["alpha", "", "beta"])
                    .with_chunks(&[2]);
                if filtered {
                    dataset.with_deflate(DeflateLevel::Six);
                }
                if extendable {
                    dataset.with_maxshape(&[MaxExtent::Unlimited]);
                }
            }
            Self::Dataset => {
                builder
                    .create_dataset("d")
                    .with_vlen_strings(&["alpha", "", "beta"]);
            }
            Self::Integers => {
                builder.create_dataset("d").with_i32_data(&[1, 2, 3]);
            }
        }
        builder
    }
}

fn complete(collections: impl IntoIterator<Item = u64>) -> GlobalHeapLiveness {
    GlobalHeapLiveness {
        referenced_collections: collections.into_iter().collect(),
        analysis_complete: true,
    }
}

fn incomplete() -> GlobalHeapLiveness {
    GlobalHeapLiveness {
        referenced_collections: BTreeSet::new(),
        analysis_complete: false,
    }
}

fn shared_message() -> Vec<u8> {
    format::encode_sohm_ref(&[0; format::FHEAP_ID_LEN])
}

const FILTER_PIPELINE_VERSION: u8 = 2;
const DEFLATE_FILTER: u16 = 1;
const UNKNOWN_FILTER: u16 = 0x7fff;
/// One element more than the editor reads from a contiguous dataset at a time.
const ELEMENTS_PAST_A_BATCH: usize = 4097;

const USERBLOCK_SIZE: u64 = 512;
