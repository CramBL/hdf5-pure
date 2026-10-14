use std::collections::BTreeSet;

use hdf5_pure::AttrValue;
use hdf5_pure::CharacterSet;
use hdf5_pure::CompoundTypeBuilder;
use hdf5_pure::Datatype;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FillTime;
use hdf5_pure::GlobalHeapLiveness;
use hdf5_pure::ReferenceType;
use hdf5_pure::StringPadding;
use hdf5_pure_format::__private as format;
use rstest::rstest;
use test_util::attribute;
use test_util::attribute::Attribute;
use test_util::data_layout;
use test_util::dataspace;
use test_util::global_heap;
use test_util::global_heap::Object;
use test_util::image::Image;
use test_util::link_message::HardLink;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags;
use test_util::object_header::MessageType;
use test_util::object_header::v2::Header;
use test_util::superblock::v2::Superblock;
use test_util::temp;
use test_util::widths::Widths;

#[rstest]
#[case::compact_attribute_of_strings(Form::CompactAttribute, vlen_string())]
#[case::compact_attribute_of_sequences(Form::CompactAttribute, vlen_sequence())]
#[case::contiguous_dataset_of_strings(Form::ContiguousDataset, vlen_string())]
#[case::contiguous_dataset_of_sequences(Form::ContiguousDataset, vlen_sequence())]
#[case::compact_dataset_of_strings(Form::CompactDataset, vlen_string())]
#[case::compact_dataset_of_sequences(Form::CompactDataset, vlen_sequence())]
#[case::fill_value_of_strings(Form::FillValue, vlen_string())]
fn each_supported_form_refers_to_its_collection(#[case] form: Form, #[case] datatype: Datatype) {
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"alpha"), Object::new(2, b"beta")]);
    let elements = [reference(5, collection, 1), reference(4, collection, 2)].concat();
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
        ("a", reference(5, shared, 1)),
        ("b", reference(4, shared, 2)),
        (
            "c",
            [reference(5, shared, 1), reference(5, other, 1)].concat(),
        ),
    ] {
        let messages = Form::ContiguousDataset.messages(&mut fixture, &vlen_string(), &elements);
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
        &vlen_string(),
        &reference(5, collection, 1),
    );
    let dataset = fixture.append_header(messages);
    let group = fixture.append_header(vec![
        hard_link("d", dataset),
        hard_link("root", Fixture::ROOT),
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
    let messages = Form::ContiguousDataset.messages(&mut fixture, &vlen_string(), &elements);
    fixture.object("d", messages);

    assert_eq!(
        fixture.liveness(),
        complete(referenced.then_some(collection))
    );
}

// `slots` holds the byte offset of each variable-length reference in an element.
#[rstest]
#[case::compound_with_a_string_member(with_integer(vlen_string()), &[4])]
#[case::compound_with_a_sequence_member(with_integer(vlen_sequence()), &[4])]
#[case::array_of_strings(array_of(vlen_string()), &[0, 16])]
#[case::array_of_compounds_with_a_sequence_member(array_of(with_integer(vlen_sequence())), &[4, 24])]
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
                element[slot..slot + REFERENCE_SIZE]
                    .copy_from_slice(&reference(5, collection, index));
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
    base_type: Box::new(vlen_sequence()),
})]
#[case::object_reference(object_reference())]
#[case::compound_with_an_object_reference(with_integer(object_reference()))]
fn a_datatype_with_unlocated_addresses_is_incomplete(#[case] datatype: Datatype) {
    let mut fixture = Fixture::new();
    let mut messages = dataset_messages(&datatype, &dataspace::scalar());
    messages.push(Message::new(
        MessageType::DATA_LAYOUT,
        &data_layout::contiguous(UNDEFINED_ADDRESS, 0),
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
    let mut messages = dataset_messages(&hdf5_pure::make_i32_type(), &dataspace::scalar());
    messages.push(message);
    fixture.object("d", messages);

    assert_eq!(fixture.liveness(), incomplete());
}

#[test]
fn a_dataset_with_a_shared_datatype_is_incomplete() {
    let mut fixture = Fixture::new();
    fixture.object(
        "d",
        vec![
            Message::new(MessageType::DATATYPE, &shared_message()).with_flags(MessageFlags::SHARED),
            Message::new(MessageType::DATASPACE, &dataspace::scalar()),
            Message::new(
                MessageType::DATA_LAYOUT,
                &data_layout::contiguous(UNDEFINED_ADDRESS, 0),
            ),
        ],
    );

    assert_eq!(fixture.liveness(), incomplete());
}

#[test]
fn every_batch_of_a_contiguous_dataset_is_read() {
    let mut fixture = Fixture::new();
    let first = fixture.collection(&[Object::new(1, b"alpha")]);
    let last = fixture.collection(&[Object::new(1, b"omega")]);
    let mut elements = reference(5, first, 1).repeat(ELEMENTS_PAST_A_BATCH - 1);
    elements.extend(reference(5, last, 1));
    let messages = Form::ContiguousDataset.messages(&mut fixture, &vlen_string(), &elements);
    fixture.object("d", messages);

    assert_eq!(fixture.liveness(), complete([first, last]));
}

#[test]
fn a_chunked_variable_length_dataset_is_incomplete() {
    let path = temp::temp_path("liveness_chunked.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("d")
        .with_vlen_strings(&["alpha", "beta", "gamma"])
        .with_chunks(&[2]);
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();

    assert_eq!(file.__global_heap_liveness().unwrap(), incomplete());
    file.close().unwrap();
}

#[rstest]
#[case::contiguous_dataset(Written::Dataset)]
#[case::compact_attribute(Written::Attributes(0))]
// The writer stores the attributes of an object with more than 8 in dense storage.
#[case::dense_attribute(Written::Attributes(12))]
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

#[derive(Clone, Copy, Debug)]
enum Form {
    CompactAttribute,
    CompactDataset,
    ContiguousDataset,
    FillValue,
}

impl Form {
    fn messages(self, fixture: &mut Fixture, datatype: &Datatype, elements: &[u8]) -> Vec<Message> {
        let element_size = datatype.type_size() as usize;
        let count = (elements.len() / element_size) as u64;
        let space = dataspace::v2(
            1,
            dataspace::Flags::NONE,
            dataspace::Kind::SIMPLE,
            &[count],
            None,
        );
        let mut messages = dataset_messages(datatype, &space);
        let layout = match self {
            Self::CompactAttribute => {
                return vec![Message::new(
                    MessageType::ATTRIBUTE,
                    &Attribute::new("a", &format::serialize_datatype(datatype), &space, elements)
                        .build(),
                )];
            }
            Self::CompactDataset => data_layout::compact(elements),
            Self::ContiguousDataset => {
                let address = fixture.image.append(elements) as u64;
                data_layout::contiguous(address, elements.len() as u64)
            }
            Self::FillValue => {
                messages.push(Message::new(
                    MessageType::FILL_VALUE,
                    &format::serialize_fill_value_message_v3(
                        FillTime::IfSet,
                        Some(&elements[..element_size]),
                    )
                    .unwrap(),
                ));
                data_layout::contiguous(UNDEFINED_ADDRESS, elements.len() as u64)
            }
        };
        messages.push(Message::new(MessageType::DATA_LAYOUT, &layout));
        messages
    }
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
            Self::Empty => reference(0, collection, 1),
            Self::NullAtUndefined => reference(0, UNDEFINED_ADDRESS, 0),
            Self::NullAtZero => reference(0, 0, 0),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Written {
    /// A variable-length attribute and this many integer attributes beside it.
    Attributes(usize),
    Dataset,
    Integers,
}

impl Written {
    fn builder(self, userblock: u64) -> FileBuilder {
        let mut builder = FileBuilder::new();
        builder.with_userblock(userblock);
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

/// A file of version 2 object headers under a version 3 superblock, whose root group refers to
/// each object through a hard link.
struct Fixture {
    image: Image,
    links: Vec<Message>,
}

impl Fixture {
    fn new() -> Self {
        let mut image = Image::new();
        image.place(0, &superblock().build());
        image.place(Self::ROOT as usize, &[0; ROOT_HEADER_SPACE]);
        Self {
            image,
            links: Vec::new(),
        }
    }

    fn collection(&mut self, objects: &[Object<'_>]) -> u64 {
        self.image
            .append(&global_heap::collection(objects, Widths::EIGHT)) as u64
    }

    fn object(&mut self, link_name: &str, messages: Vec<Message>) {
        let address = self.append_header(messages);
        self.link(link_name, address);
    }

    fn append_header(&mut self, messages: Vec<Message>) -> u64 {
        self.image.append(&Header::new().messages(messages).build()) as u64
    }

    fn link(&mut self, link_name: &str, address: u64) {
        self.links.push(hard_link(link_name, address));
    }

    fn liveness(mut self) -> GlobalHeapLiveness {
        let root = Header::new().messages(self.links).build();
        assert!(
            root.len() <= ROOT_HEADER_SPACE,
            "the root header outgrows its space"
        );
        self.image.place(Self::ROOT as usize, &root);
        let eof = self.image.as_bytes().len() as u64;
        self.image.place(0, &superblock().eof_address(eof).build());
        let path = temp::temp_path("liveness_fixture.h5");
        std::fs::write(&path, self.image.build()).unwrap();
        let file = File::open_rw(&path).unwrap();
        let liveness = file.__global_heap_liveness().unwrap();
        file.close().unwrap();
        liveness
    }

    /// The address of the root group's object header, directly after the superblock.
    const ROOT: u64 = 48;
}

fn superblock() -> Superblock {
    Superblock::new(Widths::EIGHT)
        .version(SUPERBLOCK_VERSION)
        .root_header_address(Fixture::ROOT)
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

fn dataset_messages(datatype: &Datatype, space: &[u8]) -> Vec<Message> {
    vec![
        Message::new(MessageType::DATATYPE, &format::serialize_datatype(datatype)),
        Message::new(MessageType::DATASPACE, space),
    ]
}

fn hard_link(link_name: &str, address: u64) -> Message {
    Message::new(
        MessageType::LINK,
        &HardLink::new(link_name, address).build(Widths::EIGHT),
    )
}

fn reference(len: u32, collection: u64, index: u32) -> Vec<u8> {
    global_heap::reference(len, collection, index, Widths::EIGHT)
}

fn vlen_string() -> Datatype {
    Datatype::VariableLength {
        is_string: true,
        padding: Some(StringPadding::NullTerminate),
        charset: Some(CharacterSet::Ascii),
        base_type: Box::new(hdf5_pure::make_u8_type()),
    }
}

fn with_integer(member: Datatype) -> Datatype {
    CompoundTypeBuilder::new()
        .field("n", hdf5_pure::make_i32_type())
        .field("v", member)
        .build()
        .unwrap()
}

fn array_of(base_type: Datatype) -> Datatype {
    Datatype::Array {
        base_type: Box::new(base_type),
        dimensions: vec![2],
    }
}

fn object_reference() -> Datatype {
    Datatype::Reference {
        size: 8,
        ref_type: ReferenceType::Object,
    }
}

fn vlen_sequence() -> Datatype {
    Datatype::VariableLength {
        is_string: false,
        padding: None,
        charset: None,
        base_type: Box::new(hdf5_pure::make_i32_type()),
    }
}

const REFERENCE_SIZE: usize = 16;
/// One element more than the editor reads from a contiguous dataset at a time.
const ELEMENTS_PAST_A_BATCH: usize = 4097;
const UNDEFINED_ADDRESS: u64 = u64::MAX;
/// The bytes reserved for the root group's object header, which the fixture writes once every
/// link is known.
const ROOT_HEADER_SPACE: usize = 512;

const SUPERBLOCK_VERSION: u8 = 3;

const USERBLOCK_SIZE: u64 = 512;
