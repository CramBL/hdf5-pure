use std::path::Path;

use hdf5_pure::CharacterSet;
use hdf5_pure::CompoundTypeBuilder;
use hdf5_pure::Datatype;
use hdf5_pure::File;
use hdf5_pure::FillTime;
use hdf5_pure::GlobalHeapLiveness;
use hdf5_pure::ReferenceType;
use hdf5_pure::StringPadding;
use hdf5_pure_format::__private as format;
use test_util::attribute::Attribute;
use test_util::data_layout;
use test_util::dataspace;
use test_util::global_heap;
use test_util::global_heap::Object;
use test_util::image::Image;
use test_util::link_message::HardLink;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v2::Header;
use test_util::superblock::v2::Superblock;
use test_util::temp;
use test_util::widths::Widths;

/// Where an object stores the elements that refer to global heap objects.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Form {
    CompactAttribute,
    CompactDataset,
    ContiguousDataset,
    /// The fill value of a dataset without data, which holds the first element.
    FillValue,
}

impl Form {
    /// Returns the header messages of an object that stores `elements` of `datatype` in this
    /// form, and appends the data of a contiguous dataset to the image of `fixture`.
    pub(crate) fn messages(
        self,
        fixture: &mut Fixture,
        datatype: &Datatype,
        elements: &[u8],
    ) -> Vec<Message> {
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

/// A file of version 2 object headers under a version 3 superblock, whose root group refers to
/// each object through a hard link.
pub(crate) struct Fixture {
    pub(crate) image: Image,
    links: Vec<Message>,
}

impl Fixture {
    pub(crate) fn new() -> Self {
        let mut image = Image::new();
        image.place(0, &superblock().build());
        image.place(Self::ROOT as usize, &[0; ROOT_HEADER_SPACE]);
        Self {
            image,
            links: Vec::new(),
        }
    }

    pub(crate) fn collection(&mut self, objects: &[Object<'_>]) -> u64 {
        self.image
            .append(&global_heap::collection(objects, Widths::EIGHT)) as u64
    }

    pub(crate) fn object(&mut self, link_name: &str, messages: Vec<Message>) {
        let address = self.append_header(messages);
        self.link(link_name, address);
    }

    pub(crate) fn append_header(&mut self, messages: Vec<Message>) -> u64 {
        self.image.append(&Header::new().messages(messages).build()) as u64
    }

    pub(crate) fn link(&mut self, link_name: &str, address: u64) {
        self.links.push(hard_link(link_name, address));
    }

    pub(crate) fn liveness(self) -> GlobalHeapLiveness {
        let path = temp::temp_path("liveness_fixture.h5");
        self.write(&path);
        let file = File::open_rw(&path).unwrap();
        let liveness = file.__global_heap_liveness().unwrap();
        file.close().unwrap();
        liveness
    }

    pub(crate) fn write(mut self, path: &Path) {
        let link_info = format::LinkInfoMessage {
            max_creation_order: None,
            fractal_heap_address: None,
            btree_name_index_address: None,
            btree_creation_order_address: None,
        };
        let root = Header::new()
            .message(Message::new(
                MessageType::LINK_INFO,
                &link_info.serialize(format::OffsetWidth::Eight),
            ))
            .messages(self.links)
            .build();
        assert!(
            root.len() <= ROOT_HEADER_SPACE,
            "the root header outgrows its space"
        );
        self.image.place(Self::ROOT as usize, &root);
        let eof = self.image.as_bytes().len() as u64;
        self.image.place(0, &superblock().eof_address(eof).build());
        std::fs::write(path, self.image.build()).unwrap();
    }

    /// The address of the root group's object header, directly after the superblock.
    pub(crate) const ROOT: u64 = 48;
}

pub(crate) fn superblock() -> Superblock {
    Superblock::new(Widths::EIGHT)
        .version(SUPERBLOCK_VERSION)
        .root_header_address(Fixture::ROOT)
}

pub(crate) fn dataset_messages(datatype: &Datatype, space: &[u8]) -> Vec<Message> {
    vec![
        Message::new(MessageType::DATATYPE, &format::serialize_datatype(datatype)),
        Message::new(MessageType::DATASPACE, space),
    ]
}

pub(crate) fn hard_link(link_name: &str, address: u64) -> Message {
    Message::new(
        MessageType::LINK,
        &HardLink::new(link_name, address).build(Widths::EIGHT),
    )
}

pub(crate) fn reference(len: u32, collection: u64, index: u32) -> Vec<u8> {
    global_heap::reference(len, collection, index, Widths::EIGHT)
}

pub(crate) fn vlen_string() -> Datatype {
    Datatype::VariableLength {
        is_string: true,
        padding: Some(StringPadding::NullTerminate),
        charset: Some(CharacterSet::Ascii),
        base_type: Box::new(hdf5_pure::make_u8_type()),
    }
}

pub(crate) fn with_integer(member: Datatype) -> Datatype {
    CompoundTypeBuilder::new()
        .field("n", hdf5_pure::make_i32_type())
        .field("v", member)
        .build()
        .unwrap()
}

pub(crate) fn array_of(base_type: Datatype) -> Datatype {
    Datatype::Array {
        base_type: Box::new(base_type),
        dimensions: vec![2],
    }
}

pub(crate) fn object_reference() -> Datatype {
    Datatype::Reference {
        size: 8,
        ref_type: ReferenceType::Object,
    }
}

pub(crate) fn vlen_sequence() -> Datatype {
    Datatype::VariableLength {
        is_string: false,
        padding: None,
        charset: None,
        base_type: Box::new(hdf5_pure::make_i32_type()),
    }
}

pub(crate) const REFERENCE_SIZE: usize = 16;
pub(crate) const UNDEFINED_ADDRESS: u64 = u64::MAX;
/// The bytes reserved for the root group's object header, which the fixture writes once every
/// link is known.
pub(crate) const ROOT_HEADER_SPACE: usize = 512;
pub(crate) const SUPERBLOCK_VERSION: u8 = 3;
