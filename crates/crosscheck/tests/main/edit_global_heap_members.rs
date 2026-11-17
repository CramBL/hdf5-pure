#![cfg(feature = "__libhdf5-1.10")]

use std::fs;
use std::ops::Range;
use std::path::Path;

use hdf5_pure::{AttrValue, EditBacking, File, FileAccessProperties, FileSpaceStrategy};
use libhdf5::file::LibraryVersion;
use libhdf5::plist::file_create::FileSpaceStrategy as NativeStrategy;
use libhdf5::types::VarLenAscii;
use rstest::rstest;
use test_util::attribute::Attribute;
use test_util::dataspace;
use test_util::global_heap::{self, Object};
use test_util::object_header::{Message, MessageType};
use test_util::widths::Widths;
use test_util_hdf5::file;
use test_util_hdf5::global_heap::attributes;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

#[rstest]
#[case::explicit(Tail::Explicit)]
#[case::full(Tail::Full)]
#[case::implied(Tail::Implied)]
fn native_attribute_removal_compacts_the_shared_parent(
    #[case] tail: Tail,
    #[values(EditBacking::Bounded, EditBacking::Mirrored)] backing: EditBacking,
) {
    let path = test_util::temp::temp_path("native-heap-member-compaction.h5");
    let retained = "r".repeat(CAPACITY - 4 * HEADER_SIZE - 2 * ALIGNMENT - tail.bytes());
    write_native(&path, &retained);
    let before = fs::read(&path).unwrap();
    let parent = Parent::read(&before);
    assert_eq!(parent.bytes.len(), CAPACITY);
    assert_eq!(parent.tail, tail.bytes());
    assert_eq!(
        parent
            .members
            .iter()
            .map(|member| member.object.data)
            .collect::<Vec<_>>(),
        [b"left".as_slice(), b"middle", retained.as_bytes()]
    );
    let pure = open_editor(&path, backing);
    let settings = pure.file_space_settings().unwrap().unwrap();
    assert_eq!(settings.strategy(), FileSpaceStrategy::FsmAggr);
    assert!(settings.persist());
    let attrs = pure.root().attrs().unwrap();
    assert_eq!(attrs.len(), 3);
    for (name, value) in [
        ("first", "left"),
        ("second", "middle"),
        ("third", &retained),
    ] {
        assert_eq!(attrs[name], AttrValue::VarLenAsciiString(value.into()));
    }
    pure.root().remove_attr("first").unwrap();
    pure.root().remove_attr("second").unwrap();
    pure.commit().unwrap();
    assert!(!pure.has_staged_edits());
    parent.assert_protected(&pure);
    pure.close().unwrap();

    let after = fs::read(&path).unwrap();
    parent.assert_compacted(&after, &parent.members[2..]);
    let pure = File::open(&path).unwrap();
    let attrs = pure.root().attrs().unwrap();
    assert_eq!(attrs.len(), 1);
    assert_eq!(
        attrs["third"],
        AttrValue::VarLenAsciiString(retained.clone())
    );
    assert_eq!(pure.dataset("data").unwrap().read_i32().unwrap(), NUMBERS);
    pure.close().unwrap();
    let native = file::libhdf5_open(&path);
    assert_eq!(native.attr_names().unwrap(), ["third"]);
    assert_eq!(
        attributes::ascii_values(&native.attr("third").unwrap()),
        [retained]
    );
    assert_eq!(
        native.dataset("data").unwrap().read_raw::<i32>().unwrap(),
        NUMBERS
    );
    native.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum Tail {
    Explicit,
    Full,
    Implied,
}

impl Tail {
    fn bytes(self) -> usize {
        match self {
            Self::Explicit => HEADER_SIZE,
            Self::Full => 0,
            Self::Implied => ALIGNMENT,
        }
    }
}

fn write_native(path: &Path, retained: &str) {
    let native = libhdf5::File::with_options()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::V110, LibraryVersion::latest())
        })
        .with_fcpl(|properties| {
            properties.file_space_strategy(NativeStrategy::FreeSpaceManager {
                paged: false,
                persist: true,
                threshold: 1,
            })
        })
        .create(path)
        .unwrap();
    native
        .new_dataset_builder()
        .with_data(&NUMBERS)
        .create("data")
        .unwrap();
    for (name, value) in [("first", "left"), ("second", "middle"), ("third", retained)] {
        attributes::write_ascii(&native, name, value);
    }
    native.close().unwrap();
}

#[rstest]
fn native_readers_resolve_sparse_members_after_compaction(
    #[values(EditBacking::Bounded, EditBacking::Mirrored)] backing: EditBacking,
) {
    let path = test_util::temp::temp_path("native-sparse-heap-members.h5");
    let objects = [
        Object::new(400, b"removed"),
        Object {
            index: u16::MAX,
            reference_count: 37,
            data: b"retained",
        },
        Object::new(7, b"obsolete"),
        Object {
            index: 91,
            reference_count: 9,
            data: b"tail!",
        },
    ];
    write_sparse_source(&path, &objects);
    let before = fs::read(&path).unwrap();
    let parent = Parent::read(&before);
    assert_eq!(parent.bytes.len(), CAPACITY);
    assert_eq!(parent.objects(), objects);
    assert_eq!(parent.members[3].bytes.last(), Some(&RECORD_PADDING));
    assert_native_sparse(&path, &[("first", "removed"), ("second", "obsolete")]);

    let pure = open_editor(&path, backing);
    assert_eq!(pure.dataset("keep").unwrap().read_string().unwrap(), KEPT);
    let mut data = pure.dataset("data").unwrap();
    let attrs = data.attrs().unwrap();
    assert_eq!(
        attrs["first"],
        AttrValue::VarLenAsciiString("removed".into())
    );
    assert_eq!(
        attrs["second"],
        AttrValue::VarLenAsciiString("obsolete".into())
    );
    data.remove_attr("first").unwrap();
    data.remove_attr("second").unwrap();
    pure.commit().unwrap();
    assert!(!pure.has_staged_edits());
    parent.assert_protected(&pure);
    drop(data);
    pure.close().unwrap();

    let after = fs::read(&path).unwrap();
    parent.assert_compacted(&after, &[parent.members[1], parent.members[3]]);
    let pure = File::open(&path).unwrap();
    assert_eq!(pure.dataset("keep").unwrap().read_string().unwrap(), KEPT);
    let data = pure.dataset("data").unwrap();
    assert_eq!(data.read_i32().unwrap(), NUMBERS);
    assert!(data.attrs().unwrap().is_empty());
    drop(data);
    pure.close().unwrap();
    assert_native_sparse(&path, &[]);
}

fn assert_native_sparse(path: &Path, attrs: &[(&str, &str)]) {
    let native = file::libhdf5_open(path);
    let kept = native
        .dataset("keep")
        .unwrap()
        .read_raw::<VarLenAscii>()
        .unwrap();
    assert_eq!(
        kept.iter().map(VarLenAscii::as_str).collect::<Vec<_>>(),
        KEPT
    );
    let data = native.dataset("data").unwrap();
    assert_eq!(data.read_raw::<i32>().unwrap(), NUMBERS);
    let names = attrs.iter().map(|&(name, _)| name).collect::<Vec<_>>();
    assert_eq!(data.attr_names().unwrap(), names);
    for &(name, value) in attrs {
        assert_eq!(attributes::ascii_values(&data.attr(name).unwrap()), [value]);
    }
    drop(data);
    native.close().unwrap();
}

fn open_editor(path: &Path, backing: EditBacking) -> File {
    let pure = File::open_rw_with_options(
        path,
        FileAccessProperties::new().with_memory_strategy(backing.into()),
    )
    .unwrap();
    assert_eq!(pure.edit_backing(), Some(backing));
    assert_eq!(pure.superblock().offset_size.get(), 8);
    assert_eq!(pure.superblock().length_size.get(), 8);
    pure
}

fn write_sparse_source(path: &Path, objects: &[Object<'_>; 4]) {
    let mut fixture = Fixture::new();
    let mut bytes = global_heap::collection(objects, Widths::EIGHT);
    let mut offset = HEADER_SIZE;
    for object in objects {
        let data = offset + HEADER_SIZE;
        let end = data + object.data.len().next_multiple_of(ALIGNMENT);
        bytes[data + object.data.len()..end].fill(RECORD_PADDING);
        offset = end;
    }
    let parent = u64::try_from(fixture.image.append(&bytes)).unwrap();
    let reference = |object: Object<'_>| {
        global_heap_fixture::reference(
            u32::try_from(object.data.len()).unwrap(),
            parent,
            u32::from(object.index),
        )
    };
    let datatype = global_heap_fixture::vlen_string();
    let mut records = Form::CompactDataset.messages(
        &mut fixture,
        &hdf5_pure::make_i32_type(),
        &NUMBERS.map(i32::to_le_bytes).concat(),
    );
    let datatype_bytes = hdf5_pure_format::__private::serialize_datatype(&datatype).unwrap();
    for (name, member) in [("first", objects[0]), ("second", objects[2])] {
        let attribute = Attribute::new(
            name,
            &datatype_bytes,
            &dataspace::scalar(),
            &reference(member),
        )
        .build();
        records.push(Message::new(MessageType::ATTRIBUTE, &attribute));
    }
    fixture.object("data", records);
    let references = [reference(objects[1]), reference(objects[3])].concat();
    let records = Form::ContiguousDataset.messages(&mut fixture, &datatype, &references);
    fixture.object("keep", records);
    fixture.write(path);
}

struct Parent<'a> {
    range: Range<u64>,
    bytes: &'a [u8],
    members: Vec<Member<'a>>,
    tail: usize,
}

#[derive(Clone, Copy)]
struct Member<'a> {
    object: Object<'a>,
    bytes: &'a [u8],
}

impl<'a> Parent<'a> {
    fn read(image: &'a [u8]) -> Self {
        let ranges = global_heap::collection_ranges(image, Widths::EIGHT);
        let [range] = ranges.as_slice() else {
            panic!("expected one heap collection, got {ranges:?}");
        };
        let start = usize::try_from(range.start).unwrap();
        let end = usize::try_from(range.end).unwrap();
        let bytes = &image[start..end];
        assert_eq!(&bytes[..4], b"GCOL");
        assert_eq!(bytes[4], 1);
        let mut members = Vec::new();
        let mut offset = HEADER_SIZE;
        while bytes.len() - offset >= HEADER_SIZE {
            let header = &bytes[offset..offset + HEADER_SIZE];
            let index = u16::from_le_bytes(header[..2].try_into().unwrap());
            let size =
                usize::try_from(u64::from_le_bytes(header[8..16].try_into().unwrap())).unwrap();
            if index == 0 {
                assert_eq!(size, bytes.len() - offset);
                break;
            }
            let start = offset + HEADER_SIZE;
            let end = start + size.next_multiple_of(ALIGNMENT);
            members.push(Member {
                object: Object {
                    index,
                    reference_count: u16::from_le_bytes(header[2..4].try_into().unwrap()),
                    data: &bytes[start..start + size],
                },
                bytes: &bytes[offset..end],
            });
            offset = end;
        }
        Self {
            range: range.clone(),
            bytes,
            members,
            tail: bytes.len() - offset,
        }
    }

    fn objects(&self) -> Vec<Object<'a>> {
        self.members.iter().map(|member| member.object).collect()
    }

    fn assert_protected(&self, file: &File) {
        let free = file.reusable_free_sections().unwrap();
        assert!(free.sections().iter().all(|section| {
            section.end_address().get() <= self.range.start
                || self.range.end <= section.address().get()
        }));
    }

    fn assert_compacted(&self, image: &[u8], retained: &[Member<'_>]) {
        let compacted = Parent::read(image);
        assert_eq!(compacted.range, self.range);
        assert_eq!(
            compacted.objects(),
            retained
                .iter()
                .map(|member| member.object)
                .collect::<Vec<_>>()
        );
        let mut expected = self.bytes.to_vec();
        let mut free = HEADER_SIZE;
        for member in retained {
            let end = free + member.bytes.len();
            expected[free..end].copy_from_slice(member.bytes);
            free = end;
        }
        let size = u64::try_from(self.bytes.len() - free).unwrap();
        expected[free..free + HEADER_SIZE].fill(0);
        expected[free + 8..free + HEADER_SIZE].copy_from_slice(&size.to_le_bytes());
        assert_ne!(compacted.bytes, self.bytes);
        assert_eq!(compacted.bytes, expected);
    }
}

const HEADER_SIZE: usize = 16;
const ALIGNMENT: usize = 8;
const CAPACITY: usize = global_heap::MIN_COLLECTION_SIZE;
const RECORD_PADDING: u8 = 0xa5;
const NUMBERS: [i32; 4] = [7, 11, 13, 17];
const KEPT: [&str; 2] = ["retained", "tail!"];
