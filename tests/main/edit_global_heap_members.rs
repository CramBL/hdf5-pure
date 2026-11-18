use std::fs;
use std::path::Path;

use hdf5_pure::{AttrValue, File, FileAccessProperties, MemoryStrategy, SyncPolicy};
use rstest::rstest;
use test_util::attribute::Attribute;
use test_util::dataspace;
use test_util::global_heap::{self, Object};
use test_util::object_header::{Message, MessageType};
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

const HEADER: usize = 16;
const ALIGNMENT: usize = 8;
const PADDING: u8 = 0xa5;

#[derive(Clone, Copy, Debug)]
enum Tail {
    Explicit,
    Full,
    Implied,
}

impl Tail {
    fn bytes(self) -> usize {
        match self {
            Self::Explicit => HEADER,
            Self::Full => 0,
            Self::Implied => ALIGNMENT,
        }
    }
}

#[rstest]
#[case::explicit_free_header(Tail::Explicit)]
#[case::full_collection(Tail::Full)]
#[case::short_implied_tail(Tail::Implied)]
fn grouped_attribute_removals_preserve_member_records_and_parent_capacity(
    #[case] tail: Tail,
    #[values(1, 3)] count: usize,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
    #[values(0, 65_536)] buffer_size: usize,
) {
    let path = test_util::temp::temp_path("member-layouts.h5");
    let mut fixture = Fixture::new();
    let mut parents = Vec::new();
    let payload_size = global_heap::MIN_COLLECTION_SIZE - 5 * HEADER - 3 * ALIGNMENT - tail.bytes();
    let datatype = global_heap_fixture::vlen_string();
    for index in 0..count {
        let payload = vec![b'q' + u8::try_from(index).unwrap(); payload_size];
        let objects = [
            Object {
                index: 400,
                reference_count: 0,
                data: b"left!",
            },
            Object {
                index: u16::MAX,
                reference_count: 0,
                data: &payload,
            },
            Object {
                index: 7,
                reference_count: u16::MAX,
                data: b"gone",
            },
            Object {
                index: 91,
                reference_count: 9,
                data: b"tail!",
            },
        ];
        let parent = Parent::append(&mut fixture, &objects, &[u16::MAX, 91]);
        assert_eq!(parent.tail, tail.bytes());
        let mut messages = Form::CompactDataset.messages(
            &mut fixture,
            &hdf5_pure::make_i32_type(),
            &17_i32.to_le_bytes(),
        );
        for (name, member, length) in [("first", 400, 5), ("second", 7, 4)] {
            let attribute = Attribute::new(
                name,
                &hdf5_pure_format::__private::serialize_datatype(&datatype).unwrap(),
                &dataspace::scalar(),
                &global_heap_fixture::reference(length, parent.address, member),
            )
            .build();
            messages.push(Message::new(MessageType::ATTRIBUTE, &attribute));
        }
        fixture.object(&format!("value_{index}"), messages);
        let references = [
            global_heap_fixture::reference(
                u32::try_from(payload.len()).unwrap(),
                parent.address,
                u32::from(u16::MAX),
            ),
            global_heap_fixture::reference(5, parent.address, 91),
        ]
        .concat();
        let messages = Form::ContiguousDataset.messages(&mut fixture, &datatype, &references);
        fixture.object(&format!("keep_{index}"), messages);
        parents.push((parent, String::from_utf8(payload).unwrap()));
    }
    fixture.write(&path);
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(memory)
            .with_sync_policy(SyncPolicy::OnClose)
            .with_page_buffer_size(buffer_size),
    )
    .unwrap();
    for index in 0..count {
        let mut dataset = file.dataset(&format!("value_{index}")).unwrap();
        dataset.remove_attr("first").unwrap();
        dataset.remove_attr("second").unwrap();
    }
    file.commit().unwrap();
    file.close().unwrap();
    for (parent, _) in &parents {
        parent.assert_compacted(&path);
    }
    let file = File::open(&path).unwrap();
    for (index, (_, payload)) in parents.iter().enumerate() {
        assert_eq!(
            file.dataset(&format!("keep_{index}"))
                .unwrap()
                .read_string()
                .unwrap(),
            [payload.as_str(), "tail!"]
        );
        let dataset = file.dataset(&format!("value_{index}")).unwrap();
        assert_eq!(dataset.read_i32().unwrap(), [17]);
        assert!(dataset.attrs().unwrap().is_empty());
    }
}

#[derive(Clone, Copy, Debug)]
enum Edit {
    RemoveAttribute,
    ReplaceAttribute,
    OverwriteDataset,
    DeleteDataset,
}

#[rstest]
#[case::attribute_removal(Edit::RemoveAttribute)]
#[case::attribute_replacement(Edit::ReplaceAttribute)]
#[case::dataset_overwrite(Edit::OverwriteDataset)]
#[case::dataset_deletion(Edit::DeleteDataset)]
fn public_edit_variants_remove_only_obsolete_members(
    #[case] edit: Edit,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let path = test_util::temp::temp_path("member-edit-variants.h5");
    let mut fixture = Fixture::new();
    let parent = Parent::append(
        &mut fixture,
        &[
            Object {
                index: 7,
                reference_count: 0,
                data: b"removed",
            },
            Object {
                index: 91,
                reference_count: 3,
                data: b"keep",
            },
        ],
        &[91],
    );
    let datatype = hdf5_pure::Datatype::VariableLength(hdf5_pure::VariableLengthType::string(
        hdf5_pure::StringPadding::NullTerminate,
        hdf5_pure::CharacterSet::Utf8,
    ));
    let reference = global_heap_fixture::reference(7, parent.address, 7);
    let messages = match edit {
        Edit::RemoveAttribute | Edit::ReplaceAttribute => {
            let mut messages = Form::CompactDataset.messages(
                &mut fixture,
                &hdf5_pure::make_i32_type(),
                &17_i32.to_le_bytes(),
            );
            messages.extend(Form::CompactAttribute.messages(&mut fixture, &datatype, &reference));
            messages
        }
        Edit::OverwriteDataset | Edit::DeleteDataset => {
            Form::ContiguousDataset.messages(&mut fixture, &datatype, &reference)
        }
    };
    fixture.object("value", messages);
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &datatype,
        &global_heap_fixture::reference(4, parent.address, 91),
    );
    fixture.object("keep", messages);
    fixture.write(&path);
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(memory),
    )
    .unwrap();
    match edit {
        Edit::RemoveAttribute => file.dataset("value").unwrap().remove_attr("a").unwrap(),
        Edit::ReplaceAttribute => file
            .dataset("value")
            .unwrap()
            .set_attr("a", AttrValue::I64(9))
            .unwrap(),
        Edit::OverwriteDataset => file
            .dataset("value")
            .unwrap()
            .write_staged(|dataset| {
                dataset.with_vlen_strings(&["after"]);
            })
            .unwrap(),
        Edit::DeleteDataset => file.root().delete("value").unwrap(),
    }
    file.commit().unwrap();
    file.close().unwrap();
    parent.assert_compacted(&path);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("keep").unwrap().read_string().unwrap(),
        ["keep"]
    );
    match edit {
        Edit::RemoveAttribute => {
            let dataset = file.dataset("value").unwrap();
            assert_eq!(dataset.read_i32().unwrap(), [17]);
            assert!(dataset.attrs().unwrap().is_empty());
        }
        Edit::ReplaceAttribute => {
            let dataset = file.dataset("value").unwrap();
            assert_eq!(dataset.read_i32().unwrap(), [17]);
            assert_eq!(dataset.attrs().unwrap().get("a"), Some(&AttrValue::I64(9)));
        }
        Edit::OverwriteDataset => assert_eq!(
            file.dataset("value").unwrap().read_string().unwrap(),
            ["after"]
        ),
        Edit::DeleteDataset => assert_eq!(file.root().datasets().unwrap(), ["keep"]),
    }
}

struct Parent {
    address: u64,
    original: Vec<u8>,
    expected: Vec<u8>,
    tail: usize,
}

impl Parent {
    fn append(fixture: &mut Fixture, objects: &[Object<'_>], retained: &[u16]) -> Self {
        let mut original = global_heap::collection_of_size(
            objects,
            Widths::EIGHT,
            global_heap::MIN_COLLECTION_SIZE,
        );
        let mut cursor = HEADER;
        let mut kept = Vec::new();
        for object in objects {
            let size = HEADER + object.data.len().next_multiple_of(ALIGNMENT);
            let payload_end = cursor + HEADER + object.data.len();
            original[payload_end..cursor + size].fill(PADDING);
            assert_eq!(&original[cursor..cursor + 2], &object.index.to_le_bytes());
            assert_eq!(
                &original[cursor + 2..cursor + 4],
                &object.reference_count.to_le_bytes()
            );
            assert_eq!(&original[cursor + HEADER..payload_end], object.data);
            if retained.contains(&object.index) {
                kept.push(original[cursor..cursor + size].to_vec());
            }
            cursor += size;
        }
        assert_eq!(kept.len(), retained.len());
        let tail = original.len() - cursor;
        let mut expected = original.clone();
        cursor = HEADER;
        for record in kept {
            expected[cursor..cursor + record.len()].copy_from_slice(&record);
            cursor += record.len();
        }
        let free_size = expected.len() - cursor;
        assert!(free_size >= HEADER);
        expected[cursor..cursor + HEADER].fill(0);
        expected[cursor + 8..cursor + HEADER]
            .copy_from_slice(&u64::try_from(free_size).unwrap().to_le_bytes());
        let address = u64::try_from(fixture.image.append(&original)).unwrap();
        Self {
            address,
            original,
            expected,
            tail,
        }
    }

    fn assert_compacted(&self, path: &Path) {
        let bytes = fs::read(path).unwrap();
        let start = usize::try_from(self.address).unwrap();
        let after = &bytes[start..start + self.original.len()];
        assert_ne!(after, self.original);
        assert_eq!(after, self.expected);
        assert_eq!(&after[..HEADER], &self.original[..HEADER]);
    }
}
