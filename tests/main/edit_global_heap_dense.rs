use std::fs;
use std::ops::Range;
use std::path::Path;

use hdf5_pure::{AttrValue, File, FileAccessProperties, MemoryStrategy};
use rstest::rstest;
use test_util::attribute::{self, Attribute};
use test_util::bytes;
use test_util::dataspace;
use test_util::dense_attributes;
use test_util::fractal_heap;
use test_util::global_heap::{self, Object};
use test_util::image::Image;
use test_util::object_header::{Message, MessageType, v2};
use test_util::superblock::v2::Superblock;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

const PARENT: u64 = 8_192;

#[rstest]
#[case::group_dataset_deletion(Owner::Group, Removal::Dataset)]
#[case::group_attribute_removal(Owner::Group, Removal::Attribute)]
#[case::dataset_deletion(Owner::Dataset, Removal::Dataset)]
#[case::dataset_attribute_removal(Owner::Dataset, Removal::Attribute)]
fn dense_attribute_members_survive_compaction_and_later_root_edits(
    #[case] owner: Owner,
    #[case] removal: Removal,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    edit_dense_owner(owner, removal, memory, Accounting::Complete);
}

#[rstest]
#[case::group(Owner::Group)]
#[case::dataset(Owner::Dataset)]
fn contradictory_dense_accounting_retains_the_original_parent(
    #[case] owner: Owner,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    edit_dense_owner(owner, Removal::Dataset, memory, Accounting::Contradictory);
}

fn edit_dense_owner(
    owner: Owner,
    removal: Removal,
    memory: MemoryStrategy,
    accounting: Accounting,
) {
    let path = test_util::temp::temp_path("dense-member-retention.h5");
    let source = DenseOwner::write(&path, owner, removal, accounting);
    let original = fs::read(&path).unwrap();
    assert!(source.direct_used < source.direct.len());
    assert!(source.leaf_used < source.leaf.len());
    assert!(
        original[source.direct.start + source.direct_used..source.direct.end]
            .iter()
            .all(|byte| *byte == 0xa5)
    );
    assert!(
        original[source.leaf.start + source.leaf_used..source.leaf.end]
            .iter()
            .all(|byte| *byte == 0xcc)
    );
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(memory),
    )
    .unwrap();
    source.assert_values(&file);
    match removal {
        Removal::Dataset => {
            assert_eq!(
                file.dataset("victim").unwrap().read_string().unwrap(),
                ["retained", "removed"]
            );
            file.root().delete("victim").unwrap();
        }
        Removal::Attribute => {
            assert_eq!(
                file.root().attrs().unwrap().get("a"),
                Some(&AttrValue::VarLenAsciiStringArray(vec![
                    "retained".into(),
                    "removed".into()
                ]))
            );
            file.root().remove_attr("a").unwrap();
        }
    }
    file.commit().unwrap();
    assert!(!file.has_staged_edits());
    source.assert_values(&file);
    source.assert_excluded(&file);
    file.root().set_attr("later", AttrValue::I32(23)).unwrap();
    file.commit().unwrap();
    source.assert_values(&file);
    source.assert_excluded(&file);
    file.close().unwrap();
    let after = fs::read(&path).unwrap();
    for range in &source.protected {
        assert_eq!(&after[range.clone()], &original[range.clone()]);
    }
    let start = usize::try_from(PARENT).unwrap();
    let before = &original[start..start + global_heap::MIN_COLLECTION_SIZE];
    let mut expected = before.to_vec();
    expected[40..64].copy_from_slice(&before[64..88]);
    expected[64..80].fill(0);
    let free_size = u64::try_from(expected.len()).unwrap() - 64;
    expected[72..80].copy_from_slice(&free_size.to_le_bytes());
    match accounting {
        Accounting::Complete => {
            assert_ne!(&after[start..start + expected.len()], before);
            assert_eq!(&after[start..start + expected.len()], expected);
        }
        Accounting::Contradictory => assert_eq!(&after[start..start + before.len()], before),
    }
    let file = File::open(&path).unwrap();
    source.assert_values(&file);
    assert!(
        file.root()
            .datasets()
            .unwrap()
            .iter()
            .all(|name| name != "victim")
    );
    let attrs = file.root().attrs().unwrap();
    assert!(!attrs.contains_key("a"));
    assert_eq!(attrs.get("later"), Some(&AttrValue::I32(23)));
}

#[derive(Clone, Copy, Debug)]
enum Owner {
    Group,
    Dataset,
}

#[derive(Clone, Copy, Debug)]
enum Removal {
    Dataset,
    Attribute,
}

#[derive(Clone, Copy, Debug)]
enum Accounting {
    Complete,
    Contradictory,
}

struct DenseOwner {
    owner: Owner,
    direct: Range<usize>,
    direct_used: usize,
    leaf: Range<usize>,
    leaf_used: usize,
    protected: Vec<Range<usize>>,
}

impl DenseOwner {
    fn write(path: &Path, owner: Owner, removal: Removal, accounting: Accounting) -> Self {
        let datatype = global_heap_fixture::vlen_string();
        let message = Attribute::new(
            "value",
            &hdf5_pure_format::__private::serialize_datatype(&datatype).unwrap(),
            &dataspace::scalar(),
            &global_heap_fixture::reference(8, PARENT, 7),
        )
        .build();
        assert!(message.len() <= 64);
        let mut bytes = dense_attributes::one_message(&message, Widths::EIGHT);
        let heap = signature(&bytes, b"FRHP");
        let direct_at = signature(&bytes, b"FHDB");
        let leaf_at = signature(&bytes, b"BTLF");
        let index = signature(&bytes, b"BTHD");
        let header = usize::try_from(bytes::u64_at(&bytes, 36)).unwrap();
        assert_eq!(&bytes[header..header + 4], b"OHDR");
        let layout = fractal_heap::Header::new(0)
            .heap_id_length(8)
            .managed_object_count(1);
        let direct_used = layout.direct_block(0, &[], Widths::EIGHT).len() + message.len();
        if matches!(accounting, Accounting::Complete) {
            // Set managed space, allocated space and the allocation iterator for one direct block.
            for field in [46, 54, 62] {
                bytes::set_uint_at(&mut bytes, heap + field, 8, layout.direct_block_size());
            }
            bytes::set_uint_at(
                &mut bytes,
                heap + 30,
                8,
                layout.direct_block_size() - u64::try_from(direct_used).unwrap(),
            );
            test_util::checksum::restamp(&mut bytes, heap, direct_at - heap);
        }
        let direct = direct_at..direct_at + usize::try_from(layout.direct_block_size()).unwrap();
        assert_eq!(direct.end, leaf_at);
        let leaf = leaf_at..leaf_at + 1_024;
        assert_eq!(leaf.end, index);
        let leaf_used = 6 + 17 + 4;
        bytes[direct.start + direct_used..direct.end].fill(0xa5);
        bytes[leaf.start + leaf_used..leaf.end].fill(0xcc);
        let mut protected = vec![heap..direct_at, direct.clone(), leaf.clone(), index..header];
        if matches!(owner, Owner::Dataset) {
            bytes.truncate(header);
            let mut builder = Fixture::new();
            let mut messages = Form::CompactDataset.messages(
                &mut builder,
                &hdf5_pure::make_i32_type(),
                &17_i32.to_le_bytes(),
            );
            messages.push(Message::new(
                MessageType::ATTRIBUTE_INFO,
                &attribute::info(
                    Some(u64::try_from(heap).unwrap()),
                    Some(u64::try_from(index).unwrap()),
                    Widths::EIGHT,
                ),
            ));
            bytes.extend(v2::Header::new().messages(messages).build());
        }
        protected.push(header..bytes.len());
        let mut image = Image::starting_with(&bytes);
        let collection = global_heap::collection(
            &[
                Object {
                    index: 7,
                    reference_count: 2,
                    data: b"retained",
                },
                Object::new(41, b"removed"),
                Object {
                    index: 91,
                    reference_count: 3,
                    data: b"keep!",
                },
            ],
            Widths::EIGHT,
        );
        image.place(usize::try_from(PARENT).unwrap(), &collection);
        let mut root = vec![global_heap_fixture::hard_link(
            "retained",
            u64::try_from(header).unwrap(),
        )];
        let mut builder = Fixture::new();
        let messages = Form::CompactDataset.messages(
            &mut builder,
            &datatype,
            &global_heap_fixture::reference(5, PARENT, 91),
        );
        let keep = image.append(&v2::Header::new().messages(messages).build());
        root.push(global_heap_fixture::hard_link(
            "keep",
            u64::try_from(keep).unwrap(),
        ));
        let reference = [
            global_heap_fixture::reference(8, PARENT, 7),
            global_heap_fixture::reference(7, PARENT, 41),
        ]
        .concat();
        match removal {
            Removal::Dataset => {
                let messages = Form::CompactDataset.messages(&mut builder, &datatype, &reference);
                let victim = image.append(&v2::Header::new().messages(messages).build());
                root.push(global_heap_fixture::hard_link(
                    "victim",
                    u64::try_from(victim).unwrap(),
                ));
            }
            Removal::Attribute => {
                root.extend(Form::CompactAttribute.messages(&mut builder, &datatype, &reference))
            }
        }
        let root = image.append(
            &v2::Header::new()
                .messages(test_util::group_info::compact_messages(&root))
                .build(),
        );
        image.place(
            0,
            &Superblock::new(Widths::EIGHT)
                .root_header_address(u64::try_from(root).unwrap())
                .eof_address(u64::try_from(image.as_bytes().len()).unwrap())
                .build(),
        );
        fs::write(path, image.build()).unwrap();
        Self {
            owner,
            direct,
            direct_used,
            leaf,
            leaf_used,
            protected,
        }
    }

    fn assert_values(&self, file: &File) {
        let attributes = match self.owner {
            Owner::Group => file.group("retained").unwrap().attrs().unwrap(),
            Owner::Dataset => {
                let dataset = file.dataset("retained").unwrap();
                assert_eq!(dataset.read_i32().unwrap(), [17]);
                dataset.attrs().unwrap()
            }
        };
        assert_eq!(attributes.len(), 1);
        assert_eq!(
            attributes.get("value"),
            Some(&AttrValue::VarLenAsciiString("retained".into()))
        );
        assert_eq!(
            file.dataset("keep").unwrap().read_string().unwrap(),
            ["keep!"]
        );
    }

    fn assert_excluded(&self, file: &File) {
        let sections = file.reusable_free_sections().unwrap();
        for range in self.protected.iter().cloned().chain(std::iter::once(
            usize::try_from(PARENT).unwrap()
                ..usize::try_from(PARENT).unwrap() + global_heap::MIN_COLLECTION_SIZE,
        )) {
            assert!(
                sections.sections().iter().all(|section| {
                    section.end_address().get() <= u64::try_from(range.start).unwrap()
                        || u64::try_from(range.end).unwrap() <= section.address().get()
                }),
                "live dense allocation {range:?} overlaps {sections:?}"
            );
        }
    }
}

fn signature(bytes: &[u8], signature: &[u8; 4]) -> usize {
    let offsets = bytes::signature_offsets(bytes, signature);
    let [offset] = offsets.as_slice() else {
        panic!("expected one {signature:?} signature, got {offsets:?}");
    };
    *offset
}
