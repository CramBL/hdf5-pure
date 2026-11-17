use std::fs;
use std::ops::Range;
use std::path::Path;

use hdf5_pure::{AttrValue, File, FileAccessProperties, MemoryStrategy};
use rstest::rstest;
use test_util::global_heap::Object;
use test_util::group_info;
use test_util::object_header::{Message, v2};
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

#[rstest]
#[case::group_removal(Owner::Group, Edit::Remove)]
#[case::group_replacement(Owner::Group, Edit::Replace)]
#[case::aliased_root_removal(Owner::AliasedRoot, Edit::Remove)]
#[case::aliased_root_replacement(Owner::AliasedRoot, Edit::Replace)]
fn root_edits_preserve_retained_continuations_and_their_heap_values(
    #[case] owner: Owner,
    #[case] edit: Edit,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let path = test_util::temp::temp_path("retained-continuation-values.h5");
    let source = ContinuedOwner::write(&path, owner);
    let original = fs::read(&path).unwrap();
    assert_eq!(
        &original[source.continuation.start..source.continuation.start + 4],
        b"OCHK"
    );
    assert!(
        original[source.continuation.clone()]
            .windows(80)
            .any(|bytes| bytes == [0xa5; 80])
    );
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(memory),
    )
    .unwrap();
    if matches!(owner, Owner::AliasedRoot) {
        assert_eq!(
            file.resolve_address(file.superblock().root_group_address)
                .unwrap()
                .get(),
            Fixture::ROOT
        );
        assert_eq!(file.reusable_free_sections().unwrap().section_count(), 0);
        assert!(
            source
                .protected
                .iter()
                .all(|range| u64::try_from(range.end).unwrap() < file.superblock().eof_address)
        );
    } else {
        file.root()
            .set_attr("a", AttrValue::VarLenAsciiString("temporary".into()))
            .unwrap();
        file.commit().unwrap();
    }
    source.assert_values(&file);
    match edit {
        Edit::Remove => file.root().remove_attr("a").unwrap(),
        Edit::Replace => file.root().set_attr("a", AttrValue::I64(9)).unwrap(),
    }
    file.commit().unwrap();
    assert!(!file.has_staged_edits());
    source.assert_excluded(&file);
    edit.assert_value(&file);
    source.assert_values(&file);
    file.root().set_attr("later", AttrValue::I32(23)).unwrap();
    file.commit().unwrap();
    source.assert_excluded(&file);
    source.assert_values(&file);
    edit.assert_value(&file);
    assert_eq!(
        file.root().attrs().unwrap().get("later"),
        Some(&AttrValue::I32(23))
    );
    file.close().unwrap();
    let after = fs::read(&path).unwrap();
    for range in &source.protected {
        assert_eq!(&after[range.clone()], &original[range.clone()]);
    }
    let file = File::open(&path).unwrap();
    if matches!(owner, Owner::AliasedRoot) {
        assert_ne!(
            file.resolve_address(file.superblock().root_group_address)
                .unwrap()
                .get(),
            Fixture::ROOT,
        );
    }
    source.assert_values(&file);
    edit.assert_value(&file);
    assert_eq!(
        file.root().attrs().unwrap().get("later"),
        Some(&AttrValue::I32(23))
    );
}

#[derive(Clone, Copy, Debug)]
enum Owner {
    Group,
    AliasedRoot,
}

#[derive(Clone, Copy, Debug)]
enum Edit {
    Remove,
    Replace,
}

impl Edit {
    fn assert_value(self, file: &File) {
        let attributes = file.root().attrs().unwrap();
        match self {
            Self::Remove => assert!(!attributes.contains_key("a")),
            Self::Replace => assert_eq!(attributes.get("a"), Some(&AttrValue::I64(9))),
        }
    }
}

struct ContinuedOwner {
    owner: Owner,
    continuation: Range<usize>,
    protected: Vec<Range<usize>>,
}

impl ContinuedOwner {
    fn write(path: &Path, owner: Owner) -> Self {
        let mut fixture = Fixture::new();
        let parent = fixture.collection(&[
            Object::new(7, b"retained"),
            Object {
                index: 91,
                reference_count: 3,
                data: b"keep!",
            },
        ]);
        let heap = usize::try_from(parent).unwrap()..fixture.image.as_bytes().len();
        let data = fixture.image.as_bytes().len();
        let messages = Form::ContiguousDataset.messages(
            &mut fixture,
            &hdf5_pure::make_i32_type(),
            &31_i32.to_le_bytes(),
        );
        let mut protected = vec![heap, data..fixture.image.as_bytes().len()];
        let child = fixture.append_header(messages);
        protected.push(usize::try_from(child).unwrap()..fixture.image.as_bytes().len());
        let messages = Form::CompactDataset.messages(
            &mut fixture,
            &global_heap_fixture::vlen_string(),
            &global_heap_fixture::reference(5, parent, 91),
        );
        let keep = fixture.append_header(messages);
        protected.push(usize::try_from(keep).unwrap()..fixture.image.as_bytes().len());
        let mut messages = Form::CompactAttribute.messages(
            &mut fixture,
            &global_heap_fixture::vlen_string(),
            &global_heap_fixture::reference(8, parent, 7),
        );
        messages.push(global_heap_fixture::hard_link("child", child));
        if matches!(owner, Owner::AliasedRoot) {
            messages.push(global_heap_fixture::hard_link("root_alias", Fixture::ROOT));
            messages.push(global_heap_fixture::hard_link("keep", keep));
        }
        let mut padding = Message::nil(248);
        padding.data.fill(0xa5);
        messages.push(padding);
        let bytes = v2::continuation_chunk(&messages, v2::HeaderFlags::default());
        let start = fixture.image.append(&bytes);
        let continuation = start..start + bytes.len();
        protected.push(continuation.clone());
        let mut header = v2::Header::new().messages(group_info::compact_messages(&[]));
        if matches!(owner, Owner::AliasedRoot) {
            header = header.message(Message::reference_count(2));
        }
        let header = header.continuation(start, &bytes, Widths::EIGHT).build();
        match owner {
            Owner::Group => {
                let address = fixture.image.append(&header);
                protected.push(address..address + header.len());
                fixture.link("retained", u64::try_from(address).unwrap());
                fixture.link("keep", keep);
                fixture.write(path);
            }
            Owner::AliasedRoot => {
                fixture.image.append(&[0; 64]);
                fixture.write(path);
                let mut image = fs::read(path).unwrap();
                let start = usize::try_from(Fixture::ROOT).unwrap();
                image[start..start + header.len()].copy_from_slice(&header);
                protected.push(start..start + header.len());
                fs::write(path, image).unwrap();
            }
        }
        Self {
            owner,
            continuation,
            protected,
        }
    }

    fn assert_values(&self, file: &File) {
        let prefix = match self.owner {
            Owner::Group => "retained",
            Owner::AliasedRoot => "root_alias",
        };
        let group = file.group(prefix).unwrap();
        assert_eq!(
            group.attrs().unwrap().get("a"),
            Some(&AttrValue::VarLenAsciiStringArray(vec!["retained".into()]))
        );
        assert_eq!(
            group.datasets().unwrap(),
            if matches!(self.owner, Owner::AliasedRoot) {
                vec!["child".to_owned(), "keep".to_owned()]
            } else {
                vec!["child".to_owned()]
            }
        );
        assert_eq!(
            file.dataset(&format!("{prefix}/child"))
                .unwrap()
                .read_i32()
                .unwrap(),
            [31]
        );
        assert_eq!(
            file.dataset("keep").unwrap().read_string().unwrap(),
            ["keep!"]
        );
        if matches!(self.owner, Owner::AliasedRoot) {
            assert_eq!(
                file.dataset("root_alias/keep")
                    .unwrap()
                    .read_string()
                    .unwrap(),
                ["keep!"]
            );
        }
    }

    fn assert_excluded(&self, file: &File) {
        let sections = file.reusable_free_sections().unwrap();
        for range in &self.protected {
            assert!(
                sections.sections().iter().all(|section| {
                    section.end_address().get() <= u64::try_from(range.start).unwrap()
                        || u64::try_from(range.end).unwrap() <= section.address().get()
                }),
                "retained allocation {range:?} overlaps {sections:?}"
            );
        }
    }
}
