use std::fs;
use std::ops::Range;
use std::path::Path;

use hdf5_pure::{AttrValue, File, FileAccessProperties, MemoryStrategy};
use rstest::rstest;
use test_util::global_heap::{self, Object};
use test_util::group_info;
use test_util::object_header::{Message, MessageType, v1, v2};
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

const BYTE_LIMIT: usize = 1_048_576;
const GROUP_LIMIT: usize = 64;

#[rstest]
#[case::numeric_target(Residual::Keep)]
#[case::exact_byte_limit(Residual::Bytes(Boundary::Exact))]
#[case::exact_group_limit(Residual::Groups(Boundary::Exact))]
#[case::group_chain(Residual::Links(GroupLinks::Chain))]
#[case::group_cycle(Residual::Links(GroupLinks::Cycle))]
fn qualified_residual_links_allow_the_heap_parent_to_be_released(
    #[case] residual: Residual,
    #[values(KeeperEdit::Unchanged, KeeperEdit::Reconstruct)] keeper: KeeperEdit,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let path = test_util::temp::temp_path("qualified-residual.h5");
    let edit = ResidualEdit::open(&path, residual, memory);
    keeper.stage(&edit.file);
    edit.commit();
    let parent = edit.parent();
    let free = edit.file.reusable_free_sections().unwrap();
    assert!(
        free.sections()
            .iter()
            .any(|section| section.address().get() <= parent.start
                && parent.end <= section.end_address().get()),
        "reclaimed parent {parent:?} is absent from {free:?}"
    );
    let protected = [edit.source.keeper.clone()];
    edit.add_group_and_reopen(&path, &protected, keeper);
}

#[rstest]
#[case::unknown_bytes(Residual::Unknown)]
#[case::unsupported_message(Residual::Unsupported)]
#[case::truncated_header(Residual::Malformed)]
#[case::legacy_header(Residual::Legacy)]
#[case::unresolved_target(Residual::Unresolved)]
#[case::exceeded_byte_limit(Residual::Bytes(Boundary::Exceeded))]
#[case::exceeded_group_limit(Residual::Groups(Boundary::Exceeded))]
#[case::group_interior(Residual::Links(GroupLinks::Interior))]
fn unqualified_residual_links_preserve_the_heap_and_old_keeper(
    #[case] residual: Residual,
    #[values(KeeperEdit::Unchanged, KeeperEdit::Reconstruct)] keeper: KeeperEdit,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let path = test_util::temp::temp_path("unqualified-residual.h5");
    let edit = ResidualEdit::open(&path, residual, memory);
    keeper.stage(&edit.file);
    edit.commit();
    edit.assert_parent_retained();
    let protected = [edit.source.keeper.clone(), bytes_range(edit.parent())];
    edit.add_group_and_reopen(&path, &protected, keeper);
}

#[rstest]
#[case::removed_target(Residual::Gone)]
#[case::cycle_to_removed_owner(Residual::Links(GroupLinks::CycleToRemovedOwner))]
fn a_residual_link_preserves_the_deleted_heap_owner(
    #[case] residual: Residual,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let path = test_util::temp::temp_path("residual-owner.h5");
    let edit = ResidualEdit::open(&path, residual, memory);
    edit.commit();
    edit.assert_parent_retained();
    let protected = [
        edit.source.keeper.clone(),
        edit.source.removed.clone(),
        bytes_range(edit.parent()),
    ];
    edit.add_group_and_reopen(&path, &protected, KeeperEdit::Unchanged);
}

#[rstest]
#[case::removed_target(Residual::Gone)]
#[case::cycle_to_removed_owner(Residual::Links(GroupLinks::CycleToRemovedOwner))]
fn reconstructing_an_unreferenced_keeper_preserves_the_residual_heap_owner(
    #[case] residual: Residual,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let path = test_util::temp::temp_path("residual-owner-replaced-keeper.h5");
    let edit = ResidualEdit::open(&path, residual, memory);
    KeeperEdit::Reconstruct.stage(&edit.file);
    edit.commit();
    edit.assert_parent_retained();
    let protected = [edit.source.removed.clone(), bytes_range(edit.parent())];
    edit.add_group_and_reopen(&path, &protected, KeeperEdit::Reconstruct);
}

#[derive(Clone, Copy, Debug)]
enum KeeperEdit {
    Unchanged,
    Reconstruct,
}

impl KeeperEdit {
    fn stage(self, file: &File) {
        if matches!(self, Self::Reconstruct) {
            file.dataset("keep")
                .unwrap()
                .set_attr("a", AttrValue::I64(9))
                .unwrap();
        }
    }

    fn assert_attributes(self, file: &File) {
        let attributes = file.dataset("keep").unwrap().attrs().unwrap();
        match self {
            Self::Reconstruct => assert_eq!(attributes.get("a"), Some(&AttrValue::I64(9))),
            Self::Unchanged => assert!(attributes.is_empty()),
        }
    }
}

struct ResidualEdit {
    source: ResidualSource,
    before: Vec<u8>,
    file: File,
}

impl ResidualEdit {
    fn open(path: &Path, residual: Residual, memory: MemoryStrategy) -> Self {
        let source = ResidualSource::write(path, residual);
        let before = fs::read(path).unwrap();
        source.assert_layout(&before, residual);
        let file = File::open_rw_with_options(
            path,
            FileAccessProperties::new().with_memory_strategy(memory),
        )
        .unwrap();
        assert_eq!(
            file.dataset("gone").unwrap().read_string().unwrap(),
            ["removed"]
        );
        assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [17]);
        file.root().delete("gone").unwrap();
        Self {
            source,
            before,
            file,
        }
    }

    fn parent(&self) -> Range<u64> {
        self.source.parent
            ..self.source.parent + u64::try_from(global_heap::MIN_COLLECTION_SIZE).unwrap()
    }

    fn commit(&self) {
        self.file.commit().unwrap();
        assert!(!self.file.has_staged_edits());
    }

    fn assert_parent_retained(&self) {
        assert_excluded(&self.file, &bytes_range(self.parent()));
    }

    fn add_group_and_reopen(self, path: &Path, protected: &[Range<usize>], keeper: KeeperEdit) {
        for range in std::iter::once(&self.source.residual).chain(protected) {
            assert_excluded(&self.file, range);
        }
        self.file.root().create_group("added").unwrap();
        self.commit();
        for range in std::iter::once(&self.source.residual).chain(protected) {
            assert_excluded(&self.file, range);
        }
        self.file.close().unwrap();
        let after = fs::read(path).unwrap();
        for range in std::iter::once(&self.source.residual).chain(protected) {
            assert_eq!(&after[range.clone()], &self.before[range.clone()]);
        }
        let file = File::open(path).unwrap();
        assert_eq!(file.root().datasets().unwrap(), ["keep"]);
        assert_eq!(file.root().groups().unwrap(), ["added"]);
        assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [17]);
        keeper.assert_attributes(&file);
    }
}

#[derive(Clone, Copy, Debug)]
enum Boundary {
    Exact,
    Exceeded,
}

impl Boundary {
    fn value(self, limit: usize) -> usize {
        match self {
            Self::Exact => limit,
            Self::Exceeded => limit + 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Residual {
    Keep,
    Gone,
    Unknown,
    Unsupported,
    Malformed,
    Legacy,
    Unresolved,
    Bytes(Boundary),
    Groups(Boundary),
    Links(GroupLinks),
}

#[derive(Clone, Copy, Debug)]
enum GroupLinks {
    Chain,
    Cycle,
    CycleToRemovedOwner,
    Interior,
}

impl GroupLinks {
    fn encode(self, start: usize, owner: u64) -> (Vec<u8>, usize) {
        let group = |links: &[Message]| {
            v2::Header::new()
                .messages(group_info::compact_messages(links))
                .build()
        };
        let first_length = group(&[global_heap_fixture::hard_link("next", 0)]).len();
        let second_address = u64::try_from(start + first_length).unwrap();
        let target = match self {
            Self::Chain | Self::Cycle | Self::CycleToRemovedOwner => second_address,
            Self::Interior => second_address + 1,
        };
        let mut bytes = group(&[global_heap_fixture::hard_link("next", target)]);
        assert_eq!(bytes.len(), first_length);
        let mut links = vec![global_heap_fixture::hard_link("child", owner)];
        if matches!(self, Self::Cycle | Self::CycleToRemovedOwner) {
            links.push(global_heap_fixture::hard_link(
                "back",
                u64::try_from(start).unwrap(),
            ));
        }
        bytes.extend(group(&links));
        (bytes, first_length)
    }
}

struct ResidualSource {
    parent: u64,
    removed: Range<usize>,
    keeper: Range<usize>,
    residual: Range<usize>,
    group_length: usize,
}

impl ResidualSource {
    fn write(path: &Path, residual: Residual) -> Self {
        let mut source = Fixture::new();
        let parent = source.collection(&[Object::new(7, b"removed")]);
        let messages = Form::CompactDataset.messages(
            &mut source,
            &global_heap_fixture::vlen_string(),
            &global_heap_fixture::reference(7, parent, 7),
        );
        let gone = source.append_header(messages);
        let removed = usize::try_from(gone).unwrap()..source.image.as_bytes().len();
        source.link("gone", gone);
        let messages = Form::CompactDataset.messages(
            &mut source,
            &hdf5_pure::make_i32_type(),
            &17_i32.to_le_bytes(),
        );
        let keep = source.append_header(messages);
        let keeper = usize::try_from(keep).unwrap()..source.image.as_bytes().len();
        source.link("keep", keep);
        let target = match residual {
            Residual::Gone | Residual::Links(GroupLinks::CycleToRemovedOwner) => gone,
            Residual::Unresolved => u64::MAX - 1,
            _ => keep,
        };
        let mut messages =
            group_info::compact_messages(&[global_heap_fixture::hard_link("child", target)]);
        let mut group_length = v2::Header::new().messages(messages.clone()).build().len();
        if matches!(residual, Residual::Unsupported) {
            messages.push(Message::new(MessageType(0x7f), &[]));
        }
        let mut bytes = v2::Header::new().messages(messages).build();
        match residual {
            Residual::Unknown => bytes.fill(0x5a),
            Residual::Malformed => bytes = b"OHDR\x02".to_vec(),
            Residual::Legacy => {
                bytes = v1::Header::new()
                    .message(Message::new(MessageType::SYMBOL_TABLE, &[0; 16]))
                    .build()
            }
            Residual::Bytes(boundary) => bytes.resize(boundary.value(BYTE_LIMIT), 0),
            Residual::Groups(boundary) => bytes = bytes.repeat(boundary.value(GROUP_LIMIT)),
            Residual::Links(links) => (bytes, group_length) = links.encode(keeper.end, target),
            _ => {}
        }
        let start = source.image.append(&bytes);
        let range = start..start + bytes.len();
        source.write(path);
        Self {
            parent,
            removed,
            keeper,
            residual: range,
            group_length,
        }
    }

    fn assert_layout(&self, bytes: &[u8], residual: Residual) {
        assert_eq!(self.keeper.end, self.residual.start);
        assert_eq!(self.residual.end, bytes.len());
        let stored = &bytes[self.residual.clone()];
        match residual {
            Residual::Bytes(boundary) => {
                assert_eq!(stored.len(), boundary.value(BYTE_LIMIT));
                assert_eq!(&stored[..4], b"OHDR");
                assert!(stored[self.group_length..].iter().all(|byte| *byte == 0));
            }
            Residual::Groups(boundary) => {
                let mut chunks = stored.chunks_exact(self.group_length);
                assert_eq!(chunks.len(), boundary.value(GROUP_LIMIT));
                assert!(chunks.all(|chunk| chunk == &stored[..self.group_length]));
                assert!(chunks.remainder().is_empty());
            }
            Residual::Links(_) => {
                assert_eq!(&stored[..4], b"OHDR");
                assert_eq!(&stored[self.group_length..self.group_length + 4], b"OHDR");
            }
            _ => {}
        }
    }
}

fn bytes_range(range: Range<u64>) -> Range<usize> {
    usize::try_from(range.start).unwrap()..usize::try_from(range.end).unwrap()
}
fn assert_excluded(file: &File, range: &Range<usize>) {
    let free = file.reusable_free_sections().unwrap();
    assert!(
        free.sections()
            .iter()
            .all(
                |section| section.end_address().get() <= u64::try_from(range.start).unwrap()
                    || u64::try_from(range.end).unwrap() <= section.address().get()
            ),
        "protected residual allocation {range:?} overlaps {free:?}"
    );
}
