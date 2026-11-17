use std::ops::Range;
use std::path::Path;

use rstest::rstest;
use test_util::datatype;
use test_util::disk_log::{self, DiskOp};
use test_util::object_header::v2::Header;
use test_util::object_header::{Message, MessageFlags, MessageType};
use test_util::superblock;
use test_util_hdf5::global_heap_fixture::{Fixture, Form};

use crate::{File, FileAccessProperties, FileLocking, MemoryStrategy};

mod postpublication;
mod prepublication;

#[rstest]
#[case::increase(2, false, false, 3)]
#[case::decrease(1, true, false, 2)]
#[case::balanced(1, false, false, 2)]
#[case::removed_dataset_user(1, false, true, 2)]
fn public_commits_order_source_counts_around_tree_publication(
    #[case] copies: usize,
    #[case] extra_alias: bool,
    #[case] remove_source: bool,
    #[case] count: u32,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let path = test_util::temp::temp_path("public-count-ordering.h5");
    let original = if extra_alias { 3 } else { 2 };
    let fixture = CountFile::write(&path, original, extra_alias);
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_locking(FileLocking::Disabled)
            .with_memory_strategy(memory)
            .with_page_buffer_size(0),
    )
    .unwrap();
    let before = std::fs::read(&path).unwrap();
    fixture.assert_header(&before, original);
    assert_eq!(file.dataset("source").unwrap().read_f64().unwrap(), [17.0]);
    stage(&file, copies, extra_alias, remove_source);
    disk_log::start();
    file.commit().unwrap();
    let recording = disk_log::finish();
    assert!(!file.has_staged_edits());
    let after = std::fs::read(&path).unwrap();
    fixture.assert_publication(&before, &recording.operations, &after, count);
    let sections = file.reusable_free_sections().unwrap();
    assert!(sections.sections().iter().all(|section| {
        section.end_address().get() <= fixture.target
            || section.address().get() >= fixture.target + type_header(count).len() as u64
    }));
    assert!(sections.sections().iter().any(|section| {
        section.address().get() <= fixture.removed.start
            && section.end_address().get() >= fixture.removed.end
    }));
    assert_values(&file, copies, remove_source);
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_values(&file, copies, remove_source);
    file.close().unwrap();
    fixture.assert_header(&std::fs::read(&path).unwrap(), count);
}

struct CountFile {
    target: u64,
    original: u32,
    removed: Range<u64>,
}

impl CountFile {
    fn write(path: &Path, original: u32, extra_alias: bool) -> Self {
        let mut fixture = Fixture::new();
        let target = fixture.image.append(&type_header(original)) as u64;
        fixture.link("type", target);
        if extra_alias {
            fixture.link("alias", target);
        }
        let mut messages = Form::CompactDataset.messages(
            &mut fixture,
            &crate::make_f64_type(),
            &17_f64.to_le_bytes(),
        );
        *messages
            .iter_mut()
            .find(|message| message.msg_type == MessageType::DATATYPE)
            .unwrap() = Message::new(
            MessageType::DATATYPE,
            &datatype::committed_reference(target),
        )
        .with_flags(MessageFlags::SHARED);
        fixture.object("source", messages);
        let messages = Form::ContiguousDataset.messages(
            &mut fixture,
            &crate::make_f64_type(),
            &17_f64.to_le_bytes(),
        );
        fixture.object("value", messages);
        let removed = fixture.image.as_bytes().len() as u64;
        let messages = Form::ContiguousDataset.messages(
            &mut fixture,
            &crate::make_i32_type(),
            &vec![7_i32.to_le_bytes(); 1_024].concat(),
        );
        fixture.object("dead", messages);
        fixture.write(path);
        Self {
            target,
            original,
            removed: removed..removed + 4_096,
        }
    }

    fn assert_header(&self, bytes: &[u8], count: u32) {
        let start = usize::try_from(self.target).unwrap();
        let expected = type_header(count);
        assert_eq!(&bytes[start..start + expected.len()], expected);
    }

    fn assert_publication(&self, before: &[u8], writes: &[DiskOp], after: &[u8], count: u32) {
        let publication = writes
            .iter()
            .position(|operation| matches!(operation, DiskOp::Write { offset: 0, .. }))
            .unwrap();
        let update = writes.iter().position(|operation| match operation {
            DiskOp::Write { offset, bytes } => {
                self.target
                    .checked_add(12)
                    .and_then(|field| field.checked_sub(*offset))
                    .and_then(|relative| usize::try_from(relative).ok())
                    .and_then(|relative| bytes.get(relative..relative + 4))
                    == Some(count.to_le_bytes().as_slice())
            }
            DiskOp::SetLen(_) => false,
        });
        assert_eq!(update.is_some(), count != self.original);
        if let Some(update) = update {
            assert_eq!(update < publication, count > self.original);
        }
        let mut replay = before.to_vec();
        for (index, operation) in writes.iter().enumerate() {
            if index == publication {
                self.assert_header(
                    &replay,
                    if count > self.original {
                        count
                    } else {
                        self.original
                    },
                );
                assert_eq!(
                    superblock::root_header_address(&replay, 0),
                    superblock::root_header_address(before, 0)
                );
            }
            operation.apply(&mut replay);
        }
        assert_ne!(
            superblock::root_header_address(&replay, 0),
            superblock::root_header_address(before, 0)
        );
        self.assert_header(&replay, count);
        assert_eq!(replay, after);
    }
}

fn stage(file: &File, copies: usize, extra_alias: bool, remove_source: bool) {
    for index in 0..copies {
        file.copy("source", &format!("copy{index}")).unwrap();
    }
    file.root().delete("type").unwrap();
    file.root().delete("dead").unwrap();
    if extra_alias {
        file.root().delete("alias").unwrap();
    }
    if remove_source {
        file.root().delete("source").unwrap();
    } else {
        file.dataset("value")
            .unwrap()
            .write_staged(|builder| {
                builder.with_f64_data(&[31.0]);
            })
            .unwrap();
    }
}

fn assert_values(file: &File, copies: usize, remove_source: bool) {
    let mut names = file.root().datasets().unwrap();
    names.sort();
    let mut expected = (0..copies)
        .map(|index| format!("copy{index}"))
        .collect::<Vec<_>>();
    if !remove_source {
        expected.push("source".into());
    }
    expected.push("value".into());
    assert_eq!(names, expected);
    assert!(file.root().named_datatypes().unwrap().is_empty());
    for index in 0..copies {
        assert_eq!(
            file.dataset(&format!("copy{index}"))
                .unwrap()
                .read_f64()
                .unwrap(),
            [17.0]
        );
    }
    if !remove_source {
        assert_eq!(file.dataset("source").unwrap().read_f64().unwrap(), [17.0]);
    }
    assert_eq!(
        file.dataset("value").unwrap().read_f64().unwrap(),
        [if remove_source { 17.0 } else { 31.0 }]
    );
}

fn type_header(count: u32) -> Vec<u8> {
    Header::new()
        .messages([
            Message::reference_count(count),
            Message::new(MessageType::DATATYPE, &datatype::f64_le()),
        ])
        .build()
}
