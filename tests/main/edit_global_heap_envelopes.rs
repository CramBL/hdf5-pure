use std::fs;
use std::ops::Range;
use std::path::Path;

use hdf5_pure::{
    AddressRange, EditStorageConflict, Error, File, FileAccessProperties, FileLocking, FormatError,
    MemoryStrategy, PhysicalAddress,
};
use rstest::rstest;
use test_util::global_heap::{self, MIN_COLLECTION_SIZE, Object};
use test_util::object_header::{Message, v2};
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

#[rstest]
#[case::overlapping_collections(Conflict::Overlap)]
#[case::collection_beyond_source(Conflict::BeyondSource)]
fn conflicting_heap_envelopes_reject_edits_without_changing_storage(
    #[case] conflict: Conflict,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let path = test_util::temp::temp_path("conflicting-heap-envelopes.h5");
    let source = Source::write(&path, conflict);
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(memory)
            .with_locking(FileLocking::Disabled),
    )
    .unwrap();
    let before = fs::read(&path).unwrap();
    source.assert_layout(&before, conflict);
    conflict.assert_values(&file);
    let free = file.reusable_free_sections().unwrap();

    for _ in 0..2 {
        file.root().delete("gone").unwrap();
        file.root().create_group("added").unwrap();
        assert!(file.has_staged_edits());
        source.assert_rejection(
            file.commit().unwrap_err(),
            conflict,
            u64::try_from(before.len()).unwrap(),
        );
        assert!(!file.has_staged_edits());
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(file.reusable_free_sections().unwrap(), free);
        conflict.assert_values(&file);

        file.commit().unwrap();
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(file.reusable_free_sections().unwrap(), free);
    }

    file.close().unwrap();
    assert_eq!(fs::read(&path).unwrap(), before);
    let file = File::open(&path).unwrap();
    conflict.assert_values(&file);
}

#[derive(Clone, Copy, Debug)]
enum Conflict {
    Overlap,
    BeyondSource,
}

impl Conflict {
    fn assert_values(self, file: &File) {
        assert!(file.root().groups().unwrap().is_empty());
        assert_eq!(file.root().datasets().unwrap(), ["gone", "keep", "bad"]);
        if matches!(self, Self::Overlap) {
            assert_eq!(
                file.dataset("bad").unwrap().read_string().unwrap(),
                ["retained"]
            );
        }
        assert_eq!(
            file.dataset("gone").unwrap().read_string().unwrap(),
            ["removed"]
        );
        assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [17]);
    }
}

struct Source {
    declared: Range<u64>,
    healthy: Range<u64>,
}

impl Source {
    fn write(path: &Path, conflict: Conflict) -> Self {
        let mut fixture = Fixture::new();
        let size = u64::try_from(MIN_COLLECTION_SIZE).unwrap();
        let first = Object::new(7, b"retained");
        let second = Object::new(7, b"removed");
        let (declared, healthy) = match conflict {
            Conflict::Overlap => {
                let parent = global_heap::collection_of_size(
                    &[first],
                    Widths::EIGHT,
                    2 * MIN_COLLECTION_SIZE,
                );
                let start = fixture.image.append(&parent);
                // The inner collection lies inside the outer collection's free-space record.
                fixture.image.place(
                    start + MIN_COLLECTION_SIZE,
                    &global_heap::collection(&[second], Widths::EIGHT),
                );
                let start = u64::try_from(start).unwrap();
                Self::add_gone_and_keep(&mut fixture, start + size);
                let records = string_records(&mut fixture, start, 8);
                fixture.object("bad", records);
                (start..start + 2 * size, start + size..start + 2 * size)
            }
            Conflict::BeyondSource => {
                let healthy = fixture.collection(&[second]);
                Self::add_gone_and_keep(&mut fixture, healthy);
                let header_length = v2::Header::new()
                    .messages(string_records(&mut fixture, 0, 8))
                    .build()
                    .len();
                let start = u64::try_from(fixture.image.as_bytes().len() + header_length).unwrap();
                let records = string_records(&mut fixture, start, 8);
                fixture.object("bad", records);
                assert_eq!(fixture.collection(&[first]), start);
                let declared_size = 16 * size;
                fixture.image.place(
                    usize::try_from(start).unwrap() + global_heap::SIZE_OFFSET,
                    &declared_size.to_le_bytes(),
                );
                (start..start + declared_size, healthy..healthy + size)
            }
        };
        fixture.write(path);
        Self { declared, healthy }
    }

    fn add_gone_and_keep(fixture: &mut Fixture, healthy: u64) {
        let messages = string_records(fixture, healthy, 7);
        fixture.object("gone", messages);
        let messages = Form::CompactDataset.messages(
            fixture,
            &hdf5_pure::make_i32_type(),
            &17_i32.to_le_bytes(),
        );
        fixture.object("keep", messages);
    }

    fn assert_layout(&self, bytes: &[u8], conflict: Conflict) {
        let length = u64::try_from(bytes.len()).unwrap();
        let ranges = global_heap::collection_ranges(bytes, Widths::EIGHT);
        assert!(self.healthy.end <= length);
        match conflict {
            Conflict::Overlap => {
                assert_eq!(ranges, [self.declared.clone(), self.healthy.clone()]);
                assert!(self.declared.start < self.healthy.start);
                assert_eq!(self.declared.end, self.healthy.end);
            }
            Conflict::BeyondSource => {
                assert_eq!(ranges, [self.healthy.clone(), self.declared.clone()]);
                assert!(self.healthy.end <= self.declared.start);
                assert_eq!(
                    self.declared.start + u64::try_from(MIN_COLLECTION_SIZE).unwrap(),
                    length
                );
                assert!(length < self.declared.end);
            }
        }
    }

    fn assert_rejection(&self, error: Error, conflict: Conflict, source_length: u64) {
        let range = |extent: &Range<u64>| {
            AddressRange::new(
                PhysicalAddress::new(extent.start),
                extent.end - extent.start,
            )
            .unwrap()
        };
        match (conflict, error) {
            (Conflict::Overlap, Error::EditStorageConflict(actual)) => assert_eq!(
                actual,
                EditStorageConflict::Overlap {
                    first: range(&self.declared),
                    second: range(&self.healthy),
                }
            ),
            (Conflict::BeyondSource, Error::Format(actual)) => assert_eq!(
                actual,
                FormatError::UnexpectedSourceEof {
                    expected: self.declared.end,
                    available: source_length,
                }
            ),
            (_, error) => panic!("unexpected commit error: {error:?}"),
        }
    }
}

fn string_records(fixture: &mut Fixture, parent: u64, length: u32) -> Vec<Message> {
    Form::CompactDataset.messages(
        fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(length, parent, 7),
    )
}
