//! Builds shared-parent fixtures and checks complete collection bytes after public edits.

use std::path::Path;

use hdf5_pure::{
    EditBacking, File, FileAccessProperties, FileLocking, PersistedFreeSpacePolicy, SyncPolicy,
};
use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::GlobalHeapCollectionLocator;
use hdf5_pure_format::LengthWidth;
use test_util::attribute::Attribute;
use test_util::dataspace;
use test_util::global_heap::{self, Object};
use test_util::object_header::{Message, MessageType};
use test_util::widths::Widths;

use super::{Fixture, Form};
use crate::global_heap_fixture;

/// Stores original parent images and expected values for grouped attribute removals.
///
/// Each parent holds two removable members and two retained members. Dataset names use the
/// parent index: `value_0` holds the removable attributes, and `keep_0` refers to retained members.
///
/// # Examples
///
/// ```
/// use hdf5_pure::{EditBacking, File};
/// use test_util_hdf5::global_heap_fixture::members::{Buffering, LayoutFixture, Terminal};
///
/// # let directory = tempfile::tempdir()?;
/// let path = directory.path().join("members.h5");
/// let fixture = LayoutFixture::write(&path, Terminal::Implied, 2);
/// let file = File::open_rw_with_options(
///     &path,
///     Buffering::Operation.properties(EditBacking::Bounded),
/// )?;
/// fixture.remove(&file);
/// file.commit()?;
/// file.close()?;
/// fixture.assert_compacted(&path);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct LayoutFixture {
    /// The complete original collections in parent-index order.
    pub parents: Vec<ParentImage>,
    /// The expected `keep_0`, `keep_1`, and subsequent dataset values, in parent-index order.
    pub values: Vec<Vec<String>>,
}

impl LayoutFixture {
    /// Writes `count` independent parents with the selected original tail and nonzero record
    /// padding.
    ///
    /// # Panics
    ///
    /// Panics if fixture construction, writing, or inspection fails.
    pub fn write(path: &Path, terminal: Terminal, count: usize) -> Self {
        let mut fixture = Fixture::new();
        let payload_size = global_heap::MIN_COLLECTION_SIZE
            - 5 * global_heap::HEADER_SIZE
            - 3 * global_heap::ALIGNMENT
            - terminal.bytes();
        let mut values = Vec::new();
        let mut parents = Vec::new();
        for index in 0..count {
            let payload = vec![b'q' + u8::try_from(index % 10).unwrap(); payload_size];
            let tail = [
                b't',
                b'a',
                b'i',
                b'l',
                b'0' + u8::try_from(index % 10).unwrap(),
            ];
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
                    data: &tail,
                },
            ];
            values.push(vec![
                String::from_utf8(payload.clone()).unwrap(),
                String::from_utf8(tail.to_vec()).unwrap(),
            ]);

            let bytes = global_heap::collection_of_size_with_padding(
                &objects,
                Widths::EIGHT,
                global_heap::MIN_COLLECTION_SIZE,
                RECORD_PADDING,
            );
            let parent = u64::try_from(fixture.image.append(&bytes)).unwrap();
            parents.push(parent);
            let mut records = Form::CompactDataset.messages(
                &mut fixture,
                &hdf5_pure::make_i32_type(),
                &17_i32.to_le_bytes(),
            );
            for (name, member, length) in [("first", 400, 5), ("second", 7, 4)] {
                let value = Attribute::new(
                    name,
                    &format::serialize_datatype(&global_heap_fixture::vlen_string()).unwrap(),
                    &dataspace::scalar(),
                    &global_heap_fixture::reference(length, parent, member),
                )
                .build();
                records.push(Message::new(MessageType::ATTRIBUTE, &value));
            }
            fixture.object(&format!("value_{index}"), records);
            let descriptors = [
                global_heap_fixture::reference(
                    u32::try_from(payload.len()).unwrap(),
                    parent,
                    u32::from(u16::MAX),
                ),
                global_heap_fixture::reference(5, parent, 91),
            ]
            .concat();
            let records = Form::ContiguousDataset.messages(
                &mut fixture,
                &global_heap_fixture::vlen_string(),
                &descriptors,
            );
            fixture.object(&format!("keep_{index}"), records);
        }
        fixture.write(path);
        let bytes = std::fs::read(path).unwrap();
        let parents = parents
            .into_iter()
            .map(|address| {
                let parent = ParentImage::read(&bytes, address);
                assert_eq!(parent.indices(), [400, u16::MAX, 7, 91]);
                assert_eq!(parent.tail_bytes(), terminal.bytes());
                assert_eq!(parent.records[3].bytes.last(), Some(&RECORD_PADDING));
                parent
            })
            .collect();
        Self { parents, values }
    }

    /// Stages removal of both attributes from each `value_0`, `value_1`, and subsequent dataset.
    ///
    /// The caller commits and closes `file` before inspecting the completed file.
    ///
    /// # Panics
    ///
    /// Panics if a dataset cannot be opened or an attribute removal cannot be staged.
    pub fn remove(&self, file: &File) {
        for index in 0..self.parents.len() {
            let mut dataset = file.dataset(&format!("value_{index}")).unwrap();
            dataset.remove_attr("first").unwrap();
            dataset.remove_attr("second").unwrap();
        }
    }

    /// Checks exact parent packing, retained values, numeric datasets, and removed attributes.
    ///
    /// # Panics
    ///
    /// Panics if reading or reopening fails, or any bytes or public values differ from
    /// expectations.
    pub fn assert_compacted(&self, path: &Path) {
        let Self { parents, values } = self;
        let bytes = std::fs::read(path).unwrap();
        for parent in parents {
            parent.assert_compacted(&bytes, &[u16::MAX, 91]);
        }
        let file = File::open(path).unwrap();
        assert_eq!(parents.len(), values.len());
        for (index, expected) in values.iter().enumerate() {
            assert_eq!(
                file.dataset(&format!("keep_{index}"))
                    .unwrap()
                    .read_string()
                    .unwrap(),
                *expected
            );
            assert_eq!(
                file.dataset(&format!("value_{index}"))
                    .unwrap()
                    .read_i32()
                    .unwrap(),
                [17]
            );
            assert!(
                file.dataset(&format!("value_{index}"))
                    .unwrap()
                    .attrs()
                    .unwrap()
                    .is_empty()
            );
        }
        file.close().unwrap();
    }
}

/// Stores a complete collection and its parsed member records before editing.
pub struct ParentImage {
    range: FileRange,
    header: usize,
    records: Vec<Record>,
    bytes: Vec<u8>,
}

impl ParentImage {
    /// Reads the complete collection at the physical `address`, using eight-byte lengths.
    ///
    /// # Panics
    ///
    /// Panics if the address or collection is out of bounds, a record is invalid, or an address
    /// cannot be represented as an index into `image`.
    pub fn read(image: &[u8], address: u64) -> Self {
        let file = FileRegion::new(
            FileRange::new(PhysicalAddress::new(0), u64::try_from(image.len()).unwrap()).unwrap(),
            image,
        )
        .unwrap();
        let locator =
            GlobalHeapCollectionLocator::new(PhysicalAddress::new(address), LengthWidth::Eight);
        let header = usize::try_from(locator.parse_range().unwrap().length()).unwrap();
        let collection = locator
            .parse(file.subregion(locator.parse_range().unwrap()).unwrap())
            .unwrap();
        let range = collection.range();
        let mut walk = collection.walk();
        let mut records = Vec::new();
        while let Some(next) = walk.parse_range() {
            if let Some(object) = walk.parse(file.subregion(next).unwrap()).unwrap() {
                records.push(Record {
                    index: object.index().get(),
                    count: object.reference_count(),
                    payload: object.payload().parse(file).unwrap().to_vec(),
                    bytes: file.subregion(object.range()).unwrap().bytes().to_vec(),
                });
            }
        }
        Self {
            range,
            header,
            records,
            bytes: file.subregion(range).unwrap().bytes().to_vec(),
        }
    }

    /// Returns allocated member indices in their original physical order.
    pub fn indices(&self) -> Vec<u16> {
        self.records.iter().map(|record| record.index).collect()
    }

    /// Returns the index of the unique member with payload bytes equal to `payload`.
    ///
    /// # Panics
    ///
    /// Panics if no member or more than one member has that payload.
    pub fn index_for(&self, payload: &[u8]) -> u16 {
        let found = self
            .records
            .iter()
            .filter(|record| record.payload == payload)
            .collect::<Vec<_>>();
        let [record] = found.as_slice() else {
            panic!("expected exactly one member for payload");
        };
        record.index
    }

    /// Counts the collection bytes after the last allocated record, including any free header.
    pub fn tail_bytes(&self) -> usize {
        let Self {
            range: _,
            header,
            records,
            bytes,
        } = self;
        bytes.len()
            - header
            - records
                .iter()
                .map(|record| record.bytes.len())
                .sum::<usize>()
    }

    /// Compares the complete original collection with the same range in `image`.
    ///
    /// # Panics
    ///
    /// Panics if the range is unavailable or any byte differs.
    pub fn assert_unchanged(&self, image: &[u8]) {
        let start = usize::try_from(self.range.start().get()).unwrap();
        let end = usize::try_from(self.range.end().get()).unwrap();
        assert_eq!(&image[start..end], self.bytes);
    }

    /// Checks packed retained records, a free header, and unchanged remaining collection bytes.
    ///
    /// Records are copied from the original image in physical order. The check includes the
    /// collection address and capacity, stored counts, payloads, and record padding. Each index
    /// in `retained` must identify one original member and appear once. The retained records must
    /// leave room for a complete free-space object header.
    ///
    /// # Panics
    ///
    /// Panics if retained indices are absent or repeated, the remaining space is shorter than a
    /// free header, parsing fails, or the result differs.
    pub fn assert_compacted(&self, image: &[u8], retained: &[u16]) {
        let Self {
            range,
            header,
            records,
            bytes,
        } = self;
        let kept = records
            .iter()
            .filter(|record| retained.contains(&record.index))
            .collect::<Vec<_>>();
        assert_eq!(kept.len(), retained.len());
        let mut expected = bytes.clone();
        let mut cursor = *header;
        for record in &kept {
            let end = cursor.checked_add(record.bytes.len()).unwrap();
            expected[cursor..end].copy_from_slice(&record.bytes);
            cursor = end;
        }
        let free = global_heap::free_space_header(
            u64::try_from(bytes.len() - cursor).unwrap(),
            Widths::EIGHT,
        );
        expected[cursor..cursor + free.len()].copy_from_slice(&free);
        let after = Self::read(image, range.start().get());
        assert_eq!(after.range, *range);
        assert_eq!(after.bytes, expected);
        assert_eq!(
            after.indices(),
            kept.iter().map(|record| record.index).collect::<Vec<_>>()
        );
        assert_eq!(
            after.records.iter().map(Record::values).collect::<Vec<_>>(),
            kept.iter()
                .map(|record| record.values())
                .collect::<Vec<_>>()
        );
        assert_eq!(after.tail_bytes(), bytes.len() - cursor);
    }
}

struct Record {
    index: u16,
    count: u16,
    payload: Vec<u8>,
    bytes: Vec<u8>,
}

impl Record {
    /// Returns the payload bytes followed by the stored reference count.
    fn values(&self) -> (&[u8], u16) {
        let Self {
            index: _,
            count,
            payload,
            bytes: _,
        } = self;
        (payload, *count)
    }
}

/// Selects the original bytes after the last allocated member.
#[derive(Clone, Copy, Debug)]
pub enum Terminal {
    /// Reserves exactly one free-space object header.
    Explicit,
    /// Fills the collection with allocated records.
    Full,
    /// Leaves eight zero bytes, shorter than a free-space object header.
    Implied,
}

impl Terminal {
    /// Returns the number of bytes reserved after the allocated records.
    pub fn bytes(self) -> usize {
        match self {
            Self::Explicit => global_heap::HEADER_SIZE,
            Self::Full => 0,
            Self::Implied => global_heap::ALIGNMENT,
        }
    }
}

/// Selects locking and page-buffer settings for public edit sessions.
#[derive(Clone, Copy, Debug)]
pub enum Buffering {
    /// Enables locking with the page buffer disabled.
    Operation,
    /// Enables locking with a 65,536-byte page buffer.
    Session,
    /// Disables locking and the page buffer.
    Unbuffered,
}

impl Buffering {
    /// Returns access properties with the selected backing and buffering settings.
    ///
    /// Uses synchronization on close and strict persisted free-space handling.
    pub fn properties(self, backing: EditBacking) -> FileAccessProperties {
        FileAccessProperties::new()
            .with_memory_strategy(backing.into())
            .with_sync_policy(SyncPolicy::OnClose)
            .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict)
            .with_locking(if matches!(self, Self::Unbuffered) {
                FileLocking::Disabled
            } else {
                FileLocking::Enabled
            })
            .with_page_buffer_size(if matches!(self, Self::Session) {
                65_536
            } else {
                0
            })
    }
}

const RECORD_PADDING: u8 = 0xa5;
