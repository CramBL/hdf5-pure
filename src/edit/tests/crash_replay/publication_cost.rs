//! Counts publication, allocation-content, and EOF-padding writes from parsed fixture ranges.

use std::collections::BTreeSet;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private as format;
use test_util::checksum;
use test_util::disk_log::DiskOp;

use crate::File;
use crate::address::AddressContext;
use crate::edit::object_header;
use crate::source::BytesSource;

use super::GATHER_PAGE;
use super::ordering_facts::DatasetGeometry;
use super::ordering_facts::Ownership;

/// The ranges written by an append that retains its index allocations and adds one payload.
pub(super) struct PublicationCost {
    /// The changed object-header suffix, including the dimension and checksum.
    pub(super) dimension: FileRange,
    /// The changed suffixes of the object header, index structures, and superblock.
    publications: Vec<FileRange>,
    /// The persisted manager metadata and the appended payload.
    content: Vec<FileRange>,
    /// The zero-filled tail from appended manager metadata to the EOF page boundary.
    padding: Option<FileRange>,
}

impl PublicationCost {
    /// Derives write roles from the starting and completed fixture metadata.
    ///
    /// # Panics
    ///
    /// Panics if the fixture metadata does not parse, the index allocations change, or the append
    /// adds a different number of payloads than one. Also panics if the dataset has more than one
    /// object-header chunk, a changed structure has an invalid checksum, or appended manager
    /// metadata has a tail that is not zero-filled through the EOF page boundary.
    pub(super) fn read(before: &[u8], after: &[u8]) -> Self {
        let DatasetGeometry {
            index,
            payloads: before_payloads,
            ..
        } = DatasetGeometry::parse(before);
        let DatasetGeometry {
            index: final_index,
            payloads,
            ..
        } = DatasetGeometry::parse(after);
        assert_eq!(index, final_index);
        let file = File::from_bytes(before.to_vec()).unwrap();
        let header = file.dataset("d").unwrap().header_address().unwrap();
        let located = crate::superblock::read(before).unwrap();
        let source = BytesSource::new(before);
        let mut chunks = object_header::read_oh_chunks(
            &source,
            header,
            AddressContext::from_effective_base(located.superblock.base_address),
        )
        .unwrap();
        assert_eq!(chunks.len(), 1);
        let dimension = changed(before, after, chunks.remove(0).span).unwrap();
        let superblock = FileRange::new(
            PhysicalAddress::new(0),
            u64::try_from(
                format::serialize_superblock(&located.superblock)
                    .unwrap()
                    .len(),
            )
            .unwrap(),
        )
        .unwrap();
        let mut publications = vec![dimension];
        publications.extend(
            index
                .into_iter()
                .chain([superblock])
                .filter_map(|range| changed(before, after, range)),
        );
        assert_eq!(publications.len(), 4);
        let mut content = Ownership::parse(after).initialized;
        let end = content.iter().map(|range| range.end().get()).max().unwrap();
        let padding =
            (end > u64::try_from(before.len()).unwrap() && end % GATHER_PAGE != 0).then(|| {
                FileRange::new(
                    PhysicalAddress::new(end),
                    end.next_multiple_of(GATHER_PAGE) - end,
                )
                .unwrap()
            });
        if let Some(padding) = padding {
            assert_eq!(padding.end().get(), u64::try_from(after.len()).unwrap());
            assert!(bytes(after, padding).iter().all(|byte| *byte == 0));
        }
        let payloads: Vec<_> = payloads
            .into_iter()
            .filter(|range| !before_payloads.contains(range))
            .collect();
        assert_eq!(payloads.len(), 1);
        content.extend(payloads);
        Self {
            dimension,
            publications,
            content,
            padding,
        }
    }

    /// Returns whether the dimension publication crosses the first gather-page boundary.
    pub(super) fn crosses_page(&self) -> bool {
        self.dimension.start().get() < GATHER_PAGE && self.dimension.end().get() > GATHER_PAGE
    }

    /// Returns the number of manager metadata ranges and new payload allocations.
    pub(super) fn content_allocations(&self) -> usize {
        self.content.len()
    }

    /// Attributes each operation to publication, allocation content, or EOF padding.
    ///
    /// Several content allocations may share one write. Each required range must be wholly
    /// written by exactly one operation.
    ///
    /// # Errors
    ///
    /// Returns [`WriteCostError::Coverage`] if a required range is not covered by one complete
    /// write, [`WriteCostError::MultipleRoles`] if a write covers both publication and content,
    /// or [`WriteCostError::UnattributedWrite`] for any remaining operation other than the exact
    /// zero-filled EOF tail.
    pub(super) fn count(&self, operations: &[DiskOp]) -> Result<WriteCounts, WriteCostError> {
        let Self {
            dimension: _,
            publications: frames,
            content: allocations,
            padding: tail,
        } = self;
        let publications = complete_writes(frames, operations).map_err(WriteCostError::Coverage)?;
        let content = complete_writes(allocations, operations).map_err(WriteCostError::Coverage)?;
        if let Some(&operation) = publications.intersection(&content).next() {
            return Err(WriteCostError::MultipleRoles { operation });
        }
        if let Some(padding) = tail {
            complete_writes(&[*padding], operations).map_err(WriteCostError::Coverage)?;
        }
        let mut padding = 0;
        for (operation, write) in operations.iter().enumerate() {
            if publications.contains(&operation) || content.contains(&operation) {
                continue;
            }
            let DiskOp::Write { bytes, .. } = write else {
                return Err(WriteCostError::UnattributedWrite { operation });
            };
            if Some(range(write)) != *tail || bytes.iter().any(|byte| *byte != 0) {
                return Err(WriteCostError::UnattributedWrite { operation });
            }
            padding += 1;
        }
        Ok(WriteCounts {
            publications: publications.len(),
            content: content.len(),
            padding,
        })
    }

    /// Splits the dimension publication before its checksum, preserving the completed bytes.
    ///
    /// # Panics
    ///
    /// Panics if the dimension suffix is not covered by one write ending at its checksum.
    pub(super) fn split_dimension(&self, operations: &[DiskOp]) -> Vec<DiskOp> {
        let mut altered = operations.to_vec();
        let write = *complete_writes(&[self.dimension], operations)
            .unwrap()
            .first()
            .unwrap();
        let DiskOp::Write { offset, bytes } = &mut altered[write] else {
            panic!("dimension publication is a write");
        };
        assert_eq!(
            *offset + u64::try_from(bytes.len()).unwrap(),
            self.dimension.end().get()
        );
        let checksum = bytes.split_off(bytes.len() - checksum::CHECKSUM);
        let offset = *offset + u64::try_from(bytes.len()).unwrap();
        altered.insert(
            write + 1,
            DiskOp::Write {
                offset,
                bytes: checksum,
            },
        );
        altered
    }
}

/// The number of distinct writes attributed to each role.
#[derive(Debug)]
pub(super) struct WriteCounts {
    pub(super) publications: usize,
    pub(super) content: usize,
    pub(super) padding: usize,
}

/// A failure to attribute a recorded write to one permitted role.
#[derive(Debug, Eq, PartialEq)]
pub(super) enum WriteCostError {
    /// A required range is not covered by one complete write.
    Coverage(WriteCoverage),
    /// One operation writes both publication and allocation content.
    MultipleRoles { operation: usize },
    /// An operation is outside the parsed ranges or is not the exact zero-filled EOF tail.
    UnattributedWrite { operation: usize },
}

/// A required range and the number of recorded writes that overlap it.
#[derive(Debug, Eq, PartialEq)]
pub(super) struct WriteCoverage {
    pub(super) range: FileRange,
    pub(super) writes: usize,
}

/// Returns the suffix from the first changed byte through the structure's checksum.
fn changed(before: &[u8], after: &[u8], range: FileRange) -> Option<FileRange> {
    let old = bytes(before, range);
    let new = bytes(after, range);
    let first = old.iter().zip(new).position(|(old, new)| old != new)?;
    let end = new.len() - checksum::CHECKSUM;
    assert!(first < end);
    assert_eq!(new[end..], checksum::lookup3(&new[..end]).to_le_bytes());
    Some(
        FileRange::new(
            range
                .start()
                .checked_add(u64::try_from(first).unwrap())
                .unwrap(),
            range.length() - u64::try_from(first).unwrap(),
        )
        .unwrap(),
    )
}

/// Locates the single write that completely covers each required range.
///
/// # Errors
///
/// Returns [`WriteCoverage`] if a range is not covered by exactly one overlapping write.
fn complete_writes(
    ranges: &[FileRange],
    operations: &[DiskOp],
) -> Result<BTreeSet<usize>, WriteCoverage> {
    ranges
        .iter()
        .map(|allocation| {
            let writes: Vec<_> = operations
                .iter()
                .enumerate()
                .filter(|(_, operation)| {
                    matches!(operation, DiskOp::Write { .. })
                        && range(operation).overlaps(*allocation)
                })
                .collect();
            if writes.len() != 1 || !range(writes[0].1).contains(*allocation) {
                return Err(WriteCoverage {
                    range: *allocation,
                    writes: writes.len(),
                });
            }
            Ok(writes[0].0)
        })
        .collect()
}

fn range(operation: &DiskOp) -> FileRange {
    let DiskOp::Write { offset, bytes } = operation else {
        panic!("expected a write, got {operation:?}");
    };
    FileRange::new(
        PhysicalAddress::new(*offset),
        u64::try_from(bytes.len()).unwrap(),
    )
    .unwrap()
}

fn bytes(image: &[u8], range: FileRange) -> &[u8] {
    &image
        [usize::try_from(range.start().get()).unwrap()..usize::try_from(range.end().get()).unwrap()]
}
