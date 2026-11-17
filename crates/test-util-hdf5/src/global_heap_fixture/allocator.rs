//! A fixed numeric source for allocator and placement tests.

use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use test_util::group_info;
use test_util::object_header::v2::Header;

use super::Fixture;
use super::Form;

/// Physical ranges in a fixed contiguous-dataset source.
///
/// The source has one dataset at `value`, with shape `[1]` and value `[3]`. Its payload follows
/// a zero-filled gap. The ranges describe the constructed source before edits.
pub struct AllocatorSource {
    /// The 64-byte interval immediately before [`Self::data`].
    pub gap: FileRange,
    /// The four-byte little-endian payload of `value`.
    pub data: FileRange,
    /// The complete original object header of `value`.
    pub header: FileRange,
    /// The complete original root-group object header.
    pub root: FileRange,
}

impl AllocatorSource {
    /// Writes the fixed source at `path` and returns its physical ranges.
    ///
    /// # Panics
    ///
    /// Panics if the file cannot be written or a fixture address or range cannot be represented.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::File;
    /// use test_util_hdf5::__private::AllocatorSource;
    ///
    /// let dir = tempfile::tempdir()?;
    /// let path = dir.path().join("allocator.h5");
    /// let AllocatorSource { gap, data, header, root } = AllocatorSource::write(&path);
    /// assert_eq!(gap.length(), 64);
    /// assert_eq!(gap.end(), data.start());
    /// assert_eq!(data.end(), header.start());
    /// let file = File::open(&path)?;
    /// assert!(root.end() <= gap.start());
    /// assert_eq!(file.root().datasets()?, ["value"]);
    /// let dataset = file.dataset("value")?;
    /// assert_eq!(dataset.shape()?, [1]);
    /// assert_eq!(dataset.read_i32()?, [3]);
    /// drop(dataset);
    /// file.close()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write(path: &Path) -> Self {
        let mut fixture = Fixture::new();
        let gap = u64::try_from(fixture.image.append(&[0; 64])).unwrap();
        let raw = u64::try_from(fixture.image.as_bytes().len()).unwrap();
        let records = Form::ContiguousDataset.messages(
            &mut fixture,
            &hdf5_pure::make_i32_type(),
            &3_i32.to_le_bytes(),
        );
        let header = Header::new().messages(records).build();
        let owner = u64::try_from(fixture.image.append(&header)).unwrap();
        let root = Header::new()
            .messages(group_info::compact_messages(&[super::hard_link(
                "value", owner,
            )]))
            .build();
        fixture.link("value", owner);
        fixture.write(path);
        Self {
            gap: FileRange::new(PhysicalAddress::new(gap), 64).unwrap(),
            data: FileRange::new(PhysicalAddress::new(raw), 4).unwrap(),
            header: FileRange::new(
                PhysicalAddress::new(owner),
                u64::try_from(header.len()).unwrap(),
            )
            .unwrap(),
            root: FileRange::new(
                PhysicalAddress::new(Fixture::ROOT),
                u64::try_from(root.len()).unwrap(),
            )
            .unwrap(),
        }
    }
}
