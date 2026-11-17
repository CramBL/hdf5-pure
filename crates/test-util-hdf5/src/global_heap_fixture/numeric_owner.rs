//! A fixed numeric source with adjacent payload and dataset header bytes.

use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use test_util::group_info;
use test_util::object_header::v2::Header;

use super::Fixture;
use super::Form;

/// Physical ranges in a file with one contiguous [`i32`] dataset.
///
/// The dataset is named `value`, with shape `[1]`, value `[19]`, and no attributes.
/// Its object header immediately follows its payload. The ranges describe the original
/// constructed source.
pub struct NumericOwnerSource {
    /// The four-byte little-endian payload of `value`.
    pub data: FileRange,
    /// The complete original object header of `value`.
    pub header: FileRange,
    /// The complete original root-group object header.
    pub root: FileRange,
}

impl NumericOwnerSource {
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
    /// use test_util_hdf5::__private::NumericOwnerSource;
    ///
    /// let dir = tempfile::tempdir()?;
    /// let path = dir.path().join("numeric-owner.h5");
    /// let NumericOwnerSource { data, header, root } = NumericOwnerSource::write(&path);
    /// assert_eq!(data.length(), 4);
    /// assert_eq!(data.end(), header.start());
    /// assert!(root.end() <= data.start());
    /// let file = File::open(&path)?;
    /// assert!(file.root().groups()?.is_empty());
    /// assert_eq!(file.root().datasets()?, ["value"]);
    /// assert!(file.root().attrs()?.is_empty());
    /// let dataset = file.dataset("value")?;
    /// assert_eq!(dataset.shape()?, [1]);
    /// assert_eq!(dataset.read_i32()?, [19]);
    /// assert!(dataset.attrs()?.is_empty());
    /// drop(dataset);
    /// file.close()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write(path: &Path) -> Self {
        let mut fixture = Fixture::new();
        let raw = u64::try_from(fixture.image.as_bytes().len()).unwrap();
        let records = Form::ContiguousDataset.messages(
            &mut fixture,
            &hdf5_pure::make_i32_type(),
            &19_i32.to_le_bytes(),
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
