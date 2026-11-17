//! A fixed nested dataset with a group continuation.

use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use test_util::group_info;
use test_util::object_header::Message;
use test_util::object_header::v2;
use test_util::object_header::v2::Header;
use test_util::object_header::v2::HeaderFlags;
use test_util::widths::Widths;

use super::Fixture;
use super::Form;

/// Physical ranges in a fixed file with a group continuation.
///
/// The file contains group `retained` and dataset `retained/child`, with shape `[1]` and
/// value `[31]`. The group stores its `child` link in a continuation chunk. The dataset
/// payload, dataset header, continuation, and group header are adjacent in that order.
pub struct RetainedContinuationSource {
    /// The four-byte little-endian payload of `retained/child`.
    pub data: FileRange,
    /// The complete original object header of `retained/child`.
    pub child: FileRange,
    /// The complete continuation chunk containing the `child` link.
    pub continuation: FileRange,
    /// The complete initial object header of `retained`.
    pub group: FileRange,
    /// The complete original root-group object header.
    pub root: FileRange,
}

impl RetainedContinuationSource {
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
    /// use test_util_hdf5::__private::RetainedContinuationSource;
    ///
    /// let dir = tempfile::tempdir()?;
    /// let path = dir.path().join("retained-continuation.h5");
    /// let RetainedContinuationSource { data, child, continuation, group, root } =
    ///     RetainedContinuationSource::write(&path);
    /// assert_eq!(data.length(), 4);
    /// assert_eq!(data.end(), child.start());
    /// assert_eq!(child.end(), continuation.start());
    /// assert_eq!(continuation.end(), group.start());
    /// assert!(root.end() <= data.start());
    /// let file = File::open(&path)?;
    /// assert_eq!(file.root().groups()?, ["retained"]);
    /// assert!(file.root().datasets()?.is_empty());
    /// let retained = file.group("retained")?;
    /// assert!(retained.groups()?.is_empty());
    /// assert_eq!(retained.datasets()?, ["child"]);
    /// let dataset = file.dataset("retained/child")?;
    /// assert_eq!(dataset.shape()?, [1]);
    /// assert_eq!(dataset.read_i32()?, [31]);
    /// drop(dataset);
    /// drop(retained);
    /// file.close()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write(path: &Path) -> Self {
        let mut fixture = Fixture::new();
        let raw = u64::try_from(fixture.image.as_bytes().len()).unwrap();
        let records = Form::ContiguousDataset.messages(
            &mut fixture,
            &hdf5_pure::make_i32_type(),
            &31_i32.to_le_bytes(),
        );
        let child = Header::new().messages(records).build();
        let owner = u64::try_from(fixture.image.append(&child)).unwrap();
        let continuation = v2::continuation_chunk(
            &[super::hard_link("child", owner), Message::nil(80)],
            HeaderFlags::default(),
        );
        let continued = fixture.image.append(&continuation);
        let header = Header::new()
            .messages(group_info::compact_messages(&[]))
            .continuation(continued, &continuation, Widths::EIGHT)
            .build();
        let group = u64::try_from(fixture.image.append(&header)).unwrap();
        let root = Header::new()
            .messages(group_info::compact_messages(&[super::hard_link(
                "retained", group,
            )]))
            .build();
        fixture.link("retained", group);
        fixture.write(path);
        Self {
            data: FileRange::new(PhysicalAddress::new(raw), 4).unwrap(),
            child: FileRange::new(
                PhysicalAddress::new(owner),
                u64::try_from(child.len()).unwrap(),
            )
            .unwrap(),
            continuation: FileRange::new(
                PhysicalAddress::new(u64::try_from(continued).unwrap()),
                u64::try_from(continuation.len()).unwrap(),
            )
            .unwrap(),
            group: FileRange::new(
                PhysicalAddress::new(group),
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
