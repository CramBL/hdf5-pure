//! Native PAGE fixtures for dense link and attribute edits.

use std::path::Path;

use hdf5::File;
use hdf5::GroupStorageType;
use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::FileSpaceStrategy;
use hdf5::plist::group_create::LinkPhaseChange;

use crate::creation_order::Indexing;
use crate::dense_links;
use crate::file;

/// The attribute sizes and storage transition of a native PAGE fixture.
#[derive(Clone, Copy, Debug)]
pub enum Contents {
    /// An empty result after removing all nine initial links and attributes.
    Empty,
    /// A ten-entry result with 70,000-byte attribute values.
    Huge,
    /// A ten-entry result with 48-byte attribute values.
    Managed,
    /// A ten-entry result with alternating 70,000-byte and 48-byte attribute values.
    Mixed,
    /// A dense result after adding a ninth link and attribute to a compact set of eight.
    Promote,
    /// A compact result after removing four of nine links and attributes.
    Revert,
}

impl Contents {
    /// Returns the number of links and attributes written by [`write_native`].
    pub fn initial(self) -> usize {
        if matches!(self, Self::Promote) { 8 } else { 9 }
    }

    /// Returns the attribute value length at `index`, in bytes.
    ///
    /// Huge values have 70,000 bytes. Managed values have 48 bytes. For [`Self::Mixed`],
    /// returns 70,000 for even indexes and 48 for odd indexes.
    pub fn length(self, index: usize) -> usize {
        match self {
            Self::Huge => 70_000,
            Self::Mixed if index % 2 == 0 => 70_000,
            _ => 48,
        }
    }

    /// Returns the number of leading links and attributes removed by the edit.
    pub fn removed(self) -> usize {
        match self {
            Self::Empty => 9,
            Self::Revert => 4,
            _ => 0,
        }
    }

    /// Returns the exclusive end of the surviving link and attribute indexes.
    ///
    /// The surviving indexes are in the range from [`Self::removed`] to this index.
    pub fn end(self) -> usize {
        match self {
            Self::Empty | Self::Promote | Self::Revert => 9,
            _ => 10,
        }
    }
}

/// Writes a persistent PAGE file with links and attributes in group `g`.
///
/// Creates `c00`, `c01`, and subsequent child groups, and attributes `a00`, `a01`, and
/// subsequent names. Each attribute contains its index repeated for [`Contents::length`]
/// bytes. Sets the page size to `page` bytes, the userblock size to `base` bytes, and
/// tracked creation order with the indexes selected by `indexing`.
///
/// # Panics
///
/// Panics if libhdf5 cannot create, write, or close the fixture.
///
/// # Examples
///
/// ```
/// use hdf5_pure::{AttrValue, File};
/// use test_util_hdf5::creation_order::Indexing;
/// use test_util_hdf5::page_dense::{self, Contents};
///
/// let directory = tempfile::tempdir()?;
/// let path = directory.path().join("dense.h5");
/// page_dense::write_native(&path, 4_096, 0, Contents::Managed, Indexing::No);
/// let file = File::open_rw(&path)?;
/// let group = file.group("g")?;
/// group.create_group("c09")?;
/// group.set_attr("a00", AttrValue::U8Array(vec![99; 48]))?;
/// group.set_attr("a09", AttrValue::U8Array(vec![9; 48]))?;
/// drop(group);
/// file.commit()?;
/// file.close()?;
/// page_dense::assert_edited(&path, 4_096, Contents::Managed, Indexing::No);
/// directory.close()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn write_native(path: &Path, page: u64, base: u64, contents: Contents, indexing: Indexing) {
    let file = File::with_options()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::V110, LibraryVersion::latest())
        })
        .with_fcpl(|properties| {
            properties
                .userblock(base)
                .file_space_page_size(page)
                .file_space_strategy(FileSpaceStrategy::FreeSpaceManager {
                    paged: true,
                    persist: true,
                    threshold: 1,
                })
        })
        .create(path)
        .unwrap();
    let group = file
        .create_group_builder()
        .with_gcpl(|properties| {
            let min = if matches!(contents, Contents::Revert) {
                6
            } else {
                0
            };
            properties
                .link_creation_order(indexing.link_order())
                .attr_creation_order(indexing.attr_order())
                .link_phase_change(LinkPhaseChange {
                    max_compact: 8,
                    min_dense: min,
                })
                .attr_phase_change(8, min)
        })
        .create("g")
        .unwrap();
    for index in 0..contents.initial() {
        group.create_group(&format!("c{index:02}")).unwrap();
        group
            .new_attr::<u8>()
            .shape([contents.length(index)])
            .create(format!("a{index:02}").as_str())
            .unwrap()
            .write_raw(&vec![u8::try_from(index).unwrap(); contents.length(index)])
            .unwrap();
    }
    drop(group);
    file.close().unwrap();
}

/// Checks edited links, attributes, and creation properties through a native PAGE reader.
///
/// Opens with a page buffer of four pages where supported. Checks names and values for indexes in
/// [`Contents::removed`] to [`Contents::end`], and checks link storage and the creation-order
/// properties selected by `contents` and `indexing`. Attribute zero, when present, must
/// contain copies of 99. Other attributes must contain copies of their index.
///
/// # Panics
///
/// Panics if an observation differs, if the page-buffer size or an index cannot be represented,
/// or if libhdf5 cannot open, read, or close the file.
pub fn assert_edited(path: &Path, page: u64, contents: Contents, indexing: Indexing) {
    let file = file::libhdf5_open_page_compatible(path, page);
    let group = file.group("g").unwrap();
    let expected = (contents.removed()..contents.end())
        .map(|index| format!("c{index:02}"))
        .collect::<Vec<_>>();
    assert_eq!(dense_links::creation_order_names(&group), expected);
    let mut names = group.member_names().unwrap();
    names.sort_unstable();
    assert_eq!(names, expected);
    assert_eq!(
        group.info().unwrap().storage_type,
        if matches!(contents, Contents::Revert) {
            GroupStorageType::Compact
        } else {
            GroupStorageType::Dense
        }
    );
    assert_eq!(
        group.gcpl().unwrap().link_creation_order(),
        indexing.link_order()
    );
    assert_eq!(
        group.gcpl().unwrap().attr_creation_order(),
        indexing.attr_order()
    );
    let mut names = group.attr_names().unwrap();
    names.sort_unstable();
    assert_eq!(
        names,
        (contents.removed()..contents.end())
            .map(|index| format!("a{index:02}"))
            .collect::<Vec<_>>()
    );
    for index in contents.removed()..contents.end() {
        let expected = if index == 0 {
            99
        } else {
            u8::try_from(index).unwrap()
        };
        assert_eq!(
            group
                .attr(&format!("a{index:02}"))
                .unwrap()
                .read_raw::<u8>()
                .unwrap(),
            vec![expected; contents.length(index)]
        );
    }
    drop(group);
    file.close().unwrap();
}
