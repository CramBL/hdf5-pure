//! Native fixtures with persistent free sections and surviving datasets.

use std::ops::Range;
use std::path::Path;

use hdf5::File as NativeFile;
use hdf5::plist::file_create::FileSpaceStrategy as NativeStrategy;

use crate::file;
use crate::file_space;

/// Creates a file with retired scratch datasets and returns the native free-section ranges.
///
/// Enables PAGE allocation when `paged` is `true`. Each surviving `spacer` dataset holds bytes
/// equal to 31 and has the length returned by [`HoleLayout::spacer_size`]. The returned ranges
/// include all sections reported after closing and reopening the file, with complete coverage
/// of each retired raw allocation.
///
/// # Panics
///
/// Panics if file creation, dataset I/O, deletion, reopening, or free-section inspection fails,
/// or a retired raw allocation is absent from the reported sections.
pub fn create(path: &Path, paged: bool, layout: HoleLayout) -> Vec<Range<u64>> {
    let native = NativeFile::with_options()
        .with_fapl(|properties| properties.libver_v110())
        .with_fcpl(|properties| {
            properties.file_space_strategy(NativeStrategy::FreeSpaceManager {
                paged,
                persist: true,
                threshold: 1,
            })
        })
        .create(path)
        .unwrap();
    let mut holes = Vec::new();
    for index in 0..if matches!(layout, HoleLayout::Contiguous) {
        1
    } else {
        16
    } {
        let scratch = native
            .new_dataset::<u8>()
            .shape([layout.extent_size()])
            .create(format!("scratch{index}").as_str())
            .unwrap();
        scratch.write_raw(&vec![13; layout.extent_size()]).unwrap();
        let start = scratch.offset().unwrap();
        holes.push(start..start + scratch.storage_size());
        drop(scratch);
        let spacer = native
            .new_dataset::<u8>()
            .shape([layout.spacer_size()])
            .create(format!("spacer{index}").as_str())
            .unwrap();
        spacer
            .write_raw(&vec![31_u8; layout.spacer_size()])
            .unwrap();
        drop(spacer);
    }
    for index in 0..holes.len() {
        native.unlink(&format!("scratch{index}")).unwrap();
    }
    native.close().unwrap();
    let native = file::libhdf5_open(path);
    let sections = file_space::libhdf5_free_sections(&native).unwrap();
    native.close().unwrap();
    let sections = sections
        .into_iter()
        .map(|(start, length)| start..start + length)
        .collect::<Vec<_>>();
    for hole in holes {
        assert!(
            sections
                .iter()
                .any(|section| section.start <= hole.start && hole.end <= section.end)
        );
    }
    sections
}

/// The scratch allocations and surviving spacers written by [`create`].
#[derive(Clone, Copy)]
pub enum HoleLayout {
    /// One 32 KiB scratch allocation followed by a 1 KiB spacer.
    Contiguous,
    /// Sixteen 1 KiB scratch allocations separated by 1 KiB spacers.
    Fragmented,
    /// Sixteen 8 KiB scratch allocations separated by 4 KiB spacers.
    WholePages,
}

impl HoleLayout {
    /// Returns the byte length of each surviving spacer dataset.
    pub fn spacer_size(self) -> usize {
        if matches!(self, Self::WholePages) {
            4_096
        } else {
            1_024
        }
    }

    fn extent_size(self) -> usize {
        match self {
            Self::Contiguous => 32_768,
            Self::Fragmented => 1_024,
            Self::WholePages => 8_192,
        }
    }
}
