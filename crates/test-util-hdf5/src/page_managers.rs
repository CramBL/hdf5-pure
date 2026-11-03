//! Native PAGE fixtures and checks for repeated manager publication.

use std::path::Path;

use hdf5::File;
use hdf5::plist::file_create::FileSpaceStrategy;

/// Page-size and userblock configurations for manager-placement checks.
#[derive(Clone, Copy, Debug)]
pub enum Geometry {
    /// An 8,192-byte page with an equally sized userblock.
    Large,
    /// A 768-byte page without a userblock.
    NonPowerOfTwo,
    /// A 512-byte page without a userblock.
    Small,
    /// A 512-byte page with an 8,192-byte userblock.
    Userblock,
    /// A 1,024-byte page with an 8,192-byte userblock.
    UserblockAlignedPage,
}

impl Geometry {
    /// Returns the page length in bytes used by the native fixture.
    pub fn page_size(self) -> u64 {
        match self {
            Self::Large => 8_192,
            Self::NonPowerOfTwo => 768,
            Self::Small | Self::Userblock => 512,
            Self::UserblockAlignedPage => 1_024,
        }
    }

    fn userblock(self) -> u64 {
        match self {
            Self::Large | Self::Userblock | Self::UserblockAlignedPage => 8_192,
            Self::NonPowerOfTwo | Self::Small => 0,
        }
    }
}

/// Creates six fixed-size datasets with persisted PAGE managers and an HDF5 1.10 lower bound.
///
/// Closes the file before returning it to the editor.
///
/// # Panics
///
/// Panics if a file operation fails.
pub fn write_native(path: &Path, geometry: Geometry) {
    let page = geometry.page_size();
    let base = geometry.userblock();
    let file = File::with_options()
        .with_fapl(|properties| properties.libver_v110())
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
    for index in 0_u8..6 {
        file.new_dataset::<u8>()
            .shape([1_000])
            .create(format!("d{index}").as_str())
            .unwrap()
            .write_raw(&vec![index; 1_000])
            .unwrap();
    }
    file.close().unwrap();
}

/// Checks surviving dataset values and the final generation through a native page buffer.
///
/// The caller deletes `d2` and writes the root attribute `generation` with the value 3.
/// Requires positive free space and closes the native file before returning.
///
/// # Panics
///
/// Panics if a file operation fails or an expected value, member name, or free-space check fails.
pub fn assert_native(path: &Path, geometry: Geometry) {
    let page = geometry.page_size();
    let file = File::with_options()
        .with_fapl(|properties| {
            properties.page_buffer_size(usize::try_from(4 * page).unwrap(), 0, 0)
        })
        .open(path)
        .unwrap();
    assert_eq!(file.member_names().unwrap(), ["d0", "d1", "d3", "d4", "d5"]);
    for index in [0_u8, 1, 3, 4, 5] {
        assert_eq!(
            file.dataset(&format!("d{index}"))
                .unwrap()
                .read_raw::<u8>()
                .unwrap(),
            vec![index; 1_000]
        );
    }
    assert_eq!(
        file.attr("generation")
            .unwrap()
            .read_scalar::<u32>()
            .unwrap(),
        3
    );
    assert!(file.free_space() > 0);
    file.close().unwrap();
}
