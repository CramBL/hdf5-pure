//! Native fixtures and value checks for persistent PAGE editing.

use std::path::Path;

use hdf5::File;
use hdf5::plist::file_create::FileSpaceStrategy;

/// Creates an empty persistent PAGE file through libhdf5.
///
/// # Panics
///
/// Panics if file creation or closure fails.
pub fn write_native(path: &Path, page_size: u64) {
    let file = File::with_options()
        .with_fapl(|properties| properties.libver_v110())
        .with_fcpl(|properties| {
            properties
                .file_space_strategy(strategy())
                .file_space_page_size(page_size)
        })
        .create(path)
        .unwrap();
    file.close().unwrap();
}

/// Checks the edited fixture's dataset values and persistent PAGE settings through libhdf5.
///
/// The fixture has datasets `first`, `second`, and `third` with values `[11, 12, 13, 14]`,
/// `[21, 22, 23]`, and `[31, 32]`, respectively.
///
/// # Panics
///
/// Panics if a value or setting differs, or if opening, reading, or closing the file fails.
pub fn assert_native(path: &Path, page_size: u64) {
    let file = File::open(path).unwrap();
    assert_eq!(
        file.dataset("first").unwrap().read_raw::<i32>().unwrap(),
        [11, 12, 13, 14]
    );
    assert_eq!(
        file.dataset("second").unwrap().read_raw::<i32>().unwrap(),
        [21, 22, 23]
    );
    assert_eq!(
        file.dataset("third").unwrap().read_raw::<i32>().unwrap(),
        [31, 32]
    );
    assert_eq!(
        file.create_plist().unwrap().file_space_page_size(),
        page_size
    );
    assert_eq!(
        file.create_plist().unwrap().file_space_strategy(),
        strategy()
    );
    file.close().unwrap();
}

fn strategy() -> FileSpaceStrategy {
    FileSpaceStrategy::FreeSpaceManager {
        paged: true,
        persist: true,
        threshold: 1,
    }
}
