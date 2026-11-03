//! Object copies through `hdf5-pure`, within one file or from another file.

use std::path::Path;
use std::path::PathBuf;

use hdf5_pure::File;
use hdf5_pure::FileBuilder;
#[cfg(feature = "__hdf5-1.10")]
use hdf5_pure::OffsetWidth;

crate::hdf5_1_10_series_tests! {
    /// Returns eight bytes for legacy object-reference storage in HDF5 1.10 fixtures.
    // H5T__init_package and H5T_set_loc in H5T.c, HDF5 1.10.11, keep legacy object references
    // at eight bytes.
    pub fn legacy_reference_storage_size(_offsets: OffsetWidth) -> u64 {
        8
    }
}

crate::hdf5_1_12_tests! {
    /// Returns the file address width in bytes for legacy object-reference storage.
    // H5T__ref_set_loc in H5Tref.c, HDF5 1.12.3, sets legacy object-reference storage
    // to the address width.
    pub fn legacy_reference_storage_size(offsets: OffsetWidth) -> u64 {
        u64::from(offsets.get())
    }
}

/// The file a copy reads its source from.
#[derive(Clone, Copy, Debug)]
pub enum Origin {
    /// Another file, copied from with [`File::copy_from`].
    OtherFile,
    /// The destination file, copied within with [`File::copy`].
    SameFile,
}

impl Origin {
    /// Returns the path of the file a copy writes to, which is `source` for [`Origin::SameFile`].
    ///
    /// For [`Origin::OtherFile`], writes `destination.h5` in `dir` with a dataset `target` whose
    /// one value is 7.
    ///
    /// # Panics
    ///
    /// Panics if the destination file cannot be written.
    pub fn destination(self, dir: &Path, source: &Path) -> PathBuf {
        match self {
            Self::OtherFile => {
                let path = dir.join("destination.h5");
                let mut builder = FileBuilder::new();
                builder.create_dataset("target").with_i32_data(&[7]);
                builder.write(&path).unwrap();
                path
            }
            Self::SameFile => source.to_path_buf(),
        }
    }

    /// Copies `src` to `dst` and commits and closes `destination`.
    ///
    /// # Panics
    ///
    /// Panics if a file cannot be opened, or if the copy, the commit, or the close fails.
    pub fn copy(self, source: &Path, destination: &Path, src: &str, dst: &str) {
        let file = File::open_rw(destination).unwrap();
        match self {
            Self::OtherFile => {
                let source = File::open(source).unwrap();
                file.copy_from(&source, src, dst).unwrap();
            }
            Self::SameFile => file.copy(src, dst).unwrap(),
        }
        file.commit().unwrap();
        file.close().unwrap();
    }
}
