//! Native sources with a shared anonymous committed datatype.

use std::path::Path;

use hdf5::File;
use hdf5::LocationToken;
use hdf5::plist::file_access::LibraryVersion;

use crate::file;
use crate::numeric::ByteOrder;

/// A native source fixture for shared anonymous committed-datatype counts.
///
/// [`Self::write`] creates `first` and `second`, each containing `[7, 11]` in one shared,
/// anonymous committed little-endian `i32` datatype.
///
/// # Examples
///
/// ```
/// use test_util_hdf5::__private::AnonymousCountSource;
///
/// let directory = tempfile::tempdir()?;
/// let path = directory.path().join("anonymous.h5");
/// AnonymousCountSource::write(&path);
/// let identity = AnonymousCountSource::assert_native(&path, &["first", "second"]);
/// let file = hdf5_pure::File::open_rw(&path)?;
/// file.copy("first", "copy")?;
/// file.commit()?;
/// file.close()?;
/// assert_eq!(
///     AnonymousCountSource::assert_native(&path, &["copy", "first", "second"]),
///     identity,
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct AnonymousCountSource;

impl AnonymousCountSource {
    /// Creates the source at `path` and releases every native handle before returning.
    ///
    /// Sets format bounds from [`LibraryVersion::V18`] through [`LibraryVersion::latest`].
    /// Sets the alignment threshold to one byte and allocation alignment to 64 bytes.
    ///
    /// # Panics
    ///
    /// Panics if creating or writing the fixture, or closing the file, fails.
    pub fn write(path: &Path) {
        let native = File::with_options()
            .with_fapl(|properties| {
                properties
                    .libver_bounds(LibraryVersion::V18, LibraryVersion::latest())
                    .alignment(1, 64)
            })
            .create(path)
            .unwrap();
        let datatype = ByteOrder::LittleEndian.i32_datatype();
        native.commit_datatype("type", &datatype).unwrap();
        for name in ["first", "second"] {
            let data = native
                .new_dataset_builder()
                .empty_as(&datatype)
                .shape([2])
                .create(name)
                .unwrap();
            data.write(&[7_i32, 11]).unwrap();
        }
        native.unlink("type").unwrap();
        drop(datatype);
        native.close().unwrap();
    }

    /// Returns the shared datatype's native identity after checking the datasets.
    ///
    /// `names` must be nonempty and list every root member in sorted order. Each dataset must
    /// contain `[7, 11]`, have no attributes, and use the same committed datatype. Its reported
    /// link count must equal `names.len()`. Closes the native file before returning.
    ///
    /// # Panics
    ///
    /// Panics if `names` is empty, an observation differs from the required shape, or a native
    /// open, read, or close operation fails.
    pub fn assert_native(path: &Path, names: &[&str]) -> LocationToken {
        assert!(!names.is_empty());
        let native = file::libhdf5_open(path);
        let mut actual = native.member_names().unwrap();
        actual.sort();
        assert_eq!(actual, names);
        let mut identity = None;
        for name in names {
            let dataset = native.dataset(name).unwrap();
            assert_eq!(dataset.read_raw::<i32>().unwrap(), [7, 11]);
            assert!(dataset.attr_names().unwrap().is_empty());
            let datatype = dataset.dtype().unwrap();
            assert!(datatype.is_committed());
            let info = datatype.as_location().unwrap().loc_info().unwrap();
            assert_eq!(info.num_links, names.len());
            assert_eq!(info.token, *identity.get_or_insert(info.token));
        }
        native.close().unwrap();
        identity.unwrap()
    }
}
