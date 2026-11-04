//! Object copies through `hdf5-pure`, within one file or from another file.

use std::path::Path;
use std::path::PathBuf;

#[cfg(feature = "hdf5")]
use hdf5::File as NativeFile;
#[cfg(feature = "hdf5")]
use hdf5::ObjectReference;
#[cfg(feature = "hdf5")]
use hdf5::ObjectReference1;
#[cfg(feature = "hdf5")]
use hdf5::ReferencedObject;
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

/// The storage location of a native legacy object-reference fixture.
#[cfg(feature = "hdf5")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReferenceSite {
    /// A root-group attribute named `reference`.
    Attribute,
    /// A dataset named `references`.
    Dataset,
    /// A root-group attribute with additional attributes for dense-storage fixtures.
    DenseAttribute,
}

/// Writes one native legacy object reference to the object named `target`.
///
/// For [`ReferenceSite::DenseAttribute`], also creates eight scalar numeric attributes on the
/// root group. The caller checks that its creation settings select dense storage.
///
/// # Panics
///
/// Panics if libhdf5 cannot create the reference, its dataset or attribute, or the extra
/// attributes.
#[cfg(feature = "hdf5")]
pub fn write_legacy_reference(file: &NativeFile, site: ReferenceSite) {
    let reference = ObjectReference1::create(file, "target").unwrap();
    match site {
        ReferenceSite::Dataset => {
            file.new_dataset::<ObjectReference1>()
                .shape(1)
                .create("references")
                .unwrap()
                .write_raw(&[reference])
                .unwrap();
        }
        ReferenceSite::Attribute | ReferenceSite::DenseAttribute => {
            file.new_attr::<ObjectReference1>()
                .shape(1)
                .create("reference")
                .unwrap()
                .write_raw(&[reference])
                .unwrap();
            if site == ReferenceSite::DenseAttribute {
                for index in 0..8_u32 {
                    file.new_attr::<u32>()
                        .create(format!("number_{index}").as_str())
                        .unwrap()
                        .write_scalar(&index)
                        .unwrap();
                }
            }
        }
    }
}

/// Checks that a native legacy reference resolves to a dataset containing the one value `expected`.
///
/// # Panics
///
/// Panics if a native read or reference resolution fails, the fixture holds a different number
/// of references, the target is not a dataset, or its complete values differ from `[expected]`.
#[cfg(feature = "hdf5")]
pub fn assert_legacy_reference(path: &Path, site: ReferenceSite, expected: u32) {
    let file = crate::file::libhdf5_open(path);
    let references = match site {
        ReferenceSite::Dataset => file
            .dataset("references")
            .unwrap()
            .read_raw::<ObjectReference1>()
            .unwrap(),
        ReferenceSite::Attribute | ReferenceSite::DenseAttribute => file
            .attr("reference")
            .unwrap()
            .read_raw::<ObjectReference1>()
            .unwrap(),
    };
    assert_eq!(references.len(), 1);
    let ReferencedObject::Dataset(target) = references[0].dereference(&file).unwrap() else {
        panic!("the reference must address a dataset");
    };
    assert_eq!(target.read_raw::<u32>().unwrap(), [expected]);
}
