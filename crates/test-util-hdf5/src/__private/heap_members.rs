//! Native fixtures with variable-length values and caller-held region tokens.

use std::path::Path;

use hdf5::Dataspace;
use hdf5::File;
use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::FileSpaceStrategy as NativeStrategy;
use hdf5::types::VarLenArray;
use hdf5_pure::File as PureFile;
use hdf5_pure_core::__private::FileRange;
use hdf5_sys::h5r;

use crate::file;
use crate::global_heap::attributes;
use crate::storage;

/// The opaque dataset-region token returned by the fixture's native reference creation.
pub struct RegionToken(h5r::hdset_reg_ref_t);

impl RegionToken {
    /// Creates a token for `selection` in the fixture's `data` dataset.
    ///
    /// # Panics
    ///
    /// Panics if `H5Rcreate` fails.
    fn create(native: &hdf5::File, selection: &Dataspace) -> Self {
        let mut token = h5r::hdset_reg_ref_t::default();
        let result = hdf5::sync::sync(|| {
            // SAFETY: The lock serializes native access. The file and selection remain open,
            // the path is terminated, and token is a writable complete legacy reference buffer.
            unsafe {
                h5r::H5Rcreate(
                    token.as_mut_ptr().cast(),
                    native.id(),
                    c"data".as_ptr(),
                    REGION_REFERENCE_TYPE,
                    selection.id(),
                )
            }
        });
        assert!(result >= 0, "H5Rcreate failed");
        Self(token)
    }

    /// Borrows the complete legacy reference buffer returned by `H5Rcreate`.
    pub fn bytes(&self) -> &[u8] {
        &self.0
    }

    /// Checks that the token resolves to shape `[4]` and a two-element selection.
    ///
    /// # Panics
    ///
    /// Panics if the native call fails, the dataspace cannot be opened, or its shape or selection
    /// size differs.
    fn assert_region(&self, native: &hdf5::File) {
        let region = hdf5::sync::sync(|| {
            // SAFETY: The lock serializes native access, the file is open, and the token has
            // the complete legacy dataset-region reference representation returned by H5Rcreate.
            let id = unsafe {
                h5r::H5Rget_region(native.id(), REGION_REFERENCE_TYPE, self.0.as_ptr().cast())
            };
            assert!(id >= 0, "H5Rget_region failed");
            // SAFETY: H5Rget_region returns a newly owned dataspace identifier on success.
            unsafe { hdf5::from_id::<Dataspace>(id) }.unwrap()
        });
        assert_eq!(region.shape(), [4]);
        assert_eq!(region.selection_size(), 2);
    }
}

/// Native region tokens and a sequence descriptor location for a lifetime control.
///
/// Associate the raw sequence descriptor with [`Self::original`] before checking the aliased
/// values.
pub struct MemberLifetime {
    /// Token for the member used in the caller's manual sequence descriptor.
    original: RegionToken,
    /// Token for a different member retained during replacement.
    retained: RegionToken,
    /// File offset of the sequence's contiguous raw descriptor.
    sequence_address: u64,
}

impl MemberLifetime {
    /// Creates a native file and two region tokens for its numeric dataset.
    ///
    /// Writes `data` with values `[7, 11, 13, 17]`, `guard` with value `"retained"`, and
    /// `sequence` with one empty byte-sequence element. Both tokens select elements `1..3` of
    /// `data`. Closes all native handles before returning.
    ///
    /// # Panics
    ///
    /// Panics if a native operation fails or the sequence has no allocated raw storage.
    pub fn write(path: &Path) -> Self {
        let native = file::libhdf5_create_v110(path);
        let data = native
            .new_dataset_builder()
            .with_data(&[7_i32, 11, 13, 17])
            .create("data")
            .unwrap();
        let selection = data.space().unwrap().select(1..3).unwrap();
        let original = RegionToken::create(&native, &selection);
        let retained = RegionToken::create(&native, &selection);
        drop(selection);
        drop(data);
        attributes::write_ascii(&native, "guard", "retained");
        let sequence = native
            .new_dataset::<VarLenArray<u8>>()
            .shape([1])
            .create("sequence")
            .unwrap();
        sequence
            .write(&[VarLenArray::<u8>::from_slice(&[])])
            .unwrap();
        let sequence_address = sequence.offset().unwrap();
        drop(sequence);
        native.flush().unwrap();
        native.close().unwrap();
        Self {
            original,
            retained,
            sequence_address,
        }
    }

    /// Writes one empty element to the fixture's `sequence` dataset.
    ///
    /// Pass the fixture's file after associating its sequence descriptor with [`Self::original`].
    /// Retains the original token buffer through the write and checks its bytes. Closes all
    /// native handles and returns the other region token.
    ///
    /// # Panics
    ///
    /// Panics if a native operation fails or the original token bytes change.
    pub fn replace_with_empty(self, path: &Path) -> RegionToken {
        let original = self.original.0;
        let native = file::libhdf5_open_rw(path);
        let sequence = native.dataset("sequence").unwrap();
        sequence
            .write(&[VarLenArray::<u8>::from_slice(&[])])
            .unwrap();
        drop(sequence);
        native.close().unwrap();
        assert_eq!(self.original.bytes(), original);
        let Self {
            original: _,
            retained,
            sequence_address: _,
        } = self;
        retained
    }

    /// Borrows the token for the member used in the manual sequence descriptor.
    pub fn original(&self) -> &RegionToken {
        &self.original
    }

    /// Borrows the token for the other region member.
    pub fn retained(&self) -> &RegionToken {
        &self.retained
    }

    /// Returns the file offset of the sequence's contiguous raw descriptor.
    pub fn sequence_address(&self) -> u64 {
        self.sequence_address
    }

    /// Checks the sequence payload, both region tokens, and the numeric and guard values.
    ///
    /// Closes all native handles before returning.
    ///
    /// # Panics
    ///
    /// Panics if a native operation fails or a payload, value, shape, or selection size differs.
    pub fn assert_aliased_values(&self, path: &Path, payload: &[u8]) {
        let Self {
            original,
            retained,
            sequence_address: _,
        } = self;
        let native = file::libhdf5_open(path);
        assert_lifetime_values(&native);
        let sequence = native.dataset("sequence").unwrap();
        assert_eq!(
            sequence.read_raw::<VarLenArray<u8>>().unwrap(),
            [VarLenArray::from_slice(payload)]
        );
        drop(sequence);
        original.assert_region(&native);
        retained.assert_region(&native);
        native.close().unwrap();
    }
}

/// Checks the empty sequence, retained token, and unchanged numeric and guard values.
///
/// Closes all native handles before returning.
///
/// # Panics
///
/// Panics if a native operation fails or a value, shape, or selection size differs.
pub fn assert_lifetime_replacement(path: &Path, retained: &RegionToken) {
    let native = file::libhdf5_open(path);
    assert_lifetime_values(&native);
    let sequence = native.dataset("sequence").unwrap();
    assert_eq!(
        sequence.read_raw::<VarLenArray<u8>>().unwrap(),
        [VarLenArray::<u8>::from_slice(&[])]
    );
    drop(sequence);
    retained.assert_region(&native);
    native.close().unwrap();
}

/// Creates a file with two variable-length attributes and an optional dataset-region token.
///
/// Writes root attributes `first` and `second` with values `"left"` and `"right"`, and a dataset
/// `data` with values `[7, 11, 13, 17]`. If `external_token` is true, returns a token for elements
/// `1..3` without writing it into a dataset or attribute. Closes all native handles before returning.
///
/// # Panics
///
/// Panics if a file, dataset, attribute, or reference operation fails.
pub fn write(path: &Path, external_token: bool) -> Option<RegionToken> {
    write_members(file::libhdf5_create_v110(path), external_token)
}

/// Creates the two-attribute fixture with persistent classic free-space managers.
///
/// Sets the threshold to 1 and the lower format bound to [`LibraryVersion::V110`]. Writes the
/// values described by [`write()`] and closes all native handles before returning.
///
/// # Panics
///
/// Panics if a native file, dataset, or attribute operation fails.
pub fn write_persisted(path: &Path) {
    let native = File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .with_fcpl(|p| {
            p.file_space_strategy(NativeStrategy::FreeSpaceManager {
                paged: false,
                persist: true,
                threshold: 1,
            })
        })
        .create(path)
        .unwrap();
    assert!(write_members(native, false).is_none());
}

/// Returns physical ranges for manager storage and superblock-extension chunks.
///
/// Includes complete section-list capacity. A present extension must use version 2 headers.
///
/// # Panics
///
/// Panics if a file operation, metadata parse, or checked range conversion fails.
pub fn persistence_allocations(path: &Path) -> Vec<FileRange> {
    let bytes = std::fs::read(path).unwrap();
    let file = PureFile::open(path).unwrap();
    storage::manager_and_extension_extents(&file, &bytes)
        .into_iter()
        .map(|range| FileRange::new(range.start(), range.length()).unwrap())
        .collect()
}

/// Checks the contents written by [`write()`] and the optional token's dataspace.
///
/// Checks shape `[4]` and selection size 2 when `token` is present. Closes the native file before
/// returning.
///
/// # Panics
///
/// Panics if a native operation fails or a value, shape, or selection size differs.
pub fn assert_values(path: &Path, token: Option<&RegionToken>) {
    let native = file::libhdf5_open(path);
    assert_eq!(
        native.dataset("data").unwrap().read_raw::<i32>().unwrap(),
        [7, 11, 13, 17]
    );
    assert_eq!(
        attributes::ascii_values(&native.attr("first").unwrap()),
        ["left"]
    );
    assert_eq!(
        attributes::ascii_values(&native.attr("second").unwrap()),
        ["right"]
    );
    if let Some(token) = token {
        token.assert_region(&native);
    }
    native.close().unwrap();
}

/// Writes the shared fixture values and closes the native file.
///
/// Returns a caller-held region token when `external_token` is true.
fn write_members(native: File, external_token: bool) -> Option<RegionToken> {
    let dataset = native
        .new_dataset::<i32>()
        .shape([4])
        .create("data")
        .unwrap();
    dataset.write(&[7, 11, 13, 17]).unwrap();
    attributes::write_ascii(&native, "first", "left");
    attributes::write_ascii(&native, "second", "right");
    let token = external_token.then(|| {
        let selection = dataset.space().unwrap().select(1..3).unwrap();
        RegionToken::create(&native, &selection)
    });
    drop(dataset);
    native.flush().unwrap();
    native.close().unwrap();
    token
}

fn assert_lifetime_values(native: &hdf5::File) {
    assert_eq!(
        native.dataset("data").unwrap().read_raw::<i32>().unwrap(),
        [7, 11, 13, 17]
    );
    assert_eq!(
        attributes::ascii_values(&native.attr("guard").unwrap()),
        ["retained"]
    );
}

#[cfg(not(feature = "__hdf5-1.12"))]
const REGION_REFERENCE_TYPE: h5r::H5R_type_t = h5r::H5R_DATASET_REGION;
#[cfg(feature = "__hdf5-1.12")]
const REGION_REFERENCE_TYPE: h5r::H5R_type_t = h5r::H5R_DATASET_REGION1;
