//! Native fixtures with variable-length attributes and an optional caller-held region token.

use std::path::Path;

use hdf5::Dataspace;
use hdf5_sys::h5r;

use crate::file;
use crate::global_heap::attributes;

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
    let native = file::libhdf5_create_v110(path);
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

#[cfg(not(feature = "__hdf5-1.12"))]
const REGION_REFERENCE_TYPE: h5r::H5R_type_t = h5r::H5R_DATASET_REGION;
#[cfg(feature = "__hdf5-1.12")]
const REGION_REFERENCE_TYPE: h5r::H5R_type_t = h5r::H5R_DATASET_REGION1;
