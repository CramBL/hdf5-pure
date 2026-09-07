//! Dimension helpers used to map between Rust shape vectors, MATLAB workspace
//! shapes, and HDF5 on-disk shapes.
//!
//! HDF5 stores datasets in C/row-major order; MATLAB sees them in
//! Fortran/column-major order. Concretely: a MATLAB matrix of shape
//! `[rows, cols]` is laid out on disk with HDF5 shape `[cols, rows]`. The
//! helpers in this module exist to keep that translation in one place.
//!
//! The hot path (scalars and 1-D vectors) returns `[usize; 2]` directly so
//! leaf writes don't allocate. The HDF5 storage shape conversion takes a
//! caller-provided buffer for the same reason.

use super::options::OneDimensionalMode;

/// Maximum number of dimensions any helper here will write into a stack
/// buffer. MATLAB v7.3 datasets are practically capped well below this.
pub const STORAGE_DIMS_BUF_LEN: usize = 8;

/// Return the MATLAB shape of a 1-D vector of length `len`, given the
/// configured 1-D mode (column or row vector).
///
/// An empty vector is `0x0` under either mode, which is MATLAB's `[]`. The mode
/// orients a vector, and an empty one has no orientation to preserve: what it
/// does have is a value in MATLAB, and that value is the canonical empty. The
/// difference is visible — `[[], 1]` is `1`, where `[zeros(0,1), 1]` is a
/// dimension-mismatch error — and `0x0` is what MATLAB itself overwhelmingly
/// writes. Counted across the MATLAB-authored fixtures in `tests/data/matlab`,
/// `double` empties are `0x0` 47 times against `0x1` once, `char` 55 times, and
/// `cell` 122 times; `matio` and h5py's `hdf5storage` both call `0x0` the
/// canonical empty.
///
/// Stating the rule here rather than at each writer is what keeps the two MAT
/// emitters from disagreeing: they had drifted to `0x0` and `0x1` for the same
/// empty `Vec`, which no test compared because the parity check used no empty
/// sequence.
#[inline]
pub fn vector_dims(len: usize, mode: OneDimensionalMode) -> [usize; 2] {
    if len == 0 {
        return [0, 0];
    }
    match mode {
        OneDimensionalMode::ColumnVector => [len, 1],
        OneDimensionalMode::RowVector => [1, len],
    }
}

/// Return the MATLAB shape for a value with the given multi-dimensional
/// `extents`. Empty extents collapse to `[1, 1]` (scalar); 1-D extents follow
/// [`vector_dims`]; everything else is returned as-is.
///
/// Returns `Vec<usize>` because the >2-D path needs to copy out of `extents`
/// regardless. For the 2-D-or-shorter common case, prefer constructing
/// `[usize; 2]` directly.
#[inline]
pub fn matrix_dims(extents: &[usize], mode: OneDimensionalMode) -> Vec<usize> {
    match extents {
        [] => vec![1, 1],
        [len] => vector_dims(*len, mode).to_vec(),
        _ => extents.to_vec(),
    }
}

/// Convert MATLAB-shape dimensions to HDF5 storage shape (`u64`) into a
/// caller-provided buffer; returns the populated slice. Pads to ≥ 2 dims with
/// 1s and reverses so MATLAB column-major appears row-major on disk.
///
/// Panics if `buf.len()` is smaller than `matlab_dims.len().max(2)`.
#[inline]
pub fn storage_dims_u64_into<'a>(matlab_dims: &[usize], buf: &'a mut [u64]) -> &'a [u64] {
    let n = matlab_dims.len().max(2);
    assert!(
        n <= buf.len(),
        "storage_dims_u64_into: need {n} slots, buf has {}",
        buf.len()
    );
    if matlab_dims.is_empty() {
        buf[0] = 1;
        buf[1] = 1;
    } else if matlab_dims.len() == 1 {
        buf[0] = 1;
        buf[1] = matlab_dims[0] as u64;
    } else {
        for (i, &d) in matlab_dims.iter().enumerate() {
            buf[n - 1 - i] = d as u64;
        }
    }
    &buf[..n]
}

/// Convert MATLAB-shape dimensions to HDF5 storage shape, allocating a
/// `Vec<u64>`. Convenience for callers who don't care about the allocation
/// (e.g., one-shot finish-time emission).
#[inline]
pub fn storage_dims_u64(matlab_dims: &[usize]) -> Vec<u64> {
    let mut buf = [0u64; STORAGE_DIMS_BUF_LEN];
    if matlab_dims.len() > STORAGE_DIMS_BUF_LEN {
        return storage_dims_u64_heap(matlab_dims);
    }
    storage_dims_u64_into(matlab_dims, &mut buf).to_vec()
}

#[cold]
fn storage_dims_u64_heap(matlab_dims: &[usize]) -> Vec<u64> {
    let n = matlab_dims.len().max(2);
    let mut out = vec![0u64; n];
    storage_dims_u64_into(matlab_dims, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_dims_modes() {
        assert_eq!(vector_dims(5, OneDimensionalMode::ColumnVector), [5, 1]);
        assert_eq!(vector_dims(5, OneDimensionalMode::RowVector), [1, 5]);
    }

    #[test]
    fn matrix_dims_handles_special_cases() {
        assert_eq!(
            matrix_dims(&[], OneDimensionalMode::ColumnVector),
            vec![1, 1]
        );
        assert_eq!(
            matrix_dims(&[7], OneDimensionalMode::ColumnVector),
            vec![7, 1]
        );
        assert_eq!(
            matrix_dims(&[3, 4], OneDimensionalMode::ColumnVector),
            vec![3, 4]
        );
    }

    #[test]
    fn storage_dims_u64_pads_and_reverses() {
        let mut buf = [0u64; STORAGE_DIMS_BUF_LEN];
        assert_eq!(storage_dims_u64_into(&[], &mut buf), &[1u64, 1]);
        assert_eq!(storage_dims_u64_into(&[5], &mut buf), &[1u64, 5]);
        assert_eq!(storage_dims_u64_into(&[3, 4], &mut buf), &[4u64, 3]);
        assert_eq!(storage_dims_u64_into(&[2, 3, 4], &mut buf), &[4u64, 3, 2]);
    }

    #[test]
    fn storage_dims_u64_alloc_path_matches() {
        assert_eq!(storage_dims_u64(&[5]), vec![1u64, 5]);
        assert_eq!(storage_dims_u64(&[3, 4]), vec![4u64, 3]);
        assert_eq!(storage_dims_u64(&[2, 3, 4]), vec![4u64, 3, 2]);
    }
}
