pub use hdf5_pure_format::__private::Narrow;
pub use hdf5_pure_format::__private::is_undefined_addr;
pub use hdf5_pure_format::__private::slice_range;

#[cfg(test)]
pub(crate) fn nz(value: usize) -> core::num::NonZeroUsize {
    core::num::NonZeroUsize::new(value).expect("a test's element size is non-zero")
}
