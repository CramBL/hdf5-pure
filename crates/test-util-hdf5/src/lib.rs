//! Test helpers that read or write a file through `hdf5-pure`'s own API, and through the
//! reference C libraries behind the `hdf5` and `__matio` features.

#[cfg(feature = "hdf5")]
pub mod absence;
pub mod aggregation;
pub mod attr;
#[cfg(feature = "__hdf5-1.10")]
pub mod chunk_filter;
#[cfg(feature = "__hdf5-1.10")]
pub mod creation_order;
pub mod dataset;
pub mod dataset_writer;
pub mod dense_attr;
#[cfg(feature = "hdf5")]
pub mod dense_links;
#[cfg(feature = "hdf5")]
pub mod file;
pub mod file_builder;
pub mod file_space;
#[cfg(feature = "__hdf5-1.10")]
pub mod file_space_oracle;
pub mod fixed_string;
pub mod fixtures;
mod free_sections;
pub mod free_space_churn;
pub mod free_space_endurance;
#[cfg(feature = "hdf5")]
pub mod global_heap;
pub mod global_heap_fixture;
#[cfg(feature = "__hdf5-1.10")]
pub mod group_filter;
pub mod heap_allocations;
#[cfg(feature = "hdf5")]
pub mod heap_interop;
pub mod lock;
pub mod mat_file;
#[cfg(feature = "hdf5")]
pub mod numeric;
#[cfg(feature = "__hdf5-1.10")]
pub mod page_publication;
pub mod paged;
pub mod reader;
#[cfg(feature = "hdf5")]
pub mod region;
pub mod session;
#[cfg(feature = "hdf5")]
pub mod soft_link;
#[cfg(feature = "__hdf5-1.10")]
pub mod space_ownership;
pub mod storage;
pub mod symbol_table;
pub mod userblock;

/// Includes the supplied items when the helper's HDF5 1.10 capability is enabled.
///
/// The helper's `__hdf5-1.10` feature is enabled for HDF5 1.10 or later. Items are omitted
/// when it is disabled. The invoking crate can use this macro without defining an HDF5
/// release feature of its own.
///
/// # Examples
///
/// ```test_harness
/// test_util_hdf5::hdf5_1_10_tests! {
///     #[test]
///     fn tracked_group_fixture_has_expected_label() {
///         assert_eq!(test_util_hdf5::dense_links::GroupOrder::Tracked.label(), "tracked");
///     }
/// }
/// ```
#[cfg(feature = "__hdf5-1.10")]
#[macro_export]
macro_rules! hdf5_1_10_tests {
    ($($item:item)*) => { $($item)* };
}

/// Includes the supplied items when the helper's HDF5 1.10 capability is enabled.
///
/// The helper's `__hdf5-1.10` feature is enabled for HDF5 1.10 or later. Items are omitted
/// when it is disabled. The invoking crate can use this macro without defining an HDF5
/// release feature of its own.
///
/// # Examples
///
/// ```test_harness
/// test_util_hdf5::hdf5_1_10_tests! {
///     #[test]
///     fn tracked_group_fixture_has_expected_label() {
///         assert_eq!(test_util_hdf5::dense_links::GroupOrder::Tracked.label(), "tracked");
///     }
/// }
/// ```
#[cfg(not(feature = "__hdf5-1.10"))]
#[macro_export]
macro_rules! hdf5_1_10_tests {
    ($($item:item)*) => {};
}
