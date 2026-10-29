//! Test helpers that read or write a file through `hdf5-pure`'s own API, and through the
//! reference C libraries behind the `hdf5` and `__matio` features.

pub mod __private;
#[cfg(feature = "hdf5")]
pub mod absence;
pub mod aggregation;
#[cfg(feature = "__hdf5-1.10")]
pub mod allocation_publication;
pub mod attr;
#[cfg(feature = "__hdf5-1.10")]
pub mod chunk_filter;
#[cfg(feature = "__hdf5-1.10")]
pub mod creation_order;
pub mod dataset;
pub mod dataset_writer;
pub mod dense_attr;
pub mod dense_attributes;
#[cfg(feature = "hdf5")]
pub mod dense_links;
#[cfg(feature = "__hdf5-1.10")]
pub mod exotic_attributes;
#[cfg(feature = "hdf5")]
pub mod file;
pub mod file_builder;
pub mod file_space;
#[cfg(feature = "__hdf5-1.10")]
pub mod file_space_oracle;
pub mod fixed_string;
pub mod fixtures;
#[cfg(feature = "__hdf5-bundled")]
pub mod fractal_heap;
mod free_sections;
pub mod free_space_churn;
pub mod free_space_endurance;
pub mod generic_large;
#[cfg(feature = "hdf5")]
pub mod global_heap;
pub mod global_heap_fixture;
#[cfg(feature = "__hdf5-1.10")]
pub mod group_filter;
pub mod heap_allocations;
#[cfg(feature = "hdf5")]
pub mod heap_interop;
pub mod huge_links;
pub mod identity;
pub mod legacy;
pub mod legacy_attributes;
pub mod legacy_chunked;
pub mod lock;
pub mod lzf;
pub mod mat_file;
pub mod mat_struct_array;
pub mod modern;
#[cfg(feature = "__hdf5-1.10")]
pub mod native_fixtures;
#[cfg(feature = "hdf5")]
pub mod numeric;
#[cfg(feature = "__hdf5-1.10")]
pub mod page_publication;
pub mod paged;
pub mod paged_index;
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

/// Includes the supplied items when only the helper's HDF5 1.10 release tier is enabled.
///
/// Items are included when `__hdf5-1.10` is enabled and `__hdf5-1.12` is disabled.
/// The invoking crate can use this macro without defining a release feature of its own.
///
/// # Examples
///
/// ```test_harness
/// test_util_hdf5::hdf5_1_10_series_tests! {
///     #[test]
///     fn linked_library_is_in_the_1_10_series() {
///         let (major, minor, _) = hdf5::library_version();
///         assert_eq!((major, minor), (1, 10));
///     }
/// }
/// ```
#[cfg(all(feature = "__hdf5-1.10", not(feature = "__hdf5-1.12")))]
#[macro_export]
macro_rules! hdf5_1_10_series_tests {
    ($($item:item)*) => { $($item)* };
}

/// Includes the supplied items when only the helper's HDF5 1.10 release tier is enabled.
///
/// Items are included when `__hdf5-1.10` is enabled and `__hdf5-1.12` is disabled.
/// The invoking crate can use this macro without defining a release feature of its own.
///
/// # Examples
///
/// ```test_harness
/// test_util_hdf5::hdf5_1_10_series_tests! {
///     #[test]
///     fn linked_library_is_in_the_1_10_series() {
///         let (major, minor, _) = hdf5::library_version();
///         assert_eq!((major, minor), (1, 10));
///     }
/// }
/// ```
#[cfg(not(all(feature = "__hdf5-1.10", not(feature = "__hdf5-1.12"))))]
#[macro_export]
macro_rules! hdf5_1_10_series_tests {
    ($($item:item)*) => {};
}

/// Includes the supplied items when the helper's HDF5 1.12 capability is enabled.
///
/// The helper's `__hdf5-1.12` feature is enabled for HDF5 1.12 or later. Items are omitted
/// when it is disabled. The invoking crate can use this macro without defining an HDF5
/// release feature of its own.
///
/// # Examples
///
/// ```test_harness
/// test_util_hdf5::hdf5_1_12_tests! {
///     #[test]
///     fn linked_library_is_at_least_1_12() {
///         assert!(hdf5::library_version() >= (1, 12, 0));
///     }
/// }
/// ```
#[cfg(feature = "__hdf5-1.12")]
#[macro_export]
macro_rules! hdf5_1_12_tests {
    ($($item:item)*) => { $($item)* };
}

/// Includes the supplied items when the helper's HDF5 1.12 capability is enabled.
///
/// The helper's `__hdf5-1.12` feature is enabled for HDF5 1.12 or later. Items are omitted
/// when it is disabled. The invoking crate can use this macro without defining an HDF5
/// release feature of its own.
///
/// # Examples
///
/// ```test_harness
/// test_util_hdf5::hdf5_1_12_tests! {
///     #[test]
///     fn linked_library_is_at_least_1_12() {
///         assert!(hdf5::library_version() >= (1, 12, 0));
///     }
/// }
/// ```
#[cfg(not(feature = "__hdf5-1.12"))]
#[macro_export]
macro_rules! hdf5_1_12_tests {
    ($($item:item)*) => {};
}

/// Includes the supplied items when the helper's `__hdf5-bundled` feature is enabled.
///
/// The invoking crate can use this macro without defining a matching feature of its own.
///
/// # Examples
///
/// ```test_harness
/// test_util_hdf5::hdf5_bundled_tests! {
///     #[test]
///     fn bundled_library_has_the_expected_version() {
///         assert_eq!(hdf5::library_version(), (2, 2, 0));
///     }
/// }
/// ```
#[cfg(feature = "__hdf5-bundled")]
#[macro_export]
macro_rules! hdf5_bundled_tests {
    ($($item:item)*) => { $($item)* };
}

/// Includes the supplied items when the helper's `__hdf5-bundled` feature is enabled.
///
/// The invoking crate can use this macro without defining a matching feature of its own.
///
/// # Examples
///
/// ```test_harness
/// test_util_hdf5::hdf5_bundled_tests! {
///     #[test]
///     fn bundled_library_has_the_expected_version() {
///         assert_eq!(hdf5::library_version(), (2, 2, 0));
///     }
/// }
/// ```
#[cfg(not(feature = "__hdf5-bundled"))]
#[macro_export]
macro_rules! hdf5_bundled_tests {
    ($($item:item)*) => {};
}
