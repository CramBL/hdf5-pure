//! libhdf5 observations of independent MATLAB struct-array images.

#![cfg(feature = "hdf5")]

use test_util::mat_struct_array;
use test_util::temp;
use test_util_hdf5::mat_struct_array::native;

#[test]
fn libhdf5_reads_the_complete_struct_array_graph() {
    let path = temp::temp_path("struct_array.mat");
    std::fs::write(&path, mat_struct_array::struct_array()).unwrap();
    native::assert_contents(&path);
}

#[test]
fn libhdf5_reads_equal_count_fields_with_different_shapes() {
    let path = temp::temp_path("mismatched_struct_array.mat");
    std::fs::write(&path, mat_struct_array::mismatched_struct_array()).unwrap();
    native::assert_mismatched_contents(&path);
}
