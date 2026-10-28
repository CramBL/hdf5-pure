//! Native observations of private verbatim attributes before and after repacking.

use test_util_hdf5::exotic_attributes;

use super::test_util::ExoticAttributes;
use crate::RepackOptions;

#[test]
fn the_c_library_reads_fresh_exotic_attributes_before_and_after_repack() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("source.h5");
    let dst = dir.path().join("repacked.h5");
    let exotic = ExoticAttributes::new();
    exotic.write(&src);
    exotic_attributes::assert_libhdf5_contents(&src);
    super::repack(&src, &dst, &RepackOptions::new()).unwrap();
    exotic.assert_repacked(&src, &dst);
    exotic_attributes::assert_libhdf5_contents(&dst);
}
