#![cfg(feature = "__hdf5-1.10")]

use hdf5::Extents;

use hdf5_pure::AttrValue;
use hdf5_pure::File;
use test_util_hdf5::file;

#[test]
fn a_null_dataspace_attribute_in_a_padded_header_record_reads_back_holding_nothing() {
    // The C library writes version 1 headers with 8-byte record padding under `libver_earliest`.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("null-earliest.h5");
    {
        let file = file::libhdf5_create_earliest(&path);
        file.new_attr::<i32>()
            .shape(Extents::Null)
            .create("empty")
            .unwrap();
        file.new_attr::<u8>()
            .shape(Extents::Null)
            .create("empty_u8")
            .unwrap();
        file.close().unwrap();
    }

    let attrs = File::open(&path).unwrap().root().attrs().unwrap();
    assert_eq!(attrs.get("empty"), Some(&AttrValue::I32Array(Vec::new())));
    assert_eq!(attrs.get("empty_u8"), Some(&AttrValue::U8Array(Vec::new())));
}
