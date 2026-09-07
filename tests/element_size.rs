//! Tests for `Dataset::element_size` — the on-disk byte width of one element
//! (HDF5's `H5Tget_size`), which lets a caller bound a read's allocation up
//! front instead of trusting an untrusted file's declared extent.

use hdf5_pure::{File, FileBuilder};
use tempfile::tempdir;

/// `element_size` reports the datatype's storage size, and `shape().product() *
/// element_size()` is exactly the length of the raw buffer a full read returns.
#[test]
fn element_size_matches_datatype_and_read_raw() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("sizes.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("i32")
            .with_i32_data(&(0..12).collect::<Vec<_>>())
            .with_shape(&[12]);
        b.create_dataset("f64")
            .with_f64_data(&[1.0, 2.0, 3.0, 4.0])
            .with_shape(&[4]);
        b.create_dataset("i64")
            .with_i64_data(&[10, 20, 30])
            .with_shape(&[3]);
        b.write(&path).unwrap();
    }

    let file = File::open(&path).unwrap();
    for (name, expected) in [("i32", 4u64), ("f64", 8), ("i64", 8)] {
        let ds = file.dataset(name).unwrap();
        assert_eq!(ds.element_size().unwrap(), expected, "{name} element size");

        let elements: u64 = ds.shape().unwrap().iter().product();
        let raw_len = ds.read_raw().unwrap().len() as u64;
        assert_eq!(
            elements * ds.element_size().unwrap(),
            raw_len,
            "{name}: element_count * element_size equals the raw read length",
        );
    }
}

/// A variable-length string element is stored as a fixed-width heap reference, so
/// `element_size` reports that descriptor width — the payload lives in the global
/// heaps and is not counted here.
#[test]
fn element_size_of_vlen_string_is_the_descriptor_width() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("vl.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("s")
            .with_vlen_strings(&["alpha", "beta", "gamma"])
            .with_shape(&[3]);
        b.write(&path).unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("s").unwrap();
    let size = ds.element_size().unwrap();
    assert!(size > 0, "vlen descriptor has a non-zero width, got {size}");
    // The raw buffer holds one descriptor per element.
    let elements: u64 = ds.shape().unwrap().iter().product();
    assert_eq!(ds.read_raw().unwrap().len() as u64, elements * size);
}

/// `element_size` surfaces the *declared* per-element size even for a malformed
/// file — here the issue #185 crash input, whose fixed-length string datatype
/// declares a ~2.86 GB element. A caller can read this and refuse the dataset
/// before a full read tries to allocate `50 * 2.86 GB`.
#[test]
fn element_size_exposes_a_hostile_declared_size() {
    let bytes =
        std::fs::read("tests/data/fuzz/oom_chunked_string_huge_elem.h5").expect("read fixture");
    let file = File::from_bytes(bytes).unwrap();
    let root = file.root();
    let name = root.datasets().unwrap().into_iter().next().unwrap();
    let ds = root.dataset(&name).unwrap();
    assert_eq!(ds.element_size().unwrap(), 0xAAAA_AAAA);
}
