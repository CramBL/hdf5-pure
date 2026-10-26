//! Checks dense heap fixture values through hdf5-pure and the C library.

use std::path::Path;

use hdf5::plist::file_create::{Sizeof, SizeofInfo};
use hdf5::plist::link_create::CharEncoding;
use hdf5_pure::File;

/// Checks dataset values through buffered, streaming, and C-library reads.
///
/// Each dataset `g/{name}` must contain the index of `name` in `names` as its only value.
///
/// # Panics
///
/// Panics if a file operation fails or a dataset differs from its expected value.
pub fn assert_links_resolve(path: &Path, names: &[String]) {
    let buffered = File::open(path).unwrap();
    let streaming = File::open_streaming(path).unwrap();
    let reference = hdf5::File::open(path).unwrap();
    for (i, name) in names.iter().enumerate() {
        let p = format!("g/{name}");
        assert_eq!(
            reference.dataset(&p).unwrap().read_raw::<i32>().unwrap(),
            vec![i32::try_from(i).unwrap()]
        );
        assert_eq!(
            buffered.dataset(&p).unwrap().read_i32().unwrap(),
            vec![i as i32],
            "buffered link {i}"
        );
        assert_eq!(
            streaming.dataset(&p).unwrap().read_i32().unwrap(),
            vec![i as i32],
            "streaming link {i}"
        );
    }
}

/// Checks an edited fixture through the C library, then checks a C-written dataset.
///
/// The fixture must contain `probe` with 32 values equal to 19. Creates `c_probe` with the value 23.
///
/// # Panics
///
/// Panics if a file operation fails or either dataset differs from its expected values.
pub fn assert_edit_round_trip(path: &Path) {
    let file = hdf5::File::open_rw(path).unwrap();
    assert_eq!(
        file.dataset("probe").unwrap().read_raw::<i32>().unwrap(),
        [19; 32]
    );
    file.new_dataset::<i32>()
        .shape([1])
        .create("c_probe")
        .unwrap()
        .write_raw(&[23])
        .unwrap();
    file.close().unwrap();
    let file = File::open(path).unwrap();
    assert_eq!(file.dataset("c_probe").unwrap().read_i32().unwrap(), [23]);
}

/// Writes a dense group with tiny-object links and returns its sorted member names.
///
/// Creates `g/a` through `g/p` in `path`, with each child's zero-based position in its `value`
/// attribute. The file uses 2-byte addresses and 8-byte lengths.
///
/// # Panics
///
/// Panics if a file operation fails or the member names differ from `a` through `p`.
pub fn write_tiny_link_group(path: &Path) -> Vec<String> {
    let expected: Vec<_> = (b'a'..=b'p')
        .map(|name| char::from(name).to_string())
        .collect();
    let file = hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_latest())
        .with_fcpl(|properties| {
            properties.sizes(SizeofInfo {
                sizeof_addr: Sizeof::Bytes2,
                sizeof_size: Sizeof::Bytes8,
            })
        })
        .create(path)
        .unwrap();
    let group = file.create_group("g").unwrap();
    for (value, name) in expected.iter().enumerate() {
        group
            .create_group_builder()
            .char_encoding(CharEncoding::Ascii)
            .create(name.as_str())
            .unwrap()
            .new_attr::<i32>()
            .shape(())
            .create("value")
            .unwrap()
            .write_scalar(&i32::try_from(value).unwrap())
            .unwrap();
    }
    let mut reference_names = group.member_names().unwrap();
    reference_names.sort();
    assert_eq!(reference_names, expected);
    drop(group);
    file.close().unwrap();
    reference_names
}
