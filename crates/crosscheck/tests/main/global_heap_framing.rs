#![cfg(feature = "hdf5")]
use std::path::Path;

use hdf5::plist::file_create::Sizeof;
use hdf5::plist::file_create::SizeofInfo;
use hdf5::types::VarLenUnicode;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::__private::PhysicalAddress;
use hdf5_pure_format::__private::GlobalHeapCollectionLocator;
use hdf5_pure_format::__private::GlobalHeapObject;
use hdf5_pure_format::__private::LengthWidth;
use rstest::rstest;

#[rstest]
#[case::two_byte_lengths(Sizeof::Bytes2, LengthWidth::Two)]
#[case::four_byte_lengths(Sizeof::Bytes4, LengthWidth::Four)]
#[case::eight_byte_lengths(Sizeof::Bytes8, LengthWidth::Eight)]
fn a_walk_reads_every_string_libhdf5_writes(
    #[case] sizeof_size: Sizeof,
    #[case] lengths: LengthWidth,
) {
    let bytes = libhdf5_file(sizeof_size);
    let image = whole_file(&bytes);
    let mut walked: Vec<String> = collection_starts(&bytes)
        .into_iter()
        .flat_map(|start| walk(image, start, lengths))
        .map(|object| String::from_utf8(object.payload().parse(image).unwrap().to_vec()).unwrap())
        .collect();

    walked.sort();
    assert_eq!(walked, strings());
}

#[test]
fn a_walk_frames_each_libhdf5_payload_after_its_object_header() {
    let bytes = libhdf5_file(Sizeof::Bytes8);
    let image = whole_file(&bytes);
    for start in collection_starts(&bytes) {
        let objects = walk(image, start, LengthWidth::Eight);
        let framed: Vec<(u16, u64, u64)> = objects
            .iter()
            .map(|object| {
                let payload = object.payload().range();
                (
                    object.index().get(),
                    payload.start().get(),
                    payload.length(),
                )
            })
            .collect();
        // A 16-byte collection header, then a 16-byte header and 16 padded bytes per string.
        let expected: Vec<(u16, u64, u64)> = (1..)
            .zip(0..objects.len() as u64)
            .map(|(index, at)| (index, start + 32 + 32 * at, 11))
            .collect();
        assert_eq!(framed, expected);
    }
}

/// Returns a file in which libhdf5 writes [`strings`] to a variable-length string dataset, with
/// lengths `sizeof_size` wide.
fn libhdf5_file(sizeof_size: Sizeof) -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vlen.h5");
    let file = create_libhdf5_file(&path, sizeof_size, 0);
    let values: Vec<VarLenUnicode> = strings().iter().map(|s| s.parse().unwrap()).collect();
    file.new_dataset::<VarLenUnicode>()
        .shape([STRING_COUNT])
        .create("strings")
        .unwrap()
        .write(&values)
        .unwrap();
    file.close().unwrap();
    std::fs::read(&path).unwrap()
}

/// Creates a file at `path` through libhdf5, with 8-byte addresses, lengths `sizeof_size` wide, and
/// a userblock of `userblock` bytes.
pub(crate) fn create_libhdf5_file(path: &Path, sizeof_size: Sizeof, userblock: u64) -> hdf5::File {
    hdf5::File::with_options()
        .with_create_plist(|p| {
            p.sizes(SizeofInfo {
                sizeof_addr: Sizeof::Bytes8,
                sizeof_size,
            })
            .userblock(userblock)
        })
        .create(path)
        .unwrap()
}

/// Returns the [`STRING_COUNT`] strings the tests write, each 11 bytes long.
pub(crate) fn strings() -> Vec<String> {
    (0..STRING_COUNT)
        .map(|i| format!("string {i:04}"))
        .collect()
}

fn whole_file(bytes: &[u8]) -> FileRegion<'_> {
    FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), bytes.len() as u64).unwrap(),
        bytes,
    )
    .unwrap()
}

/// Returns the offset of each `GCOL` signature in `bytes`, the start of a global heap collection.
pub(crate) fn collection_starts(bytes: &[u8]) -> Vec<u64> {
    bytes
        .windows(4)
        .enumerate()
        .filter(|(_, window)| *window == b"GCOL")
        .map(|(start, _)| start as u64)
        .collect()
}

fn walk(image: FileRegion<'_>, start: u64, lengths: LengthWidth) -> Vec<GlobalHeapObject> {
    let locator = GlobalHeapCollectionLocator::new(PhysicalAddress::new(start), lengths);
    let mut walk = locator
        .parse(image.subregion(locator.parse_range().unwrap()).unwrap())
        .unwrap()
        .walk();
    let mut objects = Vec::new();
    while let Some(range) = walk.parse_range() {
        objects.extend(walk.parse(image.subregion(range).unwrap()).unwrap());
    }
    objects
}

pub(crate) const STRING_COUNT: usize = 300;
