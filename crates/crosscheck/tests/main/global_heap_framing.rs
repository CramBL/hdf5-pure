#![cfg(feature = "hdf5")]

use hdf5::plist::file_create::Sizeof;
use hdf5::types::VarLenUnicode;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::__private::PhysicalAddress;
use hdf5_pure_format::__private::GlobalHeapCollectionLocator;
use hdf5_pure_format::__private::GlobalHeapObject;
use hdf5_pure_format::__private::LengthWidth;
use rstest::rstest;
use test_util::global_heap;
use test_util_hdf5::global_heap as libhdf5_heap;

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
    let mut walked: Vec<String> = global_heap::collection_starts(&bytes)
        .into_iter()
        .flat_map(|start| walk(image, start, lengths))
        .map(|object| String::from_utf8(object.payload().parse(image).unwrap().to_vec()).unwrap())
        .collect();

    walked.sort();
    assert_eq!(walked, libhdf5_heap::strings());
}

#[test]
fn a_walk_frames_each_libhdf5_payload_after_its_object_header() {
    let bytes = libhdf5_file(Sizeof::Bytes8);
    let image = whole_file(&bytes);
    for start in global_heap::collection_starts(&bytes) {
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
    let file = libhdf5_heap::create_libhdf5_file(&path, sizeof_size, 0);
    let values: Vec<VarLenUnicode> = libhdf5_heap::strings()
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    file.new_dataset::<VarLenUnicode>()
        .shape([libhdf5_heap::STRING_COUNT])
        .create("strings")
        .unwrap()
        .write(&values)
        .unwrap();
    file.close().unwrap();
    std::fs::read(&path).unwrap()
}

fn whole_file(bytes: &[u8]) -> FileRegion<'_> {
    FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), bytes.len() as u64).unwrap(),
        bytes,
    )
    .unwrap()
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
