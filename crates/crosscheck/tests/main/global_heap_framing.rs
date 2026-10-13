#![cfg(feature = "hdf5")]
use hdf5::plist::file_create::Sizeof;
use hdf5::plist::file_create::SizeofInfo;
use hdf5::types::VarLenUnicode;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::__private::PhysicalAddress;
use hdf5_pure_format::__private::GlobalHeapCollectionLocator;
use hdf5_pure_format::__private::GlobalHeapIndex;
use hdf5_pure_format::__private::GlobalHeapObject;
use hdf5_pure_format::__private::GlobalHeapObjectInfo;
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
fn a_walk_agrees_with_the_directory_parse_on_a_libhdf5_collection() {
    let bytes = libhdf5_file(Sizeof::Bytes8);
    let image = whole_file(&bytes);
    for start in collection_starts(&bytes) {
        let walked: Vec<GlobalHeapObjectInfo> = walk(image, start, LengthWidth::Eight)
            .iter()
            .map(|object| GlobalHeapObjectInfo {
                index: object.index().get(),
                data_address: object.payload().range().start().get(),
                size: object.payload().range().length(),
            })
            .collect();
        assert_eq!(
            walked,
            GlobalHeapIndex::parse(bytes.as_slice(), start, 8)
                .unwrap()
                .objects
        );
    }
}

/// Returns a file in which libhdf5 writes [`strings`] to a variable-length string dataset, with
/// lengths `sizeof_size` wide.
fn libhdf5_file(sizeof_size: Sizeof) -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vlen.h5");
    let file = hdf5::File::with_options()
        .with_create_plist(|p| {
            p.sizes(SizeofInfo {
                sizeof_addr: Sizeof::Bytes8,
                sizeof_size,
            })
        })
        .create(&path)
        .unwrap();
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

fn strings() -> Vec<String> {
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

fn collection_starts(bytes: &[u8]) -> Vec<u64> {
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

const STRING_COUNT: usize = 300;
