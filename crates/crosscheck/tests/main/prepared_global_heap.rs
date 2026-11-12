#![cfg(feature = "hdf5")]

use core::num::NonZeroU16;
use std::path::Path;

use hdf5::plist::file_create::Sizeof;
use hdf5::types::VarLenUnicode;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::LengthWidth;
use hdf5_pure_core::OffsetWidth;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::GlobalHeapCollection;
use hdf5_pure_format::__private::GlobalHeapCollectionLocator;
use hdf5_pure_format::__private::GlobalHeapMember;
use hdf5_pure_format::__private::GlobalHeapObject;
use hdf5_pure_format::__private::PreparedGlobalHeapCollection;
use hdf5_pure_format::__private::VariableLengthDescriptor;
use rstest::rstest;
use test_util::global_heap;
use test_util_hdf5::global_heap as native_heap;
use test_util_hdf5::lock;

#[rstest]
#[case::two(Sizeof::Bytes2, LengthWidth::Two)]
#[case::four(Sizeof::Bytes4, LengthWidth::Four)]
#[case::eight(Sizeof::Bytes8, LengthWidth::Eight)]
fn prepared_global_heap_reproduces_native_terminal_forms(
    #[case] sizeof_size: Sizeof,
    #[case] lengths: LengthWidth,
    #[values(Terminal::Explicit, Terminal::Full, Terminal::Implied)] terminal: Terminal,
) {
    let _guard = lock::libhdf5_guard();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("terminal.h5");
    let values = ["x".repeat(terminal.payload_size())];
    let (mut image, _) = native_image(&path, sizeof_size, &values);
    let (collection, objects) = parse_collection(&image, lengths);
    let region = region(&image);
    assert_eq!(objects.len(), 1);
    let object = objects[0];
    assert_eq!(object.reference_count(), NATIVE_INITIAL_REFERENCE_COUNT);
    assert_eq!(
        object.payload().parse(region).unwrap(),
        values[0].as_bytes()
    );
    assert_eq!(
        collection.range().length(),
        global_heap::MIN_COLLECTION_SIZE as u64
    );
    let members = [GlobalHeapMember::new(
        object.index(),
        object.reference_count(),
        object.payload().parse(region).unwrap(),
    )];
    let prepared =
        PreparedGlobalHeapCollection::new(lengths, collection.range().length(), &members).unwrap();
    let encoded = prepared.serialize().unwrap();
    let start = usize::try_from(collection.range().start().get()).unwrap();
    let end = start + encoded.len();
    assert_eq!(encoded, image[start..end]);
    image[start..end].copy_from_slice(&encoded);
    std::fs::write(&path, image).unwrap();
    let file = hdf5::File::open(&path).unwrap();
    let dataset = file.dataset("values").unwrap();
    let actual = dataset.read_raw::<VarLenUnicode>().unwrap();
    assert_eq!(
        actual.iter().map(VarLenUnicode::as_str).collect::<Vec<_>>(),
        values
    );
    drop(dataset);
    file.close().unwrap();
}

#[rstest]
#[case::two(Sizeof::Bytes2, LengthWidth::Two)]
#[case::four(Sizeof::Bytes4, LengthWidth::Four)]
#[case::eight(Sizeof::Bytes8, LengthWidth::Eight)]
fn native_reader_reads_sparse_members_in_supplied_physical_order(
    #[case] sizeof_size: Sizeof,
    #[case] lengths: LengthWidth,
) {
    let _guard = lock::libhdf5_guard();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("sparse.h5");
    let values = ["alpha".to_owned(), "beta".to_owned(), "gamma".to_owned()];
    let (mut image, data_start) = native_image(&path, sizeof_size, &values);
    let (collection, original) = parse_collection(&image, lengths);
    assert_eq!(original.len(), values.len());
    assert_eq!(
        original
            .iter()
            .map(GlobalHeapObject::reference_count)
            .collect::<Vec<_>>(),
        vec![NATIVE_INITIAL_REFERENCE_COUNT; values.len()]
    );
    let members = [
        GlobalHeapMember::new(NonZeroU16::new(u16::MAX).unwrap(), 0, values[2].as_bytes()),
        GlobalHeapMember::new(NonZeroU16::new(7).unwrap(), u16::MAX, values[0].as_bytes()),
        GlobalHeapMember::new(NonZeroU16::new(400).unwrap(), 9, values[1].as_bytes()),
    ];
    let prepared =
        PreparedGlobalHeapCollection::new(lengths, collection.range().length(), &members).unwrap();
    let encoded = prepared.serialize().unwrap();
    let start = usize::try_from(collection.range().start().get()).unwrap();
    image[start..start + encoded.len()].copy_from_slice(&encoded);
    let stride = VariableLengthDescriptor::serialized_size(OffsetWidth::Eight);
    for (ordinal, index) in [7u16, 400, u16::MAX].into_iter().enumerate() {
        let at = data_start + ordinal * stride;
        let (descriptor, consumed) =
            VariableLengthDescriptor::parse(OffsetWidth::Eight, &image[at..at + stride]).unwrap();
        assert_eq!(consumed, stride);
        assert_eq!(
            descriptor.collection_address().get(),
            collection.range().start().get()
        );
        let replacement = VariableLengthDescriptor::new(
            OffsetWidth::Eight,
            descriptor.element_count(),
            descriptor.collection_address(),
            u32::from(index),
        )
        .unwrap();
        assert_eq!(
            replacement.write_into(&mut image[at..at + stride]).unwrap(),
            stride
        );
    }
    let (_, parsed) = parse_collection(&image, lengths);
    assert_eq!(
        parsed
            .iter()
            .map(|object| (object.index().get(), object.reference_count()))
            .collect::<Vec<_>>(),
        [(u16::MAX, 0), (7, u16::MAX), (400, 9)]
    );
    std::fs::write(&path, image).unwrap();
    let file = hdf5::File::open(&path).unwrap();
    let dataset = file.dataset("values").unwrap();
    let actual = dataset.read_raw::<VarLenUnicode>().unwrap();
    assert_eq!(
        actual.iter().map(VarLenUnicode::as_str).collect::<Vec<_>>(),
        values
    );
    drop(dataset);
    file.close().unwrap();
}

#[derive(Clone, Copy)]
enum Terminal {
    Explicit,
    Full,
    Implied,
}

impl Terminal {
    fn payload_size(self) -> usize {
        let free = match self {
            Self::Explicit => HEADER_SIZE,
            Self::Full => 0,
            Self::Implied => ALIGNMENT,
        };
        global_heap::MIN_COLLECTION_SIZE - 2 * HEADER_SIZE - free
    }
}

fn native_image(path: &Path, sizeof_size: Sizeof, values: &[String]) -> (Vec<u8>, usize) {
    let file = native_heap::create_libhdf5_file(path, sizeof_size, 0);
    let dataset = file
        .new_dataset::<VarLenUnicode>()
        .shape(values.len())
        .create("values")
        .unwrap();
    let values: Vec<VarLenUnicode> = values.iter().map(|value| value.parse().unwrap()).collect();
    dataset.write_raw(&values).unwrap();
    assert_eq!(dataset.read_raw::<VarLenUnicode>().unwrap(), values);
    let start = usize::try_from(dataset.offset().unwrap()).unwrap();
    drop(dataset);
    file.close().unwrap();
    (std::fs::read(path).unwrap(), start)
}

fn parse_collection(
    image: &[u8],
    lengths: LengthWidth,
) -> (GlobalHeapCollection, Vec<GlobalHeapObject>) {
    let starts = global_heap::collection_starts(image);
    let [start] = starts.as_slice() else {
        panic!("expected one collection, got {starts:?}");
    };
    let region = region(image);
    let locator = GlobalHeapCollectionLocator::new(PhysicalAddress::new(*start), lengths);
    let collection = locator
        .parse(region.subregion(locator.parse_range().unwrap()).unwrap())
        .unwrap();
    let mut walk = collection.walk();
    let mut objects = Vec::new();
    while let Some(range) = walk.parse_range() {
        objects.extend(walk.parse(region.subregion(range).unwrap()).unwrap());
    }
    (collection, objects)
}

fn region(bytes: &[u8]) -> FileRegion<'_> {
    FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), u64::try_from(bytes.len()).unwrap()).unwrap(),
        bytes,
    )
    .unwrap()
}

// `H5HG_SIZEOF_HDR` and `H5HG_SIZEOF_OBJHDR` in `H5HGpkg.h`, HDF5 2.2.0.
const HEADER_SIZE: usize = 16;

// Version 1 collections use this geometry ("Global Heap", format specification, version 4.0).
const ALIGNMENT: usize = 8;

// `H5HG_insert` in `H5HG.c`, HDF5 2.0.0, initializes each object's reference count to zero.
const NATIVE_INITIAL_REFERENCE_COUNT: u16 = 0;
