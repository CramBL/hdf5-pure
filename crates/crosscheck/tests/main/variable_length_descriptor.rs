#![cfg(feature = "hdf5")]

use core::num::NonZeroU16;

use hdf5::Dataset;
use hdf5::plist::file_create::Sizeof;
use hdf5::plist::file_create::SizeofInfo;
use hdf5::types::VarLenArray;
use hdf5::types::VarLenAscii;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::LengthWidth;
use hdf5_pure_core::OffsetWidth;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::GlobalHeapCollectionLocator;
use hdf5_pure_format::__private::VariableLengthDescriptor;
use rstest::rstest;
use test_util::bytes;
use test_util_hdf5::lock;

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two, Sizeof::Bytes2, Sizeof::Bytes2)]
#[case::two_four(OffsetWidth::Two, LengthWidth::Four, Sizeof::Bytes2, Sizeof::Bytes4)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight, Sizeof::Bytes2, Sizeof::Bytes8)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two, Sizeof::Bytes4, Sizeof::Bytes2)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four, Sizeof::Bytes4, Sizeof::Bytes4)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight, Sizeof::Bytes4, Sizeof::Bytes8)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two, Sizeof::Bytes8, Sizeof::Bytes2)]
#[case::eight_four(OffsetWidth::Eight, LengthWidth::Four, Sizeof::Bytes8, Sizeof::Bytes4)]
#[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight, Sizeof::Bytes8, Sizeof::Bytes8)]
fn variable_length_descriptors_preserve_native_string_and_sequence_records(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[case] sizeof_addr: Sizeof,
    #[case] sizeof_size: Sizeof,
    #[values(Payload::Sequence, Payload::String)] payload: Payload,
    #[values(Value::Empty, Value::Nonempty)] value: Value,
) {
    let _guard = lock::libhdf5_guard();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("descriptors.h5");
    let file = hdf5::File::with_options()
        .with_create_plist(|properties| {
            properties.sizes(SizeofInfo {
                sizeof_addr,
                sizeof_size,
            })
        })
        .create(&path)
        .unwrap();
    let dataset = payload.create(&file);
    let start = usize::try_from(dataset.offset().unwrap()).unwrap();
    let count_size = 0u32.to_le_bytes().len();
    let descriptor_size = count_size * 2 + usize::from(offsets.get());
    assert_eq!(
        dataset.storage_size(),
        u64::try_from(2 * descriptor_size).unwrap()
    );
    drop(dataset);
    file.close().unwrap();

    let image = std::fs::read(&path).unwrap();
    let start = start + value.index() * descriptor_size;
    let raw = &image[start..start + descriptor_size];
    let (descriptor, consumed) = VariableLengthDescriptor::parse(offsets, raw).unwrap();
    let expected_count = match value {
        Value::Empty => 0,
        Value::Nonempty => payload.count(),
    };
    assert_eq!(consumed, descriptor_size);
    assert_eq!(descriptor.encoded_size(), descriptor_size);
    assert_eq!(descriptor.element_count(), expected_count);
    assert_eq!(descriptor.offset_width(), offsets);
    assert_eq!(
        descriptor.collection_address().get(),
        bytes::uint_at(raw, count_size, usize::from(offsets.get())),
    );
    assert_eq!(
        u64::from(descriptor.object_index()),
        bytes::uint_at(raw, count_size + usize::from(offsets.get()), count_size),
    );
    let mut encoded = vec![0xab; descriptor_size];
    assert_eq!(
        descriptor.write_into(&mut encoded).unwrap(),
        descriptor_size
    );
    assert_eq!(encoded, raw);
    assert_eq!(
        VariableLengthDescriptor::new(
            offsets,
            expected_count,
            descriptor.collection_address(),
            descriptor.object_index(),
        )
        .unwrap(),
        descriptor,
    );
    let expected = match value {
        Value::Empty => Vec::new(),
        Value::Nonempty => payload.bytes(),
    };
    if descriptor.collection_address().get() == 0 {
        assert_eq!(expected, Vec::<u8>::new());
        assert_eq!(descriptor.element_count(), 0);
        assert_eq!(descriptor.object_index(), 0);
    } else {
        assert_eq!(heap_payload(&image, lengths, &descriptor), expected);
    }
}

#[derive(Clone, Copy)]
enum Payload {
    Sequence,
    String,
}

impl Payload {
    fn create(self, file: &hdf5::File) -> Dataset {
        match self {
            Self::Sequence => {
                let values = [
                    VarLenArray::from_slice(&[3u16, 5]),
                    VarLenArray::from_slice(&[]),
                ];
                let dataset = file
                    .new_dataset_builder()
                    .with_data(&values)
                    .create("values")
                    .unwrap();
                assert_eq!(dataset.read_raw::<VarLenArray<u16>>().unwrap(), values);
                dataset
            }
            Self::String => {
                let values = [
                    VarLenAscii::from_ascii("alpha").unwrap(),
                    VarLenAscii::from_ascii("").unwrap(),
                ];
                let dataset = file
                    .new_dataset_builder()
                    .with_data(&values)
                    .create("values")
                    .unwrap();
                assert_eq!(dataset.read_raw::<VarLenAscii>().unwrap(), values);
                dataset
            }
        }
    }

    fn count(self) -> u32 {
        match self {
            Self::Sequence => 2,
            Self::String => 5,
        }
    }

    fn bytes(self) -> Vec<u8> {
        match self {
            Self::Sequence => [3u16, 5].into_iter().flat_map(u16::to_ne_bytes).collect(),
            Self::String => b"alpha".to_vec(),
        }
    }
}

#[derive(Clone, Copy)]
enum Value {
    Empty,
    Nonempty,
}

impl Value {
    fn index(self) -> usize {
        match self {
            Self::Empty => 1,
            Self::Nonempty => 0,
        }
    }
}

fn heap_payload(
    bytes: &[u8],
    lengths: LengthWidth,
    descriptor: &VariableLengthDescriptor,
) -> Vec<u8> {
    let region = FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), u64::try_from(bytes.len()).unwrap()).unwrap(),
        bytes,
    )
    .unwrap();
    let locator = GlobalHeapCollectionLocator::new(
        PhysicalAddress::new(descriptor.collection_address().get()),
        lengths,
    );
    let collection = locator
        .parse(region.subregion(locator.parse_range().unwrap()).unwrap())
        .unwrap();
    let index = NonZeroU16::new(u16::try_from(descriptor.object_index()).unwrap()).unwrap();
    let mut lookup = collection.lookup(index);
    let mut found = None;
    while let Some(range) = lookup.parse_range() {
        found = lookup.parse(region.subregion(range).unwrap()).unwrap();
    }
    found.unwrap().payload().parse(region).unwrap().to_vec()
}
