use core::num::NonZeroU16;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::FormatError;
use hdf5_pure_core::LengthWidth;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::GlobalHeapCollectionLocator;
use hdf5_pure_format::__private::GlobalHeapMember;
use hdf5_pure_format::__private::PreparedGlobalHeapCollection;
use rstest::rstest;
use test_util::global_heap;
use test_util::widths::Widths;

#[rstest]
#[case::full(global_heap::MIN_COLLECTION_SIZE - 2 * HEADER_SIZE, global_heap::MIN_COLLECTION_SIZE)]
#[case::implied(global_heap::MIN_COLLECTION_SIZE - 2 * HEADER_SIZE - ALIGNMENT, global_heap::MIN_COLLECTION_SIZE)]
#[case::explicit_header(global_heap::MIN_COLLECTION_SIZE - 3 * HEADER_SIZE, global_heap::MIN_COLLECTION_SIZE)]
#[case::payload_padding(global_heap::MIN_COLLECTION_SIZE - 3 * HEADER_SIZE - 1, global_heap::MIN_COLLECTION_SIZE)]
#[case::empty_payload(0, global_heap::MIN_COLLECTION_SIZE)]
#[case::selected_capacity(7, 2 * global_heap::MIN_COLLECTION_SIZE)]
#[case::narrow_capacity_limit(65_496, 65_528)]
fn prepared_global_heap_encodes_every_terminal_form(
    #[case] payload_size: usize,
    #[case] capacity: usize,
    #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] lengths: LengthWidth,
    #[values(0, 1, u16::MAX)] count: u16,
) {
    let payload = vec![0xab; payload_size];
    let objects = [global_heap::Object {
        index: u16::MAX,
        reference_count: count,
        data: &payload,
    }];
    assert_encoding(lengths, capacity, &objects);
}

#[rstest]
fn prepared_global_heap_preserves_sparse_indices_counts_and_physical_order(
    #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] lengths: LengthWidth,
) {
    let objects = [
        global_heap::Object {
            index: 400,
            reference_count: 0,
            data: b"alpha",
        },
        global_heap::Object {
            index: 7,
            reference_count: u16::MAX,
            data: b"",
        },
        global_heap::Object {
            index: u16::MAX,
            reference_count: 9,
            data: b"gamma-long",
        },
    ];
    assert_encoding(lengths, global_heap::MIN_COLLECTION_SIZE, &objects);
}

#[rstest]
fn prepared_global_heap_encodes_an_empty_collection(
    #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] lengths: LengthWidth,
) {
    assert_encoding(lengths, global_heap::MIN_COLLECTION_SIZE, &[]);
}

#[rstest]
#[case::zero(0, FormatError::GlobalHeapCapacityTooSmall { capacity: 0, minimum: global_heap::MIN_COLLECTION_SIZE as u64 })]
#[case::below_minimum(global_heap::MIN_COLLECTION_SIZE - ALIGNMENT, FormatError::GlobalHeapCapacityTooSmall { capacity: (global_heap::MIN_COLLECTION_SIZE - ALIGNMENT) as u64, minimum: global_heap::MIN_COLLECTION_SIZE as u64 })]
#[case::unaligned(global_heap::MIN_COLLECTION_SIZE + 1, FormatError::UnalignedGlobalHeapCapacity { capacity: (global_heap::MIN_COLLECTION_SIZE + 1) as u64, alignment: ALIGNMENT as u64 })]
fn prepared_global_heap_rejects_invalid_capacity(
    #[case] capacity: usize,
    #[case] expected: FormatError,
    #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] lengths: LengthWidth,
) {
    assert_eq!(
        PreparedGlobalHeapCollection::new(lengths, u64::try_from(capacity).unwrap(), &[])
            .unwrap_err(),
        expected,
    );
}

#[rstest]
#[case::two(LengthWidth::Two, 0x1_0000)]
#[case::four(LengthWidth::Four, 0x1_0000_0000)]
fn prepared_global_heap_rejects_capacity_outside_the_length_field(
    #[case] lengths: LengthWidth,
    #[case] capacity: u64,
) {
    assert_eq!(
        PreparedGlobalHeapCollection::new(lengths, capacity, &[]).unwrap_err(),
        FormatError::LengthTooLarge {
            length: capacity,
            length_size: lengths.get()
        },
    );
}

#[rstest]
#[case::after_complete_member(9)]
#[case::after_zero_count_member(0)]
fn prepared_global_heap_rejects_duplicate_member_indices(#[case] count: u16) {
    let index = NonZeroU16::new(400).unwrap();
    let members = [
        GlobalHeapMember::new(index, count, b"alpha"),
        GlobalHeapMember::new(NonZeroU16::new(7).unwrap(), 1, b"beta"),
        GlobalHeapMember::new(index, 1, b"gamma"),
    ];
    assert_eq!(
        PreparedGlobalHeapCollection::new(
            LengthWidth::Eight,
            global_heap::MIN_COLLECTION_SIZE as u64,
            &members
        )
        .unwrap_err(),
        FormatError::DuplicateGlobalHeapMemberIndex { index: index.get() },
    );
}

#[rstest]
#[case::payload_padding(global_heap::MIN_COLLECTION_SIZE - 2 * HEADER_SIZE + 1)]
#[case::complete_record(global_heap::MIN_COLLECTION_SIZE - HEADER_SIZE)]
fn prepared_global_heap_rejects_members_beyond_capacity(#[case] payload_size: usize) {
    let payload = vec![0xab; payload_size];
    let members = [GlobalHeapMember::new(
        NonZeroU16::new(7).unwrap(),
        0,
        &payload,
    )];
    let required = 2 * HEADER_SIZE + payload_size.next_multiple_of(ALIGNMENT);
    assert_eq!(
        PreparedGlobalHeapCollection::new(
            LengthWidth::Eight,
            global_heap::MIN_COLLECTION_SIZE as u64,
            &members
        )
        .unwrap_err(),
        FormatError::GlobalHeapCapacityExceeded {
            capacity: global_heap::MIN_COLLECTION_SIZE as u64,
            required: u64::try_from(required).unwrap(),
        },
    );
}

#[test]
fn prepared_global_heap_rejects_more_than_the_member_index_domain() {
    let members = vec![GlobalHeapMember::new(NonZeroU16::MIN, 0, b""); usize::from(u16::MAX) + 1];
    assert_eq!(
        PreparedGlobalHeapCollection::new(
            LengthWidth::Eight,
            global_heap::MIN_COLLECTION_SIZE as u64,
            &members
        )
        .unwrap_err(),
        FormatError::ValueTooLargeForRepresentation {
            value: u64::from(u16::MAX) + 1,
            target: "u16"
        },
    );
}

#[rstest]
#[case::beyond_32_bit(0x1_0000_0000)]
#[case::largest_aligned(u64::MAX - (ALIGNMENT as u64 - 1))]
fn prepared_global_heap_checks_host_capacity_before_emission(#[case] capacity: u64) {
    let result = PreparedGlobalHeapCollection::new(LengthWidth::Eight, capacity, &[]);
    match usize::try_from(capacity) {
        Ok(expected) => {
            let prepared = result.unwrap();
            assert_eq!(prepared.encoded_size(), expected);
            assert_eq!(prepared.length_width(), LengthWidth::Eight);
            assert_eq!(prepared.member_count(), 0);
        }
        Err(_) => assert_eq!(
            result.unwrap_err(),
            FormatError::ValueTooLargeForPlatform {
                value: capacity,
                target: "usize",
            }
        ),
    }
}

#[test]
fn prepared_global_heap_accepts_the_complete_nonzero_index_domain() {
    let members: Vec<_> = (1..=u16::MAX)
        .rev()
        .map(|index| GlobalHeapMember::new(NonZeroU16::new(index).unwrap(), 0, b""))
        .collect();
    let capacity = HEADER_SIZE * (members.len() + 1);
    let prepared = PreparedGlobalHeapCollection::new(
        LengthWidth::Four,
        u64::try_from(capacity).unwrap(),
        &members,
    )
    .unwrap();
    assert_eq!(prepared.member_count(), usize::from(u16::MAX));
    assert_eq!(prepared.encoded_size(), capacity);
    let encoded = prepared.serialize().unwrap();
    assert_eq!(encoded.len(), capacity);
    let actual = decoded(LengthWidth::Four, &encoded);
    assert_eq!(
        actual,
        (1..=u16::MAX)
            .rev()
            .map(|index| (index, 0, Vec::new()))
            .collect::<Vec<_>>()
    );
}

fn assert_encoding(lengths: LengthWidth, capacity: usize, objects: &[global_heap::Object<'_>]) {
    let members: Vec<_> = objects
        .iter()
        .map(|object| {
            GlobalHeapMember::new(
                NonZeroU16::new(object.index).unwrap(),
                object.reference_count,
                object.data,
            )
        })
        .collect();
    assert_eq!(
        members
            .iter()
            .map(|member| (
                member.index().get(),
                member.reference_count(),
                member.payload()
            ))
            .collect::<Vec<_>>(),
        objects
            .iter()
            .map(|object| (object.index, object.reference_count, object.data))
            .collect::<Vec<_>>(),
    );
    let prepared =
        PreparedGlobalHeapCollection::new(lengths, u64::try_from(capacity).unwrap(), &members)
            .unwrap();
    assert_eq!(prepared.length_width(), lengths);
    assert_eq!(prepared.encoded_size(), capacity);
    assert_eq!(prepared.member_count(), objects.len());
    let expected = global_heap::collection_of_size(
        objects,
        Widths::new(Widths::EIGHT.offset, usize::from(lengths.get())),
        capacity,
    );
    let encoded = prepared.serialize().unwrap();
    assert_eq!(encoded, expected);
    assert_eq!(prepared.serialize().unwrap(), encoded);
    assert_eq!(
        decoded(lengths, &encoded),
        objects
            .iter()
            .map(|object| (object.index, object.reference_count, object.data.to_vec()))
            .collect::<Vec<_>>()
    );
}

fn decoded(lengths: LengthWidth, bytes: &[u8]) -> Vec<(u16, u16, Vec<u8>)> {
    let base = PhysicalAddress::new(512);
    let region = FileRegion::new(
        FileRange::new(base, u64::try_from(bytes.len()).unwrap()).unwrap(),
        bytes,
    )
    .unwrap();
    let locator = GlobalHeapCollectionLocator::new(base, lengths);
    let mut walk = locator
        .parse(region.subregion(locator.parse_range().unwrap()).unwrap())
        .unwrap()
        .walk();
    let mut objects = Vec::new();
    while let Some(range) = walk.parse_range() {
        if let Some(object) = walk.parse(region.subregion(range).unwrap()).unwrap() {
            objects.push((
                object.index().get(),
                object.reference_count(),
                object.payload().parse(region).unwrap().to_vec(),
            ));
        }
    }
    objects
}

// `H5HG_SIZEOF_HDR` and `H5HG_SIZEOF_OBJHDR` in `H5HGpkg.h`, HDF5 2.2.0.
const HEADER_SIZE: usize = 16;

// Version 1 collections use this geometry ("Global Heap", format specification, version 4.0).
const ALIGNMENT: usize = 8;
