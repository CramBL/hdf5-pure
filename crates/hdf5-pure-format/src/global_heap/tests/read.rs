use core::num::NonZeroU16;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;

use super::fixture;
use crate::error::FormatError;
use crate::global_heap::collection::{GCOL_SIGNATURE, GCOL_VERSION, GlobalHeapCollectionLocator};
use crate::global_heap::object::GlobalHeapObject;
use crate::global_heap::write;
use crate::width::LengthWidth;

#[rstest]
#[case::empty(LengthWidth::Eight, 0, &[])]
#[case::two_byte_lengths(LengthWidth::Two, 0, &[b"alpha".as_slice(), b"".as_slice(), b"gamma!!!".as_slice()])]
#[case::four_byte_lengths(LengthWidth::Four, 0, &[b"alpha".as_slice(), b"".as_slice(), b"gamma!!!".as_slice()])]
#[case::eight_byte_lengths(LengthWidth::Eight, 0, &[b"alpha".as_slice(), b"".as_slice(), b"gamma!!!".as_slice()])]
#[case::large_object(LengthWidth::Four, 0, &[LARGE_OBJECT.as_slice()])]
#[case::above_four_gib(LengthWidth::Eight, (1 << 32) + 4096, &[b"alpha".as_slice(), b"beta".as_slice()])]
fn a_walk_reads_back_the_objects_the_encoder_writes(
    #[case] lengths: LengthWidth,
    #[case] address: u64,
    #[case] objects: &[&[u8]],
) {
    let encoded = write::encode_global_heap_collection(lengths, objects).unwrap();
    let expected: Vec<(u16, &[u8])> = (1..).zip(objects.iter().copied()).collect();
    assert_eq!(
        fixture::walk_objects(fixture::region(address, &encoded), lengths).unwrap(),
        expected
    );
}

#[test]
fn a_walk_reads_many_objects() {
    let payloads: Vec<[u8; 2]> = (0..2000u16).map(u16::to_le_bytes).collect();
    let objects: Vec<&[u8]> = payloads.iter().map(<[u8; 2]>::as_slice).collect();
    let encoded = write::encode_global_heap_collection(LengthWidth::Two, &objects).unwrap();
    let expected: Vec<(u16, &[u8])> = (1..).zip(objects.iter().copied()).collect();
    assert_eq!(
        fixture::walk_objects(fixture::region(0, &encoded), LengthWidth::Two).unwrap(),
        expected
    );
}

#[rstest]
#[case::empty(&[], &[])]
#[case::two(&[b"alpha".as_slice(), b"beta".as_slice()], &[(1, 32, 5), (2, 56, 4)])]
#[case::large(
    &[b"head".as_slice(), [7u8; 5000].as_slice(), b"tail".as_slice()],
    &[(1, 32, 4), (2, 56, 5000), (3, 5072, 4)]
)]
fn a_walk_frames_each_payload_after_its_object_header(
    #[case] objects: &[&[u8]],
    #[case] expected: &[(u16, u64, u64)],
) {
    let encoded = write::encode_global_heap_collection(LengthWidth::Eight, objects).unwrap();
    let image = fixture::region(0, &encoded);
    let mut walk = fixture::collection_at(image, LengthWidth::Eight).walk();
    let mut walked = Vec::new();
    while let Some(range) = walk.parse_range() {
        walked.extend(walk.parse(image.subregion(range).unwrap()).unwrap());
    }
    let framed: Vec<(u16, u64, u64)> = walked
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
    assert_eq!(framed, expected);
}

#[test]
fn a_lookup_reads_the_object_headers_and_its_own_payload_alone() {
    let objects: [&[u8]; 4] = [b"one", &[2; 100], b"three", &[4; 200]];
    let encoded = write::encode_global_heap_collection(LengthWidth::Eight, &objects).unwrap();
    let image = fixture::region(512, &encoded);
    let mut lookup =
        fixture::collection_at(image, LengthWidth::Eight).lookup(NonZeroU16::new(3).unwrap());
    let mut requested = Vec::new();
    let object = loop {
        let range = lookup.parse_range().unwrap();
        requested.push(range.start().get()..range.end().get());
        if let Some(object) = lookup.parse(image.subregion(range).unwrap()).unwrap() {
            break object;
        }
    };

    // The headers of the 3-, 100-, 5-, and 200-byte objects start at 528, 552, 672, and 696,
    // and the free space header at 912.
    assert_eq!(
        requested,
        [528..544, 552..568, 672..688, 696..712, 912..928]
    );
    assert_eq!(object.index().get(), 3);
    assert_eq!(object.reference_count(), 1);
    assert_eq!(
        object.range().start().get()..object.range().end().get(),
        672..696
    );
    assert_eq!(
        object
            .payload()
            .parse(fixture::region(688, b"three"))
            .unwrap(),
        b"three"
    );
    assert_eq!(lookup.parse_range(), None);
}

#[test]
fn a_walk_frames_an_object_in_an_8_gib_collection_from_headers_alone() {
    // An 8 GiB collection at 1 TiB, filled by one object up to a 16-byte free space object.
    let address = 1 << 40;
    let size = 1u64 << 33;
    let mut collection_header = GCOL_SIGNATURE.to_vec();
    collection_header.extend_from_slice(&[GCOL_VERSION, 0, 0, 0]);
    collection_header.extend_from_slice(&size.to_le_bytes());
    let mut object = vec![1, 0, 1, 0, 0, 0, 0, 0];
    object.extend_from_slice(&(size - 48).to_le_bytes());
    let mut free_space = vec![0; 8];
    free_space.extend_from_slice(&16u64.to_le_bytes());
    let headers = [
        fixture::region(address + 16, &object),
        fixture::region(address + size - 16, &free_space),
    ];

    let mut walk = fixture::collection_at(
        fixture::region(address, &collection_header),
        LengthWidth::Eight,
    )
    .walk();
    let objects: Vec<GlobalHeapObject> = headers
        .iter()
        .filter_map(|&header| {
            assert_eq!(walk.parse_range(), Some(header.range()));
            walk.parse(header).unwrap()
        })
        .collect();

    assert_eq!(walk.parse_range(), None);
    let [object] = objects.as_slice() else {
        panic!("expected one object, got {objects:?}");
    };
    let payload = object.payload().range();
    assert_eq!(payload.start().get(), address + 32);
    assert_eq!(payload.length(), size - 48);
}

#[rstest]
#[case::requested_first(1)]
#[case::requested_between(2)]
fn a_lookup_rejects_a_duplicate_index_after_the_requested_object(#[case] index: u16) {
    let bytes = fixture::padded_collection(
        LengthWidth::Eight,
        4096,
        &[(1, b"first"), (2, b"second"), (1, b"again")],
    );
    let image = fixture::region(0, &bytes);
    let mut lookup =
        fixture::collection_at(image, LengthWidth::Eight).lookup(NonZeroU16::new(index).unwrap());
    let err = loop {
        let range = lookup.parse_range().unwrap();
        match lookup.parse(image.subregion(range).unwrap()) {
            Ok(found) => assert_eq!(found, None),
            Err(err) => break err,
        }
    };
    assert_eq!(
        err,
        FormatError::DuplicateGlobalHeapObjectIndex {
            collection_address: 0,
            index: 1,
        }
    );
}

#[test]
fn a_lookup_of_an_absent_index_walks_to_the_end_and_reports_it() {
    let bytes =
        fixture::padded_collection(LengthWidth::Eight, 4096, &[(1, b"first"), (3, b"third")]);
    let image = fixture::region(0, &bytes);
    let mut lookup =
        fixture::collection_at(image, LengthWidth::Eight).lookup(NonZeroU16::new(2).unwrap());
    let mut steps = 0;
    let err = loop {
        steps += 1;
        let range = lookup.parse_range().unwrap();
        if let Err(err) = lookup.parse(image.subregion(range).unwrap()) {
            break err;
        }
    };
    assert_eq!(steps, 3);
    assert_eq!(
        err,
        FormatError::GlobalHeapObjectNotFound {
            collection_address: 0,
            index: 2,
        }
    );
}

#[rstest]
#[case::free_space_ends_at_the_collection_end(4096, 64, 0, 4032, Ok(vec![1, 2]))]
#[case::free_space_followed_by_bytes(
    4096,
    64,
    0,
    4024,
    Err(FormatError::MisplacedGlobalHeapFreeSpace { address: 64, size: 4024, collection_end: 4096 })
)]
#[case::free_space_past_the_collection_end(
    4096,
    64,
    0,
    4040,
    Err(FormatError::MisplacedGlobalHeapFreeSpace { address: 64, size: 4040, collection_end: 4096 })
)]
#[case::free_space_followed_by_an_object(
    4096,
    16,
    0,
    24,
    Err(FormatError::MisplacedGlobalHeapFreeSpace { address: 16, size: 24, collection_end: 4096 })
)]
#[case::object_past_the_collection_end(
    4096,
    40,
    2,
    4041,
    Err(FormatError::GlobalHeapObjectOutsideCollection {
        address: 40,
        index: 2,
        collection_end: 4096,
    })
)]
#[case::object_size_overflows(
    4096,
    40,
    2,
    u64::MAX,
    Err(FormatError::GlobalHeapObjectOutsideCollection {
        address: 40,
        index: 2,
        collection_end: 4096,
    })
)]
#[case::object_fills_the_collection(4104, 40, 2, 4048, Ok(vec![1, 2]))]
#[case::object_one_byte_past_the_collection_end(
    4104,
    40,
    2,
    4049,
    Err(FormatError::GlobalHeapObjectOutsideCollection {
        address: 40,
        index: 2,
        collection_end: 4104,
    })
)]
fn a_walk_checks_each_extent_against_the_collection_end(
    #[case] collection_size: usize,
    #[case] at: usize,
    #[case] index: u16,
    #[case] size: u64,
    #[case] expected: Result<Vec<u16>, FormatError>,
) {
    let mut bytes = fixture::padded_collection(
        LengthWidth::Eight,
        collection_size,
        &[(1, b"first"), (2, b"second")],
    );
    bytes[at..at + 2].copy_from_slice(&index.to_le_bytes());
    bytes[at + 8..at + 16].copy_from_slice(&size.to_le_bytes());
    assert_eq!(
        fixture::walk_objects(fixture::region(0, &bytes), LengthWidth::Eight)
            .map(|objects| objects.iter().map(|&(index, _)| index).collect()),
        expected
    );
    if let Err(expected) = expected {
        let image = fixture::region(0, &bytes);
        let mut lookup =
            fixture::collection_at(image, LengthWidth::Eight).lookup(NonZeroU16::new(1).unwrap());
        let err = loop {
            let range = lookup.parse_range().unwrap();
            match lookup.parse(image.subregion(range).unwrap()) {
                Ok(found) => assert_eq!(found, None),
                Err(err) => break err,
            }
        };
        assert_eq!(err, expected);
    }
}

#[rstest]
#[case::zero_tail(4096, 4056, [0; 8])]
#[case::nonzero_tail(4096, 4056, [7, 0, 0, 0, 0, 0, 0, 0])]
#[case::all_nonzero_tail(4096, 4056, [0xff; 8])]
#[case::larger_collection(4104, 4064, [7, 0, 0, 0, 0, 0, 0, 0])]
#[case::no_free_space(4096, 4064, [0; 8])]
fn a_tail_shorter_than_an_object_header_is_free_space(
    #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] lengths: LengthWidth,
    #[case] collection_size: usize,
    #[case] object_len: usize,
    #[case] tail: [u8; 8],
) {
    let payload = vec![9; object_len];
    let mut bytes = fixture::padded_collection(lengths, collection_size, &[(1, &payload)]);
    let member_end = 32 + object_len;
    if member_end < collection_size {
        assert_eq!(collection_size - member_end, tail.len());
        bytes[member_end..].copy_from_slice(&tail);
    }
    let image = fixture::region(0, &bytes);
    assert_eq!(
        fixture::walk_objects(image, lengths).unwrap(),
        [(1, payload.as_slice())]
    );

    let mut lookup = fixture::collection_at(image, lengths).lookup(NonZeroU16::new(1).unwrap());
    let mut requested = Vec::new();
    let found = loop {
        let range = lookup.parse_range().unwrap();
        requested.push(range.start().get()..range.end().get());
        if let Some(object) = lookup.parse(image.subregion(range).unwrap()).unwrap() {
            break object;
        }
    };
    assert_eq!(found.index().get(), 1);
    assert_eq!(found.payload().parse(image).unwrap(), payload);
    assert_eq!(lookup.parse_range(), None);
    let expected: Vec<_> = core::iter::once(16..32)
        .chain((member_end < collection_size).then_some(member_end as u64..collection_size as u64))
        .collect();
    assert_eq!(requested, expected);
}

#[rstest]
#[case::two_byte_lengths(LengthWidth::Two)]
#[case::four_byte_lengths(LengthWidth::Four)]
#[case::eight_byte_lengths(LengthWidth::Eight)]
fn a_walk_ignores_the_reserved_bytes_of_every_header(#[case] lengths: LengthWidth) {
    let objects: [&[u8]; 2] = [b"alpha", b"beta"];
    let mut bytes = write::encode_global_heap_collection(lengths, &objects).unwrap();
    bytes[5..8].fill(0xff);
    for object_header in [16, 40, 64] {
        bytes[object_header + 4..object_header + 8].fill(0xff);
    }
    assert_eq!(
        fixture::walk_objects(fixture::region(0, &bytes), lengths).unwrap(),
        vec![(1, objects[0]), (2, objects[1])]
    );
}

#[rstest]
#[case::complete_free_header(5)]
#[case::implied_free_tail(4056)]
fn each_step_rejects_a_region_short_of_its_range_and_accepts_the_range_after(
    #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] lengths: LengthWidth,
    #[case] payload_len: usize,
) {
    let payload_bytes = vec![9; payload_len];
    let bytes = fixture::padded_collection(lengths, 4096, &[(1, &payload_bytes)]);
    let image = fixture::region(256, &bytes);
    let short = |range: FileRange| {
        image
            .subregion(FileRange::new(range.start(), range.length() - 1).unwrap())
            .unwrap()
    };
    let insufficient = |range: FileRange| FormatError::InsufficientRegion {
        required_start: range.start().get(),
        required_length: range.length(),
        provided_start: range.start().get(),
        provided_length: range.length() - 1,
    };

    let locator = GlobalHeapCollectionLocator::new(PhysicalAddress::new(256), lengths);
    let collection_header = locator.parse_range().unwrap();
    assert_eq!(
        locator.parse(short(collection_header)).unwrap_err(),
        insufficient(collection_header)
    );
    let mut walk = locator
        .parse(image.subregion(collection_header).unwrap())
        .unwrap()
        .walk();

    let object_header = walk.parse_range().unwrap();
    assert_eq!(
        walk.parse(short(object_header)).unwrap_err(),
        insufficient(object_header)
    );
    assert_eq!(walk.parse_range(), Some(object_header));
    let object = walk
        .parse(image.subregion(object_header).unwrap())
        .unwrap()
        .unwrap();

    let free_space_header = walk.parse_range().unwrap();
    assert_eq!(
        walk.parse(short(free_space_header)).unwrap_err(),
        insufficient(free_space_header)
    );
    assert_eq!(
        walk.parse(image.subregion(free_space_header).unwrap()),
        Ok(None)
    );
    assert_eq!(walk.parse_range(), None);

    let payload = object.payload().range();
    assert_eq!(
        object.payload().parse(short(payload)).unwrap_err(),
        insufficient(payload)
    );
    assert_eq!(object.payload().parse(image).unwrap(), payload_bytes);
}

static LARGE_OBJECT: [u8; 70_000] = [7; 70_000];
