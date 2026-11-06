use alloc::vec;
use core::ops::Range;

use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::bytes;
use test_util::global_heap;
use test_util::widths::Widths;

use super::*;
use crate::global_heap::GlobalHeapCollectionLocator;
use crate::width::LengthWidth;

#[rstest]
#[case::first(LengthWidth::Two, &[7], &[(400, 40, 16, 16), (65_535, 56, 32, 32)], Some(64))]
#[case::middle(LengthWidth::Four, &[400], &[(65_535, 56, 40, 32)], Some(72))]
#[case::last(LengthWidth::Eight, &[65_535], &[], Some(56))]
#[case::separated(LengthWidth::Eight, &[7, 65_535], &[(400, 40, 16, 16)], Some(32))]
#[case::all(LengthWidth::Two, &[7, 400, 65_535], &[], Some(16))]
#[case::unchanged(LengthWidth::Four, &[], &[], None)]
fn compaction_preserves_complete_records_and_collection_capacity(
    #[case] lengths: LengthWidth,
    #[case] removed: &[u16],
    #[case] expected_moves: &[(u16, u64, u64, u64)],
    #[case] free_start: Option<u64>,
) {
    let objects = [
        global_heap::Object {
            index: 7,
            reference_count: 0,
            data: b"alpha",
        },
        global_heap::Object {
            index: 400,
            reference_count: 9,
            data: b"",
        },
        global_heap::Object {
            index: 65_535,
            reference_count: 65_535,
            data: b"gamma-long",
        },
    ];
    let mut original =
        global_heap::collection(&objects, Widths::new(8, usize::from(lengths.get())));
    for &start in &[16, 40, 56] {
        original[start + 4..start + 8].fill(0x8e);
    }
    original[39] = 0x3d;
    original[87] = 0x3d;
    let base = 512;
    let members = collect(region(base, &original), lengths);
    assert_eq!(
        members
            .members()
            .iter()
            .map(|object| (object.index().get(), object.reference_count()))
            .collect::<Vec<_>>(),
        [(7, 0), (400, 9), (65_535, 65_535)]
    );
    let plan = members
        .compact(|object| removed.contains(&object.index().get()))
        .unwrap();
    assert_eq!(plan.range(), members.range());
    assert_eq!(
        plan.range(),
        FileRange::new(PhysicalAddress::new(base), 4096).unwrap()
    );
    assert_eq!(plan.retained(), 3 - removed.len());
    let moves: Vec<_> = plan
        .moves()
        .iter()
        .map(|movement| {
            (
                movement.object().index().get(),
                movement.object().range().start().get() - base,
                movement.destination().start().get() - base,
                movement.destination().length(),
            )
        })
        .collect();
    assert_eq!(moves, expected_moves);
    for movement in plan.moves() {
        assert_eq!(
            movement.destination().length(),
            movement.object().range().length()
        );
        assert!(movement.destination().start() < movement.object().range().start());
    }
    match (plan.free_space(), free_start) {
        (Some(free), Some(start)) => {
            assert_eq!(
                free.range(),
                FileRange::new(PhysicalAddress::new(base + start), 4096 - start).unwrap()
            );
            assert_eq!(
                free.header_range(),
                FileRange::new(PhysicalAddress::new(base + start), 16).unwrap()
            );
            let mut expected = [0; 16];
            bytes::set_uint_at(&mut expected, 8, usize::from(lengths.get()), 4096 - start);
            assert_eq!(free.bytes(), &expected);
        }
        (None, None) => {}
        _ => panic!("unexpected free-space header"),
    }
    let mut output = original.clone();
    for movement in plan.moves() {
        output.copy_within(
            local(movement.object().range(), base),
            usize::try_from(movement.destination().start().get() - base).unwrap(),
        );
    }
    if let Some(free) = plan.free_space() {
        output[local(free.header_range(), base)].copy_from_slice(free.bytes());
    }
    assert_eq!(&output[..16], &original[..16]);
    let mut cursor = 16;
    for (object, source) in objects.iter().zip([16..40, 40..56, 56..88]) {
        if !removed.contains(&object.index) {
            let end = cursor + source.len();
            assert_eq!(&output[cursor..end], &original[source]);
            cursor = end;
        }
    }
    if let Some(start) = free_start {
        assert_eq!(
            &output[usize::try_from(start).unwrap() + 16..],
            &original[usize::try_from(start).unwrap() + 16..]
        );
    } else {
        assert_eq!(output, original);
    }
    let parsed = collect(region(base, &output), lengths);
    assert_eq!(
        parsed
            .members()
            .iter()
            .map(|object| object.index().get())
            .collect::<Vec<_>>(),
        objects
            .iter()
            .filter(|object| !removed.contains(&object.index))
            .map(|object| object.index)
            .collect::<Vec<_>>()
    );
}

#[rstest]
#[case::empty(0)]
#[case::short_tail(4056)]
#[case::exactly_full(4064)]
fn removing_all_members_plans_free_space_only_for_nonempty_collections(#[case] payload: usize) {
    let data = vec![0x53; payload];
    let objects = if payload == 0 {
        vec![]
    } else {
        vec![global_heap::Object::new(17, &data)]
    };
    let original = global_heap::collection_of_size(&objects, Widths::new(8, 8), 4096);
    let members = collect(region(0, &original), LengthWidth::Eight);
    let plan = members.compact(|_| true).unwrap();
    assert_eq!(plan.retained(), 0);
    assert!(plan.moves().is_empty());
    if payload == 0 {
        assert!(plan.free_space().is_none());
    } else {
        let free = plan.free_space().unwrap();
        assert_eq!(
            free.range(),
            FileRange::new(PhysicalAddress::new(16), 4080).unwrap()
        );
        assert_eq!(bytes::u64_at(free.bytes(), 8), 4080);
    }
}

#[test]
fn compaction_plans_multi_gib_records_from_headers_alone() {
    let base = 1u64 << 40;
    let size = 1u64 << 33;
    let mut header = global_heap::collection(&[], Widths::new(8, 8));
    header.truncate(16);
    bytes::set_uint_at(&mut header, 8, 8, size);
    let mut first = [0; 16];
    bytes::set_u16_at(&mut first, 0, 11);
    let mut second = [0; 16];
    bytes::set_u16_at(&mut second, 0, 61_000);
    bytes::set_uint_at(&mut second, 8, 8, size - 64);
    let mut free = [0; 16];
    bytes::set_uint_at(&mut free, 8, 8, 16);
    let collection =
        GlobalHeapCollectionLocator::new(PhysicalAddress::new(base), LengthWidth::Eight)
            .parse(region(base, &header))
            .unwrap();
    let mut collector = collection.collect_members();
    for image in [
        region(base + 16, &first),
        region(base + 32, &second),
        region(base + size - 16, &free),
    ] {
        assert_eq!(collector.parse_range(), Some(image.range()));
        collector.parse(image).unwrap();
    }
    let members = collector.finish().unwrap();
    let plan = members
        .compact(|object| object.index().get() == 11)
        .unwrap();
    let [movement] = plan.moves() else {
        panic!("expected one moved record");
    };
    assert_eq!(movement.object().index().get(), 61_000);
    assert_eq!(
        movement.object().range(),
        FileRange::new(PhysicalAddress::new(base + 32), size - 48).unwrap()
    );
    assert_eq!(
        movement.destination(),
        FileRange::new(PhysicalAddress::new(base + 16), size - 48).unwrap()
    );
    assert_eq!(plan.retained(), 1);
    assert_eq!(
        plan.free_space().unwrap().range(),
        FileRange::new(PhysicalAddress::new(base + size - 32), 32).unwrap()
    );
}

#[rstest]
#[case::incomplete(false)]
#[case::duplicate(true)]
fn incomplete_or_invalid_framing_cannot_produce_members(#[case] duplicate: bool) {
    let mut original = global_heap::collection(
        &[
            global_heap::Object::new(7, b"alpha"),
            global_heap::Object::new(9, b"beta"),
        ],
        Widths::new(8, 8),
    );
    let mut collector =
        GlobalHeapCollectionLocator::new(PhysicalAddress::new(0), LengthWidth::Eight)
            .parse(region(0, &original[..16]))
            .unwrap()
            .collect_members();
    collector.parse(region(16, &original[16..32])).unwrap();
    assert_eq!(
        collector.parse_range(),
        Some(FileRange::new(PhysicalAddress::new(40), 16).unwrap())
    );
    if duplicate {
        bytes::set_u16_at(&mut original, 40, 7);
        assert_eq!(
            collector.parse(region(40, &original[40..56])).unwrap_err(),
            FormatError::DuplicateGlobalHeapObjectIndex {
                collection_address: 0,
                index: 7
            }
        );
    }
    assert!(collector.finish().is_none());
}

#[rstest]
#[case::short(528, 15)]
#[case::misassociated(529, 16)]
fn collector_retries_insufficient_and_misassociated_regions(
    #[case] start: u64,
    #[case] length: usize,
) {
    let original = global_heap::collection(&[], Widths::new(8, 2));
    let mut collector =
        GlobalHeapCollectionLocator::new(PhysicalAddress::new(512), LengthWidth::Two)
            .parse(region(512, &original[..16]))
            .unwrap()
            .collect_members();
    let image = region(start, &original[16..16 + length]);
    assert_eq!(
        collector.parse(image).unwrap_err(),
        FormatError::InsufficientRegion {
            required_start: 528,
            required_length: 16,
            provided_start: image.range().start().get(),
            provided_length: image.range().length(),
        }
    );
    assert_eq!(
        collector.parse_range(),
        Some(FileRange::new(PhysicalAddress::new(528), 16).unwrap())
    );
    collector.parse(region(528, &original[16..32])).unwrap();
    assert_eq!(collector.parse_range(), None);
    assert!(collector.finish().unwrap().members().is_empty());
}

fn collect(image: FileRegion<'_>, lengths: LengthWidth) -> GlobalHeapMembers {
    let locator = GlobalHeapCollectionLocator::new(image.range().start(), lengths);
    let mut collector = locator.parse(image).unwrap().collect_members();
    while let Some(range) = collector.parse_range() {
        collector.parse(image.subregion(range).unwrap()).unwrap();
    }
    collector.finish().unwrap()
}

fn region(start: u64, bytes: &[u8]) -> FileRegion<'_> {
    FileRegion::new(
        FileRange::new(PhysicalAddress::new(start), bytes.len() as u64).unwrap(),
        bytes,
    )
    .unwrap()
}

fn local(range: FileRange, base: u64) -> Range<usize> {
    usize::try_from(range.start().get() - base).unwrap()
        ..usize::try_from(range.end().get() - base).unwrap()
}
