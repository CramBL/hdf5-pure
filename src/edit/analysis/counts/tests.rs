use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::ObjectReferenceCount;
use rstest::rstest;
use test_util::object_header::Message;
use test_util_hdf5::global_heap_fixture::Fixture;

use super::{CountObservation, HeaderCountAllocation, IncomingLinks};
use crate::edit::test_util as edit_test_util;
use crate::object_graph::{ObjectId, ObjectIndex};

#[rstest]
#[case::empty(None, 0, 1)]
#[case::one(Some(1), 1, 2)]
#[case::maximum(Some(u32::MAX), u32::MAX, u64::from(u32::MAX) + 1)]
fn root_count_minima_preserve_the_full_visible_count_range(
    #[case] incoming: Option<u32>,
    #[case] visible: u32,
    #[case] minimum: u64,
) {
    let count = IncomingLinks::new(ObjectIndex::ROOT, incoming).unwrap();
    assert_eq!(count.visible(), visible);
    assert_eq!(count.minimum(), minimum);
}

#[rstest]
#[case::empty(AllocationGeometry::Empty)]
#[case::field_only(AllocationGeometry::FieldOnly)]
#[case::duplicate(AllocationGeometry::Duplicate)]
#[case::overlap(AllocationGeometry::Overlap)]
#[case::missing(AllocationGeometry::Missing)]
#[case::truncated(AllocationGeometry::Truncated)]
#[case::zero_length(AllocationGeometry::ZeroLength)]
#[case::unordered(AllocationGeometry::Unordered)]
fn header_allocation_summary_requires_complete_disjoint_bounded_chunks(
    #[case] geometry: AllocationGeometry,
) {
    let path = test_util::temp::temp_path("count-allocation-geometry.h5");
    let mut fixture = Fixture::new();
    let address = fixture.append_header(vec![Message::reference_count(2)]);
    fixture.link("a", address);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let ObjectReferenceCount::Version2(field) =
        CountObservation::read(&engine, ObjectId::new(PhysicalAddress::new(address)))
            .unwrap()
            .count
    else {
        panic!("expected version 2 framing")
    };
    let mut eof = field.chunk().end().get() + 8;
    let extra = FileRange::new(field.chunk().end(), 8).unwrap();
    let mut allocations = match geometry {
        AllocationGeometry::Duplicate => vec![field.chunk(), field.chunk()],
        AllocationGeometry::Empty => vec![],
        AllocationGeometry::FieldOnly => vec![field.range()],
        AllocationGeometry::Missing => vec![extra],
        AllocationGeometry::Overlap => vec![
            field.chunk(),
            FileRange::new(field.chunk().start().checked_add(1).unwrap(), 1).unwrap(),
        ],
        AllocationGeometry::Truncated => {
            eof = field.chunk().end().get() - 1;
            vec![field.chunk()]
        }
        AllocationGeometry::Unordered => vec![extra, field.chunk()],
        AllocationGeometry::ZeroLength => vec![
            field.chunk(),
            FileRange::new(field.chunk().end(), 0).unwrap(),
        ],
    };
    let owned = HeaderCountAllocation::new(field, &mut allocations, eof);
    assert_eq!(
        owned.map(HeaderCountAllocation::range),
        matches!(geometry, AllocationGeometry::Unordered).then_some(field.chunk())
    );
    assert_eq!(field.value(), 2);
}

#[derive(Clone, Copy)]
enum AllocationGeometry {
    Duplicate,
    Empty,
    FieldOnly,
    Missing,
    Overlap,
    Truncated,
    Unordered,
    ZeroLength,
}
