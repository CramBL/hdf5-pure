use core::num::NonZeroU64;

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::{
    AllocationClass, Extent, PageType, ReservationRequest, ReservationWork, SessionSpace,
};
use rstest::rstest;
use test_util::snapshot::{Report, Value};

use crate::address::{StoredAddress, StoredAddressExt};
use crate::edit::block::{BlockRequest, EncodingDependency};
use crate::edit::observations::{self, ZeroWork};
use crate::edit::test_util as edit_test_util;
use crate::free_space_manager::{ManagerProjection, ManagerWork};
use crate::image::WriteBuffering;
use crate::source::Source;
use crate::{FileBuilder, FileSpacePageSize, FileSpaceSettings, FileSpaceStrategy, UserblockSize};

#[derive(Clone, Copy)]
enum Layout {
    Appended,
    Fragmented,
    Paged,
    PagedUserblock,
}

#[rstest]
#[case::appended(Layout::Appended, "foundation_reservations_appended")]
#[case::fragmented(Layout::Fragmented, "foundation_reservations_fragmented")]
#[case::paged(Layout::Paged, "foundation_reservations_paged")]
#[case::paged_userblock(Layout::PagedUserblock, "foundation_reservations_paged_userblock")]
fn foundation_reservations_preserve_capacity_geometry_and_contents(
    #[case] layout: Layout,
    #[case] snapshot: &str,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reservations.h5");
    let (strategy, base) = match layout {
        Layout::Appended => (FileSpaceStrategy::None, 0),
        Layout::Fragmented => (FileSpaceStrategy::FsmAggr, 0),
        Layout::Paged => (FileSpaceStrategy::Page, 0),
        Layout::PagedUserblock => (FileSpaceStrategy::Page, 4096),
    };
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(base).unwrap());
    builder.with_file_space_settings(
        FileSpaceSettings::new(strategy).with_persistence(strategy == FileSpaceStrategy::Page),
    );
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine.aggregation = None;
    if matches!(layout, Layout::Paged | Layout::PagedUserblock) {
        engine.space = SessionSpace::from_strategy(Some(strategy), 1, FileSpacePageSize::DEFAULT);
    }
    let start = engine.image.len();
    if matches!(layout, Layout::Fragmented) {
        engine.append(&[13; 512]).unwrap();
        engine.space = SessionSpace::from_persisted(
            Some(strategy),
            1,
            FileSpacePageSize::DEFAULT,
            [(None, extent(start, 128)), (None, extent(start + 256, 128))],
        )
        .unwrap();
    }
    engine
        .set_write_buffering(WriteBuffering::Session {
            page_size: 256,
            max_bytes: 1024,
        })
        .unwrap();
    let original_len = engine.image.len();
    let mut expected = engine
        .image()
        .read_exact_at(0, usize::try_from(original_len).unwrap())
        .unwrap();
    let before = engine.image.buffering_work();
    observations::reset_zeros();
    let requests = [
        request(64, 16, PageType::Raw),
        request(48, 48, PageType::Meta),
        request(80, 32, PageType::Raw),
    ];
    let addresses = engine.addresses();
    let mut plan = engine
        .plan_blocks(&requests)
        .unwrap_or_else(|failure| panic!("planning failed: {:?}", failure.error));
    let end = plan.end().get();
    let placements: Vec<_> = plan
        .reservations()
        .iter()
        .zip(&requests)
        .map(|(entry, request)| {
            let range = entry.range();
            (
                addresses.relative(range.start()).unwrap().get(),
                range.start().get(),
                range.length(),
                request.encoded_length(),
                entry.request().page_type(),
            )
        })
        .collect();
    let expected_offsets: &[u64] = match layout {
        Layout::Appended => &[0, 64, 112],
        Layout::Fragmented => &[0, 64, 256],
        Layout::Paged | Layout::PagedUserblock => &[0, 4096, 8192],
    };
    assert_eq!(
        placements.iter().map(|entry| entry.1).collect::<Vec<_>>(),
        expected_offsets
            .iter()
            .map(|offset| start + offset)
            .collect::<Vec<_>>()
    );
    expected.resize(usize::try_from(end).unwrap().max(expected.len()), 0);
    for (index, request) in requests.iter().enumerate() {
        let bytes = vec![
            u8::try_from(7 + index).unwrap();
            usize::try_from(request.encoded_length()).unwrap()
        ];
        let at = plan.write_next(&bytes).unwrap().get();
        let local = usize::try_from(at).unwrap();
        expected[local..local + bytes.len()].copy_from_slice(&bytes);
    }
    let work = plan.finish().unwrap();
    assert_eq!(engine.image.len(), end);
    assert_eq!(
        engine.image().read_exact_at(0, expected.len()).unwrap(),
        expected
    );
    if matches!(layout, Layout::Fragmented) {
        assert_eq!(
            engine.space.reusable_sections().collect::<Vec<_>>(),
            [extent(start + 112, 16), extent(start + 336, 48)]
        );
        assert_eq!(work.indexed_extents, 2);
        assert_eq!(engine.space.reusable_bytes().unwrap(), 64);
    } else {
        assert_eq!(work.indexed_extents, 0);
    }
    engine.image.flush().unwrap();
    let after = engine.image.buffering_work();
    let zeros = observations::zero_work();
    assert_eq!(
        zeros,
        match layout {
            Layout::Appended => ZeroWork {
                calls: 2,
                bytes: 96,
                scratch_peak: 48
            },
            Layout::Fragmented => ZeroWork::default(),
            Layout::Paged | Layout::PagedUserblock => ZeroWork {
                calls: 4,
                bytes: 8176,
                scratch_peak: 4048
            },
        }
    );
    let projection =
        ManagerProjection::new(engine.space.admitted_snapshot(), engine.widths()).unwrap();
    let manager_work = projection.work();
    assert_eq!(
        manager_work.managers,
        u64::from(matches!(layout, Layout::Fragmented))
    );
    assert_eq!(
        manager_work.sections,
        if matches!(layout, Layout::Fragmented) {
            2
        } else {
            0
        }
    );
    let manager_start = addresses.relative(PhysicalAddress::new(end)).unwrap();
    let manager_plan = projection.into_plan(manager_start).unwrap();
    let manager_end = manager_plan.end_of_managers;
    let encoded = manager_plan
        .encode(manager_end)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let encoded_len: usize = encoded
        .iter()
        .map(|manager| manager.header.len() + manager.section_list.len())
        .sum();
    let encoded_capacity: usize = encoded
        .iter()
        .map(|manager| manager.header.capacity() + manager.section_list.capacity())
        .sum();
    assert_eq!(encoded_len as u64, manager_work.encoded_bytes);
    assert!(encoded_capacity >= encoded_len);
    let sections: Vec<_> = engine
        .space
        .admitted_snapshot()
        .sections()
        .map(|(extent, class)| (extent.start().get(), extent.len(), class))
        .collect();
    let mut report = Report::new();
    report.metrics(
        "File geometry",
        "Bytes",
        [
            ("Address base", base),
            ("Original image", original_len),
            ("Planned end", end),
            ("File growth", end - original_len),
        ],
    );
    report.table(
        "Reservations",
        [
            "Stored address",
            "Physical address",
            "Allocation bytes",
            "Encoding bytes",
            "PAGE type",
        ],
        placements
            .iter()
            .map(|&(stored, physical, allocation, encoding, page)| {
                [
                    Value::Number(stored),
                    Value::Number(physical),
                    Value::Number(allocation),
                    Value::Number(encoding),
                    Value::text(format!("{page:?}")),
                ]
            }),
    );
    let ReservationWork {
        indexed_extents,
        lookups,
        removed_extents,
        inserted_extents,
        restored_extents,
    } = work;
    report.metrics(
        "Reservation work",
        "Count",
        [
            ("Indexed extents", indexed_extents),
            ("Lookups", lookups),
            ("Removed extents", removed_extents),
            ("Inserted extents", inserted_extents),
            ("Restored extents", restored_extents),
        ],
    );
    report.table(
        "Free sections",
        ["Stored address", "Bytes", "Class"],
        sections.iter().map(|&(address, length, class)| {
            [
                Value::Number(address),
                Value::Number(length),
                Value::text(format!("{class:?}")),
            ]
        }),
    );
    let ManagerWork {
        managers,
        sections,
        section_capacity,
        encoded_bytes,
    } = manager_work;
    report.metrics(
        "Manager projection",
        "Value",
        [
            ("Managers (count)", managers),
            ("Sections (count)", sections),
            ("Section capacity (elements)", section_capacity),
            ("Measured encoding (bytes)", encoded_bytes),
            (
                "Encoded metadata (bytes)",
                u64::try_from(encoded_len).unwrap(),
            ),
            (
                "Encoded metadata capacity (bytes)",
                u64::try_from(encoded_capacity).unwrap(),
            ),
        ],
    );
    let ZeroWork {
        calls,
        bytes,
        scratch_peak,
    } = zeros;
    report.metrics(
        "Zero generation",
        "Value",
        [
            ("Requests (count)", calls),
            ("Requested bytes", bytes),
            ("Scratch peak (bytes)", scratch_peak),
        ],
    );
    report.metrics(
        "Image I/O",
        "Value",
        [
            (
                "Issued writes (calls)",
                after.write_calls - before.write_calls,
            ),
            (
                "Issued writes (bytes)",
                after.write_bytes - before.write_bytes,
            ),
            (
                "Clean-gap reads (bytes)",
                after.clean_gap_bytes - before.clean_gap_bytes,
            ),
            (
                "Explicit buffer copies (bytes)",
                after.copied_bytes - before.copied_bytes,
            ),
        ],
    );
    let report = report.finish();
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), expected);
    insta::assert_snapshot!(snapshot, report);
}

#[test]
fn aggregate_zero_observations_include_complete_scratch_capacity() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("aggregate.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Aggr));
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    let start = engine.image.len();
    observations::reset_zeros();
    assert_eq!(
        engine
            .alloc_or_append_typed(&[7], AllocationClass::Raw)
            .unwrap(),
        start
    );
    assert_eq!(
        observations::zero_work(),
        ZeroWork {
            calls: 1,
            bytes: 2047,
            scratch_peak: 4096
        }
    );
    assert_eq!(engine.image.len(), start + 2048);
    let mut expected = vec![0; 2048];
    expected[0] = 7;
    assert_eq!(engine.image().read_exact_at(start, 2048).unwrap(), expected);
}

fn extent(start: u64, length: u64) -> Extent {
    Extent::new(StoredAddress::new(start), length).unwrap()
}

fn request(allocation: u64, encoding: u64, page_type: PageType) -> BlockRequest {
    let class = match page_type {
        PageType::Meta => AllocationClass::Metadata,
        PageType::Raw => AllocationClass::Raw,
    };
    BlockRequest::new(
        ReservationRequest::with_page_type(NonZeroU64::new(allocation).unwrap(), class, page_type),
        encoding,
        EncodingDependency::Independent,
    )
    .unwrap()
}
