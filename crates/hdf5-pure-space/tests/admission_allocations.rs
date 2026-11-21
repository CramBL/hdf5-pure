//! Allocation bounds for returning fragmented free space.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use std::hint::black_box;
use std::mem::size_of;

use hdf5_pure_core::__private::{ByteLength, StoredAddressExt};
use hdf5_pure_core::{FileSpacePageSize, FileSpaceStrategy, StoredAddress};
use hdf5_pure_space::__private::{
    Extent, FreeClass, FreeList, ManagerKind, PageType, PagedPostFree, SessionSpace, TrackedSpace,
};
use rstest::rstest;
use test_util::allocation::{Measured, measure};

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

const PAGE: u64 = FileSpacePageSize::DEFAULT.get();
const RAW: FreeClass = FreeClass::Page(PageType::Raw);

#[rstest]
#[case::sorted(64, false)]
#[case::large(1_024, false)]
#[case::reversed(1_024, true)]
fn flat_return_allocations_scale_with_fragmented_extents(
    #[case] count: u64,
    #[case] reverse: bool,
) {
    let FragmentedReturns {
        tracked,
        mut returned,
        expected,
    } = fragmented_returns(count, 8);
    if reverse {
        returned.reverse();
    }
    let mut list = FreeList::from_sorted(tracked).unwrap();
    let region = format!("flat_fragmented_{count}_{reverse}");
    let limit = 2_048 + 176 * count;
    let (dropped, usage) = measure(&region, || {
        list.release_each(
            ByteLength::new(16),
            returned.iter().map(|&extent| (extent, ())),
        )
    });
    assert!(dropped.is_empty());
    assert_eq!(list.extents(), expected);
    assert_eq!(list.total_bytes(), total(&expected));
    assert_allocations(usage, expected.len(), limit, 48);
    if count == 1_024 && !reverse {
        assert_rebuilding_pending_exceeds_bounds(&region, &returned, limit, 48);
    }
}

#[rstest]
#[case::sorted(32, false)]
#[case::large(1_024, false)]
#[case::reversed(1_024, true)]
fn page_return_allocations_scale_with_fragmented_extents(
    #[case] count: u64,
    #[case] reverse: bool,
) {
    let FragmentedReturns {
        tracked,
        mut returned,
        expected,
    } = fragmented_returns(count, PAGE);
    if reverse {
        returned.reverse();
    }
    let space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        ByteLength::new(3 * PAGE),
        FileSpacePageSize::DEFAULT,
        tracked
            .iter()
            .map(|&extent| (Some(ManagerKind::PagedLargeGeneric), extent)),
    )
    .unwrap();
    let eoa = StoredAddress::new((2 * count + 6) * PAGE);
    let region = format!("paged_fragmented_{count}_{reverse}");
    let limit = 4_096 + 512 * count;
    let (result, usage) = measure(&region, || {
        space.paged_post_free(eoa, returned.iter().map(|&extent| (extent, RAW)), [])
    });
    assert_page_sections(result, eoa, &expected);
    assert_allocations(usage, expected.len(), limit, 72 + count / 8);
    if count == 1_024 && !reverse {
        assert_rebuilding_pending_exceeds_bounds(&region, &returned, limit, 72 + count / 8);
    }
}

#[rstest]
#[case::one_page(1)]
#[case::billion_pages(1_000_000_000)]
fn whole_page_returns_do_not_allocate_by_page_count(#[case] pages: u64) {
    let space = SessionSpace::from_strategy(
        Some(FileSpaceStrategy::Page),
        ByteLength::new(1),
        FileSpacePageSize::DEFAULT,
    );
    let run = extent(PAGE, pages * PAGE);
    let isolated = extent((pages + 3) * PAGE, PAGE);
    let eoa = StoredAddress::new((pages + 10) * PAGE);
    let (result, usage) = measure(&format!("paged_run_{pages}"), || {
        space.paged_post_free(eoa, [(run, PageType::Meta.into()), (isolated, RAW)], [])
    });
    assert_page_sections(result, eoa, &[run, isolated]);
    assert_allocations(usage, 2, 4_096, 24);
}

struct FragmentedReturns {
    tracked: Vec<Extent>,
    returned: Vec<Extent>,
    expected: Vec<Extent>,
}

fn fragmented_returns(count: u64, size: u64) -> FragmentedReturns {
    let seed = (count + 1) * size;
    let tracked: Vec<_> = std::iter::once(extent(seed, size))
        .chain((0..count / 2).map(|index| extent(seed + (3 + 2 * index) * size, size)))
        .collect();
    let returned = (1..=count)
        .map(|index| extent(index * size, size))
        .collect();
    let mut expected = tracked.clone();
    expected[0] = extent(size, (count + 1) * size);
    FragmentedReturns {
        tracked,
        returned,
        expected,
    }
}

fn assert_page_sections(
    result: Option<(PagedPostFree, StoredAddress)>,
    eoa: StoredAddress,
    expected: &[Extent],
) {
    let (post, result_eoa) = result.unwrap();
    assert_eq!(result_eoa, eoa);
    let sections = post.sections();
    assert!(sections.metadata().is_empty());
    assert_eq!(sections.raw(), expected);
    assert!(sections.unclassified().is_empty());
    assert_eq!(
        post.admitted_snapshot().reusable_bytes().unwrap(),
        total(expected)
    );
}

fn assert_allocations(usage: Measured, retained_extents: usize, byte_limit: u64, count_limit: u64) {
    let output_bytes = (retained_extents * size_of::<Extent>()) as u64;
    assert!(usage.live_bytes >= output_bytes, "{usage}");
    assert!(usage.bytes <= byte_limit, "{usage}");
    assert!(usage.blocks <= count_limit, "{usage}");
}

fn assert_rebuilding_pending_exceeds_bounds(
    region: &str,
    returned: &[Extent],
    byte_limit: u64,
    count_limit: u64,
) {
    // A low-to-high scan admits one high-end neighbor per pass in this fixture.
    let ((), usage) = measure(&format!("{region}_rebuild"), || {
        for pending in (1..=returned.len()).rev() {
            black_box(returned[..pending].to_vec());
        }
    });
    assert!(
        usage.bytes > byte_limit && usage.blocks > count_limit,
        "{usage}"
    );
}

fn total(extents: &[Extent]) -> ByteLength {
    ByteLength::new(extents.iter().map(|extent| extent.len().get()).sum())
}

fn extent(start: u64, len: u64) -> Extent {
    Extent::new(StoredAddress::new(start), ByteLength::new(len)).unwrap()
}
