use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::FileSpacePageSize;
use hdf5_pure_core::StoredAddress;
use rstest::rstest;

use super::PageClassEvidence;
use super::PartialPageClasses;
use crate::extent::Extent;
use crate::list::FreeList;
use crate::paged::PageType;

#[rstest]
#[case::small(16)]
#[case::many(4_096)]
fn partial_page_queries_probe_canonical_lists_logarithmically(#[case] count: u64) {
    let list = FreeList::from_sorted((0..count).map(|page| extent(page * 512 + 400, 16))).unwrap();
    let classes = PartialPageClasses {
        metadata: list.extents(),
        raw: &[],
        page_size: FileSpacePageSize::MIN,
    };
    let query = extent(100, (count - 1) * 512 + 200);
    let mut visits = 0;
    let result = classes.classify(query, || visits += 1);
    assert_eq!(result, PageClassEvidence::Known(PageType::Meta));
    assert!(visits > 0);
    assert!(
        visits <= 4 * (u64::BITS - count.leading_zeros() + 2),
        "{visits} probes for {count} extents"
    );
    assert!(
        u64::from(visits) < count,
        "{visits} probes for {count} extents"
    );
    assert_eq!(u64::try_from(list.extents().len()).unwrap(), count);
}

#[rstest]
#[case::one(1)]
#[case::many(1_000_000_000)]
fn whole_page_interiors_add_no_queries(#[case] pages: u64) {
    let list = FreeList::from_sorted([extent(50, 16), extent(pages * 512 + 400, 16)]).unwrap();
    let classes = PartialPageClasses {
        metadata: list.extents(),
        raw: &[],
        page_size: FileSpacePageSize::MIN,
    };
    let mut visits = 0;
    assert_eq!(
        classes.classify(extent(100, pages * 512 + 200), || visits += 1),
        PageClassEvidence::Known(PageType::Meta)
    );
    assert!(
        (4..=8).contains(&visits),
        "{visits} probes for two partial edges"
    );
}

#[rstest]
#[case::power_of_two(512)]
#[case::other(513)]
fn the_last_address_page_needs_no_representable_following_boundary(#[case] page: u64) {
    let start = u64::MAX / page * page;
    let list = FreeList::from_sorted([extent(start + 450, 16)]).unwrap();
    let classes = PartialPageClasses {
        metadata: list.extents(),
        raw: &[],
        page_size: FileSpacePageSize::try_from(page).unwrap(),
    };
    assert_eq!(
        classes.classify(extent(start + 100, 300), || {}),
        PageClassEvidence::Known(PageType::Meta)
    );
    assert_eq!(
        classes.classify(extent(u64::MAX - 100, 100), || {}),
        PageClassEvidence::Known(PageType::Meta)
    );
}

#[test]
fn aligned_whole_allocations_need_no_partial_class_probes() {
    let list = FreeList::from_sorted((0..4_096).map(|page| extent(page * 512 + 400, 16))).unwrap();
    let classes = PartialPageClasses {
        metadata: list.extents(),
        raw: &[],
        page_size: FileSpacePageSize::MIN,
    };
    let mut visits = 0;
    assert_eq!(
        classes.classify(extent(512, 512_000_000_000), || visits += 1),
        PageClassEvidence::Unknown
    );
    assert_eq!(visits, 0);
    assert_eq!(
        classes.classify(extent(100, 20), || visits += 1),
        PageClassEvidence::Known(PageType::Meta)
    );
    assert!(visits > 0);
}

fn extent(start: u64, length: u64) -> Extent {
    Extent::new(StoredAddress::new(start), ByteLength::new(length)).unwrap()
}
