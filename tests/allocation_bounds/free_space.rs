use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::FileSpacePageSize;
use hdf5_pure_core::FileSpaceStrategy;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::ManagerKind;
use hdf5_pure_space::__private::PageClassEvidence;
use hdf5_pure_space::__private::PageType;
use hdf5_pure_space::__private::SessionSpace;
use rstest::rstest;
use test_util::allocation;

#[rstest]
#[case::few_extents(16, 1)]
#[case::many_extents(4_096, 1)]
#[case::long_interior(16, 1_000_000_000)]
fn partial_page_queries_borrow_canonical_state(#[case] count: u64, #[case] stride: u64) {
    let space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        1,
        FileSpacePageSize::MIN,
        (0..count).map(|page| {
            (
                Some(ManagerKind::PagedSmallMetadata),
                extent(page * stride * 512 + 400, 16),
            )
        }),
    )
    .unwrap();
    let snapshot = space.admitted_snapshot();
    let query = extent(100, (count - 1) * stride * 512 + 200);
    let (copied, control) =
        allocation::measure(&format!("page_class_control_{count}_{stride}"), || {
            snapshot.sections().collect::<Vec<_>>()
        });
    assert_eq!(u64::try_from(copied.len()).unwrap(), count);
    assert!(
        copied
            .iter()
            .all(|(_, class)| *class == FreeClass::Page(PageType::Meta))
    );
    assert!(control.blocks > 0, "{control:?}");
    assert!(
        control.live_bytes >= count * u64::try_from(size_of::<(Extent, FreeClass)>()).unwrap(),
        "{control:?}"
    );
    let (class, measured) =
        allocation::measure(&format!("partial_page_class_{count}_{stride}"), || {
            snapshot.partial_page_class(query)
        });
    assert_eq!(class, PageClassEvidence::Known(PageType::Meta));
    assert_eq!(measured.blocks, 0, "{measured:?}");
    assert_eq!(measured.bytes, 0, "{measured:?}");
    assert_eq!(measured.peak_bytes, 0, "{measured:?}");
    assert_eq!(measured.live_bytes, 0, "{measured:?}");
}

fn extent(start: u64, length: u64) -> Extent {
    Extent::new(StoredAddress::new(start), ByteLength::new(length)).unwrap()
}
