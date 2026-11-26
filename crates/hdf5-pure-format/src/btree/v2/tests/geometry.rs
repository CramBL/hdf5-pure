use alloc::vec;

use rstest::rstest;

use super::*;
use crate::width::OffsetWidth;

#[rstest]
#[case::minimum_node(10, 1)]
#[case::zero_internal_capacity(20, 1)]
#[case::ordinary(512, 11)]
#[case::large(4096, 8)]
#[case::maximum_node(u32::MAX, 1)]
#[case::record_exceeds_leaf(512, u16::MAX)]
fn collection_geometry_stabilizes_without_depth_sized_tables(
    #[case] node_size: u32,
    #[case] record_size: u16,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let compact = BTreeV2CollectionInfo::compute(node_size, record_size, width, u16::MAX);
    let reference = BTreeV2NodeInfo::compute(node_size, record_size, width, 255);
    let levels = compact.info.max_nrec.len();
    assert!(levels > 1);
    assert!(levels <= usize::try_from(u64::BITS).unwrap() + 3);
    assert_eq!(compact.info.cum_max_nrec.len(), levels);
    assert_eq!(compact.info.cum_max_nrec_size.len(), levels);
    for depth in 1..=255 {
        assert_eq!(
            compact.count_widths(depth),
            BTreeV2CountWidths {
                direct: reference.max_nrec_size,
                subtree: reference.total_nrec_size(depth),
            }
        );
    }
    assert_eq!(compact.count_widths(u16::MAX), compact.count_widths(255));
}

#[test]
fn node_info_matches_hdf5_widths() {
    // The name index of a dense group: 512-byte nodes, 11-byte records, and 8-byte offsets,
    // with the widths computed by hand from `H5B2__hdr_init`. A child pointer of a depth-3
    // root is 11 bytes: the address (8), `max_nrec_size` (1), and `cum_max_nrec_size[2]` (2),
    // since `cum_max_nrec[2]` is 26449.
    let ni = BTreeV2NodeInfo::compute(512, 11, OffsetWidth::Eight, 3);
    assert_eq!(ni.max_nrec_size, 1);
    assert_eq!(ni.cum_max_nrec_size, vec![0, 2, 2, 3]);
    let geometry = BTreeV2Geometry::Indexed(ni);
    assert_eq!(
        geometry
            .count_widths(1)
            .unwrap()
            .pointer_size(OffsetWidth::Eight),
        9
    ); // depth-1 children are leaves
    assert_eq!(
        geometry
            .count_widths(2)
            .unwrap()
            .pointer_size(OffsetWidth::Eight),
        11
    );
    assert_eq!(
        geometry
            .count_widths(3)
            .unwrap()
            .pointer_size(OffsetWidth::Eight),
        11
    );

    // A leaf of (4096 - 10) / 8 = 510 records needs a 2-byte record count.
    let big = BTreeV2NodeInfo::compute(4096, 8, OffsetWidth::Eight, 1);
    assert_eq!(big.max_nrec_size, 2);

    let (node_info, depth) =
        BTreeV2NodeInfo::for_record_count(512, 17, OffsetWidth::Eight, 30).unwrap();
    assert_eq!(depth, 1);
    assert_eq!(node_info.max_nrec(0), 29);
}
