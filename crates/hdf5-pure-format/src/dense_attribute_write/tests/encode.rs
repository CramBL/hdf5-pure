use alloc::format;
use alloc::vec;

use rstest::rstest;

use crate::address::{StoredAddress, StoredAddressExt};
use crate::dense_attribute_write::{DenseAttrCreationOrder, PreparedDenseAttrs, test_support};

#[rstest]
#[case::managed(
    CanonicalShape::Managed,
    11_054,
    0xc3e7_19d6_79bd_1897,
    0x9c7d_22d3_63fa_dd12
)]
#[case::tracked(
    CanonicalShape::Tracked,
    11_604,
    0xf339_b08a_cd6a_9d44,
    0x6660_3de0_96e5_effe
)]
#[case::mixed(
    CanonicalShape::Mixed,
    73_374,
    0x2dd0_53a6_3871_24e4,
    0xf5b8_cc1e_4fed_c90d
)]
// Expected lengths and hashes were produced at revision 534712d2 (see the test data notice).
fn prepared_dense_output_matches_baseline_bytes(
    #[case] shape: CanonicalShape,
    #[case] len: usize,
    #[case] blob_hash: u64,
    #[case] info_hash: u64,
) {
    let attrs = match shape {
        CanonicalShape::Managed | CanonicalShape::Tracked => (0..10)
            .map(|i| test_support::canonical_attribute(&format!("a{i}"), 512))
            .collect::<Vec<_>>(),
        CanonicalShape::Mixed => vec![
            test_support::canonical_attribute("small", 512),
            test_support::canonical_attribute("huge", 70_000),
            test_support::canonical_attribute("after", 512),
        ],
    };
    let creation = match shape {
        CanonicalShape::Tracked => DenseAttrCreationOrder::Tracked {
            indices: vec![19, 4, 8, 1, 10, 3, 9, 6, 12, 7],
            max: 30,
            indexed: true,
        },
        CanonicalShape::Managed | CanonicalShape::Mixed => DenseAttrCreationOrder::Untracked,
    };
    let prepared = PreparedDenseAttrs::new(&attrs, creation, []).unwrap();
    let ready = prepared
        .finalize(
            |reference| panic!("unexpected reference {reference:?}"),
            |attribute| panic!("unexpected payload {attribute}"),
        )
        .unwrap();
    let built = ready.build(StoredAddress::new(0x8000)).unwrap();
    assert_eq!(
        ready.geometry().huge_count(),
        usize::from(matches!(shape, CanonicalShape::Mixed))
    );
    assert_eq!(built.blob.len(), len);
    assert_eq!(canonical_hash(&built.blob), blob_hash);
    assert_eq!(canonical_hash(&built.attribute_info.to_bytes()), info_hash);
}

#[derive(Clone, Copy, Debug)]
enum CanonicalShape {
    Managed,
    Mixed,
    Tracked,
}

fn canonical_hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    })
}
