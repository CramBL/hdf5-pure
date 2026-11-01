use core::num::NonZeroU16;

use hdf5_pure::{CompactReversion, LinkPhaseChange};
use rstest::rstest;

const THRESHOLDS: LinkPhaseChange = LinkPhaseChange::builder()
    .max_compact::<12>()
    .min_dense::<8>()
    .build();
const ONE_WAY: LinkPhaseChange = LinkPhaseChange::builder()
    .max_compact::<12>()
    .disable_compact_reversion()
    .build();
const EQUAL: LinkPhaseChange = LinkPhaseChange::builder()
    .max_compact::<1>()
    .min_dense::<1>()
    .build();
const MAXIMUM: LinkPhaseChange = LinkPhaseChange::builder()
    .max_compact::<65_535>()
    .min_dense::<65_535>()
    .build();

#[rstest]
#[case::default(LinkPhaseChange::default(), 8, 6)]
#[case::thresholds(THRESHOLDS, 12, 8)]
#[case::one_way(ONE_WAY, 12, 0)]
#[case::always_dense(LinkPhaseChange::AlwaysDense, 0, 0)]
#[case::equal(EQUAL, 1, 1)]
#[case::maximum(MAXIMUM, 65_535, 65_535)]
fn completed_policies_have_effective_thresholds(
    #[case] policy: LinkPhaseChange,
    #[case] max_compact: u16,
    #[case] min_dense: u16,
) {
    assert_eq!(
        (policy.max_compact(), policy.min_dense()),
        (max_compact, min_dense)
    );
    match policy {
        LinkPhaseChange::AlwaysDense => assert_eq!((max_compact, min_dense), (0, 0)),
        LinkPhaseChange::Thresholds(thresholds) => {
            assert_eq!(
                thresholds.max_compact(),
                NonZeroU16::new(max_compact).unwrap()
            );
            assert_eq!(
                thresholds.compact_reversion(),
                match NonZeroU16::new(min_dense) {
                    Some(min_dense) => CompactReversion::Below(min_dense),
                    None => CompactReversion::Disabled,
                }
            );
        }
    }
}
