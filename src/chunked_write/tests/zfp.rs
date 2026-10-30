#![cfg(feature = "zfp")]

use super::*;

#[rstest]
#[case::scale_offset(
    FilterKind::ScaleOffset(ScaleOffset::FloatDScale(2), FillAvailability::Defined),
    "scale-offset and ZFP cannot be combined on one dataset"
)]
#[case::shuffle(
    FilterKind::Shuffle,
    "shuffle and ZFP cannot be combined on one dataset"
)]
#[case::lzf(FilterKind::Lzf, "lzf and ZFP cannot be combined on one dataset")]
#[case::deflate(
    FilterKind::Deflate(DeflateLevel::Six),
    "deflate and ZFP cannot be combined on one dataset"
)]
fn zfp_conflicts_report_both_filter_names(#[case] other: FilterKind, #[case] expected: &str) {
    let options = filtered(&[other, FilterKind::Zfp(ZfpRate::try_from(16.0).unwrap())]);
    let error = options
        .build_pipeline(&f64_ctx(&[64]), FillPattern::ZERO)
        .unwrap_err();
    assert_eq!(error, FormatError::FilterError(expected.into()));
}
