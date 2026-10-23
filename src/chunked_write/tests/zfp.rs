#![cfg(feature = "zfp")]

use super::*;

#[rstest]
#[case::scale_offset(
    FilterKind::ScaleOffset(ScaleOffset::FloatDScale(2), FillAvailability::Defined),
    "scale-offset"
)]
#[case::shuffle(FilterKind::Shuffle, "shuffle")]
#[case::lzf(FilterKind::Lzf, "lzf")]
#[case::deflate(FilterKind::Deflate(DeflateLevel::Six), "deflate")]
fn zfp_conflicts_report_both_filter_names(#[case] other: FilterKind, #[case] name: &str) {
    let options = filtered(&[other, FilterKind::Zfp(ZfpRate::try_from(16.0).unwrap())]);
    let error = options
        .build_pipeline(&f64_ctx(&[64]), FillPattern::ZERO)
        .unwrap_err();
    let FormatError::FilterError(message) = &error else {
        panic!("expected a filter error, got {error:?}");
    };
    assert!(
        message.contains(name) && message.contains("ZFP"),
        "{message}"
    );
}
