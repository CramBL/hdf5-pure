use rstest::rstest;

use hdf5_pure_core::{FileSpacePageSize, FileSpaceSettings, FileSpaceStrategy, FormatError};

#[rstest]
#[case::fsm_aggr(FileSpaceStrategy::FsmAggr)]
#[case::page(FileSpaceStrategy::Page)]
#[case::aggr(FileSpaceStrategy::Aggr)]
#[case::none(FileSpaceStrategy::None)]
fn constructors_use_default_manager_settings_and_page_size(#[case] strategy: FileSpaceStrategy) {
    let settings = FileSpaceSettings::new(strategy);
    assert_eq!(settings.strategy(), strategy);
    assert!(!settings.persist());
    assert_eq!(settings.threshold(), 1);
    assert_eq!(settings.page_size(), FileSpacePageSize::DEFAULT);
}

#[test]
fn default_uses_fsm_aggr() {
    assert_eq!(
        FileSpaceSettings::default(),
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
    );
}

#[rstest]
#[case::fsm_aggr(FileSpaceStrategy::FsmAggr, true)]
#[case::page(FileSpaceStrategy::Page, true)]
#[case::aggr(FileSpaceStrategy::Aggr, false)]
#[case::none(FileSpaceStrategy::None, false)]
fn setters_normalize_manager_settings_and_retain_page_size(
    #[case] strategy: FileSpaceStrategy,
    #[case] has_managers: bool,
    #[values(false, true)] persist: bool,
    #[values(0, 1, 128, u64::MAX)] threshold: u64,
) {
    let page_size = FileSpacePageSize::try_from(8192).unwrap();
    let settings = FileSpaceSettings::new(strategy)
        .with_persistence(persist)
        .with_threshold(threshold)
        .with_page_size(page_size);
    assert_eq!(settings.strategy(), strategy);
    assert_eq!(settings.persist(), has_managers && persist);
    assert_eq!(
        settings.threshold(),
        if has_managers { threshold } else { 1 }
    );
    assert_eq!(settings.page_size(), page_size);
    assert_eq!(
        settings,
        FileSpaceSettings::new(strategy)
            .with_page_size(page_size)
            .with_threshold(threshold)
            .with_persistence(persist)
    );
    assert_eq!(
        settings.with_persistence(false).with_threshold(1),
        FileSpaceSettings::new(strategy).with_page_size(page_size)
    );
}

#[rstest]
#[case::minimum(FileSpacePageSize::MIN.get())]
#[case::irregular(3000)]
#[case::default(FileSpacePageSize::DEFAULT.get())]
#[case::maximum(FileSpacePageSize::MAX.get())]
fn settings_accept_checked_page_sizes(#[case] size: u64) {
    let settings =
        FileSpaceSettings::default().with_page_size(FileSpacePageSize::try_from(size).unwrap());
    assert_eq!(settings.page_size().get(), size);
}

#[rstest]
#[case::zero(0)]
#[case::below_minimum(FileSpacePageSize::MIN.get() - 1)]
#[case::above_maximum(FileSpacePageSize::MAX.get() + 1)]
fn page_size_construction_rejects_values_outside_supported_range(#[case] size: u64) {
    assert_eq!(
        FileSpacePageSize::try_from(size).unwrap_err(),
        FormatError::InvalidFileSpacePageSize(size)
    );
}

#[test]
fn settings_support_const_construction_and_accessors() {
    const SETTINGS: FileSpaceSettings = FileSpaceSettings::new(FileSpaceStrategy::Page)
        .with_persistence(true)
        .with_threshold(0)
        .with_page_size(FileSpacePageSize::MAX);
    const VALUES: (FileSpaceStrategy, bool, u64, FileSpacePageSize) = (
        SETTINGS.strategy(),
        SETTINGS.persist(),
        SETTINGS.threshold(),
        SETTINGS.page_size(),
    );
    assert_eq!(
        VALUES,
        (FileSpaceStrategy::Page, true, 0, FileSpacePageSize::MAX)
    );
}
