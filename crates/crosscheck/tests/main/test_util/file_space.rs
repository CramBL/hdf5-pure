//! Paired free-space checkpoints and bundled-library snapshots.

use std::path::Path;

use test_util_hdf5::file_space_oracle;

/// Asserts exact section-map equality and snapshots the paired report with the bundled build.
///
/// # Panics
///
/// Panics if observation fails, section maps or C totals differ, or a bundled snapshot differs.
pub(crate) fn checkpoint(path: &Path, case: &str, step: &str) {
    let observation = file_space_oracle::observe(path, case, step);
    snapshot(
        &format!("{case}_{step}").replace('/', "_"),
        &observation.render(),
    );
}

/// Checks `report` against a named snapshot when using bundled libhdf5.
///
/// Uses the `main__file_space_oracle__` prefix and the suite's snapshot directory independently
/// of this module's path. Has no effect with an external libhdf5 build.
///
/// # Panics
///
/// Panics if snapshot comparison or snapshot I/O fails.
pub(crate) fn snapshot(name: &str, report: &str) {
    #[cfg(feature = "__hdf5-bundled")]
    insta::with_settings!({
        snapshot_path => concat!(env!("CARGO_MANIFEST_DIR"), "/tests/main/snapshots"),
        prepend_module_to_snapshot => false,
    }, {
        insta::assert_snapshot!(format!("main__file_space_oracle__{name}"), report);
    });
    #[cfg(not(feature = "__hdf5-bundled"))]
    let _ = (name, report);
}
