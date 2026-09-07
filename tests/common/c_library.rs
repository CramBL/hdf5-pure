//! Helpers that call into the reference HDF5 C library.
//!
//! Split from `common/mod.rs` so the 32-bit gate lives in one `#[cfg]` on the
//! module rather than on every item, and so pure-Rust tests can use
//! [`super::heap`] without pulling the C dev-dependency in.

use hdf5::MinorErrorCode;

/// A file the C library creates in the 1.8 format or newer: version 2 object
/// headers and link-message groups. HDF5 2.0 made that the default, and every
/// earlier release writes version 1 headers unless told otherwise.
#[cfg(feature = "__hdf5-1.10")]
pub fn create_v18(path: &std::path::Path) -> hdf5::File {
    use hdf5::file::LibraryVersion;
    hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V18, LibraryVersion::latest()))
        .create(path)
        .unwrap_or_else(|e| panic!("create {}: {e}", path.display()))
}

/// Minor codes that mean the C library failed to *read* a metadata block, as
/// opposed to reading it successfully and finding nothing.
const LOAD_FAILURES: [MinorErrorCode; 6] = [
    MinorErrorCode::CantLoad,
    MinorErrorCode::ReadError,
    MinorErrorCode::CantProtect,
    MinorErrorCode::CantUnprotect,
    MinorErrorCode::CantGet,
    MinorErrorCode::CantDecode,
];

/// Assert that `err` is the C library reporting `what` **absent** from a file it
/// could otherwise traverse — not merely failing for some unrelated reason.
///
/// This is the assertion a delete test wants, and `is_err()` is not it: a file
/// so damaged the C library cannot walk the group at all also fails, so
/// `is_err()` passes on the corruption a delete was supposed to avoid. That is
/// the [#201] failure mode — a write path returning `Ok` on a file the C library
/// cannot read.
///
/// `H5E_NOTFOUND` alone is not it either. A damaged link table reports
/// `NotFound` too, because an unreadable index and a missing link are
/// indistinguishable at the traversal layer. Sweeping single-byte through
/// 32-byte corruptions across every metadata offset of a 40-dataset file, 885
/// produced a detectable failure and **176 of those carried `NotFound`**.
///
/// What separates the two is the metadata cache: a genuine absence reads every
/// block it needs and finds no matching link, so it never reports a load or
/// decode failure. Adding that conjunction took the same sweep from 176 false
/// accepts to zero, while still accepting a genuine absence in a compact group,
/// a dense group, a subgroup, a missing group, and a missing attribute.
///
/// `tests/c_absence_predicate.rs` guards that this stays able to tell the two
/// apart; read it before changing either condition.
///
/// [#201]: https://github.com/stephenberry/hdf5-pure/issues/201
#[track_caller]
pub fn assert_c_absent(err: &hdf5::Error, what: &str) {
    if let Some(reason) = not_absent_because(err) {
        let stack: Vec<String> = match err.stack() {
            Some(s) => s.minor_codes().map(|c| format!("{c:?}")).collect(),
            None => vec!["<no HDF5 error stack>".to_string()],
        };
        panic!("`{what}` {reason}: {err}\n  minor codes: {stack:?}");
    }
}

/// Whether the C library reports a genuine absence, as a `bool`.
///
/// This and [`assert_c_absent`] must share one implementation, or the guard test
/// in `tests/c_absence_predicate.rs` would be measuring a copy of the predicate
/// rather than the one the delete crosschecks actually rely on.
pub fn c_reports_absent(err: &hdf5::Error) -> bool {
    not_absent_because(err).is_none()
}

/// `None` when `err` is a genuine absence, otherwise why it is not.
fn not_absent_because(err: &hdf5::Error) -> Option<String> {
    if !err.contains_minor(MinorErrorCode::NotFound) {
        return Some(
            "was expected to be absent, but the C library did not report it \
                     as not found"
                .to_string(),
        );
    }
    for code in LOAD_FAILURES {
        if err.contains_minor(code) {
            return Some(format!(
                "is reported missing, but the C library also failed to read metadata \
                 ({code:?}) — that is damage, not absence"
            ));
        }
    }
    None
}
