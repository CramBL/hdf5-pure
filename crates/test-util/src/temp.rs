//! Scratch paths that two concurrent runs of the suite cannot collide on
//! (issue #334).
//!
//! A fixture named directly in the shared system temporary directory resolves to
//! the same path in every process, so two runs of the same test binary at once —
//! two agents, two terminals, a `cargo test` beside a `cargo nextest run` — write
//! over each other's files. The tests then fail in ways that read like real
//! defects: measured before this helper landed, two concurrent runs of
//! `edit_free_space` failed 16 and 19 of their 26 tests, every one of which
//! passes alone. [`temp_path`] gives each call a directory of its own, so a
//! fixture name only has to be unique within its test, and removes that
//! directory when the test ends whether it passed, failed, or panicked.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// A fixture path inside a temporary directory owned by the test.
///
/// Stands in for the [`PathBuf`] a test would otherwise hold: it derefs to
/// [`Path`] and implements `AsRef<Path>`, so `&path` reaches both the `&Path`
/// parameters and the `impl AsRef<Path>` ones without a conversion at the call
/// site.
pub struct TempPath {
    /// Held for its `Drop`, which removes the directory and everything written
    /// inside it. Never read — the `path` beside it already names the fixture.
    _dir: tempfile::TempDir,
    path: PathBuf,
}

/// A path named `name` in a temporary directory created for this call.
///
/// Bind the result for as long as the file is needed. The directory is removed
/// when the returned value drops, so passing this straight into a call rather
/// than through a `let` leaves nothing behind for the next statement to open.
pub fn temp_path(name: &str) -> TempPath {
    let dir = tempfile::tempdir().expect("create a temporary directory for a fixture");
    fixture_in(dir, name)
}

/// The same, with the temporary directory created inside `parent`.
///
/// For the tests that keep their fixtures in the repository's gitignored `tmp/`
/// rather than the system temporary directory. `parent` has to exist.
pub fn temp_path_in(parent: &Path, name: &str) -> TempPath {
    let dir = tempfile::TempDir::new_in(parent).unwrap_or_else(|e| {
        panic!(
            "create a temporary directory under {}: {e}",
            parent.display()
        )
    });
    fixture_in(dir, name)
}

fn fixture_in(dir: tempfile::TempDir, name: &str) -> TempPath {
    let path = dir.path().join(name);
    TempPath { _dir: dir, path }
}

impl std::ops::Deref for TempPath {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.path
    }
}

impl AsRef<Path> for TempPath {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

impl std::fmt::Debug for TempPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.path, f)
    }
}
