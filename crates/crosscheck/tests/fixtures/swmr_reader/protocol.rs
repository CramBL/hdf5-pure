//! Requests and observations for native reads of the rank-one dataset `d`.
//!
//! The fixture accepts a file path and a JSON-encoded [`Mode`] as arguments. It
//! writes and flushes [`START`] before opening the file. After reading the dataset
//! and closing the file, it writes one JSON-encoded [`Observation`]. Native and
//! infrastructure failures terminate the fixture unsuccessfully.

use serde::{Deserialize, Serialize};

/// The stdout prefix written before opening the native file.
pub(crate) const START: &[u8] = b"HDF5_SWMR_READER\n";

/// The native file-open mode requested by the parent.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) enum Mode {
    /// A normal read-only open.
    Normal,
    /// A SWMR read with a dataset refresh before observation.
    Swmr,
}

/// The open mode, shape, and complete [`i32`] values from one native reader process.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Observation {
    pub(crate) mode: Mode,
    pub(crate) shape: [usize; 1],
    pub(crate) values: Vec<i32>,
}

#[cfg(test)]
/// An invalid reader outcome with the child's exit status and captured diagnostics.
#[derive(Debug)]
pub(crate) struct Failure {
    pub(crate) kind: FailureKind,
    output: std::process::Output,
}

#[cfg(test)]
/// The reason a child's output cannot be used as a native observation.
#[derive(Debug)]
pub(crate) enum FailureKind {
    Exit,
    Handshake,
    Json(serde_json::Error),
    Mode,
    Shape,
}

#[cfg(test)]
impl std::fmt::Display for Failure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            FailureKind::Json(error) => write!(formatter, "invalid reader JSON: {error}")?,
            kind => write!(formatter, "{kind:?}")?,
        }
        write!(
            formatter,
            "; status: {status}; stdout: {stdout:?}; stderr: {stderr:?}",
            status = self.output.status,
            stdout = String::from_utf8_lossy(&self.output.stdout),
            stderr = String::from_utf8_lossy(&self.output.stderr),
        )
    }
}

#[cfg(test)]
/// Parses a native reader's completed output for `mode`.
///
/// The shape must match the value count. The caller checks the expected shape and
/// values after parsing.
///
/// # Errors
///
/// Returns [`Failure`] for an unsuccessful exit, a missing handshake, malformed
/// JSON, a different open mode, or a shape that differs from the value count.
pub(crate) fn observe(mode: Mode, output: std::process::Output) -> Result<Observation, Failure> {
    if !output.status.success() {
        return Err(Failure {
            kind: FailureKind::Exit,
            output,
        });
    }
    let Some(body) = output.stdout.strip_prefix(START) else {
        return Err(Failure {
            kind: FailureKind::Handshake,
            output,
        });
    };
    let observation: Observation = match serde_json::from_slice(body) {
        Ok(observation) => observation,
        Err(error) => {
            return Err(Failure {
                kind: FailureKind::Json(error),
                output,
            });
        }
    };
    if observation.mode != mode {
        return Err(Failure {
            kind: FailureKind::Mode,
            output,
        });
    }
    if observation.shape != [observation.values.len()] {
        return Err(Failure {
            kind: FailureKind::Shape,
            output,
        });
    }
    Ok(observation)
}

#[cfg(all(test, any(unix, windows)))]
#[path = "tests.rs"]
mod tests;
