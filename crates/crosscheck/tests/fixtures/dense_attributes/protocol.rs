//! Requests and observations shared by the dense-attribute fixture and its parent.
//!
//! The fixture accepts a file path and a JSON-encoded operation as arguments. It
//! writes and flushes [`START`] before native operations, then writes one JSON
//! response on completion. A native rejection includes the operation stage and
//! expanded error stack in [`Rejection`].
//! Infrastructure failures terminate the fixture unsuccessfully.

use serde::{Deserialize, Serialize};

/// The stdout prefix written before any native operation.
pub(crate) const START: &[u8] = b"HDF5_DENSE_NATIVE\n";

/// An attribute operation on the file's root group.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) enum Operation {
    Count,
    Details,
    InsertHuge,
    InsertManaged,
    Lookup(Vec<String>),
}

/// A completed observation or a native library rejection.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) enum Response {
    Count(usize),
    Details(Vec<Attribute>),
    Lookup(Vec<Lookup>),
    Rejected(Rejection),
}

/// An attribute's name and total data size in bytes, with a wrapping sum for `i64` values.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) enum Attribute {
    Integer { name: String, size: usize, sum: i64 },
    Other { name: String, size: usize },
}

#[cfg(test)]
impl Attribute {
    fn name(&self) -> &str {
        match self {
            Self::Integer { name, .. } | Self::Other { name, .. } => name,
        }
    }
}

/// The result of a named lookup, with a wrapping sum of the found attribute's `i64` values.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) enum Lookup {
    Found { name: String, sum: i64 },
    Missing(String),
}

/// The failed native operation and its expanded error stack.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Rejection {
    pub(crate) stage: Stage,
    pub(crate) frames: Vec<Frame>,
}

/// A native error-stack frame, with the major and minor error codes formatted as text.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Frame {
    pub(crate) function: String,
    pub(crate) major: String,
    pub(crate) minor: String,
    pub(crate) description: String,
}

/// The native operation that reported an error.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) enum Stage {
    AttributeOpen,
    AttributeRead,
    AttributeType,
    Close,
    Create,
    Flush,
    Names,
    Open,
    Write,
}

#[cfg(test)]
/// The native assertion abort permitted for an observation.
#[derive(Clone, Copy)]
pub(crate) enum AbortPolicy {
    /// Permits an abort with `H5A__attr_release_table` in standard error and no response body.
    AttributeRelease,
    Forbid,
}

#[cfg(test)]
/// A parsed native observation or a permitted native assertion abort.
#[derive(Debug)]
pub(crate) enum Outcome {
    Aborted,
    Completed(Response),
}

#[cfg(test)]
/// A protocol failure with the child's exit status and captured diagnostics.
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
    Shape,
}

#[cfg(test)]
impl std::fmt::Display for Failure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            FailureKind::Json(error) => write!(formatter, "invalid fixture JSON: {error}")?,
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
/// Parses a completed child process according to its requested operation and abort policy.
///
/// Successful responses must match `operation`. Attribute details have unique names,
/// and lookup results preserve the names and order in [`Operation::Lookup`]. A native
/// rejection has at least one frame, with nonempty major and minor error codes.
/// The caller asserts the expected values or rejection after parsing.
///
/// # Errors
///
/// Returns [`Failure`] for a missing handshake, an unexpected exit, malformed JSON,
/// or an incomplete or mismatched response. An allowed assertion abort requires the
/// handshake, an empty response body, the platform's abort status, and
/// `H5A__attr_release_table` in standard error.
pub(crate) fn observe(
    operation: &Operation,
    abort: AbortPolicy,
    output: std::process::Output,
) -> Result<Outcome, Failure> {
    let Some(body) = output.stdout.strip_prefix(START) else {
        return Err(Failure {
            kind: FailureKind::Handshake,
            output,
        });
    };
    if !output.status.success() {
        if matches!(abort, AbortPolicy::AttributeRelease)
            && body.is_empty()
            && is_abort(output.status)
            && output
                .stderr
                .windows(b"H5A__attr_release_table".len())
                .any(|bytes| bytes == b"H5A__attr_release_table")
        {
            return Ok(Outcome::Aborted);
        }
        return Err(Failure {
            kind: FailureKind::Exit,
            output,
        });
    }
    let response = match serde_json::from_slice(body) {
        Ok(response) => response,
        Err(error) => {
            return Err(Failure {
                kind: FailureKind::Json(error),
                output,
            });
        }
    };
    let matches = match (operation, &response) {
        (_, Response::Rejected(rejection)) => {
            !rejection.frames.is_empty()
                && rejection
                    .frames
                    .iter()
                    .all(|frame| !frame.major.is_empty() && !frame.minor.is_empty())
        }
        (Operation::Count, Response::Count(_)) => true,
        (
            Operation::Details | Operation::InsertHuge | Operation::InsertManaged,
            Response::Details(attributes),
        ) => {
            let names: std::collections::BTreeSet<_> =
                attributes.iter().map(Attribute::name).collect();
            names.len() == attributes.len()
        }
        (Operation::Lookup(names), Response::Lookup(attributes)) => {
            names.len() == attributes.len()
                && names.iter().zip(attributes).all(|(expected, attribute)| {
                    expected
                        == match attribute {
                            Lookup::Found { name, .. } | Lookup::Missing(name) => name,
                        }
                })
        }
        _ => false,
    };
    if !matches {
        return Err(Failure {
            kind: FailureKind::Shape,
            output,
        });
    }
    Ok(Outcome::Completed(response))
}

#[cfg(test)]
fn is_abort(status: std::process::ExitStatus) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        status.signal() == Some(ABORT_SIGNAL)
    }
    #[cfg(windows)]
    {
        status.code() == Some(ABORT_EXIT)
    }
    #[cfg(not(any(unix, windows)))]
    {
        false
    }
}

#[cfg(all(test, unix))]
const ABORT_SIGNAL: i32 = 6;
#[cfg(all(test, windows))]
const ABORT_EXIT: i32 = 3;

#[cfg(all(test, any(unix, windows)))]
#[path = "tests.rs"]
mod tests;
