//! Child processes for explicit test fixture binaries.
//!
//! [`Fixture`] captures standard output and standard error in temporary files. Its
//! readiness and completion waits share a deadline measured from [`Fixture::spawn`].
//! Dropping the fixture attempts to kill and reap the direct child, including during
//! unwinding. Call [`Fixture::finish`] or [`Fixture::interrupt`] to check cleanup errors.

use std::env;
use std::ffi::OsStr;
use std::fmt;
use std::io::{self, Read};
use std::process::{Child, Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use tempfile::NamedTempFile;

/// A child process with captured diagnostics and a deadline for waits.
///
/// The child has an open standard-input pipe. Readiness fixtures can wait for input
/// until the parent interrupts them. Output is captured in temporary files, avoiding
/// pipe-buffer limits. The output size is not limited.
///
/// The deadline is checked when waiting. Cleanup applies to the direct child,
/// including a runner. Descendant processes require separate cleanup.
#[derive(Debug)]
pub struct Fixture {
    child: Child,
    stdout: NamedTempFile,
    stderr: NamedTempFile,
    deadline: Instant,
}

impl Fixture {
    /// Starts the child with a shared readiness and completion timeout.
    ///
    /// Replaces the command's standard streams with an open input pipe and temporary
    /// output files. Other command settings are preserved.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the deadline overflows, temporary-file setup fails,
    /// or the child cannot be started.
    pub fn spawn(mut command: Command, timeout: Duration) -> io::Result<Self> {
        let deadline = Instant::now().checked_add(timeout).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "fixture deadline overflows")
        })?;
        let stdout = NamedTempFile::new()?;
        let stderr = NamedTempFile::new()?;
        let child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::from(stdout.reopen()?))
            .stderr(Stdio::from(stderr.reopen()?))
            .spawn()?;
        Ok(Self {
            child,
            stdout,
            stderr,
            deadline,
        })
    }

    /// Waits until the child's complete standard output equals `expected`.
    ///
    /// Partial prefixes are permitted while the child is running. The child must
    /// remain alive until readiness is observed. On failure, cleanup is attempted
    /// before returning. Each readiness read is limited to the expected length plus
    /// one byte.
    ///
    /// # Errors
    ///
    /// Returns [`Failure::Exited`] if the child exits before readiness is observed,
    /// [`Failure::Readiness`] for unexpected output, or [`Failure::Timeout`] when
    /// the shared deadline expires. Returns [`Failure::Io`] if reading output,
    /// inspecting the child, or cleanup fails, or the read limit overflows.
    pub fn wait_ready(mut self, expected: Readiness<'_>) -> Result<Self, Failure> {
        let limit = u64::try_from(expected.0.len())
            .map_err(io::Error::other)?
            .checked_add(1)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "readiness length overflows")
            })?;
        loop {
            if Instant::now() >= self.deadline {
                return Err(Failure::Timeout(self.terminate()?));
            }
            if let Some(status) = self.child.try_wait()? {
                return Err(Failure::Exited(self.output(status)?));
            }
            let mut stdout = Vec::new();
            self.stdout.reopen()?.take(limit).read_to_end(&mut stdout)?;
            if stdout == expected.0 {
                return Ok(self);
            }
            if !expected.0.starts_with(&stdout) {
                return Err(Failure::Readiness(self.terminate()?));
            }
            self.pause();
        }
    }

    /// Waits for the child to exit and returns its status and captured output.
    ///
    /// An unsuccessful exit is returned as [`Output`]. The caller checks its status
    /// and parses the response according to the fixture's protocol. The input pipe
    /// stays open while waiting.
    ///
    /// # Errors
    ///
    /// Returns [`Failure::Timeout`] if the child is running when the shared deadline
    /// expires, or [`Failure::Io`] if inspecting the child, cleanup, or reading output
    /// fails. On failure, cleanup is attempted before returning.
    pub fn finish(mut self) -> Result<Output, Failure> {
        loop {
            if let Some(status) = self.child.try_wait()? {
                return Ok(self.output(status)?);
            }
            if Instant::now() >= self.deadline {
                return Err(Failure::Timeout(self.terminate()?));
            }
            self.pause();
        }
    }

    /// Kills and reaps a running child, returning its status and captured output.
    ///
    /// # Errors
    ///
    /// Returns [`Failure::Exited`] if the child has already exited, or [`Failure::Io`]
    /// if inspecting the child, cleanup, or reading output fails. On failure,
    /// cleanup is attempted before returning.
    pub fn interrupt(mut self) -> Result<Output, Failure> {
        if let Some(status) = self.child.try_wait()? {
            return Err(Failure::Exited(self.output(status)?));
        }
        Ok(self.terminate()?)
    }

    /// Returns the child's operating-system process identifier.
    pub fn id(&self) -> u32 {
        self.child.id()
    }

    fn terminate(&mut self) -> io::Result<Output> {
        self.child.kill()?;
        let status = self.child.wait()?;
        self.output(status)
    }

    fn output(&self, status: std::process::ExitStatus) -> io::Result<Output> {
        Ok(Output {
            status,
            stdout: read(&self.stdout)?,
            stderr: read(&self.stderr)?,
        })
    }

    fn pause(&self) {
        thread::sleep(
            self.deadline
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(10)),
        );
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.child.kill().ok();
        self.child.wait().ok();
    }
}

/// A nonempty sequence of bytes expected on standard output before child completion.
///
/// # Examples
///
/// ```
/// use test_util::process::Readiness;
///
/// let readiness = Readiness::try_from(b"READY\n".as_slice())?;
/// let error = Readiness::try_from(b"".as_slice()).unwrap_err();
/// assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
/// # Ok::<(), std::io::Error>(())
/// ```
#[derive(Debug)]
pub struct Readiness<'a>(&'a [u8]);

impl<'a> TryFrom<&'a [u8]> for Readiness<'a> {
    type Error = io::Error;

    /// Borrows the expected readiness bytes.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] with [`io::ErrorKind::InvalidInput`] if `bytes` is empty.
    fn try_from(bytes: &'a [u8]) -> Result<Self, Self::Error> {
        if bytes.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "fixture readiness must contain bytes",
            ));
        }
        Ok(Self(bytes))
    }
}

/// A lifecycle or readiness failure, with captured output when available.
#[derive(Debug)]
pub enum Failure {
    /// The child exited before readiness or before an interruption request.
    Exited(Output),
    /// An operating-system operation or capture-size calculation failed.
    Io(io::Error),
    /// The child's standard output differed from the expected readiness bytes.
    Readiness(Output),
    /// The shared deadline expired while waiting for readiness or completion.
    Timeout(Output),
}

impl From<io::Error> for Failure {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exited(output) => diagnostics(formatter, "fixture exited early", output),
            Self::Io(error) => error.fmt(formatter),
            Self::Readiness(output) => {
                diagnostics(formatter, "unexpected fixture readiness", output)
            }
            Self::Timeout(output) => diagnostics(formatter, "fixture deadline expired", output),
        }
    }
}

impl std::error::Error for Failure {}

/// Configures a fixture command with the optional `CROSS_TARGET_RUNNER` wrapper.
///
/// Pass the explicit fixture binary's `CARGO_BIN_EXE_<name>` path. A nonempty runner
/// is split on whitespace into a program and arguments, followed by `binary`.
/// Shell quoting is not interpreted. With an absent or whitespace-only runner, the
/// command executes `binary` directly. Add fixture arguments to the returned
/// [`Command`].
///
/// # Errors
///
/// Returns [`io::Error`] with [`io::ErrorKind::InvalidInput`] if the runner is not
/// Unicode.
pub fn command(binary: impl AsRef<OsStr>) -> io::Result<Command> {
    command_with_runner(
        binary.as_ref(),
        env::var_os("CROSS_TARGET_RUNNER").as_deref(),
    )
}

fn command_with_runner(binary: &OsStr, runner: Option<&OsStr>) -> io::Result<Command> {
    let runner = runner
        .map(|runner| {
            runner.to_str().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "cross runner is not Unicode")
            })
        })
        .transpose()?;
    let mut args = runner.unwrap_or_default().split_whitespace();
    let Some(program) = args.next() else {
        return Ok(Command::new(binary));
    };
    let mut command = Command::new(program);
    command.args(args).arg(binary);
    Ok(command)
}

fn read(file: &NamedTempFile) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    file.reopen()?.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn diagnostics(formatter: &mut fmt::Formatter<'_>, kind: &str, output: &Output) -> fmt::Result {
    write!(
        formatter,
        "{kind}: {status}; stdout: {stdout:?}; stderr: {stderr:?}",
        status = output.status,
        stdout = String::from_utf8_lossy(&output.stdout),
        stderr = String::from_utf8_lossy(&output.stderr),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use rstest::rstest;

    #[rstest]
    #[case::absent(None, "fixture", vec![])]
    #[case::empty(Some(""), "fixture", vec![])]
    #[case::wrapper(Some("runner --arg value"), "runner", vec!["--arg", "value", "fixture"])]
    fn runner_preserves_the_fixture_and_arguments(
        #[case] runner: Option<&str>,
        #[case] program: &str,
        #[case] args: Vec<&str>,
    ) {
        let command = command_with_runner(OsStr::new("fixture"), runner.map(OsStr::new)).unwrap();
        assert_eq!(command.get_program(), program);
        assert_eq!(command.get_args().collect::<Vec<_>>(), args);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_non_unicode_runners() {
        use std::os::unix::ffi::OsStrExt;

        let error = command_with_runner(OsStr::new("fixture"), Some(OsStr::from_bytes(&[0xff])))
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }
}
