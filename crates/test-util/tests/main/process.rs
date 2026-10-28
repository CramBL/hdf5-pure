use std::time::Duration;

use rstest::rstest;

use test_util::process::{self, Failure, Fixture, Readiness};

fn fixture(mode: &str, timeout: Duration) -> Fixture {
    let mut command = process::command(env!("CARGO_BIN_EXE_process-fixture")).unwrap();
    command.arg(mode);
    Fixture::spawn(command, timeout).unwrap()
}

#[test]
fn captures_output_larger_than_pipe_buffers() {
    let output = fixture("output", Duration::from_secs(10)).finish().unwrap();
    assert!(output.status.success(), "{output:?}");
    assert_eq!(output.stdout, vec![b'o'; 256 * 1024]);
    assert_eq!(output.stderr, vec![b'e'; 256 * 1024]);
}

#[test]
fn preserves_unsuccessful_exit_and_diagnostics() {
    let output = fixture("failure", Duration::from_secs(10))
        .finish()
        .unwrap();
    assert_eq!(output.status.code(), Some(17));
    assert!(output.stdout.is_empty());
    assert_eq!(output.stderr, b"fixture failed\n");
}

#[rstest]
#[case::readiness(true)]
#[case::completion(false)]
fn bounds_silent_children(#[case] readiness: bool) {
    let child = fixture("silent", Duration::from_millis(100));
    let error = if readiness {
        child
            .wait_ready(Readiness::try_from(b"READY\n".as_slice()).unwrap())
            .unwrap_err()
    } else {
        child.finish().unwrap_err()
    };
    let Failure::Timeout(output) = error else {
        panic!("expected Timeout, got {error:?}");
    };
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(output.stderr.is_empty());
}

#[test]
fn rejects_unexpected_readiness() {
    let child = fixture("wrong", Duration::from_secs(10));
    let error = child
        .wait_ready(Readiness::try_from(b"READY\n".as_slice()).unwrap())
        .unwrap_err();
    let Failure::Readiness(output) = error else {
        panic!("expected Readiness, got {error:?}");
    };
    assert!(!output.status.success());
    assert_eq!(output.stdout, b"WRONG\n");
    assert_eq!(output.stderr, b"fixture not ready\n");
}

#[test]
fn rejects_and_reaps_continuously_writing_children() {
    let child = fixture("growing", Duration::from_secs(10));
    #[cfg(target_os = "linux")]
    let path = std::path::PathBuf::from(format!("/proc/{}", child.id()));
    let started = std::time::Instant::now();
    let error = child
        .wait_ready(Readiness::try_from(b"READY\n".as_slice()).unwrap())
        .unwrap_err();
    assert!(started.elapsed() < Duration::from_secs(10));
    let Failure::Readiness(output) = error else {
        panic!("expected Readiness, got {error:?}");
    };
    assert!(!output.status.success());
    assert!(!output.stdout.is_empty());
    assert!(output.stdout.iter().all(|&byte| byte == b'o'));
    assert_eq!(output.stderr, b"fixture keeps writing\n");
    #[cfg(target_os = "linux")]
    assert!(!path.exists());
}

#[test]
fn rejects_exit_before_readiness() {
    let child = fixture("exit", Duration::from_secs(10));
    let error = child
        .wait_ready(Readiness::try_from(b"READY\n".as_slice()).unwrap())
        .unwrap_err();
    let Failure::Exited(output) = error else {
        panic!("expected Exited, got {error:?}");
    };
    assert!(output.status.success());
    assert!(output.stdout.is_empty());
    assert!(output.stderr.is_empty());
}

#[rstest]
#[case::complete("ready")]
#[case::fragmented("fragmented")]
fn interrupts_ready_children(#[case] mode: &str) {
    let child = fixture(mode, Duration::from_secs(10));
    let child = child
        .wait_ready(Readiness::try_from(b"READY\n".as_slice()).unwrap())
        .unwrap();
    let output = child.interrupt().unwrap();
    assert!(!output.status.success());
    assert_eq!(output.stdout, b"READY\n");
    assert!(output.stderr.is_empty());
}

#[test]
fn rejects_empty_readiness() {
    let error = Readiness::try_from(b"".as_slice()).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn reports_launch_failure() {
    let missing = test_util::temp::temp_path("missing-process");
    let error = Fixture::spawn(
        std::process::Command::new(&*missing),
        Duration::from_secs(10),
    )
    .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
}

#[cfg(target_os = "linux")]
#[rstest]
#[case::drop(false)]
#[case::panic(true)]
fn reaps_children_during_cleanup(#[case] panic: bool) {
    let child = fixture("ready", Duration::from_secs(10));
    let child = child
        .wait_ready(Readiness::try_from(b"READY\n".as_slice()).unwrap())
        .unwrap();
    let path = std::path::PathBuf::from(format!("/proc/{}", child.id()));
    assert!(path.exists());
    if panic {
        let result = std::panic::catch_unwind(move || {
            let _child = child;
            panic!("parent test failed");
        });
        let payload = result.unwrap_err();
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"parent test failed"));
    } else {
        drop(child);
    }
    assert!(!path.exists());
}
