use std::process::{ExitStatus, Output};

use rstest::rstest;
use serde_json::error::Category;

use super::{FailureKind, Mode, Observation};

#[rstest]
#[case::empty_swmr(Mode::Swmr, br#"{"mode":"Swmr","shape":[0],"values":[]}"#, vec![])]
#[case::populated_swmr(Mode::Swmr, br#"{"mode":"Swmr","shape":[3],"values":[0,1,2]}"#, vec![0, 1, 2])]
#[case::normal(Mode::Normal, br#"{"mode":"Normal","shape":[3],"values":[0,1,2]}"#, vec![0, 1, 2])]
fn parses_complete_observations(#[case] mode: Mode, #[case] body: &[u8], #[case] values: Vec<i32>) {
    assert_eq!(
        super::observe(mode, output(0, body)).unwrap(),
        Observation {
            mode,
            shape: [values.len()],
            values,
        },
    );
}

#[rstest]
#[case::empty(b"", Category::Eof)]
#[case::truncated(br#"{"mode":"Swmr""#, Category::Eof)]
#[case::trailing(br#"{"mode":"Swmr","shape":[0],"values":[]} false"#, Category::Syntax)]
#[case::missing_shape(br#"{"mode":"Swmr","values":[]}"#, Category::Data)]
#[case::missing_values(br#"{"mode":"Swmr","shape":[0]}"#, Category::Data)]
#[case::unknown_field(
    br#"{"mode":"Swmr","shape":[0],"values":[],"extra":true}"#,
    Category::Data
)]
#[case::unknown_mode(br#"{"mode":"Other","shape":[0],"values":[]}"#, Category::Data)]
#[case::invalid_shape(br#"{"mode":"Swmr","shape":[-1],"values":[]}"#, Category::Data)]
#[case::invalid_values(
    br#"{"mode":"Swmr","shape":[1],"values":[2147483648]}"#,
    Category::Data
)]
#[case::duplicate_field(
    br#"{"mode":"Swmr","mode":"Swmr","shape":[0],"values":[]}"#,
    Category::Data
)]
#[case::scalar(br#"{"mode":"Swmr","shape":[],"values":[0]}"#, Category::Data)]
#[case::rank_two(br#"{"mode":"Swmr","shape":[1,1],"values":[0]}"#, Category::Syntax)]
fn rejects_malformed_observations(#[case] body: &[u8], #[case] category: Category) {
    let error = super::observe(Mode::Swmr, output(0, body)).unwrap_err();
    let FailureKind::Json(json) = &error.kind else {
        panic!("expected JSON failure, got {error}");
    };
    assert_eq!(json.classify(), category, "{error}");
}

#[rstest]
#[case::short_values(br#"{"mode":"Swmr","shape":[2],"values":[0]}"#)]
#[case::long_values(br#"{"mode":"Swmr","shape":[0],"values":[0]}"#)]
fn rejects_inconsistent_shapes(#[case] body: &[u8]) {
    let error = super::observe(Mode::Swmr, output(0, body)).unwrap_err();
    assert!(matches!(error.kind, FailureKind::Shape), "{error}");
}

#[rstest]
#[case::normal_instead_of_swmr(Mode::Swmr, br#"{"mode":"Normal","shape":[0],"values":[]}"#)]
#[case::swmr_instead_of_normal(Mode::Normal, br#"{"mode":"Swmr","shape":[0],"values":[]}"#)]
fn rejects_wrong_mode(#[case] mode: Mode, #[case] body: &[u8]) {
    let error = super::observe(mode, output(0, body)).unwrap_err();
    assert!(matches!(error.kind, FailureKind::Mode), "{error}");
}

#[rstest]
#[case::missing(b"")]
#[case::partial(b"HDF5_SWMR_READ")]
#[case::unprefixed_json(br#"{"mode":"Swmr","shape":[0],"values":[]}"#)]
fn rejects_missing_handshake(#[case] stdout: &[u8]) {
    let mut output = output(0, b"");
    output.stdout = stdout.to_vec();
    let error = super::observe(Mode::Swmr, output).unwrap_err();
    assert!(matches!(error.kind, FailureKind::Handshake), "{error}");
}

#[rstest]
#[case::failure(1)]
#[case::panic(101)]
#[case::missing_runner_binary(127)]
fn rejects_unsuccessful_exit_with_valid_response(#[case] code: i32) {
    let error = super::observe(
        Mode::Swmr,
        output(code, br#"{"mode":"Swmr","shape":[0],"values":[]}"#),
    )
    .unwrap_err();
    assert!(matches!(error.kind, FailureKind::Exit), "{error}");
    let diagnostic = error.to_string();
    assert!(diagnostic.contains("reader diagnostic"), "{diagnostic}");
    assert!(diagnostic.contains("HDF5_SWMR_READER"), "{diagnostic}");
}

fn output(code: i32, body: &[u8]) -> Output {
    #[cfg(unix)]
    let status = {
        use std::os::unix::process::ExitStatusExt;
        ExitStatus::from_raw(code << 8)
    };
    #[cfg(windows)]
    let status = {
        use std::os::windows::process::ExitStatusExt;
        ExitStatus::from_raw(u32::try_from(code).unwrap())
    };
    Output {
        status,
        stdout: [super::START, body].concat(),
        stderr: b"reader diagnostic".to_vec(),
    }
}
