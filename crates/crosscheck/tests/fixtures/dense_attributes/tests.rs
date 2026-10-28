use std::process::{ExitStatus, Output};

use rstest::rstest;
use serde_json::error::Category;

use super::*;

#[rstest]
#[case::count(Operation::Count, br#"{"Count":9}"#, Response::Count(9))]
#[case::integer(Operation::Details, br#"{"Details":[{"Integer":{"name":"big","size":16,"sum":-7}}]}"#,
    Response::Details(vec![Attribute::Integer { name: "big".to_string(), size: 16, sum: -7 }]))]
#[case::other(Operation::Details, br#"{"Details":[{"Other":{"name":"text","size":4}}]}"#,
    Response::Details(vec![Attribute::Other { name: "text".to_string(), size: 4 }]))]
#[case::lookup(Operation::Lookup(vec!["found".to_string(), "missing".to_string()]),
    br#"{"Lookup":[{"Found":{"name":"found","sum":5}},{"Missing":"missing"}]}"#,
    Response::Lookup(vec![Lookup::Found { name: "found".to_string(), sum: 5 }, Lookup::Missing("missing".to_string())]))]
fn parses_complete_native_observations(
    #[case] operation: Operation,
    #[case] body: &[u8],
    #[case] expected: Response,
) {
    let outcome = observe(&operation, AbortPolicy::Forbid, output(exit(0), body)).unwrap();
    let Outcome::Completed(response) = outcome else {
        panic!("expected complete observation, got {outcome:?}");
    };
    assert_eq!(response, expected);
}

#[rstest]
#[case::missing_reply(Operation::Count, b"", Invalid::Json(Category::Eof))]
#[case::malformed_reply(Operation::Count, b"invalid", Invalid::Json(Category::Syntax))]
#[case::truncated_reply(Operation::Count, br#"{"Count":"#, Invalid::Json(Category::Eof))]
#[case::wrong_count_type(Operation::Count, br#"{"Count":"9"}"#, Invalid::Json(Category::Data))]
#[case::trailing_bytes(
    Operation::Count,
    br#"{"Count":9} extra"#,
    Invalid::Json(Category::Syntax)
)]
#[case::wrong_operation(Operation::Count, br#"{"Details":[]}"#, Invalid::Shape)]
#[case::empty_rejection(
    Operation::Count,
    br#"{"Rejected":{"stage":"Open","frames":[]}}"#,
    Invalid::Shape
)]
#[case::missing_sum(
    Operation::Details,
    br#"{"Details":[{"Integer":{"name":"big","size":16}}]}"#,
    Invalid::Json(Category::Data)
)]
#[case::unknown_field(
    Operation::Details,
    br#"{"Details":[{"Other":{"name":"text","size":4,"extra":0}}]}"#,
    Invalid::Json(Category::Data)
)]
#[case::duplicate_details(
    Operation::Details,
    br#"{"Details":[{"Other":{"name":"a","size":1}},{"Other":{"name":"a","size":2}}]}"#,
    Invalid::Shape
)]
#[case::wrong_lookup_name(Operation::Lookup(vec!["a".to_string()]), br#"{"Lookup":[{"Missing":"b"}]}"#, Invalid::Shape)]
#[case::incomplete_lookup(Operation::Lookup(vec!["a".to_string()]), br#"{"Lookup":[]}"#, Invalid::Shape)]
fn rejects_incomplete_or_malformed_responses(
    #[case] operation: Operation,
    #[case] body: &[u8],
    #[case] expected: Invalid,
) {
    let error = observe(&operation, AbortPolicy::Forbid, output(exit(0), body)).unwrap_err();
    match (error.kind, expected) {
        (FailureKind::Json(error), Invalid::Json(category)) => {
            assert_eq!(error.classify(), category)
        }
        (FailureKind::Shape, Invalid::Shape) => {}
        (actual, expected) => panic!("expected {expected:?}, got {actual:?}"),
    }
}

#[test]
fn rejects_missing_native_handshakes() {
    let mut output = output(exit(0), br#"{"Count":9}"#);
    output.stdout = br#"{"Count":9}"#.to_vec();
    let error = observe(&Operation::Count, AbortPolicy::Forbid, output).unwrap_err();
    assert!(matches!(error.kind, FailureKind::Handshake), "{error:?}");
}

#[test]
fn preserves_native_rejection_frames() {
    let body = br#"{"Rejected":{"stage":"Names","frames":[{"function":"H5Aiterate2","major":"Attribute","minor":"CantLoad","description":"metadata checksum"}]}}"#;
    let outcome = observe(
        &Operation::Count,
        AbortPolicy::Forbid,
        output(exit(0), body),
    )
    .unwrap();
    let Outcome::Completed(Response::Rejected(rejection)) = outcome else {
        panic!("expected native rejection, got {outcome:?}");
    };
    assert_eq!(
        rejection,
        Rejection {
            stage: Stage::Names,
            frames: vec![Frame {
                function: "H5Aiterate2".to_string(),
                major: "Attribute".to_string(),
                minor: "CantLoad".to_string(),
                description: "metadata checksum".to_string(),
            }],
        }
    );
}

#[rstest]
#[case::rust_panic(101)]
#[case::unrelated_exit(1)]
#[case::shell_abort_code(134)]
fn rejects_unsuccessful_exits(#[case] code: i32) {
    let mut output = output(exit(code), b"");
    output.stderr = b"H5A__attr_release_table assertion".to_vec();
    let error = observe(&Operation::Count, AbortPolicy::AttributeRelease, output).unwrap_err();
    assert!(matches!(error.kind, FailureKind::Exit), "{error:?}");
}

#[test]
fn accepts_only_the_permitted_native_abort() {
    let mut output = output(aborted(), b"");
    output.stderr = b"H5A__attr_release_table assertion".to_vec();
    let outcome = observe(&Operation::Count, AbortPolicy::AttributeRelease, output).unwrap();
    assert!(matches!(outcome, Outcome::Aborted), "{outcome:?}");
}

#[rstest]
#[case::forbidden(AbortPolicy::Forbid, b"", b"H5A__attr_release_table assertion")]
#[case::unrelated(AbortPolicy::AttributeRelease, b"", b"allocation failed")]
#[case::partial_reply(
    AbortPolicy::AttributeRelease,
    b"{",
    b"H5A__attr_release_table assertion"
)]
fn rejects_aborts_outside_the_contract(
    #[case] policy: AbortPolicy,
    #[case] body: &[u8],
    #[case] stderr: &[u8],
) {
    let mut output = output(aborted(), body);
    output.stderr = stderr.to_vec();
    let error = observe(&Operation::Count, policy, output).unwrap_err();
    assert!(matches!(error.kind, FailureKind::Exit), "{error:?}");
}

#[derive(Debug)]
enum Invalid {
    Json(Category),
    Shape,
}

fn output(status: ExitStatus, body: &[u8]) -> Output {
    Output {
        status,
        stdout: [START, body].concat(),
        stderr: Vec::new(),
    }
}

fn exit(code: i32) -> ExitStatus {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        ExitStatus::from_raw(code << 8)
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::ExitStatusExt;
        ExitStatus::from_raw(u32::try_from(code).unwrap())
    }
}

fn aborted() -> ExitStatus {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        ExitStatus::from_raw(6)
    }
    #[cfg(windows)]
    {
        exit(3)
    }
}
