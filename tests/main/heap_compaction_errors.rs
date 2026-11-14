use std::error::Error as StdError;
use std::io;

use hdf5_pure::Error;
use hdf5_pure::HeapCompactionState;
use rstest::rstest;

#[rstest]
#[case::restored(
    HeapCompactionState::Restored,
    "the failed collection is unchanged or restored"
)]
#[case::uncertain(
    HeapCompactionState::Uncertain,
    "restoration or allocation permission is uncertain"
)]
fn heap_compaction_errors_preserve_progress_state_and_source(
    #[case] state: HeapCompactionState,
    #[case] detail: &str,
) {
    let error = Error::HeapCompactionFailed {
        completed: 2,
        state,
        source: Box::new(Error::Io(io::Error::other("collection write failed"))),
    };
    let Error::HeapCompactionFailed {
        completed,
        state: actual,
        source,
    } = &error
    else {
        panic!("expected a heap compaction failure");
    };
    assert_eq!(*completed, 2);
    assert_eq!(*actual, state);
    let Error::Io(inner) = source.as_ref() else {
        panic!("unexpected source: {source:?}");
    };
    assert_eq!(inner.kind(), io::ErrorKind::Other);
    assert_eq!(inner.to_string(), "collection write failed");
    assert_eq!(
        error.to_string(),
        format!(
            "global heap compaction failed after 2 completed collections: I/O error: collection write failed. {detail}"
        )
    );
    let cause = error.source().unwrap().downcast_ref::<Error>().unwrap();
    assert!(std::ptr::eq(cause, source.as_ref()));
    assert_eq!(
        cause
            .source()
            .unwrap()
            .downcast_ref::<io::Error>()
            .unwrap()
            .kind(),
        io::ErrorKind::Other
    );
}

#[rstest]
#[case::unpublished(false)]
#[case::published(true)]
fn restoration_diagnostics_preserve_scope_and_both_errors(#[case] published: bool) {
    let restoration = Error::CommitPartiallyApplied {
        refusal: Box::new(Error::Io(io::Error::new(
            io::ErrorKind::WriteZero,
            "operation failed",
        ))),
        restore: Box::new(Error::Io(io::Error::new(
            io::ErrorKind::BrokenPipe,
            "restoration failed",
        ))),
    };
    let error = if published {
        Error::HeapCompactionFailed {
            completed: 2,
            state: HeapCompactionState::Uncertain,
            source: Box::new(restoration),
        }
    } else {
        restoration
    };
    let diagnostic = "an operation failed (I/O error: operation failed) and restoration could not be confirmed (I/O error: restoration failed)";
    let restoration = if published {
        assert_eq!(
            error.to_string(),
            format!(
                "global heap compaction failed after 2 completed collections: {diagnostic}. restoration or allocation permission is uncertain"
            )
        );
        let Error::HeapCompactionFailed {
            completed,
            state,
            source,
        } = &error
        else {
            panic!("expected a published compaction failure");
        };
        assert_eq!(*completed, 2);
        assert_eq!(*state, HeapCompactionState::Uncertain);
        let cause = error.source().unwrap().downcast_ref::<Error>().unwrap();
        assert!(std::ptr::eq(cause, source.as_ref()));
        cause
    } else {
        assert_eq!(error.to_string(), diagnostic);
        &error
    };
    let Error::CommitPartiallyApplied { refusal, restore } = restoration else {
        panic!("expected a restoration failure: {restoration:?}");
    };
    let Error::Io(operation) = refusal.as_ref() else {
        panic!("unexpected operation error: {refusal:?}");
    };
    assert_eq!(operation.kind(), io::ErrorKind::WriteZero);
    assert_eq!(operation.to_string(), "operation failed");
    let Error::Io(restored) = restore.as_ref() else {
        panic!("unexpected restoration error: {restore:?}");
    };
    assert_eq!(restored.kind(), io::ErrorKind::BrokenPipe);
    assert_eq!(restored.to_string(), "restoration failed");
    let cause = restoration
        .source()
        .unwrap()
        .downcast_ref::<Error>()
        .unwrap();
    assert!(std::ptr::eq(cause, restore.as_ref()));
    let cause = cause.source().unwrap().downcast_ref::<io::Error>().unwrap();
    assert_eq!(cause.kind(), io::ErrorKind::BrokenPipe);
    assert_eq!(cause.to_string(), "restoration failed");
}
