use std::collections::BTreeMap;
use std::io;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::EditStorageConflict;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::RetainedKind;
use rstest::rstest;

use super::super::OriginalStorage;
use super::super::ParentPreparationError;
use super::super::SpaceClaim;
use super::super::StorageAllocation;
use super::super::StorageConflict;
use super::super::StorageIntegrityError;
use super::super::StorageOrigin;
use crate::edit::analysis::CountProtectionError;
use crate::edit::retirement::RetirementMergeError;
use crate::error::Error;
use crate::error::FormatError;
use crate::object_graph::ObjectId;

#[rstest]
#[case::protected_bounds(Geometry::Protection)]
#[case::parent_bounds(Geometry::Parent)]
#[case::batch_bounds(Geometry::Batch)]
#[case::retirement_overlap(Geometry::Retirement)]
#[case::protected_overlap(Geometry::Protected)]
#[case::allocation_overlap(Geometry::Allocation)]
#[case::count_overlap(Geometry::Count)]
#[case::count_reuse(Geometry::Reuse)]
#[case::count_aggregate(Geometry::Aggregate)]
#[case::count_retained(Geometry::Retained)]
#[case::end(Geometry::End)]
fn storage_failure_conversion_preserves_conflict_geometry(#[case] geometry: Geometry) {
    let first = FileRange::new(PhysicalAddress::new(4_294_967_600), 16).unwrap();
    let second = FileRange::new(PhysicalAddress::new(4_294_967_608), 16).unwrap();
    let allocation = StorageAllocation {
        range: first,
        origin: StorageOrigin::Unobserved,
    };
    let object = ObjectId::new(first.start());
    let end = second.start();
    let error = match geometry {
        Geometry::Protection => StorageIntegrityError::ProtectionBounds { allocation, end }.into(),
        Geometry::Parent => super::parent_error(
            empty_storage(),
            ParentPreparationError::Bounds { range: first, end },
        ),
        Geometry::Batch => super::parent_error(
            empty_storage(),
            ParentPreparationError::Batch(RetirementMergeError::Bounds { range: first, end }),
        ),
        Geometry::Retirement => super::parent_error(
            empty_storage(),
            ParentPreparationError::Batch(RetirementMergeError::Overlap { first, second }),
        ),
        Geometry::Protected => StorageIntegrityError::ConflictingProtection {
            first: allocation,
            second: StorageAllocation {
                range: second,
                ..allocation
            },
        }
        .into(),
        Geometry::Allocation => StorageConflict::Overlap {
            allocation,
            range: second,
            claim: SpaceClaim::Reusable,
        }
        .into(),
        Geometry::Count => StorageConflict::CountAssociation(CountProtectionError::Overlap {
            first: (object, first),
            second: (ObjectId::new(second.start()), second),
        })
        .into(),
        Geometry::Reuse => StorageConflict::CountAssociation(CountProtectionError::Reusable {
            object,
            allocation: first,
            reusable: second,
        })
        .into(),
        Geometry::Aggregate => StorageConflict::CountAssociation(CountProtectionError::Aggregate {
            object,
            allocation: first,
            spare: second,
        })
        .into(),
        Geometry::Retained => StorageConflict::CountAssociation(CountProtectionError::Retained {
            object,
            allocation: first,
            retained: second,
            kind: RetainedKind::Dead,
        })
        .into(),
        Geometry::End => StorageConflict::End {
            proposed: first.start(),
            required: first.end(),
        }
        .into(),
    };
    let Error::EditStorageConflict(actual) = error else {
        panic!("unexpected geometry error: {error:?}");
    };
    assert_eq!(
        actual,
        match geometry {
            Geometry::Protection | Geometry::Parent | Geometry::Batch =>
                EditStorageConflict::Bounds {
                    range: first.range(),
                    end
                },
            Geometry::End => EditStorageConflict::Truncation {
                proposed: first.start(),
                required: first.end()
            },
            _ => EditStorageConflict::Overlap {
                first: first.range(),
                second: second.range()
            },
        }
    );
}

#[rstest]
#[case::source(Cause::Source)]
#[case::parent_allocation(Cause::Parent)]
#[case::count_publication(Cause::Count)]
fn storage_failure_conversion_returns_original_io_errors(#[case] cause: Cause) {
    let source = Error::Io(io::Error::new(io::ErrorKind::BrokenPipe, "source failed"));
    let error: Error = match cause {
        Cause::Source => StorageConflict::Source(source).into(),
        Cause::Parent => {
            super::parent_error(empty_storage(), ParentPreparationError::Allocation(source))
        }
        Cause::Count => {
            StorageConflict::CountAssociation(CountProtectionError::Publication(source)).into()
        }
    };
    let Error::Io(error) = error else {
        panic!("unexpected source error: {error:?}");
    };
    assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
    assert_eq!(error.to_string(), "source failed");
}

#[rstest]
#[case::object_format(FormatCause::Object)]
#[case::parent_address(FormatCause::Parent)]
#[case::count_source(FormatCause::Count)]
#[case::truncated_source(FormatCause::Truncated)]
fn storage_failure_conversion_preserves_format_error_fields(#[case] cause: FormatCause) {
    let source = FormatError::UnexpectedSourceEof {
        expected: 4_294_967_616,
        available: 4_294_967_600,
    };
    let error: Error = match cause {
        FormatCause::Object => StorageIntegrityError::ObjectFormat {
            object: ObjectId::new(PhysicalAddress::new(1)),
            error: Box::new(source),
        }
        .into(),
        FormatCause::Parent => {
            super::parent_error(empty_storage(), ParentPreparationError::Address(source))
        }
        FormatCause::Count => {
            StorageConflict::CountAssociation(CountProtectionError::Source(source)).into()
        }
        FormatCause::Truncated => StorageIntegrityError::TruncatedSource {
            end: PhysicalAddress::new(4_294_967_600),
            required: PhysicalAddress::new(4_294_967_616),
        }
        .into(),
    };
    let Error::Format(FormatError::UnexpectedSourceEof {
        expected,
        available,
    }) = error
    else {
        panic!("unexpected format error: {error:?}");
    };
    assert_eq!((expected, available), (4_294_967_616, 4_294_967_600));
}

fn empty_storage() -> OriginalStorage {
    OriginalStorage {
        source: None,
        allocations: vec![],
        parents: Ok(BTreeMap::new()),
        integrity: Ok(()),
        unavailable: None,
    }
}

#[derive(Clone, Copy, Debug)]
enum Geometry {
    Aggregate,
    Allocation,
    Batch,
    Count,
    End,
    Parent,
    Protected,
    Protection,
    Retained,
    Retirement,
    Reuse,
}

#[derive(Clone, Copy, Debug)]
enum Cause {
    Count,
    Parent,
    Source,
}

#[derive(Clone, Copy, Debug)]
enum FormatCause {
    Count,
    Object,
    Parent,
    Truncated,
}
