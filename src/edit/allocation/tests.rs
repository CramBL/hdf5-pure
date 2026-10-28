use hdf5_pure_core::__private::PhysicalAddress;
use rstest::rstest;

use super::*;

mod io;
mod persistent;

#[rstest]
#[case::one_allocation(1)]
#[case::batch(16)]
fn complete_publication_confirms_every_allocation_in_its_batch(#[case] count: u64) {
    let mut publication = AllocationPublication::new();
    let allocations: Vec<_> = (0..count)
        .map(|index| {
            publication
                .provision(FileRange::new(PhysicalAddress::new(64 + index * 16), 16).unwrap())
                .unwrap()
        })
        .collect();
    let PublicationAction::Publish(attempt) = publication.begin().unwrap() else {
        panic!("expected pending allocation publication");
    };
    let confirmed = publication.complete(attempt).unwrap();
    for (index, allocation) in allocations.into_iter().enumerate() {
        let allocation = confirmed.confirm(allocation).unwrap();
        assert_eq!(
            publication.require(&allocation).unwrap(),
            FileRange::new(PhysicalAddress::new(64 + index as u64 * 16), 16).unwrap()
        );
    }
}

#[rstest]
#[case::same_session(false)]
#[case::other_session(true)]
fn uncertain_publication_grants_no_reference_permission(#[case] other_session: bool) {
    let mut owner = AllocationPublication::new();
    let allocation = owner
        .provision(FileRange::new(PhysicalAddress::new(64), 32).unwrap())
        .unwrap();
    let PublicationAction::Publish(attempt) = owner.begin().unwrap() else {
        panic!("expected pending allocation publication");
    };
    let confirmed = owner
        .complete(attempt)
        .unwrap()
        .confirm(allocation)
        .unwrap();
    let mut other = AllocationPublication::new();
    let publication = if other_session {
        &mut other
    } else {
        &mut owner
    };
    publication.make_uncertain();
    assert!(matches!(
        publication.require(&confirmed),
        Err(Error::AllocationPublicationUncertain)
    ));
    assert!(matches!(
        publication.provision(FileRange::new(PhysicalAddress::new(96), 8).unwrap()),
        Err(Error::AllocationPublicationUncertain)
    ));
    assert!(matches!(
        publication.begin(),
        Err(Error::AllocationPublicationUncertain)
    ));
}

#[rstest]
#[case::unrelated_session(true)]
#[case::later_publication(false)]
fn allocation_permission_is_bound_to_its_session_and_publication(#[case] unrelated: bool) {
    let mut first = AllocationPublication::new();
    let allocation = first
        .provision(FileRange::new(PhysicalAddress::new(64), 16).unwrap())
        .unwrap();
    let PublicationAction::Publish(attempt) = first.begin().unwrap() else {
        panic!("expected pending allocation publication");
    };
    let confirmed = first.complete(attempt).unwrap();
    let mut second = AllocationPublication::new();
    let source = if unrelated { &mut second } else { &mut first };
    let next = source
        .provision(FileRange::new(PhysicalAddress::new(96), 16).unwrap())
        .unwrap();
    let error = match confirmed.confirm(next) {
        Ok(_) => panic!("expected a mismatched allocation publication"),
        Err(error) => error,
    };
    let Error::Format(FormatError::Internal(reason)) = error else {
        panic!("expected an allocation publication mismatch, got {error:?}");
    };
    assert_eq!(
        reason,
        "allocation belongs to another session or publication"
    );
    let allocation = confirmed.confirm(allocation).unwrap();
    assert_eq!(first.require(&allocation).unwrap().length(), 16);
    let error = second.require(&allocation).unwrap_err();
    let Error::Format(FormatError::Internal(reason)) = error else {
        panic!("expected an allocation publication mismatch, got {error:?}");
    };
    assert_eq!(
        reason,
        "allocation belongs to another session or publication"
    );
}

#[test]
fn an_empty_publication_cannot_authorize_a_later_allocation() {
    let mut publication = AllocationPublication::new();
    let PublicationAction::Current(empty) = publication.begin().unwrap() else {
        panic!("expected a publication without pending allocations");
    };
    let allocation = publication
        .provision(FileRange::new(PhysicalAddress::new(64), 16).unwrap())
        .unwrap();
    let error = match empty.confirm(allocation) {
        Ok(_) => panic!("expected a mismatched allocation publication"),
        Err(error) => error,
    };
    let Error::Format(FormatError::Internal(reason)) = error else {
        panic!("expected an allocation publication mismatch, got {error:?}");
    };
    assert_eq!(
        reason,
        "allocation belongs to another session or publication"
    );
}

#[rstest]
#[case::before_attempt(false)]
#[case::unpublished_attempt(true)]
fn later_batch_confirmation_rejects_canceled_allocations(#[case] attempted: bool) {
    let mut state = AllocationPublication::new();
    let range = FileRange::new(PhysicalAddress::new(64), 16).unwrap();
    let discarded = state.provision(range).unwrap();
    if attempted {
        let PublicationAction::Publish(attempt) = state.begin().unwrap() else {
            panic!("expected a publication attempt");
        };
        state.cancel(attempt).unwrap();
    } else {
        state.discard_provisional().unwrap();
    }
    let replacement = state.provision(range).unwrap();
    let PublicationAction::Publish(attempt) = state.begin().unwrap() else {
        panic!("expected a replacement publication attempt");
    };
    let publication = state.complete(attempt).unwrap();
    let error = match publication.confirm(discarded) {
        Ok(_) => panic!("a canceled allocation was confirmed"),
        Err(error) => error,
    };
    let Error::Format(FormatError::Internal(reason)) = error else {
        panic!("expected an allocation generation mismatch, got {error:?}");
    };
    assert_eq!(
        reason,
        "allocation belongs to another session or publication"
    );
    let allocation = publication.confirm(replacement).unwrap();
    assert_eq!(state.require(&allocation).unwrap(), range);
}
